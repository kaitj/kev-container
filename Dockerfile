# Multi-arch (arm64 + x86_64) image for serving kev.
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu130
ARG PYTHON_VERSION=3.13
ARG UV_VERSION=latest

# ---- BUILDER: venv with CUDA torch and kev ----
FROM docker.io/library/ubuntu:24.04 AS builder
ARG TORCH_INDEX
ARG PYTHON_VERSION
ARG UV_VERSION
ARG KEV_REPO=https://github.com/jaredpalmer/kev
ARG KEV_REF=2074cc4
# Must match the pin in kev/fused_qwen35.py. Check after cloning kev.
ARG FLA_VERSION=0.5.2

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:${UV_VERSION} /uv /usr/local/bin/uv
ENV UV_PYTHON_INSTALL_DIR=/opt/python UV_NO_CACHE=1
RUN uv python install ${PYTHON_VERSION} && uv venv --python ${PYTHON_VERSION} /opt/venv
ENV PATH=/opt/venv/bin:$PATH VIRTUAL_ENV=/opt/venv
RUN python -c "import sys; assert sys.version_info[:2] == tuple(map(int, '${PYTHON_VERSION}'.split('.'))), sys.version"

# Drop .git and directories serving never reads. runs/ stays: the default --run runs/kev reads it.
RUN git clone ${KEV_REPO} /src/kev && git -C /src/kev checkout ${KEV_REF} \
    && rm -rf /src/kev/.git /src/kev/playground /src/kev/docs /src/kev/tests /src/kev/experiments /src/kev/evals
RUN uv pip install /src/kev "flash-linear-attention==${FLA_VERSION}"

RUN uv pip install --index-url ${TORCH_INDEX} torch --reinstall \
    && python -c "import torch; assert torch.version.cuda is not None, 'CPU-only torch: ' + torch.__version__; print(torch.__version__, torch.version.cuda)"

# Re-check: nothing above may have swapped torch.
RUN python -c "import torch; assert torch.version.cuda is not None, torch.__version__"

# Trim bytecode and test dirs from the venv. Do not remove nvidia-* packages: torch links them at import.
RUN find /opt/venv -name __pycache__ -type d -prune -exec rm -rf {} + \
    && find /opt/venv/lib -name tests -type d -prune -exec rm -rf {} +

# ---- RUNTIME ----
# The wheels bundle the CUDA libraries. The host driver is injected by the NVIDIA container toolkit.
FROM docker.io/library/ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /opt/python /opt/python
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /src/kev /src/kev
ENV PATH=/opt/venv/bin:$PATH \
    HF_HOME=/hf-cache
VOLUME /hf-cache
WORKDIR /src/kev
