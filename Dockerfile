# Multi-arch (arm64 + x86_64) image for serving kev.
ARG CUDA_VERSION=13.0.0
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
ENV UV_PYTHON_INSTALL_DIR=/opt/python
RUN uv python install ${PYTHON_VERSION} && uv venv --python ${PYTHON_VERSION} /opt/venv
ENV PATH=/opt/venv/bin:$PATH VIRTUAL_ENV=/opt/venv
RUN python -c "import sys; assert sys.version_info[:2] == tuple(map(int, '${PYTHON_VERSION}'.split('.'))), sys.version"

RUN git clone ${KEV_REPO} /src/kev && git -C /src/kev checkout ${KEV_REF}
RUN uv pip install /src/kev "flash-linear-attention==${FLA_VERSION}"

RUN uv pip install --index-url ${TORCH_INDEX} torch --reinstall \
    && python -c "import torch; assert torch.version.cuda is not None, 'CPU-only torch: ' + torch.__version__; print(torch.__version__, torch.version.cuda)"

# Re-check: nothing above may have swapped torch.
RUN python -c "import torch; assert torch.version.cuda is not None, torch.__version__"

# ---- RUNTIME ----
FROM docker.io/nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /opt/python /opt/python
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /src/kev /src/kev
ENV PATH=/opt/venv/bin:$PATH \
    HF_HOME=/hf-cache
VOLUME /hf-cache
WORKDIR /src/kev
