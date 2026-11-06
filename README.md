# kev deployment

Container image for serving [kev](https://github.com/jaredpalmer/kev) on a GPU host (arm64 or x86_64).

## Build

Native build on the target host (x86_64 or arm64):

    docker build -t kev:dev .

Explicit platform:

    docker build --platform linux/amd64 -t kev:dev .
    docker build --platform linux/arm64 -t kev:dev .

> [!WARNING]
> The build fails if torch has no CUDA (`torch.version.cuda is None`). Run the CUDA check below on the target host.

## Run

The image has no entrypoint. Run the serve command yourself, with your own flags:

    docker run --rm --gpus all -p 127.0.0.1:18009:8009 -v hf-cache:/hf-cache kev:dev \
        python -m kev.serve --run <checkpoint> --host 0.0.0.0 --port 8009

`kev.serve` takes `--host` and defaults to `127.0.0.1`. Inside a container that address is not reachable through `-p`, so pass `--host 0.0.0.0` and restrict exposure on the host side (`-p 127.0.0.1:...`).

Needs the NVIDIA container toolkit on the host. Check CUDA first:

    docker run --rm --gpus all kev:dev python -c "import torch; print(torch.cuda.is_available())"
