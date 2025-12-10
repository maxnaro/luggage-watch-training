# Luggage Watch Training (Docker)

Training and export pipeline for YOLO-based suspicious luggage detection using Docker.

## Quick Start (Windows)

Run everything (build + train + export) with the helper script:

```powershell
.\source\helpers\run.ps1 -b -t -e -c .\source\helpers\config.json
```

## Prerequisites

- Docker Desktop (Windows) or Docker Engine (Linux)
- NVIDIA GPU with drivers and CUDA runtime
- WSL 2 for Windows
- (Windows) Optional: run `/luggage-watch/helpers/setup.ps1` from the main repo to install prerequisites automatically.

## Data Layout

- Container expects data at `/app/data` and outputs runs to `/app/runs`.
- Ensure `config/data.yaml` points to `/app/data/train` and `/app/data/val` (absolute inside the container).

### Windows volume option

Mounting Windows folders to Linux containers can be slow. To avoid the I/O bottleneck, create a Docker volume that holds your dataset:

```powershell
.\source\helpers\create_docker_volume.ps1 -v luggage-watch-data -p "C:\path\to\your\dataset"
```

Use the volume name (`luggage-watch-data`) when mounting to `/app/data`.

## Build the image

From the repo root (where the Dockerfile lives):

```bash
docker build -t luggage-watch-training .
```

Or via the helper script (Windows):

```powershell
.\source\helpers\run.ps1 -b
```

> First build may take a few minutes; subsequent builds are cached.

## Train

Mount local data (or volume) and a local `runs` directory so results persist.

**PowerShell (Windows):**

```powershell
docker run --gpus all --ipc=host -it `
  -v "C:\Path\To\Your\Dataset:/app/data" `
  -v "${PWD}\runs:/app/runs" `
  luggage-watch-training
```

**Bash (Linux/WSL):**

```bash
docker run --gpus all --ipc=host -it \
  -v /path/to/dataset:/app/data \
  -v $(pwd)/runs:/app/runs \
  luggage-watch-training
```

Append training args to override defaults in `source/train.py` (example):

```bash
... luggage-watch-training --epochs 50 --batch 32
```

### Train via helper script (Windows)

`run.ps1` reads settings from `source/helpers/config.json`:

```json
{
  "projectName": "luggage-watch-training",
  "paths": {
    "data": "luggage-watch-data",
    "runs": "runs",
    "model": "model"
  },
  "train": {
    "model": "yolo11s.pt",
    "epochs": 100,
    "batch": 16,
    "imgsz": 640,
    "device": "0",
    "name": "yolo11s_101225"
  },
  "export": {
    "opset": 12,
    "simplify": true,
    "out": "/app/model/yolo11s_101225.onnx"
  }
}
```

Start training:

```powershell
.\source\helpers\run.ps1 -t -c .\source\helpers\config.json
```

## Export to ONNX (Jetson/TensorRT)

Run the export script inside the same container environment:

**PowerShell (Windows):**

```powershell
docker run --rm --gpus all `
  -v "${PWD}\runs:/app/runs" `
  --entrypoint python3 `
  luggage-watch-training `
  /app/source/export.py --weights /app/runs/exp/weights/best.pt --format onnx
```

**Bash (Linux/WSL):**

```bash
docker run --rm --gpus all \
  -v $(pwd)/runs:/app/runs \
  --entrypoint python3 \
  luggage-watch-training \
  /app/source/export.py --weights /app/runs/exp/weights/best.pt --format onnx
```

### Export via helper script (Windows)

Use the same `config.json` (export block shown above):

```powershell
.\source\helpers\run.ps1 -e -c .\source\helpers\config.json
```

## Full workflow with helper script (Windows)

```powershell
.\source\helpers\run.ps1 -b -t -e -c .\source\helpers\config.json
```

## Notes

- WSL memory limits (Windows): if training fails unexpectedly, ensure WSL has enough RAM.
  - Create or edit `%UserProfile%\.wslconfig` (example):
    ```ini
    [wsl2]
    memory=12GB  # adjust to available RAM
    processors=4 # optional
    swap=8GB     # optional; can be 0 to disable swap
    ```
  - Apply changes: close all WSL sessions, then run `wsl --shutdown` in PowerShell.
  - Reopen your WSL distro and rerun training; verify with `free -h` inside WSL.
- TensorRT engines: build ONNX here, then compile `.engine` on the target device (e.g., Jetson Orin Nano) to match hardware.
- Persistence: always mount `runs` (and `model` if exporting). Anything left inside the container is lost when it stops.

## Non-Docker setup (not recommended)

If you must run locally without Docker, create a virtual environment and install deps (WSL/Linux example):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Train directly:

```bash
python source/train.py \
  --model yolov8n.pt \
  --imgsz 640 \
  --epochs 100 \
  --batch 16 \
  --data config/data.yaml \
  --project runs \
  --name y8n-baseline
```

If GPU is not used, install the CUDA 12.1 PyTorch build:

```bash
source .venv/bin/activate
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Export to ONNX locally:

```bash
python source/export.py \
  --weights runs/train/y8n-baseline/weights/best.pt \
  --imgsz 640 \
  --opset 12 \
  --simplify \
  --out model/yolov8n.onnx
```