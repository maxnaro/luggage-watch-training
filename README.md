# Luggage Watch Training

Training and export pipeline for YOLO-based suspicious luggage detection using Docker.

Everything is driven by a single **config.json** — edit it once, then tell the container what to do: `train` or `export`.

## Quick Start

```powershell
# 0. Start Docker Desktop
docker desktop start

# 1. Build the image
docker build -t luggage-watch-training .

# 2. Train (single-phase config uses "train" key automatically)
docker run --gpus all --ipc=host --rm -it `
  -v "${PWD}\source\config.json:/app/config.json:ro" `
  -v "my-dataset:/app/data" `
  -v "${PWD}\runs:/app/runs" `
  -v "${PWD}\model:/app/model" `
  luggage-watch-training train

# 2b. Or select a phase for multi-phase configs
docker run --gpus all --ipc=host --rm -it `
  -v "${PWD}\source\config.json:/app/config.json:ro" `
  -v "my-dataset:/app/data" `
  -v "${PWD}\runs:/app/runs" `
  -v "${PWD}\model:/app/model" `
  luggage-watch-training train --phase phase1

# 3. Export to ONNX
docker run --gpus all --rm -it `
  -v "${PWD}\source\config.json:/app/config.json:ro" `
  -v "${PWD}\runs:/app/runs" `
  -v "${PWD}\model:/app/model" `
  luggage-watch-training export
```

## Prerequisites

- Docker Desktop (Windows) or Docker Engine (Linux)
- NVIDIA GPU with drivers and CUDA runtime
- WSL 2 for Windows
- (Windows, optional) Run `/luggage-watch/helpers/setup.ps1` from the main repo to install prerequisites automatically.

## Configuration

The config supports **single-phase** (one `train` key) or **multi-phase** training (multiple `phase*` keys). All train parameters are passed directly to `YOLO.train()`.

### Single-phase config

```json
{
    "train": {
        "model": "yolo26s.pt",
        "data": "/app/data/dataset.yaml",
        "epochs": 150,
        "imgsz": 640,
        "name": "my_run"
    },
    "export": { ... }
}
```

### Multi-phase config

Use named `phase*` keys for staged training (e.g. baseline then fine-tune). Phase 2 references phase 1's output weights:

```json
{
    "phase1_coco_baseline": {
        "model": "yolo26s.pt",
        "data": "/app/data/dataset.yaml",
        "epochs": 150,
        "freeze": 0,
        "lr0": 0.0008,
        "name": "coco_baseline_26s"
    },
    "phase2_finetune": {
        "model": "/app/runs/detect/coco_baseline_26s/weights/best.pt",
        "data": "/app/data/dataset.yaml",
        "epochs": 80,
        "freeze": 10,
        "lr0": 0.0002,
        "name": "finetune_mixed_26s"
    },
    "export": { ... }
}
```

Select the phase at runtime with `--phase`:

```bash
# Phase 1: train baseline
docker run --gpus all --ipc=host --rm -it \
  -v ./source/config.json:/app/config.json:ro \
  -v coco-data:/app/data \
  -v ./runs:/app/runs \
  luggage-watch-training train --phase phase1

# Phase 2: fine-tune (same runs volume, swap data volume)
docker run --gpus all --ipc=host --rm -it \
  -v ./source/config.json:/app/config.json:ro \
  -v mixed-data:/app/data \
  -v ./runs:/app/runs \
  luggage-watch-training train --phase phase2
```

Prefix matching is supported — `--phase phase1` matches `phase1_coco_baseline`. If `--phase` is omitted, the `train` key is used (or the first `phase*` key).

| Section       | Purpose                                                                                                                  |
| ------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `train` or `phase*` | Passed directly to `YOLO.train()` — add any [Ultralytics parameter](https://docs.ultralytics.com/modes/train/#arguments) |
| `export`      | Passed directly to `YOLO.export()` — `weights` auto-resolved from `train.name` if omitted                                |

### Data Layout

The container expects the dataset at `/app/data`. Ensure your `data.yaml` inside the dataset points to `/app/data/train` and `/app/data/val`.

#### Windows Volume Option

Mounting Windows paths into Linux containers can be slow. Create a Docker volume instead:

```powershell
.\source\helpers\create_docker_volume.ps1 -Volume luggage-watch-data -Path "C:\path\to\dataset"
```

Then set `paths.data` to the volume name (`luggage-watch-data`).

## Usage

The container accepts `train` or `export` as its command. Mount your config and volumes:

**PowerShell (Windows):**

```powershell
docker run --gpus all --ipc=host --rm -it `
  -v "${PWD}\source\config.json:/app/config.json:ro" `
  -v "my-dataset:/app/data" `
  -v "${PWD}\runs:/app/runs" `
  -v "${PWD}\model:/app/model" `
  luggage-watch-training train
```

**Bash (Linux/WSL):**

```bash
docker run --gpus all --ipc=host --rm -it \
  -v "$(pwd)/source/config.json:/app/config.json:ro" \
  -v /path/to/dataset:/app/data \
  -v "$(pwd)/runs:/app/runs" \
  -v "$(pwd)/model:/app/model" \
  luggage-watch-training train
```

Replace `train` with `export` to export the model.

## Notes

- **WSL memory**: If training fails, ensure WSL has enough RAM. Create or edit `%UserProfile%\.wslconfig`:
  ```ini
  [wsl2]
  memory=12GB
  processors=4
  swap=8GB
  ```
  Then run `wsl --shutdown` and reopen WSL.
- **TensorRT**: Build ONNX here, then compile `.engine` on the target device (e.g., Jetson Orin Nano).
- **Persistence**: Always mount `runs` and `model`. Anything inside the container is lost when it stops.

## Non-Docker Setup (Not Recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Train
python source/train.py --model yolo11m.pt --data config/data.yaml --epochs 100

# Export
python source/export.py --weights runs/exp/weights/best.pt --out model/model.onnx --simplify
```