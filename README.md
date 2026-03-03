# Luggage Watch Training

Training and export pipeline for YOLO-based suspicious luggage detection using Docker.

Everything is driven by a single **config.json** — edit it once, then tell the container what to do: `train` or `export`.

## Quick Start

```powershell
# 0. Start Docker Desktop
docker desktop start

# 1. Build the image
docker build -t luggage-watch-training .

# 2. Train
docker run --gpus all --ipc=host --rm -it `
  -v "${PWD}\source\config.json:/app/config.json:ro" `
  -v "my-dataset:/app/data" `
  -v "${PWD}\runs:/app/runs" `
  -v "${PWD}\model:/app/model" `
  luggage-watch-training train

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

All parameters live in `source/config.json`:

```json
{
    "train": {
        "data": "/app/data/dataset.yaml",
        "model": "/app/runs/detect/yolo26s_1280_020326/weights/best.pt",
        "epochs": 100,
        "patience": 30,
        "batch": 0.8,
        "imgsz": 1280,
        "optimizer": "AdamW",
        "lr0": 0.0008,
        "lrf": 0.01,
        "warmup_epochs": 3.0,
        "freeze": 10,
        "workers": 8,
        "cache": "disk",
        "seed": 0,
        "resume": true,
        "exist_ok": false,
        "save_period": -1,
        "close_mosaic": 15,
        "iou": 0.8,
        "label_smoothing": 0.05,
        "cls": 0.8,
        "scale": 0.7,
        "device": "0",
        "name": "yolo26s_1280_020326"
    },
    "export": {
        "format": "onnx",
        "imgsz": 1280,
        "opset": 12,
        "simplify": true,
        "dynamic": false,
        "half": true,
        "int8": false,
        "batch": 1,
        "out": "/app/model/yolo26s_1280_020326.onnx"
    }
}
```

| Section       | Purpose                                                                                                                  |
| ------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `projectName` | Docker image name                                                                                                        |
| `paths.data`  | Host path (or Docker volume) to the dataset                                                                              |
| `paths.runs`  | Host directory for training outputs                                                                                      |
| `paths.model` | Host directory for exported ONNX models                                                                                  |
| `train.*`     | Passed directly to `YOLO.train()` — add any [Ultralytics parameter](https://docs.ultralytics.com/modes/train/#arguments) |
| `export.*`    | Passed directly to `YOLO.export()` — `weights` auto-resolved from `train.name` if omitted                                |

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