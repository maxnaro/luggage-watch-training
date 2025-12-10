# Luggage Watch Training (Docker)

Training and export pipeline for YOLO-based suspicious luggage detection using Docker.

## Prerequisites

- **Docker Desktop** (Windows) or **Docker Engine** (Linux)
- **NVIDIA GPU** with correctly installed drivers
- **WSL 2** (if on Windows)

*If on Windows,* run the `setup.ps1` script from the luggage-watch repository (`/luggage-watch/helpers/setup.ps1`) to sort the prerequisites out for you.

## 1. Build the Docker Image

Run this command from the directory containing the `Dockerfile`:

```bash
docker build -t luggage-watch-training .
```

> **Note**: The first build may take a few minutes. Subsequent builds will be faster due to caching.

## 2. Data Setup

The container expects the dataset to be mounted at /app/data.

Ensure your local dataset folder is ready.

> **Important**: Check that your `config/data.yaml` file points to absolute paths (`/app/data/train`, `/app/data/val`) or relative paths that resolve correctly inside the container.

## 3. Training

Run the training container by mounting your local dataset folder and a local runs folder (to save results).

**PowerShell** (Windows):

```PowerShell
docker run --gpus all --ipc=host -it `
  -v "C:\Path\To\Your\Dataset:/app/data" `
  -v "${PWD}\runs:/app/runs" `
  luggage-watch-training
```

**Bash** (Linux/WSL):

```Bash
docker run --gpus all --ipc=host -it \
  -v /path/to/dataset:/app/data \
  -v $(pwd)/runs:/app/runs \
  luggage-watch-training
```

### Passing Arguments

The container runs source/train.py by default. You can append arguments to the end of the command to override defaults defined in the Dockerfile:

```Bash
# Example: Overriding epochs and batch size
... luggage-watch-training --epochs 50 --batch 32
```

## 4. Export to ONNX (for Jetson/TensorRT)

To export the trained model, override the container's entrypoint to run the export script. This ensures the export happens in the exact same environment as the training.

**PowerShell** (Windows):

```PowerShell
docker run --rm --gpus all `
  -v "${PWD}\runs:/app/runs" `
  --entrypoint python3 `
  luggage-watch-training `
  /app/source/export.py --weights /app/runs/exp/weights/best.pt --format onnx
```

**Bash** (Linux/WSL):

```Bash
docker run --rm --gpus all \
  -v $(pwd)/runs:/app/runs \
  --entrypoint python3 \
  luggage-watch-training \
  /app/source/export.py --weights /app/runs/exp/weights/best.pt --format onnx
```

## Notes

**TensorRT Engines**: Do not generate .engine files inside this container. Export to ONNX here, then copy the .onnx file to the target device (e.g., Jetson Orin Nano) and compile the engine there to ensure hardware compatibility.

**Persistence**: Always mount the runs volume. Any data saved inside the container (like trained weights) will be lost when the container stops if it is not mounted to a local folder.