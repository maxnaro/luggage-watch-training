# Luggage Watch Training

Training and export pipeline for YOLO-based suspicious-luggage detection.

## Quickstart (WSL or Linux)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Data

- Use a mounted path or symlink (e.g., `datasets/` -> `/mnt/data/luggage`).
- `configs/data.yaml` should point to your train/val image/label folders and list class names.

## Training

```bash
python scripts/train.py \
  --model yolov8n.pt \
  --imgsz 640 \
  --epochs 100 \
  --batch 16 \
  --data config/data.yaml \
  --project runs \
  --name y8n-baseline
```

### Fix: GPU is not being utilised in training

```bash
source .venv\Scripts\activate
pip uninstall -y torch torchvision torchaudio

# Install CUDA 12.1 build (includes CUDA runtime; no separate toolkit needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Export to ONNX (for DeepStream/TensorRT)

```bash
python scripts/export.py \
  --weights runs/train/y8n-baseline/weights/best.pt \
  --imgsz 640 \
  --opset 12 \
  --simplify \
  --out model/yolov8n.onnx
```

## Notes

- Engines (`*.engine`) and checkpoints (`*.pt`, `*.ckpt`) should not be committed. See `.gitignore`.
- Build TensorRT engines on the target device (e.g., Jetson Orin Nano) for compatibility.