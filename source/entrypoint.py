"""
Unified entrypoint for the luggage-watch-training container.

Usage inside the container:
    python3 source/entrypoint.py train
    python3 source/entrypoint.py export

Reads all parameters from /app/config.json (mounted at runtime).
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

from ultralytics import YOLO

CONFIG_PATH = Path("/app/config.json")


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        print(f"Error: config file not found at {CONFIG_PATH}", file=sys.stderr)
        print("Mount your config.json into the container:", file=sys.stderr)
        print('  -v "./source/config.json:/app/config.json"', file=sys.stderr)
        sys.exit(1)

    with open(CONFIG_PATH) as f:
        return json.load(f)


def run_train(config: dict) -> None:
    """Run YOLO training from the 'train' section of the config."""
    train_cfg = config.get("train", {})

    model_name = train_cfg.pop("model", "yolo11n.pt")
    print(f"[train] Loading base model: {model_name}")
    model = YOLO(model_name)

    # Map config keys directly to YOLO .train() kwargs
    defaults = {
        "data": "/app/data/data.yaml",
        "imgsz": 640,
        "epochs": 100,
        "patience": 50,
        "batch": 16,
        "optimizer": "auto",
        "lr0": 0.01,
        "workers": 8,
        "cache": False,
        "seed": 0,
        "resume": False,
        "exist_ok": False,
        "save_period": -1,
        "close_mosaic": 10,
        "project": "runs",
        "name": "exp",
        "device": "",
    }
    defaults.update(train_cfg)

    print(f"[train] Parameters: {defaults}")
    model.train(**defaults)


def run_export(config: dict) -> None:
    """Export a trained model to ONNX from the 'export' section of the config."""
    export_cfg = config.get("export", {})

    weights = export_cfg.pop("weights", None)
    out = export_cfg.pop("out", "/app/model/model.onnx")

    if not weights:
        # Auto-resolve from training run name
        run_name = config.get("train", {}).get("name", "exp")
        weights = f"/app/runs/{run_name}/weights/best.pt"

    if not Path(weights).exists():
        print(f"Error: weights not found at {weights}", file=sys.stderr)
        sys.exit(1)

    print(f"[export] Loading weights: {weights}")
    model = YOLO(weights)

    defaults = {
        "format": "onnx",
        "imgsz": 640,
        "opset": 12,
        "simplify": True,
        "dynamic": False,
        "half": False,
        "int8": False,
        "batch": 1,
    }
    defaults.update(export_cfg)

    print(f"[export] Parameters: {defaults}")
    path = model.export(**defaults)

    # Move to output path
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(path, out_path)
    print(f"[export] Saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Luggage Watch Training Container Entrypoint"
    )
    parser.add_argument(
        "mode",
        choices=["train", "export"],
        help="Action to perform: 'train' or 'export'",
    )
    args = parser.parse_args()

    config = load_config()

    if args.mode == "train":
        run_train(config)
    elif args.mode == "export":
        run_export(config)


if __name__ == "__main__":
    main()
