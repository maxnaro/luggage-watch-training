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
        "name": "exp",
        "device": "",
    }
    defaults.update(train_cfg)

    print(f"[train] Parameters: {defaults}")
    model.train(**defaults)


def run_export(config: dict) -> None:
    """Export a trained model to ONNX using DeepStream export script."""
    import export as ds_export
    
    export_cfg = config.get("export", {})

    weights = export_cfg.get("weights", None)
    out = export_cfg.get("out", "/app/model/model.onnx")

    if not weights:
        # Auto-resolve from training run name
        run_name = config.get("train", {}).get("name", "exp")
        weights = f"/app/runs/detect/{run_name}/weights/best.pt"

    if not Path(weights).exists():
        print(f"Error: weights not found at {weights}", file=sys.stderr)
        sys.exit(1)

    print(f"[export] Using DeepStream export for weights: {weights}")
    
    # Build arguments for the DeepStream export script
    import types
    args = types.SimpleNamespace()
    args.weights = weights
    args.size = export_cfg.get("size", export_cfg.get("imgsz", [640]))
    if isinstance(args.size, int):
        args.size = [args.size]
    args.opset = export_cfg.get("opset", 17)
    args.simplify = export_cfg.get("simplify", False)
    args.dynamic = export_cfg.get("dynamic", False)
    args.batch = export_cfg.get("batch", 1)
    
    print(f"[export] Parameters: size={args.size}, opset={args.opset}, "
          f"simplify={args.simplify}, dynamic={args.dynamic}, batch={args.batch}")
    
    # Run the DeepStream export
    ds_export.main(args)
    
    # Move to output path if different from default
    default_out = weights.rsplit(".", 1)[0] + ".onnx"
    if out != default_out:
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if Path(default_out).exists():
            shutil.move(default_out, out_path)
            print(f"[export] Moved to {out_path}")
        
        # Also move labels.txt if created
        if Path("labels.txt").exists():
            labels_dest = out_path.parent / "labels.txt"
            shutil.move("labels.txt", labels_dest)
            print(f"[export] Moved labels.txt to {labels_dest}")


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
