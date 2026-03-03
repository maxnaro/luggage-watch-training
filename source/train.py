"""Standalone training script.

Accepts --model and any Ultralytics YOLO .train() parameter as CLI
arguments.  Unknown flags are parsed as key=value pairs and forwarded
directly, so the script never needs updating when new YOLO parameters
are added.

Usage:
    python train.py --model yolo26s.pt --imgsz 1280 --epochs 100 --cls 0.8
"""

import argparse
import json

from ultralytics import YOLO

from helpers.cli import parse_extra_args


def main():
    ap = argparse.ArgumentParser(
        description="Train a YOLO model (any .train() kwarg accepted)",
    )
    ap.add_argument("--model", default="yolo11n.pt", help="Base model checkpoint")
    args, extra = ap.parse_known_args()

    train_kwargs = parse_extra_args(extra)
    model = YOLO(args.model)

    print(f"[train] Model : {args.model}")
    print(f"[train] Params: {json.dumps(train_kwargs, indent=2, default=str)}")

    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
