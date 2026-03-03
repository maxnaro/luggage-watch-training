"""Standalone export script.

Accepts --weights, --out, and any Ultralytics YOLO .export() parameter
as CLI arguments.  Unknown flags are forwarded directly, so the script
never needs updating when new export parameters are added.

Usage:
    python export.py --weights runs/detect/exp/weights/best.pt --format onnx --imgsz 1280
"""

import argparse
import json
import shutil
from pathlib import Path

from ultralytics import YOLO

from helpers.cli import parse_extra_args


def main():
    ap = argparse.ArgumentParser(
        description="Export a YOLO model (any .export() kwarg accepted)",
    )
    ap.add_argument("--weights", required=True, help="Path to trained .pt weights")
    ap.add_argument("--out", default="model/model.onnx", help="Output path")
    args, extra = ap.parse_known_args()

    export_kwargs = parse_extra_args(extra)
    model = YOLO(args.weights)

    print(f"[export] Weights: {args.weights}")
    print(f"[export] Params : {json.dumps(export_kwargs, indent=2, default=str)}")

    path = model.export(**export_kwargs)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(path, out_path)
    print(f"[export] Saved to {out_path}")


if __name__ == "__main__":
    main()
