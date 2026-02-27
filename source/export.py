import argparse
import shutil
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="Path to trained .pt weights")
    ap.add_argument("--format", default="onnx", help="Export format (onnx, torchscript, engine, etc.)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--opset", type=int, default=12)
    ap.add_argument("--simplify", action="store_true")
    ap.add_argument("--dynamic", action="store_true")
    ap.add_argument("--half", action="store_true", help="FP16 quantization")
    ap.add_argument("--int8", action="store_true", help="INT8 quantization")
    ap.add_argument("--batch", type=int, default=1, help="Export batch size")
    ap.add_argument("--out", default="model/model.onnx")
    args = ap.parse_args()

    model = YOLO(args.weights)
    path = model.export(
        format=args.format,
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=args.simplify,
        dynamic=args.dynamic,
        half=args.half,
        int8=args.int8,
        batch=args.batch,
    )

    shutil.move(path, args.out)


if __name__ == "__main__":
    main()
