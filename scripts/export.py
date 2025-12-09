import argparse
from ultralytics import YOLO

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="Path to trained .pt weights")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--opset", type=int, default=12)
    ap.add_argument("--simplify", action="store_true")
    ap.add_argument("--dynamic", action="store_true")
    ap.add_argument("--out", default="model/model.onnx")
    args = ap.parse_args()

    model = YOLO(args.weights)
    model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=args.simplify,
        dynamic=args.dynamic,
        save=True,
        export_path=args.out,
    )

if __name__ == "__main__":
    main()