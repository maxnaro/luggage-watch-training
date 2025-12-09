import argparse
from ultralytics import YOLO

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8n.pt", help="Base model checkpoint")
    ap.add_argument("--data", default="configs/data.yaml", help="Data config")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--project", default="runs")
    ap.add_argument("--name", default="exp")
    ap.add_argument("--device", default="")  # "" = auto, or "0", "0,1"
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        project=args.project,
        name=args.name,
        device=args.device,
    )

if __name__ == "__main__":
    main()