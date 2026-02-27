import argparse
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11n.pt", help="Base model checkpoint")
    ap.add_argument("--data", default="config/data.yaml", help="Data config")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=50, help="Early stopping patience")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--optimizer", default="auto", help="Optimizer (SGD, Adam, AdamW, auto)")
    ap.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate")
    ap.add_argument("--workers", type=int, default=8, help="Dataloader workers")
    ap.add_argument("--cache", action="store_true", help="Cache images in RAM")
    ap.add_argument("--seed", type=int, default=0, help="Random seed")
    ap.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    ap.add_argument("--exist-ok", action="store_true", help="Overwrite existing project/name")
    ap.add_argument("--save-period", type=int, default=-1, help="Save checkpoint every N epochs (-1 = disabled)")
    ap.add_argument("--close-mosaic", type=int, default=10, help="Disable mosaic for final N epochs")
    ap.add_argument("--project", default="runs")
    ap.add_argument("--name", default="exp")
    ap.add_argument("--device", default="")  # "" = auto, or "0", "0,1"
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        patience=args.patience,
        batch=args.batch,
        optimizer=args.optimizer,
        lr0=args.lr0,
        workers=args.workers,
        cache=args.cache,
        seed=args.seed,
        resume=args.resume,
        exist_ok=args.exist_ok,
        save_period=args.save_period,
        close_mosaic=args.close_mosaic,
        project=args.project,
        name=args.name,
        device=args.device,
    )


if __name__ == "__main__":
    main()
