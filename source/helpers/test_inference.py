from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO(
        r"\path\to\runs\detect\tuned_26s\weights\best.pt"
    )
    metrics = model.val(
        data=r"\path\to\dataset\data.yaml",
        split="test",
        device=0,
        imgsz=640,
        conf=0.25,
    )

    print(f"mAP50:    {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")
