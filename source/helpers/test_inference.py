from ultralytics import YOLO

model = YOLO("C:\\path\\to\\trained\\model")
results = model.predict(
    source="C:\\path\\to\\test\\video",
    device=0,
    imgsz=640,
    stream=True,
    show=True,  # Opens a window
    conf=0.25,
)

for r in results:
    # r.orig_img is the original frame; r.plot() gives annotated frame
    pass
