from ultralytics import YOLO

model = YOLO("")
results = model.predict(
    source="",         
    device=0,
    imgsz=640,
    stream=True,
    show=True, # Opens a window
    conf=0.25
)

for r in results:
    # r.orig_img is the original frame; r.plot() gives annotated frame
    pass