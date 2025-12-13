from ultralytics import YOLO

model = YOLO("C:\\source\\luggage-watch\\training\\model\\yolo11s_131225.onnx")
results = model.predict(
    source="C:\\Users\\maxna\\Downloads\\drive-download-20251213T120438Z-3-001\\AVSS_E2.avi",         
    device=0,
    imgsz=640,
    stream=True,
    show=True, # Opens a window
    conf=0.25
)

for r in results:
    # r.orig_img is the original frame; r.plot() gives annotated frame
    pass