from ultralytics import YOLO

model = YOLO("best.pt")

results = model.predict(source='video.mp4', show=True, conf=0.6)