import ultralytics

from ultralytics import YOLO
model=YOLO("../models/yolov8n.pt", task="Detect")

model.predict("../DATA/video (1080p).mp4",
              conf=0.2, show=True,)