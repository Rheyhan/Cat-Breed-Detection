import ultralytics

from ultralytics import YOLO
model=YOLO("runs/detect/yolotest5/weights/best.pt", task="Detect")

model.predict("../DATA/kucingkamping.png",
              conf=0.1, show=True, save=True)