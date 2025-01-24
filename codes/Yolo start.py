from ultralytics import YOLO


model = YOLO("last (train3).pt")
results = model(source='video2.mp4' , conf=0.55, show=True, save=True)
