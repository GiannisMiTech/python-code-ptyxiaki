from ultralytics import YOLO
from playsound import playsound

model = YOLO("last (train3).pt")
results = model(source=0 , show=True , conf=0.5 , save=True)

for r in results:
  if 0 in r.boxes.cls:   # 0 is cane class
    playsound('Ding Sound Effect.mp3')
  elif 1 in r.boxes.cls:    # 1 is guide dog class
    playsound('Tick Sound.mp3')

    #to exit close vscode(if using vscode)
