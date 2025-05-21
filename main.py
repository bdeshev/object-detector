from datetime import datetime
import tkinter as tk
from tkinter import messagebox
from ultralytics import YOLO
from ultralytics.engine.results import Results
import cv2

root = tk.Tk()
root.withdraw()

model = YOLO("yolo11n.pt")

cap = cv2.VideoCapture(0)

last_alert = datetime.now()

def bear_detected():
    global last_alert

    now = datetime.now()
    if (now - last_alert).total_seconds() > 2:
        print("Open secret door")
        last_alert = now
        # Show a message box
        messagebox.showinfo("SURPRISE!", "Secret door opened!")
        root.update()
    else:
        print("Already alerted, waiting for next detection.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    results = model(frame)
    
    annotated_frame = results[0].plot()
    
    cv2.imshow('YOLO Detection', annotated_frame)
    
    boxes = results[0].boxes
    names = results[0].names
    if boxes is not None:
        for box in boxes:
            class_id = int(box.cls[0])
            class_name = names[class_id]
            confidence = float(box.conf[0])
            if class_name == "teddy bear" and confidence >= 0.6:
                bear_detected()
                break
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()