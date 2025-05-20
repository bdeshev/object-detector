import tkinter as tk
from tkinter import messagebox
from ultralytics import YOLO

root = tk.Tk()
root.withdraw()  # Hide the main window


# Load a model
model = YOLO("yolo11n.pt")  # load an official model

# Predict with the model (webcam)
results = model(source="0", stream=True, show = True)  

def bear_detected():
    print("Open secret door")
    messagebox.showinfo("SURPRISE!", "Secret door opened!")

for result in results:
    boxes = result.boxes
    names = result.names
    if boxes is not None:
        for box in boxes:
            class_id = int(box.cls[0])
            class_name = names[class_id]
            confidence = float(box.conf[0])
            if class_name == "teddy bear" and confidence >= 0.6:
                bear_detected()
                break

