<<<<<<< HEAD
=======
from datetime import datetime
>>>>>>> e39e0a9bab0fe00a46c8251b3d977b4f918c2794
import tkinter as tk
from tkinter import messagebox
from ultralytics import YOLO
from ultralytics.engine.results import Results

root = tk.Tk()
root.withdraw()  # Hide the main window


root = tk.Tk()
root.withdraw()  # Hide the main window


# Load a model
model = YOLO("yolo11n.pt")  # load an official model

# Predict with the model (webcam)
<<<<<<< HEAD
results = model(source="0", stream=True, show = True)  

def bear_detected():
    print("Open secret door")
    messagebox.showinfo("SURPRISE!", "Secret door opened!")
=======
results: list[Results] = model(source="0", stream=True, show = True)

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
>>>>>>> e39e0a9bab0fe00a46c8251b3d977b4f918c2794

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

