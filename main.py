# from datetime import datetime
# import tkinter as tk
# from tkinter import messagebox
# from typing import Callable
# from ultralytics import YOLO
# from ultralytics.engine.results import Results, Boxes
# import cv2


# class Detector:
#     def __init__(self, target_class: str, confidence_threshold: float, action: Callable):
#         self.target_class = target_class
#         self.confidence_threshold = confidence_threshold
#         self.last_alert = datetime.now()
#         self.action = action

#     def detect(self, box: Boxes, names: dict[int, str]) -> bool:
#         class_id = int(box.cls[0])
#         class_name = names[class_id]
#         confidence = float(box.conf[0])
#         if class_name == self.target_class and confidence >= self.confidence_threshold:
#             self._trigger_action()
#             return False
#         return True

#     def _trigger_action(self) -> None:
#         now = datetime.now()
#         if (now - self.last_alert).total_seconds() > 2:
#             print("Open secret door")
#             self.last_alert = now
#             self.action()
#         else:
#             print("Already alerted, waiting for next detection.")


# def main():
#     root = tk.Tk()
#     root.withdraw()

#     model = YOLO("yolo11n.pt")

#     cap = cv2.VideoCapture(0)

#     def messagebox_action():
#         # Show a message box
#         messagebox.showinfo("SURPRISE!", "Secret door opened!")
#         root.update()

#     detector = Detector(target_class="book", confidence_threshold=0.6, action=messagebox_action)

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         results: Results = model(frame)

#         annotated_frame = results[0].plot()

#         cv2.imshow('YOLO Detection', annotated_frame)

#         boxes = results[0].boxes
#         names = results[0].names
#         if boxes is not None:
#             for box in boxes:
#                 if not detector.detect(box, names):
#                     break

#         if cv2.waitKey(1) == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()
#     root.destroy()


# if __name__ == "__main__":
#     main()


import numpy as np
import gradio as gr

def sepia(input_img, text, slider_value):
    sepia_filter = np.array([
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131]
    ])
    sepia_img = input_img.dot(sepia_filter.T)
    sepia_img /= sepia_img.max()
    # Optionally use 'text' and 'slider_value' for additional processing
    return sepia_img


demo = gr.Interface(
    fn = sepia,
    title = "Image Filter",
    inputs = [
        gr.Image(label="Input Image"),
        gr.Textbox(label="Testing", value="Enter text here"),
        gr.Slider(label="Intensity", value=1, minimum=0, maximum=2, step=.2)
    ],
    outputs = "image"
)


demo.launch()