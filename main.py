from datetime import datetime
import tkinter as tk
from tkinter import messagebox
from typing import Callable
from ultralytics import YOLO
from ultralytics.engine.results import Results, Boxes
import cv2


class Detector:
    def __init__(self, target_class: str, confidence_threshold: float, action: Callable):
        self.target_class = target_class
        self.confidence_threshold = confidence_threshold
        self.last_alert = datetime.now()
        self.action = action

    def detect(self, box: Boxes, names: dict[int, str]) -> bool:
        class_id = int(box.cls[0])
        class_name = names[class_id]
        confidence = float(box.conf[0])
        if class_name == self.target_class and confidence >= self.confidence_threshold:
            self._trigger_action()
            return False
        return True

    def _trigger_action(self) -> None:
        now = datetime.now()
        if (now - self.last_alert).total_seconds() > 2:
            print("Open secret door")
            self.last_alert = now
            self.action()
        else:
            print("Already alerted, waiting for next detection.")


def main():
    root = tk.Tk()
    root.withdraw()

    model = YOLO("yolo11n.pt")

    cap = cv2.VideoCapture(0)

    def messagebox_action():
        # Show a message box
        messagebox.showinfo("SURPRISE!", "Secret door opened!")
        root.update()

    detector = Detector(target_class="book", confidence_threshold=0.6, action=messagebox_action)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results: Results = model(frame)

        annotated_frame = results[0].plot()

        cv2.imshow('YOLO Detection', annotated_frame)

        boxes = results[0].boxes
        names = results[0].names
        if boxes is not None:
            for box in boxes:
                if not detector.detect(box, names):
                    break

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    root.destroy()


if __name__ == "__main__":
    main()
