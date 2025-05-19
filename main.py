from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")  # load an official model

# Predict with the model (webcam)
results = model(source="0", stream=True, show = True)  

def phone_detected():
    print("Open secret door")

for result in results:
    boxes = result.boxes
    names = result.names
    if boxes is not None:
        for box in boxes:
            class_id = int(box.cls[0])
            class_name = names[class_id]
            if class_name == "cell phone":
                phone_detected()
                wait = input("Press Enter to continue...")  # practically crashes the server
                break  