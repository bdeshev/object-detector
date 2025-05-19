from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")  # load an official model

# Predict with the model
results = model(source = "0", show = True)  # predict on an image

# Access the results
# for result in results:
#     boxes = result.boxes  # get boxes
#     classes = result.names  # get class names