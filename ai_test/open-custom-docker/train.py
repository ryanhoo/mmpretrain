from ultralytics import YOLO

# load a pretrained model (recommended for training)
model = YOLO('yolov8m.pt')

# Train the model
# results = model.train(data='OpenImagesV7.yaml', epochs=600, imgsz=640, batch=32, workers=32)
# results = model.train(data='coco.yaml', epochs=600, imgsz=640, batch=64, workers=32)
results = model.train(data='inat-2017.yaml', epochs=600, imgsz=640, batch=32, workers=32)