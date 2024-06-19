from ultralytics import YOLO

# Load a model
model = YOLO('./runs/detect/train13/weights/best.pt')

# Customize validation settings
validation_results = model.val(
  # data='inat-2017.yaml',
  imgsz=640,
  batch=32,
  conf=0.25,
  iou=0.6,
  # device='0'
)