from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")

# Customize validation settings
validation_results = model.val(data="coco8.yaml", 
                               imgsz=640, 
                               batch=16, 
                               conf=0.25, 
                               iou=0.6,
                               save_txt=True,
                               save_hybrid=True,
                               save_json=True)
