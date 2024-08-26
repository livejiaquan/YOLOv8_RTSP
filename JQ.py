
from ultralytics import YOLO

# model = YOLO(r'D:\YOLOv8_data\YOLO_tools\yolov8-streamlit-detection-tracking\weights\0719_ppl_car_model_L.pt')
model = YOLO(r'D:\YOLOv8_data\YOLO_tools\yolov8-streamlit-detection-tracking\weights\0730_ppe.pt')

source = r'D:\YOLOv8_data\CAM13-images_part_1'

model.predict( source, 
               save= True, 
               imgsz= 640, 
               conf= 0.5,
            #    classes = 0,  # 僅檢測person
               save_txt= True,   # 保存符合Darklabel的YOLO txt格式
               save_frames= False,# 關閉即時串流視頻時同時輸出圖像和標籤
               )