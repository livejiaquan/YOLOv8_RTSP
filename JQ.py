from ultralytics import YOLO

model = YOLO(r'C:\Users\limjiaquan\AppData\Local\anaconda3\envs\YOLOv8env\Lib\site-packages\ultralytics\yolov8n.pt')

rtsp_url = 'URL'
for result in model.predict( source = rtsp_url, 
               save= True, 
               imgsz= 640, 
               conf= 0.5,
               classes = 0,  # 僅檢測person
               save_txt= True,   # 保存符合Darklabel的YOLO txt格式
               save_frames= False,# 關閉即時串流視頻時同時輸出圖像和標籤
               ):
    pass