# 修改詳情

在 `predictor.py` 檔案中進行了重要的修改，主要涉及 `save_txt` 和 `save_frame` 兩個函數。具體修改如下：

## save_txt

- 對 `save_txt` 功能進行了調整，以符合 DarkLabel 的格式要求。
- 根據源視頻的幀數來命名保存的文本檔案，採用八位數字的格式，以便於在 DarkLabel 中進行後續處理。

## save_frame

- 修改了 `save_frame` 函數，以支持檢測即時串流視頻時同時輸出圖像和標籤。
- 對部分代碼進行了修改，確保輸出的檔案名以八位數字順序開始命名，以便後續處理。

這些修改使得 `predictor.py` 更適用於使用預訓練的 YOLO 模型進行人員檢測，並根據 DarkLabel 的要求調整輸出。

## 注意事項

- 使用 `save_txt` 和 `save_frame` 功能時，請確保已正確配置 YOLO 預訓練模型和 DarkLabel。
- 在保存標籤檔案時，請確保使用正確的命名格式，以便後續處理。

## 示例用法 JQ.py

```python
# 使用預訓練的 YOLO 模型進行人員檢測，並保存標籤檔案和圖像
from ultralytics import YOLO
# Load a pretrained YOLOv8n model
model = YOLO('yolov8n.pt')

source = '路徑'
for result in model.predict( source = rtsp_url, 
               save= True, 
               imgsz= 640, 
               conf= 0.5,
               classes = 0,  # 僅檢測person
               save_txt= True,   # 保存符合Darklabel的YOLO txt格式
               save_frames= True,# 關閉即時串流視頻時同時輸出圖像和標籤
               ):
    pass
```

## 參考

