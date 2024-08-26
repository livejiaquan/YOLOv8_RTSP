# 專案名稱

此專案使用 YOLO 模型進行人員檢測，並依據 DarkLabel 的格式保存輸出。

## 功能

- 使用預訓練 YOLO 模型進行人員檢測。
- 支援保存符合 DarkLabel 格式的標籤檔案。
- 支援檢測即時串流視頻時同時輸出圖像和標籤。

## 修改詳情

### save_txt

- 調整 `save_txt` 功能以符合 DarkLabel 的格式要求。
- 根據源視頻的幀數來命名保存的文本檔案，採用八位數字的格式，以便於在 DarkLabel 中進行後續處理。

### save_frame

- 修改 `save_frame` 函數，以支持檢測即時串流視頻時同時輸出圖像和標籤。
- 確保輸出的檔案名以八位數字順序開始命名，以便後續處理。

## 安裝

在開始使用之前，請確保你已經安裝了所需的 Python 套件。你可以通過以下指令安裝：

```bash
pip install ultralytics==8.1.4
```

**為什麼需要安裝特定版本？**  
本專案依賴於 ultralytics 版本 8.1.4，因為我們的 RTSP 功能需要在這個版本下才可以確保正確執行。在其他版本中，可能存在不兼容的變更，導致 RTSP 相關功能無法正常運作。因此，請務必使用指定的版本以避免潛在的問題。

## 使用方法

使用預訓練的 YOLO 模型進行人員檢測，並保存標籤檔案和圖像。請參考以下範例代碼：

```python
from ultralytics import YOLO

# 載入預訓練的 YOLOv8n 模型
model = YOLO('yolov8n.pt')

source = '路徑'
for result in model.predict(source=rtsp_url, 
                            save=True, 
                            imgsz=640, 
                            conf=0.5,
                            classes=0,  # 僅檢測person
                            save_txt=True,   # 保存符合 DarkLabel 的 YOLO txt 格式
                            save_frames=True # 檢測即時串流視頻時同時輸出圖像和標籤
                            ):
    pass
```

## 注意事項

- 使用 `save_txt` 和 `save_frame` 功能時，請確保已正確配置 YOLO 預訓練模型和 DarkLabel。
- 在保存標籤檔案時，請確保使用正確的命名格式，以便後續處理。

## 版本要求

- Python 版本: 3.8+
- Ultralytics: 8.1.4
