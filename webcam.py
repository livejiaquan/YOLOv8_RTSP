import cv2
import torch
from ultralytics import YOLO

# 加载YOLOv8模型
model = YOLO('yolov8n-pose.pt')  # 替换为您的YOLOv8模型路径

# 初始化网络摄像头
cap = cv2.VideoCapture(0)  # 0表示第一个摄像头

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 将帧转换为YOLO输入并进行预测
    results = model(frame)

    # 获取姿态检测结果
    for result in results:
        # 获取检测框和关键点
        bboxes = result.boxes.xyxy
        keypoints = result.keypoints.xy

        # 绘制检测框和关键点
        for bbox, kpts in zip(bboxes, keypoints):
            x1, y1, x2, y2 = map(int, bbox[:4])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            for x, y in kpts:
                cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)

    # 显示结果帧
    cv2.imshow('YOLOv8 Pose Detection', frame)

    # 按q键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头并关闭窗口
cap.release()
cv2.destroyAllWindows()
