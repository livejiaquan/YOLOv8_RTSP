import argparse
from ultralytics import YOLO
 
def main(rtsp_url):
    print("OKOKOKOK")
    model = YOLO(r'C:\Users\limjiaquan\AppData\Local\anaconda3\envs\YOLOv8env\Lib\site-packages\ultralytics\yolov8n.pt')
 
    for result in model.predict(source=rtsp_url,
                                save=True,
                                imgsz=640,
                                conf=0.5,
                                classes=0,  # 仅检测person
                                save_txt=True,   # 保存符合Darklabel的YOLO txt格式
                                save_frames=True,
                                stream=True):  # 关闭即时串流视频时同时输出图像和标签
        pass
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RTSP Stream YOLO Detection')
    parser.add_argument('--rtsp_url', type=str, required=True, help='RTSP URL of the stream')
    args = parser.parse_args()
    main(args.rtsp_url)