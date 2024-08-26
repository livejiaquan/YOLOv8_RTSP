#coding:utf-8
import argparse
import cv2
import os
import numpy as np
import onnxruntime as ort
import torch
import time
from ultralytics.utils import ASSETS, yaml_load
from ultralytics.utils.checks import check_requirements, check_yaml
from ultralytics import YOLO

class YOLOv8:
    """YOLOv8目标检测模型类，用于处理推理和可视化操作。"""
    def __init__(self, onnx_model, confidence_thres, iou_thres):
        """
        初始化YOLOv8类的实例。
        参数:
            onnx_model: ONNX模型的路径。
            confidence_thres: 过滤检测的置信度阈值。
            iou_thres: 非极大抑制的IoU（交并比）阈值。
        """
        self.onnx_model = onnx_model
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres

        # 从COCO数据集的配置文件加载类别名称
        self.classes = yaml_load(check_yaml("data_ppl.yaml"))["names"]

        # 为类别生成颜色调色板
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))

        # 初始化ONNX会话
        self.initialize_session(self.onnx_model)

    def draw_detections(self, img, box, score, class_id):
        """
        根据检测到的对象在输入图像上绘制边界框和标签。
        参数:
            img: 要绘制检测的输入图像。
            box: 检测到的边界框。
            score: 对应的检测得分。
            class_id: 检测到的对象的类别ID。
        返回:
            None
        """

        x1, y1, w, h = box
        color = self.color_palette[class_id]
        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)
        label = f"{self.classes[class_id]}: {score:.2f}"
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10
        cv2.rectangle(
            img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED
        )
        cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    def preprocess(self, img_path):
        """
        在进行推理之前，对输入图像进行预处理。
        返回:
            image_data: 预处理后的图像数据，准备好进行推理。
        """
        self.img = cv2.imread(img_path)
        self.img_height, self.img_width = self.img.shape[:2]
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.input_width, self.input_height))
        image_data = np.array(img) / 255.0
        image_data = np.transpose(image_data, (2, 0, 1))
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
        return image_data

    def postprocess(self, input_image, output):
        """
        对模型的输出进行后处理，以提取边界框、分数和类别ID。
        返回:
            numpy.ndarray: 输入图像，上面绘制了检测结果。
        """
        outputs = np.transpose(np.squeeze(output[0]))
        rows = outputs.shape[0]
        boxes, scores, class_ids = [], [], []
        x_factor = self.img_width / self.input_width
        y_factor = self.img_height / self.input_height

        for i in range(rows):
            classes_scores = outputs[i][4:]
            max_score = np.amax(classes_scores)

            if max_score >= self.confidence_thres:
                class_id = np.argmax(classes_scores)
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]
                left = int((x - w / 2) * x_factor)
                top = int((y - h / 2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                class_ids.append(class_id)
                scores.append(max_score)
                boxes.append([left, top, width, height])

        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_thres, self.iou_thres)

        for i in indices:
            box = boxes[i]
            score = scores[i]
            class_id = class_ids[i]
            self.draw_detections(input_image, box, score, class_id)
        return input_image

    def initialize_session(self, onnx_model):
        """
        初始化ONNX模型会话。
        :return:
        """
        providers = ["CUDAExecutionProvider"] if torch.cuda.is_available() else ["CPUExecutionProvider"]
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(onnx_model, session_options=session_options, providers=providers)
        return self.session

    def run_inference_on_folder(self, input_folder, output_folder):
        """
        在指定的文件夹中运行推理，并将带有检测结果的图像保存到输出文件夹中。
        参数:
            input_folder: 包含输入图像的文件夹路径。
            output_folder: 保存检测结果图像的文件夹路径。
        """
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        model_inputs = self.session.get_inputs()
        input_shape = model_inputs[0].shape
        self.input_width, self.input_height = input_shape[2], input_shape[3]

        for img_name in os.listdir(input_folder):
            img_path = os.path.join(input_folder, img_name)
            output_path = os.path.join(output_folder, img_name)
            
            start_time = time.time()
            img_data = self.preprocess(img_path)
            outputs = self.session.run(None, {model_inputs[0].name: img_data})
            output_image = self.postprocess(self.img, outputs)
            cv2.imwrite(output_path, output_image)
            print(f"Processed {img_name} in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    onnx_model_name = r"D:\YOLOv8_data\YOLO_tools\yolov8-streamlit-detection-tracking\weights\0802_ppl_car.onnx"
    input_folder = r"D:\YOLOv8_data\0802_ppl_car_split\images\val"
    output_folder = r"D:\YOLOv8_data\output_results"
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=onnx_model_name, help="请输入您的ONNX模型路径.")
    parser.add_argument("--input-folder", type=str, default=input_folder, help="输入图像的文件夹路径.")
    parser.add_argument("--output-folder", type=str, default=output_folder, help="输出图像的文件夹路径.")
    parser.add_argument("--conf-thres", type=float, default=0.3, help="置信度阈值.")
    parser.add_argument("--iou-thres", type=float, default=0.7, help="IoU(交并比)阈值.")
    args = parser.parse_args()

    detection = YOLOv8(args.model, args.conf_thres, args.iou_thres)
    detection.run_inference_on_folder(args.input_folder, args.output_folder)
