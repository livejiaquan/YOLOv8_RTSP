#%%
import cv2
import onnxruntime as ort
import numpy as np
from PIL import Image
from PIL import ImageDraw

def intersection(box1,box2):
    box1_x1,box1_y1,box1_x2,box1_y2 = box1[:4]
    box2_x1,box2_y1,box2_x2,box2_y2 = box2[:4]
    x1 = max(box1_x1,box2_x1)
    y1 = max(box1_y1,box2_y1)
    x2 = min(box1_x2,box2_x2)
    y2 = min(box1_y2,box2_y2)
    return (x2-x1)*(y2-y1)
    
def union(box1,box2):
    box1_x1,box1_y1,box1_x2,box1_y2 = box1[:4]
    box2_x1,box2_y1,box2_x2,box2_y2 = box2[:4]
    box1_area = (box1_x2-box1_x1)*(box1_y2-box1_y1)
    box2_area = (box2_x2-box2_x1)*(box2_y2-box2_y1)
    return box1_area + box2_area - intersection(box1,box2)
    
def iou(box1,box2):
    return intersection(box1,box2)/union(box1,box2)


ort_model = ort.InferenceSession(r'D:\YOLOv8_data\YOLO_tools\yolov8-streamlit-detection-tracking\weights\0802_ppl_car.onnx')
#%%
for input in ort_model.get_inputs():
    print("input name: ", input.name)
    print("input shape: ", input.shape)
    print("input type: ", input.type)
# %%
img = Image.open(r"D:\YOLOv8_data\0802_ppl_car_split\images\val\00000004.jpg")
img_width, img_height = img.size
img = img.resize((640,640))
img = img.convert("RGB")
#%%
input = np.array(img)
input = input.transpose(2,0,1)
input = input.reshape(1,3,640,640) #或者用expand_dims
input = input/255.0

# output:(640, 640, 3)
# %%
outputs = ort_model.get_outputs()
output = outputs[0]
print("Name:", output.name)
print("Type:", output.type)
print("Shape:", output.shape)

# %%
input = input.astype(np.float32) # input代表预处理后的数据，这里先转成单精度浮点
outputs = ort_model.run(["output0"], {"images":input})
output = outputs[0]
output.shape # (1, 7, 8400)
# %%
output = output[0]
output.shape # (7, 8400)
output = output.transpose() # 转置
output.shape # (8400, 7)
# %%
row = output[0]
print(row) 
# [     14.623      22.475      29.157      45.249  1.4901e-07  3.2783e-07  5.9605e-08]
# %%
yolo_classes = ['person', 'truck']

def parse_row(row):
    xc,yc,w,h = row[:4]
    x1 = (xc-w/2)/640*img_width
    y1 = (yc-h/2)/640*img_height
    x2 = (xc+w/2)/640*img_width
    y2 = (yc+h/2)/640*img_height
    prob = row[4:].max()
    class_id = row[4:].argmax()
    label = yolo_classes[class_id]
    return [x1,y1,x2,y2,label,prob]
# %%
boxes = [row for row in [parse_row(row) for row in output] if row[5]>0.5]

len(boxes) # 20
# %%

img = Image.open(r"D:\YOLOv8_data\0802_ppl_car_split\images\val\00000004.jpg")
draw = ImageDraw.Draw(img)

# NMS
boxes.sort(key=lambda x: x[5], reverse=True)

result = []

while len(boxes)>0:
    result.append(boxes[0])
    boxes = [box for box in boxes if iou(box,boxes[0])<0.7] #<0.7则不是同一物体，要保留
for box in boxes:
    x1,y1,x2,y2,class_id,prob = box
    draw.rectangle((x1,y1,x2,y2),None,"#00ff00")

# 已有代碼...

# 這裡繪製框框的代碼...
for box in result:
    x1, y1, x2, y2, label, prob = box
    draw.rectangle((x1, y1, x2, y2), outline="#00ff00")
    draw.text((x1, y1), f"{label}: {prob:.2f}", fill="#00ff00")

# 將PIL圖像轉換為OpenCV格式
img_cv2 = np.array(img)
img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_RGB2BGR)

# 顯示圖像
cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
cv2.imshow("Output", img_cv2)
cv2.waitKey(0)
cv2.destroyAllWindows()


# %%
