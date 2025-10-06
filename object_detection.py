
# object_detection.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import glob
import random
import time
import torch
import torchvision
import torchvision.transforms as T
from torchvision.transforms import functional as F
from ultralytics import YOLO
from torch.utils.data import DataLoader
from torchvision.datasets import VOCDetection
from torchvision.transforms import ToTensor

# Set dataset paths
LABEL_PATH = r'C:\Users\ASUS\Music\Object Detection Project\coco128\labels\train2017'
IMAGE_PATH = r'C:\Users\ASUS\Music\Object Detection Project\coco128\images\train2017'

# ------------------ PART 1: EDA ------------------
print("Performing EDA...")
annotations = []
label_files = glob.glob(os.path.join(LABEL_PATH, '*.txt'))

for file in label_files:
    image_name = os.path.basename(file).replace('.txt', '.jpg')
    with open(file, 'r') as f:
        for line in f.readlines():
            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            annotations.append([image_name, int(class_id), x_center, y_center, width, height])

df = pd.DataFrame(annotations, columns=["image", "class_id", "x_center", "y_center", "width", "height"])
df['area'] = df['width'] * df['height']

# Plot class distribution
plt.figure(figsize=(12, 5))
sns.countplot(x='class_id', data=df)
plt.title("Class Distribution")
plt.tight_layout()
plt.savefig("class_distribution.png")
plt.close()

# Box per image
bbox_count = df.groupby('image').size()
plt.figure(figsize=(10, 4))
sns.histplot(bbox_count, bins=15, kde=True)
plt.title("Bounding Boxes per Image")
plt.tight_layout()
plt.savefig("bboxes_per_image.png")
plt.close()

# Area distribution
plt.figure(figsize=(10, 4))
sns.histplot(df['area'], bins=20, kde=True)
plt.title("Bounding Box Area Distribution")
plt.tight_layout()
plt.savefig("bbox_area_distribution.png")
plt.close()

# ------------------ PART 2: MODEL TRAINING ------------------
print("Training models...")

# 1. Train YOLOv5
print("Training YOLOv5...")
def detect_with_yolov5(image_path, weights='yolov5s.pt'):
    device = select_device('')
    model = torch.load(weights, map_location=device)['model'].float()
    model.eval()

    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (640, 640))
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0

    pred = model(img_tensor)[0]
    pred = non_max_suppression(pred)[0]

    if pred is not None:
        for *xyxy, conf, cls in pred:
            label = f'{int(cls.item())} {conf:.2f}'
            xyxy = list(map(int, xyxy))
            cv2.rectangle(img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
            cv2.putText(img, label, (xyxy[0], xyxy[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    return img

# 2. Train YOLOv8
print("Training YOLOv8...")
yolo8 = YOLO('yolov8s.pt')
yolo8.train(data='coco128.yaml', epochs=3, imgsz=640, project='yolo8_train', name='yolov8_results', batch=8)

# 3. Train Faster R-CNN (using a simplified dummy VOC loader for demonstration)
print("Training Faster R-CNN on dummy VOC subset (for demo)...")

# Transform function (dummy here, just ToTensor)
transform = T.Compose([ToTensor()])

# Load VOC dataset (2007 train subset)
train_dataset = VOCDetection(root='.', year='2007', image_set='train', download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

# Load Faster R-CNN model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

model.train()

# Use just 1 batch for demo
for images, targets_raw in train_loader:
    images = [img.to(device) for img in images]
    targets = []

    for t in targets_raw:
        ann = t['annotation']
        boxes = []
        labels = []

        objects = ann['object']
        if not isinstance(objects, list):
            objects = [objects]

        for obj in objects:
            bbox = obj['bndbox']
            xmin = float(bbox['xmin'])
            ymin = float(bbox['ymin'])
            xmax = float(bbox['xmax'])
            ymax = float(bbox['ymax'])
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(1)  # Use dummy label "1" for simplicity

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        targets.append({'boxes': boxes.to(device), 'labels': labels.to(device)})

    try:
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        print("Loss:", losses.item())
    except Exception as e:
        print("Error during training:", e)

    break  # Only one batch for demo
print("Training complete (limited epochs).")

# ------------------ PART 3: MODEL INFERENCE ------------------
sample_image_name = random.choice(df['image'].unique())
sample_image_path = os.path.join(IMAGE_PATH, sample_image_name)
img = cv2.imread(sample_image_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_pil = F.to_pil_image(img_rgb)

results_table = []

#YOLOv5 inference
print("Running YOLOv5 inference...")
yolo5 = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
start = time.time()
results = yolo5(img_rgb)
end = time.time()
results.save(save_dir='yolo5_output')
results_table.append(['YOLOv5s', round(end - start, 4)])

# YOLOv8 inference
print("Running YOLOv8 inference...")
start = time.time()
yolo8_res = yolo8.predict(source=sample_image_path, save=True, save_txt=True, project='yolo8_output', name='result')
end = time.time()
results_table.append(['YOLOv8s', round(end - start, 4)])

# Faster R-CNN inference
print("Running Faster R-CNN inference...")
rcnn = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
rcnn.eval()
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
img_tensor = transform(img_rgb)
start = time.time()
with torch.no_grad():
    pred = rcnn([img_tensor])[0]
end = time.time()
for box, score in zip(pred['boxes'], pred['scores']):
    if score > 0.5:
        x1, y1, x2, y2 = box.int().tolist()
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
cv2.imwrite('rcnn_output.jpg', img)
results_table.append(['Faster R-CNN', round(end - start, 4)])

# ------------------ PART 4: COMPARISON ------------------
df_compare = pd.DataFrame(results_table, columns=['Model', 'Inference Time (s)'])
print("\nModel Inference Time Comparison:")
print(df_compare.to_string(index=False))
df_compare.to_csv("model_comparison.csv", index=False)

print("\nEDA plots, training logs, and inference outputs saved.")
