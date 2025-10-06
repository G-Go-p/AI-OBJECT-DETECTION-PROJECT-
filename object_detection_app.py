
# object_detection_app.py

import streamlit as st
import torch
import cv2
import tempfile
import os
from PIL import Image
import numpy as np
from ultralytics import YOLO
import torchvision
from torchvision.transforms import functional as F

st.set_page_config(page_title="Object Detection Demo", layout="centered")
st.title("📦 Object Detection App (YOLOv5, YOLOv8, Faster R-CNN)")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Select model
model_choice = st.selectbox("Select Detection Model", ["YOLOv5s", "YOLOv8s", "Faster R-CNN"])

if uploaded_file is not None:
    # Read image
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    # Temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    image.save(temp_file.name)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Run Detection"):
        with st.spinner("Running model..."):
            if model_choice == "YOLOv5s":
                model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
                results = model(img_array)
                results.render()
                st.image(results.ims[0], caption="YOLOv5s Detection", use_column_width=True)

            elif model_choice == "YOLOv8s":
                model = YOLO("yolov8s.pt")
                results = model.predict(source=temp_file.name, save=False)
                res_img = results[0].plot()
                st.image(res_img, caption="YOLOv8s Detection", use_column_width=True)

            elif model_choice == "Faster R-CNN":
                model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
                model.eval()
                img_tensor = F.to_tensor(image)
                with torch.no_grad():
                    pred = model([img_tensor])[0]
                for box, score in zip(pred['boxes'], pred['scores']):
                    if score > 0.5:
                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(img_array, (x1, y1), (x2, y2), (0, 255, 0), 2)
                st.image(img_array, caption="Faster R-CNN Detection", use_column_width=True)

        os.unlink(temp_file.name)
