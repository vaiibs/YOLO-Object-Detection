import streamlit as st
import torch
import os
from PIL import Image
import cv2
import numpy as np

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Pre-trained YOLOv5 model

# App title
st.title("Object Detection with YOLOv5")

# File uploader
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Save the uploaded file temporarily
    input_image_path = f"uploaded_{uploaded_file.name}"
    with open(input_image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Display the uploaded image
    st.image(input_image_path, caption="Uploaded Image", use_column_width=True)
    st.write("Performing object detection...")

    # Perform object detection
    results = model(input_image_path)

    # Convert the annotated results (NumPy array) to a displayable image
    annotated_image = results.render()[0]  # YOLOv5 annotates directly on the image
    annotated_image = Image.fromarray(annotated_image)  # Convert to PIL format

    # Save the annotated image for download
    output_image_path = f"annotated_{uploaded_file.name}"
    annotated_image.save(output_image_path)

    # Display the annotated image
    st.image(annotated_image, caption="Annotated Image", use_column_width=True)

    # Provide download link for the annotated image
    with open(output_image_path, "rb") as f:
        st.download_button(
            label="Download Annotated Image",
            data=f,
            file_name=f"annotated_{uploaded_file.name}",
            mime="image/jpeg"
        )
