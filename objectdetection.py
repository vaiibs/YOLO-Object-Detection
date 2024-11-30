import streamlit as st
import torch
import os
from PIL import Image
import cv2
import numpy as np
from io import BytesIO

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Pre-trained YOLOv5 model

# App title
st.title("Object Detection with YOLOv5")

# File uploader (image or video)
uploaded_file = st.file_uploader("Upload an Image or Video", type=["jpg", "jpeg", "png", "mp4", "avi"])

if uploaded_file:
    # Check if the file is an image or video
    file_extension = uploaded_file.name.split('.')[-1].lower()

    # Save the uploaded file temporarily
    input_file_path = f"uploaded_{uploaded_file.name}"
    with open(input_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Handle image file
    if file_extension in ["jpg", "jpeg", "png"]:
        # Display the uploaded image
        st.image(input_file_path, caption="Uploaded Image", use_column_width=True)
        st.write("Performing object detection on image...")

        # Perform object detection on the image
        results = model(input_file_path)

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

    # Handle video file
    elif file_extension in ["mp4", "avi"]:
        # Display video
        st.video(uploaded_file, format="video/mp4", start_time=0)
        st.write("Performing object detection on video...")

        # Open the uploaded video file with OpenCV
        cap = cv2.VideoCapture(input_file_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        
        # Save the annotated video
        output_video_path = f"annotated_{uploaded_file.name}"
        out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (frame_width, frame_height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Perform object detection on the frame
            results = model(frame)

            # Render results on the frame
            annotated_frame = results.render()[0]  # YOLOv5 annotates directly on the frame

            # Write the annotated frame to the output video
            out.write(annotated_frame)

        # Release video objects
        cap.release()
        out.release()

        # Provide download link for the annotated video
        with open(output_video_path, "rb") as f:
            st.download_button(
                label="Download Annotated Video",
                data=f,
                file_name=f"annotated_{uploaded_file.name}",
                mime="video/mp4"
            )
