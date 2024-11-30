import streamlit as st
import torch
import cv2
from PIL import Image

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

st.title("Object Detection with YOLOv5")

uploaded_file = st.file_uploader("Upload an Image or Video", type=["jpg", "jpeg", "png", "mp4", "avi"])

if uploaded_file:
    file_extension = uploaded_file.name.split('.')[-1].lower()
    input_file_path = f"uploaded_{uploaded_file.name}"
    with open(input_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if file_extension in ["jpg", "jpeg", "png"]:
        st.image(input_file_path, caption="Uploaded Image", use_column_width=True)
        results = model(input_file_path)
        annotated_image = results.render()[0]
        annotated_image = Image.fromarray(annotated_image)
        output_image_path = f"annotated_{uploaded_file.name}"
        annotated_image.save(output_image_path)
        st.image(annotated_image, caption="Annotated Image", use_column_width=True)
        with open(output_image_path, "rb") as f:
            st.download_button(label="Download Annotated Image", data=f, file_name=f"annotated_{uploaded_file.name}", mime="image/jpeg")

    elif file_extension in ["mp4", "avi"]:
        st.video(uploaded_file, format="video/mp4", start_time=0)
        cap = cv2.VideoCapture(input_file_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        output_video_path = f"annotated_{uploaded_file.name}"
        out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (frame_width, frame_height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame)
            annotated_frame = results.render()[0]
            out.write(annotated_frame)

        cap.release()
        out.release()
        with open(output_video_path, "rb") as f:
            st.download_button(label="Download Annotated Video", data=f, file_name=f"annotated_{uploaded_file.name}", mime="video/mp4")
