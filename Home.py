import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

# Load your trained YOLO model
model = YOLO(".\\yolo_project\\exp1\\weights\\best.pt")  # Replace with your model path

st.set_page_config(page_title="YOLOv8 Detection App", layout="centered")
st.title("🔍 YOLOv8 Object Detection")

# Sidebar mode selection
mode = st.sidebar.selectbox("Choose Mode", ["📁 Upload Image", "📷 Live Camera"])

# ------------------------- Mode 1: Upload Image -------------------------
if mode == "📁 Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Convert image to OpenCV format
        image_cv = np.array(image)
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

        # Run YOLO inference
        results = model.predict(source=image_cv, conf=0.3)
        annotated_image = results[0].plot()
        annotated_image=cv2.cvtColor(annotated_image,cv2.COLOR_BGR2RGB)

        # Show result
        st.image(annotated_image, caption="Detected Objects", use_container_width=True)

# ------------------------- Mode 2: Live Camera -------------------------
elif mode == "📷 Live Camera":

    class YOLOVideoProcessor(VideoProcessorBase):
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            results = model.predict(img, conf=0.3)
            annotated = results[0].plot()
            return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    webrtc_streamer(
        key="yolo-camera",
        video_processor_factory=YOLOVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
