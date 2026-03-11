import os
import tempfile

from PIL import Image
import streamlit as st

from utils.ensemble_predictor import EnsembleDetector
from video_detector import analyze_video


st.set_page_config(page_title="AI Detector", layout="centered")


@st.cache_resource
def load_detector():
    return EnsembleDetector()


detector = load_detector()

st.title("AI Media Detector")
st.write("Upload an image or video to classify it as fake or real.")

image_tab, video_tab = st.tabs(["Image Detection", "Video Detection"])

with image_tab:
    uploaded_image = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"],
        key="image_uploader",
    )

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("Run Image Detection"):
            temp_path = None

            with st.spinner("Analyzing image..."):
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                        image.save(tmp.name)
                        temp_path = tmp.name

                    result = detector.predict(temp_path)
                finally:
                    if temp_path and os.path.exists(temp_path):
                        os.remove(temp_path)

            st.success("Detection Complete")
            st.json(result)

with video_tab:
    uploaded_video = st.file_uploader(
        "Choose a video",
        type=["mp4", "mov", "avi", "mkv"],
        key="video_uploader",
    )

    if uploaded_video is not None:
        st.video(uploaded_video)

        if st.button("Run Video Detection"):
            temp_path = None

            with st.spinner("Analyzing video..."):
                try:
                    suffix = os.path.splitext(uploaded_video.name)[1] or ".mp4"
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(uploaded_video.getbuffer())
                        temp_path = tmp.name

                    result = analyze_video(temp_path)
                finally:
                    if temp_path and os.path.exists(temp_path):
                        os.remove(temp_path)

            st.success("Detection Complete")
            st.json(result)
