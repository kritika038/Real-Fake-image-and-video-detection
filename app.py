import os
import tempfile

from PIL import Image
import streamlit as st
import pandas as pd

from utils.ensemble_predictor import EnsembleDetector
from video_detector import analyze_video


st.set_page_config(page_title="AI Detector", layout="centered")


@st.cache_resource
def load_detector():
    return EnsembleDetector()


detector = load_detector()

st.title("AI Media Detector")
st.write("Upload an image or video to classify it as fake or real.")

mode_label = st.selectbox(
    "Image detection mode",
    options=["Balanced", "Strict Fake Detection"],
    index=1,
    help=(
        "Balanced is safer for real photos. Strict Fake Detection is more "
        "aggressive for suspicious images and is the current default."
    ),
)

image_detection_mode = {
    "Balanced": "balanced",
    "Strict Fake Detection": "strict_fake_detection",
}[mode_label]


def render_result_card(result, media_type):
    prediction = result["prediction"]
    confidence = result["confidence_percent"]
    fake_probability = result["fake_probability"]
    real_probability = result["real_probability"]

    if prediction == "REAL":
        badge_color = "#1f7a1f"
        label = "Likely Real"
    else:
        badge_color = "#b42318"
        label = "Likely Fake"

    st.markdown(
        f"""
        <div style="padding:20px;border-radius:16px;background:#f7f9fc;border:1px solid #dbe4f0;margin-top:12px;">
            <div style="font-size:14px;color:#4a5568;">{media_type} Analysis Result</div>
            <div style="display:inline-block;margin-top:10px;padding:8px 14px;border-radius:999px;background:{badge_color};color:white;font-weight:600;">
                {label}
            </div>
            <div style="font-size:32px;font-weight:700;margin-top:16px;color:#101828;">{prediction}</div>
            <div style="font-size:16px;color:#475467;margin-top:4px;">
                Confidence: {confidence}%
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    col1.metric("Fake Probability", f"{fake_probability}%")
    col2.metric("Real Probability", f"{real_probability}%")


def render_image_result(result):
    render_result_card(result, "Image")

    st.caption(f"Mode: {result.get('detection_mode', 'balanced')}")

    extra_cols = st.columns(2)
    if "sharpness_score" in result:
        extra_cols[0].metric("Sharpness Score", result["sharpness_score"])
    if "exif_entries" in result:
        extra_cols[1].metric("EXIF Entries", result["exif_entries"])

    if "heuristic_override" in result:
        st.info(f"Fallback rule used: {result['heuristic_override']}")


def render_video_result(result):
    render_result_card(result, "Video")

    col1, col2 = st.columns(2)
    col1.metric("Frames Analyzed", result["frames_analyzed"])
    col2.metric("Suspicious Frames", result["suspicious_frames"])

    timeline = result.get("timeline", [])
    if timeline:
        with st.expander("Frame-by-frame analysis"):
            frame_df = pd.DataFrame(timeline)
            st.dataframe(frame_df, use_container_width=True, hide_index=True)

image_tab, video_tab = st.tabs(["Image Detection", "Video Detection"])

with image_tab:
    uploaded_image = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"],
        key="image_uploader",
    )

    if uploaded_image is not None:
        image_bytes = uploaded_image.getvalue()
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("Run Image Detection"):
            temp_path = None

            with st.spinner("Analyzing image..."):
                try:
                    suffix = os.path.splitext(uploaded_image.name)[1] or ".jpg"
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(image_bytes)
                        temp_path = tmp.name

                    try:
                        result = detector.predict(
                            temp_path,
                            detection_mode=image_detection_mode,
                        )
                    except TypeError:
                        # Backward compatibility for deployments still loading
                        # an older detector signature during refresh.
                        result = detector.predict(temp_path)
                finally:
                    if temp_path and os.path.exists(temp_path):
                        os.remove(temp_path)

            st.success("Detection Complete")
            render_image_result(result)

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
            render_video_result(result)
