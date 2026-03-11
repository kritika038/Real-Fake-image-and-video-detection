import gradio as gr
from utils.ensemble_predictor import EnsembleDetector
from video_detector import VideoDeepfakeDetector

print("Loading models...")

image_detector = EnsembleDetector()
video_detector = VideoDeepfakeDetector()

print("Models loaded")


def detect_image(image):

    result = image_detector.predict(image)

    return result


def detect_video(video):

    result = video_detector.analyze_video(video)

    return result


with gr.Blocks() as demo:

    gr.Markdown("# Deepfake Detection System")

    with gr.Tab("Image Detection"):

        image_input = gr.Image(type="filepath")

        image_output = gr.JSON()

        image_button = gr.Button("Detect")

        image_button.click(
            detect_image,
            inputs=image_input,
            outputs=image_output
        )

    with gr.Tab("Video Detection"):

        video_input = gr.Video()

        video_output = gr.JSON()

        video_button.click(
            detect_video,
            inputs=video_input,
            outputs=video_output
        )


demo.launch()