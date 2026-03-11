import os
import tempfile

import cv2
import numpy as np

from utils.ensemble_predictor import DeepfakeDetector


detector = DeepfakeDetector()


def analyze_video(video_path, sample_every_n_frames=None):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return {"error": "Could not open video"}

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = int(fps) if fps and fps > 0 else 0
    frame_interval = sample_every_n_frames or max(fps, 1)

    frame_count = 0
    analyzed_frames = 0

    fake_scores = []
    real_scores = []
    timeline = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            temp_path = None

            try:
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                    temp_path = tmp.name

                cv2.imwrite(temp_path, frame)
                result = detector.predict(temp_path, apply_image_heuristics=True)
            finally:
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)

            analyzed_frames += 1

            fake_prob = float(result["fake_probability"])
            real_prob = float(result["real_probability"])

            fake_scores.append(fake_prob)
            real_scores.append(real_prob)
            timeline.append({
                "frame_index": frame_count,
                "prediction": result["prediction"],
                "fake_probability": fake_prob,
                "real_probability": real_prob,
            })

        frame_count += 1

    cap.release()

    if analyzed_frames == 0:
        return {"error": "No frames processed"}

    avg_fake = float(np.mean(fake_scores))
    avg_real = float(np.mean(real_scores))
    suspicious_frames = sum(1 for item in timeline if item["prediction"] == "FAKE")
    fake_ratio = suspicious_frames / analyzed_frames
    real_ratio = 1 - fake_ratio

    if fake_ratio >= 0.5:
        final_prediction = "FAKE"
        confidence = fake_ratio * 100
    else:
        final_prediction = "REAL"
        confidence = real_ratio * 100

    return {
        "prediction": final_prediction,
        "confidence_percent": round(confidence, 2),
        "fake_probability": round(avg_fake, 2),
        "real_probability": round(avg_real, 2),
        "frames_analyzed": analyzed_frames,
        "suspicious_frames": suspicious_frames,
        "timeline": timeline,
    }


class VideoDeepfakeDetector:
    """Backward-compatible wrapper for older app/test imports."""

    def __init__(self, image_detector=None):
        self.detector = image_detector or detector

    def analyze_video(self, video_path, sample_every_n_frames=None):
        global detector

        previous_detector = detector
        detector = self.detector

        try:
            return analyze_video(video_path, sample_every_n_frames=sample_every_n_frames)
        finally:
            detector = previous_detector
