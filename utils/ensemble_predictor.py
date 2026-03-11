from pathlib import Path

import cv2
import torch
from PIL import Image
from torchvision import transforms

from models.baseline_model import EfficientNetBinary


class DeepfakeDetector:
    """Single-model image detector.

    The training set is loaded with ImageFolder from data/train/{fake,real},
    so class index 0 maps to fake and class index 1 maps to real.
    """

    FAKE_CLASS_INDEX = 0
    REAL_CLASS_INDEX = 1
    LOW_SHARPNESS_THRESHOLD = 75.0
    STRICT_NO_EXIF_SHARPNESS_THRESHOLD = 120.0
    REAL_CAMERA_EXIF_THRESHOLD = 5
    REAL_CAMERA_SHARPNESS_THRESHOLD = 100.0
    DETECTION_MODES = ("balanced", "strict_fake_detection")

    def __init__(self, model_path=None):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.model_path = self._resolve_model_path(model_path)
        self.model = EfficientNetBinary().to(self.device)

        state = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def _resolve_model_path(self, model_path):
        if model_path is not None:
            candidate = Path(model_path)
            if candidate.exists():
                return str(candidate)
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Prefer the checkpoint saved by training/train_baseline.py.
        # A root-level best_model.pth may exist, but in this project it is not
        # the correct EfficientNet weights for the image/video detector.
        candidates = [
            Path("models/efficientnet_model.pth"),
            Path("best_model.pth"),
        ]

        for candidate in candidates:
            if candidate.exists():
                return str(candidate)

        raise FileNotFoundError(
            "No model weights found. Expected models/efficientnet_model.pth "
            "or best_model.pth."
        )

    def preprocess(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image)
        return image_tensor.unsqueeze(0).to(self.device)

    def sharpness_score(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            return None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def exif_entry_count(self, image_path):
        try:
            image = Image.open(image_path)
            exif = image.getexif()
            return len(exif) if exif else 0
        except Exception:
            return 0

    def predict(
        self,
        image_path,
        apply_image_heuristics=True,
        detection_mode="balanced",
    ):
        if detection_mode not in self.DETECTION_MODES:
            raise ValueError(
                f"Unsupported detection mode: {detection_mode}. "
                f"Expected one of {self.DETECTION_MODES}."
            )

        image_tensor = self.preprocess(image_path)

        with torch.no_grad():
            output = self.model(image_tensor)
            probs = torch.softmax(output, dim=1)[0]

        fake_prob = probs[self.FAKE_CLASS_INDEX].item()
        real_prob = probs[self.REAL_CLASS_INDEX].item()

        if fake_prob >= real_prob:
            prediction = "FAKE"
            confidence = fake_prob
        else:
            prediction = "REAL"
            confidence = real_prob

        result = {
            "prediction": prediction,
            "confidence_percent": round(confidence * 100, 2),
            "fake_probability": round(fake_prob * 100, 2),
            "real_probability": round(real_prob * 100, 2),
            "model_path": self.model_path,
            "detection_mode": detection_mode,
        }

        sharpness = self.sharpness_score(image_path)
        exif_entries = self.exif_entry_count(image_path)
        result["exif_entries"] = exif_entries

        if apply_image_heuristics and sharpness is not None:
            result["sharpness_score"] = round(sharpness, 2)
            if (
                prediction == "REAL"
                and sharpness < self.LOW_SHARPNESS_THRESHOLD
            ):
                result["prediction"] = "FAKE"
                result["confidence_percent"] = round(
                    max(confidence * 100, 75.0),
                    2,
                )
                result["heuristic_override"] = "low_sharpness"

        if (
            apply_image_heuristics
            and detection_mode == "strict_fake_detection"
            and result["prediction"] == "REAL"
            and exif_entries == 0
            and sharpness is not None
            and sharpness < self.STRICT_NO_EXIF_SHARPNESS_THRESHOLD
        ):
            result["prediction"] = "FAKE"
            result["confidence_percent"] = max(
                result["confidence_percent"],
                80.0,
            )
            result["heuristic_override"] = "no_exif_strict_mode"

        if (
            apply_image_heuristics
            and exif_entries >= self.REAL_CAMERA_EXIF_THRESHOLD
            and sharpness is not None
            and sharpness >= self.REAL_CAMERA_SHARPNESS_THRESHOLD
        ):
            result["prediction"] = "REAL"
            result["confidence_percent"] = max(
                result["confidence_percent"],
                85.0,
            )
            result["heuristic_override"] = "camera_exif_real"

        return result


class EnsembleDetector(DeepfakeDetector):
    """Backward-compatible alias for older app/test imports."""
