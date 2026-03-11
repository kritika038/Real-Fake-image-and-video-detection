from utils.ensemble_predictor import EnsembleDetector
import sys

if len(sys.argv) < 2:
    print("Usage: python test_image.py <image_path>")
    exit()

image_path = sys.argv[1]

detector = EnsembleDetector()

result = detector.predict(image_path)

print("\nRESULT")
print("Prediction:", result["prediction"])
print("Confidence:", result["confidence_percent"], "%")

for k, v in result.items():
    if k not in ["prediction", "confidence_percent"]:
        print(k, ":", v)
