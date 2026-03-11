import os
from utils.ensemble_predictor import EnsembleDetector

detector = EnsembleDetector()

folder = "test_images"

for img in os.listdir(folder):

    path = os.path.join(folder, img)

    result = detector.predict(path)

    print(img, "->", result["prediction"], result["confidence_percent"], "%")
    
