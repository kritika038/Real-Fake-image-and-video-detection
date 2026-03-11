from utils.ensemble_predictor import EnsembleDetector

detector = EnsembleDetector()

print("\nFAKE DATASET IMAGE:")
print(detector.predict("data/test/fake/0.jpg"))

print("\nREAL DATASET IMAGE:")
print(detector.predict("data/test/real/0000.jpg"))
