import sys
from video_detector import VideoDeepfakeDetector

video_path = sys.argv[1]

detector = VideoDeepfakeDetector()

result = detector.analyze_video(video_path)

print("\nVIDEO RESULT")
print(result)
