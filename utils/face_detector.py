from facenet_pytorch import MTCNN
from PIL import Image
import torch


class FaceDetector:

    def __init__(self):

        # force CPU to avoid MPS bug
        self.mtcnn = MTCNN(keep_all=False, device="cpu")


    def detect_face(self,image_path):

        image = Image.open(image_path).convert("RGB")

        boxes, _ = self.mtcnn.detect(image)

        if boxes is None:
            return image

        x1,y1,x2,y2 = boxes[0]

        face=image.crop((x1,y1,x2,y2))

        return face