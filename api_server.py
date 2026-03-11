from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import shutil
import os

from utils.ensemble_predictor import DeepfakeDetector
from video_detector import analyze_video

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

detector = DeepfakeDetector()

UPLOAD_FOLDER = "temp"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.mount("/temp", StaticFiles(directory="."), name="temp")


@app.post("/detect-image")
async def detect_image(file: UploadFile = File(...)):

    temp_path = f"{UPLOAD_FOLDER}/{file.filename}"

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = detector.predict(temp_path)

    return result


@app.post("/detect-video")
async def detect_video(file: UploadFile = File(...)):

    temp_path = f"{UPLOAD_FOLDER}/{file.filename}"

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = analyze_video(temp_path)

    return result