# AI Media Detector

This project classifies uploaded images and videos as `FAKE` or `REAL`.

## What this repo contains

- `app.py`: main Streamlit interface
- `launch.py`: one-command local launcher
- `utils/ensemble_predictor.py`: image detection logic
- `video_detector.py`: video frame-by-frame detection logic
- `models/baseline_model.py`: EfficientNet-B0 model definition
- `models/efficientnet_model.pth`: trained model weights used by the app

## Features

- Image detection
- Video detection
- Streamlit UI
- FastAPI backend in `api_server.py`

## Quick Start

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd agentic_ai_detector_v2
```

### 2. Create a virtual environment

Mac/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

Option 1:

```bash
python3 launch.py
```

Option 2:

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

## Run the API instead

```bash
uvicorn api_server:app --reload
```

Usually available at:

```text
http://127.0.0.1:8000
```

## Notes

- The current app uses `models/efficientnet_model.pth`.
- Video detection works by extracting frames and analyzing each frame as an image.
- Some image predictions also use heuristic checks like sharpness and EXIF metadata.

## Important repo notes

The `.gitignore` already excludes large local-only folders and files such as:

- `data/`
- `data_fast/`
- `venv/`
- temp folders
- unused large artifacts

So someone downloading the repo gets only the code and required model weights for running the app.

## Deploy on Streamlit Community Cloud

1. Push this project to GitHub.
2. Go to `https://share.streamlit.io/`
3. Click `New app`
4. Select your GitHub repo
5. Set the main file path to `app.py`
6. Deploy

After deployment, Streamlit gives a public link that opens the running interface directly.
