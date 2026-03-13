# AI Media Detector

This project classifies uploaded images and videos as `FAKE` or `REAL`.

## Links

- GitHub Repository: `https://github.com/kritika038/Real-Fake-image-and-video-detection`
- Website: `https://kritika038.github.io/Real-Fake-image-and-video-detection/`
- Live App: `https://real-fake-image-and-video-detection-e7pum2tuwtmhjhsk3yfpwe.streamlit.app/`
- Website Source: `docs/index.html` for free GitHub Pages hosting

If the Streamlit live app shows a sleep screen, click `Yes, get this app back up!` and wait a few moments for it to wake up.

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
git clone https://github.com/kritika038/Real-Fake-image-and-video-detection.git
cd Real-Fake-image-and-video-detection
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

## Create a real public website for free

This repo now includes a static website in `docs/` so you can publish a proper homepage with GitHub Pages.

### Why this helps

- The website can be indexed by Google
- Visitors get a professional landing page first
- The page links to your live Streamlit detector
- GitHub Pages hosting is free

### Deploy the website with GitHub Pages

1. Push this repo to GitHub.
2. Open the repository on GitHub.
3. Go to `Settings` -> `Pages`.
4. Under `Build and deployment`, choose:
   - `Source`: `Deploy from a branch`
   - `Branch`: `main`
   - `Folder`: `/docs`
5. Save.

GitHub will publish the site at a URL similar to:

```text
https://kritika038.github.io/Real-Fake-image-and-video-detection/
```

### Exact git commands

```bash
git add docs/index.html docs/styles.css docs/robots.txt docs/sitemap.xml README.md
git commit -m "Add public website landing page"
git push origin main
```

### Important note

- The website in `docs/` stays awake and public.
- Your Streamlit detector may still sleep on the free tier after inactivity.
- This setup is still useful because people can find your website on Google and then click into the detector.
