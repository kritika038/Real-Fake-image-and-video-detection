# AI Media Detector

This project detects whether an uploaded image or video is `FAKE` or `REAL`.

## Active app

The deployable interface is the Streamlit app in `app.py`.

It supports:
- image detection
- video detection

## Main files

- `app.py`: Streamlit UI
- `utils/ensemble_predictor.py`: image detector logic
- `video_detector.py`: video frame analysis
- `models/baseline_model.py`: EfficientNet-B0 model definition
- `models/efficientnet_model.pth`: trained model weights

## Local run

```bash
./venv/bin/streamlit run app.py
```

## GitHub upload

Do not upload local training data or virtualenv. `.gitignore` already excludes:
- `data/`
- `data_fast/`
- `venv/`
- temp folders

## Deploy on Streamlit Community Cloud

1. Push this project to a GitHub repository.
2. Go to `https://share.streamlit.io/`
3. Click `New app`
4. Select your GitHub repo
5. Set main file path to `app.py`
6. Deploy

After deploy, Streamlit will give you a live public link that opens the running interface directly.
