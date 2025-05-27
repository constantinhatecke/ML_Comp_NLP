from pathlib import Path
import gdown

MODEL_URL = "https://drive.google.com/uc?id=1A8xh9HGVOvVB-yw-vFGe9bA8uqoLwNw8"

LABEL_URL = "https://drive.google.com/uc?id=1g3fJl4ia6qHYCnrCoYJxo20ArcaTbJ


def download_model():
    model_path = Path("emotion_model.pt")
    label_path = Path("label_encoder.pkl")

    if not model_path.exists():
        print("Downloading model...")
        gdown.download(MODEL_URL, str(model_path), quiet=False)

    if not label_path.exists():
        print("Downloading label encoder...")
        gdown.download(LABEL_URL, str(label_path), quiet=False)

def get_kaggle_id():
    return "constantinhatecke"
