from pathlib import Path
import gdown

MODEL_URL = "https://drive.google.com/file/d/1A8xh9HGVOvVB-yw-vFGe9bA8uqoLwNw8/view?usp=sharing"
LABEL_URL = "https://drive.google.com/file/d/1g3fJl4ia6qHYCnrCoYJxo20ArcaTbJD7/view?usp=sharing"


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
