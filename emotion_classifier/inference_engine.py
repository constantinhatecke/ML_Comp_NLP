import torch
import pickle
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
from emotion_classifier.utils import download_model


def load_model_and_tokenizer():
    model_path = Path("emotion_model.pt")
    label_path = Path("label_encoder.pkl")

    if not model_path.exists() or not label_path.exists():
        download_model()

    model = AutoModelForSequenceClassification.from_pretrained("vinai/bertweet-base", num_labels=6)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")

    with open(label_path, "rb") as f:
        label_encoder = pickle.load(f)

    return model, tokenizer, label_encoder

def predict_label(text, model, tokenizer, label_encoder):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    pred = torch.argmax(outputs.logits, dim=1).item()
    return label_encoder.inverse_transform([pred])[0]
