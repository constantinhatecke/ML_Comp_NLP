import torch
import pickle
from pathlib import Path
from emotion_classifier.utils import download_model
from emotion_classifier.model import MyModel
import torch.nn.functional as F

# Replace with your real vocab (or load it from file if saved)
vocab = {
    "<PAD>": 0,
    "<OOV>": 1,
    # Add more tokens if you saved vocab to disk
}

def load_model_and_tokenizer():
    model_path = Path("emotion_model.pt")
    label_path = Path("label_encoder.pkl")

    if not model_path.exists() or not label_path.exists():
        download_model()

    # You need to match these with training values
    embedding_dim = 100
    hidden_dim = 256
    output_dim = 6
    pad_idx = vocab["<PAD>"]

    # Dummy embedding matrix just for loading (won’t be used at inference)
    embedding_matrix = torch.randn(len(vocab), embedding_dim)

    model = MyModel(embedding_matrix, hidden_dim, output_dim, pad_idx)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()

    with open(label_path, "rb") as f:
        label_encoder = pickle.load(f)

    return model, vocab, label_encoder

def predict_label(text, model, vocab, label_encoder, max_length=200):
    from nltk.tokenize import TweetTokenizer
    tokenizer = TweetTokenizer()
    tokens = tokenizer.tokenize(text.lower())
    ids = [vocab.get(tok, vocab["<OOV>"]) for tok in tokens]

    if len(ids) < max_length:
        ids += [vocab["<PAD>"]] * (max_length - len(ids))
    else:
        ids = ids[:max_length]

    inputs = torch.tensor([ids])
    outputs = model(inputs)
    pred = torch.argmax(F.softmax(outputs, dim=1), dim=1).item()
    return label_encoder.inverse_transform([pred])[0]
