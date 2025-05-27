import argparse
import sys
from emotion_classifier.inference_engine import load_model_and_tokenizer, predict_label
from emotion_classifier.utils import get_kaggle_id

def main():
    parser = argparse.ArgumentParser(description="Emotion Inference CLI")
    parser.add_argument("--input", type=str, help="Input text for emotion prediction")
    parser.add_argument("--kaggle", action="store_true", help="Print Kaggle username")
    args = parser.parse_args()

    if args.kaggle:
        print(get_kaggle_id())
        sys.stdout.flush()
    elif args.input:
        model, tokenizer, label_encoder = load_model_and_tokenizer()
        prediction = predict_label(args.input, model, tokenizer, label_encoder)
        print(prediction)
        sys.stdout.flush()
    else:
        print("Please provide either --input or --kaggle.")
