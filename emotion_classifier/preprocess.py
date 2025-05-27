import re
import torch
from collections import Counter
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')

STOPWORDS = set(stopwords.words('english'))
tokenizer = TweetTokenizer()

# Clean text: lowercase, remove URLs, mentions, hashtags, punctuation, stopwords
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = " ".join([w for w in text.split() if w not in STOPWORDS])
    return text

def clean_data(df, text_column='text'):
    df[text_column] = df[text_column].astype(str).apply(clean_text)
    return df

def remove_outliers(df, text_column='text', min_len=3, max_len=50):
    df['text_len'] = df[text_column].apply(lambda x: len(x.split()))
    df = df[(df['text_len'] >= min_len) & (df['text_len'] <= max_len)]
    return df.drop(columns=['text_len'])

def build_vocab(texts, max_vocab_size=10000):
    counter = Counter()
    for text in texts:
        counter.update(text.lower().split())
    most_common = counter.most_common(max_vocab_size - 2)
    vocab = {word: idx + 2 for idx, (word, _) in enumerate(most_common)}
    vocab['<PAD>'] = 0
    vocab['<OOV>'] = 1
    return vocab

def encode_text(texts, vocab, max_length=200):
    sequences = []
    for text in texts:
        tokens = tokenizer.tokenize(text.lower())
        seq = [vocab.get(tok, vocab['<OOV>']) for tok in tokens]
        if len(seq) < max_length:
            seq += [vocab['<PAD>']] * (max_length - len(seq))
        else:
            seq = seq[:max_length]
        sequences.append(seq)
    return torch.tensor(sequences)