import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

nltk.download("punkt")
nltk.download("stopwords")


def flatten_1d(x):
    return x.ravel()


def custom_text_tokenizer(text):
    stop_words = set(stopwords.words("english")) - {"not", "no"}
    stem = PorterStemmer()
    text = text.lower().strip()
    text = re.sub(r"n['’]t?", " not", text)
    text = re.sub(r"(\d)([a-zA-Z])", r"\1 \2", text)
    text = re.sub(r"([a-zA-Z])(\d)", r"\1 \2", text)
    text = re.sub(r"[!?\-_.,;()\[\]{}$&+%]", " ", text)
    text = re.sub(r"\s+", " ", text)
    tokens = [word.strip("'").strip('"').strip() for word in text.split()]
    tokens = [
        stem.stem(word) for word in tokens if word not in stop_words and len(word) > 1
    ]
    return tokens
