import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
nltk.download("punkt")  
nltk.download("stopwords")

def flatten_1d(x):
    return x.ravel()


def custom_text_tokenizer(text):
    stop_words = set(stopwords.words("english"))
    stem = PorterStemmer()
    text = text.lower().strip()
    text = re.sub(r"n['’]t", " not", text)
    text = re.sub(r"['’]re", " are", text)
    text = re.sub(r"['’]s", " is", text)
    text = re.sub(r"['’]ve", " have", text)
    text = re.sub(r"['’]m", " am", text)
    text = re.sub(r"['’]ll", " will", text)
    text = re.sub(r"['’]d", " would", text)
    text = re.sub(r"(\d)([a-zA-Z])", r"\1 \2", text)
    text = re.sub(r"[!?\-_.,;()\[\]{}]", " ", text)
    text = re.sub(r"\s+", " ", text)
    tokens = text.split()
    tokens = [stem.stem(word.strip()) for word in tokens if word not in stop_words]
    return tokens
