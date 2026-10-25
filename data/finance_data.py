"""
finance_data.py — Download and load the finance sentiment training data.

  - Twitter Financial News Sentiment (zeroshot/twitter-financial-news-sentiment).
    Labels are 0=bearish, 1=bullish, 2=neutral. Its validation split is our test set.
  - Financial PhraseBank, 75%-agreement version (non-commercial licence, CC BY-NC-SA).
    80% can be added to training; the other 20% is a second, unseen test set.

Files are downloaded into data/raw/ on first use (that folder is not committed).
Run scripts from the project root.
"""

import csv
import re
import urllib.request
import zipfile
from pathlib import Path

from sklearn.model_selection import train_test_split

RAW = Path("data/raw")

# Dataset label -> our label.
LABEL_MAP = {"0": "negative", "1": "positive", "2": "neutral"}

URL_RE = re.compile(r"https?://\S+")
CASHTAG_RE = re.compile(r"\$[A-Za-z]{1,6}\b")

TWITTER_URL = "https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment/resolve/main"
PHRASEBANK_URL = "https://huggingface.co/datasets/takala/financial_phrasebank/resolve/main/data/FinancialPhraseBank-v1.0.zip"


def clean(text: str) -> str:
    """Remove tweet-only noise (links, $TICKER tags) that real headlines don't have."""
    text = URL_RE.sub(" ", text)
    text = CASHTAG_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def download_data() -> None:
    """Fetch the datasets on first run so training is reproducible."""
    RAW.mkdir(parents=True, exist_ok=True)
    for name in ("sent_train.csv", "sent_valid.csv"):
        if not (RAW / name).exists():
            print(f"Downloading {name}...")
            urllib.request.urlretrieve(f"{TWITTER_URL}/{name}", RAW / name)
    if not (RAW / "pb").exists():
        print("Downloading Financial PhraseBank...")
        zip_path = RAW / "pb.zip"
        urllib.request.urlretrieve(PHRASEBANK_URL, zip_path)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(RAW / "pb")
        zip_path.unlink()


def _load_twitter(name: str):
    with open(RAW / name, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return [clean(r["text"]) for r in rows], [LABEL_MAP[r["label"]] for r in rows]


def _load_phrasebank():
    """PhraseBank split 80/20 with a fixed seed, so every model sees the same unseen 20%."""
    path = RAW / "pb" / "FinancialPhraseBank-v1.0" / "Sentences_75Agree.txt"
    lines = path.read_text(encoding="latin-1").splitlines()
    pairs = [l.rsplit("@", 1) for l in lines if "@" in l]
    texts, labels = [clean(t) for t, _ in pairs], [lab for _, lab in pairs]
    return train_test_split(texts, labels, test_size=0.2, random_state=42, stratify=labels)


def load_all(use_phrasebank: bool = True) -> dict:
    """Return train data plus both held-out test sets."""
    download_data()
    train_x, train_y = _load_twitter("sent_train.csv")
    tw_test_x, tw_test_y = _load_twitter("sent_valid.csv")
    pb_train_x, pb_test_x, pb_train_y, pb_test_y = _load_phrasebank()
    if use_phrasebank:
        train_x, train_y = train_x + pb_train_x, train_y + pb_train_y
    return {
        "train_x": train_x, "train_y": train_y,
        "tests": {
            "Twitter test split": (tw_test_x, tw_test_y),
            "PhraseBank held-out": (pb_test_x, pb_test_y),
        },
    }
