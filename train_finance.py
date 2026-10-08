"""
train_finance.py — Train a 3-class (negative / neutral / positive) finance sentiment model.

Data (downloaded into data/raw/ on first run):
  - Twitter Financial News Sentiment (zeroshot/twitter-financial-news-sentiment).
    Labels are 0=bearish, 1=bullish, 2=neutral. Its validation split is our test set.
  - Financial PhraseBank, 75%-agreement version (non-commercial licence, CC BY-NC-SA).
    80% is added to training; the other 20% is a second, unseen test set.

The eval/ benchmark headlines are NEVER used here, so the benchmark stays honest.
Run from the project root:  python train_finance.py [--twitter-only] [--out model_dir]
"""

import csv
import pickle
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

USE_PHRASEBANK = "--twitter-only" not in sys.argv
OUT_ARG = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "model"

RAW = Path("data/raw")
MODEL_DIR = Path(OUT_ARG)

# Dataset label -> our label. Order matters: classes are sorted alphabetically by sklearn.
LABEL_MAP = {"0": "negative", "1": "positive", "2": "neutral"}

URL_RE = re.compile(r"https?://\S+")
CASHTAG_RE = re.compile(r"\$[A-Za-z]{1,6}\b")


def clean(text: str) -> str:
    """Remove tweet-only noise (links, $TICKER tags) that real headlines don't have."""
    text = URL_RE.sub(" ", text)
    text = CASHTAG_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


TWITTER_URL = "https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment/resolve/main"
PHRASEBANK_URL = "https://huggingface.co/datasets/takala/financial_phrasebank/resolve/main/data/FinancialPhraseBank-v1.0.zip"


def download_data():
    """Fetch the datasets on first run so the script is reproducible (data/raw/ is not committed)."""
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


def load(name):
    with open(RAW / name, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return [clean(r["text"]) for r in rows], [LABEL_MAP[r["label"]] for r in rows]


def load_phrasebank():
    """Financial PhraseBank (75% annotator agreement), split 80/20 so 20% stays unseen."""
    path = RAW / "pb" / "FinancialPhraseBank-v1.0" / "Sentences_75Agree.txt"
    lines = path.read_text(encoding="latin-1").splitlines()
    pairs = [l.rsplit("@", 1) for l in lines if "@" in l]
    texts, labels = [clean(t) for t, _ in pairs], [lab for _, lab in pairs]
    return train_test_split(texts, labels, test_size=0.2, random_state=42, stratify=labels)


download_data()
train_x, train_y = load("sent_train.csv")
test_x, test_y = load("sent_valid.csv")

# Always split PhraseBank the same way, so every model is scored on the same unseen 20%.
pb_train_x, pb_test_x, pb_train_y, pb_test_y = load_phrasebank()
if USE_PHRASEBANK:
    train_x, train_y = train_x + pb_train_x, train_y + pb_train_y
print(f"Train: {len(train_x):,}  Twitter test: {len(test_x):,}"
      + (f"  PhraseBank test: {len(pb_test_x):,}" if pb_test_x else ""))

vectorizer = TfidfVectorizer(max_features=50_000, ngram_range=(1, 2), sublinear_tf=True, min_df=2)
X_train = vectorizer.fit_transform(train_x)
X_test = vectorizer.transform(test_x)

# class_weight="balanced": neutral is the biggest class, so without this the model
# drifts toward always guessing neutral.
model = LogisticRegression(max_iter=2000, C=5.0, class_weight="balanced")
model.fit(X_train, train_y)

def report(name, xs, ys):
    preds = model.predict(vectorizer.transform(xs))
    print(f"{name}: accuracy {accuracy_score(ys, preds):.1%}, macro-F1 {f1_score(ys, preds, average='macro'):.3f}")


print()
report("Twitter test split", test_x, test_y)
if pb_test_x:
    report("PhraseBank held-out", pb_test_x, pb_test_y)
print()
print(classification_report(test_y, model.predict(X_test)))

MODEL_DIR.mkdir(exist_ok=True)
with open(MODEL_DIR / "vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
with open(MODEL_DIR / "classifier.pkl", "wb") as f:
    pickle.dump(model, f)
print("Saved model/ (classes:", list(model.classes_), ")")
