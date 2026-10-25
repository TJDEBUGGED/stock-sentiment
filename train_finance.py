"""
train_finance.py — Train a 3-class (negative / neutral / positive) finance sentiment model.

Data comes from data/finance_data.py (Twitter Financial News Sentiment + Financial PhraseBank).
The eval/ benchmark headlines are NEVER used here, so the benchmark stays honest.
Run from the project root:  python train_finance.py [--twitter-only] [--out model_dir]
"""

import pickle
import sys
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import FeatureUnion

from data.finance_data import load_all


def build_vectorizer(char_ngrams: bool = False):
    """Word 1-2grams; optionally also character 3-5grams (robust to wording/typos)."""
    words = TfidfVectorizer(max_features=50_000, ngram_range=(1, 2), sublinear_tf=True, min_df=2)
    if not char_ngrams:
        return words
    chars = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=3, max_features=100_000)
    return FeatureUnion([("words", words), ("chars", chars)])


def build_model(C: float = 5.0):
    # class_weight="balanced": neutral is the biggest class, so without this the model
    # drifts toward always guessing neutral.
    return LogisticRegression(max_iter=2000, C=C, class_weight="balanced")


def fit(train_x, train_y, char_ngrams: bool = False, C: float = 5.0, sample_weight=None):
    vectorizer = build_vectorizer(char_ngrams)
    model = build_model(C)
    model.fit(vectorizer.fit_transform(train_x), train_y, sample_weight=sample_weight)
    return vectorizer, model


# Settings of the model that ships. Change these (and re-run) to ship a different one.
CHAR_NGRAMS = False
C = 5.0


def main():
    use_phrasebank = "--twitter-only" not in sys.argv
    model_dir = Path(sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "model")

    data = load_all(use_phrasebank)
    print(f"Train: {len(data['train_x']):,}")
    vectorizer, model = fit(data["train_x"], data["train_y"], CHAR_NGRAMS, C)

    print()
    for name, (xs, ys) in data["tests"].items():
        preds = model.predict(vectorizer.transform(xs))
        print(f"{name}: accuracy {accuracy_score(ys, preds):.1%}, macro-F1 {f1_score(ys, preds, average='macro'):.3f}")
    xs, ys = data["tests"]["Twitter test split"]
    print()
    print(classification_report(ys, model.predict(vectorizer.transform(xs))))

    model_dir.mkdir(exist_ok=True)
    with open(model_dir / "vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    with open(model_dir / "classifier.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Saved", model_dir, "(classes:", list(model.classes_), ")")


if __name__ == "__main__":
    main()
