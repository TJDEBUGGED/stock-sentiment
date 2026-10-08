"""
sentiment.py — Loads the trained model and scores text.

The score is signed: P(positive) - P(negative), so it runs from -1 (very
negative) to +1 (very positive). A signed score is easy to average and plot.
"""

import pickle
from pathlib import Path

MODEL_DIR = Path(__file__).parent.parent / "model"


class SentimentScorer:
    def __init__(self, model_dir: Path = MODEL_DIR):
        with open(model_dir / "vectorizer.pkl", "rb") as f:
            self.vectorizer = pickle.load(f)
        with open(model_dir / "classifier.pkl", "rb") as f:
            self.model = pickle.load(f)

    def score(self, text: str) -> dict:
        X = self.vectorizer.transform([text])
        probs = dict(zip(self.model.classes_, self.model.predict_proba(X)[0]))
        signed = float(probs["positive"] - probs["negative"])

        # The model has a real neutral class, so the label is the most likely class.
        label = max(probs, key=probs.get)

        return {"score": round(signed, 4), "label": str(label)}
