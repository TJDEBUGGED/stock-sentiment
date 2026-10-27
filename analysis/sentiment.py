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
        """Score one text. Returns only {score, label}: this is what gets stored per headline."""
        full = self.score_many([text])[0]
        return {"score": full["score"], "label": full["label"]}

    def score_many(self, texts: list[str]) -> list[dict]:
        """Score a batch in one vectorizer/model call. Also returns the class probabilities."""
        X = self.vectorizer.transform(texts)
        classes = [str(c) for c in self.model.classes_]
        results = []
        for row in self.model.predict_proba(X):
            probs = dict(zip(classes, (float(p) for p in row)))
            signed = probs["positive"] - probs["negative"]
            # The model has a real neutral class, so the label is the most likely class.
            label = max(probs, key=probs.get)
            results.append({
                "label": label,
                "score": round(signed, 4),
                "probabilities": {k: round(v, 4) for k, v in probs.items()},
            })
        return results
