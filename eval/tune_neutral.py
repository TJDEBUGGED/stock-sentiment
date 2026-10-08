"""
tune_neutral.py — Does down-weighting the "neutral" class improve label recall, honestly?

The model under-calls positive/negative headlines (it plays safe and says neutral).
A cheap fix is to multiply P(neutral) by a weight w < 1 before taking the arg-max.

To avoid tuning on the same data we report on, this uses a nested split:
for each of 20 random splits, pick the best w on one half (selecting by macro-F1),
then score it on the other, unseen half. The reported gain is averaged over the test halves.

Run from the project root:  python -m eval.tune_neutral
"""

import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent))
from analysis.sentiment import SentimentScorer  # noqa: E402

HERE = Path(__file__).parent
WEIGHTS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]


def read(name):
    with open(HERE / name, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def predict(probs, classes, w):
    """probs: (n, 3) array in `classes` order. Scale the neutral column by w, take arg-max."""
    scaled = probs.copy()
    scaled[:, classes.index("neutral")] *= w
    return np.array(classes)[scaled.argmax(axis=1)]


def main():
    labels = {r["id"]: r["label"] for r in read("labels.csv")}
    rows = [r for r in read("headlines_raw.csv") if r["id"] in labels]
    scorer = SentimentScorer()
    classes = [str(c) for c in scorer.model.classes_]
    probs = scorer.model.predict_proba(scorer.vectorizer.transform([r["title"] for r in rows]))
    y = np.array([labels[r["id"]] for r in rows])

    print("Whole-benchmark view (descriptive only, NOT a fair estimate because w is judged on the same data):")
    for w in WEIGHTS:
        p = predict(probs, classes, w)
        print(f"  w={w:.1f}  accuracy {accuracy_score(y, p):.1%}  macro-F1 {f1_score(y, p, average='macro'):.3f}")

    base, tuned, chosen = [], [], []
    base_acc, tuned_acc = [], []
    for seed in range(20):
        idx_dev, idx_test = train_test_split(np.arange(len(y)), test_size=0.5, random_state=seed, stratify=y)
        best_w = max(WEIGHTS, key=lambda w: f1_score(y[idx_dev], predict(probs[idx_dev], classes, w), average="macro"))
        chosen.append(best_w)
        for w, f1s, accs in ((1.0, base, base_acc), (best_w, tuned, tuned_acc)):
            p = predict(probs[idx_test], classes, w)
            f1s.append(f1_score(y[idx_test], p, average="macro"))
            accs.append(accuracy_score(y[idx_test], p))

    print("\nNested estimate over 20 random dev/test splits (score on the unseen half):")
    print(f"  w=1.0 (current)   macro-F1 {np.mean(base):.3f} +/- {np.std(base):.3f}   accuracy {np.mean(base_acc):.1%}")
    print(f"  w chosen on dev   macro-F1 {np.mean(tuned):.3f} +/- {np.std(tuned):.3f}   accuracy {np.mean(tuned_acc):.1%}")
    print(f"  chosen weights: {sorted(chosen)}  (median {np.median(chosen):.2f})")
    wins = sum(t > b for t, b in zip(tuned, base))
    print(f"  tuned beat current on {wins}/20 test halves")


if __name__ == "__main__":
    main()
