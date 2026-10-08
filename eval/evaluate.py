"""
evaluate.py — Score the benchmark headlines with the current model and compare to labels.csv.

Labels were written by Claude (not a human annotator) using one rule:
"is this news good, bad or neutral for the stock?" Treat results as indicative.
Run from the project root:  python -m eval.evaluate
"""

import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from analysis.sentiment import SentimentScorer  # noqa: E402

HERE = Path(__file__).parent
LABELS = ["negative", "neutral", "positive"]


def read(name):
    with open(HERE / name, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    labels = {r["id"]: r["label"] for r in read("labels.csv")}
    rows = [r for r in read("headlines_raw.csv") if r["id"] in labels]
    # Optional: python -m eval.evaluate model_candidate   (defaults to model/)
    scorer = SentimentScorer(Path(sys.argv[1])) if len(sys.argv) > 1 else SentimentScorer()

    pairs = [(labels[r["id"]], scorer.score(r["title"])["label"]) for r in rows]
    n = len(pairs)
    acc = sum(t == p for t, p in pairs) / n

    # Binary view: ignore neutral rows, ask only "did it get the direction right?"
    polar = [(t, p) for t, p in pairs if t != "neutral"]
    polar_acc = sum(t == p for t, p in polar) / len(polar)

    # Fairer direction check: only where truth is polar AND the model picked a side.
    sided = [(t, p) for t, p in polar if p != "neutral"]
    sided_acc = sum(t == p for t, p in sided) / len(sided)

    print(f"Headlines: {n}   label mix: {dict(Counter(t for t, _ in pairs))}")
    print(f"3-class accuracy: {acc:.1%}")
    print(f"Direction accuracy, counting model 'neutral' as wrong: {polar_acc:.1%} ({len(polar)} headlines)")
    print(f"Direction accuracy when model picked a side: {sided_acc:.1%} ({len(sided)} headlines; chance = 50%)")
    print("\nConfusion (rows = true label, columns = model):")
    print(f"{'':10}" + "".join(f"{l:>10}" for l in LABELS))
    for t in LABELS:
        print(f"{t:10}" + "".join(f"{sum(1 for a, b in pairs if a == t and b == p):>10}" for p in LABELS))


if __name__ == "__main__":
    main()
