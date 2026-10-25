"""
compare_indomain.py — Does training on real Yahoo-style headlines help?

The finance training data is tweets and earnings sentences, not news headlines. Here we add
387 extra real headlines (eval/indomain_raw.csv + indomain_labels.csv; none overlap the
benchmark) as extra training data, optionally up-weighted.

Honest procedure:
  1. Choose the weight by 5-fold cross-validation INSIDE the 387 headlines (benchmark not used).
  2. Train once with the chosen weight on finance data + all 387 headlines.
  3. Score the 119-headline benchmark once, versus the shipped model, with a paired bootstrap.

Caveat: Claude labelled both the training headlines and the benchmark, so part of any gain could be
the model learning Claude's labelling habits rather than a better view of the world.

Run from the project root:  python -m eval.compare_indomain
"""

import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.finance_data import load_all  # noqa: E402
from train_finance import C, CHAR_NGRAMS, fit  # noqa: E402

HERE = Path(__file__).parent
WEIGHTS = [0, 1, 3, 5, 10]   # 0 = ignore the in-domain headlines (the shipped model)


def read(name):
    with open(HERE / name, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def train(base_x, base_y, extra_x, extra_y, w):
    xs = list(base_x) + (list(extra_x) if w else [])
    ys = list(base_y) + (list(extra_y) if w else [])
    weights = np.concatenate([np.ones(len(base_x)), np.full(len(extra_x), float(w))]) if w else None
    return fit(xs, ys, CHAR_NGRAMS, C, sample_weight=weights)


def main():
    data = load_all(use_phrasebank=True)
    base_x, base_y = data["train_x"], data["train_y"]

    lab = {r["id"]: r["label"] for r in read("indomain_labels.csv")}
    rows = read("indomain_raw.csv")
    in_x = np.array([r["title"] for r in rows])
    in_y = np.array([lab[r["id"]] for r in rows])
    print(f"In-domain training headlines: {len(in_x)}")

    # 1. Weight selection by CV inside the in-domain set.
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    cv = {w: [] for w in WEIGHTS}
    for tr, te in skf.split(in_x, in_y):
        for w in WEIGHTS:
            vec, model = train(base_x, base_y, in_x[tr], in_y[tr], w)
            pred = model.predict(vec.transform(in_x[te]))
            cv[w].append(f1_score(in_y[te], pred, average="macro"))
    print("\n5-fold CV on the in-domain headlines (macro-F1 on held-out folds):")
    for w in WEIGHTS:
        print(f"  weight {w:>2}{' (shipped model)' if w == 0 else ''}: {np.mean(cv[w]):.3f} +/- {np.std(cv[w]):.3f}")
    best_w = max(WEIGHTS, key=lambda w: np.mean(cv[w]))
    print(f"Chosen weight: {best_w}")
    if best_w == 0:
        print("Extra in-domain data does not help even in cross-validation. Nothing to change.")
        return

    # 2-3. Final training and one benchmark comparison.
    bl = {r["id"]: r["label"] for r in read("labels.csv")}
    brow = [r for r in read("headlines_raw.csv") if r["id"] in bl]
    titles = [r["title"] for r in brow]
    y = np.array([bl[r["id"]] for r in brow])
    pred = {}
    for name, w in (("shipped", 0), ("with in-domain", best_w)):
        vec, model = train(base_x, base_y, in_x, in_y, w)
        pred[name] = model.predict(vec.transform(titles))

    print("\nYahoo benchmark (119 headlines, untouched until now):")
    for name in pred:
        p = pred[name]
        polar = [(a, b) for a, b in zip(y, p) if a != "neutral" and b != "neutral"]
        direction = np.mean([a == b for a, b in polar]) if polar else float("nan")
        print(f"  {name:15} accuracy {accuracy_score(y, p):.1%}  macro-F1 {f1_score(y, p, average='macro'):.3f}"
              f"  direction when it picks a side {direction:.1%} (n={len(polar)})")
    for cls in ("positive", "negative"):
        m = y == cls
        print(f"  recall on {cls:8}: shipped {np.mean(pred['shipped'][m] == cls):.0%} -> "
              f"with in-domain {np.mean(pred['with in-domain'][m] == cls):.0%}  (n={m.sum()})")

    rng = np.random.default_rng(0)
    da, df = [], []
    for _ in range(2000):
        idx = rng.integers(0, len(y), len(y))
        da.append(accuracy_score(y[idx], pred["with in-domain"][idx]) - accuracy_score(y[idx], pred["shipped"][idx]))
        df.append(f1_score(y[idx], pred["with in-domain"][idx], average="macro") - f1_score(y[idx], pred["shipped"][idx], average="macro"))
    print(f"  accuracy difference 95% CI: [{np.percentile(da, 2.5):+.1%}, {np.percentile(da, 97.5):+.1%}]"
          f"   macro-F1 difference 95% CI: [{np.percentile(df, 2.5):+.3f}, {np.percentile(df, 97.5):+.3f}]")
    print("  (a CI that includes 0 means the gain could be noise)")


if __name__ == "__main__":
    main()
