"""
compare_models.py — Is a different model configuration genuinely better?

1. Train each configuration on the finance training data.
2. SELECT the best one using only the two held-out finance test sets
   (Twitter validation split + PhraseBank held-out 20%). The Yahoo benchmark is not used here.
3. Score the Yahoo benchmark ONCE, for the current shipped settings versus the selected one,
   and bootstrap the difference to see whether it is real or within noise.

Run from the project root:  python -m eval.compare_models
"""

import csv
import itertools
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.finance_data import load_all  # noqa: E402
from train_finance import C as SHIPPED_C, CHAR_NGRAMS as SHIPPED_CHARS, fit  # noqa: E402

HERE = Path(__file__).parent
GRID = list(itertools.product([False, True], [1.0, 5.0, 20.0]))   # (char_ngrams, C)


def read(name):
    with open(HERE / name, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    data = load_all(use_phrasebank=True)

    print("Held-out finance test sets (selection uses ONLY these):")
    print(f"{'config':26} {'Twitter F1':>11} {'PhraseBank F1':>14} {'mean F1':>9}")
    results = {}
    for chars, c in GRID:
        vec, model = fit(data["train_x"], data["train_y"], chars, c)
        f1s = []
        for xs, ys in data["tests"].values():
            f1s.append(f1_score(ys, model.predict(vec.transform(xs)), average="macro"))
        results[(chars, c)] = (vec, model, float(np.mean(f1s)))
        tag = " <- shipped" if (chars, c) == (SHIPPED_CHARS, SHIPPED_C) else ""
        print(f"{'word+char' if chars else 'word only':10} C={c:<12} {f1s[0]:>11.3f} {f1s[1]:>14.3f} {np.mean(f1s):>9.3f}{tag}")

    best = max(results, key=lambda k: results[k][2])
    shipped = (SHIPPED_CHARS, SHIPPED_C)
    print(f"\nSelected on held-out sets: char_ngrams={best[0]}, C={best[1]}")
    if best == shipped:
        print("The shipped configuration is already the best of the grid. Nothing to change.")
        return

    # --- Yahoo benchmark, once ---
    labels = {r["id"]: r["label"] for r in read("labels.csv")}
    rows = [r for r in read("headlines_raw.csv") if r["id"] in labels]
    titles = [r["title"] for r in rows]
    y = np.array([labels[r["id"]] for r in rows])
    pred = {}
    for name, key in (("shipped", shipped), ("selected", best)):
        vec, model, _ = results[key]
        pred[name] = model.predict(vec.transform(titles))

    print("\nYahoo benchmark (119 headlines):")
    for name in ("shipped", "selected"):
        print(f"  {name:9} accuracy {accuracy_score(y, pred[name]):.1%}   macro-F1 {f1_score(y, pred[name], average='macro'):.3f}")

    fixed = [(t, l, pred['selected'][i]) for i, (t, l) in enumerate(zip(titles, y))
             if pred['shipped'][i] != l and pred['selected'][i] == l]
    broken = [(t, l, pred['selected'][i]) for i, (t, l) in enumerate(zip(titles, y))
              if pred['shipped'][i] == l and pred['selected'][i] != l]
    print(f"  headlines the selected model fixed: {len(fixed)}   broke: {len(broken)}")

    rng = np.random.default_rng(0)
    diffs_acc, diffs_f1 = [], []
    for _ in range(2000):
        idx = rng.integers(0, len(y), len(y))
        diffs_acc.append(accuracy_score(y[idx], pred["selected"][idx]) - accuracy_score(y[idx], pred["shipped"][idx]))
        diffs_f1.append(f1_score(y[idx], pred["selected"][idx], average="macro") - f1_score(y[idx], pred["shipped"][idx], average="macro"))
    lo_a, hi_a = np.percentile(diffs_acc, [2.5, 97.5])
    lo_f, hi_f = np.percentile(diffs_f1, [2.5, 97.5])
    print(f"  accuracy difference 95% CI: [{lo_a:+.1%}, {hi_a:+.1%}]   macro-F1 difference 95% CI: [{lo_f:+.3f}, {hi_f:+.3f}]")
    print("  (a CI that includes 0 means the gain could be noise)")

    print("\nExamples fixed:")
    for t, l, p in fixed[:6]:
        print(f"  [{l}] {t[:85]}")
    print("Examples broken:")
    for t, l, p in broken[:6]:
        print(f"  [{l}] now {p}: {t[:75]}")


if __name__ == "__main__":
    main()
