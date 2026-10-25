"""
export_indomain.py — Pull extra real headlines (different tickers) to use as in-domain TRAINING data.

Any headline that also appears in the benchmark (headlines_raw.csv) is dropped, so the
benchmark stays unseen. Writes eval/indomain_raw.csv (no model scores, label added separately).
Run from the project root:  python -m eval.export_indomain
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.fetcher import get_news  # noqa: E402

TICKERS = ["ORCL", "INTC", "AMD", "QCOM", "CSCO", "IBM", "ADBE", "CRM", "PYPL", "V", "MA", "BAC",
           "GS", "MS", "C", "WFC", "CVX", "COP", "UNH", "JNJ", "MRK", "LLY", "ABBV", "HD", "LOW",
           "COST", "TGT", "PEP", "MCD", "SBUX", "NKE", "F", "GM", "UBER", "SHOP", "SNOW", "PLTR",
           "COIN", "GE", "CAT", "LMT", "T", "VZ", "CMCSA"]

HERE = Path(__file__).parent
bench = {r["title"] for r in csv.DictReader(open(HERE / "headlines_raw.csv", encoding="utf-8"))}

seen, rows = set(bench), []
for t in TICKERS:
    try:
        arts = get_news(t)
    except Exception as exc:  # one flaky ticker shouldn't stop the run
        print(f"skip {t}: {exc}")
        continue
    for a in arts:
        if a["title"] in seen:
            continue
        seen.add(a["title"])
        rows.append({"id": len(rows) + 1, "ticker": t, "title": a["title"]})

with open(HERE / "indomain_raw.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["id", "ticker", "title"])
    w.writeheader()
    w.writerows(rows)
print(f"Wrote {len(rows)} new headlines (benchmark titles excluded)")
