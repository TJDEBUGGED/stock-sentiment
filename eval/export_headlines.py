"""
export_headlines.py — Pull real headlines for many tickers into eval/headlines_raw.csv.

No model scores are included on purpose, so labelling can't be biased by the model.
Run from the project root:  python -m eval.export_headlines
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.fetcher import get_news  # noqa: E402

TICKERS = ["AAPL", "MSFT", "TSLA", "NVDA", "AMZN", "GOOGL", "META", "JPM",
           "XOM", "PFE", "DIS", "BA", "NFLX", "WMT", "KO"]

OUT = Path(__file__).parent / "headlines_raw.csv"

seen, rows = set(), []
for t in TICKERS:
    for a in get_news(t):
        if a["title"] in seen:
            continue
        seen.add(a["title"])
        rows.append({"id": len(rows) + 1, "ticker": t, "title": a["title"],
                     "publisher": a["publisher"], "date": (a["published_at"] or "")[:10]})

with open(OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["id", "ticker", "title", "publisher", "date"])
    w.writeheader()
    w.writerows(rows)

print(f"Wrote {len(rows)} unique headlines to {OUT}")
