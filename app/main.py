"""
main.py — FastAPI app for the Stock Sentiment Dashboard.
Run: uvicorn app.main:app --reload   (from the project root)
"""

import re
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from analysis.pipeline import analyze_ticker

app = FastAPI(
    title="Stock Sentiment Dashboard",
    description="Scores recent news headlines for a stock and plots sentiment against price.",
    version="1.0.0",
)

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

TICKER_RE = re.compile(r"^[A-Za-z0-9.\-^=]{1,12}$")
PERIODS = {"5d", "1mo", "3mo", "6mo", "1y"}


@app.get("/")
def root():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


@app.get("/api/analyze/{ticker}")
def analyze(ticker: str, period: str = Query("1mo")):
    if not TICKER_RE.match(ticker):
        raise HTTPException(status_code=400, detail="Invalid ticker symbol.")
    if period not in PERIODS:
        raise HTTPException(status_code=400, detail=f"period must be one of {sorted(PERIODS)}")

    try:
        result = analyze_ticker(ticker, period)
    except Exception as exc:  # upstream (Yahoo) failures shouldn't look like our bugs
        raise HTTPException(status_code=502, detail=f"Could not fetch data from Yahoo Finance: {exc}")

    if not result["prices"]:
        raise HTTPException(status_code=404, detail=f"No price data found for '{ticker.upper()}'.")
    return result


@app.get("/health")
def health():
    return {"status": "ok"}
