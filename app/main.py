"""
main.py — FastAPI app for the Stock Sentiment Dashboard.
Run: uvicorn app.main:app --reload   (from the project root)
"""

import logging
import re
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from analysis.pipeline import analyze_ticker, get_scorer

log = logging.getLogger("stock-sentiment")

app = FastAPI(
    title="Stock Sentiment Dashboard",
    description="Scores recent news headlines for a stock and plots sentiment against price.",
    version="1.1.0",
)

MAX_TEXTS = 50          # per /predict request
MAX_TEXT_CHARS = 1000   # per text

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
    except Exception:  # upstream (Yahoo) failures shouldn't look like our bugs
        log.exception("analyze_ticker failed for %s (%s)", ticker, period)   # details stay in the server log
        raise HTTPException(status_code=502, detail="Could not fetch price data from Yahoo Finance. Please try again shortly.")

    if not result["prices"]:
        raise HTTPException(status_code=404, detail=f"No price data found for '{ticker.upper()}'.")
    return result


class PredictRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=MAX_TEXTS)

    @field_validator("texts")
    @classmethod
    def clean_texts(cls, texts: list[str]) -> list[str]:
        cleaned = [t.strip() for t in texts]
        if any(not t for t in cleaned):
            raise ValueError("texts must not contain empty strings")
        if any(len(t) > MAX_TEXT_CHARS for t in cleaned):
            raise ValueError(f"each text must be at most {MAX_TEXT_CHARS} characters")
        return cleaned


@app.post("/predict")
def predict(request: PredictRequest):
    """Score arbitrary text. Returns label, signed score (P(pos) - P(neg)) and class probabilities."""
    scored = get_scorer().score_many(request.texts)
    return {"results": [{"text": t, **s} for t, s in zip(request.texts, scored)]}


@app.get("/health")
def health():
    return {"status": "ok"}
