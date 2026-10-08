"""
Tests for the FastAPI layer (app/main.py). analyze_ticker is replaced with a fake,
so there is no network and no model load.
"""

import pytest
from fastapi.testclient import TestClient

import app.main as main

client = TestClient(main.app)

FAKE_RESULT = {
    "ticker": "AAPL", "period": "1mo",
    "prices": [{"date": "2026-10-01", "close": 100.0}],
    "articles": [], "daily_sentiment": [],
    "summary": {"headline_count": 0, "mean_score": 0.0, "positive": 0, "neutral": 0, "negative": 0},
}


@pytest.fixture
def fake_analyze(monkeypatch):
    calls = []

    def fake(ticker, period):
        calls.append((ticker, period))
        return FAKE_RESULT

    monkeypatch.setattr(main, "analyze_ticker", fake)
    return calls


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_valid_request_returns_result(fake_analyze):
    r = client.get("/api/analyze/AAPL?period=5d")
    assert r.status_code == 200 and r.json()["ticker"] == "AAPL"
    assert fake_analyze == [("AAPL", "5d")]


def test_invalid_period_is_rejected(fake_analyze):
    assert client.get("/api/analyze/AAPL?period=10y").status_code == 400
    assert fake_analyze == []   # never reached the pipeline


@pytest.mark.parametrize("bad", ["AA PL", "AAPL;DROP", "WAYTOOLONGTICKER1"])
def test_invalid_ticker_is_rejected(fake_analyze, bad):
    assert client.get(f"/api/analyze/{bad}").status_code == 400
    assert fake_analyze == []


def test_unknown_ticker_with_no_prices_is_404(monkeypatch):
    monkeypatch.setattr(main, "analyze_ticker", lambda t, p: {**FAKE_RESULT, "prices": []})
    assert client.get("/api/analyze/ZZZZ").status_code == 404


def test_upstream_failure_becomes_502(monkeypatch):
    def boom(t, p):
        raise RuntimeError("yahoo down")

    monkeypatch.setattr(main, "analyze_ticker", boom)
    r = client.get("/api/analyze/AAPL")
    assert r.status_code == 502 and "yahoo down" in r.json()["detail"]
