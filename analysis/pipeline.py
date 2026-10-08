"""
pipeline.py — Ties it together: fetch -> score -> store -> aggregate.

analyze_ticker() returns one JSON-friendly dict the API and dashboard consume:
  prices           daily closing prices
  articles         every stored headline for the ticker, with its sentiment
  daily_sentiment  average sentiment per day (the line plotted against price)
  summary          overall mean score + positive/neutral/negative counts
"""

from collections import defaultdict

from data import fetcher, store
from analysis.sentiment import SentimentScorer

_scorer = None


def get_scorer() -> SentimentScorer:
    """Load the model lazily, once."""
    global _scorer
    if _scorer is None:
        _scorer = SentimentScorer()
    return _scorer


def score_articles(articles: list[dict], scorer: SentimentScorer) -> list[dict]:
    return [{**a, **scorer.score(a["title"])} for a in articles]


def daily_average(articles: list[dict]) -> list[dict]:
    """Group scored headlines by calendar day and average their scores."""
    buckets = defaultdict(list)
    for a in articles:
        if a.get("published_at"):
            buckets[a["published_at"][:10]].append(a["score"])

    return [
        {"date": day, "avg_score": round(sum(s) / len(s), 4), "count": len(s)}
        for day, s in sorted(buckets.items())
    ]


def summarize(articles: list[dict]) -> dict:
    n = len(articles)
    counts = {"positive": 0, "neutral": 0, "negative": 0}
    for a in articles:
        counts[a["label"]] += 1
    mean = round(sum(a["score"] for a in articles) / n, 4) if n else 0.0
    return {"headline_count": n, "mean_score": mean, **counts}


def analyze_ticker(
    ticker: str,
    period: str = "1mo",
    *,
    get_news=fetcher.get_news,
    get_prices=fetcher.get_price_history,
    db_path=store.DB_PATH,
    scorer: SentimentScorer | None = None,
) -> dict:
    ticker = ticker.upper().strip()
    scorer = scorer or get_scorer()

    # 1. Fetch fresh headlines, score them, and save (duplicates are ignored).
    fresh = score_articles(get_news(ticker), scorer)
    store.save_headlines(ticker, fresh, db_path)

    # 2. Read back everything stored so far, so history accumulates over time.
    prices = get_prices(ticker, period)
    articles = store.load_headlines(ticker, db_path)

    # 3. Only keep headlines inside the selected period (from the first price date on).
    #    No upper bound: a weekend headline is newer than the last close but still relevant.
    if prices:
        start = prices[0]["date"]
        articles = [a for a in articles if (a.get("published_at") or "")[:10] >= start]

    return {
        "ticker": ticker,
        "period": period,
        "prices": prices,
        "articles": articles,
        "daily_sentiment": daily_average(articles),
        "summary": summarize(articles),
    }
