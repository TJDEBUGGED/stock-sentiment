"""
fetcher.py — Pulls stock news headlines + price history using yfinance.
No API key required.

Run directly to test: python data/fetcher.py AAPL
"""

import sys
import yfinance as yf
from datetime import datetime, timezone


def get_news(ticker: str, count: int = 10) -> list[dict]:
    """
    Fetch recent news headlines for a ticker.

    NOTE: We use yf.Search() here instead of yf.Ticker(ticker).news —
    the Ticker.news endpoint was returning empty results (a known Yahoo
    backend issue), while Search() hits a different, working endpoint.

    Returns a list of dicts: {title, publisher, published_at, link}
    """
    search = yf.Search(ticker, news_count=count)
    raw_news = search.news  # list of article dicts from Yahoo Finance

    articles = []
    for item in raw_news:
        title = item.get("title")
        if not title:
            continue

        # providerPublishTime is a Unix timestamp (seconds since epoch)
        publish_ts = item.get("providerPublishTime")
        published_at = (
            datetime.fromtimestamp(publish_ts, tz=timezone.utc).isoformat()
            if publish_ts
            else None
        )

        articles.append({
            "uuid": item.get("uuid", title),  # unique id, used to avoid storing duplicates
            "title": title,
            "publisher": item.get("publisher", "Unknown"),
            "published_at": published_at,
            "link": item.get("link", ""),
        })

    return articles


def get_price_history(ticker: str, period: str = "1mo") -> list[dict]:
    """
    Fetch daily price history for a ticker.
    period options: '5d', '1mo', '3mo', '6mo', '1y'
    Returns a list of dicts: {date, close}
    """
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period)

    prices = []
    for date, row in hist.iterrows():
        prices.append({
            "date": date.strftime("%Y-%m-%d"),
            "close": round(float(row["Close"]), 2),
        })

    return prices


if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"

    print(f"📰 Fetching news for {ticker}...\n")
    news = get_news(ticker)
    print(f"Found {len(news)} articles:\n")
    for article in news[:5]:
        print(f"  • {article['title']}")
        print(f"    ({article['publisher']}, {article['published_at']})\n")

    print(f"\n📈 Fetching 1-month price history for {ticker}...\n")
    prices = get_price_history(ticker)
    for p in prices[-5:]:
        print(f"  {p['date']}: ${p['close']}")
