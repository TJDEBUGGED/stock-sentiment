"""
Tests for analysis/pipeline.py. No network, no real model: the pipeline accepts
fake get_news / get_prices / scorer / db_path, so everything here is deterministic.
"""

from analysis.pipeline import analyze_ticker, daily_average, summarize


class FakeScorer:
    """Scores by keyword so tests control exactly what each headline gets."""

    def score(self, text):
        if "good" in text:
            return {"score": 0.8, "label": "positive"}
        if "bad" in text:
            return {"score": -0.8, "label": "negative"}
        return {"score": 0.0, "label": "neutral"}


def article(uuid, title, published_at, **extra):
    return {"uuid": uuid, "title": title, "publisher": "Test", "published_at": published_at, "link": "", **extra}


def scored(score, label, published_at):
    return {"score": score, "label": label, "published_at": published_at}


# ---------- daily_average ----------

def test_daily_average_groups_by_day_and_averages():
    arts = [
        scored(0.5, "positive", "2026-10-05T09:00:00+00:00"),
        scored(-0.1, "neutral", "2026-10-05T15:00:00+00:00"),
        scored(0.2, "positive", "2026-10-06T10:00:00+00:00"),
    ]
    result = daily_average(arts)
    assert result == [
        {"date": "2026-10-05", "avg_score": 0.2, "count": 2},
        {"date": "2026-10-06", "avg_score": 0.2, "count": 1},
    ]


def test_daily_average_is_sorted_by_date():
    arts = [scored(0.1, "neutral", "2026-10-06T10:00:00+00:00"), scored(0.1, "neutral", "2026-10-01T10:00:00+00:00")]
    assert [d["date"] for d in daily_average(arts)] == ["2026-10-01", "2026-10-06"]


def test_daily_average_ignores_articles_without_a_date():
    arts = [scored(0.9, "positive", None), scored(0.3, "positive", "2026-10-06T10:00:00+00:00")]
    result = daily_average(arts)
    assert len(result) == 1 and result[0]["count"] == 1


# ---------- summarize ----------

def test_summarize_counts_labels_and_mean():
    arts = [scored(0.6, "positive", None), scored(-0.4, "negative", None), scored(0.0, "neutral", None), scored(0.4, "positive", None)]
    s = summarize(arts)
    assert s == {"headline_count": 4, "mean_score": 0.15, "positive": 2, "neutral": 1, "negative": 1}


def test_summarize_with_no_articles():
    # Guards the divide-by-zero: a ticker with no headlines must not crash the API.
    assert summarize([]) == {"headline_count": 0, "mean_score": 0.0, "positive": 0, "neutral": 0, "negative": 0}


# ---------- analyze_ticker ----------

PRICES = [{"date": "2026-09-30", "close": 100.0}, {"date": "2026-10-02", "close": 102.0}]


def run(tmp_path, news, ticker="aapl", prices=PRICES):
    return analyze_ticker(
        ticker,
        "5d",
        get_news=lambda t: news,
        get_prices=lambda t, p: prices,
        db_path=tmp_path / "test.db",
        scorer=FakeScorer(),
    )


def test_analyze_scores_and_returns_articles(tmp_path):
    result = run(tmp_path, [article("a", "good news", "2026-10-02T10:00:00+00:00"),
                            article("b", "bad news", "2026-10-02T11:00:00+00:00")])
    labels = {a["title"]: a["label"] for a in result["articles"]}
    assert labels == {"good news": "positive", "bad news": "negative"}
    assert result["summary"]["headline_count"] == 2
    assert result["daily_sentiment"] == [{"date": "2026-10-02", "avg_score": 0.0, "count": 2}]
    assert result["prices"] == PRICES


def test_analyze_cleans_the_ticker(tmp_path):
    assert run(tmp_path, [], ticker="  aapl ")["ticker"] == "AAPL"


def test_running_twice_does_not_duplicate_headlines(tmp_path):
    news = [article("a", "good news", "2026-10-02T10:00:00+00:00")]
    run(tmp_path, news)
    second = run(tmp_path, news)
    assert second["summary"]["headline_count"] == 1


def test_history_accumulates_across_runs(tmp_path):
    run(tmp_path, [article("a", "good news", "2026-10-01T10:00:00+00:00")])
    second = run(tmp_path, [article("b", "bad news", "2026-10-02T10:00:00+00:00")])
    assert second["summary"]["headline_count"] == 2


def test_period_filter_drops_headlines_before_the_first_price_date(tmp_path):
    result = run(tmp_path, [article("old", "good old", "2026-08-01T10:00:00+00:00"),
                            article("new", "good new", "2026-10-01T10:00:00+00:00")])
    assert [a["title"] for a in result["articles"]] == ["good new"]
    assert [d["date"] for d in result["daily_sentiment"]] == ["2026-10-01"]


def test_weekend_headline_after_last_price_is_kept(tmp_path):
    result = run(tmp_path, [article("w", "good weekend", "2026-10-04T10:00:00+00:00")])
    assert [d["date"] for d in result["daily_sentiment"]] == ["2026-10-04"]


def test_headlines_are_kept_per_ticker(tmp_path):
    run(tmp_path, [article("a", "good news", "2026-10-02T10:00:00+00:00")], ticker="AAPL")
    other = run(tmp_path, [], ticker="MSFT")
    assert other["summary"]["headline_count"] == 0
