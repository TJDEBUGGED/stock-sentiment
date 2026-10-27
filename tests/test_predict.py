"""
Tests for POST /predict. Most use the real shipped model (model/*.pkl), so they also catch a
broken or incompatible pickle. No network is involved.
"""

import pytest
from fastapi.testclient import TestClient

import app.main as main
from analysis.sentiment import SentimentScorer

client = TestClient(main.app)

TEXTS = [
    "Boeing wins $14.7 billion missile contract",
    "Company misses earnings and cuts full-year guidance",
    "Quarterly report scheduled for Thursday",
]


def post(texts):
    return client.post("/predict", json={"texts": texts})


def test_predict_returns_one_result_per_text_in_order():
    r = post(TEXTS)
    assert r.status_code == 200
    results = r.json()["results"]
    assert [x["text"] for x in results] == TEXTS


def test_predict_result_is_internally_consistent():
    for res in post(TEXTS).json()["results"]:
        probs = res["probabilities"]
        assert set(probs) == {"negative", "neutral", "positive"}
        assert sum(probs.values()) == pytest.approx(1.0, abs=1e-3)
        assert res["label"] == max(probs, key=probs.get)                       # label is the most likely class
        assert res["score"] == pytest.approx(probs["positive"] - probs["negative"], abs=1e-3)
        assert -1 <= res["score"] <= 1


def test_predict_strips_whitespace():
    res = post(["  padded headline  "]).json()["results"][0]
    assert res["text"] == "padded headline"


def test_batch_scoring_matches_single_scoring():
    scorer = SentimentScorer()
    batch = scorer.score_many(TEXTS)
    for text, from_batch in zip(TEXTS, batch):
        single = scorer.score(text)
        assert single == {"score": from_batch["score"], "label": from_batch["label"]}


def test_predict_uses_the_scorer_it_is_given(monkeypatch):
    class Fake:
        def score_many(self, texts):
            return [{"label": "positive", "score": 0.5, "probabilities": {"negative": 0.1, "neutral": 0.3, "positive": 0.6}}
                    for _ in texts]

    monkeypatch.setattr(main, "get_scorer", lambda: Fake())
    res = post(["anything"]).json()["results"][0]
    assert res["label"] == "positive" and res["score"] == 0.5


@pytest.mark.parametrize("body", [
    {"texts": []},                         # nothing to score
    {"texts": [""]},                       # empty string
    {"texts": ["   "]},                    # whitespace only
    {"texts": ["x"] * (main.MAX_TEXTS + 1)},          # too many
    {"texts": ["x" * (main.MAX_TEXT_CHARS + 1)]},     # one text too long
    {"texts": "not a list"},
    {},                                    # missing field
])
def test_predict_rejects_bad_input(body):
    assert client.post("/predict", json=body).status_code == 422


def test_predict_accepts_the_maximum_allowed_batch():
    r = post(["fine headline"] * main.MAX_TEXTS)
    assert r.status_code == 200 and len(r.json()["results"]) == main.MAX_TEXTS
