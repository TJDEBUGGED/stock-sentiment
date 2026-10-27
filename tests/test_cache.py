"""Tests for data/cache.py. A fake clock replaces real time, so nothing sleeps."""

import pytest

from data.cache import ttl_cache


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def make(clock, ttl=10, max_items=256):
    calls = []

    @ttl_cache(ttl, clock=clock, max_items=max_items)
    def fetch(key, extra=None):
        calls.append(key)
        return [key]

    return fetch, calls


def test_repeat_call_within_ttl_uses_the_cache():
    clock = Clock()
    fetch, calls = make(clock)
    assert fetch("AAPL") == ["AAPL"]
    clock.now = 9.9
    fetch("AAPL")
    assert calls == ["AAPL"]


def test_call_after_ttl_refetches():
    clock = Clock()
    fetch, calls = make(clock)
    fetch("AAPL")
    clock.now = 10.1
    fetch("AAPL")
    assert calls == ["AAPL", "AAPL"]


def test_different_arguments_are_cached_separately():
    clock = Clock()
    fetch, calls = make(clock)
    fetch("AAPL")
    fetch("MSFT")
    fetch("AAPL", extra=1)          # keyword arguments are part of the key too
    assert calls == ["AAPL", "MSFT", "AAPL"]


def test_exceptions_are_not_cached():
    clock = Clock()
    attempts = []

    @ttl_cache(10, clock=clock)
    def flaky():
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("yahoo hiccup")
        return ["ok"]

    with pytest.raises(RuntimeError):
        flaky()
    assert flaky() == ["ok"]        # the failure was not remembered
    assert len(attempts) == 2


def test_empty_results_are_not_cached():
    clock = Clock()
    results = [[], ["data"]]

    @ttl_cache(10, clock=clock)
    def fetch():
        return results.pop(0)

    assert fetch() == []
    assert fetch() == ["data"]      # empty was treated as a hiccup and retried


def test_oldest_entry_is_evicted_when_full():
    clock = Clock()
    fetch, calls = make(clock, max_items=2)
    fetch("A"); fetch("B"); fetch("C")   # "A" is evicted
    fetch("B")                            # still cached
    fetch("A")                            # fetched again
    assert calls == ["A", "B", "C", "A"]


def test_cache_clear():
    clock = Clock()
    fetch, calls = make(clock)
    fetch("AAPL")
    fetch.cache_clear()
    fetch("AAPL")
    assert calls == ["AAPL", "AAPL"]
