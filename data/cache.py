"""
cache.py — A tiny in-memory time-to-live cache for upstream (Yahoo) calls.

Repeated lookups of the same ticker within the TTL are served from memory, which is faster
and keeps us well under Yahoo's unofficial rate limits.

Design choices:
  - Exceptions are never cached (the next call retries).
  - Empty results are not cached either: an empty list is usually a transient upstream hiccup.
  - The wrapped function runs outside the lock, so a slow call for one ticker doesn't block others.
  - `clock` is injectable so tests don't need to sleep.
"""

import threading
import time
from functools import wraps


def ttl_cache(seconds: float, clock=time.monotonic, max_items: int = 256):
    def decorator(fn):
        store: dict = {}
        lock = threading.Lock()

        @wraps(fn)
        def wrapper(*args, **kwargs):
            key = (args, tuple(sorted(kwargs.items())))
            with lock:
                hit = store.get(key)
                if hit and clock() - hit[0] < seconds:
                    return hit[1]

            value = fn(*args, **kwargs)

            if value:
                with lock:
                    if len(store) >= max_items:
                        store.pop(next(iter(store)))      # drop the oldest entry
                    store[key] = (clock(), value)
            return value

        wrapper.cache_clear = store.clear
        return wrapper

    return decorator
