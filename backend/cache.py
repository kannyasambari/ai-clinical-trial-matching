"""
backend/cache.py
─────────────────
Simple LRU in-memory cache for trial-match results.

Why? Bio_ClinicalBERT + FAISS + GPT-4o-mini takes ~5-15 s for 5 trials.
Identical (or near-identical) queries from the same session should be instant.

In production, swap for Redis:
    cache = Redis(host=REDIS_HOST, ...)
    cache.set(key, json.dumps(value), ex=TTL_SECONDS)
    cached = cache.get(key)
"""

import hashlib
import json
import logging
import time
from collections import OrderedDict
from typing import Any, Optional

logger = logging.getLogger(__name__)

_TTL_SECONDS = 300          # 5-minute TTL
_MAX_ENTRIES = 200           # max cached queries


class _LRUCache:
    def __init__(self, max_size: int = _MAX_ENTRIES, ttl: int = _TTL_SECONDS):
        self._store: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._max   = max_size
        self._ttl   = ttl

    def _make_key(self, **kwargs) -> str:
        """Deterministic hash of the query parameters."""
        canonical = json.dumps(kwargs, sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    def get(self, **kwargs) -> Optional[Any]:
        key = self._make_key(**kwargs)
        entry = self._store.get(key)
        if entry is None:
            return None
        value, timestamp = entry
        if time.time() - timestamp > self._ttl:
            del self._store[key]
            return None
        # Move to end (most recently used)
        self._store.move_to_end(key)
        logger.debug("Cache HIT for key %s", key)
        return value

    def set(self, value: Any, **kwargs) -> None:
        key = self._make_key(**kwargs)
        self._store[key] = (value, time.time())
        self._store.move_to_end(key)
        if len(self._store) > self._max:
            evicted = next(iter(self._store))
            del self._store[evicted]
            logger.debug("Cache evicted key %s", evicted)

    def clear(self) -> None:
        self._store.clear()

    @property
    def size(self) -> int:
        return len(self._store)


# Module-level singleton
query_cache = _LRUCache()