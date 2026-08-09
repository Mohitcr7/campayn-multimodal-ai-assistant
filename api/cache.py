"""
cache.py
=========
Campayn AI Assistant — Redis cache for intent parsing

WHY THIS EXISTS
----------------
Stage 1 of the intent parser is the most cacheable call in the system:

  * it runs at temperature 0, so the same query yields the same extraction
  * its output is small JSON (extracted / missing_fields / confidence)
  * demo and evaluation traffic repeats the same handful of queries

Caching it removes an entire LLM round trip on a repeat query. That matters
here specifically: the primary provider's free tier allows 20 requests per
day, and exhausting it is what pushed every request onto the fallback chain
earlier in this project's life.

The cache key includes the model and prompt version, so changing either
invalidates old entries instead of serving an extraction produced by a
different prompt.

Redis is shared with conversation history (api/graph.py) but uses a separate
key namespace and a separate TTL. Every function here fails soft — a cache
miss and a dead Redis are the same thing to the caller.

Author : Mohit Chauhan — Campayn Multimodal AI Assistant
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Optional

REDIS_URL         = os.environ.get("REDIS_URL", "redis://localhost:6379")
CACHE_ENABLED     = os.environ.get("CACHE_ENABLED", "true").lower() == "true"
INTENT_CACHE_TTL  = int(os.environ.get("INTENT_CACHE_TTL_SECONDS", "86400"))  # 24h

# Bump this whenever the Stage 1 prompt changes, so stale extractions
# produced by an older prompt are never served.
PROMPT_VERSION = "v1"

_client: Optional[Any] = None
_client_failed = False

_stats = {"hits": 0, "misses": 0}


def _redis():
    """Shared Redis client, or None if unavailable/disabled."""
    global _client, _client_failed

    if not CACHE_ENABLED or _client_failed:
        return None
    if _client is not None:
        return _client

    try:
        import redis
        _client = redis.from_url(REDIS_URL, decode_responses=True,
                                 socket_connect_timeout=2, socket_timeout=2)
        _client.ping()
        print("[Cache] Redis connected — intent-parse caching active")
        return _client
    except Exception as e:
        print(f"[Cache] ⚠ Redis unavailable ({e}) — caching disabled, every parse hits the LLM")
        _client_failed = True
        _client = None
        return None


def _key(query: str, role: str, model: str) -> str:
    """
    Build a stable cache key.

    The query is normalised (trimmed + lowercased) so trivial differences
    in whitespace or capitalisation still hit. Model and prompt version are
    part of the hash so a model or prompt change can't serve stale results.
    """
    raw    = f"{PROMPT_VERSION}|{model}|{role}|{query.strip().lower()}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]
    return f"campayn:intent:{digest}"


def get_intent(query: str, role: str, model: str) -> Optional[dict]:
    """Return a cached Stage 1 result, or None on miss/unavailable."""
    r = _redis()
    if r is None:
        return None
    try:
        raw = r.get(_key(query, role, model))
        if raw:
            _stats["hits"] += 1
            print(f"[Cache] HIT  intent parse — skipped an LLM call ({query[:40]!r})")
            return json.loads(raw)
        _stats["misses"] += 1
        return None
    except Exception as e:
        print(f"[Cache] read failed: {e}")
        return None


def set_intent(query: str, role: str, model: str, value: dict) -> None:
    """
    Store a Stage 1 result.

    Low-confidence extractions are NOT cached: those usually come from the
    JSON-parse fallback, and caching a degraded result for 24 hours would
    make one bad model response permanently sticky.
    """
    r = _redis()
    if r is None:
        return
    if value.get("confidence") == "low":
        return
    try:
        r.setex(_key(query, role, model), INTENT_CACHE_TTL, json.dumps(value))
    except Exception as e:
        print(f"[Cache] write failed: {e}")


def stats() -> dict:
    """Hit/miss counters for this process — surfaced on GET /health."""
    total = _stats["hits"] + _stats["misses"]
    rate  = round(_stats["hits"] / total * 100, 1) if total else 0.0
    return {**_stats, "hit_rate_pct": rate}


def health() -> str:
    """Health-check string for GET /health."""
    if not CACHE_ENABLED:
        return "disabled"
    r = _redis()
    if r is None:
        return "unavailable (no caching)"
    s = stats()
    return f"ok (hits={s['hits']} misses={s['misses']} rate={s['hit_rate_pct']}%)"
