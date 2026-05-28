"""
cache.py — In-memory cache (development mode)

Uses a simple Python dict instead of Redis.
Fast enough for testing, but resets when the
server restarts. Switch to Redis for production.
"""

import time

# Simple in-memory store
_store = {}
_expiry = {}

def get(key: str):
    """Get a value from cache. Returns None if missing or expired."""
    if key not in _store:
        return None
    if key in _expiry and time.time() > _expiry[key]:
        del _store[key]
        del _expiry[key]
        return None
    return _store[key]

def set(key: str, value, ttl_seconds: int = 3600) -> bool:
    """Store a value in cache with optional TTL."""
    try:
        _store[key] = value
        if ttl_seconds:
            _expiry[key] = time.time() + ttl_seconds
        return True
    except Exception as e:
        print(f"Cache set error: {e}")
        return False

def delete(key: str) -> bool:
    """Delete a key from cache."""
    _store.pop(key, None)
    _expiry.pop(key, None)
    return True

def exists(key: str) -> bool:
    """Check if a key exists and hasn't expired."""
    return get(key) is not None

def ping() -> bool:
    """Always reachable since it's in-memory."""
    return True