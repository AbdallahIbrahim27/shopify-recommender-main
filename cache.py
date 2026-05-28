"""
cache.py — Artifact + key/value store.

Backs state with Redis when REDIS_URL is configured so it is shared across
workers/replicas and the web process stays stateless (P1-2). Falls back to an
in-process dict for local development only; that fallback is NOT shared across
processes, so multi-worker deployments must set REDIS_URL.

Artifacts are namespaced per shop (P0-5) and (de)serialized off the event loop.
"""
import asyncio
import logging
import pickle
import time

from config import get_settings

logger = logging.getLogger(__name__)
_settings = get_settings()

_redis_client = None
_redis_init = False

# In-process fallback (development only).
_store: dict = {}
_expiry: dict = {}


def _get_redis():
    global _redis_client, _redis_init
    if _redis_init:
        return _redis_client
    _redis_init = True
    if not _settings.redis_url:
        logger.warning(
            "REDIS_URL not set; using in-process cache fallback "
            "(not shared across workers)."
        )
        return None
    try:
        import redis
        client = redis.Redis.from_url(_settings.redis_url)
        client.ping()
        _redis_client = client
        logger.info("Connected to Redis artifact store.")
    except Exception as e:
        logger.error("Redis unavailable (%s); using in-process fallback.", e)
        _redis_client = None
    return _redis_client


# --- Sync primitives (run via to_thread from the async helpers) ---
def _sync_get(key):
    r = _get_redis()
    if r is not None:
        try:
            return r.get(key)
        except Exception as e:
            logger.error("Redis get failed for %s: %s", key, e)
            return None
    if key not in _store:
        return None
    exp = _expiry.get(key)
    if exp is not None and time.time() > exp:
        _store.pop(key, None)
        _expiry.pop(key, None)
        return None
    return _store[key]


def _sync_set(key, value, ttl):
    r = _get_redis()
    if r is not None:
        try:
            r.set(key, value, ex=ttl if ttl else None)
            return True
        except Exception as e:
            logger.error("Redis set failed for %s: %s", key, e)
            return False
    _store[key] = value
    _expiry[key] = (time.time() + ttl) if ttl else None
    return True


def _sync_delete(key):
    r = _get_redis()
    if r is not None:
        try:
            r.delete(key)
            return True
        except Exception as e:
            logger.error("Redis delete failed for %s: %s", key, e)
            return False
    _store.pop(key, None)
    _expiry.pop(key, None)
    return True


def _artifact_key(shop_domain: str) -> str:
    return f"recsvc:artifact:{shop_domain}"


# --- Async artifact API used by the service ---
async def get_artifact(shop_domain: str):
    raw = await asyncio.to_thread(_sync_get, _artifact_key(shop_domain))
    if not raw:
        return None
    try:
        return await asyncio.to_thread(pickle.loads, raw)
    except Exception as e:
        logger.error("Failed to deserialize artifact for %s: %s", shop_domain, e)
        return None


async def set_artifact(shop_domain: str, artifact, ttl: int = None) -> bool:
    ttl = _settings.artifact_ttl_seconds if ttl is None else ttl
    try:
        raw = await asyncio.to_thread(
            pickle.dumps, artifact, pickle.HIGHEST_PROTOCOL
        )
    except Exception as e:
        logger.error("Failed to serialize artifact for %s: %s", shop_domain, e)
        return False
    return await asyncio.to_thread(_sync_set, _artifact_key(shop_domain), raw, ttl)


async def delete_artifact(shop_domain: str) -> bool:
    return await asyncio.to_thread(_sync_delete, _artifact_key(shop_domain))


def ping() -> bool:
    r = _get_redis()
    if r is None:
        return True  # in-memory store is always reachable
    try:
        return bool(r.ping())
    except Exception:
        return False
