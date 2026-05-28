"""
semantic_tags.py — Semantic similarity between a query and product text using
sentence-transformer embeddings.

Embeddings are computed per product and keyed by a content hash, so only new or
changed products are re-encoded (P0-5). The single global embeddings.pkl is
gone — callers own per-shop persistence, eliminating cross-shop cache bleed.
Scoring is vectorized, and an empty query yields no signal instead of noise.
"""
import hashlib
import logging

import numpy as np

from config import get_settings

logger = logging.getLogger(__name__)
_settings = get_settings()

_model = None


def _get_model():
    """Load the embedding model lazily so import stays cheap and testable."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model %s", _settings.embedding_model)
        _model = SentenceTransformer(_settings.embedding_model)
    return _model


def _normalize_tags(tags):
    if isinstance(tags, str):
        return [t.strip() for t in tags.split(",") if t.strip()]
    if isinstance(tags, list):
        return [str(t).strip() for t in tags if str(t).strip()]
    return []


def _product_text(product):
    tags = _normalize_tags(product.get("tags", []))
    return (
        f"{product.get('title', '')} "
        f"{' '.join(tags)} "
        f"{product.get('product_type', '')}"
    ).strip()


def content_hash(product) -> str:
    return hashlib.sha1(_product_text(product).encode("utf-8")).hexdigest()


def encode_products(products, existing=None):
    """
    Return {product_id: {"hash": h, "vec": ndarray}} for all products, reusing
    cached embeddings whose content hash is unchanged. Only missing or changed
    products are encoded, so new products are picked up incrementally.
    """
    existing = existing or {}
    result = {}
    pending_texts = []
    pending_keys = []

    for p in products:
        pid = p["id"]
        h = content_hash(p)
        cached = existing.get(pid)
        if cached and cached.get("hash") == h:
            result[pid] = cached
        else:
            pending_texts.append(_product_text(p))
            pending_keys.append((pid, h))

    if pending_texts:
        model = _get_model()
        vecs = model.encode(pending_texts, normalize_embeddings=True)
        for (pid, h), vec in zip(pending_keys, vecs):
            result[pid] = {"hash": h, "vec": np.asarray(vec, dtype=np.float32)}

    return result


def score(query, products, embeddings):
    """
    Vectorized cosine similarity between the query and each candidate product.
    Returns {product_id: similarity}. An empty query returns {} (no signal),
    rather than scoring every product against a meaningless empty embedding.
    """
    query = (query or "").strip()
    if not query or not embeddings:
        return {}

    pids, mat = [], []
    for p in products:
        entry = embeddings.get(p["id"])
        if entry is None:
            continue
        pids.append(p["id"])
        mat.append(entry["vec"])

    if not mat:
        return {}

    model = _get_model()
    q = np.asarray(
        model.encode([query], normalize_embeddings=True)[0], dtype=np.float32
    )
    matrix = np.vstack(mat)          # rows already L2-normalized
    sims = matrix @ q                # cosine similarity (both normalized)
    return {pid: float(s) for pid, s in zip(pids, sims)}
