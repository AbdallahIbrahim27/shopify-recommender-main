"""
main.py — Shopify recommendation service.

/recommend serves from precomputed per-shop artifacts (built by /build-map, or
lazily on first miss). Heavy compute (SVD, embedding inference) runs off the
event loop, and the endpoint degrades gracefully — full hybrid -> popularity ->
empty — instead of returning 500s.
"""
import asyncio
import logging
import time
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException

import cache
from browse import browsed_product_ids, normalize_title, score_browse_intent
from collab import (
    build_customer_product_matrix,
    build_item_vectors,
    score_collaborative,
)
from config import get_settings
from models import RecommendedProduct, RecommendRequest, RecommendResponse
from shopify import WorkerError, get_all_orders, get_products
from tags import encode_product_embeddings, semantic_tag_recommend

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("recsvc")

settings = get_settings()
app = FastAPI(title="Shopify Recommendation Engine")


# ----------------------------------------------------------------------
# Inbound auth: enforced only when INTERNAL_API_KEY is configured, so local
# development and /health stay frictionless (P2-1).
# ----------------------------------------------------------------------
async def require_api_key(x_api_key: Optional[str] = Header(default=None)):
    expected = settings.internal_api_key
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ----------------------------------------------------------------------
# Artifact build — fetch data, run heavy compute off-loop, persist per shop.
# ----------------------------------------------------------------------
async def build_shop_artifact(shop_domain):
    products = await get_products(shop_domain)
    if not products:
        return None

    orders = await get_all_orders(shop_domain)

    # Reuse prior embeddings for incremental encoding (P0-5).
    prev = await cache.get_artifact(shop_domain)
    existing_embeddings = prev.get("embeddings") if prev else None

    def _compute_collab(orders):
        matrix, product_index, popularity = build_customer_product_matrix(orders)
        item_vecs = build_item_vectors(matrix, product_index, settings.svd_components)
        return product_index, popularity, item_vecs

    product_index, popularity, item_vecs = await asyncio.to_thread(
        _compute_collab, orders
    )
    embeddings = await encode_product_embeddings(products, existing_embeddings)

    artifact = {
        "products": products,
        "product_index": product_index,
        "item_vecs": item_vecs,
        "popularity": popularity,
        "embeddings": embeddings,
        "title_to_id": {normalize_title(p["title"]): p["id"] for p in products},
        "built_at": time.time(),
    }
    await cache.set_artifact(shop_domain, artifact)
    return artifact


def _minmax(scores):
    """Scale scores to [0, 1] so heterogeneous signals are comparable (P0-3)."""
    if not scores:
        return {}
    lo, hi = min(scores.values()), max(scores.values())
    if hi == lo:
        return {k: 0.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


# ----------------------------------------------------------------------
# RECOMMENDATION ENDPOINT
# ----------------------------------------------------------------------
@app.post(
    "/recommend",
    response_model=RecommendResponse,
    dependencies=[Depends(require_api_key)],
)
async def recommend(request: RecommendRequest):
    started = time.time()
    limit = request.limit or settings.recommend_limit_default

    # Load (or lazily build) the per-shop artifact; never surface a 500 (P1-1).
    try:
        artifact = await cache.get_artifact(request.shop_domain)
        if artifact is None:
            artifact = await build_shop_artifact(request.shop_domain)
    except WorkerError as e:
        logger.error("Worker unavailable for %s: %s", request.shop_domain, e)
        return RecommendResponse(recommendations=[], status="degraded")
    except Exception as e:
        logger.exception("Artifact load/build failed for %s: %s", request.shop_domain, e)
        return RecommendResponse(recommendations=[], status="error")

    if not artifact or not artifact.get("products"):
        return RecommendResponse(recommendations=[], status="empty")

    products = artifact["products"]
    product_index = artifact["product_index"]
    item_vecs = artifact["item_vecs"]
    popularity = artifact["popularity"]
    embeddings = artifact["embeddings"]
    title_to_id = artifact["title_to_id"]

    # ---- Signals (each guarded: a failure degrades that signal only) ----
    # Collaborative seed = prior purchases + browsed products (P0-2).
    seed = set(request.purchased_product_ids or [])
    seed |= browsed_product_ids(request.browse_history, title_to_id)

    collab_scores = {}
    try:
        collab_scores = await asyncio.to_thread(
            score_collaborative, item_vecs, product_index, seed
        ) or {}
    except Exception as e:
        logger.error("Collaborative scoring failed: %s", e)

    # Bias the semantic query with the customer's top tags / types (P0-2).
    query_parts = [request.query] if request.query else []
    query_parts += request.top_tags or []
    query_parts += request.top_product_types or []
    query = " ".join(query_parts).strip()

    tag_scores = {}
    try:
        tag_scores = await semantic_tag_recommend(query, products, embeddings) or {}
    except Exception as e:
        logger.error("Semantic scoring failed: %s", e)

    browse_scores = {}
    try:
        browse_scores = score_browse_intent(
            request.browse_history,
            title_to_id=title_to_id,
            decay=settings.browse_decay,
        )
    except Exception as e:
        logger.error("Browse scoring failed: %s", e)

    # ---- Fusion: normalize each signal, then weight (P0-3). Fall back to
    #      popularity when collaborative produced nothing (cold start). ----
    use_collab = bool(collab_scores)
    collab_norm = _minmax(collab_scores if use_collab else popularity)
    collab_source = "collab" if use_collab else "popularity"
    tag_norm = _minmax(tag_scores)
    browse_norm = _minmax(browse_scores)

    final_scores = {}
    source_of = {}
    for product in products:
        pid = product["id"]
        contributions = {
            collab_source: settings.weight_collab * collab_norm.get(pid, 0.0),
            "tags": settings.weight_tags * tag_norm.get(pid, 0.0),
            "browse": settings.weight_browse * browse_norm.get(pid, 0.0),
        }
        final_scores[pid] = sum(contributions.values())
        # Surface which signal contributed most, for prompt tuning/debugging.
        source_of[pid] = max(contributions, key=contributions.get)

    ranked = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    product_map = {p["id"]: p for p in products}

    recommendations = []
    for pid, sc in ranked[:limit]:
        p = product_map.get(pid)
        if not p:
            continue
        recommendations.append(
            RecommendedProduct(
                product_id=pid,
                title=p.get("title", ""),
                url=p.get("url", ""),
                image=p.get("image"),
                price=p.get("price"),
                product_type=p.get("product_type", ""),
                tags=p.get("tags", []),
                source=source_of.get(pid, collab_source),
                score=round(sc, 4),
            )
        )

    logger.info(
        "recommend shop=%s n=%d limit=%d collab=%s latency_ms=%d",
        request.shop_domain,
        len(recommendations),
        limit,
        use_collab,
        int((time.time() - started) * 1000),
    )
    return RecommendResponse(recommendations=recommendations, status="ok")


# ----------------------------------------------------------------------
# BUILD PRECOMPUTED MAP
# ----------------------------------------------------------------------
@app.post("/build-map", dependencies=[Depends(require_api_key)])
async def build_map(request: RecommendRequest):
    try:
        artifact = await build_shop_artifact(request.shop_domain)
    except WorkerError as e:
        raise HTTPException(status_code=502, detail=f"Upstream data unavailable: {e}")
    except Exception as e:
        logger.exception("build_map failed for %s", request.shop_domain)
        raise HTTPException(status_code=500, detail=str(e))  # now shows real error
    if not artifact:
        return {"status": "no products", "products_indexed": 0}
    return {
        "status": "recommendation maps built",
        "products_indexed": len(artifact["products"]),
        "has_collaborative": artifact["item_vecs"] is not None,
        "built_at": artifact["built_at"],
    }
# ----------------------------------------------------------------------
# HEALTH CHECK
# ----------------------------------------------------------------------
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "shopify-recommender",
        "cache": "redis" if settings.redis_url else "in-memory",
        "cache_reachable": cache.ping(),
        "worker_configured": bool(settings.worker_url),
    }
