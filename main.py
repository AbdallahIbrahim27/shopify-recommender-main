from fastapi import FastAPI
from models import RecommendRequest
from shopify import get_products
from collab import collaborative_recommend
from tags import semantic_tag_recommend
import asyncio

app = FastAPI(title="Shopify Recommendation Engine")


# -----------------------------
# RECOMMENDATION ENDPOINT
# -----------------------------
@app.post("/recommend")
async def recommend(request: RecommendRequest):

    # 1. Fetch all products from Shopify
    products = await get_products(request.shop_domain)

    if not products:
        return {"recommendations": []}

    # 2. Run both recommendation engines in parallel
    collab_task = collaborative_recommend(request.shop_domain)

    tag_task = semantic_tag_recommend(
        request.query or "",
        products
    )

    collab_scores, tag_scores = await asyncio.gather(
        collab_task,
        tag_task
    )

    # 3. Merge scores (hybrid ranking)
    final_scores = {}

    for product in products:
        pid = product["id"]

        collab_score = collab_scores.get(pid, 0)
        tag_score = tag_scores.get(pid, 0)

        # Weighted hybrid model
        final_scores[pid] = (
            0.7 * collab_score +
            0.3 * tag_score
        )

    # 4. Sort products by final score
    ranked = sorted(
        final_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # 5. Build response with product metadata
    product_map = {p["id"]: p for p in products}

    recommendations = []

    for pid, score in ranked[:10]:
        product = product_map.get(pid)

        if not product:
            continue

        recommendations.append({
            "product_id": pid,
            "title": product.get("title"),
            "score": round(score, 4),
            "tags": product.get("tags", []),
            "product_type": product.get("product_type", ""),
            "price": product.get("price")
        })

    return {
        "recommendations": recommendations
    }


# -----------------------------
# OPTIONAL: BUILD PRECOMPUTED MAP
# -----------------------------
@app.post("/build-map")
async def build_map(request: RecommendRequest):

    # Precompute embeddings + collab matrix
    products = await get_products(request.shop_domain)

    await collaborative_recommend(request.shop_domain)
    await semantic_tag_recommend("", products)

    return {
        "status": "recommendation maps built",
        "products_indexed": len(products)
    }


# -----------------------------
# HEALTH CHECK
# -----------------------------
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "shopify-recommender"
    }