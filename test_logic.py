import os
os.environ["INTERNAL_API_KEY"] = "testkey"   # set before importing config (lru_cache)
# WORKER_URL intentionally unset -> exercises graceful degradation

import numpy as np
from fastapi.testclient import TestClient

import collab, browse
import main
from main import _minmax, app

AUTH = {"X-API-Key": "testkey"}

# ---------- 1. collaborative filtering unit test ----------
# 6 customers, 6 products. Customers who buy 1 also buy 2; buyers of 4 also buy 5.
orders = [
    {"customer_id": "c1", "line_items": [{"product_id": 1}, {"product_id": 2}]},
    {"customer_id": "c2", "line_items": [{"product_id": 1}, {"product_id": 2}]},
    {"customer_id": "c3", "line_items": [{"product_id": 2}, {"product_id": 3}]},
    {"customer_id": "c4", "line_items": [{"product_id": 4}, {"product_id": 5}]},
    {"customer_id": "c5", "line_items": [{"product_id": 4}, {"product_id": 5}]},
    {"customer_id": "c6", "line_items": [{"product_id": 5}, {"product_id": 6}]},
]
matrix, pindex, popularity = collab.build_customer_product_matrix(orders)
item_vecs = collab.build_item_vectors(matrix, pindex, n_components=20)
assert item_vecs is not None, "expected item vectors"
assert item_vecs.shape[0] == 6, item_vecs.shape  # 6 products
# Seed with product 1 -> product 2 should rank above product 5 (different cluster)
scores = collab.score_collaborative(item_vecs, pindex, seed_pids={1})
assert scores[2] > scores[5], f"CF not personalizing: {scores}"
assert collab.score_collaborative(item_vecs, pindex, seed_pids=set()) is None
print("1. collaborative filtering: PASS  (seed=1 -> p2=%.3f > p5=%.3f)" % (scores[2], scores[5]))

# n_components guard: tiny catalog must not crash
m2, pi2, _ = collab.build_customer_product_matrix(
    [{"customer_id": "a", "line_items": [{"product_id": 1}, {"product_id": 2}]},
     {"customer_id": "b", "line_items": [{"product_id": 1}, {"product_id": 3}]}]
)
iv2 = collab.build_item_vectors(m2, pi2, n_components=20)  # 3 products, asks for 20
print("2. n_components guard: PASS  (3 products, asked 20, got k=%s)" % (None if iv2 is None else iv2.shape[1]))

# ---------- 3. browse recency + id mapping ----------
from datetime import datetime, timezone, timedelta
now = datetime.now(timezone.utc)
recent = (now - timedelta(hours=1)).isoformat().replace("+00:00", "Z")
old = (now - timedelta(hours=200)).isoformat().replace("+00:00", "Z")
hist = [
    {"event": "product_viewed", "data": {"productTitle": "Red Shoes - My Store"}, "timestamp": recent},
    {"event": "product_viewed", "data": {"productTitle": "Blue Hat"}, "timestamp": old},
    {"event": "checkout", "data": {}, "timestamp": recent},  # ignored
]
title_to_id = {"red shoes": 10, "blue hat": 11}
bscores = browse.score_browse_intent(hist, title_to_id=title_to_id)
assert bscores[10] > bscores[11], bscores       # recent weighted higher
assert browse.browsed_product_ids(hist, title_to_id) == {10, 11}
print("3. browse recency + id-keying: PASS  (recent=%.3f > old=%.3f)" % (bscores[10], bscores[11]))

# ---------- 4. minmax normalization ----------
assert _minmax({}) == {}
assert _minmax({"a": 5, "a2": 5}) == {"a": 0.0, "a2": 0.0}
mm = _minmax({"a": 10, "b": 20, "c": 30})
assert mm == {"a": 0.0, "b": 0.5, "c": 1.0}, mm
print("4. minmax normalization: PASS")

# ---------- build a synthetic artifact for endpoint tests ----------
products = [{"id": i, "title": f"P{i}", "url": f"/p{i}", "image": None,
            "price": "9.99", "product_type": "thing", "tags": ["a"]} for i in [1,2,3,4,5,6]]
artifact = {
    "products": products, "product_index": pindex, "item_vecs": item_vecs,
    "popularity": popularity, "embeddings": {}, "built_at": 0.0,
    "title_to_id": {p["title"].lower(): p["id"] for p in products},
}

async def fake_get_artifact(shop): return artifact
async def fake_get_none(shop): return None

client = TestClient(app)

# ---------- 5. auth enforced ----------
r = client.post("/recommend", json={"shop_domain": "s"})
assert r.status_code == 401, r.status_code
print("5. auth (missing key -> 401): PASS")

# ---------- 6. personalized recommend honors limit, emits source/url ----------
main.cache.get_artifact = fake_get_artifact
r = client.post("/recommend",
                headers=AUTH,
                json={"shop_domain": "s", "purchased_product_ids": [1], "limit": 3})
assert r.status_code == 200, r.text
body = r.json()
assert body["status"] == "ok"
assert len(body["recommendations"]) == 3, "limit not honored"
top = body["recommendations"][0]
assert set(top) >= {"product_id", "url", "source", "score", "image", "tags"}, top.keys()
assert all("source" in x for x in body["recommendations"])
# Already-purchased product 1 must not be recommended back.
assert all(x["product_id"] != 1 for x in body["recommendations"]), body["recommendations"]
print("6. recommend (limit honored=%d, source=%s, no purchased echo): PASS"
      % (len(body["recommendations"]), top["source"]))

# ---------- 7. cold-start fallback to popularity ----------
cold = dict(artifact); cold["item_vecs"] = None
async def fake_cold(shop): return cold
main.cache.get_artifact = fake_cold
r = client.post("/recommend", headers=AUTH, json={"shop_domain": "s"})  # no seed, no query
assert r.status_code == 200, r.text
srcs = {x["source"] for x in r.json()["recommendations"]}
assert srcs == {"popularity"}, srcs
print("7. cold-start -> popularity fallback: PASS  (sources=%s)" % srcs)

# ---------- 8. graceful degradation when worker is unreachable ----------
main.cache.get_artifact = fake_get_none   # forces build, which hits unset WORKER_URL
r = client.post("/recommend", headers=AUTH, json={"shop_domain": "s"})
assert r.status_code == 200, r.text
assert r.json()["status"] == "degraded", r.json()
print("8. worker unreachable -> 200 'degraded' (no 500): PASS")

# ---------- 9. health ----------
r = client.get("/health")
assert r.status_code == 200 and r.json()["status"] == "ok"
print("9. health: PASS  (%s)" % r.json())

# ---------- 10. _base_url preserves an explicit scheme ----------
import shopify
shopify._settings.worker_url = "http://localhost:9000"
assert shopify._base_url() == "http://localhost:9000", shopify._base_url()
shopify._settings.worker_url = "example.com/api"          # no scheme -> https
assert shopify._base_url() == "https://example.com/api", shopify._base_url()
shopify._settings.worker_url = None
print("10. _base_url scheme handling (http preserved, bare -> https): PASS")

print("\nALL TESTS PASSED")
