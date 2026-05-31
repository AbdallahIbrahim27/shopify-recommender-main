# Shopify Recommendation Engine

A FastAPI service that produces **personalized product recommendations** for a Shopify
store by blending three independent signals — collaborative filtering, semantic
similarity, and browse intent — into a single ranked list.

The service is built around two ideas:

1. **Precompute the heavy work.** Expensive compute (SVD factorization, embedding
   inference) is run off the request path and persisted as a per-shop *artifact*.
   The serve path just loads the artifact and fuses signals.
2. **Degrade, never crash.** The `/recommend` endpoint always returns `200`. If a
   data source or a signal fails, it falls back gracefully
   (full hybrid → popularity → empty) instead of surfacing a `500`.

---

## Table of contents

- [How recommendations work](#how-recommendations-work)
- [Architecture](#architecture)
- [Data flow](#data-flow)
- [Project structure](#project-structure)
- [Configuration](#configuration)
- [Running locally](#running-locally)
- [API reference](#api-reference)
- [Caching & artifacts](#caching--artifacts)
- [Testing](#testing)
- [Production notes](#production-notes)

---

## How recommendations work

Each candidate product is scored by **three signals**, which are individually
normalized to `[0, 1]` and combined with configurable weights.

| Signal | Module | What it captures | Default weight |
|---|---|---|---|
| **Collaborative** | [collab.py](collab.py) | "Customers who bought what you bought also bought…" | `0.55` |
| **Semantic tags** | [semantic_tags.py](semantic_tags.py) | Text similarity between a query/intent and each product | `0.30` |
| **Browse intent** | [browse.py](browse.py) | Recency-weighted products the shopper recently viewed | `0.15` |

### Collaborative filtering
A sparse **customer × product** interaction matrix is built from order history and
factored with **truncated SVD** ([scikit-learn `randomized_svd`](https://scikit-learn.org/))
to obtain latent **item vectors**. A shopper is represented by the **centroid** of the
latent vectors of their *seed* products (prior purchases + browsed products), and every
product is scored by **cosine similarity** to that centroid. This yields real
personalization rather than ranking everyone by global popularity.

> **Cold start:** if there is no usable seed (a brand-new shopper), the engine falls
> back to **popularity** (total quantity purchased) for this signal.

### Semantic similarity
Each product's `title + tags + product_type` is embedded with a
**sentence-transformer** (`all-MiniLM-L6-v2` by default). The query — built from the
caller's free-text `query`, `top_tags`, and `top_product_types` — is embedded and
compared by cosine similarity. Embeddings are keyed by a **content hash**, so only new
or changed products are re-encoded between builds.

### Browse intent
Recently viewed products score higher via **exponential recency decay**
(`weight = exp(-decay · hours_ago)`), with more frequent views accumulating score.

### Fusion
```
final(p) = w_collab · norm(collab)[p]
         + w_tags   · norm(tags)[p]
         + w_browse · norm(browse)[p]
```
Products are ranked by `final`, already-purchased products are excluded, and the top
`limit` are returned. Each result is tagged with the `source` signal that contributed
most (useful for debugging / tuning).

---

## Architecture

```
                          ┌──────────────────────────────────────────────┐
                          │                  FastAPI app                  │
                          │                   (main.py)                   │
                          │                                              │
   client ──X-API-Key──▶  │  /recommend   /build-map   /health           │
                          │      │             │                          │
                          │      ▼             ▼                          │
                          │   load artifact   build artifact              │
                          │      │             │                          │
                          └──────┼─────────────┼──────────────────────────┘
                                 │             │
                  ┌──────────────┘             └───────────────┐
                  ▼                                            ▼
        ┌───────────────────┐                       ┌───────────────────────┐
        │   cache.py        │                       │  compute (off-loop)   │
        │  Redis / in-mem   │                       │  collab.py            │
        │  per-shop artifact│                       │  semantic_tags.py     │
        └───────────────────┘                       │  browse.py            │
                                                     └───────────┬───────────┘
                                                                 │ needs data
                                                                 ▼
                                                     ┌───────────────────────┐
                                                     │     shopify.py        │
                                                     │  products + orders    │
                                                     └───────────┬───────────┘
                                                                 │
                                   ┌─────────────────────────────┴───────────────┐
                                   ▼                                              ▼
                     Direct Shopify Admin API                      Cloudflare Worker proxy
                  (SHOPIFY_ADMIN_TOKEN configured)                 (WORKER_URL configured)
                  GET /admin/api/{ver}/products.json               GET {WORKER_URL}/products
                  GET /admin/api/{ver}/orders.json                 GET {WORKER_URL}/orders
```

### Two data-source modes
`shopify.py` fetches products and orders from one of two backends, chosen automatically:

- **Direct Shopify Admin API** — used when `SHOPIFY_ADMIN_TOKEN` **and**
  `SHOPIFY_STORE_DOMAIN` are set. Talks straight to the store, no proxy.
- **Cloudflare Worker proxy** — the fallback. Expects a worker exposing
  `GET /products` and `GET /orders` (see [mock_worker.py](mock_worker.py) for the shape).

Both backends are normalized to the same internal shape (`id` coerced to `int`,
Shopify GIDs handled, `tags` coerced to `list[str]`, `price` flattened), so the rest of
the pipeline is backend-agnostic.

### Design principles
- **I/O at the edges.** `collab.py` / `semantic_tags.py` / `browse.py` are pure compute
  and take already-fetched data — they're independently unit-testable.
- **Heavy work off the event loop.** All CPU-bound work runs via `asyncio.to_thread`.
- **Stateless web tier.** State lives in the artifact store (Redis), so the app scales
  horizontally — provided `REDIS_URL` is set (the in-process fallback is **dev-only**
  and not shared across workers).

---

## Data flow

**Build (precompute) — `/build-map` or a lazy first miss:**
1. Fetch products + full order history from the active backend (`shopify.py`).
2. Build the customer×product matrix and SVD item vectors (`collab.py`).
3. Encode product embeddings, reusing unchanged ones by content hash (`semantic_tags.py`).
4. Persist the artifact (`products`, `item_vecs`, `popularity`, `embeddings`,
   `title_to_id`, …) per shop in the cache (`cache.py`).

**Serve — `/recommend`:**
1. Load the per-shop artifact (build it once under a per-shop lock on a miss).
2. Score the three signals (each guarded — a failure degrades only that signal).
3. Normalize, weight, and fuse; exclude already-purchased products.
4. Return the top `limit` with `source` + `score`.

---

## Project structure

| File | Responsibility |
|---|---|
| [main.py](main.py) | FastAPI app, endpoints, signal fusion, build orchestration, auth |
| [config.py](config.py) | Typed settings (`pydantic-settings`), loaded once and cached |
| [shopify.py](shopify.py) | Fetch + normalize products/orders (direct Admin API **or** worker proxy) |
| [collab.py](collab.py) | Collaborative filtering: interaction matrix, SVD item vectors, scoring |
| [semantic_tags.py](semantic_tags.py) | Sentence-transformer embeddings + cosine scoring |
| [tags.py](tags.py) | Async wrappers (`to_thread`) around the embedding routines |
| [browse.py](browse.py) | Recency-weighted browse-intent scoring + title→id resolution |
| [cache.py](cache.py) | Artifact store: Redis when configured, in-process fallback for dev |
| [models.py](models.py) | Pydantic request/response schemas |
| [mock_worker.py](mock_worker.py) | Local stand-in for the Cloudflare Worker (sample products/orders) |
| [test_logic.py](test_logic.py) | End-to-end + unit assertions for the core logic |
| [Procfile](Procfile) | Process definition for Heroku/Railway-style deploys |

---

## Configuration

All settings are read from environment variables or a local `.env` file
(see [config.py](config.py)). 

> ⚠️ **`.env` parsing caveat:** for an **empty** value, do **not** add an inline
> `# comment` on the same line — the parser will treat the comment text as the value.
> Inline comments after a *non-empty* value are stripped correctly.

### Inbound auth
| Variable | Default | Description |
|---|---|---|
| `INTERNAL_API_KEY` | _(unset)_ | If set, callers must send header `X-API-Key`. If unset, auth is disabled (dev convenience). |

### Direct Shopify Admin API (preferred for a single store)
| Variable | Default | Description |
|---|---|---|
| `SHOPIFY_STORE_DOMAIN` | _(unset)_ | e.g. `your-store.myshopify.com` |
| `SHOPIFY_ADMIN_TOKEN` | _(unset)_ | Admin API access token (`shpat_…`) with `read_products` + `read_orders` |
| `SHOPIFY_API_VERSION` | `2024-10` | Admin API version |

### Cloudflare Worker proxy (fallback)
| Variable | Default | Description |
|---|---|---|
| `WORKER_URL` | _(unset)_ | Base URL of the data proxy exposing `/products` and `/orders` |
| `WORKER_API_KEY` | _(unset)_ | Optional bearer token sent to the worker |
| `WORKER_TIMEOUT_RECOMMEND` | `5.0` | Per-request timeout on the serve path (seconds) |
| `WORKER_TIMEOUT_BUILD` | `60.0` | Per-request timeout on the precompute path (seconds) |
| `WORKER_MAX_ORDER_PAGES` | `200` | Cap on order-history pagination |

### Artifact store
| Variable | Default | Description |
|---|---|---|
| `REDIS_URL` | _(unset)_ | If set, artifacts are stored in Redis (shared, multi-worker safe). Otherwise an **in-process** dict is used (dev only). |
| `ARTIFACT_TTL_SECONDS` | `21600` (6h) | Artifact expiry, matching a ~6h rebuild cadence |

### Ranking knobs
| Variable | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformer model |
| `SVD_COMPONENTS` | `20` | Latent dimensions for collaborative filtering |
| `RECOMMEND_LIMIT_DEFAULT` | `4` | Default result count |
| `RECOMMEND_LIMIT_MAX` | `50` | Hard cap on a caller's `limit` |
| `WEIGHT_COLLAB` | `0.55` | Collaborative signal weight |
| `WEIGHT_TAGS` | `0.30` | Semantic signal weight |
| `WEIGHT_BROWSE` | `0.15` | Browse-intent signal weight |
| `BROWSE_DECAY` | `0.1` | Recency decay rate (per hour) |

---

## Running locally

### Prerequisites
- Python 3.10+
- Dependencies: `pip install -r requirements.txt`
  (installs `torch` + `sentence-transformers`; the first build downloads the embedding model)

### Option A — against the mock worker (no Shopify account needed)
```bash
# terminal 1 — sample data worker on :9000
python mock_worker.py

# terminal 2 — the recommender on :8000
# (PowerShell)
$env:WORKER_URL="http://localhost:9000"; $env:INTERNAL_API_KEY="devkey"
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

### Option B — against a real Shopify store (direct mode)
Set the Shopify variables (e.g. in `.env`):
```
SHOPIFY_STORE_DOMAIN=your-store.myshopify.com
SHOPIFY_ADMIN_TOKEN=shpat_xxxxxxxxxxxxxxxxxxxxxxxx
INTERNAL_API_KEY=devkey
```
Then:
```bash
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```
Check the active backend:
```bash
curl http://127.0.0.1:8000/health   # -> "data_source": "shopify-direct"
```

---

## API reference

Base URL (local): `http://127.0.0.1:8000`
Protected endpoints require header `X-API-Key: <INTERNAL_API_KEY>` (only when that
variable is set).

### `GET /health`
Liveness + active configuration. **No auth.**

**Response**
```json
{
  "status": "ok",
  "service": "shopify-recommender",
  "cache": "redis",
  "cache_reachable": true,
  "data_source": "shopify-direct",
  "shopify_direct": true,
  "shopify_store": "your-store.myshopify.com",
  "worker_configured": true
}
```

---

### `POST /build-map`
Precomputes and caches the per-shop artifact (fetch → SVD → embeddings → store). Run
this before serving, or on a schedule (~every 6h). **Auth required.**

**Request**
```json
{ "shop_domain": "your-store.myshopify.com" }
```

**Response**
```json
{
  "status": "recommendation maps built",
  "products_indexed": 42,
  "has_collaborative": true,
  "built_at": 1780181159.72
}
```

**Errors**
- `502` — upstream data source unreachable (the `detail` field explains why).
- `500` — other build failure.

---

### `POST /recommend`
Returns a ranked, personalized recommendation list. Always responds `200` (degrades
gracefully). On a cache miss it builds the artifact once under a per-shop lock.
**Auth required.**

**Request** — only `shop_domain` is required; all personalization fields are optional.

| Field | Type | Default | Description |
|---|---|---|---|
| `shop_domain` | string | **required** | Target shop |
| `purchased_product_ids` | `int[]` | `[]` | Seeds collaborative filtering; excluded from results |
| `top_product_types` | `string[]` | `[]` | Biases the semantic query |
| `top_tags` | `string[]` | `[]` | Biases the semantic query |
| `query` | string | `null` | Free-text semantic query |
| `limit` | int | `4` | Result count (clamped to `RECOMMEND_LIMIT_MAX`) |
| `browse_history` | `BrowseEvent[]` | `[]` | Recently viewed products |

`BrowseEvent`:
```json
{
  "event": "product_viewed",
  "data": { "productTitle": "Running Shorts" },
  "timestamp": "2026-05-31T10:00:00Z"
}
```

**Example**
```json
{
  "shop_domain": "your-store.myshopify.com",
  "purchased_product_ids": [101],
  "query": "summer running gear",
  "top_tags": ["summer", "sport"],
  "limit": 4,
  "browse_history": [
    { "event": "product_viewed",
      "data": { "productTitle": "Running Shorts" },
      "timestamp": "2026-05-31T10:00:00Z" }
  ]
}
```

**Response**
```json
{
  "recommendations": [
    {
      "product_id": 105,
      "title": "Running Shorts",
      "url": "https://your-store.myshopify.com/products/running-shorts",
      "image": "https://cdn.shopify.com/...jpg",
      "price": "34.99",
      "product_type": "shorts",
      "tags": ["sport", "summer"],
      "source": "collab",
      "score": 0.7236
    }
  ],
  "status": "ok"
}
```

**`status` values**
| Value | Meaning |
|---|---|
| `ok` | Recommendations returned |
| `empty` | No products for this shop |
| `degraded` | Data source unavailable (worker/Shopify error) |
| `error` | Unexpected failure during load/build |

**`source` values** (winning signal per item): `collab` · `popularity` · `tags` · `browse`

---

## Caching & artifacts

- An **artifact** is the precomputed per-shop bundle: normalized products, SVD item
  vectors, popularity map, embeddings (by content hash), and a `title→id` index.
- Stored at key `recsvc:artifact:{shop_domain}`, pickled, with a TTL
  (`ARTIFACT_TTL_SECONDS`, default 6h).
- **Redis** when `REDIS_URL` is set (shared across workers/replicas — required for
  multi-worker deployments). Otherwise an **in-process dict** is used; this is
  **development-only** and not shared across processes.
- A per-shop in-process lock serializes concurrent first-misses so the heavy build
  runs once per process.

---

## Testing

```bash
python test_logic.py
```
Covers collaborative personalization, the SVD `n_components` guard, browse recency +
id-keying, min-max normalization, auth enforcement, limit honoring, purchased-product
exclusion, cold-start → popularity fallback, graceful degradation when the data source
is down, the health endpoint, and URL-scheme handling.

---

## Production notes

This service is a strong MVP. Before production traffic, consider:

- **Pin dependencies** — `torch` / `sentence-transformers` are breaking-change prone,
  and pickled numpy artifacts can fail to deserialize across versions.
- **Shopify rate limits** — the Admin REST API is a leaky bucket (~2 req/s). Add
  `429` / `Retry-After` handling and backoff to the pagination loops for large stores.
- **Validate `shop_domain` in direct mode** — in direct mode the store is fixed by
  `SHOPIFY_STORE_DOMAIN`; the per-request `shop_domain` should be validated against it
  rather than used to construct the Admin API URL.
- **Require Redis in production** — fail loudly if `REDIS_URL` is set but unreachable;
  the in-process fallback is unbounded and per-process.
- **Distributed build lock** — the in-process lock doesn't span replicas; use a Redis
  lock to fully prevent cache stampedes.
- **Pre-build, don't lazy-build** — schedule `/build-map`; have `/recommend` return
  `degraded` on a miss instead of building inline on the request path.
- **Observability** — add metrics, request IDs, and a deeper health check
  (verify the data source is reachable).
