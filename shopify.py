"""
shopify.py — Fetch and normalize Shopify product/order data from the
Cloudflare Worker proxy.

Network failures raise WorkerError so the API layer can degrade gracefully
instead of returning a 500 (P1-1). Product/order shapes are normalized at the
boundary (P1-3): tags coerced to list[str], price flattened, ids coerced to a
canonical int (handling Shopify GIDs).
"""
import asyncio
import logging
from urllib.parse import parse_qs, urlparse

import httpx

from config import get_settings

logger = logging.getLogger(__name__)
_settings = get_settings()


class WorkerError(RuntimeError):
    """Raised when the Shopify data source is unreachable or returns an error."""


# ----------------------------------------------------------------------
# Direct Shopify Admin API mode. Enabled when a store domain + admin token
# are configured; otherwise the Cloudflare Worker proxy path is used.
# ----------------------------------------------------------------------
def _use_direct() -> bool:
    return bool(_settings.shopify_admin_token and _settings.shopify_store_domain)


def _clean_domain(shop_domain=None) -> str:
    dom = (shop_domain or _settings.shopify_store_domain or "").strip().rstrip("/")
    return dom.removeprefix("https://").removeprefix("http://")


def _admin_base(shop_domain=None) -> str:
    dom = _clean_domain(shop_domain)
    if not dom:
        raise WorkerError("SHOPIFY_STORE_DOMAIN is not configured.")
    return f"https://{dom}/admin/api/{_settings.shopify_api_version}"


def _admin_headers() -> dict:
    return {
        "X-Shopify-Access-Token": _settings.shopify_admin_token or "",
        "Accept": "application/json",
    }


def _next_page_info(resp) -> "str | None":
    """Extract the page_info cursor from Shopify's Link header (rel=next)."""
    link = resp.headers.get("link") or resp.headers.get("Link")
    if not link:
        return None
    for part in link.split(","):
        segments = part.split(";")
        if len(segments) < 2 or 'rel="next"' not in segments[1]:
            continue
        url = segments[0].strip().strip("<>")
        return parse_qs(urlparse(url).query).get("page_info", [None])[0]
    return None


def _map_shopify_product(p, shop) -> dict:
    """Map a native Shopify Admin product into the worker-equivalent shape."""
    img = p.get("image")
    image = img.get("src") if isinstance(img, dict) else img
    handle = p.get("handle")
    return {
        "id": p.get("id"),
        "title": p.get("title"),
        "url": f"https://{shop}/products/{handle}" if handle else "",
        "image": image,
        "price": None,                      # normalize_product pulls from variants
        "product_type": p.get("product_type"),
        "tags": p.get("tags"),              # comma string -> _normalize_tags handles
        "variants": p.get("variants"),
    }


def _map_shopify_order(o) -> dict:
    """Map a native Shopify Admin order into the worker-equivalent shape."""
    return {
        "customer_id": (o.get("customer") or {}).get("id"),
        "line_items": o.get("line_items", []),
    }


def _base_url() -> str:
    if not _settings.worker_url:
        raise WorkerError("WORKER_URL is not configured.")
    url = _settings.worker_url.rstrip("/")
    # Preserve an explicit scheme so a local http worker (e.g. mock_worker on
    # http://localhost:9000) stays reachable. Only default to https:// when no
    # scheme is given, guarding against WORKER_URL being set without a protocol
    # in Railway env vars.
    if url.startswith(("http://", "https://")):
        return url
    return f"https://{url}"


def _headers() -> dict:
    if _settings.worker_api_key:
        return {"Authorization": f"Bearer {_settings.worker_api_key}"}
    return {}


def _normalize_tags(tags):
    if isinstance(tags, str):
        return [t.strip() for t in tags.split(",") if t.strip()]
    if isinstance(tags, list):
        return [str(t).strip() for t in tags if str(t).strip()]
    return []


def _to_int_id(value):
    """Coerce numeric ids and Shopify GIDs (gid://shopify/Product/123) to int."""
    if value is None:
        return None
    if isinstance(value, int):
        return value
    s = str(value)
    if "/" in s:
        s = s.rstrip("/").split("/")[-1]
    try:
        return int(s)
    except (ValueError, TypeError):
        return None


def normalize_product(p):
    pid = _to_int_id(p.get("id"))
    if pid is None:
        return None
    price = p.get("price")
    if price is None:
        variants = p.get("variants") or []
        if isinstance(variants, list) and variants:
            price = variants[0].get("price")
    return {
        "id": pid,
        "title": p.get("title", "") or "",
        "url": p.get("url") or p.get("handle_url") or "",
        "image": p.get("image"),
        "price": str(price) if price is not None else None,
        "product_type": p.get("product_type", "") or "",
        "tags": _normalize_tags(p.get("tags", [])),
    }


def normalize_order(o):
    items = []
    for it in o.get("line_items", []) or []:
        pid = _to_int_id(it.get("product_id"))
        if pid is None:
            continue
        items.append({"product_id": pid, "quantity": it.get("quantity", 1) or 1})
    return {"customer_id": o.get("customer_id"), "line_items": items}


async def _get_products_direct(shop_domain):
    shop = _clean_domain(shop_domain)
    base = _admin_base(shop_domain)
    raw, page_info, pages = [], None, 0
    try:
        async with httpx.AsyncClient(
            timeout=_settings.worker_timeout_build, headers=_admin_headers()
        ) as client:
            while pages < _settings.worker_max_order_pages:
                # page_info is mutually exclusive with other filters on Shopify.
                params = {"page_info": page_info} if page_info else {}
                params["limit"] = 250
                resp = await client.get(f"{base}/products.json", params=params)
                resp.raise_for_status()
                batch = resp.json().get("products", [])
                if not batch:
                    break
                raw.extend(batch)
                page_info = _next_page_info(resp)
                pages += 1
                if not page_info:
                    break
                await asyncio.sleep(0.05)
    except (httpx.HTTPError, ValueError) as e:
        raise WorkerError(f"Failed to fetch products from Shopify: {e}") from e

    return [
        n for n in (normalize_product(_map_shopify_product(p, shop)) for p in raw) if n
    ]


async def get_products(shop_domain):
    if _use_direct():
        return await _get_products_direct(shop_domain)

    base = _base_url()
    try:
        async with httpx.AsyncClient(
            timeout=_settings.worker_timeout_recommend, headers=_headers()
        ) as client:
            resp = await client.get(f"{base}/products", params={"shop": shop_domain})
            resp.raise_for_status()
            raw = resp.json().get("products", [])
    except (httpx.HTTPError, httpx.UnsupportedProtocol, ValueError) as e:
        raise WorkerError(f"Failed to fetch products: {e}") from e

    return [n for n in (normalize_product(p) for p in raw) if n]


async def _get_orders_direct(shop_domain):
    base = _admin_base(shop_domain)
    all_orders, page_info, pages = [], None, 0
    try:
        async with httpx.AsyncClient(
            timeout=_settings.worker_timeout_build, headers=_admin_headers()
        ) as client:
            while pages < _settings.worker_max_order_pages:
                if page_info:
                    params = {"page_info": page_info, "limit": 250}
                else:
                    # status=any so closed/archived orders count too.
                    params = {"status": "any", "limit": 250}
                resp = await client.get(f"{base}/orders.json", params=params)
                resp.raise_for_status()
                batch = resp.json().get("orders", [])
                if not batch:
                    break
                all_orders.extend(normalize_order(_map_shopify_order(o)) for o in batch)
                page_info = _next_page_info(resp)
                pages += 1
                if not page_info:
                    break
                await asyncio.sleep(0.05)
    except (httpx.HTTPError, ValueError) as e:
        raise WorkerError(f"Failed to fetch orders from Shopify: {e}") from e

    if pages >= _settings.worker_max_order_pages and page_info:
        logger.warning(
            "Order pagination capped at %s pages for %s",
            _settings.worker_max_order_pages,
            shop_domain,
        )
    return all_orders


async def get_all_orders(shop_domain):
    if _use_direct():
        return await _get_orders_direct(shop_domain)

    base = _base_url()
    all_orders = []
    cursor = None
    pages = 0
    try:
        async with httpx.AsyncClient(
            timeout=_settings.worker_timeout_build, headers=_headers()
        ) as client:
            while pages < _settings.worker_max_order_pages:
                params = {"shop": shop_domain, "limit": 250}
                if cursor:
                    params["cursor"] = cursor
                resp = await client.get(f"{base}/orders", params=params)
                resp.raise_for_status()
                data = resp.json()
                orders = data.get("orders", [])
                if not orders:
                    break
                all_orders.extend(normalize_order(o) for o in orders)
                cursor = data.get("next_cursor")
                pages += 1
                if not cursor:
                    break
                await asyncio.sleep(0.05)
    except (httpx.HTTPError, httpx.UnsupportedProtocol, ValueError) as e:
        raise WorkerError(f"Failed to fetch orders: {e}") from e

    if pages >= _settings.worker_max_order_pages and cursor:
        logger.warning(
            "Order pagination capped at %s pages for %s",
            _settings.worker_max_order_pages,
            shop_domain,
        )
    return all_orders