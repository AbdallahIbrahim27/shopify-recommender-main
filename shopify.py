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

import httpx

from config import get_settings

logger = logging.getLogger(__name__)
_settings = get_settings()


class WorkerError(RuntimeError):
    """Raised when the Shopify data proxy is unreachable or returns an error."""


def _base_url() -> str:
    if not _settings.worker_url:
        raise WorkerError("WORKER_URL is not configured.")
    return _settings.worker_url.rstrip("/")


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


async def get_products(shop_domain):
    base = _base_url()
    try:
        async with httpx.AsyncClient(
            timeout=_settings.worker_timeout_recommend, headers=_headers()
        ) as client:
            resp = await client.get(f"{base}/products", params={"shop": shop_domain})
            resp.raise_for_status()
            raw = resp.json().get("products", [])
    except (httpx.HTTPError, ValueError) as e:
        raise WorkerError(f"Failed to fetch products: {e}") from e

    return [n for n in (normalize_product(p) for p in raw) if n]


async def get_all_orders(shop_domain):
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
    except (httpx.HTTPError, ValueError) as e:
        raise WorkerError(f"Failed to fetch orders: {e}") from e

    if pages >= _settings.worker_max_order_pages and cursor:
        logger.warning(
            "Order pagination capped at %s pages for %s",
            _settings.worker_max_order_pages,
            shop_domain,
        )
    return all_orders
