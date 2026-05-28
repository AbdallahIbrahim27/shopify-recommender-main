import httpx
import asyncio
import os

WORKER_URL = os.getenv("WORKER_URL", "http://localhost:8000")
TIMEOUT = 60


async def fetch_orders_page(client, shop_domain, cursor=None):
    params = {
        "shop": shop_domain,
        "limit": 250
    }

    if cursor:
        params["cursor"] = cursor

    response = await client.get(
        f"{WORKER_URL}/orders",
        params=params
    )

    response.raise_for_status()
    return response.json()


async def get_all_orders(shop_domain):
    all_orders = []
    cursor = None

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        while True:
            data = await fetch_orders_page(
                client,
                shop_domain,
                cursor
            )

            orders = data.get("orders", [])

            if not orders:
                break

            all_orders.extend(orders)

            cursor = data.get("next_cursor")

            if not cursor:
                break

            await asyncio.sleep(0.1)

    return all_orders


async def get_products(shop_domain):
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        response = await client.get(
            f"{WORKER_URL}/products",
            params={"shop": shop_domain}
        )

        response.raise_for_status()
        return response.json().get("products", [])