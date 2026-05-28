"""
tags.py — Public surface for the semantic tag signal.

Wraps the (CPU-bound) embedding routines in asyncio.to_thread so they never
block the event loop on the request path (P0-4).
"""
import asyncio

from semantic_tags import encode_products, score


async def encode_product_embeddings(products, existing=None):
    return await asyncio.to_thread(encode_products, products, existing)


async def semantic_tag_recommend(query, products, embeddings):
    return await asyncio.to_thread(score, query, products, embeddings)
