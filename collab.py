"""
collab.py — Collaborative filtering via item-item similarity in a latent space.

The customer x product matrix is factored with truncated SVD to obtain product
(item) vectors. Candidates are scored by similarity to the centroid of the
customer's seed products (purchased + browsed) — real personalization, instead
of ranking everyone by global popularity (P0-1).

I/O is deliberately kept out of this module so the heavy compute can be
precomputed and persisted (P0-4); callers pass in already-fetched orders.
"""
import logging

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from sklearn.utils.extmath import randomized_svd

logger = logging.getLogger(__name__)


def build_customer_product_matrix(orders):
    """
    Build a sparse customer x product interaction matrix from raw orders.

    Returns (matrix, product_index, popularity):
      - matrix: csr_matrix [n_customers x n_products]
      - product_index: {product_id: column_index}
      - popularity: {product_id: total_quantity_purchased}
    """
    customer_index = {}
    product_index = {}
    rows, cols, data = [], [], []
    popularity = {}

    for order in orders or []:
        customer_id = order.get("customer_id")
        if customer_id is None:
            continue
        if customer_id not in customer_index:
            customer_index[customer_id] = len(customer_index)
        r = customer_index[customer_id]

        for item in order.get("line_items", []) or []:
            pid = item.get("product_id")
            if pid is None:
                continue
            qty = float(item.get("quantity", 1) or 1)
            if pid not in product_index:
                product_index[pid] = len(product_index)
            rows.append(r)
            cols.append(product_index[pid])
            data.append(qty)
            popularity[pid] = popularity.get(pid, 0.0) + qty

    if not data:
        return csr_matrix((0, 0)), {}, {}

    matrix = csr_matrix(
        (data, (rows, cols)),
        shape=(len(customer_index), len(product_index)),
    )
    return matrix, product_index, popularity


def build_item_vectors(matrix, product_index, n_components=20):
    """
    Factor the matrix and return L2-normalized item vectors aligned to the
    product_index column order. Returns None when the matrix is too small to
    factor meaningfully.
    """
    if matrix.shape[0] < 2 or matrix.shape[1] < 2:
        return None
    # n_components must be strictly < n_features and <= n_samples (P1-1 guard).
    k = min(n_components, matrix.shape[1] - 1, matrix.shape[0] - 1)
    if k < 1:
        return None
    try:
        _, _, vt = randomized_svd(matrix, n_components=k, random_state=42)
    except Exception as e:
        logger.error("randomized_svd failed: %s", e)
        return None
    return normalize(vt.T)  # [n_products x k], one row per product column


def score_collaborative(item_vecs, product_index, seed_pids):
    """
    Score every product by cosine similarity to the centroid of the seed
    products' latent vectors.

    Returns {product_id: score}, or None when there is no usable seed (caller
    should fall back to popularity for a genuine cold start).
    """
    if item_vecs is None or not product_index or not seed_pids:
        return None
    seed_rows = [product_index[p] for p in seed_pids if p in product_index]
    if not seed_rows:
        return None
    profile = item_vecs[seed_rows].mean(axis=0).reshape(1, -1)
    profile = normalize(profile)
    sims = (item_vecs @ profile.T).ravel()
    return {pid: float(sims[idx]) for pid, idx in product_index.items()}
