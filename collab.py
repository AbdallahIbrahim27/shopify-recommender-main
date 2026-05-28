import pandas as pd
from sklearn.decomposition import TruncatedSVD
from shopify import get_all_orders


async def build_customer_product_matrix(shop_domain):
    orders = await get_all_orders(shop_domain)

    rows = []

    for order in orders:
        customer_id = order.get("customer_id")

        for item in order.get("line_items", []):
            rows.append({
                "customer_id": customer_id,
                "product_id": item["product_id"]
            })

    df = pd.DataFrame(rows)

    matrix = pd.crosstab(
        df["customer_id"],
        df["product_id"]
    )

    return matrix


async def collaborative_recommend(shop_domain):
    matrix = await build_customer_product_matrix(shop_domain)

    if matrix.shape[0] < 5 or matrix.shape[1] < 5:
        return matrix.sum().to_dict()

    svd = TruncatedSVD(n_components=20)

    latent = svd.fit_transform(matrix)

    similarity_scores = latent @ latent.T

    return {
        pid: score
        for pid, score in zip(
            matrix.columns,
            matrix.sum(axis=0)
        )
    }