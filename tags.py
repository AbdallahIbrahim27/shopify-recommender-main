from semantic_tags import SemanticTagMatcher

matcher = SemanticTagMatcher()
matcher.load_embeddings()


async def semantic_tag_recommend(query, products):
    if not matcher.product_embeddings:
        matcher.encode_products(products)

    return matcher.score(
        query=query,
        products=products
    )