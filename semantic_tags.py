from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

MODEL_NAME = "all-MiniLM-L6-v2"
CACHE_FILE = "embeddings.pkl"

model = SentenceTransformer(MODEL_NAME)


class SemanticTagMatcher:
    def __init__(self):
        self.product_embeddings = {}

    def encode_products(self, products):
        texts = [
            f"{p.get('title', '')} "
            f"{' '.join(p.get('tags', []))} "
            f"{p.get('product_type', '')}"
            for p in products
        ]

        embeddings = model.encode(texts)

        for product, emb in zip(products, embeddings):
            self.product_embeddings[product["id"]] = emb

        self.save_embeddings()

    def score(self, query, products):
        query_embedding = model.encode([query])[0]

        scores = {}

        for product in products:
            emb = self.product_embeddings.get(product["id"])

            if emb is None:
                continue

            similarity = cosine_similarity(
                [query_embedding],
                [emb]
            )[0][0]

            scores[product["id"]] = float(similarity)

        return scores

    def save_embeddings(self):
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(self.product_embeddings, f)

    def load_embeddings(self):
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, "rb") as f:
                self.product_embeddings = pickle.load(f)