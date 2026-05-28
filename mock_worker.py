from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/products")
def products():
    return {
        "products": [
            {"id": 1, "title": "Red T-Shirt",   "tags": ["clothing", "summer"], "product_type": "shirt",  "price": "29.99"},
            {"id": 2, "title": "Blue Jeans",     "tags": ["clothing", "denim"],  "product_type": "pants",  "price": "59.99"},
            {"id": 3, "title": "White Sneakers", "tags": ["shoes", "casual"],    "product_type": "shoes",  "price": "89.99"},
            {"id": 4, "title": "Black Hoodie",   "tags": ["clothing", "winter"], "product_type": "hoodie", "price": "49.99"},
            {"id": 5, "title": "Running Shorts", "tags": ["sport", "summer"],    "product_type": "shorts", "price": "34.99"},
        ]
    }

@app.get("/orders")
def orders():
    return {
        "orders": [
            {"customer_id": "c1", "line_items": [{"product_id": 1}, {"product_id": 2}]},
            {"customer_id": "c2", "line_items": [{"product_id": 2}, {"product_id": 3}]},
            {"customer_id": "c3", "line_items": [{"product_id": 1}, {"product_id": 4}]},
            {"customer_id": "c4", "line_items": [{"product_id": 3}, {"product_id": 5}]},
            {"customer_id": "c5", "line_items": [{"product_id": 4}, {"product_id": 5}]},
        ],
        "next_cursor": None
    }

if __name__ == "__main__":
    uvicorn.run(app, port=9000)