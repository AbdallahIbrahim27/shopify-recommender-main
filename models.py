from pydantic import BaseModel
from typing import Optional


class BrowseEvent(BaseModel):
    event: str
    data: dict
    timestamp: str


class RecommendRequest(BaseModel):
    shop_domain: str
    purchased_product_ids: list[int]
    top_product_types: list[str]
    top_tags: list[str]
    browse_history: list[BrowseEvent] = []
    limit: Optional[int] = 4
    query: Optional[str] = None


class Product(BaseModel):
    id: int
    title: str
    url: str
    image: Optional[str]
    price: Optional[str]
    source: str
    score: float = 0


class RecommendResponse(BaseModel):
    recommendations: list[Product]