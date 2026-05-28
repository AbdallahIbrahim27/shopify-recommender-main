from typing import List, Optional

from pydantic import BaseModel


class BrowseEvent(BaseModel):
    event: str
    data: dict = {}
    timestamp: Optional[str] = None


class RecommendRequest(BaseModel):
    shop_domain: str
    # Personalization signals are genuinely optional — the contract no longer
    # forces callers to send data the endpoint might not use (P1-1).
    purchased_product_ids: List[int] = []
    top_product_types: List[str] = []
    top_tags: List[str] = []
    browse_history: List[BrowseEvent] = []
    limit: Optional[int] = 4
    query: Optional[str] = None


class RecommendedProduct(BaseModel):
    # Field names reconciled with what the endpoint actually emits, and the
    # fields a consumer needs to render a product are included (P1-1).
    product_id: int
    title: str
    url: str = ""
    image: Optional[str] = None
    price: Optional[str] = None
    product_type: str = ""
    tags: List[str] = []
    source: str = ""          # which signal won: collab / popularity / tags / browse
    score: float = 0.0


class RecommendResponse(BaseModel):
    recommendations: List[RecommendedProduct]
    status: str = "ok"        # ok / empty / degraded / error
