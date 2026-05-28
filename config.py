"""
config.py — Typed application settings.

Centralizes configuration so modules stop reading bare os.getenv() values with
no defaults or validation (P2-1). Settings are loaded once and cached.
"""
from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- External Shopify data proxy (Cloudflare Worker) ---
    worker_url: Optional[str] = None          # WORKER_URL
    worker_api_key: Optional[str] = None      # WORKER_API_KEY (sent to the worker)
    worker_timeout_recommend: float = 5.0     # tight budget for the serve path
    worker_timeout_build: float = 60.0        # generous budget for precompute
    worker_max_order_pages: int = 200         # cap full-history pagination

    # --- Inbound auth (Helm -> recommender). Caller sends RECOMMENDER_API_KEY. ---
    internal_api_key: Optional[str] = None    # INTERNAL_API_KEY

    # --- Artifact / cache store ---
    redis_url: Optional[str] = None           # REDIS_URL
    artifact_ttl_seconds: int = 6 * 60 * 60   # ~matches the 6h rebuild cadence

    # --- Embedding model + ranking knobs (no more magic numbers) ---
    embedding_model: str = "all-MiniLM-L6-v2"
    svd_components: int = 20
    recommend_limit_default: int = 4
    weight_collab: float = 0.55
    weight_tags: float = 0.30
    weight_browse: float = 0.15
    browse_decay: float = 0.1


@lru_cache
def get_settings() -> Settings:
    return Settings()
