"""
browse.py — Browse intent scoring with recency weighting

Products viewed more recently and more frequently
get a higher intent score.
"""

from datetime import datetime, timezone
from collections import defaultdict
import math

def score_browse_intent(browse_history: list) -> dict:
    """
    Score each viewed product by:
    - Frequency: how many times viewed
    - Recency: more recent views weighted higher
    
    Returns dict of {product_title: score}
    """
    scores = defaultdict(float)
    now = datetime.now(timezone.utc)

    for event in browse_history:
        if event.event != "product_viewed":
            continue

        title = event.data.get("productTitle", "")
        if not title:
            continue

        # Clean store name suffix
        import re
        title = re.sub(r'\s*[–-]\s*.*$', '', title).strip()

        # Recency weight — exponential decay
        # Views from 1 hour ago worth more than views from 1 week ago
        try:
            viewed_at = datetime.fromisoformat(
                event.timestamp.replace('Z', '+00:00')
            )
            hours_ago = (now - viewed_at).total_seconds() / 3600
            recency_weight = math.exp(-0.1 * hours_ago)  # decay factor
        except:
            recency_weight = 0.5  # default if timestamp missing

        scores[title] += recency_weight

    return dict(scores)