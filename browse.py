"""
browse.py — Browse-intent scoring with recency weighting.

Products viewed more recently and more frequently get a higher intent score.
This signal was previously dead code (never imported); it is now wired into the
ranker (P0-2). Scores can be keyed by product id for merging with the other
signals when a title->id resolver is supplied.
"""
import logging
import math
import re
from collections import defaultdict
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Compiled once at module load rather than per-iteration (P2-1).
_SUFFIX_RE = re.compile(r"\s*[–-]\s*.*$")


def _event_fields(event):
    """Support both pydantic BrowseEvent objects and plain dicts."""
    if isinstance(event, dict):
        return event.get("event"), event.get("data") or {}, event.get("timestamp")
    return (
        getattr(event, "event", None),
        getattr(event, "data", None) or {},
        getattr(event, "timestamp", None),
    )


def _clean_title(title):
    return _SUFFIX_RE.sub("", title or "").strip()


def normalize_title(title):
    return _clean_title(title).lower()


def score_browse_intent(browse_history, title_to_id=None, decay=0.1):
    """
    Score each viewed product by frequency + exponential recency decay.

    If title_to_id is provided, returns {product_id: score} (unresolvable
    titles are dropped); otherwise returns {clean_title: score}.
    """
    scores = defaultdict(float)
    now = datetime.now(timezone.utc)

    for event in browse_history or []:
        event_type, data, timestamp = _event_fields(event)
        if event_type != "product_viewed":
            continue

        clean = _clean_title(data.get("productTitle", ""))
        if not clean:
            continue

        recency_weight = 0.5  # default when the timestamp is missing/unparseable
        if timestamp:
            try:
                viewed_at = datetime.fromisoformat(
                    str(timestamp).replace("Z", "+00:00")
                )
                if viewed_at.tzinfo is None:
                    viewed_at = viewed_at.replace(tzinfo=timezone.utc)
                hours_ago = max(0.0, (now - viewed_at).total_seconds() / 3600.0)
                recency_weight = math.exp(-decay * hours_ago)
            except (ValueError, TypeError) as e:  # no more bare except (P2-1)
                logger.debug("Unparseable browse timestamp %r: %s", timestamp, e)

        if title_to_id is not None:
            pid = title_to_id.get(clean.lower())
            if pid is None:
                continue
            scores[pid] += recency_weight
        else:
            scores[clean] += recency_weight

    return dict(scores)


def browsed_product_ids(browse_history, title_to_id):
    """Resolve browse-history titles to known product ids (collaborative seed)."""
    ids = set()
    for event in browse_history or []:
        event_type, data, _ = _event_fields(event)
        if event_type != "product_viewed":
            continue
        pid = title_to_id.get(normalize_title(data.get("productTitle", "")))
        if pid is not None:
            ids.add(pid)
    return ids
