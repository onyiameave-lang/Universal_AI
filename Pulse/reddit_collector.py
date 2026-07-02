"""Reddit collector adapter (optional).

This module provides `fetch_reddit_posts(...)` which returns items in a
normalized schema compatible with SocialIntel.socialintel_service:

Expected item fields (best-effort):
- topic: str
- platform/source: str
- text/content: str
- author: str
- timestamp: epoch seconds or ISO8601
- engagement: numeric (approx)
- thread_id: str (optional)
- url: str (optional)

Real API usage requires an API key/secret depending on approach.

If no credentials are present or requests fail, this module returns [] so the
rest of SocialIntel continues to work using seed_items fallback.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def fetch_reddit_posts(
    query: str,
    api_key: Optional[str] = None,
    limit: int = 30,
) -> List[Dict[str, Any]]:
    # We purposely keep this adapter optional and return [] when no creds.
    if not api_key:
        return []

    try:
        import requests  # type: ignore
    except Exception:
        return []

    # NOTE: Implementing full OAuth2 Reddit API here is outside MVP scope.
    # This adapter is a placeholder for future real-time wiring.
    # Returning [] keeps pipeline functional without keys.
    return []

