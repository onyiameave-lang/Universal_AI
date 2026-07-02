"""NewsAPI collector adapter (optional).

This module provides `fetch_newsapi_articles(...)` which returns items in a
normalized schema compatible with NewsIntel.newsintel_service:

Expected item fields:
- title: str
- summary: str (optional)
- raw_text: str (optional)
- url: str (optional)
- topic: str (optional)
- source: str (optional)

Real API usage requires an API key.

If no key is present or requests fail, this module returns [] so the rest of
NewsIntel continues to work using seed_items fallback.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def fetch_newsapi_articles(
    query: str,
    api_key: Optional[str] = None,
    country: str = "us",
    page_size: int = 20,
) -> List[Dict[str, Any]]:
    api_key = api_key or (None)

    # No hard dependency on requests at import time.
    try:
        import requests  # type: ignore
    except Exception:
        return []

    if not api_key:
        return []

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "country": country,
        "pageSize": page_size,
        "apiKey": api_key,
    }

    try:
        resp = requests.get(url, params=params, timeout=20)
        if resp.status_code != 200:
            return []
        data = resp.json() if resp is not None else {}
        articles = data.get("articles") or []
    except Exception:
        return []

    out: List[Dict[str, Any]] = []
    for a in articles:
        if not isinstance(a, dict):
            continue
        title = str(a.get("title") or "")
        source_obj = a.get("source") or {}
        if isinstance(source_obj, dict):
            source = str(source_obj.get("name") or "")
        else:
            source = str(a.get("source") or "")

        item = {
            "title": title,
            "summary": str(a.get("description") or ""),
            "raw_text": str(a.get("content") or ""),
            "url": str(a.get("url") or ""),
            "topic": query,
            "source": source or "newsapi",
        }
        out.append(item)

    return out

