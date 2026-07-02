from __future__ import annotations

from typing import Any, Dict, List


def _join_sources(items: List[Dict[str, Any]], max_sources: int = 5) -> List[str]:
    out: List[str] = []
    for it in items[:max_sources]:
        url = str(it.get("url") or it.get("source") or "").strip()
        if url and url not in out:
            out.append(url)
    return out


def generate_summaries(cluster: Dict[str, Any]) -> Dict[str, str]:
    """
    Deterministic summaries:
      - short_summary (retrieval-optimized)
      - technical_summary (mentions detected signals)
      - market_summary (event/market relevance heuristic)
    """
    event_type = str(cluster.get("event_type") or "general_news")
    items = list(cluster.get("items") or [])
    topics = list(cluster.get("topics") or [])

    top_topic = topics[0] if topics else "general"

    # Credibility/misinformation signals are computed elsewhere; here we only use deterministic properties.
    urls = _join_sources(items, max_sources=6)
    count = len(items)

    short_summary = f"{event_type}: {top_topic} (clustered from {count} articles)."
    technical_summary = (
        f"Signals: type={event_type}, topic={top_topic}, articles={count}. "
        f"Source hints: {', '.join(urls) if urls else 'none'}."
    )

    # Market relevance heuristic
    market_summary = f"Market relevance: potential sensitivity via {top_topic} related flows; monitor headlines and guidance around {top_topic}."

    return {
        "short_summary": short_summary,
        "technical_summary": technical_summary,
        "market_summary": market_summary,
    }
