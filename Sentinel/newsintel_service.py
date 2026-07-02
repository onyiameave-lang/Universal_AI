"""
NewsIntel Service (Phase A/B MVP)

Implements a deterministic, non-network news intelligence pipeline that:
- clusters related articles into events
- computes credibility + confidence
- detects misinformation risk (heuristic)
- generates retrieval-optimized summaries
- stores event intelligence into MemoryAI knowledge_records

No real external collection is implemented in this MVP.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Dict, List, Optional

from NewsIntel.event_clustering import cluster_items
from NewsIntel.credibility import score_cluster_credibility
from NewsIntel.misinformation import detect_misinformation
from NewsIntel.summarization import generate_summaries


REQUIRED_OUTPUT_KEYS = {
    "event_type",
    "importance_score",
    "credibility_score",
    "confidence_score",
    "summary",
    "sources",
}


def _to_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def collect_events(seed_items: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """Collect raw news events.

    Current behavior:
    - If `seed_items` is provided, return them.
    - Otherwise attempt a NewsAPI-backed fetch if an API key is available.
    - If API keys / network are missing, return [] (so pipeline still works).

    Seed/adapter normalized schema is best-effort; downstream clustering
    tolerates missing fields by stringifying values.
    """

    if seed_items:
        return list(seed_items)

    # Optional NewsAPI fetch (no keys => [])
    try:
        import os
        api_key = os.getenv("NEWSAPI_KEY")
    except Exception:
        api_key = None

    # Simple deterministic fallback query list when no seeds are provided.
    # Later you can pass real queries in as seed_items if you want full control.
    queries = ["CPI inflation", "FOMC rate decision"]

    try:
        from NewsIntel.newsapi_collector import fetch_newsapi_articles
    except Exception:
        return []

    collected: List[Dict[str, Any]] = []
    for q in queries:
        items = fetch_newsapi_articles(query=q, api_key=api_key)
        collected.extend(items)

    # Normalize minimal fields expected by event_clustering
    normalized: List[Dict[str, Any]] = []
    for it in collected:
        if not isinstance(it, dict):
            continue
        normalized.append(
            {
                "title": str(it.get("title") or ""),
                "summary": str(it.get("summary") or ""),
                "raw_text": str(it.get("raw_text") or ""),
                "url": str(it.get("url") or ""),
                "source": str(it.get("source") or "newsapi"),
                "topic": str(it.get("topic") or "news"),
            }
        )

    return normalized



def verify_and_score(
    events: List[Dict[str, Any]],
    max_clusters: int = 10,
) -> List[Dict[str, Any]]:
    """
    Verify credibility, detect misinformation, and produce event outputs.

    Output schema (per event):
    {
      "event_type": "",
      "importance_score": 0,
      "credibility_score": 0,
      "confidence_score": 0,
      "summary": "",
      "sources": []
    }
    """
    clusters = cluster_items(events, max_clusters=max_clusters)

    outputs: List[Dict[str, Any]] = []
    for cl in clusters:
        # Credibility + confidence
        credibility_score, confidence_score, _cred_notes = score_cluster_credibility(
            {"items": cl.items}
        )

        # Misinformation risk
        mis = detect_misinformation({"items": cl.items})
        misinformation_risk = _to_float(mis.get("misinformation_risk"), default=0.5)
        confidence_score = _to_float(confidence_score, default=0.5)
        # Reduce confidence slightly when misinformation risk is high
        confidence_score = max(0.05, min(0.99, confidence_score * (1.0 - 0.35 * misinformation_risk)))

        # Importance heuristic: cluster size + presence of key tokens
        combined_parts: List[str] = []
        for it in cl.items:
            title = str(it.get("title") or "")
            summary = str(it.get("summary") or "")
            raw_text = str(it.get("raw_text") or "")
            combined_parts.append(f"{title} {summary} {raw_text}")
        combined_text = " ".join(combined_parts).lower()

        impact_words = ["inflation", "cpi", "employment", "jobs", "rate", "decision", "fomc", "guidance", "regulation", "cyber", "breach", "sanction"]
        impact_hits = sum(1 for w in impact_words if w in combined_text)
        importance_score = 0.35 + 0.05 * min(8, len(cl.items)) + 0.07 * min(6, impact_hits)
        importance_score = max(0.0, min(1.0, importance_score))

        # Summaries
        sums = generate_summaries({"event_type": cl.event_type, "topics": cl.topics, "items": cl.items})

        # Retrieval-optimized summary string includes misinfo risk + flags
        flags = mis.get("flags") or []
        sources = []
        for it in cl.items[:8]:
            src = str(it.get("url") or it.get("source") or "").strip()
            if src and src not in sources:
                sources.append(src)

        summary_obj = {
            "event_type": cl.event_type,
            "short_summary": sums["short_summary"],
            "technical_summary": sums["technical_summary"],
            "market_summary": sums["market_summary"],
            "credibility_score": credibility_score,
            "confidence_score": confidence_score,
            "misinformation_risk": misinformation_risk,
            "misinformation_flags": flags,
            "topics": cl.topics,
            "cluster_size": len(cl.items),
        }
        summary_text = json.dumps(summary_obj, ensure_ascii=False)

        out = {
            "event_type": cl.event_type,
            "importance_score": float(importance_score),
            "credibility_score": float(credibility_score),
            "confidence_score": float(confidence_score),
            "summary": summary_text,
            "sources": sources,
        }

        # Enforce required keys
        if not REQUIRED_OUTPUT_KEYS.issubset(set(out.keys())):
            for k in (REQUIRED_OUTPUT_KEYS - set(out.keys())):
                out[k] = "" if k == "summary" else 0 if k.endswith("_score") else []
        outputs.append(out)

    return outputs


def store_to_memory(
    memory_ai: Any,
    domain: str = "news",
    seed_items: Optional[List[Dict[str, Any]]] = None,
    max_clusters: int = 10,
) -> Dict[str, Any]:
    """
    Store clustered/scored events inside MemoryAI via knowledge_records.

    Returns:
      { "stored": int, "events_seen": int, "clusters": int }
    """
    events = collect_events(seed_items=seed_items)
    outputs = verify_and_score(events, max_clusters=max_clusters)

    stored = 0
    for ev in outputs:
        if not hasattr(memory_ai, "store_knowledge_record"):
            continue

        knowledge_id = f"news_{uuid.uuid4().hex}"[:64]

        # category should be stable; use news_event
        category = "news_event"

        # Put output JSON into summary string for later retrieval.
        summary_text = ev.get("summary", "")

        relationships = []
        # Create relationships to original sources (if needed later)
        for i, src in enumerate(ev.get("sources") or []):
            relationships.append({"type": "source_citation", "index": i, "url": src})

        memory_ai.store_knowledge_record(
            knowledge_id=knowledge_id,
            domain=domain,
            source="NewsIntelService",
            title=str(ev.get("event_type", "")),
            author="",
            category=category,
            symbol="",
            strategy_type="",
            market_regime="",
            importance_score=_to_float(ev.get("importance_score"), 0.0),
            embedding=[],
            summary=summary_text,
            relationships=relationships,
        )
        stored += 1

    return {"stored": stored, "events_seen": len(events), "clusters": len(outputs)}

