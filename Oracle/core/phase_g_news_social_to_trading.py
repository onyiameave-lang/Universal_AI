"""Phase G — News/Social → Trading opportunities (MVP)

This module bridges non-trading intelligence (NewsIntel/SocialIntel) into
trading opportunity queries against MemoryAI.

MVP constraint:
- MemoryAI integration point is the current ai-memory-system-/core/OPTIMIZED_memory_ai_system.py
- That system exposes:
  - get_domain_knowledge(domain)
  - get_concept(domain, concept)
  - receive_contribution(...)
- It may not expose a true vector similarity/search API.

So we implement "best-effort semantic retrieval" using:
- keyword extraction + matching against MemoryAI concepts stored in the
  'news' and 'social' domains
- returning structured "opportunities" consumable by MarketOracle.

The long-term goal (per architecture spec) is to replace this with:
- embedding generation + vector search + metadata filtering + regime linking
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class IntelligenceContext:
    source: str  # "news" | "social"
    event_type: str
    summary: str
    sentiment: Optional[float] = None
    topics: Optional[List[str]] = None
    detected_at: Optional[str] = None


def _tokenize(text: str, max_tokens: int = 12) -> List[str]:
    if not text:
        return []
    tokens: List[str] = []
    for part in text.lower().replace("/", " ").replace("-", " ").split():
        w = "".join(ch for ch in part if ch.isalnum() or ch in {"_", "."})
        if not w:
            continue
        # small stop set
        if w in {"the", "and", "for", "with", "that", "this", "from", "into", "about"}:
            continue
        if w not in tokens:
            tokens.append(w)
        if len(tokens) >= max_tokens:
            break
    return tokens


def _best_effort_retrieve_matches(memory_ai: Any, domain: str, query_text: str, top_k: int = 8) -> List[Dict[str, Any]]:
    """
    Best-effort: pull domain knowledge and rank items by token overlap
    on concept content fields.

    Returns list of dicts.
    """
    if memory_ai is None:
        return []

    try:
        dom = memory_ai.get_domain_knowledge(domain)
    except Exception:
        return []

    concepts = dom.get("concepts") if isinstance(dom, dict) else None
    if not concepts:
        return []

    tokens = _tokenize(query_text)
    if not tokens:
        # if no tokens, just return first concepts slice as fallback
        return concepts[:top_k] if isinstance(concepts, list) else []

    ranked: List[tuple[float, Dict[str, Any]]] = []
    for c in concepts:
        if not isinstance(c, dict):
            continue
        blob = " ".join(
            [
                str(c.get("concept", "")),
                str(c.get("what", "")),
                str(c.get("why", "")),
                str(c.get("when_to_use", "")),
            ]
        ).lower()

        score = 0.0
        for t in tokens:
            if t in blob:
                score += 1.0
        # incorporate possible precomputed effectiveness/confidence
        score += float(c.get("effectiveness_score", 0.0) or 0.0) * 0.25
        score += float(c.get("confidence", 0.0) or 0.0) * 0.1

        ranked.append((score, c))

    ranked.sort(key=lambda x: x[0], reverse=True)
    return [x[1] for x in ranked[:top_k]]


def query_trading_opportunities(
    memory_ai: Any,
    intelligence: List[IntelligenceContext],
    regime_hint: Optional[str] = None,
    top_k_per_item: int = 5,
) -> Dict[str, Any]:
    """
    Convert intelligence contexts into trading opportunity payload.

    Output schema:
    {
      "generated_at": "...Z",
      "opportunities": [
         {
           "opportunity_id": "...",
           "source": "news|social",
           "event_type": "...",
           "regime_hint": "...",
           "memory_matches": [ ... ],
           "trading_query": "string suitable for MarketOracle/Strategy evolution"
         }
      ]
    }
    """
    generated_at = datetime.utcnow().isoformat() + "Z"
    opportunities: List[Dict[str, Any]] = []

    for idx, ctx in enumerate(intelligence):
        query_text = ctx.summary or ""
        if ctx.topics:
            query_text += " " + " ".join(ctx.topics)

        matches = _best_effort_retrieve_matches(
            memory_ai=memory_ai,
            domain="news" if ctx.source == "news" else "social",
            query_text=query_text,
            top_k=top_k_per_item,
        )

        # Create a trading query that MarketOracle can interpret later
        regime_str = regime_hint or ""
        topics_str = ", ".join((ctx.topics or [])[:6]) if ctx.topics else ""
        trading_query = (
            f"From {ctx.source} intelligence event_type={ctx.event_type}. "
            f"Topics={topics_str}. "
            f"Sentiment={ctx.sentiment}. "
            f"RegimeHint={regime_str}. "
            f"Find historical trading strategies and outcomes most similar to this event."
        )

        opportunities.append(
            {
                "opportunity_id": f"opp_{idx}_{int(datetime.utcnow().timestamp())}",
                "source": ctx.source,
                "event_type": ctx.event_type,
                "regime_hint": regime_hint or "",
                "memory_matches": matches,
                "trading_query": trading_query,
            }
        )

    return {"generated_at": generated_at, "opportunities": opportunities}
