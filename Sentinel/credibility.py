from __future__ import annotations

from typing import Any, Dict, List, Tuple


def _source_reputation_stub(source: str) -> float:
    """
    Deterministic stub:
    - unknown => 0.55
    - recognizable/exchange/official => 0.75
    - social/aggregator => 0.45
    """
    s = (source or "").lower()
    if any(k in s for k in ["reuters", "bloomberg", "ft", "wsj", "sec", "federalreserve", "bank", "centralbank", "imf", "worldbank", "ft.com"]):
        return 0.78
    if any(k in s for k in ["company", "pressrelease", "prnewswire", "globenewswire", "company"]):
        return 0.65
    if any(k in s for k in ["twitter", "x.com", "blog", "substack", "reddit", "telegram", "t.me", "random"]):
        return 0.45
    return 0.55


def score_item_credibility(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Credibility/confidence stub for a single item.
    Returns:
      - credibility_score (0..1)
      - confidence_score (0..1)
      - notes (list[str])
    """
    source = str(item.get("source") or item.get("publisher") or "")
    title = str(item.get("title") or "")
    summary = str(item.get("summary") or "")
    raw_text = str(item.get("raw_text") or "")

    rep = _source_reputation_stub(source)

    # Consistency proxy: look for explicit numbers/dates
    text = " ".join([title, summary, raw_text]).lower()
    has_numbers = any(ch.isdigit() for ch in text)
    has_dates = any(k in text for k in ["202", "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])

    consistency = 0.08 if has_numbers else 0.0
    consistency += 0.08 if has_dates else 0.0

    # Misinformation heuristics (coarse) affect confidence but also nudge credibility down.
    sensational = any(k in text for k in ["shocking", "breaking", "unbelievable", "secret", "rumor", "allegedly"])
    sensational_penalty = 0.10 if sensational else 0.0

    credibility = rep + consistency - sensational_penalty
    credibility = max(0.05, min(0.99, credibility))

    # Confidence: more context => more confidence
    length = len(text.strip())
    context_conf = 0.1 if length > 400 else 0.05 if length > 150 else 0.02

    confidence = 0.5 + context_conf + (0.15 if not sensational else -0.05)
    confidence = max(0.05, min(0.99, confidence))

    notes: List[str] = []
    notes.append(f"source_reputation={rep:.2f}")
    if has_numbers:
        notes.append("has_numbers")
    if has_dates:
        notes.append("has_dates")
    if sensational:
        notes.append("sensational_language")

    return {
        "credibility_score": float(credibility),
        "confidence_score": float(confidence),
        "notes": notes,
    }


def score_cluster_credibility(cluster: Dict[str, Any]) -> Tuple[float, float, List[str]]:
    """
    Cluster credibility:
      - credibility is median-ish across items
      - confidence increases with number of items and diversity of sources
    cluster must have: items: List[Dict[str, Any]]
    """
    items = list(cluster.get("items") or [])
    if not items:
        return 0.5, 0.5, ["empty_cluster"]

    item_scores = [score_item_credibility(it)["credibility_score"] for it in items]
    item_conf = [score_item_credibility(it)["confidence_score"] for it in items]

    # Median without importing statistics (deterministic)
    sorted_scores = sorted(item_scores)
    mid = len(sorted_scores) // 2
    credibility = sorted_scores[mid] if sorted_scores else 0.5

    # Confidence boost: more items => higher confidence
    sources = {str(it.get("source") or "") for it in items}
    sources = {s for s in sources if s}
    source_diversity = 0.05 * min(4, len(sources))  # cap

    base_conf = sum(item_conf) / len(item_conf) if item_conf else 0.5
    size_conf = 0.05 * min(6, len(items))

    confidence = base_conf + size_conf + source_diversity
    confidence = max(0.05, min(0.99, confidence))

    notes: List[str] = [
        f"cluster_size={len(items)}",
        f"source_diversity={len(sources)}",
    ]
    return float(credibility), float(confidence), notes
