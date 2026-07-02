from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class NewsItem:
    title: str
    summary: str = ""
    url: str = ""
    published_at: str = ""
    author: str = ""
    topic: str = ""
    source: str = ""
    raw_text: str = ""
    metadata: Dict[str, Any] = None


@dataclass
class NewsCluster:
    event_type: str
    topics: List[str]
    items: List[Dict[str, Any]]
    importance_hint: float = 0.0


def _extract_keywords(text: str) -> List[str]:
    if not text:
        return []
    # Deterministic lightweight keyword extraction: split on non-alnum, filter short tokens
    import re

    toks = re.split(r"[^a-zA-Z0-9]+", (text or "").lower())
    toks = [t for t in toks if len(t) >= 3]
    # Deduplicate while preserving order
    seen = set()
    out = []
    for t in toks:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _infer_event_type(item: Dict[str, Any]) -> str:
    q = " ".join(
        [
            str(item.get("topic") or ""),
            str(item.get("title") or ""),
            str(item.get("summary") or ""),
            str(item.get("raw_text") or ""),
        ]
    ).lower()

    # Heuristic event type mapping (Phase A/B stub)
    if any(k in q for k in ["cpi", "inflation", "ppi"]):
        return "inflation_report"
    if any(k in q for k in ["employment", "jobs", "nonfarm", "unemployment", "payroll"]):
        return "employment_report"
    if any(k in q for k in ["rate decision", "interest rate", "fomc", "e c b", "central bank", "cb"]):
        return "interest_rate_decision"
    if any(k in q for k in ["regulation", "regulatory", "rule", "guidance", "amendment"]):
        return "regulatory_change"
    if any(k in q for k in ["breach", "hack", "ransomware", "vulnerability", "cve", "exploit", "cyberattack"]):
        return "security_incident"
    if any(k in q for k in ["geopolit", "sanction", "war", "ceasefire", "treaty", "missile"]):
        return "major_geopolitical_event"
    if any(k in q for k in ["merger", "acquisition", "earnings", "guidance", "earnings call"]):
        return "corporate_announcement"

    return "general_news"


def cluster_items(items: List[Dict[str, Any]], max_clusters: int = 10) -> List[NewsCluster]:
    """
    Deterministic event clustering stub:
    - infer event_type per item
    - cluster by (event_type, main_topic_keyword)
    - produce up to max_clusters clusters
    """
    clusters_by_key: Dict[str, NewsCluster] = {}

    for it in items:
        event_type = _infer_event_type(it)
        topic = str(it.get("topic") or "").strip()
        title = str(it.get("title") or "")
        summary = str(it.get("summary") or "")
        raw_text = str(it.get("raw_text") or "")

        keywords = _extract_keywords(" ".join([topic, title, summary, raw_text]))
        main_kw = keywords[0] if keywords else (topic.split()[0] if topic else "general")

        key = f"{event_type}::{main_kw}"
        if key not in clusters_by_key:
            clusters_by_key[key] = NewsCluster(
                event_type=event_type,
                topics=[main_kw],
                items=[],
                importance_hint=0.0,
            )
        clusters_by_key[key].items.append(it)
        if main_kw not in clusters_by_key[key].topics:
            clusters_by_key[key].topics.append(main_kw)

    # Sort clusters by size (importance proxy) then cap
    clusters = list(clusters_by_key.values())
    clusters.sort(key=lambda c: len(c.items), reverse=True)
    return clusters[:max_clusters]
