"""Phase D — MemoryAI retrieval + evolution loop (MVP)

Goal: replace the current "strategy generation" fallback with a loop that
queries MemoryAI for:
- historically related strategies (by symbol + regime + features)
- strategy optimization attempts and failures
- best/worst performing strategies under similar conditions

MVP constraint (given current codebase):
- MarketOracle's experts/db_handler.py currently implements file IO only.
- There is an OPTIMIZED_memory_ai_system.py with receive_contribution/get_concept APIs.

So this MVP implements an adaptation layer:
- If memory_ai exposes get_concept/get_domain_knowledge/query-style methods,
  we use them. Otherwise we fall back to file-based strategies but keep the
  evolution loop interface.

This module is written to be imported by StrategyAgent in Phase D/F.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import random


@dataclass
class EvolutionCandidate:
    strategy: Dict[str, Any]
    score: float
    reason: str


def _memory_query_best_effort(memory_ai: Any, question: str, domain: str = "trading") -> List[Dict[str, Any]]:
    """Best-effort retrieval from MemoryAI.

    Tries common APIs:
    - memory_ai.query_memory_ai(question)
    - memory_ai.retrieve(domain, question)
    - memory_ai.get_domain_knowledge(domain) then naive keyword match

    Returns a list of candidate dicts (possibly empty).
    """
    if memory_ai is None:
        return []

    # Best guess: a generic query method exists
    for method_name in ("query_memory_ai", "retrieve", "semantic_search", "search"):
        m = getattr(memory_ai, method_name, None)
        if callable(m):
            try:
                res = m(question=question) if method_name == "query_memory_ai" else m(domain=domain, query=question)
                # allow dict->list normalization
                if isinstance(res, dict):
                    return res.get("candidates", []) or res.get("results", []) or []
                if isinstance(res, list):
                    return res
            except Exception:
                pass

    # Fallback: use domain knowledge and keyword-match on JSON string
    try:
        dom = memory_ai.get_domain_knowledge(domain)
        concepts = dom.get("concepts") if isinstance(dom, dict) else None
        if concepts is None:
            return []
        ql = question.lower()
        out = []
        for c in concepts[:200]:
            blob = str(c).lower()
            if any(tok in blob for tok in ql.split()[:10]):
                out.append(c)
        return out[:25]
    except Exception:
        return []


def retrieve_strategy_candidates(
    memory_ai: Any,
    symbol: str,
    regime_label: str,
    top_k: int = 5,
) -> List[EvolutionCandidate]:
    """Retrieve candidate strategies from MemoryAI."""
    q = f"What strategies historically worked for symbol={symbol} during regime={regime_label}? What failures should be avoided?"
    records = _memory_query_best_effort(memory_ai, q, domain="trading")

    # MVP scoring: use importance/effectiveness if present else random small
    candidates: List[EvolutionCandidate] = []
    for r in records:
        if not isinstance(r, dict):
            continue
        # heuristic fields
        score = float(r.get("effectiveness_score", r.get("importance_score", 0.3)))
        reason = "memory_ai_retrieved"
        # if it looks like a strategy record, keep as strategy
        candidates.append(EvolutionCandidate(strategy=r, score=score, reason=reason))

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[:top_k]


def evolve_strategy_via_known_candidates(
    memory_ai: Any,
    symbol: str,
    regime_label: str,
    candidate_pool: List[EvolutionCandidate],
    fallback_strategies: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Generate an evolved strategy payload.

    MVP: since we may not have a full LLM/mutation layer available here,
    we return the best candidate merged with metadata describing planned
    improvements.
    """
    now = datetime.utcnow().isoformat() + "Z"

    if candidate_pool:
        best = candidate_pool[0]
        evolved = dict(best.strategy)
        evolved.setdefault("_source_type", "memory")
        evolved.setdefault("name", f"Evolved Strategy for {symbol}")
        evolved["_evolved_from"] = best.reason
        evolved["_regime"] = regime_label
        evolved["_symbol"] = symbol
        evolved["_evolved_at"] = now
        evolved["_improvement_plan"] = [
            "apply known optimizations from MemoryAI records",
            "avoid historically flagged failures",
            "revalidate under regime-similar walk-forward splits",
        ]
        return evolved

    # Fallback path
    if fallback_strategies:
        base = fallback_strategies[0]
        evolved = dict(base)
        evolved.setdefault("name", f"Evolved Strategy for {symbol}")
        evolved["_source_type"] = evolved.get("_source_type", "fallback")
        evolved["_regime"] = regime_label
        evolved["_symbol"] = symbol
        evolved["_evolved_at"] = now
        evolved["_improvement_plan"] = [
            "later: replace with MemoryAI-based mutation/evolution",
            "for now: ensure we still store strategy lifecycle into MemoryAI",
        ]
        return evolved

    return {
        "name": f"Evolved Strategy for {symbol}",
        "symbol": symbol,
        "market_regime": regime_label,
        "entry_conditions": [],
        "exit_conditions": [],
        "risk_management": [],
        "indicators": [],
        "market_structure": [],
        "psychology": [],
        "adaptations": [],
        "overall_confidence": 0.0,
        "_source_type": "memoryai_mvp_empty",
        "_evolved_at": now,
    }

