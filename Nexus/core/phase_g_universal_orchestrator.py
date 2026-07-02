"""Phase G — UniversalAI orchestrator wiring (MVP)

Light Coordinator/Observer/Auditor layer for:
- taking news/social intelligence contexts
- querying MemoryAI for related trading opportunities
- returning a structured payload downstream

Long-term design intent:
- Coordinator routes
- Observer detects inefficiencies/opportunities (here: observation input)
- Auditor validates retrieved shapes/match presence and flags anomalies

This MVP does shape validation and minimal auditing only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import importlib.util
import os

# Dynamic import because the MarketOracle-workspace folder name contains '-'
# which isn't a valid Python package identifier.
def _load_marketoracle_phase_g_module():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    module_path = os.path.join(
        base_dir,
        "MarketOracle-workspace",
        "core",
        "phase_g_news_social_to_trading.py",
    )
    spec = importlib.util.spec_from_file_location("phase_g_news_social_to_trading", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {module_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_phase_g_news_social_to_trading = _load_marketoracle_phase_g_module()
IntelligenceContext = _phase_g_news_social_to_trading.IntelligenceContext
query_trading_opportunities = _phase_g_news_social_to_trading.query_trading_opportunities


@dataclass
class AuditingResult:
    is_valid: bool
    issues: List[str]


def _audit_payload(payload: Dict[str, Any]) -> AuditingResult:
    issues: List[str] = []
    if not isinstance(payload, dict):
        return AuditingResult(False, ["payload_not_dict"])
    if "opportunities" not in payload:
        issues.append("missing_opportunities")
    opps = payload.get("opportunities", [])
    if not isinstance(opps, list):
        issues.append("opportunities_not_list")
        return AuditingResult(len(issues) == 0, issues)

    if opps:
        # flag if no matches
        for i, o in enumerate(opps[:25]):
            if not isinstance(o, dict):
                issues.append(f"opportunity_{i}_not_dict")
                continue
            matches = o.get("memory_matches", None)
            if not matches:
                issues.append(f"opportunity_{i}_no_memory_matches")

    return AuditingResult(len(issues) == 0, issues)


class PhaseGUniversalOrchestrator:
    """
    UniversalAI wiring adapter to call MemoryAI opportunity retrieval.
    """

    def __init__(self, memory_ai: Any):
        self.memory_ai = memory_ai

    def plan_mission(self, intelligence: List[Dict[str, Any]], regime_hint: Optional[str] = None) -> Dict[str, Any]:
        contexts = [
            {
                "source": item.get("source", "news"),
                "event_type": item.get("event_type", item.get("type", "general_news")),
                "summary": item.get("summary", item.get("title", "")),
                "sentiment": item.get("sentiment"),
                "topics": item.get("topics"),
                "detected_at": item.get("detected_at"),
            }
            for item in intelligence
        ]
        return {
            "mission": "observe_and_generate_opportunities",
            "regime_hint": regime_hint,
            "contexts": contexts,
            "evidence": [ctx.get("summary", "") for ctx in contexts if ctx.get("summary")],
            "confidence": 0.75 if contexts else 0.0,
        }

    def observe_and_generate_opportunities(
        self,
        intelligence: List[Dict[str, Any]],
        regime_hint: Optional[str] = None,
    ) -> Dict[str, Any]:
        contexts: List[IntelligenceContext] = []
        for it in intelligence:
            contexts.append(
                IntelligenceContext(
                    source=it.get("source", "news"),
                    event_type=it.get("event_type", it.get("type", "general_news")),
                    summary=it.get("summary", it.get("title", "")),
                    sentiment=it.get("sentiment", None),
                    topics=it.get("topics", None),
                    detected_at=it.get("detected_at", None),
                )
            )

        mission_plan = self.plan_mission(intelligence, regime_hint=regime_hint)
        opportunities_payload = query_trading_opportunities(
            memory_ai=self.memory_ai,
            intelligence=contexts,
            regime_hint=regime_hint,
        )

        audit = _audit_payload(opportunities_payload)
        opportunities_payload["mission_plan"] = mission_plan
        opportunities_payload["audit"] = {
            "is_valid": audit.is_valid,
            "issues": audit.issues,
        }
        return opportunities_payload
