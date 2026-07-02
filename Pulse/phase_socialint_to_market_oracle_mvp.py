"""SocialIntel → MemoryAI → MarketOracle pipeline (MVP)

Goal (MVP): connect SocialIntel outputs into the existing
UniversalAI Phase G (News/Social → Trading opportunities) and then
run MarketOracle Phase D pipeline.

Constraints:
- SocialIntel has deterministic offline collectors only (no network).
- UniversalAI Phase G uses MemoryAI best-effort retrieval.
- MarketOracle Phase D runner expects a MarketOracle data_bundle; in MVP
  we can run the retrieval step and then return opportunities without
  executing MT5 orders.

This module is an integration driver; it does NOT trade.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import importlib.util
import os
import pathlib

# Local import without relying on SocialIntel as a package
from socialintel_service import collect_signals, store_to_memory



# ---- dynamic loader for ai-memory-system- (folder name contains '-') ---------

def _load_memory_ai_system():
    """Dynamically import OPTIMIZED_memory_ai_system.py.

    Returns the OPTIMIZED_memory_ai_system class.
    """
    root = pathlib.Path(__file__).resolve().parents[1]
    mem_path = root / "ai-memory-system-" / "core" / "OPTIMIZED_memory_ai_system.py"

    # Add core dir to sys.path so relative imports work.
    import sys

    core_dir = mem_path.parent
    import sys
    # Ensure both the core dir and its parent are on sys.path so
    # OPTIMIZED_memory_ai_database can be imported.
    sys.path.insert(0, str(core_dir))
    sys.path.insert(0, str(root / "ai-memory-system-" / "core"))

    # Ensure parent 'ai-memory-system-' is importable for consistent module loading
    sys.path.insert(0, str(root / "ai-memory-system-"))


    spec = importlib.util.spec_from_file_location("memory_ai_system", str(mem_path))

    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load MemoryAI system from {mem_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OPTIMIZED_memory_ai_system


def _load_universal_orchestrator_phase_g(memory_ai: Any):
    root = pathlib.Path(__file__).resolve().parents[1]
    uni_path = root / "universal-ai" / "core" / "phase_g_universal_orchestrator.py"

    spec = importlib.util.spec_from_file_location("phase_g_universal_orchestrator", str(uni_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load orchestrator from {uni_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.PhaseGUniversalOrchestrator(memory_ai=memory_ai)


def _load_market_oracle_phase_d_runner():
    root = pathlib.Path(__file__).resolve().parents[1]
    mo_runner_path = root / "MarketOracle-workspace" / "core" / "phase_d_marketoracle_agents_runner.py"
    spec = importlib.util.spec_from_file_location("phase_d_marketoracle_agents_runner", str(mo_runner_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load MarketOracle runner from {mo_runner_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_social_to_marketoracle_mvp(
    seed_items: Optional[List[Dict[str, Any]]] = None,
    regime_hint: Optional[str] = None,
    market_data_bundle: Optional[Dict[str, Any]] = None,
    max_symbols: int = 10,
) -> Dict[str, Any]:
    """Runs integration and returns opportunities + (optional) Phase D decisions.

    If `market_data_bundle` is not provided, Phase D will be skipped.
    """
    memory_ai_system_cls = _load_memory_ai_system()
    memory_ai = memory_ai_system_cls()

    # Store SocialIntel topic profiles into MemoryAI
    store_to_memory(memory_ai=memory_ai, domain="social", seed_items=seed_items)

    # Build intelligence payload from topic profiles
    # Use SocialIntel's own deterministic collect_signals to produce topic outputs.
    signals = collect_signals(seed_items=seed_items)

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for s in signals:
        topic = s.get("topic") or "general"
        grouped.setdefault(str(topic), []).append(s)

    # Convert topic profiles using SocialIntel internals
    from SocialIntel.socialintel_service import build_topic_profile, build_topic_output

    intelligence: List[Dict[str, Any]] = []
    for topic, evs in grouped.items():
        profile = build_topic_profile(topic=str(topic), events=evs)
        out = build_topic_output(profile)
        intelligence.append(
            {
                "source": "social",
                "event_type": "social_topic",
                "summary": out["summary"],
                "sentiment": out["sentiment_score"],
                "topics": [topic],
                "detected_at": None,
            }
        )

    orchestrator = _load_universal_orchestrator_phase_g(memory_ai)
    opportunities_payload = orchestrator.observe_and_generate_opportunities(
        intelligence=intelligence,
        regime_hint=regime_hint,
    )

    result: Dict[str, Any] = {
        "status": "ok",
        "intelligence": intelligence,
        "opportunities_payload": opportunities_payload,
    }

    # Optional: run MarketOracle Phase D if a data_bundle exists
    if market_data_bundle is not None:
        mo_runner_mod = _load_market_oracle_phase_d_runner()
        result["phase_d"] = mo_runner_mod.run_phase_d_pipeline(
            data_bundle=market_data_bundle,
            memory_ai=memory_ai,
            max_symbols=max_symbols,
        )
    else:
        result["phase_d"] = {
            "status": "skipped",
            "reason": "No market_data_bundle provided. Phase D not executed in this MVP driver.",
        }

    return result


if __name__ == "__main__":
    # Minimal offline run
    out = run_social_to_marketoracle_mvp(regime_hint="unknown", market_data_bundle=None)
    import json

    print(json.dumps(out, ensure_ascii=False, indent=2))

