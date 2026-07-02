"""Phase D orchestrator stub for MarketOracle agents.

This runner wires Scout/Trader/Risk/Execution/Strategy agents in a simple
pipeline without requiring UniversalAI.

MVP:
- Uses existing StrategyTester/SymbolAnalyzer/chart features.
- Does NOT place MT5 orders.

Later:
- Replace with UniversalAI Coordinator orchestration.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def run_phase_d_pipeline(data_bundle: Dict[str, Any], memory_ai: Any = None, max_symbols: int = 10) -> Dict[str, Any]:
    from core.phase_d_marketoracle_agents import ScoutAgent, TraderAgent, RiskAgent, ExecutionAgent, StrategyAgent

    symbols = sorted(list(data_bundle.keys()))[:max_symbols]

    scout = ScoutAgent(memory_ai=memory_ai)
    risk = RiskAgent()
    trader = TraderAgent(risk_agent=risk)
    executor = ExecutionAgent(mt5_interface=None)
    strategy_agent = StrategyAgent(memory_ai=memory_ai)

    ranked_setups = scout.scan(symbols, data_bundle=data_bundle, top_k=5)

    decisions: List[Dict[str, Any]] = []
    for setup in ranked_setups:
        plan = trader.plan_trade(setup)
        validated = executor.validate_execution(plan)
        decisions.append({
            "setup": {
                "symbol": setup.symbol,
                "action": setup.action,
                "confidence": setup.confidence,
                "timeframe": setup.timeframe,
                "reason": setup.reason,
            },
            "plan": plan,
            "execution": validated,
        })

    return {
        "status": "ok",
        "ranked_setups": [d["setup"] for d in decisions],
        "decisions": decisions,
    }

