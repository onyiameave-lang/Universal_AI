"""RL environment audit report generator (Phase D)

MVP implementation:
- Produces a structured audit JSON using static inspection of the current
  MarketOracle chart_expert.py environment.
- Intended to be invoked by UniversalAI Coordinator/Auditor later.

Outputs:
- reward_function summary
- action_space semantics
- observation_space feature breakdown
- risk management hard limits present
- validation harness missing (walk-forward/OOS/MC)

This is an MVP that does not require parsing Python AST.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


def generate_rl_audit_report() -> Dict[str, Any]:
    # NOTE: This is a static MVP report aligned with current
    # experts/chart_expert.py implementation.
    report: Dict[str, Any] = {
        "agent": "PhaseD_RL_Auditor",
        "status": "mvp_static_report",
        "target": {
            "module": "MarketOracle-workspace/experts/chart_expert.py",
            "class": "MultiSymbolChartExpert",
        },
        "reward_function": {
            "type": "risk_adjusted_shaped",
            "key_terms": [
                "differential_sharpe_increment (sliding window)",
                "trade pnl with R-multiple clipping",
                "quadratic drawdown penalty",
                "overtrading penalty",
                "switch penalty",
                "excess leverage penalty",
                "time decay",
            ],
            "notes": [
                "Reward is computed both on close and per-step baseline.",
                "Differential Sharpe term encourages risk-adjusted improvements.",
            ],
        },
        "action_space": {
            "space": "Discrete(5)",
            "mapping": {
                "0": "Hold",
                "1": "Buy (open long)",
                "2": "Sell (open short)",
                "3": "Close position",
                "4": "Switch symbol",
            },
            "switch_guardrails_present": True,
            "notes": [
                "Switch allowed only after minimum dwell bars.",
                "Switches capped per episode.",
                "Switch closes open position and applies shaped reward.",
            ],
        },
        "state_space": {
            "observation_space": "Box continuous",
            "feature_groups": {
                "entry_tf_window": "window_size * 5 OHLCV flattened",
                "higher_tf_context": "len(HIGHER_TIMEFRAMES)*5 OHLCV (most recent snapshot)",
                "technical_indicators": "10 indicator features at current bar",
                "trend_alignment": "Daily close vs SMA50 -> 1 value",
                "knowledge_signals": "KNOWLEDGE_FEATURE_SIZE derived from strategy rules",
                "augmented_state": "session + regime + vol/spread + position + pnl/distances + counters",
            },
            "notes": [
                "No bfill leakage in most indicator rolling computations.",
                "Per-step observation cache implemented.",
            ],
        },
        "risk_management": {
            "hard_limits": [
                "max_drawdown_pct",
                "max_daily_loss_pct",
                "max_position_pct",
                "max_daily_trades",
                "max_correlated_exposure (placeholder/derived)",
            ],
            "notes": [
                "Hard limits can terminate episode early.",
            ],
        },
        "backtesting_and_validation": {
            "walk_forward": "not implemented in MVP",
            "rolling_window": "not implemented in MVP",
            "out_of_sample": "partial (train/test split used in trainer.evaluate)",
            "monte_carlo_robustness": "not implemented in MVP",
            "metrics": ["win_rate (reported)", "max_drawdown (computed)"],
            "notes": [
                "Need Sharpe/profit factor/expectancy and MC robustness.",
            ],
        },
        "audited_failure_hypotheses": [
            "Overfitting to indicator regime without cross-validation",
            "Win-rate as insufficient objective (already partially addressed via differential Sharpe)",
            "Knowledge-signal bridge quality depends on rule keywords coverage",
            "Symbol switching may still dominate exploration if rewards are not calibrated",
        ],
        "recommended_next_tasks": [
            "Implement walk-forward + rolling-window + OOS + MC harness.",
            "Persist regime history to MemoryAI and link regime->strategy performance.",
            "Implement MemoryAI retrieval-based strategy evolution loop.",
            "Add Sharpe/profit factor/expectancy metrics and retire objective based on win-rate only.",
        ],
    }
    return report

