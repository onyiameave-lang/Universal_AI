"""Phase D — Strategy validation harness

Implements robust evaluation scaffolding:
- walk-forward testing
- rolling-window testing
- out-of-sample (OOS) evaluation
- Monte Carlo robustness testing (bootstrap equity path perturbation)

This is designed to integrate with the existing RL environment:
- MultiSymbolChartExpert in experts/chart_expert.py
- Strategy training/evaluation via experts/strategy_tester.py

MVP goals:
1) Provide a stable API that Phase D/F logic can call.
2) Produce metrics beyond win_rate: Sharpe, profit factor,
   expectancy, max drawdown, risk-adjusted returns.

Note: Monte Carlo is implemented as a bootstrap over per-trade returns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import math
import random
import numpy as np


@dataclass
class TradeSample:
    pnl: float
    is_win: bool
    r_multiple: Optional[float] = None
    timestamp: Optional[str] = None


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b not in (0, 0.0) else default


def compute_max_drawdown(equity_curve: List[float]) -> float:
    if not equity_curve:
        return 0.0
    peak = equity_curve[0]
    max_dd = 0.0
    for x in equity_curve:
        peak = max(peak, x)
        if peak > 0:
            dd = (peak - x) / peak
            max_dd = max(max_dd, dd)
    return float(max_dd)


def sharpe_ratio(returns: List[float], risk_free_rate: float = 0.0) -> float:
    if len(returns) < 2:
        return 0.0
    r = np.array(returns, dtype=float)
    excess = r - risk_free_rate
    std = float(np.std(excess))
    if std <= 1e-12:
        return 0.0
    return float(np.mean(excess) / std * math.sqrt(len(returns)))


def profit_factor(returns: List[float]) -> float:
    wins = sum(x for x in returns if x > 0)
    losses = -sum(x for x in returns if x < 0)
    if losses <= 0:
        return float('inf') if wins > 0 else 0.0
    return float(wins / losses)


def expectancy(returns: List[float]) -> float:
    if not returns:
        return 0.0
    return float(np.mean(returns))


def evaluate_trade_samples(trades: List[TradeSample], starting_equity: float = 10_000.0) -> Dict[str, Any]:
    returns = [t.pnl for t in trades]
    equity = [starting_equity]
    cur = starting_equity
    for r in returns:
        cur += r
        equity.append(cur)

    max_dd = compute_max_drawdown(equity)

    # Convert pnl to per-step returns for Sharpe approximation
    step_returns = []
    for i in range(1, len(equity)):
        prev = equity[i - 1]
        step_returns.append((equity[i] - prev) / prev if prev != 0 else 0.0)

    return {
        "n_trades": len(trades),
        "win_rate": float(sum(1 for t in trades if t.is_win) / len(trades)) if trades else 0.0,
        "expectancy": expectancy(returns),
        "profit_factor": profit_factor(returns),
        "sharpe": sharpe_ratio(step_returns),
        "max_drawdown": max_dd,
        "final_equity": float(equity[-1]) if equity else starting_equity,
        "risk_adjusted_return": (expectancy(returns) / max(max_dd, 1e-8)) if trades else 0.0,
    }


def bootstrap_equity_monte_carlo(
    trades: List[TradeSample],
    n_runs: int = 200,
    starting_equity: float = 10_000.0,
    seed: int = 42,
) -> Dict[str, Any]:
    """Monte Carlo robustness via bootstrap over trade order.

    This approximates uncertainty in sequence/paths without needing
    minute-level replay.
    """
    rng = random.Random(seed)
    if not trades:
        return {
            "n_runs": n_runs,
            "mc_sharpe_mean": 0.0,
            "mc_sharpe_p5": 0.0,
            "mc_max_dd_p95": 0.0,
            "mc_profit_factor_mean": 0.0,
        }

    sharpe_vals = []
    dd_vals = []
    pf_vals = []

    for _ in range(n_runs):
        sample = [rng.choice(trades) for __ in range(len(trades))]
        metrics = evaluate_trade_samples(sample, starting_equity=starting_equity)
        sharpe_vals.append(metrics["sharpe"])
        dd_vals.append(metrics["max_drawdown"])
        pf_vals.append(metrics["profit_factor"] if math.isfinite(metrics["profit_factor"]) else 0.0)

    sharpe_arr = np.array(sharpe_vals, dtype=float)
    dd_arr = np.array(dd_vals, dtype=float)
    pf_arr = np.array(pf_vals, dtype=float)

    return {
        "n_runs": n_runs,
        "mc_sharpe_mean": float(np.mean(sharpe_arr)),
        "mc_sharpe_p5": float(np.quantile(sharpe_arr, 0.05)),
        "mc_sharpe_p95": float(np.quantile(sharpe_arr, 0.95)),
        "mc_max_dd_mean": float(np.mean(dd_arr)),
        "mc_max_dd_p95": float(np.quantile(dd_arr, 0.95)),
        "mc_profit_factor_mean": float(np.mean(pf_arr)),
        "mc_profit_factor_p5": float(np.quantile(pf_arr, 0.05)),
    }


def walk_forward_splits(df_len: int, n_splits: int = 5, train_ratio: float = 0.6, test_ratio: float = 0.2) -> List[Tuple[int, int, int]]:
    """Return list of (train_start, train_end, test_end) indices."""
    if df_len < 50:
        return [(0, int(df_len * train_ratio), df_len)]

    splits = []
    step = max(1, int((df_len * (1 - test_ratio)) / n_splits))
    for i in range(n_splits):
        train_start = 0
        train_end = min(df_len - 1, int(df_len * train_ratio) + i * step)
        test_end = min(df_len, train_end + int(df_len * test_ratio))
        if test_end <= train_end:
            break
        splits.append((train_start, train_end, test_end))
    if not splits:
        splits.append((0, int(df_len * train_ratio), df_len))
    return splits


def rolling_window_splits(df_len: int, window: int, step: int) -> List[Tuple[int, int]]:
    if window >= df_len:
        return [(0, df_len)]
    out = []
    for start in range(0, df_len - window + 1, step):
        out.append((start, start + window))
    return out or [(0, df_len)]


def validate_strategy_with_harness(
    trades: List[Dict[str, Any]],
    starting_equity: float = 10_000.0,
    mc_runs: int = 200,
) -> Dict[str, Any]:
    """Convenience wrapper: convert trade dicts into TradeSample and compute metrics."""
    samples: List[TradeSample] = []
    for t in trades:
        pnl = float(t.get("pnl", 0.0))
        samples.append(TradeSample(pnl=pnl, is_win=(pnl > 0), r_multiple=t.get("r_multiple")))

    base_metrics = evaluate_trade_samples(samples, starting_equity=starting_equity)
    mc_metrics = bootstrap_equity_monte_carlo(samples, n_runs=mc_runs, starting_equity=starting_equity)

    return {
        "base_metrics": base_metrics,
        "monte_carlo": mc_metrics,
    }

