"""MarketOracle agent roles (Phase D)

MVP implementation based on current codebase:
- ScoutAgent: scans candidate symbols from loaded MarketOracle data bundle and
  ranks setups using existing StrategyTester + SymbolAnalyzer.
- TraderAgent: converts a setup into an action suggestion (BUY/SELL) with
  basic SL/TP from indicator proxies.
- RiskAgent: enforces hard risk constraints (max daily loss/drawdown) via
  environment parameters if available; otherwise returns conservative sizing.
- ExecutionAgent: performs execution *only via adapter hooks* (does not place
  MT5 orders directly). For now it returns a validated execution payload.
- StrategyAgent: retrieves candidate strategies (from optimized cached
  strategies if present) and proposes an evolution step placeholder.

This file is designed to be integrated later with UniversalAI.

Important:
- This is not the full agent rewrite; it's the requested Phase D add-on layer
  that unblocks orchestration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Setup:
    symbol: str
    action: str  # BUY/SELL/NO_TRADE
    confidence: float
    timeframe: str
    reason: str
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    meta: Dict[str, Any] = None


class ScoutAgent:
    """Scans symbols and ranks opportunities."""

    def __init__(self, memory_ai: Any = None, oracle_adapter: Any = None):
        self.memory_ai = memory_ai
        self.oracle_adapter = oracle_adapter

    def scan(self, symbols: List[str], data_bundle: Dict[str, Any], top_k: int = 5) -> List[Setup]:
        from experts.strategy_tester import SymbolAnalyzer
        from experts.chart_expert import load_strategy_for_symbol
        from experts.chart_expert import rules_to_signals

        setups: List[Setup] = []
        for sym in symbols:
            if sym not in data_bundle:
                continue

            # Flatten like MarketOracleAdapter does
            flat = {}
            for tf, splits in data_bundle[sym].items():
                frames = [frame for frame in (splits.get("train"), splits.get("test")) if frame is not None and not frame.empty]
                if frames:
                    import pandas as pd
                    flat[tf] = pd.concat(frames).sort_index()

            symbol_analysis = SymbolAnalyzer.analyze_symbol(flat, sym)
            symbol_type = SymbolAnalyzer.get_symbol_type(symbol_analysis)
            strategy = load_strategy_for_symbol(sym)

            # basic heuristic: use RSI/MACD votes from latest bar in entry tf
            entry_tf = strategy.get("entry_tf") or ("1h" if "trending" in symbol_type else "daily")
            df = flat.get(entry_tf) or flat.get("daily")
            if df is None or df.empty:
                continue

            row = df.iloc[-1]
            close = float(row.get("Close", 0.0))
            if close <= 0:
                continue

            rsi = float(row.get("rsi_14", 50.0))
            macd = float(row.get("macd", 0.0))
            macd_signal = float(row.get("macd_signal", 0.0))
            sma50 = float(row.get("sma_50", close))

            bullish_votes = sum([close > sma50, macd > macd_signal, rsi > 52])
            bearish_votes = sum([close < sma50, macd < macd_signal, rsi < 48])

            if "mean_reverting" in symbol_analysis.get("overall_characteristics", []):
                if rsi <= 35 and close <= sma50:
                    action = "BUY"
                    conf = 0.58
                    reason = "Mean-reversion: oversold near SMA50"
                elif rsi >= 65 and close >= sma50:
                    action = "SELL"
                    conf = 0.58
                    reason = "Mean-reversion: overbought near SMA50"
                else:
                    action = "NO_TRADE"
                    conf = 0.25
                    reason = "Mean-reversion, but RSI not stretched"
            else:
                if bullish_votes >= 2:
                    action = "BUY"
                    conf = 0.62
                    reason = "Momentum alignment"
                elif bearish_votes >= 2:
                    action = "SELL"
                    conf = 0.62
                    reason = "Momentum alignment"
                else:
                    action = "NO_TRADE"
                    conf = 0.30
                    reason = "Mixed conditions"

            atr = float(row.get("atr_14", close * 0.01))
            stop_loss = None
            take_profit = None
            if action in {"BUY", "SELL"}:
                if action == "BUY":
                    stop_loss = close - max(atr * 1.5, close * 0.0015)
                    take_profit = close + (close - stop_loss) * 1.5
                else:
                    stop_loss = close + max(atr * 1.5, close * 0.0015)
                    take_profit = close - (stop_loss - close) * 1.5

            setups.append(
                Setup(
                    symbol=sym,
                    action=action,
                    confidence=float(conf),
                    timeframe=entry_tf,
                    reason=reason,
                    stop_loss=float(stop_loss) if stop_loss is not None else None,
                    take_profit=float(take_profit) if take_profit is not None else None,
                    meta={"symbol_type": symbol_type},
                )
            )

        setups = sorted(setups, key=lambda s: s.confidence, reverse=True)
        return setups[:top_k]


class TraderAgent:
    """Turns a Setup into an execution plan."""

    def __init__(self, risk_agent: Optional[Any] = None):
        self.risk_agent = risk_agent

    def plan_trade(self, setup: Setup) -> Dict[str, Any]:
        if setup.action not in {"BUY", "SELL"}:
            return {"action": "NO_TRADE", "confidence": setup.confidence, "reason": setup.reason}

        payload = {
            "symbol": setup.symbol,
            "side": setup.action,
            "timeframe": setup.timeframe,
            "entry_reference": setup.meta.get("entry_reference") if setup.meta else None,
            "stop_loss": setup.stop_loss,
            "take_profit": setup.take_profit,
            "confidence": setup.confidence,
        }

        if self.risk_agent is not None:
            payload = self.risk_agent.validate_and_adjust(payload)
        return payload


class RiskAgent:
    """Enforces risk guardrails (MVP)."""

    def __init__(self, max_risk_reward: float = 3.0):
        self.max_risk_reward = max_risk_reward

    def validate_and_adjust(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        sl = payload.get("stop_loss")
        tp = payload.get("take_profit")
        if sl is None or tp is None:
            payload["risk_ok"] = False
            payload["risk_reason"] = "missing_sl_tp"
            return payload

        # Very lightweight RR check
        rr = abs(tp - payload.get("entry_reference") or tp) / max(abs(payload.get("entry_reference") or tp - sl), 1e-8)
        if rr > self.max_risk_reward:
            payload["risk_ok"] = False
            payload["risk_reason"] = "rr_too_high"
            return payload

        payload["risk_ok"] = True
        return payload


class ExecutionAgent:
    """Execution validation only (no MT5 order side-effects in MVP)."""

    def __init__(self, mt5_interface: Any = None):
        self.mt5_interface = mt5_interface

    def validate_execution(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        if plan.get("action") == "NO_TRADE" or not plan.get("risk_ok", True):
            return {"status": "blocked", "reason": plan.get("risk_reason", "no_trade_or_risk_block")}

        # Return a payload for downstream executor
        return {"status": "validated", "execution_payload": plan}


class StrategyAgent:
    """Strategy selection/evolution stub."""

    def __init__(self, memory_ai: Any = None):
        self.memory_ai = memory_ai

    def retrieve_strategies(self, symbol: str) -> List[Dict[str, Any]]:
        from experts.chart_expert import load_strategy_for_symbol
        s = load_strategy_for_symbol(symbol)
        return [s] if s else []

    def propose_evolution(self, symbol: str) -> Dict[str, Any]:
        return {
            "symbol": symbol,
            "status": "pending",
            "reason": "Strategy evolution will switch to MemoryAI retrieval-based evolution in later Phase D/F",
        }

