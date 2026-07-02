"""Regime detector (Phase D) for MarketOracle.

MVP implementation:
- Detects a small set of regimes from OHLCV + indicators already computed
  by experts/chart_expert.py / strategy_tester.py environments.
- Persists regime history to MemoryAI (knowledge_records + relationships).

This is an add-on module used by the future MarketOracle redesign.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class Regime:
    label: str
    confidence: float
    features: Dict[str, float]


class MarketRegimeDetector:
    """Detect market regime for a single symbol/timeframe."""

    def __init__(
        self,
        trend_threshold: float = 0.005,
        breakout_atr_multiplier: float = 1.2,
        high_vol_quantile: float = 0.7,
        low_vol_quantile: float = 0.3,
    ):
        self.trend_threshold = trend_threshold
        self.breakout_atr_multiplier = breakout_atr_multiplier
        self.high_vol_quantile = high_vol_quantile
        self.low_vol_quantile = low_vol_quantile

    def detect_last(self, df: pd.DataFrame) -> Regime:
        """Detect regime for the most recent row of df."""
        if df is None or df.empty:
            return Regime("unknown", 0.0, {})

        df = df.copy()
        if "atr_14" not in df.columns:
            df["atr_14"] = (df["High"] - df["Low"]).rolling(14, min_periods=1).mean()
        if "sma_50" not in df.columns:
            df["sma_50"] = df["Close"].rolling(50, min_periods=1).mean()

        row = df.iloc[-1]
        close = float(row.get("Close", 0.0))
        if close == 0.0:
            return Regime("unknown", 0.0, {})

        sma50 = float(row.get("sma_50", close))
        atr = float(row.get("atr_14", close * 0.01))
        atr_pct = atr / max(close, 1e-8)

        # Trend proxy
        trend_strength = abs(close - sma50) / max(close, 1e-8)
        is_trending = trend_strength > self.trend_threshold

        # Volatility proxy
        atr_series = df["atr_14"].astype(float)
        atr_pct_series = atr_series / max(close, 1e-8)
        q_high = float(np.nanquantile(atr_pct_series, self.high_vol_quantile)) if len(atr_pct_series) > 5 else atr_pct
        q_low = float(np.nanquantile(atr_pct_series, self.low_vol_quantile)) if len(atr_pct_series) > 5 else atr_pct
        is_high_vol = atr_pct > q_high
        is_low_vol = atr_pct < q_low

        # Breakout proxy: ATR spike relative to recent mean
        recent_atr_mean = float(atr_series.iloc[max(0, len(df) - 60):].mean())
        is_breakout = (atr / max(recent_atr_mean, 1e-8)) > self.breakout_atr_multiplier

        # Decide regime label
        if is_breakout and is_high_vol:
            label = "breakout"
            conf = min(1.0, 0.4 + 0.4 * (atr_pct / max(q_high, 1e-8)))
        elif is_trending and is_high_vol:
            label = "trend_high_vol"
            conf = min(1.0, 0.3 + 0.4 * (trend_strength / (self.trend_threshold * 2)))
        elif is_trending:
            label = "trend"
            conf = min(1.0, 0.25 + 0.5 * (trend_strength / (self.trend_threshold * 2)))
        elif is_low_vol:
            label = "accumulation_low_vol"
            conf = 0.55
        elif not is_trending and not is_high_vol:
            label = "ranging"
            conf = 0.45
        else:
            label = "neutral"
            conf = 0.35

        features = {
            "close": close,
            "sma50": sma50,
            "trend_strength": float(trend_strength),
            "atr_pct": float(atr_pct),
            "is_trending": float(1.0 if is_trending else 0.0),
            "is_high_vol": float(1.0 if is_high_vol else 0.0),
            "is_breakout": float(1.0 if is_breakout else 0.0),
        }
        return Regime(label=label, confidence=float(conf), features=features)


def persist_regime_history(
    memory_ai: Any,
    knowledge_records_domain: str,
    symbol: str,
    timeframe: str,
    regime_history: List[Dict[str, Any]],
    source: str = "phase_d_regime_detector",
) -> Dict[str, Any]:
    """Persist regime history to MemoryAI.

    MVP strategy:
    - store one knowledge_record per detected regime snapshot.
    - relationship inference is left for later Phase D/E.

    regime_history item format:
      {"timestamp": ISO, "regime": str, "confidence": float, "features": {...}}
    """

    stored = 0
    if memory_ai is None:
        return {"status": "no_memory_ai", "stored": 0}

    for idx, item in enumerate(regime_history):
        ts = item.get("timestamp") or datetime.utcnow().isoformat() + "Z"
        regime = item.get("regime") or "unknown"
        conf = float(item.get("confidence", 0.0))
        features = item.get("features") or {}
        summary = f"{symbol} {timeframe} regime={regime} conf={conf:.2f}"
        embedding = []

        knowledge_id = f"{symbol}_{timeframe}_regime_{idx}"[:64]

        if hasattr(memory_ai, "store_knowledge_record"):
            memory_ai.store_knowledge_record(
                knowledge_id=knowledge_id,
                domain=knowledge_records_domain,
                source=source,
                title=f"{symbol}::{timeframe}::{regime}",
                author="",
                category="market_regime",
                symbol=symbol,
                strategy_type="",
                market_regime=regime,
                importance_score=conf,
                embedding=embedding,
                summary=summary,
                relationships=[],
            )
            stored += 1

    return {"status": "stored", "stored": stored}

