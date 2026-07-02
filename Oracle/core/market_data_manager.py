"""market_data_manager.py

MarketDataManager centralizes all MT5 communication, adaptive history
acquisition, raw/indicator/observation validation, and MemoryAI
self-optimization.

Downstream components must treat MarketDataManager as the single source of
truth for validated market datasets.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Callable

import pandas as pd

from core.mt5_data_validation_layer import (
    ValidationRequirement,
    DEFAULT_PROGRESSION,
    DATA_VALID,
)

from core.mt5_data_validation_layer_v2 import adaptive_fetch_and_validate_v2


from experts.mt5_expert import connect_mt5, get_mt5_data

from experts.chart_expert import add_technical_indicators


@dataclass
class FetchRecommendation:
    requested_history: int
    timestamp: str
    status: str


class MarketDataManager:
    """Single entry point for validated historical market data."""

    def __init__(
        self,
        memory_ai: Optional[Any] = None,
        mt5_connector: Callable[[], bool] = connect_mt5,
        mt5_getter: Callable[[str, str, int], Optional[pd.DataFrame]] = get_mt5_data,
        indicator_fn: Callable[[pd.DataFrame], pd.DataFrame] = None,
        logger=print,
    ):
        self.memory_ai = memory_ai
        self.mt5_getter = mt5_getter
        self.logger = logger

        # Ensure MT5 connectivity only here.
        self._connected = False
        self._mt5_connector = mt5_connector
        self._connect_if_needed()

        if indicator_fn is not None:
            # adaptive_fetch_and_validate takes indicator_fn as kwarg.
            self.indicator_fn = indicator_fn
        else:
            self.indicator_fn = None

    def _connect_if_needed(self) -> None:
        if self._connected:
            return
        if self._mt5_connector is not None:
            ok = bool(self._mt5_connector())
            if not ok:
                raise RuntimeError("Could not connect to MT5 terminal")
        self._connected = True

    # ------------------------------
    # MemoryAI integration (minimal)
    # ------------------------------

    def _memoryai_learn_fetch(
        self,
        *,
        symbol: str,
        timeframe: str,
        requested_history: int,
        returned_history: int,
        usable_history: int,
        warmup_loss: int,
        indicator_loss: int,
        recommended_history: int,
        status: str,
        timestamp: Optional[str] = None,
        attempts: int = 0,
        diagnostics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Best-effort MemoryAI persistence.

        This repository contains multiple MemoryAI implementations with
        differing method names. We keep this defensive and non-blocking.
        """

        if self.memory_ai is None:
            return

        ts = timestamp or datetime.now().isoformat()

        payload = {
            "symbol": symbol,
            "timeframe": timeframe,
            "requested_history": requested_history,
            "returned_history": returned_history,
            "usable_history": usable_history,
            "warmup_loss": warmup_loss,
            "indicator_loss": indicator_loss,
            "recommended_history": recommended_history,
            "timestamp": ts,
            "status": status,
            "attempts": attempts,
            "diagnostics": diagnostics or {},
        }

        try:
            # OPTIMIZED_memory_ai_system.py tends to expose concept/learn-like APIs.
            # We store under a consistent concept namespace.
            concept = f"mt5_fetch_profile::{symbol}::{timeframe}"
            if hasattr(self.memory_ai, "learn_concept"):
                self.memory_ai.learn_concept(
                    domain="market_data",
                    concept=concept,
                    source_text=str(payload),
                    agent_name="market_oracle_data_manager",
                    confidence=1.0,
                )
                return
            if hasattr(self.memory_ai, "log_concept"):
                self.memory_ai.log_concept(
                    domain="market_data",
                    concept=concept,
                    source_text=str(payload),
                    agent_name="market_oracle_data_manager",
                    confidence=1.0,
                )
                return
        except Exception as e:
            self.logger(f"    ⚠️ MemoryAI learn failed: {e}")

    # ------------------------------
    # Public API
    # ------------------------------

    def get_validated_bundle(
        self,
        symbol: str,
        timeframes: List[str],
        *,
        window_size: int,
        warmup_buffer: int = 50,
        progression=None,
        max_attempts: Optional[int] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Return a dict {timeframe: indicator_df} with validated indicators.

        Raises RuntimeError if no timeframe can be validated.
        """

        if not timeframes:
            raise ValueError("timeframes must be non-empty")

        progression = progression or DEFAULT_PROGRESSION
        req = ValidationRequirement(window_size=window_size, warmup_buffer=warmup_buffer)

        bundle: Dict[str, pd.DataFrame] = {}

        for tf in timeframes:
            result = adaptive_fetch_and_validate_v2(

                symbol=symbol,
                timeframe=tf,
                mt5_getter_fn=self.mt5_getter,
                req=req,
                progression=progression,
                max_attempts=max_attempts,
                indicator_fn=(self.indicator_fn or None),
                logger=self._log_adaptive,
            )

            status = result.get("status")
            if status == DATA_VALID and result.get("data") is not None:
                df = result["data"]
                bundle[tf] = df

                # Learning inputs (best-effort; compute what we can)
                requested = int(result.get("diagnostics", {}).get("requested_candles", 0) or 0)
                returned = int(result.get("diagnostics", {}).get("returned_candles", 0) or 0)
                rows_after = int(result.get("diagnostics", {}).get("rows_after_indicators", 0) or 0)
                required_rows_min = int(result.get("diagnostics", {}).get("required_rows_min", 0) or 0)
                warmup_loss = max(returned - rows_after, 0)
                indicator_loss = max(required_rows_min - rows_after, 0)

                # Simple recommendation: use returned if valid, else fallback to returned.
                recommended = max(returned, requested)

                self._memoryai_learn_fetch(
                    symbol=symbol,
                    timeframe=tf,
                    requested_history=requested,
                    returned_history=returned,
                    usable_history=rows_after,
                    warmup_loss=warmup_loss,
                    indicator_loss=indicator_loss,
                    recommended_history=recommended,
                    status=status,
                    attempts=int(result.get("attempts", 0) or 0),
                    diagnostics=result.get("diagnostics", {}),
                )
            else:
                self._structured_log_failure(symbol, tf, result)

        if not bundle:
            raise RuntimeError(f"No validated timeframes for {symbol}")

        return bundle

    def _log_adaptive(self, msg: str) -> None:
        self.logger(msg)

    def _structured_log_failure(self, symbol: str, timeframe: str, result: Dict[str, Any]) -> None:
        diag = result.get("diagnostics", {}) if isinstance(result.get("diagnostics"), dict) else {}
        self.logger(
            "================================\n"
            f"{symbol} {timeframe}\n"
            f"Requested: {diag.get('requested_candles', 'NA')}\n"
            f"Returned: {diag.get('returned_candles', 'NA')}\n"
            f"Rows after indicators: {diag.get('rows_after_indicators', 'NA')}\n"
            f"Required rows (effective): {diag.get('required_rows_effective', 'NA')}\n"
            f"Status: {result.get('status')}\n"
            f"Attempts: {result.get('attempts', 'NA')}\n"
            "================================"
        )

