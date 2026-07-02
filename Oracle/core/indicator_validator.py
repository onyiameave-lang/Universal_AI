from __future__ import annotations

"""indicator_validator.py

Fault-tolerant indicator validation and recovery classification.

This module introduces:
- Failure taxonomy with human-readable explanations
- IndicatorValidator class that validates:
  * raw OHLCV dataframe
  * indicator generation output
  * required indicator columns
  * NaN levels
  * usable row count after indicator warm-up drops
  * dropna losses (before/after)

The validator also recommends recovery actions and whether the failure
is recoverable or unrecoverable.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# Status taxonomy (replace generic INDICATOR_FAILURE)
RAW_DATA_EMPTY = "RAW_DATA_EMPTY"
MT5_FETCH_FAILED = "MT5_FETCH_FAILED"
INSUFFICIENT_HISTORY = "INSUFFICIENT_HISTORY"
INDICATOR_EXCEPTION = "INDICATOR_EXCEPTION"
MISSING_INDICATOR_COLUMNS = "MISSING_INDICATOR_COLUMNS"
TOO_MANY_NANS = "TOO_MANY_NANS"
EMPTY_AFTER_DROPNA = "EMPTY_AFTER_DROPNA"
OBSERVATION_TOO_SHORT = "OBSERVATION_TOO_SHORT"
VALID = "VALID"

# Unrecoverable errors
INVALID_SYMBOL = "INVALID_SYMBOL"
MT5_UNAVAILABLE = "MT5_UNAVAILABLE"
CORRUPTED_OHLCV = "CORRUPTED_OHLCV"
REPEATED_INDICATOR_EXCEPTION = "REPEATED_INDICATOR_EXCEPTION"
REPEATED_MT5_FAILURE = "REPEATED_MT5_FAILURE"


@dataclass
class ValidationResult:
    status: str
    ok: bool
    explanation: str
    diagnostics: Dict[str, Any]
    recoverable: bool
    recovery_actions: List[str]


class IndicatorValidator:
    """Validates indicator generation for MarketOracle.

    Expected indicator columns are aligned with experts/chart_expert.py.
    """

    REQUIRED_RAW_COLS = ["Open", "High", "Low", "Close", "Volume"]

    # These names match chart_expert.add_technical_indicators output.
    REQUIRED_INDICATOR_COLS = [
        "sma_50",
        "rsi_14",
        "atr_14",
        "macd",
        "macd_signal",
        "macd_hist",
        "stoch_k",
        "stoch_d",
    ]

    # Additional columns that chart_expert uses for observation
    # (not all are strictly required, but help detect unexpected issues)
    OPTIONAL_INDICATOR_COLS = [
        "williams_r",
        "volume_ratio",
        "body_size",
        "upper_shadow",
        "lower_shadow",
        "bb_upper",
        "bb_lower",
        "bb_mid",
    ]

    def __init__(
        self,
        *,
        required_indicator_cols: Optional[List[str]] = None,
        required_raw_cols: Optional[List[str]] = None,
        max_nan_fraction: float = 1e-6,
        min_usable_rows: int = 10,
        logger=print,
    ):
        self.required_indicator_cols = required_indicator_cols or list(
            self.REQUIRED_INDICATOR_COLS
        )
        self.required_raw_cols = required_raw_cols or list(self.REQUIRED_RAW_COLS)
        self.max_nan_fraction = float(max_nan_fraction)
        self.min_usable_rows = int(min_usable_rows)
        self.logger = logger

    def _nan_fraction(self, df: pd.DataFrame, cols: List[str]) -> float:
        if df is None or df.empty:
            return 1.0
        denom = max(len(df) * max(len(cols), 1), 1)
        nan_total = int(df[cols].isna().sum().sum()) if cols else 0
        return nan_total / denom

    def validate_raw_df(
        self,
        df_raw: Optional[pd.DataFrame],
        *,
        requested_candles: Optional[int] = None,
    ) -> ValidationResult:
        if df_raw is None:
            return ValidationResult(
                status=RAW_DATA_EMPTY,
                ok=False,
                explanation="Raw dataframe is None.",
                diagnostics={"requested_candles": requested_candles},
                recoverable=True,
                recovery_actions=["Fetch more MT5 history"],
            )

        if not isinstance(df_raw, pd.DataFrame) or df_raw.empty:
            return ValidationResult(
                status=RAW_DATA_EMPTY,
                ok=False,
                explanation="Raw dataframe is empty.",
                diagnostics={"requested_candles": requested_candles},
                recoverable=True,
                recovery_actions=["Fetch more MT5 history"],
            )

        missing = [c for c in self.required_raw_cols if c not in df_raw.columns]
        if missing:
            return ValidationResult(
                status=CORRUPTED_OHLCV,
                ok=False,
                explanation=f"Missing raw OHLCV columns: {missing}",
                diagnostics={"missing_columns": missing},
                recoverable=False,
                recovery_actions=[],
            )

        # Basic sanity: at least one non-null close
        if df_raw["Close"].notna().sum() <= 1:
            return ValidationResult(
                status=CORRUPTED_OHLCV,
                ok=False,
                explanation="Close series contains too few non-null values.",
                diagnostics={"close_non_null": int(df_raw["Close"].notna().sum())},
                recoverable=False,
                recovery_actions=[],
            )

        return ValidationResult(
            status=VALID,
            ok=True,
            explanation="Raw dataframe validation passed.",
            diagnostics={"rows": int(len(df_raw)), "requested_candles": requested_candles},
            recoverable=True,
            recovery_actions=[],
        )

    def validate_after_indicator_generation(
        self,
        *,
        df_ind: Optional[pd.DataFrame],
        df_raw_before_dropna: Optional[pd.DataFrame],
        df_raw_after_dropna: Optional[pd.DataFrame],
        required_rows_min: int,
    ) -> ValidationResult:
        diagnostics: Dict[str, Any] = {}

        # Dropna loss diagnostics (pre indicator computation)
        if df_raw_before_dropna is not None and isinstance(df_raw_before_dropna, pd.DataFrame):
            diagnostics["rows_before_clean"] = int(len(df_raw_before_dropna))
        if df_raw_after_dropna is not None and isinstance(df_raw_after_dropna, pd.DataFrame):
            diagnostics["rows_after_clean"] = int(len(df_raw_after_dropna))

        if df_ind is None or not isinstance(df_ind, pd.DataFrame) or df_ind.empty:
            return ValidationResult(
                status=EMPTY_AFTER_DROPNA,
                ok=False,
                explanation="Indicator output is empty after indicator generation / NaN dropping.",
                diagnostics={"rows_after_indicators": 0, **diagnostics},
                recoverable=True,
                recovery_actions=[
                    "Fetch more MT5 history (warm-up losses)",
                    "Recompute indicators",
                ],
            )

        missing = [c for c in self.required_indicator_cols if c not in df_ind.columns]
        diagnostics["missing_indicator_columns"] = missing
        diagnostics["rows_after_indicators"] = int(len(df_ind))

        if missing:
            return ValidationResult(
                status=MISSING_INDICATOR_COLUMNS,
                ok=False,
                explanation=f"Missing required indicator columns: {missing}",
                diagnostics=diagnostics,
                recoverable=True,
                recovery_actions=[
                    "Recalculate indicators",
                    "Verify indicator function mapping / naming",
                ],
            )

        nan_total = int(df_ind[self.required_indicator_cols].isna().sum().sum())
        nan_fraction = self._nan_fraction(df_ind, self.required_indicator_cols)
        diagnostics["nan_total_required_cols"] = nan_total
        diagnostics["nan_fraction_required_cols"] = float(nan_fraction)

        if nan_total > 0 and nan_fraction > self.max_nan_fraction:
            return ValidationResult(
                status=TOO_MANY_NANS,
                ok=False,
                explanation=(
                    "Too many NaNs in required indicator columns. "
                    f"nan_total={nan_total}, nan_fraction={nan_fraction}"
                ),
                diagnostics=diagnostics,
                recoverable=True,
                recovery_actions=[
                    "Fetch more MT5 history (indicator warm-up)",
                    "Recalculate indicators",
                ],
            )

        usable_rows = int(len(df_ind))
        diagnostics["required_rows_min"] = int(required_rows_min)

        if usable_rows < required_rows_min:
            return ValidationResult(
                status=INSUFFICIENT_HISTORY,
                ok=False,
                explanation=(
                    f"Not enough usable rows after indicator warm-up. "
                    f"usable_rows={usable_rows}, required_rows_min={required_rows_min}"
                ),
                diagnostics=diagnostics,
                recoverable=True,
                recovery_actions=[
                    "Fetch more MT5 history",
                    "Recompute indicators",
                ],
            )

        return ValidationResult(
            status=VALID,
            ok=True,
            explanation="Indicator output validation passed.",
            diagnostics=diagnostics,
            recoverable=True,
            recovery_actions=[],
        )

