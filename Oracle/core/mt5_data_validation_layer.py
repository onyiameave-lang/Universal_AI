"""mt5_data_validation_layer.py

Adaptive MT5 history fetch + validation layer.

Guarantees downstream consumers receive:
- valid raw OHLCV (ordered, de-duplicated, required columns)
- sufficient history for indicator warm-up + RL observation windows
- indicator outputs without excessive NaNs

If not possible, returns explicit statuses and detailed diagnostics.

This module is intentionally reusable by both:
- training pipeline (main.py step_load_data in MT5 mode)
- live trading (live_trader.py MT5MarketData)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from experts.chart_expert import add_technical_indicators, ALL_TIMEFRAMES

from core.indicator_validator import (
    IndicatorValidator,
    ValidationResult,
    VALID,
    RAW_DATA_EMPTY,
    INSUFFICIENT_HISTORY,
    EMPTY_AFTER_DROPNA,
    MISSING_INDICATOR_COLUMNS,
    TOO_MANY_NANS,
    INDICATOR_EXCEPTION,
    CORRUPTED_OHLCV,
)



# ---------------------------------------------------------
# Statuses
# ---------------------------------------------------------

DATA_VALID = "DATA_VALID"
DATA_UNAVAILABLE = "DATA_UNAVAILABLE"
INSUFFICIENT_HISTORY = "INSUFFICIENT_HISTORY"
RAW_INVALID = "RAW_INVALID"
INDICATOR_FAILURE = "INDICATOR_FAILURE"
OBSERVATION_INVALID = "OBSERVATION_INVALID"


@dataclass
class ValidationRequirement:
    """What downstream requires from the history."""

    # RL observation builder (MultiSymbolChartExpert) needs:
    # - entry window of OHLCV values length = window_size*5
    # - indicator features derived at current bar (with warm-up rows dropped)
    # For safety we require at least:
    #   (longest_indicator_lookback - 1) + window_size + warmup_buffer
    window_size: int
    warmup_buffer: int = 50


@dataclass
class FetchProgression:
    # Progression counts passed to MT5 copy_rates...
    counts: List[int]


DEFAULT_PROGRESSION = FetchProgression(
    counts=[500, 1000, 1500, 2500, 5000]
)


def _estimate_longest_indicator_lookback() -> int:
    """Derive from experts/chart_expert.py add_technical_indicators().

    Longest rolling windows:
      - sma_50 => 50
      - bb_upper/mid/lower => 20
      - rsi_14 => 14
      - atr_14 => 14
      - stoch_k => 14

    We also compute other indicators but they use smaller windows.

    Return the max window used among the *required* NaN-dropped columns.
    """

    return 50


def compute_required_history(req: ValidationRequirement) -> int:
    """Compute dynamic required raw candle count.

    Formula (conservative):
        required_raw >= longest_lookback + window_size + warmup_buffer

    We also account for the fact that some indicators may fill early rows
    but add_technical_indicators() ultimately drops rows with:
        sma_50, rsi_14, atr_14, stoch_k
    """

    longest = _estimate_longest_indicator_lookback()
    required = int(longest + req.window_size + req.warmup_buffer)
    # Ensure a minimum lower bound
    return max(required, 200)


def validate_raw_df(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validate raw OHLCV before indicator computation."""

    issues: List[str] = []

    if df is None:
        issues.append("df is None")
        return False, issues

    if not isinstance(df, pd.DataFrame) or df.empty:
        issues.append("df is empty")
        return False, issues

    # Ensure required columns
    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        issues.append(f"missing_columns={missing}")
        return False, issues

    # Timestamp ordering: index should be datetime-like and sorted
    if not isinstance(df.index, pd.DatetimeIndex):
        issues.append("index is not DatetimeIndex")
        return False, issues

    if not df.index.is_monotonic_increasing:
        issues.append("timestamps_not_sorted")
        return False, issues

    # De-duplicate timestamps
    dup_count = int(df.index.duplicated().sum())
    if dup_count > 0:
        issues.append(f"duplicate_timestamps={dup_count}")
        # not fatal; we will de-dup

    # NaNs in OHLCV
    ohlcv = df[required_cols]
    nan_count = int(ohlcv.isna().sum().sum())
    if nan_count > 0:
        issues.append(f"nan_in_ohlcv_total={nan_count}")
        # not fatal; we will coerce and drop later

    return True, issues


def _coerce_and_clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Coerce OHLCV numeric
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Remove duplicates, keep last
    if isinstance(df.index, pd.DatetimeIndex):
        df = df[~df.index.duplicated(keep="last")]

    # Drop rows with any critical NaNs
    df = df.dropna(subset=["Open", "High", "Low", "Close"])

    # Sort
    df = df.sort_index()
    return df


def validate_indicator_output(df_ind: pd.DataFrame, required_rows_min: int) -> Tuple[bool, Dict[str, Any]]:
    """Validate indicator-augmented output."""

    diagnostics: Dict[str, Any] = {}

    if df_ind is None or not isinstance(df_ind, pd.DataFrame) or df_ind.empty:
        diagnostics["rows_after_indicators"] = 0
        return False, diagnostics

    required_indicator_cols = [
        "sma_50", "rsi_14", "atr_14", "stoch_k",
        "macd", "macd_signal", "macd_hist",
        "bb_upper", "bb_mid", "bb_lower",
        "volume_ratio", "body_size", "upper_shadow", "lower_shadow",
        "williams_r",
    ]

    missing = [c for c in required_indicator_cols if c not in df_ind.columns]
    diagnostics["missing_columns"] = missing
    if missing:
        return False, diagnostics

    # NaN percentage on required columns
    nan_total = int(df_ind[required_indicator_cols].isna().sum().sum())
    nan_pct = nan_total / max(len(df_ind) * len(required_indicator_cols), 1)
    diagnostics["nan_total_required_cols"] = nan_total
    diagnostics["nan_pct_required_cols"] = float(nan_pct)

    usable_rows = int(len(df_ind))
    diagnostics["rows_after_indicators"] = usable_rows
    diagnostics["required_rows_min"] = int(required_rows_min)

    if nan_total > 0:
        # add_technical_indicators is designed to drop NaN rows for key columns,
        # so any NaNs here indicate unexpected drift.
        # Treat as invalid if > small threshold.
        if nan_pct > 1e-6:
            return False, diagnostics

    if usable_rows < required_rows_min:
        return False, diagnostics

    return True, diagnostics


def _default_observation_validation(bundle_df: pd.DataFrame, feature_cols: List[str]) -> Tuple[bool, Dict[str, Any]]:
    """For live RLSignalGenerator we need a minimal non-NaN feature vector.

    In chart_expert.MultiSymbolChartExpert, observation creation happens inside
    the gym env; it already handles NaNs by nan_to_num, but it still assumes
    slices are non-empty and columns exist.

    Here we validate minimal required columns for the live observation builder.
    """

    diagnostics: Dict[str, Any] = {}
    if bundle_df is None or bundle_df.empty:
        return False, {"rows": 0}

    missing = [c for c in feature_cols if c not in bundle_df.columns]
    diagnostics["missing_columns"] = missing
    if missing:
        return False, diagnostics

    last = bundle_df.iloc[-1]
    vals = last[feature_cols]
    nan_total = int(vals.isna().sum())
    diagnostics["nan_in_last_feature_vec"] = nan_total
    if nan_total > 0:
        return False, diagnostics

    diagnostics["feature_vector_len"] = len(feature_cols)
    return True, diagnostics


def adaptive_fetch_and_validate(
    symbol: str,
    timeframe: str,
    mt5_getter_fn,
    req: ValidationRequirement,
    progression: FetchProgression = DEFAULT_PROGRESSION,
    max_attempts: Optional[int] = None,
    indicator_fn=add_technical_indicators,
    logger=print,
) -> Dict[str, Any]:
    """Fault-tolerant MT5 history fetch + self-healing indicator validation.

    Recovery loop:
      1) request more MT5 history
      2) recompute indicators (wrapped with exception capture)
      3) revalidate using IndicatorValidator

    Only terminate early for unrecoverable raw issues.
    """






    validator = IndicatorValidator(max_nan_fraction=1e-6, min_usable_rows=10, logger=logger)

    required_history_estimate = compute_required_history(req)
    required_rows_min = max(10, int(req.window_size + req.warmup_buffer / 5))


    counts = progression.counts
    if max_attempts is not None:
        counts = counts[:max_attempts]

    final: Dict[str, Any] = {
        "symbol": symbol,
        "timeframe": timeframe,
        "status": DATA_UNAVAILABLE,
        "diagnostics": {},
    }

    attempts = 0

    for count in counts:
        attempts += 1
        if count < 50:
            continue

        logger(f"    [MT5-VALID] {symbol} {timeframe}: request_count={count} (attempt {attempts})")

        df_raw = mt5_getter_fn(symbol, timeframe, count)
        raw_rows = 0 if df_raw is None else int(len(df_raw))

        raw_ok, raw_issues = validate_raw_df(df_raw)

        diag_base = {
            "requested_candles": int(count),
            "returned_candles": raw_rows,
            "raw_valid": bool(raw_ok),
            "raw_issues": raw_issues,
            "required_history_estimate": int(required_history),
        }

        if not raw_ok:
            # Still attempt to clean if dataframe exists
            if isinstance(df_raw, pd.DataFrame) and not df_raw.empty:
                try:
                    df_raw_clean = _coerce_and_clean_raw(df_raw)
                    if df_raw_clean.empty:
                        diag_base["after_clean_rows"] = 0
                        final["diagnostics"] = diag_base
                        continue
                    df_raw = df_raw_clean
                    diag_base["after_clean_rows"] = int(len(df_raw))
                except Exception:
                    diag_base["after_clean_rows"] = 0
                    final["diagnostics"] = diag_base
                    continue
            else:
                final["diagnostics"] = diag_base
                continue

        if isinstance(df_raw, pd.DataFrame):
            df_raw = _coerce_and_clean_raw(df_raw)

        if df_raw is None or df_raw.empty:
            final["diagnostics"] = diag_base
            continue

        # Indicator computation
        try:
            df_ind = indicator_fn(df_raw)
        except Exception as e:
            diag = dict(diag_base)
            diag["indicator_exception"] = f"{type(e).__name__}: {e}"
            final["status"] = INDICATOR_FAILURE
            final["diagnostics"] = diag
            continue

        # Validate indicator output
        ok_ind, ind_diag = validate_indicator_output(df_ind, required_rows_min=required_rows_min)

        # Build final log diagnostics similar to requirements
        diag = dict(diag_base)
        diag.update(ind_diag)
        diag["required_rows_effective"] = int(required_rows_min)
        
        logger(
            "      Raw candles: {raw_rows}\n"
            "      Rows after indicators: {rows_after}\n"
            "      Required (effective): {req_rows}\n"
            "      Status: {status}".format(
                raw_rows=raw_rows,
                rows_after=ind_diag.get("rows_after_indicators", 0),
                req_rows=required_rows_min,
                status=(DATA_VALID if ok_ind else INSUFFICIENT_HISTORY),
            )
        )

        if ok_ind:
            # Minimal live observation sanity check (for live trader's feature set)
            live_feature_cols = [
                "rsi_14", "atr_14", "macd", "macd_signal", "stoch_k",
                "stoch_d", "sma_50",
            ]
            obs_ok, obs_diag = _default_observation_validation(df_ind, live_feature_cols)
            diag["observation_validation"] = obs_diag

            if not obs_ok:
                final = {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "status": OBSERVATION_INVALID,
                    "data": None,
                    "diagnostics": diag,
                    "attempts": attempts,
                }
                return final

            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "status": DATA_VALID,
                "data": df_ind,
                "diagnostics": diag,
                "attempts": attempts,
            }

        # Not enough usable rows -> try next larger request
        final = {
            "symbol": symbol,
            "timeframe": timeframe,
            "status": INSUFFICIENT_HISTORY,
            "data": None,
            "diagnostics": diag,
            "attempts": attempts,
        }

    return final


def make_mt5_getter(mt5_get_mt5_data_fn):
    """Adapt get_mt5_data(symbol, timeframe, count) into a uniform callable."""

    def _getter(symbol: str, timeframe: str, count: int) -> Optional[pd.DataFrame]:
        return mt5_get_mt5_data_fn(symbol, timeframe, count)

    return _getter

