"""mt5_data_validation_layer_v2.py

Self-healing, fault-tolerant MT5 history acquisition + indicator validation.

This is a drop-in v2 replacement for the legacy logic in
core/mt5_data_validation_layer.py.

Key behaviors:
- Uses IndicatorValidator for detailed failure classification
- Wraps indicator generation with exception + stack trace capture
- Recovery loop retries recoverable failures by requesting more MT5 history
- Stops only after exhaustion or unrecoverable classification
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

from core.indicator_validator import IndicatorValidator
from core.mt5_data_validation_layer import (
    ValidationRequirement,
    FetchProgression,
    DEFAULT_PROGRESSION,
    compute_required_history,
    make_mt5_getter,
    OBSERVATION_INVALID,
    DATA_UNAVAILABLE,
    DATA_VALID,
    INDICATOR_FAILURE,
)


def adaptive_fetch_and_validate_v2(
    symbol: str,
    timeframe: str,
    mt5_getter_fn,
    req: ValidationRequirement,
    progression: FetchProgression = DEFAULT_PROGRESSION,
    max_attempts: Optional[int] = None,
    indicator_fn=None,
    logger=print,
    *,
    max_nan_fraction: float = 1e-6,
    min_usable_rows_fallback: int = 10,
) -> Dict[str, Any]:
    """Fault-tolerant MT5 history fetch + indicator self-healing."""

    if indicator_fn is None:
        # Late import to avoid circulars
        from experts.chart_expert import add_technical_indicators as indicator_fn

    validator = IndicatorValidator(
        max_nan_fraction=max_nan_fraction,
        min_usable_rows=min_usable_rows_fallback,
        logger=logger,
    )

    required_history_estimate = compute_required_history(req)
    required_rows_min = max(10, int(req.window_size + req.warmup_buffer / 5))

    counts = list(progression.counts)
    if max_attempts is not None:
        counts = counts[:max_attempts]

    attempts = 0
    last_result: Dict[str, Any] = {
        "symbol": symbol,
        "timeframe": timeframe,
        "status": DATA_UNAVAILABLE,
        "data": None,
        "diagnostics": {},
        "attempts": 0,
    }

    for count in counts:
        if count < 50:
            continue
        attempts += 1

        logger(f"    [MT5-VALID-v2] {symbol} {timeframe}: request_count={count} (attempt {attempts})")

        df_raw = mt5_getter_fn(symbol, timeframe, count)
        returned_candles = 0 if df_raw is None else int(len(df_raw))

        raw_res = validator.validate_raw_df(df_raw, requested_candles=count)

        diag_base: Dict[str, Any] = {
            "requested_candles": int(count),
            "returned_candles": int(returned_candles),
            "required_history_estimate": int(required_history_estimate),
            "raw_validation_status": raw_res.status,
            "raw_valid": raw_res.ok,
            "raw_explanation": raw_res.explanation,
            "raw_diagnostics": raw_res.diagnostics,
        }

        # Best-effort cleaning only for recoverable raw issues
        if (not raw_res.ok) and raw_res.recoverable and isinstance(df_raw, pd.DataFrame) and not df_raw.empty:
            try:
                df_raw = df_raw.copy()
                for c in ["Open", "High", "Low", "Close", "Volume"]:
                    if c in df_raw.columns:
                        df_raw[c] = pd.to_numeric(df_raw[c], errors="coerce")
                if isinstance(df_raw.index, pd.DatetimeIndex):
                    df_raw = df_raw[~df_raw.index.duplicated(keep="last")]
                df_raw = df_raw.dropna(subset=["Open", "High", "Low", "Close"]).sort_index()
                diag_base["after_clean_rows"] = int(len(df_raw))
            except Exception:
                diag_base["after_clean_rows"] = 0

        if not raw_res.ok and raw_res.recoverable is False:
            last_result = {
                "symbol": symbol,
                "timeframe": timeframe,
                "status": raw_res.status,
                "data": None,
                "diagnostics": diag_base,
                "attempts": attempts,
            }
            return last_result

        if df_raw is None or not isinstance(df_raw, pd.DataFrame) or df_raw.empty:
            last_result = {
                "symbol": symbol,
                "timeframe": timeframe,
                "status": raw_res.status,
                "data": None,
                "diagnostics": diag_base,
                "attempts": attempts,
            }
            continue

        df_raw_before = df_raw.copy()
        rows_before = int(len(df_raw_before))

        # Indicator generation wrapped
        try:
            df_ind = indicator_fn(df_raw)
        except Exception as e:
            import traceback

            last_result = {
                "symbol": symbol,
                "timeframe": timeframe,
                "status": INDICATOR_FAILURE,
                "data": None,
                "diagnostics": {
                    **diag_base,
                    "indicator_generation_exception_type": type(e).__name__,
                    "indicator_generation_exception_message": str(e),
                    "stack_trace": traceback.format_exc(),
                },
                "attempts": attempts,
            }
            continue

        # Revalidate with detailed taxonomy
        ind_res = validator.validate_after_indicator_generation(
            df_ind=df_ind,
            df_raw_before_dropna=df_raw_before,
            df_raw_after_dropna=df_raw,
            required_rows_min=required_rows_min,
        )

        rows_after = int(len(df_ind)) if isinstance(df_ind, pd.DataFrame) else 0

        diagnostics: Dict[str, Any] = {
            **diag_base,
            **ind_res.diagnostics,
            "rows_before_indicators": rows_before,
            "rows_after_indicators": rows_after,
            "dropna_rows_removed": max(rows_before - rows_after, 0),
            "required_rows_effective": int(required_rows_min),
            "validation_status": ind_res.status,
            "validation_ok": ind_res.ok,
            "validation_explanation": ind_res.explanation,
            "recovery_actions": ind_res.recovery_actions,
        }

        logger(
            "      Raw candles: {rows_before}\n"
            "      Rows after indicators: {rows_after}\n"
            "      Required (effective): {req_rows}\n"
            "      Status: {status}\n"
            "      Explanation: {exp}".format(
                rows_before=rows_before,
                rows_after=rows_after,
                req_rows=required_rows_min,
                status=ind_res.status,
                exp=ind_res.explanation,
            )
        )

        if ind_res.ok:
            # Minimal live observation sanity check
            from core.mt5_data_validation_layer import _default_observation_validation

            live_feature_cols = [
                "rsi_14",
                "atr_14",
                "macd",
                "macd_signal",
                "stoch_k",
                "stoch_d",
                "sma_50",
            ]
            obs_ok, obs_diag = _default_observation_validation(df_ind, live_feature_cols)
            diagnostics["observation_validation"] = obs_diag

            if not obs_ok:
                return {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "status": OBSERVATION_INVALID,
                    "data": None,
                    "diagnostics": diagnostics,
                    "attempts": attempts,
                }

            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "status": DATA_VALID,
                "data": df_ind,
                "diagnostics": diagnostics,
                "attempts": attempts,
            }

        # If unrecoverable indicator failure -> stop early
        if ind_res.recoverable is False:
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "status": ind_res.status,
                "data": None,
                "diagnostics": diagnostics,
                "attempts": attempts,
            }

        last_result = {
            "symbol": symbol,
            "timeframe": timeframe,
            "status": ind_res.status,
            "data": None,
            "diagnostics": diagnostics,
            "attempts": attempts,
        }

    return last_result

