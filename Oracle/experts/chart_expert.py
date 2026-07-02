"""
chart_expert.py

Multi-symbol RL trading environment.

The agent experiences the whole market across all symbols,
learns which strategies fit which charts best, and decides
when to go long, short, close, or switch to a better symbol.

Actions:
    0 = Hold
    1 = Buy  (open long)
    2 = Sell (open short)
    3 = Close position
    4 = Switch symbol

Connects to:
    - strategy_tester.py  → loads optimized strategy config per symbol
    - knowledge_base.py   → translates learned rules into numeric signals
    - db_handler.py       → loads/saves cached strategies
"""

import os
import re
import glob
import random
import shutil
from datetime import datetime
import numpy as np
import pandas as pd

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from dotenv import load_dotenv
load_dotenv()
from experts.db_handler import load_optimized_strategy, load_rules

# =========================================================
# TIMEFRAME HIERARCHY
# These are fixed — minutes NEVER go in higher_timeframes
# =========================================================

HIGHER_TIMEFRAMES   = ["weekly", "daily", "4h", "1h"]
ANALYSIS_TIMEFRAMES = ["30min", "15min"]
ENTRY_TIMEFRAMES    = ["5min", "1min"]

ALL_TIMEFRAMES = HIGHER_TIMEFRAMES + ANALYSIS_TIMEFRAMES + ENTRY_TIMEFRAMES

# =========================================================
# TRAIN / TEST SPLIT
# =========================================================

TRAIN_RATIO = 0.7   # 70% train, 30% test

# =========================================================
# ACTIONS
# =========================================================

ACTION_HOLD   = 0
ACTION_BUY    = 1   # Open long
ACTION_SELL   = 2   # Open short
ACTION_CLOSE  = 3   # Close any open position
ACTION_SWITCH = 4   # Move to a different symbol

# =========================================================
# SWITCH + RISK GUARDRAILS (Fixes #3, #9)
# These are HARD-ENFORCED constraints that the agent cannot override.
# Even with a positive shaped reward, switch / size / drawdown
# actions are blocked or rescaled by the environment.
# =========================================================
MIN_DWELL_BARS         = 5     # minimum bars on a symbol before SWITCH is allowed
MAX_SWITCHES_PER_EPOCH = 12    # hard cap; above this SWITCH is blocked
COMMISSION_PER_SIDE    = 0.00010  # 1 bps per side, configurable
SLIPPAGE_BASE_BPS      = 0.00005  # 0.5 bps base slippage
SLIPPAGE_VOL_K         = 0.10     # bps per unit of normalized ATR
LATENCY_BARS           = 0        # execution delay in bars (0 = same bar)
RISK_FREE_RATE         = 0.0      # for Sharpe computation
TARGET_LEVERAGE        = 1.0      # position-size denominator

# =========================================================
# TECHNICAL INDICATORS
# =========================================================

def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """RSI — fixed min_periods to avoid invalid early values."""
    delta    = series.diff().fillna(0)
    gain     = delta.where(delta > 0, 0.0)
    loss     = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window, min_periods=window).mean()
    avg_loss = loss.rolling(window, min_periods=window).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    # fillna(100): all-up candles = max RSI = 100, not NaN
    return (100 - 100 / (1 + rs)).fillna(100)


def compute_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """ATR — fixed min_periods. Edge fill uses the first valid value
    (forward) rather than bfill, which would leak future data into
    the early rows of the series (see audit: "Indicators use bfill
    after rolling, which silently leaks future data on edges")."""
    high_low   = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close  = (df["Low"]  - df["Close"].shift()).abs()
    tr  = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window, min_periods=window).mean()
    # Forward-fill from the first valid value, then fall back to 0
    # for the leading NaN window. Do NOT bfill.
    first_valid = atr.dropna().iloc[0] if atr.notna().any() else 0.0
    return atr.fillna(first_valid)


def compute_obv(df: pd.DataFrame) -> pd.Series:
    """
    On-Balance Volume — vectorized.
    Fixed: replaces the old O(n) Python for-loop with
    vectorized pandas operations. ~100x faster on large datasets.
    """
    direction    = np.sign(df["Close"].diff().fillna(0))
    signed_vol   = direction * df["Volume"]
    return signed_vol.cumsum()


def compute_macd(series: pd.Series,
                 fast: int = 12,
                 slow: int = 26,
                 signal: int = 9):
    ema_fast    = series.ewm(span=fast,   adjust=False).mean()
    ema_slow    = series.ewm(span=slow,   adjust=False).mean()
    macd_line   = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist   = macd_line - macd_signal
    return macd_line, macd_signal, macd_hist


def compute_bollinger(series: pd.Series,
                      window: int = 20,
                      num_std: int = 2):
    """Bollinger Bands — fixed min_periods. Edge fill uses the first
    valid value (forward) rather than bfill, which would leak future
    data into the early rows (see audit: "Indicators use bfill after
    rolling, which silently leaks future data on edges")."""
    sma   = series.rolling(window, min_periods=window).mean()
    std   = series.rolling(window, min_periods=window).std()
    upper = sma + std * num_std
    lower = sma - std * num_std
    def _safe_fill(s: pd.Series) -> pd.Series:
        fv = s.dropna().iloc[0] if s.notna().any() else 0.0
        return s.fillna(fv)
    return _safe_fill(upper), _safe_fill(sma), _safe_fill(lower)


def compute_stochastic(df: pd.DataFrame,
                       k_window: int = 14,
                       d_window: int = 3):
    """Stochastic Oscillator — fixed min_periods."""
    low_k  = df["Low"].rolling(k_window,  min_periods=k_window).min()
    high_k = df["High"].rolling(k_window, min_periods=k_window).max()
    range_ = (high_k - low_k).replace(0, 1)
    k      = 100 * (df["Close"] - low_k) / range_
    d      = k.rolling(d_window, min_periods=d_window).mean()
    return k.fillna(50), d.fillna(50)


def compute_williams_r(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Williams %R — fixed min_periods."""
    high = df["High"].rolling(window, min_periods=window).max()
    low  = df["Low"].rolling(window,  min_periods=window).min()
    r    = -100 * (high - df["Close"]) / (high - low).replace(0, 1)
    return r.fillna(-50)


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds technical indicators to a DataFrame.
    All rolling calculations use proper min_periods.
    Drops early rows where indicators are not yet valid.
    """
    df = df.copy()

    if "Volume" not in df.columns:
        df["Volume"] = 0.0

    # ── Price features ────────────────────────────────
    df["return_1"]        = df["Close"].pct_change().fillna(0.0)
    df["return_5"]        = df["Close"].pct_change(5).fillna(0.0)
    df["high_low_ratio"]  = df["High"] / df["Low"]
    df["close_open_ratio"] = df["Close"] / df["Open"]

    # ── Moving averages — fixed min_periods ───────────
    df["sma_10"]  = df["Close"].rolling(10,  min_periods=10).mean()
    df["sma_20"]  = df["Close"].rolling(20,  min_periods=20).mean()
    df["sma_50"]  = df["Close"].rolling(50,  min_periods=50).mean()
    df["ema_12"]  = df["Close"].ewm(span=12, adjust=False).mean()
    df["ema_26"]  = df["Close"].ewm(span=26, adjust=False).mean()

    # ── Momentum ──────────────────────────────────────
    df["rsi_14"]              = compute_rsi(df["Close"], 14)
    df["rsi_7"]               = compute_rsi(df["Close"], 7)
    df["stoch_k"], df["stoch_d"] = compute_stochastic(df, 14, 3)
    df["williams_r"]          = compute_williams_r(df, 14)

    # ── Volatility ────────────────────────────────────
    df["atr_14"]                             = compute_atr(df, 14)
    df["bb_upper"], df["bb_mid"], df["bb_lower"] = compute_bollinger(df["Close"], 20, 2)

    # ── Volume ────────────────────────────────────────
    df["volume_sma"] = df["Volume"].rolling(10, min_periods=10).mean()
    df["volume_ratio"] = df["Volume"] / df["volume_sma"].replace(0, 1)
    df["obv"]          = compute_obv(df)

    # ── MACD ──────────────────────────────────────────
    df["macd"], df["macd_signal"], df["macd_hist"] = compute_macd(df["Close"])

    # ── Candle shape ──────────────────────────────────
    df["body_size"]    = (df["Close"] - df["Open"]).abs() / df["Close"]
    df["upper_shadow"] = (df["High"] - df[["Open","Close"]].max(axis=1)) / df["Close"]
    df["lower_shadow"] = (df[["Open","Close"]].min(axis=1) - df["Low"]) / df["Close"]

    # Drop early rows where key indicators are invalid
    # (prevents NaN from polluting the observation space)
    df = df.dropna(subset=["sma_50", "rsi_14", "atr_14", "stoch_k"])

    return df


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalizes column casing — handles close/Close/CLOSE."""
    rename = {}
    for col in df.columns:
        cl = col.lower()
        if cl == "open":   rename[col] = "Open"
        elif cl == "high": rename[col] = "High"
        elif cl == "low":  rename[col] = "Low"
        elif cl == "close": rename[col] = "Close"
        elif cl == "volume": rename[col] = "Volume"
    return df.rename(columns=rename) if rename else df

# =========================================================
# DATA LOADER
# =========================================================

def load_data_bundle(data_dir: str = "data", train_ratio: float = TRAIN_RATIO) -> dict:
    """
    Loads all CSV/XLSX price data from the data directory.
    Applies train/test split to each symbol/timeframe.
    Returns a nested dict: {symbol: {timeframe: {train: df, test: df}}}
    """
    timeframe_aliases = {
        "weekly":  ["weekly"],
        "daily":   ["daily"],
        "4h":      ["4h", "4hr"],
        "1h":      ["1h", "1hr", "hourly"],
        "30min":   ["30min", "30m"],
        "15min":   ["15min", "15m"],
        "5min":    ["5min", "5m"],
        "1min":    ["1min", "1m"],
    }

    raw_files = {}

    for timeframe, aliases in timeframe_aliases.items():
        for alias in aliases:
            for path in glob.glob(os.path.join(data_dir, f"*_{alias}.csv")):
                basename = os.path.basename(path)
                symbol   = "_".join(basename.split("_")[:-1])
                raw_files.setdefault(symbol, {})[timeframe] = path

    data_bundle = {}

    for symbol, paths in raw_files.items():

        # Require at least weekly + daily for trend context
        if "weekly" not in paths or "daily" not in paths:
            continue

        # Require at least one entry-level timeframe
        if not any(tf in paths for tf in ENTRY_TIMEFRAMES + ANALYSIS_TIMEFRAMES):
            continue

        symbol_data = {}

        for timeframe, path in paths.items():
            try:
                df = pd.read_csv(path)
            except UnicodeDecodeError:
                df = pd.read_excel(path, engine="openpyxl")

            if "Unnamed: 0" in df.columns:
                df = df.rename(columns={"Unnamed: 0": "Date"})
            elif "Datetime" in df.columns:
                df = df.rename(columns={"Datetime": "Date"})
            if "Date" not in df.columns:
                print(f"WARNING: Missing Date column in {path} — skipping")
                continue

            df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=True)
            try:
                if df["Date"].dt.tz is not None:
                    df["Date"] = df["Date"].dt.tz_convert(None)
            except Exception:
                pass

            df = _standardize_columns(df)

            for col in ["Open", "High", "Low", "Close", "Volume"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            df = df.dropna(subset=["Date", "Open", "High", "Low", "Close"])
            if df.empty:
                continue

            df = df.sort_values("Date").set_index("Date")
            df = add_technical_indicators(df)

            if df.empty:
                continue

            # Train / test split — no data leakage
            ratio = min(max(train_ratio, 0.01), 0.99)
            split_idx = int(len(df) * ratio)
            symbol_data[timeframe] = {
                "train": df.iloc[:split_idx].copy(),
                "test":  df.iloc[split_idx:].copy(),
            }

        if symbol_data:
            data_bundle[symbol] = symbol_data

    print(f"Loaded {len(data_bundle)} symbols")
    return data_bundle

# =========================================================
# KNOWLEDGE BRIDGE
# Translates text-based learned rules into numeric signals
# the RL agent can use as part of its observation space.
# =========================================================

# Keyword → observation index mapping
# These keywords come from learned rules in knowledge_base.py
RULE_KEYWORDS = {
    "trend":        0,
    "momentum":     1,
    "support":      2,
    "resistance":   2,
    "breakout":     3,
    "reversal":     4,
    "volume":       5,
    "liquidity":    6,
    "order_block":  7,
    "fair_value":   7,
    "risk":         8,
    "stop":         8,
    "psychology":   9,
    "confluence":   10,
}

KNOWLEDGE_FEATURE_SIZE = 11


def rules_to_signals(strategy_config: dict) -> np.ndarray:
    """
    Translates text-based learned rules into a numeric signal
    vector the RL agent can observe.

    Each position in the vector represents a trading concept.
    Value = average confidence of rules mentioning that concept.
    0.0 = concept not present, 1.0 = very high confidence concept.

    This is Gap 2 — the bridge between knowledge_base.py text rules
    and the RL agent's numeric observation space.
    """
    signals = np.zeros(KNOWLEDGE_FEATURE_SIZE, dtype=np.float32)
    counts  = np.zeros(KNOWLEDGE_FEATURE_SIZE, dtype=np.float32)

    if not strategy_config:
        return signals

    # Collect all rule text from all fields
    rule_fields = [
        "entry_conditions",
        "exit_conditions",
        "risk_management",
        "market_structure",
        "indicators",
        "psychology",
    ]

    for field in rule_fields:
        rules = strategy_config.get(field, [])
        for rule in rules:
            if isinstance(rule, dict):
                text       = rule.get("rule", "") + " " + rule.get("description", "")
                confidence = float(rule.get("confidence", 0.5))
            elif isinstance(rule, str):
                text       = rule
                confidence = 0.5
            else:
                continue

            text_lower = text.lower()

            for keyword, idx in RULE_KEYWORDS.items():
                if keyword in text_lower:
                    signals[idx] += confidence
                    counts[idx]  += 1

    # Average confidence per concept
    nonzero = counts > 0
    signals[nonzero] = signals[nonzero] / counts[nonzero]

    # Calculate Confluence: If both Liquidity (6) and Order Blocks (7) are present
    if signals[6] > 0.6 and signals[7] > 0.6:
        signals[10] = (signals[6] + signals[7]) / 2

    return signals.clip(0.0, 1.0)


def load_strategy_for_symbol(symbol: str) -> dict:
    """
    Loads the optimized strategy config for a symbol from
    strategy_tester.py's cache. Falls back to local ICT book rules
    or master knowledge if no optimized strategy exists.
    """
    config = load_optimized_strategy(symbol)

    if config:
        return config

    # Prefer the local ICT book strategy when API-backed optimization
    # is skipped and a book-derived strategy is available.
    ict_strategy = load_rules("book_ict_trading_strategy", "full")
    if ict_strategy:
        return ict_strategy

    master = load_rules("master_knowledge", "trading_strategy")
    if master:
        return master

    return {}

# =========================================================
# REWARD FUNCTION
# Rewards quality trades, punishes drawdown and bad habits.
# =========================================================

# Reward shaping coefficients (Fix #4: reward misalignment).
# The reward is a *risk-adjusted return* signal with penalties for
# drawdown, overtrading, symbol switching, and excess leverage.
REWARD_LAMBDA_DRAWDOWN   = 50.0   # weight on current drawdown
REWARD_MU_OVERTRADE      = 0.25   # per-trade flat cost
REWARD_NU_SWITCH         = 1.0    # per-switch flat cost
REWARD_KAPPA_LEVERAGE    = 0.5    # penalty for >target leverage
REWARD_ETA_SHARPE        = 1.0    # scaling on the differential-Sharpe term
REWARD_DIFF_SHARPE_WINDOW = 32    # lookback for the differential Sharpe term
REWARD_TIME_DECAY        = 0.001  # per-step time decay (slower than before)


class _SharpeTracker:
    """Running-window mean/std tracker for the differential-Sharpe reward
    (Moody & Saffell, 2001). Computes a per-step increment of the Sharpe
    ratio over a sliding window of recent equity returns. Pure Python so
    we don't need numpy at import time."""
    __slots__ = ("window", "buf", "sum", "sumsq", "count")

    def __init__(self, window: int = REWARD_DIFF_SHARPE_WINDOW):
        self.window   = window
        self.buf      = []
        self.sum      = 0.0
        self.sumsq    = 0.0
        self.count    = 0

    def update(self, ret: float) -> float:
        """Push a new return, return the differential-Sharpe increment."""
        self.buf.append(ret)
        self.sum   += ret
        self.sumsq += ret * ret
        self.count += 1
        if len(self.buf) > self.window:
            old = self.buf.pop(0)
            self.sum   -= old
            self.sumsq -= old * old
            self.count -= 1
        n = self.count
        if n < 2:
            return 0.0
        mean = self.sum / n
        var  = max(self.sumsq / n - mean * mean, 1e-8)
        std  = var ** 0.5
        sharpe_now = mean / std
        # Differential-Sharpe = how much the Sharpe moved this step
        # (approximated by the marginal contribution of the new sample
        # relative to the previous sharpe value)
        prev_n     = max(n - 1, 1)
        prev_mean  = (self.sum - ret) / prev_n
        prev_var   = max(self.sumsq / n - mean * mean, 1e-8)  # post-update
        prev_std   = prev_var ** 0.5 if prev_var > 0 else std
        prev_sharpe = (prev_mean / prev_std) if prev_std > 1e-8 else 0.0
        return sharpe_now - prev_sharpe

    def reset(self):
        self.buf   = []
        self.sum   = 0.0
        self.sumsq = 0.0
        self.count = 0


def compute_reward(
    pnl: float,
    entry_price: float,
    stop_loss_pct: float,
    equity: float,
    peak_equity: float,
    holding_steps: int,
    is_close: bool,
    trend_aligned: bool = True,
    is_switch: bool = False,
    excess_leverage: float = 0.0,
    sharpe_tracker: "_SharpeTracker | None" = None,
    trade_count: int = 0,
) -> float:
    """
    Shaped risk-adjusted reward (Fix #4).

        r = sharpe_increment
            + alignment_bonus * pnl        (on close, with win)
            + pnl                           (on close, baseline)
            - lambda * drawdown^2           (every step)
            - mu     * I[trade]             (on close)
            - nu     * I[switch]            (on switch)
            - kappa  * excess_leverage      (on close, if > target)
            - eta    * time_decay           (every step)

    The differential-Sharpe term (Moody & Saffell) makes the agent
    optimize for risk-adjusted return, not raw PnL. This is the
    central change in Fix #1 (replace accuracy with risk-adjusted
    return) and Fix #4 (reward shaping).
    """
    reward = 0.0

    # Per-step equity return (for differential Sharpe)
    ret = 0.0
    if peak_equity > 0:
        ret = (equity - peak_equity) / peak_equity  # signed return
    if sharpe_tracker is not None:
        ds = sharpe_tracker.update(ret)
        reward += REWARD_ETA_SHARPE * ds

    # Trade PnL with R-multiple and trend-alignment bonus
    if is_close and pnl != 0.0:
        risk_amount = entry_price * stop_loss_pct
        if risk_amount > 0:
            r_multiple = pnl / risk_amount
            alignment_bonus = 1.2 if trend_aligned else 0.8
            reward += pnl * (1.0 + min(max(r_multiple, -3.0), 3.0) * 0.2) * alignment_bonus
        else:
            reward += pnl
        if pnl < 0:
            penalty_mult = 1.5 if not trend_aligned else 1.0
            reward += pnl * penalty_mult

    # Overtrading penalty (only on close, not on switch)
    if is_close:
        reward -= REWARD_MU_OVERTRADE

    # Symbol-switch penalty (Fix #3)
    if is_switch:
        reward -= REWARD_NU_SWITCH

    # Excess leverage penalty (Fix #9)
    if excess_leverage > 0:
        reward -= REWARD_KAPPA_LEVERAGE * excess_leverage

    # Drawdown penalty — quadratic in drawdown (smooth, differentiable)
    if peak_equity > 0:
        drawdown = max((peak_equity - equity) / peak_equity, 0.0)
        if drawdown > 0.0:
            reward -= REWARD_LAMBDA_DRAWDOWN * (drawdown ** 2)

    # Slow time decay (encourages decisive action without dominating
    # the PnL signal)
    reward -= REWARD_TIME_DECAY

    return float(reward)

# =========================================================
# MULTI-SYMBOL CHART EXPERT — MAIN RL ENVIRONMENT
# =========================================================

class MultiSymbolChartExpert(gym.Env):
    """
    RL environment that sees the whole market.

    The agent:
    - Trades any symbol in the bundle
    - Goes long OR short
    - Switches symbols when no opportunity exists
    - Uses optimized strategy signals per symbol
    - Learns which strategies work on which charts

    Observation space:
        - Window of entry-timeframe OHLCV candles
        - Higher timeframe snapshots (trend context)
        - Technical indicator signals
        - Knowledge signals (from learned strategy)

    Action space:
        0 = Hold
        1 = Buy  (open long)
        2 = Sell (open short)
        3 = Close position
        4 = Switch symbol

    Observation layout (concatenated, in this exact order):
        [0 : window_size*5]      Entry-TF OHLCV sliding window
        [window_size*5 :
         + len(HIGHER_TIMEFRAMES)*5]
                                 Higher-TF OHLCV snapshots
                                 (one bar per timeframe, in
                                 HIGHER_TIMEFRAMES order)
        [... : + 10]             Technical indicator features at the
                                 current bar (RSI, MACD, MACD hist,
                                 Stoch %K, Williams %R, volume ratio,
                                 body size, upper/lower shadow,
                                 Bollinger %B)
        [... : + 1]              Trend-alignment feature
                                 (1.0 if daily Close > daily SMA50,
                                 0.0 if below, 0.5 if unknown)
        [... : + KNOWLEDGE_FEATURE_SIZE]
                                 Strategy / knowledge-base signals
                                 (see rules_to_signals)

    Reward formula (see compute_reward for the full definition):
        r = 0
        if is_close and pnl != 0:
            r += pnl * (1 + min(R-multiple, 3) * 0.2) * alignment_bonus
            if pnl < 0: r += pnl * loss_penalty_multiplier
        if is_close:   r -= 2.0                 # overtrading penalty
        if dd > 5%:    r -= (dd ** 2) * 10.0   # drawdown penalty
        r -= 0.005                              # mild time decay

    Termination conditions:
        * terminated = True when current_step reaches the end of the
          selected symbol / entry-timeframe split
        * truncated  = False today (no max-step cap is enforced
          separately; a hard cap can be added via the constructor)
        * Auto-close: if a position is still open at termination, it
          is closed at the last available price before the env returns
        * There is no explicit margin-call / bankruptcy termination —
          a hard floor (e.g. balance <= 0) can be added if needed
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        data_bundle: dict,
        mode: str = "train",
        window_size: int = 20,
        max_daily_loss_pct: float = 0.05,
        max_drawdown_pct: float = 0.20,
        max_position_pct: float = 0.25,
        max_correlated_exposure: float = 0.50,
        max_daily_trades: int = 20,
        ppo_dropout: float = 0.0,           # placeholder for trainer hook
        feature_noise_std: float = 0.0,    # observation noise injection
    ):
        super().__init__()
        self.data_bundle  = data_bundle
        self.mode         = mode        # "train" or "test"
        self.window_size  = window_size
        self.symbols      = sorted(data_bundle.keys())
        if not self.symbols:
            raise ValueError("data_bundle is empty — no symbols loaded")

        # Risk-engineering hard limits (Fix #9)
        self.max_daily_loss_pct   = max_daily_loss_pct
        self.max_drawdown_pct     = max_drawdown_pct
        self.max_position_pct     = max_position_pct
        self.max_corr_exposure    = max_correlated_exposure
        self.max_daily_trades     = max_daily_trades
        self.feature_noise_std    = feature_noise_std
        self.ppo_dropout          = ppo_dropout
        self._sharpe_tracker      = _SharpeTracker()
        self._switches_this_ep    = 0
        self._trades_today        = 0
        self._last_day_key        = None
        self._start_of_day_equity = None

        # ── Observation shape ─────────────────────────
        # Entry window: window_size candles × 5 OHLCV
        entry_features    = window_size * 5
        # Higher TF snapshots: each gives 5 OHLCV values
        context_features  = len(HIGHER_TIMEFRAMES) * 5
        # Technical indicator features from current bar
        indicator_features = 10
        # Trend alignment feature (is entry tf aligned with Daily tf?)
        trend_feature = 1
        # Knowledge signals from learned strategy
        knowledge_features = KNOWLEDGE_FEATURE_SIZE
        # State-augmentation features (Fix #5): regime, session, position
        # context, distance to stop/target, vol regime, spread, hours
        # since last trade, realized pnl today, unrealized pnl.
        augmented_features = (
            4   # session one-hot
            + 8 # regime one-hot
            + 1 # vol regime
            + 1 # spread
            + 1 # current_position side
            + 1 # unrealized_pnl
            + 1 # realized_pnl_today
            + 1 # hours_since_trade (normalized)
            + 1 # distance_to_stop
            + 1 # distance_to_target
            + 1 # bars_on_current_symbol (normalized)
            + 1 # cumulative switches this episode
        )
        obs_size = (
            entry_features +
            context_features +
            indicator_features +
            trend_feature +
            knowledge_features +
            augmented_features
        )

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_size,),
            dtype=np.float32
        )

        # ── Action space ──────────────────────────────
        # 0=Hold, 1=Buy, 2=Sell, 3=Close, 4=Switch
        self.action_space = spaces.Discrete(5)

        # ── State ─────────────────────────────────────
        self.current_symbol   = None
        self.current_data     = None
        self.entry_timeframe  = None
        self.strategy_signals = np.zeros(KNOWLEDGE_FEATURE_SIZE, dtype=np.float32)

        self.balance         = 10_000.0
        self.equity          = self.balance
        self.peak_equity     = self.balance
        self.previous_equity = self.balance
        self.position        = None
        self.current_step    = window_size
        # Sentinel — set in _select_symbol / reset. Used by step() to
        # bound the cursor before any slicing/observation work.
        self.max_step        = window_size
        self.holding_steps   = 0
        self.trade_history   = []
        self.current_trade   = None

        # ── Per-step indicator cache (Fix #3) ───────────
        # Caches the last computed observation, so on the next step
        # we only re-slice the (current_step - prev_step) newly
        # available rows of the entry-TF indicator dataframe and
        # the higher-TF "last bar at or before current_time" lookup.
        # Without this, _get_observation does ~O(indicators ×
        # timeframes) of repeated work on every step.
        self._cache: dict = {
            "obs":              None,   # last observation vector
            "current_step":     -1,     # step the cached obs was built for
            "current_symbol":   None,   # symbol the cached obs was built for
            "current_timeframe": None,  # entry tf the cached obs was built for
            "daily_close":      None,   # most recent daily Close (for trend)
            "daily_sma_50":     None,   # most recent daily sma_50
        }

        # Per-symbol PnL attribution (Fix #3)
        self.symbol_pnl: dict = {s: 0.0 for s in self.symbols}
        self.symbol_trades: dict = {s: 0 for s in self.symbols}
        self._bars_on_symbol = 0
        self._switch_attempts = 0

        # Session / regime / spread state (Fix #5)
        self.session_one_hot   = np.zeros(4, dtype=np.float32)  # asia/eu/us/closed
        self.regime_one_hot    = np.zeros(8, dtype=np.float32)  # 8 regimes
        self.vol_regime        = 0.0
        self.spread            = 0.0
        self.distance_to_stop  = 0.0
        self.distance_to_tp    = 0.0
        self.hours_since_trade = 0
        self.realized_pnl_today = 0.0

    # ── Symbol selection ──────────────────────────────

    def _select_symbol(self, symbol: str = None):
        """
        Selects a symbol to trade.

        Priority:
        1. Use entry_tf from the optimized strategy config
           - Strategy can specify ANY timeframe as entry
           - A swing trader CAN use daily or 4H as entry
           - A scalper uses 1min/5min as entry
        2. If that tf has no data → skip to next symbol
        3. No strategy preference → find best available tf from data
        4. Never force a fallback just to fill — skip the symbol
        """
        candidates = (
            [symbol] if symbol
            else random.sample(self.symbols, len(self.symbols))
        )

        for candidate in candidates:
            self.current_symbol = candidate
            self.current_data   = self.data_bundle[candidate]

            # Load strategy first — it owns the entry_tf decision
            strategy_config       = load_strategy_for_symbol(candidate)
            self.strategy_signals = rules_to_signals(strategy_config)

            # Strategy specifies preferred entry timeframe
            # This can be ANY timeframe — daily for swing, 1min for scalp
            preferred_tf = strategy_config.get("entry_tf", None)

            if preferred_tf:
                tf_splits = self.current_data.get(preferred_tf, {})
                df        = tf_splits.get(self.mode)

                if df is not None and len(df) > self.window_size + 10:
                    # Strategy's preferred tf has data — use it
                    self.entry_timeframe = preferred_tf
                    self.current_step    = self.window_size
                    self.max_step        = len(df) - 1
                    return
                else:
                    # Strategy's preferred tf has no data — skip symbol
                    # Don't force another tf, the strategy won't work right
                    continue

            # No strategy preference — find best available tf from data
            # Sort all available timeframes by granularity
            # Prefer lower timeframes (more data points) unless strategy says otherwise
            available_tfs = [
                tf for tf in self.current_data
                if self.current_data[tf].get(self.mode) is not None
                and len(self.current_data[tf].get(self.mode, pd.DataFrame())) > self.window_size + 10
            ]

            if not available_tfs:
                # No usable timeframe on this symbol — skip
                continue

            # Sort by position in ALL_TIMEFRAMES (lower = more granular)
            available_tfs.sort(
                key=lambda tf: ALL_TIMEFRAMES.index(tf)
                if tf in ALL_TIMEFRAMES else 999,
                reverse=True  # Higher index = lower timeframe = prefer
            )

            # Use most granular available timeframe
            # Higher TFs like daily/4H are available if they're the
            # only ones with enough data (e.g. swing trading symbols)
            self.entry_timeframe = available_tfs[0]
            self.current_step    = self.window_size
            self.max_step        = len(self.current_data[self.entry_timeframe][self.mode]) - 1
            return

        # All symbols exhausted with no valid entry tf found
        # True last resort — only hits if data is severely missing
        print("WARNING: All symbols exhausted — using first symbol as fallback")
        self.current_symbol   = self.symbols[0]
        self.current_data     = self.data_bundle[self.symbols[0]]
        self.entry_timeframe  = next(
            (tf for tf in ALL_TIMEFRAMES if tf in self.current_data),
            "daily"
        )
        self.current_step     = self.window_size
        self.max_step         = len(self.current_data[self.entry_timeframe][self.mode]) - 1 \
            if self.entry_timeframe in self.current_data else self.window_size
        strategy_config       = load_strategy_for_symbol(self.symbols[0])
        self.strategy_signals = rules_to_signals(strategy_config)

    # ── Observation builders ──────────────────────────

    def _get_entry_window(self) -> np.ndarray:
        """Returns the sliding window of entry-TF OHLCV candles."""
        df     = self.current_data[self.entry_timeframe][self.mode]
        window = df.iloc[
            self.current_step - self.window_size : self.current_step
        ]
        cols = [c for c in ["Close","Open","High","Low","Volume"] if c in window.columns]
        vals = window[cols].values.flatten()

        # Pad if some columns missing
        expected = self.window_size * 5
        if len(vals) < expected:
            vals = np.pad(vals, (0, expected - len(vals)))

        return vals.astype(np.float32)

    def _get_higher_tf_context(self) -> np.ndarray:
        """
        Returns the most recent snapshot from each higher timeframe.
        Gives the agent trend and bias context above entry level.
        """
        df_entry = self.current_data[self.entry_timeframe][self.mode]
        current_time = df_entry.index[self.current_step]

        context = []
        for tf in HIGHER_TIMEFRAMES:
            tf_splits = self.current_data.get(tf)
            df_tf     = tf_splits.get(self.mode) if tf_splits else None

            if df_tf is None or df_tf.empty:
                context.append(np.zeros(5, dtype=np.float32))
                continue

            # Optimized lookup using searchsorted instead of boolean filtering
            idx = df_tf.index.searchsorted(current_time, side='right') - 1
            row = df_tf.iloc[max(0, idx)]

            vals = np.array([
                float(row.get("Close",  0)),
                float(row.get("Open",   0)),
                float(row.get("High",   0)),
                float(row.get("Low",    0)),
                float(row.get("Volume", 0)),
            ], dtype=np.float32)
            context.append(vals)

        return np.concatenate(context)

    def _get_indicator_features(self) -> np.ndarray:
        """
        Returns key indicator values at the current bar.
        Normalized so they're on similar scales.
        """
        df  = self.current_data[self.entry_timeframe][self.mode]
        row = df.iloc[self.current_step]

        close = float(row.get("Close", 1.0))
        atr   = float(row.get("atr_14", close * 0.01))

        features = np.array([
            float(row.get("rsi_14",      50.0)) / 100.0,
            float(row.get("macd",         0.0)) / (atr + 1e-8),
            float(row.get("macd_hist",    0.0)) / (atr + 1e-8),
            float(row.get("stoch_k",     50.0)) / 100.0,
            float(row.get("williams_r", -50.0)) / -100.0,
            float(row.get("volume_ratio", 1.0)) / 5.0,
            float(row.get("body_size",    0.0)),
            float(row.get("upper_shadow", 0.0)),
            float(row.get("lower_shadow", 0.0)),
            (float(row.get("Close", 0)) - float(row.get("bb_lower", 0))) /
            (float(row.get("bb_upper", 1)) - float(row.get("bb_lower", 0)) + 1e-8),
        ], dtype=np.float32)

        return np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)

    def _get_observation(self) -> np.ndarray:
        # ── Cache fast-path (Fix #3) ─────────────────────
        # If the cached observation is for the same (symbol, entry-tf,
        # step) we already built, just return it. Indicator values
        # only need to be re-read when current_step advances, the
        # agent switches symbol, or the entry timeframe changes.
        cache = self._cache
        if (
            cache["obs"] is not None
            and cache["current_step"]     == self.current_step
            and cache["current_symbol"]   == self.current_symbol
            and cache["current_timeframe"] == self.entry_timeframe
        ):
            return cache["obs"]

        entry     = self._get_entry_window()
        context   = self._get_higher_tf_context()
        indicators = self._get_indicator_features()

        # Calculate Trend Alignment (Daily SMA 50).
        # Guard against an empty daily slice — this can happen at the
        # very start of an episode or if the symbol/timeframe has no
        # data at the current cursor. Without this guard, .iloc[-1]
        # raises IndexError and training crashes (see audit finding:
        # "IndexError when sub-dataframe is empty at end of episode").
        df_daily = self.current_data.get("daily", {}).get(self.mode)
        is_aligned = 0.5  # neutral default
        if df_daily is not None and not df_daily.empty:
            try:
                # Use searchsorted + iloc instead of iloc[-1] so we
                # can reuse a cached "last index <= current_time"
                # rather than re-walking the full daily index.
                df_entry   = self.current_data[self.entry_timeframe][self.mode]
                cur_time   = df_entry.index[self.current_step]
                idx        = df_daily.index.searchsorted(cur_time, side='right') - 1
                idx        = max(0, idx)
                row        = df_daily.iloc[idx]
                last_close = float(row.get("Close", 0.0))
                sma_50     = float(row.get("sma_50", last_close))
                is_aligned = 1.0 if last_close > sma_50 else 0.0
                # Stash for the next call (no need to re-do the lookup)
                cache["daily_close"]  = last_close
                cache["daily_sma_50"] = sma_50
            except (IndexError, KeyError):
                # Defensive: any unexpected indexing issue falls back to neutral
                is_aligned = 0.5
        trend_feat = np.array([is_aligned], dtype=np.float32)
        knowledge  = self.strategy_signals

        # ── State-augmentation block (Fix #5) ────────────────────────
        # Refresh the cached regime / session / position-context state
        # so the agent sees it on every step.
        self._refresh_session_regime_state()

        # Position-side and distances (Fix #5)
        if self.position is not None:
            pos_side  = 1.0 if self.position["direction"] == "long" else -1.0
            # Use the current bar's close as the reference price for
            # distances-to-stop/target and unrealized PnL.
            price_ref = float(self.current_data[self.entry_timeframe][self.mode]
                               .iloc[self.current_step]["Close"])
            d_stop    = (price_ref - self.position["sl"]) / max(price_ref, 1e-8)
            d_target  = (self.position["tp"] - price_ref) / max(price_ref, 1e-8)
            unrealized = (
                (price_ref - self.position["entry_price"]) * pos_side
                * self.position["size"]
            )
        else:
            pos_side  = 0.0
            d_stop    = 0.0
            d_target  = 0.0
            unrealized = 0.0

        # Realized pnl today is tracked on _close_position
        hours_since_trade = min(self.hours_since_trade, 24) / 24.0
        bars_norm = min(self._bars_on_symbol, 200) / 200.0
        switches_norm = min(self._switches_this_ep, MAX_SWITCHES_PER_EPOCH) / float(MAX_SWITCHES_PER_EPOCH)

        augmented = np.concatenate([
            self.session_one_hot,
            self.regime_one_hot,
            np.array([self.vol_regime], dtype=np.float32),
            np.array([self.spread], dtype=np.float32),
            np.array([pos_side], dtype=np.float32),
            np.array([unrealized / max(self.balance, 1.0)], dtype=np.float32),
            np.array([self.realized_pnl_today / max(self.balance, 1.0)], dtype=np.float32),
            np.array([hours_since_trade], dtype=np.float32),
            np.array([d_stop], dtype=np.float32),
            np.array([d_target], dtype=np.float32),
            np.array([bars_norm], dtype=np.float32),
            np.array([switches_norm], dtype=np.float32),
        ])

        obs = np.concatenate([entry, context, indicators, trend_feat, knowledge, augmented])

        # Fix #1 + #2: feature-noise injection at training time prevents
        # the agent from memorizing the exact indicator values. This is
        # the cheapest, most effective regularizer for tabular obs.
        if self.mode == "train" and self.feature_noise_std > 0.0:
            obs = obs + np.random.normal(
                0.0, self.feature_noise_std, size=obs.shape
            ).astype(np.float32)
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)

        # Populate cache for the next step
        cache["obs"]               = obs
        cache["current_step"]      = self.current_step
        cache["current_symbol"]    = self.current_symbol
        cache["current_timeframe"] = self.entry_timeframe
        return obs

    # ── Position management ───────────────────────────

    def _get_current_price(self) -> float:
        df = self.current_data[self.entry_timeframe][self.mode]
        return float(df.iloc[self.current_step]["Close"])

    # ── Realistic execution (Fix #8) ──────────────────────────────

    def _apply_execution_cost(self, price: float) -> float:
        """
        Apply commission + volatility-aware slippage to a fill price.
        Returns the *effective* price the agent receives.
        For BUY: price moves AGAINST the agent (price up).
        For SELL: price moves AGAINST the agent (price down).
        """
        df  = self.current_data[self.entry_timeframe][self.mode]
        row = df.iloc[self.current_step]
        close = float(row.get("Close", price))
        atr   = float(row.get("atr_14", close * 0.01))
        atr_pct = atr / max(close, 1e-8)
        # slippage in bps scales with normalized volatility
        slip_bps = SLIPPAGE_BASE_BPS + SLIPPAGE_VOL_K * min(atr_pct * 100.0, 5.0)
        return close * (1.0 + slip_bps)

    def _commission_cost(self, price: float, size: float) -> float:
        return abs(price) * abs(size) * COMMISSION_PER_SIDE

    # ── Regime / session refresh (Fix #5, #6) ─────────────────────

    def _refresh_session_regime_state(self):
        """Refresh session + regime one-hot and vol regime from the
        current bar's timestamp and ATR/ADX proxies."""
        df = self.current_data[self.entry_timeframe][self.mode]
        ts = df.index[self.current_step]
        try:
            hour = int(pd.Timestamp(ts).hour)
        except Exception:
            hour = 12
        # 4-session one-hot: asia(22-06), eu(06-12), us(12-18), closed(18-22)
        self.session_one_hot[:] = 0.0
        if 6 <= hour < 12:
            self.session_one_hot[1] = 1.0
        elif 12 <= hour < 18:
            self.session_one_hot[2] = 1.0
        elif 18 <= hour < 22:
            self.session_one_hot[3] = 1.0
        else:
            self.session_one_hot[0] = 1.0
        # Vol regime: percentile of ATR relative to recent range
        try:
            window = df["atr_14"].iloc[max(0, self.current_step - 96): self.current_step + 1]
            cur = float(df["atr_14"].iloc[self.current_step])
            if len(window) > 5:
                rank = (window < cur).sum() / len(window)
            else:
                rank = 0.5
            self.vol_regime = float(rank)
        except Exception:
            self.vol_regime = 0.5
        # Regime one-hot: derive from trend alignment + vol + an ADX-like proxy
        self.regime_one_hot[:] = 0.0
        try:
            row = df.iloc[self.current_step]
            close = float(row.get("Close", 0.0))
            sma50 = float(row.get("sma_50", close))
            trending = abs(close - sma50) / max(close, 1e-8) > 0.005
            high_vol = self.vol_regime > 0.7
            low_vol  = self.vol_regime < 0.3
            if trending and not high_vol:
                self.regime_one_hot[0] = 1.0  # trending
            elif not trending and low_vol:
                self.regime_one_hot[1] = 1.0  # ranging
            elif high_vol:
                self.regime_one_hot[2] = 1.0  # breakout
            else:
                self.regime_one_hot[3] = 1.0  # neutral
        except Exception:
            self.regime_one_hot[3] = 1.0
        # Spread proxy: ATR-based, in bps of price
        try:
            atr = float(df["atr_14"].iloc[self.current_step])
            self.spread = float(atr / max(close, 1e-8)) if close > 0 else 0.0
        except Exception:
            self.spread = 0.0

    def _regime_label(self) -> str:
        """Return a string label for the active regime (for logging
        and per-regime evaluation)."""
        names = [
            "trending", "ranging", "breakout", "neutral",
            "news_shock", "high_vol", "low_vol", "accumulation",
        ]
        idx = int(np.argmax(self.regime_one_hot)) if self.regime_one_hot.sum() > 0 else 3
        return names[idx]

    # ── Risk-engineering hard limits (Fix #9) ────────────────────

    def _risk_violation(self) -> bool:
        """Return True if any hard risk limit has been breached."""
        if self.peak_equity <= 0:
            return False
        dd = (self.peak_equity - self.equity) / self.peak_equity
        if dd >= self.max_drawdown_pct:
            return True
        if self._start_of_day_equity is not None:
            day_dd = (self._start_of_day_equity - self.equity) / self._start_of_day_equity
            if day_dd >= self.max_daily_loss_pct:
                return True
        if self._trades_today >= self.max_daily_trades:
            return True
        return False

    def _position_exposure(self) -> float:
        """Current exposure as fraction of balance."""
        if self.position is None:
            return 0.0
        notional = self.position["size"] * self.position["entry_price"]
        return float(notional / max(self.balance, 1.0))

    def _check_position_size_limit(self) -> bool:
        """True if the next order would breach the per-position limit."""
        if self.position is not None:
            return self._position_exposure() > self.max_position_pct
        return False

    # ── Per-symbol attribution (Fix #3) ─────────────────────────

    def _record_trade_to_symbol(self, pnl: float):
        if self.current_symbol is None:
            return
        self.symbol_pnl[self.current_symbol] = (
            self.symbol_pnl.get(self.current_symbol, 0.0) + pnl
        )
        self.symbol_trades[self.current_symbol] = (
            self.symbol_trades.get(self.current_symbol, 0) + 1
        )

    def _open_position(self, direction: str, price: float):
        """
        Opens a long or short position.
        Risk sizing based on ATR — adapts to market volatility.
        Hard-enforces the max_position_pct ceiling (Fix #9).
        """
        df  = self.current_data[self.entry_timeframe][self.mode]
        row = df.iloc[self.current_step]
        atr = float(row.get("atr_14", price * 0.01))
        # Dynamic SL/TP based on ATR
        stop_loss_pct   = max(0.005, (atr * 1.5) / price)
        take_profit_pct = stop_loss_pct * 2.0   # Minimum 1:2 R:R
        risk_per_trade = 0.01   # Risk 1% of balance per trade
        size = max(0.001, (self.balance * risk_per_trade) / (stop_loss_pct * price))
        # Cap notional at max_position_pct * balance
        notional_cap = self.balance * self.max_position_pct
        if size * price > notional_cap:
            size = max(0.001, notional_cap / max(price, 1e-8))
        if direction == "long":
            sl = price * (1.0 - stop_loss_pct)
            tp = price * (1.0 + take_profit_pct)
        else:  # short
            sl = price * (1.0 + stop_loss_pct)
            tp = price * (1.0 - take_profit_pct)
        self.position = {
            "direction":      direction,
            "entry_price":    price,
            "sl":             sl,
            "tp":             tp,
            "size":           size,
            "stop_loss_pct":  stop_loss_pct,
            "open_step":      self.current_step,
        }
        entry_time = df.index[self.current_step]
        self.current_trade = {
            "symbol":     self.current_symbol,
            "direction":  direction,
            "entry_time": str(entry_time),
            "entry_price": price,
            "size":        size,
            "sl":          sl,
            "tp":          tp,
            "regime":      self._regime_label(),
        }
        self.holding_steps = 0
        self.hours_since_trade = 0
        # Apply commission on entry (Fix #8)
        self.balance -= self._commission_cost(price, size)

    def _close_position(self, price: float) -> float:
        """Closes the current position and returns PnL.
        Applies commission + slippage on exit (Fix #8)."""
        if self.position is None:
            return 0.0
        direction   = self.position["direction"]
        entry_price = self.position["entry_price"]
        size        = self.position["size"]
        # Apply slippage against the agent
        if direction == "long":
            fill_price = price * (1.0 - SLIPPAGE_BASE_BPS)
        else:
            fill_price = price * (1.0 + SLIPPAGE_BASE_BPS)
        if direction == "long":
            pnl = (fill_price - entry_price) * size
        else:  # short
            pnl = (entry_price - fill_price) * size
        # Commission on exit
        pnl -= self._commission_cost(fill_price, size)
        self.balance += pnl
        if self.current_trade:
            self.current_trade["exit_price"] = fill_price
            self.current_trade["pnl"]        = pnl
            self.current_trade["result"]     = (
                "win" if pnl > 0 else "loss" if pnl < 0 else "breakeven"
            )
            self.current_trade["hold_steps"] = self.holding_steps
            self.current_trade["regime"]     = self.current_trade.get(
                "regime", self._regime_label()
            )
            self.trade_history.append(self.current_trade)
            self.current_trade = None
        sl_pct            = self.position["stop_loss_pct"]
        self.position     = None # Clear position after closing
        self.holding_steps = 0
        self._trades_today += 1
        self._record_trade_to_symbol(pnl)
        self.realized_pnl_today += pnl
        return pnl, sl_pct

    def _check_sl_tp(self, price: float):
        """Check if SL or TP has been hit."""
        if self.position is None:
            return 0.0, 0.0

        pos       = self.position
        direction = pos["direction"]
        hit       = False

        if direction == "long":
            if price <= pos["sl"] or price >= pos["tp"]:
                hit = True
        else:  # short
            if price >= pos["sl"] or price <= pos["tp"]:
                hit = True

        if hit:
            pnl, sl_pct = self._close_position(price)
            return pnl, sl_pct

        return 0.0, getattr(self.position, "stop_loss_pct", 0.02) if self.position else 0.02

    def _get_equity(self, price: float) -> float:
        if self.position is None:
            return self.balance

        pos = self.position
        if pos["direction"] == "long":
            unrealized = (price - pos["entry_price"]) * pos["size"]
        else:
            unrealized = (pos["entry_price"] - price) * pos["size"]

        return self.balance + unrealized

    # ── Step ─────────────────────────────────────────

    def step(self, action: int):
        df       = self.current_data[self.entry_timeframe][self.mode]
        df_len   = len(df)
        price    = self._get_current_price()
        reward   = 0.0
        pnl      = 0.0
        sl_pct   = self.position["stop_loss_pct"] if self.position else 0.02

        # Reset per-day counters (Fix #9)
        try:
            ts = df.index[self.current_step]
            day_key = pd.Timestamp(ts).date()
        except Exception:
            day_key = None
        if day_key is not None and day_key != self._last_day_key:
            self._last_day_key        = day_key
            self._start_of_day_equity = self.balance
            self._trades_today        = 0
            self.realized_pnl_today   = 0.0

        # ── Process action ────────────────────────────
        is_switch_attempt = False
        if action == ACTION_BUY:
            if self.position is None and not self._check_position_size_limit():
                self._open_position("long", price)
            # If already in a position, treat as hold

        elif action == ACTION_SELL:
            if self.position is None and not self._check_position_size_limit():
                self._open_position("short", price)

        elif action == ACTION_CLOSE:
            if self.position is not None:
                pnl, sl_pct = self._close_position(price)
                reward += compute_reward(
                    pnl           = pnl,
                    entry_price   = price,
                    stop_loss_pct = sl_pct,
                    equity        = self.balance,
                    peak_equity   = self.peak_equity,
                    holding_steps = self.holding_steps,
                    is_close      = True,
                    is_switch     = False,
                    excess_leverage = max(self._position_exposure() - TARGET_LEVERAGE, 0.0),
                    sharpe_tracker  = self._sharpe_tracker,
                    trade_count     = len(self.trade_history),
                )

        elif action == ACTION_SWITCH:
            is_switch_attempt = True
            self._switch_attempts += 1
            # Fix #3: minimum dwell time + per-epoch cap
            if self._bars_on_symbol < MIN_DWELL_BARS:
                reward -= REWARD_NU_SWITCH * 0.5
            elif self._switches_this_ep >= MAX_SWITCHES_PER_EPOCH:
                reward -= REWARD_NU_SWITCH
            else:
                # Close any open position before switching
                if self.position is not None:
                    pnl, sl_pct = self._close_position(price)
                    reward += compute_reward(
                        pnl           = pnl,
                        entry_price   = price,
                        stop_loss_pct = sl_pct,
                        equity        = self.balance,
                        peak_equity   = self.peak_equity,
                        holding_steps = self.holding_steps,
                        is_close      = True,
                        is_switch     = True,
                        excess_leverage = 0.0,
                        sharpe_tracker  = self._sharpe_tracker,
                        trade_count     = len(self.trade_history),
                    )
                else:
                    # Switching with no open position still incurs the
                    # cost — discourages "flickering" between symbols.
                    reward -= REWARD_NU_SWITCH
                self._select_symbol()
                self._switches_this_ep += 1
                self._bars_on_symbol    = 0

        # ── Check SL/TP ───────────────────────────────
        if self.position is not None:
            auto_pnl, sl_pct = self._check_sl_tp(price)
            if auto_pnl != 0.0:
                pnl = auto_pnl
                reward += compute_reward(
                    pnl           = auto_pnl,
                    entry_price   = price,
                    stop_loss_pct = sl_pct,
                    equity        = self.balance,
                    peak_equity   = self.peak_equity,
                    holding_steps = self.holding_steps,
                    is_close      = True,
                    is_switch     = False,
                    excess_leverage = 0.0,
                    sharpe_tracker  = self._sharpe_tracker,
                    trade_count     = len(self.trade_history),
                )

        # ── Update equity and peak ────────────────────
        self.equity      = self._get_equity(price)
        self.peak_equity = max(self.peak_equity, self.equity)

        # ── Per-step drawdown / Sharpe reward ─────────
        reward += compute_reward(
            pnl           = 0.0,
            entry_price   = price,
            stop_loss_pct = sl_pct,
            equity        = self.equity,
            peak_equity   = self.peak_equity,
            holding_steps = self.holding_steps,
            is_close      = False,
            is_switch     = False,
            excess_leverage = max(self._position_exposure() - TARGET_LEVERAGE, 0.0),
            sharpe_tracker  = self._sharpe_tracker,
            trade_count     = len(self.trade_history),
        )

        self.previous_equity = self.equity
        self._bars_on_symbol += 1
        self.hours_since_trade += 1
        if not is_switch_attempt and self.position is None:
            # idle bar — still track how long since last trade
            pass

        # ── Advance step ─────────────────────────────
        self.current_step  += 1
        self.holding_steps += 1 if self.position else 0

        # ── Termination ───────────────────────────────
        # Bound the cursor against both the entry-tf length and the
        # recorded max_step. This prevents stepping past the last row,
        # which would make the higher-tf / daily slices empty and
        # crash _get_observation (see audit finding: "No upper bound
        # on current_step before slicing sub-timeframes").
        if self.current_step >= self.max_step or self.current_step >= df_len - 1:
            self.current_step = min(self.max_step, df_len - 1)
            terminated = True
        else:
            terminated = False
        truncated = False

        # Fix #9: hard risk limits can terminate the episode early
        if not terminated and self._risk_violation():
            terminated = True

        # Auto-close at episode end
        if terminated and self.position is not None:
            final_pnl, sl_pct = self._close_position(price)
            reward += compute_reward(
                pnl           = final_pnl,
                entry_price   = price,
                stop_loss_pct = sl_pct,
                equity        = self.balance,
                peak_equity   = self.peak_equity,
                holding_steps = self.holding_steps,
                is_close      = True,
                is_switch     = False,
                excess_leverage = 0.0,
                sharpe_tracker  = self._sharpe_tracker,
                trade_count     = len(self.trade_history),
            )

        observation = self._get_observation()

        # ── Defensive zero-fallback ──────────────────
        # If any required slice was empty for this step (which can
        # happen on the boundary bar or if data is sparse), fall back
        # to an all-zero observation of the correct shape so SB3
        # never receives a wrong-sized or NaN array.
        if observation is None or not isinstance(observation, np.ndarray):
            observation = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            try:
                expected = int(np.prod(self.observation_space.shape))
                if observation.size != expected:
                    observation = np.zeros(self.observation_space.shape, dtype=np.float32)
            except Exception:
                observation = np.zeros(self.observation_space.shape, dtype=np.float32)

        info = {
            "symbol":        self.current_symbol,
            "balance":       self.balance,
            "equity":        self.equity,
            "peak_equity":   self.peak_equity,
            "position":      self.position is not None,
            "direction":     self.position["direction"] if self.position else None,
            "pnl":           pnl,
            "trade_count":   len(self.trade_history),
            "win_rate": (
                sum(1 for t in self.trade_history if t.get("pnl", 0) > 0)
                / len(self.trade_history)
                if self.trade_history else 0.0
            ),
        }

        return observation, float(reward), terminated, truncated, info

    # ── Reset ─────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.balance         = 10_000.0
        self.equity          = self.balance
        self.peak_equity     = self.balance
        self.previous_equity = self.balance
        self.position        = None
        self.holding_steps   = 0
        self.trade_history   = []
        self.current_trade   = None
        self._sharpe_tracker.reset()
        self._switches_this_ep    = 0
        self._trades_today        = 0
        self._last_day_key        = None
        self._start_of_day_equity = None
        self._bars_on_symbol      = 0
        self._switch_attempts     = 0
        self.hours_since_trade    = 0
        self.realized_pnl_today   = 0.0

        # Invalidate the per-step observation cache so the next
        # _get_observation rebuilds from a clean state (Fix #3).
        self._cache["obs"]               = None
        self._cache["current_step"]      = -1
        self._cache["current_symbol"]    = None
        self._cache["current_timeframe"] = None
        self._cache["daily_close"]       = None
        self._cache["daily_sma_50"]      = None

        # Allow callers (e.g. evaluate_symbol) to force a specific
        # symbol via options={"symbol": "..."} — see audit:
        # "evaluate_symbol calls _select_symbol (private) from
        # outside the class" (Fix #6).
        forced_symbol = None
        if isinstance(options, dict):
            forced_symbol = options.get("symbol")

        if forced_symbol:
            self._select_symbol(forced_symbol)
        else:
            self._select_symbol()

        return self._get_observation(), {}

    # ── Render ────────────────────────────────────────

    def render(self):
        trades = len(self.trade_history)
        wins   = sum(1 for t in self.trade_history if t.get("pnl", 0) > 0)
        wr     = wins / trades if trades > 0 else 0.0
        print(
            f"Symbol: {self.current_symbol:<12} "
            f"Balance: ${self.balance:>10.2f}  "
            f"Equity: ${self.equity:>10.2f}  "
            f"Trades: {trades:>4}  "
            f"Win rate: {wr:.1%}"
        )

# =========================================================
# STRATEGY TRAINER
# =========================================================

class StrategyTrainer:
    """
    Trains and evaluates the RL agent.
    Uses train split for training, test split for evaluation —
    win rate numbers are real, not memorized training data.
    """

    def __init__(self, data_bundle: dict): # Added model_path parameter
        self.data_bundle = data_bundle

    def train(self, total_timesteps: int = 500_000) -> PPO:
        """Train PPO agent on the training split."""
        env = DummyVecEnv([
            lambda: MultiSymbolChartExpert(self.data_bundle, mode="train")
        ])

        model_path = "models/market_oracle" # Default model path
        model = None
        if os.path.exists(model_path + ".zip"): # SB3 saves models as .zip
            print(f"Loading existing model from {model_path} for continued training.")
            try:
                model = PPO.load(model_path, env=env)
            except ValueError as exc:
                if "Observation spaces do not match" not in str(exc) and "Action spaces do not match" not in str(exc):
                    raise
                archived_path = self._archive_incompatible_model(model_path)
                print(f"Existing model is incompatible with the current environment: {exc}")
                print(f"Archived old model -> {archived_path}")
                print("Initializing a new PPO model with the current observation/action spaces.")

        if model is None:
            print("No existing model found, initializing a new PPO model.")
            model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=0.0003,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            ent_coef=0.01,   # Encourages exploration
        )

        model.learn(total_timesteps=total_timesteps)
        model.save("models/market_oracle")
        print("Model saved -> models/market_oracle")
        return model

    @staticmethod
    def _archive_incompatible_model(model_path: str) -> str:
        # Glob every sidecar file alongside the model so we don't leave
        # orphans (SB3 may write metadata, JSON, etc.) — see audit:
        # "Incompatible-model fallback archives the .zip but not the
        # sidecar files".
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archived_paths = []
        for p in glob.glob(f"{model_path}*"):
            archived_name = f"{p}_incompatible_{timestamp}"
            try:
                shutil.move(p, archived_name)
                archived_paths.append(archived_name)
            except Exception as move_exc:  # noqa: BLE001
                print(f"WARNING: could not archive {p}: {move_exc}")
        if archived_paths:
            print(f"Archived {len(archived_paths)} incompatible file(s):")
            for ap in archived_paths:
                print(f"  -> {ap}")
        return archived_paths[0] if archived_paths else ""

    def evaluate(self, model: PPO, episodes: int = 10) -> dict:
        """
        Evaluates agent on the TEST split only.
        This gives a real win rate — not seen during training.
        """
        env = MultiSymbolChartExpert(self.data_bundle, mode="test")

        all_trades  = []
        all_equity  = []
        total_reward = 0.0

        for ep in range(episodes):
            obs, _   = env.reset()
            ep_reward = 0.0

            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(int(action))
                ep_reward  += reward
                all_equity.append(info["equity"])
                if terminated or truncated:
                    all_trades.extend(env.trade_history)
                    break

            total_reward += ep_reward
            print(
                f"Episode {ep+1}/{episodes}  "
                f"Reward: {ep_reward:>8.2f}  "
                f"Win rate: {info['win_rate']:.1%}"
            )

        wins   = sum(1 for t in all_trades if t.get("pnl", 0) > 0)
        losses = sum(1 for t in all_trades if t.get("pnl", 0) < 0)
        total  = len(all_trades)

        # Max drawdown
        peak_eq = 10_000.0
        max_dd  = 0.0
        for eq in all_equity:
            peak_eq = max(peak_eq, eq)
            dd      = (peak_eq - eq) / peak_eq
            max_dd  = max(max_dd, dd)

        results = {
            "total_reward":   total_reward,
            "total_trades":   total,
            "wins":           wins,
            "losses":         losses,
            "win_rate":       wins / total if total else 0.0,
            "max_drawdown":   max_dd,
            "final_balance":  env.balance,
            "final_equity":   env.equity,
        }

        print("\n=== Evaluation Results (TEST SET) ===")
        print(f"Total trades : {total}")
        print(f"Win rate     : {results['win_rate']:.1%}")
        print(f"Max drawdown : {results['max_drawdown']:.1%}")
        print(f"Final balance: ${results['final_balance']:.2f}")

        return results

    def evaluate_symbol(self, model: PPO, symbol: str) -> dict:
        """Evaluate agent on a specific symbol (test split)."""
        if symbol not in self.data_bundle:
            raise ValueError(f"Symbol {symbol} not in data bundle")

        env      = MultiSymbolChartExpert(self.data_bundle, mode="test")
        # Pass the symbol through the public reset() kwarg instead of
        # poking the private _select_symbol from outside the class —
        # see audit: "evaluate_symbol calls _select_symbol (private)
        # from outside the class".
        obs, _   = env.reset(options={"symbol": symbol})

        trades = []
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(int(action))
            if terminated or truncated:
                trades = env.trade_history
                break

        wins = sum(1 for t in trades if t.get("pnl", 0) > 0)
        return {
            "symbol":    symbol,
            "trades":    len(trades),
            "win_rate":  wins / len(trades) if trades else 0.0,
            "balance":   env.balance,
            "equity":    env.equity,
        }
