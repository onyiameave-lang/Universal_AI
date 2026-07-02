"""
mt5_expert.py - MetaTrader 5 API Interface

Handles:
- Connection to MT5 terminal
- Historical data fetching for training/testing
- Live order execution (Long/Short/Close)
"""

import time
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime

# A unique identifier for orders placed by this EA
MAGIC_NUMBER = 202405

# Mapping MarketOracle timeframes to MT5 constants
TIMEFRAME_MAP = {
    "weekly": mt5.TIMEFRAME_W1,
    "daily":  mt5.TIMEFRAME_D1,
    "4h":     mt5.TIMEFRAME_H4,
    "1h":     mt5.TIMEFRAME_H1,
    "30min":  mt5.TIMEFRAME_M30,
    "15min":  mt5.TIMEFRAME_M15,
    "5min":   mt5.TIMEFRAME_M5,
    "1min":   mt5.TIMEFRAME_M1,
}

def connect_mt5():
    """Initializes connection to the MT5 terminal."""
    if not mt5.initialize():
        print(f"MT5 initialization failed, error code: {mt5.last_error()}")
        return False
    print("MT5 Connected Successfully")
    return True

def shutdown_mt5():
    """Properly closes the MT5 connection."""
    mt5.shutdown()
    print("MT5 Connection Closed")


def list_tradable_symbols(
    max_symbols: int = 50,
    min_volume: int = 0,
    include_groups: tuple = ("Forex", "CFD", "Crypto"),
):
    """
    Auto-discover tradable symbols from the connected MT5 terminal.

    Filters:
      * `visible=True` in Market Watch (i.e. selectable / tradeable)
      * `trade_mode != 0` (SYMBOL_TRADE_MODE_DISABLED)
      * Optional path-based group filter (e.g. "Forex\\", "Crypto\\")

    Returns a list of symbol names, capped at `max_symbols`.
    """
    if not mt5.initialize():
        print("MT5 not initialized — cannot list symbols.")
        return []

    selected = []
    for sym in mt5.symbols_get():
        info = mt5.symbol_info(sym.name)
        if info is None:
            continue
        if not info.visible:
            continue
        if getattr(info, "trade_mode", 0) == 0:
            continue
        if info.path and include_groups and not any(
            info.path.startswith(g) for g in include_groups
        ):
            continue
        if min_volume and (info.volume_min or 0) < min_volume:
            continue
        # Quick sanity check: must be able to fetch at least one bar on H1
        rates = mt5.copy_rates_from_pos(sym.name, mt5.TIMEFRAME_H1, 0, 2)
        if rates is None or len(rates) == 0:
            continue
        selected.append(sym.name)
        if len(selected) >= max_symbols:
            break

    return selected

def get_symbol_info(symbol: str):
    """Fetches symbol information from MT5."""
    if not mt5.initialize():
        return None
    return mt5.symbol_info(symbol)

def get_filling_mode(symbol: str):
    """
    Detects the supported filling mode for a symbol.
    Prevents 'Unsupported filling mode' (10030) errors common with some brokers.

    MT5 reports supported filling modes as a bitmask in `symbol_info.filling_mode`
    using these constants:
        SYMBOL_FILLING_FOK = 1
        SYMBOL_FILLING_IOC = 2
    So a symbol supporting BOTH reports 3, FOK-only reports 1, IOC-only reports 2.
    We prefer FOK > IOC > RETURN, matching what most brokers expect.
    """
    symbol_info = mt5.symbol_info(symbol)
    if not symbol_info:
        return mt5.ORDER_FILLING_IOC  # Default fallback

    filling_mode = symbol_info.filling_mode
    # Bitmask in symbol_info.filling_mode uses the same integer values as
    # ORDER_FILLING_* in the MetaTrader5 Python API.
    fok_ok = bool(filling_mode & mt5.ORDER_FILLING_FOK)
    ioc_ok = bool(filling_mode & mt5.ORDER_FILLING_IOC)

    if fok_ok:
        return mt5.ORDER_FILLING_FOK
    if ioc_ok:
        return mt5.ORDER_FILLING_IOC
    return mt5.ORDER_FILLING_RETURN

def get_mt5_data(symbol: str, timeframe_str: str, count: int = 10000):
    """
    Fetches historical bars from MT5 and returns a formatted DataFrame.

    Important: this function should be treated as *raw acquisition* only.
    Downstream indicator warm-up, NaN handling, row sufficiency checks, etc.
    are handled by core/mt5_data_validation_layer.py.

    Returns:
        DataFrame with index=DatetimeIndex named 'Date' and columns:
        Open, High, Low, Close, Volume
        or None if MT5 returns no data.
    """
    mt5_tf = TIMEFRAME_MAP.get(timeframe_str)
    if mt5_tf is None:
        print(f"Unsupported timeframe: {timeframe_str}")
        return None

    # Attempt 1: Fetch from a start date to encourage history download
    start_date = datetime(2010, 1, 1)  # reasonable start for most symbols
    rates = mt5.copy_rates_from(symbol, mt5_tf, start_date, count)

    if rates is None or len(rates) == 0:
        print(
            f"  Fetching from start date failed for {symbol} on {timeframe_str}. "
            "Falling back to position-based fetch."
        )
        # Attempt 2: Fallback to fetching from the current position
        rates = mt5.copy_rates_from_pos(symbol, mt5_tf, 0, count)
        if rates is None or len(rates) == 0:
            error_code, error_message = mt5.last_error()
            print(
                f"  No data for {symbol} on {timeframe_str}. "
                f"Error: ({error_code}, '{error_message}')"
            )
            if error_code == -1 and "Call failed" in error_message:
                print(
                    "    Hint: This often means the MT5 terminal does not have the "
                    "historical data for this symbol/timeframe."
                )
                print(
                    "    Try opening the chart in the terminal (e.g., drag the symbol "
                    "onto a chart window and select the timeframe) to force a download."
                )
            return None

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")

    # MT5 can return data with timezone info. We'll make it timezone-naive.
    if df["time"].dt.tz is not None:
        try:
            df["time"] = df["time"].dt.tz_convert(None)
        except Exception as e:
            print(f"Warning: Could not convert timezone for {symbol} on {timeframe_str}: {e}")

    df = df.rename(
        columns={
            "time": "Date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "tick_volume": "Volume",
        }
    )

    df = df.set_index("Date")

    # Ensure ordering and column set; keep duplicates for validation layer to decide.
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df = df.sort_index()

    return df


def execute_mt5_order(symbol: str, action: int, lot_size: float = 0.01,
                      sl_price: float = None, tp_price: float = None, ticket: int = 0):
    """
    Executes trade on MT5 terminal.
    Actions: 1=Buy, 2=Sell, 3=Close
    """
    if not mt5.initialize():
        print("MT5 not initialized for trade execution.")
        return None

    # Ensure symbol is selected
    if not mt5.symbol_select(symbol, True):
        print(f"Failed to select symbol: {symbol}. Ensure it is visible in Market Watch.")
        return None

    # Current market info
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        print(f"Failed to get tick for {symbol}. Error: {mt5.last_error()}")
        return None

    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"Failed to get symbol info for {symbol}. Error: {mt5.last_error()}")
        return None

    # Normalize SL / TP to the symbol's precision
    sl_price = normalize_price(sl_price, symbol_info) if sl_price is not None else None
    tp_price = normalize_price(tp_price, symbol_info) if tp_price is not None else None

    # Detect the correct filling mode for the broker
    filling_mode = get_filling_mode(symbol)

    # Retcodes that justify a retry (price moved or requote)
    RETRYABLE_RETCODES = {
        getattr(mt5, "TRADE_RETCODE_REQUOTE", 10004),
        getattr(mt5, "TRADE_RETCODE_PRICE_CHANGED", 10005),
        getattr(mt5, "TRADE_RETCODE_PRICE_OFF", 10021),
    }

    attempt = 0
    last_result = None
    while attempt <= max_retries:
        # Refresh tick for each attempt
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            print(f"Failed to refresh tick for {symbol}. Error: {mt5.last_error()}")
            return None

        request = {
            "symbol":        symbol,
            "volume":        lot_size,
            "type_time":     mt5.ORDER_TIME_GTC,
            "type_filling":  filling_mode,
            "deviation":     deviation,
            "magic":         MAGIC_NUMBER,
        }

        if action == 1:  # Buy
            request.update({
                "action":  mt5.TRADE_ACTION_DEAL,
                "type":    mt5.ORDER_TYPE_BUY,
                "price":   tick.ask, # Use ask price for buys
                "comment": "MarketOracle Long",
            })
            if sl_price is not None:
                request["sl"] = sl_price
            if tp_price is not None:
                request["tp"] = tp_price

        elif action == 2:  # Sell
            request.update({
                "action":  mt5.TRADE_ACTION_DEAL,
                "type":    mt5.ORDER_TYPE_SELL,
                "price":   tick.bid, # Use bid price for sells
                "comment": "MarketOracle Short",
            })
            if sl_price is not None:
                request["sl"] = sl_price
            if tp_price is not None:
                request["tp"] = tp_price

        elif action == 3:  # Close
            # Close must target the specific position ticket requested.
            if ticket == 0:
                print("Error: Close action requires a valid position ticket.")
                return None

            # Find the exact position by ticket.
            pos = mt5.positions_get(ticket=ticket)
            if not pos:
                # Fallback: try symbol+magic scan (should not happen often)
                positions = mt5.positions_get(symbol=symbol, magic=MAGIC_NUMBER)
                if not positions:
                    print(f"Error: Position with ticket {ticket} not found.")
                    return None
                # pick the first as last resort
                pos = positions[0]
            else:
                pos = pos[0]

            order_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
            price = tick.bid if pos.type == mt5.POSITION_TYPE_BUY else tick.ask

            request.update({
                "action":    mt5.TRADE_ACTION_DEAL,
                "type":      order_type,
                "position":  pos.ticket,
                "volume":    pos.volume,
                "price":     price,
                "comment":   "MarketOracle Close", # Clear SL/TP on close
                "sl":        0.0,
                "tp":        0.0,
            })
        else:
            print(f"Unsupported MT5 action: {action}")
            return None

        result = mt5.order_send(request)
        last_result = result

        if result is None:
            print(f"MT5 order_send returned None. Last error: {mt5.last_error()}")
            return None

        if result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"Order executed successfully! Ticket: {result.order}")
            return result

        # Retry only on transient retcodes
        if result.retcode in RETRYABLE_RETCODES and attempt < max_retries:
            print(f"  Retcode {result.retcode} ({result.comment}) — retrying ({attempt + 1}/{max_retries})...")
            attempt += 1
            time.sleep(0.5 * attempt) # Exponential backoff
            continue

        # Permanent failure
        print(f"Order failed! Retcode: {result.retcode}")
        print(f"  Reason: {result.comment}")
        print(f"  MT5 Last Error: {mt5.last_error()}")
        return result

    return last_result
