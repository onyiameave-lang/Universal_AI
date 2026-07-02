import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


class MarketOracleAgent:
    """
    Safe adapter for the existing MarketOracle trading system.

    This adapter performs cached analysis and status/report lookups. It does not
    start the live trading loop or place MT5 orders.
    """

    SYMBOL_ALIASES = {
        "EURUSD": "EURUSDX",
        "GBPUSD": "GBPUSDX",
        "BTCUSD": "BTC_USD",
        "ETHUSD": "ETH_USD",
        "LTCUSD": "LTC_USD",
    }

    def __init__(self, root_dir: Optional[str] = None, memory_ai: Any = None):
        self.root_dir = Path(root_dir or self._default_root()).resolve()
        self.memory_ai = memory_ai
        self._data_bundle = None
        if str(self.root_dir) not in sys.path:
            sys.path.insert(0, str(self.root_dir))

    @staticmethod
    def _default_root() -> str:
        return os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "MarketOracle-workspace")
        )

    def answer(self, query: str) -> Dict[str, Any]:
        symbol = self._extract_symbol(query)
        if not symbol:
            return self.get_status()
        return self.analyze_symbol(symbol)

    def get_status(self) -> Dict[str, Any]:
        return {
            "agent": "MarketOracle",
            "status": "available",
            "mode": "analysis_only",
            "live_trading_started": False,
            "root": str(self.root_dir),
            "available_symbols": self._available_symbols(),
            "note": "Use a request like 'analyze EURUSD' to run cached MarketOracle analysis.",
        }

    def analyze_symbol(self, requested_symbol: str) -> Dict[str, Any]:
        symbol = self._normalize_symbol(requested_symbol)
        data_bundle = self._load_data_bundle()

        if symbol not in data_bundle:
            return {
                "agent": "MarketOracle",
                "status": "not_found",
                "requested_symbol": requested_symbol,
                "normalized_symbol": symbol,
                "available_symbols": sorted(data_bundle.keys()),
                "message": f"No cached MarketOracle data found for {requested_symbol}.",
            }

        from experts.strategy_tester import SymbolAnalyzer
        from experts.chart_expert import load_strategy_for_symbol

        flat_data = self._flatten_symbol_data(data_bundle[symbol])
        symbol_analysis = SymbolAnalyzer.analyze_symbol(flat_data, symbol)
        symbol_type = SymbolAnalyzer.get_symbol_type(symbol_analysis)
        strategy = load_strategy_for_symbol(symbol)
        trade_idea = self._build_trade_idea(symbol, flat_data, symbol_analysis, symbol_type, strategy)

        result = {
            "agent": "MarketOracle",
            "status": "analyzed",
            "requested_symbol": requested_symbol,
            "symbol": symbol,
            "symbol_type": symbol_type,
            "overall_characteristics": symbol_analysis.get("overall_characteristics", []),
            "volatility_profile": symbol_analysis.get("volatility_profile", {}),
            "trend_strength": symbol_analysis.get("trend_strength", {}),
            "mean_reversion": symbol_analysis.get("mean_reversion", {}),
            "breakout_frequency": symbol_analysis.get("breakout_frequency", {}),
            "strategy_loaded": bool(strategy),
            "strategy_name": strategy.get("strategy_name") or strategy.get("name") or strategy.get("_source"),
            "trade_idea": trade_idea,
            "mode": "analysis_only",
            "live_trading_started": False,
        }

        if self.memory_ai and hasattr(self.memory_ai, "receive_contribution"):
            try:
                self.memory_ai.receive_contribution(
                    agent_id="market_oracle_adapter",
                    domain="trading",
                    concept=f"{symbol}_market_oracle_analysis",
                    three_ws={
                        "what": f"MarketOracle analyzed {symbol}",
                        "when": result.get("trade_idea", {}).get("timeframe", "cached data"),
                        "why": str(result.get("trade_idea", {})),
                    },
                    confidence=0.9,
                )
            except Exception:
                pass

        return result

    def _build_trade_idea(
        self,
        symbol: str,
        flat_data: Dict[str, pd.DataFrame],
        symbol_analysis: Dict[str, Any],
        symbol_type: str,
        strategy: Dict[str, Any],
    ) -> Dict[str, Any]:
        entry_tf = strategy.get("entry_tf") or self._default_entry_timeframe(symbol_type)
        df = flat_data.get(entry_tf)
        if df is None or df.empty:
            df = flat_data.get("1h")
        if df is None or df.empty:
            df = flat_data.get("daily")

        if df is None or df.empty:
            return {
                "action": "NO_TRADE",
                "confidence": 0.0,
                "timeframe": entry_tf,
                "reason": "No usable cached dataframe for trade idea.",
            }

        row = df.iloc[-1]
        price = self._float(row.get("Close"))
        atr = self._float(row.get("atr_14"), price * 0.005 if price else 0.0)
        rsi = self._float(row.get("rsi_14"), 50.0)
        macd = self._float(row.get("macd"), 0.0)
        macd_signal = self._float(row.get("macd_signal"), 0.0)
        sma_50 = self._float(row.get("sma_50"), price)

        if price is None:
            return {
                "action": "NO_TRADE",
                "confidence": 0.0,
                "timeframe": entry_tf,
                "reason": "Latest close price is unavailable.",
            }

        daily_trend = symbol_analysis.get("trend_strength", {}).get("daily", {}).get("trend_type", "weak")
        characteristics = symbol_analysis.get("overall_characteristics", [])
        bullish_votes = sum([
            price > sma_50,
            macd > macd_signal,
            rsi > 52,
        ])
        bearish_votes = sum([
            price < sma_50,
            macd < macd_signal,
            rsi < 48,
        ])

        action = "NO_TRADE"
        confidence = 0.35
        reason = "Mixed conditions; wait for cleaner confirmation."

        if "mean_reverting" in characteristics:
            if rsi <= 35 and price <= sma_50:
                action = "BUY"
                confidence = 0.58
                reason = "Mean-reversion setup: price is below/near its 50 SMA and RSI is oversold."
            elif rsi >= 65 and price >= sma_50:
                action = "SELL"
                confidence = 0.58
                reason = "Mean-reversion setup: price is above/near its 50 SMA and RSI is overbought."
            else:
                reason = "Mean-reverting/choppy profile, but RSI is not stretched enough for a high-quality entry."
        elif bullish_votes >= 2 and daily_trend != "weak":
            action = "BUY"
            confidence = 0.62
            reason = "Trend/momentum alignment: price, MACD, and RSI lean bullish."
        elif bearish_votes >= 2 and daily_trend != "weak":
            action = "SELL"
            confidence = 0.62
            reason = "Trend/momentum alignment: price, MACD, and RSI lean bearish."

        stop_loss = None
        take_profit = None
        if action == "BUY":
            stop_loss = price - max(atr * 1.5, price * 0.0015)
            take_profit = price + (price - stop_loss) * 1.5
        elif action == "SELL":
            stop_loss = price + max(atr * 1.5, price * 0.0015)
            take_profit = price - (stop_loss - price) * 1.5

        return {
            "action": action,
            "confidence": round(confidence, 2),
            "symbol": symbol,
            "timeframe": entry_tf,
            "entry_reference": round(price, 5),
            "stop_loss": round(stop_loss, 5) if stop_loss is not None else None,
            "take_profit": round(take_profit, 5) if take_profit is not None else None,
            "risk_reward": 1.5 if action in {"BUY", "SELL"} else None,
            "reason": reason,
            "indicators": {
                "rsi_14": round(rsi, 2),
                "macd": round(macd, 6),
                "macd_signal": round(macd_signal, 6),
                "sma_50": round(sma_50, 5),
                "atr_14": round(atr, 5),
            },
            "execution": "analysis_only_no_order_placed",
        }

    def _load_data_bundle(self) -> Dict[str, Any]:
        if self._data_bundle is None:
            from experts.chart_expert import load_data_bundle

            self._data_bundle = load_data_bundle(str(self.root_dir / "data"))
        return self._data_bundle

    @staticmethod
    def _flatten_symbol_data(symbol_data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, pd.DataFrame]:
        flat = {}
        for timeframe, splits in symbol_data.items():
            frames = [
                frame for frame in (splits.get("train"), splits.get("test"))
                if frame is not None and not frame.empty
            ]
            if frames:
                flat[timeframe] = pd.concat(frames).sort_index()
        return flat

    def _available_symbols(self):
        try:
            return sorted(self._load_data_bundle().keys())
        except Exception:
            return []

    @staticmethod
    def _default_entry_timeframe(symbol_type: str) -> str:
        if "trending" in symbol_type:
            return "1h"
        if "mean_reverting" in symbol_type:
            return "daily"
        return "15min"

    @staticmethod
    def _float(value, default=None):
        try:
            if pd.isna(value):
                return default
            return float(value)
        except Exception:
            return default

    @classmethod
    def _normalize_symbol(cls, symbol: str) -> str:
        clean = re.sub(r"[^A-Za-z0-9_]", "", symbol or "").upper()
        return cls.SYMBOL_ALIASES.get(clean, clean)

    @classmethod
    def _extract_symbol(cls, query: str) -> Optional[str]:
        query_upper = query.upper()
        for symbol in cls.SYMBOL_ALIASES:
            if symbol in query_upper or f"{symbol[:3]}/{symbol[3:]}" in query_upper:
                return symbol

        match = re.search(r"\b[A-Z]{6}\b", query_upper)
        if match:
            return match.group(0)

        crypto_match = re.search(r"\b(BTC|ETH|LTC)[/_-]?(USD)\b", query_upper)
        if crypto_match:
            return "".join(crypto_match.groups())

        return None
