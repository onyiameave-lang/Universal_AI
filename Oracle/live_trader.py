
"""live_trader.py — Orchestrates live execution by connecting MT5, ChartExpert, and the trained RL Model.
Leverages learned strategies and RL model for dynamic signal generation and risk management."""

import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod
from enum import Enum
import pandas as pd
import statistics
from typing import Any
import numpy as np
from stable_baselines3 import PPO

from experts.mt5_expert import connect_mt5, get_mt5_data, execute_mt5_order, get_symbol_info
from experts.chart_expert import add_technical_indicators, load_strategy_for_symbol

class TradeSignal(Enum):
    """Trading signal types."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CLOSE = "CLOSE"
    SWITCH = "SWITCH"


class TradingState(Enum):
    """Position states."""
    NO_POSITION = "NO_POSITION"
    LONG = "LONG"
    SHORT = "SHORT"
    WAITING = "WAITING"


class MarketDataFetcher(ABC):
    """Abstract base for market data sources."""
    
    @abstractmethod
    def fetch_price(self, symbol: str) -> float:
        """Get current price."""
        pass
    
    @abstractmethod
    def fetch_ohlcv(self, symbol: str, timeframe: str) -> List[Dict]:
        """Get OHLCV data."""
        pass
    
    @abstractmethod
    def fetch_bid_ask(self, symbol: str) -> Tuple[float, float]:
        """Get bid/ask prices."""
        pass


class MT5MarketData(MarketDataFetcher):
    """Real market data from MetaTrader 5."""
    
    def __init__(self):
        pass # Connection is handled dynamically in fetch_price/ohlcv if needed

    def fetch_price(self, symbol: str) -> float:
        import MetaTrader5 as mt5
        # Ensure symbol is visible in MarketWatch
        if not mt5.symbol_select(symbol, True):
            return None
        tick = mt5.symbol_info_tick(symbol)
        return tick.ask if tick else None

    def fetch_ohlcv(self, symbol: str, timeframe: str = "1h") -> List[Dict]:
        import MetaTrader5 as mt5
        mt5.symbol_select(symbol, True)
        df = get_mt5_data(symbol, timeframe, count=100)
        if df is None:
            return []
        return df.reset_index().to_dict('records')

    def fetch_bid_ask(self, symbol: str) -> Tuple[float, float]:
        import MetaTrader5 as mt5
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            return None, None
        return tick.bid, tick.ask

    def fetch_multi_tf_data(self, symbol: str, timeframes: List[str]) -> Dict[str, pd.DataFrame]:
        """Fetches and validates timeframes using the adaptive MT5 layer."""
        from core.market_data_manager import MarketDataManager

        mdm = MarketDataManager(
            memory_ai=None,
            mt5_connector=None,
            mt5_getter=get_mt5_data,
            logger=print,
        )

        bundle: Dict[str, pd.DataFrame] = mdm.get_validated_bundle(
            symbol=symbol,
            timeframes=timeframes,
            window_size=20,
            warmup_buffer=50,
        )

        return bundle


class RLSignalGenerator:
    """Uses the trained Brain (PPO Model) to generate signals based on learned knowledge."""
    
    def __init__(self, model_path: str, memory_ai: Optional[Any] = None):
        self.model = PPO.load(model_path)
        self.action_map = {
            0: TradeSignal.HOLD,
            1: TradeSignal.BUY,
            2: TradeSignal.SELL,
            3: TradeSignal.CLOSE,
            4: TradeSignal.SWITCH
        }
        self.memory_ai = memory_ai

    def generate_signal(self, symbol: str, data_bundle: Dict[str, pd.DataFrame]) -> Dict:
        """Uses the RL model to decide the next action."""
        strat_config = load_strategy_for_symbol(symbol)
        obs = self._build_observation(data_bundle, strat_config)
        
        if obs is None:
            return {"signal": TradeSignal.HOLD, "confidence": 0.0, "reason": "Data building failed"}

        action, _ = self.model.predict(obs, deterministic=True)

        # Log decision to Memory AI if available (optional; don't fail trading)
        if self.memory_ai:
            try:
                # Some MemoryAI implementations expose different method names.
                # Keep it defensive.
                if hasattr(self.memory_ai, "learn_concept"):
                    self.memory_ai.learn_concept(
                        domain="trading",
                        concept=f"{symbol}_{self.action_map.get(int(action), TradeSignal.HOLD).value}",
                        source_text=f"RL Agent decision: {self.action_map.get(int(action), TradeSignal.HOLD).value}, Confidence: 1.0, Reason: RL Agent decision for {symbol}",
                        agent_name="market_oracle_rl_agent",
                        confidence=1.0
                    )
                elif hasattr(self.memory_ai, "log_concept"):
                    self.memory_ai.log_concept(
                        domain="trading",
                        concept=f"{symbol}_{self.action_map.get(int(action), TradeSignal.HOLD).value}",
                        source_text=f"RL Agent decision: {self.action_map.get(int(action), TradeSignal.HOLD).value}, Confidence: 1.0, Reason: RL Agent decision for {symbol}",
                        agent_name="market_oracle_rl_agent",
                        confidence=1.0
                    )
                else:
                    # No compatible API; silently skip.
                    pass
            except Exception as e:
                print(f"✗ Memory AI logging error: {e}")

        
        return {
            "signal": self.action_map.get(int(action), TradeSignal.HOLD),
            "confidence": 1.0,
            "reason": f"RL Agent decision for {symbol}"
        }

    def _build_observation(self, data_bundle, config):
        """Reconstructs the feature vector exactly as the RL model expects."""
        # Primary timeframe selection:
        # 1) If strategy config specifies an explicit entry_tf/primary_timeframe, use it.
        # 2) Else, infer from what the strategy learned/contains.
        #    - Many configs produced by the RL training pipeline use `entry_tf`.
        # 3) Else, fall back to `1h`.
        primary_tf = (
            config.get("primary_timeframe")
            or config.get("entry_tf")
            or config.get("preferred_timeframe")
            or "1h"
        )
        df = data_bundle.get(primary_tf)

        
        if df is None or df.empty:
            return None
            
        # NOTE:
        # The PPO model in chart_expert.py was trained on MultiSymbolChartExpert
        # which expects a fixed observation vector size of 164.
        # This live trader previously built only 7 features, causing:
        #   Unexpected observation shape (7,) for Box environment.
        #
        # To keep the bot from crashing (and to allow trading/close execution),
        # we pad/truncate the live observation to match the expected length.
        #
        # Source 7 features (quick proxy):
        features = ["rsi_14", "atr_14", "macd", "macd_signal", "stoch_k", "stoch_d", "sma_50"]
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            return None

        base = df[features].iloc[-1].values.astype(np.float32)
        expected_obs_size = 164
        if base.shape[0] == expected_obs_size:
            return base
        if base.shape[0] > expected_obs_size:
            return base[:expected_obs_size]
        # pad with zeros to expected length
        padded = np.zeros(expected_obs_size, dtype=np.float32)
        padded[: base.shape[0]] = base
        return padded



class PositionManager:
    """Manage open positions and trades."""
    
    def __init__(self, initial_capital: float = 10000.0):
        self.capital = initial_capital
        self.positions = {}  # symbol -> position data
        self.trade_history = []
        self.equity_history = []
        self.use_live_broker = True
    
    def open_position(self, symbol: str, entry_price: float, 
                     quantity: int, trade_type: str = "LONG",
                     stop_loss: Optional[float] = None,
                     take_profit: Optional[float] = None) -> Dict:
        """Open a new position."""
        
        """Executes a real trade via mt5_expert and tracks it."""
        cost = entry_price * quantity
        if cost > self.capital:
            return {"status": "failed", "reason": "Insufficient capital"}

        # MT5 Execution
        # REAL EXECUTION via mt5_expert
        if self.use_live_broker:
            import MetaTrader5 as mt5
            symbol_info = get_symbol_info(symbol)
            if symbol_info is None:
                return {"status": "failed", "reason": f"MT5 Error: Symbol info for {symbol} not found"}
            
            # Convert quantity (units) to MT5 lot_size
            # Assuming quantity is in base currency units, trade_contract_size is usually 100000 for forex pairs
            mt5_lot_size = quantity / symbol_info.trade_contract_size
            
            action_code = 1 if trade_type == "LONG" else 2
            res = execute_mt5_order(symbol, action_code, lot_size=mt5_lot_size, sl_price=stop_loss, tp_price=take_profit)
            if not res or res.retcode != 10009:
                return {"status": "failed", "reason": f"MT5 Error: {getattr(res, 'retcode', 'Unknown')}"}
        
        print(f"  💰 LIVE ORDER PLACED: {trade_type} {symbol} @ {entry_price}")

        self.positions[symbol] = {
            "type": trade_type,
            "entry_price": entry_price,
            "quantity": quantity,
            "entry_time": datetime.now().isoformat(),
            "stop_loss": stop_loss,
            "take_profit": take_profit, 
            "ticket": getattr(res, 'order', 0) if self.use_live_broker else 0,
            "status": "OPEN",
            "cost": cost
        }
        
        self.capital -= cost
        
        return {
            "status": "success",
            "symbol": symbol,
            "type": trade_type,
            "quantity": quantity,
            "entry_price": entry_price,
            "cost": cost
        }
    
    def close_position(self, symbol: str, exit_price: float) -> Dict:
        if symbol not in self.positions:
            return {"status": "failed", "reason": "No position"}
        
        position = self.positions[symbol]
        quantity = position["quantity"]
        entry_price = position["entry_price"]
        
        revenue = exit_price * quantity
        cost = position["cost"]
        profit = revenue - cost

        # REAL EXECUTION via mt5_expert
        if self.use_live_broker:
            position_ticket = position.get("ticket")
            if position_ticket:
                execute_mt5_order(symbol, 3, ticket=position_ticket)
            else:
                print(f"  WARNING: No MT5 ticket found for {symbol} to close.")

        profit_percent = (profit / cost) * 100
        
        self.capital += revenue
        
        # Record trade
        trade_record = {
            "symbol": symbol,
            "type": position["type"],
            "entry_price": entry_price,
            "exit_price": exit_price,
            "quantity": quantity,
            "profit": profit,
            "profit_percent": profit_percent,
            "duration": (datetime.now() - datetime.fromisoformat(position["entry_time"])).total_seconds(),
            "closed_at": datetime.now().isoformat()
        }
        self.trade_history.append(trade_record)
        
        del self.positions[symbol]
        
        return {
            "status": "success",
            "symbol": symbol,
            "profit": profit,
            "profit_percent": profit_percent
        }
    
    def update_position_prices(self, symbol: str, current_price: float):
        """Update position with current price for P&L calculation."""
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        position["current_price"] = current_price
        
        # Check stop loss
        if position["stop_loss"]:
            if position["type"] == "LONG" and current_price <= position["stop_loss"]:
                return {"action": "STOP_LOSS", "trigger_price": position["stop_loss"]}
            elif position["type"] == "SHORT" and current_price >= position["stop_loss"]:
                return {"action": "STOP_LOSS", "trigger_price": position["stop_loss"]}
        
        # Check take profit
        if position["take_profit"]:
            if position["type"] == "LONG" and current_price >= position["take_profit"]:
                return {"action": "TAKE_PROFIT", "trigger_price": position["take_profit"]}
            elif position["type"] == "SHORT" and current_price <= position["take_profit"]:
                return {"action": "TAKE_PROFIT", "trigger_price": position["take_profit"]}
        
        return None
    
    def get_portfolio_stats(self) -> Dict:
        """Get current portfolio statistics."""
        
        total_value = self.capital
        open_positions_value = 0
        
        for symbol, position in self.positions.items():
            if "current_price" in position:
                open_positions_value += position["current_price"] * position["quantity"]
        
        total_value += open_positions_value
        
        # Calculate returns
        win_trades = [t for t in self.trade_history if t["profit"] > 0]
        loss_trades = [t for t in self.trade_history if t["profit"] < 0]
        
        win_rate = len(win_trades) / len(self.trade_history) * 100 if self.trade_history else 0
        total_profit = sum(t["profit"] for t in self.trade_history)
        
        return {
            "total_capital": self.capital,
            "open_positions_value": open_positions_value,
            "total_portfolio_value": total_value,
            "total_profit": total_profit,
            "win_rate": win_rate,
            "trades_completed": len(self.trade_history),
            "open_positions": len(self.positions),
            "equity": total_value
        }


class RiskManager:
    """Manage trading risk."""
    
    def __init__(self, max_position_size: float = 0.05,
                 max_daily_loss: float = 0.02,
                 position_risk_percent: float = 1.0):
        """
        max_position_size: Max % of capital per trade
        max_daily_loss: Max % daily loss
        position_risk_percent: Risk % for position sizing
        """
        self.max_position_size = max_position_size
        self.max_daily_loss = max_daily_loss
        self.position_risk_percent = position_risk_percent
        self.daily_loss = 0.0
    
    def calculate_position_size(self, capital: float, entry_price: float,
                               stop_loss_price: float) -> int:
        """Calculate position size based on risk."""
        
        risk_amount = capital * self.position_risk_percent / 100
        price_risk = abs(entry_price - stop_loss_price)
        
        if price_risk == 0:
            return 0
        
        quantity = int(risk_amount / price_risk)
        
        # Don't exceed max position size
        max_quantity = int((capital * self.max_position_size) / entry_price)
        
        return min(quantity, max_quantity)
    
    def calculate_stop_loss(self, entry_price: float, atr: float = None) -> float:
        """Calculate stop loss price."""
        
        if atr is None:
            atr = entry_price * 0.02  # 2% default
        
        return entry_price - atr
    
    def calculate_take_profit(self, entry_price: float, stop_loss: float,
                            risk_reward_ratio: float = 2.0) -> float:
        """Calculate take profit price."""
        
        risk = entry_price - stop_loss
        return entry_price + (risk * risk_reward_ratio)
    
    def check_daily_loss_limit(self, current_loss: float) -> bool:
        """Check if daily loss limit exceeded."""
        return abs(current_loss) > self.max_daily_loss
    
    def update_daily_loss(self, loss: float):
        """Update daily loss."""
        self.daily_loss += loss
    
    def reset_daily_loss(self):
        """Reset daily loss (at market open)."""
        self.daily_loss = 0.0


class LiveTradingBot:
    """Main live trading bot."""
    
    def __init__(self, symbols: List[str], capital: float = 10000.0,
                 data_fetcher: MarketDataFetcher = None,
                 model_path: str = "models/market_oracle.zip",
                 memory_ai = None):
        self.symbols = symbols
        self.capital = capital
        self.data_fetcher = data_fetcher
        self.memory_ai = memory_ai
        
        self.rl_signal_generator = RLSignalGenerator(model_path, memory_ai=self.memory_ai)
        
        self.position_manager = PositionManager(capital)
        self.risk_manager = RiskManager()
        self.current_symbol_idx = 0
        self.running = False
        self.trading_thread = None
        self.timeframes = ["weekly", "daily", "4h", "1h", "15min"]
        self.price_history = {symbol: [] for symbol in symbols}
        self.last_signals = {}
    

    def start(self, interval_seconds: int = 60):
        """Start live trading."""
        self.running = True
        self.trading_thread = threading.Thread(
            target=self._trading_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self.trading_thread.start()
        print(f"🚀 Live trading started for {self.symbols}")
    
    def stop(self):
        """Stop live trading."""
        self.running = False
        if self.trading_thread:
            self.trading_thread.join(timeout=10)
        print("⏸️  Live trading stopped")
    
    def _trading_loop(self, interval_seconds: int):
        """Main trading loop."""
        
        while self.running:
            try:
                print(f"\n{'='*80}")
                print(f"⏰ Trading Cycle: {datetime.now().isoformat()}")
                print(f"{'='*80}")
                
                symbol = self.symbols[self.current_symbol_idx]
                self._trade_symbol(symbol)
                
                # Print stats
                stats = self.position_manager.get_portfolio_stats()
                print(f"\n📊 Portfolio Stats:")
                print(f"  Capital: ${stats['total_capital']:.2f}")
                print(f"  Total Value: ${stats['total_portfolio_value']:.2f}")
                print(f"  Profit: ${stats['total_profit']:.2f}")
                print(f"  Win Rate: {stats['win_rate']:.1f}%")
                

                time.sleep(interval_seconds)
                
            except Exception as e:
                print(f"✗ Trading loop error: {e}")
                time.sleep(interval_seconds)
    
    def _trade_symbol(self, symbol: str):
        """Trade a single symbol."""
        
        print(f"\n📈 Trading {symbol}...")
        
        # Fetch data
        current_price = self.data_fetcher.fetch_price(symbol)
        if current_price is None:
            print(f"  ✗ Could not fetch price for {symbol}. Check if the symbol name matches your MT5 terminal exactly (e.g., EURUSD vs EURUSD.m).")
            return
        
        print(f"  Current price: ${current_price:.2f}")
        
        data_bundle = self.data_fetcher.fetch_multi_tf_data(symbol, self.timeframes)
        if not data_bundle:
            print(f"  ✗ No data bundle available for {symbol}")
            return
            
        signal_data = self.rl_signal_generator.generate_signal(symbol, data_bundle)
        
        sig_val = signal_data['signal'].value
        print(f"  RL Signal: {sig_val}")
        print(f"  Reason: {signal_data['reason']}")
        
        # Execute trades
        if symbol in self.position_manager.positions:
            # Update existing position
            check = self.position_manager.update_position_prices(symbol, current_price)
            if check:
                print(f"  Action: {check['action']}")
                close_result = self.position_manager.close_position(symbol, current_price)
                if close_result.get("status") == "success":
                    self._on_trade_closed(symbol, close_result)


            
            # Consider closing on signal
            if signal_data['signal'] == TradeSignal.CLOSE:
                print(f"  Signal to CLOSE position for {symbol}.")
                close_result = self.position_manager.close_position(symbol, current_price)  # This will use the ticket if available
                if close_result.get("status") == "success":
                    self._on_trade_closed(symbol, close_result)



        elif signal_data['signal'] == TradeSignal.SWITCH:
            self.current_symbol_idx = (self.current_symbol_idx + 1) % len(self.symbols)
            print(f"  🔄 Switching to next symbol: {self.symbols[self.current_symbol_idx]}")
        else:
            # Entry logic
            if signal_data['signal'] == TradeSignal.BUY and signal_data['confidence'] >= 0.6:
                strat_config = load_strategy_for_symbol(symbol)
                atr_value = strat_config.get("atr_multiplier", 2.0) * current_price * 0.01
                
                stop_loss = self.risk_manager.calculate_stop_loss(current_price, atr=atr_value)
                take_profit = self.risk_manager.calculate_take_profit(
                    current_price, stop_loss, risk_reward_ratio=strat_config.get("risk_reward_ratio", 1.5)
                )
                quantity = self.risk_manager.calculate_position_size(
                    self.position_manager.capital, current_price, stop_loss
                )
                
                if quantity > 0:
                    # Convert quantity to lot_size for MT5 (e.g., 1 lot = 100,000 units for forex)
                    result = self.position_manager.open_position(
                        symbol, current_price, quantity,
                        stop_loss=stop_loss,
                        take_profit=take_profit
                    )
                    print(f"  Opened: {quantity} @ ${current_price:.2f}")
                    print(f"  Stop Loss: ${stop_loss:.2f}, Take Profit: ${take_profit:.2f}")
    
    def _on_trade_closed(self, symbol: str, close_result: Dict[str, Any]):
        """Autonomous online symbol-specific strategy re-optimization after every closed trade.

        Keeps per-symbol configs as rolling best-fit by re-running
        the existing strategy fitting pipeline for that symbol.
        """
        # Track rolling fit state (in-memory)
        if not hasattr(self, "strategy_fit_state"):
            self.strategy_fit_state = {}

        profit = float(close_result.get("profit", 0.0) or 0.0)
        tag = "win" if profit > 0 else "loss" if profit < 0 else "breakeven"

        state = self.strategy_fit_state.setdefault(symbol, {
            "wins": 0,
            "losses": 0,
            "breakeven": 0,
            "total_trades": 0,
            "total_profit": 0.0,
            "_last_reopt_ts": None,
        })

        state["total_trades"] += 1
        state["total_profit"] += profit
        if tag == "win":
            state["wins"] += 1
        elif tag == "loss":
            state["losses"] += 1
        else:
            state["breakeven"] += 1

        win_rate = state["wins"] / state["total_trades"] if state["total_trades"] else 0.0
        print(f"  ↪ Trade closed for {symbol}: {tag} profit={profit:.2f} win_rate={win_rate:.1%}")

        # Rolling policy: re-opt after any loss early, or if win-rate drops
        lookback = 8
        recent = [t for t in self.position_manager.trade_history if t.get("symbol") == symbol][-lookback:]
        if len(recent) >= lookback:
            recent_wr = sum(1 for t in recent if t.get("profit", 0) > 0) / len(recent)
        else:
            recent_wr = win_rate

        should_reopt = (tag == "loss") or (recent_wr < 0.45)
        if not should_reopt:
            return

        # Throttle to avoid hammering data downloads
        min_seconds_between_reopt = 120
        now = time.time()
        last = state.get("_last_reopt_ts")
        if last is not None and (now - last) < min_seconds_between_reopt:
            return
        state["_last_reopt_ts"] = now

        print(f"\n{'!'*20} AUTONOMOUS RE-OPTIMIZE ({symbol}) {'!'*20}")
        print(f"Recent win-rate: {recent_wr:.1%}. Re-running symbol-fit optimizer...")

        try:
            from main import step_optimize_strategies, step_load_data

            data_bundle = step_load_data(use_mt5=True, symbols=[symbol], max_symbols=1)
            if symbol not in data_bundle:
                print(f"Could not fetch data for {symbol} to re-optimize.")
                return

            # Symbol-aware optimization pipeline writes optimized configs
            # that chart_expert will load.
            step_optimize_strategies(data_bundle, skip_gemini=True)

            # Ensure latest config is picked up immediately
            from experts.chart_expert import load_strategy_for_symbol
            _ = load_strategy_for_symbol(symbol)

            print(f"Re-optimization for {symbol} complete. Updated best-fit strategy config active.")
        except Exception as e:
            print(f"  ✗ Re-optimization failed: {e}")
        print(f"{'!'*63}\n")

    def _check_and_reoptimize(self, symbol: str, lookback: int = 10, win_rate_threshold: float = 0.4):

        """
        Checks recent performance for a symbol and triggers re-optimization if it's poor.
        This creates a continuous learning feedback loop.
        """
        recent_trades = [
            t for t in self.position_manager.trade_history 
            if t.get("symbol") == symbol
        ][-lookback:]

        if len(recent_trades) < lookback:
            return # Not enough data to make a decision

        wins = sum(1 for t in recent_trades if t.get("profit", 0) > 0)
        current_win_rate = wins / len(recent_trades)

        if current_win_rate < win_rate_threshold:
            print(f"\n{'!'*20} RE-OPTIMIZATION TRIGGERED {'!'*20}")
            print(f"Symbol {symbol} performance is poor (Win Rate: {current_win_rate:.1%}).")
            print("Running strategy optimization to find a better fit...")

            # This is where we invoke the optimization logic from main.py
            try:
                from main import step_optimize_strategies, step_load_data
                
                # We need the data bundle for the symbol to run optimization
                data_bundle = step_load_data(use_mt5=True, symbols=[symbol], max_symbols=1)
                if symbol in data_bundle:
                    step_optimize_strategies(data_bundle, skip_gemini=True) # Use skip_gemini to avoid API costs
                    print(f"Re-optimization for {symbol} complete. The new strategy will be used on the next cycle.")
                else:
                    print(f"Could not fetch data for {symbol} to re-optimize.")

            except Exception as e:
                print(f"  ✗ Re-optimization failed: {e}")
            print(f"{'!'*63}\n")

    def get_report(self) -> Dict:
        """Get trading report."""
        stats = self.position_manager.get_portfolio_stats()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "portfolio": stats,
            "open_positions": self.position_manager.positions,
            "recent_trades": self.position_manager.trade_history[-10:],
            "symbols_traded": self.symbols
        }
    
    def print_report(self):
        """Print formatted report."""
        report = self.get_report()
        
        print("\n" + "="*80)
        print("📊 TRADING REPORT")
        print("="*80)
        
        portfolio = report['portfolio']
        print(f"\nPortfolio Value: ${portfolio['total_portfolio_value']:.2f}")
        print(f"Capital: ${portfolio['total_capital']:.2f}")
        print(f"Open Positions Value: ${portfolio['open_positions_value']:.2f}")
        print(f"Total Profit: ${portfolio['total_profit']:.2f}")
        print(f"Win Rate: {portfolio['win_rate']:.1f}%")
        print(f"Trades: {portfolio['trades_completed']}")
        
        print(f"\nOpen Positions: {portfolio['open_positions']}")
        for symbol, position in report['open_positions'].items():
            if "current_price" in position:
                pnl = (position["current_price"] - position["entry_price"]) * position["quantity"]
                print(f"  {symbol}: {position['quantity']} @ ${position['entry_price']:.2f} (PnL: ${pnl:.2f})")
        
        print("\n" + "="*80)


# Example: Running live trading
if __name__ == "__main__":
    # Check if we should use memory_ai
    memory_ai = None
    try:
        import os, sys
        # MemoryAI lives here:
        #   c:/Users/.../AI-ECOSYSTEM/ai-memory-system-/core/OPTIMIZED_memory_ai_system.py
        mem_core = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "ai-memory-system-", "core")
        )
        if mem_core not in sys.path:
            sys.path.insert(0, mem_core)

        from OPTIMIZED_memory_ai_system import MemoryAISystem
        memory_ai = MemoryAISystem()
        print("Memory AI System connected.")
    except Exception as e:
        # Fix: MemoryAI init failures should never block live trading.
        print(f"Memory AI init failed: {e}. Continuing without it.")
        memory_ai = None




    # Initialize MT5 + auto-discover tradable symbols from Market Watch
    # (uses the exact broker symbol names).
    if not connect_mt5():
        raise RuntimeError("Could not connect to MT5 terminal")

    discovered = []
    try:
        from experts.mt5_expert import list_tradable_symbols
        discovered = list_tradable_symbols(max_symbols=20)
    except Exception as e:
        print(f"Symbol discovery failed: {e}")

    if not discovered:
        raise RuntimeError(
            "No tradable MT5 symbols discovered. "
            "Make sure symbols are visible in Market Watch and allow trading."
        )

    print(f"Discovered {len(discovered)} tradable MT5 symbols (showing up to 10): {discovered[:10]}")

    # Initialize real MT5 data fetcher
    data_fetcher = MT5MarketData()

    # Create trading bot
    print("Creating trading bot...")
    bot = LiveTradingBot(
        # Use exact MT5 symbol names from discovery
        symbols=discovered,
        capital=5000.0,
        data_fetcher=data_fetcher,
        memory_ai=memory_ai,
    )

    # Start trading (demo/live depends on which MT5 account you are logged into)
    bot.start(interval_seconds=5)

    # Run for a while
    try:
        for i in range(12):  # ~1 minute
            time.sleep(5)
            print(f"Running... ({i+1}/12)")
    except KeyboardInterrupt:
        print("Stopped by user")
    finally:
        print("⏸️  Live trading stopped")
        bot.stop()
        bot.print_report()
