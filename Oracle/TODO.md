# MarketOracle — Adaptive MT5 Data Acquisition Refactor (TODO)

## Plan checkpoints
- [ ] Create `MarketDataManager` to centralize all MT5 communication, adaptive fetch, validation, indicator warm-up, observation checks, and structured logging.
- [ ] Update `mt5_data_validation_layer.py` to compute dynamic required history from real indicator lookbacks (derive from `chart_expert.add_technical_indicators`).
- [ ] Wire training (`main.py` step_load_data when `--mt5`) to use `MarketDataManager.get_validated_bundle(...)` instead of calling validator directly.
- [ ] Wire live trading (`live_trader.py`) so `fetch_ohlcv` / any OHLCV retrieval routes through `MarketDataManager` (no direct MT5 calls outside manager).
- [ ] Implement MemoryAI integration for self-optimizing fetch recommendations:
  - store per (symbol,timeframe): requested_history, returned_history, usable_history, warm-up_loss, indicator_loss, recommended_history, timestamp, status/attempts.
  - seed adaptive retrieval using MemoryAI recommendations when available.
- [ ] Structured logging format across all validation outcomes.
- [ ] Smoke test: run a minimal MT5 training data load for a single symbol/timeframe.
- [ ] Smoke test: run live trading cycle for one symbol (short loop) ensuring observations build without invalid shapes.

