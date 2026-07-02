"""Example runner for Phase D add-on pipeline.

Usage:
  python phase_d_run_example.py

This will:
- load MarketOracle data bundle from MarketOracle-workspace/data
- run Scout/Trader/Risk/Execution (no MT5 side effects)

It is safe to run in environments where no MT5 terminal is present.
"""

from __future__ import annotations

from pathlib import Path


def main():
    repo_root = Path(__file__).resolve().parent
    mo_root = repo_root / "MarketOracle-workspace"

    # Load data bundle using existing chart_expert
    from MarketOracle_workspace_import_fallback import load_bundle  # type: ignore

    data_bundle = load_bundle(str(mo_root / "data"))

    from MarketOracle_workspace_import_fallback import run_phase_d_pipeline  # type: ignore

    result = run_phase_d_pipeline(data_bundle=data_bundle, memory_ai=None, max_symbols=5)
    print(result)


if __name__ == "__main__":
    main()


