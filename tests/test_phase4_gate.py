# tests/test_phase4_gate.py
"""
Phase 4 acceptance gate:
  - Out-of-sample Sharpe is computed and reported.
  - Walk-forward produces at least 4 windows.
  - Both with-cost and without-cost results are reported.
"""
import pandas as pd
import numpy as np
from backtester.metrics import sharpe_ratio, WalkForwardRunner
from run import run_backtest
from backtester.strategy import BuyAndHoldStrategy


def _make_run_fn(with_costs):
    def run_fn(train_start, train_end, test_start, test_end):
        cost = 0.001 if with_costs else 0.0
        slippage = 5.0 if with_costs else 0.0
        eq = run_backtest(
            ["SPY"], test_start, test_end,
            commission=cost, slippage_bps=slippage,
        )
        returns = eq["equity"].pct_change().dropna()
        sr = sharpe_ratio(returns)
        return {"equity_df": eq, "sharpe": sr}
    return run_fn


def test_walkforward_produces_4_plus_windows():
    runner = WalkForwardRunner(
        run_fn=_make_run_fn(with_costs=True),
        data_start="2019-01-01",
        data_end="2022-12-31",
        train_months=12,
        test_months=3,
    )
    results = runner.run()
    assert len(results) >= 4, f"Only {len(results)} windows produced"


def test_sharpe_is_finite_in_all_windows():
    runner = WalkForwardRunner(
        run_fn=_make_run_fn(with_costs=True),
        data_start="2019-01-01",
        data_end="2022-12-31",
        train_months=12,
        test_months=3,
    )
    results = runner.run()
    for r in results:
        assert np.isfinite(r["sharpe"]), f"Non-finite Sharpe in window {r}"


def test_costs_reduce_sharpe():
    """With-cost Sharpe should be <= without-cost Sharpe (or within noise)."""
    eq_with = run_backtest(["SPY"], "2021-01-01", "2022-12-31",
                           commission=0.001, slippage_bps=5.0)
    eq_without = run_backtest(["SPY"], "2021-01-01", "2022-12-31",
                              commission=0.0, slippage_bps=0.0)
    sr_with = sharpe_ratio(eq_with["equity"].pct_change().dropna())
    sr_without = sharpe_ratio(eq_without["equity"].pct_change().dropna())
    # With costs should not outperform without costs
    assert sr_with <= sr_without + 0.01
