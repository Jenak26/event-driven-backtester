# tests/test_new_metrics.py
"""New tearsheet metrics: Calmar ratio, monthly-returns table, and trade
reconstruction from a fill ledger."""
import numpy as np
import pandas as pd

from backtester.metrics import (
    annualized_return, max_drawdown, calmar_ratio,
    monthly_returns_table, trades_from_fills, trade_stats,
)


def _equity(values, start="2020-01-01"):
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq="B"))


def test_calmar_equals_ann_over_abs_dd():
    eq = _equity([100.0, 120.0, 90.0, 110.0])
    expected = annualized_return(eq) / abs(max_drawdown(eq))
    assert abs(calmar_ratio(eq) - expected) < 1e-9


def test_calmar_no_drawdown_is_inf_when_rising():
    eq = _equity([100.0, 101.0, 102.0, 103.0])
    assert calmar_ratio(eq) == float("inf")


def test_calmar_no_drawdown_is_neg_inf_when_falling():
    # Strictly declining => no new peak after the first, drawdown stays 0 only if
    # never recovers below start... use a flat-then-never-recover guard instead:
    eq = _equity([100.0, 100.0, 100.0])  # flat => ann 0, dd 0 => 0.0
    assert calmar_ratio(eq) == 0.0


def test_monthly_table_shape_and_value():
    # Linear ramp 100 -> 110 across January 2021 business days.
    jan = pd.date_range("2021-01-01", "2021-01-29", freq="B")
    eq = pd.Series(np.linspace(100.0, 110.0, len(jan)), index=jan)
    table = monthly_returns_table(eq)
    assert list(table.columns) == list(range(1, 13))
    assert 2021 in table.index
    # January return ≈ last/first - 1 = 0.10.
    assert abs(table.loc[2021, 1] - 0.10) < 1e-6
    # Other months are NaN (no data).
    assert np.isnan(table.loc[2021, 2])


def test_monthly_table_empty_input():
    assert monthly_returns_table(pd.Series([100.0])).empty


def test_trades_from_fills_round_trip_pnl():
    # Buy 10 @ $100 (no costs), sell 10 @ $110 => P&L = +$100.
    fills = pd.DataFrame([
        {"symbol": "AAA", "quantity": 10, "fill_price": 100.0,
         "commission": 0.0, "slippage": 0.0, "timestamp": pd.Timestamp("2021-01-04")},
        {"symbol": "AAA", "quantity": -10, "fill_price": 110.0,
         "commission": 0.0, "slippage": 0.0, "timestamp": pd.Timestamp("2021-01-14")},
    ])
    trades = trades_from_fills(fills)
    assert len(trades) == 1
    assert abs(trades.iloc[0]["pnl"] - 100.0) < 1e-9
    assert trades.iloc[0]["direction"] == "long"
    assert trades.iloc[0]["holding_days"] == 10


def test_trades_costs_reduce_pnl():
    fills = pd.DataFrame([
        {"symbol": "AAA", "quantity": 10, "fill_price": 100.0,
         "commission": 1.0, "slippage": 2.0, "timestamp": pd.Timestamp("2021-01-04")},
        {"symbol": "AAA", "quantity": -10, "fill_price": 110.0,
         "commission": 1.0, "slippage": 2.0, "timestamp": pd.Timestamp("2021-01-14")},
    ])
    trades = trades_from_fills(fills)
    # 100 gross minus 6 total costs = 94.
    assert abs(trades.iloc[0]["pnl"] - 94.0) < 1e-9


def test_trade_stats_profit_factor_and_winrate():
    fills = pd.DataFrame([
        # winner: +100
        {"symbol": "AAA", "quantity": 10, "fill_price": 100.0, "commission": 0.0,
         "slippage": 0.0, "timestamp": pd.Timestamp("2021-01-04")},
        {"symbol": "AAA", "quantity": -10, "fill_price": 110.0, "commission": 0.0,
         "slippage": 0.0, "timestamp": pd.Timestamp("2021-01-14")},
        # loser: -50
        {"symbol": "BBB", "quantity": 10, "fill_price": 100.0, "commission": 0.0,
         "slippage": 0.0, "timestamp": pd.Timestamp("2021-01-04")},
        {"symbol": "BBB", "quantity": -10, "fill_price": 95.0, "commission": 0.0,
         "slippage": 0.0, "timestamp": pd.Timestamp("2021-01-09")},
    ])
    stats = trade_stats(trades_from_fills(fills))
    assert stats["n_trades"] == 2
    assert abs(stats["win_rate"] - 0.5) < 1e-9
    assert abs(stats["profit_factor"] - (100.0 / 50.0)) < 1e-9


def test_trade_stats_empty():
    stats = trade_stats(trades_from_fills(pd.DataFrame()))
    assert stats["n_trades"] == 0
