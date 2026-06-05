# tests/test_metrics.py
import numpy as np
import pandas as pd
import pytest
from backtester.metrics import (
    sharpe_ratio, sortino_ratio, max_drawdown, annualized_return,
)


def _equity(values):
    return pd.Series(values, index=pd.date_range("2020-01-01", periods=len(values), freq="B"))


def test_sharpe_flat_returns_zero():
    eq = _equity([100.0] * 252)
    returns = eq.pct_change().dropna()
    assert sharpe_ratio(returns) == 0.0


def test_sharpe_positive_drift():
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0.0005, 0.01, 252))
    sr = sharpe_ratio(returns)
    assert sr > 0


def test_sortino_ignores_upside():
    # All positive returns → sortino should be very high (no downside vol)
    returns = pd.Series([0.001] * 252)
    sr = sortino_ratio(returns)
    assert sr > 10


def test_max_drawdown_negative():
    eq = _equity([100, 110, 90, 95, 105])
    dd = max_drawdown(eq)
    assert dd < 0
    # Peak = 110, trough = 90 → drawdown = (90-110)/110 ≈ -0.1818
    assert abs(dd - (-20 / 110)) < 0.001


def test_max_drawdown_no_drawdown():
    eq = _equity([100, 101, 102, 103])
    assert max_drawdown(eq) == 0.0


def test_annualized_return_doubles_in_252():
    # Double in 252 trading days → annualized = 100%
    eq = _equity([100.0] + [200.0 / 251 * i + 100 for i in range(1, 252)])
    eq = _equity([100.0, 200.0])
    ret = annualized_return(eq, periods_per_year=1)  # 1 period = 1 year
    assert abs(ret - 1.0) < 0.01
