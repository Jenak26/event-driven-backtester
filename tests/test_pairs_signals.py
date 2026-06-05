# tests/test_pairs_signals.py
import numpy as np
import pandas as pd
import pytest
from strategies.pairs import compute_zscore, check_rolling_coint


def _spread_series(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return pd.Series(
        np.cumsum(rng.normal(0, 0.1, n)),
        index=pd.date_range("2021-01-01", periods=n, freq="B"),
    )


def _cointegrated(n=200, seed=0):
    rng = np.random.default_rng(seed)
    common = np.cumsum(rng.normal(0, 1, n))
    a = pd.Series(common + rng.normal(0, 0.3, n),
                  index=pd.date_range("2021-01-01", periods=n, freq="B"))
    b = pd.Series(common * 0.8 + rng.normal(0, 0.3, n),
                  index=pd.date_range("2021-01-01", periods=n, freq="B"))
    return a, b


def test_zscore_shape():
    spread = _spread_series()
    z = compute_zscore(spread, window=60)
    assert isinstance(z, pd.Series)
    assert len(z) == len(spread)


def test_zscore_nan_before_window():
    spread = _spread_series()
    z = compute_zscore(spread, window=60)
    assert z.iloc[:59].isna().all()
    assert not z.iloc[60:].isna().all()


def test_zscore_mean_zero_approximately():
    rng = np.random.default_rng(1)
    spread = pd.Series(
        rng.normal(0, 1, 500),
        index=pd.date_range("2021-01-01", periods=500, freq="B"),
    )
    z = compute_zscore(spread, window=60).dropna()
    assert abs(z.mean()) < 0.2


def test_check_rolling_coint_returns_float():
    a, b = _cointegrated()
    pval = check_rolling_coint(a, b, window=60)
    assert isinstance(pval, float)
    assert 0.0 <= pval <= 1.0


def test_cointegrated_pair_low_pvalue():
    a, b = _cointegrated(n=300, seed=42)
    pval = check_rolling_coint(a, b, window=120)
    assert pval < 0.1


def test_insufficient_data_returns_one():
    a = pd.Series([1.0, 2.0])
    b = pd.Series([1.0, 2.0])
    pval = check_rolling_coint(a, b, window=60)
    assert pval == 1.0  # sentinel: not enough data = assume broken
