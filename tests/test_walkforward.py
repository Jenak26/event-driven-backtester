# tests/test_walkforward.py
import pandas as pd
from backtester.metrics import WalkForwardRunner, sharpe_ratio


def _dummy_run(train_start, train_end, test_start, test_end):
    """Stub: returns a flat equity curve for the test window."""
    dates = pd.date_range(test_start, test_end, freq="B")
    equity = pd.Series(100_000.0, index=dates)
    return {"equity_df": equity.to_frame("equity"), "sharpe": 0.5}


def test_walk_forward_produces_correct_windows():
    runner = WalkForwardRunner(
        run_fn=_dummy_run,
        data_start="2019-01-01",
        data_end="2022-12-31",
        train_months=12,
        test_months=3,
    )
    results = runner.run()
    assert len(results) >= 4  # 4 years of data with 12+3 windows


def test_walk_forward_result_keys():
    runner = WalkForwardRunner(
        run_fn=_dummy_run,
        data_start="2019-01-01",
        data_end="2022-12-31",
        train_months=12,
        test_months=3,
    )
    results = runner.run()
    for r in results:
        assert "train_start" in r
        assert "train_end" in r
        assert "test_start" in r
        assert "test_end" in r
        assert "sharpe" in r


def test_windows_do_not_overlap():
    runner = WalkForwardRunner(
        run_fn=_dummy_run,
        data_start="2019-01-01",
        data_end="2022-12-31",
        train_months=12,
        test_months=3,
    )
    results = runner.run()
    for i in range(1, len(results)):
        prev_test_end = pd.Timestamp(results[i - 1]["test_end"])
        curr_test_start = pd.Timestamp(results[i]["test_start"])
        # Test windows are contiguous: next starts where prev ends (boundary day
        # is excluded from the prior window since yfinance `end` is exclusive).
        assert curr_test_start >= prev_test_end
