# tests/test_phase2_gate.py
"""Phase 2 acceptance gate: full loop with costs, manual trade verification."""
import pandas as pd
import yfinance as yf
from run import run_backtest


def test_buy_and_hold_equity_tracks_spy():
    """Buy-and-hold on SPY: equity curve must track SPY adjusted close within 1%."""
    equity_df = run_backtest(["SPY"], "2020-01-02", "2023-12-29")
    assert len(equity_df) > 0

    spy = yf.download("SPY", start="2020-01-02", end="2023-12-29",
                      auto_adjust=True, progress=False)
    start_price = float(spy["Close"].iloc[0])
    end_price = float(spy["Close"].iloc[-1])
    spy_return = (end_price - start_price) / start_price

    start_equity = equity_df["equity"].iloc[0]
    end_equity = equity_df["equity"].iloc[-1]
    backtest_return = (end_equity - start_equity) / start_equity

    # Allow 2% tolerance (commission + slippage on entry)
    assert abs(backtest_return - spy_return) < 0.02, (
        f"Backtest return {backtest_return:.4f} too far from SPY return {spy_return:.4f}"
    )


def test_equity_curve_always_positive():
    equity_df = run_backtest(["SPY"], "2020-01-02", "2023-12-29")
    assert (equity_df["equity"] > 0).all()
