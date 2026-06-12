# tests/test_momentum.py
"""Cross-sectional momentum signal generation.

Uses a FakeDataHandler exposing ONLY the point-in-time API (no `_data`), so any
look-ahead reach would fail. Verifies the strategy plugs into the engine's
SignalEvent contract: long the top quintile, long-only, equal-dollar, monthly.
"""
import numpy as np
import pandas as pd

from backtester.queue import EventQueue
from backtester.events import EventType, SignalDirection
from strategies.momentum import CrossSectionalMomentumStrategy


class FakeDataHandler:
    def __init__(self, frames):
        self.symbols = list(frames)
        self._frames = frames
        self._dates = frames[self.symbols[0]].index
        self._cursor = 0

    def get_latest_bars(self, symbol, N=1):
        if self._cursor == 0:
            return None
        start = max(0, self._cursor - N)
        return self._frames[symbol].iloc[start:self._cursor]

    @property
    def current_date(self):
        if self._cursor == 0:
            return None
        return self._dates[self._cursor - 1]

    def advance(self, n=1):
        self._cursor = min(self._cursor + n, len(self._dates))


def _ramped_frames(n=120, n_symbols=10):
    """Symbol i compounds at growth rate proportional to i, so momentum rank
    increases with i: S9 strongest, S0 weakest."""
    idx = pd.date_range("2021-01-01", periods=n, freq="B")
    frames = {}
    for i in range(n_symbols):
        g = 0.0005 * i  # per-bar growth; higher i => higher trailing return
        closes = 100.0 * np.power(1.0 + g, np.arange(n))
        sym = f"S{i}"
        frames[sym] = pd.DataFrame({"Open": closes, "Close": closes}, index=idx)
    return frames


def _drain_signals(queue):
    out = []
    while not queue.empty():
        ev = queue.get()
        if ev.type == EventType.SIGNAL:
            out.append(ev)
    return out


def test_longs_top_quintile_only():
    frames = _ramped_frames(n=120, n_symbols=10)
    q = EventQueue()
    data = FakeDataHandler(frames)
    strat = CrossSectionalMomentumStrategy(
        data, q, lookback=60, skip=5, top_quantile=0.2, book_capital=100_000.0,
    )
    data.advance(120)  # plenty of history
    strat.calculate_signals()
    sigs = _drain_signals(q)

    # top quintile of 10 = 2 names; the two strongest are S9, S8.
    longed = {s.symbol for s in sigs}
    assert longed == {"S8", "S9"}
    # Long-only: all opening trades are buys.
    assert all(s.quantity > 0 for s in sigs)
    assert all(s.direction == SignalDirection.LONG for s in sigs)


def test_equal_dollar_sizing():
    frames = _ramped_frames(n=120, n_symbols=10)
    q = EventQueue()
    data = FakeDataHandler(frames)
    book = 100_000.0
    strat = CrossSectionalMomentumStrategy(
        data, q, lookback=60, skip=5, top_quantile=0.2, book_capital=book,
    )
    data.advance(120)
    strat.calculate_signals()
    sigs = _drain_signals(q)

    # Each leg targets book/n_long dollars (n_long=2 => $50k each).
    per_name = book / 2
    for s in sigs:
        price = float(frames[s.symbol]["Close"].iloc[119])
        assert abs(s.quantity - int(per_name / price)) == 0
        notional = s.quantity * price
        assert abs(notional - per_name) < price  # within one share of target


def test_rebalances_once_per_month():
    frames = _ramped_frames(n=120, n_symbols=10)
    q = EventQueue()
    data = FakeDataHandler(frames)
    strat = CrossSectionalMomentumStrategy(data, q, lookback=60, skip=5)
    data.advance(120)
    strat.calculate_signals()
    first = _drain_signals(q)
    assert len(first) > 0
    # Same bar / same month: no second rebalance.
    strat.calculate_signals()
    assert _drain_signals(q) == []


def test_insufficient_history_stays_flat():
    frames = _ramped_frames(n=40, n_symbols=10)  # < lookback
    q = EventQueue()
    data = FakeDataHandler(frames)
    strat = CrossSectionalMomentumStrategy(data, q, lookback=60, skip=5)
    data.advance(40)
    strat.calculate_signals()
    assert _drain_signals(q) == []


def test_start_after_gate_blocks_early_trading():
    frames = _ramped_frames(n=120, n_symbols=10)
    q = EventQueue()
    data = FakeDataHandler(frames)
    strat = CrossSectionalMomentumStrategy(
        data, q, lookback=60, skip=5, start_after="2099-01-01",
    )
    data.advance(120)
    strat.calculate_signals()
    assert _drain_signals(q) == []
