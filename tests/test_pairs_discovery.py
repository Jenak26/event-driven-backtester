# tests/test_pairs_discovery.py
"""Pair discovery must be point-in-time and exits must unwind exact quantities.

FakeDataHandler exposes ONLY the public point-in-time API and deliberately
has no `_data` attribute: if the strategy reaches into raw frames (a
look-ahead hazard), these tests fail with AttributeError.
"""
import numpy as np
import pandas as pd
import pytest
from backtester.queue import EventQueue
from backtester.events import EventType, SignalDirection
from strategies.pairs import CointegrationPairsStrategy

SECTORS_SYNTH = {"synthetic": ["AAA", "BBB"]}


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


def _frames_from(prices_a, prices_b, start="2021-01-01"):
    idx = pd.date_range(start, periods=len(prices_a), freq="B")
    return {
        "AAA": pd.DataFrame({"Open": prices_a, "Close": prices_a}, index=idx),
        "BBB": pd.DataFrame({"Open": prices_b, "Close": prices_b}, index=idx),
    }


def _cointegrated_frames(n=400, seed=7):
    rng = np.random.default_rng(seed)
    common = np.cumsum(rng.normal(0, 0.5, n))
    a = 100.0 + common + rng.normal(0, 0.3, n)
    b = 100.0 + 0.8 * common + rng.normal(0, 0.3, n)
    return _frames_from(a, b)


def _drain_signals(queue):
    signals = []
    while not queue.empty():
        ev = queue.get()
        if ev.type == EventType.SIGNAL:
            signals.append(ev)
    return signals


def test_discovery_finds_pair_using_only_elapsed_bars():
    frames = _cointegrated_frames()
    q = EventQueue()
    data = FakeDataHandler(frames)
    train_end = data._dates[300]
    strat = CointegrationPairsStrategy(
        data, q, train_end_date=str(train_end.date()),
        sectors=SECTORS_SYNTH,
    )
    data.advance(301)  # cursor sits exactly on train_end
    strat.calculate_signals()
    assert strat._discovered
    assert len(strat._pairs) == 1
    sym_a, sym_b, beta = strat._pairs[0]
    assert {sym_a, sym_b} == {"AAA", "BBB"}
    assert beta > 0


def test_no_discovery_before_train_end():
    frames = _cointegrated_frames()
    q = EventQueue()
    data = FakeDataHandler(frames)
    train_end = data._dates[300]
    strat = CointegrationPairsStrategy(
        data, q, train_end_date=str(train_end.date()),
        sectors=SECTORS_SYNTH,
    )
    data.advance(200)  # well before train_end
    strat.calculate_signals()
    assert not strat._discovered
    assert _drain_signals(q) == []


def test_exit_unwinds_exact_entry_quantities():
    n = 160
    rng = np.random.default_rng(3)
    # Stable spread oscillating +/-0.5, spike at bar 80-89, convergence after
    spread = np.where(np.arange(n) % 2 == 0, 0.5, -0.5).astype(float)
    spread[80:90] = 6.0
    b = 100.0 + rng.normal(0, 0.01, n)
    a = b + spread
    frames = _frames_from(a, b)

    q = EventQueue()
    data = FakeDataHandler(frames)
    strat = CointegrationPairsStrategy(
        data, q, train_end_date="2020-01-01",  # before data: never discovers
        sectors=SECTORS_SYNTH,
        z_window=20,
        stop_coint_pval=1.5,  # disable stop-out (p-values never exceed 1)
    )
    # Inject the pair directly; this test exercises signal logic only.
    strat._discovered = True
    strat._pairs = [("AAA", "BBB", 1.0)]
    strat._active = {("AAA", "BBB"): None}

    entry_signals, exit_signals = [], []
    for _ in range(n):
        data.advance()
        strat.calculate_signals()
        sigs = _drain_signals(q)
        if not sigs:
            continue
        if not entry_signals:
            entry_signals = sigs
        elif not exit_signals:
            exit_signals = sigs

    assert len(entry_signals) == 2, "entry should emit one signal per leg"
    assert len(exit_signals) == 2, "exit should emit one signal per leg"

    entry_qty = {s.symbol: s.quantity for s in entry_signals}
    exit_qty = {s.symbol: s.quantity for s in exit_signals}
    # Spread spiked high: short A, long B
    assert entry_qty["AAA"] < 0
    assert entry_qty["BBB"] > 0
    # Exit must negate the exact entry quantities — not flatten the symbol —
    # so pairs sharing a leg cannot corrupt each other's positions.
    for sym in ("AAA", "BBB"):
        assert exit_qty[sym] == -entry_qty[sym]
    for s in exit_signals:
        assert s.direction != SignalDirection.EXIT
        assert s.quantity is not None
