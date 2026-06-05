# tests/test_phase1_gate.py
"""Phase 1 acceptance gate: verify loop mechanics, no look-ahead."""
from backtester.queue import EventQueue
from backtester.data import DataHandler
from backtester.strategy import BuyAndHoldStrategy
from backtester.events import EventType


def test_market_event_count_equals_trading_days():
    """Exactly one MarketEvent per trading day."""
    q = EventQueue()
    # yfinance `end` is exclusive, so use 2023-02-01 to cover all of January.
    data = DataHandler(["SPY"], "2023-01-01", "2023-02-01", q)
    strategy = BuyAndHoldStrategy(data, q)

    market_count = 0
    while not data.is_exhausted:
        data.update_bars()
        while not q.empty():
            event = q.get()
            if event.type == EventType.MARKET:
                market_count += 1
                strategy.calculate_signals()

    # January 2023 has 20 trading days
    assert market_count == 20


def test_no_lookahead_at_bar_5():
    """On bar 5, get_latest_bars(N=10) returns exactly 5 rows."""
    q = EventQueue()
    data = DataHandler(["SPY"], "2023-01-01", "2023-03-31", q)

    for _ in range(5):
        data.update_bars()
        q.get()  # drain MarketEvent

    bars = data.get_latest_bars("SPY", 10)
    assert len(bars) == 5, f"Expected 5 bars, got {len(bars)}"


def test_cursor_never_exceeds_dataset():
    q = EventQueue()
    data = DataHandler(["SPY"], "2023-01-01", "2023-01-31", q)
    count = 0
    while not data.is_exhausted:
        data.update_bars()
        q.get()
        count += 1
    assert data._cursor == count
    assert data.is_exhausted
