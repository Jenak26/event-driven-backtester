import pandas as pd
import pytest
from backtester.queue import EventQueue
from backtester.data import DataHandler
from backtester.events import EventType


@pytest.fixture
def spy_handler():
    q = EventQueue()
    return DataHandler(["SPY"], "2023-01-01", "2023-03-31", q), q


def test_initial_cursor_returns_none(spy_handler):
    handler, _ = spy_handler
    assert handler.get_latest_bars("SPY", 1) is None


def test_update_bars_emits_market_event(spy_handler):
    handler, q = spy_handler
    handler.update_bars()
    assert not q.empty()
    event = q.get()
    assert event.type == EventType.MARKET


def test_get_latest_bars_respects_cursor(spy_handler):
    handler, q = spy_handler
    # Advance 5 bars
    for _ in range(5):
        handler.update_bars()
        q.get()  # drain queue
    bars = handler.get_latest_bars("SPY", 5)
    assert bars is not None
    assert len(bars) == 5


def test_no_lookahead(spy_handler):
    """get_latest_bars must never return more rows than cursor position."""
    handler, q = spy_handler
    for _ in range(3):
        handler.update_bars()
        q.get()
    # Ask for 10 bars when only 3 exist in the cursor window
    bars = handler.get_latest_bars("SPY", 10)
    assert len(bars) == 3


def test_prices_are_adjusted(spy_handler):
    """Adjusted close should exist and be positive."""
    handler, q = spy_handler
    handler.update_bars()
    q.get()
    bars = handler.get_latest_bars("SPY", 1)
    # Adjusted prices exist and are positive
    assert bars["Close"].iloc[0] > 0
    # Column set matches adjusted OHLCV (no separate 'Adj Close' — auto_adjust merges it)
    assert "Close" in bars.columns
    assert "Open" in bars.columns


def test_get_next_open_advances_one_bar(spy_handler):
    handler, q = spy_handler
    handler.update_bars()
    q.get()
    next_open = handler.get_next_open("SPY")
    assert next_open is not None
    assert next_open > 0


def test_exhausted_after_all_bars(spy_handler):
    handler, q = spy_handler
    while not handler.is_exhausted:
        handler.update_bars()
        while not q.empty():
            q.get()
    assert handler.is_exhausted


def _synthetic_frames(n=20):
    idx = pd.date_range("2023-01-02", periods=n, freq="B")
    prices = pd.Series(range(100, 100 + n), index=idx, dtype=float)
    return {"XYZ": pd.DataFrame({"Open": prices, "Close": prices}, index=idx)}


def test_preloaded_data_skips_download():
    """Ticker XYZ does not exist — passing preloaded frames must avoid yfinance."""
    q = EventQueue()
    handler = DataHandler(["XYZ"], "2023-01-02", "2023-02-01", q, preloaded=_synthetic_frames())
    handler.update_bars()
    bars = handler.get_latest_bars("XYZ", 1)
    assert float(bars["Close"].iloc[-1]) == 100.0


def test_preloaded_data_sliced_to_window():
    """Preloaded frames are cropped to [start, end) like a yfinance download."""
    q = EventQueue()
    handler = DataHandler(["XYZ"], "2023-01-09", "2023-01-20", q, preloaded=_synthetic_frames())
    assert handler._dates[0] >= pd.Timestamp("2023-01-09")
    assert handler._dates[-1] < pd.Timestamp("2023-01-20")
    # Bars before the window start are not visible
    handler.update_bars()
    bars = handler.get_latest_bars("XYZ", 10)
    assert len(bars) == 1
    assert bars.index[0] >= pd.Timestamp("2023-01-09")
