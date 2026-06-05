# tests/test_portfolio.py
import pytest
from backtester.queue import EventQueue
from backtester.data import DataHandler
from backtester.portfolio import Portfolio
from backtester.events import SignalEvent, FillEvent, SignalDirection, EventType, OrderType
from datetime import datetime


@pytest.fixture
def setup():
    q = EventQueue()
    data = DataHandler(["SPY"], "2023-01-01", "2023-03-31", q)
    # Advance 2 bars so data is available
    data.update_bars(); q.get()
    data.update_bars(); q.get()
    portfolio = Portfolio(data, q, initial_capital=100_000.0)
    return data, q, portfolio


def test_initial_cash(setup):
    _, _, portfolio = setup
    assert portfolio.cash == 100_000.0


def test_signal_long_emits_order(setup):
    data, q, portfolio = setup
    portfolio.update_signal(SignalEvent(symbol="SPY", direction=SignalDirection.LONG))
    assert not q.empty()
    order = q.get()
    assert order.type == EventType.ORDER
    assert order.quantity > 0


def test_signal_short_emits_negative_order(setup):
    data, q, portfolio = setup
    portfolio.update_signal(SignalEvent(symbol="SPY", direction=SignalDirection.SHORT))
    order = q.get()
    assert order.quantity < 0


def test_fill_updates_cash_and_position(setup):
    _, _, portfolio = setup
    fill = FillEvent(
        symbol="SPY", quantity=10, fill_price=400.0,
        commission=0.01, slippage=0.20, timestamp=datetime(2023, 1, 4),
    )
    portfolio.update_fill(fill)
    # cost = 10 * 400 + 0.01 + 0.20 = 4000.21
    assert abs(portfolio.cash - (100_000.0 - 4000.21)) < 0.01
    assert portfolio.positions["SPY"] == 10


def test_fill_short_reduces_position(setup):
    _, _, portfolio = setup
    # First establish a long position
    portfolio.update_fill(FillEvent(
        symbol="SPY", quantity=10, fill_price=400.0,
        commission=0.0, slippage=0.0, timestamp=datetime(2023, 1, 4),
    ))
    # Now short (sell) 10
    portfolio.update_fill(FillEvent(
        symbol="SPY", quantity=-10, fill_price=410.0,
        commission=0.0, slippage=0.0, timestamp=datetime(2023, 1, 5),
    ))
    assert portfolio.positions["SPY"] == 0


def test_update_equity_appends_record(setup):
    data, q, portfolio = setup
    portfolio.update_equity()
    df = portfolio.get_equity_df()
    assert len(df) == 1
    assert "equity" in df.columns


def test_signal_with_explicit_quantity(setup):
    data, q, portfolio = setup
    portfolio.update_signal(SignalEvent(symbol="SPY", direction=SignalDirection.LONG, quantity=50))
    order = q.get()
    assert order.quantity == 50


def test_exit_signal_closes_position(setup):
    _, q, portfolio = setup
    portfolio.positions["SPY"] = 25
    portfolio.update_signal(SignalEvent(symbol="SPY", direction=SignalDirection.EXIT))
    order = q.get()
    assert order.quantity == -25
