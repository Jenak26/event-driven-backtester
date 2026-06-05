# tests/test_execution.py
import pytest
from backtester.queue import EventQueue
from backtester.data import DataHandler
from backtester.execution import SimBroker
from backtester.events import OrderEvent, OrderType, SignalDirection, EventType


@pytest.fixture
def setup():
    q = EventQueue()
    data = DataHandler(["SPY"], "2023-01-01", "2023-03-31", q)
    data.update_bars(); q.get()  # advance to bar 1
    broker = SimBroker(data, q, commission_per_share=0.001, slippage_bps=5.0)
    return data, q, broker


def test_fill_emitted_after_order(setup):
    data, q, broker = setup
    order = OrderEvent(symbol="SPY", order_type=OrderType.MARKET, quantity=100, direction=SignalDirection.LONG)
    broker.execute_order(order)
    assert not q.empty()
    fill = q.get()
    assert fill.type == EventType.FILL


def test_fill_price_is_next_bar_open_plus_slippage(setup):
    data, q, broker = setup
    next_open = data.get_next_open("SPY")
    order = OrderEvent(symbol="SPY", order_type=OrderType.MARKET, quantity=100, direction=SignalDirection.LONG)
    broker.execute_order(order)
    fill = q.get()
    expected_slippage = next_open * 5.0 / 10_000
    assert abs(fill.fill_price - (next_open + expected_slippage)) < 0.001


def test_short_fill_price_below_open(setup):
    data, q, broker = setup
    next_open = data.get_next_open("SPY")
    order = OrderEvent(symbol="SPY", order_type=OrderType.MARKET, quantity=-100, direction=SignalDirection.SHORT)
    broker.execute_order(order)
    fill = q.get()
    assert fill.fill_price < next_open


def test_commission_calculated_correctly(setup):
    data, q, broker = setup
    order = OrderEvent(symbol="SPY", order_type=OrderType.MARKET, quantity=200, direction=SignalDirection.LONG)
    broker.execute_order(order)
    fill = q.get()
    assert abs(fill.commission - 200 * 0.001) < 1e-9


def test_no_fill_when_no_next_bar():
    q = EventQueue()
    data = DataHandler(["SPY"], "2023-01-01", "2023-01-05", q)
    broker = SimBroker(data, q)
    # Exhaust all bars
    while not data.is_exhausted:
        data.update_bars()
        q.get()
    order = OrderEvent(symbol="SPY", order_type=OrderType.MARKET, quantity=10, direction=SignalDirection.LONG)
    broker.execute_order(order)
    assert q.empty()  # no fill — no next bar exists
