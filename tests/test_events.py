from backtester.events import (
    MarketEvent, SignalEvent, OrderEvent, FillEvent,
    EventType, SignalDirection, OrderType,
)
from datetime import datetime

def test_market_event_type():
    e = MarketEvent()
    assert e.type == EventType.MARKET

def test_signal_event_fields():
    e = SignalEvent(symbol="AAPL", direction=SignalDirection.LONG)
    assert e.type == EventType.SIGNAL
    assert e.symbol == "AAPL"
    assert e.direction == SignalDirection.LONG
    assert e.quantity is None  # no explicit quantity = let portfolio decide

def test_order_event_fields():
    e = OrderEvent(symbol="AAPL", order_type=OrderType.MARKET, quantity=100, direction=SignalDirection.LONG)
    assert e.type == EventType.ORDER
    assert e.quantity == 100

def test_fill_event_fields():
    e = FillEvent(
        symbol="AAPL", quantity=100, fill_price=150.0,
        commission=0.10, slippage=0.075, timestamp=datetime(2023, 1, 3),
    )
    assert e.type == EventType.FILL
    assert e.fill_price == 150.0

def test_signal_event_with_quantity():
    e = SignalEvent(symbol="MSFT", direction=SignalDirection.SHORT, quantity=-50)
    assert e.quantity == -50
