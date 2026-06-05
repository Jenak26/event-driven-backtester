from backtester.queue import EventQueue
from backtester.data import DataHandler
from backtester.strategy import BuyAndHoldStrategy
from backtester.events import EventType, SignalDirection


def test_buy_and_hold_emits_long_once():
    q = EventQueue()
    data = DataHandler(["SPY"], "2023-01-01", "2023-01-31", q)
    strategy = BuyAndHoldStrategy(data, q)

    signals = []
    for _ in range(3):
        data.update_bars()
        # drain the MarketEvent
        while not q.empty():
            event = q.get()
            if event.type == EventType.MARKET:
                strategy.calculate_signals()
                # collect any signals emitted
                while not q.empty():
                    e2 = q.get()
                    if e2.type == EventType.SIGNAL:
                        signals.append(e2)
            elif event.type == EventType.SIGNAL:
                signals.append(event)

    # Should emit exactly one LONG signal for SPY, never repeat
    long_signals = [s for s in signals if s.symbol == "SPY" and s.direction == SignalDirection.LONG]
    assert len(long_signals) == 1
