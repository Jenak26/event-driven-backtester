from backtester.events import EventType
from backtester.queue import EventQueue
from backtester.data import DataHandler
from backtester.strategy import BuyAndHoldStrategy


def run_loop(symbols, start, end, strategy_cls=BuyAndHoldStrategy, **strategy_kwargs):
    """Minimal event loop for Phase 1 — no portfolio or execution yet."""
    queue = EventQueue()
    data = DataHandler(symbols, start, end, queue)
    strategy = strategy_cls(data, queue, **strategy_kwargs)

    market_count = 0
    while not data.is_exhausted:
        data.update_bars()
        while not queue.empty():
            event = queue.get()
            if event.type == EventType.MARKET:
                market_count += 1
                strategy.calculate_signals()

    return market_count


if __name__ == "__main__":
    count = run_loop(["SPY"], "2023-01-01", "2023-01-31")
    print(f"Phase 1 complete: processed {count} market events")
