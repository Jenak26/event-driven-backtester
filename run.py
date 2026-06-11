import pandas as pd
from backtester.events import EventType
from backtester.queue import EventQueue
from backtester.data import DataHandler
from backtester.strategy import BuyAndHoldStrategy
from backtester.portfolio import Portfolio
from backtester.execution import SimBroker


def run_backtest(
    symbols,
    start: str,
    end: str,
    strategy_cls=BuyAndHoldStrategy,
    initial_capital: float = 100_000.0,
    commission: float = 0.001,
    slippage_bps: float = 5.0,
    strategy_kwargs: dict = None,
    preloaded: dict = None,
    return_strategy: bool = False,
):
    """Run a full backtest. Returns equity curve DataFrame indexed by date.

    With return_strategy=True, returns (equity_df, strategy) so callers can
    inspect discovered pairs.
    """
    if strategy_kwargs is None:
        strategy_kwargs = {}

    queue = EventQueue()
    data = DataHandler(symbols, start, end, queue, preloaded=preloaded)
    strategy = strategy_cls(data, queue, **strategy_kwargs)
    portfolio = Portfolio(data, queue, initial_capital)
    broker = SimBroker(data, queue, commission, slippage_bps)

    while not data.is_exhausted:
        data.update_bars()
        while not queue.empty():
            event = queue.get()
            if event.type == EventType.MARKET:
                strategy.calculate_signals()
                portfolio.update_equity()
            elif event.type == EventType.SIGNAL:
                portfolio.update_signal(event)
            elif event.type == EventType.ORDER:
                broker.execute_order(event)
            elif event.type == EventType.FILL:
                portfolio.update_fill(event)

    equity = portfolio.get_equity_df()
    if return_strategy:
        return equity, strategy
    return equity


if __name__ == "__main__":
    df = run_backtest(["SPY"], "2020-01-02", "2023-12-29")
    print(df.tail())
    print(f"Final equity: ${df['equity'].iloc[-1]:,.2f}")
