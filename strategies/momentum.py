"""Cross-sectional 12-1 momentum — a demonstration strategy.

This exists to prove the engine is genuinely strategy-agnostic. It is a
completely different shape from the pairs trade:

  * single-leg (not paired), long-only,
  * rebalancing on a calendar (monthly), not event-triggered entries/exits,
  * holds many positions at once (a whole quintile).

It plugs into the same event loop with **no changes to the engine core**
(`data.py`, `execution.py`, `portfolio.py`, `queue.py`): it only subclasses
`BaseStrategy`, reads bars through the point-in-time `get_latest_bars` API, and
emits `SignalEvent`s with explicit share quantities.

It is deliberately NOT tuned to perform well — the point is plumbing, not alpha.
Parameters are textbook defaults (12-month lookback, skip the most recent
month, long the top quintile). Results are reported honestly, good or bad.
"""
from typing import Dict, List, Optional

import pandas as pd

from backtester.events import SignalEvent, SignalDirection
from backtester.strategy import BaseStrategy
from backtester.queue import EventQueue


class CrossSectionalMomentumStrategy(BaseStrategy):
    def __init__(
        self,
        data_handler,
        queue: EventQueue,
        lookback: int = 252,        # ~12 months of trading days
        skip: int = 21,             # ~1 month: the "12-1" exclusion window
        top_quantile: float = 0.2,  # long the top quintile
        book_capital: float = 100_000.0,
        start_after: Optional[str] = None,  # only trade strictly after this date
    ):
        super().__init__(data_handler, queue)
        self.lookback = lookback
        self.skip = skip
        self.top_quantile = top_quantile
        self.book_capital = book_capital
        self.start_after = pd.Timestamp(start_after) if start_after else None
        self._last_rebalance_month: Optional[tuple] = None
        # Current target share count per symbol (0 = flat).
        self._target: Dict[str, int] = {s: 0 for s in data_handler.symbols}

    def _momentum(self, symbol: str) -> Optional[float]:
        """12-1 momentum: total return from `lookback` bars ago up to `skip`
        bars ago (i.e. excluding the most recent month). None if insufficient
        history. Uses only elapsed bars via the point-in-time API."""
        bars = self.data.get_latest_bars(symbol, self.lookback + 5)
        if bars is None or len(bars) < self.lookback:
            return None
        closes = bars["Close"]
        p_start = float(closes.iloc[-self.lookback])
        p_end = float(closes.iloc[-self.skip])
        if p_start <= 0:
            return None
        return p_end / p_start - 1.0

    def calculate_signals(self) -> None:
        date = self.data.current_date
        if date is None:
            return
        if self.start_after is not None and date <= self.start_after:
            return
        month_key = (date.year, date.month)
        if month_key == self._last_rebalance_month:
            return  # already rebalanced this month — rebalance on first bar only
        self._last_rebalance_month = month_key
        self._rebalance()

    def _rebalance(self) -> None:
        """Rank the universe by 12-1 momentum, target an equal-dollar long book
        across the top quintile, and emit the *delta* trades to get there."""
        moms = {}
        for sym in self.data.symbols:
            m = self._momentum(sym)
            if m is not None:
                moms[sym] = m
        if not moms:
            return  # not enough history yet — stay flat rather than force trades

        ranked = sorted(moms, key=moms.get, reverse=True)
        n_long = max(1, int(round(len(ranked) * self.top_quantile)))
        longs = ranked[:n_long]
        per_name = self.book_capital / n_long

        new_target: Dict[str, int] = {s: 0 for s in self.data.symbols}
        for sym in longs:
            bars = self.data.get_latest_bars(sym, 1)
            if bars is None or len(bars) == 0:
                continue
            price = float(bars["Close"].iloc[-1])
            if price <= 0:
                continue
            new_target[sym] = int(per_name / price)

        # Emit deltas so existing holdings are adjusted, not churned wholesale.
        for sym in self.data.symbols:
            delta = new_target[sym] - self._target[sym]
            if delta == 0:
                continue
            direction = SignalDirection.LONG if delta > 0 else SignalDirection.SHORT
            self.queue.put(SignalEvent(sym, direction, quantity=delta))
        self._target = new_target
