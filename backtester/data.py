from typing import Dict, List, Optional
import pandas as pd
import yfinance as yf
from .events import MarketEvent
from .queue import EventQueue


class DataHandler:
    def __init__(self, symbols: List[str], start: str, end: str, queue: EventQueue):
        self.symbols = symbols
        self.queue = queue
        self._data: Dict[str, pd.DataFrame] = {}
        self._cursor: int = 0
        self._dates: pd.DatetimeIndex = pd.DatetimeIndex([])
        self._fetch(start, end)

    def _fetch(self, start: str, end: str) -> None:
        raw = yf.download(
            self.symbols, start=start, end=end,
            auto_adjust=True, progress=False,
        )
        # yfinance always returns a MultiIndex DataFrame (Price x Ticker).
        # Use xs to extract a flat per-symbol DataFrame regardless of how many symbols.
        for sym in self.symbols:
            self._data[sym] = raw.xs(sym, axis=1, level=1)
        self._dates = raw.index

    def get_latest_bars(self, symbol: str, N: int = 1) -> Optional[pd.DataFrame]:
        """Return up to N bars ending at the current cursor. Never reveals future data."""
        if self._cursor == 0:
            return None
        start = max(0, self._cursor - N)
        return self._data[symbol].iloc[start:self._cursor]

    def get_next_open(self, symbol: str) -> Optional[float]:
        """Broker-only: returns next bar's open for fill simulation."""
        if self._cursor >= len(self._dates):
            return None
        return float(self._data[symbol]["Open"].iloc[self._cursor])

    def update_bars(self) -> bool:
        """Advance cursor by one and put a MarketEvent on the queue."""
        if self._cursor >= len(self._dates):
            return False
        self._cursor += 1
        self.queue.put(MarketEvent())
        return True

    @property
    def current_date(self) -> Optional[pd.Timestamp]:
        if self._cursor == 0:
            return None
        return self._dates[self._cursor - 1]

    @property
    def is_exhausted(self) -> bool:
        return self._cursor >= len(self._dates)
