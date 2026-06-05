from abc import ABC, abstractmethod
from typing import Dict
from .events import SignalEvent, SignalDirection
from .queue import EventQueue


class BaseStrategy(ABC):
    def __init__(self, data_handler, queue: EventQueue):
        self.data = data_handler
        self.queue = queue

    @abstractmethod
    def calculate_signals(self) -> None:
        raise NotImplementedError


class BuyAndHoldStrategy(BaseStrategy):
    def __init__(self, data_handler, queue: EventQueue):
        super().__init__(data_handler, queue)
        self._bought: Dict[str, bool] = {sym: False for sym in data_handler.symbols}

    def calculate_signals(self) -> None:
        for symbol in self.data.symbols:
            if not self._bought[symbol]:
                bars = self.data.get_latest_bars(symbol, 1)
                if bars is not None and len(bars) > 0:
                    self.queue.put(SignalEvent(symbol=symbol, direction=SignalDirection.LONG))
                    self._bought[symbol] = True
