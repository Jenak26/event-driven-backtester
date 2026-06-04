from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class EventType(Enum):
    MARKET = "MARKET"
    SIGNAL = "SIGNAL"
    ORDER = "ORDER"
    FILL = "FILL"


class SignalDirection(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    EXIT = "EXIT"


class OrderType(Enum):
    MARKET = "MARKET"


@dataclass
class MarketEvent:
    type: EventType = field(default=EventType.MARKET, init=False)


@dataclass
class SignalEvent:
    symbol: str
    direction: SignalDirection
    quantity: Optional[float] = None  # None = let Portfolio decide sizing
    type: EventType = field(default=EventType.SIGNAL, init=False)


@dataclass
class OrderEvent:
    symbol: str
    order_type: OrderType
    quantity: float  # positive = buy, negative = sell (short)
    direction: SignalDirection
    type: EventType = field(default=EventType.ORDER, init=False)


@dataclass
class FillEvent:
    symbol: str
    quantity: float
    fill_price: float
    commission: float
    slippage: float
    timestamp: datetime
    type: EventType = field(default=EventType.FILL, init=False)
