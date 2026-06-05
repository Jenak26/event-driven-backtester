from typing import Dict, List, Optional
import pandas as pd
from .events import SignalEvent, OrderEvent, FillEvent, SignalDirection, OrderType, EventType
from .queue import EventQueue

_MAX_EXPOSURE = 0.95  # never commit more than 95% of cash to a single order


class Portfolio:
    def __init__(self, data_handler, queue: EventQueue,
                 initial_capital: float = 100_000.0):
        self.data = data_handler
        self.queue = queue
        self.initial_capital = initial_capital
        self.cash: float = initial_capital
        self.positions: Dict[str, float] = {sym: 0.0 for sym in data_handler.symbols}
        self._equity_curve: List[dict] = []

    def update_signal(self, event: SignalEvent) -> None:
        order = self._generate_order(event)
        if order is not None:
            self.queue.put(order)

    def _generate_order(self, signal: SignalEvent) -> Optional[OrderEvent]:
        if signal.direction == SignalDirection.EXIT:
            qty = -self.positions.get(signal.symbol, 0.0)
            if qty == 0.0:
                return None
            return OrderEvent(
                symbol=signal.symbol,
                order_type=OrderType.MARKET,
                quantity=qty,
                direction=signal.direction,
            )

        if signal.quantity is not None:
            qty = signal.quantity
        else:
            bars = self.data.get_latest_bars(signal.symbol, 1)
            if bars is None or len(bars) == 0:
                return None
            price = float(bars["Close"].iloc[-1])
            max_spend = self.cash * _MAX_EXPOSURE
            raw_qty = int(max_spend / price)
            qty = raw_qty if signal.direction == SignalDirection.LONG else -raw_qty

        if qty == 0:
            return None

        return OrderEvent(
            symbol=signal.symbol,
            order_type=OrderType.MARKET,
            quantity=qty,
            direction=signal.direction,
        )

    def update_fill(self, event: FillEvent) -> None:
        self.positions[event.symbol] = self.positions.get(event.symbol, 0.0) + event.quantity
        cost = event.quantity * event.fill_price + event.commission + event.slippage
        self.cash -= cost

    def update_equity(self) -> None:
        market_value = 0.0
        for sym, qty in self.positions.items():
            if qty == 0.0:
                continue
            bars = self.data.get_latest_bars(sym, 1)
            if bars is not None and len(bars) > 0:
                market_value += qty * float(bars["Close"].iloc[-1])
        self._equity_curve.append({
            "date": self.data.current_date,
            "equity": self.cash + market_value,
            "cash": self.cash,
        })

    def get_equity_df(self) -> pd.DataFrame:
        return pd.DataFrame(self._equity_curve).set_index("date")
