from .events import OrderEvent, FillEvent, SignalDirection
from .queue import EventQueue


class SimBroker:
    def __init__(self, data_handler, queue: EventQueue,
                 commission_per_share: float = 0.001,
                 slippage_bps: float = 5.0):
        self.data = data_handler
        self.queue = queue
        self.commission_per_share = commission_per_share
        self.slippage_bps = slippage_bps

    def execute_order(self, event: OrderEvent) -> None:
        next_open = self.data.get_next_open(event.symbol)
        if next_open is None:
            return  # no next bar — cannot fill

        slippage_per_share = next_open * self.slippage_bps / 10_000
        if event.quantity > 0:
            fill_price = next_open + slippage_per_share
        else:
            fill_price = next_open - slippage_per_share

        commission = abs(event.quantity) * self.commission_per_share
        slippage_cost = abs(event.quantity) * slippage_per_share

        self.queue.put(FillEvent(
            symbol=event.symbol,
            quantity=event.quantity,
            fill_price=fill_price,
            commission=commission,
            slippage=slippage_cost,
            timestamp=self.data._dates[self.data._cursor],
        ))
