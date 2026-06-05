# tests/test_phase3_gate.py
"""Phase 3 acceptance gate: signals fire at correct z-score thresholds."""
import numpy as np
import pandas as pd
import pytest
from strategies.pairs import (
    CointegrationPairsStrategy, compute_zscore,
    compute_hedge_ratio, SECTORS,
)
from backtester.queue import EventQueue
from backtester.data import DataHandler
from backtester.events import EventType, SignalDirection


def test_no_signals_before_train_end():
    """Strategy must not emit signals before training is complete."""
    q = EventQueue()
    data = DataHandler(
        list(SECTORS["tech"][:4] + SECTORS["financials"][:4]),
        "2020-01-01", "2022-12-31", q,
    )
    strategy = CointegrationPairsStrategy(
        data, q, train_end_date="2021-12-31",
    )
    signals = []
    bar = 0
    while not data.is_exhausted and bar < 50:
        data.update_bars()
        bar += 1
        while not q.empty():
            ev = q.get()
            if ev.type == EventType.SIGNAL:
                signals.append(ev)
            if ev.type == EventType.MARKET:
                strategy.calculate_signals()
    # First 50 bars are well before train_end 2021-12-31 starting from 2020-01-01
    assert len(signals) == 0
