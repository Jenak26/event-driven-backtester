from itertools import combinations
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

from backtester.events import SignalEvent, SignalDirection
from backtester.strategy import BaseStrategy
from backtester.queue import EventQueue

SECTORS: Dict[str, List[str]] = {
    "tech": [
        "AAPL", "MSFT", "NVDA", "GOOGL", "META",
        "AVGO", "ORCL", "AMD", "QCOM", "TXN", "INTC", "CRM",
    ],
    "financials": [
        "JPM", "BAC", "WFC", "GS", "MS",
        "C", "BLK", "SCHW", "AXP", "USB", "PNC", "TFC", "COF",
    ],
}

# Type alias
PairSpec = Tuple[str, str, float]  # (sym_a, sym_b, beta)


def screen_pairs(
    in_sample_prices: Dict[str, pd.Series],
    sectors: Dict[str, List[str]],
    oos_prices: Optional[Dict[str, pd.Series]],
    bonferroni_alpha: float = 0.05,
    oos_alpha: float = 0.05,
    min_bars: int = 60,
) -> List[PairSpec]:
    """
    Return pairs that pass:
    1. Engle-Granger cointegration at Bonferroni-adjusted threshold (in-sample).
    2. Engle-Granger re-qualification at oos_alpha (out-of-sample), if oos_prices provided.
    """
    candidate_pairs = [
        (a, b)
        for syms in sectors.values()
        for a, b in combinations(
            [s for s in syms if s in in_sample_prices], 2
        )
    ]
    if not candidate_pairs:
        return []

    n_tests = len(candidate_pairs)
    bonferroni_threshold = bonferroni_alpha / n_tests

    qualified: List[PairSpec] = []
    for sym_a, sym_b in candidate_pairs:
        sa = in_sample_prices[sym_a].dropna()
        sb = in_sample_prices[sym_b].dropna()
        common = sa.index.intersection(sb.index)
        if len(common) < min_bars:
            continue
        _, pval, _ = coint(sa.loc[common], sb.loc[common])
        if pval < bonferroni_threshold:
            beta = compute_hedge_ratio(sa.loc[common], sb.loc[common])
            qualified.append((sym_a, sym_b, beta))

    if not qualified or oos_prices is None:
        return qualified

    # OOS requalification
    final: List[PairSpec] = []
    for sym_a, sym_b, beta in qualified:
        if sym_a not in oos_prices or sym_b not in oos_prices:
            continue
        oa = oos_prices[sym_a].dropna()
        ob = oos_prices[sym_b].dropna()
        common = oa.index.intersection(ob.index)
        if len(common) < 20:
            continue
        _, pval_oos, _ = coint(oa.loc[common], ob.loc[common])
        if pval_oos < oos_alpha:
            final.append((sym_a, sym_b, beta))

    return final


def compute_hedge_ratio(series_a: pd.Series, series_b: pd.Series) -> float:
    """OLS regression: A = alpha + beta * B + eps. Returns beta."""
    X = add_constant(series_b.values)
    model = OLS(series_a.values, X).fit()
    return float(model.params[1])


def compute_zscore(spread: pd.Series, window: int = 60) -> pd.Series:
    """Rolling z-score of the spread series."""
    mu = spread.rolling(window).mean()
    sigma = spread.rolling(window).std()
    return (spread - mu) / sigma


def check_rolling_coint(
    series_a: pd.Series, series_b: pd.Series, window: int = 60
) -> float:
    """
    Engle-Granger p-value on the last `window` observations.
    Returns 1.0 (worst) if fewer than 30 observations are available.
    """
    common = series_a.index.intersection(series_b.index)
    tail = common[-window:]
    if len(tail) < 30:
        return 1.0
    _, pval, _ = coint(series_a.loc[tail], series_b.loc[tail])
    return float(pval)


class CointegrationPairsStrategy(BaseStrategy):
    def __init__(
        self,
        data_handler,
        queue: EventQueue,
        train_end_date: str,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        stop_coint_pval: float = 0.1,
        z_window: int = 60,
        coint_window: int = 60,
        leg_capital: float = 10_000.0,
    ):
        super().__init__(data_handler, queue)
        self.train_end = pd.Timestamp(train_end_date)
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.stop_coint_pval = stop_coint_pval
        self.z_window = z_window
        self.coint_window = coint_window
        self.leg_capital = leg_capital
        self._pairs: List[PairSpec] = []
        self._active: Dict[Tuple[str, str], Optional[str]] = {}
        self._discovered = False

    def _get_close(self, symbol: str, n: int) -> Optional[pd.Series]:
        bars = self.data.get_latest_bars(symbol, n)
        if bars is None or len(bars) == 0:
            return None
        return bars["Close"]

    def _discover(self) -> None:
        prices = {}
        for sym in self.data.symbols:
            bars = self.data._data[sym]
            mask = bars.index <= self.train_end
            prices[sym] = bars.loc[mask, "Close"]

        # Split in-sample / OOS requalification window (first 3 months after train_end)
        oos_end = self.train_end + pd.DateOffset(months=3)
        oos_prices = {}
        for sym in self.data.symbols:
            bars = self.data._data[sym]
            mask = (bars.index > self.train_end) & (bars.index <= oos_end)
            oos_prices[sym] = bars.loc[mask, "Close"]

        # Filter universe to symbols actually in the data
        active_sectors = {
            sector: [s for s in syms if s in self.data.symbols]
            for sector, syms in SECTORS.items()
        }

        self._pairs = screen_pairs(prices, active_sectors, oos_prices)
        self._active = {(a, b): None for a, b, _ in self._pairs}
        self._discovered = True

    def calculate_signals(self) -> None:
        if self.data.current_date is None:
            return
        if not self._discovered:
            if self.data.current_date >= self.train_end:
                self._discover()
            return

        for sym_a, sym_b, beta in self._pairs:
            pair = (sym_a, sym_b)
            trade_side = self._active[pair]

            # Stop-out check
            if trade_side is not None:
                sa = self._get_close(sym_a, self.coint_window)
                sb = self._get_close(sym_b, self.coint_window)
                if sa is not None and sb is not None:
                    pval = check_rolling_coint(sa, sb, self.coint_window)
                    if pval > self.stop_coint_pval:
                        self.queue.put(SignalEvent(symbol=sym_a, direction=SignalDirection.EXIT))
                        self.queue.put(SignalEvent(symbol=sym_b, direction=SignalDirection.EXIT))
                        self._active[pair] = None
                        continue

            # Z-score signal
            sa = self._get_close(sym_a, self.z_window + 20)
            sb = self._get_close(sym_b, self.z_window + 20)
            if sa is None or sb is None:
                continue

            common = sa.index.intersection(sb.index)
            if len(common) < self.z_window:
                continue

            spread = sa.loc[common] - beta * sb.loc[common]
            z_series = compute_zscore(spread, self.z_window)
            if z_series.isna().iloc[-1]:
                continue
            z = float(z_series.iloc[-1])

            # Determine position sizes using hedge-ratio neutral sizing
            price_a = float(sa.iloc[-1])
            price_b = float(sb.iloc[-1])
            qty_a = int(self.leg_capital / price_a)
            qty_b = int(self.leg_capital * beta / price_b)

            if trade_side is None:
                if z > self.entry_z:
                    # Spread too high: short A, long B
                    self.queue.put(SignalEvent(sym_a, SignalDirection.SHORT, quantity=-qty_a))
                    self.queue.put(SignalEvent(sym_b, SignalDirection.LONG, quantity=qty_b))
                    self._active[pair] = "short_a"
                elif z < -self.entry_z:
                    # Spread too low: long A, short B
                    self.queue.put(SignalEvent(sym_a, SignalDirection.LONG, quantity=qty_a))
                    self.queue.put(SignalEvent(sym_b, SignalDirection.SHORT, quantity=-qty_b))
                    self._active[pair] = "long_a"
            else:
                if abs(z) < self.exit_z:
                    self.queue.put(SignalEvent(symbol=sym_a, direction=SignalDirection.EXIT))
                    self.queue.put(SignalEvent(symbol=sym_b, direction=SignalDirection.EXIT))
                    self._active[pair] = None
