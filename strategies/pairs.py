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
