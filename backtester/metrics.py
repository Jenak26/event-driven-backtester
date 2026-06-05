import numpy as np
import pandas as pd


def sharpe_ratio(
    returns: pd.Series,
    risk_free: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    excess = returns - risk_free / periods_per_year
    std = excess.std()
    if std == 0:
        return 0.0
    return float(np.sqrt(periods_per_year) * excess.mean() / std)


def sortino_ratio(
    returns: pd.Series,
    risk_free: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    excess = returns - risk_free / periods_per_year
    downside = excess[excess < 0]
    downside_std = downside.std()
    # No (or insufficient) downside observations => no downside risk.
    if downside_std == 0 or np.isnan(downside_std):
        return float("inf")
    return float(np.sqrt(periods_per_year) * excess.mean() / downside_std)


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = (equity - peak) / peak
    return float(dd.min())


def annualized_return(equity: pd.Series, periods_per_year: int = 252) -> float:
    total = float(equity.iloc[-1] / equity.iloc[0])
    # A series of length n spans n-1 periods (one return per step).
    n_periods = max(len(equity) - 1, 1)
    return total ** (periods_per_year / n_periods) - 1
