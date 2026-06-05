from typing import Callable, Dict, List
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta


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


class WalkForwardRunner:
    def __init__(
        self,
        run_fn: Callable,
        data_start: str,
        data_end: str,
        train_months: int = 12,
        test_months: int = 3,
    ):
        self.run_fn = run_fn
        self.data_start = pd.Timestamp(data_start)
        self.data_end = pd.Timestamp(data_end)
        self.train_months = train_months
        self.test_months = test_months

    def run(self) -> List[Dict]:
        results = []
        train_start = self.data_start

        while True:
            train_end = train_start + relativedelta(months=self.train_months)
            test_start = train_end
            test_end = test_start + relativedelta(months=self.test_months)

            if test_end > self.data_end:
                break

            result = self.run_fn(
                train_start=train_start.strftime("%Y-%m-%d"),
                train_end=train_end.strftime("%Y-%m-%d"),
                test_start=test_start.strftime("%Y-%m-%d"),
                test_end=test_end.strftime("%Y-%m-%d"),
            )
            result.update({
                "train_start": train_start.strftime("%Y-%m-%d"),
                "train_end": train_end.strftime("%Y-%m-%d"),
                "test_start": test_start.strftime("%Y-%m-%d"),
                "test_end": test_end.strftime("%Y-%m-%d"),
            })
            results.append(result)

            train_start = train_start + relativedelta(months=self.test_months)

        return results
