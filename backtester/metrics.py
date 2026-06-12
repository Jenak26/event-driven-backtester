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


def calmar_ratio(equity: pd.Series, periods_per_year: int = 252) -> float:
    """Annualized return divided by the absolute value of max drawdown.

    Convention: with no drawdown the ratio is undefined — we return +inf for a
    positive/flat curve and -inf for a declining one, so it never silently reads
    as a finite, flattering number.
    """
    ann = annualized_return(equity, periods_per_year)
    mdd = max_drawdown(equity)
    if mdd == 0.0:
        if ann > 0:
            return float("inf")
        if ann < 0:
            return float("-inf")
        return 0.0
    return float(ann / abs(mdd))


def monthly_returns_table(equity: pd.Series) -> pd.DataFrame:
    """Calendar monthly returns laid out as years (rows) x months 1-12 (cols).

    Each cell is that month's compounded return as a fraction (0.012 = +1.2%).
    Months with no data are NaN. This is the data behind the tearsheet's
    monthly-returns heatmap.
    """
    if equity is None or len(equity) < 2:
        return pd.DataFrame()
    equity = equity.sort_index()
    returns = equity.pct_change().dropna()
    if returns.empty:
        return pd.DataFrame()
    # Compound daily returns within each calendar month.
    monthly = (1.0 + returns).resample("ME").prod() - 1.0
    frame = pd.DataFrame({
        "year": monthly.index.year,
        "month": monthly.index.month,
        "ret": monthly.values,
    })
    table = frame.pivot(index="year", columns="month", values="ret")
    # Ensure all 12 month columns are present and ordered.
    table = table.reindex(columns=range(1, 13))
    return table


def trades_from_fills(fills: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct round-trip trades from a fill ledger.

    A trade is one symbol's journey from flat (position 0) back to flat. Its
    P&L is the sum of cash deltas over that journey — which, because the journey
    nets to zero shares, is the realized P&L net of commission and slippage
    (cash delta per fill = -(qty*price + commission + slippage), exactly as the
    Portfolio books it).

    Expected columns in `fills`: symbol, quantity, fill_price, commission,
    slippage, timestamp.
    Returns columns: symbol, direction, pnl, holding_days, entry, exit, n_fills.
    """
    cols = ["symbol", "direction", "pnl", "holding_days", "entry", "exit", "n_fills"]
    if fills is None or len(fills) == 0:
        return pd.DataFrame(columns=cols)

    trades = []
    for symbol, grp in fills.groupby("symbol", sort=False):
        grp = grp.sort_values("timestamp")
        pos = 0.0
        cash = 0.0
        first_ts = None
        first_qty = 0.0
        n_fills = 0
        for _, row in grp.iterrows():
            qty = float(row["quantity"])
            price = float(row["fill_price"])
            comm = float(row.get("commission", 0.0))
            slip = float(row.get("slippage", 0.0))
            if pos == 0.0:
                first_ts = row["timestamp"]
                first_qty = qty
                cash = 0.0
                n_fills = 0
            cash += -(qty * price + comm + slip)
            pos += qty
            n_fills += 1
            if abs(pos) < 1e-9:  # back to flat → close the round trip
                trades.append({
                    "symbol": symbol,
                    "direction": "long" if first_qty > 0 else "short",
                    "pnl": cash,
                    "holding_days": (pd.Timestamp(row["timestamp"])
                                     - pd.Timestamp(first_ts)).days,
                    "entry": pd.Timestamp(first_ts),
                    "exit": pd.Timestamp(row["timestamp"]),
                    "n_fills": n_fills,
                })
                pos = 0.0
    if not trades:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(trades, columns=cols)


def trade_stats(trades: pd.DataFrame) -> Dict[str, float]:
    """Summary statistics for a reconstructed trade ledger: win rate, average
    win/loss, profit factor (gross profit / gross loss), and average holding
    period in days."""
    if trades is None or len(trades) == 0:
        return {
            "n_trades": 0, "win_rate": float("nan"),
            "avg_win": float("nan"), "avg_loss": float("nan"),
            "profit_factor": float("nan"), "avg_holding_days": float("nan"),
        }
    pnl = trades["pnl"]
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    gross_profit = float(wins.sum())
    gross_loss = float(-losses.sum())
    if gross_loss == 0.0:
        profit_factor = float("inf") if gross_profit > 0 else float("nan")
    else:
        profit_factor = gross_profit / gross_loss
    return {
        "n_trades": int(len(trades)),
        "win_rate": float(len(wins) / len(trades)),
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "profit_factor": profit_factor,
        "avg_holding_days": float(trades["holding_days"].mean()),
    }


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
