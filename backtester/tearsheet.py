import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from .metrics import (
    sharpe_ratio, max_drawdown, annualized_return, calmar_ratio,
    monthly_returns_table, trades_from_fills, trade_stats,
)


def plot_tearsheet(
    equity_df: pd.DataFrame,
    title: str = "Backtest Results",
    rolling_sharpe_window: int = 126,
    save_path: str = None,
) -> plt.Figure:
    """Three-panel tearsheet: equity curve, drawdown, rolling Sharpe."""
    equity = equity_df["equity"]
    returns = equity.pct_change().dropna()

    # Rolling Sharpe (annualized, 126-day ~ 6 months)
    rolling_sr = returns.rolling(rolling_sharpe_window).apply(
        lambda r: sharpe_ratio(pd.Series(r)), raw=False
    )

    # Drawdown
    peak = equity.cummax()
    drawdown = (equity - peak) / peak

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(title, fontsize=14)

    # Panel 1: Equity curve
    axes[0].plot(equity.index, equity.values, linewidth=1.2, color="#1f77b4")
    axes[0].set_ylabel("Portfolio Value ($)")
    axes[0].set_title(
        f"Ann. Return: {annualized_return(equity):.1%} | "
        f"Sharpe: {sharpe_ratio(returns):.2f} | "
        f"Calmar: {calmar_ratio(equity):.2f} | "
        f"Max DD: {max_drawdown(equity):.1%}"
    )
    axes[0].grid(alpha=0.3)

    # Panel 2: Drawdown
    axes[1].fill_between(drawdown.index, drawdown.values, 0, alpha=0.5, color="#d62728")
    axes[1].set_ylabel("Drawdown")
    axes[1].grid(alpha=0.3)

    # Panel 3: Rolling Sharpe
    axes[2].plot(rolling_sr.index, rolling_sr.values, linewidth=1.0, color="#2ca02c")
    axes[2].axhline(0, color="black", linewidth=0.5, linestyle="--")
    axes[2].set_ylabel(f"Rolling {rolling_sharpe_window}d Sharpe")
    axes[2].set_xlabel("Date")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_monthly_heatmap(equity_df: pd.DataFrame, ax=None, title: str = "Monthly Returns (%)"):
    """Monthly-returns heatmap: years (rows) x months (cols), diverging colormap
    (red negative, green positive). Standard in professional tearsheets."""
    import seaborn as sns

    equity = equity_df["equity"] if isinstance(equity_df, pd.DataFrame) else equity_df
    table = monthly_returns_table(equity) * 100.0  # to percent

    if ax is None:
        _, ax = plt.subplots(figsize=(11, max(2.0, 0.5 * len(table) + 1)))

    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    vmax = float(np.nanmax(np.abs(table.values))) if table.size else 1.0
    sns.heatmap(
        table, ax=ax, annot=True, fmt=".1f", cmap="RdYlGn", center=0.0,
        vmin=-vmax, vmax=vmax, linewidths=0.5, linecolor="white",
        cbar_kws={"label": "return %"},
        xticklabels=month_labels, yticklabels=table.index,
    )
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("")
    return ax


def plot_trade_analysis(fills_df: pd.DataFrame, ax=None, title: str = "Trade P&L (cost-adjusted)"):
    """Histogram of individual round-trip trade P&L (net of costs), annotated
    with win rate, average win/loss, profit factor and average holding period."""
    trades = trades_from_fills(fills_df)
    stats = trade_stats(trades)

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    if stats["n_trades"] == 0:
        ax.text(0.5, 0.5, "No completed round-trip trades",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return ax, stats

    pnl = trades["pnl"]
    ax.hist(pnl, bins=min(30, max(5, len(pnl))), color="#4c78a8",
            edgecolor="white", alpha=0.85)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    pf = stats["profit_factor"]
    pf_str = "inf" if pf == float("inf") else f"{pf:.2f}"
    ax.set_title(
        f"{title}\n"
        f"n={stats['n_trades']} | win rate {stats['win_rate']:.0%} | "
        f"avg win ${stats['avg_win']:.0f} | avg loss ${stats['avg_loss']:.0f} | "
        f"PF {pf_str} | avg hold {stats['avg_holding_days']:.0f}d"
    )
    ax.set_xlabel("Trade P&L ($, net of commission + slippage)")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.3)
    return ax, stats
