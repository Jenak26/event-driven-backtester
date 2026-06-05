import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from .metrics import sharpe_ratio, max_drawdown, annualized_return


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
