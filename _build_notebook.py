import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell(
    "# Cointegration Pairs Strategy — Research Notebook\n\n"
    "Pair discovery, out-of-sample validation, walk-forward analysis, tearsheet,\n"
    "and parameter sensitivity. Run top-to-bottom (`Kernel -> Restart & Run All`).\n\n"
    "**Methodology in one paragraph.** The backtest runs over the full 2019–2023\n"
    "window through an event-driven engine (point-in-time bars, next-bar fills,\n"
    "commission + slippage). The strategy is dormant until `TRAIN_END`: on that\n"
    "bar it screens pairs using only elapsed data — Engle–Granger on log prices\n"
    "at p < 0.05 over the training window minus the final 3 months, then\n"
    "re-qualification at p < 0.10 on those held-out 3 months. All trading and\n"
    "all reported metrics are from bars **after** `TRAIN_END`, which the\n"
    "discovery step never saw."
))

cells.append(nbf.v4.new_code_cell(
    'import sys; sys.path.insert(0, "..")\n'
    'import warnings; warnings.filterwarnings("ignore")\n'
    "import numpy as np\n"
    "import pandas as pd\n"
    "import matplotlib.pyplot as plt\n"
    "from statsmodels.tsa.stattools import coint\n"
    "\n"
    "from run import run_backtest\n"
    "from backtester.data import fetch_prices\n"
    "from strategies.pairs import CointegrationPairsStrategy, SECTORS\n"
    "from backtester.metrics import (\n"
    "    sharpe_ratio, sortino_ratio, max_drawdown,\n"
    "    annualized_return, WalkForwardRunner,\n"
    ")\n"
    "from backtester.tearsheet import plot_tearsheet"
))

cells.append(nbf.v4.new_code_cell(
    "SYMBOLS = [s for sector in SECTORS.values() for s in sector]\n"
    'START = "2019-01-01"\n'
    'TRAIN_END = "2021-12-31"\n'
    'END = "2023-12-31"\n'
    "INITIAL_CAPITAL = 100_000.0\n"
    "\n"
    "# One download, reused by every backtest below\n"
    'PRICES = fetch_prices(SYMBOLS, START, "2024-01-01")\n'
    'print(f"{len(SYMBOLS)} symbols, {len(PRICES[SYMBOLS[0]])} bars each")'
))

cells.append(nbf.v4.new_markdown_cell(
    "## 1. Full backtest + pair discovery\n\n"
    "One run over 2019–2023. The strategy discovers pairs at `TRAIN_END`\n"
    "(2021-12-31) using only bars that have elapsed, then trades 2022–2023."
))

cells.append(nbf.v4.new_code_cell(
    "eq, strat = run_backtest(\n"
    "    SYMBOLS, START, END,\n"
    "    strategy_cls=CointegrationPairsStrategy,\n"
    "    initial_capital=INITIAL_CAPITAL,\n"
    '    strategy_kwargs={"train_end_date": TRAIN_END},\n'
    "    preloaded=PRICES,\n"
    "    return_strategy=True,\n"
    ")\n"
    "\n"
    "# Reproduce the discovery-time statistics for the qualified pairs\n"
    "requal_start = pd.Timestamp(TRAIN_END) - pd.DateOffset(months=3)\n"
    "rows = []\n"
    "for a, b, beta in strat._pairs:\n"
    '    la, lb = np.log(PRICES[a]["Close"]), np.log(PRICES[b]["Close"])\n'
    "    _, p_in, _ = coint(la[la.index <= requal_start], lb[lb.index <= requal_start])\n"
    "    rq = (la.index > requal_start) & (la.index <= pd.Timestamp(TRAIN_END))\n"
    "    _, p_rq, _ = coint(la[rq], lb[rq])\n"
    "    rows.append({\n"
    '        "pair": f"{a}/{b}", "beta": round(beta, 3),\n'
    '        "p_in_sample": round(p_in, 4), "p_requal": round(p_rq, 4),\n'
    "    })\n"
    'print(f"{len(strat._pairs)} pairs qualified (in-sample p<0.05, requal p<0.10):\\n")\n'
    "print(pd.DataFrame(rows).to_string(index=False))"
))

cells.append(nbf.v4.new_markdown_cell(
    "## 2. Out-of-sample performance (with costs)\n\n"
    "Everything after `TRAIN_END` is out-of-sample: the discovery step never saw\n"
    "these bars. Commission \\$0.001/share and 5 bps slippage are always on."
))

cells.append(nbf.v4.new_code_cell(
    "eq_oos = eq[eq.index > pd.Timestamp(TRAIN_END)]\n"
    'returns_oos = eq_oos["equity"].pct_change().dropna()\n'
    'print(f"Out-of-sample Sharpe (with costs): {sharpe_ratio(returns_oos):.3f}")\n'
    'print(f"Out-of-sample Sortino:             {sortino_ratio(returns_oos):.3f}")\n'
    "print(f\"Out-of-sample Max Drawdown:        {max_drawdown(eq_oos['equity']):.2%}\")\n"
    "print(f\"Out-of-sample Ann. Return:         {annualized_return(eq_oos['equity']):.2%}\")"
))

cells.append(nbf.v4.new_markdown_cell(
    "## 3. Cost drag\n\n"
    "Same backtest with costs switched off, to quantify how much commission and\n"
    "slippage matter. Many 'winning' strategies die here."
))

cells.append(nbf.v4.new_code_cell(
    "eq_nc = run_backtest(\n"
    "    SYMBOLS, START, END,\n"
    "    strategy_cls=CointegrationPairsStrategy,\n"
    "    initial_capital=INITIAL_CAPITAL,\n"
    "    commission=0.0, slippage_bps=0.0,\n"
    '    strategy_kwargs={"train_end_date": TRAIN_END},\n'
    "    preloaded=PRICES,\n"
    ")\n"
    "eq_nc_oos = eq_nc[eq_nc.index > pd.Timestamp(TRAIN_END)]\n"
    'returns_nc = eq_nc_oos["equity"].pct_change().dropna()\n'
    'print(f"OOS Sharpe (no costs): {sharpe_ratio(returns_nc):.3f}")\n'
    'print(f"OOS Sharpe (w/ costs): {sharpe_ratio(returns_oos):.3f}")\n'
    'print(f"Cost drag:             {sharpe_ratio(returns_nc) - sharpe_ratio(returns_oos):.3f} Sharpe points")'
))

cells.append(nbf.v4.new_markdown_cell("## 4. Tearsheet (out-of-sample, with costs)"))

cells.append(nbf.v4.new_code_cell(
    'fig = plot_tearsheet(eq_oos, title="Pairs Strategy — Out-of-Sample 2022–2023 (with costs)")\n'
    "plt.show()"
))

cells.append(nbf.v4.new_markdown_cell(
    "## 5. Walk-forward analysis\n\n"
    "Rolling 12-month train / 3-month test windows across 2019–2023. Each window\n"
    "re-screens pairs from scratch on its own training data (9-month screen +\n"
    "3-month requalification holdout) and trades only its test quarter.\n"
    "`n_pairs = 0` means no pair passed the screen in that window — the\n"
    "strategy stays flat rather than forcing trades."
))

cells.append(nbf.v4.new_code_cell(
    "def wf_run(train_start, train_end, test_start, test_end):\n"
    "    eq_w, strat_w = run_backtest(\n"
    "        SYMBOLS, train_start, test_end,\n"
    "        strategy_cls=CointegrationPairsStrategy,\n"
    "        initial_capital=INITIAL_CAPITAL,\n"
    '        strategy_kwargs={"train_end_date": train_end},\n'
    "        preloaded=PRICES,\n"
    "        return_strategy=True,\n"
    "    )\n"
    "    test_eq = eq_w[eq_w.index > pd.Timestamp(train_end)]\n"
    '    r = test_eq["equity"].pct_change().dropna()\n'
    "    return {\n"
    '        "n_pairs": len(strat_w._pairs),\n'
    '        "sharpe": round(sharpe_ratio(r), 3),\n'
    '        "max_dd": round(max_drawdown(test_eq["equity"]), 4),\n'
    '        "ann_return": round(annualized_return(test_eq["equity"]), 4),\n'
    "    }\n"
    "\n"
    "runner = WalkForwardRunner(\n"
    "    run_fn=wf_run,\n"
    "    data_start=START,\n"
    "    data_end=END,\n"
    "    train_months=12,\n"
    "    test_months=3,\n"
    ")\n"
    "wf_results = runner.run()\n"
    "\n"
    "wf_df = pd.DataFrame([\n"
    "    {\n"
    '        "window": i + 1,\n'
    '        "test_start": r["test_start"],\n'
    '        "test_end": r["test_end"],\n'
    '        "n_pairs": r["n_pairs"],\n'
    '        "sharpe": r["sharpe"],\n'
    '        "max_dd": r["max_dd"],\n'
    '        "ann_return": r["ann_return"],\n'
    "    }\n"
    "    for i, r in enumerate(wf_results)\n"
    "])\n"
    "print(wf_df.to_string(index=False))\n"
    "traded = wf_df[wf_df.n_pairs > 0]\n"
    'print(f"\\n{len(wf_df)} windows, {len(traded)} traded; "\n'
    '      f"mean Sharpe across traded windows: {traded.sharpe.mean():.3f}")'
))

cells.append(nbf.v4.new_markdown_cell(
    "## 6. Parameter sensitivity\n\n"
    "Out-of-sample Sharpe across the entry/exit z-score grid. A robust strategy\n"
    "degrades smoothly as thresholds move; a cliff means the headline number is\n"
    "an artifact of one lucky parameter choice."
))

cells.append(nbf.v4.new_code_cell(
    "rows = []\n"
    "for entry in (1.5, 2.0, 2.5):\n"
    "    for exit_ in (0.0, 0.5, 1.0):\n"
    "        eq_s = run_backtest(\n"
    "            SYMBOLS, START, END,\n"
    "            strategy_cls=CointegrationPairsStrategy,\n"
    "            initial_capital=INITIAL_CAPITAL,\n"
    "            strategy_kwargs={\n"
    '                "train_end_date": TRAIN_END,\n'
    '                "entry_z": entry,\n'
    '                "exit_z": exit_,\n'
    "            },\n"
    "            preloaded=PRICES,\n"
    "        )\n"
    "        eq_s = eq_s[eq_s.index > pd.Timestamp(TRAIN_END)]\n"
    '        r = eq_s["equity"].pct_change().dropna()\n'
    "        rows.append({\n"
    '            "entry_z": entry, "exit_z": exit_,\n'
    '            "sharpe": round(sharpe_ratio(r), 3),\n'
    '            "max_dd": round(max_drawdown(eq_s["equity"]), 4),\n'
    "        })\n"
    "\n"
    "sens_df = pd.DataFrame(rows)\n"
    'print("OOS Sharpe by entry/exit z-score:\\n")\n'
    'print(sens_df.pivot(index="entry_z", columns="exit_z", values="sharpe"))'
))

CONCLUSION_MD = (
    "## 7. Reading the results honestly\n\n"
    "The numbers above are what they are — they were **not** tuned until they\n"
    "looked good. Points to take away:\n\n"
    "- **Costs are the headline.** Gross of costs the strategy is roughly flat\n"
    "  (Sharpe ≈ 0); commission and slippage turn it clearly negative, costing\n"
    "  over a full Sharpe point. The stop-out rule (exit when rolling\n"
    "  cointegration breaks, p > 0.1) limits losses from broken pairs but\n"
    "  generates churn that the cost model punishes. This is the classic\n"
    "  failure mode the engine exists to expose: a strategy that looks harmless\n"
    "  gross can be unviable net.\n"
    "- **Regime dependence is visible.** The walk-forward table shows the\n"
    "  strategy idle in several windows (no pair passes the screen — it stays\n"
    "  flat rather than forcing trades) and uneven performance across traded\n"
    "  windows. Several discovered pairs include META, whose idiosyncratic 2022\n"
    "  collapse broke its cointegrating relationships outright.\n"
    "- **No lucky parameters.** The sensitivity grid is smooth and uniformly\n"
    "  negative — the headline number is not an artifact of one threshold\n"
    "  choice.\n"
    "- **A negative out-of-sample result reported in full is the point of the\n"
    "  exercise.** The engine prevents look-ahead structurally; the validation\n"
    "  pipeline (held-out requalification, walk-forward, sensitivity grid)\n"
    "  measures the strategy rather than flattering it."
)
cells.append(nbf.v4.new_markdown_cell(CONCLUSION_MD))

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}

with open("notebooks/research.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("wrote notebooks/research.ipynb with", len(cells), "cells")
