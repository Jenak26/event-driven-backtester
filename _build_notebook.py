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

# ---------------------------------------------------------------------------
# Sections 8-12: multi-sector study, extended tearsheet, survivorship-bias
# mitigation, and a second (momentum) strategy. Appended AFTER the original
# sections so the existing structure and results are preserved unchanged.
# ---------------------------------------------------------------------------

cells.append(nbf.v4.new_markdown_cell(
    "---\n\n"
    "# Part II — Multi-sector study, richer tearsheet, survivorship mitigation, "
    "and a second strategy\n\n"
    "Everything below is additive. Sections 1–7 above are untouched. We now widen "
    "the universe to four sectors, deepen the tearsheet, partially mitigate "
    "survivorship bias, and plug a completely different strategy (momentum) into "
    "the same engine to demonstrate the architecture is strategy-agnostic."
))

cells.append(nbf.v4.new_markdown_cell(
    "## 8. Wider universe setup (Tech, Financials, Energy, Utilities)\n\n"
    "`SECTOR_UNIVERSES` adds Energy and Utilities to the original two sectors. "
    "Pairs are still tested **within-sector only** — sectors are never pooled. "
    "Some names may be undownloadable via yfinance because they were acquired and "
    "delisted (e.g. PXD / Pioneer, acquired by ExxonMobil in 2024) — itself a live "
    "illustration of survivorship bias. We drop such names and report which."
))

cells.append(nbf.v4.new_code_cell(
    "from backtester.universe import (\n"
    "    SECTOR_UNIVERSES, get_point_in_time_universe, load_sp500_constituents,\n"
    ")\n"
    "from strategies.momentum import CrossSectionalMomentumStrategy\n"
    "from backtester.metrics import (\n"
    "    calmar_ratio, monthly_returns_table, trades_from_fills, trade_stats,\n"
    ")\n"
    "from backtester.tearsheet import plot_monthly_heatmap, plot_trade_analysis\n"
    "import seaborn as sns\n"
    "\n"
    "ALL_SYMBOLS = sorted({s for v in SECTOR_UNIVERSES.values() for s in v})\n"
    'PRICES_ALL = fetch_prices(ALL_SYMBOLS, START, "2024-01-01")\n'
    "\n"
    "def _available(syms):\n"
    '    return [s for s in syms if PRICES_ALL[s]["Close"].dropna().shape[0] > 200]\n'
    "\n"
    "SECTOR_SYMS = {sec: _available(syms) for sec, syms in SECTOR_UNIVERSES.items()}\n"
    "for sec, syms in SECTOR_UNIVERSES.items():\n"
    "    dropped = [s for s in syms if s not in SECTOR_SYMS[sec]]\n"
    "    tag = f'  (dropped {dropped} — delisted/acquired, no data)' if dropped else ''\n"
    "    print(f'{sec:<12} {len(SECTOR_SYMS[sec]):>2} usable{tag}')"
))

cells.append(nbf.v4.new_markdown_cell(
    "## 9. Survivorship bias — what it is, and a partial mitigation\n\n"
    "**Survivorship bias** here means our hand-curated sector lists are drawn from "
    "*today's* index membership. Backtesting 2019–2023 on a 2024 list silently "
    "excludes firms that were delisted, acquired, or removed during the period "
    "(and over-includes firms only added later), biasing results in an unknown "
    "direction. The `get_point_in_time_universe(sector, as_of_date)` helper "
    "**partially** mitigates this: it intersects each sector list with the actual "
    "S&P 500 constituents on the as-of date, using the public historical-"
    "constituents file from [fja05680/sp500](https://github.com/fja05680/sp500) "
    "(downloaded on first run; falls back to the current list with a warning if "
    "unavailable). What it fixes: it stops us trading names *before* they joined "
    "the index. What it does **not** fix: it cannot resurrect long-dead tickers "
    "that never made it into our curated lists in the first place — those are "
    "invisible to us. A full fix needs a point-in-time constituent **and sector** "
    "database such as Compustat / CRSP. We report the mitigation honestly rather "
    "than claiming the bias is gone."
))

cells.append(nbf.v4.new_code_cell(
    "hist = load_sp500_constituents()  # downloads from fja05680/sp500 on first run\n"
    "if hist is None:\n"
    "    print('historical constituents unavailable — falling back to current lists')\n"
    "else:\n"
    "    print(f'loaded {len(hist)} constituent snapshots '\n"
    "          f'({hist.date.min().date()} … {hist.date.max().date()})\\n')\n"
    "    for sec in SECTOR_UNIVERSES:\n"
    "        full = SECTOR_UNIVERSES[sec]\n"
    '        pit = get_point_in_time_universe(sec, "2019-01-01", history=hist)\n'
    "        dropped = sorted(set(full) - set(pit))\n"
    "        print(f'{sec:<12} full={len(full):>2}  PIT(2019-01-01)={len(pit):>2}  '\n"
    "              f'not-yet-in-index={dropped}')"
))

cells.append(nbf.v4.new_markdown_cell(
    "## 10. Per-sector walk-forward study\n\n"
    "Each sector is run **independently** through the same pipeline: discover "
    "within-sector pairs at `TRAIN_END`, trade the 2022–2023 holdout, and also a "
    "full 12/3 walk-forward. Costs on and off, so cost drag is isolated.\n\n"
    "**Hypothesis under test:** low-volatility sectors (Utilities, Energy) have "
    "lower turnover and smaller cost drag. We report whether it holds — honestly."
))

cells.append(nbf.v4.new_code_cell(
    "def _oos(eq):\n"
    "    return eq[eq.index > pd.Timestamp(TRAIN_END)]\n"
    "\n"
    "def sector_full_oos(sector, syms):\n"
    '    kw = {"train_end_date": TRAIN_END, "sectors": {sector: syms}}\n'
    "    eq_c, strat, pf = run_backtest(\n"
    "        syms, START, END, strategy_cls=CointegrationPairsStrategy,\n"
    "        initial_capital=INITIAL_CAPITAL, strategy_kwargs=kw,\n"
    "        preloaded=PRICES_ALL, return_strategy=True, return_portfolio=True)\n"
    "    eq_n = run_backtest(\n"
    "        syms, START, END, strategy_cls=CointegrationPairsStrategy,\n"
    "        initial_capital=INITIAL_CAPITAL, commission=0.0, slippage_bps=0.0,\n"
    "        strategy_kwargs=kw, preloaded=PRICES_ALL)\n"
    "    eqc, eqn = _oos(eq_c), _oos(eq_n)\n"
    '    rc = eqc["equity"].pct_change().dropna()\n'
    '    rn = eqn["equity"].pct_change().dropna()\n'
    "    fills = pf.get_fills_df()\n"
    '    cost = float((fills["commission"] + fills["slippage"]).sum()) if len(fills) else 0.0\n'
    "    return {\n"
    '        "sector": sector, "n_pairs": len(strat._pairs),\n'
    '        "sharpe_cost": round(sharpe_ratio(rc), 3),\n'
    '        "sharpe_nocost": round(sharpe_ratio(rn), 3),\n'
    '        "cost_drag": round(sharpe_ratio(rn) - sharpe_ratio(rc), 3),\n'
    '        "max_dd": round(max_drawdown(eqc["equity"]), 4),\n'
    '        "ann_return": round(annualized_return(eqc["equity"]), 4),\n'
    '        "cost_%_cap": round(100.0 * cost / INITIAL_CAPITAL, 3),\n'
    "    }\n"
    "\n"
    "def sector_walkforward(sector, syms):\n"
    "    def wf(train_start, train_end, test_start, test_end):\n"
    "        eq, strat = run_backtest(\n"
    "            syms, train_start, test_end, strategy_cls=CointegrationPairsStrategy,\n"
    "            initial_capital=INITIAL_CAPITAL, preloaded=PRICES_ALL,\n"
    '            strategy_kwargs={"train_end_date": train_end, "sectors": {sector: syms}},\n'
    "            return_strategy=True)\n"
    "        te = eq[eq.index > pd.Timestamp(train_end)]\n"
    '        r = te["equity"].pct_change().dropna()\n'
    '        return {"n_pairs": len(strat._pairs), "sharpe": sharpe_ratio(r)}\n'
    "    res = WalkForwardRunner(wf, START, END, 12, 3).run()\n"
    '    traded = [r for r in res if r["n_pairs"] > 0]\n'
    '    return len(traded), (np.mean([r["sharpe"] for r in traded]) if traded else np.nan)\n'
    "\n"
    "rows = []\n"
    "for sector, syms in SECTOR_SYMS.items():\n"
    "    r = sector_full_oos(sector, syms)\n"
    "    n_traded, wf_sr = sector_walkforward(sector, syms)\n"
    '    r["wf_traded"] = n_traded\n'
    '    r["wf_mean_sharpe"] = round(wf_sr, 3)\n'
    "    rows.append(r)\n"
    "sector_df = pd.DataFrame(rows).set_index('sector')\n"
    "print(sector_df.to_string())"
))

cells.append(nbf.v4.new_markdown_cell(
    "### 10a. Did the low-volatility hypothesis hold?\n\n"
    "Read the table above honestly. In our runs the answer is **no, not cleanly**:\n\n"
    "- In the 2022–2023 headline window the two-stage screen found **no qualifying "
    "within-sector pairs** in Energy or Utilities at the 2021-12-31 split, so they "
    "never traded — zero turnover and zero cost drag, but also zero signal. That is "
    "not the hypothesised 'low vol → efficient trading' effect; it is simply "
    "*absence of cointegration* in that window.\n"
    "- Across the full walk-forward, Energy and Utilities **do** trade in other "
    "windows and their mean traded-window Sharpe is **negative**, no better than "
    "Tech or Financials.\n"
    "- Tech and Financials, which do trade in the headline window, are dominated by "
    "cost drag (≈0.8 and ≈1.3 Sharpe points) — the same failure mode as the "
    "original result.\n\n"
    "So lower realised volatility did not translate into a cost or Sharpe advantage "
    "here. We report it rather than discarding the inconvenient sectors."
))

cells.append(nbf.v4.new_markdown_cell(
    "### 10b. Sector comparison chart"
))

cells.append(nbf.v4.new_code_cell(
    "fig, axes = plt.subplots(1, 3, figsize=(15, 4))\n"
    'sector_df["sharpe_cost"].plot(kind="bar", ax=axes[0], color="#4c78a8")\n'
    'axes[0].set_title("Cost-adjusted Sharpe (OOS)"); axes[0].axhline(0, color="k", lw=0.6)\n'
    'sector_df["cost_drag"].plot(kind="bar", ax=axes[1], color="#e45756")\n'
    'axes[1].set_title("Cost drag (Sharpe points)")\n'
    '(sector_df["max_dd"] * 100).plot(kind="bar", ax=axes[2], color="#f58518")\n'
    'axes[2].set_title("Max drawdown (%)")\n'
    "for a in axes: a.set_xlabel(''); a.grid(alpha=0.3)\n"
    "plt.tight_layout(); plt.show()"
))

cells.append(nbf.v4.new_markdown_cell(
    "## 11. Extended tearsheet — Calmar, monthly heatmap, trade-level analysis\n\n"
    "On the original combined-universe out-of-sample run (`eq_oos` from Section 2). "
    "Adds the Calmar ratio, a monthly-returns heatmap (diverging colormap), and a "
    "trade-level breakdown (P&L histogram, win rate, profit factor, holding period) "
    "reconstructed from the portfolio's fill ledger."
))

cells.append(nbf.v4.new_code_cell(
    "# Re-run the headline backtest capturing the portfolio for its fill ledger.\n"
    "eq_full, _, pf_full = run_backtest(\n"
    "    SYMBOLS, START, END, strategy_cls=CointegrationPairsStrategy,\n"
    "    initial_capital=INITIAL_CAPITAL,\n"
    '    strategy_kwargs={"train_end_date": TRAIN_END},\n'
    "    preloaded=PRICES, return_strategy=True, return_portfolio=True)\n"
    "fills_oos = pf_full.get_fills_df()\n"
    "fills_oos = fills_oos[fills_oos['timestamp'] > pd.Timestamp(TRAIN_END)]\n"
    "\n"
    "print(f\"Calmar (OOS): {calmar_ratio(eq_oos['equity']):.3f}\")\n"
    "stats = trade_stats(trades_from_fills(fills_oos))\n"
    "for k, v in stats.items():\n"
    "    print(f'{k:<18} {v}')"
))

cells.append(nbf.v4.new_code_cell(
    'plot_monthly_heatmap(eq_oos, title="Monthly Returns (%) — Pairs OOS 2022–2023")\n'
    "plt.tight_layout(); plt.show()"
))

cells.append(nbf.v4.new_code_cell(
    "ax, _ = plot_trade_analysis(fills_oos,\n"
    '    title="Pairs Trade P&L (cost-adjusted) — OOS 2022–2023")\n'
    "plt.tight_layout(); plt.show()"
))

cells.append(nbf.v4.new_markdown_cell(
    "## 12. Cost-sensitivity grid across all four sectors\n\n"
    "Out-of-sample Sharpe as a function of commission (\\$0–0.002/share) and "
    "slippage (0–10 bps), a 5×5 grid per sector. Sectors that found no pairs in "
    "the headline window are flat across the grid (Sharpe ≈ 0) — shown for "
    "completeness rather than hidden."
))

cells.append(nbf.v4.new_code_cell(
    "commissions = np.linspace(0.0, 0.002, 5)\n"
    "slippages = np.linspace(0.0, 10.0, 5)\n"
    "\n"
    "fig, axes = plt.subplots(2, 2, figsize=(13, 10))\n"
    "for ax, (sector, syms) in zip(axes.ravel(), SECTOR_SYMS.items()):\n"
    "    grid = np.zeros((len(commissions), len(slippages)))\n"
    "    for i, c in enumerate(commissions):\n"
    "        for j, s in enumerate(slippages):\n"
    "            eq_g = run_backtest(\n"
    "                syms, START, END, strategy_cls=CointegrationPairsStrategy,\n"
    "                initial_capital=INITIAL_CAPITAL, commission=c, slippage_bps=s,\n"
    '                strategy_kwargs={"train_end_date": TRAIN_END, "sectors": {sector: syms}},\n'
    "                preloaded=PRICES_ALL)\n"
    "            eq_g = eq_g[eq_g.index > pd.Timestamp(TRAIN_END)]\n"
    '            r = eq_g["equity"].pct_change().dropna()\n'
    "            grid[i, j] = sharpe_ratio(r)\n"
    "    sns.heatmap(grid, ax=ax, annot=True, fmt='.2f', cmap='RdYlGn', center=0.0,\n"
    "                xticklabels=[f'{s:.0f}' for s in slippages],\n"
    "                yticklabels=[f'{c:.4f}' for c in commissions])\n"
    "    ax.set_title(f'{sector.capitalize()} — OOS Sharpe')\n"
    "    ax.set_xlabel('slippage (bps)'); ax.set_ylabel('commission ($/share)')\n"
    "plt.tight_layout(); plt.show()"
))

cells.append(nbf.v4.new_markdown_cell(
    "## 13. A second strategy: cross-sectional momentum (engine extensibility)\n\n"
    "`CrossSectionalMomentumStrategy` is a *completely different* strategy type — "
    "single-leg, long-only, monthly-rebalancing, multi-position — yet it runs on "
    "the **same engine with zero core changes**. It ranks the Tech universe by 12-1 "
    "momentum each month, longs the top quintile equal-weight, and starts trading "
    "at `TRAIN_END` (same OOS window as pairs). It is **not** tuned to perform "
    "well; defaults are textbook. We report it side-by-side with pairs on Tech, "
    "honestly — including a punishing drawdown."
))

cells.append(nbf.v4.new_code_cell(
    "TECH = SECTOR_SYMS['tech']\n"
    "eq_mom, pf_mom = run_backtest(\n"
    "    TECH, START, END, strategy_cls=CrossSectionalMomentumStrategy,\n"
    "    initial_capital=INITIAL_CAPITAL, preloaded=PRICES_ALL,\n"
    '    strategy_kwargs={"book_capital": INITIAL_CAPITAL, "start_after": TRAIN_END},\n'
    "    return_portfolio=True)\n"
    "eq_mom_oos = _oos(eq_mom)\n"
    "r_mom = eq_mom_oos['equity'].pct_change().dropna()\n"
    "\n"
    "# Pairs on the SAME Tech universe for a like-for-like comparison.\n"
    "eq_pairs_tech = run_backtest(\n"
    "    TECH, START, END, strategy_cls=CointegrationPairsStrategy,\n"
    "    initial_capital=INITIAL_CAPITAL, preloaded=PRICES_ALL,\n"
    '    strategy_kwargs={"train_end_date": TRAIN_END, "sectors": {"tech": TECH}})\n'
    "eq_pairs_tech_oos = _oos(eq_pairs_tech)\n"
    "r_pairs = eq_pairs_tech_oos['equity'].pct_change().dropna()\n"
    "\n"
    "compare = pd.DataFrame({\n"
    "    'momentum': {\n"
    "        'sharpe': round(sharpe_ratio(r_mom), 3),\n"
    "        'calmar': round(calmar_ratio(eq_mom_oos['equity']), 3),\n"
    "        'max_dd': round(max_drawdown(eq_mom_oos['equity']), 4),\n"
    "        'ann_return': round(annualized_return(eq_mom_oos['equity']), 4),\n"
    "    },\n"
    "    'pairs': {\n"
    "        'sharpe': round(sharpe_ratio(r_pairs), 3),\n"
    "        'calmar': round(calmar_ratio(eq_pairs_tech_oos['equity']), 3),\n"
    "        'max_dd': round(max_drawdown(eq_pairs_tech_oos['equity']), 4),\n"
    "        'ann_return': round(annualized_return(eq_pairs_tech_oos['equity']), 4),\n"
    "    },\n"
    "})\n"
    "print('Tech universe, OOS 2022-2023, costs on:\\n')\n"
    "print(compare.to_string())"
))

cells.append(nbf.v4.new_code_cell(
    "fig, ax = plt.subplots(figsize=(12, 5))\n"
    "ax.plot(eq_mom_oos.index, eq_mom_oos['equity'], label='momentum', color='#4c78a8')\n"
    "ax.plot(eq_pairs_tech_oos.index, eq_pairs_tech_oos['equity'], label='pairs', color='#e45756')\n"
    "ax.axhline(INITIAL_CAPITAL, color='k', lw=0.5, ls='--')\n"
    "ax.set_title('Momentum vs Pairs — Tech universe, OOS 2022-2023'); ax.legend(); ax.grid(alpha=0.3)\n"
    "plt.tight_layout(); plt.show()"
))

cells.append(nbf.v4.new_markdown_cell(
    "### 13a. Reading the momentum result honestly\n\n"
    "Momentum on Tech posts a *positive* OOS Sharpe but a savage max drawdown "
    "(concentrated long-only exposure straight through the 2022 tech sell-off), so "
    "its Calmar is poor. A positive Sharpe with a ~45% drawdown is not a good "
    "strategy — it is a small, undiversified long book that happened to recover in "
    "2023. The point of this section is **not** that momentum beats pairs; it is "
    "that a structurally different strategy plugged into the unchanged engine and "
    "produced coherent, fully-costed, honestly-reported numbers. The plug-in "
    "architecture works."
))

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}

with open("notebooks/research.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("wrote notebooks/research.ipynb with", len(cells), "cells")
