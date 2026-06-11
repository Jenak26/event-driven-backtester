# Event-Driven Backtesting Engine

A reusable event-driven backtesting engine in Python with a cointegration-based
statistical-arbitrage strategy, validated via walk-forward out-of-sample analysis.

---

## What it is

A two-part project:

1. **The engine** (`backtester/`) — a reusable event-driven backtest loop.
   New strategies plug in by subclassing `BaseStrategy`. The engine itself is
   strategy-agnostic.

2. **The strategy** (`strategies/pairs.py`) — a pairs trade on cointegrated
   S&P 500 stocks, discovered in-sample and validated out-of-sample.

---

## Strategy thesis

Two stocks in the same sector that share a common fundamental driver tend to move
together over time. When their price ratio diverges beyond a statistical threshold,
mean reversion is likely: the spread is expected to return to its long-run equilibrium.
We trade this divergence by going long the cheap leg and short the expensive leg,
profiting from convergence.

The key mathematical condition is **cointegration**: a stationary linear combination
of two I(1) price series. We test this with the Engle-Granger procedure on in-sample
data only.

---

## Engine design — why event-driven prevents look-ahead

A vectorized backtest operates over arrays of data that exist all at once in memory.
It is trivially easy to introduce look-ahead bias — using tomorrow's price today —
simply by indexing one row forward.

An event-driven loop processes one bar at a time:

```
while not exhausted:
    data.update_bars()        # advances cursor by ONE bar; emits MarketEvent
    strategy.calculate_signals()  # can only call get_latest_bars() — never future bars
    portfolio.size_orders()
    broker.fill_at_next_open()    # fills at the NEXT bar's open, not the current close
```

Two structural guards eliminate look-ahead:

1. `DataHandler.get_latest_bars(symbol, N)` returns only rows `[cursor-N : cursor]`.
   Requesting 100 bars when only 5 have elapsed returns 5 rows. Future data does not
   exist in the slice.

2. **Next-bar fills.** The simulated broker (`SimBroker`) fills every order at the
   *following* bar's open. You cannot trade on a price you just observed.

All prices use `yfinance auto_adjust=True`: split- and dividend-adjusted throughout.
Unadjusted prices introduce artificial jumps in the spread on corporate action dates,
manufacturing false cointegration signals.

---

## Statistical methodology

### Pair discovery (point-in-time, two-stage)

- Universe: ~25 liquid S&P 500 names from Tech and Financials sectors.
- Only **within-sector** pairs are tested (~144 combinations), limiting the
  multiple-testing universe.
- All statistics run on **log prices** (the literature standard — log spreads
  are scale-free and stabler across price levels).
- **Two-stage screen:**
  1. Engle-Granger at p < 0.05 on the training window *minus its final
     3 months*.
  2. **Held-out requalification:** survivors must also pass Engle-Granger at
     p < 0.10 on those final 3 months, which the screen never saw. Pairs that
     cannot replicate their cointegration on unseen data are discarded.
- **Why not Bonferroni?** We tried it first: with 144 tests the threshold is
  p < 0.000347, and on 2019–2021 data (which contains the COVID structural
  break) *zero* pairs pass — the strategy never trades. Bonferroni assumes
  independent tests, but pair tests share symbols and are strongly dependent,
  so it over-corrects. The held-out stage is the multiple-testing control
  instead: the joint false-positive probability per pair is roughly
  0.05 × 0.10 = 0.005, i.e. ~0.7 expected false positives across 144 tests —
  stricter in expectation than the naive p < 0.05 screen (~7 false positives).
- **Discovery itself is point-in-time.** The screen runs *inside* the event
  loop on the bar the training window closes, reads bars only through the
  engine's `get_latest_bars` API, and therefore structurally cannot touch
  post-training data. The requalification window is the tail of the training
  window — not future data.

### Signal and sizing

- Spread: `S_t = log(Price_A_t) - β · log(Price_B_t)`, where β is the OLS
  hedge ratio in log space.
- Rolling 60-day z-score: `z_t = (S_t - μ_{60}) / σ_{60}`.
- Entry when |z| > 2.0; exit when |z| < 0.5.
- Stop-out when rolling 60-day cointegration p-value > 0.1 (cointegration
  has broken down; the trade thesis no longer holds).
- **Hedge-ratio sizing:** with `q_A = C / P_A` shares of A, the dollar-neutral
  hedge for a log spread is `q_B = C · β / P_B` shares of B. Equal-dollar
  sizing is incorrect when β ≠ 1 — it leaves residual directional exposure.
- **Exits unwind exact entry quantities** rather than flattening the symbol,
  so pairs that share a leg (e.g. META appearing in two pairs) cannot corrupt
  each other's positions.

### Transaction costs

- Commission: $0.001/share (interactive-brokers style).
- Slippage: fill price = next open ± 5 basis points.
- All reported results include these costs. The notebook also shows results
  without costs to quantify cost drag.

---

## Results

Pairs were discovered on 2019–2021 training data; everything below is measured
on the 2022–2023 holdout the discovery step never saw. Full tearsheet and
walk-forward table in `notebooks/research.ipynb`.

Discovered pairs: META/AVGO, META/TXN, BAC/PNC, AXP/PNC.

| Metric (out-of-sample) | Value |
|---|---|
| Sharpe (with costs) | **-1.25** |
| Sharpe (without costs) | -0.02 |
| Cost drag | 1.23 Sharpe points |
| Sortino (with costs) | -0.84 |
| Max drawdown | -5.7% |
| Annualized return | -2.6% |
| Walk-forward windows | 15 (9 traded; mean Sharpe across traded windows 0.08) |

**The result is negative, and we report it anyway — that is the point.**
Gross of costs the strategy is flat; commission and slippage make it clearly
unviable. The stop-out rule (exit when rolling cointegration breaks) limits
losses from broken pairs but generates churn that the cost model punishes.
The sensitivity grid is smooth and uniformly negative, so this is not an
artifact of one threshold choice — and no parameter was tuned to flatter the
headline. A pipeline that can only ever produce good-looking numbers is a
curve-fitting machine, not a backtest.

---

## Honest limitations

1. **Survivorship bias.** The universe is drawn from today's S&P 500 constituents.
   Backtesting 2019–2023 on this list means we implicitly exclude companies that
   were delisted or removed during that period — an upward bias of unknown magnitude.
   Mitigating this properly requires a point-in-time constituent file.

2. **Regime dependence.** Pairs strategies perform best in range-bound markets and
   tend to bleed in sustained trending regimes (e.g., COVID crash, 2022 rate-hike
   sell-off). The walk-forward table shows performance by window.

3. **Capacity.** The strategy trades liquid large-caps at modest size. Real
   institutional capacity would be limited by market impact, which this model
   does not simulate.

---

## Reproducing the results

```bash
pip install -r requirements.txt
jupyter notebook notebooks/research.ipynb
# Kernel → Restart & Run All
```

---

## Running the tests

```bash
pytest tests/ -v
```
