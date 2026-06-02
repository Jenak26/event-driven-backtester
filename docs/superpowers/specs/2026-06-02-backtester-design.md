# Backtester Design Spec
**Date:** 2026-06-02
**Strategy:** Cointegration Pairs / Statistical Arbitrage
**Data:** yfinance (US large-caps, daily adjusted OHLCV)
**Structure:** Python package + research notebook

---

## 1. Project Structure

```
backtester/
├── backtester/
│   ├── __init__.py
│   ├── events.py          # MarketEvent, SignalEvent, OrderEvent, FillEvent
│   ├── queue.py           # EventQueue wrapper
│   ├── data.py            # DataHandler (yfinance auto_adjust=True, point-in-time)
│   ├── strategy.py        # BaseStrategy ABC + BuyAndHoldStrategy (dummy)
│   ├── portfolio.py       # Portfolio (positions, cash, equity curve)
│   ├── execution.py       # SimBroker (next-bar fills, commission, slippage)
│   └── metrics.py         # Sharpe, Sortino, max drawdown, WalkForwardRunner
├── strategies/
│   └── pairs.py           # CointegrationPairsStrategy
├── notebooks/
│   └── research.ipynb     # Pair discovery, walk-forward, tearsheet
├── run.py                 # Entry point: wires engine + strategy, runs backtest loop
├── tests/
│   ├── test_events.py
│   ├── test_data.py
│   ├── test_portfolio.py
│   └── test_execution.py
├── requirements.txt
└── README.md
```

The `backtester/` package is the reusable engine. `strategies/` holds domain-specific strategies — swappable, not part of the engine. The notebook is for research and tearsheet output only, not engine logic.

---

## 2. Architecture — Event-Driven Loop

```
   ┌──────────────┐  MarketEvent   ┌────────────┐
   │ Data Handler │ ──────────────►│  Strategy  │
   │ (point-in-   │                └─────┬──────┘
   │  time bars)  │                 SignalEvent
   └──────┬───────┘                      ▼
          ▲                         ┌────────────┐
          │                         │ Portfolio  │
       FillEvent                    └─────┬──────┘
          │                          OrderEvent
   ┌──────┴───────┐                       ▼
   │  SimBroker   │◄──────────────── (event queue)
   │ (next-bar    │
   │  fills)      │──────────────── FillEvent ──► Portfolio
   └──────────────┘
```

Event sequence per bar: `MarketEvent → SignalEvent → OrderEvent → FillEvent`. Each tick advances by exactly one bar.

---

## 3. The Five Phases

### Phase 1 — Event Loop + Data Handler (~1 week)

**Goal:** Prove the loop runs correctly with no look-ahead.

**Deliverables:**
- `events.py`: `MarketEvent`, `SignalEvent`, `OrderEvent`, `FillEvent` dataclasses
- `queue.py`: `EventQueue` (thin wrapper over `collections.deque`)
- `data.py`: `DataHandler` — fetches daily adjusted OHLCV from yfinance (`auto_adjust=True`), serves bars point-in-time via `get_latest_bars(symbol, N)`
- `strategy.py`: `BaseStrategy` ABC + `BuyAndHoldStrategy` (always long, never exits)
- `run.py`: minimal backtest loop wiring all components together

**Acceptance gate:** Run the loop over 10 bars of SPY data. Assert that on bar 5, `get_latest_bars('SPY', 5)` returns exactly 5 bars ending at bar 5 — and `get_latest_bars('SPY', 6)` also returns only 5 bars (not 6, because there are only 5 bars so far). Assert that exactly 10 `MarketEvent`s were emitted over the full run. The data handler must never serve a bar beyond the current cursor. No equity curve yet — that's Phase 2.

---

### Phase 2 — Portfolio + Execution Handler (~1 week)

**Goal:** Prove cash/position accounting and cost models are correct before any real strategy runs on them.

**Deliverables:**
- `portfolio.py`: `Portfolio` — tracks cash, positions (long/short), and equity curve. Turns `SignalEvent` → `OrderEvent`. Enforces max-exposure cap.
- `execution.py`: `SimBroker` — fills `OrderEvent` at next bar's open. Applies commission and slippage. Emits `FillEvent` back to portfolio.
- Commission model: `$0.001` per share (configurable)
- Slippage model: fill price = next open + N basis points (default 5 bps, configurable)

**Acceptance gate:** Run buy-and-hold on SPY for 2020–2023 using adjusted prices. The equity curve must match SPY's adjusted price movement within rounding error. Then manually verify 5 trades by hand — compute expected cash and position values from the raw adjusted data and assert the portfolio matches.

---

### Phase 3 — Pairs Strategy (~1–1.5 weeks)

**Goal:** Implement the cointegration pairs strategy wired into the engine.

**Deliverables:**
- `strategies/pairs.py`: `CointegrationPairsStrategy`
  - Universe: 20–30 liquid S&P 500 names from 2 sectors (default: ~12 Tech + ~13 Financials). Hardcoded list, easily swappable. Test only **within-sector pairs** to limit the multiple-testing universe to ~144 tests.
  - Pair discovery: Engle-Granger cointegration test on `data[:train_end_date]` only. Bonferroni-adjusted threshold: `p < 0.05 / N_tests` where `N_tests` = number of within-sector pairs tested. Any pair passing this threshold must also re-qualify on a held-out OOS window at `p < 0.05` before being traded.
  - Hedge ratio: OLS regression `Price_A = α + β·Price_B + ε` on in-sample data. Spread = `Price_A - β·Price_B`.
  - Rolling z-score: `z_t = (spread_t − μ(spread, 60d)) / σ(spread, 60d)`
  - Signal: enter long A / short B when z < −2.0; enter short A / long B when z > +2.0; exit when |z| < 0.5
  - Sizing: long 1 unit of A, short β units of B (hedge-ratio neutral, not equal-dollar)
  - Stop-out: exit when rolling 60-day cointegration p-value > 0.1
  - Constructor requires `train_end_date` — cannot accidentally use full history

**Acceptance gate:** Run on in-sample data, plot spread + z-score + entry/exit markers for one pair, visually confirm signals fire at correct z-score thresholds and hedge ratio is non-trivial (β ≠ 1.0).

---

### Phase 4 — Validation + Metrics (~1 week)

**Goal:** Produce honest, credible out-of-sample results.

**Deliverables:**
- In-sample / out-of-sample split: 70% train / 30% test by calendar time
- `metrics.py`:
  - `WalkForwardRunner(strategy_factory, data, train_months=12, test_months=3)` — rolls the window, re-runs pair discovery (including Bonferroni screening + OOS requalification) on each in-sample slice
  - Metrics: annualized return, Sharpe ratio, Sortino ratio, max drawdown, turnover, hit rate, profit factor
- Tearsheet (matplotlib): equity curve, drawdown chart, rolling 6-month Sharpe
- Results reported both with and without transaction costs

**Acceptance gate:** Out-of-sample Sharpe computed and reported. Walk-forward shows results across at least 4 windows. Both cost scenarios reported.

---

### Phase 5 — Polish + README (~1 week)

**Goal:** Make the project reproducible and self-explanatory to a quant reader.

**Deliverables:**
- Sensitivity analysis: vary entry threshold (1.5 / 2.0 / 2.5σ) and exit threshold (0.0 / 0.5 / 1.0σ), report Sharpe sensitivity table
- `notebooks/research.ipynb`: runs top-to-bottom from a clean kernel, reproduces tearsheet exactly
- `README.md` structure: What it is → Strategy thesis → Math/statistical model → Engine design → Results → Honest limitations
- Limitations section must address: survivorship bias (hardcoded current-day S&P names backfills into 2020–2023), regime dependence, capacity constraints

**Acceptance gate:** Clone repo on a fresh environment, `pip install -r requirements.txt`, run notebook top-to-bottom, exact tearsheet reproduced.

---

## 4. Key Design Decisions

### Adjusted prices throughout
`DataHandler` fetches with `yfinance.download(..., auto_adjust=True)`. All OHLCV columns are split- and dividend-adjusted. This prevents artificial jumps in the spread on corporate action dates, which would manufacture or destroy apparent cointegration in unadjusted data. This is stated explicitly in the README.

### Point-in-time data serving
`DataHandler.get_latest_bars(symbol, N)` returns only bars up to and including the current cursor position — never beyond. Look-ahead is structurally prevented, not avoided by discipline.

### Next-bar fills
Signals emitted at bar `t` fill at bar `t+1`'s open. Buying at the close you just saw is not possible by construction.

### In-sample pair discovery
`CointegrationPairsStrategy` requires `train_end_date` as a constructor argument. The cointegration test only runs on `data[:train_end_date]`. Running on full history is not the default path.

### Multiple-testing guard
With ~144 within-sector pair candidates, the naive p < 0.05 threshold yields ~7 false positives by chance. Two-layer guard:
1. **Bonferroni screen (in-sample):** require `p < 0.05 / N_tests` from Engle-Granger on the in-sample slice.
2. **OOS requalification:** any pair passing the Bonferroni screen must also pass Engle-Granger at `p < 0.05` on a separate held-out OOS window before being traded. Pairs that pass Bonferroni in-sample but fail OOS requalification are discarded.

### Hedge-ratio sizing
Position sizing uses the OLS-estimated hedge ratio β, not equal-dollar legs. Long 1 unit of A, short β units of B. Equal-dollar sizing is incorrect when β ≠ 1.0 — it leaves residual directional exposure.

### Commission + slippage model
- Commission: `$0.001/share`, configurable via `commission_per_share` parameter
- Slippage: fill at next open + `N` bps (default 5), configurable via `slippage_bps` parameter
- `no_costs=True` flag available only for the "costs matter" comparison chart — not the default

### Walk-forward harness
`WalkForwardRunner` re-runs pair discovery (including both multiple-testing layers) on each window's own in-sample slice. Pairs are not discovered once and reused across all future windows.

---

## 5. Dependencies

```
pandas>=2.0
numpy>=1.24
yfinance>=0.2
statsmodels>=0.14      # Engle-Granger, Johansen cointegration
matplotlib>=3.7
scipy>=1.10
jupyter>=1.0
pytest>=7.0
```

---

## 6. Validation Checklist (report all of these)

- [ ] In-sample / out-of-sample split documented with exact dates
- [ ] Walk-forward results across ≥4 windows
- [ ] Results with AND without transaction costs
- [ ] Look-ahead prevention explained (event-driven + next-bar fills)
- [ ] Adjusted prices used throughout — stated explicitly
- [ ] Honest out-of-sample Sharpe (target ~0.8–1.5)
- [ ] Bonferroni screen + OOS requalification both applied and documented
- [ ] Survivorship bias acknowledged (current-day index names backtested into 2020–2023)
- [ ] Sensitivity analysis table included
