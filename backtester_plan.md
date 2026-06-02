# Backtester — Project Plan
### Event-Driven Backtesting Engine + Statistical-Arbitrage Strategy (Python)
*Your finance/quant standout. Slots into the finance resume at #2 (Vol Engine → Backtester → NSE Screener), replacing the Deterministic Algo Debugger. Build it on top of your Volatility Surface Engine for narrative continuity.*

---

## Context — why this project, and what actually separates it from the pile

Almost every applicant has "a trading strategy backtest." Almost all of them are worthless to a quant screener, for one reason: they're built with look-ahead bias and report a fantasy return (the classic moving-average crossover showing 300%). A quant researcher discards those on sight.

Your entire edge here is **methodological rigor**, not flashy returns. The project's selling point — the sentence that gets you the interview and wins it — is: *"here's how I structurally prevented look-ahead bias, modeled real costs, and validated out-of-sample."* A project that honestly reports a modest Sharpe of 1.2 net of costs beats one claiming a Sharpe of 4, because the screener trusts the first and distrusts the second.

**Why pairs / statistical arbitrage:** it's a named, respected portfolio project, the math is tractable, and it demonstrates real statistical thinking (cointegration, mean reversion) rather than curve-fitting. **Why event-driven:** a vectorized backtest is easy to write and easy to cheat with — it's trivial to accidentally use future data. An event-driven loop processes one timestamp at a time and *physically cannot see the future*, which both removes look-ahead bias and mirrors how live trading works. Building it event-driven is itself the signal.

---

## What you're building

A reusable event-driven backtesting engine (the infrastructure) plus one rigorously-validated strategy running on it (the research). The engine is the part you can reuse and show as systems work; the strategy is the part that shows you can think like a quant.

---

## Architecture (classic event-driven loop)

```
   ┌──────────────┐  MarketEvent   ┌────────────┐
   │ Data Handler │ ──────────────►│  Strategy  │
   │ (point-in-   │                └─────┬──────┘
   │  time bars)  │                 SignalEvent
   └──────┬───────┘                      ▼
          ▲                         ┌────────────┐
          │                         │ Portfolio  │  sizing, risk,
       FillEvent                    └─────┬──────┘  positions, equity
          │                          OrderEvent
   ┌──────┴───────┐                       ▼
   │  Execution   │◄──────────────── (event queue)
   │  Handler     │  applies costs + slippage,
   │ (sim broker) │  fills at NEXT bar ── FillEvent ──► back to Portfolio
   └──────────────┘
```

**The event queue** processes one event at a time: `MarketEvent → SignalEvent → OrderEvent → FillEvent`. Each loop tick advances time by exactly one bar.

**Components:**
- **Data handler** — serves bars **point-in-time**: the strategy only ever sees data up to and including the current bar. This is where look-ahead bias is *structurally* prevented, not just avoided by discipline.
- **Strategy** — consumes market events, emits signals. Knows nothing about cash or position sizing. (Swappable, so the engine is reusable.)
- **Portfolio** — turns signals into sized orders; tracks positions, cash, and the equity curve; enforces risk limits.
- **Execution handler (sim broker)** — turns orders into fills with **realistic transaction costs and slippage**, and **fills at the next bar's open, not the current close** (another structural look-ahead guard).

---

## The strategy — cointegration pairs trade

1. **Universe & screening.** Pick a liquid universe (NIFTY constituents, or US large-caps via free data). Identify candidate pairs (same sector is a reasonable prior).
2. **Cointegration test.** Use Engle–Granger or Johansen (`statsmodels`) to find pairs with a stationary spread. **Run this on in-sample data only** — finding pairs on the full history is itself a look-ahead leak.
3. **Signal.** Trade the z-score of the spread: enter when it diverges beyond a threshold (short the rich leg, long the cheap leg), exit on mean reversion toward zero.
4. **Risk.** Position sizing, a stop-out when cointegration breaks down, and a max-exposure cap.

**Alternative that compounds with your Vol Engine:** a **volatility-risk-premium** strategy — systematically harvesting the gap between implied and realized vol, using your existing vol surface as the implied-vol source. Choosing this makes your two finance projects one continuous body of work, which is a strong story. Pairs is the safer first build; vol-risk-premium is the higher-continuity option.

---

## Validation methodology — report ALL of this explicitly

This section *is* the project. Each item below is a line in your README and a sentence in your interview.

- **In-sample / out-of-sample split.** Discover pairs and tune thresholds on in-sample data only; report out-of-sample results separately.
- **Walk-forward analysis.** Roll the train/test window forward through time and report each window. This is the single most credibility-building thing you can show.
- **Transaction costs + slippage, always on.** Show results with and without costs to prove you understand they matter (many "winning" strategies die once costs are included).
- **Look-ahead prevention, stated.** Explain that the event-driven design + next-bar fills make future data structurally inaccessible.
- **Honest metrics.** Annualized return, **Sharpe, Sortino, max drawdown, turnover, hit rate, profit factor.** Report a believable Sharpe (roughly 0.8–1.5 out-of-sample). A reported Sharpe of 4 is a red flag, not a brag.
- **Overfitting / multiple-testing awareness.** Note that testing many pairs inflates false positives, and how you guard against it (out-of-sample confirmation, limited parameter search, not cherry-picking the best pair).

---

## Build sequence (~4–6 weeks, part-time)

**Week 1 — Engine core.** Event queue, event classes, data handler serving point-in-time bars. Prove the loop runs correctly on a trivial buy-and-hold.

**Week 2 — Portfolio + execution.** Position/cash tracking, equity curve, commission + slippage models, next-bar fills.

**Week 3 — Strategy.** Implement cointegration pair discovery (in-sample) + the z-score signal logic. Wire it into the engine.

**Week 4 — Validation.** Out-of-sample test, walk-forward harness, the full metrics suite, and a tearsheet (equity curve, drawdown chart, rolling Sharpe).

**Weeks 5–6 (polish).** Sensitivity analysis (how robust are results to threshold/parameter changes?). A README that *leads with methodology*. A reproducible notebook so anyone can rerun it.

---

## README (half the value — do not skip)

Structure it as a story, the way quant practitioners expect:
**What it is → the strategy thesis (why cointegration mean-reverts) → the math/statistical model (cointegration test, z-score) → the engine design (why event-driven prevents look-ahead) → results with the tearsheet → honest limitations (survivorship bias if present, regime dependence, capacity).**
Acknowledging limitations *raises* credibility — it signals you understand the difference between a backtest and a live edge.

---

## Resume line + interview defense

**Resume (Finance):**
> Built an event-driven backtesting engine in Python (point-in-time data, next-bar fills, realistic costs/slippage) and a cointegration-based statistical-arbitrage strategy; validated via walk-forward out-of-sample analysis, achieving an out-of-sample Sharpe of ~X net of costs.

**Be ready to defend:** how event-driven structurally prevents look-ahead; why you fill at the next bar; what cointegration means and why a cointegrated spread can still break down; how you avoided overfitting across many candidate pairs; why your Sharpe is believable; what you'd add for live trading (real fills, latency, regime detection, capacity limits).

---

## Data & tooling
- **Stack:** Python, pandas/NumPy, `statsmodels` (cointegration), matplotlib or Plotly (tearsheet).
- **Data:** free EOD/daily data is fine (yfinance, or NSE data you already work with from the screener). Daily bars keep look-ahead reasoning clean; you don't need intraday.
- **Build the engine yourself.** Studying `backtrader`/`zipline` for design ideas is fine, but using them as your engine defeats the purpose — the engine *is* the systems signal.

---

## Stretch goals (only after honest results exist)
- Extend to a basket / multiple simultaneous pairs with portfolio-level risk.
- Add a simple regime filter (only trade pairs currently passing a rolling cointegration test).
- Swap in the vol-risk-premium strategy to link this project to your Vol Engine.
