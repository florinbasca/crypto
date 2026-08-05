# Crypto Trading System - Agent Guidelines

## Critical Rules

### DO NOT RUN SCRIPTS
Never run ETL, `research/signals/evaluate.py`, `research/portfolio/walk_forward.py`, or other
long-running scripts yourself. Only make code changes. The user runs scripts
manually. (Fast synthetic checks in `tests/` are fine.)

### NO HARDCODED PARAMETERS
All configuration values MUST be in `config.py`. Never hardcode:
- Thresholds (Sharpe, IC, FDR alpha, correlation, etc.)
- Time periods (lookbacks, windows, train/test dates, frequencies)
- Costs (bps, slippage), position limits, leverage
- Any tunable parameter

### TIMING CONVENTIONS (do not break these)
- All bars are BAR-END stamped: r[t] and factor[t] cover (t-1bar, t].
- Features may use data through bar t INCLUSIVE (targets start at t+1).
  Causality is enforced by the truncation test in tests/sanity_checks.py -
  any new feature must pass it. Do NOT add defensive shift(1) lags.
- Forward targets: fwd_res_h[t] = sum of residuals over bars t+1..t+p
  (`rolling(p).sum().shift(-p)`). Never hedge a forward return with
  factor[t..t+p-1] (off-by-one) and never `shift(-1)` the fwd_res columns
  again (double shift) - both were real bugs in the previous pipeline.
- Betas/covariances: estimated from data STRICTLY BEFORE the day they are
  applied. The universe is the full current candidate set at every timestamp.
- residual_returns.py prints acceptance checks (variance reduction,
  residual-raw correlation). If they FAIL, the factor model is broken - fix it
  before anything downstream.

## Project Structure

- `config.py` - central configuration (ALL parameters)
- `dbutil.py` - Parquet/Polars storage interface (per-table Parquet datasets
  under `db/`; tables with a `symbol` column are stored one file per symbol).
  Same public API as before (save_data/load_data/iter_data_batches/…)
- `etl/` - universe (Hyperliquid candidates + point-in-time
  universe_membership spells), prices (1m raw -> 10min panel), marketcap,
  funding, futures
- `risk_model/` - factor_returns (market EW + size/momentum/vol rank-weighted
  spreads), residual_returns (causal betas + multi-horizon targets), features
  (10min feature panel)
- `research/` - lib/spaces (cross-sectional statarb spaces), lib/portfolio_opt
  (Ledoit-Wolf MVO + rank/equal-weight sizing), signals/evaluate (multi-horizon IC + daily aggregates),
  signals/signal_lab (KEEP/KILL scorecard)
- `research/portfolio/` - walk_forward (walk-forward FDR selection +
  market-neutral backtest; rank/equal-weight sizing by default, optional MVO benchmark foil)
- `research/signals/` - plugin-style experimental signal families and templates

## Data Flow

1. universe (HL candidates) -> prices_raw (1m) -> prices (10min)
2. prices + universe + marketcap -> risk_factors (market, size)
3. prices + risk_factors -> factor_loadings (daily betas) + residual_returns
   (single-bar residual + fwd_res_{10min,1h,1d} targets)
4. prices + residual_returns + factor_loadings (+ funding/futures) -> features
5. features + residual_returns -> signal_daily_stats + signal_metrics
6. signal_daily_stats + features + residuals + betas -> walk-forward
   market-neutral portfolio, rank/equal-weight sizing (optional MVO benchmark)
   (wf_portfolio_* tables)

## Important Notes

- Discovery builds its panel PER ROLL (`build_panel(start=, end=)`), not once
  for all history: a whole-history panel is ~12.4 GB and grows with the
  universe (66 tradeable names in 2022, 136 by 2025), which on a 24 GB box
  forces the kernel to compress memory and slows scoring ~15x while still
  reporting ~100% CPU. The windowed build pads by `liquidity_window_bars`
  before and `target_lag_bars` after, then trims - drop either pad and the
  is_liquid flag or the forward target silently changes. tests/
  discovery_panel_window_checks.py asserts the two paths agree.
- Discovery candidates compile against exact AST columns from the roll feature
  store (normally 2-3 columns), and survivor signals are shared-grid float32
  matrices. Do not restore per-candidate long timestamp/symbol DataFrames:
  dozens of survivors duplicate several million keys each and force memory
  compression. Response curves, turnover and correlation consume the matrices
  directly; the public long-frame helpers remain only for compatibility/tests.
- Base frequency 10min (144 bars/day); horizons 10min / 1h / 1d
- Signals are cross-sectional; predictions target forward RESIDUAL returns
- Final backtest PnL uses RAW returns - neutrality constraints do the hedging;
  realized factor exposures are the acceptance check (~0). Net PnL also
  accrues perp funding on held positions at settlement stamps.
- A month can confirm NO signals (every promotion fails the persistence gate).
  That means no new book, not a gap in the record: the carried book is held and
  simulated (`_hold_window`), then unwound after
  `portfolio.max_hold_months_no_promotion` months. Never let an unpromoted month
  silently skip its bars - the equity curve then reads flat across a stretch
  where real positions were live.
- That hold applies ONLY to rolls discovery actually evaluated (present in the
  discovery ledger). The walk-forward truncates its schedule there. A roll that
  never ran has no verdict, so "no promotions" is missing evidence, not a
  confirmed absence of alpha - holding a book through it would report a
  fabricated result. Finish the rolls (`discovery.py --resume`) instead.
- Universe: ~130 Hyperliquid-tradeable candidates, no stablecoins. Membership
  is point-in-time where universe_membership spells exist; pre-snapshot
  history is seeded as member-since-data-start.
- The walk-forward selection speed floor is DERIVED from the execution layer
  (min_holding_lag_bars: 'auto'); do not hand-tune it against the execution
  parameters independently.
- The per-symbol volume-participation cap is the ONLY hard fill constraint in
  the backtest. There is no whole-book turnover cap or turnover budget - do not
  reintroduce one. Trading is priced (cost_bps, GP trade rate), not rationed.
