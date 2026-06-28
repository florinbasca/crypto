# Pipeline

Run in order. All steps are incremental unless noted (re-runs skip existing data).

## 1. ETL

```bash
uv run python etl/universe.py        # Hyperliquid candidates, Binance-availability probed (~130, no stables)
uv run python etl/prices_raw.py      # 1m Binance spot klines, last 3y (big download)
uv run python etl/prices.py          # resample 1m -> 10min `prices` table
uv run python etl/marketcap.py       # daily mcap via CoinGecko (needs .keys) - size factor
uv run python etl/funding.py         # perp funding rates (optional, funding features)
uv run python etl/futures.py         # OI (optional)
```

## 2. Risk model

```bash
uv run python risk_model/factor_returns.py    # market (EW) + size (SMB) factor returns
uv run python risk_model/residual_returns.py  # causal betas, residuals, fwd targets
```

## 3. Features

```bash
uv run python risk_model/features.py          # ~165-column feature panel (regenerates from scratch)
```

## 4. Research

```bash
uv run python research/signals/ml_signal.py   # optional Class-5 walk-forward GBM predictions
                                              # incremental - extends from last refit
uv run python research/signals/evaluate.py d_res_zscore --no-save
                                              # optional: quick one-signal diagnostic
uv run python research/signals/evaluate.py --fresh
                                              # evaluate all signals x lag grid from scratch
uv run python research/portfolio/walk_forward.py
                                              # walk-forward selection + market-neutral backtest (rank sizing, MVO benchmark)
                                              # (--lockbox to include held-out months: run ONCE)
```

## Develop a new signal idea

Edit the sandbox signal:

```bash
research/signals/my_signal.py
```

Inside that file, change:

- `FEATURE_COLUMNS`: which feature columns the signal reads.
- `PARAMETERS`: direction and expected decay/turnover labels.
- `compute_my_signal(...)`: the actual signal logic.

Run a one-signal diagnostic without writing production research tables:

```bash
uv run python research/signals/evaluate.py my_signal --no-save
```

If the idea looks sane, persist that signal's stats (drop `--no-save` to write
its rows into signal_daily_stats / signal_metrics):

```bash
uv run python research/signals/evaluate.py my_signal
```

For production evaluation after changing signal definitions, rerun:

```bash
uv run python research/signals/evaluate.py --fresh
uv run python research/portfolio/walk_forward.py
```

## Notes

- Order matters: each stage reads tables written by the previous one
  (universe -> prices_raw -> prices + marketcap ->
  risk_factors -> residual_returns -> features -> optional ml_predictions ->
  signal_daily_stats -> wf_portfolio_*).
- If `residual_returns.py` acceptance checks FAIL, stop and debug the factor
  model before running features/signals.
- After changing the universe, refresh price coverage and market cap before
  rebuilding downstream tables. If only prices changed, re-run from
  `prices.py` downward as appropriate.
- `signals/evaluate.py` is incremental per signal; use `--fresh` after any
  change to features, residuals, ML predictions, or signal definitions.
  `--limit N` for a quick test run.
- New experimental signal families can live in `research/signals/`; see
  `research/signals/README.md`.
- The full current Hyperliquid-tradeable candidate universe is used at every
  timestamp. Newer names simply have missing/NaN history before listing.
- The repo intentionally does not implement the MTV crypto-value article idea:
  it requires clean on-chain transaction volume, which is not in the data store.
