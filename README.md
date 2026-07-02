# Crypto Stat-Arb: Market-Neutral Residual Prediction

Cross-sectional stat-arb on ~100 Hyperliquid-tradeable crypto names.
1-minute Binance spot data (last 3 years) -> 10-minute base panel ->
market+size+momentum+vol factor model -> multi-horizon residual prediction
(10min / 1h / 1d) -> rank / equal-weight sizing (Ledoit-Wolf MVO kept as a
benchmark), dollar / market / size / momentum / vol-beta neutral.

> **Data and keys are not included.** A fresh clone has no data — the `db/`
> directory is gitignored and built locally by the ETL. The ETL downloads
> ~3 years of  1-minute Binance klines and pulls market-cap history from
> CoinGecko, which needs a free API key in an (untracked) `.keys` file at the
> repo root: `{"coingecko_api_key": "CG-..."}` (see `etl/marketcap.py`).

## Pipeline (run in order)

```bash
# 1. ETL
uv run python etl/universe.py        # Hyperliquid perp candidates (~130, no stables)
uv run python etl/prices_raw.py      # 1m Binance spot klines, last 3y (~big download)
uv run python etl/prices.py          # resample 1m -> 10min `prices` table
uv run python etl/marketcap.py       # daily mcap (CoinGecko, needs .keys) - size factor
uv run python etl/funding.py         # perp funding rates (funding features + backtest funding accrual)
uv run python etl/futures.py         # OI / positioning metrics (optional)

# 2. Risk model
uv run python risk_model/factor_returns.py    # market (EW) + size/momentum/vol rank-weighted factor returns
uv run python risk_model/residual_returns.py  # causal betas, residuals, fwd targets
                                              # (prints acceptance checks - must PASS)
uv run python risk_model/features.py          # ~170-column feature panel
                                              # (--spaces-only: persist only space-referenced columns)

# 3. Signals
uv run python research/signals/evaluate.py   # score the spaces (research/lib/spaces.py) x lag grid
                                              # one signal only: evaluate.py NAME [--no-save]

# 4. Backtest
uv run python research/portfolio/walk_forward.py    # walk-forward selection + MVO backtest
```

Each stage reads the tables the previous one writes, so run in order. If
`residual_returns.py` acceptance checks FAIL, stop and debug the factor model
before building features or signals. After changing the universe, refresh prices
and market-cap before rebuilding downstream. `evaluate.py` is incremental per
signal — use `--fresh` after changing features, residuals, or signal definitions
(`--limit N` for a quick run).

Once the pipeline has run, open `notebooks/portfolio.ipynb` for the full
walk-forward backtest summary and risk diagnostics — equity/drawdown, per-window
and per-signal attribution, cost/exposure, factor-neutrality, and the rank-vs-MVO
comparison — all read from the persisted `wf_portfolio_*` tables.

## Architecture

- **Universe**: candidates = the live Hyperliquid perp listing (tradability
  filter), mapped to Binance spot symbols, stablecoins/pegged assets excluded,
  capped at 130 by HL volume. The full current candidate set is used at every
  bar; names without data at a given bar are simply NaN.
- **Factor model** — market + size + momentum + vol, all tradable portfolio
  returns in return units. The three characteristic factors share one
  rank-weighting machinery: daily weights proportional to the centered rank of a
  per-name characteristic, long one tail / short the other, each side scaled to
  gross 1 (every member, no tercile-boundary churn; dollar-neutral). The
  characteristic is always computed from data STRICTLY BEFORE the bar's date.
  - *Market*: equal-weight mean member return per 10-min bar.
  - *Size*: small-minus-big on lagged market-cap rank (long smalls).
  - *Momentum*: winners-minus-losers on the trailing cumulative return
    (`momentum.lookback_days`, skipping the most-recent `skip_days`; long winners).
  - *Vol*: low-minus-high on trailing realized volatility (`vol.lookback_days`;
    long low-vol). For momentum/vol the sign is arbitrary — the optimizer
    constrains beta to each factor near zero rather than trading it as standalone
    alpha (same convention as size).
  - *Betas*: daily-refreshed exponentially-weighted OLS (30d window, 10d
    half-life) over all factors, estimated on strictly-past bars.
  - *Fit*: residual_returns.py prints the acceptance checks (variance reduction,
    residual-raw corr) — adding momentum/vol should raise variance reduction over
    the prior two-factor fit (market+size baseline: mean R² ~0.61 at 10min,
    var(residual)/var(raw) ~0.44). Re-measure after rebuilding; watch factor
    collinearity (VIF) as factors are added.
- **Targets**: `fwd_res_{10min,1h,1d}[t]` = sum of single-bar residuals over
  bars t+1..t+p. Bar-end timestamps throughout; a signal at t may use data
  through bar t (forward targets start at t+1, so no overlap).
- **Signals**: ~130 curated cross-sectional **spaces** (`research/lib/spaces.py`),
  each one named economic hypothesis (residual-reversion, liquidity, order-flow,
  funding, OI, positioning, vol-structure, …). Each is scored against the 14-lag
  forward grid on an hourly screening grid; only compact aggregates are persisted
  (`signal_daily_stats`, `signal_metrics`) — raw per-bar values are recomputed on
  demand. See **Signals: generation & selection** below.
- **Portfolio**:
  - *Alpha*: Grinold — `(1 − ic_shrink) × IC × residual-vol × z / √horizon`,
    summed across horizons. The IC is shrunk toward 0 (noisy/non-stationary
    realized IC) and a no-trade zone zeros any name whose edge over its holding
    horizon can't clear a round-trip cost ("lazy trading").
  - *Risk*: Ledoit-Wolf shrunk residual covariance (daily refresh) plus a soft
    cluster-exposure penalty (clusters from trailing residual correlations; the
    Marchenko-Pastur diagnostic shows stable super-noise structure — e.g. a
    meme-coin factor — beyond market+size).
  - *Size* (`portfolio.weight_scheme`): default `equal_weight` ranks alpha
    cross-sectionally (covariance-free); `mvo` uses the shrunk-covariance
    optimizer above and runs as a monitored benchmark. The walk-forward
    EW-vs-MVO comparison found the covariance weighting net-destructive on this
    low-breadth, negatively-skewed book, so rank sizing is the default. Both
    impose exact dollar / market / size / momentum / vol-beta neutrality, 5% position cap,
    gross leverage 1, and trade partially toward the aim portfolio at a
    cost-responsive Garleanu-Pedersen rate; liquidity-aware per-name multipliers
    (trailing ADV) make illiquid names cost more and fill slower.
  - *Test*: backtested on RAW forward returns, net of trading costs AND perp
    funding accrued on held positions at settlement stamps; realized factor
    exposures are reported as the market-neutrality check, alongside a
    half-alpha diagnostic (`cost / expected gross alpha`, ~0.5 at the optimum).

## Signals: generation & selection

**Generation** (`research/signals/evaluate.py`):

- The signal universe is the curated **spaces** in `research/lib/spaces.py` —
  ~130 cross-sectional hypotheses across 14 economic themes (residual-reversion,
  funding, market-structure, open-interest, order-flow, factor-loading, liquidity,
  cross-sectional, vol-structure, momentum, efficiency, fundamental, positioning,
  volume). Each space is one
  vectorized expression over feature columns; add one = add one `_S(...)` line.
- Each signal is computed at full 10-min resolution, smoothed, and
  cross-sectionally z-scored, then scored by **rank IC** against the forward
  residual targets at every lag in the 14-lag grid
  `signals.decay_lag_grid` (1..432 bars, ~10min to 3d), on a non-overlapping
  hourly screening grid (overlap would inflate IC t-stats).
- Only compact per-day aggregates (`signal_daily_stats`) and whole-period
  diagnostics (`signal_metrics`) are stored — the raw panels are recomputed on
  demand. Incremental: re-running only re-evaluates new/changed signals.

**Selection** (per walk-forward window, train data only — `research/portfolio/walk_forward.py`):

1. **Best lag per space** — pick the strongest forward lag by HAC IC t-stat,
   Bonferroni-adjusted for searching the grid.
2. **Benjamini-Hochberg FDR** (loose) across the spaces — sweep out the
   clearly-spurious tail.
3. **Threshold gates** — IC floor, ICIR, daily-return Sharpe, turnover, IC
   stability across window thirds, recent-third sign, liquid-half IC, minimum
   holding lag. The holding-lag floor is *derived from the execution layer*
   (`min_holding_lag_bars: 'auto'`): a lag must retain at least
   `min_monetizable_alpha_fraction` of its alpha after the Garleanu-Pedersen
   aim discount at the book's effective fill rate, so the selector cannot
   spend slots on signals the (turnover-budgeted) executor then scales to ~0.
4. **Standardized composite ranking**, then a **per-theme cap** + **greedy
   de-correlation** on daily returns, bucketed by holding lag.

The selected spaces are recomputed at full resolution on the test window,
**covariance-aware combined** into per-horizon composites, and handed to the MVO.

## Adding your own signal

Add one `_S(...)` line to `SPACES` in `research/lib/spaces.py` (name, feature
column, theme, rationale), then:

```bash
# 1. Dev loop — compute + print IC/decay diagnostics, no table writes:
uv run python research/signals/evaluate.py space_<name> --no-save

# 2. KEEP/KILL scorecard (significance + orthogonality to the live book):
uv run python research/signals/signal_lab.py space_<name>

# 3. Persist its stats, then it enters selection on the next walk-forward run:
uv run python research/signals/evaluate.py
uv run python research/portfolio/walk_forward.py
```

Browse existing spaces with `evaluate.py --list [--category X] [--contains Y]`.
See `research/signals/README.md`.


## Using the agentic signal generator

An LLM-in-the-loop discovery engine (`research/signals/agent/`, design in
`agent.md` there): an evolutionary search proposes signal programs in a
bounded DSL over the feature panel, a deterministic harness compiles,
causality-checks and scores them against forward **residual** returns on a
train/select/OOS walk-forward, and survivors that pass the promotion gates
(BY-FDR, deflation for search overfit, N-consecutive-roll persistence,
orthogonality to the book) are traded through each OOS month as a
dollar+factor-neutral portfolio. The LLM only ever sees compressed
diagnostics and emits DSL JSON — evaluation, windows and promotion are fixed
code it cannot touch.

### What a signal is

A candidate signal is a small **program** in a bounded DSL: an arithmetic
expression over whitelisted feature columns, optionally gated by conditions —
e.g. *"annualized funding divided by its rolling vol, z-scored, active only
when funding just moved"*:

```json
{"expression": ["cs_zscore", ["div", ["col", "fr_annualized"],
                              ["roll_std", ["col", "fr_rate_zscore"], 36]]],
 "conditions": [["abs_gt", ["col", "fr_rate_change"], 0.001]]}
```

The program compiles to a value per (10-min bar, coin), which is then
demeaned + z-scored **across the ~130-coin cross-section** at every bar and
clipped to ±3. A signal is never "BTC will go up" — it is a *ranking of
coins against each other* at each moment (long the top, short the bottom,
~0 net in signal space). It is judged against the **forward residual return
over the next 36 bars (6h)** — market/size/momentum/vol moves stripped out —
so it must predict coin-specific mispricing, not beta. The traded direction
is fixed on TRAIN, never on the scoring window.

### The walk-forward windows

Each **roll** = TRAIN 5mo (fit + diagnostics) → SELECT 1mo (the ONLY window
the search reward sees) → OOS 1mo (traded by the promoted book; the search
never sees it), advancing one month at a time from `discovery.start_date`
(2023-08) until OOS reaches `discovery.end_date` — e.g. roll 0 trains
Aug–Dec 2023, selects on Jan 2024, trades Feb 2024. Purge+embargo bars are
dropped at every boundary so forward targets cannot leak. Stitching all OOS
months end-to-end gives the system's equity curve.

### How the search selects ("survivors")

Within a roll, the search runs a fixed number of generations. Each
generation: the LLM proposes a batch of new programs (per feature family,
allocated by a UCB bandit) → each is compiled, causality-checked and scored
into one **reward** (SELECT-month IC t-stat + cost-aware Sharpe, minus
penalties for turnover, complexity, train/select instability, similarity) →
the population is then **culled to the top `search.survivors` (12)**, with a
diversity rule that skips near-duplicates (signal corr > 0.8). The 12 that
remain are the *survivors* — the parents the LLM mutates next generation.
The next roll re-tests them from scratch on its own windows; surviving twice
in a row is what the `min_rolls_survived` promotion gate measures. Hierarchy
of trust: tried → survivor (one roll, could be luck) → persistent survivor
(two fresh select months) → **promoted** (persistent + FDR/deflation
significance + uncorrelated with the book) — only promoted signals ever
trade OOS.

### Run discovery

```bash
uv run research/signals/agent/run_discovery.py --max-rolls 2   # test the system first
uv run research/signals/agent/run_discovery.py                 # full history
```

**LLM & cost**: `gemini-2.5-flash` (change in config `discovery.llm`). API
key goes in the gitignored `.env` at the repo root: `LLM_KEY=...`. Cost is
roughly **$0.05 per roll ≈ $1.50 for the full history**; actuals are tracked
per run in `discovery_llm_usage`.

Defaults: proposer = the config LLM (`discovery.llm.provider`, currently
gemini) and a **fresh start** (the discovery tables are cleared first). Flags:

- `--proposer random|llm|anthropic|gemini` — `random` is the no-API baseline
  / control experiment (the LLM must beat it to be earning its cost)
- `--ml-probe` — also fit a gradient-boosting ceiling per roll: how much
  predictability the feature set contains at all
- `--no-fresh` — keep existing discovery tables; `--no-save` — dry run
- Each roll's search is seeded with the previous roll's survivors, which
  re-earn survival on the new windows — that is what makes the
  `min_rolls_survived` promotion gate measurable.

All knobs live in `config.py` under `discovery.*` (families/input space, DSL
bounds, search budget, reward weights, promotion gates, backtest, LLM
provider/model/prices).

### Review what was generated

```bash
uv run research/signals/agent/inspect_discovery.py                  # summary
uv run research/signals/agent/inspect_discovery.py --top 20 --survivors-only
uv run research/signals/agent/inspect_discovery.py --expressions --curve
```

Prints the per-roll summary, top candidates with their DSL programs and
select-window stats, the promoted book, the stitched OOS equity curve, and
LLM token usage/cost. The underlying tables (readable with
`dbutil.load_data`): `discovery_ledger` (every candidate ever evaluated —
the audit trail), `discovery_promotions`, `discovery_oos_returns` (daily OOS
PnL), `discovery_llm_usage`.

Synthetic end-to-end checks (planted signal found, look-ahead caught, noise
promotes nothing, reproducibility): `uv run python tests/discovery_checks.py`.


## Limitations

How far the backtested numbers should be trusted out of sample:

- **Partially point-in-time universe.** Candidates are *today's* Hyperliquid
  listing, so history is conditioned on names that survived to now
  (survivorship / tradability bias); historical listing/delisting dates aren't
  available. Each `etl/universe.py` run now records membership spells in
  `universe_membership` (symbol, valid_from, valid_to), so membership becomes
  genuinely point-in-time as snapshots accrue — but everything before the
  first snapshot is still seeded as "member since the data start".
  `walk_forward.min_listing_age_days` (off by default) excludes names newly
  listed at each test day, bounding how much PnL depends on them.
- **Market data only.** Inputs are crypto prices plus market-cap, funding, and
  open-interest — no fundamentals, on-chain, order-book/L2, sentiment/news, or
  macro data. The entire edge is price-derived.
- **Approximate transaction costs.** A 2 bps/side base cost is charged on
  turnover, scaled per name by a trailing-ADV liquidity multiplier (illiquid
  names cost more, fill slower), and perp funding on held positions is accrued
  at settlement stamps (Binance USDT-perp rates as a Hyperliquid proxy; longs
  pay positive rates). The model still does *not* use realized execution data,
  calibrated impact curves, or capacity caps, and assumes fills at 10-min bar
  stamps. Real-world costs are almost certainly higher.
- **Single short sample / regime.** ~3 years of one asset class, no
  out-of-regime or crisis validation — and crypto regimes shift fast.
- **Four-factor risk model.** Market, size, momentum and vol are hedged; other
  systematic exposures (liquidity, sector/meme) still stay in the residual (only
  partly addressed by the soft cluster penalty).
- **Multiple testing.** ~130 spaces x 14 lags are screened; per-window FDR +
  gates mitigate but do not eliminate the risk of selecting overfit signals.

## License

Licensed under the **GNU Affero General Public License v3.0** (AGPL-3.0) — see
[`LICENSE`](LICENSE). In short: you are free to use, study, modify, and
redistribute this code, but any derivative you distribute **or run as a network
service** must also be released under the AGPL with its complete source. This
keeps the project and anything built on it open.

A separate **commercial license** (without the AGPL's copyleft obligations) is
available on request — open an issue or contact the author.
