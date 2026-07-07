# Crypto Stat-Arb: Factor-Neutral Strategy

Cross-sectional stat-arb on ~100 Hyperliquid-tradeable crypto names.
1-minute Binance spot data (last 3 years) -> 10-minute base panel ->
market+size+momentum+vol+meme factor model -> multi-horizon residual prediction
(10min / 1h / 1d) -> Ledoit-Wolf MVO sizing, dollar / market / size /
momentum / vol-beta neutral, net of a 5 bps/side all-in cost model.

> **Data and keys are not included.** A fresh clone has no data — the `db/`
> directory is gitignored and built locally by the ETL. The ETL downloads
> ~3 years of  1-minute Binance klines and pulls market-cap history from
> CoinGecko, which needs a free API key in an (untracked) `.keys` file at the
> repo root: `{"coingecko_api_key": "CG-..."}` (see `etl/marketcap.py`).

## Pipeline (run in order)

```bash
# 1. ETL
uv run etl/universe.py        # Hyperliquid perp candidates (~130, no stables)
uv run etl/prices_raw.py      # 1m Binance spot klines, last 3y (~big download)
uv run etl/prices.py          # resample 1m -> 10min `prices` table
uv run etl/marketcap.py       # daily mcap (CoinGecko, needs .keys) - size factor
uv run etl/funding.py         # perp funding rates (funding features + backtest funding accrual)
uv run etl/futures.py         # OI / positioning metrics (optional)
uv run etl/macro.py           # macro/event data: FRED market series (needs
                              # .env FRED_KEY), CPI/NFP release dates (fetched),
                              # FOMC schedule, stablecoin supply, DeFi TVL,
                              # Fear&Greed - all point-in-time (availability-dated)

# 2. Risk model
uv run risk_model/factor_returns.py    # market (EW) + size/momentum/vol rank-weighted factor returns
uv run risk_model/residual_returns.py  # causal betas, residuals, fwd targets
                                              # (prints acceptance checks - must PASS)
uv run risk_model/features.py          # ~170-column feature panel

# 3. Signal discovery (the ONLY signal source): one multi-lag LLM search
#    over the feature panel; promoted candidates enter the registry as
#    disc_* signals, each selectable only from its promotion date.
uv run research/signals/agent/run_discovery.py

# 4. Portfolio walk-forward: scores the promoted signals in memory, then
#    monthly-retrain selection + market-neutral backtest.
uv run research/portfolio/walk_forward.py
```

Each stage reads the tables the previous one writes, so run in order. If
`residual_returns.py` acceptance checks FAIL, stop and debug the factor model
before building features or signals. After changing the universe, refresh
prices and market-cap before rebuilding downstream.

Once the pipeline has run, open `notebooks/portfolio.ipynb` for the full
walk-forward backtest summary and risk diagnostics — equity/drawdown, per-window
and per-signal attribution, cost/exposure, and factor-neutrality — all read
from the persisted `wf_portfolio_*` tables.

## Architecture

- **Universe**: candidates = the live Hyperliquid perp listing (tradability
  filter), mapped to Binance spot symbols, stablecoins/pegged assets excluded,
  capped at 130 by HL volume. The full current candidate set is used at every
  bar; names without data at a given bar are simply NaN.
- **Factor model** — market + size + momentum + vol + meme, all tradable
  portfolio returns in return units. The characteristic factors share one
  rank-weighting machinery: daily weights proportional to the centered rank of a
  per-name characteristic, long one tail / short the other, each side scaled to
  gross 1 (every member, no tercile-boundary churn; dollar-neutral). The
  characteristic is always computed from data STRICTLY BEFORE the bar's date.
  - *Market*: equal-weight mean member return per 10-min bar.
  - *Size*: small-minus-big on lagged market-cap rank (long smalls).
  - *Momentum*: winners-minus-losers on the trailing cumulative return
    (`momentum.lookback_days`, skipping the most-recent `skip_days`; long winners).
  - *Vol*: low-minus-high on trailing realized volatility (`vol.lookback_days`;
    long low-vol).
  - *Meme*: meme-minus-nonmeme, where meme-ness = trailing correlation of a
    name's market-adjusted daily returns with a fixed anchor index of
    pre-sample canonical memes (`meme.anchor_symbols`: DOGE/SHIB/PEPE) —
    point-in-time safe and self-updating (new memes acquire high anchor
    correlation within weeks of listing). Captures the residual meme-basket
    co-movement the Marchenko-Pastur diagnostic kept finding. Measured VIF
    ~1.1: nearly orthogonal to the other factors.
    For momentum/vol/meme the sign is arbitrary — the optimizer constrains
    beta to each factor near zero rather than trading it as standalone alpha
    (same convention as size).
  - *Betas*: daily-refreshed exponentially-weighted OLS (30d window, 10d
    half-life) over all factors, estimated on strictly-past bars.
  - *Fit*: residual_returns.py prints the acceptance checks (variance
    reduction, residual-raw corr) plus per-factor VIF — watch collinearity as
    factors are added. Five-factor fit: mean R² ~0.63 at 10min,
    var(residual)/var(raw) ~0.42 (market+size two-factor baseline was ~0.61 /
    ~0.44).
- **Targets**: `fwd_res_{10min,1h,1d}[t]` = sum of single-bar residuals over
  bars t+1..t+p. Bar-end timestamps throughout; a signal at t may use data
  through bar t (forward targets start at t+1, so no overlap).
- **Signals**: discovery is the only signal source — promoted DSL candidates
  enter the registry as `disc_*` entries (`research/lib/discovered.py`; the
  hand-curated spaces library is retired, its machinery kept in
  `research/lib/spaces.py`). Each signal is scored against the 4-lag forward
  grid on an hourly screening grid; only compact aggregates are persisted
  (`signal_daily_stats`, `signal_metrics`).
- **Portfolio**:
  - *Alpha*: Grinold — `(1 − ic_shrink) × IC × residual-vol × z / √horizon`,
    summed across horizons. The IC is shrunk toward 0 (noisy/non-stationary
    realized IC) and a no-trade zone zeros any name whose edge over its holding
    horizon can't clear a round-trip cost ("lazy trading").
  - *Risk*: Ledoit-Wolf shrunk residual covariance (daily refresh) plus a soft
    cluster-exposure penalty (clusters from trailing residual correlations; the
    Marchenko-Pastur diagnostic shows stable super-noise structure — e.g. a
    meme-coin factor — beyond market+size).
  - *Size*: Ledoit-Wolf shrunk-covariance MVO under dollar / market / size /
    momentum / vol-beta neutrality within `neutrality_band`, a per-name cap
    (`max_position`), gross leverage 1 scaled by expected-edge-vs-cost
    (`edge_scaled_gross`), a hard turnover budget, and a volume-participation
    cap; the book trades toward the aim at a cost-responsive
    Garleanu-Pedersen rate with liquidity-aware per-name multipliers.
  - *Test*: backtested on RAW forward returns, net of trading costs AND perp
    funding accrued on held positions at settlement stamps; realized factor
    exposures are reported as the market-neutrality check, alongside a
    half-alpha diagnostic (`cost / expected gross alpha`, ~0.5 at the optimum).

## Signals: generation & selection

**Scoring** (`research/lib/signal_eval.py`, run in-memory by the walk-forward):

- The signal universe is whatever discovery promoted: `disc_*` DSL programs
  (see "Using the agentic signal generator"). There are no hand-written
  signals; a manual hypothesis can be re-added as one `_S(...)` line in
  `research/lib/spaces.py` if ever needed.
- Each signal is computed at full 10-min resolution, smoothed **at a halflife
  matched to the scored lag** (`signals.lag_smoothing`: fast lags get fast
  smoothing, slow lags slow — measured, this cuts slow-lag turnover 2.5–3x for
  a ~20% IC give-up), cross-sectionally z-scored, then scored by **rank IC**
  against the forward residual targets at every lag in the deliberately small
  4-lag grid `signals.decay_lag_grid` = [3, 6, 24, 144] bars (30min / 1h / 4h /
  1d — chosen from the measured decay: fast core ≤ 4h, funding at ~1d; every
  extra lag inflates the Bonferroni correction applied to every signal), on a
  non-overlapping hourly screening grid (overlap would inflate IC t-stats).
  The walk-forward recomputes each selected signal at the halflife of its
  selected lag.
- Only compact per-day aggregates (`signal_daily_stats`) and whole-period
  diagnostics (`signal_metrics`) are stored — the raw panels are recomputed on
  demand. Incremental: re-running only re-evaluates new/changed signals.

**Selection** (per walk-forward window, train data only —
`research/portfolio/walk_forward.py`). Retraining happens every `test_days`
(monthly), mirroring the production retrain cadence; the training window is
**expanding** by default (`walk_forward.train_window`): every retrain uses ALL
data from `start_date` to that window's train end, so later windows select
from ~10x the observations of the legacy rolling 6-month slice ('rolling'
restores the fixed lookback):

1. **Best lag per signal** — pick the strongest forward lag by HAC IC t-stat,
   Bonferroni-adjusted for searching the grid.
2. **Benjamini-Yekutieli FDR** (loose) across the signals — sweep out the
   clearly-spurious tail.
3. **Gates** — one economics gate (training net Sharpe after amortized
   costs >= `min_net_sharpe_threshold`) plus robustness (IC stability across
   window thirds, recent-third sign, liquid-half IC) and a minimum holding
   period. The holding floor is *derived from the execution layer*
   (`min_holding_lag_bars: 'auto'`): a lag must retain at least
   `min_monetizable_alpha_fraction` of its alpha after the Garleanu-Pedersen
   aim discount at the book's effective fill rate, so the selector cannot
   spend slots on signals the (turnover-budgeted) executor then scales to ~0.
4. **Standardized composite ranking**, then a **per-theme cap** + **greedy
   de-correlation** on daily returns (capped at `max_signals_per_window`
   total). Execution buckets are the **distinct selected lags** (e.g. `6b`,
   `144b`) — each bucket's composite refreshes at its own lag-matched cadence.

The selected signals are recomputed at full resolution on the test window
(each at its selected lag's smoothing halflife), **covariance-aware combined**
(`C⁻¹·netSharpe`) into per-lag composites, and handed to the optimizer.


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

**The search space is combinatorial, not the column count**: the ~100
feature columns are the alphabet; the features actually tested are the DSL
programs over them — ~44k distinct programs at depth 2, ~18 million with a
single condition gate, trillions at the full depth-4 grammar. All depth-1
programs (every raw column) are exhaustively IC'd each roll for the
proposer's diagnostics; the deeper space is searched (`discovery.search`
budget) against the TRAIN window only, and promotion prices its own
multiplicity with a √(2·ln n_looks) deflation haircut over the looks it
actually takes at the held-out select month (survivors × horizons) —
coverage is bought at an honest multiple-testing price, never for free.

The program compiles to a value per (10-min bar, coin), which is then
demeaned + z-scored **across the ~130-coin cross-section** at every bar and
clipped to ±3. A signal is never "BTC will go up" — it is a *ranking of
coins against each other* at each moment (long the top, short the bottom,
~0 net in signal space). It is judged against **forward residual returns**
— market/size/momentum/vol moves stripped out — so it must predict
coin-specific mispricing, not beta. There is **no pinning to a single
horizon**: each candidate is evaluated at every horizon in
`discovery.horizon_lags_bars` (1h/6h/12h/1d) and keeps the whole per-lag IC
**profile** — its alpha term structure — plus an alpha **half-life** fitted
to those four points (snapped to a fixed grid from 30min to 2 weeks). Fast
reversal and slow carry candidates are found in the same run with no
horizon prior. Duration enters the score as the **capture weight**
`1/(1 + φ/κ)` (φ = ln2/half-life, κ = the book's per-bar trade rate): the
fraction of a signal's IC a book trading at κ can actually be exposed to —
a 6h-half-life signal outscores an equally-strong 1h one ~2.4×. Not a cost
model — duration, never bps. The traded direction and the reward come from
TRAIN only — the search never sees the select month, so promotion's select
t-stats are measurements, not the maximum of a search on themselves.
Cross-horizon comparisons use **day-equivalent t-stats** (t / √(stamps per
day)), so a 1-hour signal gets no mechanical sample-size advantage over a
24-hour one.

### The walk-forward windows

Each **roll** = TRAIN 5mo (fit + diagnostics + the search reward — the only
window the search ever optimizes) → SELECT 1mo (held out from the search;
promotion tests it exactly once per survivor) → OOS 1mo (the promotion's
`valid_from` date — discovery itself never trades; the walk-forward is the
only money judge), advancing one month at a time from `discovery.start_date`
(2023-08) until OOS reaches `discovery.end_date` — e.g. roll 0 trains
Aug–Dec 2023, selects on Jan 2024, trades Feb 2024. Purge+embargo bars are
dropped at every boundary so forward targets cannot leak. Stitching all OOS
months end-to-end gives the system's equity curve.

### How the search selects ("survivors")

Within a roll, the search runs a fixed number of generations. Each
generation: the LLM proposes a batch of new programs (per feature family,
allocated by a UCB bandit) → each is compiled, causality-checked and scored
into one **reward** (TRAIN-month capture-weighted day-equivalent IC t-stat
+ liquid-IC ratio, minus penalties for complexity, train-thirds
instability, similarity — the select month is never consulted) → the
population is then
**culled to the top `search.survivors` (12)**, with a diversity rule that
skips near-duplicates (train-signal corr > 0.8). The 12 that remain are the
*survivors* — the parents the LLM mutates next generation.
The next roll re-tests them from scratch on its own windows; surviving twice
in a row is what the `min_rolls_survived` promotion gate measures. Hierarchy
of trust: tried → survivor (one roll, could be luck) → persistent survivor
(two fresh train windows) → **promoted** (persistent + any profile lag
clears FDR/deflation on the held-out select month + profile sign agreement
+ capture weight ≥ `promotion.min_capture` + uncorrelated with the book).
Discovery emits promotions and nothing else — PnL, costs, and the equity
curve live exclusively in `research/portfolio/walk_forward.py`.

### Run discovery

```bash
uv run research/signals/agent/run_discovery.py --max-rolls 2   # test the system first
uv run research/signals/agent/run_discovery.py                 # full history
```

**LLM & cost**: `gemini-2.5-flash` (change in config `discovery.llm`). API
key goes in the gitignored `.env` at the repo root: `LLM_KEY=...`. Cost is
roughly **$0.05 per roll ≈ $1.50 for the full history**; actuals are tracked
per run in `discovery_llm_usage`.

The proposer is the config LLM (`discovery.llm.provider`); every run is a
**fresh start** (the discovery tables are cleared first). Flags:

- `--no-fresh` — keep existing discovery tables; `--no-save` — dry run

Every roll also prints the **ML ceiling** per lag (a gradient-boosting fit on
ALL features): the upper bound on what any search over this feature set could
find at that horizon — barren features and a failed search look identical
without it.
- Each roll's search is seeded with the previous roll's survivors, which
  re-earn survival on the new windows — that is what makes the
  `min_rolls_survived` promotion gate measurable.

All knobs live in `config.py` under `discovery.*` (families/input space, DSL
bounds, search budget, reward weights, promotion gates, backtest, LLM
provider/model/prices).

The LLM fills exactly one slot — *which programs to try next*; everything
else (compile, causality, scoring, selection, promotion, backtest) is
deterministic code, so a wrong LLM can only waste budget, never corrupt
results.

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
promotes nothing, reproducibility): `uv run tests/discovery_checks.py`.

### From discovery to the portfolio walk-forward

Promoted candidates ARE the signal universe (`research/lib/discovered.py`;
toggle `signals.include_discovered`): every promotions-table row, deduped by
program hash, becomes a `disc_<family>_<hash>` registry entry. After a
discovery run:

```bash
uv run research/portfolio/walk_forward.py    # scores, selects, backtests
```

The walk-forward scores them in memory, then selects/trades them under the
FDR, economics gate, costs and participation cap. One honesty rule: each
discovered signal is only selectable in windows whose training end is at or
after its **promotion date** — its expression was chosen by a search that saw
data up to that roll, so earlier windows would be trading a formula chosen
with future knowledge. The discovery system's own stitched OOS curve remains
the cleanest read on the search itself; the walk-forward answers whether the
promoted book earns a slot in the production portfolio.


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
- **Market + macro data.** Inputs are crypto prices, market-cap, funding and
  open-interest, plus point-in-time macro/event data (etl/macro.py): FRED
  market-priced series, the FOMC/CPI/NFP calendar, stablecoin supply, DeFi
  TVL, Fear&Greed. Macro enters as event/state conditioners (ev_/mx_, DSL
  gate material) and per-name sensitivities (mb_). Still no order-book/L2,
  news, or per-name on-chain data.
- **Costs: conservative 5 bps/side + volume-participation cap.**
  `portfolio.cost_bps = 5.0` is a deliberately conservative all-in per-side
  assumption (for reference, Hyperliquid perp maker is 0.000% at tier 4+ and
  0.4–1.5 bps below; taker 2.4–4.5 bps). Signals are scored NET of this cost
  (selection amortizes it by the Garleanu-Pedersen fill factor so it prices
  the turnover the executor actually trades), and the backtest additionally
  enforces `portfolio.participation`: no name trades more than 10% of its
  trailing 10-bar average $ volume per bar at the configured
  `book_size_usd` — re-run at several book sizes for a capacity curve. Perp
  funding on held positions is accrued at settlement stamps (Binance
  USDT-perp rates as a Hyperliquid proxy; longs pay positive rates). The
  model still does *not* price passive-fill risk (missed fills, adverse
  selection) or nonlinear impact, and assumes fills at 10-min bar stamps —
  the 1-bar-lag Sharpe is the closest stress for this. Set cost_bps to 0
  to model pure top-tier maker execution.
- **Single short sample / regime.** ~3 years of one asset class, no
  out-of-regime or crisis validation — and crypto regimes shift fast.
- **Five-factor risk model.** Market, size, momentum, vol and meme are
  hedged; other systematic exposures (liquidity, narrative sectors beyond
  meme) still stay in the residual (only partly addressed by the soft
  cluster penalty).
- **Multiple testing.** The discovery search screens hundreds of candidate
  programs per roll across 4 lags; the FDR + deflation + persistence gates
  mitigate but do not eliminate the risk of promoting overfit signals.

## License

Licensed under the **GNU Affero General Public License v3.0** (AGPL-3.0) — see
[`LICENSE`](LICENSE). In short: you are free to use, study, modify, and
redistribute this code, but any derivative you distribute **or run as a network
service** must also be released under the AGPL with its complete source. This
keeps the project and anything built on it open.

A separate **commercial license** (without the AGPL's copyleft obligations) is
available on request — open an issue or contact the author.
