# Crypto Stat-Arb: Factor-Neutral Strategy

Cross-sectional stat-arb on ~130 Hyperliquid-tradeable crypto names.
1-minute Binance spot data (last 3 years) -> 10-minute base panel ->
market+size+momentum+vol+meme factor model -> residual-return prediction -> MVO optimized, dollar / market / size /
momentum / vol / meme-beta neutral.

> **Data and api keys are not included.** A fresh clone has no data, but should be built locally by the ETL, starting with 1-minute Binance klines. Two free keys are required in the gitignored repo-root `.env`: `COINGECKO_KEY=CG-...` (market-cap, see `etl/marketcap.py`) and `FRED_KEY=...` (macro series, see `etl/macro.py`).

## Pipeline (run in order)

```bash
# 1. ETL
uv run etl/universe.py        # Hyperliquid perp candidates (~130, no stables)
uv run etl/prices_raw.py      # 1m Binance spot klines, last 3y (~big download)
uv run etl/prices.py          # resample 1m -> 10min `prices` table
uv run etl/marketcap.py       # daily mcap
uv run etl/funding.py         # perp funding rates
uv run etl/futures.py         # OI / positioning metrics (optional)
uv run etl/macro.py           # macro/event data: FRED market series (needs .env FRED_KEY)

# 2. Risk model
uv run risk_model/factor_returns.py    # builds factor returns
uv run risk_model/residual_returns.py  # causal betas, residuals, fwd targets
uv run risk_model/features.py

# 3. Agentic signal discovery: multi-lag LLM search over the feature panel
uv run research/signals/agent/discovery.py

# 4. Portfolio walk-forward: scores the discovered signals, then
#    builds and backtests a factor-neutral portfolio.
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
  - *Momentum*: winners-minus-losers on the trailing 30-day cumulative
    return, skipping the most-recent 2 days (long winners).
  - *Vol*: low-minus-high on trailing 30-day realized volatility (long low-vol).
  - *Meme*: meme-minus-nonmeme, where a name's meme-ness is its trailing
    60-day correlation with a fixed anchor basket of three canonical memes
    (DOGE/SHIB/PEPE). Because the anchors are fixed and were memes before the
    sample begins, no look-ahead and no list to maintain — a new meme earns
    membership by correlating with the anchors within weeks of listing.
- **Targets**: `fwd_res_{10min,1h,1d}[t]` = sum of single-bar residuals over
  bars t+1..t+p. Bar-end timestamps throughout; a signal at t may use data
  through bar t (forward targets start at t+1, so no overlap).
- **Signals** — found by agentic discovery: an LLM proposes candidates, while a deterministic genetic search evolves and filters them. Full description below.
- **Portfolio** a mean-variance optimization with dollar and factor-neutral, 1× leverage.

## Signals: how they're found

Signals aren't written by hand. An LLM proposes small formulas (feature columns
combined with a few operators and optional gates); a deterministic search scores
them on rolling monthly windows and promotes only those that stay statistically
significant on a held-out month. The promoted formulas are the only signals the
portfolio trades — discovery itself never trades.

```bash
uv run research/signals/agent/discovery.py --max-rolls 2   # quick test
uv run research/signals/agent/discovery.py                 # full history
uv run research/signals/agent/inspect_discovery.py             # review a run
```

Uses `gemini-2.5-flash` (config `discovery.llm`, key in `.env` as `LLM_KEY=...`).
Promotions land in `discovery_promotions` and become `disc_*` registry entries
(`research/lib/discovered.py`), which the walk-forward then scores and trades —
each only from its promotion date onward (using it earlier would be look-ahead).

**The full design — DSL grammar, the LLM prompt, the reward, the promotion
gates, the walk-forward windows — is documented in
[`research/signals/agent/agent.md`](research/signals/agent/agent.md).**


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
  programs per roll across 4 horizons; the FDR + deflation gates on the
  held-out month mitigate but do not eliminate the risk of promoting overfit
  signals.

## License

Licensed under the **GNU Affero General Public License v3.0** (AGPL-3.0) — see
[`LICENSE`](LICENSE). In short: you are free to use, study, modify, and
redistribute this code, but any derivative you distribute **or run as a network
service** must also be released under the AGPL with its complete source. This
keeps the project and anything built on it open.

A separate **commercial license** (without the AGPL's copyleft obligations) is
available on request — open an issue or contact the author.
