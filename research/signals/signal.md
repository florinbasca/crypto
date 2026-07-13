# Agentic Signal Discovery

What do I mean by signals: formulas that rank the universe each bar by expected residual return. The ranking is cross-sectional or scored against each other. Promoted signals are used in the walk-forward backtest, which sizes a dollar and factor-neutral book. The walk-forward is the only
place P&L is judged; discovery is purely statistical, refers to residuals and can't be traded directly.

The formula space is infinite and naive search overfits. Three controls: a bounded expression, an LLM proposing candidates, and promotion only of signals that stay significant on a held-out month. Scoring and selection are deterministic.

"Agentic" describes the loop, not the model. Each LLM call is one stateless
prompt → JSON completion: no tools, no memory, no data access, no control over
what happens next. The feedback (scored parents, failures, over-mined blocks in
the next prompt), the budget allocation, and every decision are deterministic
code — the LLM is the idea generator inside an evolutionary search, nothing more.

## Grammar or the language of the signal

Formulas are written in a small fixed **DSL** (domain-specific language: a fixed set of operators over whitelisted feature columns). A candidate is an expression plus gate conditions:

```json
{"expression": ["cs_zscore", ["div", ["col","fr_annualized"],
                              ["roll_std", ["col","fr_rate_zscore"], 36]]],
 "conditions": [["abs_gt", ["col","fr_rate_change"], 0.001]]}
```

Operators: elementwise (`neg abs sign square sqrt log1p tanh + − × ÷`),
per-coin rolling (`mean std sum zscore delta` over fixed windows),
cross-sectional (`rank zscore demean`), and `where` gates. Bounds: depth ≤ 4,
≤ 2 conditions, ≤ 24 nodes (`discovery.dsl`).

The compiler evaluates the expression, applies gates, then per bar demeans,
z-scores, and clips to ±3 (the demean is what makes each bar sum to ~0). Columns are grouped into families, themed sets like volatility, funding, order flow, trend (`discovery.families`).

## Search loop

A **roll** is one train/select window pair. Per roll, for 16 generations:

1. The LLM proposes a batch of 32 candidates, its slots split across families by the rule below.
2. Each candidate is compiled and scored on train by its **per-bet return** (see Reward)
3. The best-by-reward survivors are kept, de-duplicated, and seed the next generation.

The 32 slots are split across families by an upper-confidence-bound rule: a family's priority = its mean reward so far + an exploration bonus that decays as it is tried, so more of the batch goes to families that keep producing high-reward candidates, without starving untried ones.

## Validation

Two windows per roll, advancing one month at a time:

```
| TRAIN 5mo | SELECT 1mo |   (OOS 1mo = the promotion's valid_from date)
  search happens here    tested once, at promotion
```

- **Train**: the entire search — scoring, reward, breeding, half-life fits.
- **Select**: held out. Promotion is its first and only read, so a select score
  is a clean measurement, not the max of a search on itself.
- A purge + embargo gap (max horizon + embargo bars) is dropped at the boundary
  so forward targets can't leak across it.
- Months are independent (`min_rolls_survived = 1`): a signal trades an OOS month
  only by re-qualifying on the 6 months ending just before it.

## Reward (train only)

The performance metric is the candidate's **per-bet return** ("alpha"):
build the dollar-neutral, gross-1 long/short book from the signal's
cross-section, hold it for the horizon, measure what it returned
(e.g. +3bps per bet).
**Rank IC is recorded as a diagnostic only and selects nothing**: this
system's first full run proved a signal can order names correctly for a day
(rank IC t≈15) while the large moves run against its positions (negative
returns). Ordering and returns are different quantities; the reward uses
returns.

```
reward = Σ wₖ · termₖ / scaleₖ

+ alpha_tstat        capture-weighted, day-equivalent t of the per-bet return
+ liquid_alpha_ratio liquid-half alpha vs the full cross-section
+ incremental        per-bet return the candidate adds to the survivor pool
− complexity         node + condition count
− instability        std of per-bet alpha across train thirds
− similarity         max correlation to kept survivors
```

Each candidate is scored at every horizon. Its per-horizon alpha profile (the
measured return term structure) and fitted half-life carry to promotion.

A fast signal looks better than it really is, for two separate reasons, so the
alpha term is corrected for both:

- It places more bets, which inflates its t-stat (~√24× for a 1h vs a 1d signal)
  with no extra skill. **Day-equivalent** t-stat, `t / √(stamps per day)`,
  cancels that — it scores skill per bet, not bet count.
- Even when the skill is real, a fast edge decays before the book can trade into
  it, so only part of it is capturable. **Capture weight**, `1 / (1 + φ/κ)`,
  scales the alpha down to the tradable fraction. φ = ln2 / *effective
  persistence* = min(alpha half-life, position life `1/turnover`) — the alpha
  and the positions must BOTH live long enough (turnover is per bar, so
  `1/turnover` is the bars until the signal has fully reshuffled itself).
  κ = the GP fill rate the walk-forward actually trades at (~0.048/bar;
  per-name fills are further capped at 10% of trailing volume). At the
  current κ the 0.5 capture floor binds at churn ≈ 0.07/bar. Duration,
  not fees.

**Incremental** rewards adding something new instead of repeating what you
already have: the pooled per-bet return of survivors *with* the candidate
included minus *without* it. A near-duplicate adds ~0. Cost is never a reward
term — a signal's real cost depends on the whole book, so it is judged only
in the walk-forward.

**Turnover** (mean per-bar book churn, `0.5·Σ|Δw|` on the gross-1 signal;
0 = positions never change, 1 = replaced every bar) is recorded per candidate
and enters selection twice: through the capture weight (position life
`1/turnover` caps effective persistence — the graded gate, binding at
~0.07/bar) and as a hard fails-open backstop `max_turnover = 0.10`/bar for
extremes. Never a cost term — cost stays in the walk-forward.

## Diversity

An evolutionary search collapses toward one idea: once a family scores well the
LLM keeps proposing variations of it, and you end up with twenty near-copies of
the same signal. That is no more useful than one, and it fakes a diversified
book. Two guards keep survivors genuinely distinct:

- **What they output**: two survivors' signal values must correlate ≤
  `diversity_max_corr`. Catches signals built differently that still rank the
  coins the same way.
- **How they're built**: their expression trees must share ≤
  `diversity_max_ast_sim` of their sub-expressions (Jaccard overlap = shared
  blocks ÷ total distinct blocks). Catches
  clones that swap one column but reuse the same structure — which output
  correlation can miss. The most over-used building blocks each roll are fed
  back to the LLM as a "vary away from these" hint.

## Promotion

Once per roll, on the held-out select month. All statistical tests are on the
**per-bet return**, not rank IC. A survivor promotes if any
horizon of its profile clears all of:

- **FDR** control (Benjamini-Yekutieli false-discovery-rate) across every
  (survivor, horizon) select p-value (Student-t, n_days−1 dof).
- **Deflation**: the select t must beat `deflation_mult · E[max|N(0,1)|]` over
  the actual looks taken (survivors × horizons) — the multiple-testing haircut.
- Minimum select alpha t-stat and minimum daily observations. All three tests
  use the **directed** t (positive in the traded direction), not |t| — a lag
  whose sign reverses out-of-sample is rejected, never admitted on magnitude.
- Sign agreement: the train profile mostly shares the traded sign.
- Capture floor: effective persistence long enough for the book to hold.
- Turnover ceiling (`max_turnover = 0.10`/bar): extremes backstop above the
  capture floor (which already binds at ~0.07/bar). Fails open when turnover
  is unknown.
- Orthogonality to signals already promoted this roll.

Passers are written to the promotions table with profile, half-life, turnover,
and direction. Promotion neither trades nor sizes.

## What the LLM sees

Train diagnostics only — never returns or the raw panel:

- A one-line description of each column.
- Per-column per-bet return (alpha) with its t-stat, decile curve,
  regime-split alpha, stability — the same currency the reward uses.
- The top columns per family, ranked by a blend of monotonic alpha, decile
  nonlinearity, regime spread, and stability (so U-shaped/threshold features
  aren't hidden by a t-stat-only rank).
- Current survivors with scores, best first; recently-culled ones and over-mined
  subtrees to avoid.

It emits DSL JSON; everything it returns is re-validated and re-scored by code.
The fixed system prompt (role, DSL rules, output format) is `prompt.md`; the
per-call user prompt is assembled from the diagnostics above in
`generation.py` (`_prompt`).

## Overfitting

The in-run controls (train/select split, FDR, deflation, held-out month) do not
cover the research process itself: tuning config across runs spends the select
window's honesty. A deflated Sharpe ratio, a backtest-overfitting probability,
and a never-touched lockbox period are the next step.

## Run

```bash
uv run research/signals/discovery.py            # full history
uv run research/signals/discovery.py --resume   # continue an interrupted run
uv run research/signals/inspect_discovery.py    # review a run
```

Promoted signals become `disc_*` registry entries (`research/lib/discovered.py`).
The walk-forward mirrors discovery month by month: each roll's promoted signals
are traded in that roll's OOS month only. All knobs live under `discovery.*` in
`config.py`.

## Cost and run time

A full run is 28 rolls (windows) × 16 generations. The model for now is Gemini
3.1-flash-lite ($0.25 / $1.50 per million input / output tokens — the
price-equivalent successor to 2.5-flash, which Google retired mid-2026). That
works out to roughly **$1–3 per roll, order $30–80 for the full run**. The
script prints the measured tokens and dollars per roll; trust that over these
estimates.

Run time is dominated by candidate scoring (~15–20s per candidate on the
5-month train panel), not the LLM: proposal calls run concurrently
(`discovery.llm.parallel_requests`, default 8). Expect roughly 2–2.5 hours per
roll, ~2–3 days for the full 28. Progress bars (per-generation, per-call,
per-candidate) show the run is alive; lower parallel_requests if the provider
rate-limits.

