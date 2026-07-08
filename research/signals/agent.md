# Agentic Signal Discovery

What do I mean by signals: formulas that rank the universe each bar by expected residual return. The ranking is cross-sectional or scored against each other. Promoted signals are used in the walk-forward backtest, which sizes a dollar and factor-neutral book. The walk-forward is the only
place P&L is judged; discovery is purely statistical, refers to residuals and can't be traded directly.

The formula space is infinite and naive search overfits. Three controls: a bounded expression, an LLM proposing candidates, and promotion only of signals that stay significant on a held-out month. Scoring and selection are deterministic.

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
2. Each candidate is compiled and scored on train dataset by rank **IC**
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

```
reward = Σ wₖ · termₖ / scaleₖ

+ ic_tstat        capture-weighted, day-equivalent train IC t-stat
+ liquid_ic_ratio IC on the liquid half vs the full cross-section
+ incremental     IC the candidate adds to the current survivor book
− complexity      node + condition count
− instability     std of IC across train thirds
− similarity      max correlation to kept survivors
```

Each candidate is scored at every horizon. Its per-horizon IC profile and fitted half-life (bars for the edge to decay by half) carry to promotion.

A fast signal looks better than it really is, for two separate reasons, so the
IC term is corrected for both:

- It places more bets, which inflates its t-stat (~√24× for a 1h vs a 1d signal)
  with no extra skill. **Day-equivalent** t-stat, `t / √(stamps per day)`,
  cancels that — it scores skill per bet, not bet count.
- Even when the skill is real, a fast edge decays before the book can trade into
  it, so only part of it is capturable. **Capture weight**, `1 / (1 + φ/κ)`
  (φ = decay rate = ln2/half-life, κ = how fast the book trades), scales the IC
  down to the tradable fraction: a 1h-half-life edge keeps ~0.29, a 12h one
  ~0.83. This is about duration, not fees.

**Incremental** rewards adding something new instead of repeating what you
already have: the **pooled IC** (the IC of all survivors averaged into one
signal) *with* the candidate included minus *without* it. A near-duplicate adds
~0; a signal that captures a mechanism the book was missing adds a lot. Cost is
never a reward term — a signal's real cost depends on the whole book, so it is
judged only in the walk-forward.

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

Once per roll, on the held-out select month. A survivor promotes if any horizon
of its profile clears all of:

- **FDR** control (Benjamini-Yekutieli false-discovery-rate) across every
  (survivor, horizon) select p-value (Student-t, n_days−1 dof).
- **Deflation**: the select t must beat `deflation_mult · E[max|N(0,1)|]` over
  the actual looks taken (survivors × horizons) — the multiple-testing haircut.
- Minimum select t-stat and minimum daily observations. All three tests use the
  **directed** t (positive in the traded direction), not |t| — a lag whose sign
  reverses out-of-sample is rejected, never admitted on magnitude alone.
- Sign agreement: the train profile mostly shares the traded sign.
- Capture floor: half-life long enough for the book to hold it.
- Orthogonality to signals already promoted this roll.

Passers are written to the promotions table with profile, half-life, and
direction. Promotion neither trades nor sizes.

## What the LLM sees

Train diagnostics only — never returns or the raw panel:

- A one-line description of each column.
- Per-column IC, decile curve, regime-split IC, stability.
- The top columns per family, ranked by a blend of monotonic IC, decile
  nonlinearity, regime spread, and stability (so U-shaped/threshold features
  aren't hidden by a t-stat-only rank).
- Current survivors with scores, best first; recently-culled ones and over-mined
  subtrees to avoid.

It emits DSL JSON; everything it returns is re-validated and re-scored by code.
The fixed system prompt (role, DSL rules, output format) is `prompt.md`; the
per-call user prompt is assembled from the diagnostics above in
`generation.py` (`_prompt`).

## ML probe

Each roll fits a gradient-boosting model on all features per horizon and prints
its IC — the predictability ceiling. ~0 means the features are barren at that
horizon; a real ceiling the search can't reach means the search, not the data,
is the bottleneck. Diagnostic only, never used as signal.

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

Promoted signals become `disc_*` registry entries (`research/lib/discovered.py`),
tradable in the walk-forward only from their promotion date. All knobs live under
`discovery.*` in `config.py`.

## Cost and run time

A full run is 28 rolls (windows) × 16 generations. The model for now is Gemini 2.5-flash ($0.30 / $2.50 per million input / output tokens), which works out to roughly **$1–3 per roll, order $30–80 for the full run**. 

Run time is about 1.5 hours per roll, total about 40 hours, dominated by the sequential API round-trips (scoring is fast by comparison).

