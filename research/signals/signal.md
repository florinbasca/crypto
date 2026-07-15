# Agentic Signal Discovery

What do I mean by signals: formulas that rank the universe each bar by expected residual return. The ranking is cross-sectional or scored against each other. Promoted signals are used in the walk-forward backtest, which sizes a dollar and factor-neutral book. The walk-forward is the only
place P&L is judged; discovery is purely statistical, refers to residuals and can't be traded directly.

The formula space is infinite and naive search overfits. Three controls: a bounded expression, an LLM proposing candidates, and promotion on held-out evidence — the POOLED directed alpha across every select month a candidate was ever measured on, never a single month (a lone month has no power against real-but-modest alpha). Scoring and selection are deterministic.

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
z-scores, and clips to ±3 (the demean is what makes each bar sum to ~0). Columns are grouped into families, themed sets like volatility, funding, order flow, trend (`discovery.families`). Beyond the fast price/volume families, slow fundamental families come from free sources (see README "Slow data"): token-unlock calendar (`un_`), developer activity (`dv_`), listing age (`ls_`), and stablecoin-supply state (`mx_stable_`, gate-only).

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
- Membership never carries over (`min_rolls_survived = 1`): a signal trades an
  OOS month only by re-qualifying on the 6 months ending just before it. What
  DOES accumulate across rolls is *evidence*: each roll's select month is a
  distinct, never-searched measurement, and promotion pools all of them
  (see Promotion).

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

+ alpha_tstat   capture-weighted, day-equivalent t of the per-bet return
− similarity    max correlation to kept survivors
```

Two terms, deliberately. Every additional hand-tuned term is a place the
search can silently optimize the wrong thing: an earlier six-term reward's
absolute-units instability penalty single-handedly culled a candidate with
train t 5.7 (and select t 6.6) at reward −1.9 while smooth near-zero-alpha
candidates survived — the search was optimizing for stable mediocrity.
Stability pressure now lives where it is measured honestly: promotion's
cross-roll pooling — a signal that is noisy across months never accumulates
pooled t.

Each candidate is scored at every horizon its FAMILY owns
(`discovery.family_horizon_lags`, intersected with `horizon_lags_bars`:
1h, 6h, 12h, 1d). Its per-horizon alpha profile (the measured return
term structure) and fitted half-life carry to promotion. The slow families
(unlocks, dev activity, listing age) are measured at 1d only, concentrating
their evidence at one lag. A 3d horizon existed for them and was removed on
ledger evidence: directed select follow-through at 3d was 23% (worse than a
coin flip), the slow families measured *better* at 1d, and the long purge
it forced (max lag + embargo) cost every horizon ~2 days of train and
select data per window. Multi-day event theses belong to
`event_study.py`, which pools the full history instead.

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

Cost is never a reward term — a signal's real cost depends on the whole
book, so it is judged only in the walk-forward.

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
book. One guard keeps survivors distinct: **what they output** — two
survivors' signal values must correlate ≤ `diversity_max_corr`. What a signal
outputs is what the book trades; two builds that rank the coins the same way
are one signal. (A structural AST-overlap check used to double this; output
correlation subsumes what mattered.) The most over-used building blocks each
roll are still fed back to the LLM as a "vary away from these" hint.

## Promotion

Once per roll. All statistical tests are on the **per-bet return**, not rank
IC, and the evidence is **pooled across months**: each roll re-evaluates
carried-over survivors on a fresh, never-searched select month, and promotion
combines every month a candidate was ever measured on (fixed-effect
meta-analysis over the ledger: per month se = mean/t, pooled t =
inverse-variance-weighted mean × √(Σ1/se²)). Prior months were directed by
that roll's own train fit, so they are sign-aligned to the *current*
direction first — a month measured trading the other way counts against.

Why pooling: one select month (~30 daily obs) gives a true Sharpe-2 signal
an expected t of only ~0.58 — a 3+ single-month bar selects luck, not skill
(the first run's gate stack demanded ~3.5 and promoted nothing, which
carried no information either way). Pooling k months scales expected t by
√k: *persistence across rolls*, not one lucky month, is what confirms.

The mechanism is **rank + K slots + sanity floors** — no significance gates:

- **Rank**: **posterior Sharpe × capture** at the best of the candidate's
  family horizons. Posterior Sharpe is the observed pooled annualized Sharpe
  shrunk by the evidence behind it:

  ```
  SR_observed = pooled_t / √n_days × √365
  posterior   = SR_observed · n / (n + 365/τ²)      τ = prior_sharpe_std
  ```

  One number, one interpretable constant (τ = the spread of true Sharpes we
  believe the pool can contain), combining the three things that matter:
  **effect size** (Sharpe — at equal evidence, the bigger edge ranks
  higher), **statistical evidence** (t and observation count — a t of 2 in
  one month is an observed Sharpe of ~7, almost certainly luck, and the
  shrinkage prices that; a modest Sharpe sustained a year keeps ~half its
  value), and **stability** (inconsistent months cancel inside the pooled t
  before this formula ever sees them). Daily Sharpe is bets-per-day-fair by
  construction, and capture prices whether the book can trade at the
  signal's speed — the product ranks the *tradable* expected Sharpe. Never
  the search reward — the first run measured reward and select alpha as
  ~uncorrelated; ranking by reward would supply stable mediocrity.
- **K slots** (`book_size`): the top K ranked survivors promote. A quota
  substitutes for a significance gate **only because the ranking metric is a
  sufficient measure of quality** — the quota caps how MANY promote, the
  metric decides WHO, and the walk-forward re-judges the book every month.
  (An earlier stack of BY-FDR × deflation × min-t compounded into an
  accidental one-month bar of t≈3.5 and promoted nothing — which carried no
  information.) A roll that promoted nothing would give the walk-forward —
  the only money judge — nothing to measure; a fixed K keeps it fed.
- **Sanity floors** reject only the *actively bad*, never enforce
  significance:
  - Pooled directed t > 0: the held-out months must not net-run against the
    traded direction. Directed, not |t| — a sign-reversed signal is
    rejected, never admitted on magnitude.
  - Minimum pooled observation days (`min_select_days`).
  - Sign agreement: the train profile mostly shares the traded sign.
  - Capture floor: effective persistence long enough for the book to hold.
  - Turnover ceiling (`max_turnover = 0.10`/bar): extremes backstop above
    the capture floor (which already binds at ~0.07/bar). Fails open when
    turnover is unknown.
  - Orthogonality to signals already promoted this roll (`max_book_corr`).

Promotions are written with the evidence lag, half-life, turnover, direction
and the evidence (`pooled_select_tstat`, `pooled_select_months`,
`pooled_sign_frac`, `posterior_sharpe`, `promotion_score`), plus a
provenance stamp (`run_id`, `config_hash`, `data_hash`, `git_sha`) so every
row is attributable to the exact run/config/data that produced it — a table
mixing runs after a config change is detectable, never silently blended.
Promotion neither trades nor sizes.

The walk-forward consumes each roll's promotions at that roll's **evidence
lag** and **fitted direction** (never the registry's deduped defaults), and
converts per-bet alpha to per-bar linearly (returns scale with time; the
√h convention belongs to correlations). Null controls
(`walk_forward.py --control shuffle|sign_flip|random`) backtest placebo
books that the real book must clearly beat; control results are printed
but never persisted.

### Power: what happens to a true Sharpe-2 signal

A Sharpe-2 signal is far too valuable to lose to a threshold, and the design
is built so it never faces one. Its expected monthly select t is
2/√12 ≈ 0.58 — invisible in any single month, which is why no single-month
significance bar exists anywhere. What it faces instead is accumulation
(τ = 1, ~30 obs days/month):

| pooled months k          | 1    | 4    | 9    | 16   |
|--------------------------|------|------|------|------|
| E[pooled t] = 0.58·√k    | 0.58 | 1.15 | 1.73 | 2.31 |
| P(directed floor, t > 0) | 72%  | 88%  | 96%  | 99%  |
| E[posterior Sharpe]      | 0.15 | 0.49 | 0.85 | 1.14 |

Meanwhile a noise candidate's pooled t random-walks around zero — half fail
the directed floor outright each roll — and it must re-win the train-side
survivor cut every month just to keep being measured. Real signals compound
evidence; noise churns. Three design choices serve this directly:

- **Non-overlapping select months** (select 1mo, step 1mo): every roll adds
  an *independent* month to the pool. Longer select windows with monthly
  steps would share months across rolls and double-count them in the
  pooling — the current split is the one where fixed-effect pooling is
  honest.
- **Retention** (`reseed_promoted_rolls`): candidates promoted in the last
  N rolls are re-seeded into the search even after missing a survivor cut.
  Every seed is re-measured and ledger-recorded whether or not it
  re-survives, so one noisy train month never discards an accumulated
  evidence stream — the signal re-enters the book when it re-earns
  survival.
- **Pooled direction fitting**: a single 5-month train window fits the
  wrong traded sign Φ(−SR·√years) of the time — ~26% for a Sharpe-1
  signal, and consecutive windows overlap so the error persists. The sign
  is therefore fitted from the candidate's pooled *train* evidence across
  every roll it was measured in (train-only — select stays unspent), so
  direction error falls with candidate age exactly as the select evidence
  grows. A candidate whose current window disagrees with its history keeps
  the historical sign and pays for the disagreement in this roll's
  directed reward.
- **Family horizons**: each candidate is only ever measured where its
  family's alpha can live, so its evidence concentrates instead of
  spreading across implausible lags.

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

The in-run controls (train/select split, held-out months, fixed promotion
count) do not cover the research process itself: tuning config across runs
spends the select window's honesty. A deflated Sharpe ratio, a
backtest-overfitting probability, and a never-touched lockbox period are the
next step.

## Run

```bash
uv run research/signals/discovery.py            # full history
uv run research/signals/discovery.py --resume   # continue an interrupted run
uv run research/signals/inspect_discovery.py    # review a run
uv run research/signals/event_study.py          # slow families, full history
```

The event-study script is the right instrument for the slow families
(unlocks / listings / dev activity): a monthly select window holds a handful
of their events, which no honest gate can confirm. It pools every event over
the full history and reports pre/post cumulative residual-return curves with
day-clustered t-stats (`discovery.event_study` in config). Diagnostic only —
a real drift there motivates a curated signal, not a DSL candidate.

Promoted signals become `disc_*` registry entries (`research/lib/discovered.py`).
The walk-forward mirrors discovery month by month: each roll's promoted signals
are traded in that roll's OOS month only. All knobs live under `discovery.*` in
`config.py`.

## Cost and run time

A full run is ~40 rolls (windows) × 16 generations. The default model is
Gemini 3.1-flash-lite ($0.25 / $1.50 per million input / output tokens),
measuring ~$1/roll — order $40 for the full run. Cheaper providers are one
config switch away (`discovery.llm.provider` + the key in `.env`):
`openrouter` (DeepSeek V4 Flash, ~$0.42/roll) and `xai` (Grok 4.1 Fast,
~$0.63/roll) share a plain OpenAI-compatible client; `base_url` covers any
other compatible endpoint. The script prints the measured tokens and dollars
per roll; trust that over these estimates.

Run time is dominated by candidate scoring (~15–20s per candidate on the
5-month train panel), not the LLM: proposal calls run concurrently
(`discovery.llm.parallel_requests`, default 8). Expect roughly 2–2.5 hours per
roll, ~2–3 days for the full 28. Progress bars (per-generation, per-call,
per-candidate) show the run is alive; lower parallel_requests if the provider
rate-limits.

