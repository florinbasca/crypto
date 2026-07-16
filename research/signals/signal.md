# Agentic Signal Discovery

What do I mean by signals: formulas that rank the universe each bar by expected residual return. The ranking is cross-sectional or scored against each other. Promoted signals are used in the walk-forward backtest, which sizes a dollar and factor-neutral book. The walk-forward is the only
place P&L is judged; discovery is purely statistical, refers to residuals and can't be traded directly.

The formula space is infinite and naive search overfits. Three controls: a bounded expression, an LLM proposing candidates, and promotion on a held-out **5-month test window** (~150 days the formula never saw — long enough for a single verdict to mean something; one month was a coin flip). Scoring and selection are deterministic.

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

Two windows per roll, advancing one month at a time (5 + 5 + 1):

```
| TRAIN 5mo | TEST 5mo  |   (OOS 1mo = the promotion's valid_from date)
  search happens here     the VERDICT: ~150 held-out days, read at promotion
```

- **Train**: the entire search — scoring, reward, breeding, half-life fits,
  and the traded direction (committed here, from pooled train evidence
  across the formula's rolls — never from the test).
- **Test**: held out. A formula's verdict is its most recent 5-month test
  window — one verdict per formula per roll, **no cross-roll pooling**: the
  long window IS the evidence. New formulas wait for their window to fill.
- **The verdict instrument is the response curve** (`discovery.curve`): the
  gross-1 book's cumulative return tracked bar-by-bar for 144 bars after
  entry, averaged over entries every 6 bars of the test window, then fitted
  (deterministically) to: **a0** (edge at the curve's peak), **half-life**
  (real decay — kills the saturated 4-point artifact), **peak_k** (where
  the response tops out; beyond it the alpha actively reverses) and
  **rev_frac** (how much is given back). A hump-shaped edge is positive at
  every legacy lag yet poison to a slow book — only the curve sees it. The
  curve never feeds the search reward (train-only) — it is read once, at
  promotion. Rows without curves (old ledgers) fall back to the 4-lag
  verdict.
- Windows slide monthly, so every month gets a fresh verdict and a fresh
  OOS month; consecutive test windows overlap 4 of 5 months (each verdict
  is still strictly causal for its own OOS month).
- A purge + embargo gap (max horizon + embargo bars) is dropped at the
  boundary so forward targets can't leak across it.
- A per-roll **feature coverage check** runs upstream of the LLM: features
  with ≤ `min_feature_nonnan` non-NaN values over the roll's window are
  dropped for that roll — never prompted, never compiled, never scored.

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
Stability pressure now lives where it is measured honestly: the 5-month
held-out verdict — a noisy formula rarely posts a clean positive verdict
over ~150 fresh days, and must re-earn one every roll.

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

## Promotion (CHOOSE)

Once per roll. A formula's **verdict** is its most recent 5-month test
window (per-bet return, not rank IC; directed by the sign committed on
train). Four filters, then the quintile — no significance gates, no fixed
counts:

1. **Made money** — the curve's peak edge a0 is positive in the committed
   direction, AND (`curve.median_gate`) the **median** entry outcome at the
   peak is positive — a formula whose whole profit is one jump day passes
   a mean, never a median. Directed, never |t|: a formula whose test ran
   backwards is rejected, not flipped — re-signing after seeing the test
   is how noise gets promoted. (~150 test days give a true Sharpe-2
   formula ~90% pass, Sharpe-1 ~74%, noise 50% — the filter halves noise;
   the quintile does the actual selecting.)
2. **Enough activity** (`min_select_days`) — enough real entry days within
   the test window for the curve to mean anything.
3. **Pays for itself** — the curve, judged at its own optimum, must cover
   a round trip at a positive rate: `max over k of
   (A(k) − roundtrip_cost)/k > 0` (round trip = `curve.roundtrip_mult` ×
   the cost rate, `econ_cost_bps` defaulting to the portfolio layer's
   cost model). AND holdable: capture at the book's measured fill rate ≥
   `min_capture`, with holding inputs **capped at the measured peak** —
   persistence past the point where the alpha reverses is worthless.
4. **Not a duplicate** (`max_book_corr`) — signal correlation vs formulas
   already chosen this roll, greedy best-first.

Then promote the **best quintile of the passers**: ceil(`book_frac` ×
n_passers), bounded by `book_min`/`book_max`, **ranked by net economic
rate at each formula's own optimal holding** (money-ordered, not
significance-ordered). Proportional — the book breathes with how much
quality exists.

Promotions are written with the verdict lag (= the curve's peak), peak
bars, half-life (**capped at the peak** — this is what the walk-forward
consumes for smoothing, holding and its capture discount, so the portfolio
is never told it may hold past the reversal), turnover, direction and the
economics (`select_alpha_tstat` = a0 vs its error bar, `test_days`,
`econ_margin` = net rate), plus a provenance stamp (`run_id`,
`config_hash`, `data_hash`, `git_sha`). Promotion neither trades nor
sizes.

The walk-forward consumes each roll's promotions at that roll's **evidence
lag** and **fitted direction** (never the registry's deduped defaults), and
converts per-bet alpha to per-bar linearly (returns scale with time; the
√h convention belongs to correlations). Null controls
(`walk_forward.py --control shuffle|sign_flip|random`) backtest placebo
books that the real book must clearly beat; control results are printed
but never persisted.

### Power: why a 5-month test window

One held-out month gives a true Sharpe-2 formula an expected t of ~0.58 —
statistically indistinguishable from noise, which is why the original
1-month-verdict design promoted junk or nothing. Five months (~150 days)
give the same formula E[t] ≈ 1.28, and the pass rates through filter 1
(net positive, directed) separate cleanly:

| true Sharpe        | 0 (noise) | 1   | 2   | 3   |
|--------------------|-----------|-----|-----|-----|
| P(pass filter 1)   | 50%       | 74% | 90% | 97% |

The filter halves the noise; the **quintile** then does the real selection
among passers, and re-qualification every roll (windows slide monthly) is
what noise cannot sustain. Supporting choices:

- **Pooled direction fitting** (train-only): the traded sign comes from the
  formula's pooled *train* evidence across its rolls, so direction error
  falls with age; the test window is never consulted for the sign.
- **Retention** (`reseed_promoted_rolls`): recently promoted formulas are
  re-seeded even after missing a survivor cut, so book members keep
  receiving fresh verdicts.
- **Family horizons**: each formula is measured only where its family's
  alpha can live, so verdicts concentrate instead of spreading across
  implausible lags.

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

A full run is ~36 rolls (windows; each spans 5+5+1 months, sliding monthly)
× 16 generations. The default model is
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

