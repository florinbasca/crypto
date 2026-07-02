# Agentic Signal Discovery

The hard part is not asking an LLM for signal ideas. The hard part is controlling the search space. You do not let an agent explore “all possible inputs.” You define a hypothesis language, a search prior, and a budgeted exploration algorithm. Without those, the search space is effectively infinite and the agent will overfit.

## Relevant Algorithms

### Symbolic Regression / Genetic Programming

This is the classic approach: search over mathematical expressions composed from allowed variables and operators.

Example search grammar:

```text
signal := zscore(expr)
expr   := feature
        | residual_path_stat
        | expr + expr
        | expr - expr
        | expr / abs(expr)
        | rank(expr)
        | rolling_mean(expr, window)
        | rolling_std(expr, window)
        | where(condition, expr1, expr2)
```

The algorithm evolves candidate expressions, scores them, mutates good ones, and kills bad ones. Symbolic regression addresses the “infinite possible formulas” problem by restricting primitives and penalizing complexity.

For this repo, the agent should not invent arbitrary code. It should search a controlled DSL over residuals, features, rolling transforms, ranks, gates, and interactions.

### LLM + Monte Carlo Tree Search

Recent alpha-mining literature uses LLMs with MCTS. The LLM proposes promising branches; MCTS decides where to spend more evaluations based on backtest feedback. This is a direct answer to “how do we explore without infinite branching?”

A close algorithmic template is:

```text
state = current partial signal expression
action = add transform / add condition / choose feature / choose horizon
reward = validation score after evaluation
```

The search expands promising branches while still reserving some exploration budget for less-tested branches.

### Evolutionary LLM Search

AlphaEvolve-style systems use LLMs to generate variants of candidate programs, evaluate them, then keep the best variants for future mutation. The important part is not the LLM. The important part is the evaluator and the selection pressure.

Pattern:

```text
population of candidate signal programs
-> LLM mutation / recombination
-> run validation
-> rank by reward
-> keep diverse survivors
-> repeat
```

### Automated Feature Engineering

AutoFeat, Deep Feature Synthesis, tsfresh-style extraction, and related methods generate many nonlinear features from primitive columns, then select a small robust subset.

This is relevant if the agent is trying to discover transformed residual features:

```text
residual volatility ratio
residual drawdown duration
residual autocorr decay
residual shock after low liquidity
cluster-relative residual spread
```

### Bandits / Bayesian Optimization

If evaluations are expensive, treat each candidate family as an arm. Allocate more budget to families that are producing useful discoveries, while keeping some exploration budget for neglected families.

Example families:

```text
residual shape
liquidity gated
funding/OI interaction
cluster relative value
volatility regime
nonlinear tree models
```

This formalizes the exploration/exploitation tradeoff.

### High-Dimensional Screening

Before searching complex combinations, use screening to reduce the candidate input set. Sure Independence Screening and iterative variants reduce ultrahigh-dimensional feature spaces before fitting more expensive models.

For this repo, first rank primitive columns by simple, robust diagnostics:

```text
rank IC by horizon
mutual information
distance correlation
conditional IC by regime
stability across time windows
liquid-half IC
turnover
missingness
```

Then let the agent explore combinations only among the top few candidates per family.

## The Critical Design Point

There are two spaces:

```text
input space = what raw data/columns/transforms are allowed
hypothesis space = what forms of relationships are allowed
```

Neither should be open-ended.

Good primitive data groups for the crypto repo:

```text
1. residual path: residual_return, cumulative residual, zscore, drawdown, runup, autocorr
2. volatility: residual vol, raw vol, vol ratios, vol shocks
3. liquidity: volume, dollar volume, spread proxies, Amihud, zero-volume share
4. derivatives: funding, OI, taker imbalance, top trader positioning
5. cross-sectional context: ranks, cluster-relative values, dispersion
6. factor context: betas, beta drift, factor residual correlations
```

Allowed transforms:

```text
rank
zscore
rolling mean/std/skew/autocorr
difference
ratio
spread
winsorize
condition/gate
interaction
regime split
cluster relative difference
```

Allowed model classes:

```text
expression
conditional expression
small tree
spline/GAM
random forest / gradient boosted tree with strict rolling training
small regularized model
ensemble of validated weak signals
```

That is how you make the infinite finite.

## Nonlinear Relationships

The current IC framework mostly tests monotonic relationships. Rank IC asks: “when signal rank is higher, are future residual returns generally higher?” That can miss nonlinear relationships like:

```text
both extreme positive and extreme negative residual shocks predict reversal
only middle liquidity names work
signal works only when vol is high
U-shaped relationship
threshold effect
interaction between residual move and OI
```

The agent should not rely only on linear/rank IC during discovery.

Use additional nonlinear discovery tests:

```text
1. Binned forward returns
   Sort candidate variable into deciles and inspect mean fwd residual by bin.
   This catches U-shapes, thresholds, and sign flips.

2. Conditional IC
   Measure IC separately in high-vol, low-vol, high-liquidity, low-liquidity,
   high-funding, low-funding regimes.

3. Mutual information
   Measures dependence beyond linear/monotonic correlation, though it is noisy.

4. Distance correlation / HSIC
   Kernel-style dependence tests that can detect nonlinear dependence.

5. Tree-based probes
   Fit shallow trees or gradient-boosted stumps on rolling training windows.
   Use them for discovery, not immediate production.

6. Partial dependence / accumulated local effects
   Ask whether the fitted nonlinear model learned a stable shape.

7. Interaction tests
   Does residual zscore matter only conditional on funding, OI, liquidity, or vol?
```

A good nonlinear agent loop:

```text
screen primitive variables nonlinearly
-> detect shape: monotonic, U-shaped, threshold, interaction, regime-specific
-> propose candidate representation
-> evaluate with walk-forward residual targets
-> penalize complexity and instability
```

Example:

```text
Discovery:
  residual_zscore alone has weak IC.
  But binned analysis shows extreme residual_zscore reverses only when
  volume shock is low and OI is flat.

Candidate:
  # genuinely nonlinear: reverses HARDER on bigger extremes (z**2, not |z|)
  signal = -sign(res_zscore) * res_zscore**2
  active only when volume_shock < median and abs(oi_change_zscore) < 0.5

Validation:
  evaluate by horizon, costs, turnover, liquid-half IC, walk-forward.
```

Watch the algebra: `-sign(z) * abs(z)` collapses to plain `-z`, a *linear*
reversal. Real nonlinearity has to come from the expression itself (the `z**2`
above, a threshold, or a `where` gate) or from the conditions - not from a formula
that only looks nonlinear. This candidate is nonlinear and conditional, but still
interpretable.

## Validation Protocol (the walk-forward backbone)

Every candidate is judged on a rolling walk-forward with THREE separate windows.
This is the anti-overfit backbone - without it, the search fools itself.

```text
|<---------- TRAIN 5 months ---------->|<-- SELECT 1 mo -->|<-- OOS 1 mo -->|
 generate / fit candidates               keep survivors      trade the book
```

- **TRAIN (5 months):** the search generates and fits candidate signals here.
  Expression signals only need their traded sign fixed; model-class candidates
  (Stage 4) fit their parameters here. Nothing is judged on train alone.
- **SELECT (1 month):** score every candidate here and keep the survivors. This is
  the ONLY window the search reward (Stage 3) is computed on. Because the search
  tries many candidates against it, this month gets "used up" (overfit) - that is
  acceptable, because it is not the final judge.
- **OOS / BACKTEST (1 month):** assemble the surviving signals into a
  market-neutral portfolio and record THIS month's PnL. The search never optimizes
  against this window. This month is the product, not a check.

Then roll forward one month and repeat. **Stitching every OOS month end-to-end
gives the historical PnL curve** for the whole discovery system - exactly how
`walk_forward.py` stitches its out-of-sample blocks today.

**Purge + embargo:** forward residual targets span up to ~2 days (288 bars), so
the tail of each window overlaps the next. Drop the last (max_horizon + embargo)
bars of TRAIN before SELECT, and of SELECT before OOS, so a candidate is never
scored on labels that leak across a boundary.

**Reused methodology (same math as production, separate system):** rank IC per
timestamp, Newey-West HAC IC t-stat, Benjamini-Yekutieli FDR, and greedy
de-correlation are the same tools `evaluate.py` / `walk_forward.py` use. The
discovery engine calls the same functions - it does not re-run those scripts.

All window lengths, the embargo, and every threshold live in `config.py` (a new
`discovery` group), read via `config.get('discovery.<...>')` - never hardcoded.

## The Harness

The harness is the deterministic Python program that surrounds the LLM and turns
"an agent proposes signals" into "tested, ranked, promoted signals with a PnL
curve." The LLM is only ONE part inside it - the idea generator. The harness is
everything else, and it is where all the correctness lives.

The single most important rule: **the LLM never touches data, never runs code, and
never decides what is good.** It only reads compressed diagnostics and emits
candidate signals in the DSL. Everything that could be gamed - evaluation,
scoring, the train/select/OOS split, promotion - is done by fixed code the LLM
cannot reach. That is what stops the search from fooling itself.

### Components

```text
1. Diagnostics builder  - computes per-feature stats (rank IC, binned forward
                          curves, conditional IC by regime, stability) and hands
                          the LLM a COMPRESSED view. The only thing the LLM sees.
2. Proposer (the LLM)   - reads diagnostics + current best candidates, emits a
                          batch of new candidates as DSL JSON. In genetic terms,
                          the mutation/recombination operator. Swappable; untrusted.
3. Compiler             - turns a DSL candidate into a per-(timestamp, symbol)
                          signal panel, then runs the causality (truncation) test;
                          anything that peeks at the future is rejected.
4. Evaluator            - runs each candidate through the train/select/OOS
                          walk-forward (see Validation Protocol), with purge/
                          embargo, using the same rank-IC / HAC-t-stat math as
                          the production pipeline.
5. Scorer / reward      - collapses the SELECT-month metrics into one reward
                          number (weighted, standardized terms; SELECT only).
6. Search controller    - the budgeted loop that decides what to try next: keep
                          the best + diverse survivors, ask the proposer to mutate
                          them, repeat until the candidate budget is spent. This
                          is where the genetic/evolutionary or MCTS logic lives.
7. Promoter / portfolio - takes survivors, checks they add INCREMENTAL edge over
                          the current book, assembles them market-neutral into the
                          OOS month, records that month's PnL, and rolls forward.
                          Stitched OOS months = the equity curve.
```

**Cross-sectional by construction (hard requirement).** Every generated signal is
**portfolio-wide demeaned at each timestamp, vectorized** - i.e. a
`groupby('timestamp')` transform that subtracts the cross-sectional mean across
all symbols in the universe. Never a per-symbol loop, and never per-symbol
time-series testing. A signal is a *cross-section*: at each bar it ranks names
against each other and sums to ~0 across the universe (dollar/market-neutral in
signal space). Scoring is likewise cross-sectional - rank IC across symbols per
timestamp - not a per-symbol IC. The compiler applies this demean (then the same
cross-sectional z-score + clip normalization `evaluate.py` uses) as the last step
before any candidate is scored; a candidate that cannot be expressed as a
cross-sectional panel is out of scope.

### The loop

The harness runs one **search loop per roll** (a roll = one train/select/OOS
window set, per the Validation Protocol). At a glance:

```text
diagnostics ──► [LLM proposes] ──► compile + causality gate
     ▲                                    │
     │                                    ▼
 search controller ◄─ reward ◄─ score(select) ◄─ evaluate(train→select)
     │
     └─(budget spent)─► promote survivors ─► trade OOS month ─► roll +1 month
```

Written out step by step:

```text
FOR each roll (train 5mo / select 1mo / OOS 1mo, advancing 1 month at a time):

  1. BUILD DIAGNOSTICS
     Compute per-feature stats (rank IC, binned forward curves, conditional IC
     by regime, stability) on the TRAIN window. Compress them for the LLM.

  2. SEARCH  ── repeat until the candidate budget for this roll is spent ──
     a. PROPOSE   the LLM reads the compressed diagnostics + the current best
                  candidates, and emits a batch of new candidates in the DSL.
     b. COMPILE   each candidate -> cross-sectional signal panel; run the
                  causality (truncation) test; drop anything that peeks ahead.
     c. EVALUATE  fit on TRAIN, score on the SELECT month (purge/embargo at the
                  boundary). Cross-sectional rank IC, cost-aware Sharpe, etc.
     d. REWARD    collapse the SELECT metrics into one reward (weighted,
                  standardized terms; SELECT only - never OOS).
     e. SELECT    keep the best + most diverse survivors; feed them back to (a)
                  as parents for the next mutation/recombination round.
     ── end SEARCH ──

  3. PROMOTE
     Take the survivors that clear the gates + add INCREMENTAL edge over the
     current book. This is done ONCE per roll.

  4. BACKTEST (OOS)
     Assemble the promoted signals into a market-neutral portfolio and trade the
     OOS month. Record that month's PnL. The search never saw this month.

  5. ROLL
     Advance the window by one month and go to step 1.

AFTER all rolls: stitch every OOS month end-to-end -> the historical PnL curve.
```

Two nested loops: the **outer** loop rolls the window forward month by month; the
**inner** SEARCH loop (step 2) is the budgeted propose->test->keep-survivors cycle
where the genetic/evolutionary or MCTS logic lives.

### What the harness guarantees

```text
Honesty          - the leakage protocol (train/select/OOS separation, purge/
                   embargo, OOS-scored-once) is enforced in code, not left to the
                   LLM's good behavior.
Boundedness      - candidates can only reference real feature columns and valid
                   transforms (the DSL), so no hallucinated inputs.
Reproducibility  - deterministic given a seed; search state is persisted so a run
                   can resume.
Comparability    - reuses the same IC / t-stat / de-correlation functions as
                   evaluate.py / walk_forward.py, so a discovered signal's numbers
                   mean the same thing as a production signal's.
```

The harness is a **separate program the user runs** (like `evaluate.py` today);
the LLM is a component invoked inside it, not the thing in charge. The stages
below detail each component.

## Proposed Agent Search

### Stage 1: Primitive Discovery

For every primitive column and residual-derived feature:

```text
compute rank IC
compute binned forward residual curve
compute mutual information
compute regime-conditional IC
compute stability across time thirds
```

Output:

```text
feature family -> promising columns -> likely shape -> likely horizon
```

### Stage 2: Hypothesis Proposal

The LLM sees only compressed diagnostics, not the full dataset. It proposes candidates in a DSL.

Example:

```json
{
  "kind": "conditional_expression",
  "name": "low_volume_residual_extreme_reversion",
  "expression": "-sign(res_zscore) * res_zscore**2",
  "conditions": [
    "cs_rel_volume < 0",
    "res_vol_short / res_vol_long > 1.2"
  ],
  "expected_horizon_bars": [6, 18, 36],
  "rationale": "Residual extremes on weak participation are less likely to reflect informed repricing."
}
```

### Stage 3: Search Algorithm

Use MCTS or evolutionary search.

Reward should not be just IC. Use a weighted sum, computed on the SELECT month only:

```text
reward = Σ  w_k · standardized(term_k)          # weights w_k live in config.py

  + IC_tstat                (HAC t-stat of the daily IC series)
  + walk_forward_sharpe     (cost-aware standalone Sharpe on SELECT)
  + liquid_half_bonus       (IC on the liquid half - robustness)
  + orthogonality_bonus     (low corr vs the already-accepted book)
  - turnover_penalty
  - complexity_penalty      (node + condition count)
  - instability_penalty     (|train_IC - select_IC|)
  - similarity_penalty      (max corr vs accepted signals + existing spaces)
```

Two things make this actually computable:

1. **Scale.** The terms live on different scales (a t-stat is O(1-5), a Sharpe
   O(0-3), penalties unbounded), so each is standardized (robust z-score across
   the current candidate batch) before weighting - otherwise one term dominates
   and the sum is meaningless.
2. **Weights are config, not hardcoded**, so they can be tuned.

Reward is computed on the SELECT window ONLY. It must never touch the OOS month -
that is what keeps the stitched backtest honest.

Similarity penalty matters. Otherwise the agent will generate many near-duplicates.

### Stage 4: Nonlinear Probe Models

For candidate families that look nonlinear, allow constrained models:

```text
shallow decision tree
monotonic gradient boosting
GAM / spline
small random forest
regularized logistic/linear model on nonlinear basis features
```

These should be trained causally:

```text
train on past window
predict next window
roll forward
never fit on full sample
```

Their output should be converted back into a single signal panel for normal portfolio validation.

### Stage 5: Promotion

Only promote candidates that survive:

```text
nonlinear diagnostic
rolling OOS evaluation
costs
liquidity
orthogonality
walk-forward portfolio contribution
```

Promotion is where survivors become the portfolio. A promoted signal joins the
OOS-month book (the Validation Protocol above): survivors are combined
market-neutral, that month's PnL is recorded, and the roll advances - the stitched
OOS months are the system's live-equivalent track record. A candidate earns its
slot only if it adds INCREMENTAL return to the current book, not just standalone
edge, after de-correlating against signals already promoted and the existing
curated spaces.

## Warning

The more nonlinear the search, the higher the overfit risk. Nonlinear methods can discover real conditional structure, but they can also manufacture convincing nonsense in noisy financial data.

Complexity must be paid for. A tree with five conditions needs much stronger out-of-sample evidence than a one-column residual reversal signal.

There is a second, sneakier overfit: the search itself. Trying thousands of
candidates against the same SELECT month overfits THAT month even if every
individual signal is simple. Track how many candidates were evaluated and deflate
the survivors' t-stats / Sharpes accordingly (a Deflated-Sharpe-style haircut),
and apply FDR across the promoted set. Per-signal complexity penalties do not
catch this - it is a property of the search, not of any one signal.

Practical recommendation:

```text
Start with:
  symbolic / conditional DSL + MCTS/evolutionary search

Add later:
  shallow tree/GAM probes for nonlinear shape discovery

Avoid initially:
  deep learning agents directly trained to predict residuals
```

Deep models will probably look impressive in-sample and fail quietly unless the validation infrastructure is extremely strict.

## Bottom Line

There are algorithms for this: symbolic regression, genetic programming, MCTS, evolutionary LLM search, automated feature engineering, Bayesian optimization, bandits, and high-dimensional screening.

The right design for this repo is not “LLM brainstorms signals.” It is:

```text
bounded DSL
+ nonlinear primitive diagnostics
+ budgeted search algorithm
+ strict causal evaluator
+ complexity/similarity penalties
+ walk-forward promotion
```

That gives the agent room to find nonlinear relationships without letting the search explode into infinite overfit.
