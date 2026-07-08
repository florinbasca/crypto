# Agentic Signal Discovery

The hard part is not asking an LLM for signal ideas. The hard part is controlling the search space. You do not let an agent explore “all possible inputs.” You define a hypothesis language, a search prior, and a budgeted exploration algorithm. Without those, the search space is effectively infinite and the agent will overfit and overspend.

## Relevant Algorithms

### Symbolic Regression / Genetic Programming

Search over mathematical expressions composed from allowed variables and operators.

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

For this repo, the agent should not invent arbitrary code. It should search a controlled space over residuals, features, rolling transforms, ranks, gates, and interactions.

### LLM + Monte Carlo Tree Search

Some alpha-mining systems grow a tree of partial formulas and spend more search
budget on the branches that keep scoring well (an explore/exploit rule decides
where — see the Bandits section).

**We do NOT use it.** MCTS only helps if a partial formula's score predicts its
completions' scores, and for signals it doesn't (`res_zscore` alone scores badly
while a gated `-res_zscore**2` scores well). v1 uses evolutionary search instead
(only complete candidates are ever scored), with a family bandit for the
budget-steering benefit at a level where scores actually transfer.

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

### Bandits / Bayesian Optimization  (this repo uses this)

("Bandit" is slang for a slot machine — a "one-armed bandit." The "multi-armed
bandit" is the classic problem of a row of machines with unknown payouts and a
limited number of pulls: which arm do you play to win the most?)

Evaluations are expensive (each candidate = an LLM call + compile + score) and
there are many feature families, so the budget can't be spread evenly or dumped
on one early winner. Treat each family as a machine ("arm") and allocate
each generation's proposal budget with **UCB (Upper Confidence Bound)**: score
each family by its average reward so far PLUS a bonus that is large when the
family has been tried little and shrinks as it is tried, then give the next slot
to the highest score. The average-reward term exploits what is working; the
bonus keeps exploring the uncertain families so a slow-starter is not written
off on a noisy early batch.

```text
score(family) = avg_reward + c * sqrt( ln(total_tries) / tries(family) )
```

Example families:

```text
residual shape
liquidity gated
funding/OI interaction
cluster relative value
volatility regime
nonlinear tree models
```

That is the exploration/exploitation tradeoff, and it is what `allocate_batch`
in `search.py` implements.

### High-Dimensional Screening

Before searching complex combinations, use screening to reduce the candidate input set. Sure Independence Screening and iterative variants reduce ultrahigh-dimensional feature spaces before fitting more expensive models.

For this repo, primitive columns are ranked each roll by robust diagnostics -
implemented: rank IC by horizon, conditional IC by regime, stability across time
thirds, liquid-half IC. (Distance correlation / mutual information are further
options, not built.) The top few per family are what the LLM sees and explores
combinations over.

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
3. liquidity/cost: dollar volume, avg trade size, spread proxies, Amihud, turnover
4. order flow: taker imbalance (OFI), signed volume, Kyle's lambda, VPIN-style
   toxicity, flow persistence — DIRECTIONAL aggressive-flow, largely orthogonal
   to price signals (independent breadth, its own bandit arm)
5. efficiency: efficiency ratio, variance ratio, reversal tendency, diffusion
   speed — clean information diffusion vs noisy overshoot (great as a gate)
6. derivatives: funding, OI, top-trader positioning
7. cross-sectional context: ranks, cluster-relative values, dispersion
8. factor context: betas, beta drift, factor residual correlations
9. distribution shape: skew, kurtosis — crash/squeeze regime gates
10. tokenomics: supply inflation / unlock pressure; log market cap (size gate)
11. trend state (classic price TA — momentum quality, RSI, MACD, Bollinger,
    ADX, ATR): weak standalone on residual returns, used mainly as GATES
12. calendar / events / macro: cross-sectionally CONSTANT — gate-only
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

Model classes:

```text
IMPLEMENTED (what a candidate can be):
  expression            e.g. cs_zscore(fr_annualized / roll_std(...))
  conditional expression  the same, gated by a `where` condition

NOT a candidate class (design options):
  small tree, spline/GAM, random forest, gradient boosted tree, ensemble.
  The only tree model in the code is the GBM ML probe (Stage 4), which
  is a diagnostic - it is never promoted or traded.
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

So the search must not rely only on rank IC. It handles nonlinearity two ways,
both live in the code:

```text
IMPLEMENTED:
1. Binned forward returns  (build_diagnostics)
   Each feature's mean forward residual by decile is shown to the LLM, so it
   can see U-shapes, thresholds and sign flips and PROPOSE a matching form
   (square, a gate) - which the DSL then compiles.
2. Conditional / regime IC  (build_diagnostics)
   Each feature's IC split high/low by vol, crowding, cross-asset risk, and
   event proximity - the material for regime-gated candidates.
```

The DSL expresses the nonlinearity (square, `where` gates, feature
interactions); rank IC then scores the resulting nonlinear signal. So the
nonlinearity is in the CANDIDATE, measured with the same rank IC - not a
separate battery of dependence tests.

```text
NOT IMPLEMENTED (design options, kept for reference):
  mutual information, distance correlation / HSIC (nonlinear dependence tests);
  partial dependence / ALE (shape stability of a fitted model);
  formal interaction tests. The LLM proposes interactions from the regime
  diagnostics instead of a test enumerating them.
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
  score rank IC per horizon and liquid-half IC on train; test on the held-out
  select month. Cost is judged later, at the portfolio layer, never here.
```

Watch the algebra: `-sign(z) * abs(z)` collapses to plain `-z`, a *linear*
reversal. Real nonlinearity has to come from the expression itself (the `z**2`
above, a threshold, or a `where` gate) or from the conditions - not from a formula
that only looks nonlinear. This candidate is nonlinear and conditional, but still
interpretable.

## Validation Protocol (the walk-forward backbone)

Every candidate is judged on a rolling walk-forward with TWO working windows.
This is the anti-overfit backbone - without it, the search fools itself.

```text
|<---------- TRAIN 5 months ---------->|<-- SELECT 1 mo -->|  (OOS 1 mo = valid_from)
 search: propose / score / breed         promotion tests    the walk-forward
 (reward, survival, direction)           this ONCE           trades from here on
```

- **TRAIN (5 months):** the ENTIRE search happens here - propose, score, reward,
  keep survivors, breed. Reward, traded direction, and the alpha half-life are all
  computed on train only. The search never sees the select month.
- **SELECT (1 month):** held out from the search. Promotion is the FIRST and ONLY
  look at it - each survivor is tested for significance here exactly once. Because
  the search never optimized against it, a select t-stat is an honest measurement,
  not the maximum of a search on itself. (This is the key change from the old
  design, where the reward was computed on SELECT and the window got "used up".)
- **OOS (1 month):** discovery does NOT trade it. It is only the promotion's
  `valid_from` date - the earliest month the walk-forward may use a promoted
  formula (its expression was chosen with data up to here, so using it earlier
  is look-ahead). Discovery emits promotions; the walk-forward is the only money
  judge. There is no discovery-side PnL, backtest, or stitched equity curve.

Then roll forward one month and repeat. **Each month is independent:** promotion
depends only on that month's own train + held-out select, never on prior months
(`min_rolls_survived: 1`).

**Purge + embargo:** forward residual targets span up to ~1 day (144 bars), so
the tail of each window overlaps the next. Drop the last (max_horizon + embargo)
bars of TRAIN before SELECT so a candidate is never scored on labels that leak
across the boundary.

**Reused methodology (same math as production):** rank IC per timestamp,
Newey-West HAC IC t-stat, Benjamini-Yekutieli FDR, and greedy de-correlation are
the same functions the production walk-forward uses (`research/lib/signal_eval.py`);
the discovery engine calls them directly.

All window lengths, the embargo, and every threshold live in `config.py` under
the `discovery` group, read via `config.get('discovery.<...>')` - never hardcoded.

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
1. Diagnostics builder  - computes per-feature stats (WHAT each column measures,
                          rank IC, binned forward curves, conditional IC by regime,
                          stability) on TRAIN and hands the LLM a COMPRESSED view.
                          The only thing the LLM sees.
2. Proposer (the LLM)   - reads diagnostics + current best candidates, emits a
                          batch of new candidates as DSL JSON. In genetic terms,
                          the mutation/recombination operator. Swappable; untrusted.
3. Compiler             - turns a DSL candidate into a per-(timestamp, symbol)
                          signal panel. Causality is guaranteed PER-OPERATOR
                          (every operator reads only data through bar t, proven
                          by tests/discovery_checks.py's truncation test over
                          the whole operator registry); the search path does not
                          re-run a per-candidate truncation check.
4. Evaluator            - scores each candidate at EVERY horizon on TRAIN and on
                          the held-out SELECT window, using the same rank-IC /
                          HAC-t-stat math as the production pipeline. Keeps the
                          full per-horizon IC profile + a fitted alpha half-life.
5. Scorer / reward      - collapses the TRAIN metrics into one reward number
                          (weighted terms, TRAIN only - see Stage 3). The select
                          window is never used for reward.
6. Search controller    - the budgeted loop that decides what to try next: keep
                          the best + diverse survivors, ask the proposer to mutate
                          them, repeat until the candidate budget is spent. This
                          is where the genetic/evolutionary logic lives (a bandit
                          over families steers the budget).
7. Promoter             - takes survivors, tests each ONCE on the held-out select
                          window (FDR / deflation / capture / orthogonality gates,
                          Stage 5), and writes those that pass to the promotions
                          table. It does NOT trade or compute PnL - the
                          walk-forward is the only money judge.
```

**Cross-sectional by construction (hard requirement).** Every generated signal is
**portfolio-wide demeaned at each timestamp, vectorized** - i.e. a
`groupby('timestamp')` transform that subtracts the cross-sectional mean across
all symbols in the universe. Never a per-symbol loop, and never per-symbol
time-series testing. A signal is a *cross-section*: at each bar it ranks names
against each other and sums to ~0 across the universe (dollar/market-neutral in
signal space). Scoring is likewise cross-sectional - rank IC across symbols per
timestamp - not a per-symbol IC. The compiler applies this demean (then the same
cross-sectional z-score + clip normalization the production signals use) as the last step
before any candidate is scored; a candidate that cannot be expressed as a
cross-sectional panel is out of scope.

### The loop

The harness runs one **search loop per roll** (a roll = one train/select/OOS
window set, per the Validation Protocol). At a glance:

```text
diagnostics(train) ──► [LLM proposes] ──► compile + causality gate
     ▲                                          │
     │                                          ▼
 search controller ◄─ reward(train) ◄─ evaluate(train, all horizons)
     │
     └─(budget spent)─► promote: test survivors ONCE on select ─► write promotions ─► roll +1
```

Written out step by step:

```text
FOR each roll (train 5mo / select 1mo, advancing 1 month at a time):

  0. ML PROBE (diagnostic)
     Fit a gradient-boosting model on ALL features per horizon (train->select):
     the upper bound on what any search could find. Barren features and a failed
     search look identical without it. Printed, never gates anything.

  1. BUILD DIAGNOSTICS
     Compute per-feature stats (what each column measures, rank IC, binned
     forward curves, conditional IC by regime, stability) on the TRAIN window.
     Compress them for the LLM.

  2. SEARCH  ── repeat until the candidate budget for this roll is spent ──
     a. PROPOSE   the LLM reads the compressed diagnostics + the current best
                  candidates, and emits a batch of new candidates in the DSL.
     b. COMPILE   each candidate -> cross-sectional signal panel (causality is
                  per-operator, proven once in tests - not re-checked here).
     c. EVALUATE  score on TRAIN at EVERY horizon (purge/embargo at the
                  boundary): cross-sectional rank IC per horizon -> the alpha
                  profile + fitted half-life. Direction fixed on train.
     d. REWARD    collapse the TRAIN metrics into one reward (capture-weighted
                  IC t-stat + liquid-IC ratio - penalties; TRAIN only).
     e. SELECT    keep the best + most diverse survivors; feed them back to (a)
                  as parents for the next mutation/recombination round.
     ── end SEARCH ──

  3. PROMOTE (once per roll)
     Test each survivor on the held-out SELECT window - FDR / deflation over the
     looks taken / min-t / capture floor / orthogonality (Stage 5). Write the
     passers to the promotions table with their profile + half-life.

  4. ROLL
     Advance the window by one month and go to step 0. Each month is independent.
```

The promoted formulas are the only output. The portfolio walk-forward
(`research/portfolio/walk_forward.py`) consumes them and is where all PnL,
costs, and the equity curve live.

Two nested loops: the **outer** loop rolls the window forward month by month; the
**inner** SEARCH loop (step 2) is the budgeted propose->test->keep-survivors cycle
where the genetic/evolutionary logic lives.

### What the harness guarantees

```text
Honesty          - the leakage protocol (train/select separation with the search
                   confined to train, purge/embargo, select-tested-once) is
                   enforced in code, not left to the LLM's good behavior.
Boundedness      - candidates can only reference real feature columns and valid
                   transforms (the DSL), so no hallucinated inputs.
Reproducibility  - deterministic given a seed; the ledger is persisted so a run
                   can resume.
Comparability    - reuses the same IC / t-stat / de-correlation functions as the
                   production walk-forward (research/lib/signal_eval.py), so a
                   discovered signal's numbers mean the same thing as a production
                   signal's.
```

The harness is a **separate program the user runs** (`discovery.py`);
the LLM is a component invoked inside it, not the thing in charge. The stages
below detail each component.

## Proposed Agent Search

### Stage 1: Primitive Discovery

For every primitive column and residual-derived feature (build_diagnostics):

```text
compute rank IC + HAC t-stat
compute binned forward residual curve (deciles)
compute regime-conditional IC (high/low vol, crowding, risk, event proximity)
compute stability across time thirds
attach a one-line description of what the column measures
```

Output:

```text
feature family -> promising columns -> likely shape -> likely horizon
```

The LLM gets FULL diagnostics only for the top few columns per family. Those
are ranked by a BLEND (each term rank-normalized within the family): monotonic
IC t-stat + decile-curve nonlinearity + regime spread + stability, plus a small
random quota. Ranking by monotonic t-stat alone would hide U-shaped,
threshold-only and regime-only features - exactly the nonlinear structure the
LLM is best placed to exploit - so the blend surfaces them instead.

### Stage 2: Hypothesis Proposal

The LLM sees only compressed diagnostics (including a one-line description of
WHAT each column measures), not the full dataset, and proposes candidates as
DSL S-expressions - nested JSON arrays, `["op", arg, ...]`, so nothing needs
parsing an expression string. The same hypothesis as before, in the real format
(reverse harder on bigger residual extremes, only on weak participation):

```json
{
  "family": "residual_shape",
  "expression": ["neg", ["square", ["col", "res_zscore"]]],
  "conditions": [["lt", ["col", "cs_rel_volume"], 0.0]],
  "rationale": "Residual extremes on weak participation are less likely to reflect informed repricing."
}
```

(`square` is sign-preserving `x*|x|`, so `neg(square(z))` is `-z*|z|` — a
reversal that bites HARDER on bigger extremes; the gate restricts it to weak
participation. Plain `neg(z)` would be a linear reversal — the nonlinearity has
to come from `square`, a threshold, or a `where` gate, never from a formula that
only looks nonlinear.)

**What the proposer is given (guided evolution, not blind mutation).** Each API
call is stateless, so the harness reconstructs the context every time. The
prompt carries:

```text
- a data dictionary: one line per column on WHAT IT MEASURES, so the model
  reasons about the mechanism instead of decoding an abbreviation
- per-column diagnostics on TRAIN (rank IC, decile curve, regime splits)
- current_parents WITH their scores (reward, day-equivalent IC t-stat, alpha
  half-life), ranked best-first - so the model pushes the directions that are
  working and drops the ones that scored near zero
- avoid_these: recently-culled low-scoring candidates - so it does not
  re-propose dead ideas
- overused_building_blocks: the most-tried subtrees this roll - so it varies
  the structure instead of piling onto one recipe
```

The LLM is the mutation/recombination OPERATOR; the harness runs the evolution
(scoring, keeping top survivors, the family bandit, culling). Without the
scores and failure memory the model could only diversify away from survivors;
with them it can steer up the reward gradient. It still never sees raw data,
and everything it returns is re-scored by fixed code — the feedback guides the
search, it cannot corrupt it. A `gemini_thinking_budget` gives the model room
to deliberate before emitting the JSON batch.

### Stage 3: Search Algorithm

Evolutionary search with a bandit over families. Reward is computed on the
TRAIN window ONLY (never SELECT), and is a weighted sum:

```text
reward = Σ  w_k · term_k / scale_k          # weights + scales live in config.py

  + ic_tstat          (CAPTURE-WEIGHTED day-equivalent train IC t-stat, see below)
  + liquid_ic_ratio   (IC on the liquid half / full-cross IC - a capacity read)
  + incremental       (train IC the candidate ADDS to the current survivor book)
  - complexity        (node + condition count)
  - instability       (std of the IC across the three thirds of TRAIN)
  - similarity        (max corr vs the survivors kept so far this generation)
```

Cost is NOT a term. A signal is scored gross - cost is a property of the
portfolio (judged in the walk-forward), never of a signal. IC is the only
performance measure.

The **incremental** term rewards MARGINAL edge over redundancy: the survivors
are pooled into one alpha and the candidate's contribution is the combined
pooled IC minus the book's own IC, at its best lag, on TRAIN. A strong signal
that just re-expresses the dominant factor scores ~0 here; a weaker but
orthogonal one scores high. This is the AlphaGen pooled-IC reward / Harvey-Liu
"Lucky Factors" incremental test, done in-sample so the select window stays
untouched. It complements (does not replace) standalone IC.

Two design choices make the IC term honest across horizons:

1. **Day-equivalent t-stat** (`t / sqrt(stamps-per-day)`): a raw t-stat grows
   with the number of bets, so a 1h signal would look ~sqrt(24)x stronger than a
   24h one for the same per-bet edge. Dividing that out puts every horizon on one
   scale. The candidate's best horizon (by this fair t) sets its reward and sign.
2. **Capture weight** `1/(1 + phi/kappa)` (phi = ln2 / fitted half-life,
   kappa = the book's trade rate): the fraction of a signal's IC a slow book can
   actually hold long enough to harvest. A 6h-half-life signal outscores an
   equally-strong 1h one, so the search breeds toward persistence. Duration, not
   a cost model.

Weights and scales are config, not hardcoded, so they can be tuned. Because the
search never touches SELECT, the promotion t-stats stay honest.

**Diversity is enforced on two axes**, because formula homogenization (the
population collapsing onto one recipe) is the field's central failure mode.
(a) OUTPUT: survivors must de-correlate (train-signal corr <= diversity_max_corr)
and the reward penalizes similarity. (b) STRUCTURE: a survivor whose AST
(subtree) overlap with a kept one exceeds diversity_max_ast_sim is dropped -
this catches clones that share a concrete building block yet slip under the
correlation ceiling. The over-mined subtrees this roll are also fed back to the
LLM each generation ("overused_building_blocks") with a vary-away instruction
(frequent-subtree avoidance).

### Stage 4: ML Probe (predictability ceiling)

A gradient-boosting model is fit on ALL features each roll (train -> select),
per horizon, and its select IC is printed. This is a **ceiling diagnostic, not
a signal path** - its predictions are never promoted or traded. It answers the
one question the DSL search cannot answer about itself: is there anything to
find at this horizon at all?

```text
IC(ceiling) ~ 0   -> the features are barren at this horizon; a search there is
                     digging in empty ground no matter how clever the DSL.
IC(ceiling) high  -> predictability exists; if the DSL search finds nothing, the
     but search      SEARCH is the bottleneck (budget, operators, prompt), not
     finds nothing   the data.
```

It is fit causally (train past, score the held-out select window, never the full
sample) and degenerate columns (all-NaN / constant in the window) are dropped so
the fit cannot crash on them. Because it only prints a number and gates nothing,
it adds no multiplicity to the promotion statistics.

### Stage 5: Promotion

Promotion is the ONE look at the held-out select window. A survivor is promoted
if ANY horizon of its select profile clears the statistical gates:

```text
FDR            Benjamini-Yekutieli across every (survivor, horizon) select p-value
               (Student-t with n_days-1 dof - a 1-month select is ~30 daily obs)
deflation      |t| must clear deflation_mult * E[max|N(0,1)|] over the ACTUAL
               looks promotion takes at select (survivors x horizons) - the
               search adds no looks because it never touched select
min-t/days      minimum select IC t-stat and minimum daily observations behind it
sign agreement  the train profile must mostly share the traded sign
capture floor   half-life persistent enough that the book can hold it (Stage 3)
orthogonality   low correlation vs the formulas already promoted this roll
```

Promotion does NOT trade, size, or compute PnL. It writes the passing formulas
(with their per-horizon profile, half-life, and direction) to the promotions
table. Those rows are the entire output of discovery. The portfolio walk-forward
decides whether a promoted formula earns a slot in the book - that is the only
place profitability is judged. Each roll re-forms its promotions from scratch on
its own train + select; there is no carryover between months.

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
