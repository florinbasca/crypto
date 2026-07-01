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
  signal = -sign(res_zscore) * abs(res_zscore)
  active only when volume_shock < median and abs(oi_change_zscore) < 0.5

Validation:
  evaluate by horizon, costs, turnover, liquid-half IC, walk-forward.
```

This is nonlinear and conditional, but still interpretable.

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
  "expression": "-sign(res_zscore) * abs(res_zscore)",
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

Reward should not be just IC. Use something like:

```text
reward =
  IC_tstat
  + walk_forward_sharpe_component
  + liquid_half_bonus
  + orthogonality_bonus
  - turnover_penalty
  - complexity_penalty
  - instability_penalty
  - similarity_penalty
```

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

## Warning

The more nonlinear the search, the higher the overfit risk. Nonlinear methods can discover real conditional structure, but they can also manufacture convincing nonsense in noisy financial data.

Complexity must be paid for. A tree with five conditions needs much stronger out-of-sample evidence than a one-column residual reversal signal.

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
