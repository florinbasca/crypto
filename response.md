# Why two rolls of discovery produced zero signals — and what to try next

Zero promotions in two rolls doesn't mean there is no strategy. Digging through the
ledger from the run, this outcome was close to mathematically guaranteed, for two
separate reasons that are both in the harness, not in the market.

## 1. The promotion gate cannot be passed by realistic alpha

Promotion demands a directed t-stat, on **~30 daily observations** (one select month),
that simultaneously clears BY-FDR at 0.10 across 60 looks (12 survivors × 5 lags), the
deflation bar `E[max|N|]` over 60 looks ≈ **2.9**, and min t 2.0
(`config.py`, `discovery.promotion`). The binding constraint is BY-FDR: with 60 looks
and the BY `ln(m)` penalty, the best candidate needs roughly p ≤ 3.6e-4, i.e.
**t ≈ 3.5** on one month.

The power math: a genuinely good signal with annualized Sharpe 2 has an *expected*
one-month t of about 2/√12 ≈ **0.58**. The probability it clears 3.5 in any given month
is ~0.2%. Even a Sharpe-3 signal passes ~0.5% of the time. Across 2 rolls × 12
survivors = 24 attempts, the expected number of promotions **in a world where every
survivor is a real Sharpe-2-to-3 signal is ~0.1**. Zero promotions is the modal outcome
whether or not alpha exists — the test carries almost no information. Three of the four
multiplicity controls (FDR, deflation, min-t) are charging for the same looks; stacked,
they price honesty so high that only a Sharpe-10 fluke month can pay it.

## 2. The search reward is culling the best candidates before promotion sees them

Sorting the ledger by select t:

- `macro_beta_4f89020e`: train t **5.66**, select t **6.60** (same sign, lag 72,
  turnover 0.7%/bar) — killed in search with reward **−1.86**.
- Meanwhile the actual survivors have train t 1.5–3.4 and select |t| mostly **< 1**,
  half sign-flipped.

For that candidate, the alpha term of the reward contributes only about +1.3
(day-equivalent t of 4, capture ~0.6, scale 2.0) — so hygiene penalties of −3+ must
have killed it, and the only terms that can get that large are **instability**
(weight −0.75, scale 0.0005) and similarity. The instability scale is 5bps in absolute
return units, so any signal with chunky per-bet alpha — precisely the event-gated,
bursty signals crypto actually offers — gets a penalty that dwarfs its alpha term. The
survivor pool this produces is visible in the inspect output: stable, low-variance,
near-zero-alpha candidates, almost all piled into lag 144 with 2,016-bar half-lives
(the capture weight selects for slowness, and the slow space appears to be empty). The
search is optimizing for *stable mediocrity*, then handing it to a gate only a miracle
can pass.

(The 13.5 and 8.0 "best select t" trials are sparse event-gated candidates — a handful
of correlated event days in one month, not real evidence. But 4f89020e is not that:
0.7%/bar turnover, consistent across both windows.)

## What to try, in order

1. **Rescale the reward's hygiene terms** so they can't dominate alpha. Make
   instability relative (std across train thirds ÷ the candidate's own |alpha|, i.e. a
   coefficient of variation) instead of absolute 5bps units, or cap each penalty term's
   magnitude. This is a config/`search.py` change and a ~$1, few-hour 2-roll rerun to
   validate — check whether high-select-t candidates now survive search.

2. **Restore statistical power at promotion by pooling evidence across rolls** instead
   of demanding a monster single month. Survivors already carry over as seeds and get a
   *fresh* select month each roll — that's accumulating independent evidence the gate
   currently throws away. Promote on pooled directed t across the months a candidate
   survived (e.g. pooled t ≥ 2.5 over ≥ 3 rolls) plus cross-month sign consistency,
   which is far more powerful for modest alphas than single-month magnitude. This is
   what `min_rolls_survived` was gesturing at before it was set to 1.

3. **Stop double-charging multiplicity.** Keep one calibrated control: either the
   deflation bar over actual looks *or* FDR — and consider BH instead of BY (lag
   profiles are positively correlated; BY's ln(m) penalty is for arbitrary dependence).
   Also cut looks by restricting each family to its natural horizons (slow families
   144/432 only, fast families 6–72), which lowers the bar and the FDR burden together.

4. **Accept what the survivors are telling you about the slow space.** Everything
   converged on lag 144 / half-life 2,016 bars with no select alpha — the capture
   weight is steering the search into a horizon band where this universe may have
   little to find. The alpha previously measured is fast (≤8h) and dies to 2bps taker
   fees. Two escapes that don't go through discovery at all: (a) **execution
   economics** — maker/passive fills change both the fee and the κ=0.048 fill-rate
   assumption jointly, and could make the already-demonstrated fast alpha capturable;
   (b) the **funding sleeve already found to be net-positive** — fixing its
   transmission through the portfolio layer is a known-positive-expectation project,
   unlike new discovery.

5. For the genuinely slow families (unlocks, listings, dev activity), the monthly-roll
   harness is the wrong instrument — a few events per select month can never clear any
   honest gate. Test those as **event studies over the full 2023–2026 history** with
   per-event stats, outside the discovery loop.

## Honesty caveat

As `signal.md` itself flags: recalibrating gates is legitimate when done from power
arithmetic (as above), but re-tuning thresholds *until specific candidates pass* spends
the select window. Fix the reward scaling and the pooling design on first principles,
rerun, and keep a lockbox period untouched.

**Headline:** the two rolls measured the harness, not the market. Fix the reward
scaling (cheap, item 1) and the promotion power (item 2) before drawing any conclusion
about whether the strategy exists.
