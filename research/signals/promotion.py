"""
PROMOTION: rank the survivors on pooled held-out evidence, promote the top K.

Discovery is purely statistical - it measures per-bet returns (alpha in
return units), never traded PnL. The promotions this module emits are
consumed by research/portfolio/walk_forward.py (via
research/lib/discovered.py), which is the ONLY money judge: costs, execution
and the equity curve live there and nowhere else.

The design is deliberately minimal: RANK + K SLOTS + SANITY FLOORS.

Evidence: the search never touches the select window (train-only reward), so
each roll's select month is a clean, never-searched measurement. Promotion
pools EVERY month a candidate was ever measured on (fixed-effect
meta-analysis over the ledger; rolls advance monthly, so the months are
distinct and independent). Prior months were directed by that roll's own
train-fitted direction and are sign-aligned to the CURRENT direction before
pooling - evidence measured trading the other way counts against. Pooling is
where stability is priced: one month (~30 daily obs) gives a true Sharpe-2
signal an expected t of only ~0.58; k consistent months scale that by
sqrt(k), so persistence across rolls - not one lucky month - is what rises
to the top.

Ranking: POSTERIOR SHARPE x CAPTURE at the best of the candidate's FAMILY
horizons (family_horizon_lags keeps every look honest). Posterior Sharpe is
the observed pooled annualized Sharpe shrunk by the evidence behind it -
n / (n + 365/tau^2) with tau = prior_sharpe_std - so effect size (Sharpe),
statistical evidence (t, obs count) and cross-month stability (inconsistent
months cancel inside the pooled t) are combined in ONE number with ONE
interpretable constant, not a weight soup. A quota only substitutes for a
significance gate if the ranking metric is a sufficient measure of quality;
this is that metric. Never the search reward - the first LLM run measured
reward and select alpha as ~uncorrelated.

K slots (promotion.book_size): a fixed promotion count per roll. With a
fixed count there is nothing for significance gates to do - the quota
already caps discoveries, and the walk-forward re-judges the book every
month. (The previous design stacked FDR + deflation + min-t into an
accidental one-month bar of t~3.5 and promoted nothing, which carried no
information.) The book re-forms from scratch every roll: what accumulates
across rolls is evidence, never membership.

Sanity floors (reject the ACTIVELY bad, never enforce significance):
  - pooled directed t > 0: held-out months must not net-run AGAINST the
    traded direction (directed, not |t| - a reversed signal is rejected,
    never admitted on magnitude)
  - min_select_days: minimum pooled daily observations behind the evidence
  - min_profile_sign_agreement: the train term structure mostly shares the
    traded sign (a mixed profile is a red flag)
  - min_capture: persistence weight 1/(1 + phi/kappa) high enough that the
    book can hold the alpha long enough to matter (duration, never a cost
    model); max_turnover is its direct-measurement backstop for churners
  - min_rolls_survived: consecutive-roll survival (via the ledger)
  - max_book_corr: orthogonality vs the already-promoted book (greedy)
"""

import logging
import math
from typing import Dict, List, Optional

import numpy as np

from config import BARS_PER_DAY, get
from research.signals.data import Roll, family_lags
from research.signals.search import (DAYS_PER_YEAR, DiscoveryLedger,
                                           effective_persistence_bars,
                                           max_signal_correlation,
                                           persistence_weight,
                                           trade_rate_per_bar)


def pooled_select_evidence(months: List[dict], direction: int = 1) -> dict:
    """Fixed-effect meta-analysis of independent select months at one lag.

    months: [{'alpha_mean', 'alpha_tstat', 'n_days', 'direction'?}, ...] -
    one entry per roll the candidate was measured in (each roll's select
    window is a distinct month). A month whose 'direction' differs from the
    direction being pooled for is SIGN-FLIPPED first: its evidence was
    measured trading the other way, so it counts against.

    Per month se = |alpha_mean / alpha_tstat|; pooled mean is the
    inverse-variance weighted average and pooled t = mean * sqrt(sum 1/se^2)
    - with one month this is exactly that month's t, so single-roll behavior
    is unchanged. Months with no measurement (t = 0, < 2 days) are skipped.

    Returns {'tstat', 'mean', 'n_months', 'n_days', 'sign_frac'} where
    sign_frac is the fraction of pooled months whose directed alpha was
    positive (the cross-month sign-consistency diagnostic)."""
    means, ses, days = [], [], 0
    for m in months or []:
        mu, t = m.get('alpha_mean'), m.get('alpha_tstat')
        n = int(m.get('n_days', 0) or 0)
        if mu is None or t is None:
            continue
        mu, t = float(mu), float(t)
        if not (np.isfinite(mu) and np.isfinite(t)) or t == 0.0 or n < 2:
            continue
        se = abs(mu / t)
        if not np.isfinite(se) or se <= 0:
            continue
        d = int(m.get('direction', direction) or direction)
        flip = 1.0 if d == int(direction) else -1.0
        means.append(flip * mu)
        ses.append(se)
        days += n
    if not means:
        return {'tstat': 0.0, 'mean': float('nan'), 'n_months': 0,
                'n_days': 0, 'sign_frac': float('nan')}
    w = 1.0 / np.array(ses) ** 2
    mean = float(np.dot(w, means) / w.sum())
    return {'tstat': float(mean * math.sqrt(w.sum())), 'mean': mean,
            'n_months': len(means), 'n_days': int(days),
            'sign_frac': float(np.mean([x > 0 for x in means]))}


def posterior_sharpe(tstat: float, calendar_days: int,
                     prior_sharpe_std: float) -> float:
    """Expected true annualized Sharpe given the pooled evidence - THE
    promotion ranking currency.

    calendar_days is the CALENDAR time the evidence covers, not the
    observation count: a t-stat's information content depends on time
    spanned (E[t] ~ SR_ann * sqrt(years)), so 9 three-day bets over a month
    carry the same evidence as 27 daily ones. Passing obs counts here would
    inflate a slow horizon's observed Sharpe by sqrt(lag/1d) AND
    over-shrink it - a double charge against exactly the horizons with the
    fewest observations (the promote() loop does the conversion).

    Observed Sharpe = tstat / sqrt(n) annualized by sqrt(365); its sampling
    se is ~sqrt(365/n). Under a N(0, tau^2) prior on true annualized
    Sharpes (tau = prior_sharpe_std, the spread of true Sharpes we believe
    the candidate pool can contain), the posterior mean shrinks the
    observation by n / (n + 365/tau^2):

        posterior = SR_observed * n / (n + 365/tau^2)

    A huge Sharpe on one month is mostly prior (a t of 2 in one month IS an
    observed annualized Sharpe of ~7 - almost certainly luck); a modest
    Sharpe sustained for a year keeps roughly half its value at tau = 1.
    Monotone in BOTH Sharpe and t: at fixed n it ranks by Sharpe (t is
    proportional), across different n it trades magnitude against evidence.
    Stability is priced upstream: inconsistent months cancel inside the
    pooled t, so a signal that flips sign across months arrives here with a
    small t and shrinks toward zero."""
    n = max(int(calendar_days), 0)
    tau2 = float(prior_sharpe_std) ** 2
    if n == 0 or tau2 <= 0 or not np.isfinite(tstat):
        return 0.0
    sr_ann = float(tstat) / math.sqrt(n) * math.sqrt(DAYS_PER_YEAR)
    return sr_ann * n / (n + DAYS_PER_YEAR / tau2)


def promote(survivors: List[dict], roll: Roll, ledger: DiscoveryLedger,
            cfg: Optional[dict] = None) -> List[dict]:
    """Form THIS roll's book: the top book_size survivors by pooled held-out
    evidence, subject to the sanity floors. Returns survivor dicts annotated
    with the pooled statistics and the lag the evidence lives at."""
    cfg = cfg or get('discovery', {})
    promo = cfg['promotion']
    if not survivors:
        return []
    rate = trade_rate_per_bar()

    def sel(s, lag):
        return s['profile_select'].get(lag, s['metrics_select'])

    min_days = int(promo.get('min_select_days', 0))
    min_agree = float(promo.get('min_profile_sign_agreement', 0.0))
    min_capture = float(promo.get('min_capture', 0.0))
    max_book_corr = float(promo['max_book_corr'])
    book_size = max(int(promo['book_size']), 0)
    # Turnover ceiling (None / non-finite = OFF). Rejects untradeable-standalone
    # churners; sibling of the capture floor. Fails OPEN when a survivor carries
    # no turnover (older in-memory dicts) - never block on missing diagnostics.
    _mt = promo.get('max_turnover')
    max_turnover = (float(_mt) if _mt is not None and np.isfinite(_mt)
                    else None)

    # Pooled directed evidence per survivor at its family's horizons: this
    # roll's select measurement plus every PRIOR roll's from the ledger,
    # sign-aligned to the current direction. (This roll's own row is already
    # in the ledger, hence up_to_roll = roll_id - 1 + the in-memory metrics:
    # no double counting, and promote() works on unpersisted survivor dicts.)
    # best_lag = the family horizon with the strongest posterior Sharpe;
    # score = posterior Sharpe x capture. Daily Sharpe is already
    # bets-per-day-fair (a fast signal's higher daily Sharpe from more
    # independent bets is real), and capture prices whether the book can
    # actually trade at that speed - together they rank the TRADABLE
    # expected Sharpe.
    def capture(s) -> float:
        hl = s.get('half_life_bars') or s.get('target_lag') or 1.0
        p_eff = effective_persistence_bars(hl, int(s.get('target_lag') or 6),
                                           s.get('turnover'))
        return persistence_weight(p_eff, rate)

    prior_tau = float(promo['prior_sharpe_std'])
    evidence: List[dict] = []
    for s in survivors:
        fam = family_lags(s['candidate'].family, cfg)
        lags = ([l for l in fam if l in s['profile_select']]
                or sorted(s['profile_select']))
        cur_dir = int(s.get('direction', 1) or 1)
        best = None
        for lag in lags:
            months = ledger.select_history(s['candidate'].hash, lag,
                                           up_to_roll=roll.roll_id - 1)
            ev = pooled_select_evidence(
                months + [{**sel(s, lag), 'direction': cur_dir}],
                direction=cur_dir)
            # Multi-day horizons place one bet per lag/144 days: convert obs
            # days to CALENDAR days before the posterior (and the min_days
            # floor), or slow horizons are charged sqrt(lag/1d) twice.
            ev['calendar_days'] = int(round(
                ev['n_days'] * max(1.0, lag / BARS_PER_DAY)))
            ev['posterior_sharpe'] = posterior_sharpe(
                ev['tstat'], ev['calendar_days'], prior_tau)
            if best is None or ev['posterior_sharpe'] > best['posterior_sharpe']:
                best = {**ev, 'lag': int(lag)}
        best['score'] = best['posterior_sharpe'] * capture(s)
        evidence.append(best)

    def sign_agreement(s) -> float:
        """Fraction of profile lags whose train alpha shares the traded sign
        (direction already applied: agreement = train alpha_mean > 0)."""
        prof = s.get('profile_train') or {}
        alphas = [m.get('alpha_mean') for m in prof.values()
                  if m.get('alpha_mean') is not None
                  and np.isfinite(m.get('alpha_mean'))]
        if not alphas:
            return 0.0
        return float(np.mean([a > 0 for a in alphas]))

    def floors(i, s) -> Dict[str, bool]:
        ev = evidence[i]
        tv = s.get('turnover')
        return {
            'directed': ev['tstat'] > 0.0,
            'min_days': ev['calendar_days'] >= min_days,
            'sign_agreement': sign_agreement(s) >= min_agree,
            'capture': capture(s) >= min_capture,
            'turnover': (max_turnover is None or tv is None
                         or not np.isfinite(tv) or float(tv) <= max_turnover),
            'persistence': ledger.consecutive_survivals(
                s['candidate'].hash, roll.roll_id)
                >= int(promo['min_rolls_survived']),
        }

    n_trials = ledger.n_trials(roll.roll_id)
    promoted: List[dict] = []
    for i in sorted(range(len(survivors)), key=lambda i: -evidence[i]['score']):
        if len(promoted) >= book_size:
            break
        s, ev = survivors[i], evidence[i]
        gates = floors(i, s)
        gates['orthogonal'] = max_signal_correlation(
            s['signal_select'],
            [p['signal_select'] for p in promoted]) <= max_book_corr
        if all(gates.values()):
            promoted.append({
                **s, 'roll_promoted': roll.roll_id,
                'select_lag': ev['lag'],
                'select_alpha_tstat': float(
                    sel(s, ev['lag']).get('alpha_tstat', 0.0) or 0.0),
                'pooled_select_tstat': float(ev['tstat']),
                'pooled_select_months': int(ev['n_months']),
                'pooled_select_days': int(ev['n_days']),
                'pooled_sign_frac': float(ev['sign_frac']),
                'posterior_sharpe': float(ev['posterior_sharpe']),
                'promotion_score': float(ev['score']),
                'capture': capture(s),
                'n_trials_at_promotion': n_trials,
            })
        else:
            logging.debug(f"{s['candidate'].name} blocked by "
                          f"{[k for k, v in gates.items() if not v]}")

    ledger.mark_promoted(roll.roll_id,
                         [p['candidate'].hash for p in promoted])
    return promoted
