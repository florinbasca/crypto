"""
PROMOTION: the gates that admit a survivor to the promotions table.

Discovery is purely statistical - it measures per-bet returns
(alpha in return units), never traded PnL. The promotions
this module emits are consumed by research/portfolio/walk_forward.py (via
research/lib/discovered.py), which is the ONLY money judge: costs, execution
and the equity curve live there and nowhere else.

The search never touches the select window (train-only reward), so promotion
is the FIRST and ONLY look at select - a survivor promotes if ANY lag of its
profile clears:
  1. BY/BH FDR across every (survivor, lag) select p-value - Student-t with
     (n_days - 1) dof, since a 1-month select is ~30 daily alpha observations
  2. directed select t (positive in the traded direction) must clear BOTH
     min_select_alpha_tstat AND the deflation haircut deflation_mult x
     E[max |N(0,1)|] over the ACTUAL looks at select (n_survivors x n_lags) -
     the multiplicity of promotion itself. Directed, not |t|: a lag that
     REVERSES sign out-of-sample is rejected, not admitted on magnitude.
  3. minimum daily observations behind the t (min_select_days)
and the candidate as a whole must pass:
  4. profile sign agreement: the train term structure must mostly share the
     traded direction's sign (a mixed profile is a red flag)
  5. capture floor: persistence weight 1/(1 + phi/kappa) at least
     min_capture - the book structurally cannot hold alpha faster than its
     own trade rate long enough to matter (duration, never a cost model)
  5b. turnover ceiling (max_turnover, OFF by default): direct-measurement
     sibling of the capture floor - rejects a signal whose per-bar book churn
     exceeds the cap (the untradeable-standalone case where alpha persists but
     positions are noisy, which the half-life-derived capture floor misses)
  6. N-consecutive-rolls persistence (by candidate hash, via the ledger)
  7. orthogonality vs the already-promoted book (incremental edge, greedy)
  8. book-size cap; slots filled in CAPTURE-WEIGHTED day-equivalent select
     strength order, so persistent evidence outranks equally-strong fast
     evidence
"""

import logging
import math
from typing import List, Optional

import numpy as np

from config import get
from research.lib.portfolio_opt import (benjamini_hochberg,
                                        benjamini_yekutieli)
from research.signals.data import Roll
from research.signals.search import (DiscoveryLedger,
                                           day_equivalent_tstat,
                                           effective_persistence_bars,
                                           max_signal_correlation,
                                           persistence_weight,
                                           trade_rate_per_bar)


def _tstat_pvalue(t: float, n_days: Optional[int] = None) -> float:
    """Two-sided p-value for a t-stat. With n_days given, Student-t with
    (n_days - 1) dof - a 1-month select window has ~30 daily alpha observations,
    where the normal approximation is anti-conservative. Falls back to
    normal when n_days is missing."""
    t = abs(float(t))
    if n_days is not None and int(n_days) > 1:
        from scipy import stats
        return float(2.0 * stats.t.sf(t, df=int(n_days) - 1))
    return math.erfc(t / math.sqrt(2.0))


def expected_max_abs_normal(n_trials: int) -> float:
    """E[max |N(0,1)|] over n trials ~ sqrt(2 ln n) - the bar random noise
    reaches when n candidates are tested against the same month."""
    n = max(int(n_trials), 1)
    if n == 1:
        return 0.8   # E|N(0,1)|
    return math.sqrt(2.0 * math.log(n))


def promote(survivors: List[dict], roll: Roll, ledger: DiscoveryLedger,
            cfg: Optional[dict] = None) -> List[dict]:
    """Form THIS roll's book: the survivors that pass the gates on this
    roll's windows. The book re-forms from scratch every roll - no
    carryover, no lifetime membership: a signal is promoted for a roll only
    if it re-qualified on the 6 months of data ending just before it
    (train 5 + select 1). Persistence across rolls is still required
    (min_rolls_survived via the ledger), and orthogonality is greedy WITHIN
    the roll's qualifiers.

    A survivor promotes if ANY lag of its select profile clears the
    statistical gates; the deflation haircut prices the (n_survivors x
    n_lags) looks promotion takes at select. Returns the roll's book
    (survivor dicts annotated with gate details, incl. promoted_lags).
    """
    cfg = cfg or get('discovery', {})
    promo = cfg['promotion']
    if not survivors:
        return []
    lags = [int(x) for x in cfg['horizon_lags_bars']]
    rate = trade_rate_per_bar()

    def sel(s, lag):
        return s['profile_select'].get(lag, s['metrics_select'])

    # FDR across EVERY (survivor, lag) select p-value - Student-t dof from
    # each cell's own daily-observation count.
    pvals = np.array([
        _tstat_pvalue(sel(s, lag).get('alpha_tstat', 0.0) or 0.0,
                      sel(s, lag).get('n_days'))
        for s in survivors for lag in lags
    ])
    fdr = (benjamini_yekutieli if promo['fdr_method'] == 'by'
           else benjamini_hochberg)
    fdr_mask = fdr(pvals, alpha=float(promo['fdr_alpha'])).reshape(
        len(survivors), len(lags))

    # Deflation over the looks promotion actually takes at select (the
    # search is train-only and adds none).
    n_looks = len(survivors) * len(lags)
    deflation_bar = (float(promo['deflation_mult'])
                     * expected_max_abs_normal(n_looks))
    n_trials = ledger.n_trials(roll.roll_id)

    min_t = float(promo['min_select_alpha_tstat'])
    min_days = int(promo.get('min_select_days', 0))
    min_agree = float(promo.get('min_profile_sign_agreement', 0.0))
    min_capture = float(promo.get('min_capture', 0.0))
    max_book_corr = float(promo['max_book_corr'])
    slots = int(promo['max_book_size'])
    # Turnover ceiling (None / non-finite = OFF). Rejects untradeable-standalone
    # churners; sibling of the capture floor. Fails OPEN when a survivor carries
    # no turnover (older in-memory dicts) - never block on missing diagnostics.
    _mt = promo.get('max_turnover')
    max_turnover = (float(_mt) if _mt is not None and np.isfinite(_mt)
                    else None)

    def turnover_ok(s) -> bool:
        if max_turnover is None:
            return True
        tv = s.get('turnover')
        if tv is None or not np.isfinite(tv):
            return True
        return float(tv) <= max_turnover

    def passing_lags(i, s) -> List[int]:
        out = []
        for j, lag in enumerate(lags):
            m = sel(s, lag)
            # DIRECTED select t: the profile is already in the traded
            # direction, so a NEGATIVE t means the signal REVERSED on the
            # hold-out month at this lag - not validated alpha, reject it.
            # (abs() here would admit anti-predictive lags on magnitude alone,
            # promoting a signal that trades the wrong way out-of-sample.)
            t = float(m.get('alpha_tstat', 0.0) or 0.0)
            if (bool(fdr_mask[i, j]) and t >= min_t
                    and (float(promo['deflation_mult']) <= 0
                         or t >= deflation_bar)
                    and int(m.get('n_days', 0) or 0) >= min_days):
                out.append(lag)
        return out

    def sign_agreement(s) -> float:
        """Fraction of profile lags whose train alpha shares the traded sign
        (direction already applied: agreement = train alpha_mean > 0)."""
        prof = s.get('profile_train') or {}
        ics = [m.get('alpha_mean') for m in prof.values()
               if m.get('alpha_mean') is not None
               and np.isfinite(m.get('alpha_mean'))]
        if not ics:
            return 0.0
        return float(np.mean([ic > 0 for ic in ics]))

    def capture(s) -> float:
        hl = s.get('half_life_bars') or s.get('target_lag') or 1.0
        p_eff = effective_persistence_bars(hl, int(s.get('target_lag') or 6),
                                           s.get('turnover'))
        return persistence_weight(p_eff, rate)

    # Slot order: CAPTURE-WEIGHTED day-equivalent select strength -
    # per-bet-fair across horizons AND persistent evidence outranks
    # equally-strong fast evidence.
    def slot_score(s) -> float:
        fair_t = max((abs(day_equivalent_tstat(sel(s, lag), lag))
                      for lag in lags), default=0.0)
        return fair_t * capture(s)

    promoted: List[dict] = []
    order = sorted(range(len(survivors)),
                   key=lambda i: -slot_score(survivors[i]))
    for i in order:
        if len(promoted) >= max(slots, 0):
            break
        s = survivors[i]
        ok_lags = passing_lags(i, s)
        gates = {
            'profile': bool(ok_lags),
            'sign_agreement': sign_agreement(s) >= min_agree,
            'capture': capture(s) >= min_capture,
            'turnover': turnover_ok(s),
            'persistence': ledger.consecutive_survivals(
                s['candidate'].hash, roll.roll_id)
                >= int(promo['min_rolls_survived']),
            'orthogonal': max_signal_correlation(
                s['signal_select'],
                [p['signal_select'] for p in promoted]) <= max_book_corr,
        }
        if all(gates.values()):
            # Report the select t at the STRONGEST lag that actually cleared
            # the gates - the evidence the signal promoted on. This is NOT
            # metrics_select's t: that is the best-TRAIN-lag select t, which is
            # frequently not a promoted lag and can be ~0 or negative even when
            # a different lag clears cleanly.
            best_ok = max(ok_lags, key=lambda lag:
                          abs(float(sel(s, lag).get('alpha_tstat', 0.0) or 0.0)))
            promoted.append({**s, 'roll_promoted': roll.roll_id,
                             'promoted_lags': ok_lags,
                             'select_lag': int(best_ok),
                             'select_alpha_tstat': float(
                                 sel(s, best_ok).get('alpha_tstat', 0.0) or 0.0),
                             'capture': capture(s),
                             'n_looks_at_promotion': n_looks,
                             'n_trials_at_promotion': n_trials})
        else:
            logging.debug(f"{s['candidate'].name} blocked by "
                          f"{[k for k, v in gates.items() if not v]}")

    ledger.mark_promoted(roll.roll_id,
                         [p['candidate'].hash for p in promoted])
    return promoted
