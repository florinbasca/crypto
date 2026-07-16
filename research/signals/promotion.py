"""
CHOOSE: which formulas get promoted, under the 5+5+1 windows.

Discovery is purely statistical - it measures per-bet returns, never traded
PnL. Promotions are consumed by research/portfolio/walk_forward.py (via
research/lib/discovered.py), the ONLY money judge.

A formula's VERDICT is its most recent 5-month test window (the roll's
select window, ~150 days the formula never saw), directed by the sign
committed during training. One verdict per formula per roll - no cross-roll
pooling, no evidence accumulation: the long window IS the evidence.

Four filters, then the quintile:

  1. MADE MONEY   - the verdict is net positive in the committed direction.
                    A sign, not a bar (nothing resembling a significance
                    threshold exists here). Directed, never |t|: a formula
                    whose test ran backwards is rejected, not flipped -
                    re-signing after seeing the test is how noise promotes.
  2. ENOUGH ACTIVITY - fired on at least min_select_days real days within
                    the test window. Dense formulas pass trivially; a tight
                    GATE on dense features (active 4 days/month) does not.
  3. PAYS FOR ITSELF - expected per-bar profit from the verdict exceeds the
                    formula's own per-bar trading cost (churn x cost rate),
                    AND the alpha is holdable at the book's measured fill
                    rate (capture floor). Measured quantities only.
  4. NOT A DUPLICATE - signal correlation vs already-chosen formulas at
                    most max_book_corr (greedy, best first).

Then promote the BEST QUINTILE of everything that passed
(ceil(book_frac x passers), bounded by book_min/book_max). Proportional -
the book breathes with how much quality exists; never a fixed count.
"""

import logging
import math
from typing import Dict, List, Optional

import numpy as np

from config import BARS_PER_DAY, get
from research.signals.data import Roll, family_lags
from research.signals.search import (DiscoveryLedger,
                                           day_equivalent_tstat,
                                           effective_persistence_bars,
                                           max_signal_correlation,
                                           persistence_weight,
                                           trade_rate_per_bar)


def econ_margin_per_bar(alpha_mean: float, lag_bars: int,
                        turnover: Optional[float],
                        cost_rate: float) -> float:
    """Filter 3's economics: per-bar expected profit minus per-bar trading
    cost. alpha_mean is the per-BET return over lag_bars (returns scale
    linearly with holding time -> per-bar = alpha/lag); the trading cost is
    the signal's own churn (fraction of gross replaced per bar) times the
    per-side cost rate. Missing/NaN turnover prices the cost at zero (fails
    open - never block on a missing diagnostic)."""
    a = float(alpha_mean) / max(int(lag_bars), 1)
    to = (float(turnover)
          if turnover is not None and np.isfinite(turnover) else 0.0)
    return a - to * float(cost_rate)


def promote(survivors: List[dict], roll: Roll, ledger: DiscoveryLedger,
            cfg: Optional[dict] = None) -> List[dict]:
    """Apply the four filters to this roll's measured formulas, promote the
    best quintile of the passers. Returns survivor dicts annotated with the
    verdict lag and the economics."""
    cfg = cfg or get('discovery', {})
    promo = cfg['promotion']
    if not survivors:
        return []
    rate = trade_rate_per_bar()

    min_days = int(promo.get('min_select_days', 0))
    min_capture = float(promo.get('min_capture', 0.0))
    max_book_corr = float(promo['max_book_corr'])
    _cb = promo.get('econ_cost_bps')
    cost_rate = (float(_cb) if _cb is not None
                 else float(get('portfolio.cost_bps'))) / 10000.0

    def sel(s, lag):
        return s['profile_select'].get(lag, s['metrics_select'])

    def capture(s) -> float:
        hl = s.get('half_life_bars') or s.get('target_lag') or 1.0
        p_eff = effective_persistence_bars(hl, int(s.get('target_lag') or 6),
                                           s.get('turnover'))
        return persistence_weight(p_eff, rate)

    # Verdict per formula: at each of its family's horizons, filters 1+2
    # per lag (net positive, enough active days - calendar-adjusted for
    # multi-day lags); best qualifying lag by day-equivalent t. Ranking
    # score = that t x capture: per-bet-fair across horizons, and holdable
    # evidence outranks equally-strong unholdable evidence.
    verdicts: List[Optional[dict]] = []
    for s in survivors:
        fam = family_lags(s['candidate'].family, cfg)
        lags = ([l for l in fam if l in s['profile_select']]
                or sorted(s['profile_select']))
        best = None
        for lag in lags:
            m = sel(s, lag)
            t = float(m.get('alpha_tstat', 0.0) or 0.0)
            days = int(round(int(m.get('n_days', 0) or 0)
                             * max(1.0, lag / BARS_PER_DAY)))
            if t <= 0.0 or days < min_days:
                continue                      # filters 1 + 2 at this lag
            fair = day_equivalent_tstat({'alpha_tstat': t}, lag)
            if best is None or fair > best['fair_t']:
                best = {'lag': int(lag), 'fair_t': fair, 'tstat': t,
                        'days': days,
                        'alpha': float(m.get('alpha_mean', 0.0) or 0.0)}
        if best is not None:
            best['margin'] = econ_margin_per_bar(
                best['alpha'], best['lag'], s.get('turnover'), cost_rate)
            best['score'] = best['fair_t'] * capture(s)
        verdicts.append(best)

    # Filters over whole formulas: 1+2 (a qualifying lag exists),
    # 3 (pays for itself + holdable). Filter 4 applies greedily below.
    passers = [i for i, v in enumerate(verdicts)
               if v is not None and v['margin'] > 0.0
               and capture(survivors[i]) >= min_capture]

    # THE QUINTILE: proportional to how much passed, bounded. frac 0 falls
    # back to the fixed book_size (tests only). Never a hardcoded 3.
    frac = float(promo.get('book_frac', 0.0) or 0.0)
    if frac > 0:
        k = int(min(max(math.ceil(frac * len(passers)),
                        int(promo.get('book_min', 1))),
                    int(promo.get('book_max', len(passers) or 1))))
    else:
        k = int(promo.get('book_size', 0))
    k = min(k, len(passers))

    n_trials = ledger.n_trials(roll.roll_id)
    promoted: List[dict] = []
    for i in sorted(passers, key=lambda i: -verdicts[i]['score']):
        if len(promoted) >= k:
            break
        s, v = survivors[i], verdicts[i]
        # Filter 4: not a duplicate of anything already chosen.
        if max_signal_correlation(
                s['signal_select'],
                [p['signal_select'] for p in promoted]) > max_book_corr:
            logging.debug(f"{s['candidate'].name} skipped: duplicates the "
                          f"chosen book")
            continue
        promoted.append({
            **s, 'roll_promoted': roll.roll_id,
            'select_lag': v['lag'],
            'select_alpha_tstat': v['tstat'],
            'test_days': v['days'],
            'econ_margin': v['margin'],
            'capture': capture(s),
            'n_trials_at_promotion': n_trials,
        })

    ledger.mark_promoted(roll.roll_id,
                         [p['candidate'].hash for p in promoted])
    return promoted
