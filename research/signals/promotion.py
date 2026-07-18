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
(ceil(book_frac x passers), bounded by book_min/book_max), RANKED BY THE
TRAIN curve's net rate - the test window gates, it never ranks (ranking
on it promoted the luckiest test windows; measured OOS anti-prediction).
Proportional - the book breathes with how much quality exists; never a
fixed count.
"""

import logging
import math
from typing import List, Optional

import numpy as np

from config import get
from research.signals.data import Roll
from research.signals.search import (DiscoveryLedger,
                                           effective_persistence_bars,
                                           max_signal_correlation,
                                           persistence_weight,
                                           trade_rate_per_bar)


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
    cost_rate = float(get('portfolio.cost_bps')) / 10000.0

    curve_cfg = cfg.get('curve') or {}
    median_gate = bool(curve_cfg.get('median_gate', False))
    rt_cost = float(curve_cfg.get('roundtrip_mult', 2.0)) * cost_rate

    def capture(s, peak_k: Optional[int] = None,
                half_life: Optional[float] = None) -> float:
        hl = (half_life if half_life
              else s.get('half_life_bars') or s.get('target_lag') or 1.0)
        # v1 reversal handling: holding inputs are CAPPED at the measured
        # peak - beyond it the alpha actively reverses, so persistence the
        # book could "use" past the peak is worthless.
        if peak_k:
            hl = min(float(hl), float(peak_k))
        p_eff = effective_persistence_bars(hl, s.get('turnover'))
        return persistence_weight(p_eff, rate)

    def curve_verdict(s) -> Optional[dict]:
        """Verdict from the fitted response curve. Filters:
        1 made money  - a0 > 0 (and, median_gate: median entry at the peak
                        positive - one jump day passes a mean, not a median;
                        NaN median fails open)
        2 activity    - entry_days >= min_select_days
        3 economics   - best net RATE over the sampled curve,
                        max_k (A(k) - roundtrip)/k, must be positive: the
                        formula judged at its own optimal holding.
        Returns None when the row has no curve or fails filters 1-2."""
        c = s.get('curve')
        if not c or not c.get('ks'):
            return None
        a0 = float(c.get('a0') or 0.0)
        med = c.get('median_peak')
        days = int(c.get('entry_days') or 0)
        if a0 <= 0 or days < min_days:
            return None
        if median_gate and med is not None and np.isfinite(med) and med <= 0:
            return None
        rates = [(float(a) - rt_cost) / int(k)
                 for k, a in zip(c['ks'], c['A'])
                 if a is not None and int(k) > 0]
        net_rate = max(rates) if rates else float('-inf')
        se = c.get('se_peak')
        t = (a0 / float(se)) if se and np.isfinite(se) and se > 0 \
            else float('nan')
        return {'lag': int(c.get('peak_k') or s.get('target_lag') or 6),
                'tstat': t, 'days': days, 'alpha': a0,
                'margin': net_rate, 'score': net_rate,
                'peak_k': int(c.get('peak_k') or 0),
                'half_life': float(c.get('half_life') or 0.0) or None}

    # Verdict per formula: the test response curve, judged at its own
    # optimal holding. The verdict GATES; it does not rank.
    verdicts: List[Optional[dict]] = [curve_verdict(s) for s in survivors]

    def rank_score(i: int) -> float:
        """Order passers by the TRAIN curve's net rate. Ranking by the test
        rate selected the luckiest test windows (measured: spearman(test,
        oos) -0.25, slope -0.91 - the biggest verdicts crashed hardest OOS).
        The train window is already spent by the search, so reusing it to
        order adds no new bias; the test window stays a pure gate."""
        c = survivors[i].get('curve_train')
        if c and c.get('ks'):
            rates = [(float(a) - rt_cost) / int(k)
                     for k, a in zip(c['ks'], c['A'])
                     if a is not None and int(k) > 0]
            if rates:
                return max(rates)
        return verdicts[i]['score']    # rows without a train curve

    # Filters over whole formulas: 1+2 (inside the verdict), 3 (pays for
    # itself + holdable, with holding capped at the measured peak).
    # Filter 4 applies greedily below.
    passers = [i for i, v in enumerate(verdicts)
               if v is not None and v['margin'] > 0.0
               and capture(survivors[i], v.get('peak_k'),
                           v.get('half_life')) >= min_capture]

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
    for i in sorted(passers, key=lambda i: -rank_score(i)):
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
        row = {
            **s, 'roll_promoted': roll.roll_id,
            'select_lag': v['lag'],
            'select_alpha_tstat': v['tstat'],
            'test_days': v['days'],
            'econ_margin': v['margin'],
            'peak_bars': v.get('peak_k'),
            'capture': capture(s, v.get('peak_k'), v.get('half_life')),
            'n_trials_at_promotion': n_trials,
        }
        # v1 reversal handling for the PORTFOLIO: the promotion row's
        # half-life - which the walk-forward uses for smoothing, holding
        # and its capture discount - is the curve's fitted half-life capped
        # at the peak. Beyond the peak the alpha reverses; the book must
        # never be told it can hold longer than that.
        if v.get('half_life'):
            row['half_life_bars'] = float(
                min(v['half_life'], v.get('peak_k') or v['half_life']))
        promoted.append(row)

    ledger.mark_promoted(roll.roll_id,
                         [p['candidate'].hash for p in promoted])
    return promoted
