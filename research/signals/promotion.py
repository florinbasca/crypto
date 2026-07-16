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

    curve_cfg = cfg.get('curve') or {}
    median_gate = bool(curve_cfg.get('median_gate', False))
    rt_cost = float(curve_cfg.get('roundtrip_mult', 2.0)) * cost_rate

    def sel(s, lag):
        return s['profile_select'].get(lag, s['metrics_select'])

    def capture(s, peak_k: Optional[int] = None,
                half_life: Optional[float] = None) -> float:
        hl = (half_life if half_life
              else s.get('half_life_bars') or s.get('target_lag') or 1.0)
        # v1 reversal handling: holding inputs are CAPPED at the measured
        # peak - beyond it the alpha actively reverses, so persistence the
        # book could "use" past the peak is worthless.
        if peak_k:
            hl = min(float(hl), float(peak_k))
        p_eff = effective_persistence_bars(hl, int(s.get('target_lag') or 6),
                                           s.get('turnover'))
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
        Returns None if the row carries no curve (old ledgers -> lag path)."""
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

    # Verdict per formula: at each of its family's horizons, filters 1+2
    # per lag (net positive, enough active days - calendar-adjusted for
    # multi-day lags); best qualifying lag by day-equivalent t. Ranking
    # score = that t x capture: per-bet-fair across horizons, and holdable
    # evidence outranks equally-strong unholdable evidence.
    verdicts: List[Optional[dict]] = []
    for s in survivors:
        # Preferred: the fitted response curve (judged at its own optimal
        # holding, ranked by net economic rate).
        v = curve_verdict(s)
        if v is None and not s.get('curve'):
            # Fallback for rows without curves (pre-curve ledgers): the
            # 4-lag verdict, exactly as before.
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
                    continue                  # filters 1 + 2 at this lag
                fair = day_equivalent_tstat({'alpha_tstat': t}, lag)
                if best is None or fair > best['fair_t']:
                    best = {'lag': int(lag), 'fair_t': fair, 'tstat': t,
                            'days': days,
                            'alpha': float(m.get('alpha_mean', 0.0) or 0.0)}
            if best is not None:
                best['margin'] = econ_margin_per_bar(
                    best['alpha'], best['lag'], s.get('turnover'), cost_rate)
                best['score'] = best['fair_t'] * capture(s)
                best['peak_k'] = None
                best['half_life'] = None
            v = best
        verdicts.append(v)

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
