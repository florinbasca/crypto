"""
Synthetic checks for CHOOSE under the 5+5+1 spec
(research/signals/promotion.py). No database required.

1. Train-side direction: raw-sign curve history round-trip + pooled
   direction.
2. Retention: promoted_candidates reseed pool query.
3. promote(): the QUINTILE of curve-passers - proportional, bounded, never
   a fixed count (frac 0 = fixed book_size for tests) - plus the capture
   floor (churn prices holdability) and the duplicate filter. Per-filter
   curve semantics (median gate, activity, net-rate economics, peak cap)
   are covered in tests/curve_checks.py.

Run: uv run tests/choose_checks.py
"""

import copy
import json
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from config import get
from research.signals import generation as gen
from research.signals import data as data_mod
from research.signals import search as search_mod
from research.signals import promotion as bt_mod

FAILURES = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name} {detail}")
    if not cond:
        FAILURES.append(name)


# ---------------------------------------------------------------------------
# 1. train-side direction (committed on train, from pooled curve evidence)
# ---------------------------------------------------------------------------
print("--- 1. pooled train direction ---")
led = search_mod.DiscoveryLedger(None)
c_hist = gen.Candidate('h', 'residual_shape', ('col', 'res_zscore'))


def _profile(alpha, t, n_days):
    """profile_json with a train curve, as try_candidate stores it (values
    directed by the roll's fitted direction; se is always positive)."""
    return json.dumps({'curve_train': {'a0': alpha,
                                       'se_peak': abs(alpha / t),
                                       'entry_days': n_days}})


for rid, (a, t, d) in enumerate([(0.001, 2.0, 30), (0.002, 1.5, 28),
                                 (-0.001, -1.0, 30)]):
    led.record(rid, 0, c_hist, direction=1 if rid < 2 else -1,
               train_metrics={}, select_metrics={}, reward=0.0, terms={},
               target_lag=144, profile_json=_profile(a, t, d),
               half_life_bars=144.0, turnover=0.01)

t_hist = led.train_history(c_hist.hash)
check("train history: directed rows un-flipped to raw sign",
      [h['alpha_tstat'] for h in t_hist] == [2.0, 1.5, 1.0]
      and t_hist[2]['alpha_mean'] == 0.001)

ptd = search_mod.pooled_train_direction
month = lambda mu, t: {'alpha_mean': mu, 'alpha_tstat': t, 'n_days': 100}
check("direction: single window -> its own sign",
      ptd([month(0.001, 2.0)]) == 1 and ptd([month(-0.001, -2.0)]) == -1)
check("direction: strong consistent history outvotes a weak contrary window",
      ptd([month(-0.002, -3.0), month(-0.002, -3.0),
           month(0.0005, 0.8)]) == -1)

# ---------------------------------------------------------------------------
# 2. retention reseed pool
# ---------------------------------------------------------------------------
print("--- 2. retention ---")
led.mark_promoted(0, [c_hist.hash])
check("retention: promoted_candidates returns distinct recent promotees",
      [c.hash for c in led.promoted_candidates(0, 1)] == [c_hist.hash]
      and led.promoted_candidates(2, 5) == [])

# ---------------------------------------------------------------------------
# 3. promote(): the quintile of curve-passers
# ---------------------------------------------------------------------------
print("--- 3. CHOOSE ---")
ROLL = data_mod.Roll(0, pd.Timestamp('2024-01-01'), pd.Timestamp('2024-06-01'),
                     pd.Timestamp('2024-11-01'), pd.Timestamp('2024-12-01'))
_rng = np.random.default_rng(5)
_ts = pd.date_range('2024-06-01', periods=30, freq='1d')


def _signal_panel():
    return pd.DataFrame({'timestamp': np.repeat(_ts, 4),
                         'symbol': np.tile(list('WXYZ'), 30),
                         'signal': _rng.normal(size=120)})


def make_survivor(i, sel_t, sel_alpha=None, n_days=140, direction=1,
                  turnover=0.01, family='residual_shape', signal=None,
                  train_alpha=None):
    """A survivor dict shaped like run_search's population entries: the
    5-month test verdict is a fitted curve (A ramps linearly to a0 at the
    horizon end). train_alpha attaches a TRAIN curve (the ranking input).
    Default edges are 50bp x t - comfortably above the global 10bp round
    trip, so filter 3 passes the ladder and the QUINTILE mechanics (what
    this file tests) are what decide."""
    sel_alpha = sel_alpha if sel_alpha is not None else 0.005 * sel_t
    ks = [1, 2, 3, 6, 12, 24, 48, 72, 96, 120, 144]
    A = [sel_alpha * k / 144 for k in ks]
    se = abs(sel_alpha / sel_t) if sel_t else 0.001
    m_sel = {'alpha_mean': sel_alpha, 'alpha_tstat': sel_t, 'n_days': n_days}
    m_trn = {'alpha_mean': 0.001, 'alpha_tstat': 3.0, 'n_days': 140}
    out = {
        'candidate': gen.Candidate(f's{i}', family, ('col', f'feat_{i}')),
        'direction': direction, 'target_lag': 144,
        'half_life_bars': 48.0,
        'profile_train': {}, 'profile_select': {},
        'metrics_train': m_trn, 'metrics_select': m_sel,
        'curve': {'a0': sel_alpha, 'half_life': 48.0, 'peak_k': 144,
                  'rev_frac': 0.0, 'se_peak': se, 'median_peak': sel_alpha,
                  'entries': 600, 'entry_days': n_days, 'n_eff': 25.0,
                  'ks': ks, 'A': A},
        'reward': 1.0, 'turnover': turnover,
        'signal_train': _signal_panel(),
        'signal_select': signal if signal is not None else _signal_panel(),
    }
    if train_alpha is not None:
        out['curve_train'] = {**out['curve'], 'a0': train_alpha,
                              'A': [train_alpha * k / 144 for k in ks]}
    return out


TCFG = copy.deepcopy(get('discovery'))
TCFG['promotion'].update({
    'min_select_days': 20, 'min_capture': 0.0, 'max_book_corr': 0.5,
    'book_frac': 0.20, 'book_min': 1, 'book_max': 50, 'book_size': 10,
})

# 10 candidates: a t-ladder of passers plus one of each failure mode.
pool = ([make_survivor(i, sel_t=0.4 * i) for i in range(1, 7)]   # 0.4..2.4
        + [make_survivor(7, sel_t=-2.5),                # filter 1: backwards
           make_survivor(8, sel_t=9.9, n_days=8),       # filter 2: thin
           make_survivor(9, sel_t=0.0),                 # filter 1: nothing
           make_survivor(10, sel_t=2.2)])               # passer
book = bt_mod.promote(pool, ROLL, search_mod.DiscoveryLedger(None), TCFG)
names = [p['candidate'].name for p in book]
# passers = s1..s6 + s10 = 7 -> quintile = ceil(0.2*7) = 2
check("quintile: ceil(frac x passers) promoted, best first",
      len(book) == 2 and set(names) == {'s6', 's10'}, f"({names})")
check("filters: backwards / thin / empty never promote",
      not any(n in names for n in ('s7', 's8', 's9')))
check("annotations: verdict lag, test days, econ margin present",
      all('select_lag' in p and 'test_days' in p and 'econ_margin' in p
          for p in book))

# no fixed count: double the passers -> the book grows
pool_wide = pool + [make_survivor(20 + i, sel_t=1.0 + 0.1 * i)
                    for i in range(7)]
book_wide = bt_mod.promote(pool_wide, ROLL,
                           search_mod.DiscoveryLedger(None), TCFG)
check("quintile: book grows with passers (never a fixed 3)",
      len(book_wide) == math.ceil(0.2 * 14), f"({len(book_wide)} promoted)")

# bounds: book_min floors thin months, book_max caps rich ones
b_cfg = copy.deepcopy(TCFG)
b_cfg['promotion'].update({'book_min': 5, 'book_max': 6})
check("quintile: book_min floors it",
      len(bt_mod.promote(pool, ROLL, search_mod.DiscoveryLedger(None),
                         b_cfg)) == 5)
check("quintile: book_max caps it",
      len(bt_mod.promote(pool_wide + [make_survivor(40 + i, sel_t=2.0)
                                      for i in range(30)],
                         ROLL, search_mod.DiscoveryLedger(None),
                         b_cfg)) == 6)

# frac 0 -> fixed book_size (test back-compat path)
f_cfg = copy.deepcopy(TCFG)
f_cfg['promotion'].update({'book_frac': 0.0, 'book_size': 3})
check("quintile: frac 0 falls back to fixed book_size",
      len(bt_mod.promote(pool_wide, ROLL, search_mod.DiscoveryLedger(None),
                         f_cfg)) == 3)

# filter 3 holdability: churn prices the capture. Same curve, one churns
# 0.5/bar (position life 2 bars -> capture collapses), one holds steady.
e_cfg = copy.deepcopy(TCFG)
e_cfg['promotion'].update({'min_capture': 0.5})
churner = make_survivor(30, sel_t=2.0, sel_alpha=0.004, turnover=0.5)
steady = make_survivor(31, sel_t=2.0, sel_alpha=0.004, turnover=0.001)
book_e = bt_mod.promote([churner, steady], ROLL,
                        search_mod.DiscoveryLedger(None), e_cfg)
check("holdability: churner rejected by the capture floor, steady twin "
      "promoted",
      [p['candidate'].name for p in book_e] == ['s31'])

# ranking: the TEST curve gates, the TRAIN curve orders. A passer with a
# spectacular test rate but weak train rate must lose the slot to a modest
# test / strong train one (ranking on test promoted the luckiest test
# windows - measured OOS anti-prediction).
lucky = make_survivor(70, sel_t=2.0, sel_alpha=0.0050, train_alpha=0.0005)
solid = make_survivor(71, sel_t=2.0, sel_alpha=0.0020, train_alpha=0.0040)
r_cfg = copy.deepcopy(TCFG)
r_cfg['promotion'].update({'book_min': 1, 'book_max': 1})
book_r = bt_mod.promote([lucky, solid], ROLL,
                        search_mod.DiscoveryLedger(None), r_cfg)
check("ranking: train rate orders the passers, test only gates",
      [p['candidate'].name for p in book_r] == ['s71'])

# filter 4: a lower-ranked duplicate of a chosen formula is skipped
top_sig = pool[5]['signal_select']
dupe = make_survivor(50, sel_t=2.2, signal=top_sig.copy())
book_d = bt_mod.promote(pool + [dupe], ROLL,
                        search_mod.DiscoveryLedger(None), TCFG)
check("duplicate: same-output formula skipped, slot passes on",
      's50' not in [p['candidate'].name for p in book_d])

# ---------------------------------------------------------------------------
print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURES: {FAILURES}")
    sys.exit(1)
print("ALL CHOOSE CHECKS PASSED")
