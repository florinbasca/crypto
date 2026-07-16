"""
Synthetic checks for CHOOSE under the 5+5+1 spec
(research/signals/promotion.py). No database required.

1. family_lags - per-family horizon restriction resolves and falls back.
2. Train-side direction: raw-sign history round-trip + pooled direction.
3. Retention: promoted_candidates reseed pool query.
4. econ_margin_per_bar - filter 3's arithmetic (linear per-bar alpha,
   churn-priced cost, NaN turnover fails open).
5. promote(): the four filters (made money / enough activity / pays for
   itself / not a duplicate) and the QUINTILE of passers - proportional,
   bounded, never a fixed count (frac 0 = fixed book_size for tests).

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
# 1. family_lags
# ---------------------------------------------------------------------------
print("--- 1. per-family horizon lags ---")
fl_cfg = {'horizon_lags_bars': [6, 36, 72, 144],
          'family_horizon_lags': {'default': [6, 36, 72, 144],
                                  'unlocks': [144]}}
check("family_lags: listed family gets its own horizons",
      data_mod.family_lags('unlocks', fl_cfg) == [144])
check("family_lags: unlisted family gets the default",
      data_mod.family_lags('order_flow', fl_cfg) == [6, 36, 72, 144])
check("family_lags: no config section -> full grid",
      data_mod.family_lags('unlocks',
                           {'horizon_lags_bars': [6, 36]}) == [6, 36])

# ---------------------------------------------------------------------------
# 2. train-side direction (unchanged by the 5+5+1 spec: committed on train)
# ---------------------------------------------------------------------------
print("--- 2. pooled train direction ---")
led = search_mod.DiscoveryLedger(None)
c_hist = gen.Candidate('h', 'residual_shape', ('col', 'res_zscore'))


def _profile(lag, alpha, t, n_days):
    return json.dumps({str(lag): {
        'train': {'alpha_mean': alpha, 'alpha_tstat': t, 'n_days': n_days},
        'select': {'alpha_mean': alpha, 'alpha_tstat': t, 'n_days': n_days},
    }})


for rid, (a, t, d) in enumerate([(0.001, 2.0, 30), (0.002, 1.5, 28),
                                 (-0.001, -1.0, 30)]):
    led.record(rid, 0, c_hist, direction=1 if rid < 2 else -1,
               train_metrics={}, select_metrics={}, reward=0.0, terms={},
               target_lag=144, profile_json=_profile(144, a, t, d),
               half_life_bars=144.0, turnover=0.01)

t_hist = led.train_history(c_hist.hash, 144)
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
# 3. retention reseed pool
# ---------------------------------------------------------------------------
print("--- 3. retention ---")
led.mark_promoted(0, [c_hist.hash])
check("retention: promoted_candidates returns distinct recent promotees",
      [c.hash for c in led.promoted_candidates(0, 1)] == [c_hist.hash]
      and led.promoted_candidates(2, 5) == [])

# ---------------------------------------------------------------------------
# 4. filter 3 economics
# ---------------------------------------------------------------------------
print("--- 4. pays-for-itself arithmetic ---")
em = bt_mod.econ_margin_per_bar
check("econ: per-bar alpha is linear in holding (alpha/lag)",
      abs(em(0.0144, 144, 0.0, 0.0) - 0.0001) < 1e-15)
check("econ: cost = churn x rate, subtracted",
      abs(em(0.0144, 144, 0.10, 0.0005) - (0.0001 - 0.00005)) < 1e-15)
check("econ: churny formula with thin alpha goes negative",
      em(0.0006, 6, 0.50, 0.0005) < 0)
check("econ: NaN/None turnover fails open (cost 0)",
      abs(em(0.0006, 6, float('nan'), 0.0005) - 0.0006 / 6) < 1e-15
      and abs(em(0.0006, 6, None, 0.0005) - 0.0006 / 6) < 1e-15)

# ---------------------------------------------------------------------------
# 5. promote(): four filters + the quintile
# ---------------------------------------------------------------------------
print("--- 5. CHOOSE ---")
LAG = 144
ROLL = data_mod.Roll(0, pd.Timestamp('2024-01-01'), pd.Timestamp('2024-06-01'),
                     pd.Timestamp('2024-11-01'), pd.Timestamp('2024-12-01'))
_rng = np.random.default_rng(5)
_ts = pd.date_range('2024-06-01', periods=30, freq='1d')


def _signal_panel():
    return pd.DataFrame({'timestamp': np.repeat(_ts, 4),
                         'symbol': np.tile(list('WXYZ'), 30),
                         'signal': _rng.normal(size=120)})


def make_survivor(i, sel_t, sel_alpha=None, n_days=140, direction=1,
                  turnover=0.01, lag=LAG, family='residual_shape',
                  signal=None):
    """A survivor dict shaped like run_search's population entries, with a
    5-month test verdict at one lag."""
    sel_alpha = sel_alpha if sel_alpha is not None else 0.0005 * sel_t
    m_sel = {'alpha_mean': sel_alpha, 'alpha_tstat': sel_t, 'n_days': n_days}
    m_trn = {'alpha_mean': 0.001, 'alpha_tstat': 3.0, 'n_days': 140}
    return {
        'candidate': gen.Candidate(f's{i}', family, ('col', f'feat_{i}')),
        'direction': direction, 'target_lag': lag,
        'half_life_bars': 288.0,
        'profile_train': {lag: m_trn}, 'profile_select': {lag: m_sel},
        'metrics_train': m_trn, 'metrics_select': m_sel,
        'reward': 1.0, 'turnover': turnover,
        'signal_train': _signal_panel(),
        'signal_select': signal if signal is not None else _signal_panel(),
    }


TCFG = copy.deepcopy(get('discovery'))
TCFG['horizon_lags_bars'] = [LAG]
TCFG['promotion'].update({
    'min_select_days': 20, 'min_capture': 0.0, 'max_book_corr': 0.5,
    'econ_cost_bps': 0.0,
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

# filter 3 economics: with a real cost rate, the churner drops out
e_cfg = copy.deepcopy(TCFG)
e_cfg['promotion']['econ_cost_bps'] = 5.0
churner = make_survivor(30, sel_t=2.0, sel_alpha=0.0002, turnover=0.5)
steady = make_survivor(31, sel_t=2.0, sel_alpha=0.0002, turnover=0.001)
book_e = bt_mod.promote([churner, steady], ROLL,
                        search_mod.DiscoveryLedger(None), e_cfg)
check("economics: churner rejected, steady twin promoted",
      [p['candidate'].name for p in book_e] == ['s31'])

# filter 4: a lower-ranked duplicate of a chosen formula is skipped
top_sig = pool[5]['signal_select']
dupe = make_survivor(50, sel_t=2.2, signal=top_sig.copy())
book_d = bt_mod.promote(pool + [dupe], ROLL,
                        search_mod.DiscoveryLedger(None), TCFG)
check("duplicate: same-output formula skipped, slot passes on",
      's50' not in [p['candidate'].name for p in book_d])

# thin-lag hijack regression: a flukey thin lag must not shadow a solid one
hj_cfg = copy.deepcopy(TCFG)
hj_cfg['horizon_lags_bars'] = [72, 144]
hijack = make_survivor(60, sel_t=1.5, n_days=140)
hijack['profile_select'][72] = {'alpha_mean': 0.004, 'alpha_tstat': 4.7,
                                'n_days': 8}
book_h = bt_mod.promote([hijack], ROLL, search_mod.DiscoveryLedger(None),
                        hj_cfg)
check("hijack: verdict lands on the qualifying lag, not the thin fluke",
      len(book_h) == 1 and book_h[0]['select_lag'] == 144)

# ---------------------------------------------------------------------------
print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURES: {FAILURES}")
    sys.exit(1)
print("ALL CHOOSE CHECKS PASSED")
