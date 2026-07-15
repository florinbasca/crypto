"""
Synthetic checks for the rank-K-floors promotion (research/signals/
promotion.py) and its pooled cross-roll evidence. No database required.

1. family_lags - per-family horizon restriction resolves and falls back.
2. pooled_select_evidence - fixed-effect meta-analysis: single-month
   passthrough, sqrt(k) pooling of consistent months, opposite-direction
   months count against, unmeasured months skipped.
3. DiscoveryLedger.select_history - per-roll select metrics round-trip.
4. promote() = rank + K slots + sanity floors: the book takes the top
   book_size by capture-weighted day-equivalent pooled t; wrong-way,
   short-history and floor-failing survivors never promote; correlated
   survivors are de-duplicated; there is NO significance bar.
5. Cross-roll pooling inside promote() - consistent history raises the
   pooled evidence (and the rank); opposite-direction history flips it
   negative and the directed floor rejects.

Run: uv run tests/promotion_pooling_checks.py
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
fl_cfg = {'horizon_lags_bars': [6, 36, 72, 144, 432],
          'family_horizon_lags': {'default': [6, 36, 72, 144],
                                  'unlocks': [144, 432]}}
check("family_lags: listed family gets its own horizons",
      data_mod.family_lags('unlocks', fl_cfg) == [144, 432])
check("family_lags: unlisted family gets the default",
      data_mod.family_lags('order_flow', fl_cfg) == [6, 36, 72, 144])
check("family_lags: intersected with the actual grid",
      data_mod.family_lags('unlocks',
                           {**fl_cfg, 'horizon_lags_bars': [6, 144]})
      == [144])
check("family_lags: no config section -> full grid",
      data_mod.family_lags('unlocks',
                           {'horizon_lags_bars': [6, 36]}) == [6, 36])
check("family_lags: empty intersection falls back to the grid",
      data_mod.family_lags('unlocks',
                           {**fl_cfg, 'horizon_lags_bars': [6, 36]})
      == [6, 36])
check("family_lags: prod config concentrates slow families at 1d",
      data_mod.family_lags('dev_activity') == [144]
      and 432 not in data_mod.family_lags('order_flow'))

# ---------------------------------------------------------------------------
# 2. pooled_select_evidence
# ---------------------------------------------------------------------------
print("--- 2. fixed-effect pooling ---")
m1 = {'alpha_mean': 0.001, 'alpha_tstat': 2.0, 'n_days': 30, 'direction': 1}
p1 = bt_mod.pooled_select_evidence([m1], direction=1)
check("pooling: single month = that month's t",
      abs(p1['tstat'] - 2.0) < 1e-9 and p1['n_months'] == 1
      and p1['n_days'] == 30 and p1['sign_frac'] == 1.0)

p2 = bt_mod.pooled_select_evidence([m1, dict(m1)], direction=1)
check("pooling: two identical consistent months -> t x sqrt(2)",
      abs(p2['tstat'] - 2.0 * math.sqrt(2)) < 1e-9
      and p2['n_months'] == 2 and p2['n_days'] == 60)

flip = {**m1, 'direction': -1}
p3 = bt_mod.pooled_select_evidence([m1, flip], direction=1)
check("pooling: opposite-direction month counts AGAINST (cancels here)",
      abs(p3['tstat']) < 1e-9 and p3['sign_frac'] == 0.5)

p4 = bt_mod.pooled_select_evidence(
    [m1, {'alpha_mean': 0.0, 'alpha_tstat': 0.0, 'n_days': 30},
     {'alpha_mean': None, 'alpha_tstat': None, 'n_days': 0},
     {'alpha_mean': 0.002, 'alpha_tstat': 1.5, 'n_days': 1}], direction=1)
check("pooling: unmeasured months (t=0 / None / <2 days) skipped",
      p4['n_months'] == 1 and abs(p4['tstat'] - 2.0) < 1e-9)

check("pooling: nothing measurable -> t 0, months 0",
      bt_mod.pooled_select_evidence([])['n_months'] == 0
      and bt_mod.pooled_select_evidence([])['tstat'] == 0.0)

# precision-weighting: a tighter (smaller-se) month dominates the pooled mean
tight = {'alpha_mean': 0.0005, 'alpha_tstat': 5.0, 'n_days': 30}   # se 1e-4
loose = {'alpha_mean': -0.004, 'alpha_tstat': -1.0, 'n_days': 30}  # se 4e-3
p5 = bt_mod.pooled_select_evidence([tight, loose], direction=1)
check("pooling: inverse-variance weights (tight month dominates)",
      p5['mean'] > 0 and p5['tstat'] > 2.0,
      f"(mean {p5['mean']:.5f}, t {p5['tstat']:.2f})")

# ---------------------------------------------------------------------------
# 3. ledger select_history
# ---------------------------------------------------------------------------
print("--- 3. ledger select history ---")
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

hist = led.select_history(c_hist.hash, 144)
check("history: one entry per roll, sorted, direction carried",
      [h['roll_id'] for h in hist] == [0, 1, 2]
      and [h['direction'] for h in hist] == [1, 1, -1]
      and hist[0]['alpha_tstat'] == 2.0 and hist[1]['n_days'] == 28)
check("history: up_to_roll bounds the window",
      len(led.select_history(c_hist.hash, 144, up_to_roll=1)) == 2)
check("history: unknown lag / hash -> empty",
      led.select_history(c_hist.hash, 6) == []
      and led.select_history('nope', 144) == [])

# retention re-seed pool: distinct promoted candidates within a roll range
led.mark_promoted(0, [c_hist.hash])
led.mark_promoted(1, [c_hist.hash])
check("retention: promoted_candidates returns distinct recent promotees",
      [c.hash for c in led.promoted_candidates(0, 1)] == [c_hist.hash]
      and led.promoted_candidates(2, 5) == [])

# ---------------------------------------------------------------------------
# 3c. pooled train direction (train-only; select never consulted)
# ---------------------------------------------------------------------------
print("--- 3c. pooled train direction ---")
# train_history un-flips the stored (directed) profile: roll 2 was recorded
# with direction -1 and directed (mean -0.001, t -1.0) -> raw (+0.001, +1.0)
t_hist = led.train_history(c_hist.hash, 144)
check("train history: directed rows un-flipped to raw sign",
      [h['alpha_tstat'] for h in t_hist] == [2.0, 1.5, 1.0]
      and t_hist[2]['alpha_mean'] == 0.001)

ptd = search_mod.pooled_train_direction
month = lambda mu, t: {'alpha_mean': mu, 'alpha_tstat': t, 'n_days': 100}
check("direction: single window -> its own sign (old behavior)",
      ptd([month(0.001, 2.0)]) == 1 and ptd([month(-0.001, -2.0)]) == -1)
check("direction: strong consistent history outvotes a weak contrary window",
      ptd([month(-0.002, -3.0), month(-0.002, -3.0),
           month(0.0005, 0.8)]) == -1)
check("direction: overwhelming new evidence can still flip the sign",
      ptd([month(-0.0005, -0.5), month(0.004, 6.0)]) == 1)
check("direction: nothing measurable -> last raw sign; empty -> +1",
      ptd([month(-0.001, 0.0)]) == -1 and ptd([]) == 1
      and ptd([{'alpha_mean': None, 'alpha_tstat': None}]) == 1)

# ---------------------------------------------------------------------------
# 3b. posterior Sharpe (the ranking currency)
# ---------------------------------------------------------------------------
print("--- 3b. posterior Sharpe ---")
TAU = 1.0
ps = bt_mod.posterior_sharpe


def _expected(t, n, tau=TAU):
    return t / math.sqrt(n) * math.sqrt(365) * n / (n + 365 / tau ** 2)


check("posterior: closed form (t=2, 30 days, tau=1)",
      abs(ps(2.0, 30, TAU) - _expected(2.0, 30)) < 1e-12
      and abs(ps(2.0, 30, TAU) - 0.53) < 0.01,
      f"({ps(2.0, 30, TAU):.3f})")
check("posterior: no data / degenerate prior / NaN -> 0",
      ps(2.0, 0, TAU) == 0.0 and ps(2.0, 30, 0.0) == 0.0
      and ps(float('nan'), 30, TAU) == 0.0)
check("posterior: directed (negative evidence stays negative)",
      ps(-2.0, 30, TAU) == -ps(2.0, 30, TAU))
# A true signal accumulating months: t grows like SR_d * sqrt(n), so the
# posterior must RISE toward the true Sharpe as evidence accumulates.
SR_D = 2.0 / math.sqrt(365)   # true Sharpe-2 daily
traj = [ps(SR_D * math.sqrt(n), n, TAU) for n in (30, 120, 270, 480)]
check("posterior: true Sharpe-2 trajectory rises with months of evidence",
      all(a < b for a, b in zip(traj, traj[1:]))
      and traj[0] < 0.2 and traj[-1] > 1.1,
      f"({[round(x, 2) for x in traj]})")
# At equal t the shorter history implies a LARGER observed Sharpe, but the
# shrinkage discounts it harder - below the one-year crossover
# (sqrt(n)/(n+365) rising) the better-evidenced signal outranks the
# flashier, thinner one. Both Sharpe and t matter; neither alone decides.
check("posterior: at equal t, more evidence outranks a flashier short "
      "history (below the 1y crossover)",
      ps(2.0, 30, TAU) > ps(2.0, 10, TAU)
      and ps(2.0, 60, TAU) > ps(2.0, 30, TAU))

# ---------------------------------------------------------------------------
# 4. promote() = rank + K + floors
# ---------------------------------------------------------------------------
print("--- 4. rank + K + floors ---")
LAG = 144
ROLL = data_mod.Roll(0, pd.Timestamp('2024-01-01'), pd.Timestamp('2024-06-01'),
                     pd.Timestamp('2024-07-01'), pd.Timestamp('2024-08-01'))
_rng = np.random.default_rng(5)
_ts = pd.date_range('2024-06-01', periods=30, freq='1d')


def _signal_panel():
    return pd.DataFrame({'timestamp': np.repeat(_ts, 4),
                         'symbol': np.tile(list('WXYZ'), 30),
                         'signal': _rng.normal(size=120)})


def make_survivor(i, sel_t, sel_alpha=None, n_days=30, direction=1,
                  train_alpha=0.001, turnover=0.01, signal=None,
                  lag=LAG, family='residual_shape'):
    """A survivor dict shaped like run_search's population entries."""
    sel_alpha = sel_alpha if sel_alpha is not None else 0.0005 * sel_t
    m_sel = {'alpha_mean': sel_alpha, 'alpha_tstat': sel_t, 'n_days': n_days}
    m_trn = {'alpha_mean': train_alpha, 'alpha_tstat': 3.0, 'n_days': 100}
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
    'book_size': 3, 'min_select_days': 20,
    'min_profile_sign_agreement': 0.75, 'min_capture': 0.0,
    'max_turnover': None, 'min_rolls_survived': 0,
})

# 10 survivors: a t-ladder plus wrong-way and too-thin ones. The book must
# be exactly the top 3 by pooled t - nothing about a significance bar.
pool = ([make_survivor(i, sel_t=0.4 * i) for i in range(1, 7)]  # 0.4..2.4
        + [make_survivor(7, sel_t=-2.5), make_survivor(8, sel_t=-0.5),
           make_survivor(9, sel_t=9.9, n_days=5),      # thin evidence
           make_survivor(10, sel_t=0.0)])              # unmeasured
book = bt_mod.promote(pool, ROLL, search_mod.DiscoveryLedger(None), TCFG)
names = [p['candidate'].name for p in book]
check("rank+K: exactly book_size promoted, top pooled t first",
      names == ['s6', 's5', 's4'], f"({names})")
check("rank+K: sub-bar t promotes when it ranks (no significance gate)",
      any(p['pooled_select_tstat'] < 2.0 for p in book),
      f"(pooled ts {[round(p['pooled_select_tstat'], 2) for p in book]})")
check("floors: wrong-way, thin and unmeasured survivors never promote",
      not any(n in names for n in ('s7', 's8', 's9', 's10')))
check("annotations: promotions carry pooled stats + evidence lag",
      all(('pooled_select_tstat' in p and 'select_lag' in p
           and p['pooled_select_months'] == 1) for p in book))

# orthogonality floor: a lower-ranked clone of the top survivor (same select
# signal) is skipped and the slot goes to the next distinct candidate
top_sig = pool[5]['signal_select']
clone = make_survivor(11, sel_t=2.2, signal=top_sig.copy())
book_c = bt_mod.promote(pool + [clone], ROLL,
                        search_mod.DiscoveryLedger(None), TCFG)
check("floors: correlated clone de-duplicated by max_book_corr",
      's11' not in [p['candidate'].name for p in book_c]
      and len(book_c) == 3)

# capture floor empties the book regardless of rank
cap_cfg = copy.deepcopy(TCFG)
cap_cfg['promotion']['min_capture'] = 0.999
check("floors: capture floor overrides the quota",
      bt_mod.promote(pool, ROLL, search_mod.DiscoveryLedger(None), cap_cfg)
      == [])

# book_size 0 disables promotion entirely
zero_cfg = copy.deepcopy(TCFG)
zero_cfg['promotion']['book_size'] = 0
check("rank+K: book_size 0 promotes nothing",
      bt_mod.promote(pool, ROLL, search_mod.DiscoveryLedger(None), zero_cfg)
      == [])

# Thin-lag hijack (regression, found live at roll 3 of the first extended
# run): a flukey t on a handful of gated days must not win best-lag and then
# kill the candidate on min_days when another lag qualifies outright.
hj_cfg = copy.deepcopy(TCFG)
hj_cfg['horizon_lags_bars'] = [72, 144]
hijack = make_survivor(12, sel_t=1.5, n_days=27)     # solid at 144
hijack['profile_select'][72] = {'alpha_mean': 0.004, 'alpha_tstat': 4.7,
                                'n_days': 8}          # flukey thin at 72
book_h = bt_mod.promote([hijack], ROLL, search_mod.DiscoveryLedger(None),
                        hj_cfg)
check("hijack: qualifying lag wins over a flukey thin lag",
      len(book_h) == 1 and book_h[0]['select_lag'] == 144
      and book_h[0]['pooled_select_days'] == 27,
      f"(promoted at lag {book_h[0]['select_lag']})" if book_h
      else "(nothing promoted)")
only_thin = make_survivor(13, sel_t=4.7, sel_alpha=0.004, n_days=8)
check("hijack: candidate with ONLY thin lags still rejected",
      bt_mod.promote([only_thin], ROLL, search_mod.DiscoveryLedger(None),
                     TCFG) == [])

# Horizon parity: the same t over the same CALENDAR month must score the
# same whether it arrived as 27 daily obs (1d lag) or 9 three-day obs
# (3d lag). promote() converts obs days to calendar days before the
# posterior and the min_days floor, so slow horizons are neither charged
# sqrt(3) twice nor blocked for their first rolls by the day floor.
par_cfg = copy.deepcopy(TCFG)
par_cfg['promotion']['book_size'] = 2
par_cfg['horizon_lags_bars'] = [144, 432]
fast_s = make_survivor(30, sel_t=1.8, n_days=27, lag=144)
slow_s = make_survivor(31, sel_t=1.8, n_days=9, lag=432, family='unlocks')
book_p = bt_mod.promote([fast_s, slow_s], ROLL,
                        search_mod.DiscoveryLedger(None), par_cfg)
post = {p['candidate'].name: p['posterior_sharpe'] for p in book_p}
check("horizon parity: 3d evidence scores equal to 1d at same t/calendar",
      len(book_p) == 2 and abs(post['s30'] - post['s31']) < 1e-9,
      f"({ {k: round(v, 3) for k, v in post.items()} })")
check("horizon parity: 3d signal passes min_select_days in its FIRST month",
      all(p['pooled_select_months'] == 1 for p in book_p))

# ---------------------------------------------------------------------------
# 5. cross-roll pooling inside promote()
# ---------------------------------------------------------------------------
print("--- 5. pooling across rolls ---")
ROLL1 = data_mod.Roll(1, pd.Timestamp('2024-02-01'), pd.Timestamp('2024-07-01'),
                      pd.Timestamp('2024-08-01'), pd.Timestamp('2024-09-01'))
one_cfg = copy.deepcopy(TCFG)
one_cfg['promotion']['book_size'] = 1

# Two equal current months (t=1.6); only one has a consistent prior month.
# The pooled evidence (1.6 x sqrt(2) = 2.26) must win the single slot.
vet = make_survivor(20, sel_t=1.6)
rookie = make_survivor(21, sel_t=1.6)
led1 = search_mod.DiscoveryLedger(None)
led1.record(0, 0, vet['candidate'], direction=1, train_metrics={},
            select_metrics={}, reward=0.0, terms={}, target_lag=LAG,
            profile_json=_profile(LAG, 0.0008, 1.6, 30),
            half_life_bars=288.0, turnover=0.01)
book1 = bt_mod.promote([rookie, vet], ROLL1, led1, one_cfg)
check("cross-roll: consistent history wins the slot over an equal rookie",
      len(book1) == 1 and book1[0]['candidate'].name == 's20'
      and abs(book1[0]['pooled_select_tstat'] - 1.6 * math.sqrt(2)) < 0.05
      and book1[0]['pooled_select_months'] == 2,
      f"(pooled t {book1[0]['pooled_select_tstat']:.2f})"
      if book1 else "(nothing promoted)")

# A prior month measured under the OPPOSITE direction counts against: equal
# magnitudes cancel, pooled t = 0 fails the directed floor.
strong = make_survivor(22, sel_t=2.5)
led2 = search_mod.DiscoveryLedger(None)
led2.record(0, 0, strong['candidate'], direction=-1, train_metrics={},
            select_metrics={}, reward=0.0, terms={}, target_lag=LAG,
            profile_json=_profile(LAG, 0.00125, 2.5, 30),
            half_life_bars=288.0, turnover=0.01)
book_clean = bt_mod.promote([strong], ROLL1,
                            search_mod.DiscoveryLedger(None), one_cfg)
book_flip = bt_mod.promote([strong], ROLL1, led2, one_cfg)
check("cross-roll: opposite-direction history fails the directed floor",
      len(book_clean) == 1 and book_flip == [],
      f"(clean {len(book_clean)}, flipped-history {len(book_flip)})")

# ---------------------------------------------------------------------------
print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURES: {FAILURES}")
    sys.exit(1)
print("ALL PROMOTION POOLING CHECKS PASSED")
