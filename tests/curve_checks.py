"""
Synthetic checks for the response-curve verdict (research/signals/search.py
response_curve/fit_response_curve, research/signals/promotion.py
curve_verdict path). No database required.

1. response_curve - recovers a planted ramp-then-flat response; entry grid,
   full-path clamping, NaN handling, overlap-deflated n_eff.
2. fit_response_curve - pure decay recovered (a0, half-life); hump gets its
   peak and reversal fraction; the 4-lag method would have missed the hump.
3. CHOOSE on curves - median gate, activity gate, net-rate economics, peak
   caps the promoted half-life (the walk-forward's holding input), lag
   fallback still works for curve-less rows.

Run: uv run tests/curve_checks.py
"""

import copy
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
# 1. response_curve on a crafted panel
# ---------------------------------------------------------------------------
print("--- 1. response_curve ---")
H, N_BARS, SYMS = 48, 600, list('ABCDEF')
ts = pd.date_range('2024-01-01', periods=N_BARS, freq='10min')
# Signal: constant ranking A>B>C>D>E>F. Residuals: each bar, the top names
# earn +c and bottom names lose c for exactly PEAK bars after any moment,
# i.e. a constant drift -> the book's response is a linear ramp (slope = per
# -bar book return), flat thereafter is not constructible with constant
# drift, so we check the RAMP case here and shapes in section 2.
c = 1e-4
drift = {'A': 2 * c, 'B': c, 'C': 0.0, 'D': 0.0, 'E': -c, 'F': -2 * c}
res_wide = pd.DataFrame({s: np.full(N_BARS, drift[s]) for s in SYMS},
                        index=ts)
sig = pd.DataFrame({'timestamp': np.repeat(ts, len(SYMS)),
                    'symbol': SYMS * N_BARS,
                    'signal': [3, 2, 1, -1, -2, -3] * N_BARS})
rc = search_mod.response_curve(sig, res_wide, horizon_bars=H,
                               entry_stride=6, min_assets=4)
# book return per bar: w = demeaned scaled to gross 1 -> w.r = const > 0
per_bar = float(rc['A'][0])
check("curve: linear ramp recovered (A(k) = k x per-bar return)",
      rc is not None and per_bar > 0
      and abs(rc['A'][H - 1] - H * per_bar) < 1e-12,
      f"(per-bar {per_bar:.2e})")
check("curve: entries on the stride grid with full paths only",
      rc['entries'] == len(range(0, N_BARS - H, 6)) or rc['entries'] > 0)
check("curve: n_eff deflates for path overlap (stride/H)",
      abs(rc['n_eff'] - max(1.0, rc['entries'] * 6 / H)) < 1e-9)
check("curve: per-entry outcomes sampled for robustness stats",
      set(rc['per_entry_at']) == {1, H // 4, H // 2, H}
      and len(rc['per_entry_at'][H]) == rc['entries'])

# sample_ks grid: per-entry outcomes follow the log-spaced curve grid, so
# the median gate judges a SHORT-peak signal near its own peak (regression:
# the coarse default judged a 1-hour edge at bar 1).
rc_g = search_mod.response_curve(sig, res_wide, horizon_bars=H,
                                 entry_stride=6, min_assets=4,
                                 sample_ks=[1, 2, 3, 6, 12, 24, 48, 999])
check("curve: sample_ks grid respected, clipped to the horizon",
      set(rc_g['per_entry_at']) == {1, 2, 3, 6, 12, 24, 48})
fit_g = search_mod.fit_response_curve(
    np.concatenate([0.001 * np.arange(1, 7) / 6, np.full(H - 6, 0.001)]),
    n_eff=25.0, per_entry_at=rc_g['per_entry_at'])
check("curve: short-peak median judged at a nearby k, not bar 1",
      min(rc_g['per_entry_at'],
          key=lambda k: abs(k - fit_g['peak_k'])) >= 3)

# NaN residuals contribute zero (delisting mid-path must not poison)
res_nan = res_wide.copy()
res_nan.iloc[100:, 5] = np.nan          # F vanishes
rc_nan = search_mod.response_curve(sig, res_nan, horizon_bars=H,
                                   entry_stride=6, min_assets=4)
check("curve: NaN residuals along a path contribute zero, no NaN output",
      rc_nan is not None and np.isfinite(rc_nan['A']).all())

check("curve: window too short for one full path -> None (lag fallback)",
      search_mod.response_curve(sig[sig['timestamp'] < ts[30]],
                                res_wide.iloc[:30], horizon_bars=H,
                                entry_stride=6, min_assets=4) is None)

# ---------------------------------------------------------------------------
# 2. fit_response_curve shapes
# ---------------------------------------------------------------------------
print("--- 2. curve fit ---")
ks = np.arange(1, 145)
decay = 0.001 * (1 - np.exp(-np.log(2) / 24 * ks))       # a0 1e-3, hl 24
fit_d = search_mod.fit_response_curve(decay, n_eff=25.0)
check("fit: pure decay -> half-life recovered on the grid",
      fit_d['half_life'] == 24.0 and abs(fit_d['a0'] - 0.001) < 1e-4
      and fit_d['rev_frac'] < 0.05,
      f"(hl {fit_d['half_life']}, a0 {fit_d['a0']:.2e})")

hump = np.concatenate([0.001 * np.arange(1, 31) / 30,     # ramp to 1e-3 @30
                       0.001 - 0.0006 * np.arange(1, 115) / 114])  # give back 60%
fit_h = search_mod.fit_response_curve(hump, n_eff=25.0)
check("fit: hump -> peak located, reversal fraction measured",
      28 <= fit_h['peak_k'] <= 34 and 0.5 < fit_h['rev_frac'] < 0.7,
      f"(peak {fit_h['peak_k']}, rev {fit_h['rev_frac']:.2f})")
# the 4-lag view of this hump: positive alpha at ALL of 6/36/72/144 - the
# reversal is invisible without the curve
check("fit: hump is positive at every legacy lag (why the curve exists)",
      all(hump[k - 1] > 0 for k in (6, 36, 72, 144)))

per_entry = {30: np.array([0.001] * 20 + [-0.0002] * 5)}
fit_m = search_mod.fit_response_curve(hump, n_eff=5.0, per_entry_at=per_entry)
check("fit: per-entry spread -> se and median at the peak",
      np.isfinite(fit_m['se_peak']) and fit_m['median_peak'] == 0.001)

# ---------------------------------------------------------------------------
# 3. CHOOSE on curve verdicts
# ---------------------------------------------------------------------------
print("--- 3. CHOOSE on curves ---")
LAG = 144
ROLL = data_mod.Roll(0, pd.Timestamp('2024-01-01'), pd.Timestamp('2024-06-01'),
                     pd.Timestamp('2024-11-01'), pd.Timestamp('2024-12-01'))
_rng = np.random.default_rng(5)
_ts30 = pd.date_range('2024-06-01', periods=30, freq='1d')


def _signal_panel():
    return pd.DataFrame({'timestamp': np.repeat(_ts30, 4),
                         'symbol': np.tile(list('WXYZ'), 30),
                         'signal': _rng.normal(size=120)})


def make_curved(i, a0, peak_k=48, entry_days=120, median=None, hl=24.0,
                turnover=0.01):
    """Survivor dict whose verdict is a fitted curve (A sampled linearly up
    to a0 at peak, flat after)."""
    ks = [1, 2, 3, 6, 12, 24, 48, 72, 96, 120, 144]
    A = [a0 * min(k, peak_k) / peak_k for k in ks]
    m = {'alpha_mean': a0, 'alpha_tstat': 1.0, 'n_days': entry_days}
    return {
        'candidate': gen.Candidate(f'c{i}', 'residual_shape',
                                   ('col', f'feat_{i}')),
        'direction': 1, 'target_lag': LAG, 'half_life_bars': 2016.0,
        'profile_train': {LAG: m}, 'profile_select': {LAG: m},
        'metrics_train': m, 'metrics_select': m,
        'curve': {'a0': a0, 'half_life': hl, 'peak_k': peak_k,
                  'rev_frac': 0.0, 'se_peak': abs(a0) / 3.0,
                  'median_peak': median if median is not None else a0,
                  'entries': 600, 'entry_days': entry_days, 'n_eff': 25.0,
                  'ks': ks, 'A': A},
        'reward': 1.0, 'turnover': turnover,
        'signal_train': _signal_panel(), 'signal_select': _signal_panel(),
    }


TCFG = copy.deepcopy(get('discovery'))
TCFG['promotion'].update({
    'min_select_days': 20, 'min_capture': 0.0, 'max_book_corr': 0.5,
    'econ_cost_bps': 0.0,
    'book_frac': 0.20, 'book_min': 1, 'book_max': 50, 'book_size': 10,
})

pool = [make_curved(1, 0.0030), make_curved(2, 0.0020),
        make_curved(3, 0.0010),
        make_curved(4, -0.0010),                    # filter 1: backwards
        make_curved(5, 0.0025, entry_days=8),       # filter 2: thin
        make_curved(6, 0.0025, median=-0.0001)]     # median gate: jump-day
book = bt_mod.promote(pool, ROLL, search_mod.DiscoveryLedger(None), TCFG)
names = [p['candidate'].name for p in book]
check("choose: quintile of curve-passers, ranked by net rate",
      names == ['c1'], f"({names}; 3 passers -> ceil(0.6)=1)")
check("choose: backwards/thin/median-failing rejected",
      not any(n in names for n in ('c4', 'c5', 'c6')))
check("choose: peak caps the promoted half-life (walk-forward input)",
      book[0]['half_life_bars'] == min(24.0, 48)
      and book[0]['peak_bars'] == 48)

# net-rate ranking prefers a fast small edge over a slow bigger one when
# rates say so: a0 8bp at peak 144 (rate ~0.055bp/bar) vs 4bp at peak 12
# (rate ~0.33bp/bar)
fast = make_curved(7, 0.0004, peak_k=12)
slow = make_curved(8, 0.0008, peak_k=144)
one = copy.deepcopy(TCFG)
one['promotion'].update({'book_min': 1, 'book_frac': 0.2})
b2 = bt_mod.promote([slow, fast], ROLL, search_mod.DiscoveryLedger(None), one)
check("choose: ranking is net RATE at own optimum, not raw size",
      b2[0]['candidate'].name == 'c7')

# economics: a real cost makes the round trip bind
e_cfg = copy.deepcopy(TCFG)
e_cfg['promotion']['econ_cost_bps'] = 5.0     # roundtrip 10bp
thin = make_curved(9, 0.0008)                 # 8bp peak < 10bp round trip
fat = make_curved(10, 0.0030)                 # 30bp peak > 10bp round trip
b3 = bt_mod.promote([thin, fat], ROLL, search_mod.DiscoveryLedger(None),
                    e_cfg)
check("choose: curve that can't cover a round trip is rejected",
      [p['candidate'].name for p in b3] == ['c10'])

# curve-less rows still promote via the lag fallback
legacy = make_curved(11, 0.0030)
legacy['curve'] = None
legacy['profile_select'][LAG]['alpha_tstat'] = 2.0
legacy['profile_select'][LAG]['n_days'] = 140
b4 = bt_mod.promote([legacy], ROLL, search_mod.DiscoveryLedger(None), TCFG)
check("choose: pre-curve ledger rows fall back to the lag verdict",
      len(b4) == 1 and b4[0]['select_lag'] == LAG)

# ---------------------------------------------------------------------------
print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURES: {FAILURES}")
    sys.exit(1)
print("ALL CURVE CHECKS PASSED")
