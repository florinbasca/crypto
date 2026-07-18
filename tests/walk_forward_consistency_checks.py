"""
Synthetic checks for the discovery -> portfolio consistency fixes and the
null-control machinery (research/portfolio/walk_forward.py,
research/lib/discovered.py). No database required.

1. per_bar_alpha - expected returns are ONE unit (per bar): per-bet alpha
   scales LINEARLY with holding time (the old /sqrt(h) rank-IC convention
   overstated slow buckets by sqrt(h)).
2. month_book - roll-specific direction (dir_of) and evidence-lag buckets.
3. entries_from_promotions - the registry entry's lag prefers the
   promotion's evidence lag (select_lag) over the train-best target_lag.
4. apply_signal_control - sign_flip is exact negation; shuffle preserves
   each bar's value multiset and NaN membership while destroying order
   (deterministic per seed); random keeps shape/NaN mask; 'none' is
   identity.
5. DiscoveryLedger.run_stamp - provenance columns reach every recorded row.

Run: uv run tests/walk_forward_consistency_checks.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from research.portfolio.walk_forward import (WalkForwardPortfolio,
                                             apply_signal_control,
                                             per_bar_alpha)
from research.lib.discovered import entries_from_promotions
from research.signals import generation as gen
from research.signals.search import DiscoveryLedger

FAILURES = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name} {detail}")
    if not cond:
        FAILURES.append(name)


# ---------------------------------------------------------------------------
# 1. per-bar alpha units
# ---------------------------------------------------------------------------
print("--- 1. expected-return units ---")
check("units: per-bet alpha converts to per-bar LINEARLY (alpha/h)",
      abs(per_bar_alpha(0.0144, 144) - 0.0001) < 1e-15
      and abs(per_bar_alpha(0.0006, 6) - 0.0001) < 1e-15)
check("units: equal per-bar edge scores equal across horizons "
      "(no sqrt(h) inflation of slow buckets)",
      abs(per_bar_alpha(0.0144, 144) - per_bar_alpha(0.0006, 6)) < 1e-15)
check("units: h below one bar clamps to 1",
      per_bar_alpha(0.001, 0.0) == 0.001)

# ---------------------------------------------------------------------------
# 2. month_book: roll-specific direction + evidence-lag buckets
# ---------------------------------------------------------------------------
print("--- 2. month_book ---")
wf = object.__new__(WalkForwardPortfolio)   # month_book uses no state
meta = [
    {'name': 'disc_a', 'lag': 36, 'direction': -1, 'ic': 0.002,
     'half_life': 288.0, 'turnover': 0.02},
    {'name': 'disc_b', 'lag': 36, 'direction': 1, 'ic': 0.001,
     'half_life': None, 'turnover': 0.05},
    {'name': 'disc_c', 'lag': 144, 'direction': 1, 'ic': 0.003,
     'half_life': 144.0, 'turnover': 0.01},
]
(selected, weights, lag_of, dir_of, bucket_ic, bucket_h,
 bucket_to, bucket_hl, bucket_cv) = WalkForwardPortfolio.month_book(wf, meta)
check("month_book: buckets keyed by the promoted (evidence) lag",
      set(selected) == {'36b', '144b'}
      and selected['36b'] == ['disc_a', 'disc_b'])
check("month_book: dir_of carries each ROLL's fitted direction",
      dir_of == {'disc_a': -1, 'disc_b': 1, 'disc_c': 1})
check("month_book: in-bucket weights proportional to |train alpha|",
      abs(weights['36b']['disc_a'] - 2 / 3) < 1e-12
      and abs(weights['36b']['disc_b'] - 1 / 3) < 1e-12)
check("month_book: lag_of matches the bucket lag per signal",
      lag_of == {'disc_a': 36, 'disc_b': 36, 'disc_c': 144})

# ---------------------------------------------------------------------------
# 3. registry lag prefers the evidence lag
# ---------------------------------------------------------------------------
print("--- 3. registry evidence-lag preference ---")
cand = gen.Candidate('x', 'residual_shape', ('col', 'res_zscore'))
row = {'cand_hash': cand.hash, 'family': 'residual_shape', 'roll_id': 0,
       'direction': -1, 'candidate_json': cand.to_json(),
       'select_alpha_tstat': 2.0, 'half_life_bars': 288.0,
       'target_lag': 6, 'select_lag': 72}
entries = entries_from_promotions(pd.DataFrame([row]))
e = list(entries.values())[0]
check("registry: entry lag = select_lag (evidence), not target_lag",
      e['signal_def'].lag == 72)
old = dict(row)
del old['select_lag']
e_old = list(entries_from_promotions(pd.DataFrame([old])).values())[0]
check("registry: pre-select_lag tables fall back to target_lag",
      e_old['signal_def'].lag == 6)

# ---------------------------------------------------------------------------
# 4. null-control transforms
# ---------------------------------------------------------------------------
print("--- 4. placebo controls ---")
idx = pd.date_range('2024-01-01', periods=50, freq='10min')
rng = np.random.default_rng(3)
panel = pd.DataFrame(rng.standard_normal((50, 6)), index=idx,
                     columns=list('ABCDEF'))
panel.iloc[::7, 2] = np.nan                     # membership holes
comps = {'36b': panel}

check("control: 'none' is identity",
      apply_signal_control(comps, 'none', 0)['36b'] is panel)

flip = apply_signal_control(comps, 'sign_flip', 0)['36b']
check("control: sign_flip is exact negation (NaNs preserved)",
      np.allclose(flip.values, -panel.values, equal_nan=True))

sh1 = apply_signal_control(comps, 'shuffle', 1)['36b']
sh1b = apply_signal_control(comps, 'shuffle', 1)['36b']
sh2 = apply_signal_control(comps, 'shuffle', 2)['36b']
row_ok = all(
    sorted(panel.iloc[i].dropna()) == sorted(sh1.iloc[i].dropna())
    and panel.iloc[i].isna().equals(sh1.iloc[i].isna())
    for i in range(len(panel)))
check("control: shuffle preserves each bar's values + NaN membership",
      row_ok)
check("control: shuffle actually reorders and is seed-deterministic",
      not np.allclose(sh1.values, panel.values, equal_nan=True)
      and np.allclose(sh1.values, sh1b.values, equal_nan=True)
      and not np.allclose(sh1.values, sh2.values, equal_nan=True))

rnd = apply_signal_control(comps, 'random', 5)['36b']
check("control: random keeps shape + NaN mask, replaces values",
      rnd.shape == panel.shape
      and rnd.isna().equals(panel.isna())
      and not np.allclose(rnd.values, panel.values, equal_nan=True))
try:
    apply_signal_control(comps, 'bogus', 0)
    check("control: unknown mode raises", False)
except ValueError:
    check("control: unknown mode raises", True)

# ---------------------------------------------------------------------------
# 5. provenance stamp
# ---------------------------------------------------------------------------
print("--- 5. run provenance ---")
led = DiscoveryLedger(None)
led.run_stamp = {'run_id': 'r1', 'config_hash': 'c1', 'data_hash': 'd1',
                 'git_sha': 'g1'}
led.record(0, 0, cand, 1, {}, {}, 0.5, {}, target_lag=36)
r = led.to_frame().iloc[0]
check("stamp: run/config/data/git columns on every ledger row",
      r['run_id'] == 'r1' and r['config_hash'] == 'c1'
      and r['data_hash'] == 'd1' and r['git_sha'] == 'g1')

# ---------------------------------------------------------------------------
print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURES: {FAILURES}")
    sys.exit(1)
print("ALL WALK-FORWARD CONSISTENCY CHECKS PASSED")
