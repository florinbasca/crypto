"""
Synthetic checks for the membership backdating that makes the 2022-01 data
extension point-in-time honest (etl/universe.py). No database required.

1. extend_membership_start - only the SEEDED cohort (spells anchored at the
   old seed stamp) is backdated; snapshot-accrued spells are untouched; each
   extension is clipped at the name's true first perp trade.
2. clip_seed_at_first_trade - a fresh seed never predates a name's first
   trade, and never goes below the data start.

Run: uv run tests/membership_extension_checks.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from etl.universe import (clip_seed_at_first_trade, evolve_membership,
                          extend_membership_start)

FAILURES = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name} {detail}")
    if not cond:
        FAILURES.append(name)


T = pd.Timestamp
OLD, NEW = T('2023-01-01'), T('2022-01-01')

# seeded cohort at OLD + one real snapshot spell opened later
mem = pd.DataFrame({
    'symbol': ['AAA', 'BBB', 'CCC', 'DDD'],
    'valid_from': [OLD, OLD, OLD, T('2026-05-01')],   # DDD = real snapshot
    'valid_to': [pd.NaT, pd.NaT, T('2026-06-01'), pd.NaT],
})
first_trade = {'AAA': T('2020-06-01'),    # pre-window -> clip to NEW
               'BBB': T('2022-07-15'),    # inside window -> its own date
               'DDD': T('2021-01-01')}    # snapshot spell: must NOT move
ext = extend_membership_start(mem, NEW, first_trade)
vf = dict(zip(ext['symbol'], pd.to_datetime(ext['valid_from'])))
check("extend: pre-window listing clipped to the new start",
      vf['AAA'] == NEW)
check("extend: in-window listing backdated to its first trade",
      vf['BBB'] == T('2022-07-15'))
check("extend: seeded name with no listings info -> new start",
      vf['CCC'] == NEW)
check("extend: snapshot-accrued spell untouched",
      vf['DDD'] == T('2026-05-01'))
check("extend: valid_to spells preserved",
      ext.loc[ext['symbol'] == 'CCC', 'valid_to'].iloc[0] == T('2026-06-01'))
check("extend: no-op when the table already starts at/<= the new start",
      extend_membership_start(ext, NEW, first_trade)['valid_from']
      .equals(ext['valid_from']))

# fresh seed: evolve from empty, then clip at first trades
seeded, n_new, _ = evolve_membership(None, ['AAA', 'BBB', 'EEE'],
                                     T('2026-07-01'), NEW)
clipped = clip_seed_at_first_trade(seeded, first_trade, NEW)
cvf = dict(zip(clipped['symbol'], pd.to_datetime(clipped['valid_from'])))
check("fresh seed: cohort seeded at the data start", n_new == 3
      and (pd.to_datetime(seeded['valid_from']) == NEW).all())
check("fresh seed: late listing raised to its first trade",
      cvf['BBB'] == T('2022-07-15'))
check("fresh seed: pre-window and unknown listings stay at the start",
      cvf['AAA'] == NEW and cvf['EEE'] == NEW)
check("fresh seed: empty/no-info inputs are no-ops",
      clip_seed_at_first_trade(seeded, {}, NEW)['valid_from']
      .equals(seeded['valid_from'])
      and clip_seed_at_first_trade(None, first_trade, NEW) is None)

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURES: {FAILURES}")
    sys.exit(1)
print("ALL MEMBERSHIP EXTENSION CHECKS PASSED")
