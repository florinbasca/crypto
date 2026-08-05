"""
Equivalence checks for the WINDOWED panel build in research/signals/data.py.

discovery.py used to build one whole-history panel (~12.4 GB at 177 float32
features) and slice it per roll. It now calls build_panel(start=, end=) once
per roll. That is only safe if the two paths produce the SAME numbers, and two
columns are window-sensitive:

  - fwd_{L}b   forward target: needs target_lag_bars AFTER the last kept bar
  - is_liquid  trailing-volume rank: needs liquidity_window_bars BEFORE the
               first kept bar

build_panel pads the load on both sides and trims afterwards. These checks
build a real panel both ways over the same span and assert they agree cell for
cell - if the padding is ever dropped or mis-sized, the target column silently
fills with NaN near the window end and the liquidity flag flips near the start,
neither of which raises.

Also checks panel_fingerprint() against a real build, because the run
provenance stamp (data_hash) is derived from it and must stay comparable with
ledger rows written before this change.

Needs the DB. Run: uv run tests/discovery_panel_window_checks.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from config import get
from research.signals import data as data_mod

FAILURES = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        FAILURES.append(name)


cfg = dict(get('discovery'))
# A short, mid-history span: far from the data edges so both paths have real
# padding available on each side (the interesting case).
OUTER_LO, OUTER_HI = pd.Timestamp('2024-05-01'), pd.Timestamp('2024-09-01')
WIN_LO, WIN_HI = pd.Timestamp('2024-06-01'), pd.Timestamp('2024-08-01')

import polars as pl
avail = pl.read_parquet('db/features', n_rows=1).columns
fam = data_mod.resolve_family_columns(avail, cfg)
cols = data_mod.all_family_columns(fam)[:25]        # 25 features is plenty
assert cols, "no feature columns resolved - the comparison would be vacuous"
print(f"comparing on {len(cols)} feature columns, "
      f"window {WIN_LO.date()}..{WIN_HI.date()}\n")

# Path A: build the OUTER span (as the old code did for all history), slice.
outer = data_mod.build_panel(cols, cfg, start=OUTER_LO, end=OUTER_HI)
sliced = data_mod.slice_window(outer, WIN_LO, WIN_HI, 0)
sliced = sliced.sort_values(['symbol', 'timestamp']).reset_index(drop=True)

# Path B: build the WINDOW directly, as discovery.py now does per roll.
windowed = data_mod.build_panel(cols, cfg, start=WIN_LO, end=WIN_HI)
windowed = windowed.sort_values(['symbol', 'timestamp']).reset_index(drop=True)

print(f"build-then-slice: {len(sliced):,} rows | windowed: {len(windowed):,} rows\n")

check("same row count", len(sliced) == len(windowed),
      f"({len(sliced):,} vs {len(windowed):,})")
check("same columns", list(sliced.columns) == list(windowed.columns))
check("same (symbol, timestamp) keys",
      sliced[['symbol', 'timestamp']].equals(windowed[['symbol', 'timestamp']]))

tcol = data_mod.target_col(int(cfg['target_lag_bars']))
num = [c for c in sliced.columns
       if c not in ('timestamp', 'symbol')
       and pd.api.types.is_numeric_dtype(sliced[c])
       and not pd.api.types.is_bool_dtype(sliced[c])]
nonnum = [c for c in sliced.columns
          if c not in ('timestamp', 'symbol') and c not in num]

# The forward target is the one column that CANNOT match bit-for-bit, and the
# reason is pandas, not the padding: rolling().sum() slides an incremental
# accumulator (add the entering bar, subtract the leaving one), so its rounding
# depends on where the window began. A build that starts earlier accumulates a
# different last bit. Everything else is a plain merge or comparison and must
# be exact. The tolerance below is ~1e4 x machine epsilon on the observed
# magnitudes and ~1e9 x tighter than anything that could move a decile bucket,
# so a genuine padding regression (NaNs, or a shifted window) still fails hard.
ROLLING_ATOL = 1e-12
diffs = {}
for c in num:
    a = sliced[c].to_numpy(dtype=float)
    b = windowed[c].to_numpy(dtype=float)
    both_nan = np.isnan(a) & np.isnan(b)
    # NaN in exactly one side => inf => a real mismatch, not silently ignored.
    d = np.abs(np.where(both_nan, 0.0, np.nan_to_num(a - b, nan=np.inf)))
    diffs[c] = float(np.max(d)) if len(d) else 0.0

exact = {c: m for c, m in diffs.items() if c != tcol}
worst_col = max(exact, key=exact.get) if exact else None
check(f"all {len(exact)} non-target numeric columns match EXACTLY",
      all(m == 0.0 for m in exact.values()),
      f"(worst |diff| {exact.get(worst_col, 0.0):.3e} in {worst_col})")

scale = float(np.nanmax(np.abs(sliced[tcol].to_numpy(dtype=float))))
check(f"forward target '{tcol}' matches to floating-point noise",
      diffs[tcol] <= ROLLING_ATOL,
      f"(|diff| {diffs[tcol]:.3e} vs values up to {scale:.3e} "
      f"= {diffs[tcol] / max(scale, 1e-300):.1e} relative)")
check(f"all {len(nonnum)} non-numeric columns match exactly",
      all(sliced[c].equals(windowed[c]) for c in nonnum),
      f"({nonnum})")
check("the comparison actually covered the features",
      len([c for c in num if c in cols]) >= 20,
      f"({len([c for c in num if c in cols])} of {len(cols)} feature cols present)")

# The two window-sensitive columns, called out explicitly. If the forward
# padding were dropped, the last target_lag_bars of every symbol would go NaN
# in the windowed build and this NaN-share comparison would catch it.
check(f"forward target '{tcol}' is padded, not truncated to NaN",
      windowed[tcol].isna().mean() == sliced[tcol].isna().mean(),
      f"(NaN share {windowed[tcol].isna().mean():.4f} vs "
      f"{sliced[tcol].isna().mean():.4f})")
check(f"'{tcol}' is not simply all-NaN (the test would be vacuous)",
      windowed[tcol].notna().any(),
      f"({windowed[tcol].notna().mean() * 100:.0f}% populated)")
check("is_liquid matches (padding BEFORE the window)",
      sliced['is_liquid'].equals(windowed['is_liquid']))
check("is_liquid is not degenerate",
      0 < windowed['is_liquid'].mean() < 1,
      f"({windowed['is_liquid'].mean() * 100:.0f}% liquid)")

# Padding must actually be REMOVED, not left in the output.
check("windowed build returns no bar before its start",
      windowed['timestamp'].min() >= WIN_LO,
      f"(min {windowed['timestamp'].min()})")
check("windowed build returns no bar at/after its end",
      windowed['timestamp'].max() < WIN_HI,
      f"(max {windowed['timestamp'].max()})")

# --- fingerprint: must equal what a real build reports -----------------------
fp_cfg = dict(cfg)
fp_cfg['start_date'], fp_cfg['end_date'] = OUTER_LO, OUTER_HI
fp = data_mod.panel_fingerprint(fp_cfg)
check("fingerprint row count == a real build's",
      fp['n_rows'] == len(outer), f"({fp['n_rows']:,} vs {len(outer):,})")
check("fingerprint span == a real build's",
      fp['ts_min'] == outer['timestamp'].min()
      and fp['ts_max'] == outer['timestamp'].max())
check("fingerprint symbol set == a real build's",
      fp['symbols'] == sorted(outer['symbol'].unique()),
      f"({len(fp['symbols'])} symbols)")

print()
if FAILURES:
    print(f"FAILED {len(FAILURES)}: {FAILURES}")
    sys.exit(1)
print("ALL PANEL-WINDOW CHECKS PASSED")
