"""
Synthetic checks for the walk-forward analysis helpers
(research/portfolio/walk_forward_analysis.py). No database required.

1. book_pnl / gross1 - pnl of aligned weights x forwards; gross-1 scaling.
2. bpd - per-bar pnl converts to bp/day with BARS_PER_DAY.
3. tstat - known mean/std case; nan on degenerate input.
4. decay_kept - linear interpolation of the decay grid; nan when fresh
   alpha is absent or non-positive.
5. alpha_half_life - exact crossing, interpolated crossing, and the
   never-crosses case (inf).
6. alpha_shape - back-loaded (growing) alpha is detected; decaying and
   dead alphas are not; a never-decaying half-life formats readably.

Run: uv run tests/walk_forward_analysis_checks.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from config import BARS_PER_DAY
from research.portfolio.walk_forward_analysis import (
    alpha_half_life, alpha_shape, book_pnl, bpd, decay_kept, fmt_bars,
    gross1, tstat)

FAILURES = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name} {detail}")
    if not cond:
        FAILURES.append(name)


# 1. book_pnl / gross1 ------------------------------------------------------
idx = pd.date_range('2025-01-01', periods=3, freq='10min')
w = pd.DataFrame({'A': [0.5, 0.5, 0.5], 'B': [-0.5, -0.5, -0.5]}, index=idx)
fwd = pd.DataFrame({'A': [0.01, 0.0, np.nan], 'B': [0.0, 0.02, 0.01]},
                   index=idx)
# 0.5*0.01 + (-0.5)*0.02 + (-0.5)*0.01 with the NaN contributing zero
check("book_pnl sums w.fwd with NaN as zero",
      np.isclose(book_pnl(w, fwd), 0.005 - 0.01 - 0.005))
check("book_pnl aligns on the intersection",
      np.isclose(book_pnl(w.iloc[:2], fwd), 0.005 - 0.01))

g = gross1(pd.DataFrame({'A': [2.0, 0.0], 'B': [-2.0, 0.0]}))
check("gross1 scales each bar to gross 1",
      np.isclose(g.abs().sum(axis=1).iloc[0], 1.0))
check("gross1 leaves all-zero bars NaN (no book)",
      g.iloc[1].isna().all())

# 2. bpd --------------------------------------------------------------------
check("bpd: 1bp per bar -> BARS_PER_DAY bp/day",
      np.isclose(bpd(1e-4 * 10, 10), BARS_PER_DAY))

# 3. tstat ------------------------------------------------------------------
x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
check("tstat matches mean/std*sqrt(n)",
      np.isclose(tstat(x), x.mean() / x.std(ddof=1) * np.sqrt(5)))
check("tstat nan on constant series", np.isnan(tstat(np.ones(5))))
check("tstat nan on short series", np.isnan(tstat(np.array([1.0, 2.0]))))

# 4. decay_kept -------------------------------------------------------------
decay = {0: 4.0, 6: 2.0, 36: 1.0}
check("decay_kept at a grid point", np.isclose(decay_kept(decay, 6), 0.5))
check("decay_kept interpolates between points",
      np.isclose(decay_kept(decay, 3), 0.75))
check("decay_kept clamps beyond the grid",
      np.isclose(decay_kept(decay, 100), 0.25))
check("decay_kept nan without positive fresh alpha",
      np.isnan(decay_kept({0: -1.0, 6: 0.5}, 3)))

# 5. alpha_half_life --------------------------------------------------------
check("half-life: exact crossing at the midpoint of a linear decay",
      np.isclose(alpha_half_life({0: 4.0, 6: 2.0, 36: 1.0}), 6.0))
# from 100% at lag 0 to 25% at lag 8: crosses 50% at lag 8 * (0.5/0.75)
check("half-life: interpolated crossing",
      np.isclose(alpha_half_life({0: 4.0, 8: 1.0}), 8 * 0.5 / 0.75))
check("half-life: never crosses -> inf",
      alpha_half_life({0: 4.0, 6: 3.0, 36: 2.5}) == float('inf'))
check("half-life: nan without positive fresh alpha",
      np.isnan(alpha_half_life({0: 0.0, 6: 1.0})))

# 6. alpha_shape / fmt_bars -------------------------------------------------
lag, best, back = alpha_shape({0: 2.0, 6: 2.5, 144: 5.0})
check("alpha_shape flags growing alpha as back-loaded",
      back and lag == 144 and np.isclose(best, 5.0))
check("alpha_shape: decaying alpha is not back-loaded",
      not alpha_shape({0: 4.0, 6: 2.0, 144: 1.0})[2])
check("alpha_shape: negative-fresh but positive-lagged is back-loaded",
      alpha_shape({0: -1.0, 6: 0.5, 144: 2.0})[2])
check("alpha_shape: dead everywhere is not back-loaded",
      not alpha_shape({0: -1.0, 6: -0.5, 144: -2.0})[2])
check("alpha_shape: small wiggle within MATERIAL is not back-loaded",
      not alpha_shape({0: 2.0, 6: 2.1, 144: 1.0})[2])
check("fmt_bars: infinite half-life formats readably, not n/a",
      "grid" in fmt_bars(float('inf')))
check("fmt_bars: nan is n/a", fmt_bars(float('nan')) == "n/a")

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURES: {FAILURES}")
    sys.exit(1)
print("all checks passed")
