"""
Equivalence checks for the walk-forward signal-selection speedups:

1. grouped_nw_tstat  vs  per-signal _nw_tstat (vectorized relevance t-stat).
2. precomputed |corr| matrix lookup  vs  pandas corrwith (de-correlation prune).

Each fast path must be bit-for-bit (within fp tolerance) equal to the original
reference on synthetic data. No database required.

Run: uv run tests/walk_forward_selection_speedups_checks.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from research.portfolio.walk_forward import _nw_tstat, grouped_nw_tstat

rng = np.random.default_rng(7)
FAILURES = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name} {detail}")
    if not cond:
        FAILURES.append(name)


# ---------------------------------------------------------------------------
# 1. grouped_nw_tstat == per-group _nw_tstat
# ---------------------------------------------------------------------------
def make_daily_stats(n_signals=40):
    """Synthetic signal_daily_stats-like frame: per (signal, date) ic_sum/n_cs.

    Mixes group lengths (drives the auto NW lag), serial correlation, NaN days,
    and short groups (<3 obs) to exercise every _nw_tstat branch."""
    rows = []
    base = pd.Timestamp('2023-01-01')
    for s in range(n_signals):
        length = int(rng.integers(2, 400))           # includes <3 -> 0.0 branch
        # AR(1) IC series so the HAC correction actually bites
        ic = np.zeros(length)
        phi = rng.uniform(-0.6, 0.6)
        for t in range(1, length):
            ic[t] = phi * ic[t - 1] + rng.normal(0, 0.05)
        ic += rng.normal(0, 0.02)
        n_cs = rng.integers(5, 50, size=length).astype(float)
        ic_sum = ic * n_cs
        # sprinkle NaN days (n_cs == 0 -> ic_day NaN) and explicit NaN ic_sum
        zero_days = rng.random(length) < 0.05
        n_cs[zero_days] = 0.0
        nan_days = rng.random(length) < 0.05
        ic_sum[nan_days] = np.nan
        dates = [base + pd.Timedelta(days=int(t)) for t in range(length)]
        for t in range(length):
            rows.append({'signal_name': f'sig_{s:03d}', 'date': dates[t],
                         'ic_sum': ic_sum[t], 'n_cs': n_cs[t]})
    df = pd.DataFrame(rows)
    # shuffle so neither path can rely on input ordering
    return df.sample(frac=1.0, random_state=3).reset_index(drop=True)


def ref_nw_per_group(dd, lags):
    """Original: sort by date, group, apply _nw_tstat (matches old window_stats)."""
    d = dd.sort_values('date')
    return d.groupby('signal_name')['ic_day'].apply(
        lambda s: _nw_tstat(s.values, lags))


d = make_daily_stats()
ic_mean_index = pd.Index(sorted(d['signal_name'].unique()))
dd = d.assign(ic_day=d['ic_sum'] / d['n_cs'].replace(0, np.nan))

for lags in ('auto', 0, 2, 5):
    ref = ref_nw_per_group(dd, lags).reindex(ic_mean_index).fillna(0.0)
    fast = grouped_nw_tstat(dd, lags).reindex(ic_mean_index).fillna(0.0)
    max_abs = float((ref - fast).abs().max())
    check(f"grouped_nw_tstat lags={lags}", max_abs < 1e-9,
          f"max|diff|={max_abs:.2e}")


# ---------------------------------------------------------------------------
# 2. precomputed |corr| matrix == corrwith, in the greedy prune
# ---------------------------------------------------------------------------
def greedy_corrwith(rets, ranked, thresh):
    selected = []
    for sig in ranked:
        if not selected:
            selected.append(sig)
            continue
        if sig in rets.columns and all(s in rets.columns for s in selected):
            if rets[selected].corrwith(rets[sig]).abs().max() > thresh:
                continue
        selected.append(sig)
    return selected


def greedy_matrix(rets, ranked, thresh):
    corr_abs = rets.corr().abs() if rets.shape[1] > 1 else pd.DataFrame()
    selected = []
    for sig in ranked:
        if not selected:
            selected.append(sig)
            continue
        cols = corr_abs.columns
        if (not corr_abs.empty and sig in cols
                and all(s in cols for s in selected)):
            if corr_abs.loc[selected, sig].max() > thresh:
                continue
        selected.append(sig)
    return selected


T, K = 200, 25
factors = rng.normal(0, 1, (T, 4))
cols = {}
for s in range(K):
    load = rng.normal(0, 1, 4) * (rng.random(4) < 0.5)   # share factors -> corr
    cols[f'sig_{s:03d}'] = factors @ load + rng.normal(0, 0.5, T)
rets = pd.DataFrame(cols, index=pd.date_range('2023-01-01', periods=T, freq='D'))
rets = rets.mask(rng.random(rets.shape) < 0.05)          # pairwise-NaN handling
ranked = list(rets.columns)
rng.shuffle(ranked)

for thresh in (0.3, 0.5, 0.7):
    a = greedy_corrwith(rets, ranked, thresh)
    b = greedy_matrix(rets, ranked, thresh)
    check(f"de-corr prune thresh={thresh}", a == b,
          f"corrwith={len(a)} matrix={len(b)}")


print("\n" + ("ALL PASSED" if not FAILURES else f"FAILURES: {FAILURES}"))
sys.exit(1 if FAILURES else 0)
