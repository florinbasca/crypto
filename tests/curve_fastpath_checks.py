"""
Equivalence checks for the ARRAY-NATIVE fast paths in the discovery engine.

Two hot functions were replaced with faster implementations that are claimed to
be equivalent, not merely similar. This file holds them to that claim, because
both sit underneath the reward, the survivor cut and the promotion verdict - a
small drift here silently changes which formulas get traded.

  response_curve()  ->  response_curve_matrix()
      The original pivots per candidate and accumulates cumsum(blk @ w) per
      entry. The new one prefixes a cumulative-residual matrix ONCE per roll
      and takes differences: sum_{j=1..k} r_{t+j} = C[t+k+1] - C[t+1], dotted
      with the weights. Equal by linearity, but NOT bit-identical: differencing
      a long cumulative sum is a different rounding path, and it is exactly the
      kind of change that can cancel catastrophically if the series is long
      enough. The tolerance below is asserted against the curve's own scale.

  compile_candidate()  ->  compile_candidate_values()
      Refactored so the normalization runs on a Series aligned to the panel
      instead of a fresh long frame. Must be bit-identical - it is the same
      arithmetic, only fewer copies.

Covers the cases that separate "equivalent" from "usually equivalent": NaN
residuals (delistings mid-path), NaN signal values, the min_assets filter, and
a long window where cumulative-sum cancellation would show up.

Run: uv run tests/curve_fastpath_checks.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from research.signals import search as search_mod
from research.signals import generation as gen

FAILURES = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        FAILURES.append(name)


H, STRIDE, MIN_ASSETS = 144, 6, 5
SAMPLE_KS = [1, 6, 36, 144]


def make_case(n_bars, n_sym, seed, nan_res=0.0, nan_sig=0.0, thin=False):
    """A (signal, residual) grid on one shared schedule, as the caller
    guarantees. Optionally punches NaNs into residuals (delisting), NaNs into
    the signal, or thins the cross-section below min_assets on some bars."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range('2024-01-01', periods=n_bars, freq='10min')
    syms = [f'S{i}' for i in range(n_sym)]
    sig = rng.standard_normal((n_bars, n_sym))
    res = rng.standard_normal((n_bars, n_sym)) * 1e-3
    if nan_res:
        res[rng.random(res.shape) < nan_res] = np.nan
    if nan_sig:
        sig[rng.random(sig.shape) < nan_sig] = np.nan
    if thin:
        # Drop most names on every 5th bar so min_assets excludes those entries.
        for row in range(0, n_bars, 5):
            sig[row, MIN_ASSETS - 1:] = np.nan
    sig_df = pd.DataFrame(sig, index=ts, columns=syms).stack(
        future_stack=True).rename('signal').reset_index()
    sig_df.columns = ['timestamp', 'symbol', 'signal']
    res_wide = pd.DataFrame(res, index=ts, columns=syms)
    return sig_df, res_wide, sig, res, ts


CASES = [
    ('plain',                dict(n_bars=900, n_sym=30, seed=1)),
    ('NaN residuals 5%',     dict(n_bars=900, n_sym=30, seed=2, nan_res=0.05)),
    ('NaN signal 10%',       dict(n_bars=900, n_sym=30, seed=3, nan_sig=0.10)),
    ('thin cross-section',   dict(n_bars=900, n_sym=30, seed=4, thin=True)),
    ('long window (6000b)',  dict(n_bars=6000, n_sym=40, seed=5)),
    ('long + NaNs',          dict(n_bars=6000, n_sym=40, seed=6,
                                  nan_res=0.03, nan_sig=0.05)),
]

print("response_curve  vs  response_curve_matrix\n")
worst_rel = 0.0
for label, kw in CASES:
    sig_df, res_wide, sig_arr, res_arr, ts = make_case(**kw)
    a = search_mod.response_curve(sig_df, res_wide, H, STRIDE, MIN_ASSETS,
                                  sample_ks=SAMPLE_KS)
    b = search_mod.response_curve_matrix(sig_arr, res_arr, ts, H, STRIDE,
                                         MIN_ASSETS, sample_ks=SAMPLE_KS)
    if a is None or b is None:
        check(f"{label}: both produce a curve", a is None and b is None,
              "(one returned None)")
        continue
    same_entries = a['entries'] == b['entries']
    same_days = a['entry_days'] == b['entry_days']
    same_neff = np.isclose(a['n_eff'], b['n_eff'])
    scale = float(np.max(np.abs(a['A']))) or 1.0
    rel = float(np.max(np.abs(a['A'] - b['A']))) / scale
    worst_rel = max(worst_rel, rel)
    per_entry_ok = all(
        np.allclose(a['per_entry_at'][k], b['per_entry_at'][k],
                    rtol=0, atol=1e-9 * scale)
        for k in a['per_entry_at'])
    check(f"{label}: same entries / days / n_eff",
          same_entries and same_days and same_neff,
          f"(entries {a['entries']} vs {b['entries']}, "
          f"days {a['entry_days']} vs {b['entry_days']})")
    check(f"{label}: curve matches to float noise", rel < 1e-9,
          f"(max rel diff {rel:.2e} on {a['entries']} entries)")
    check(f"{label}: per-entry outcomes match", per_entry_ok)

print(f"\nworst relative curve difference across all cases: {worst_rel:.2e}\n")

# --- the cumulative-difference trick must not cancel on a long series -------
sig_df, res_wide, sig_arr, res_arr, ts = make_case(n_bars=20000, n_sym=40,
                                                   seed=11)
a = search_mod.response_curve(sig_df, res_wide, H, STRIDE, MIN_ASSETS,
                              sample_ks=SAMPLE_KS)
b = search_mod.response_curve_matrix(sig_arr, res_arr, ts, H, STRIDE,
                                     MIN_ASSETS, sample_ks=SAMPLE_KS)
scale = float(np.max(np.abs(a['A'])))
rel = float(np.max(np.abs(a['A'] - b['A']))) / scale
cum_mag = float(np.max(np.abs(np.cumsum(np.nan_to_num(res_arr), axis=0))))
check("20k-bar window: no catastrophic cancellation", rel < 1e-9,
      f"(rel {rel:.2e}; cumulative reaches {cum_mag:.3f} vs curve scale "
      f"{scale:.2e}, so ~{np.log10(cum_mag / scale):.0f} digits are spent)")

# --- precomputed cumulative must equal computing it inline ------------------
cum = np.vstack([np.zeros((1, res_arr.shape[1])),
                 np.cumsum(np.nan_to_num(res_arr), axis=0)])
c = search_mod.response_curve_matrix(sig_arr, res_arr, ts, H, STRIDE,
                                     MIN_ASSETS, sample_ks=SAMPLE_KS,
                                     residual_cumulative=cum)
check("passing residual_cumulative changes nothing",
      np.array_equal(b['A'], c['A']))

# --- compile_candidate must be bit-identical to its values fast path --------
print()
rng = np.random.default_rng(0)
n_bars, n_sym = 400, 25
ts = pd.date_range('2024-03-01', periods=n_bars, freq='10min')
panel = pd.DataFrame({
    'timestamp': np.repeat(ts, n_sym),
    'symbol': np.tile([f'S{i}' for i in range(n_sym)], n_bars),
    'res_zscore': rng.standard_normal(n_bars * n_sym),
    'vol_ratio': rng.standard_normal(n_bars * n_sym),
}).sort_values(['symbol', 'timestamp']).reset_index(drop=True)
panel.loc[panel.sample(frac=0.05, random_state=1).index, 'res_zscore'] = np.nan

# compile_candidate is compile_candidate_values + dropna + reset_index (the
# dropna predates this refactor). So the values must be identical ON THE ROWS
# THE LONG FRAME KEEPS, and the rows it drops must be exactly the NaN ones -
# a NaN row and a dropped row both become NaN once pivoted, which is why the
# array path can carry NaN in place instead.
for name, expr in [('plain column', ('col', 'res_zscore')),
                   ('cs_zscore', ('cs_zscore', ('col', 'res_zscore'))),
                   ('product', ('mul', ('cs_zscore', ('col', 'res_zscore')),
                                ('cs_zscore', ('col', 'vol_ratio'))))]:
    cand = gen.Candidate(f'probe_{name}', 'residual_shape', expr)
    long_frame = gen.compile_candidate(cand, panel)
    values = gen.compile_candidate_values(cand, panel)
    kept = values.dropna()
    check(f"compile_candidate == compile_candidate_values ({name})",
          len(long_frame) == len(kept)
          and np.array_equal(long_frame['signal'].to_numpy(),
                             kept.to_numpy()),
          f"({len(long_frame)} kept of {len(values)})")
    check(f"dropped rows are exactly the NaN ones ({name})",
          len(values) - len(long_frame) == int(values.isna().sum()))
    check(f"the case is not vacuous - NaNs present ({name})",
          int(values.isna().sum()) > 0,
          f"({int(values.isna().sum())} NaN)")

print()
if FAILURES:
    print(f"FAILED {len(FAILURES)}: {FAILURES}")
    sys.exit(1)
print("ALL CURVE FAST-PATH CHECKS PASSED")
