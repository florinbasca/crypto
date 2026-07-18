"""
Unit checks for the sn_ (per-symbol seasonality) and ll_ (leader lead-lag)
feature primitives in risk_model/features.py. No database required.

Covers: causality (truncation test on the two calculators), correctness on
constructed patterns (a deterministic hour-of-day residual profile must be
recovered; the day-of-week feature must exclude the running day; a lagged
leader-follower must show high ll_lag_corr), degenerate inputs (leader symbol
itself, missing leader), and the wiring (discovery families resolve the new
prefixes; the new spaces reference real columns).

Run: uv run tests/new_feature_checks.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from config import get, BARS_PER_DAY
from risk_model.features import (FEATURE_CONFIG, SN_FEATURE_NAMES,
                                 calculate_leadlag_features,
                                 calculate_residual_features,
                                 calculate_seasonality_features,
                                 calculate_time_features,
                                 leadlag_feature_names)

FAILURES = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        FAILURES.append(name)


LL_NAMES = leadlag_feature_names(FEATURE_CONFIG)
N_DAYS = 30
N = N_DAYS * BARS_PER_DAY
rng = np.random.default_rng(3)
TS = pd.date_range('2024-01-01', periods=N, freq='10min')


def make_df(close):
    return pd.DataFrame({'timestamp': TS, 'symbol': 'SYN', 'close': close})


# --- causality: truncation test on both calculators ---------------------------
close = 100 * np.exp(np.cumsum(rng.normal(0, 1e-3, N)))
res = pd.Series(rng.normal(0, 8e-4, N))
leader = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 1e-3, N))), index=TS)
df = make_df(close)

cut = N - 3 * BARS_PER_DAY - 71   # mid-day cut, not on a day boundary
sn_full = calculate_seasonality_features(df, res, FEATURE_CONFIG)
sn_trunc = calculate_seasonality_features(df.iloc[:cut], res.iloc[:cut],
                                          FEATURE_CONFIG)
ll_full = calculate_leadlag_features(df, 'SYN', leader, FEATURE_CONFIG)
ll_trunc = calculate_leadlag_features(df.iloc[:cut], 'SYN',
                                      leader.iloc[:cut], FEATURE_CONFIG)

for names, full, trunc, label in [(SN_FEATURE_NAMES, sn_full, sn_trunc, 'sn_'),
                                  (LL_NAMES, ll_full, ll_trunc, 'll_')]:
    leaking = []
    for c in names:
        a, b = float(full[c].iloc[cut - 1]), float(trunc[c].iloc[cut - 1])
        if not (np.isclose(a, b, atol=1e-10, rtol=1e-8)
                or (np.isnan(a) and np.isnan(b))):
            leaking.append(c)
    check(f"{label} truncation test (no look-ahead)", not leaking,
          f"leaking: {leaking}")

# --- sn_tod_res recovers a deterministic hour-of-day profile ------------------
hour = TS.hour.values
mu = (hour - 11.5) * 1e-4                       # linear hour profile
res_pattern = pd.Series(mu)
sn = calculate_seasonality_features(make_df(close), res_pattern, FEATURE_CONFIG)
tail = slice(-BARS_PER_DAY, None)               # last day: warmed up
got = sn['sn_tod_res'].iloc[tail].values
want = mu[tail]
check("sn_tod_res recovers the hour-of-day profile",
      np.allclose(got, want, atol=1e-12),
      f"(max err {np.abs(got - want).max():.2e})")

# --- sn_tod_vol_ratio: a high-vol hour scores > 1, quiet hours < 1 ------------
scale = np.where(hour == 0, 4e-3, 4e-4)
res_vol = pd.Series(rng.normal(0, 1, N) * scale)
sn = calculate_seasonality_features(make_df(close), res_vol, FEATURE_CONFIG)
ratio = sn['sn_tod_vol_ratio'].iloc[tail]
hr_tail = hour[tail]
check("sn_tod_vol_ratio flags the high-vol hour",
      float(ratio[hr_tail == 0].mean()) > 2.0
      and float(ratio[hr_tail != 0].mean()) < 1.0,
      f"(hour0 {ratio[hr_tail == 0].mean():.2f}, "
      f"others {ratio[hr_tail != 0].mean():.2f})")

# --- sn_dow_res excludes the running (partial) day -----------------------------
c0 = 5e-5
res_flat = pd.Series(np.full(N, c0))
res_shock = res_flat.copy()
res_shock.iloc[-BARS_PER_DAY:] = 1.0            # huge shock on the final day
sn_flat = calculate_seasonality_features(make_df(close), res_flat, FEATURE_CONFIG)
sn_shock = calculate_seasonality_features(make_df(close), res_shock, FEATURE_CONFIG)
last_day = slice(-BARS_PER_DAY, None)
check("sn_dow_res = trailing same-weekday full-day sum",
      np.allclose(sn_flat['sn_dow_res'].iloc[last_day].dropna(),
                  c0 * BARS_PER_DAY, atol=1e-12))
check("sn_dow_res ignores the running day's own residuals",
      np.allclose(sn_shock['sn_dow_res'].iloc[last_day].values,
                  sn_flat['sn_dow_res'].iloc[last_day].values,
                  atol=1e-12, equal_nan=True))

# --- ll_: leader-follower behaviour --------------------------------------------
lag = int(FEATURE_CONFIG['lead_lag']['lag_bars'])
lead_ret = rng.normal(0, 2e-3, N)
leader_px = pd.Series(100 * np.exp(np.cumsum(lead_ret)), index=TS)
# Follower: responds to the leader's PREVIOUS lag-bar move + small noise
lagged_move = pd.Series(lead_ret).rolling(lag).sum().shift(1).fillna(0.0).values
own_ret = 0.5 * lagged_move / lag + rng.normal(0, 2e-4, N)
own_px = 100 * np.exp(np.cumsum(own_ret))
ll = calculate_leadlag_features(make_df(own_px), 'SYN', leader_px, FEATURE_CONFIG)
check("ll_lag_corr is strongly positive for a lagged follower",
      float(ll['ll_lag_corr'].iloc[-1]) > 0.3,
      f"(corr {ll['ll_lag_corr'].iloc[-1]:.2f})")

# Independent asset: no lagged relationship
indep_px = 100 * np.exp(np.cumsum(rng.normal(0, 2e-3, N)))
ll_ind = calculate_leadlag_features(make_df(indep_px), 'SYN', leader_px,
                                    FEATURE_CONFIG)
check("ll_lag_corr ~ 0 for an independent asset",
      abs(float(ll_ind['ll_lag_corr'].iloc[-1])) < 0.1,
      f"(corr {ll_ind['ll_lag_corr'].iloc[-1]:.2f})")

# Gap identity: ll_leader_gap_wb == beta * leader w-bar move - own w-bar move
w = int(FEATURE_CONFIG['lead_lag']['gap_windows_bars'][0])
own_s = pd.Series(own_px).pct_change()
lead_s = leader_px.reset_index(drop=True).pct_change()
expected = (ll['ll_leader_beta']
            * lead_s.rolling(w, min_periods=w).sum()
            - own_s.rolling(w, min_periods=w).sum())
got = ll[f'll_leader_gap_{w}b']
ok_mask = expected.notna()
check("ll_leader_gap matches its formula",
      np.allclose(got[ok_mask], expected[ok_mask], atol=1e-12))

# Degenerate inputs -> all NaN
leader_sym = FEATURE_CONFIG['lead_lag']['leader_symbol']
ll_self = calculate_leadlag_features(make_df(own_px), leader_sym, leader_px,
                                     FEATURE_CONFIG)
ll_none = calculate_leadlag_features(make_df(own_px), 'SYN', None,
                                     FEATURE_CONFIG)
check("leader symbol itself -> all NaN",
      all(ll_self[c].isna().all() for c in LL_NAMES))
check("missing leader series -> all NaN",
      all(ll_none[c].isna().all() for c in LL_NAMES))

# --- cyclical features -------------------------------------------------------
# sn_fph_res recovers a planted funding-cycle profile (same trailing
# same-bucket construction as sn_tod_res, bucketed by cycle phase).
f_hours = FEATURE_CONFIG['funding_hours_utc']
cycle_h = 24 // len(f_hours)
phase = (hour - f_hours[0]) % cycle_h
mu_f = (phase - (cycle_h - 1) / 2) * 1e-4
sn_f = calculate_seasonality_features(make_df(close), pd.Series(mu_f),
                                      FEATURE_CONFIG)
got_f = sn_f['sn_fph_res'].iloc[tail].values
check("sn_fph_res recovers the funding-cycle profile",
      np.allclose(got_f, mu_f[tail], atol=1e-12),
      f"(max err {np.abs(got_f - mu_f[tail]).max():.2e})")

# tm_ funding phase: deterministic from the schedule - dist counts down to
# the next settlement and the sin/cos repeat exactly every cycle.
tm = calculate_time_features(make_df(close), FEATURE_CONFIG)
bars_per_cycle = BARS_PER_DAY // len(f_hours)
d = tm['tm_funding_dist'].values
at_settle = phase == 0
check("tm_funding_dist: full cycle at settlement, counts down to 1",
      np.allclose(np.asarray(d)[at_settle & (TS.minute == 0)],
                  bars_per_cycle)
      and float(np.min(d)) >= 1.0,
      f"(range {np.min(d):.0f}-{np.max(d):.0f} bars)")
check("tm_funding_sin/cos: exactly periodic at the cycle length",
      np.allclose(tm['tm_funding_sin'].values[bars_per_cycle:],
                  tm['tm_funding_sin'].values[:-bars_per_cycle])
      and np.allclose(tm['tm_funding_cos'].values[bars_per_cycle:],
                      tm['tm_funding_cos'].values[:-bars_per_cycle]))

# res_ cycle-scale autocorr: a 24h-periodic residual shows positive AC at
# the period (lag 144) and negative at the half-period (lag 72).
wave = pd.Series(np.sin(2 * np.pi * np.arange(N) / BARS_PER_DAY) * 1e-3
                 + rng.normal(0, 2e-4, N))
res_f = calculate_residual_features(make_df(close), wave, FEATURE_CONFIG)
ac144 = float(res_f['res_autocorr_lag144'].iloc[-1])
ac72 = float(res_f['res_autocorr_lag72'].iloc[-1])
check("res_autocorr: daily cycle -> positive at 24h lag, negative at 12h",
      ac144 > 0.5 and ac72 < -0.5,
      f"(lag144 {ac144:+.2f}, lag72 {ac72:+.2f})")

# --- wiring: discovery families resolve the new prefixes -----------------------
from research.signals.data import resolve_family_columns

available = SN_FEATURE_NAMES + LL_NAMES + ['res_zscore', 'timestamp', 'symbol']
fams = resolve_family_columns(available, get('discovery'))
check("discovery family 'seasonality' resolves sn_ columns",
      set(fams.get('seasonality', [])) == set(SN_FEATURE_NAMES))
check("discovery family 'lead_lag' resolves ll_ columns",
      set(fams.get('lead_lag', [])) == set(LL_NAMES))

# ---------------------------------------------------------------------------
print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURES: {FAILURES}")
    sys.exit(1)
print("ALL NEW-FEATURE CHECKS PASSED")
