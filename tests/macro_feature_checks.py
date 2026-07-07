"""
Unit checks for the macro/event data layer: etl/macro.py parsers and the
ev_/mx_/mb_ feature calculators in risk_model/features.py. No database or
network required.

Covers: FRED/DeFiLlama response parsing, event-timing math (hours to/since a
known event, day flags, event window), macro-state values (rate change,
event shock stamped on the right day), per-name macro-beta recovery on a
planted sensitivity, the event-volume profile (volume doubled on event days
-> ratio ~2), post-event drift availability (a value may change only once
the event's response window has fully elapsed), and truncation causality
for all three groups (events pass in full - schedules are known ahead;
macro data is truncated).

Run: uv run tests/macro_feature_checks.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from config import BARS_PER_DAY
from etl.macro import parse_fred_observations, parse_llama_stables
from risk_model.features import (EV_FEATURE_NAMES, MB_FEATURE_NAMES,
                                 MX_FEATURE_NAMES, FEATURE_CONFIG,
                                 calculate_event_features,
                                 calculate_macro_state_features,
                                 calculate_macrobeta_features)

FAILURES = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        FAILURES.append(name)


# --- ETL parsers ---------------------------------------------------------------
obs = parse_fred_observations({'observations': [
    {'date': '2024-01-02', 'value': '4.32'},
    {'date': '2024-01-03', 'value': '.'},          # FRED missing marker
    {'date': '2024-01-04', 'value': '4.40'},
]})
check("fred parser: values parsed, '.' dropped",
      len(obs) == 2 and obs[pd.Timestamp('2024-01-04')] == 4.40)

st = parse_llama_stables([
    {'date': '1704153600', 'totalCirculatingUSD': {'peggedUSD': 1.3e11,
                                                   'peggedEUR': 5e8}},
    {'date': '1704240000', 'totalCirculatingUSD': {'peggedUSD': 1.31e11}},
])
check("llama parser: circulating summed per day",
      len(st) == 2 and abs(st.iloc[0] - 1.305e11) < 1e6)

# --- synthetic panel + macro world ---------------------------------------------
rng = np.random.default_rng(9)
N_DAYS = 80
N = N_DAYS * BARS_PER_DAY
TS = pd.date_range('2024-01-01', periods=N, freq='10min')
DF = pd.DataFrame({'timestamp': TS, 'symbol': 'SYN',
                   'close': 100.0,
                   'volume': np.full(N, 10.0),
                   'quote_asset_volume': np.full(N, 1e5)})

# Events: one every 7 days at 19:00 UTC (FOMC-style), alternating types
ev_times = pd.date_range('2024-01-04 19:00', periods=10, freq='7D')
EVENTS = pd.DataFrame({
    'event_time_utc': ev_times,
    'event_type': ['fomc', 'cpi'] * 5,
})
EV_DATES = set(ev_times.normalize())

# Macro daily: rates ramp with a known step; other series present
cal = pd.date_range('2023-12-01', periods=N_DAYS + 40, freq='D')
rates = pd.Series(4.0, index=cal)
rates.loc[pd.Timestamp('2024-02-01'):] = 4.5      # one 50bp step on Feb 1
MACRO = pd.DataFrame({
    'date': cal,
    'rates2y': rates.values,
    'rates10y': rates.values + 0.2,
    'vix': 15 + rng.normal(0, 1, len(cal)).cumsum() * 0.05,
    'dollar': 120 + rng.normal(0, 0.2, len(cal)).cumsum(),
    'stables_mcap': 1.3e11 * (1 + 0.001 * np.arange(len(cal))),
    'fed_bs': np.full(len(cal), 7.5e6),            # $mn
    'rrp': np.full(len(cal), 500.0),               # $bn
    'breakeven10': np.full(len(cal), 2.3),
    'fear_greed': np.full(len(cal), 60.0),
})
DOM = pd.Series(0.55, index=cal)

# --- ev_: timing math -----------------------------------------------------------
ev = calculate_event_features(DF, EVENTS, FEATURE_CONFIG)
i0 = TS.get_indexer([pd.Timestamp('2024-01-04 09:00')])[0]
check("ev_hours_to_event: 10h before a 19:00 event",
      abs(ev['ev_hours_to_event'].iloc[i0] - 10.0) < 1e-9,
      f"({ev['ev_hours_to_event'].iloc[i0]:.2f}h)")
i1 = TS.get_indexer([pd.Timestamp('2024-01-05 07:00')])[0]
check("ev_hours_since_event: 12h after",
      abs(ev['ev_hours_since_event'].iloc[i1] - 12.0) < 1e-9)
day_flag = ev['ev_fomc_day'][TS.normalize() == pd.Timestamp('2024-01-04')]
check("ev_fomc_day flags the whole event UTC day", (day_flag == 1.0).all())
check("ev flags are 0 off events",
      float(ev['ev_fomc_day'][TS.normalize() ==
                              pd.Timestamp('2024-01-06')].max()) == 0.0)
iw = TS.get_indexer([pd.Timestamp('2024-01-04 19:30')])[0]
check("ev_event_window on right after the event",
      ev['ev_event_window'].iloc[iw] == 1.0)
check("hours clip respected",
      float(ev['ev_hours_to_event'].max()) <=
      FEATURE_CONFIG['macro']['hours_clip'] + 1e-9)

# --- mx_: values + event shock ---------------------------------------------------
mx = calculate_macro_state_features(DF, MACRO, DOM, EVENTS, FEATURE_CONFIG)
istep = TS.get_indexer([pd.Timestamp('2024-02-01 12:00')])[0]
check("mx_rates2y_chg_1d shows the 50bp step on its date",
      abs(mx['mx_rates2y_chg_1d'].iloc[istep] - 0.5) < 1e-9)
check("mx_curve_2s10s = 10y - 2y",
      abs(mx['mx_curve_2s10s'].iloc[istep] - 0.2) < 1e-9)
# event shock: |2y 1d change| stamped on event_date + 1, zero elsewhere
shock = pd.Series(mx['mx_event_shock'].values, index=TS.normalize())
on_next = sorted({d + pd.Timedelta(days=1) for d in EV_DATES})
check("mx_event_shock only on event_date + 1",
      float(shock[~shock.index.isin(on_next)].abs().max()) == 0.0)
check("mx_fear_greed level mapped", float(mx['mx_fear_greed'].iloc[istep]) == 60.0)

# --- mb_: planted beta recovery ---------------------------------------------------
B = 3.0
x_d = pd.Series(rng.normal(0, 0.02, len(cal)), index=cal)   # daily rate impulse
macro_beta = MACRO.copy()
macro_beta['rates2y'] = (4.0 + x_d.cumsum()).values
res_bars = np.zeros(N)
day_idx = TS.normalize()
x_by_day = x_d.reindex(day_idx).values
res_bars = B * x_by_day / BARS_PER_DAY + rng.normal(0, 1e-5, N)
RES = pd.Series(res_bars, index=DF.index)

mb = calculate_macrobeta_features(DF, RES, macro_beta, DOM, EVENTS,
                                  FEATURE_CONFIG)
beta_est = float(mb['mb_beta_rates'].iloc[-1])
check("mb_beta_rates recovers a planted sensitivity",
      abs(beta_est - B) < 0.15 * B, f"(est {beta_est:.2f}, true {B})")

# --- mb_: event volume profile (volume doubled on event days -> ratio ~2) --------
df_vol = DF.copy()
on_event_day = np.asarray(day_idx.isin(EV_DATES))
df_vol['quote_asset_volume'] = np.where(on_event_day, 2e5, 1e5)
mb_v = calculate_macrobeta_features(df_vol, RES, MACRO, DOM, EVENTS,
                                    FEATURE_CONFIG)
ratio = float(mb_v['mb_event_volume_ratio'].iloc[-1])
check("mb_event_volume_ratio ~ 2 when event-day volume doubles",
      1.6 < ratio < 2.4, f"(ratio {ratio:.2f})")

# --- mb_event_drift: availability + sign -----------------------------------------
# Plant: +1% residual over the 24h after every event, ~0 elsewhere
res_drift = np.full(N, 0.0)
for e in ev_times:
    lo = np.searchsorted(TS.values, e.to_datetime64(), side='right')
    hi = np.searchsorted(TS.values, (e + pd.Timedelta(hours=24)).to_datetime64(),
                         side='right')
    res_drift[lo:hi] += 0.01 / max(hi - lo, 1)
mb_d = calculate_macrobeta_features(DF, pd.Series(res_drift, index=DF.index),
                                    MACRO, DOM, EVENTS, FEATURE_CONFIG)
drift = mb_d['mb_event_drift']
check("mb_event_drift converges to the planted +1% post-event return",
      abs(float(drift.iloc[-1]) - 0.01) < 1e-3,
      f"(est {float(drift.iloc[-1]):.4f})")
# availability: NaN until event_min_events events have fully elapsed
first_ok = drift.first_valid_index()
min_e = FEATURE_CONFIG['macro']['event_min_events']
avail_time = ev_times[min_e - 1] + pd.Timedelta(hours=24)
check("mb_event_drift only available once enough events fully elapsed",
      first_ok is not None and TS[first_ok] >= avail_time,
      f"(first {TS[first_ok]} vs {avail_time})")

# --- truncation causality for all three groups -----------------------------------
cut = N - 20 * BARS_PER_DAY - 55
cut_date = TS[cut - 1].normalize()
macro_tr = MACRO[MACRO['date'] <= cut_date]
dom_tr = DOM[DOM.index <= cut_date]
full_sets = [
    (EV_FEATURE_NAMES, calculate_event_features(DF, EVENTS, FEATURE_CONFIG),
     calculate_event_features(DF.iloc[:cut], EVENTS, FEATURE_CONFIG), 'ev_'),
    (MX_FEATURE_NAMES,
     calculate_macro_state_features(DF, MACRO, DOM, EVENTS, FEATURE_CONFIG),
     calculate_macro_state_features(DF.iloc[:cut], macro_tr, dom_tr, EVENTS,
                                    FEATURE_CONFIG), 'mx_'),
    (MB_FEATURE_NAMES,
     calculate_macrobeta_features(DF, RES, MACRO, DOM, EVENTS, FEATURE_CONFIG),
     calculate_macrobeta_features(DF.iloc[:cut], RES.iloc[:cut], macro_tr,
                                  dom_tr, EVENTS, FEATURE_CONFIG), 'mb_'),
]
for names, full, trunc, label in full_sets:
    leaking = []
    for c in names:
        a, b = float(full[c].iloc[cut - 1]), float(trunc[c].iloc[cut - 1])
        if not (np.isclose(a, b, atol=1e-10, rtol=1e-8)
                or (np.isnan(a) and np.isnan(b))):
            leaking.append(c)
    check(f"{label} truncation test (no look-ahead)", not leaking,
          f"leaking: {leaking}")

# --- wiring: discovery families resolve the new prefixes --------------------------
from config import get
from research.signals.agent.data import resolve_family_columns

fams = resolve_family_columns(EV_FEATURE_NAMES + MX_FEATURE_NAMES
                              + MB_FEATURE_NAMES + ['res_zscore'],
                              get('discovery'))
check("families events/macro/macro_beta resolve their columns",
      set(fams['events']) == set(EV_FEATURE_NAMES)
      and set(fams['macro']) == set(MX_FEATURE_NAMES)
      and set(fams['macro_beta']) == set(MB_FEATURE_NAMES))

# ---------------------------------------------------------------------------
print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURES: {FAILURES}")
    sys.exit(1)
print("ALL MACRO-FEATURE CHECKS PASSED")
