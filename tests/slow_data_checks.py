"""
Checks for the slow/fundamental data pipeline (stablecoin_supply,
token_unlocks_daily, dev_activity, listings) and their feature builders.

Two layers:
  1. Synthetic feature-builder tests (no database) - correctness of the
     math and the causal-lag conventions.
  2. Table schema/date-sanity checks - SKIPPED per table if the table
     does not exist yet (ETLs may not have run on a fresh clone).

Run: uv run tests/slow_data_checks.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from unittest import mock

import numpy as np
import pandas as pd

FAILURES = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name} {detail}")
    if not cond:
        FAILURES.append(name)


def _panel(start, periods, freq='10min'):
    ts = pd.date_range(start, periods=periods, freq=freq)
    return pd.DataFrame({'timestamp': ts, 'close': 100.0 + np.arange(periods)})


# ---------------------------------------------------------------------------
# 1. Stablecoin supply features (mx_stable_) - synthetic
# ---------------------------------------------------------------------------
import risk_model.features as rf

days = pd.date_range('2023-11-01', periods=120, freq='D')
supply = pd.concat([
    pd.DataFrame({'date': days, 'asset': 'USDT', 'chain': 'all',
                  # 0.1%/day compounding growth
                  'supply_usd': 1e11 * 1.001 ** np.arange(120)}),
    pd.DataFrame({'date': days, 'asset': 'ALL', 'chain': 'Ethereum',
                  'supply_usd': 5e10 * 1.002 ** np.arange(120)}),
    pd.DataFrame({'date': days, 'asset': 'ALL', 'chain': 'Solana',
                  'supply_usd': 1e10}),
    pd.DataFrame({'date': days, 'asset': 'ALL', 'chain': 'Tron',
                  'supply_usd': 3e10}),
], ignore_index=True)

df = _panel('2024-02-01', 3 * 144)
rf._STABLE_CACHE = None
with mock.patch.object(rf, 'load_data', return_value=supply):
    mx = rf.calculate_stablecoin_features(df)
rf._STABLE_CACHE = None

check("mx_stable: 30d total growth ~ 3.04%",
      np.isclose(mx['mx_stable_total_chg_30d'].dropna().iloc[-1],
                 1.001 ** 30 - 1, rtol=1e-6))
check("mx_stable: flat Solana -> 7d change 0",
      np.isclose(mx['mx_stable_sol_chg_7d'].dropna().iloc[-1], 0.0))
check("mx_stable: constant across the day (daily source)",
      mx['mx_stable_total_chg_30d'].iloc[:144].nunique() <= 2)

# ---------------------------------------------------------------------------
# 2. Unlock features (un_) - synthetic
# ---------------------------------------------------------------------------
# Schedule: cliffs at 2024-01-10 (5%) and 2024-02-15 (10%); dust below the
# 0.1% cliff threshold in between (must be ignored by cliff features).
ul_days = pd.date_range('2024-01-01', periods=60, freq='D')
daily_pct = np.full(60, 0.0001)
daily_pct[9] = 0.05    # 2024-01-10
daily_pct[45] = 0.10   # 2024-02-15
unlocks = pd.DataFrame({'date': ul_days, 'symbol': 'TEST',
                        'unlocked_amt': daily_pct * 1e9,
                        'daily_pct': daily_pct,
                        'cum_pct': np.cumsum(daily_pct)})

df = _panel('2024-01-20', 2 * 144)  # sits between the two cliffs
with mock.patch.object(rf, 'load_data', return_value=unlocks):
    un = rf.calculate_unlock_features(df, 'TEST')

check("un: days_to_next counts to the NEXT cliff (Feb 15)",
      np.isclose(un['un_days_to_next'].iloc[0], 26.0))
check("un: next_pct is the coming cliff's size",
      np.isclose(un['un_next_pct'].iloc[0], 0.10))
check("un: days_since_last counts from Jan 10",
      np.isclose(un['un_days_since_last'].iloc[0], 10.0))
check("un: sub-threshold dust is not a cliff",
      np.isclose(un['un_next_pct'].iloc[-1], 0.10))
check("un: trailing 30d pct sums cliff + dust",
      np.isclose(un['un_trailing_30d_pct'].dropna().iloc[0],
                 0.05 + 0.0001 * 20, atol=5e-4))

# On a cliff day itself: days_to_next == 0 (searchsorted side='left').
df_cliff = _panel('2024-02-15', 144)
with mock.patch.object(rf, 'load_data', return_value=unlocks):
    un2 = rf.calculate_unlock_features(df_cliff, 'TEST')
check("un: cliff day -> days_to_next == 0",
      np.isclose(un2['un_days_to_next'].iloc[0], 0.0))

# No schedule -> all-NaN (meaning: no known vesting), not zeros.
with mock.patch.object(rf, 'load_data', return_value=pd.DataFrame()):
    un3 = rf.calculate_unlock_features(df, 'TEST')
check("un: no schedule -> all-NaN",
      all(un3[n].isna().all() for n in rf.UN_FEATURE_NAMES))

# ---------------------------------------------------------------------------
# 3. Dev-activity features (dv_) - synthetic
# ---------------------------------------------------------------------------
# 200 daily rows: devs ramp 10 -> 30; constant shares.
dv_days = pd.date_range('2023-10-01', periods=200, freq='D')
devs = np.round(np.linspace(10, 30, 200))
dev_tbl = pd.DataFrame({'date': dv_days, 'symbol': 'TEST',
                        'all_devs': devs,
                        'exclusive_devs': devs // 2,
                        'full_time_devs': devs // 5,
                        'one_time_devs': 1, 'devs_2y_plus': 2,
                        'num_commits': devs * 10})

df = _panel('2024-04-01', 2 * 144)
rf._DEV_CACHE = None
with mock.patch.object(rf, 'load_data', return_value=dev_tbl):
    dv = rf.calculate_dev_activity_features(df, 'TEST')
rf._DEV_CACHE = None

last_day = dv_days.max()
check("dv: 30d lag (bar reads the value from 30 days before)",
      np.isclose(dv['dv_active_devs'].iloc[0],
                 np.log1p(devs[(pd.Timestamp('2024-04-01') -
                                pd.Timedelta(days=30) - dv_days[0]).days])))
check("dv: full-time share ~ 1/5",
      abs(dv['dv_full_time_share'].dropna().iloc[0] - 0.2) < 0.05)
check("dv: 3m change positive on a ramp",
      dv['dv_devs_chg_3m'].dropna().iloc[0] > 0)

# Unmapped symbol -> all NaN.
rf._DEV_CACHE = None
with mock.patch.object(rf, 'load_data', return_value=dev_tbl):
    dv2 = rf.calculate_dev_activity_features(df, 'MEMECOIN')
rf._DEV_CACHE = None
check("dv: unmapped symbol -> all-NaN",
      all(dv2[n].isna().all() for n in rf.DV_FEATURE_NAMES))

# ---------------------------------------------------------------------------
# 4. Listing-age features (ls_) - synthetic
# ---------------------------------------------------------------------------
listings = pd.DataFrame({'symbol': ['TEST', 'OLD'],
                         'first_trade': [pd.Timestamp('2024-01-15'),
                                         pd.Timestamp('2020-01-01')],
                         'venue': ['binance', 'binance']})

df = _panel('2024-02-01', 144)
rf._LISTINGS_CACHE = None
with mock.patch.object(rf, 'load_data', return_value=listings):
    ls_new = rf.calculate_listing_features(df, 'TEST')
    ls_old = rf.calculate_listing_features(df, 'OLD')
rf._LISTINGS_CACHE = None

check("ls: 17 days since a Jan-15 listing on Feb 1",
      np.isclose(ls_new['ls_days_since_listing'].iloc[0], 17.0))
check("ls: newly_listed=1 inside 30d", ls_new['ls_newly_listed'].iloc[0] == 1.0)
check("ls: newly_listed=0 for a 2020 listing", ls_old['ls_newly_listed'].iloc[0] == 0.0)
check("ls: age not clipped by the backtest window",
      ls_old['ls_days_since_listing'].iloc[0] > 1400)

# ---------------------------------------------------------------------------
# 5. Config / description registration
# ---------------------------------------------------------------------------
from config import config as cfg
from research.signals.data import describe_column

fams = cfg['discovery']['families']
check("config: unlocks family", fams.get('unlocks') == ['un_'])
check("config: dev_activity family", fams.get('dev_activity') == ['dv_'])
check("config: listing family", fams.get('listing') == ['ls_'])
for col in ['mx_stable_total_chg_30d',
            'un_days_to_next', 'un_trailing_30d_pct',
            'dv_active_devs', 'ls_days_since_listing']:
    check(f"describe: {col}", 'unknown' not in describe_column(col).lower())

# ---------------------------------------------------------------------------
# 6. Table schema + date sanity (skipped when a table is absent)
# ---------------------------------------------------------------------------
from dbutil import load_data

TODAY = pd.Timestamp.now().normalize()

def table_checks(name, cols, date_col, min_syms=None, allow_future=False):
    try:
        t = load_data(name)
    except Exception:
        t = None
    if t is None or t.empty:
        print(f"[SKIP] table {name}: not built yet")
        return
    check(f"{name}: schema", set(cols) <= set(t.columns),
          f"missing={set(cols) - set(t.columns)}")
    d = pd.to_datetime(t[date_col])
    if not allow_future:
        check(f"{name}: no future timestamps", d.max() <= TODAY + pd.Timedelta('1D'))
    if min_syms:
        check(f"{name}: >= {min_syms} symbols", t['symbol'].nunique() >= min_syms,
              f"got {t['symbol'].nunique()}")

table_checks('stablecoin_supply', ['date', 'asset', 'chain', 'supply_usd'], 'date')
# unlock calendar legitimately extends into the future
table_checks('token_unlocks_daily',
             ['date', 'symbol', 'unlocked_amt', 'cum_pct', 'daily_pct'],
             'date', min_syms=20, allow_future=True)
table_checks('dev_activity',
             ['date', 'symbol', 'all_devs', 'exclusive_devs',
              'full_time_devs', 'num_commits'],
             'date', min_syms=80)
table_checks('listings', ['symbol', 'first_trade', 'venue'],
             'first_trade', min_syms=120)

try:
    ul = load_data('token_unlocks_daily')
except Exception:
    ul = None
if ul is not None and not ul.empty:
    check("token_unlocks_daily: daily_pct in [0, 1]",
          bool(((ul['daily_pct'] >= 0) & (ul['daily_pct'] <= 1)).all()))
    check("token_unlocks_daily: cum_pct monotone per symbol",
          bool(ul.sort_values(['symbol', 'date'])
                 .groupby('symbol')['cum_pct'].apply(
                     lambda s: (s.diff().fillna(0) >= -1e-9).all()).all()))

# ---------------------------------------------------------------------------
print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURES: {FAILURES}")
    sys.exit(1)
print("all slow-data checks passed")
