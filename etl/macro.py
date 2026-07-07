"""
ETL for macro / event data (single job, idempotent full refresh).

Point-in-time discipline - ONLY revision-immune data:
  1. Market-priced daily series from FRED (yields, VIX, breakevens, RRP,
     Fed balance sheet, broad dollar): prices/positions are never revised.
     A FRED API key is required (free): .env  FRED_KEY=...
  2. Scheduled release calendars: CPI and Employment Situation (NFP) release
     DATES are fetched from the FRED releases API (not typed by hand); FOMC
     statement dates are the Fed's published schedule (embedded below -
     scheduled years in advance, so "hours until FOMC" is legitimately
     knowable ahead of time). Release times are fixed by convention
     (CPI/NFP 08:30 ET, FOMC statement 14:00 ET) and converted to UTC.
  3. Crypto aggregates from DeFiLlama (no key): total stablecoin
     circulating supply and total DeFi TVL (TVL history can be restated
     when adapters change - mildly weaker PIT; used as a slow conditioner
     only). Fear & Greed index from alternative.me (no key; index values
     are published once and not revised).

AVAILABILITY CONVENTION (critical for causality): every macro_daily value is
stamped at the FIRST UTC DATE IT IS USABLE, i.e. observation date + the
series' publication lag (per-series below). Downstream features may use a
value on all bars of its stamped date - no further shifting needed. Event
times in macro_events are exact UTC datetimes.

Tables:
  macro_daily  [date, <one column per series>]   (wide, availability-dated)
  macro_events [event_time_utc, event_type]      (fomc | cpi | nfp)

Usage:
    uv run etl/macro.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from dbutil import save_data
from config import get, load_env_key

FRED_API = 'https://api.stlouisfed.org/fred'
START_DATE = get('data.start_date', '2023-01-01')
# Fetch extra history so trailing windows (betas, event profiles) are warm
# at the panel start.
FETCH_START = (pd.Timestamp(START_DATE) - pd.Timedelta(days=400)).date().isoformat()

# FRED series -> (output column, publication lag in calendar days).
# Lags are conservative: H.15 yields / CBOE VIX post next business day (1);
# H.4.1 (WALCL, RRP is same-day but keep 1) releases Thursday for the
# Wednesday level (2); the broad dollar index (DTWEXBGS) posts with ~a week.
FRED_SERIES = {
    'DGS2':      ('rates2y', 1),
    'DGS10':     ('rates10y', 1),
    'VIXCLS':    ('vix', 1),
    'T10YIE':    ('breakeven10', 1),
    'RRPONTSYD': ('rrp', 1),
    'WALCL':     ('fed_bs', 2),
    'DTWEXBGS':  ('dollar', 7),
}

# FOMC statement dates (meeting second day), from the Fed's published
# schedule (federalreserve.gov/monetarypolicy/fomccalendars.htm). Statements
# at 14:00 America/New_York. Extend this list when new schedules publish.
FOMC_DATES = [
    # 2022 (warmup history)
    '2022-01-26', '2022-03-16', '2022-05-04', '2022-06-15',
    '2022-07-27', '2022-09-21', '2022-11-02', '2022-12-14',
    # 2023
    '2023-02-01', '2023-03-22', '2023-05-03', '2023-06-14',
    '2023-07-26', '2023-09-20', '2023-11-01', '2023-12-13',
    # 2024
    '2024-01-31', '2024-03-20', '2024-05-01', '2024-06-12',
    '2024-07-31', '2024-09-18', '2024-11-07', '2024-12-18',
    # 2025
    '2025-01-29', '2025-03-19', '2025-05-07', '2025-06-18',
    '2025-07-30', '2025-09-17', '2025-10-29', '2025-12-10',
    # 2026
    '2026-01-28', '2026-03-18', '2026-04-29', '2026-06-17',
    '2026-07-29', '2026-09-16', '2026-10-28', '2026-12-09',
]

ET = ZoneInfo('America/New_York')


def _get(url: str, params: dict, timeout: int = 30) -> dict:
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


# =============================================================================
# FRED
# =============================================================================

def fred_observations(series_id: str, api_key: str) -> pd.Series:
    """Daily observations [observation date -> value]; '.' (missing) dropped."""
    js = _get(f'{FRED_API}/series/observations', {
        'series_id': series_id, 'api_key': api_key, 'file_type': 'json',
        'observation_start': FETCH_START,
    })
    return parse_fred_observations(js)


def parse_fred_observations(js: dict) -> pd.Series:
    rows = [(o['date'], o['value']) for o in js.get('observations', [])
            if o.get('value') not in ('.', '', None)]
    if not rows:
        return pd.Series(dtype=float)
    s = pd.Series({pd.Timestamp(d): float(v) for d, v in rows}).sort_index()
    s.index.name = 'date'
    return s


def fred_release_id(name: str, api_key: str) -> int:
    """Look the release id up by name (never hardcode ids)."""
    js = _get(f'{FRED_API}/releases', {
        'api_key': api_key, 'file_type': 'json', 'limit': 1000,
    })
    for rel in js.get('releases', []):
        if rel.get('name', '').strip().lower() == name.strip().lower():
            return int(rel['id'])
    raise LookupError(f'FRED release not found: {name!r}')


def fred_release_dates(release_id: int, api_key: str) -> list:
    """Historical + scheduled release dates for a FRED release."""
    js = _get(f'{FRED_API}/release/dates', {
        'release_id': release_id, 'api_key': api_key, 'file_type': 'json',
        'realtime_start': '1776-07-04', 'realtime_end': '9999-12-31',
        'sort_order': 'asc', 'limit': 10000,
        'include_release_dates_with_no_data': 'true',
    })
    return sorted({d['date'] for d in js.get('release_dates', [])
                   if d.get('date', '') >= FETCH_START})


# =============================================================================
# Free crypto aggregates
# =============================================================================

def defillama_stables() -> pd.Series:
    """Total stablecoin circulating USD, daily."""
    js = _get('https://stablecoins.llama.fi/stablecoincharts/all', {})
    return parse_llama_stables(js)


def parse_llama_stables(js: list) -> pd.Series:
    rows = {}
    for p in js:
        circ = p.get('totalCirculatingUSD') or {}
        total = sum(v for v in circ.values() if isinstance(v, (int, float)))
        if total > 0:
            rows[pd.Timestamp(int(p['date']), unit='s').normalize()] = float(total)
    s = pd.Series(rows).sort_index()
    s.index.name = 'date'
    return s


def defillama_tvl() -> pd.Series:
    """Total DeFi TVL (all chains), daily. NOTE: history may be restated
    when protocol adapters change - slow conditioner use only."""
    js = _get('https://api.llama.fi/v2/historicalChainTvl', {})
    s = pd.Series({pd.Timestamp(int(p['date']), unit='s').normalize():
                   float(p['tvl']) for p in js if p.get('tvl')}).sort_index()
    s.index.name = 'date'
    return s


def fear_greed() -> pd.Series:
    """Crypto Fear & Greed index (0-100), daily; published once, not revised."""
    js = _get('https://api.alternative.me/fng/', {'limit': 0, 'format': 'json'})
    s = pd.Series({pd.Timestamp(int(p['timestamp']), unit='s').normalize():
                   float(p['value']) for p in js.get('data', [])}).sort_index()
    s.index.name = 'date'
    return s


# =============================================================================
# Assembly
# =============================================================================

def availability_shift(s: pd.Series, lag_days: int) -> pd.Series:
    """Restamp a series from observation date to first-usable UTC date."""
    out = s.copy()
    out.index = out.index + pd.Timedelta(days=lag_days)
    return out


def build_events(api_key) -> pd.DataFrame:
    rows = [{'event_time_utc': pd.Timestamp(
                datetime.fromisoformat(f'{d}T14:00').replace(tzinfo=ET)
             ).tz_convert('UTC').tz_localize(None),
             'event_type': 'fomc'} for d in FOMC_DATES]

    if api_key:
        for name, etype in [('Consumer Price Index', 'cpi'),
                            ('Employment Situation', 'nfp')]:
            try:
                rid = fred_release_id(name, api_key)
                for d in fred_release_dates(rid, api_key):
                    rows.append({'event_time_utc': pd.Timestamp(
                        datetime.fromisoformat(f'{d}T08:30').replace(tzinfo=ET)
                    ).tz_convert('UTC').tz_localize(None),
                        'event_type': etype})
                print(f"  {etype}: release dates fetched ({name})")
            except Exception as e:
                print(f"  WARNING: {etype} release dates unavailable ({e})")
    else:
        print("  WARNING: no FRED key - macro_events has FOMC dates only")

    ev = pd.DataFrame(rows).sort_values('event_time_utc').reset_index(drop=True)
    return ev.drop_duplicates()


def build_daily(api_key) -> pd.DataFrame:
    cols = {}
    if api_key:
        for sid, (col, lag) in FRED_SERIES.items():
            try:
                cols[col] = availability_shift(fred_observations(sid, api_key), lag)
                print(f"  {col:12s} <- FRED {sid} ({len(cols[col])} obs, lag {lag}d)")
            except Exception as e:
                print(f"  WARNING: FRED {sid} failed ({e})")
    else:
        print("  WARNING: no FRED key (.env FRED_KEY=...) - skipping FRED series")

    for name, fn, lag in [('stables_mcap', defillama_stables, 1),
                          ('defi_tvl', defillama_tvl, 1),
                          ('fear_greed', fear_greed, 1)]:
        try:
            cols[name] = availability_shift(fn(), lag)
            print(f"  {name:12s} <- {fn.__name__} ({len(cols[name])} obs, lag {lag}d)")
        except Exception as e:
            print(f"  WARNING: {name} failed ({e})")

    if not cols:
        return pd.DataFrame()
    wide = pd.DataFrame(cols).sort_index()
    wide = wide[wide.index >= pd.Timestamp(FETCH_START)]
    return wide.reset_index().rename(columns={'index': 'date'})


def main():
    print("=" * 60)
    print("Macro / event ETL (PIT: market-priced series + schedules)")
    print("=" * 60)
    api_key = load_env_key('FRED_KEY')

    print("\nEvents (macro_events):")
    events = build_events(api_key)
    n_by_type = events['event_type'].value_counts().to_dict()
    print(f"  {len(events)} events: {n_by_type}")

    print("\nDaily series (macro_daily, availability-dated):")
    daily = build_daily(api_key)
    if not daily.empty:
        span = f"{daily['date'].min().date()} -> {daily['date'].max().date()}"
        print(f"  {len(daily)} rows x {len(daily.columns) - 1} series ({span})")

    # Idempotent full refresh: these are small daily tables.
    save_data('macro_events', events, mode='overwrite',
              datetime_columns=['event_time_utc'])
    if not daily.empty:
        save_data('macro_daily', daily, mode='overwrite',
                  datetime_columns=['date'])
    print("\nSaved: macro_events" + (", macro_daily" if not daily.empty else
                                     " (macro_daily skipped - no sources)"))


if __name__ == '__main__':
    main()
