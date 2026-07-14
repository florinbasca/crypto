"""
EVENT STUDY: the slow families, tested the way slow signals must be.

Unlocks, listings and dev activity produce a handful of events per month -
the discovery loop's one-month select window can never confirm them at any
honest bar (a few events is not a sample). This script tests each event type
ONCE over the FULL history instead: pool every occurrence 2023-2026, measure
the mean cumulative forward RESIDUAL return around the event, and price the
cross-event dependence honestly (events on the same calendar day share
market shocks, so t-stats are clustered by event day).

Purely diagnostic - nothing here feeds discovery, promotion or the
walk-forward. If an event type shows a real, persistent post-event drift,
the follow-up is a dedicated curated signal (research/lib/spaces.py), not a
DSL candidate.

Event definitions live in config discovery.event_study.events - a named
trigger on a daily feature column ('cross_below'/'cross_above' fire on the
crossing day, 'below'/'above' on the first day of a contiguous spell), plus
optional 'require' AND-clauses, deduped per symbol by cooldown_days.
Features are bar-end stamped and taken at each day's LAST bar, so an event
on day D uses data through D and the measured window starts at D+1 - causal
by construction. Pre-event drift (anticipation/leakage view) is reported
alongside; CAR(-k) sums the k days ENDING at the event day.

Usage:
    uv run research/signals/event_study.py [--event NAME] [--save]

    --event NAME   run one event type only (default: all configured)
    --save         persist per-horizon stats to the event_study_results table
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import logging

import numpy as np
import pandas as pd

from config import config as global_config, get

logging.basicConfig(level=logging.WARNING,
                    format=global_config['logging']['format'],
                    datefmt=global_config['logging']['datefmt'])


# =============================================================================
# Pure core (synthetic-testable, no database)
# =============================================================================

def find_events(daily: pd.DataFrame, spec: dict,
                cooldown_days: int) -> pd.DataFrame:
    """Event days for one spec on a per-symbol DAILY frame.

    daily: [date, symbol, <feature columns...>], one row per (symbol, date).
    spec: {'column', 'op', 'threshold', 'require': [[col, 'above'|'below',
    thr], ...]}. Ops: 'below'/'above' fire on the FIRST day of a contiguous
    spell beyond the threshold; 'cross_below'/'cross_above' require the
    previous day's value on the other side (NaN yesterday never fires a
    cross - a data gap is not a crossing). Events within cooldown_days of a
    kept same-symbol event are dropped (first wins).

    Returns [symbol, date] sorted by (symbol, date).
    """
    col = spec['column']
    op = spec['op']
    thr = float(spec['threshold'])
    if col not in daily.columns:
        return pd.DataFrame(columns=['symbol', 'date'])

    d = daily.sort_values(['symbol', 'date'])
    x = d[col]
    prev = d.groupby('symbol')[col].shift(1)
    if op == 'below':
        hit = (x < thr) & ~(prev < thr)
    elif op == 'above':
        hit = (x > thr) & ~(prev > thr)
    elif op == 'cross_below':
        hit = (x < thr) & (prev >= thr)
    elif op == 'cross_above':
        hit = (x > thr) & (prev <= thr)
    else:
        raise ValueError(f"unknown event op: {op}")

    for rcol, rop, rthr in spec.get('require', []):
        if rcol not in d.columns:
            return pd.DataFrame(columns=['symbol', 'date'])
        hit &= (d[rcol] > float(rthr)) if rop == 'above' \
            else (d[rcol] < float(rthr))

    ev = d.loc[hit.fillna(False), ['symbol', 'date']]
    if ev.empty or cooldown_days <= 0:
        return ev.reset_index(drop=True)

    keep = []
    last: dict = {}
    for row in ev.itertuples(index=False):
        prev_date = last.get(row.symbol)
        if prev_date is None or (row.date - prev_date).days > cooldown_days:
            keep.append(row)
            last[row.symbol] = row.date
    return pd.DataFrame(keep, columns=['symbol', 'date'])


def event_cars(events: pd.DataFrame, daily_res: pd.DataFrame,
               horizons: list) -> pd.DataFrame:
    """Per-event cumulative residual returns at each horizon.

    daily_res: [date, symbol, res] with res = the day's summed residual
    return. horizons: signed day counts; CAR(+h) = sum over event day +1..+h
    (the tradeable window), CAR(-k) = sum over the k days ENDING at the event
    day (pre-event drift). Events without full forward coverage at a horizon
    get NaN there (never zero-filled).

    Returns [symbol, date, car_<h>...], one row per event.
    """
    wide = daily_res.pivot_table(index='date', columns='symbol',
                                 values='res', aggfunc='first').sort_index()
    cum = wide.fillna(0.0).cumsum()
    has = wide.notna()
    dates = cum.index
    pos = {d: i for i, d in enumerate(dates)}

    rows = []
    for ev in events.itertuples(index=False):
        if ev.symbol not in cum.columns or ev.date not in pos:
            continue
        i = pos[ev.date]
        row = {'symbol': ev.symbol, 'date': ev.date}
        c = cum[ev.symbol]
        h_col = has[ev.symbol]
        for h in horizons:
            j = i + int(h)
            if h > 0:
                ok = 0 <= j < len(dates) and h_col.iloc[i + 1:j + 1].any()
                row[f'car_{h}'] = (float(c.iloc[j] - c.iloc[i])
                                   if ok else np.nan)
            else:
                k = i + int(h)   # h negative: window [k+1 .. i]
                ok = k >= -1 and h_col.iloc[max(k + 1, 0):i + 1].any()
                base = float(c.iloc[k]) if k >= 0 else 0.0
                row[f'car_{h}'] = (float(c.iloc[i]) - base
                                   if ok else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def clustered_stats(cars: pd.DataFrame, horizons: list) -> pd.DataFrame:
    """Mean CAR and DAY-CLUSTERED t-stat per horizon.

    Events sharing a calendar day are one observation: same-day events sit in
    the same cross-section and share shocks, so treating them as independent
    would overstate n. Within-day means first, then t over the day means.
    """
    out = []
    for h in horizons:
        col = f'car_{h}'
        sub = cars[['date', col]].dropna()
        day_means = sub.groupby('date')[col].mean()
        n = len(day_means)
        mean = float(sub[col].mean()) if len(sub) else np.nan
        if n > 1 and day_means.std(ddof=1) > 0:
            t = float(day_means.mean()
                      / (day_means.std(ddof=1) / np.sqrt(n)))
        else:
            t = np.nan
        out.append({'horizon_days': int(h), 'mean_car': mean,
                    'tstat_clustered': t, 'n_events': int(len(sub)),
                    'n_event_days': n})
    return pd.DataFrame(out)


# =============================================================================
# Data assembly (the only db-touching code here)
# =============================================================================

def load_daily_panels(event_cols: list) -> tuple:
    """(daily feature frame, daily residual frame), universe-filtered.

    Features are bar-end stamped; the day's LAST bar value uses data through
    that day only, so day-D events are causal for a D+1 entry. Residuals are
    summed per (symbol, day).
    """
    from dbutil import load_data
    from research.lib.signal_eval import (load_universe_membership,
                                          universe_member_mask)

    res = load_data('residual_returns',
                    columns=['timestamp', 'symbol', 'residual_return'])
    if res is None or res.empty:
        raise SystemExit("residual_returns is empty - run "
                         "risk_model/residual_returns.py first")
    res['timestamp'] = pd.to_datetime(res['timestamp'])
    membership = load_universe_membership()
    if membership is not None:
        res = res[universe_member_mask(res, membership)]
    res['date'] = res['timestamp'].dt.normalize()
    daily_res = (res.groupby(['date', 'symbol'])['residual_return']
                 .sum().rename('res').reset_index())

    feats = load_data('features',
                      columns=['timestamp', 'symbol'] + sorted(set(event_cols)))
    if feats is None or feats.empty:
        raise SystemExit("features table is empty - run the feature ETL first")
    feats['timestamp'] = pd.to_datetime(feats['timestamp'])
    feats['date'] = feats['timestamp'].dt.normalize()
    daily_feat = (feats.sort_values('timestamp')
                  .groupby(['date', 'symbol'], as_index=False).last()
                  .drop(columns=['timestamp']))
    return daily_feat, daily_res


def main():
    parser = argparse.ArgumentParser(
        description='Full-history event studies for the slow families '
                    '(diagnostic only - see discovery.event_study in config)')
    parser.add_argument('--event', default=None,
                        help='Run one configured event type only')
    parser.add_argument('--save', action='store_true',
                        help='Persist stats to event_study_results')
    args = parser.parse_args()

    cfg = get('discovery.event_study')
    specs = cfg['events']
    if args.event:
        if args.event not in specs:
            raise SystemExit(f"unknown event '{args.event}' - configured: "
                             f"{sorted(specs)}")
        specs = {args.event: specs[args.event]}

    horizons = ([-int(cfg['pre_days'])]
                + [int(d) for d in cfg['report_days']
                   if int(d) <= int(cfg['horizon_days'])])
    event_cols = [s['column'] for s in specs.values()] + [
        r[0] for s in specs.values() for r in s.get('require', [])]

    print("Building daily panels (features last-bar-of-day, residuals "
          "day-summed, universe-filtered)...")
    daily_feat, daily_res = load_daily_panels(event_cols)
    print(f"{daily_res['date'].nunique():,} days, "
          f"{daily_res['symbol'].nunique()} symbols")

    all_stats = []
    for name, spec in specs.items():
        events = find_events(daily_feat, spec, int(cfg['cooldown_days']))
        print("\n" + "=" * 72)
        print(f"{name}: {spec['column']} {spec['op']} {spec['threshold']}"
              + (f" AND {spec['require']}" if spec.get('require') else ""))
        print("=" * 72)
        if events.empty:
            print("no events found (column missing or trigger never fired)")
            continue
        cars = event_cars(events, daily_res, horizons)
        stats = clustered_stats(cars, horizons)
        n = int(events.shape[0])
        flag = ("" if n >= int(cfg['min_events'])
                else f"  [LOW POWER: {n} < min_events "
                     f"{int(cfg['min_events'])}]")
        print(f"{n} events across {events['symbol'].nunique()} symbols, "
              f"{events['date'].nunique()} distinct days{flag}")
        with pd.option_context('display.float_format',
                               lambda x: f'{x:,.5f}'):
            print(stats.to_string(index=False))
        print("(horizon -k = the k days ENDING at the event day: "
              "anticipation, not tradeable)")
        stats['event'] = name
        all_stats.append(stats)

    if args.save and all_stats:
        from dbutil import save_data, delete_table
        out = pd.concat(all_stats, ignore_index=True)
        delete_table('event_study_results')
        save_data('event_study_results', out, mode='overwrite')
        print("\nSaved: event_study_results")


if __name__ == '__main__':
    main()
