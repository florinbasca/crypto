"""
Synthetic checks for the event-study core (research/signals/event_study.py).
No database required - the db-touching assembly is exercised only by the
script itself.

1. find_events - trigger ops (spell vs crossing), NaN discipline (a data gap
   is not a crossing), require-clauses, per-symbol cooldown.
2. event_cars - forward CAR sums D+1..D+h (causal: never includes the event
   day), pre-event CAR sums the k days ending AT the event day, missing
   forward coverage -> NaN.
3. clustered_stats - same-day events collapse to one observation; a planted
   post-event drift is recovered with a significant clustered t.

Run: uv run tests/event_study_checks.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from research.signals import event_study as es

FAILURES = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name} {detail}")
    if not cond:
        FAILURES.append(name)


D = pd.date_range('2024-01-01', periods=20, freq='1D')

# ---------------------------------------------------------------------------
# 1. find_events
# ---------------------------------------------------------------------------
print("--- 1. event triggers ---")
# days_to: counts down 10..1, resets to 12 (an unlock passing), NaN head
days_to = [np.nan, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 12, 11, 10, 9, 8, 7, 6, 5, 4]
size = [0.01] * 20
daily = pd.DataFrame({'date': D, 'symbol': 'AAA',
                      'un_days_to_next': days_to, 'un_next_pct': size})

ev = es.find_events(daily, {'column': 'un_days_to_next', 'op': 'cross_below',
                            'threshold': 7.0}, cooldown_days=0)
check("cross_below: fires on each crossing day (7 -> 6), not the spell",
      list(ev['date']) == [D[5], D[17]], f"({list(ev['date'].dt.day)})")

ev_cd = es.find_events(daily, {'column': 'un_days_to_next',
                               'op': 'cross_below', 'threshold': 7.0},
                       cooldown_days=30)
check("cooldown: second same-symbol event within the window dropped",
      list(ev_cd['date']) == [D[5]])

# NaN yesterday: 'below' may fire (spell start), a crossing may NOT
head = pd.DataFrame({'date': D[:3], 'symbol': 'BBB',
                     'un_days_to_next': [3.0, 3.0, 3.0],
                     'un_next_pct': [0.01] * 3})
first = pd.concat([head.assign(un_days_to_next=[np.nan, 3.0, 3.0])])
check("NaN discipline: gap -> no crossing, but a spell can start",
      es.find_events(first, {'column': 'un_days_to_next',
                             'op': 'cross_below', 'threshold': 7.0},
                     0).empty
      and list(es.find_events(first, {'column': 'un_days_to_next',
                                      'op': 'below', 'threshold': 7.0},
                              0)['date']) == [D[1]])

ev_req = es.find_events(daily, {'column': 'un_days_to_next',
                                'op': 'cross_below', 'threshold': 7.0,
                                'require': [['un_next_pct', 'above', 0.05]]},
                        cooldown_days=0)
check("require clause: AND condition filters events", ev_req.empty)

up = pd.DataFrame({'date': D, 'symbol': 'CCC',
                   'x': np.r_[np.zeros(10), np.ones(10) * 0.5]})
check("cross_above + spell 'above' both fire once at the step",
      list(es.find_events(up, {'column': 'x', 'op': 'cross_above',
                               'threshold': 0.25}, 0)['date']) == [D[10]]
      and list(es.find_events(up, {'column': 'x', 'op': 'above',
                                   'threshold': 0.25}, 0)['date']) == [D[10]])

check("missing column -> no events (not a crash)",
      es.find_events(daily, {'column': 'nope', 'op': 'below',
                             'threshold': 1.0}, 0).empty)

# ---------------------------------------------------------------------------
# 2. event_cars
# ---------------------------------------------------------------------------
print("--- 2. CAR windows ---")
# AAA returns 1% on every day; event at D[9]
res = pd.DataFrame({'date': np.tile(D, 1), 'symbol': 'AAA',
                    'res': np.full(20, 0.01)})
events = pd.DataFrame({'symbol': ['AAA'], 'date': [D[9]]})
cars = es.event_cars(events, res, horizons=[-5, 1, 3, 5, 15])
r = cars.iloc[0]
check("forward CAR: D+1..D+h, event day excluded",
      abs(r['car_1'] - 0.01) < 1e-12 and abs(r['car_3'] - 0.03) < 1e-12
      and abs(r['car_5'] - 0.05) < 1e-12)
check("pre CAR: the k days ending AT the event day",
      abs(r['car_-5'] - 0.05) < 1e-12)
check("insufficient forward coverage -> NaN (never zero-filled)",
      np.isnan(r['car_15']))
check("event for an unknown symbol/date is skipped",
      es.event_cars(pd.DataFrame({'symbol': ['ZZZ'], 'date': [D[9]]}),
                    res, [1]).empty)

# ---------------------------------------------------------------------------
# 3. clustered stats
# ---------------------------------------------------------------------------
print("--- 3. day-clustered t-stats ---")
rng = np.random.default_rng(11)
n_days, n_per_day = 40, 5
dates = pd.date_range('2024-01-01', periods=n_days, freq='3D')
rows = []
for d in dates:
    day_shock = rng.normal(0, 0.02)          # shared same-day shock
    for k in range(n_per_day):
        rows.append({'date': d, 'car_3': 0.01 + day_shock
                     + rng.normal(0, 0.001)})
cc = pd.DataFrame(rows)
stats = es.clustered_stats(cc, [3]).iloc[0]
check("clustering: n counts events, day clusters set the dof",
      stats['n_events'] == n_days * n_per_day
      and stats['n_event_days'] == n_days)
# Naive (per-event) t would claim sqrt(5)x the precision; the clustered t
# must be close to the day-level t, not the naive one.
day_means = cc.groupby('date')['car_3'].mean()
t_day = day_means.mean() / (day_means.std(ddof=1) / np.sqrt(n_days))
check("clustering: t equals the day-level t (same-day events are 1 obs)",
      abs(stats['tstat_clustered'] - t_day) < 1e-9,
      f"(t {stats['tstat_clustered']:.2f})")
check("clustering: planted 1% drift detected",
      stats['mean_car'] > 0.005 and stats['tstat_clustered'] > 2.0,
      f"(mean {stats['mean_car']:.4f}, t {stats['tstat_clustered']:.2f})")

check("degenerate: single day -> NaN t, not a crash",
      np.isnan(es.clustered_stats(cc[cc['date'] == dates[0]], [3])
               .iloc[0]['tstat_clustered']))

# ---------------------------------------------------------------------------
print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURES: {FAILURES}")
    sys.exit(1)
print("ALL EVENT STUDY CHECKS PASSED")
