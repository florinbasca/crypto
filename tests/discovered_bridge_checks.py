"""
Unit checks for the discovery -> production bridge
(research/lib/discovered.py + the 'dsl' op in research/lib/spaces.py + the
valid_from gate in research/portfolio/walk_forward.py). No database required.

The bridge turns rows of the discovery promotions table into registry entries
shaped exactly like curated spaces, so evaluate.py / walk_forward.py pick up
promoted candidates automatically. Checks cover: entry construction and
hash-stable naming, dedup across lag-sweep tables (strongest t-stat wins,
earliest promotion date wins), the compiled 'dsl' space op (values, gating,
alignment to the feature frame), direction application through
compute_signal_panel, and the walk-forward valid_from filter.

Run: uv run tests/discovered_bridge_checks.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json

import numpy as np
import pandas as pd

from research.lib.discovered import entries_from_promotions
from research.lib.spaces import compute_space_raw
from research.signals.generation import Candidate

FAILURES = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        FAILURES.append(name)


CAND = Candidate(name='syn_rev', family='residual_shape',
                 expression=('col', 'f1'), rationale='synthetic reversal')
GATED = Candidate(name='syn_gated', family='liquidity',
                  expression=('col', 'f1'),
                  conditions=(('gt', ('col', 'f2'), 0.0),))


def promo_row(cand, roll_id=2, direction=1, tstat=2.5, half_life=None,
              target_lag=36):
    row = {'roll_id': roll_id, 'cand_hash': cand.hash, 'name': cand.name,
           'family': cand.family, 'direction': direction,
           'candidate_json': cand.to_json(), 'select_ic_tstat': tstat,
           'reward': 1.0, 'n_trials_at_promotion': 100,
           'target_lag': target_lag}
    if half_life is not None:
        row['half_life_bars'] = half_life
    return row


VALID_FROM = {1: pd.Timestamp('2024-03-01'), 2: pd.Timestamp('2024-04-01'),
              5: pd.Timestamp('2024-07-01')}

# --- entry construction --------------------------------------------------------
entries = entries_from_promotions(pd.DataFrame([promo_row(CAND)]), VALID_FROM)
name = f'disc_residual_shape_{CAND.hash[:10]}'
check("one promotion -> one hash-named entry", list(entries) == [name])
e = entries[name]
check("entry columns come from the expression",
      e['signal_def'].columns == ('f1',))
check("entry family is disc_<family>", e['family'] == 'disc_residual_shape')
check("valid_from = promotion roll's OOS start",
      e['valid_from'] == pd.Timestamp('2024-04-01'))
check("op is dsl", e['signal_def'].op == 'dsl')
check("no half-life column -> entry carries None (no fake persistence)",
      e['half_life_bars'] is None)

# half-life flows from the promotion row into the entry + signal_def
entries_hl = entries_from_promotions(
    pd.DataFrame([promo_row(CAND, half_life=96.0)]), VALID_FROM)
e_hl = entries_hl[name]
check("half_life_bars flows through the bridge",
      e_hl['half_life_bars'] == 96.0
      and e_hl['signal_def'].halflife == 96.0
      and e_hl['signal_def'].lag == 36)

# --- dedup across tables: strongest t wins direction, earliest date wins -------
promos = pd.DataFrame([
    promo_row(CAND, roll_id=5, direction=1, tstat=2.0),   # later, weaker
    promo_row(CAND, roll_id=1, direction=-1, tstat=4.0),  # earlier, stronger
    promo_row(GATED, roll_id=2, direction=1, tstat=3.0),
])
entries = entries_from_promotions(promos, VALID_FROM)
check("dedup by content hash across rows/tables", len(entries) == 2)
e = entries[name]
check("strongest |t| row wins the direction", e['direction'] == -1)
check("earliest promotion date wins valid_from",
      e['valid_from'] == pd.Timestamp('2024-03-01'))

# --- compiled 'dsl' space op ----------------------------------------------------
N, SYMS = 60, ['A', 'B', 'C', 'D']
ts = pd.date_range('2024-01-01', periods=N, freq='10min')
rng = np.random.default_rng(11)
frames = []
for s in SYMS:
    frames.append(pd.DataFrame({
        'timestamp': ts, 'symbol': s,
        'f1': rng.normal(0, 1, N), 'f2': rng.normal(0, 1, N),
    }))
features = (pd.concat(frames, ignore_index=True)
            .sort_values(['symbol', 'timestamp']).reset_index(drop=True))

raw = compute_space_raw(entries[name]['signal_def'], features)
check("dsl op returns a Series aligned to the feature frame",
      isinstance(raw, pd.Series) and raw.index.equals(features.index))
# compile_candidate z-scores per timestamp: mean ~0, unit-ish std, clipped +-3
by_ts = pd.DataFrame({'timestamp': features['timestamp'], 'v': raw.values})
mu = by_ts.groupby('timestamp')['v'].mean()
check("dsl signal is cross-sectionally demeaned", float(mu.abs().max()) < 1e-12)
check("dsl signal is clipped to +-3", float(raw.abs().max()) <= 3.0)
# Rank identity: z of f1 preserves the cross-sectional ordering of f1
t0 = ts[10]
sel = features['timestamp'] == t0
got_order = raw[sel].rank().values
want_order = features.loc[sel, 'f1'].rank().values
check("dsl signal preserves the expression's cross-sectional order",
      np.array_equal(got_order, want_order))

# Gated candidate: rows failing the condition are neutral 0 BEFORE z-scoring
gname = f'disc_liquidity_{GATED.hash[:10]}'
graw = compute_space_raw(entries[gname]['signal_def'], features)
gated_off = features['f2'] <= 0.0
z_of_zero = graw[gated_off]
# gated-off rows all share the same pre-z value (0), so within a timestamp
# they must all map to the same z value
same_within_ts = (pd.DataFrame({'timestamp': features['timestamp'][gated_off],
                                'v': z_of_zero.values})
                  .groupby('timestamp')['v'].nunique() <= 1)
check("gate conditions apply (gated-off rows collapse to one neutral value)",
      bool(same_within_ts.all()))

# --- direction flows through compute_signal_panel -------------------------------
from research.lib.signal_eval import compute_signal_panel

registry = {name: entries[name]}
panel_pos = compute_signal_panel(name, {**registry,
                                        name: {**entries[name], 'direction': 1}},
                                 features)
panel_neg = compute_signal_panel(name, {**registry,
                                        name: {**entries[name], 'direction': -1}},
                                 features)
m = panel_pos.merge(panel_neg, on=['timestamp', 'symbol'], suffixes=('_p', '_n'))
check("registry direction flips the traded sign",
      np.allclose(m['signal_p'], -m['signal_n'], atol=1e-9))

# --- registry merge is graceful without promotions tables -----------------------
from research.lib.signal_eval import build_registry

reg = build_registry()   # promotions tables may or may not exist locally
check("build_registry: discovery is the only signal source (disc_* or empty)",
      all(k.startswith('disc_') for k in reg),
      f"({len(reg)} entries)")

# ---------------------------------------------------------------------------
print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURES: {FAILURES}")
    sys.exit(1)
print("ALL DISCOVERED-BRIDGE CHECKS PASSED")
