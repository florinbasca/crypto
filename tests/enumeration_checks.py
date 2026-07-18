"""
Checks for INVENT's deterministic pair-sweep lane
(research/signals/enumeration.py). No database required.

1. A planted pair effect (forward residuals ~ z(A) x z(B)) is ranked at the
   top of the sweep; a contemporaneous-only (non-predictive) effect is not.
2. Emitted candidates validate and compile under the normal grammar.
3. top_n and the enabled flag are respected.

Run: uv run tests/enumeration_checks.py
"""

import copy
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from config import get
from research.signals import data as data_mod
from research.signals import generation as gen
from research.signals.enumeration import enumerate_candidates

FAILURES = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        FAILURES.append(name)


def make_cfg(top_n=5):
    cfg = copy.deepcopy(get('discovery'))
    cfg['target_lag_bars'] = 6
    cfg['embargo_bars'] = 6
    cfg['min_assets_per_timestamp'] = 10
    cfg['families'] = {'residual_shape': ['res_']}
    cfg['curve'] = {**cfg['curve'], 'horizon_bars': 6}
    cfg['enumeration'] = {'enabled': True, 'top_n': top_n,
                          'agg_bars': 6, 'horizon_steps': 8}
    return cfg


N_FEATS = 6
FEAT_COLS = [f'res_f{i}' for i in range(N_FEATS)]


def make_panel(lead=1, persist=True, plant=0.004, seed=3, n_sym=20,
               n_days=20, template='product'):
    """residual_return[t] = plant * planted_signal[t - lead] + noise, with
    decoy features. template: 'product' plants z(f0) x z(f1); 'single'
    plants z(f0) alone. persist=True: slow AR features (a real, holdable
    effect). persist=False with lead=0: the effect exists ONLY on the same
    bar and features have no memory - zero predictive content, the screen
    must not rank it."""
    r = np.random.default_rng(seed)
    n = n_days * 144
    ts = pd.date_range('2024-01-01', periods=n, freq='10min')
    fs = []
    for _ in range(N_FEATS):
        sh = r.normal(size=(n, n_sym))
        if persist:
            x = np.zeros((n, n_sym))
            for t in range(1, n):
                x[t] = 0.99 * x[t - 1] + 0.1 * sh[t]
        else:
            x = sh
        fs.append(x)
    za = (fs[0] - fs[0].mean(1, keepdims=True)) / fs[0].std(1, keepdims=True)
    zb = (fs[1] - fs[1].mean(1, keepdims=True)) / fs[1].std(1, keepdims=True)
    prod = za * zb if template == 'product' else za
    prod = prod - prod.mean(1, keepdims=True)
    ret = r.normal(0, 1e-2, (n, n_sym))
    if lead:
        ret[lead:] += plant * prod[:-lead]
    else:
        ret += plant * prod
    frames = [pd.DataFrame({
        'timestamp': ts, 'symbol': f'S{i:02d}',
        **{c: fs[k][:, i] for k, c in enumerate(FEAT_COLS)},
        'residual_return': ret[:, i]}) for i in range(n_sym)]
    return (pd.concat(frames, ignore_index=True)
            .sort_values(['symbol', 'timestamp']).reset_index(drop=True))


ROLL = data_mod.Roll(0, pd.Timestamp('2024-01-01'),
                     pd.Timestamp('2024-01-19'), pd.Timestamp('2024-01-20'),
                     pd.Timestamp('2024-01-21'))
CFG = make_cfg()
FAMS = {'residual_shape': FEAT_COLS}

cands = enumerate_candidates(make_panel(lead=1), ROLL, FAMS, CFG)
check("sweep: emits top_n candidates", len(cands) == 5, f"({len(cands)})")
top_cols = gen.candidate_columns(cands[0]) if cands else set()
check("sweep: planted pair ranked first",
      top_cols == {'res_f0', 'res_f1'}, f"({sorted(top_cols)})")

# a planted SINGLE-feature effect is found by the single template (not
# forced into a pair costume)
cands1 = enumerate_candidates(make_panel(lead=1, template='single'), ROLL,
                              FAMS, CFG)
check("templates: planted single-feature effect found as a single",
      bool(cands1)
      and gen.candidate_columns(cands1[0]) == {'res_f0'},
      f"({sorted(gen.candidate_columns(cands1[0])) if cands1 else []})")

# same-bar effect on MEMORYLESS features carries zero forward information -
# the screen (which only reads forward steps) must not rank it first.
cands0 = enumerate_candidates(make_panel(lead=0, persist=False), ROLL,
                              FAMS, CFG)
top0 = gen.candidate_columns(cands0[0]) if cands0 else set()
check("causality: same-bar-only effect not ranked first",
      top0 != {'res_f0', 'res_f1'}, f"({sorted(top0)})")

# emitted candidates pass the normal grammar and compile
panel = make_panel(lead=1)
ok = True
for c in cands:
    try:
        gen.validate_candidate(c, FEAT_COLS, CFG['dsl'])
        ok = ok and not gen.compile_candidate(c, panel).empty
    except gen.ValidationError:
        ok = False
check("grammar: sweep candidates validate and compile", ok)
check("config: disabled -> no candidates",
      enumerate_candidates(
          panel, ROLL, FAMS,
          {**CFG, 'enumeration': {'enabled': False}}) == [])

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURES: {FAILURES}")
    sys.exit(1)
print("ALL ENUMERATION CHECKS PASSED")
