"""
Synthetic-data checks for the agentic signal discovery engine
(research/signals/). No database required.

1. Evaluator calibration - the measurement instrument recovers a planted
   IC (and reports ~0 on a zero-IC panel).
2. Operator causality - truncation test over the WHOLE DSL registry, plus a
   deliberately look-ahead operator that MUST be caught.
3. Window discipline - roll construction, purge/embargo slicing, and
   boundedness (unknown columns/windows/depth rejected).
4. Noise in, nothing out - promotion gates on pure-noise candidates.
5. End-to-end - the random-proposer search finds and promotes a planted
   effect, the ledger is complete, runs are reproducible, and the OOS book
   is dollar-neutral.

Run: uv run tests/discovery_checks.py
"""

import copy
import math
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from config import get
from research.signals import generation as gen
from research.signals import data as data_mod
from research.signals import search as search_mod
from research.signals import promotion as bt_mod

rng = np.random.default_rng(7)
FAILURES = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name} {detail}")
    if not cond:
        FAILURES.append(name)


# ---------------------------------------------------------------------------
# Test config: the global discovery config with small, fast overrides.
# Windows are passed as explicit Roll timestamps (day-granular) so the whole
# suite runs on ~1 month of synthetic 10-minute data.
# ---------------------------------------------------------------------------

def test_cfg():
    cfg = copy.deepcopy(get('discovery'))
    cfg['target_lag_bars'] = 6
    cfg['embargo_bars'] = 6
    cfg['min_assets_per_timestamp'] = 10
    cfg['families'] = {
        'residual_shape': ['res_'],
        'volatility_regime': ['vol_'],
    }
    cfg['search'].update({'seed': 7, 'n_generations': 2, 'batch_size': 8,
                          'survivors': 6, 'diversity_max_corr': 0.9})
    # min_select_days 0: the synthetic select window is only ~6 days long.
    # min_capture 0: a single-lag test grid fits every half-life to the
    # shortest grid value, which the production capture floor would block.
    # Pooling/ranking specifics are covered by
    # tests/promotion_pooling_checks.py.
    # book_frac 0 -> fixed book_size (deterministic small-sample tests);
    # econ_cost_bps 0 -> filter 3's cost side off (synthetic alphas are
    # tiny; the sign/activity/duplicate filters are what these checks
    # exercise). Quintile/econ specifics live in tests/choose_checks.py.
    cfg['promotion'].update({'book_frac': 0.0, 'book_size': 10,
                             'econ_cost_bps': 0.0,
                             'min_select_days': 0, 'min_capture': 0.0})
    # Small synthetic windows: short curve horizon (also keeps the purge at
    # the legacy 12 bars the window-discipline checks pin down).
    cfg['curve'] = {**cfg['curve'], 'horizon_bars': 6,
                    'entry_stride_bars': 2,
                    'sample_ks': [1, 2, 3, 6]}
    return cfg


N_SYM, N_DAYS, BPD = 20, 30, 144


def make_panel(plant=0.0, seed=0, n_sym=N_SYM, n_days=N_DAYS):
    """Synthetic (symbol, timestamp)-sorted panel.

    Features: res_zscore (AR(0.95) state), res_mom / vol_ratio / vol_noise
    (pure noise). residual_return[t] = plant * res_zscore[t-1] + noise, so
    res_zscore predicts forward residuals with strength `plant` and CAUSAL
    alignment (the feature at t predicts bars t+1..). raw_return adds a
    common market move so dollar neutrality actually matters.
    """
    r = np.random.default_rng(seed)
    n_bars = n_days * BPD
    ts = pd.date_range('2024-01-01', periods=n_bars, freq='10min')
    market = r.normal(0, 5e-4, n_bars)

    frames = []
    for i in range(n_sym):
        s = np.zeros(n_bars)
        shocks = r.normal(size=n_bars)
        for t in range(1, n_bars):
            s[t] = 0.95 * s[t - 1] + shocks[t]
        eps = r.normal(0, 1e-2, n_bars)
        res_ret = np.empty(n_bars)
        res_ret[0] = eps[0]
        res_ret[1:] = plant * s[:-1] + eps[1:]
        frames.append(pd.DataFrame({
            'timestamp': ts, 'symbol': f'S{i:02d}',
            'res_zscore': s,
            'res_mom': r.normal(size=n_bars),
            'vol_ratio': np.abs(r.normal(size=n_bars)) + 0.5,
            'vol_noise': r.normal(size=n_bars),
            'residual_return': res_ret,
            'raw_return': res_ret + market,
            'is_liquid': i < n_sym // 2,
        }))
    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values(['symbol', 'timestamp']).reset_index(drop=True)
    return data_mod.attach_targets(panel, [6])


ROLL = data_mod.Roll(
    roll_id=0,
    train_start=pd.Timestamp('2024-01-01'),
    select_start=pd.Timestamp('2024-01-19'),
    oos_start=pd.Timestamp('2024-01-25'),
    oos_end=pd.Timestamp('2024-01-30'),
)

# ---------------------------------------------------------------------------
# 1. Evaluator calibration: recovers a planted IC / reports ~0 on noise
# ---------------------------------------------------------------------------
print("--- 1. evaluator calibration (response curve) ---")
n_stamps, n_assets = 720, 40
cal_ts = pd.date_range('2024-01-01', periods=n_stamps, freq='10min')
cal_sym = [f'A{i}' for i in range(n_assets)]

for plant_c in (0.2, 0.0):
    sig_mat = rng.normal(size=(n_stamps, n_assets))
    # Residual return one bar AFTER the signal: r[t] = plant * sig[t-1] + eps
    r_mat = rng.normal(size=(n_stamps, n_assets))
    r_mat[1:] += plant_c * sig_mat[:-1]
    sig_df = pd.DataFrame({'timestamp': np.repeat(cal_ts, n_assets),
                           'symbol': np.tile(cal_sym, n_stamps),
                           'signal': sig_mat.ravel()})
    res_wide = pd.DataFrame(r_mat, index=cal_ts, columns=cal_sym)
    rc = search_mod.response_curve(sig_df, res_wide, horizon_bars=12,
                                   entry_stride=2, min_assets=10,
                                   sample_ks=[1, 2, 3, 6, 12])
    # Error bar at k=1 (the bar the effect lives on), overlap-deflated -
    # the smoothed PEAK of a one-bar effect sits on the noise plateau, so
    # significance is judged where the effect is, not where noise tops out.
    _v1 = np.asarray(rc['per_entry_at'][1], dtype=float)
    se1 = float(np.std(_v1[np.isfinite(_v1)], ddof=1)
                / np.sqrt(max(rc['n_eff'], 1.0)))
    # Expected one-bar book return: plant * E[(sig - mean) . sig] / E[gross]
    # = plant * (n-1)/(n * E|N(0,1)|) ~ plant * 1.22 for n = 40.
    expected = plant_c * (n_assets - 1) / (n_assets * np.sqrt(2 / np.pi))
    if plant_c > 0:
        check(f"calibration: planted effect {plant_c} recovered at k=1",
              abs(rc['A'][0] - expected) < 0.3 * expected,
              f"(A(1) {rc['A'][0]:+.4f} vs expected {expected:+.4f})")
        check("calibration: planted edge positive and significant",
              rc['A'][0] > 0 and rc['A'][0] / se1 > 5,
              f"(A(1) {rc['A'][0]:+.4f}, t {rc['A'][0] / se1:.1f})")
    else:
        check("calibration: zero-signal panel reads ~0 edge",
              abs(rc['A'][0]) < 4 * se1,
              f"(A(1) {rc['A'][0]:+.5f}, se {se1:.5f})")

# ---------------------------------------------------------------------------
# 2. Operator causality: truncation test over the WHOLE registry
# ---------------------------------------------------------------------------
print("--- 2. operator causality (truncation) ---")
op_panel = make_panel(plant=0.0, seed=3, n_sym=12, n_days=6)
F1, F2 = ('col', 'res_zscore'), ('col', 'vol_ratio')
W = 6

op_exprs = {}
for op, spec in gen.OPERATORS.items():
    if spec['kind'] == 'elementwise':
        op_exprs[op] = (op, F1) if spec['n_args'] == 1 else (op, F1, F2)
    elif spec['kind'] == 'rolling':
        op_exprs[op] = (op, F1, W)
    elif spec['kind'] == 'cross_sectional':
        op_exprs[op] = (op, F1)
    elif spec['kind'] == 'where':
        op_exprs[op] = ('where', ('gt', F2, 1.0), F1, ('neg', F1))

leaking = []
for op, expr in op_exprs.items():
    cand = gen.Candidate(name=f'op_{op}', family='residual_shape',
                         expression=expr)
    if not gen.truncation_check(cand, op_panel):
        leaking.append(op)
check("causality: every DSL operator passes truncation",
      len(leaking) == 0, f"leaking: {leaking}")

# a condition gate must also be causal
gated = gen.Candidate(name='gated', family='residual_shape', expression=F1,
                      conditions=(('abs_lt', ('roll_zscore', F2, W), 1.0),))
check("causality: gate conditions pass truncation",
      gen.truncation_check(gated, op_panel))

# a deliberately look-ahead operator MUST be caught
gen.OPERATORS['lead'] = {'kind': 'rolling', 'n_args': 1, 'extra': ('window',),
                         'fn': lambda g, w: g.shift(-w)}
try:
    bad = gen.Candidate(name='bad', family='residual_shape',
                        expression=('lead', F1, W))
    check("causality: look-ahead operator caught",
          not gen.truncation_check(bad, op_panel))
finally:
    del gen.OPERATORS['lead']

# ---------------------------------------------------------------------------
# 3. Window discipline: rolls, purge/embargo, boundedness
# ---------------------------------------------------------------------------
print("--- 3. window discipline ---")
roll_cfg = {'start_date': '2023-08-01', 'end_date': '2024-02-01',
            'train_months': 3, 'select_months': 1, 'oos_months': 1,
            'roll_step_months': 1}
rolls = data_mod.make_rolls(roll_cfg)
check("rolls: count for 6 months of data with 5-month windows",
      len(rolls) == 2, f"({len(rolls)} rolls)")
r0 = rolls[0]
check("rolls: contiguous boundaries",
      r0.select_start == pd.Timestamp('2023-11-01')
      and r0.oos_start == pd.Timestamp('2023-12-01')
      and r0.oos_end == pd.Timestamp('2024-01-01'))
check("rolls: no OOS beyond end_date",
      all(r.oos_end <= pd.Timestamp('2024-02-01') for r in rolls))

CFG = test_cfg()
pb = data_mod.purge_bars(CFG)
check("purge: max target lag + embargo", pb == 12, f"({pb} bars)")

disc_panel = make_panel(plant=0.0, seed=4, n_sym=12, n_days=8)
cut_end = pd.Timestamp('2024-01-05')
sliced = data_mod.slice_window(disc_panel, '2024-01-01', cut_end, pb)
bar = pd.Timedelta('10min')
check("purge: sliced window ends exactly purge bars before the boundary",
      sliced['timestamp'].max() == cut_end - (pb + 1) * bar,
      f"(last stamp {sliced['timestamp'].max()})")
# no forward target computed at any kept train stamp reaches past the boundary
last_kept = sliced['timestamp'].max()
check("purge: last kept stamp's 6-bar target ends before the boundary",
      last_kept + 6 * bar < cut_end)

# boundedness: unknown columns / windows / depth are rejected
dsl_cfg = CFG['dsl']
allowed = ['res_zscore', 'res_mom']
for label, cand in [
    ("unknown column", gen.Candidate('x', 'f', ('col', 'not_a_feature'))),
    ("window not allowed", gen.Candidate('x', 'f',
                                         ('roll_mean', ('col', 'res_zscore'), 7))),
    ("too deep", gen.Candidate('x', 'f',
        ('neg', ('neg', ('neg', ('neg', ('neg', ('col', 'res_zscore')))))))),
]:
    try:
        gen.validate_candidate(cand, allowed, dsl_cfg)
        check(f"boundedness: {label} rejected", False)
    except gen.ValidationError:
        check(f"boundedness: {label} rejected", True)

# ---------------------------------------------------------------------------
# 4. Noise in, bounded out: rank + K slots on a pure-noise panel. There is
#    deliberately NO significance gate anymore - the fixed book_size caps
#    what noise can supply, the directed floor rejects wrong-way evidence,
#    and the walk-forward is the judge.
# ---------------------------------------------------------------------------
print("--- 4. noise -> bounded, directed-only promotion ---")
t0 = time.perf_counter()
noise_panel = make_panel(plant=0.0, seed=11)
noise_ledger = search_mod.DiscoveryLedger(None)
noise_proposer = gen.RandomProposer(dsl_cfg=CFG['dsl'],
                                    mutation_prob=CFG['search']['mutation_prob'])
family_cols = data_mod.resolve_family_columns(
    [c for c in noise_panel.columns], CFG)
noise_survivors = search_mod.run_search(noise_panel, ROLL, family_cols,
                                        noise_proposer, noise_ledger, CFG)
noise_promoted = bt_mod.promote(noise_survivors, ROLL, noise_ledger, CFG)
check("noise: search ran a full budget", noise_ledger.n_trials(0) >= 10,
      f"({noise_ledger.n_trials(0)} trials)")
check("noise: promotions capped at book_size",
      len(noise_promoted) <= CFG['promotion']['book_size'],
      f"({len(noise_promoted)} promoted, "
      f"{time.perf_counter() - t0:,.1f}s)")
check("noise: every promotion made money on its test (directed)",
      all(p['select_alpha_tstat'] > 0 for p in noise_promoted))

# ---------------------------------------------------------------------------
# 5. End-to-end: planted effect found, ledger complete, reproducible,
#    promoted book trades OOS dollar-neutral
# ---------------------------------------------------------------------------
print("--- 5. end-to-end on a planted effect ---")
t0 = time.perf_counter()
planted_panel = make_panel(plant=0.002, seed=21)

ledger_a = search_mod.DiscoveryLedger(None)
proposer = gen.RandomProposer(dsl_cfg=CFG['dsl'],
                              mutation_prob=CFG['search']['mutation_prob'])
survivors_a = search_mod.run_search(planted_panel, ROLL, family_cols,
                                    proposer, ledger_a, CFG)
check("e2e: survivors found", len(survivors_a) > 0,
      f"({len(survivors_a)} survivors, {ledger_a.n_trials(0)} trials)")
check("e2e: ledger has one row per evaluation",
      ledger_a.n_trials(0) == len(ledger_a.to_frame()))

best = max(survivors_a, key=lambda s: s['reward'])
check("e2e: best survivor detects the planted effect",
      best['metrics_select']['alpha_tstat'] > 3
      and 'res_zscore' in gen.candidate_columns(best['candidate']),
      f"(t {best['metrics_select']['alpha_tstat']:.1f}, "
      f"cols {sorted(gen.candidate_columns(best['candidate']))})")

# reproducibility: identical run -> identical survivors and rewards
ledger_b = search_mod.DiscoveryLedger(None)
survivors_b = search_mod.run_search(planted_panel, ROLL, family_cols,
                                    gen.RandomProposer(
                                        dsl_cfg=CFG['dsl'],
                                        mutation_prob=CFG['search']['mutation_prob']),
                                    ledger_b, CFG)
check("e2e: reproducible (same seed -> same survivors + rewards)",
      [s['candidate'].hash for s in survivors_a]
      == [s['candidate'].hash for s in survivors_b]
      and np.allclose([s['reward'] for s in survivors_a],
                      [s['reward'] for s in survivors_b]))

promoted = bt_mod.promote(survivors_a, ROLL, ledger_a, CFG)
check("e2e: planted effect promoted through the gates", len(promoted) >= 1,
      f"({len(promoted)} promoted)")
check("e2e: ledger promoted flags set",
      len(ledger_a.to_frame().query('promoted')) == len(promoted))

# Directional filter (regression): the verdict is DIRECTED, never |t|. A
# formula whose test ran backwards is rejected, not flipped. (an earlier
# gate once used abs(ic_tstat) - two prod signals promoted while trading
# the WRONG way on the hold-out window, at directed t of -5.2 and -3.8.)
check("directional: every promotion made money in its committed direction",
      all(p['select_alpha_tstat'] > 0 for p in promoted),
      f"({len(promoted)} promoted)")

# Flip the winner's select profile: same magnitude, opposite sign - abs()
# would still pass, the directed gate must reject.
reversed_s = copy.deepcopy(best)
# metrics_select IS profile_select[best_lag] (same object), so dedupe by id -
# flipping the same dict twice would cancel out.
_seen = set()
for m in (list(reversed_s['profile_select'].values())
          + [reversed_s['metrics_select']]):
    if id(m) in _seen:
        continue
    _seen.add(id(m))
    for k in ('alpha_mean', 'alpha_tstat', 'alpha_ir',
              'rank_ic_mean', 'rank_ic_tstat'):
        if m.get(k) is not None:
            m[k] = -m[k]
# ... and the response curve (a truly reversed hold-out reverses its curve).
if reversed_s.get('curve'):
    _c = dict(reversed_s['curve'])
    _c['a0'] = -_c['a0'] if _c.get('a0') is not None else None
    _c['A'] = [None if a is None else -a for a in _c.get('A', [])]
    if _c.get('median_peak') is not None:
        _c['median_peak'] = -_c['median_peak']
    reversed_s['curve'] = _c
reversed_promoted = bt_mod.promote([reversed_s], ROLL, ledger_a, CFG)
check("directional: sign-reversed hold-out signal is NOT promoted",
      len(reversed_promoted) == 0,
      f"(|t| {abs(best['metrics_select']['alpha_tstat']):.1f} but reversed -> "
      f"{len(reversed_promoted)} promoted)")

# Effective persistence: capture prices min(alpha half-life, position life
# 1/turnover). Turnover is PER BAR, so 1/turnover = bars until the signal has
# fully reshuffled itself (regression: lag/turnover mixed units and credited
# a 0.1/bar churner with 1,440 bars of persistence instead of 10).
_ep = search_mod.effective_persistence_bars
check("persistence: full-churn signal (turnover 1) -> 1 bar",
      _ep(2016.0, 1.0) == 1.0)
check("persistence: slow churn (0.04/bar) -> 25 bars, beats half-life 96",
      _ep(96.0, 0.04) == 25.0)
check("persistence: half-life binds when churn is slower still",
      _ep(96.0, 0.005) == 96.0)
check("persistence: missing turnover -> half-life alone",
      _ep(288.0, float('nan')) == 288.0 and _ep(288.0, None) == 288.0)
check("persistence: hyperactive turnover clipped at 2 -> 0.5 bars",
      _ep(2016.0, 5.0) == 0.5)
# Turnover diagnostic: a property of the signal alone (0 = never retrades,
# 1 = fully replaced each bar). Ledger-only; never touches reward, promotion,
# or the walk-forward.
_ts = pd.date_range('2024-01-01', periods=4, freq='10min')
_static = pd.DataFrame([{'timestamp': t, 'symbol': s, 'signal': v}
                        for t in _ts
                        for s, v in [('A', 1.0), ('B', -1.0)]])
check("turnover: constant signal -> ~0",
      abs(search_mod.signal_turnover(_static)) < 1e-9,
      f"({search_mod.signal_turnover(_static):.4f})")
_flip = pd.DataFrame([{'timestamp': t, 'symbol': s, 'signal': v * (1 if i % 2 == 0 else -1)}
                      for i, t in enumerate(_ts)
                      for s, v in [('A', 1.0), ('B', -1.0)]])
check("turnover: sign-flipping every bar -> ~1 (full replacement)",
      abs(search_mod.signal_turnover(_flip) - 1.0) < 1e-9,
      f"({search_mod.signal_turnover(_flip):.4f})")
_led_df = ledger_a.to_frame()
# Coverage-floor rejects are recorded WITHOUT turnover (their evaluation is
# cut short by design) - the finiteness guarantee applies to fully-evaluated
# rows only.
_full = _led_df[_led_df['reward'] != CFG['search']['sparse_reward']]
check("turnover: recorded in the ledger for every evaluated candidate",
      'turnover' in _full.columns
      and _full['turnover'].notna().all()
      and (_full['turnover'] >= 0).all(),
      f"(range {_full['turnover'].min():.3f}-{_full['turnover'].max():.3f}, "
      f"{len(_led_df) - len(_full)} coverage rejects excluded)")
check("turnover: survivors carry it in-memory too",
      all('turnover' in s and np.isfinite(s['turnover']) for s in survivors_a))

# PAYS-FOR-ITSELF (filter 3) on the curve: the round-trip cost is judged
# against the curve at its own optimum. A round trip larger than any
# measured edge must reject everything; NaN turnover never blocks on the
# curve path (turnover only enters capture, which fails open).
_base = bt_mod.promote(survivors_a, ROLL, ledger_a, CFG)
assert _base, "e2e produced no promotions to gate"
_ec_cfg = copy.deepcopy(CFG)
# The planted synthetic edge is huge (~thousands of bp per bet), so the
# unambiguous kill-cost is absurd on purpose.
_ec_cfg['promotion']['econ_cost_bps'] = 1e6
check("economics: a round-trip cost above any measured edge rejects all",
      bt_mod.promote(survivors_a, ROLL, ledger_a, _ec_cfg) == [],
      f"(base {len(_base)} -> 0 at 1e6bps)")
check("economics: promotions carry a positive net rate at cost 0",
      all(p['econ_margin'] > 0 for p in _base))
_no_tv = [{**s, 'turnover': float('nan')} for s in survivors_a]
_openq = bt_mod.promote(_no_tv, ROLL, ledger_a, CFG)
check("economics: NaN turnover never blocks (capture fails open)",
      len(_openq) == len(_base),
      f"({len(_openq)} promoted with NaN turnover vs {len(_base)} baseline)")

# survivor carry-over: the next roll is seeded with this roll's survivors,
# they re-earn survival on the new windows, and the N-consecutive-rolls
# persistence gate becomes satisfiable (without seeding it never can be).
ROLL_B = data_mod.Roll(
    roll_id=1,
    train_start=pd.Timestamp('2024-01-04'),
    select_start=pd.Timestamp('2024-01-22'),
    oos_start=pd.Timestamp('2024-01-27'),
    oos_end=pd.Timestamp('2024-01-30'),
)
seed_cands = [s['candidate'] for s in survivors_a]
survivors_roll1 = search_mod.run_search(planted_panel, ROLL_B, family_cols,
                                        gen.RandomProposer(
                                            dsl_cfg=CFG['dsl'],
                                            mutation_prob=CFG['search']['mutation_prob']),
                                        ledger_a, CFG,
                                        seed_candidates=seed_cands)
carried = ({s['candidate'].hash for s in survivors_roll1}
           & {c.hash for c in seed_cands})
check("carry-over: seeded survivors re-survive the next roll",
      len(carried) >= 1, f"({len(carried)} carried)")
check("carry-over: seeds recorded in the ledger (generation -1)",
      (ledger_a.to_frame().query('roll_id == 1 and generation == -1')
       ['cand_hash'].nunique()) >= len(carried))
check("carry-over: ledger survivor_candidates round-trips",
      {c.hash for c in ledger_a.survivor_candidates(0)}
      == {s['candidate'].hash for s in survivors_a})

# Discovery is purely statistical: promotions carry everything the
# walk-forward (the only money judge) needs to trade them.
if promoted:
    p0 = promoted[0]
    check("e2e: promotion carries verdict lag + half-life + capture + "
          "economics",
          'half_life_bars' in p0 and 'capture' in p0
          and 'select_lag' in p0 and 'test_days' in p0
          and 'econ_margin' in p0
          and 0.0 < p0['capture'] <= 1.0,
          f"(hl {p0['half_life_bars']:.0f}b, capture {p0['capture']:.2f}, "
          f"lag {p0['select_lag']}, test days {p0['test_days']}, "
          f"margin {p0['econ_margin']:.2e})")
print(f"(end-to-end block: {time.perf_counter() - t0:,.1f}s)")

# ---------------------------------------------------------------------------
# 5b. The response curve finds each effect's true horizon: a fast impulse
#     peaks early with a short half-life; a slow impulse (alpha arriving at
#     bar k=20) ramps until ~k and peaks late. No lag menu anywhere.
# ---------------------------------------------------------------------------
print("--- 5b. response curve finds each effect's true horizon ---")
CFG_ML = test_cfg()
CFG_ML['curve'] = {**CFG_ML['curve'], 'horizon_bars': 36,
                   'sample_ks': [1, 2, 3, 6, 12, 24, 36]}
CFG_ML['search'] = {**CFG_ML['search'], 'n_generations': 0}  # seeds only


def make_impulse_panel(plant=0.01, seed=33, k=20):
    """vol_noise (WHITE noise) predicts the residual exactly k bars ahead:
    a one-bar impulse of alpha at horizon k. k=1 -> all the alpha sits
    inside the 6-bar target (fast); k=20 -> only the 36-bar target sees it
    (slow). White noise, not an AR state, so the horizon separation is
    exact."""
    r = np.random.default_rng(seed)
    n_bars = N_DAYS * BPD
    ts = pd.date_range('2024-01-01', periods=n_bars, freq='10min')
    frames = []
    for i in range(N_SYM):
        w = r.normal(size=n_bars)
        res_ret = r.normal(0, 1e-2, n_bars)
        res_ret[k:] += plant * w[:-k]
        frames.append(pd.DataFrame({
            'timestamp': ts, 'symbol': f'S{i:02d}',
            'res_zscore': r.normal(size=n_bars),
            'res_mom': r.normal(size=n_bars),
            'vol_ratio': np.abs(r.normal(size=n_bars)) + 0.5,
            'vol_noise': w,
            'residual_return': res_ret,
            'raw_return': res_ret,
            'is_liquid': i < N_SYM // 2,
        }))
    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values(['symbol', 'timestamp']).reset_index(drop=True)
    return data_mod.attach_targets(panel, [6, 36])


fast_seed = gen.Candidate(name='fast_probe', family='residual_shape',
                          expression=('col', 'res_zscore'))
wn_seed = gen.Candidate(name='wn_probe', family='volatility_regime',
                        expression=('col', 'vol_noise'))

fast_panel_ml = make_impulse_panel(plant=0.01, seed=31, k=1)
led_fast = search_mod.DiscoveryLedger(None)
pop_fast = search_mod.run_search(
    fast_panel_ml, ROLL, family_cols,
    gen.RandomProposer(dsl_cfg=CFG_ML['dsl'], mutation_prob=0.6),
    led_fast, CFG_ML, seed_candidates=[wn_seed])
e_fast = next(s for s in pop_fast if s['candidate'].hash == wn_seed.hash)
check("curve: fast impulse -> curves measured on train and test",
      e_fast['curve'] is not None and e_fast['curve_train'] is not None)
check("curve: fast effect -> short fitted half-life",
      e_fast['half_life_bars'] <= 12,
      f"(half-life {e_fast['half_life_bars']:.0f} bars)")

slow_panel_ml = make_impulse_panel()
led_slow = search_mod.DiscoveryLedger(None)
pop_slow = search_mod.run_search(
    slow_panel_ml, ROLL, family_cols,
    gen.RandomProposer(dsl_cfg=CFG_ML['dsl'], mutation_prob=0.6),
    led_slow, CFG_ML, seed_candidates=[wn_seed])
e_slow = next(s for s in pop_slow if s['candidate'].hash == wn_seed.hash)
# The slow effect's alpha only arrives at bar 20: its curve ramps until
# ~20+ and its peak sits far beyond the fast effect's.
check("curve: slow impulse peaks late, fast peaks early",
      e_slow['curve_train']['peak_k'] >= 15
      and e_fast['curve_train']['peak_k'] < e_slow['curve_train']['peak_k'],
      f"(fast peak {e_fast['curve_train']['peak_k']}, "
      f"slow peak {e_slow['curve_train']['peak_k']})")
check("curve: slow effect is real on its held-out curve",
      e_slow['curve']['a0'] > 0
      and e_slow['metrics_select']['alpha_mean'] > 0,
      f"(test a0 {e_slow['curve']['a0']:.2e})")
check("curve: slow effect outlives the fast one",
      e_slow['half_life_bars'] > e_fast['half_life_bars'],
      f"(slow {e_slow['half_life_bars']:.0f} vs fast "
      f"{e_fast['half_life_bars']:.0f} bars)")
check("curve: ledger records peak + half-life + both curves in the json",
      led_slow.to_frame()['half_life_bars'].notna().all()
      and all('curve_train' in pj and '"curve"' in pj
              for pj in led_slow.to_frame()['profile_json']))

# ---------------------------------------------------------------------------
# 5b2. Sign-agnostic capture: a strongly NEGATIVE-IC effect must be found
#      and traded with direction -1 (the flip is fixed on TRAIN; SELECT is
#      then scored on the flipped signal, so its IC comes out positive).
# ---------------------------------------------------------------------------
print("--- 5b2. negative-IC effect captured with flipped direction ---")
neg_panel = data_mod.attach_targets(make_panel(plant=-0.002, seed=41), [36])
led_neg = search_mod.DiscoveryLedger(None)
pop_neg = search_mod.run_search(
    neg_panel, ROLL, family_cols,
    gen.RandomProposer(dsl_cfg=CFG_ML['dsl'], mutation_prob=0.6),
    led_neg, CFG_ML, seed_candidates=[fast_seed])
e_neg = next(s for s in pop_neg if s['candidate'].hash == fast_seed.hash)
check("negative effect: traded direction is -1", e_neg['direction'] == -1)
check("negative effect: select IC strongly positive AFTER the flip",
      e_neg['metrics_select']['alpha_tstat'] > 3,
      f"(t {e_neg['metrics_select']['alpha_tstat']:.1f})")
check("negative effect: train IC positive after the flip too",
      e_neg['metrics_train']['alpha_mean'] > 0)

# ---------------------------------------------------------------------------
# 5c. Coverage floor: a feature alive on only ~2 of the train window's 18
#     days yields a curve with almost no entry days - a lottery ticket
#     promotion can never accept. It is recorded with the penalty reward
#     (bandit + failure memory learn) but never enters the population.
#     (It has thousands of non-NaN VALUES on its live days, so it passes
#     the upstream per-roll feature filter - this floor is what catches it.)
# ---------------------------------------------------------------------------
print("--- 5c. train coverage floor ---")
cov_cfg = test_cfg()
cov_cfg['search'] = {**cov_cfg['search'], 'n_generations': 0}
sparse_panel = planted_panel.copy()
sparse_panel['res_sparse'] = sparse_panel['res_zscore'].where(
    sparse_panel['timestamp'] < pd.Timestamp('2024-01-03'))
family_cols_cov = data_mod.resolve_family_columns(
    list(sparse_panel.columns), cov_cfg)
dense_probe = gen.Candidate(name='dense_probe', family='residual_shape',
                            expression=('col', 'res_zscore'))
sparse_probe = gen.Candidate(name='sparse_probe', family='residual_shape',
                             expression=('col', 'res_sparse'))
led_cov = search_mod.DiscoveryLedger(None)
pop_cov = search_mod.run_search(
    sparse_panel, ROLL, family_cols_cov,
    gen.RandomProposer(dsl_cfg=cov_cfg['dsl'], mutation_prob=0.6),
    led_cov, cov_cfg, seed_candidates=[sparse_probe, dense_probe])
_cov_hashes = {s['candidate'].hash for s in pop_cov}
_cov_row = led_cov.to_frame().query('cand_hash == @sparse_probe.hash')
check("coverage: sparse candidate ledger-recorded with the penalty reward",
      len(_cov_row) == 1
      and _cov_row['reward'].iloc[0] == cov_cfg['search']['sparse_reward'])
check("coverage: sparse candidate never enters the population",
      sparse_probe.hash not in _cov_hashes)
check("coverage: dense candidate unaffected",
      dense_probe.hash in _cov_hashes)

# ---------------------------------------------------------------------------
# 5d. Unit checks: persistence weight, reward, capture floor.
#     (Curve fitting - a0/half-life/peak/reversal - is unit-checked in
#     tests/curve_checks.py.)
# ---------------------------------------------------------------------------
print("--- 5d. reward/persistence unit checks ---")

# persistence weight 1/(1 + phi/rate): matches the closed form at any rate,
# monotone in persistence, and crushes alpha much faster than the fill rate.
rate = search_mod.trade_rate_per_bar()
w_slow = search_mod.persistence_weight(1008, rate)
w_fast = search_mod.persistence_weight(3, rate)
check("persistence weight: closed form, fast crushed, monotone",
      abs(w_slow - 1 / (1 + math.log(2) / 1008 / rate)) < 1e-12
      and abs(w_fast - 1 / (1 + math.log(2) / 3 / rate)) < 1e-12
      and w_fast < 0.5 * w_slow,
      f"(rate {rate:.4f}/bar: hl 1008 -> {w_slow:.3f}, hl 3 -> {w_fast:.3f})")

# net-rate reward: exactly TWO terms (net_rate + similarity - anything more
# is a hand-tuned constant that can silently dominate alpha); the net rate
# is the same number promotion ranks by, so a better rate scores higher and
# similarity to the kept pool lowers the reward.
t_hi = search_mod.reward_terms(2e-6, 0.0)
check("reward: exactly two terms (net_rate, similarity)",
      set(t_hi) == {'net_rate', 'similarity'}
      and set(get('discovery.reward.weights')) == {'net_rate', 'similarity'})
_r_hi, _ = search_mod.compute_reward(2e-6, 0.0)
_r_lo, _ = search_mod.compute_reward(1e-6, 0.0)
_r_dup, _ = search_mod.compute_reward(2e-6, 1.0)
check("reward: better net rate scores higher, similarity lowers the reward",
      _r_hi > _r_lo and _r_dup < _r_hi)
check("reward: non-finite net rate floored to zero, never NaN",
      search_mod.reward_terms(float('nan'), 0.0)['net_rate'] == 0.0)

# capture floor: with min_capture above the fast fit's capture,
# nothing may promote (the gate that replaced the PnL scoreboard).
cap_cfg = copy.deepcopy(CFG)
cap_cfg['promotion']['min_capture'] = 0.99
check("capture floor: min_capture blocks fast-alpha promotion",
      bt_mod.promote(survivors_a, ROLL, ledger_a, cap_cfg) == [])

# ---------------------------------------------------------------------------
# 5e. diagnostic blend + output-correlation diversity
# ---------------------------------------------------------------------------
print("--- 5e. diagnostic blend / output-correlation diversity ---")

# #1 nonlinearity + regime helpers: a U-shaped decile curve scores high even
# with ~0 monotonic IC; a flat one scores ~0.
check("diag: decile nonlinearity catches U-shape, ignores flat",
      data_mod._decile_nonlinearity([0.01, -0.01, -0.02, -0.01, 0.01]) > 0.02
      and data_mod._decile_nonlinearity([0.0, 0.0, 0.0]) == 0.0)
check("diag: regime spread = max high-vs-low gap",
      abs(data_mod._regime_spread({'r': {'high': 0.03, 'low': -0.01}}) - 0.04) < 1e-9
      and data_mod._regime_spread({'r': {'high': None, 'low': 0.0}}) == 0.0)
_r01 = data_mod._rank01({'a': 1.0, 'b': 3.0, 'c': 2.0})
check("diag: rank01 percentile-ranks within the group",
      _r01['b'] == 1.0 and _r01['a'] < _r01['c'] < _r01['b'])

# candidate_subtrees still feeds the over-mined-blocks prompt hint
_C = gen.Candidate('c', 'f', ('mul', ('square', ('col', 'res_zscore')),
                              ('col', 'res_mom')))
check("subtrees: candidate_subtrees excludes bare leaves",
      all(gen._depth(s) >= 2 for s in gen.candidate_subtrees(_C)))

# select_survivors: the SINGLE diversity guard is output correlation - a
# clone that outputs the same ranking is dropped (best reward wins), an
# uncorrelated candidate is kept.
_A = gen.Candidate('a', 'f', ('neg', ('square', ('col', 'res_zscore'))))
_D = gen.Candidate('d', 'f', ('neg', ('square', ('col', 'res_mom'))))
_ts = pd.date_range('2024-01-01', periods=40, freq='1h')
def _sig_panel(vals):
    return pd.DataFrame({'timestamp': np.repeat(_ts, 3),
                         'symbol': np.tile(['X', 'Y', 'Z'], 40),
                         'signal': vals})
rng_s = np.random.default_rng(3)
v1 = rng_s.normal(size=120)
pop_div = [
    {'candidate': _A, 'reward': 2.0, 'signal_train': _sig_panel(v1)},
    {'candidate': _C, 'reward': 1.0, 'signal_train': _sig_panel(v1 * 2)},
    {'candidate': _D, 'reward': 0.5,
     'signal_train': _sig_panel(rng_s.normal(size=120))},
]
kept_hashes = {s['candidate'].hash for s in
               search_mod.select_survivors(pop_div, 3, 0.8)}
check("select_survivors: output-correlated clone dropped, best reward wins",
      _A.hash in kept_hashes and _C.hash not in kept_hashes
      and _D.hash in kept_hashes)

# ---------------------------------------------------------------------------
# 6. Proposer providers + cost tracking (no API calls: clients are lazy)
# ---------------------------------------------------------------------------
print("--- 6. providers + cost tracking ---")


class _BrokenSDKProposer(gen.GeminiProposer):
    """Simulates a missing SDK: _complete raises ImportError."""
    def __init__(self):
        super().__init__(llm_cfg={'candidates_per_call': 8})
    provider = 'gemini'

    def _prompt(self, *args, **kwargs):
        return 'x'

    def _complete(self, prompt):
        raise ImportError("cannot import name 'genai' from 'google'")


# parent scores + failure memory reach the prompt (guided evolution)
import json as _json
_pp = gen.GeminiProposer.__new__(gen.GeminiProposer)
_pp.llm_cfg = {'candidates_per_call': 8}
_pp.dsl_cfg = get('discovery.dsl')
_pa = gen.Candidate('a', 'residual_shape', ('col', 'res_zscore'))
_pb = gen.Candidate('b', 'residual_shape', ('neg', ('col', 'res_zscore')))
_payload = _json.loads(_pp._prompt(
    4, 'residual_shape',
    {'target': 'fwd_36b', 'features': {}, 'top_by_family': {}}, [_pb, _pa],
    {'residual_shape': ['res_zscore']},
    parent_scores={_pa.hash: {'reward': 2.1, 'alpha_tstat': 4.3, 'half_life_bars': 288},
                   _pb.hash: {'reward': 0.1, 'alpha_tstat': 0.4, 'half_life_bars': 6}},
    failures=[{'expression': ['col', 'res_mom'], 'conditions': [], 'reward': -0.5}]))
check("prompt: parents carry scores, ranked best-first",
      _payload['current_parents'][0]['score']['reward'] == 2.1
      and _payload['current_parents'][1]['score']['reward'] == 0.1)
check("prompt: failure memory reaches avoid_these",
      _payload['avoid_these'] == [{'expression': ['col', 'res_mom'],
                                   'conditions': [], 'reward': -0.5}])
check("prompt: columns carry descriptions (data dictionary)",
      _payload['columns'].get('res_zscore', '') != '')

try:
    _BrokenSDKProposer().propose(4, 'residual_shape', {}, [], {}, rng)
    check("providers: missing SDK aborts the run (fail fast)", False,
          "(no exception raised)")
except SystemExit as e:
    check("providers: missing SDK aborts the run (fail fast)",
          'uv add google-genai' in str(e), f"({e})")
except Exception as e:
    check("providers: missing SDK aborts the run (fail fast)", False,
          f"(wrong exception: {type(e).__name__}: {e})")


class _RetiredModelProposer(gen.GeminiProposer):
    """Simulates a retired/mis-named model: the API 404s on every call
    (Google retired gemini-2.5-flash mid-2026 exactly this way). Permanent,
    not transient - must abort, not degrade to empty batches per family."""
    def __init__(self):
        super().__init__(llm_cfg={'candidates_per_call': 8,
                                  'model': {'gemini': 'gemini-2.5-flash'}})

    def _prompt(self, *args, **kwargs):
        return 'x'

    def _complete(self, prompt):
        raise Exception(
            "404 NOT_FOUND. {'error': {'code': 404, 'message': 'This model "
            "models/gemini-2.5-flash is no longer available.'}}")


try:
    _RetiredModelProposer().propose(4, 'residual_shape', {}, [], {}, rng)
    check("providers: retired model (404) aborts the run (fail fast)", False,
          "(no exception raised)")
except SystemExit as e:
    check("providers: retired model (404) aborts the run (fail fast)",
          'discovery.llm.model' in str(e), f"({str(e)[:80]}...)")
except Exception as e:
    check("providers: retired model (404) aborts the run (fail fast)", False,
          f"(wrong exception: {type(e).__name__}: {e})")

class _DepletedCreditsProposer(gen.GeminiProposer):
    """Simulates prepay credits running out mid-run: 429 RESOURCE_EXHAUSTED
    with a BILLING message. Permanent for the run - must abort with resume
    guidance (a live run skipped every batch for hours on this)."""
    def __init__(self):
        super().__init__(llm_cfg={'candidates_per_call': 8})

    def _prompt(self, *args, **kwargs):
        return 'x'

    def _complete(self, prompt):
        raise Exception(
            "429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': "
            "'Your prepayment credits are depleted. Please go to AI Studio "
            "to manage your project and billing.', "
            "'status': 'RESOURCE_EXHAUSTED'}}")


try:
    _DepletedCreditsProposer().propose(4, 'residual_shape', {}, [], {}, rng)
    check("providers: depleted credits (429 billing) aborts with resume hint",
          False, "(no exception raised)")
except SystemExit as e:
    check("providers: depleted credits (429 billing) aborts with resume hint",
          '--resume' in str(e), f"({str(e)[:80]}...)")
except Exception as e:
    check("providers: depleted credits (429 billing) aborts with resume hint",
          False, f"(wrong exception: {type(e).__name__}: {e})")


class _RateLimitedProposer(gen.GeminiProposer):
    """An ORDINARY 429 rate limit (no billing language) is transient: the
    proposer must keep the retry -> skip-batch degradation, never abort."""
    def __init__(self):
        super().__init__(llm_cfg={'candidates_per_call': 8})

    def _prompt(self, *args, **kwargs):
        return 'x'

    def _complete(self, prompt):
        raise Exception(
            "429 RESOURCE_EXHAUSTED. Quota exceeded for metric "
            "generate_requests_per_minute_per_project. Retry in 12s.")


check("providers: plain 429 rate limit degrades to an empty batch (no abort)",
      _RateLimitedProposer().propose(4, 'residual_shape', {}, [], {}, rng)
      == [])


class _FlakyProposer(gen.GeminiProposer):
    """First call truncates INSIDE the first object (nothing salvageable);
    the retry returns a clean array."""
    def __init__(self, fail_times=1):
        super().__init__(llm_cfg={'candidates_per_call': 8})
        self._fails_left = fail_times

    def _prompt(self, *args, **kwargs):
        return 'x'

    def _complete(self, prompt):
        if self._fails_left > 0:
            self._fails_left -= 1
            return '[ {"family": "residual_shape", "rationale": "cut mid-str'
        return ('[{"family": "residual_shape", '
                '"expression": ["col", "res_zscore"], "conditions": []}]')


flaky = _FlakyProposer(fail_times=1)
got = flaky.propose(4, 'residual_shape', {}, [], {}, rng)
check("providers: unusable response retried once and recovered",
      len(got) == 1 and flaky.usage['calls'] == 2,
      f"({len(got)} candidates from {flaky.usage['calls']} attempts)")
dead = _FlakyProposer(fail_times=2)
check("providers: unusable twice -> empty batch, no exception",
      dead.propose(4, 'residual_shape', {}, [], {}, rng) == []
      and dead.usage['calls'] == 2)


# Parallel proposals: within a generation the per-family API calls run
# concurrently (they share one immutable snapshot); scoring stays sequential.
# Verify the pool actually fans out, results are collected per family, and
# the locked usage counters stay exact under concurrency.
class _ParallelProposer(gen.GeminiProposer):
    """Records the thread of each call; returns one distinct candidate per
    family (column varies so hashes differ)."""
    def __init__(self):
        super().__init__(llm_cfg={'candidates_per_call': 8})
        self.threads = set()

    def _prompt(self, n, family, *args, **kwargs):
        return family   # _complete keys its response off the family

    def _complete(self, prompt):
        import threading as _th, time as _t
        self.threads.add(_th.current_thread().name)
        _t.sleep(0.05)          # force overlap so the pool must use >1 thread
        col = {'residual_shape': 'res_zscore',
               'volatility_regime': 'vol_ratio'}.get(prompt, 'res_zscore')
        return (f'[{{"family": "{prompt}", '
                f'"expression": ["col", "{col}"], "conditions": []}}]')


par = _ParallelProposer()
par_cfg = copy.deepcopy(CFG)
par_cfg['llm']['parallel_requests'] = 8
par_cfg['search'].update({'n_generations': 1, 'batch_size': 8})
par_ledger = search_mod.DiscoveryLedger(None)
par_pop = search_mod.run_search(make_panel(plant=0.002, seed=33), ROLL,
                                family_cols, par, par_ledger, par_cfg)
check("parallel: per-family proposals fan out across threads",
      len(par.threads) >= 2, f"({len(par.threads)} threads)")
check("parallel: both families' candidates evaluated",
      par_ledger.n_trials(0) >= 2
      and {s['candidate'].family for s in par_pop}
      == {'residual_shape', 'volatility_regime'},
      f"({par_ledger.n_trials(0)} trials)")
check("parallel: locked usage counters exact under concurrency",
      par.usage['calls'] == len(family_cols),
      f"({par.usage['calls']} calls for {len(family_cols)} families)")

# OpenAI-compatible providers (OpenRouter / xAI): chat-completions parsing,
# usage accounting, and HTTP errors carrying .code for the fail-fast rules.
import requests as _rq


class _FakeHTTPResp:
    def __init__(self, status=200, payload=None, text=''):
        self.status_code = status
        self._payload = payload
        self.text = text

    def json(self):
        return self._payload


_orig_post = _rq.post
try:
    _rq.post = lambda *a, **k: _FakeHTTPResp(payload={
        'choices': [{'message': {'content':
            '[{"family": "residual_shape", '
            '"expression": ["col", "res_zscore"], "conditions": []}]'}}],
        'usage': {'prompt_tokens': 100, 'completion_tokens': 20}})
    _orp = gen.make_proposer('openrouter')
    _got = _orp.propose(4, 'residual_shape', {}, [], {}, rng)
    check("openai-compat: chat completion parsed + usage counted",
          len(_got) == 1 and _orp.usage['input_tokens'] == 100
          and _orp.usage['output_tokens'] == 20 and _orp.usage['calls'] == 1)

    _rq.post = lambda *a, **k: _FakeHTTPResp(
        status=429, text="Your prepayment credits are depleted; check billing")
    try:
        gen.make_proposer('xai').propose(4, 'residual_shape', {}, [], {}, rng)
        check("openai-compat: 429 billing aborts via .code fail-fast", False,
              "(no exception raised)")
    except SystemExit as e:
        check("openai-compat: 429 billing aborts via .code fail-fast",
              '--resume' in str(e))
finally:
    _rq.post = _orig_post

check("providers: 'openrouter'/'xai' -> OpenAI-compatible proposers",
      isinstance(gen.make_proposer('openrouter'), gen.OpenRouterProposer)
      and isinstance(gen.make_proposer('xai'), gen.XAIProposer)
      and gen.make_proposer('openrouter').model
      == get('discovery.llm.model')['openrouter'])

check("providers: 'random' -> RandomProposer",
      isinstance(gen.make_proposer('random'), gen.RandomProposer))
check("providers: explicit 'gemini' -> GeminiProposer",
      isinstance(gen.make_proposer('gemini'), gen.GeminiProposer))
check("providers: explicit 'anthropic' -> AnthropicProposer",
      isinstance(gen.make_proposer('anthropic'), gen.AnthropicProposer))
llm_kind = gen.make_proposer('llm')
check("providers: 'llm' resolves provider from config",
      llm_kind.provider == get('discovery.llm.provider'),
      f"({llm_kind.provider})")
check("providers: per-provider model resolution",
      gen.make_proposer('gemini').model
      == get('discovery.llm.model')['gemini']
      and gen.make_proposer('anthropic').model
      == get('discovery.llm.model')['anthropic'])

check("usage: non-API proposer reports zero usage",
      gen.RandomProposer().usage_snapshot()
      == {'calls': 0, 'input_tokens': 0, 'output_tokens': 0})
gp = gen.make_proposer('gemini')
gp.usage.update({'calls': 3, 'input_tokens': 2_000_000,
                 'output_tokens': 500_000})
snap = gp.usage_snapshot()
snap['calls'] = 99
check("usage: snapshot is a copy, not the live accumulator",
      gp.usage['calls'] == 3)

priced_cfg = {'price_per_mtok': {'gemini': {'input': 1.0, 'output': 4.0}}}
cost = gen.estimate_cost_usd(gp.usage, priced_cfg, 'gemini')
check("cost: estimate = tokens x configured $/Mtok",
      cost is not None and abs(cost - (2.0 * 1.0 + 0.5 * 4.0)) < 1e-9,
      f"(${cost})")
check("cost: unpriced provider -> None (never a guessed price)",
      gen.estimate_cost_usd(gp.usage, {'price_per_mtok': {
          'gemini': {'input': None, 'output': None}}}, 'gemini') is None
      and gen.estimate_cost_usd(gp.usage, {}, 'gemini') is None)

# .env key loading (secrets live in the gitignored repo-root .env; the LLM
# key name is GENERIC - discovery.llm.key_name - so switching providers never
# renames anything)
import tempfile
from config import load_env_key
check("env: llm key name is the generic config value",
      isinstance(get('discovery.llm.key_name'), str)
      and get('discovery.llm.key_name') == 'LLM_KEY')
with tempfile.NamedTemporaryFile('w', suffix='.env', delete=False) as fh:
    fh.write('# comment\nLLM_KEY = g-123 \n\nCOINGECKO_KEY="CG-x"\n'
             'BROKEN LINE\n')
    env_path = Path(fh.name)
try:
    check("env: value read and stripped",
          gen.load_api_key('LLM_KEY', env_path) == 'g-123')
    check("env: quotes stripped (COINGECKO_KEY as used by etl/marketcap)",
          load_env_key('COINGECKO_KEY', env_path) == 'CG-x')
    check("env: missing name -> None from load_env_key",
          load_env_key('NOPE_KEY', env_path) is None)
    try:
        gen.load_api_key('OTHER_KEY', env_path)
        check("env: proposer key missing raises with instructions", False)
    except RuntimeError as e:
        check("env: proposer key missing raises with instructions",
              'OTHER_KEY' in str(e))
    try:
        gen.load_api_key('LLM_KEY', env_path.with_suffix('.nope'))
        check("env: missing file raises with instructions", False)
    except RuntimeError as e:
        check("env: missing file raises with instructions",
              'LLM_KEY' in str(e))
finally:
    env_path.unlink()

# LLM response parsing: plain array, fenced markdown, wrapped object
check("parse: plain JSON array",
      gen._parse_json_array('[{"a": 1}]') == [{'a': 1}])
check("parse: markdown-fenced array",
      gen._parse_json_array('```json\n[{"a": 1}]\n```') == [{'a': 1}])
check("parse: prose around the array",
      gen._parse_json_array('Here you go:\n[{"a": 1}]\nEnjoy!') == [{'a': 1}])
check("parse: object-wrapped array",
      gen._parse_json_array('{"candidates": [{"a": 1}]}') == [{'a': 1}])
# max_tokens truncation: the complete prefix is salvaged, not the whole
# batch dropped
check("parse: truncated array salvages complete prefix",
      gen._parse_json_array('[{"a": 1}, {"b": [2, {"c": 3}]}, {"d": "trunc')
      == [{'a': 1}, {'b': [2, {'c': 3}]}])
check("parse: truncation inside a string (with escapes) salvaged",
      gen._parse_json_array('[{"a": "x\\"y"}, {"b": "cut }he')
      == [{'a': 'x"y'}])
try:
    gen._parse_json_array('no array here')
    check("parse: garbage raises", False)
except ValueError:
    check("parse: garbage raises", True)
try:
    gen._parse_json_array('[{"a": tru')
    check("parse: truncated with NO complete object raises", False)
except ValueError:
    check("parse: truncated with NO complete object raises", True)

# ---------------------------------------------------------------------------
print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURES: {FAILURES}")
    sys.exit(1)
print("ALL DISCOVERY CHECKS PASSED")
