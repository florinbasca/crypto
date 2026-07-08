"""
Synthetic-data checks for the agentic signal discovery engine
(research/signals/agent/). No database required.

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
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from config import get
from research.signals.agent import generation as gen
from research.signals.agent import data as data_mod
from research.signals.agent import search as search_mod
from research.signals.agent import promotion as bt_mod

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
    cfg['horizon_lags_bars'] = [6]
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
    cfg['promotion'].update({'min_rolls_survived': 1, 'deflation_mult': 1.0,
                             'max_book_size': 10, 'min_select_days': 0,
                             'min_capture': 0.0})
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
print("--- 1. evaluator calibration ---")
n_stamps, n_assets = 360, 40
cal_ts = np.repeat(pd.date_range('2024-01-01', periods=n_stamps, freq='1h'),
                   n_assets)
cal_sym = np.tile([f'A{i}' for i in range(n_assets)], n_stamps)

for rho in (0.3, 0.0):
    sig_vals = rng.normal(size=n_stamps * n_assets)
    tgt = rho * sig_vals + np.sqrt(1 - rho ** 2) * rng.normal(
        size=n_stamps * n_assets)
    sig_df = pd.DataFrame({'timestamp': cal_ts, 'symbol': cal_sym,
                           'signal': sig_vals})
    win = pd.DataFrame({'timestamp': cal_ts, 'symbol': cal_sym,
                        'fwd_6b': tgt, 'is_liquid': True})
    m = search_mod.evaluate_window(sig_df, win, 'fwd_6b', lag_bars=1,
                                   min_assets=10)
    # Spearman of a bivariate normal: (6/pi) * asin(rho/2)
    expected = 6.0 / np.pi * np.arcsin(rho / 2.0)
    if rho > 0:
        check(f"calibration: planted rho={rho} recovered",
              abs(m['ic_mean'] - expected) < 0.03,
              f"(ic {m['ic_mean']:.4f} vs expected {expected:.4f})")
        check("calibration: t-stat strongly significant", m['ic_tstat'] > 5,
              f"(t {m['ic_tstat']:.1f})")
    else:
        check("calibration: zero-IC panel reads ~0",
              abs(m['ic_mean']) < 0.02, f"(ic {m['ic_mean']:.4f})")
        check("calibration: zero-IC t-stat in bounds",
              abs(m['ic_tstat']) < 3.5, f"(t {m['ic_tstat']:.2f})")

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
# 4. Noise in, nothing out: FDR + deflation on a pure-noise panel
# ---------------------------------------------------------------------------
print("--- 4. noise -> no promotion ---")
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
check("noise: ~nothing promoted from pure noise", len(noise_promoted) <= 1,
      f"({len(noise_promoted)} promoted, "
      f"{time.perf_counter() - t0:,.1f}s)")

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
      best['metrics_select']['ic_tstat'] > 3
      and 'res_zscore' in gen.candidate_columns(best['candidate']),
      f"(t {best['metrics_select']['ic_tstat']:.1f}, "
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
persist_cfg = copy.deepcopy(CFG)
persist_cfg['promotion']['min_rolls_survived'] = 2
promoted_r1 = bt_mod.promote(survivors_roll1, ROLL_B, ledger_a,
                             persist_cfg)
check("carry-over: 2-roll persistence gate now passes",
      len(promoted_r1) >= 1
      and all(ledger_a.consecutive_survivals(p['candidate'].hash, 1) >= 2
              for p in promoted_r1),
      f"({len(promoted_r1)} promoted with 2-roll persistence)")
check("carry-over: ledger survivor_candidates round-trips",
      {c.hash for c in ledger_a.survivor_candidates(0)}
      == {s['candidate'].hash for s in survivors_a})

# Discovery is purely statistical: promotions carry everything the
# walk-forward (the only money judge) needs to trade them.
if promoted:
    p0 = promoted[0]
    check("e2e: promotion carries profile + half-life + capture",
          'half_life_bars' in p0 and 'capture' in p0
          and 'promoted_lags' in p0 and len(p0['promoted_lags']) >= 1
          and 0.0 < p0['capture'] <= 1.0,
          f"(hl {p0['half_life_bars']:.0f}b, capture {p0['capture']:.2f}, "
          f"lags {p0['promoted_lags']})")
print(f"(end-to-end block: {time.perf_counter() - t0:,.1f}s)")

# ---------------------------------------------------------------------------
# 5b. Profile (no pinning): each candidate carries its full per-lag IC
#     profile; best_lag (day-equivalent t, per-bet-fair) lands on the horizon
#     where the effect actually lives, and the fitted half-life separates
#     fast from slow alpha.
# ---------------------------------------------------------------------------
print("--- 5b. per-lag profile finds each effect's true horizon ---")
CFG_ML = test_cfg()
CFG_ML['horizon_lags_bars'] = [6, 36]
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
check("profile: fast impulse effect -> best per-bet lag 6",
      e_fast['target_lag'] == 6,
      f"(chose {e_fast['target_lag']}, t {e_fast['metrics_select']['ic_tstat']:.1f})")
check("profile: fast effect -> short fitted half-life",
      e_fast['half_life_bars'] <= 24,
      f"(half-life {e_fast['half_life_bars']:.0f} bars)")

slow_panel_ml = make_impulse_panel()
led_slow = search_mod.DiscoveryLedger(None)
pop_slow = search_mod.run_search(
    slow_panel_ml, ROLL, family_cols,
    gen.RandomProposer(dsl_cfg=CFG_ML['dsl'], mutation_prob=0.6),
    led_slow, CFG_ML, seed_candidates=[wn_seed])
e_slow = next(s for s in pop_slow if s['candidate'].hash == wn_seed.hash)
check("profile: slow impulse effect -> best per-bet lag 36",
      e_slow['target_lag'] == 36,
      f"(chose {e_slow['target_lag']}, t {e_slow['metrics_select']['ic_tstat']:.1f})")
check("profile: slow effect is real at its lag",
      e_slow['metrics_select']['ic_tstat'] > 2,
      f"(select t {e_slow['metrics_select']['ic_tstat']:.1f})")
check("profile: slow effect outlives the fast one",
      e_slow['half_life_bars'] > e_fast['half_life_bars'],
      f"(slow {e_slow['half_life_bars']:.0f} vs fast "
      f"{e_fast['half_life_bars']:.0f} bars)")
check("profile: ledger records best lag + half-life + profile json",
      set(led_fast.to_frame()['target_lag']) == {6}
      and set(led_slow.to_frame()['target_lag']) == {36}
      and led_slow.to_frame()['half_life_bars'].notna().all()
      and (led_slow.to_frame()['profile_json'].str.len() > 2).all())
check("profile: BOTH lags scored on train and select",
      all(set(e['profile_train']) == {6, 36}
          and set(e['profile_select']) == {6, 36}
          for e in (e_fast, e_slow)))

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
      e_neg['metrics_select']['ic_tstat'] > 3,
      f"(t {e_neg['metrics_select']['ic_tstat']:.1f})")
check("negative effect: train IC positive after the flip too",
      e_neg['metrics_train']['ic_mean'] > 0)

# ---------------------------------------------------------------------------
# 5c. ML probe robustness: degenerate feature columns (all-NaN / constant in
#     the training slice, e.g. futures columns before their data starts) must
#     be dropped, not crash sklearn's binner ("window shape cannot be larger
#     than input array shape").
# ---------------------------------------------------------------------------
print("--- 5c. ml probe on degenerate columns ---")
probe_panel = fast_panel_ml.copy()
probe_panel['dead_all_nan'] = np.nan
probe_panel['dead_constant'] = 1.0
probe_cols = ['res_zscore', 'res_mom', 'vol_ratio', 'vol_noise',
              'dead_all_nan', 'dead_constant']
try:
    probe = search_mod.run_ml_probe(probe_panel, ROLL, probe_cols, CFG_ML)
    m6 = probe['metrics_by_lag'].get(6, {})
    check("ml probe: survives all-NaN + constant columns",
          np.isfinite(m6.get('ic_mean', np.nan)),
          f"(ic {m6.get('ic_mean', float('nan')):.4f})")
    check("ml probe: one metrics entry per search lag",
          set(probe['metrics_by_lag']) == {6, 36})
    check("ml probe: planted effect visible in the ceiling",
          m6.get('ic_tstat', 0.0) > 2, f"(t {m6.get('ic_tstat', 0.0):.1f})")
except Exception as e:
    check("ml probe: survives all-NaN + constant columns", False, f"({e})")

# ---------------------------------------------------------------------------
# 5d. Unit checks: day-equivalent t, half-life fit, persistence weight,
#     analytic flip, Student-t p-values, half-alpha gate.
# ---------------------------------------------------------------------------
print("--- 5d. profile/persistence unit checks ---")

# day-equivalent t: the same raw t is worth sqrt(24) less at lag 6 (24
# stamps/day) than at lag 144 (1 stamp/day) - the anti-pinning-bias scale.
t6 = search_mod.day_equivalent_tstat({'ic_tstat': 10.0}, 6)
t144 = search_mod.day_equivalent_tstat({'ic_tstat': 10.0}, 144)
check("day-equivalent t: lag-6 discounted by sqrt(24)",
      abs(t6 - 10.0 / np.sqrt(24)) < 1e-9 and abs(t144 - 10.0) < 1e-9,
      f"(t6 {t6:.2f}, t144 {t144:.2f})")

# half-life fit: a flat cumulative profile (all alpha in the first bars) is
# FAST; a linearly growing one (constant per-bar alpha) is SLOW.
hl_fast = search_mod.fit_half_life({6: 1.0, 36: 1.0, 72: 1.0, 144: 1.0})
hl_slow = search_mod.fit_half_life({6: 6.0, 36: 36.0, 72: 72.0, 144: 144.0})
check("half-life fit: flat profile -> fast, linear profile -> slow",
      hl_fast <= 12 and hl_slow >= 720,
      f"(fast {hl_fast:.0f}, slow {hl_slow:.0f} bars)")
check("half-life fit: degenerate profile priced as fastest, never free",
      search_mod.fit_half_life({}) == search_mod.HALF_LIFE_GRID[0]
      and search_mod.fit_half_life({6: -1.0, 36: -2.0})
      == search_mod.HALF_LIFE_GRID[0])

# persistence weight 1/(1 + phi/rate): persistent alpha keeps ~full weight,
# alpha faster than the trade rate is crushed toward 0.
rate = search_mod.trade_rate_per_bar()
w_slow = search_mod.persistence_weight(1008, rate)
w_fast = search_mod.persistence_weight(3, rate)
check("persistence weight: slow ~1, fast ~0, monotone",
      w_slow > 0.9 and w_fast < 0.25 and w_slow > w_fast,
      f"(rate {rate:.4f}/bar: hl 1008 -> {w_slow:.3f}, hl 3 -> {w_fast:.3f})")

# capture-weighted reward: a slow signal outscores an equally-strong fast
# one; the IC term is exactly (day-equivalent t) x capture.
m_eq = {'ic_mean': 0.02, 'ic_tstat': 5.0, 'icir': 0.3,
        'liquid_ic_ratio': 0.5, 'target_dispersion': 0.01,
        'n_cross_sections': 100, 'n_days': 20}
cand_r = gen.Candidate(name='r', family='residual_shape',
                       expression=('col', 'res_zscore'))
t_fast = search_mod.reward_terms(m_eq, 144, 6, 0.0, cand_r, 0.0)
t_slow = search_mod.reward_terms(m_eq, 144, 288, 0.0, cand_r, 0.0)
check("reward: capture-weighted IC term (slow beats equally-strong fast)",
      t_slow['ic_tstat'] > 2.5 * t_fast['ic_tstat']
      and abs(t_slow['ic_tstat']
              - 5.0 * search_mod.persistence_weight(288, rate)) < 1e-9,
      f"(fast {t_fast['ic_tstat']:.2f} vs slow {t_slow['ic_tstat']:.2f})")

# analytic flip mirrors IC metrics exactly
m0 = {'ic_mean': 0.02, 'ic_tstat': 3.0, 'icir': 0.4,
      'liquid_ic_ratio': 0.7, 'target_dispersion': 0.01,
      'n_cross_sections': 100, 'n_days': 10}
mf = search_mod.flip_metrics(m0)
check("flip: ic/t/icir negate; ratio/dispersion/counts unchanged",
      mf['ic_mean'] == -0.02 and mf['ic_tstat'] == -3.0
      and mf['icir'] == -0.4 and mf['liquid_ic_ratio'] == 0.7
      and mf['target_dispersion'] == 0.01 and mf['n_days'] == 10)

# Student-t p-values: on 30 daily obs the normal approximation is
# anti-conservative - the t p-value must be LARGER.
p_t = bt_mod._tstat_pvalue(2.5, n_days=30)
p_norm = bt_mod._tstat_pvalue(2.5)
check("p-values: Student-t (30 days) > normal for the same t",
      p_t > p_norm, f"(t-dist {p_t:.4f} vs normal {p_norm:.4f})")

# capture floor: with min_capture above the single-lag fit's capture,
# nothing may promote (the gate that replaced the PnL scoreboard).
cap_cfg = copy.deepcopy(CFG)
cap_cfg['promotion']['min_capture'] = 0.99
check("capture floor: min_capture blocks fast-alpha promotion",
      bt_mod.promote(survivors_a, ROLL, ledger_a, cap_cfg) == [])

# ---------------------------------------------------------------------------
# 5e. feedback.md #1/#2/#3: diagnostic blend, AST diversity, incremental reward
# ---------------------------------------------------------------------------
print("--- 5e. diagnostic blend / AST diversity / incremental reward ---")

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

# #2 AST similarity: identical structure = 1; a shared concrete subtree gives
# partial overlap; a pure column swap (no shared concrete block) = 0.
_A = gen.Candidate('a', 'f', ('neg', ('square', ('col', 'res_zscore'))))
_A2 = gen.Candidate('a2', 'f', ('neg', ('square', ('col', 'res_zscore'))))
_Bclone = gen.Candidate('b', 'f',                       # shares 2 of _A's blocks
                        ('tanh', ('neg', ('square', ('col', 'res_zscore')))))
_C = gen.Candidate('c', 'f', ('mul', ('square', ('col', 'res_zscore')),
                              ('col', 'res_mom')))
_D = gen.Candidate('d', 'f', ('neg', ('square', ('col', 'res_mom'))))
check("ast: identical structure -> 1.0", gen.ast_similarity(_A, _A2) == 1.0)
check("ast: shared concrete subtree -> partial", 0.0 < gen.ast_similarity(_A, _C) < 1.0)
check("ast: structural clone -> high (> 0.5)", gen.ast_similarity(_A, _Bclone) > 0.5)
check("ast: column swap (no shared block) -> 0.0", gen.ast_similarity(_A, _D) == 0.0)
check("ast: candidate_subtrees excludes bare leaves",
      all(gen._depth(s) >= 2 for s in gen.candidate_subtrees(_C)))

# select_survivors: a structural clone that is OUTPUT-uncorrelated with the
# original - the AST gate must still drop the lower-reward one.
_ts = pd.date_range('2024-01-01', periods=40, freq='1h')
def _sig_panel(vals):
    return pd.DataFrame({'timestamp': np.repeat(_ts, 3),
                         'symbol': np.tile(['X', 'Y', 'Z'], 40),
                         'signal': vals})
rng_s = np.random.default_rng(3)
v1 = rng_s.normal(size=120); v2 = rng_s.normal(size=120)  # ~uncorrelated
pop_ast = [
    {'candidate': _A, 'reward': 2.0, 'signal_train': _sig_panel(v1)},
    {'candidate': _Bclone, 'reward': 1.0, 'signal_train': _sig_panel(v2)},
    {'candidate': _D, 'reward': 0.5, 'signal_train': _sig_panel(rng_s.normal(size=120))},
]
kept_hashes = {s['candidate'].hash for s in
               search_mod.select_survivors(pop_ast, 3, 0.99, max_ast_sim=0.5)}
check("select_survivors: AST gate drops a structural clone",
      _A.hash in kept_hashes and _Bclone.hash not in kept_hashes
      and _D.hash in kept_hashes)
check("select_survivors: AST gate off (1.0) keeps both clones",
      len(search_mod.select_survivors(pop_ast, 3, 0.99, 1.0)) == 3)

# #3 incremental reward: pooled_signal averages; the reward carries the term
# and it lifts the reward for a positive marginal contribution.
_pooled = search_mod.pooled_signal([_sig_panel(np.ones(120)),
                                    _sig_panel(3 * np.ones(120))])
check("pooled_signal: averages the panels", bool((_pooled['signal'] == 2.0).all()))
_m = {'ic_mean': 0.02, 'ic_tstat': 4.0, 'liquid_ic_ratio': 0.5,
      'target_dispersion': 0.01, 'n_days': 20}
_t_hi = search_mod.reward_terms(_m, 36, 288, 0.0, _A, 0.0, incremental=0.01)
_t_lo = search_mod.reward_terms(_m, 36, 288, 0.0, _A, 0.0, incremental=0.0)
check("reward: incremental term present and passed through",
      _t_hi['incremental'] == 0.01 and _t_lo['incremental'] == 0.0)
_rwd_hi, _ = search_mod.compute_reward(_m, 36, 288, 0.0, _A, 0.0,
                                       incremental=0.01, reward_cfg=CFG['reward'])
_rwd_lo, _ = search_mod.compute_reward(_m, 36, 288, 0.0, _A, 0.0,
                                       incremental=0.0, reward_cfg=CFG['reward'])
check("reward: positive incremental raises the reward",
      _rwd_hi > _rwd_lo)

# ---------------------------------------------------------------------------
# 6. Proposer providers + cost tracking (no API calls: clients are lazy)
# ---------------------------------------------------------------------------
print("--- 6. providers + cost tracking ---")


class _BrokenSDKProposer(gen.GeminiProposer):
    """Simulates a missing SDK: _complete raises ImportError."""
    def __init__(self):
        pass
    provider = 'gemini'
    llm_cfg = {'candidates_per_call': 8}
    usage = {'calls': 0, 'input_tokens': 0, 'output_tokens': 0}

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
    parent_scores={_pa.hash: {'reward': 2.1, 'ic_tstat': 4.3, 'half_life_bars': 288},
                   _pb.hash: {'reward': 0.1, 'ic_tstat': 0.4, 'half_life_bars': 6}},
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

class _FlakyProposer(gen.GeminiProposer):
    """First call truncates INSIDE the first object (nothing salvageable);
    the retry returns a clean array."""
    def __init__(self, fail_times=1):
        self.llm_cfg = {'candidates_per_call': 8}
        self.usage = {'calls': 0, 'input_tokens': 0, 'output_tokens': 0}
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
