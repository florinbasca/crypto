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

Run: uv run python tests/discovery_checks.py
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
from research.signals.agent import backtest as bt_mod

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
    cfg['promotion'].update({'min_rolls_survived': 1, 'deflation_mult': 1.0,
                             'max_promoted_per_roll': 3, 'max_book_size': 10})
    cfg['backtest'].update({'funding_pnl': False, 'min_assets': 10,
                            'cost_bps': 2.0})
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
                                   min_assets=10, cost_bps=2.0)
    # Spearman of a bivariate normal: (6/pi) * asin(rho/2)
    expected = 6.0 / np.pi * np.arcsin(rho / 2.0)
    if rho > 0:
        check(f"calibration: planted rho={rho} recovered",
              abs(m['ic_mean'] - expected) < 0.03,
              f"(ic {m['ic_mean']:.4f} vs expected {expected:.4f})")
        check("calibration: t-stat strongly significant", m['ic_tstat'] > 5,
              f"(t {m['ic_tstat']:.1f})")
        check("calibration: gross sharpe positive", m['gross_sharpe'] > 0)
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
noise_promoted = bt_mod.promote(noise_survivors, ROLL, noise_ledger, [], CFG)
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

promoted = bt_mod.promote(survivors_a, ROLL, ledger_a, [], CFG)
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
promoted_r1 = bt_mod.promote(survivors_roll1, ROLL_B, ledger_a, [],
                             persist_cfg)
check("carry-over: 2-roll persistence gate now passes",
      len(promoted_r1) >= 1
      and all(ledger_a.consecutive_survivals(p['candidate'].hash, 1) >= 2
              for p in promoted_r1),
      f"({len(promoted_r1)} promoted with 2-roll persistence)")
check("carry-over: ledger survivor_candidates round-trips",
      {c.hash for c in ledger_a.survivor_candidates(0)}
      == {s['candidate'].hash for s in survivors_a})

if promoted:
    result = bt_mod.backtest_oos(planted_panel, ROLL, promoted, CFG)
    daily = result['daily']
    check("e2e: OOS backtest produced daily PnL", len(daily) >= 3,
          f"({len(daily)} days, {result['stamps']} rebalances)")
    check("e2e: PnL finite and costs positive",
          np.isfinite(daily['net']).all() and (daily['cost'] >= 0).all())
    check("e2e: book is dollar-neutral",
          result['exposures'].get('dollar', 1.0) < 1e-6,
          f"(mean |net| {result['exposures'].get('dollar'):.2e})")
    check("e2e: planted alpha earns positive OOS gross",
          daily['gross'].sum() > 0,
          f"(gross {daily['gross'].sum() * 1e4:,.1f} bps)")
    curve = bt_mod.stitch_oos([result])
    check("e2e: stitched curve has cum_net",
          'cum_net' in curve.columns and len(curve) == len(daily))
print(f"(end-to-end block: {time.perf_counter() - t0:,.1f}s)")

# ---------------------------------------------------------------------------
# 6. Proposer providers + cost tracking (no API calls: clients are lazy)
# ---------------------------------------------------------------------------
print("--- 6. providers + cost tracking ---")
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
try:
    gen._parse_json_array('no array here')
    check("parse: garbage raises", False)
except ValueError:
    check("parse: garbage raises", True)

# ---------------------------------------------------------------------------
print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURES: {FAILURES}")
    sys.exit(1)
print("ALL DISCOVERY CHECKS PASSED")
