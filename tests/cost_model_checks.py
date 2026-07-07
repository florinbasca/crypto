"""
Unit checks for the execution-matched selection cost and the edge-scaled-gross
deployment floor in research/portfolio/walk_forward.py. No database required.

- selection_cost_rate: the per-side cost charged to a signal's own turnover in
  selection is amortized by the Garleanu-Pedersen fill factor h/(h + 1/kappa)
  at the signal's holding lag - the same factor that discounts the alpha side
  - so the gate prices the turnover the executor actually trades, not full
  stamp-by-stamp replication. Falls back to the full cost when amortization is
  disabled, gp_trading is off, or the lag is unknown.
- edge_gross_multiplier(min_mult): a multiplier below the floor snaps to 0.0
  (unwind) instead of deploying a sliver of gross that pays full relative
  costs on ~no expected alpha.

Run: uv run tests/cost_model_checks.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from config import get   # import runs validate_config
from research.portfolio.walk_forward import (PORT, WF, _lag_bars_from_label,
                                             edge_gross_multiplier,
                                             effective_fill_rate,
                                             selection_cost_rate)

FAILURES = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        FAILURES.append(name)


# --- horizon-label parsing ---------------------------------------------------
check("'36b' -> 36", _lag_bars_from_label('36b') == 36)
check("'6b' -> 6", _lag_bars_from_label('6b') == 6)
check("non-bar labels -> None",
      all(_lag_bars_from_label(s) is None for s in ('1h', '10min', 'b', '', 'xb')))

# --- selection_cost_rate: amortization by the GP fill factor ------------------
FULL = PORT['cost_bps'] / 10000.0
_, KAPPA = effective_fill_rate()
gp_enabled = PORT.get('gp_trading', {}).get('enabled', False)

check("unknown lag -> full cost (conservative)",
      selection_cost_rate(None) == FULL)

if gp_enabled and FULL > 0:
    r6 = selection_cost_rate(6)
    r144 = selection_cost_rate(144)
    check("amortized cost matches closed form h/(h + 1/kappa) at h=6",
          abs(r6 - FULL * 6 / (6 + 1 / KAPPA)) < 1e-15,
          f"(rate {r6:.3e}, factor {r6 / FULL:.3f})")
    check("amortized < full cost at short lags", r6 < FULL)
    check("monotone increasing in holding lag", r6 < r144 < FULL + 1e-15,
          f"(factors {r6 / FULL:.3f} < {r144 / FULL:.3f})")
    check("amortized cost matches closed form at h=144",
          abs(r144 - FULL * 144 / (144 + 1 / KAPPA)) < 1e-15,
          f"(factor {r144 / FULL:.3f} at 1/kappa={1 / KAPPA:.0f} bars)")
    check("lag floored at 1 bar",
          selection_cost_rate(0) == selection_cost_rate(1))
else:
    print("[SKIP] amortization checks (gp_trading disabled or zero cost)")

# Fallback: gp off -> full cost. Mutate config dict and restore.
_saved_gp = PORT['gp_trading'].get('enabled', False)
PORT['gp_trading']['enabled'] = False
check("gp_trading disabled -> full cost (no aim fill modeled)",
      selection_cost_rate(6) == FULL)
PORT['gp_trading']['enabled'] = _saved_gp

# --- edge_gross_multiplier: linear clip + deployment floor --------------------
check("free execution -> always full size",
      edge_gross_multiplier(0.0, 0.0, 2.0, min_mult=0.9) == 1.0)
check("edge covers edge_mult round trips -> full size",
      edge_gross_multiplier(0.002, 0.001, 2.0) == 1.0)
check("min_mult=0 reproduces the legacy linear clip",
      abs(edge_gross_multiplier(0.0005, 0.001, 2.0) - 0.25) < 1e-15)
check("multiplier below the floor snaps to zero",
      edge_gross_multiplier(0.0004, 0.001, 2.0, min_mult=0.25) == 0.0,
      f"(raw mult {0.0004 / 0.002:.2f} < floor 0.25 -> 0)")
check("multiplier at the floor is kept",
      edge_gross_multiplier(0.0005, 0.001, 2.0, min_mult=0.25) == 0.25)
check("multiplier above the floor passes through",
      abs(edge_gross_multiplier(0.0012, 0.001, 2.0, min_mult=0.25) - 0.6) < 1e-15)
check("zero / negative edge -> zero (with or without floor)",
      edge_gross_multiplier(0.0, 0.001, 2.0) == 0.0
      and edge_gross_multiplier(-0.001, 0.001, 2.0, min_mult=0.25) == 0.0)

# --- cost_holding_bars: turnover-implied holding period ------------------------
from research.portfolio.walk_forward import cost_holding_bars

MAXB = 1008
check("churny signal (turnover 1) -> the scoring lag",
      cost_holding_bars(6, 1.0, MAXB) == 6.0)
check("slow carry (turnover 0.04) -> lag/turnover",
      cost_holding_bars(6, 0.04, MAXB) == 150.0,
      f"({cost_holding_bars(6, 0.04, MAXB):.0f} bars)")
check("near-zero turnover capped at cost_holding_max_bars",
      cost_holding_bars(6, 0.001, MAXB) == 1008.0)
check("hyperactive signal (turnover > 1) credited LESS than the lag",
      cost_holding_bars(6, 2.0, MAXB) == 3.0
      and cost_holding_bars(6, 5.0, MAXB) == 3.0)   # turnover clipped at 2
check("missing/invalid turnover -> the scoring lag (fallback)",
      cost_holding_bars(6, None, MAXB) == 6.0
      and cost_holding_bars(6, float('nan'), MAXB) == 6.0
      and cost_holding_bars(6, 0.0, MAXB) == 6.0)
check("lag floored at 1 bar", cost_holding_bars(0, 1.0, MAXB) == 1.0)
check("max_bars=None reads portfolio.cost_holding_max_bars",
      cost_holding_bars(6, 0.001) ==
      float(get('portfolio.cost_holding_max_bars')))

import pandas as pd

# --- combination weights: net-economics strength basis --------------------------
from research.portfolio.walk_forward import SignalSelector

rng_c = np.random.default_rng(5)
comb_stats = pd.DataFrame({
    'signal_name': ['sleeve', 'churny'],
    'sign': [1.0, 1.0],
    'ic_mean': [0.004, 0.010],        # churny is 2.5x "stronger" by IC...
    'sharpe_net': [1.0, 0.1],         # ...but the sleeve earns after costs
})
comb_rets = pd.DataFrame({'sleeve': rng_c.normal(size=200),
                          'churny': rng_c.normal(size=200)})
_saved_comb = dict(WF.get('signal_combination', {}))
WF['signal_combination'] = {'enabled': True, 'corr_shrink': 0.5}
w_net = SignalSelector.combination_weights(comb_stats, ['sleeve', 'churny'],
                                           comb_rets)
check("weights follow after-cost value, not IC magnitude",
      w_net['sleeve'] > w_net['churny'],
      f"(sleeve {w_net['sleeve']:.2f} vs churny {w_net['churny']:.2f})")
check("weights sum to 1", abs(sum(w_net.values()) - 1) < 1e-12)

neg_stats = comb_stats.assign(sharpe_net=[1.0, -0.5])
w_neg = SignalSelector.combination_weights(neg_stats, ['sleeve', 'churny'],
                                           comb_rets)
check("negative training net -> zero combination weight",
      w_neg['churny'] == 0.0 and abs(w_neg['sleeve'] - 1.0) < 1e-12)
nan_stats = comb_stats.assign(sharpe_net=[1.0, np.nan])
w_nan = SignalSelector.combination_weights(nan_stats, ['sleeve', 'churny'],
                                           comb_rets)
check("NaN training net -> zero strength (not a crash)",
      w_nan['churny'] == 0.0)
# Fallback path (<2 signals with return history) uses the same strength
w_fb = SignalSelector._ic_weights(comb_stats, ['sleeve', 'churny'])
check("knob-free fallback weights by after-cost value too",
      w_fb['sleeve'] > w_fb['churny'],
      f"(sleeve {w_fb['sleeve']:.2f})")
WF['signal_combination'] = _saved_comb

# --- config wiring -------------------------------------------------------------
check("edge_scaled_gross.min_mult present and valid",
      0.0 <= float(get('portfolio.edge_scaled_gross.min_mult', -1)) < 1.0,
      f"(min_mult {get('portfolio.edge_scaled_gross.min_mult')})")
check("cost-aware net-Sharpe gate is active",
      get('walk_forward.min_net_sharpe_threshold') is not None,
      f"(threshold {get('walk_forward.min_net_sharpe_threshold')})")

# ---------------------------------------------------------------------------
print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURES: {FAILURES}")
    sys.exit(1)
print("ALL COST-MODEL CHECKS PASSED")
