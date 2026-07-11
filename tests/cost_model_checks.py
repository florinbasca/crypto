"""
Cost-model checks for the walk-forward execution layer (no database):
edge_gross_multiplier (edge-scaled gross) and cost_holding_bars
(turnover-implied holding period).

Run: uv run tests/cost_model_checks.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get   # import runs validate_config
from research.portfolio.walk_forward import (edge_gross_multiplier,
                                             cost_holding_bars)

FAILURES = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        FAILURES.append(name)


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

# --- config wiring -------------------------------------------------------------
check("edge_scaled_gross.min_mult present and valid",
      0.0 <= float(get('portfolio.edge_scaled_gross.min_mult', -1)) < 1.0,
      f"(min_mult {get('portfolio.edge_scaled_gross.min_mult')})")

# ---------------------------------------------------------------------------
print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURES: {FAILURES}")
    sys.exit(1)
print("ALL COST-MODEL CHECKS PASSED")
