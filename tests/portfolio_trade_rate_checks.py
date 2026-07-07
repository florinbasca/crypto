"""
Unit checks for the multi-period (Garleanu-Pedersen) trade rate in
research/portfolio/walk_forward.py. No database required.

The trade rate is the per-bar fraction of the gap to the gross-1 aim that the
book trades each bar. It must be cost-responsive (trade slower when costs are
higher), reproduce the legacy halflife rate at the reference cost, fall back to
the halflife when disabled, and stay a valid fraction in (0, 1).

Run: uv run tests/portfolio_trade_rate_checks.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from research.portfolio.walk_forward import gp_trade_rate

FAILURES = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        FAILURES.append(name)


HALFLIFE = 6
legacy = float(1.0 - np.exp(-np.log(2) / HALFLIFE))
GP = {'enabled': True, 'trade_urgency': 0.1223, 'ref_cost_bps': 5.0}

# --- default calibration: at the reference cost the rate ~= legacy halflife ---
r_ref = gp_trade_rate(5.0, GP, HALFLIFE)
check("default reproduces legacy rate at ref cost", abs(r_ref - legacy) < 1e-3,
      f"(rate {r_ref:.4f} vs legacy {legacy:.4f})")

# --- cost responsiveness: cheaper -> trade faster, costlier -> trade slower ---
r_cheap = gp_trade_rate(2.5, GP, HALFLIFE)
r_dear = gp_trade_rate(10.0, GP, HALFLIFE)
check("cheaper cost trades faster than ref", r_cheap > r_ref, f"({r_cheap:.4f} > {r_ref:.4f})")
check("costlier cost trades slower than ref", r_dear < r_ref, f"({r_dear:.4f} < {r_ref:.4f})")
check("rate is monotone decreasing in cost", r_cheap > r_ref > r_dear)

# --- limits: huge urgency -> trade fully (->1), tiny urgency -> barely (->0) ---
r_hi = gp_trade_rate(5.0, {'enabled': True, 'trade_urgency': 1e6, 'ref_cost_bps': 5.0}, HALFLIFE)
r_lo = gp_trade_rate(5.0, {'enabled': True, 'trade_urgency': 1e-6, 'ref_cost_bps': 5.0}, HALFLIFE)
check("huge urgency -> rate ~ 1", r_hi > 0.999, f"({r_hi:.4f})")
check("tiny urgency -> rate ~ 0", r_lo < 1e-3, f"({r_lo:.6f})")

# --- fallbacks: disabled or unset urgency -> legacy halflife rate ---
check("disabled -> legacy fallback",
      abs(gp_trade_rate(5.0, {'enabled': False}, HALFLIFE) - legacy) < 1e-12)
check("urgency unset -> legacy fallback",
      abs(gp_trade_rate(5.0, {'enabled': True}, HALFLIFE) - legacy) < 1e-12)
check("empty gp config -> legacy fallback",
      abs(gp_trade_rate(5.0, {}, HALFLIFE) - legacy) < 1e-12)

# --- always a valid fraction in (0, 1) across a wide cost / urgency sweep ---
valid = True
for cost in (0.1, 1.0, 5.0, 20.0, 100.0):
    for u in (1e-4, 0.01, 0.1223, 1.0, 100.0):
        r = gp_trade_rate(cost, {'enabled': True, 'trade_urgency': u, 'ref_cost_bps': 5.0}, HALFLIFE)
        valid &= (0.0 < r < 1.0)
check("rate stays in (0,1) across cost/urgency sweep", valid)

# --- doubling cost halves omega -> rate matches omega/(1+omega) closed form ---
u = GP['trade_urgency']
omega_10 = u * (5.0 / 10.0)
check("closed form omega/(1+omega) holds",
      abs(r_dear - omega_10 / (1.0 + omega_10)) < 1e-12)

# ---------------------------------------------------------------------------
print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURES: {FAILURES}")
    sys.exit(1)
print("ALL TRADE-RATE CHECKS PASSED")
