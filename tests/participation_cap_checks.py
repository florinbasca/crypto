"""
Unit checks for the volume-participation trade cap in
research/portfolio/walk_forward.py (portfolio.participation). No database
required.

The cap says: in any single bar, a name's voluntary trade may not exceed
max_participation x its trailing average bar $ volume, converted to weight
units by book_size_usd (max |dw_i| = max_participation * avg_$vol_i /
book_size_usd). Names with missing volume history are not tradeable (cap 0).

Checks cover the two helpers (participation_caps, clamp_to_participation) and
a sequential mini-simulation that mirrors the _backtest_window ordering
(GP step -> clamp -> clip/neutralize loop -> gross scale -> final clamp),
asserting the cap holds bar by bar after all adjustments.

Run: uv run python tests/participation_cap_checks.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from research.portfolio.walk_forward import (clamp_to_participation,
                                             participation_caps)

FAILURES = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        FAILURES.append(name)


# --- participation_caps: weight-unit conversion and missing-volume handling ---
PART = 0.10
BOOK = 1_000_000.0

caps = participation_caps(np.array([30_000.0, 1_800.0, 740_000.0]), PART, BOOK)
check("cap = participation * $vol / book",
      np.allclose(caps, [0.003, 0.00018, 0.074]),
      f"(caps {caps})")
caps_bad = participation_caps(np.array([np.nan, 0.0, -5.0, np.inf]), PART, BOOK)
check("NaN / zero / negative / inf volume -> cap 0",
      np.allclose(caps_bad[:3], 0.0) and caps_bad[3] == 0.0,
      f"(caps {caps_bad})")

# --- clamp_to_participation: trades bounded, within-cap trades untouched ---
w_prev = np.array([0.02, -0.01, 0.00, 0.03])
w_new = np.array([0.05, -0.05, 0.04, 0.031])
max_dw = np.array([0.01, 0.01, 0.01, 0.01])
clamped = clamp_to_participation(w_new, w_prev, max_dw)
check("over-cap trades clamped to the boundary",
      np.allclose(clamped, [0.03, -0.02, 0.01, 0.031]),
      f"(clamped {clamped})")
check("within-cap trade passes through unchanged", clamped[3] == w_new[3])
check("trade direction preserved",
      np.all(np.sign(clamped - w_prev) == np.sign(w_new - w_prev)))
check("cap 0 -> no trade at all",
      np.allclose(clamp_to_participation(w_new, w_prev, np.zeros(4)), w_prev))
check("clamp is symmetric in trade sign",
      np.allclose(clamp_to_participation(w_prev - (w_new - w_prev), w_prev, max_dw) - w_prev,
                  -(clamped - w_prev)))

# --- sequential mini-simulation mirroring the _backtest_window ordering ---
# GP step toward the aim -> participation clamp -> 3x (position-cap clip +
# clamp + neutrality projection) -> gross soft-ceiling scale -> final hard
# clamp. Asserts: the cap holds for every name on every bar AFTER all
# adjustments; no-volume names never trade; the book still converges toward
# the aim; neutrality slack from the final clamp stays small.
N, T = 30, 60
CAP_W = 0.05 * 0.999
A_EFF = 0.44
GROSS_TARGET = 1.0

ones = np.ones((N, 1))
neutralizer = np.linalg.solve(ones.T @ ones, ones.T)   # dollar neutrality


def make_aims(seed=11):
    """Deterministic per-bar aim sequence (re-drawn every 15 bars)."""
    rng = np.random.default_rng(seed)
    aims, aim = [], None
    for t in range(T):
        if aim is None or (t and t % 15 == 0):
            a = rng.normal(size=N)
            a -= a.mean()
            aim = np.clip(a / np.abs(a).sum(), -CAP_W, CAP_W)
        aims.append(aim)
    return aims


def make_vols(nan_names=0, seed=7):
    """Volumes: wide lognormal spread (like the real panel), first
    `nan_names` columns missing entirely."""
    rng = np.random.default_rng(seed)
    vols = np.exp(rng.normal(np.log(20_000), 1.5, size=(T, N)))
    vols[:, :nan_names] = np.nan
    return vols


def run_sim(book_size, vols, apply_cap=True, aims=None):
    """Mirror of the _backtest_window per-bar ordering."""
    aims = make_aims() if aims is None else aims
    w = np.zeros(N)
    breaches = 0
    max_nan_trade = 0.0
    slack = 0.0
    for t in range(T):
        max_dw = participation_caps(vols[t], PART, book_size)
        w_prev = w.copy()
        v = (1 - A_EFF) * w_prev + A_EFF * aims[t]             # GP step
        if apply_cap:
            v = clamp_to_participation(v, w_prev, max_dw)      # clamp
        for _ in range(3):                                     # cap + neutralize
            v = np.clip(v, -CAP_W, CAP_W)
            if apply_cap:
                v = clamp_to_participation(v, w_prev, max_dw)
            v = v - ones @ (neutralizer @ v)
        g = np.abs(v).sum()                                    # gross ceiling
        if g > 1e-12:
            v = v * min(1.0, GROSS_TARGET / g)
        if apply_cap:
            v = clamp_to_participation(v, w_prev, max_dw)      # final guarantee
            breaches += int((np.abs(v - w_prev) > max_dw + 1e-12).sum())
            nan_cols = ~np.isfinite(vols[t])
            if nan_cols.any():
                max_nan_trade = max(max_nan_trade,
                                    float(np.abs(v - w_prev)[nan_cols].max()))
        slack = max(slack, abs(float(v.sum())))
        w = v
    return w, aims[-1], breaches, max_nan_trade, slack


vols = make_vols(nan_names=3)
w_end, aim_end, breaches, nan_traded, slack = run_sim(BOOK, vols)
check("cap never breached on any (bar, name)", breaches == 0,
      f"({breaches} breaches)")
check("no-volume names never trade", nan_traded == 0.0)
check("dollar-neutrality slack from the final clamp stays small",
      slack < 0.05, f"(max |sum w| {slack:.5f})")

# Convergence: under a FIXED aim, the tradeable (finite-volume) names must
# close most of their gap to the aim despite the per-bar cap. The frozen
# no-volume names are excluded - they can never converge by construction.
fixed_aims = [make_aims()[0]] * T
w_fix, aim_fix, _, _, _ = run_sim(BOOK, vols, aims=fixed_aims)
tradeable = np.isfinite(vols[0])
gap_end = np.abs(w_fix - aim_fix)[tradeable].sum()
gap_start = np.abs(aim_fix)[tradeable].sum()
check("tradeable names converge toward a fixed aim despite the cap",
      gap_end < 0.5 * gap_start,
      f"(gap {gap_start:.4f} -> {gap_end:.4f} over {T} bars)")

# Huge book -> caps bind hard -> executed turnover per bar respects the tiny
# caps but is still nonzero (the book is throttled, not frozen).
w_huge, _, breaches_huge, _, _ = run_sim(1e10, vols)
check("tiny caps (huge book): still zero breaches", breaches_huge == 0)
check("tiny caps throttle but do not freeze the book",
      0 < np.abs(w_huge).sum() < 0.05,
      f"(gross after {T} bars {np.abs(w_huge).sum():.5f})")

# Tiny book, full volume coverage -> caps are huge and non-binding -> the
# trajectory must be IDENTICAL to running with the cap disabled.
vols_full = make_vols(nan_names=0)
w_capped, _, _, _, _ = run_sim(1.0, vols_full, apply_cap=True)
w_uncapped, _, _, _, _ = run_sim(1.0, vols_full, apply_cap=False)
check("non-binding caps reproduce the uncapped trajectory",
      np.allclose(w_capped, w_uncapped, atol=1e-12),
      f"(max diff {np.abs(w_capped - w_uncapped).max():.2e})")

# --- config wiring: block present, validated, and costs live in scoring ---
from config import get, config as global_config   # import runs validate_config

part_cfg = get('portfolio.participation', {})
check("portfolio.participation block exists with all keys",
      all(k in part_cfg for k in
          ('enabled', 'book_size_usd', 'max_participation',
           'volume_window_bars')))
check("discovery backtest cost falls back to portfolio.cost_bps",
      global_config['discovery']['backtest']['cost_bps'] is None)
check("portfolio.cost_bps is set (signals scored net of cost)",
      float(get('portfolio.cost_bps')) > 0.0,
      f"(cost_bps {get('portfolio.cost_bps')})")

# ---------------------------------------------------------------------------
print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURES: {FAILURES}")
    sys.exit(1)
print("ALL PARTICIPATION-CAP CHECKS PASSED")
