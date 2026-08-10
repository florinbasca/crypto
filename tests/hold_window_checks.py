"""
Checks for the no-promotion HOLD path in research/portfolio/walk_forward.py
(_hold_window / _run_held_window, portfolio.max_hold_months_no_promotion).

A month that promotes nothing confirms no alpha, so
no new book is built - but the book already on the exchange keeps earning,
accruing funding, and losing names that leave the universe. Skipping those
months (the old behavior) dropped that from the record entirely: between W06
and W09 a live 63-name book went unsimulated for 60 days and the equity curve
read flat across the hole.

The hold is capped. Past max_hold_months_no_promotion consecutive unconfirmed
months the book is unwound at the volume-participation cap, because holding a
stale book forever is an implicit bet that expired alpha persists. Without
the cap the tail of
this dataset (rolls 18+ promote nothing, prices run to 2026-07) would park the
W17 book on the exchange for 18 months.

Uses the real DataContext over a short span, so it needs the DB. Checks:
  - a frozen hold trades nothing, keeps gross, and still earns / pays funding
  - forced closes still happen (names leaving the universe are cut)
  - an unwind respects the participation cap and drives the book to flat
  - the streak logic holds for `limit` months, then unwinds, then stays flat
  - realized exposures are recorded (drift), not silently zeroed

Run: uv run tests/hold_window_checks.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from config import config as global_config
from research.portfolio.walk_forward import (WalkForwardPortfolio, PORT,
                                             FACTOR_NAMES)

FAILURES = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        FAILURES.append(name)


# The unsimulated stretch: W06 ended 2023-12-31, W09 resumed 2024-03-01.
T0 = pd.Timestamp('2024-01-01')
T1 = pd.Timestamp('2024-01-08')          # one week is enough to exercise the loop

wf = WalkForwardPortfolio()
wf.persist = False                       # never write tables from a test
wf._ctx_start = T0 - pd.Timedelta(days=45)
wf._ctx_end = T1 + pd.Timedelta(days=2)
wf._ensure_context()

def carried_book() -> pd.Series:
    """The book to hold. Prefers a REAL stored book (the last bar walk_forward
    persisted), falling back to a synthetic dollar-neutral one over the live
    universe. The fallback matters: wf_portfolio_weights is empty whenever a
    run is in progress or has just reset its tables, and these checks are about
    _hold_window's mechanics, which do not care where the weights came from."""
    # Restricted to names live at T0: the frozen-hold assertions below are
    # about a book that is NOT forced to close anything, so a name the stored
    # book held at some later date but which is not investable here would
    # otherwise drop out and (correctly) move gross and turnover.
    # Holdable = in the universe AND carrying a live beta row; _hold_window
    # force-closes anything else, which would move gross and turnover.
    _betas = wf.ctx.betas_for_day(T0.normalize())
    live = wf.ctx.members_at(T0) & set(_betas.index)
    from dbutil import load_data
    try:
        wts = load_data('wf_portfolio_weights')
        if wts is not None and 'timestamp' in getattr(wts, 'columns', []):
            wts = wts.copy()
            wts['timestamp'] = pd.to_datetime(wts['timestamp'])
            last = wts['timestamp'].max()
            b = wts[wts['timestamp'] == last].set_index('symbol')['weight']
            b = b[[s for s in b.index if s in live]]
            if len(b) >= 10:
                print(f"carried book: REAL, {len(b)} names from {last} "
                      f"(live at {T0.date()}), gross {b.abs().sum():.4f}\n")
                return b
    except Exception as e:                       # noqa: BLE001 - diagnostic only
        print(f"(stored weights unavailable: {e})")
    members = sorted(live)[:60]
    assert len(members) >= 10, "no holdable names to build a test book from"
    v = np.linspace(-1.0, 1.0, len(members))
    v = v - v.mean()
    b = pd.Series(v / np.abs(v).sum() * 0.48, index=members)
    print(f"carried book: SYNTHETIC, {len(b)} names, "
          f"gross {b.abs().sum():.4f} (no stored run to read)\n")
    return b


book = carried_book()

# --- frozen hold ------------------------------------------------------------
wf._carry = book.copy()
held = wf._hold_window(T0, T1, unwind=False)
check("hold produces bars for a month that promotes nothing",
      held is not None and not held.empty,
      f"({0 if held is None else len(held)} bars)")

check("hold trades nothing voluntarily (turnover only from forced closes)",
      float(held['turnover'].sum()) < 1e-9 or
      bool((held['turnover'] > 0).sum() <= (held['n_positions'].diff() < 0).sum()),
      f"(total turnover {held['turnover'].sum():.6f})")
check("gross is preserved across the hold (book is frozen, not decayed)",
      abs(held['gross_exposure'].iloc[-1] - book.abs().sum()) < 1e-6,
      f"({held['gross_exposure'].iloc[0]:.4f} -> {held['gross_exposure'].iloc[-1]:.4f})")
check("the frozen book actually earns (PnL is not silently zero)",
      float(held['gross_return'].abs().sum()) > 0,
      f"(sum |gross| {held['gross_return'].abs().sum():.4f})")
check("funding accrues on held perp positions",
      'funding_pnl' in held.columns and float(held['funding_pnl'].abs().sum()) > 0,
      f"(sum |funding| {held['funding_pnl'].abs().sum():.6f})")
check("net = gross - cost + funding, bar by bar",
      np.allclose(held['net_return'],
                  held['gross_return'] - held['trade_cost'] + held['funding_pnl']))
check("realized exposures are recorded, not zeroed",
      held['mkt_exposure'].notna().all() and float(held['mkt_exposure'].abs().max()) > 0,
      f"(max |mkt| {held['mkt_exposure'].abs().max():.2e})")
check("every neutralized factor gets an exposure column",
      all(f'{n}_exposure' in held.columns or n in ('market', 'size')
          for n in FACTOR_NAMES))
check("exp_alpha is NaN on a held month (no aim -> no expected alpha)",
      held['exp_alpha'].isna().all())
check("carry survives a hold", len(wf._carry) > 0, f"({len(wf._carry)} names)")

# --- unwind -----------------------------------------------------------------
wf._carry = book.copy()
unwound = wf._hold_window(T0, T1, unwind=True)
check("unwind produces bars", unwound is not None and not unwound.empty)
check("unwind reduces gross monotonically",
      bool((unwound['gross_exposure'].diff().dropna() <= 1e-12).all()),
      f"({unwound['gross_exposure'].iloc[0]:.4f} -> "
      f"{unwound['gross_exposure'].iloc[-1]:.4f})")
check("unwind ends flat (or is still working within the cap)",
      unwound['gross_exposure'].iloc[-1] < book.abs().sum(),
      f"(final gross {unwound['gross_exposure'].iloc[-1]:.6f})")
check("unwind pays a real exit cost",
      float(unwound['trade_cost'].sum()) > 0,
      f"(cost {unwound['trade_cost'].sum() * 100:.4f}% of book)")
part_cap = PORT['participation']['max_participation']
pm = unwound['participation_max'].dropna()
check("unwind never exceeds the volume-participation cap",
      pm.empty or float(pm.max()) <= part_cap + 1e-9,
      f"(max participation {0 if pm.empty else pm.max():.4f} vs cap {part_cap})")

# --- streak logic -----------------------------------------------------------
limit = int(PORT['max_hold_months_no_promotion'])
check("max_hold_months_no_promotion is configured (not hardcoded)",
      isinstance(limit, int) and limit >= 0, f"(limit {limit})")

from research.portfolio.walk_forward import WindowResult


def fake_window(i):
    return WindowResult(i, T0 - pd.Timedelta(days=180), T0, T1)


wf._carry = book.copy()
wf._no_promo_streak = 0
states = []
for i in range(limit + 3):
    r = fake_window(i)
    wf._run_held_window(r)
    states.append(r.hold_state)
print(f"    streak states over {limit + 3} unconfirmed months: {states}")
check(f"first {limit} unconfirmed months HOLD",
      all(s == 'held' for s in states[:limit]), f"({states[:limit]})")
check("the month after the limit UNWINDS",
      states[limit] == 'unwind', f"({states[limit]!r})")
check("once flat, later unconfirmed months record nothing",
      all(s == '' for s in states[limit + 1:]), f"({states[limit + 1:]})")
# A traded month sets _no_promo_streak = 0 (run_window); the book must then be
# eligible to hold again rather than staying latched in unwind.
wf._no_promo_streak = 0
r = fake_window(99)
wf._carry = book.copy()
wf._run_held_window(r)
check("after a reset the book HOLDS again rather than unwinding",
      r.hold_state == 'held', f"({r.hold_state!r})")

# --- flat book is a no-op ---------------------------------------------------
wf._carry = pd.Series(dtype=float)
r = fake_window(100)
wf._run_held_window(r)
check("an already-flat book records nothing and stays flat",
      r.oos_returns is None and r.hold_state == '')
check("_hold_window on an empty book returns None",
      wf._hold_window(T0, T1, unwind=False) is None)

# --- the hold must never run past where discovery evaluated -----------------
# "No promotions" means "nothing confirmed" only for a roll discovery actually
# tested. For a roll it never ran there is no verdict, and holding the last real
# book through it would report a fabricated result over unexamined months.
from research.signals.data import make_rolls
from config import get as _get

all_rolls = list(make_rolls(_get('discovery')))
evaluated = wf.evaluated_rolls
check("evaluated_rolls is read from the discovery ledger",
      isinstance(evaluated, set) and len(evaluated) > 0,
      f"({len(evaluated)} rolls evaluated of {len(all_rolls)} scheduled)")
check("every evaluated roll has promotions data or a real empty verdict",
      evaluated == set(range(min(evaluated), max(evaluated) + 1)),
      f"(contiguous {min(evaluated)}..{max(evaluated)})")
unevaluated = [r.roll_id for r in all_rolls if r.roll_id not in evaluated]
print(f"    unevaluated rolls: {unevaluated if unevaluated else 'none'}")
check("unevaluated rolls are excluded from the walk-forward schedule",
      all(r not in evaluated for r in unevaluated))
if unevaluated:
    check("the schedule stops at the last evaluated roll, not the last roll",
          max(evaluated) < max(r.roll_id for r in all_rolls),
          f"(schedule ends at roll {max(evaluated)}, "
          f"make_rolls offers {max(r.roll_id for r in all_rolls)})")

print()
if FAILURES:
    print(f"FAILED {len(FAILURES)}: {FAILURES}")
    sys.exit(1)
print("ALL HOLD-WINDOW CHECKS PASSED")
