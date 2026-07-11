"""
Walk-forward analysis: compare what the promoted signals offered in each OOS
month with what the backtest book captured, and show where the gap comes from.

Everything here is a real, measured object:
  - the HELD book: the backtest's persisted per-bar weights
  - the SIGNAL book: the month's composite ranks as dollar-neutral gross-1
    weights, re-formed every bar (a pure function of the signals - no
    execution parameters)
  - slice IC: fresh-signal IC per horizon slice on the discovery lag grid
  - the signal book's own turnover, priced at portfolio.cost_bps

Run: uv run research/portfolio/walk_forward_analysis.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import logging

import numpy as np
import pandas as pd

from config import config as global_config, get
from dbutil import load_data, table_exists
from research.portfolio.walk_forward import WalkForwardPortfolio, WARMUP_DAYS

logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True,
                    format=global_config['logging']['format'],
                    datefmt=global_config['logging']['datefmt'])

COST_RATE = get('portfolio.cost_bps') / 10000.0
LAG_GRID = [int(x) for x in get('discovery.horizon_lags_bars')]


def _pnl(w: pd.DataFrame, fwd: pd.DataFrame) -> float:
    a, b = w.align(fwd, join='inner')
    return float(np.nansum(a.values * b.values))


def _gross1(w: pd.DataFrame) -> pd.DataFrame:
    g = w.abs().sum(axis=1).replace(0, np.nan)
    return w.div(g, axis=0)


def _slice_ics(comp: pd.DataFrame, fwd: pd.DataFrame, start, end,
               stride: int) -> list:
    ts = comp.index[(comp.index >= start) & (comp.index < end)][::stride]
    out = []
    for t in ts:
        if t not in fwd.index:
            continue
        a, b = comp.loc[t], fwd.loc[t]
        m = a.notna() & b.notna()
        if m.sum() >= 10:
            ic = a[m].rank().corr(b[m].rank())
            if np.isfinite(ic):
                out.append(ic)
    return out


def main():
    wf = WalkForwardPortfolio()
    from research.signals.data import make_rolls
    rolls = make_rolls(get('discovery'))

    res = load_data('residual_returns',
                    columns=['timestamp', 'symbol', 'residual_return',
                             'fwd_raw_10min'])
    res['timestamp'] = pd.to_datetime(res['timestamp'])
    res_w = res.pivot_table(index='timestamp', columns='symbol',
                            values='residual_return',
                            aggfunc='first').sort_index()
    raw_fwd = res.pivot_table(index='timestamp', columns='symbol',
                              values='fwd_raw_10min',
                              aggfunc='first').sort_index()
    res_fwd = res_w.shift(-1)
    # cumulative forward sums over (t, t+L] for the slice returns
    cum = {L: res_w.iloc[::-1].rolling(L, min_periods=L).sum()
                 .iloc[::-1].shift(-1) for L in LAG_GRID}
    slices = {}
    prev = 0
    for L in LAG_GRID:
        lab = f'({prev},{L}]'
        slices[lab] = (cum[L] - cum[prev] if prev else cum[L], L - prev)
        prev = L

    if not table_exists('wf_portfolio_weights'):
        raise SystemExit("no wf_portfolio_weights - run walk_forward.py first")
    hw = load_data('wf_portfolio_weights')
    hw['timestamp'] = pd.to_datetime(hw['timestamp'])
    held = hw.pivot_table(index='timestamp', columns='symbol',
                          values='weight', aggfunc='first')

    rows = []
    slice_agg = {k: [] for k in slices}
    sig_turnover = 0.0
    held_gross_sum, held_bars = 0.0, 0
    for i, r in enumerate(rolls):
        meta = wf.month_meta.get(pd.Timestamp(r.oos_start), [])
        if not meta:
            continue
        selected, weights, lag_of, *_ = wf.month_book(meta)
        comp = wf.composite_scores(
            selected, weights, r.oos_start - pd.Timedelta(days=WARMUP_DAYS),
            r.oos_start, r.oos_end, lag_of=lag_of)
        if not comp:
            continue
        z = None
        for c in comp.values():
            z = c if z is None else z.add(c, fill_value=0.0)
        z = z / len(comp)
        sig = _gross1(z.sub(z.mean(axis=1), axis=0))
        sig = sig.loc[(sig.index >= r.oos_start) & (sig.index < r.oos_end)]
        hb = held.loc[(held.index >= r.oos_start) & (held.index < r.oos_end)]

        rows.append({
            'window': i, 'oos': str(r.oos_start.date()), 'n_sig': len(meta),
            'held.raw': _pnl(hb, raw_fwd), 'held.res': _pnl(hb, res_fwd),
            'sig.res': _pnl(sig, res_fwd), 'sig.raw': _pnl(sig, raw_fwd),
        })
        sig_turnover += float(sig.diff().abs().sum(axis=1).iloc[1:].sum())
        held_gross_sum += float(hb.abs().sum(axis=1).sum())
        held_bars += len(hb)
        for lab, (fwd, width) in slices.items():
            slice_agg[lab].extend(
                _slice_ics(z, fwd, r.oos_start, r.oos_end, stride=width))

    df = pd.DataFrame(rows).set_index('window')
    tot = df[['held.raw', 'held.res', 'sig.res', 'sig.raw']].sum()
    n_bars_sig = sum(1 for _ in ())  # per-bar sig turnover below uses count
    sig_bars = int(df['n_sig'].count() and held_bars)  # same OOS bar count

    print("=" * 78)
    print("PNL PER OOS MONTH (sum of w . fwd, before costs)")
    print("=" * 78)
    print("""columns:
  n_sig     signals promoted for this OOS month (traded that month only)
  held.raw  the ACTUAL backtest book (persisted weights) on raw forward
            returns - must equal the walk-forward's gross PnL
  held.res  the same held book on RESIDUAL forward returns - a gap vs
            held.raw means the factor hedge leaks
  sig.res   the SIGNAL book (composite ranks as dollar-neutral gross-1
            weights, re-formed every bar; a pure function of the signals)
            on residual returns - what the signals offered
  sig.raw   the same signal book on raw returns (unhedged read)
""")
    print(df.to_string(float_format=lambda x: f'{x:+.4f}'))
    print("-" * 78)
    print("SUM " + "  ".join(f"{k}={v:+.4f}" for k, v in tot.items()))

    # ------- verdicts, computed from the numbers ------------------------------
    print()
    print("=" * 78)
    print("VERDICTS (material = 20% of the largest leg)")
    print("=" * 78)
    scale = max(abs(tot['sig.res']), abs(tot['held.raw']), 1e-9)
    mat = 0.2 * scale

    if table_exists('wf_portfolio_returns'):
        ret = load_data('wf_portfolio_returns')
        persisted_gross = float(ret['gross_return'].sum())
        persisted_cost = float(ret['gross_return'].sum()
                               - ret['net_return'].sum()
                               + ret['funding_pnl'].sum())
        ok = abs(persisted_gross - tot['held.raw']) < max(
            0.02, 0.1 * abs(persisted_gross))
        print(f"sanity : held.raw {tot['held.raw']:+.4f} vs persisted gross "
              f"{persisted_gross:+.4f} -> "
              f"{'MATCH' if ok else 'MISMATCH - investigate before reading on'}")
        print(f"costs  : the backtest paid {persisted_cost:.4f} in trading "
              f"costs over the period (gross - net, ex funding)")

    gap_hedge = tot['held.res'] - tot['held.raw']
    print(f"hedge  : held.res - held.raw = {gap_hedge:+.4f} -> "
          + ("clean (same PnL on raw and residual returns)"
             if abs(gap_hedge) < mat else
             "MATERIAL - the factor hedge leaks; check betas/neutrality"))

    if tot['sig.res'] > mat:
        print(f"alpha  : sig.res = {tot['sig.res']:+.4f} -> the promoted "
              f"signals DID predict their OOS months")
    elif tot['sig.res'] < -mat:
        print(f"alpha  : sig.res = {tot['sig.res']:+.4f} -> the promoted "
              f"signals ANTI-predicted; signal quality, not execution")
    else:
        print(f"alpha  : sig.res = {tot['sig.res']:+.4f} ~ 0 -> nothing to "
              f"capture")

    held_avg_gross = held_gross_sum / max(held_bars, 1)
    held_per_gross = tot['held.res'] / max(held_avg_gross, 1e-9)
    gap_capture = tot['sig.res'] - held_per_gross
    print(f"capture: per unit gross, signal book {tot['sig.res']:+.4f} vs "
          f"held book {held_per_gross:+.4f} (held avg gross "
          f"{held_avg_gross:.2f}) -> the book captured "
          f"{held_per_gross / tot['sig.res']:+.1%} of the signal PnL"
          if abs(tot['sig.res']) > 1e-9 else "capture: n/a (no signal PnL)")

    # ------- where the alpha lives + can the signal book pay its costs -------
    print()
    print("=" * 78)
    print("SIGNAL ANATOMY (fresh-signal IC per horizon slice, non-overlapping)")
    print("=" * 78)
    front, back = None, None
    for lab, v in slice_agg.items():
        v = np.array(v)
        t = (v.mean() / v.std() * np.sqrt(len(v))
             if len(v) > 3 and v.std() > 0 else float('nan'))
        print(f"  {lab:<10} mean IC {v.mean():+.4f}  t {t:+6.1f}  n {len(v)}")
        if front is None:
            front = v.mean()
        back = v.mean()
    if front and back is not None and front > 0:
        if back > 0.5 * front:
            print("  -> alpha is present across the whole horizon grid "
                  "(not front-loaded); the horizon labels are honest")
        else:
            print("  -> alpha concentrates in the shortest slice; long-lag "
                  "promotions are riding front-loaded alpha")

    to_bar = sig_turnover / max(held_bars, 1)
    sig_cost = sig_turnover * COST_RATE
    sig_net = tot['sig.res'] - sig_cost
    print()
    print(f"signal book churn: {to_bar:.3f}/bar of gross ({1 / max(to_bar, 1e-9):.0f} "
          f"bars to fully re-shuffle)")
    print(f"signal book net of {get('portfolio.cost_bps'):.0f}bps/side: "
          f"{tot['sig.res']:+.4f} gross - {sig_cost:.4f} costs = "
          f"{sig_net:+.4f}")
    if sig_net > 0:
        print("-> the signals pay for their own trading at the configured "
              "cost; the gap to the held book is implementation, not alpha")
    else:
        print("-> at the configured cost the signal book's own churn burns "
              "more than its alpha earns; the improvement must come from "
              "signals whose RANKING changes more slowly (lower churn), "
              "not from execution")


if __name__ == '__main__':
    main()
