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


def _verdict_vs_oos(wf, rolls, res_w: pd.DataFrame) -> None:
    """One line per PROMOTION (a signal promoted at a roll): what its
    5-month test verdict promised vs what the same signal earned in its
    OOS month, measured with the SAME instrument that produced the verdict
    (the response curve on the signal's own gross-1 dollar-neutral book -
    no portfolio sizing, no cost blending). Diagnostic only; separates
    'the search finds nothing' from 'promotion ranks luck' from 'one
    family drags everything'."""
    from research.signals.search import response_curve

    tables = get('discovery.tables')
    promos = load_data(tables['promotions']) if table_exists(
        tables['promotions']) else None
    if promos is None or promos.empty:
        print("(no promotions - nothing to grade)")
        return
    curve_cfg = get('discovery.curve', {})
    H = int(curve_cfg.get('horizon_bars', 144))
    stride = int(curve_cfg.get('entry_stride_bars', 6))
    sample_ks = [int(k) for k in (curve_cfg.get('sample_ks')
                                  or range(1, H + 1))]
    min_assets = int(get('discovery.min_assets_per_timestamp', 10))
    cost_rate = float(get('portfolio.cost_bps')) / 10000.0
    rt = float(curve_cfg.get('roundtrip_mult', 2.0)) * cost_rate
    bar = pd.Timedelta('10min')

    oos_of = {r.roll_id: (pd.Timestamp(r.oos_start), pd.Timestamp(r.oos_end))
              for r in rolls}
    seen: dict = {}
    lines = []
    for roll_id, grp in promos.sort_values('roll_id').groupby('roll_id',
                                                              sort=True):
        if int(roll_id) not in oos_of:
            continue
        o_start, o_end = oos_of[int(roll_id)]
        sel, wts, lag_of, dir_of, metas = {}, {}, {}, {}, []
        for _, p in grp.iterrows():
            name = f"disc_{p['family']}_{str(p['cand_hash'])[:10]}"
            if name not in wf.registry:
                continue
            lag = int(p.get('select_lag') or 0) or H
            sel[name] = [name]
            wts[name] = {name: 1.0}
            lag_of[name] = lag
            dir_of[name] = int(p.get('direction', 1) or 1)
            metas.append((name, p, lag))
        if not metas:
            continue
        # One bucket per signal -> composite_scores returns each signal's
        # own traded-orientation panel (roll direction applied), features
        # loaded once per roll.
        comps = wf.composite_scores(
            sel, wts, o_start - pd.Timedelta(days=WARMUP_DAYS),
            o_start, o_end, lag_of=lag_of, dir_of=dir_of)
        # Paths of late-month entries may spill up to H bars past o_end -
        # same convention as holding a position opened on the last day.
        res_slice = res_w[(res_w.index >= o_start)
                          & (res_w.index < o_end + H * bar)]
        for name, p, lag in metas:
            rep = seen.get(p['cand_hash'], 0) + 1
            seen[p['cand_hash']] = rep
            oos_edge, oos_rate = np.nan, np.nan
            panel = comps.get(name)
            if panel is not None and not panel.empty:
                sig = panel.stack().rename('signal').reset_index()
                sig.columns = ['timestamp', 'symbol', 'signal']
                rc = response_curve(sig, res_slice, H, stride, min_assets,
                                    sample_ks=sample_ks)
                if rc is not None:
                    A = rc['A']
                    k_hold = min(lag, len(A))
                    oos_edge = float(A[k_hold - 1])
                    rates = [(float(A[k - 1]) - rt) / k
                             for k in sample_ks if k <= len(A)]
                    if rates:
                        oos_rate = max(rates)
            lines.append({
                'roll': int(roll_id), 'name': p['name'],
                'family': p['family'], 'rep': rep,
                'test_t': float(p.get('select_alpha_tstat', np.nan)),
                'test_rate': float(p.get('econ_margin', np.nan)),
                'oos_rate': oos_rate, 'oos_edge': oos_edge,
                'agree': (bool(oos_edge > 0) if np.isfinite(oos_edge)
                          else None),
            })

    d = pd.DataFrame(lines)
    if d.empty:
        print("(no gradeable promotions)")
        return

    print()
    print("=" * 78)
    print("VERDICT vs OOS - one line per promotion")
    print("=" * 78)
    print("""columns (rates in bp/day on the gross-1 book; edge in bp/bet):
  test_t     test-curve t (a0/se) that earned the promotion
  test_rate  the verdict: net economic rate on the 5-month test window
  oos_rate   the SAME number measured on the OOS month
  oos_edge   per-bet edge at the promoted holding, OOS month, traded
             direction - 'agree' = it made money the way it was traded
  rep        1 = first promotion, 2+ = re-promoted (consecutive verdicts)
""")
    show = d.copy()
    for c in ('test_rate', 'oos_rate'):
        show[c] = show[c] * BPD_RATE
    show['oos_edge'] = show['oos_edge'] * 1e4
    print(show[['roll', 'name', 'family', 'rep', 'test_t', 'test_rate',
                'oos_rate', 'oos_edge', 'agree']]
          .to_string(index=False, float_format=lambda x: f'{x:+.2f}'))

    v = d.dropna(subset=['oos_edge'])
    if len(v) < 3:
        print("(too few measurable promotions for summary stats)")
        return
    agree = float((v['oos_edge'] > 0).mean())
    spear = float(v['test_rate'].corr(v['oos_rate'], method='spearman'))
    slope = float(np.polyfit(v['test_rate'], v['oos_rate'], 1)[0]) \
        if v['test_rate'].std() > 0 else float('nan')
    print("-" * 78)
    print(f"headline: n={len(v)}  sign-agreement {agree:.0%} (null 50%)  "
          f"spearman(test,oos) {spear:+.2f}  slope {slope:+.2f} "
          f"(1 = verdicts carry, 0 = luck)")

    def _split(label, groups):
        print(f"\nby {label}:")
        for key, g in groups:
            if not len(g):
                continue
            print(f"  {str(key):<18} n={len(g):<3d} "
                  f"agree {float((g['oos_edge'] > 0).mean()):>4.0%}  "
                  f"mean oos_edge {float(g['oos_edge'].mean()) * 1e4:+.1f}"
                  f"bp/bet  mean oos_rate "
                  f"{float(g['oos_rate'].mean()) * BPD_RATE:+.2f}bp/day")

    _split('family', v.groupby('family'))
    _split('repeat', [('1st promotion', v[v['rep'] == 1]),
                      ('re-promoted (2+)', v[v['rep'] >= 2])])
    if v['test_t'].notna().sum() >= 6 and v['test_t'].nunique() > 3:
        try:
            terc = pd.qcut(v['test_t'], 3,
                           labels=['weak t', 'mid t', 'strong t'])
            _split('test-t tercile', v.groupby(terc, observed=True))
        except ValueError:
            pass


BPD_RATE = 144 * 1e4   # per-bar rate -> bp per day


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
        selected, weights, lag_of, dir_of, *_ = wf.month_book(meta)
        comp = wf.composite_scores(
            selected, weights, r.oos_start - pd.Timedelta(days=WARMUP_DAYS),
            r.oos_start, r.oos_end, lag_of=lag_of, dir_of=dir_of)
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

    # ------- realized toll: what a holding cycle ACTUALLY costs ---------------
    # Filter 3 charges every candidate an ASSUMED round trip
    # (roundtrip_mult x cost_bps). The traded book pays cost per unit
    # TRADED, with GP laziness and cross-signal netting included - the
    # honest toll is what it paid per unit gross per holding cycle.
    print()
    print("=" * 78)
    print("REALIZED TOLL (per holding cycle, per unit gross)")
    print("=" * 78)
    if table_exists('wf_portfolio_returns'):
        ret = load_data('wf_portfolio_returns')
        traded = ret[ret['gross_exposure'] > 1e-6]
        cost = float((traded['gross_return'] - traded['net_return']
                      + traded['funding_pnl']).sum())
        gross_bars = float(traded['gross_exposure'].sum())
        promos_t = load_data(get('discovery.tables')['promotions'])
        hold = float(promos_t['select_lag'].median()) if promos_t is not None \
            and not promos_t.empty else 144.0
        cycle_bp = cost / max(gross_bars, 1e-9) * hold * 1e4
        assumed = (get('discovery.curve.roundtrip_mult', 2.0)
                   * get('portfolio.cost_bps'))
        print(f"paid {cost:.4f} over {len(traded):,} traded bars; per unit "
              f"gross per bar {cost / max(gross_bars, 1e-9) * 1e4:.3f}bp; "
              f"x median holding {hold:.0f} bars = {cycle_bp:.1f}bp/cycle "
              f"vs assumed {assumed:.0f}bp")
        print(f"-> honest one-side cost_bps ~ {cycle_bp / 2:.1f}; "
              f"configured portfolio.cost_bps = {get('portfolio.cost_bps')}")
    else:
        print("(no wf_portfolio_returns - run walk_forward.py first)")

    _verdict_vs_oos(wf, rolls, res_w)


if __name__ == '__main__':
    main()
