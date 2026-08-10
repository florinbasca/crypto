"""
Walk-forward analysis: WHERE does the pipeline make or lose money?

One report, four stages, one question each. Every number is in bp/day per
unit gross, so the stages chain into a single "alpha funnel":

  STAGE 1  SELECTION     Do the promoted signals work OOS at all?
                         (per-promotion: test verdict vs OOS outcome)
  STAGE 2  SIGNAL BOOK   Do the combined signals predict their OOS months,
                         and how fast does the edge decay if traded late?
  STAGE 3  CONSTRUCTION  Does the built (held) book capture the signal
                         book's alpha, before costs? Is the gap explained
                         by trading speed or by construction losses?
  STAGE 4  COSTS         Does the held book survive trading costs and
                         funding? Is turnover or participation the problem?

Each stage ends with a VERDICT (what works, what does not, why) and ACTION
lines (what to do about it), all derived from the measured numbers. The
report closes with the alpha funnel and a BOTTOM LINE: net result, the
causal story, and the consolidated to-do list in pipeline order.

Books measured (all real, persisted or recomputed objects):
  - SIGNAL book: the month's composite ranks as dollar-neutral gross-1
    weights, re-formed every bar. A pure function of the signals: no
    optimizer, no caps, no costs. This is what the signals OFFERED.
  - HELD book: the backtest's persisted per-bar weights
    (wf_portfolio_weights). This is what the portfolio CAPTURED.

Run: uv run research/portfolio/walk_forward_analysis.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import logging

import numpy as np
import pandas as pd

from config import (config as global_config, get, BARS_PER_DAY,
                    BASE_FREQUENCY)
from dbutil import load_data, table_exists
from research.portfolio.walk_forward import WalkForwardPortfolio, WARMUP_DAYS

logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True,
                    format=global_config['logging']['format'],
                    datefmt=global_config['logging']['datefmt'])

BPD = BARS_PER_DAY * 1e4                      # per-bar return -> bp/day
COST_RATE = get('portfolio.cost_bps') / 1e4   # per side, per unit traded
# Implementation-lag grid for the decay view: the backtest's actual lag
# first, then the discovery holding grid.
DECAY_LAGS = sorted({0, int(get('walk_forward.implementation_lag_bars', 1))}
                    | {int(x) for x in get('discovery.horizon_lags_bars')})
# Display threshold: a leg is called MATERIAL when it moves the reference
# alpha by more than this fraction (report formatting only, not a model
# parameter).
MATERIAL = 0.2
# Minimum promotions before the report says anything about a family.
FAMILY_MIN_N = 3


# ---------------------------------------------------------------- helpers
def book_pnl(w: pd.DataFrame, fwd: pd.DataFrame) -> float:
    """Sum of w[t] . fwd[t] over the common index/columns (NaN = 0)."""
    a, b = w.align(fwd, join='inner')
    return float(np.nansum(a.values * b.values))


def gross1(w: pd.DataFrame) -> pd.DataFrame:
    """Scale each bar's weights to gross exposure 1."""
    g = w.abs().sum(axis=1).replace(0, np.nan)
    return w.div(g, axis=0)


def bpd(pnl: float, bars: float) -> float:
    """Total pnl over `bars` bars -> bp/day (per unit gross for gross-1)."""
    return pnl / max(bars, 1) * BPD


def tstat(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 3 or x.std(ddof=1) == 0:
        return float('nan')
    return float(x.mean() / x.std(ddof=1) * np.sqrt(len(x)))


def decay_kept(decay: dict, lag: float) -> float:
    """Fraction of fresh (lag-0) alpha left when the book is `lag` bars
    stale, linearly interpolated on the measured decay grid."""
    lags = sorted(decay)
    fresh = decay.get(0, np.nan)
    if not np.isfinite(fresh) or fresh <= 0:
        return float('nan')
    vals = [decay[L] / fresh for L in lags]
    return float(np.interp(lag, lags, vals))


def alpha_shape(decay: dict) -> tuple:
    """(best_lag, best_alpha, backloaded) from a {lag: alpha} decay grid.
    backloaded = the alpha at some positive lag MATERIALLY beats the fresh
    alpha, i.e. the edge grows with staleness instead of decaying."""
    best_lag = max(decay, key=lambda L: decay[L])
    best = decay[best_lag]
    backloaded = (best > 0 and best_lag > 0
                  and best > max(decay.get(0, 0.0), 0.0) * (1 + MATERIAL))
    return best_lag, best, backloaded


def alpha_half_life(decay: dict) -> float:
    """First lag (bars) at which the alpha falls below half of fresh;
    inf if it never does within the grid."""
    lags = sorted(decay)
    fresh = decay.get(0, np.nan)
    if not np.isfinite(fresh) or fresh <= 0:
        return float('nan')
    prev_l, prev_f = 0, 1.0
    for L in lags[1:]:
        f = decay[L] / fresh
        if f < 0.5:
            if prev_f == f:
                return float(L)
            return float(prev_l + (prev_f - 0.5) / (prev_f - f) * (L - prev_l))
        prev_l, prev_f = L, f
    return float('inf')


def fmt_bars(bars: float) -> str:
    if bars == float('inf'):
        return f"beyond the {max(DECAY_LAGS)}-bar grid"
    if not np.isfinite(bars):
        return "n/a"
    return f"{bars:.0f} bars ({bars / BARS_PER_DAY:.1f} days)"


# ---------------------------------------------------------------- data
def load_returns_panels():
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
    # weights at t earn the (t, t+1] bar: forward residual = residual at t+1
    return res_w, res_w.shift(-1), raw_fwd


def load_held_weights() -> pd.DataFrame:
    if not table_exists('wf_portfolio_weights'):
        raise SystemExit("no wf_portfolio_weights - run walk_forward.py first")
    hw = load_data('wf_portfolio_weights')
    hw['timestamp'] = pd.to_datetime(hw['timestamp'])
    return hw.pivot_table(index='timestamp', columns='symbol',
                          values='weight', aggfunc='first').sort_index()


def collect_months(wf, rolls, res_fwd, raw_fwd, held) -> list:
    """One dict per OOS month: signal book and held book measured on the
    same forward returns. This is the raw material for stages 2 and 3."""
    months = []
    for r in rolls:
        meta = wf.month_meta.get(pd.Timestamp(r.oos_start), [])
        if not meta:
            continue
        logging.info(f"measuring OOS month {pd.Timestamp(r.oos_start).date()}"
                     f" ({len(meta)} signals)")
        selected, weights, lag_of, dir_of, *_ = wf.month_book(meta)
        comp = wf.composite_scores(
            selected, weights, r.oos_start - pd.Timedelta(days=WARMUP_DAYS),
            r.oos_start, r.oos_end, lag_of=lag_of, dir_of=dir_of)
        if not comp:
            continue
        z = None
        for c in comp.values():
            z = c if z is None else z.add(c, fill_value=0.0)
        sig = gross1(z.sub(z.mean(axis=1), axis=0))
        sig = sig.loc[(sig.index >= r.oos_start) & (sig.index < r.oos_end)]
        hb = held.loc[(held.index >= r.oos_start) & (held.index < r.oos_end)]

        decay = {L: (book_pnl(sig.shift(L), res_fwd), max(len(sig) - L, 0))
                 for L in DECAY_LAGS}
        months.append({
            'oos': str(pd.Timestamp(r.oos_start).date()),
            'n_sig': len(meta),
            'sig_bars': len(sig),
            'sig_res': decay[0][0],
            'sig_turnover': float(sig.diff().abs().sum(axis=1).iloc[1:].sum()),
            'decay': decay,                       # {lag: (pnl, bars)}
            'held_bars': len(hb),
            'held_gross': float(hb.abs().sum(axis=1).sum()),
            'held_res': book_pnl(hb, res_fwd),
            'held_raw': book_pnl(hb, raw_fwd),
        })
    return months


# ---------------------------------------------------------------- stage 1
def stage1_selection(wf, rolls, res_w) -> dict:
    """Grade every promotion: the 5-month test verdict (the number that
    earned the promotion) vs the same measurement on its OOS month, using
    the same instrument (response curve of the signal's own gross-1
    dollar-neutral book; no portfolio, no sizing)."""
    from research.signals.search import response_curve

    print()
    print("=" * 72)
    print("STAGE 1  SELECTION: do the promoted signals work OOS?")
    print("=" * 72)

    tables = get('discovery.tables')
    promos = (load_data(tables['promotions'])
              if table_exists(tables['promotions']) else None)
    if promos is None or promos.empty:
        print("(no promotions: nothing to grade)")
        return {}

    curve_cfg = get('discovery.curve', {})
    H = int(curve_cfg.get('horizon_bars', 144))
    stride = int(curve_cfg.get('entry_stride_bars', 6))
    sample_ks = [int(k) for k in (curve_cfg.get('sample_ks')
                                  or range(1, H + 1))]
    min_assets = int(get('discovery.min_assets_per_timestamp', 10))
    rt_cost = float(curve_cfg.get('roundtrip_mult', 2.0)) * COST_RATE
    bar = pd.Timedelta(BASE_FREQUENCY)

    oos_of = {r.roll_id: (pd.Timestamp(r.oos_start), pd.Timestamp(r.oos_end))
              for r in rolls}
    groups = [(rid, grp) for rid, grp in
              promos.sort_values('roll_id').groupby('roll_id', sort=True)
              if int(rid) in oos_of]
    n_promos = sum(len(g) for _, g in groups)
    logging.info(f"stage 1: grading {n_promos} promotions across "
                 f"{len(groups)} rolls (features reload + response curve "
                 f"per promotion; this is the slow part)")
    seen: dict = {}
    lines = []
    for roll_id, grp in groups:
        o_start, o_end = oos_of[int(roll_id)]
        # One singleton bucket per signal so composite_scores returns each
        # signal's own traded-orientation panel (features loaded once/roll).
        sel, wts, lag_of, dir_of, metas = {}, {}, {}, {}, []
        for _, p in grp.iterrows():
            name = f"disc_{p['family']}_{str(p['cand_hash'])[:10]}"
            if name not in wf.registry:
                continue
            lag = int(p.get('select_lag') or 0) or H
            sel[name], wts[name] = [name], {name: 1.0}
            lag_of[name] = lag
            dir_of[name] = int(p.get('direction', 1) or 1)
            metas.append((name, p, lag))
        if not metas:
            continue
        logging.info(f"stage 1: roll {int(roll_id)} "
                     f"(OOS {o_start.date()}), {len(metas)} promotions")
        comps = wf.composite_scores(
            sel, wts, o_start - pd.Timedelta(days=WARMUP_DAYS),
            o_start, o_end, lag_of=lag_of, dir_of=dir_of)
        # Late-month entries may hold up to H bars past o_end.
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
                    oos_edge = float(A[k_hold - 1])   # per-bet, traded dir
                    # best net per-bar rate over the holding grid, same
                    # instrument as the promotion verdict (econ_margin)
                    rates = [(float(A[k - 1]) - rt_cost) / k
                             for k in sample_ks if k <= len(A)]
                    if rates:
                        oos_rate = max(rates)
            lines.append({
                'roll': int(roll_id), 'name': p['name'],
                'family': p['family'], 'rep': rep,
                'test_rate': float(p.get('econ_margin', np.nan)),
                'oos_rate': oos_rate, 'oos_edge': oos_edge,
            })

    if not lines:
        print("(no measurable promotions)")
        return {}
    d = pd.DataFrame(lines).dropna(subset=['oos_edge'])
    if len(d) < 3:
        print("(too few measurable promotions to grade)")
        return {}

    n = len(d)
    worked = float((d['oos_edge'] > 0).mean())
    # binomial z against the coin-flip null of 50% working by luck
    z = (worked - 0.5) * np.sqrt(n) / 0.5
    spear = float(d['test_rate'].corr(d['oos_rate'], method='spearman'))
    test_bpd = float(d['test_rate'].mean()) * BPD
    oos_bpd = float(d['oos_rate'].mean()) * BPD
    kept = oos_bpd / test_bpd if abs(test_bpd) > 1e-12 else np.nan

    print(f"promotions graded: {n}  (signal promoted for an OOS month, "
          f"measured on that month)")
    print(f"worked OOS (made money as traded):  {worked:.0%}   "
          f"(coin flip = 50%, z = {z:+.1f})")
    print(f"test-vs-OOS rank correlation:       {spear:+.2f}   "
          f"(1 = verdicts carry over, 0 = luck)")
    print(f"promised edge (test window):        {test_bpd:+.2f} bp/day")
    print(f"delivered edge (OOS month):         {oos_bpd:+.2f} bp/day   "
          f"(kept {kept:.0%})" if np.isfinite(kept) else
          f"delivered edge (OOS month):         {oos_bpd:+.2f} bp/day")

    def _split(label, groups):
        print(f"  by {label}:")
        for key, g in groups:
            if len(g) == 0:
                continue
            print(f"    {str(key):<18} n={len(g):<3d} worked "
                  f"{float((g['oos_edge'] > 0).mean()):>4.0%}  delivered "
                  f"{float(g['oos_rate'].mean()) * BPD:+.2f} bp/day")

    _split('family', d.groupby('family'))
    _split('repetition', [('1st promotion', d[d['rep'] == 1]),
                          ('re-promoted', d[d['rep'] >= 2])])

    # Data-driven family lists (n >= FAMILY_MIN_N to say anything).
    fam = d.groupby('family').agg(
        n=('oos_edge', 'size'), worked=('oos_edge', lambda s: (s > 0).mean()),
        delivered=('oos_rate', 'mean'))
    fam = fam[fam['n'] >= FAMILY_MIN_N]
    dead = fam[(fam['worked'] <= 0.25) & (fam['delivered'] < 0)]
    good = fam[(fam['worked'] >= 0.6) & (fam['delivered'] > 0)]

    actions = []
    if z > 2 and oos_bpd > 0:
        verdict = ("selection WORKS: promoted signals beat the coin flip "
                   "OOS and keep a usable share of the promised edge")
    else:
        verdict = ("selection FAILS on the full pool: the promotion verdict "
                   "does not separate real signals from luck "
                   f"(rank corr {spear:+.2f}, worked {worked:.0%})")
        actions.append(
            "replace/augment the promotion verdict (econ_margin, test t): "
            f"it has ~no OOS ranking power (rank corr {spear:+.2f})")
    if len(dead):
        actions.append(
            "stop promoting families with no OOS delivery: "
            + ", ".join(f"{f} ({int(r.n)} tries, worked {r.worked:.0%})"
                        for f, r in dead.iterrows()))
    if len(good):
        actions.append(
            "point the discovery budget at the families that deliver OOS: "
            + ", ".join(f"{f} (worked {r.worked:.0%}, "
                        f"{r.delivered * BPD:+.1f} bp/day)"
                        for f, r in good.iterrows()))
    print(f"VERDICT: {verdict}")
    for a in actions:
        print(f"ACTION: {a}")
    return {'test_bpd': test_bpd, 'oos_bpd': oos_bpd, 'worked': worked,
            'spearman': spear, 'n': n, 'z': z, 'actions': actions}


# ---------------------------------------------------------------- stage 2
def stage2_signal(months: list) -> dict:
    print()
    print("=" * 72)
    print("STAGE 2  SIGNAL BOOK: do the combined signals predict OOS?")
    print("=" * 72)
    print("(signal book = promoted signals combined into dollar-neutral "
          "gross-1\n weights each bar; no optimizer, no caps, no costs)")

    monthly = np.array([bpd(m['sig_res'], m['sig_bars']) for m in months])
    total_bars = sum(m['sig_bars'] for m in months)
    alpha = bpd(sum(m['sig_res'] for m in months), total_bars)
    hit = float((monthly > 0).mean())
    print(f"alpha on residual returns: {alpha:+.2f} bp/day   "
          f"t = {tstat(monthly):+.1f} across {len(monthly)} months, "
          f"{hit:.0%} of months positive")

    # Decay: the same book traded L bars late.
    agg = {L: (sum(m['decay'][L][0] for m in months),
               sum(m['decay'][L][1] for m in months)) for L in DECAY_LAGS}
    decay = {L: bpd(p, b) for L, (p, b) in agg.items()}
    print("alpha if every trade happens L bars late:")
    for L in sorted(decay):
        if L == 0:
            continue
        keptL = decay[L] / decay[0] if decay[0] > 0 else np.nan
        kept_s = f"(keeps {keptL:.0%})" if np.isfinite(keptL) else ""
        print(f"    {L:>4} bars late ({L / BARS_PER_DAY * 24:>4.1f}h): "
              f"{decay[L]:+.2f} bp/day  {kept_s}")

    best_lag, best, backloaded = alpha_shape(decay)
    hl = alpha_half_life(decay)
    if backloaded:
        print("alpha shape: BACK-LOADED, it grows with lag "
              f"(best {best:+.2f} bp/day at {best_lag} bars stale)")
    else:
        print(f"alpha half-life: {fmt_bars(hl)}")

    # Can the signal book pay its own trading bill?
    churn = sum(m['sig_turnover'] for m in months) / max(total_bars, 1)
    own_cost = churn * COST_RATE * BPD
    own_net = alpha - own_cost
    print(f"signal book churn: {churn:.3f}/bar of gross "
          f"(full reshuffle every {fmt_bars(2 / max(churn, 1e-9))})")
    print(f"traded 1:1 at {get('portfolio.cost_bps'):.0f} bp/side it nets "
          f"{alpha:+.2f} - {own_cost:.2f} = {own_net:+.2f} bp/day")

    actions = []
    if best <= 0:
        verdict = ("NO ALPHA at any measured staleness: the combined "
                   "signals do not predict their OOS months")
        actions.append("nothing to construct or execute; fix stage 1 "
                       "(selection) first")
    elif backloaded:
        verdict = ("alpha is BACK-LOADED: entry timing is noise, the edge "
                   "accrues over days; trading slow costs nothing and "
                   "earns more")
        max_lag = max(DECAY_LAGS)
        if best_lag == max_lag:
            actions.append(
                f"the edge is still growing at the {max_lag}-bar end of "
                f"the grid; extend discovery.horizon_lags_bars past "
                f"{max_lag} bars to find where it peaks, and score/hold "
                f"at those longer lags")
        actions.append(
            f"do not chase fresh scores: the fresh book pays {own_cost:.2f}"
            f" bp/day in churn for LESS alpha than a stale one; smooth or "
            f"lag the signal at the source")
        if tstat(monthly) < 2:
            actions.append(
                f"treat the level with care: monthly t = {tstat(monthly):+.1f}"
                f", the signal book's edge is not yet statistically solid")
    elif own_net > 0:
        verdict = ("alpha is REAL and pays its own trading bill; any "
                   "further loss is construction or execution")
    else:
        verdict = ("alpha is real but decays and CHURNS TOO FAST to pay "
                   "the configured cost")
        actions.append(
            "need signals whose RANKING moves more slowly (lower churn at "
            "the source); no execution layer can outrun a decaying, "
            "fast-churning signal")
    print(f"VERDICT: {verdict}")
    for a in actions:
        print(f"ACTION: {a}")
    return {'alpha': alpha, 'decay': decay, 'half_life': hl,
            'backloaded': backloaded, 'best_lag': best_lag, 'best': best,
            'own_net': own_net, 'monthly': monthly, 'actions': actions}


# ---------------------------------------------------------------- stage 3
def stage3_construction(months: list, s2: dict, ret) -> dict:
    print()
    print("=" * 72)
    print("STAGE 3  CONSTRUCTION: does the held book capture that alpha? "
          "(pre-cost)")
    print("=" * 72)

    gross_bars = sum(m['held_gross'] for m in months)
    held_res = sum(m['held_res'] for m in months)
    held_raw = sum(m['held_raw'] for m in months)
    held_bpd = held_res / max(gross_bars, 1e-9) * BPD
    raw_bpd = held_raw / max(gross_bars, 1e-9) * BPD
    sig_bpd = s2['alpha']
    capture = held_bpd / sig_bpd if abs(sig_bpd) > 1e-12 else np.nan

    monthly = np.array([m['held_res'] / m['held_gross'] * BPD
                        for m in months if m['held_gross'] > 0])
    print(f"signal book:              {sig_bpd:+.2f} bp/day per unit gross")
    print(f"held book (residual):     {held_bpd:+.2f} bp/day per unit gross"
          + (f"   -> captures {capture:.0%} of the signal"
             if np.isfinite(capture) else ""))
    print(f"held book significance:   t = {tstat(monthly):+.1f} across "
          f"{len(monthly)} months, {float((monthly > 0).mean()):.0%} of "
          f"months positive")
    leak = raw_bpd - held_bpd
    leak_ok = abs(leak) < MATERIAL * max(abs(held_bpd), 1e-9)
    print(f"held book (raw, actual):  {raw_bpd:+.2f} bp/day per unit gross"
          f"   hedge leak {leak:+.2f} bp/day: "
          + ("clean" if leak_ok else "MATERIAL, check factor neutrality"))

    # Is the capture gap explained by trading speed alone? The held book's
    # average position age ~ half its holding time; the signal decay curve
    # says how much alpha survives that staleness.
    expected_kept = np.nan
    hold_bars = np.nan
    if ret is not None:
        to_sum = float(ret['turnover'].sum())
        gr_sum = float(ret['gross_exposure'].sum())
        if to_sum > 0:
            hold_bars = 2.0 * gr_sum / to_sum   # in+out = 2x gross traded
            expected_kept = decay_kept(s2['decay'], hold_bars / 2.0)
    if np.isfinite(hold_bars):
        print(f"held book speed: a position lives ~{fmt_bars(hold_bars)}; "
              f"at that staleness the signal itself still keeps "
              f"~{expected_kept:.0%} of its alpha"
              if np.isfinite(expected_kept) else
              f"held book speed: a position lives ~{fmt_bars(hold_bars)}")

    actions = []
    if not leak_ok:
        actions.append("close the hedge leak: held pnl differs on raw vs "
                       "residual returns; check betas/neutrality bands")
    if not np.isfinite(capture) or sig_bpd <= 0:
        if held_bpd > 0 and s2.get('backloaded'):
            verdict = ("the fresh signal book has no alpha but the held "
                       "book earns: its edge comes from trading the "
                       "back-loaded part of the alpha slowly, not from "
                       "fresh signal ranks")
        else:
            verdict = "n/a: no signal alpha to capture (see stage 2)"
    elif capture <= 0:
        verdict = ("construction DESTROYS the alpha: the held book loses "
                   "money on a signal that makes money")
        actions.append("look at sizing, neutrality bands and caps before "
                       "costs; the loss happens before any trading cost")
    elif capture > 1 + MATERIAL:
        exp_s = (f"~{expected_kept:.0%} of that is explained by the "
                 f"alpha being worth more at the book's staleness"
                 if np.isfinite(expected_kept) else
                 "no speed-based expectation available")
        verdict = (f"held book earns MORE than its signal inputs "
                   f"({capture:.0%} capture); {exp_s}; the remainder is "
                   f"construction (vol sizing, caps, bucket weights) or "
                   f"luck")
        actions.append(
            "verify the over-capture is not luck: run the null controls "
            "(walk_forward.py --control shuffle / sign_flip) and require "
            "the real book to clearly beat them")
    elif np.isfinite(expected_kept) and capture < expected_kept - MATERIAL:
        verdict = (f"construction UNDER-CAPTURES: speed alone predicts "
                   f"~{expected_kept:.0%} capture but the book gets "
                   f"{capture:.0%}")
        actions.append("the extra loss is construction (caps/neutrality/"
                       "sizing), not slowness; relax whichever binds")
    elif (np.isfinite(expected_kept) and expected_kept < 0.5
          and not s2.get('backloaded')):
        verdict = (f"the book is TOO SLOW for this alpha: holding "
                   f"{fmt_bars(hold_bars)} vs alpha half-life "
                   f"{fmt_bars(s2['half_life'])}; capture {capture:.0%} "
                   f"is what slowness predicts")
        actions.append("speed the book up (trade rate / participation) or "
                       "select slower alpha; pick one deliberately")
    else:
        verdict = (f"construction is FINE: capture {capture:.0%} is in "
                   f"line with the book's speed")
    print(f"VERDICT: {verdict}")
    for a in actions:
        print(f"ACTION: {a}")
    return {'held_bpd': held_bpd, 'raw_bpd': raw_bpd, 'capture': capture,
            'hold_bars': hold_bars, 'expected_kept': expected_kept,
            'held_t': tstat(monthly), 'actions': actions}


# ---------------------------------------------------------------- stage 4
def stage4_costs(ret) -> dict:
    print()
    print("=" * 72)
    print("STAGE 4  COSTS: does the held book survive trading?")
    print("=" * 72)
    if ret is None:
        print("(no wf_portfolio_returns: run walk_forward.py first)")
        return {}

    traded = ret[ret['gross_exposure'] > 1e-6]
    gr_sum = float(traded['gross_exposure'].sum())
    gross = float(traded['gross_return'].sum())
    net = float(traded['net_return'].sum())
    funding = float(traded['funding_pnl'].sum())
    cost = gross + funding - net                 # net = gross - cost + funding
    to_sum = float(traded['turnover'].sum())

    def per_gross(x):
        return x / max(gr_sum, 1e-9) * BPD

    ann = np.sqrt(BARS_PER_DAY * 365)
    std = float(traded['net_return'].std())
    sharpe = (float(traded['net_return'].mean()) / std * ann
              if std > 0 else np.nan)

    print(f"gross:    {per_gross(gross):+.2f} bp/day per unit gross")
    print(f"costs:    {per_gross(-cost):+.2f} bp/day")
    print(f"funding:  {per_gross(funding):+.2f} bp/day")
    print(f"net:      {per_gross(net):+.2f} bp/day   "
          f"(annualized Sharpe {sharpe:+.2f})")

    hold_bars = 2.0 * gr_sum / to_sum if to_sum > 0 else np.nan
    to_day = to_sum / max(gr_sum, 1e-9) * BARS_PER_DAY
    print(f"turnover: {to_day:.2f}x gross per day  "
          f"-> a position lives ~{fmt_bars(hold_bars)}")
    breakeven = gross / to_sum * 1e4 if to_sum > 0 else np.nan
    cfg_cost = float(get('portfolio.cost_bps'))
    print(f"breakeven cost (gross pnl / $ traded): {breakeven:.1f} bp/side "
          f"vs configured {cfg_cost:.1f} bp/side")

    part_cfg = get('portfolio.participation', {})
    cap = float(part_cfg.get('max_participation', np.nan))
    pm = traded['participation_max'].dropna()
    if len(pm) and np.isfinite(cap):
        binding = float((pm >= cap * (1 - 1e-6)).mean())
        print(f"participation: peak {pm.median():.4f} median / {pm.max():.4f}"
              f" max of bar volume (cap {cap}); cap binds on "
              f"{binding:.0%} of traded bars")
    else:
        binding = np.nan

    actions = []
    if net > 0:
        headroom = breakeven / max(cfg_cost, 1e-9)
        verdict = (f"the book MAKES MONEY net of costs and funding; cost "
                   f"headroom {headroom:.1f}x (fills can be {headroom:.1f}x"
                   f" worse before net hits zero)")
    elif gross <= 0:
        verdict = ("gross is already negative: the problem is UPSTREAM "
                   "(stage 3), not costs")
        actions.append("do not tune execution: there is no gross alpha "
                       "for it to protect")
    else:
        need = cfg_cost / max(breakeven, 1e-9)
        verdict = (f"gross alpha is real but TRADING EATS IT: each unit "
                   f"traded earns {breakeven:.1f} bp vs {cfg_cost:.1f} bp "
                   f"cost")
        actions.append(f"need ~{need:.1f}x cheaper fills or ~{need:.1f}x "
                       f"less turnover at the same gross pnl")
    if np.isfinite(binding) and binding > MATERIAL:
        actions.append(
            f"the participation cap binds on {binding:.0%} of traded bars:"
            f" the book is at liquidity capacity for the configured "
            f"book_size_usd; do not scale the book up and expect these "
            f"numbers to hold")
    print(f"VERDICT: {verdict}")
    for a in actions:
        print(f"ACTION: {a}")
    return {'gross_bpd': per_gross(gross), 'cost_bpd': per_gross(cost),
            'funding_bpd': per_gross(funding), 'net_bpd': per_gross(net),
            'sharpe': sharpe, 'breakeven': breakeven,
            'turnover_per_day': to_day, 'binding': binding,
            'actions': actions}


# ---------------------------------------------------------------- funnel
def print_funnel(s1: dict, s2: dict, s3: dict, s4: dict) -> None:
    print()
    print("=" * 72)
    print("ALPHA FUNNEL (bp/day per unit gross)")
    print("=" * 72)
    steps = []
    if s1:
        steps.append(("1 selection promised (test window)", s1['test_bpd']))
        steps.append(("1 selection delivered OOS (per signal)",
                      s1['oos_bpd']))
    if s2:
        steps.append(("2 combined signal book, fresh", s2['alpha']))
    if s3:
        steps.append(("3 held book before costs", s3['held_bpd']))
    if s4:
        steps.append(("4 net after costs + funding", s4['net_bpd']))

    prev = None
    biggest, biggest_drop = None, 0.0
    died = None
    for label, val in steps:
        keep = ""
        if prev is not None and np.isfinite(prev) and prev > 0 \
                and np.isfinite(val):
            keep = f"   kept {val / prev:.0%}"
            if prev - val > biggest_drop:
                biggest_drop, biggest = prev - val, label
        if died is None and np.isfinite(val) and val <= 0:
            died = label
        print(f"  {label:<40} {val:+.2f}{keep}")
        prev = val
    if died:
        print(f"-> first stage in the red: {died}")
    elif biggest:
        print(f"-> everything survives; biggest leak: {biggest} "
              f"(-{biggest_drop:.2f} bp/day)")


def print_bottom_line(s1: dict, s2: dict, s3: dict, s4: dict) -> None:
    """The one-paragraph answer: does the book make money, why, and what
    to do next. Every sentence is derived from the stage metrics."""
    print()
    print("=" * 72)
    print("BOTTOM LINE")
    print("=" * 72)
    net = s4.get('net_bpd', np.nan)
    n_months = len(s2.get('monthly', []))
    if np.isfinite(net):
        print(f"the book {'MAKES' if net > 0 else 'LOSES'} {net:+.2f} "
              f"bp/day per unit gross net of costs and funding "
              f"(annualized Sharpe {s4['sharpe']:+.2f}, {n_months} OOS "
              f"months)")
    selection_fails = bool(s1) and not (s1['z'] > 2 and s1['oos_bpd'] > 0)
    if np.isfinite(net) and net > 0 and selection_fails:
        helpers = ["combining survivors into one diversified book"]
        if s2.get('backloaded'):
            helpers.append("a slow execution layer that fits the "
                           "back-loaded alpha")
        print("it does so DESPITE selection, not because of it: the "
              "promotion verdict carries no OOS information, and the "
              "profit is produced downstream by "
              + ", ".join(helpers)
              + "; that chain is fragile because nothing guarantees the "
                "downstream layers keep compensating")
    elif np.isfinite(net) and net > 0:
        print("the pipeline is healthy end to end: selection carries OOS "
              "and the downstream stages keep most of it")
    elif np.isfinite(net):
        print("read the funnel above: fix the first stage in the red "
              "before tuning anything after it")
    actions = [a for s in (s1, s2, s3, s4) for a in s.get('actions', [])]
    if actions:
        print("WHAT TO DO, in pipeline order (upstream fixes first):")
        for i, a in enumerate(actions, 1):
            print(f"  {i}. {a}")


# ---------------------------------------------------------------- main
def main():
    wf = WalkForwardPortfolio()
    from research.signals.data import make_rolls
    rolls = make_rolls(get('discovery'))

    res_w, res_fwd, raw_fwd = load_returns_panels()
    held = load_held_weights()
    ret = (load_data('wf_portfolio_returns')
           if table_exists('wf_portfolio_returns') else None)

    months = collect_months(wf, rolls, res_fwd, raw_fwd, held)
    if not months:
        raise SystemExit("no OOS months with promoted signals")

    # Sanity: the held book recomputed on raw forwards must reproduce the
    # backtest's own persisted gross pnl, or nothing below is trustworthy.
    if ret is not None:
        mine = sum(m['held_raw'] for m in months)
        theirs = float(ret['gross_return'].sum())
        ok = abs(mine - theirs) < max(0.02, 0.1 * abs(theirs))
        print(f"sanity: held book on raw forwards {mine:+.4f} vs persisted "
              f"gross {theirs:+.4f} -> "
              + ("MATCH" if ok else "MISMATCH, investigate before reading on"))

    # Per-month overview, one row per OOS month, all in bp/day.
    rows = []
    for m in months:
        rows.append({
            'oos': m['oos'], 'n_sig': m['n_sig'],
            'signal': bpd(m['sig_res'], m['sig_bars']),
            'held_gross': (m['held_res'] / m['held_gross'] * BPD
                           if m['held_gross'] > 0 else np.nan),
        })
    df = pd.DataFrame(rows)
    print()
    print("per OOS month (bp/day per unit gross): 'signal' = what the "
          "signals\noffered, 'held_gross' = what the book captured before "
          "costs")
    print(df.to_string(index=False, float_format=lambda x: f'{x:+.2f}'))

    s1 = stage1_selection(wf, rolls, res_w)
    s2 = stage2_signal(months)
    s3 = stage3_construction(months, s2, ret)
    s4 = stage4_costs(ret)
    print_funnel(s1, s2, s3, s4)
    print_bottom_line(s1, s2, s3, s4)


if __name__ == '__main__':
    main()
