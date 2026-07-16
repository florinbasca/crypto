"""
Inspect what a discovery run generated (read-only; no API, no search).

Reads the three persisted tables and prints a review that leads with what
matters, so the run can be judged without cross-referencing raw rows:

  1. WHAT MATTERS         - computed flags: where the CHOOSE funnel binds,
                            idea concentration among survivors, train->test
                            generalization, degenerate fits
  2. per-roll summary     - trials, survivors, the FILTER FUNNEL (how many
                            survivors passed made-money / activity /
                            economics, then promoted), best net rate
  3. top candidates       - curve verdicts (test AND train) + the DSL
                            program + rationale
  4. promoted book        - everything that made it through CHOOSE
  5. feature usage        - which columns the survivors actually use
                            (idea-concentration view)
  6. LLM usage / cost     - tokens per roll, dollar estimate if priced

Funnel and flags re-price the stored curves with the CURRENT config's
thresholds (cost, min days, capture floor); the run stamp columns make any
config drift attributable. (Discovery is purely statistical - there is no
PnL here. The walk-forward is the only money judge.)

Usage:
    python research/signals/inspect_discovery.py [--top N] [--roll N]
        [--survivors-only] [--expressions]

    --top N            candidates to show (default 10, ranked by reward)
    --roll N           restrict the candidate view to one roll
    --survivors-only   only show search survivors
    --expressions      print the full DSL JSON for each shown candidate
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
from collections import Counter

import numpy as np
import pandas as pd

from config import get
from dbutil import load_data, table_exists
from research.signals.search import (HALF_LIFE_GRID,
                                           effective_persistence_bars,
                                           persistence_weight,
                                           trade_rate_per_bar)


def _load(table: str) -> pd.DataFrame:
    return load_data(table) if table_exists(table) else pd.DataFrame()


def _fmt_expr(cand_json: str, full: bool) -> str:
    c = json.loads(cand_json)
    expr = json.dumps(c['expression'])
    conds = json.dumps(c.get('conditions', []))
    if not full and len(expr) > 90:
        expr = expr[:87] + '...'
    out = f"expr: {expr}"
    if c.get('conditions'):
        out += f"\n      gates: {conds}"
    if c.get('rationale'):
        out += f"\n      why: {c['rationale'][:110]}"
    return out


def _curves(profile_json) -> tuple:
    """(test_curve, train_curve) dicts from a ledger row, or (None, None)."""
    if not isinstance(profile_json, str) or not profile_json:
        return None, None
    try:
        prof = json.loads(profile_json)
    except (ValueError, TypeError):
        return None, None
    return prof.get('curve'), prof.get('curve_train')


def _net_rate(curve, rt_cost: float):
    """Promotion's ranking number: max_k (A(k) - roundtrip) / k."""
    if not curve or not curve.get('ks'):
        return None
    rates = [(float(a) - rt_cost) / int(k)
             for k, a in zip(curve['ks'], curve['A'])
             if a is not None and int(k) > 0]
    return max(rates) if rates else None


def _tstat(curve):
    if not curve:
        return float('nan')
    a0, se = curve.get('a0'), curve.get('se_peak')
    if a0 is None or not se or not np.isfinite(se) or se <= 0:
        return float('nan')
    return float(a0) / float(se)


def _capture(curve, turnover, rate) -> float:
    """Promotion's holdability number: half-life capped at the peak."""
    hl = float(curve.get('half_life') or 1.0)
    if curve.get('peak_k'):
        hl = min(hl, float(curve['peak_k']))
    return persistence_weight(effective_persistence_bars(hl, turnover), rate)


def _cols_used(cand_json) -> set:
    """Feature columns referenced by a candidate (expression + gates)."""
    try:
        c = json.loads(cand_json)
    except (ValueError, TypeError):
        return set()

    def walk(node, out):
        if isinstance(node, (list, tuple)):
            if len(node) == 2 and node[0] == 'col':
                out.add(node[1])
            else:
                for x in node[1:]:
                    walk(x, out)

    out = set()
    walk(c.get('expression'), out)
    for cond in c.get('conditions') or []:
        walk(cond, out)
    return out


def _funnel(led: pd.DataFrame, promo_cfg: dict, curve_cfg: dict,
            rt_cost: float, rate: float) -> pd.DataFrame:
    """Per-roll CHOOSE funnel over the SURVIVORS, re-priced at the current
    config: made money (a0>0 + median) -> active (entry days) -> economics
    (net rate > 0 and capture >= floor) -> promoted (quintile + dedup)."""
    min_days = int(promo_cfg.get('min_select_days', 0))
    min_capture = float(promo_cfg.get('min_capture', 0.0))
    median_gate = bool(curve_cfg.get('median_gate', False))
    rows = []
    for roll_id, grp in led[led['survivor']].groupby('roll_id'):
        n = {'survivors': len(grp), 'money': 0, 'active': 0, 'pays': 0,
             'holdable': 0, 'promoted': int(grp['promoted'].sum())}
        for _, r in grp.iterrows():
            c, _ = _curves(r.get('profile_json'))
            if not c:
                continue
            a0 = float(c.get('a0') or 0.0)
            med = c.get('median_peak')
            if a0 <= 0 or (median_gate and med is not None
                           and np.isfinite(med) and med <= 0):
                continue
            n['money'] += 1
            if int(c.get('entry_days') or 0) < min_days:
                continue
            n['active'] += 1
            nr = _net_rate(c, rt_cost)
            if nr is None or nr <= 0:
                continue
            n['pays'] += 1
            if _capture(c, r.get('turnover'), rate) < min_capture:
                continue
            n['holdable'] += 1
        rows.append({'roll_id': roll_id, **n})
    return pd.DataFrame(rows).set_index('roll_id') if rows else pd.DataFrame()


def _flags(led: pd.DataFrame, funnel: pd.DataFrame,
           promo_cfg: dict) -> list:
    """The 'what matters' lines: only noteworthy, always plain."""
    out = []
    surv = led[led['survivor']]

    # 1. Where the funnel binds, per the latest roll.
    if not funnel.empty:
        f = funnel.iloc[-1]
        drops = {'made money (filter 1)': f['survivors'] - f['money'],
                 'activity (filter 2)': f['money'] - f['active'],
                 'pays for itself (filter 3, cost)': f['active'] - f['pays'],
                 'holdable (filter 3, capture floor)':
                     f['pays'] - f['holdable'],
                 'quintile/duplicates (filter 4)':
                     f['holdable'] - f['promoted']}
        worst = max(drops, key=drops.get)
        if f['promoted'] < int(promo_cfg.get('book_min', 1)):
            out.append(f"BOOK BELOW book_min: roll {funnel.index[-1]} "
                       f"promoted {f['promoted']} of {f['survivors']} "
                       f"survivors; biggest cut is {worst} "
                       f"(-{drops[worst]}).")
        elif drops[worst] > 0:
            out.append(f"Funnel (latest roll): {f['survivors']} survivors -> "
                       f"{f['money']} made money -> {f['active']} active -> "
                       f"{f['pays']} pay for themselves -> {f['holdable']} "
                       f"holdable -> {f['promoted']} promoted; biggest cut "
                       f"is {worst} (-{drops[worst]}).")

    # 2. Idea concentration among survivors.
    if not surv.empty:
        counts = Counter()
        for cj in surv['candidate_json']:
            counts.update(_cols_used(cj))
        if counts:
            col, n = counts.most_common(1)[0]
            share = n / len(surv)
            if share > 1 / 3:
                out.append(f"IDEA CONCENTRATION: {n} of {len(surv)} "
                           f"survivors ({share:.0%}) use '{col}' - the "
                           f"search may be re-mining one mechanism.")

    # 3. Train -> test generalization among survivors.
    gaps = 0
    measured = 0
    for _, r in surv.iterrows():
        c_test, c_train = _curves(r.get('profile_json'))
        if not c_test or not c_train:
            continue
        measured += 1
        if _tstat(c_train) >= 3.0 and float(c_test.get('a0') or 0.0) <= 0:
            gaps += 1
    if measured and gaps:
        out.append(f"OVERFIT WATCH: {gaps} of {measured} survivors had "
                   f"train t >= 3 but a non-positive test edge - strong "
                   f"train fits that did not carry.")

    # 4. Degenerate decay fits (half-life pinned at the grid floor).
    if not surv.empty and 'half_life_bars' in surv.columns:
        hl = surv['half_life_bars'].dropna()
        floor_share = float((hl <= HALF_LIFE_GRID[0]).mean()) if len(hl) else 0.0
        if floor_share > 0.4:
            out.append(f"FAST-DECAY PILEUP: {floor_share:.0%} of survivors "
                       f"fit the minimum half-life ({HALF_LIFE_GRID[0]} "
                       f"bars) - little real persistence structure found.")

    # 5. Reward outlier: the top candidate towering over the field.
    if len(led) > 5:
        rw = led['reward'].astype(float)
        top, med = rw.max(), rw[rw > 0].median()
        if np.isfinite(med) and med > 0 and top > 3 * med:
            name = led.loc[rw.idxmax(), 'name']
            out.append(f"REWARD OUTLIER: {name} scores {top:.1f} vs a "
                       f"positive-median of {med:.1f} - a fat but possibly "
                       f"noisy train edge; check its test t before trusting "
                       f"it.")
    return out


def main():
    parser = argparse.ArgumentParser(
        description='Review the output of a discovery run')
    parser.add_argument('--top', type=int, default=10,
                        help='Candidates to show (default 10, by reward)')
    parser.add_argument('--roll', type=int, default=None,
                        help='Restrict the candidate view to one roll')
    parser.add_argument('--survivors-only', action='store_true',
                        help='Only show search survivors')
    parser.add_argument('--expressions', action='store_true',
                        help='Print full DSL programs (no truncation)')
    args = parser.parse_args()

    tables = get('discovery.tables')
    led = _load(tables['ledger'])
    promos = _load(tables['promotions'])
    usage = _load(tables['llm_usage'])

    if led.empty:
        raise SystemExit(f"No {tables['ledger']} table - run "
                         "research/signals/discovery.py first")

    promo_cfg = get('discovery.promotion')
    curve_cfg = get('discovery.curve', {})
    _cb = promo_cfg.get('econ_cost_bps')
    cost_rate = (float(_cb) if _cb is not None
                 else float(get('portfolio.cost_bps'))) / 10000.0
    rt_cost = float(curve_cfg.get('roundtrip_mult', 2.0)) * cost_rate
    rate = trade_rate_per_bar()

    pd.set_option('display.width', 220)
    pd.set_option('display.float_format', lambda x: f'{x:,.4f}')

    funnel = _funnel(led, promo_cfg, curve_cfg, rt_cost, rate)

    # 1. what matters -------------------------------------------------------
    print("=" * 76)
    print("WHAT MATTERS")
    print("=" * 76)
    flags = _flags(led, funnel, promo_cfg)
    if flags:
        for fl in flags:
            print(f"- {fl}")
    else:
        print("(nothing noteworthy: funnel healthy, ideas diverse, "
              "train edges carried to test)")

    # 2. per-roll summary + funnel ------------------------------------------
    print()
    print("=" * 76)
    print("PER ROLL (funnel re-priced at current config: "
          f"roundtrip {rt_cost * 1e4:.0f}bp, min {promo_cfg['min_select_days']} "
          f"days, capture >= {promo_cfg['min_capture']})")
    print("=" * 76)
    seeded = (led[led['generation'] == -1].groupby('roll_id')['cand_hash']
              .nunique().rename('seeded'))
    summary = led.groupby('roll_id').agg(
        trials=('cand_hash', 'size'),
        best_reward=('reward', 'max'),
    ).join(seeded).fillna({'seeded': 0}).astype({'seeded': int})
    if not funnel.empty:
        summary = summary.join(funnel)
    if 'turnover' in led.columns:
        summary = summary.join(
            led[led['survivor']].groupby('roll_id')['turnover']
            .median().rename('surv_turnover'))
    print(summary.to_string())

    # 3. top candidates ------------------------------------------------------
    view = led if args.roll is None else led[led['roll_id'] == args.roll]
    if args.survivors_only:
        view = view[view['survivor']]
    view = view.sort_values('reward', ascending=False).head(args.top)
    print()
    print("=" * 76)
    scope = f"roll {args.roll}" if args.roll is not None else "all rolls"
    print(f"TOP {len(view)} CANDIDATES BY REWARD ({scope}"
          f"{', survivors only' if args.survivors_only else ''})")
    print("=" * 76)
    for _, r in view.iterrows():
        flags_s = ('PROMOTED' if r['promoted']
                   else 'survivor' if r['survivor'] else '')
        c_test, c_train = _curves(r.get('profile_json'))

        def _cline(c):
            if not c:
                return "no curve"
            nr = _net_rate(c, rt_cost)
            nr_s = (f", net {nr * 144 * 1e4:+.1f}bp/day"
                    if nr is not None else "")
            return (f"a0 {float(c.get('a0') or 0) * 1e4:+.1f}bp/bet "
                    f"(t={_tstat(c):.1f}), peak {c.get('peak_k', '?')}b, "
                    f"hl {c.get('half_life', float('nan')):.0f}b, "
                    f"rev {float(c.get('rev_frac') or 0):.0%}, "
                    f"{c.get('entry_days', '?')}d{nr_s}")

        tv = r.get('turnover')
        tv_s = (f" | turnover {tv:.1%}/bar"
                if tv is not None and np.isfinite(tv) else "")
        print(f"\n{r['name']}  [{r['family']}]  roll {r['roll_id']}  "
              f"dir {r['direction']:+d}  {flags_s}")
        print(f"      reward {r['reward']:.3f}{tv_s}")
        print(f"      test:  {_cline(c_test)}")
        print(f"      train: {_cline(c_train)}")
        print(f"      {_fmt_expr(r['candidate_json'], args.expressions)}")

    # 4. promoted book -------------------------------------------------------
    print()
    print("=" * 76)
    print(f"PROMOTED BOOK ({len(promos)} signals)")
    print("=" * 76)
    if promos.empty:
        print("(nothing promoted yet - promotion takes the best "
              f"{get('discovery.promotion.book_frac'):.0%} of formulas "
              "passing the four filters on their 5-month test; an empty "
              "book means nothing passed)")
    else:
        cols = ['roll_id', 'name', 'family', 'direction', 'select_lag',
                'peak_bars', 'half_life_bars', 'capture', 'turnover',
                'select_alpha_tstat', 'test_days', 'econ_margin', 'reward',
                'n_trials_at_promotion']
        print(promos[[c for c in cols if c in promos.columns]]
              .to_string(index=False))
        print("\n(select_lag/peak_bars/half_life_bars in bars; econ_margin = "
              "net rate per bar; PnL lives in the walk-forward - run "
              "research/portfolio/walk_forward.py)")

    # 5. feature usage among survivors ---------------------------------------
    surv = led[led['survivor']]
    if not surv.empty:
        counts = Counter()
        for cj in surv['candidate_json']:
            counts.update(_cols_used(cj))
        print()
        print("=" * 76)
        print(f"FEATURE USAGE ({len(surv)} survivor rows, "
              "expression + gates)")
        print("=" * 76)
        for col, n in counts.most_common(10):
            print(f"  {n:3d}  ({n / len(surv):4.0%})  {col}")

    # 6. LLM usage -----------------------------------------------------------
    print()
    print("=" * 76)
    print("LLM USAGE")
    print("=" * 76)
    if usage.empty:
        print("(none recorded - random proposer run?)")
    else:
        # Rows persisted before prices were configured have NaN estimates;
        # re-price them from the CURRENT config rates.
        from research.signals.generation import estimate_cost_usd
        llm_cfg = get('discovery.llm')
        usage['est_cost_usd'] = [
            estimate_cost_usd({'input_tokens': r.input_tokens,
                               'output_tokens': r.output_tokens},
                              llm_cfg, r.provider)
            for r in usage.itertuples()
        ]
        print(usage.to_string(index=False))
        priced = usage['est_cost_usd'].notna()
        print(f"totals: {usage['calls'].sum()} calls, "
              f"{usage['input_tokens'].sum():,} in / "
              f"{usage['output_tokens'].sum():,} out tokens"
              + (f", ~${usage['est_cost_usd'].dropna().sum():,.2f}"
                 f" (at current config rates)" if priced.any()
                 else "  (set discovery.llm.price_per_mtok for $ estimates)"))


if __name__ == '__main__':
    main()
