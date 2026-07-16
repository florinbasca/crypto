"""
Inspect what a discovery run generated (read-only; no API, no search).

Reads the three persisted tables and prints a review:
  1. per-roll summary        - trials, survivors, carried-over, promoted
  2. top candidates          - metrics + the actual DSL program + rationale
  3. promotions              - everything that made it through the gates
  4. LLM usage / cost        - tokens per roll, dollar estimate if priced

(Discovery is purely statistical - there is no PnL here. The walk-forward
is the only money judge.)

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

import numpy as np
import pandas as pd

from config import get
from dbutil import load_data, table_exists


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

    pd.set_option('display.width', 220)
    pd.set_option('display.float_format', lambda x: f'{x:,.4f}')

    # 1. per-roll summary ----------------------------------------------------
    print("=" * 76)
    print("PER ROLL")
    print("=" * 76)
    seeded = (led[led['generation'] == -1].groupby('roll_id')['cand_hash']
              .nunique().rename('seeded'))
    summary = led.groupby('roll_id').agg(
        trials=('cand_hash', 'size'),
        survivors=('survivor', 'sum'),
        promoted=('promoted', 'sum'),
        best_select_t=('select_alpha_tstat'
                       if 'select_alpha_tstat' in led.columns
                       else 'select_ic_tstat', lambda s: s.abs().max()),
        best_reward=('reward', 'max'),
    ).join(seeded).fillna({'seeded': 0}).astype({'seeded': int})
    # Median survivor book turnover per bar (diagnostic column; absent on runs
    # from before it was added, NaN on their rows in a resumed run).
    if 'turnover' in led.columns:
        summary = summary.join(
            led[led['survivor']].groupby('roll_id')['turnover']
            .median().rename('surv_turnover'))
    print(summary.to_string())

    # 2. top candidates ------------------------------------------------------
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
        flags = ('PROMOTED' if r['promoted']
                 else 'survivor' if r['survivor'] else '')
        hl = r.get('half_life_bars')
        hl_str = (f" | half-life {hl:,.0f}b"
                  if hl is not None and np.isfinite(hl) else "")
        tv = r.get('turnover')
        tv_str = (f" | turnover {tv:.1%}/bar"
                  if tv is not None and np.isfinite(tv) else "")
        print(f"\n{r['name']}  [{r['family']}]  roll {r['roll_id']}  "
              f"dir {r['direction']:+d}  best lag {r.get('target_lag', '?')}b"
              f"  {flags}")
        print(f"      reward {r['reward']:.3f} | select $/bet "
              f"{r['select_alpha_mean']:+.5f} (t={r['select_alpha_tstat']:.2f}) | "
              f"train t={r['train_alpha_tstat']:.2f} | "
              f"rank IC {r.get('select_rank_ic_mean', float('nan')):.4f} | "
              f"liquid ratio {r.get('select_liquid_alpha_ratio', float('nan')):.2f}"
              f"{hl_str}{tv_str}")
        print(f"      {_fmt_expr(r['candidate_json'], args.expressions)}")

    # 3. promoted book -------------------------------------------------------
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
        print("\n(PnL lives in the walk-forward - run "
              "research/portfolio/walk_forward.py)")

    # 4. LLM usage -----------------------------------------------------------
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
