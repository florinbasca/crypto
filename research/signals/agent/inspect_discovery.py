"""
Inspect what a discovery run generated (read-only; no API, no search).

Reads the four persisted tables and prints a review:
  1. per-roll summary        - trials, survivors, carried-over, promoted
  2. top candidates          - metrics + the actual DSL program + rationale
  3. promoted book           - everything that made it through the gates
  4. OOS equity curve        - stitched daily PnL of the promoted book
  5. LLM usage / cost        - tokens per roll, dollar estimate if priced

Usage:
    python research/signals/agent/inspect_discovery.py [--top N] [--roll N]
        [--survivors-only] [--expressions] [--curve]

    --top N            candidates to show (default 10, ranked by reward)
    --roll N           restrict the candidate view to one roll
    --survivors-only   only show search survivors
    --expressions      print the full DSL JSON for each shown candidate
    --curve            print the full daily OOS PnL table (default: summary)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

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
    parser.add_argument('--curve', action='store_true',
                        help='Print the full daily OOS PnL table')
    args = parser.parse_args()

    tables = get('discovery.tables')
    led = _load(tables['ledger'])
    promos = _load(tables['promotions'])
    oos = _load(tables['oos_returns'])
    usage = _load(tables['llm_usage'])

    if led.empty:
        raise SystemExit(f"No {tables['ledger']} table - run "
                         "research/signals/agent/run_discovery.py first")

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
        best_select_t=('select_ic_tstat', lambda s: s.abs().max()),
        best_reward=('reward', 'max'),
    ).join(seeded).fillna({'seeded': 0}).astype({'seeded': int})
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
        print(f"\n{r['name']}  [{r['family']}]  roll {r['roll_id']}  "
              f"dir {r['direction']:+d}  {flags}")
        print(f"      reward {r['reward']:.3f} | select IC "
              f"{r['select_ic_mean']:.4f} (t={r['select_ic_tstat']:.2f}) | "
              f"train t={r['train_ic_tstat']:.2f} | "
              f"net Sharpe {r['select_net_sharpe']:.2f} | "
              f"turnover {r['select_turnover']:.2f}")
        print(f"      {_fmt_expr(r['candidate_json'], args.expressions)}")

    # 3. promoted book -------------------------------------------------------
    print()
    print("=" * 76)
    print(f"PROMOTED BOOK ({len(promos)} signals)")
    print("=" * 76)
    if promos.empty:
        print("(nothing promoted yet - candidates must survive "
              f"{get('discovery.promotion.min_rolls_survived')} consecutive "
              "rolls and clear FDR/deflation/orthogonality)")
    else:
        cols = ['roll_id', 'name', 'family', 'direction', 'select_ic_tstat',
                'reward', 'n_trials_at_promotion']
        print(promos[[c for c in cols if c in promos.columns]]
              .to_string(index=False))

    # 4. OOS equity curve ----------------------------------------------------
    print()
    print("=" * 76)
    print("OOS EQUITY CURVE (stitched, the search never saw these months)")
    print("=" * 76)
    if oos.empty:
        print("(no OOS returns - nothing was in the book during any OOS month)")
    else:
        daily = oos.groupby('date', as_index=False)[
            ['gross', 'cost', 'funding', 'net']].sum().sort_values('date')
        net = daily['net']
        sharpe = (net.mean() / net.std() * np.sqrt(365)
                  if len(net) > 2 and net.std() > 0 else 0.0)
        dd = (net.cumsum() - net.cumsum().cummax()).min()
        print(f"{len(daily)} days: net {net.sum() * 1e4:,.1f} bps | "
              f"ann. Sharpe {sharpe:.2f} | max drawdown {dd * 1e4:,.1f} bps | "
              f"costs {daily['cost'].sum() * 1e4:,.1f} bps | "
              f"funding {daily['funding'].sum() * 1e4:,.1f} bps")
        if args.curve:
            daily['cum_net_bps'] = daily['net'].cumsum() * 1e4
            print(daily.to_string(index=False))

    # 5. LLM usage -----------------------------------------------------------
    print()
    print("=" * 76)
    print("LLM USAGE")
    print("=" * 76)
    if usage.empty:
        print("(none recorded - random proposer run?)")
    else:
        # Rows persisted before prices were configured have NaN estimates;
        # re-price them from the CURRENT config rates.
        from research.signals.agent.generation import estimate_cost_usd
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
