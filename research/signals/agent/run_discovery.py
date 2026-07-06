"""
Agentic signal discovery - the deterministic harness (see agent.md).

Outer loop per roll (train 5mo / select 1mo / OOS 1mo, advancing monthly):
  1. build diagnostics on TRAIN (compressed - the proposer's entire view)
  2. SEARCH: budgeted propose -> compile -> evaluate(train->select) -> reward
     -> keep best+diverse survivors (evolutionary loop, family bandit)
  3. PROMOTE survivors through the gates (FDR, deflation, persistence,
     orthogonality vs the book)
  4. BACKTEST the promoted book through the OOS month (dollar+factor neutral,
     ex-post costs, funding) - the search never saw this month
  5. roll forward; stitched OOS months = the equity curve

The LLM is only the idea generator inside step 2. Defaults: the proposer is
the config LLM (discovery.llm.provider, currently gemini) and every run is a
FRESH start (discovery tables cleared first). --proposer random is the no-API
baseline / control experiment; --no-fresh keeps existing tables.

Usage:
    python research/signals/agent/run_discovery.py
        [--proposer random|llm|anthropic|gemini] [--max-rolls N]
        [--ml-probe] [--no-fresh] [--no-save] [--target-lag BARS]

The search is MULTI-LAG by default: every candidate is evaluated at every lag
in discovery.search_lags_bars ('all' -> the horizon_lags_bars grid) on TRAIN
and pinned to its strongest horizon there, so one run finds signals wherever
on the speed spectrum they live. --target-lag restricts to a single lag
(writes to _h<lag>-suffixed tables) for focused runs.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import argparse
import logging
import time

import pandas as pd

from config import config as global_config, get
from research.signals.agent import backtest as bt
from research.signals.agent import data as data_mod
from research.signals.agent import search as search_mod
from research.signals.agent.generation import Candidate, make_proposer

logging.basicConfig(level=logging.INFO,
                    format=global_config['logging']['format'],
                    datefmt=global_config['logging']['datefmt'])


def _resolve_columns(cfg):
    from dbutil import get_table_columns
    available = get_table_columns('features')
    if not available:
        raise SystemExit("features table unavailable - run the feature ETL first")
    family_columns = data_mod.resolve_family_columns(available, cfg)
    for family, cols in family_columns.items():
        print(f"  {family:18s} {len(cols):3d} columns")
    return family_columns


def _save(table: str, df: pd.DataFrame, fresh: bool, save: bool):
    if not save or df.empty:
        return
    from dbutil import save_data, delete_table
    if fresh:
        delete_table(table)
    save_data(table, df, mode='append')


def main():
    parser = argparse.ArgumentParser(
        description='Agentic signal discovery: bounded-DSL search + '
                    'walk-forward promotion + OOS backtest')
    parser.add_argument('--proposer', default='llm',
                        choices=['random', 'llm', 'anthropic', 'gemini'],
                        help='Candidate generator (default llm = provider '
                             'from config discovery.llm.provider, currently '
                             'gemini). random = no-API baseline / control; '
                             'or name the provider explicitly')
    parser.add_argument('--max-rolls', type=int, default=0,
                        help='Only run the first N rolls (0 = all)')
    parser.add_argument('--ml-probe', action='store_true',
                        help='Also fit the GBM predictability ceiling per roll')
    parser.add_argument('--no-fresh', action='store_true',
                        help='Keep existing discovery tables (default is a '
                             'fresh start: tables cleared before running)')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not persist ledger/promotions/returns')
    parser.add_argument('--target-lag', type=int, default=0,
                        help='RESTRICT the search to one lag (must be in '
                             'discovery.horizon_lags_bars); output tables get '
                             'a _h<lag> suffix. Default 0 = multi-lag: every '
                             'candidate is scored at every lag in '
                             'discovery.search_lags_bars and pinned to its '
                             'strongest one on TRAIN - one run searches the '
                             'whole speed spectrum.')
    args = parser.parse_args()

    cfg = get('discovery')
    if args.target_lag:
        allowed = [int(x) for x in cfg['horizon_lags_bars']]
        if args.target_lag not in allowed:
            raise SystemExit(
                f"--target-lag {args.target_lag} is not in "
                f"discovery.horizon_lags_bars {allowed}: the panel only "
                "builds forward targets for those lags (add it there first)")
        # Shallow copy: this run scores at the override lag only and writes
        # to suffixed tables; the global config object stays untouched.
        cfg = {**cfg,
               'search_lags_bars': [int(args.target_lag)],
               'target_lag_bars': int(args.target_lag),
               'tables': {k: f"{v}_h{args.target_lag}"
                          for k, v in cfg['tables'].items()}}
        print(f"Target-lag override: search restricted to {args.target_lag} "
              f"bars, tables suffixed _h{args.target_lag}")
    tables = cfg['tables']
    save = not args.no_save
    fresh = not args.no_fresh

    if fresh and save:
        from dbutil import delete_table
        for t in tables.values():
            delete_table(t)
        print("Fresh start: discovery tables cleared")

    print("Resolving the bounded input space from the features table:")
    family_columns = _resolve_columns(cfg)
    feature_cols = data_mod.all_family_columns(family_columns)

    print(f"Building the panel ({len(feature_cols)} features + targets "
          f"{cfg['horizon_lags_bars']} bars)...")
    t0 = time.perf_counter()
    panel = data_mod.build_panel(feature_cols, cfg)
    print(f"Panel ready: {len(panel):,} rows, "
          f"{panel['symbol'].nunique()} symbols "
          f"({time.perf_counter() - t0:,.1f}s)")

    rolls = data_mod.make_rolls(cfg)
    if args.max_rolls > 0:
        rolls = rolls[:args.max_rolls]
    print(f"{len(rolls)} rolls: train {cfg['train_months']}mo / "
          f"select {cfg['select_months']}mo / OOS {cfg['oos_months']}mo")

    ledger = search_mod.DiscoveryLedger(tables['ledger'] if save else None)
    proposer = make_proposer(args.proposer)
    provider = getattr(proposer, 'provider', '')
    if provider:
        print(f"LLM proposer: {provider} / {proposer.model}")

    book = []          # currently promoted entries (candidate, direction, ...)
    oos_results = []
    promo_rows = []
    usage_rows = []
    seeds = []         # previous roll's survivors, re-tested each new roll

    for roll in rolls:
        print(f"\n=== roll {roll.roll_id}: train {roll.train_start.date()} "
              f"select {roll.select_start.date()} OOS {roll.oos_start.date()}"
              f"..{roll.oos_end.date()} ===")

        if args.ml_probe:
            probe = search_mod.run_ml_probe(panel, roll, feature_cols, cfg)
            for lag_i, m in sorted(probe['metrics_by_lag'].items()):
                print(f"ML ceiling @ {lag_i:>3d} bars: IC {m['ic_mean']:.4f} "
                      f"(t={m['ic_tstat']:.2f}, "
                      f"net Sharpe {m['net_sharpe']:.2f})")

        if not seeds and roll.roll_id > 0:
            # resumed/partial runs: recover the previous roll's survivors
            seeds = ledger.survivor_candidates(roll.roll_id - 1)

        usage_before = proposer.usage_snapshot()
        survivors = search_mod.run_search(panel, roll, family_columns,
                                          proposer, ledger, cfg,
                                          seed_candidates=seeds)
        n_reseeded = len({s['candidate'].hash for s in survivors}
                         & {c.hash for c in seeds})
        print(f"search: {ledger.n_trials(roll.roll_id)} candidates tried, "
              f"{len(survivors)} survivors "
              f"({n_reseeded} carried over from the previous roll)")
        seeds = [s['candidate'] for s in survivors]

        usage_after = proposer.usage_snapshot()
        roll_usage = {k: usage_after[k] - usage_before[k] for k in usage_after}
        if roll_usage['calls'] > 0:
            from research.signals.agent.generation import estimate_cost_usd
            cost = estimate_cost_usd(roll_usage, cfg['llm'], provider)
            cost_str = f", ~${cost:,.2f}" if cost is not None else ""
            print(f"LLM usage: {roll_usage['calls']} calls, "
                  f"{roll_usage['input_tokens']:,} in / "
                  f"{roll_usage['output_tokens']:,} out tokens{cost_str}")
            usage_rows.append({
                'roll_id': roll.roll_id, 'provider': provider,
                'model': proposer.model, **roll_usage,
                'est_cost_usd': cost,
            })

        # Recompile the current book on THIS roll's select window so the
        # orthogonality gate compares like with like.
        pb = data_mod.purge_bars(cfg)
        roll_panel = data_mod.slice_window(panel, roll.train_start,
                                           roll.oos_start, 0).reset_index(drop=True)
        select_start = roll.select_start
        for entry in book:
            from research.signals.agent.generation import compile_candidate
            sig = compile_candidate(entry['candidate'], roll_panel)
            sig['signal'] *= entry['direction']
            entry['signal_select'] = sig[sig['timestamp'] >= select_start]

        promoted = bt.promote(survivors, roll, ledger, book, cfg)
        book.extend(promoted)
        print(f"promoted {len(promoted)} (book size {len(book)})")
        for p in promoted:
            c = p['candidate']
            promo_rows.append({
                'roll_id': roll.roll_id, 'cand_hash': c.hash, 'name': c.name,
                'family': c.family, 'direction': p['direction'],
                'target_lag': int(p.get('target_lag', 0) or 0),
                'candidate_json': c.to_json(),
                'select_ic_tstat': p['metrics_select']['ic_tstat'],
                'reward': p['reward'],
                'n_trials_at_promotion': p['n_trials_at_promotion'],
            })
            print(f"  + {c.name} ({c.family}) "
                  f"t={p['metrics_select']['ic_tstat']:.2f} "
                  f"@ {p.get('target_lag', '?')} bars")

        result = bt.backtest_oos(panel, roll, book, cfg)
        result['roll_id'] = roll.roll_id
        if len(result['daily']) > 0:
            result['daily']['roll_id'] = roll.roll_id
            net = result['daily']['net'].sum()
            print(f"OOS month: net {net * 1e4:,.1f} bps over "
                  f"{result['stamps']} rebalances; "
                  f"mean |exposures| { {k: round(v, 4) for k, v in result['exposures'].items()} }")
        else:
            print("OOS month: no book to trade")
        oos_results.append(result)

        if save:
            ledger.flush()

    curve = bt.stitch_oos(oos_results)
    print("\n=== stitched OOS equity curve ===")
    if curve.empty:
        print("No promoted book was ever traded.")
    else:
        total = curve['net'].sum()
        daily = curve['net']
        sharpe = search_mod._annualized_sharpe(daily.values)
        print(f"{len(curve)} days, net {total * 1e4:,.1f} bps, "
              f"annualized Sharpe {sharpe:.2f}, "
              f"costs {curve['cost'].sum() * 1e4:,.1f} bps, "
              f"funding {curve['funding'].sum() * 1e4:,.1f} bps")

    # Always shown, every run: what this run cost in LLM tokens/dollars.
    if usage_rows:
        from research.signals.agent.generation import estimate_cost_usd
        total = {k: sum(r[k] for r in usage_rows)
                 for k in ('calls', 'input_tokens', 'output_tokens')}
        cost = estimate_cost_usd(total, cfg['llm'], provider)
        cost_str = (f", ~${cost:,.2f}" if cost is not None
                    else " (set discovery.llm.price_per_mtok for a $ estimate)")
        print(f"\n=== LLM cost ({provider} / {proposer.model}) ===")
        print(f"{total['calls']} calls, {total['input_tokens']:,} in / "
              f"{total['output_tokens']:,} out tokens{cost_str}")
    else:
        print("\n=== LLM cost ===\n$0.00 (no API calls - random proposer)")

    _save(tables['llm_usage'], pd.DataFrame(usage_rows), fresh, save)
    _save(tables['promotions'], pd.DataFrame(promo_rows), fresh, save)
    _save(tables['oos_returns'],
          pd.concat([r['daily'] for r in oos_results if len(r['daily'])],
                    ignore_index=True) if any(len(r['daily']) for r in oos_results)
          else pd.DataFrame(), fresh, save)
    if save:
        ledger.flush()
        print(f"Saved: {tables['ledger']}, {tables['promotions']}, "
              f"{tables['oos_returns']}")


if __name__ == '__main__':
    main()
