"""
Agentic signal discovery - the deterministic harness (see signal.md).

Discovery is PURELY STATISTICAL: it measures each candidate's per-bet return
(alpha, in return units; rank IC kept as a diagnostic), fits alpha
half-lives, and emits promotions. It never charges costs or trades -
research/portfolio/walk_forward.py is the ONLY money judge.

Outer loop per roll (train 5mo / select 1mo, advancing monthly; the roll's
OOS month exists only as the promotion's valid_from date):
  1. build diagnostics on TRAIN (compressed - the proposer's entire view)
  2. SEARCH: budgeted propose -> compile -> evaluate -> reward -> keep
     best+diverse survivors (evolutionary loop, family bandit). The search
     is TRAIN-ONLY: reward, survival, breeding and direction never see the
     select window. The reward's alpha term is the candidate's
     PER-BET RETURN (not rank IC), CAPTURE-WEIGHTED (x 1/(1 + phi/kappa)), so
     persistent signals outscore equally-strong fast ones.
  3. PROMOTE survivors through the gates - the first and only look at
     select (per-lag profile FDR, deflation over the actual looks,
     persistence, sign agreement, capture floor, orthogonality)
  4. roll forward. Output: the promotions table, consumed by the
     walk-forward via research/lib/discovered.py.

The LLM (config discovery.llm.provider) is the idea generator inside step 2 -
everything it emits is re-validated, compiled, causality-checked and scored
by fixed code, so it can only ever waste budget, not corrupt results. Every
run is a FRESH start (discovery tables cleared first); --no-fresh keeps
existing tables. Ledger and promotions are flushed EVERY roll - a run
killed at roll 20 keeps its first 20 rolls.

Usage:
    python research/signals/discovery.py
        [--max-rolls N] [--no-fresh] [--resume] [--no-save]

    --resume continues an interrupted run: keeps the tables and skips the
    rolls already completed (the ledger flushes per roll, so on-disk rolls
    are complete), picking up at the first missing roll.

NO PINNING: every candidate is evaluated at every lag in
discovery.horizon_lags_bars (train AND select) - the per-lag profile is its
alpha term structure, and its fitted half-life sets the capture weight in
the reward here and the persistence discount in the walk-forward.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import logging
import time

import pandas as pd

from config import config as global_config, get
from research.signals import promotion as bt
from research.signals import data as data_mod
from research.signals import search as search_mod
from research.signals.generation import Candidate, make_proposer

# WARNING level: the run's narrative is the print/tqdm output; warnings
# (proposer retries, salvage) stay visible.
logging.basicConfig(level=logging.WARNING,
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
                    'statistical promotion (the walk-forward is the money judge)')
    parser.add_argument('--max-rolls', type=int, default=0,
                        help='Only run the first N rolls (0 = all)')
    parser.add_argument('--no-fresh', action='store_true',
                        help='Keep existing discovery tables (default is a '
                             'fresh start: tables cleared before running)')
    parser.add_argument('--resume', action='store_true',
                        help='Continue an interrupted run: keep existing '
                             'tables and SKIP already-completed rolls, picking '
                             'up at the first roll not in the ledger (seeded '
                             'from the last completed roll\'s survivors)')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not persist ledger/promotions/returns')
    args = parser.parse_args()

    cfg = get('discovery')
    tables = cfg['tables']
    save = not args.no_save
    # --resume implies --no-fresh (never wipe the completed rolls).
    fresh = not (args.no_fresh or args.resume)

    if fresh and save:
        from dbutil import delete_table
        for t in tables.values():
            delete_table(t)
        # legacy scoreboard table (discovery no longer produces PnL)
        delete_table('discovery_oos_returns')
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

    # --resume: skip rolls already fully completed (the ledger flushes only at
    # the END of a roll, so every roll on disk is complete). Continue at the
    # first missing roll; the loop below re-seeds it from the previous roll's
    # survivors via ledger.survivor_candidates.
    if args.resume:
        done = ledger.to_frame()
        last_done = int(done['roll_id'].max()) if not done.empty else -1
        rolls = [r for r in rolls if r.roll_id > last_done]
        if not rolls:
            print(f"Resume: all rolls already complete (last = {last_done}); "
                  "nothing to do.")
            return
        print(f"Resume: {last_done + 1} rolls already complete; continuing "
              f"from roll {rolls[0].roll_id} ({len(rolls)} remaining)")

    proposer = make_proposer('llm')
    provider = getattr(proposer, 'provider', '')
    if provider:
        print(f"LLM proposer: {provider} / {proposer.model}")

    promo_rows = []
    usage_rows = []
    seeds = []         # previous roll's survivors, re-tested each new roll

    for roll in rolls:
        print(f"\n=== roll {roll.roll_id}: train {roll.train_start.date()} "
              f"select {roll.select_start.date()} OOS {roll.oos_start.date()}"
              f"..{roll.oos_end.date()} ===")

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
            from research.signals.generation import estimate_cost_usd
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

        # THIS roll's book: re-formed from scratch every roll - a signal
        # trades the OOS month only if it re-qualified on the 6 months of
        # data ending just before it. No carryover, no lifetime membership;
        # failing to re-qualify IS the demotion.
        book = bt.promote(survivors, roll, ledger, cfg)
        print(f"promoted this roll: {len(book)} signals")
        roll_promo_rows = []
        for p in book:
            c = p['candidate']
            roll_promo_rows.append({
                'roll_id': roll.roll_id, 'cand_hash': c.hash, 'name': c.name,
                'family': c.family, 'direction': p['direction'],
                'target_lag': int(p.get('target_lag', 0) or 0),
                'half_life_bars': float(p.get('half_life_bars', 0) or 0),
                'capture': float(p.get('capture', 0) or 0),
                'turnover': float(p.get('turnover', float('nan'))),
                'promoted_lags': ','.join(str(l) for l in
                                          p.get('promoted_lags', [])),
                'candidate_json': c.to_json(),
                # Select alpha t at the STRONGEST promoted lag (the evidence
                # the signal actually cleared on), not the best-TRAIN-lag t.
                'select_alpha_tstat': p.get('select_alpha_tstat',
                                            p['metrics_select']['alpha_tstat']),
                'reward': p['reward'],
                'n_looks_at_promotion': p.get('n_looks_at_promotion', 0),
                'n_trials_at_promotion': p['n_trials_at_promotion'],
            })
            print(f"  + {c.name} ({c.family}) "
                  f"select $t={roll_promo_rows[-1]['select_alpha_tstat']:.2f} "
                  f"@ lag {p.get('select_lag', p.get('target_lag'))} "
                  f"of [{roll_promo_rows[-1]['promoted_lags']}] "
                  f"half-life {p.get('half_life_bars', 0):,.0f} bars "
                  f"(capture {p.get('capture', 0):.2f}, "
                  f"turnover {p.get('turnover', float('nan')):.3f}/bar)")
        promo_rows.extend(roll_promo_rows)

        # Flush EVERY roll (ledger, promotions AND llm usage): a run killed
        # at roll N keeps rolls 0..N-1, and inspect sees usage mid-run.
        if save:
            ledger.flush()
            _save(tables['promotions'], pd.DataFrame(roll_promo_rows),
                  fresh=False, save=True)
            if usage_rows and usage_rows[-1]['roll_id'] == roll.roll_id:
                _save(tables['llm_usage'], pd.DataFrame(usage_rows[-1:]),
                      fresh=False, save=True)

    print(f"\n=== promotions ===")
    if not promo_rows:
        print("Nothing promoted. (Statistical gates only - see "
              "inspect_discovery.py for near-misses.)")
    else:
        pf = pd.DataFrame(promo_rows)
        print(f"{len(pf)} promotions across "
              f"{pf['roll_id'].nunique()} rolls, "
              f"{pf['cand_hash'].nunique()} distinct signals; "
              f"half-life range "
              f"{pf['half_life_bars'].min():,.0f}-"
              f"{pf['half_life_bars'].max():,.0f} bars")
        print("Next: uv run research/portfolio/walk_forward.py "
              "(the only money judge)")

    # Always shown, every run: what this run cost in LLM tokens/dollars.
    if usage_rows:
        from research.signals.generation import estimate_cost_usd
        total = {k: sum(r[k] for r in usage_rows)
                 for k in ('calls', 'input_tokens', 'output_tokens')}
        cost = estimate_cost_usd(total, cfg['llm'], provider)
        cost_str = (f", ~${cost:,.2f}" if cost is not None
                    else " (set discovery.llm.price_per_mtok for a $ estimate)")
        print(f"\n=== LLM cost ({provider} / {proposer.model}) ===")
        print(f"{total['calls']} calls, {total['input_tokens']:,} in / "
              f"{total['output_tokens']:,} out tokens{cost_str}")
    else:
        print("\n=== LLM cost ===\n$0.00 (no API calls)")

    # ledger/promotions/usage were all flushed per roll.
    if save:
        ledger.flush()
        print(f"Saved: {tables['ledger']}, {tables['promotions']}, "
              f"{tables['llm_usage']}")


if __name__ == '__main__':
    main()
