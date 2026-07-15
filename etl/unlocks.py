"""
ETL for token unlock/emission schedules (self-run DefiLlama
emissions-adapters). The adapter repo is a gitignored clone under
vendor/ (NOT tracked); only the exporter etl/export_schedules.ts is ours.

SELF-CONTAINED: this script runs the whole chain itself - it derives the
protocol list from the universe, copies the exporter into the vendor repo,
runs it via npx ts-node (installing the repo's node deps on first use),
then converts the JSON. No manual TypeScript step. --no-export converts an
existing data_unlock_schedules.json without touching node.

Intermediate: data_unlock_schedules.json ({protocol: {sections: [{label,
        continuous, series: [[unix_s, cumulative_unlocked], ...]}],
        dropped?}} at daily resolution, past + forward)
Output: `token_unlocks_daily` (date, symbol, unlocked_amt, cum_pct,
        daily_pct) - the summed-across-sections daily unlock series per
        universe symbol. Forward dates are INCLUDED: an unlock calendar is
        legitimately knowable ahead of time (same convention as the macro
        event calendar).

PIT caveat (documented in the plan): schedules are the CURRENT known
version; revisions are not versioned here. schedule_hash is stored per run
so future snapshots can detect changes.

One-time setup (upstream DefiLlama/emissions-adapters went private, hence
the fork; everything after the clone is automatic):
  git clone https://github.com/Omni-Chain-Protocols/emissions-adapters \
      vendor/emissions-adapters
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import hashlib
import json
import re
import shutil
import subprocess

import pandas as pd

from dbutil import load_data, save_data

SCHEDULES_JSON = Path(__file__).parent.parent / 'data_unlock_schedules.json'
VENDOR_DIR = Path(__file__).parent.parent / 'vendor/emissions-adapters'
ADAPTER_DIR = VENDOR_DIR / 'protocols'
EXPORTER_SRC = Path(__file__).parent / 'export_schedules.ts'


def run_exporter(protocols: list) -> None:
    """Run the TS exporter end-to-end: node deps on first use, fresh copy of
    our exporter into the vendor repo (imports resolve there), npx ts-node.
    Raises SystemExit with setup instructions when the vendor clone or node
    are missing."""
    if not (VENDOR_DIR / 'package.json').exists():
        raise SystemExit(
            "vendor/emissions-adapters missing - one-time setup:\n"
            "  git clone https://github.com/Omni-Chain-Protocols/"
            "emissions-adapters vendor/emissions-adapters\n"
            "then rerun this script (node deps install automatically).")
    if shutil.which('npx') is None:
        raise SystemExit("node/npx not found on PATH - install node, then "
                         "rerun (the exporter is TypeScript).")
    if not (VENDOR_DIR / 'node_modules').exists():
        print("installing vendor node deps (first use, one-time)...")
        subprocess.run(['npm', 'install', '--ignore-scripts'],
                       cwd=VENDOR_DIR, check=True)
        subprocess.run(['npm', 'install', '-D', 'typescript@5.3'],
                       cwd=VENDOR_DIR, check=True)
    shutil.copy2(EXPORTER_SRC, VENDOR_DIR / 'export_schedules.ts')
    print(f"exporting {len(protocols)} protocol schedules "
          "(network-bound, a few minutes)...")
    subprocess.run(
        ['npx', 'ts-node', '--transpile-only', 'export_schedules.ts',
         ','.join(sorted(protocols)), str(SCHEDULES_JSON.resolve())],
        cwd=VENDOR_DIR, check=True)


def symbol_protocol_map() -> dict:
    """Universe symbol -> adapter protocol name, via the same CoinGecko-id
    mapping etl/marketcap.py maintains."""
    from etl.marketcap import COINGECKO_ID_MAP
    u = sorted(set(load_data('universe')['symbol'].str.upper()))
    cached = load_data('coingecko_ids')
    ids = {**COINGECKO_ID_MAP,
           **dict(zip(cached['symbol'], cached['coingecko_id']))}
    protos = {re.sub(r'\.ts$', '', p.name).lower():
              re.sub(r'\.ts$', '', p.name) for p in ADAPTER_DIR.glob('*.ts')}
    out = {}
    for s in u:
        cid = str(ids.get(s, s)).lower()
        hit = protos.get(cid) or protos.get(s.lower())
        if hit:
            out[s] = hit
    return out


def main():
    parser = argparse.ArgumentParser(
        description='Token unlock schedules: runs the TS exporter and '
                    'converts to token_unlocks_daily (see module docstring)')
    parser.add_argument('--no-export', action='store_true',
                        help='Skip the exporter; convert the existing '
                             'data_unlock_schedules.json as-is')
    args = parser.parse_args()

    sym_map = symbol_protocol_map()
    if args.no_export:
        if not SCHEDULES_JSON.exists():
            raise SystemExit(f"{SCHEDULES_JSON} missing - rerun without "
                             "--no-export")
    else:
        run_exporter(sorted(set(sym_map.values())))
    schedules = json.loads(SCHEDULES_JSON.read_text())

    rows, failed, partial = [], [], []
    for symbol, protocol in sorted(sym_map.items()):
        entry = schedules.get(protocol)
        if not entry or entry.get('error') or not entry.get('sections'):
            failed.append((symbol, protocol,
                           (entry or {}).get('error', 'not exported')))
            continue
        # Exporter salvage: broken sections (usually staking-reward feeds)
        # were dropped so the vesting cliffs survive - keep it visible.
        if entry.get('dropped'):
            partial.append((symbol, entry['dropped']))
        cum = None
        for sec in entry['sections']:
            if not sec['series']:
                continue
            s = pd.Series({pd.to_datetime(int(t), unit='s').normalize(): v
                           for t, v in sec['series']}).sort_index()
            cum = s if cum is None else cum.add(s, fill_value=0.0)
        if cum is None or cum.empty or cum.max() <= 0:
            failed.append((symbol, protocol, 'empty schedule'))
            continue
        cum = cum.groupby(level=0).last()
        total = float(cum.max())
        daily = cum.diff().fillna(cum.iloc[0]).clip(lower=0.0)
        df = pd.DataFrame({
            'date': cum.index, 'symbol': symbol,
            'unlocked_amt': daily.values,
            'cum_pct': (cum / total).values,
            'daily_pct': (daily / total).values,
        })
        rows.append(df)

    if not rows:
        raise SystemExit("no schedules converted - inspect the exporter log")
    out = pd.concat(rows, ignore_index=True)
    out = out.sort_values(['symbol', 'date']).reset_index(drop=True)
    save_data('token_unlocks_daily', out, mode='overwrite')

    sched_hash = hashlib.sha256(
        SCHEDULES_JSON.read_bytes()).hexdigest()[:16]
    n_syms = out['symbol'].nunique()
    print(f"token_unlocks_daily: {len(out):,} rows, {n_syms} symbols, "
          f"{out['date'].min().date()} -> {out['date'].max().date()} "
          f"(forward dates included), schedule_hash={sched_hash}")
    if partial:
        print(f"partial ({len(partial)}, broken sections dropped by the "
              "exporter): "
              + ", ".join(f"{s}(-{','.join(d)})" for s, d in partial[:10]))
    if failed:
        print(f"not converted ({len(failed)}): "
              + ", ".join(f"{s}({r[:30]})" for s, _, r in failed[:10]))
    assert n_syms >= 20, "suspiciously few symbols converted"
    assert (out['daily_pct'] >= 0).all() and (out['cum_pct'] <= 1.0 + 1e-9).all()


if __name__ == '__main__':
    main()
