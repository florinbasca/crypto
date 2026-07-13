"""
ETL for token unlock/emission schedules (self-run DefiLlama
emissions-adapters; see vendor/emissions-adapters + export_schedules.ts).

Input : data_unlock_schedules.json (produced by the exporter:
        {protocol: {sections: [{label, continuous, series: [[unix_s,
        cumulative_unlocked], ...]}]}} at daily resolution, past + forward)
Output: `token_unlocks_daily` (date, symbol, unlocked_amt, cum_pct,
        daily_pct) - the summed-across-sections daily unlock series per
        universe symbol. Forward dates are INCLUDED: an unlock calendar is
        legitimately knowable ahead of time (same convention as the macro
        event calendar).

PIT caveat (documented in the plan): schedules are the CURRENT known
version; revisions are not versioned here. schedule_hash is stored per run
so future snapshots can detect changes.

Run the exporter first:
  cd vendor/emissions-adapters && npx ts-node --transpile-only \
      export_schedules.ts <protocols_csv> ../../data_unlock_schedules.json
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import hashlib
import json
import re

import pandas as pd

from dbutil import load_data, save_data

SCHEDULES_JSON = Path(__file__).parent.parent / 'data_unlock_schedules.json'
ADAPTER_DIR = (Path(__file__).parent.parent
               / 'vendor/emissions-adapters/protocols')


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
    if not SCHEDULES_JSON.exists():
        raise SystemExit(f"{SCHEDULES_JSON} missing - run the exporter first "
                         "(see module docstring)")
    schedules = json.loads(SCHEDULES_JSON.read_text())
    sym_map = symbol_protocol_map()

    rows, failed = [], []
    for symbol, protocol in sorted(sym_map.items()):
        entry = schedules.get(protocol)
        if not entry or entry.get('error') or not entry.get('sections'):
            failed.append((symbol, protocol,
                           (entry or {}).get('error', 'not exported')))
            continue
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
    if failed:
        print(f"not converted ({len(failed)}): "
              + ", ".join(f"{s}({r[:30]})" for s, _, r in failed[:10]))
    assert n_syms >= 20, "suspiciously few symbols converted"
    assert (out['daily_pct'] >= 0).all() and (out['cum_pct'] <= 1.0 + 1e-9).all()


if __name__ == '__main__':
    main()
