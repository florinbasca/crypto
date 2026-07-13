"""
ETL for stablecoin circulating supply (DefiLlama, free, no key).

Writes `stablecoin_supply` (date, asset, chain, supply_usd):
  - per-asset totals across all chains (USDT, USDC, DAI)
  - per-chain totals across all stablecoins (Ethereum, Solana, Tron, BSC,
    Arbitrum, Base)

Daily history back to 2017 for USDT. Values are as-of end of `date` (UTC),
so a value stamped D is knowable during D+1; the feature layer applies a
1-day lag on top (publication safety).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import urllib.request

import pandas as pd

from config import get_data_end_date, get_data_start_date
from dbutil import save_data

BASE = 'https://stablecoins.llama.fi'
ASSETS = {'USDT': 1, 'USDC': 2, 'DAI': 5}
CHAINS = ['Ethereum', 'Solana', 'Tron', 'BSC', 'Arbitrum', 'Base']


def _get(url: str):
    req = urllib.request.Request(url, headers={'User-Agent': 'crypto-etl'})
    return json.loads(urllib.request.urlopen(req, timeout=30).read())


def _series(points: list, asset: str, chain: str) -> pd.DataFrame:
    rows = []
    for p in points:
        usd = (p.get('totalCirculatingUSD') or {}).get('peggedUSD')
        if usd is None:
            continue
        rows.append({'date': pd.to_datetime(int(p['date']), unit='s'),
                     'asset': asset, 'chain': chain,
                     'supply_usd': float(usd)})
    return pd.DataFrame(rows)


def main():
    start_dt = pd.Timestamp(get_data_start_date())
    end_dt = pd.Timestamp(get_data_end_date('daily'))
    frames = []
    for asset, sid in ASSETS.items():
        frames.append(_series(
            _get(f'{BASE}/stablecoincharts/all?stablecoin={sid}'),
            asset, 'all'))
    for chain in CHAINS:
        frames.append(_series(
            _get(f'{BASE}/stablecoincharts/{chain}'), 'ALL', chain))

    df = pd.concat(frames, ignore_index=True)
    df = df[(df['date'] >= start_dt) & (df['date'] <= end_dt)]
    df = df.drop_duplicates(['date', 'asset', 'chain'])
    df = df.sort_values(['asset', 'chain', 'date']).reset_index(drop=True)
    if df.empty:
        raise SystemExit("no stablecoin data in the configured window")
    save_data('stablecoin_supply', df, mode='overwrite')

    assert (df['date'] <= pd.Timestamp.now()).all(), "future dates"
    assert (df['supply_usd'] > 0).all(), "non-positive supply values"
    summary = df.groupby(['asset', 'chain']).agg(
        rows=('supply_usd', 'size'), first=('date', 'min'),
        last=('date', 'max'), latest_usd=('supply_usd', 'last'))
    print(summary.to_string())
    print(f"total rows: {len(df):,}")


if __name__ == '__main__':
    main()
