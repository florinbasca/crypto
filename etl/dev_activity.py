"""
ETL for developer activity (Electric Capital Open Dev Data, CC BY 4.0).

Source: https://data.opendevdata.org/manifest.json -> eco_mads.parquet
(daily Monthly-Active-Developer metrics per ecosystem: 28d-window dev
counts, tenure and commitment segments) + ecosystems.parquet (id -> name).
Free, no key; snapshots regenerated periodically (~75MB download).

Mapping: universe symbol -> ecosystem name, first via CoinGecko ids
(etl/marketcap.py), then MANUAL_ECOSYSTEM_MAP for renames/rebrands.
Memecoins with no dev ecosystem are legitimately unmapped -> NaN features.

Output: `dev_activity` (date, symbol, all_devs, exclusive_devs,
full_time_devs, one_time_devs, devs_2y_plus, num_commits).

PIT caveat (same class as unlocks): each snapshot is built with the
CURRENT repo taxonomy, so historical rows can be restated when repos are
added to an ecosystem later. Commits themselves are public within days;
the feature layer applies a 30-day lag as publication/backfill margin.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import tempfile
import urllib.request

import pandas as pd

from dbutil import load_data, save_data

MANIFEST_URL = 'https://data.opendevdata.org/manifest.json'
BASE_URL = 'https://data.opendevdata.org'
# CDN 403s the default python User-Agent
_UA = {'User-Agent': 'crypto-statarb-etl/1.0'}

# Universe symbol -> Electric Capital ecosystem name, where the CoinGecko
# id doesn't match the taxonomy name (renames, rebrands, "Network" suffix).
MANUAL_ECOSYSTEM_MAP = {
    '0G': '0g Labs',
    'ATOM': 'Cosmos Network',
    'AVAX': 'Avalanche',
    'BABY': 'Babylon Chain',
    'BERA': 'Berachain Foundation',
    'BLUR': 'Blur Exchange',
    'CAKE': 'PancakeSwap',
    'CFX': 'Conflux Network',
    'COMP': 'Compound',
    'CRV': 'Curve',
    'DOT': 'Polkadot Network',
    'ETHFI': 'ether.fi',
    'FET': 'Fetch.ai',
    'GMT': 'stepn (GMT)',
    'HBAR': 'Hedera',
    'IMX': 'Immutable',
    'INIT': 'Initia Labs',
    'INJ': 'Injective',
    'JTO': 'Jito Network',
    'JUP': 'Jupiter Exchange',
    'LAYER': 'Solayer Labs',
    'LDO': 'Lido',
    'LUNC': 'Terra Classic',
    'ORDI': 'Ordinals (Bitcoin)',
    'POL': 'Polygon',
    'RESOLV': 'Resolv USR',
    'S': 'Sonic',
    'SKY': 'Maker',            # MakerDAO -> Sky rebrand; dev history is Maker's
    'STX': 'Stacks',
    'SUSHI': 'Sushi Swap',
    'SYRUP': 'Maple',
    'WLD': 'Worldcoin',
    'ZEN': 'Horizen',
}

KEEP_COLS = ['all_devs', 'exclusive_devs', 'full_time_devs',
             'one_time_devs', 'devs_2y_plus', 'num_commits']


def _fetch(path: str, dest: Path) -> Path:
    if not dest.exists():
        req = urllib.request.Request(f"{BASE_URL}{path}", headers=_UA)
        with urllib.request.urlopen(req) as r:
            dest.write_bytes(r.read())
    return dest


def download_snapshot() -> tuple:
    """Resolve the latest snapshot from the manifest; cache by version."""
    import json
    req = urllib.request.Request(MANIFEST_URL, headers=_UA)
    with urllib.request.urlopen(req) as r:
        manifest = json.load(r)
    ds = manifest['dataset']
    version = ds['version']
    paths = {Path(res['path']).name: res['path'] for res in ds['resources']}
    cache = Path(tempfile.gettempdir()) / f"opendevdata_{version}"
    cache.mkdir(exist_ok=True)
    mads = _fetch(paths['eco_mads.parquet'], cache / 'eco_mads.parquet')
    ecos = _fetch(paths['ecosystems.parquet'], cache / 'ecosystems.parquet')
    return mads, ecos, version


def symbol_ecosystem_map(ecosystems: pd.DataFrame) -> dict:
    """Universe symbol -> ecosystem id (manual map wins over CoinGecko id)."""
    from etl.marketcap import COINGECKO_ID_MAP
    eco = ecosystems[ecosystems['is_crypto'] == 1]
    by_name = {n.lower(): i for i, n in zip(eco['id'], eco['name'])}

    u = sorted(set(load_data('universe')['symbol'].str.upper()))
    cached = load_data('coingecko_ids')
    ids = {**COINGECKO_ID_MAP,
           **dict(zip(cached['symbol'], cached['coingecko_id']))}

    out = {}
    for s in u:
        if s in MANUAL_ECOSYSTEM_MAP:
            hit = by_name.get(MANUAL_ECOSYSTEM_MAP[s].lower())
        else:
            cid = str(ids.get(s, s)).lower()
            hit = next((by_name[c] for c in
                        (cid, cid.replace('-', ' '), s.lower())
                        if c in by_name), None)
        if hit is not None:
            out[s] = hit
    return out


def main():
    mads_path, ecos_path, version = download_snapshot()
    print(f"snapshot {version}")
    ecosystems = pd.read_parquet(ecos_path)
    sym_map = symbol_ecosystem_map(ecosystems)
    print(f"mapped {len(sym_map)} universe symbols to ecosystems")

    mads = pd.read_parquet(mads_path)
    id_to_sym = {}
    for s, i in sym_map.items():
        id_to_sym.setdefault(i, s)   # 1:1; first symbol wins on collisions
    out = mads[mads['ecosystem_id'].isin(id_to_sym)].copy()
    out['symbol'] = out['ecosystem_id'].map(id_to_sym)
    out['date'] = pd.to_datetime(out['day'])
    out = (out[['date', 'symbol'] + KEEP_COLS]
           .sort_values(['symbol', 'date']).reset_index(drop=True))

    save_data('dev_activity', out, mode='overwrite')

    n_syms = out['symbol'].nunique()
    print(f"dev_activity: {len(out):,} rows, {n_syms} symbols, "
          f"{out['date'].min().date()} -> {out['date'].max().date()}")
    unmapped = sorted(set(load_data('universe')['symbol'].str.upper())
                      - set(sym_map))
    print(f"unmapped ({len(unmapped)}): {', '.join(unmapped)}")
    assert n_syms >= 80, "suspiciously few symbols mapped"
    assert out['date'].max() <= pd.Timestamp.now(), "future dev-activity dates"


if __name__ == '__main__':
    main()
