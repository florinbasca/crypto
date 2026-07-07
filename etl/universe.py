"""
Universe ETL: Hyperliquid-tradeable candidate universe.

Fetches the live Hyperliquid perp listing (the tradability constraint for the
strategy), maps coin names to Binance spot base symbols (our historical data
source), removes stablecoins/pegged assets, PROBES Binance Vision for actual
data availability (a name with no Binance spot history anywhere in the window
cannot be backtested - e.g. HYPE, KAS, MNT, HL-native and meme listings),
then keeps the top `universe.max_candidates` by Hyperliquid 24h notional
volume among the AVAILABLE names - so dead candidates don't consume slots.

The result is the CANDIDATE list used by the downloaders, factor model,
research, and portfolio (missing history naturally produces NaNs). Each run
also updates `universe_membership` - a point-in-time interval table
[symbol, valid_from, valid_to] of candidate spells - so membership becomes
genuinely point-in-time as snapshots accrue (see update_membership_history).

Caveat (documented, unavoidable without historical listing dates): history
BEFORE the first snapshot is seeded as "member since the data start", so the
backtest there is still conditioned on the Hyperliquid listing as of that
first snapshot. Use walk_forward.min_listing_age_days to bound the
sensitivity to newly listed (survivor-biased) names.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import datetime

import aiohttp
import pandas as pd
import requests
from dateutil.relativedelta import relativedelta

from dbutil import load_data, save_data, table_exists
from config import get, get_data_start_date

BASE_URL = 'https://data.binance.vision'
SPOT_PATH = '/data/spot/monthly/klines'


def fetch_hyperliquid_universe() -> pd.DataFrame:
    """Fetch perp universe + 24h notional volume from the Hyperliquid info API."""
    api_url = get('universe.hyperliquid_api')
    resp = requests.post(api_url, json={'type': 'metaAndAssetCtxs'}, timeout=30)
    resp.raise_for_status()
    meta, ctxs = resp.json()

    rows = []
    for asset, ctx in zip(meta['universe'], ctxs):
        if asset.get('isDelisted'):
            continue
        rows.append({
            'hl_name': asset['name'],
            'max_leverage': asset.get('maxLeverage'),
            'day_ntl_vlm': float(ctx.get('dayNtlVlm', 0.0)),
            # Mark price enables ticker-identity diagnostics:
            # recycled tickers (e.g. LIT: Binance=Litentry, HL=Lighter) make
            # Binance history belong to a different asset than the HL perp
            'hl_mark_price': float(ctx.get('markPx') or 0.0),
        })
    return pd.DataFrame(rows)


def map_to_binance_symbol(hl_name: str) -> str:
    """Map a Hyperliquid coin name to a Binance spot base symbol."""
    aliases = get('universe.symbol_aliases', {})
    if hl_name in aliases:
        return aliases[hl_name]
    # k-prefixed contracts trade 1000x units of the underlying (kPEPE -> PEPE)
    if hl_name.startswith('k') and hl_name[1:].isupper():
        return hl_name[1:]
    return hl_name.upper()


def _probe_months() -> list:
    """Months to probe across the data window (recent, mid, old).

    A symbol counts as available if ANY probed month exists for ANY quote
    currency - recently delisted names keep their history, new listings only
    have recent months.
    """
    now = datetime.datetime.now().replace(day=1)
    start = get_data_start_date().replace(day=1)
    months_since_start = max((now.year - start.year) * 12 + (now.month - start.month), 0)
    offsets = [1, 7, 19, min(months_since_start, 31)]
    months = {(now - relativedelta(months=m)).strftime('%Y-%m') for m in offsets}
    months.add(start.strftime('%Y-%m'))
    return sorted(months)


async def probe_binance_availability(symbols: list) -> dict:
    """HEAD-probe Binance Vision monthly 1m files. Returns {symbol: bool}."""
    interval = get('data.raw_interval', '1m')
    quotes = get('data.quote_currencies', ['USDT', 'USDC'])
    months = _probe_months()
    semaphore = asyncio.Semaphore(get('data.max_concurrent_downloads', 50))

    async def probe_symbol(session, symbol):
        for quote in quotes:
            pair = f"{symbol}{quote}"
            for month in months:
                url = (f"{BASE_URL}{SPOT_PATH}/{pair}/{interval}/"
                       f"{pair}-{interval}-{month}.zip")
                async with semaphore:
                    try:
                        async with session.head(url) as resp:
                            if resp.status == 200:
                                return symbol, True
                    except Exception:
                        continue
        return symbol, False

    timeout = aiohttp.ClientTimeout(total=60)
    connector = aiohttp.TCPConnector(limit=50, limit_per_host=25)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        results = await asyncio.gather(*[probe_symbol(session, s) for s in symbols])
    return dict(results)


def build_universe() -> pd.DataFrame:
    """Build the candidate universe table (availability-validated)."""
    blacklist = set(get('universe.stablecoin_blacklist', [])) | \
        set(get('universe.symbol_blacklist', []))
    max_candidates = get('universe.max_candidates', 130)
    fetched_at = datetime.datetime.utcnow()

    hl = fetch_hyperliquid_universe()
    print(f"Hyperliquid perps fetched: {len(hl)}")

    hl['symbol'] = hl['hl_name'].apply(map_to_binance_symbol)
    # k-prefixed HL contracts quote 1000 units of the underlying (kPEPE),
    # while Binance spot prices are base-unit. Normalize the mark so the
    # ticker-identity check compares like with like.
    k_mask = hl['hl_name'].str.startswith('k') & (hl['symbol'] == hl['hl_name'].str[1:])
    hl.loc[k_mask, 'hl_mark_price'] = hl.loc[k_mask, 'hl_mark_price'] / 1000.0
    hl = hl[~hl['symbol'].isin(blacklist)]
    hl = hl.drop_duplicates(subset=['symbol'], keep='first')
    hl = hl.sort_values('day_ntl_vlm', ascending=False).reset_index(drop=True)

    # Probe ALL mapped HL names so unavailable ones don't consume candidate
    # slots - the list refills with the next-ranked available names
    print(f"Probing Binance Vision availability for {len(hl)} symbols "
          f"(months {_probe_months()})...")
    availability = asyncio.run(probe_binance_availability(hl['symbol'].tolist()))
    hl['has_binance_data'] = hl['symbol'].map(availability)

    dropped = hl[~hl['has_binance_data']]
    if not dropped.empty:
        print(f"\nDropped {len(dropped)} HL names with NO Binance spot data "
              f"(top by HL volume): "
              f"{', '.join(dropped.sort_values('day_ntl_vlm', ascending=False)['symbol'].head(15))}")

    hl = hl[hl['has_binance_data']].head(max_candidates).reset_index(drop=True)
    hl['rank'] = hl.index + 1
    hl['fetched_at'] = fetched_at

    return hl[['symbol', 'hl_name', 'day_ntl_vlm', 'hl_mark_price', 'rank',
               'fetched_at']]


def evolve_membership(mem, current_symbols, as_of, seed_from) -> tuple:
    """Pure membership-spell evolution (no IO): -> (mem, n_new, n_closed).

    One row per spell: [symbol, valid_from, valid_to] (valid_to = NaT while
    open). mem=None seeds the initial cohort from `seed_from` (historical HL
    listing dates are unavailable, so pre-snapshot history keeps the legacy
    "current set at every timestamp" assumption). Subsequent calls OPEN a
    spell for newly appeared candidates and CLOSE the spell of names that
    dropped out (delisted, or pushed out of the volume-capped top
    `universe.max_candidates`); a later re-listing opens a new spell.
    """
    as_of = pd.Timestamp(as_of).normalize()
    current = set(current_symbols)

    if mem is None or mem.empty:
        seeded = pd.DataFrame({
            'symbol': sorted(current),
            'valid_from': pd.Timestamp(seed_from),
            'valid_to': pd.NaT,
        })
        return seeded, len(seeded), 0

    mem = mem.copy()
    mem['valid_from'] = pd.to_datetime(mem['valid_from'])
    mem['valid_to'] = pd.to_datetime(mem['valid_to'])
    open_mask = mem['valid_to'].isna()
    open_syms = set(mem.loc[open_mask, 'symbol'])
    closed = sorted(open_syms - current)
    if closed:
        mem.loc[open_mask & mem['symbol'].isin(closed), 'valid_to'] = as_of
    new = sorted(current - open_syms)
    if new:
        mem = pd.concat([mem, pd.DataFrame({
            'symbol': new, 'valid_from': as_of, 'valid_to': pd.NaT,
        })], ignore_index=True)
    mem = mem.sort_values(['symbol', 'valid_from']).reset_index(drop=True)
    return mem, len(new), len(closed)


def update_membership_history(current_symbols, as_of) -> pd.DataFrame:
    """Maintain the point-in-time `universe_membership` interval table.

    IO wrapper around `evolve_membership` (see its docstring for semantics).
    Consumed automatically by research/lib/signal_eval.py
    (universe_member_mask) and the walk-forward DataContext.
    """
    mem = load_data('universe_membership') if table_exists('universe_membership') \
        else None
    mem, n_new, n_closed = evolve_membership(
        mem, current_symbols, as_of, get_data_start_date())
    save_data('universe_membership', mem, mode='overwrite',
              datetime_columns=['valid_from', 'valid_to'])
    print(f"universe_membership: {len(mem)} spells "
          f"({n_new} opened, {n_closed} closed this run)")
    return mem


def main():
    df = build_universe()
    save_data(table_name='universe', data=df, mode='overwrite',
              datetime_columns=['fetched_at'])
    update_membership_history(df['symbol'], df['fetched_at'].iloc[0])

    print(f"\nSaved {len(df)} candidate symbols to 'universe' table "
          f"(all validated to have Binance spot data)")
    print(f"Top 20 by HL volume: {', '.join(df['symbol'].head(20))}")
    print(f"Bottom 5: {', '.join(df['symbol'].tail(5))}")
    print("\nNext: run etl/prices_raw.py (incremental - already-downloaded "
          "symbols are skipped), then etl/prices.py, etl/marketcap.py, and "
          "risk_model/factor_returns.py.")


if __name__ == '__main__':
    main()
