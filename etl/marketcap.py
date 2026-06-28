"""
ETL: daily market capitalization per symbol -> `marketcap` (size-factor input).

Resolves each universe symbol to its CoinGecko id (manual map > cached
auto-resolution in `coingecko_ids` > scored disambiguation), then fetches daily
market-cap history from the CoinGecko API (requires a key in .keys). The size
factor in risk_model/factor_returns.py ranks universe names by this market cap.

Incremental: cached id resolutions are reused and existing mcap rows are
extended rather than refetched.

Usage:
    python etl/marketcap.py             # run AFTER etl/universe.py; needs .keys
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import aiohttp
import asyncio
import time
import logging
import json
from typing import Optional, Tuple
from datetime import datetime, timedelta, timezone
from dbutil import (
    save_data,
    load_data,
    get_table_symbols,
)
from config import get_data_start_date
from tqdm.asyncio import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# CoinGecko ID mapping
COINGECKO_ID_MAP = {
    'BTC': 'bitcoin',
    'ETH': 'ethereum',
    'BNB': 'binancecoin',
    'SOL': 'solana',
    'XRP': 'ripple',
    'DOGE': 'dogecoin',
    'ADA': 'cardano',
    'AVAX': 'avalanche-2',
    'TRX': 'tron',
    'DOT': 'polkadot',
    'LINK': 'chainlink',
    'MATIC': 'matic-network',
    'POL': 'matic-network',  # Polygon rebrand
    'BCH': 'bitcoin-cash',
    'LTC': 'litecoin',
    'NEAR': 'near',
    'UNI': 'uniswap',
    'ICP': 'internet-computer',
    'ETC': 'ethereum-classic',
    'XLM': 'stellar',
    'APT': 'aptos',
    'FIL': 'filecoin',
    'ATOM': 'cosmos',
    'ARB': 'arbitrum',
    'VET': 'vechain',
    'OP': 'optimism',
    'GRT': 'the-graph',
    'FTM': 'fantom',
    'PEPE': 'pepe',
    'HBAR': 'hedera-hashgraph',
    'SUI': 'sui',
    'TON': 'the-open-network',
    'RUNE': 'thorchain',
    'ALGO': 'algorand',
    'XMR': 'monero',
    'AAVE': 'aave',
    'MKR': 'maker',
    'INJ': 'injective-protocol',
    'RNDR': 'render-token',
    'RENDER': 'render-token',
    'IMX': 'immutable-x',
    'KAS': 'kaspa',
    'SEI': 'sei-network',
    'TIA': 'celestia',
    'FET': 'fetch-ai',
    'AR': 'arweave',
    'FLOW': 'flow',
    'EGLD': 'elrond-erd-2',
    'QNT': 'quant-network',
    'MINA': 'mina-protocol',
    'AXS': 'axie-infinity',
    'SAND': 'the-sandbox',
    'LDO': 'lido-dao',
    'STX': 'blockstack',
    'PENDLE': 'pendle',
    'WLD': 'worldcoin-wld',
    'JUP': 'jupiter-exchange-solana',
    'ALT': 'altlayer',
    'PYTH': 'pyth-network',
    'WIF': 'dogwifcoin',
    'MANTA': 'manta-network',
    'RAY': 'raydium',
    'THETA': 'theta-token',
    'CHZ': 'chiliz',
    'ENS': 'ethereum-name-service',
    'LRC': 'loopring',
    'CRV': 'curve-dao-token',
    'COMP': 'compound-governance-token',
    'SNX': 'havven',
    '1INCH': '1inch',
    'NOT': 'notcoin',
    'ORDI': 'ordinals',
    'BONK': 'bonk',
    'FLOKI': 'floki',
    'AGIX': 'singularitynet',
    'OCEAN': 'ocean-protocol',
    'GALA': 'gala',
    'AXL': 'axelar',
    'DYDX': 'dydx-chain',
    'SHIB': 'shiba-inu',
    'TAO': 'bittensor',
    'ENA': 'ethena',
    # Hyperliquid-universe additions
    'HYPE': 'hyperliquid',
    'ONDO': 'ondo-finance',
    'JTO': 'jito-governance-token',
    'W': 'wormhole',
    'STRK': 'starknet',
    'ZK': 'zksync',
    'ZRO': 'layerzero',
    'EIGEN': 'eigenlayer',
    'ETHFI': 'ether-fi',
    'MOVE': 'movement',
    'ME': 'magic-eden',
    'PENGU': 'pudgy-penguins',
    'VIRTUAL': 'virtual-protocol',
    'AI16Z': 'ai16z',
    'FARTCOIN': 'fartcoin',
    'GOAT': 'goatseus-maximus',
    'PNUT': 'peanut-the-squirrel',
    'MOODENG': 'moo-deng',
    'POPCAT': 'popcat',
    'MEW': 'cat-in-a-dogs-world',
    'BRETT': 'based-brett',
    'TURBO': 'turbo',
    'OM': 'mantra-dao',
    'SAGA': 'saga-2',
    'DYM': 'dymension',
    'PIXEL': 'pixels',
    'AEVO': 'aevo-exchange',
    'BLUR': 'blur',
    'S': 'sonic-3',
    'FTM': 'fantom',
    'TRUMP': 'official-trump',
    'BERA': 'berachain-bera',
    'IP': 'story-2',
    'KAITO': 'kaito',
    'GRASS': 'grass',
    'IO': 'io',
    'SPX': 'spx6900',
    'BOME': 'book-of-meme',
    'DOGS': 'dogs-2',
    'NEIRO': 'neiro-3',
    'PEOPLE': 'constitutiondao',
    'APE': 'apecoin',
    'GMT': 'stepn',
    'CAKE': 'pancakeswap-token',
    'EOS': 'eos',
    'XTZ': 'tezos',
    'IOTA': 'iota',
    'NEO': 'neo',
    'ZEC': 'zcash',
    'WAVES': 'waves',
    'OMG': 'omisego',
    'MANA': 'decentraland',
    'PENDLE': 'pendle',
    'SUSHI': 'sushi',
    'KAVA': 'kava',
    'ANIME': 'anime',
    'USUAL': 'usual',
    'WCT': 'connect-token-wct',
    'INIT': 'initia',
    'SOPH': 'sophon',
    'LAYER': 'solayer',
}

# Rate limiting
REQUEST_DELAY = 2.1  # Max 28 requests/minute
COINGECKO_BASE_URL = 'https://api.coingecko.com/api/v3'

# Auto-resolution: id substrings that indicate wrapped/bridged duplicates
ID_BLOCKLIST_SUBSTRINGS = ['binance-peg', 'wrapped', 'bridged', 'wormhole',
                           '-peg-', 'osmosis-allbtc', 'harrypotter']


_KEYS_HELP = (
    "Create a '.keys' file at the repo root with your (free) CoinGecko Demo "
    "API key:\n"
    '  {\n    "coingecko_api_key": "CG-..."\n  }\n'
    "Get a key at https://www.coingecko.com/en/developers/dashboard"
)


def load_api_key():
    """Load the CoinGecko API key from the .keys file at the repo root.

    Returns the key string, or None after printing an actionable message when
    the file is missing, unreadable, not valid JSON, or has no key set.
    """
    path = Path('.keys')
    if not path.exists():
        print(f"Missing '{path}' file.\n{_KEYS_HELP}")
        return None
    try:
        keys = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        print(f"'{path}' is not valid JSON ({e}).\n{_KEYS_HELP}")
        return None
    except OSError as e:
        print(f"Could not read '{path}': {e}")
        return None
    key = (keys.get('coingecko_api_key') or '').strip()
    if not key:
        print(f"'{path}' has no 'coingecko_api_key' set.\n{_KEYS_HELP}")
        return None
    return key


def _cg_get(path: str, api_key: str, params: dict = None):
    """Synchronous CoinGecko GET with rate limiting."""
    import requests
    import time as _time
    headers = {'x-cg-demo-api-key': api_key} if api_key else {}
    _time.sleep(REQUEST_DELAY)
    resp = requests.get(f"{COINGECKO_BASE_URL}{path}", params=params or {},
                        headers=headers, timeout=60)
    resp.raise_for_status()
    return resp.json()


def fetch_markets_for_ids(ids: list, api_key: str) -> dict:
    """Current {id: (market_cap, current_price)} via /coins/markets (250/page)."""
    out = {}
    for i in range(0, len(ids), 250):
        chunk = ids[i:i + 250]
        try:
            rows = _cg_get('/coins/markets', api_key,
                           {'vs_currency': 'usd', 'ids': ','.join(chunk),
                            'per_page': 250, 'page': 1})
        except Exception as e:
            logging.error(f"/coins/markets failed: {e}")
            continue
        for r in rows:
            out[r['id']] = (float(r.get('market_cap') or 0),
                            float(r.get('current_price') or 0))
    return out


def resolve_symbol_ids(symbols: list, api_key: str) -> Tuple[dict, dict]:
    """
    Resolve symbol -> CoinGecko id for the whole universe.

    Priority: manual COINGECKO_ID_MAP > cached auto-resolutions (coingecko_ids
    table) > fresh auto-resolution via /coins/list with max-market-cap
    disambiguation (ticker collisions are rampant on CoinGecko; the largest
    market cap among same-symbol candidates is the right coin in practice).

    Returns:
        id_map: {symbol: coingecko_id}
        supply_hints: {symbol: circulating_supply_now} (mcap/price) - used to
            anchor the historical backfill for symbols whose Binance history
            doesn't overlap CoinGecko's free 365d window (e.g. delisted XMR).
    """
    id_map = {s: COINGECKO_ID_MAP[s] for s in symbols if s in COINGECKO_ID_MAP}

    # Cached auto-resolutions
    try:
        cache = load_data('coingecko_ids')
        if not cache.empty:
            for _, row in cache.iterrows():
                if row['symbol'] in symbols and row['symbol'] not in id_map:
                    id_map[row['symbol']] = row['coingecko_id']
    except Exception:
        pass

    need = sorted(set(symbols) - set(id_map))
    if need:
        print(f"Auto-resolving {len(need)} symbols via CoinGecko /coins/list...")
        coins = _cg_get('/coins/list', api_key)
        by_symbol = {}
        for c in coins:
            by_symbol.setdefault(str(c.get('symbol', '')).upper(), []).append(c['id'])

        candidates = {}
        for s in need:
            ids = [i for i in by_symbol.get(s, [])
                   if not any(b in i for b in ID_BLOCKLIST_SUBSTRINGS)]
            if ids:
                candidates[s] = ids

        all_ids = sorted({i for ids in candidates.values() for i in ids})
        markets = fetch_markets_for_ids(all_ids, api_key)

        resolved_rows = []
        for s, ids in candidates.items():
            best = max(ids, key=lambda i: markets.get(i, (0, 0))[0])
            if markets.get(best, (0, 0))[0] > 0:
                id_map[s] = best
                resolved_rows.append({'symbol': s, 'coingecko_id': best,
                                      'source': 'auto',
                                      'resolved_at': datetime.now()})
                print(f"  {s} -> {best} "
                      f"(mcap ${markets[best][0] / 1e6:,.0f}M)")

        if resolved_rows:
            save_data('coingecko_ids', pd.DataFrame(resolved_rows), mode='append',
                      datetime_columns=['resolved_at'])

    unresolved = sorted(set(symbols) - set(id_map))
    if unresolved:
        logging.warning(f"Unresolvable on CoinGecko (no candidate with mcap): "
                        f"{', '.join(unresolved)}")

    # Fresh supply hints for ALL mapped ids (supply changes; never cache)
    markets_all = fetch_markets_for_ids(sorted(set(id_map.values())), api_key)
    supply_hints = {}
    for s, cid in id_map.items():
        mcap, price = markets_all.get(cid, (0, 0))
        if mcap > 0 and price > 0:
            supply_hints[s] = mcap / price

    return id_map, supply_hints


def load_universe_from_prices():
    """Load symbols from universe/prices"""
    try:
        # Try loading from universe table
        df_universe = load_data(table_name='universe')
        if not df_universe.empty and 'symbol' in df_universe.columns:
            symbols = df_universe['symbol'].tolist()
        else:
            # Fall back to prices_raw table
            symbols = get_table_symbols('prices_raw', use_universe_cache=False)
        print(f"Loaded {len(symbols)} symbols from universe")
        return sorted(symbols)
    except Exception as e:
        logging.error(f"Error loading universe: {e}")
        return []


def _price_source() -> Tuple[str, str]:
    """(table, timestamp column) for price reads - prefer the resampled
    `prices` table (10x smaller than prices_raw; daily-resolution logic here
    is identical on either)."""
    from dbutil import table_exists
    if table_exists('prices'):
        return 'prices', 'timestamp'
    return 'prices_raw', 'open_time'


def get_symbol_date_ranges(symbol: str) -> Tuple[Optional[datetime], Optional[datetime], Optional[datetime]]:
    """Get price date range and latest market cap date for a symbol"""
    # Get price date range - only load timestamp column for speed
    price_start, price_end = None, None
    table, ts_col = _price_source()
    try:
        price_data = load_data(table_name=table, filters={'symbol': symbol}, columns=[ts_col])
        if ts_col in price_data.columns and len(price_data) > 0:
            price_start = pd.to_datetime(price_data[ts_col].min())
            price_end = pd.to_datetime(price_data[ts_col].max())
            price_start = max(price_start, get_data_start_date())
    except Exception:
        pass

    # Get latest market cap date - only load date column for speed
    mcap_latest = None
    try:
        mcap_data = load_data(table_name='marketcap', filters={'symbol': symbol}, columns=['date'])
        if 'date' in mcap_data.columns and len(mcap_data) > 0:
            mcap_latest = pd.to_datetime(mcap_data['date'].max())
    except Exception:
        pass

    return price_start, price_end, mcap_latest


def save_marketcap_to_db(symbol: str, df: pd.DataFrame) -> bool:
    """Save ONLY NEW market cap data to DB (silent - no logging to avoid breaking tqdm)"""
    start_dt = get_data_start_date()
    df = df[pd.to_datetime(df['date']) >= start_dt].copy() if not df.empty else df

    try:
        # Check if symbol already has data
        try:
            existing = load_data(table_name='marketcap', filters={'symbol': symbol})
            existing_len = len(existing)
            # Remove symbol column
            if 'symbol' in existing.columns:
                existing = existing.drop(columns=['symbol'])
            existing = existing[pd.to_datetime(existing['date']) >= start_dt].copy()
            existing_dates = set(pd.to_datetime(existing['date']).dt.date)
            new_data = df[~pd.to_datetime(df['date']).dt.date.isin(existing_dates)]

            if new_data.empty:
                if len(existing) == existing_len:
                    return True
                combined = existing
            else:
                combined = pd.concat([existing, new_data])
            combined = combined.drop_duplicates(subset=['date']).sort_values('date').reset_index(drop=True)
            save_data(
                table_name='marketcap',
                data=combined,
                mode='replace',
                partition_key='symbol',
                partition_value=symbol,
                datetime_columns=['date'],
                sort_by=['date']
            )
        except KeyError:
            if df.empty:
                return True
            # First time saving
            save_data(
                table_name='marketcap',
                data=df,
                mode='replace',
                partition_key='symbol',
                partition_value=symbol,
                datetime_columns=['date'],
                sort_by=['date']
            )

        return True
    except Exception as e:
        logging.error(f"{symbol}: Save error - {e}")
        return False


def prune_marketcap_to_window(symbol: str) -> None:
    """Keep persisted market cap rows inside the configured ETL window."""
    try:
        existing = load_data(table_name='marketcap', filters={'symbol': symbol})
    except Exception:
        return
    if existing is None or existing.empty or 'date' not in existing.columns:
        return

    start_dt = get_data_start_date()
    kept = existing[pd.to_datetime(existing['date']) >= start_dt].copy()
    if len(kept) == len(existing):
        return
    if kept.empty:
        logging.warning(f"{symbol}: pruning would empty marketcap; leaving unchanged")
        return
    save_data(
        table_name='marketcap',
        data=kept.drop(columns=['symbol'], errors='ignore'),
        mode='replace',
        partition_key='symbol',
        partition_value=symbol,
        datetime_columns=['date'],
        sort_by=['date']
    )


class CoinGeckoClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = None
        self.last_request_time = 0
        
    async def __aenter__(self):
        import ssl
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        connector = aiohttp.TCPConnector(ssl=ssl_context)
        headers = {'x-cg-demo-api-key': self.api_key} if self.api_key else {}
        self.session = aiohttp.ClientSession(connector=connector, headers=headers)
        return self
        
    async def __aexit__(self, *args):
        if self.session:
            await self.session.close()
    
    async def get_market_cap(self, coin_id: str, days_back: int) -> Optional[pd.DataFrame]:
        """Get market cap data for last N days from today"""
        # Rate limit
        elapsed = time.time() - self.last_request_time
        if elapsed < REQUEST_DELAY:
            await asyncio.sleep(REQUEST_DELAY - elapsed)
        
        url = f"{COINGECKO_BASE_URL}/coins/{coin_id}/market_chart"
        params = {'vs_currency': 'usd', 'days': min(days_back, 365)}
        
        try:
            async with self.session.get(url, params=params) as resp:
                self.last_request_time = time.time()
                if resp.status == 200:
                    data = await resp.json()
                    market_caps = data.get('market_caps', [])
                    if not market_caps:
                        return None
                    
                    records = []
                    for ts, mcap in market_caps:
                        # UTC, not local tz - dates must align with UTC price bars
                        date = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
                        records.append({'date': date.date(), 'market_cap': float(mcap or 0)})
                    
                    df = pd.DataFrame(records)
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.groupby('date').last().reset_index().sort_values('date')
                    return df
                else:
                    logging.error(f"API error {resp.status} for {coin_id}")
                    return None
        except Exception as e:
            logging.error(f"Request error for {coin_id}: {e}")
            return None


def calculate_historical_marketcap(symbol: str, start_date: datetime, end_date: datetime,
                                   reference_df: pd.DataFrame,
                                   supply_hint: Optional[float] = None) -> Optional[pd.DataFrame]:
    """
    Calculate historical market cap using price ratios (constant-supply
    approximation).

    Anchor priority:
    1. A date where Binance prices and CoinGecko mcap overlap (implied supply
       at that date).
    2. supply_hint = current circulating supply from /coins/markets - used
       when there is NO date overlap, e.g. symbols delisted from Binance
       before CoinGecko's free 365d window (XMR).
    """
    # Get price data
    table, ts_col = _price_source()
    try:
        price_data = load_data(table_name=table, filters={'symbol': symbol},
                               columns=[ts_col, 'close'])
    except Exception as e:
        logging.warning(f"{symbol}: Not found in prices: {e}")
        return None

    try:
        # Convert to daily prices
        price_data['date'] = pd.to_datetime(price_data[ts_col]).dt.date
        daily_prices = price_data.groupby('date')['close'].last().reset_index()
        daily_prices['date'] = pd.to_datetime(daily_prices['date'])

        implied_supply = None

        if reference_df is not None and not reference_df.empty:
            ref_date = reference_df['date'].min()
            ref_price_data = daily_prices[daily_prices['date'] >= ref_date]
            if not ref_price_data.empty:
                ref_mcap = reference_df.loc[reference_df['date'] == ref_date, 'market_cap'].iloc[0]
                ref_price = ref_price_data['close'].iloc[0]
                if ref_mcap > 0 and ref_price > 0:
                    implied_supply = ref_mcap / ref_price
            else:
                overlapping_dates = set(daily_prices['date'].dt.date) & \
                    set(reference_df['date'].dt.date)
                if overlapping_dates:
                    overlap_date = pd.to_datetime(min(overlapping_dates))
                    ref_mcap = reference_df.loc[reference_df['date'] == overlap_date,
                                                'market_cap'].iloc[0]
                    ref_price = daily_prices.loc[daily_prices['date'] == overlap_date,
                                                 'close'].iloc[0]
                    if ref_mcap > 0 and ref_price > 0:
                        implied_supply = ref_mcap / ref_price

        # Fallback: anchor with current supply (no date overlap available)
        if implied_supply is None and supply_hint and supply_hint > 0:
            implied_supply = supply_hint
            logging.info(f"{symbol}: anchoring historical mcap with current "
                         f"supply ({supply_hint:,.0f})")

        if implied_supply is None:
            logging.warning(f"{symbol}: No mcap anchor (no date overlap, no supply hint)")
            return None

        mask = (daily_prices['date'] >= start_date) & (daily_prices['date'] <= end_date)
        hist_prices = daily_prices[mask].copy()
        if hist_prices.empty:
            return None

        hist_prices['market_cap'] = hist_prices['close'] * implied_supply
        return hist_prices[['date', 'market_cap']]

    except Exception as e:
        logging.error(f"{symbol}: Error calculating historical - {e}")
        return None


async def process_symbol(client: CoinGeckoClient, symbol: str, coin_id: Optional[str],
                         supply_hint: Optional[float] = None,
                         incremental: bool = True) -> bool:
    """Process a single symbol with TRUE incremental updates"""
    if not coin_id:
        logging.warning(f"{symbol}: No CoinGecko ID mapping")
        return False
    
    # Get date ranges
    price_start, price_end, mcap_latest = get_symbol_date_ranges(symbol)
    
    if not price_start:
        logging.warning(f"{symbol}: No price data found")
        return False
    prune_marketcap_to_window(symbol)
    
    # Determine what data we need
    today = datetime.now()
    yesterday = today - timedelta(days=1)

    # Target end date is the end of the last complete month from price data
    # Since Binance data is monthly, we shouldn't fetch beyond the price data end date
    target_end_date = price_end if price_end else yesterday

    # Get API data for recent dates (within 365 days from today)
    api_cutoff = today - timedelta(days=365)

    # Check if we need historical data (before API cutoff)
    # We need to backfill from price_start to api_cutoff if we don't have that data yet
    need_historical = False
    if not mcap_latest:
        # No market cap at all - need full history
        need_historical = True
    else:
        # Check if we have data back to price_start - only load date column for speed
        try:
            existing = load_data(table_name='marketcap', filters={'symbol': symbol}, columns=['date'])
            earliest_mcap = pd.to_datetime(existing['date'].min())

            # If we don't have data all the way back to price_start (or close to it)
            # we need to calculate historical market cap
            if earliest_mcap > price_start + timedelta(days=30):  # Allow 30 day buffer
                need_historical = True
        except Exception:
            # Error loading - assume we need historical
            need_historical = True

    # Check if already up to date (both recent AND historical)
    if incremental and mcap_latest and mcap_latest.date() >= target_end_date.date() and not need_historical:
        return True

    # Collect all new data
    all_new_data = []

    # Determine what date ranges we need to fill
    # 1. Recent data (API): from api_cutoff to yesterday
    # 2. Historical data (calculated): from price_start to api_cutoff

    # Check what we're missing in the recent period (API can provide this)
    recent_start = api_cutoff if not mcap_latest else max(mcap_latest + timedelta(days=1), api_cutoff)
    if recent_start <= target_end_date:
        days_back = (today - recent_start).days + 1
        api_data = await client.get_market_cap(coin_id, days_back)

        if api_data is not None:
            # Filter to only NEW dates we need
            if mcap_latest:
                api_data = api_data[(api_data['date'] > mcap_latest) & (api_data['date'] <= target_end_date)]
            else:
                api_data = api_data[api_data['date'] <= target_end_date]

            if not api_data.empty:
                all_new_data.append(api_data)

    # Calculate historical data if needed (already determined above)
    if need_historical:
        hist_start = price_start
        hist_end = min(api_cutoff - timedelta(days=1), price_end)

        if hist_start < hist_end and price_end >= price_start:
            # Get reference data for calculation
            reference_data = None
            if all_new_data:
                reference_data = all_new_data[0]
            elif mcap_latest:
                # Load some existing data as reference
                try:
                    existing = load_data(table_name='marketcap', filters={'symbol': symbol})
                    reference_data = existing.tail(30)
                except Exception:
                    pass
            else:
                # Try to get any API data as reference
                temp_api_data = await client.get_market_cap(coin_id, 365)
                if temp_api_data is not None and not temp_api_data.empty:
                    reference_data = temp_api_data

            hist_data = calculate_historical_marketcap(
                symbol, hist_start, hist_end, reference_data, supply_hint=supply_hint)
            if hist_data is not None and not hist_data.empty:
                all_new_data.append(hist_data)
    
    # Save only new data
    if all_new_data:
        combined_new = pd.concat(all_new_data).drop_duplicates(subset=['date']).sort_values('date')
        return save_marketcap_to_db(symbol, combined_new)
    else:
        return True


async def main():
    """Fetch market cap data for all symbols (incremental only)"""
    api_key = load_api_key()
    if not api_key:
        return  # load_api_key already printed an actionable message

    symbols = load_universe_from_prices()
    if not symbols:
        print("No symbols found")
        return

    # Resolve symbol -> CoinGecko id (manual map > cache > auto by max mcap)
    # and fetch fresh supply hints for the delisted-symbol backfill anchor
    id_map, supply_hints = resolve_symbol_ids(symbols, api_key)
    print(f"Resolved {len(id_map)}/{len(symbols)} symbols to CoinGecko ids")

    print(f"Fetching market cap data for {len(symbols)} symbols (incremental mode)")

    results = {}
    async with CoinGeckoClient(api_key) as client:
        with tqdm(total=len(symbols), desc="Fetching market cap data") as pbar:
            for symbol in symbols:
                try:
                    success = await process_symbol(
                        client, symbol, id_map.get(symbol),
                        supply_hint=supply_hints.get(symbol), incremental=True)
                    results[symbol] = success
                except Exception as e:
                    logging.error(f"{symbol}: {e}")
                    results[symbol] = False
                pbar.update(1)

    # Summary using rich
    from rich.console import Console
    from rich.table import Table

    console = Console()
    successful = sum(results.values())
    failed = [s for s, r in results.items() if not r]

    # Per-symbol date ranges table
    detail_table = Table(title="Market Cap Data Summary", show_header=True, header_style="bold magenta")
    detail_table.add_column("Symbol", style="cyan", width=8)
    detail_table.add_column("Status", style="white", width=8)
    detail_table.add_column("Rows", justify="right", style="blue", width=8)
    detail_table.add_column("Start Date", style="green", width=12)
    detail_table.add_column("End Date", style="green", width=12)

    for symbol in sorted(symbols):
        try:
            data = load_data(table_name='marketcap', filters={'symbol': symbol})
            status = "[green]OK[/green]" if results.get(symbol, False) else "[red]FAIL[/red]"
            rows = len(data)
            start_date = str(data['date'].min())[:10]
            end_date = str(data['date'].max())[:10]
            detail_table.add_row(symbol, status, f"{rows:,}", start_date, end_date)
        except Exception:
            detail_table.add_row(symbol, "[red]FAIL[/red]", "0", "-", "-")

    console.print("\n")
    console.print(detail_table)

    # Summary stats
    summary_table = Table(show_header=True, header_style="bold magenta")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="green", justify="right")

    summary_table.add_row("Total Symbols", str(len(symbols)))
    summary_table.add_row("Successful", str(successful))
    summary_table.add_row("Failed", str(len(failed)))

    console.print("\n")
    console.print(summary_table)

    if failed:
        console.print(f"\n[red]Failed symbols:[/red] {', '.join(sorted(failed))}")


if __name__ == "__main__":
    asyncio.run(main())
