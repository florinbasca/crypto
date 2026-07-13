"""
ETL for perp listing dates (first-trade timestamps).

The prices table is clipped to the backtest window, so "first bar" is
left-censored for anything listed before it. This pulls the TRUE first
perp trade date per universe symbol:
  1. Binance USD-M futures: earliest 1d kline (startTime=0, limit=1),
     with the 1000-contract fallback (1000PEPEUSDT etc.).
  2. Hyperliquid candleSnapshot (earliest 1d candle) for HL-only names.

Listing dates are PIT by construction: the date was known on the date.

Output: `listings` (symbol, first_trade, venue).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import time

import aiohttp
import pandas as pd

from dbutil import load_data, save_data

BINANCE_URL = 'https://fapi.binance.com/fapi/v1/klines'
HL_URL = 'https://api.hyperliquid.xyz/info'


async def binance_first_trade(session, symbol):
    for perp in (f"{symbol}USDT", f"1000{symbol}USDT"):
        params = {'symbol': perp, 'interval': '1d',
                  'startTime': 0, 'limit': 1}
        try:
            async with session.get(BINANCE_URL, params=params) as r:
                if r.status != 200:
                    continue
                rows = await r.json()
                if rows:
                    return pd.to_datetime(rows[0][0], unit='ms'), 'binance'
        except Exception:
            continue
    return None, None


async def hl_first_trade(session, symbol):
    # HL returns at most ~5000 candles; 1d covers any listing age.
    body = {'type': 'candleSnapshot',
            'req': {'coin': symbol, 'interval': '1d',
                    'startTime': 0, 'endTime': int(time.time() * 1000)}}
    try:
        async with session.post(HL_URL, json=body) as r:
            if r.status != 200:
                return None, None
            rows = await r.json()
            if rows:
                return pd.to_datetime(rows[0]['t'], unit='ms'), 'hyperliquid'
    except Exception:
        pass
    return None, None


async def main():
    symbols = sorted(set(load_data('universe')['symbol'].str.upper()))
    sem = asyncio.Semaphore(8)

    async def one(session, s):
        async with sem:
            ts, venue = await binance_first_trade(session, s)
            if ts is None:
                ts, venue = await hl_first_trade(session, s)
            return s, ts, venue

    timeout = aiohttp.ClientTimeout(total=60)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        results = await asyncio.gather(*[one(session, s) for s in symbols])

    rows = [(s, ts, v) for s, ts, v in results if ts is not None]
    missing = [s for s, ts, _ in results if ts is None]
    out = pd.DataFrame(rows, columns=['symbol', 'first_trade', 'venue'])
    out = out.sort_values('symbol').reset_index(drop=True)
    save_data('listings', out, mode='overwrite')

    print(f"listings: {len(out)}/{len(symbols)} symbols "
          f"({(out['venue'] == 'binance').sum()} binance, "
          f"{(out['venue'] == 'hyperliquid').sum()} hyperliquid), "
          f"{out['first_trade'].min().date()} -> {out['first_trade'].max().date()}")
    if missing:
        print(f"missing: {', '.join(missing)}")
    assert len(out) >= len(symbols) - 5, "too many symbols without a listing date"
    assert out['first_trade'].max() <= pd.Timestamp.now()


if __name__ == '__main__':
    asyncio.run(main())
