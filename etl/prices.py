#!/usr/bin/env python3
"""
Resample 1-minute price data to the base panel frequency - single streaming pass.

Streams prices_raw once via DuckDB record batches (no per-symbol scans, no
per-symbol table rewrites) and aggregates with one vectorized groupby per
batch. Buckets that straddle a batch boundary are carried over so the
aggregation is exact.

Bar-end convention: raw timestamps are close_time (bar end), so the bucket is
timestamp.ceil(rule) - identical to resample(label='right', closed='right').

Correctness requires per-symbol time-sorted storage order, which prices_raw
guarantees (whole-symbol sorted replaces). Each batch is checked and a final
duplicate-key assertion raises rather than silently corrupting bars.

Usage:
    python etl/prices.py                # prices_raw -> `prices` at base frequency
    python etl/prices.py --minutes 60   # -> prices_60m
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging
import time

import pandas as pd

from dbutil import (
    count_duplicate_keys,
    count_rows,
    delete_table,
    get_table_columns,
    get_table_symbols,
    iter_data_batches,
    save_data,
    table_exists,
)
from config import config as global_config, get_frequency_config

logging.basicConfig(
    level=logging.INFO,
    format=global_config['logging']['format'],
    datefmt=global_config['logging']['datefmt'],
)

BASE_MINUTES = int(24 * 60 / get_frequency_config(global_config['base_frequency'])['bars_per_day'])
RAW_MINUTES = int(
    24 * 60
    / get_frequency_config(global_config['data']['raw_interval'])['bars_per_day']
)

BATCH_ROWS = 2_000_000        # raw rows per streamed batch
FLUSH_ROWS = 5_000_000        # aggregated rows per append to the target table

AGG_SPEC = {
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum',
    'quote_asset_volume': 'sum',
    'number_of_trades': 'sum',
    'taker_buy_base_asset_volume': 'sum',
    'taker_buy_quote_asset_volume': 'sum',
}


def aggregate_chunk(df: pd.DataFrame, rule: str, agg_cols: dict) -> pd.DataFrame:
    """Vectorized OHLCV aggregation of a (symbol-contiguous, time-sorted) chunk."""
    bucket = df['timestamp'].dt.ceil(rule)
    g = df.groupby([df['symbol'], bucket], sort=False, observed=True)
    out = g.agg(agg_cols)
    out.index.names = ['symbol', 'timestamp']
    return out.reset_index()


def aggregate_stream(batches, rule: str, agg_cols: dict):
    """
    Consume an iterable of raw-bar DataFrames (symbol-contiguous, time-sorted
    storage order) and yield exact N-minute aggregations.

    The final (symbol, bucket) group of each batch is carried into the next
    batch so buckets split across batch boundaries aggregate exactly.
    """
    carry = None
    for df in batches:
        if df.empty:
            continue
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Guard the order assumption: within-symbol timestamps must be
        # non-decreasing in storage order (whole-symbol sorted replaces).
        sym = df['symbol']
        bad = (df['timestamp'].diff() < pd.Timedelta(0)) & (sym == sym.shift())
        if bad.any():
            logging.warning(f"Out-of-order rows in batch ({int(bad.sum())}) - sorting")
            df = df.sort_values(['symbol', 'timestamp'], kind='stable')

        if carry is not None and not carry.empty:
            df = pd.concat([carry, df], ignore_index=True)

        # Hold back the final (symbol, bucket) group - it may continue in the
        # next batch
        last_sym = df['symbol'].iloc[-1]
        last_bucket = df['timestamp'].iloc[-1].ceil(rule)
        tail = (df['symbol'] == last_sym) & (df['timestamp'].dt.ceil(rule) == last_bucket)
        carry = df[tail]
        body = df[~tail]

        if not body.empty:
            yield aggregate_chunk(body, rule, agg_cols)

    if carry is not None and not carry.empty:
        yield aggregate_chunk(carry, rule, agg_cols)


def resample_streaming(minutes: int, target_table: str) -> int:
    """Stream prices_raw once, aggregate to N-minute bars, write target table."""
    rule = f'{minutes}min'
    schema_cols = set(get_table_columns('prices_raw'))
    agg_cols = {c: f for c, f in AGG_SPEC.items() if c in schema_cols}
    read_cols = ['symbol', 'timestamp'] + list(agg_cols)

    total_raw = count_rows('prices_raw')
    logging.info(f"Streaming {total_raw:,} raw rows -> {rule} bars ({target_table})")

    delete_table(target_table)

    out_buffer = []
    out_rows = 0
    total_out = 0
    t0 = time.time()

    def flush():
        nonlocal out_buffer, out_rows, total_out
        if not out_buffer:
            return
        chunk = pd.concat(out_buffer, ignore_index=True)
        save_data(target_table, chunk, mode='append', datetime_columns=['timestamp'])
        total_out += len(chunk)
        out_buffer, out_rows = [], 0

    # read-write config: this pass streams prices_raw while appending to the
    # target table in the same process, so the read cursor must share the
    # writer's read-write connection config (DuckDB forbids mixing them).
    batches = iter_data_batches(
        'prices_raw',
        columns=read_cols,
        batch_size=BATCH_ROWS,
        order_by=['symbol', 'timestamp'],
        read_only=False,
    )
    for agg in aggregate_stream(batches, rule, agg_cols):
        out_buffer.append(agg)
        out_rows += len(agg)
        if out_rows >= FLUSH_ROWS:
            flush()
            rate = total_out * minutes / max(time.time() - t0, 1e-9)
            logging.info(f"  ~{total_out * minutes:,}/{total_raw:,} raw rows "
                         f"({rate / 1e6:.1f}M rows/s)")
    flush()

    # Integrity: (symbol, timestamp) must be unique - a violation means the
    # storage-order assumption broke and bars were split
    n_dupes = count_duplicate_keys(target_table, ['symbol', 'timestamp'])
    if n_dupes:
        raise RuntimeError(
            f"{n_dupes} duplicate (symbol, timestamp) bars in {target_table} - "
            f"prices_raw storage order is not symbol-contiguous/sorted")

    elapsed = time.time() - t0
    symbol_count = len(get_table_symbols(target_table, use_universe_cache=False))
    logging.info(f"Done: {total_out:,} bars, {symbol_count} symbols "
                 f"in {elapsed:.0f}s ({total_raw / max(elapsed, 1e-9) / 1e6:.1f}M raw rows/s)")
    return total_out


def main():
    parser = argparse.ArgumentParser(
        description='Resample raw 1-minute price data (single streaming pass)')
    parser.add_argument('--minutes', '-m', type=int, default=BASE_MINUTES,
                        help=f'Bar size in minutes (default: {BASE_MINUTES} = base frequency)')
    args = parser.parse_args()
    minutes = args.minutes

    if minutes < RAW_MINUTES:
        logging.error(f"Minutes must be >= {RAW_MINUTES} (raw data interval)")
        return
    if minutes % RAW_MINUTES != 0:
        logging.warning(f"Minutes ({minutes}) is not a multiple of {RAW_MINUTES}")

    if not table_exists('prices_raw'):
        logging.error("prices_raw not found - run etl/prices_raw.py first")
        return

    target_table = 'prices' if minutes == BASE_MINUTES else f'prices_{minutes}m'
    resample_streaming(minutes, target_table)


if __name__ == '__main__':
    main()
