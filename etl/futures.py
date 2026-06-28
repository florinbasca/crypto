"""
ETL for Binance Perpetual Futures Metrics.

Downloads historical metrics data from Binance Data Vision (free, no API key required).
Data available from 2021-12 onwards (daily granularity, 5-minute resolution within day).

Metrics include:
- sum_open_interest: Total open interest in contracts
- sum_open_interest_value: Total open interest in USD
- count_toptrader_long_short_ratio: Top trader positioning
- sum_toptrader_long_short_ratio: Aggregated top trader L/S ratio
- count_long_short_ratio: Retail long/short ratio
- sum_taker_long_short_vol_ratio: Taker buy/sell volume ratio

IMPORTANT: Metrics are reported every 5 minutes. To avoid look-ahead bias,
features should use the prior period's metrics when predicting.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import aiohttp
import asyncio
import datetime
import pandas as pd
import zipfile
import io
from tqdm import tqdm

from dbutil import load_data, save_data
from config import config as global_config, get_data_end_date, get_data_start_date

# Binance Data Vision base URL (free, no API key)
BASE_URL = 'https://data.binance.vision'
METRICS_PATH = '/data/futures/um/daily/metrics'

# Data collection settings - limited to the configured data window
MAX_CONCURRENT_DOWNLOADS = global_config['data']['max_concurrent_downloads']

# Resample to base panel frequency on save (raw metrics are 5-minute)
RESAMPLE_FREQUENCY = global_config['base_frequency']

# Quote currency for perpetuals
QUOTE_CURRENCY = 'USDT'


def load_universe():
    """Load symbols from universe table."""
    try:
        df = load_data(table_name='universe')
        if df is None or df.empty:
            return None
        symbols = df['symbol'].tolist()
        return [str(s).strip().upper() for s in symbols if pd.notna(s)]
    except Exception as e:
        print(f"Error loading universe: {e}")
        return None


def get_date_range(start_date, end_date=None):
    """Generate list of dates between start and end."""
    if end_date is None:
        end_date = datetime.datetime.now()

    if isinstance(start_date, str):
        start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    if isinstance(end_date, str):
        end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d")

    dates = []
    current = start_date

    while current <= end_date:
        dates.append(current.strftime('%Y-%m-%d'))
        current += datetime.timedelta(days=1)

    return dates


def get_existing_dates(symbol: str) -> set:
    """Get COMPLETE dates already in database for a symbol.

    Expect bars_per_day rows per day at the resample frequency;
    mark complete if >= 75% (allows minor gaps).
    """
    from config import get_frequency_config
    bars_per_day = get_frequency_config(RESAMPLE_FREQUENCY)['bars_per_day']
    try:
        df = load_data(table_name='futures_metrics', filters={'symbol': symbol},
                       columns=['timestamp'])
        if df is None or df.empty:
            return set()

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['timestamp'].dt.strftime('%Y-%m-%d')

        date_counts = df.groupby('date').size()
        complete_dates = date_counts[date_counts >= int(bars_per_day * 0.75)].index.tolist()
        return set(complete_dates)
    except Exception:
        return set()


def prune_to_window(start_dt: datetime.datetime, end_dt: datetime.datetime) -> None:
    """Keep persisted futures metrics inside the configured ETL window."""
    df = load_data(table_name='futures_metrics')
    if df is None or df.empty:
        return

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    kept = df[(df['timestamp'] >= start_dt) & (df['timestamp'] <= end_dt)].copy()
    if len(kept) == len(df):
        return
    if kept.empty:
        print("Warning: pruning would empty futures_metrics; leaving table unchanged.")
        return

    kept = kept.drop_duplicates(subset=['timestamp', 'symbol'])
    kept = kept.sort_values(['symbol', 'timestamp']).reset_index(drop=True)
    save_data('futures_metrics', kept, mode='overwrite')


async def download_metrics(session, semaphore, symbol, date_str):
    """Download daily metrics data from Binance Vision."""
    async with semaphore:
        perp_symbol = f"{symbol}{QUOTE_CURRENCY}"
        url = f"{BASE_URL}{METRICS_PATH}/{perp_symbol}/{perp_symbol}-metrics-{date_str}.zip"

        try:
            async with session.get(url) as resp:
                if resp.status == 404:
                    return None
                resp.raise_for_status()
                zip_data = await resp.read()
        except Exception:
            return None

        try:
            with zipfile.ZipFile(io.BytesIO(zip_data)) as zip_file:
                csv_filename = f"{perp_symbol}-metrics-{date_str}.csv"
                with zip_file.open(csv_filename) as csv_file:
                    df = pd.read_csv(csv_file)
        except Exception:
            return None

        if df.empty:
            return None

        # Rename columns for consistency
        df = df.rename(columns={
            'create_time': 'timestamp',
            'sum_open_interest': 'open_interest',
            'sum_open_interest_value': 'open_interest_value',
            'count_toptrader_long_short_ratio': 'toptrader_ls_accounts',
            'sum_toptrader_long_short_ratio': 'toptrader_ls_positions',
            'count_long_short_ratio': 'retail_ls_ratio',
            'sum_taker_long_short_vol_ratio': 'taker_buy_sell_ratio'
        })

        # Parse timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Add base symbol
        df['symbol'] = symbol

        # Drop the symbol column from original data if present
        if 'symbol' in df.columns and df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated()]

        # Resample the 5-minute snapshots onto the base panel grid.
        # BAR-END convention (must match prices.py / the panel): a bar stamped
        # T covers (T-1bar, T], so the bin must be RIGHT-labeled/right-closed -
        # the value at T is the last snapshot AT or BEFORE T. pandas' default
        # (left-labeled) would stamp the [T, T+1bar) snapshot at T, leaking
        # ~5min of future OI/positioning into the bar that predicts (T, T+1bar].
        df = df.set_index('timestamp')
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        resampled = df[numeric_cols].resample(
            RESAMPLE_FREQUENCY, label='right', closed='right').last()
        resampled['symbol'] = symbol
        resampled = resampled.dropna(subset=numeric_cols[:1])  # Drop rows where all numeric are NaN
        df = resampled.reset_index()

        return df


async def fetch_symbol_metrics(session, semaphore, symbol, dates_to_download, pbar=None):
    """Fetch all metrics data for a symbol."""

    async def fetch_date(date_str):
        try:
            df = await download_metrics(session, semaphore, symbol, date_str)
            return date_str, df
        finally:
            if pbar:
                pbar.update(1)

    tasks = [fetch_date(d) for d in dates_to_download]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    downloaded_frames = []
    for result in results:
        if isinstance(result, Exception):
            continue
        _, df = result
        if df is not None and not df.empty:
            downloaded_frames.append(df)

    if not downloaded_frames:
        return None

    combined = pd.concat(downloaded_frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=['timestamp']).sort_values('timestamp')

    return combined


async def main():
    """Download futures metrics for all symbols in universe."""
    from rich.console import Console
    from rich.table import Table

    console = Console()

    base_symbols = load_universe()
    if base_symbols is None:
        console.print("[red]Error: universe table not found. Run 'python etl/universe.py' first.[/red]")
        return

    console.print(f"\n[bold]Futures Metrics ETL[/bold]")
    console.print(f"Symbols: {len(base_symbols)}")

    # Date range (metrics available from 2021-12)
    start_dt = get_data_start_date()
    end_dt = get_data_end_date('daily')

    console.print(f"Date range: {start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}")

    all_dates = get_date_range(start_dt, end_dt)
    prune_to_window(start_dt, end_dt)
    console.print(f"Total days in range: {len(all_dates)}")

    # Calculate what needs to be downloaded
    download_plan = {}
    total_downloads = 0

    for symbol in base_symbols:
        existing = get_existing_dates(symbol)
        missing = [d for d in all_dates if d not in existing]
        if missing:
            download_plan[symbol] = missing
            total_downloads += len(missing)

    if total_downloads == 0:
        console.print("[green]All futures metrics data is up to date![/green]")
        return

    # Show download plan
    plan_table = Table(title="Download Plan")
    plan_table.add_column("Symbol", style="cyan")
    plan_table.add_column("Missing Days", justify="right")

    for symbol, dates in sorted(download_plan.items()):
        plan_table.add_row(symbol, str(len(dates)))

    console.print(plan_table)
    console.print(f"\nTotal downloads: {total_downloads}")

    # Download - all symbols in parallel, save incrementally
    connector = aiohttp.TCPConnector(limit=200, limit_per_host=50)
    timeout = aiohttp.ClientTimeout(total=120)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)
        pbar = tqdm(total=total_downloads, desc="Downloading futures metrics")

        # Process symbols and save incrementally to avoid losing progress
        async def fetch_and_save(symbol, dates):
            df = await fetch_symbol_metrics(session, semaphore, symbol, dates, pbar)
            if df is not None and not df.empty:
                # Save this symbol's data immediately (append mode with file lock for concurrency)
                save_data('futures_metrics', df, mode='append', use_file_lock=True)
                return symbol, len(df)
            return symbol, 0

        tasks = [fetch_and_save(symbol, dates) for symbol, dates in download_plan.items()]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        saved_count = 0
        for result in results:
            if isinstance(result, Exception):
                console.print(f"[yellow]Warning: {result}[/yellow]")
                continue
            symbol, count = result
            saved_count += count

        pbar.close()

    if saved_count == 0:
        console.print("[yellow]No new futures metrics downloaded.[/yellow]")
        return

    # Load final data for summary
    combined_df = load_data(table_name='futures_metrics')
    if combined_df is None or combined_df.empty:
        console.print("[yellow]No data in table.[/yellow]")
        return

    combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'])
    combined_df = combined_df[
        (combined_df['timestamp'] >= start_dt) &
        (combined_df['timestamp'] <= end_dt)
    ].copy()
    # Deduplicate in case of any overlaps
    combined_df = combined_df.drop_duplicates(subset=['timestamp', 'symbol'])
    combined_df = combined_df.sort_values(['symbol', 'timestamp']).reset_index(drop=True)
    save_data('futures_metrics', combined_df, mode='overwrite')

    # Summary
    summary_table = Table(title="Download Summary")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="green", justify="right")

    summary_table.add_row("Total rows", f"{len(combined_df):,}")
    summary_table.add_row("Symbols", str(combined_df['symbol'].nunique()))
    summary_table.add_row("Date range",
                          f"{combined_df['timestamp'].min().strftime('%Y-%m-%d')} to "
                          f"{combined_df['timestamp'].max().strftime('%Y-%m-%d')}")

    console.print(summary_table)

    # Show per-symbol summary
    symbol_summary = combined_df.groupby('symbol').agg({
        'timestamp': ['min', 'max', 'count'],
        'open_interest_value': 'mean',
        'retail_ls_ratio': 'mean'
    }).round(2)
    symbol_summary.columns = ['first', 'last', 'count', 'avg_oi_usd', 'avg_ls_ratio']

    sym_table = Table(title="Per-Symbol Summary")
    sym_table.add_column("Symbol", style="cyan")
    sym_table.add_column("First", style="dim")
    sym_table.add_column("Last", style="dim")
    sym_table.add_column("Count", justify="right")
    sym_table.add_column("Avg OI ($M)", justify="right")
    sym_table.add_column("Avg L/S", justify="right")

    for symbol, row in symbol_summary.iterrows():
        avg_oi_m = row['avg_oi_usd'] / 1e6
        ls_style = "green" if row['avg_ls_ratio'] > 1 else "red"
        sym_table.add_row(
            symbol,
            row['first'].strftime('%Y-%m-%d'),
            row['last'].strftime('%Y-%m-%d'),
            f"{row['count']:,}",
            f"${avg_oi_m:.1f}M",
            f"[{ls_style}]{row['avg_ls_ratio']:.2f}[/{ls_style}]"
        )

    console.print(sym_table)


if __name__ == "__main__":
    asyncio.run(main())
