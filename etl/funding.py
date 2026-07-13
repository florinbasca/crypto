"""
ETL for Binance Perpetual Futures Funding Rates.

Downloads historical funding rate data from Binance Data Vision (free, no API key required).
Data is available from 2020-01 for major perpetuals.

IMPORTANT: Funding rates are settled every 8h (or 4h for some contracts).
The rate is KNOWN at settlement time, so calc_time is the timestamp. Under
the pipeline's bar-end convention a value stamped T is knowable at bar T
(features may use data through bar t INCLUSIVE; forward targets start at
t+1), so NO extra shift is applied downstream - features_futures.py aligns
with a right-labeled/right-closed resample and the truncation test in
tests/sanity_checks.py enforces causality.
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
from dateutil.relativedelta import relativedelta
from tqdm import tqdm

from dbutil import load_data, save_data
from config import get_data_end_date, get_data_start_date

# Binance Data Vision base URL (free, no API key)
BASE_URL = 'https://data.binance.vision'
FUTURES_PATH = '/data/futures/um/monthly/fundingRate'

# Data collection settings
MAX_CONCURRENT_DOWNLOADS = 10

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


def get_month_range(start_date, end_date=None):
    """Generate list of year-month strings between start and end dates."""
    if end_date is None:
        end_date = datetime.datetime.now()

    if isinstance(start_date, str):
        start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    if isinstance(end_date, str):
        end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d")

    months = []
    current = start_date.replace(day=1)

    while current <= end_date:
        months.append(current.strftime('%Y-%m'))
        current += relativedelta(months=1)

    return months


def get_existing_months(symbol: str) -> set:
    """Get months already in database for a symbol."""
    try:
        df = load_data(table_name='funding_rates', filters={'symbol': symbol})
        if df is None or df.empty:
            return set()

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        months = df['timestamp'].dt.strftime('%Y-%m').unique()
        return set(months)
    except Exception:
        return set()


def prune_to_window(start_dt: datetime.datetime, end_dt: datetime.datetime) -> None:
    """Keep persisted funding data inside the configured ETL window."""
    df = load_data(table_name='funding_rates')
    if df is None or df.empty:
        return

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    kept = df[(df['timestamp'] >= start_dt) & (df['timestamp'] <= end_dt)].copy()
    if len(kept) == len(df):
        return
    if kept.empty:
        print("Warning: pruning would empty funding_rates; leaving table unchanged.")
        return

    kept = kept.drop_duplicates(subset=['timestamp', 'symbol'])
    kept = kept.sort_values(['symbol', 'timestamp']).reset_index(drop=True)
    save_data('funding_rates', kept, mode='overwrite')


async def download_funding_rate(session, semaphore, symbol, year_month):
    """Download monthly funding rate data from Binance Vision.

    Tiny-price names have no {SYM}USDT perp - Binance lists them as
    1000-contracts (1000PEPEUSDT). Funding RATES are unitless, so the
    fallback needs no rescaling."""
    async with semaphore:
        zip_data = None
        for perp_symbol in (f"{symbol}{QUOTE_CURRENCY}",
                            f"1000{symbol}{QUOTE_CURRENCY}"):
            url = (f"{BASE_URL}{FUTURES_PATH}/{perp_symbol}/"
                   f"{perp_symbol}-fundingRate-{year_month}.zip")
            try:
                async with session.get(url) as resp:
                    if resp.status == 404:
                        continue
                    resp.raise_for_status()
                    zip_data = await resp.read()
                    break
            except Exception:
                continue
        if zip_data is None:
            return None

        try:
            with zipfile.ZipFile(io.BytesIO(zip_data)) as zip_file:
                csv_filename = f"{perp_symbol}-fundingRate-{year_month}.csv"
                with zip_file.open(csv_filename) as csv_file:
                    df = pd.read_csv(csv_file)
        except Exception:
            return None

        if df.empty:
            return None

        # Expected columns: calc_time, funding_interval_hours, last_funding_rate
        # Rename for consistency
        df = df.rename(columns={
            'calc_time': 'timestamp',
            'last_funding_rate': 'funding_rate',
            'funding_interval_hours': 'interval_hours'
        })

        # Convert timestamp (milliseconds)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        # Add symbol
        df['symbol'] = symbol

        return df


async def fetch_symbol_funding(session, semaphore, symbol, months_to_download, pbar=None):
    """Fetch all funding rate data for a symbol."""

    async def fetch_month(year_month):
        try:
            df = await download_funding_rate(session, semaphore, symbol, year_month)
            return year_month, df
        finally:
            if pbar:
                pbar.update(1)

    tasks = [fetch_month(month) for month in months_to_download]
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
    """Download funding rates for all symbols in universe."""
    from rich.console import Console
    from rich.table import Table

    console = Console()

    base_symbols = load_universe()
    if base_symbols is None:
        console.print("[red]Error: universe table not found. Run 'python etl/universe.py' first.[/red]")
        return

    console.print(f"\n[bold]Funding Rate ETL[/bold]")
    console.print(f"Symbols: {len(base_symbols)}")

    # Date range
    start_dt = get_data_start_date()
    end_dt = get_data_end_date('monthly')

    console.print(f"Date range: {start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}")

    all_months = get_month_range(start_dt, end_dt)
    prune_to_window(start_dt, end_dt)

    # Calculate what needs to be downloaded
    download_plan = {}
    total_downloads = 0

    for symbol in base_symbols:
        existing = get_existing_months(symbol)
        missing = [m for m in all_months if m not in existing]
        if missing:
            download_plan[symbol] = missing
            total_downloads += len(missing)

    if total_downloads == 0:
        console.print("[green]All funding rate data is up to date![/green]")
        return

    # Show download plan
    plan_table = Table(title="Download Plan")
    plan_table.add_column("Symbol", style="cyan")
    plan_table.add_column("Missing Months", justify="right")
    plan_table.add_column("Sample", style="dim")

    for symbol, months in sorted(download_plan.items()):
        sample = ', '.join(months[:3]) + ('...' if len(months) > 3 else '')
        plan_table.add_row(symbol, str(len(months)), sample)

    console.print(plan_table)
    console.print(f"\nTotal downloads: {total_downloads}")

    # Download - all symbols in parallel, save incrementally
    connector = aiohttp.TCPConnector(limit=50, limit_per_host=10)
    timeout = aiohttp.ClientTimeout(total=120)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)
        pbar = tqdm(total=total_downloads, desc="Downloading funding rates")

        # Process symbols and save incrementally to avoid losing progress
        async def fetch_and_save(symbol, months):
            df = await fetch_symbol_funding(session, semaphore, symbol, months, pbar)
            if df is not None and not df.empty:
                # Save this symbol's data immediately (append mode with file lock for concurrency)
                save_data('funding_rates', df, mode='append', use_file_lock=True)
                return symbol, len(df)
            return symbol, 0

        tasks = [fetch_and_save(symbol, months) for symbol, months in download_plan.items()]
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
        console.print("[yellow]No new funding rate data downloaded.[/yellow]")
        return

    # Load final data for summary and deduplication
    combined_df = load_data(table_name='funding_rates')
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

    # Save deduplicated data
    save_data('funding_rates', combined_df, mode='overwrite')

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
        'funding_rate': 'mean'
    }).round(6)
    symbol_summary.columns = ['first', 'last', 'count', 'avg_rate']

    sym_table = Table(title="Per-Symbol Summary")
    sym_table.add_column("Symbol", style="cyan")
    sym_table.add_column("First", style="dim")
    sym_table.add_column("Last", style="dim")
    sym_table.add_column("Count", justify="right")
    sym_table.add_column("Avg Rate", justify="right")

    for symbol, row in symbol_summary.iterrows():
        avg_rate_pct = row['avg_rate'] * 100
        rate_style = "green" if avg_rate_pct > 0 else "red"
        sym_table.add_row(
            symbol,
            row['first'].strftime('%Y-%m'),
            row['last'].strftime('%Y-%m'),
            str(row['count']),
            f"[{rate_style}]{avg_rate_pct:.4f}%[/{rate_style}]"
        )

    console.print(sym_table)


if __name__ == "__main__":
    asyncio.run(main())
