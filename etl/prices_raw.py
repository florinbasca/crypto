"""
ETL: 1-minute Binance spot klines -> `prices_raw` (the raw history source).

Asynchronously downloads monthly kline archives from Binance Data Vision (free,
no API key) for every candidate in the universe table, over the configured
data window, and stores them whole-symbol time-sorted in
`prices_raw`. Incremental: months already present are skipped. Symbols with no
Binance spot history anywhere in the window are reported, not fatal.

Downstream etl/prices.py resamples this 1m table to the 10-minute base panel and
relies on the per-symbol sorted storage order this script guarantees.

Usage:
    python etl/prices_raw.py            # run AFTER etl/universe.py
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
from dbutil import (
    load_data,
    save_data,
    get_existing_months as db_get_existing_months,
    get_table_symbols,
)
from tqdm import tqdm

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.precision', 4)


from config import get, get_data_end_date, get_data_start_date

BASE_URL = 'https://data.binance.vision'
SPOT_PATH = '/data/spot/monthly/klines'

# Data collection settings (from config)
RAW_INTERVAL = get('data.raw_interval', '1m')
MAX_CONCURRENT_DOWNLOADS = get('data.max_concurrent_downloads', 50)
QUOTE_CURRENCIES = get('data.quote_currencies', ['USDT', 'USDC'])


def get_start_date() -> datetime.datetime:
    """Data window start from central config."""
    return get_data_start_date()


def trim_symbol_to_window(base_symbol: str, start_dt: datetime.datetime,
                          end_dt: datetime.datetime) -> bool:
    """Rewrite a raw-price symbol partition to the configured ETL window."""
    try:
        df = load_data(table_name='prices_raw', filters={'symbol': base_symbol})
    except Exception:
        return False
    if df is None or df.empty:
        return False

    for col in ['open_time', 'close_time', 'timestamp']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])

    ts_col = 'timestamp' if 'timestamp' in df.columns else 'open_time'
    kept = df[(df[ts_col] >= start_dt) & (df[ts_col] <= end_dt)].copy()
    if len(kept) == len(df):
        return False
    if kept.empty:
        print(f"Warning: pruning would empty prices_raw for {base_symbol}; leaving unchanged.")
        return False

    save_data(
        table_name='prices_raw',
        data=kept.drop(columns=['symbol'], errors='ignore'),
        mode='replace',
        partition_key='symbol',
        partition_value=base_symbol,
        datetime_columns=['open_time', 'close_time', 'timestamp']
    )
    return True


def trim_symbols_to_window(base_symbols: list, start_dt: datetime.datetime,
                           end_dt: datetime.datetime) -> int:
    return sum(trim_symbol_to_window(symbol, start_dt, end_dt) for symbol in base_symbols)


async def download_monthly_klines(session, semaphore, symbol, year_month, interval=RAW_INTERVAL):
    """Download monthly klines data from Binance Vision (default: 1m interval)"""
    async with semaphore:
        url = f"{BASE_URL}{SPOT_PATH}/{symbol}/{interval}/{symbol}-{interval}-{year_month}.zip"
        
        async with session.get(url) as resp:
            if resp.status == 404:
                return None
            
            resp.raise_for_status()
            zip_data = await resp.read()
            
        # Extract CSV from ZIP
        with zipfile.ZipFile(io.BytesIO(zip_data)) as zip_file:
            csv_filename = f"{symbol}-{interval}-{year_month}.csv"
            with zip_file.open(csv_filename) as csv_file:
                df = pd.read_csv(csv_file, header=None)
                
        # Check if first row looks like header (contains strings)
        first_row = df.iloc[0]
        if any(isinstance(val, str) and not val.replace('.', '').replace('-', '').isdigit() for val in first_row):
            df = df.iloc[1:].reset_index(drop=True)
        
        df.columns = [
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ]
        
        # Convert data types
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume',
                        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
        df[numeric_cols] = df[numeric_cols].astype(float)
        
        # Validate and convert timestamps
        open_times_numeric = pd.to_numeric(df['open_time'])
        
        # Check if timestamps are in microseconds (16 digits) or milliseconds (13 digits)
        # Milliseconds max value is 9,999,999,999,999 (13 digits)
        milliseconds_max = 10**13 - 1
        if open_times_numeric.max() > milliseconds_max:
            # Convert from microseconds to milliseconds
            df['open_time'] = (open_times_numeric / 1000).astype(int)
            df['close_time'] = (pd.to_numeric(df['close_time']) / 1000).astype(int)
            open_times_numeric = df['open_time']
        
        # Validate timestamp range (1970-2099 in milliseconds)
        invalid_mask = (open_times_numeric < 0) | (open_times_numeric > 4102444800000)
        
        if invalid_mask.any():
            df = df[~invalid_mask].copy()
        
        # Check for NaN values in price columns
        price_cols = ['open', 'high', 'low', 'close']
        nan_mask = df[price_cols].isna().any(axis=1)
        
        if nan_mask.any():
            df = df[~nan_mask].copy()
        
        if len(df) == 0:
            return None
        
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        # Use close_time as timestamp (bar-end time when data is actually available)
        df['timestamp'] = df['close_time']
        df['number_of_trades'] = df['number_of_trades'].astype(int)
        df = df.drop('ignore', axis=1)
        return df


def get_month_range(start_date, end_date=None):
    """Generate list of year-month strings between start and end dates"""
    if end_date is None:
        end_date = datetime.datetime.now()
    
    months = []
    current = start_date.replace(day=1)  # Start from first day of month
    
    while current <= end_date:
        months.append(current.strftime('%Y-%m'))
        current += relativedelta(months=1)
    
    return months


def load_universe():
    """Load base symbols from the universe table (e.g., BTC, ETH, not BTCUSDT)."""
    try:
        df = load_data(table_name='universe')
        if df is None or df.empty:
            return None
        symbols = df['symbol'].tolist()
        # Clean up symbols - remove any whitespace and convert to uppercase
        symbols = [str(s).strip().upper() for s in symbols if pd.notna(s)]
        return symbols
    except Exception as e:
        print(f"Error loading universe from database: {e}")
        return None


async def fetch_all_klines(session, semaphore, base_symbol, interval, start_ts, end_ts=None, pbar=None):
    """
    Fetch klines data from Binance Vision monthly files and store them in SQLite.
    Downloads all months in parallel for optimal performance.
    """
    start_dt = pd.to_datetime(start_ts, unit='ms')
    end_dt = pd.to_datetime(end_ts, unit='ms') if end_ts is not None else datetime.datetime.now()

    existing_months = db_get_existing_months(base_symbol)
    all_months = get_month_range(start_dt, end_dt)
    months_to_download = [month for month in all_months if month not in existing_months]

    if not months_to_download:
        trim_symbol_to_window(base_symbol, start_dt.to_pydatetime(), end_dt.to_pydatetime())
        return True, f"ALREADY_IN_DB_{len(existing_months)}_months"

    quote_options = [(quote, f"{base_symbol}{quote}") for quote in QUOTE_CURRENCIES]
    months_attempted = len(months_to_download)
    earliest_month = months_to_download[0]

    async def fetch_single_month(year_month: str):
        """Try all quote currencies for a given month, return first successful download."""
        try:
            for quote_currency, symbol_pair in quote_options:
                df = await download_monthly_klines(session, semaphore, symbol_pair, year_month, interval)
                if df is not None:
                    return year_month, quote_currency, df
            return year_month, None, None
        finally:
            if pbar:
                pbar.update(1)

    # Download all months in parallel (semaphore controls concurrency)
    month_tasks = [fetch_single_month(month) for month in months_to_download]
    month_results = await asyncio.gather(*month_tasks, return_exceptions=True)

    # Process results
    downloaded_frames = []
    quotes_used = set()

    for result in month_results:
        if isinstance(result, Exception):
            continue  # Skip failed downloads

        year_month, quote_currency, df = result
        if df is None:
            continue

        # Filter to start date for earliest month
        if year_month == earliest_month:
            df = df[df['open_time'] >= start_dt]

        if df.empty:
            continue

        downloaded_frames.append(df)
        if quote_currency:
            quotes_used.add(quote_currency)

    if not downloaded_frames:
        return False, f"NO_BINANCE_DATA_attempted_{months_attempted}_months"

    # Combine all downloaded data
    new_data = pd.concat(downloaded_frames, ignore_index=True)
    new_data = new_data.drop_duplicates(subset=['open_time']).sort_values('open_time').reset_index(drop=True)

    # Merge with existing data if present
    try:
        existing_df = load_data(table_name='prices_raw', filters={'symbol': base_symbol})
    except Exception:
        existing_df = None

    if existing_df is not None and not existing_df.empty:
        # Remove symbol column for merging
        if 'symbol' in existing_df.columns:
            existing_df = existing_df.drop(columns=['symbol'])
        combined_df = pd.concat([existing_df, new_data], ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=['open_time']).sort_values('open_time').reset_index(drop=True)
    else:
        combined_df = new_data

    # Ensure timestamp column exists for all data
    if 'timestamp' not in combined_df.columns:
        combined_df['timestamp'] = combined_df['close_time']

    combined_df = combined_df[
        (combined_df['timestamp'] >= start_dt) &
        (combined_df['timestamp'] <= end_dt)
    ].copy()

    # Save to database
    save_data(
        table_name='prices_raw',
        data=combined_df,
        mode='replace',
        partition_key='symbol',
        partition_value=base_symbol,
        datetime_columns=['open_time', 'close_time', 'timestamp']
    )

    quote_summary = ",".join(sorted(quotes_used)) if quotes_used else "UNKNOWN"
    return True, f"DOWNLOADED_{quote_summary}_{len(downloaded_frames)}/{months_attempted}_months"


async def main():
    base_symbols = load_universe()
    if base_symbols is None:
        print("Error: universe table not found. Run 'python etl/universe.py' first.")
        return
    print(f"Using {len(base_symbols)} base symbols for data collection")

    start_dt = get_start_date()
    start_ts = int(start_dt.timestamp() * 1000)  # Binance API uses ms

    # Dynamic end date: last day of previous month
    end_dt = get_data_end_date('monthly')
    end_ts = int(end_dt.timestamp() * 1000)

    print(f"\n=== DATE RANGE ===")
    print(f"Start: {start_dt.strftime('%Y-%m-%d')}")
    print(f"End:   {end_dt.strftime('%Y-%m-%d')}")
    print(f"Frequency: {RAW_INTERVAL} candles")

    # Create aiohttp session for downloading files
    connector = aiohttp.TCPConnector(limit=50, limit_per_host=10)
    timeout = aiohttp.ClientTimeout(total=120)  # Longer timeout for file downloads

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        # Calculate total number of months to download and show detailed breakdown
        all_months_range = get_month_range(start_dt, end_dt)
        total_months = 0
        symbols_with_data = []
        symbols_missing_data = []
        symbol_details = {}

        # Build summary data
        summary_data = []
        for base_symbol in base_symbols:
            existing_months = db_get_existing_months(base_symbol)
            months_to_download = [month for month in all_months_range if month not in existing_months]

            total_months += len(months_to_download)

            symbol_details[base_symbol] = {
                'existing': len(existing_months),
                'missing': len(months_to_download),
                'total': len(all_months_range)
            }

            if len(months_to_download) > 0:
                symbols_missing_data.append(base_symbol)
                summary_data.append({
                    'symbol': base_symbol,
                    'have': len(existing_months),
                    'need': len(months_to_download),
                    'missing': ', '.join(months_to_download[:3]) + (f' +{len(months_to_download)-3}' if len(months_to_download) > 3 else '')
                })
            else:
                symbols_with_data.append(base_symbol)

        # Print missing data table using rich
        from rich.console import Console
        from rich.table import Table

        console = Console()

        if summary_data:
            missing_table = Table(title="Missing Data by Symbol", show_header=True, header_style="bold magenta")
            missing_table.add_column("Symbol", style="cyan", width=8)
            missing_table.add_column("Have", justify="right", style="green", width=6)
            missing_table.add_column("Need", justify="right", style="yellow", width=6)
            missing_table.add_column("Missing Months", style="blue")

            for item in summary_data:
                missing_table.add_row(
                    item['symbol'],
                    str(item['have']),
                    str(item['need']),
                    item['missing']
                )

            console.print("\n")
            console.print(missing_table)

        if symbols_with_data:
            console.print(f"\n[green]Complete:[/green] {', '.join(symbols_with_data)}")

        # Summary table
        summary_table = Table(show_header=True, header_style="bold magenta")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green", justify="right")

        summary_table.add_row("Complete data", str(len(symbols_with_data)))
        summary_table.add_row("Partial/Missing", str(len(symbols_missing_data)))
        summary_table.add_row("Downloads needed", str(total_months))

        console.print("\n")
        console.print(summary_table)

        pruned_symbols = trim_symbols_to_window(base_symbols, start_dt, end_dt)
        if pruned_symbols:
            console.print(f"\n[yellow]Pruned old rows before {start_dt.strftime('%Y-%m-%d')} "
                          f"from {pruned_symbols} prices_raw symbol partitions.[/yellow]")

        if total_months == 0:
            console.print("\n[green]No downloads needed - all data is up to date![/green]")
            return

        # Create semaphore for rate limiting downloads
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)

        # Symbol-level concurrency cap: each in-flight symbol holds its monthly
        # frames in memory until saved (~150MB/symbol at 1m over 3y)
        symbol_semaphore = asyncio.Semaphore(get('data.max_concurrent_symbols', 8))

        # Create shared progress bar for all downloads
        pbar = tqdm(total=total_months, desc="Downloading monthly files", unit="file")

        async def fetch_symbol_data(base_symbol):
            async with symbol_semaphore:
                try:
                    result = await fetch_all_klines(session, semaphore, base_symbol, RAW_INTERVAL, start_ts, end_ts, pbar)
                    data_found, info = result
                    return base_symbol, data_found, info
                except Exception as e:
                    return base_symbol, False, str(e)

        # Execute all symbols in parallel with optimized concurrency
        tasks = [fetch_symbol_data(base_symbol) for base_symbol in symbols_missing_data]
        results_list = await asyncio.gather(*tasks, return_exceptions=True)

        pbar.close()

        # Add results for symbols that were already complete
        for symbol in symbols_with_data:
            existing_count = symbol_details[symbol]['existing']
            results_list.append((symbol, True, f"ALREADY_IN_DB_{existing_count}_months"))
        
        # Process results
        successful_downloads = []
        already_in_db = []
        no_binance_data = []
        error_symbols = []
        symbols_with_details = {}
        
        for result in results_list:
            if isinstance(result, Exception):
                error_symbols.append(("unknown", str(result)))
            else:
                base_symbol, _, info = result
                symbols_with_details[base_symbol] = info

                if "ALREADY_IN_DB" in info:
                    already_in_db.append(base_symbol)
                elif "PARTIAL_IN_DB" in info:
                    already_in_db.append(base_symbol)
                elif "DOWNLOADED" in info:
                    successful_downloads.append(base_symbol)
                elif "NO_BINANCE_DATA" in info:
                    no_binance_data.append(base_symbol)
                else:
                    error_symbols.append((base_symbol, info))
        
        # Download summary using rich
        download_summary_table = Table(title="Download Summary", show_header=True, header_style="bold magenta")
        download_summary_table.add_column("Metric", style="cyan")
        download_summary_table.add_column("Value", style="green", justify="right")

        download_summary_table.add_row("Total symbols", str(len(base_symbols)))
        download_summary_table.add_row("Already complete", str(len(already_in_db)))
        download_summary_table.add_row("Downloaded", str(len(successful_downloads)))
        download_summary_table.add_row("Errors", str(len(error_symbols)))

        console.print("\n")
        console.print(download_summary_table)

        if successful_downloads:
            downloads_table = Table(title="Newly Downloaded", show_header=True, header_style="bold magenta")
            downloads_table.add_column("Symbol", style="cyan", width=8)
            downloads_table.add_column("Quote", style="green", width=6)
            downloads_table.add_column("Got", justify="right", style="blue", width=5)
            downloads_table.add_column("Attempted", justify="right", style="yellow", width=10)
            downloads_table.add_column("Success Rate", justify="right", style="magenta", width=12)

            for symbol in sorted(successful_downloads):
                info = symbols_with_details.get(symbol, '')
                # Parse info string: "DOWNLOADED_USDT_62/70_months"
                if "DOWNLOADED_" in info:
                    parts = info.replace("DOWNLOADED_", "").split("_")
                    quote = parts[0] if len(parts) > 0 else "?"
                    months_info = parts[1] if len(parts) > 1 else "?/?"
                    if "/" in months_info:
                        got, attempted = months_info.replace("_months", "").split("/")
                        success_rate = f"{int(got)/int(attempted)*100:.1f}%" if attempted != "0" else "N/A"
                        downloads_table.add_row(symbol, quote, got, attempted, success_rate)
                    else:
                        downloads_table.add_row(symbol, quote, "?", "?", "N/A")
                else:
                    downloads_table.add_row(symbol, "?", "?", "?", "N/A")

            console.print("\n")
            console.print(downloads_table)

        if error_symbols:
            errors_table = Table(title="Errors", show_header=True, header_style="bold magenta")
            errors_table.add_column("Symbol", style="cyan", width=8)
            errors_table.add_column("Error", style="red")

            for symbol, error in error_symbols:
                errors_table.add_row(symbol, error)

            console.print("\n")
            console.print(errors_table)

        # Verify what's actually in the database
        actual_symbols = get_table_symbols('prices_raw', use_universe_cache=False)

        # Show which symbols from universe are missing from DB
        expected_but_missing = set(base_symbols) - set(actual_symbols)
        if expected_but_missing:
            missing_table = Table(title="Symbols Not in Database", show_header=True, header_style="bold magenta")
            missing_table.add_column("Symbol", style="cyan", width=8)
            missing_table.add_column("Status", style="yellow")

            for symbol in sorted(expected_but_missing):
                info = symbols_with_details.get(symbol, 'Not processed')
                missing_table.add_row(symbol, info)

            console.print("\n")
            console.print(missing_table)
        else:
            console.print(f"\n[green]All {len(base_symbols)} symbols successfully stored in database[/green]")


if __name__ == "__main__":
    asyncio.run(main())
