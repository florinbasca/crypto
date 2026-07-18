"""
Parquet + Polars interface for the crypto pipeline: typed table I/O + parallel
helpers.

Storage model (replaces the previous single DuckDB file):
- Each table is its own location under `database.data_dir`:
  - Tables WITH a `symbol` column are **symbol-partitioned**: a directory
    `<root>/<table>/` holding one Parquet file per symbol (`<symbol>.parquet`).
    Per-symbol writes touch only that symbol's file, so ETL stays cheap.
  - Tables WITHOUT a `symbol` column are a single file `<root>/<table>.parquet`.
- Reads use `polars.scan_parquet` (lazy, predicate/projection pushdown) and
  return pandas DataFrames, so existing callers are unchanged.
- Writes are atomic (temp file + `os.replace`) and guarded by a per-file lock,
  so concurrent ETL workers writing *different* symbols never contend, and
  concurrent appends to the *same* file can't corrupt it. Readers are lock-free
  (they only ever see fully-written files).

The public API (save_data / load_data / iter_data_batches / table introspection
/ parallel helpers) matches the previous DuckDB implementation.

Run directly for an interactive table viewer:  python dbutil.py
"""

import calendar
import fcntl
import multiprocessing
import os
import re
import shutil
import threading
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, as_completed, wait
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple

from config import config as global_config, get, get_frequency_config

# Polars' multithreaded Rust runtime is NOT fork-safe. The signal evaluator
# forks worker processes AFTER the parent has already used Polars; a forked
# child then deadlocks on the inherited (dead) thread-pool locks the first time
# it touches Polars. Capping Polars to a single thread BEFORE it is imported
# avoids spawning that work-stealing pool, making the fork pattern safe. This
# pipeline parallelizes process-per-core anyway (cf. blas_threads_per_worker).
os.environ.setdefault('POLARS_MAX_THREADS', str(get('compute.polars_max_threads', 1)))

import pandas as pd  # noqa: E402
import polars as pl  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.table import Table  # noqa: E402
from tqdm import tqdm  # noqa: E402


_configured_dir = Path(global_config['database'].get('data_dir', 'db'))
ROOT = (
    _configured_dir
    if _configured_dir.is_absolute()
    else Path(__file__).resolve().parent / _configured_dir
)
_LOCKS_DIR = ROOT / '.locks'
_LOCAL_LOCK = threading.RLock()

_SYMBOL_RE = re.compile(r'^[A-Za-z0-9._-]+$')


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
def _safe_symbol(symbol: str) -> str:
    """Filename-safe symbol. Crypto tickers are alnum (+ . _ -); reject paths."""
    s = str(symbol)
    if not _SYMBOL_RE.match(s):
        # Fall back to a sanitized form; collisions are astronomically unlikely
        # for the project's ticker namespace, but keep it deterministic.
        s = re.sub(r'[^A-Za-z0-9._-]', '_', s)
    return s


def _table_dir(table_name: str) -> Path:
    return ROOT / table_name


def _single_file(table_name: str) -> Path:
    return ROOT / f"{table_name}.parquet"


def _symbol_file(table_name: str, symbol: str) -> Path:
    return _table_dir(table_name) / f"{_safe_symbol(symbol)}.parquet"


def _is_partitioned(table_name: str) -> bool:
    d = _table_dir(table_name)
    return d.is_dir() and any(d.glob('*.parquet'))


def _parts(table_name: str) -> List[Path]:
    """All Parquet files backing a table, sorted (symbol order for partitioned)."""
    if _is_partitioned(table_name):
        return sorted(_table_dir(table_name).glob('*.parquet'))
    f = _single_file(table_name)
    return [f] if f.exists() else []


def table_exists(table_name: str) -> bool:
    return bool(_parts(table_name))


def _scan(table_name: str) -> Optional[pl.LazyFrame]:
    """Lazy scan over all parts of a table, or None if it doesn't exist."""
    parts = _parts(table_name)
    if not parts:
        return None
    return pl.scan_parquet([str(p) for p in parts])


# ---------------------------------------------------------------------------
# Per-file write locking + atomic writes
# ---------------------------------------------------------------------------
@contextmanager
def _file_lock(target: Path):
    """Exclusive lock keyed to one target Parquet file (cross-process)."""
    _LOCKS_DIR.mkdir(parents=True, exist_ok=True)
    key = str(target.relative_to(ROOT)).replace(os.sep, '__')
    lock_path = _LOCKS_DIR / f"{key}.lock"
    with _LOCAL_LOCK:
        handle = open(lock_path, 'w')
        try:
            fcntl.flock(handle, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
            handle.close()


def _write_atomic(target: Path, df: pl.DataFrame) -> None:
    """Write df to target atomically (temp + os.replace). Caller holds the lock."""
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f"{target.name}.tmp.{os.getpid()}.{threading.get_ident()}")
    df.write_parquet(tmp, compression='zstd', statistics=True)
    os.replace(tmp, target)


def _read_file(path: Path) -> Optional[pl.DataFrame]:
    if not path.exists():
        return None
    return pl.read_parquet(path)


def _concat(existing: Optional[pl.DataFrame], incoming: pl.DataFrame) -> pl.DataFrame:
    """Append-merge by column name (like DuckDB INSERT BY NAME)."""
    if existing is None or existing.is_empty():
        return incoming
    return pl.concat([existing, incoming], how='diagonal_relaxed')


# ---------------------------------------------------------------------------
# Introspection
# ---------------------------------------------------------------------------
def get_table_columns(table_name: str) -> List[str]:
    lf = _scan(table_name)
    if lf is None:
        return []
    return list(lf.collect_schema().names())


def table_has_columns(table_name: str, columns: List[str]) -> bool:
    existing = set(get_table_columns(table_name))
    if not existing:
        return False
    return set(columns).issubset(existing)


def count_rows(table_name: str) -> int:
    lf = _scan(table_name)
    if lf is None:
        return 0
    return int(lf.select(pl.len()).collect().item())


def count_duplicate_keys(table_name: str, columns: List[str]) -> int:
    lf = _scan(table_name)
    if lf is None:
        return 0
    counts = lf.group_by(columns).len().filter(pl.col('len') > 1)
    extra = counts.select((pl.col('len') - 1).sum()).collect().item()
    return int(extra or 0)


def get_table_symbols(table_name: str, use_universe_cache: bool = True) -> List[str]:
    # Partitioned tables encode symbols as filenames — list them directly.
    if _is_partitioned(table_name):
        return sorted(p.stem for p in _table_dir(table_name).glob('*.parquet'))
    if not table_has_columns(table_name, ['symbol']):
        return []
    lf = _scan(table_name)
    syms = (
        lf.select('symbol').drop_nulls().unique().collect()['symbol'].to_list()
    )
    return sorted(syms)


def get_table_symbols_cached(table_name: str) -> List[str]:
    return get_table_symbols(table_name, use_universe_cache=True)


def get_tables() -> List[str]:
    if not ROOT.exists():
        return []
    names = set()
    for child in ROOT.iterdir():
        if child.name == '.locks':
            continue
        if child.is_dir() and any(child.glob('*.parquet')):
            names.add(child.name)
        elif child.is_file() and child.suffix == '.parquet':
            names.add(child.stem)
    return sorted(names)


def get_existing_months(symbol: str) -> Set[str]:
    """Return complete raw-price months for a symbol (>=90% of expected bars)."""
    path = _symbol_file('prices_raw', symbol)
    if not path.exists():
        return set()
    bars_per_day = get_frequency_config(get('data.raw_interval', '1m'))['bars_per_day']
    df = (
        pl.scan_parquet(path)
        .select(pl.col('open_time').dt.strftime('%Y-%m').alias('month'))
        .group_by('month')
        .len()
        .collect()
    )
    complete = set()
    for month, count in zip(df['month'].to_list(), df['len'].to_list()):
        year, month_number = map(int, month.split('-'))
        expected = calendar.monthrange(year, month_number)[1] * bars_per_day
        if count >= expected * 0.9:
            complete.add(month)
    return complete


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------
def _enforce_datetime_types(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    df = df.copy()
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column]).astype('datetime64[ns]')
    return df


def _normalize_datetimes_ns(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce every datetime column to datetime64[ns].

    pandas 3 creates [us]/[s]-unit datetimes (date_range from strings,
    to_datetime of strings, etc). All legacy Parquet files in db/ are
    nanosecond; writing a microsecond frame would create mixed-unit files
    inside one table (a polars scan-schema error waiting to happen), and
    non-ns frames also break merge_asof joins and int64-nanosecond arithmetic
    downstream. Normalizing at this single storage boundary keeps the whole
    pipeline ns-uniform on both pandas 2 and 3."""
    dt_cols = [c for c in df.columns
               if pd.api.types.is_datetime64_any_dtype(df[c])
               and str(df[c].dtype) != 'datetime64[ns]']
    if not dt_cols:
        return df
    df = df.copy()
    for c in dt_cols:
        df[c] = df[c].astype('datetime64[ns]')
    return df


def _remove_table(table_name: str) -> None:
    d = _table_dir(table_name)
    if d.is_dir():
        shutil.rmtree(d)
    f = _single_file(table_name)
    if f.exists():
        f.unlink()


def _append_partitioned(table_name: str, incoming: pl.DataFrame) -> None:
    for (symbol,), group in incoming.group_by(['symbol'], maintain_order=True):
        target = _symbol_file(table_name, symbol)
        with _file_lock(target):
            merged = _concat(_read_file(target), group)
            _write_atomic(target, merged)


def _append_single(table_name: str, incoming: pl.DataFrame) -> None:
    target = _single_file(table_name)
    with _file_lock(target):
        merged = _concat(_read_file(target), incoming)
        _write_atomic(target, merged)


def _append(table_name: str, incoming: pl.DataFrame, partitioned: bool) -> None:
    if partitioned:
        _append_partitioned(table_name, incoming)
    else:
        _append_single(table_name, incoming)


def _delete_partition(table_name: str, key: str, value: Any) -> None:
    """Remove rows where key == value (the pre-step of mode='replace')."""
    if not table_exists(table_name):
        return
    if _is_partitioned(table_name) and key == 'symbol':
        target = _symbol_file(table_name, value)
        if target.exists():
            with _file_lock(target):
                if target.exists():
                    target.unlink()
        return
    # General case: rewrite each affected part with the matching rows removed.
    for part in _parts(table_name):
        with _file_lock(part):
            df = _read_file(part)
            if df is None or key not in df.columns:
                continue
            kept = df.filter(pl.col(key) != value)
            if kept.height == df.height:
                continue
            if kept.is_empty():
                part.unlink()
            else:
                _write_atomic(part, kept)


def save_data(
    table_name: str,
    data: pd.DataFrame,
    mode: str = 'replace',
    partition_key: Optional[str] = None,
    partition_value: Optional[str] = None,
    datetime_columns: Optional[List[str]] = None,
    use_file_lock: bool = False,
    required_columns: Optional[List[str]] = None,
    sort_by: Optional[List[str]] = None,
) -> None:
    """Write a pandas DataFrame to the table's Parquet store.

    Modes:
      - 'overwrite': replace the whole table.
      - 'replace': delete rows where partition_key == partition_value, then add
        the incoming rows (requires partition_key + partition_value).
      - 'append': add the incoming rows.

    Tables with a 'symbol' column are stored one Parquet file per symbol; all
    other tables as a single file. (`use_file_lock` is accepted for backward
    compatibility — locking is always applied per file.)
    """
    del use_file_lock  # writes are always per-file locked

    if required_columns:
        missing = set(required_columns) - set(data.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
    if datetime_columns:
        data = _enforce_datetime_types(data, datetime_columns)
    if partition_key and partition_key not in data.columns:
        if partition_value is None:
            raise ValueError(
                f"partition_value required when partition_key {partition_key!r} is absent"
            )
        data = data.copy()
        data[partition_key] = partition_value
    if sort_by:
        data = data.sort_values(sort_by).reset_index(drop=True)
    if data.empty:
        return
    if mode not in {'replace', 'append', 'overwrite'}:
        raise ValueError(f"Invalid mode: {mode}. Use 'replace', 'append', or 'overwrite'")
    if mode == 'replace' and (partition_key is None or partition_value is None):
        raise ValueError("mode='replace' requires partition_key and partition_value")

    data = _normalize_datetimes_ns(data)
    incoming = pl.from_pandas(data)
    partitioned = 'symbol' in data.columns

    if mode == 'overwrite':
        _remove_table(table_name)
        _append(table_name, incoming, partitioned)
    elif mode == 'replace':
        _delete_partition(table_name, partition_key, partition_value)
        _append(table_name, incoming, partitioned)
    else:  # append
        _append(table_name, incoming, partitioned)


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------
def load_data(
    table_name: str,
    filters: Optional[Dict[str, Any]] = None,
    columns: Optional[List[str]] = None,
    limit: Optional[int] = None,
    drop_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    filters = filters or {}

    def _is_condition(value: Any) -> bool:
        return (isinstance(value, tuple) and len(value) == 2 and
                str(value[0]) in {'=', '==', '!=', '>', '>=', '<', '<=',
                                  'in', 'not in'})

    def _is_condition_list(value: Any) -> bool:
        return isinstance(value, list) and all(_is_condition(v) for v in value)

    def _apply_filter(lf: pl.LazyFrame, key: str, value: Any) -> pl.LazyFrame:
        conditions = value if _is_condition_list(value) else [value]
        for cond in conditions:
            if _is_condition(cond):
                op, rhs = cond
                op = str(op)
                if op in ('=', '=='):
                    lf = lf.filter(pl.col(key) == rhs)
                elif op == '!=':
                    lf = lf.filter(pl.col(key) != rhs)
                elif op == '>':
                    lf = lf.filter(pl.col(key) > rhs)
                elif op == '>=':
                    lf = lf.filter(pl.col(key) >= rhs)
                elif op == '<':
                    lf = lf.filter(pl.col(key) < rhs)
                elif op == '<=':
                    lf = lf.filter(pl.col(key) <= rhs)
                elif op == 'in':
                    lf = lf.filter(pl.col(key).is_in(rhs))
                elif op == 'not in':
                    lf = lf.filter(~pl.col(key).is_in(rhs))
            else:
                lf = lf.filter(pl.col(key) == cond)
        return lf

    # If filtering by symbol on a partitioned table, read only that file.
    if ('symbol' in filters and _is_partitioned(table_name) and
            not _is_condition(filters['symbol']) and
            not _is_condition_list(filters['symbol'])):
        path = _symbol_file(table_name, filters['symbol'])
        lf = pl.scan_parquet(path) if path.exists() else None
        remaining = {k: v for k, v in filters.items() if k != 'symbol'}
    else:
        lf = _scan(table_name)
        remaining = filters

    if lf is None:
        return pd.DataFrame()

    for key, value in remaining.items():
        lf = _apply_filter(lf, key, value)
    if columns:
        lf = lf.select(columns)
    if limit is not None:
        lf = lf.limit(int(limit))

    df = lf.collect().to_pandas()
    if drop_columns:
        df = df.drop(columns=[c for c in drop_columns if c in df.columns])
    return _normalize_datetimes_ns(df)


def iter_data_batches(
    table_name: str,
    columns: Optional[List[str]] = None,
    batch_size: int = 1_000_000,
    order_by: Optional[List[str]] = None,
    read_only: bool = True,
) -> Iterator[pd.DataFrame]:
    """Stream a table as pandas batches.

    For symbol-partitioned tables this yields one symbol's file at a time (in
    symbol order), so batches are naturally symbol-contiguous; `order_by` keys
    other than 'symbol' sort within each file. This preserves the
    "symbol-contiguous, time-sorted" guarantee `etl/prices.py` relies on.
    """
    del read_only  # no connection state in the Parquet backend
    parts = _parts(table_name)
    if not parts:
        return

    order_by = order_by or []
    if _is_partitioned(table_name):
        within_sort = [c for c in order_by if c != 'symbol']
    else:
        within_sort = order_by

    for part in parts:
        lf = pl.scan_parquet(part)
        if columns:
            lf = lf.select(columns)
        if within_sort:
            lf = lf.sort(within_sort)
        df = lf.collect()
        height = df.height
        if height == 0:
            continue
        for start in range(0, height, batch_size):
            yield _normalize_datetimes_ns(df.slice(start, batch_size).to_pandas())


# ---------------------------------------------------------------------------
# Mutation
# ---------------------------------------------------------------------------
def delete_table(table_name: str) -> bool:
    if not table_exists(table_name):
        return False
    row_count = count_rows(table_name)
    symbols = get_table_symbols(table_name)
    # Per-symbol tables report their symbol partitions; single-file tables
    # (e.g. portfolio-level series) have no symbol dimension to report.
    what = f"{len(symbols)} symbols, " if symbols else ""
    _remove_table(table_name)
    print(f"Deleted table '{table_name}' ({what}{row_count:,} rows)")
    return True


def clear_table(table_name: str) -> bool:
    return delete_table(table_name)


def delete_rows_where(table_name: str, column: str, value: Any) -> int:
    """Delete rows where one column equals a value. Returns rows removed."""
    if not table_exists(table_name):
        return 0
    if _is_partitioned(table_name) and column == 'symbol':
        path = _symbol_file(table_name, value)
        if not path.exists():
            return 0
        with _file_lock(path):
            n = count_rows_in_file(path)
            if path.exists():
                path.unlink()
        return n
    removed = 0
    for part in _parts(table_name):
        with _file_lock(part):
            df = _read_file(part)
            if df is None or column not in df.columns:
                continue
            kept = df.filter(pl.col(column) != value)
            removed += df.height - kept.height
            if kept.height == df.height:
                continue
            if kept.is_empty():
                part.unlink()
            else:
                _write_atomic(part, kept)
    return removed


def count_rows_in_file(path: Path) -> int:
    if not path.exists():
        return 0
    return int(pl.scan_parquet(path).select(pl.len()).collect().item())


def delete_rows_in(table_name: str, column: str, values) -> int:
    """Delete rows where ``column`` is in ``values`` (one file pass). Returns
    rows removed. Like calling delete_rows_where for each value but rewriting
    each part at most once — used to make batched writes idempotent."""
    vals = list(dict.fromkeys(values))  # de-dup, preserve order
    if not table_exists(table_name) or not vals:
        return 0
    if _is_partitioned(table_name) and column == 'symbol':
        removed = 0
        for v in vals:
            path = _symbol_file(table_name, v)
            if path.exists():
                with _file_lock(path):
                    removed += count_rows_in_file(path)
                    if path.exists():
                        path.unlink()
        return removed
    removed = 0
    for part in _parts(table_name):
        with _file_lock(part):
            df = _read_file(part)
            if df is None or column not in df.columns:
                continue
            kept = df.filter(~pl.col(column).is_in(vals))
            if kept.height == df.height:
                continue
            removed += df.height - kept.height
            if kept.is_empty():
                part.unlink()
            else:
                _write_atomic(part, kept)
    return removed


def remove_symbol_from_table(table_name: str, symbol: str) -> bool:
    if not table_has_columns(table_name, ['symbol']):
        return False
    removed = delete_rows_where(table_name, 'symbol', symbol)
    if not removed:
        return False
    print(f"Removed {removed:,} rows for '{symbol}' from '{table_name}'")
    return True


def remove_symbol_from_all_tables(symbol: str):
    console = Console()
    results = {}
    for table_name in get_tables():
        if not table_has_columns(table_name, ['symbol']):
            continue
        try:
            results[table_name] = (
                'Removed' if remove_symbol_from_table(table_name, symbol) else 'Not found'
            )
        except Exception as exc:
            results[table_name] = f'Error: {exc}'

    summary = Table(
        title=f"Symbol Removal Summary: {symbol}",
        show_header=True,
        header_style="bold magenta",
    )
    summary.add_column("Table", style="cyan")
    summary.add_column("Status", style="green")
    for table_name, status in results.items():
        summary.add_row(table_name, status)
    console.print(summary)


def ensure_symbol_index(table_name: str, column: str = 'symbol') -> None:
    """No-op: symbol partitioning is the index in the Parquet backend."""


def close_connection():
    """Compatibility no-op: the Parquet backend holds no open connection."""


# ---------------------------------------------------------------------------
# Viewers
# ---------------------------------------------------------------------------
def _table_size_bytes(table_name: str) -> int:
    return sum(p.stat().st_size for p in _parts(table_name))


def get_database_summary():
    console = Console()
    tables = get_tables()
    if not tables:
        console.print("[yellow]No data found[/yellow]")
        return
    total_mb = sum(_table_size_bytes(t) for t in tables) / (1024**2)
    output = Table(
        title=f"Parquet store: {ROOT} ({total_mb:,.1f} MB)",
        show_header=True,
        header_style="bold magenta",
    )
    output.add_column("Table", style="cyan")
    output.add_column("Rows", justify="right", style="green")
    output.add_column("Files", justify="right", style="green")
    for table_name in tables:
        output.add_row(table_name, f"{count_rows(table_name):,}", str(len(_parts(table_name))))
    console.print(output)


def show_table_details(table_name: str):
    console = Console()
    if not table_exists(table_name):
        console.print(f"[yellow]Table '{table_name}' not found[/yellow]")
        return
    lf = _scan(table_name)
    schema = lf.collect_schema()
    sample = lf.limit(1).collect().to_pandas()
    rows = count_rows(table_name)
    output = Table(
        title=f"{table_name.upper()} - {rows:,} rows",
        show_header=True,
        header_style="bold magenta",
    )
    output.add_column("Column", style="cyan")
    output.add_column("Type", style="green")
    output.add_column("Sample", style="blue")
    for column in schema.names():
        value = "" if sample.empty else str(sample[column].iloc[0])
        output.add_row(column, str(schema[column]), value[:50])
    console.print(output)


def view_table_data(table_name: str, limit: int = 100) -> pd.DataFrame:
    df = load_data(table_name, limit=limit)
    print(df)
    return df


# ---------------------------------------------------------------------------
# Parallelism (storage-agnostic helpers; unchanged behavior)
# ---------------------------------------------------------------------------
def _limit_worker_blas_threads() -> None:
    # Force (not setdefault) the thread count: a stray OMP_NUM_THREADS already
    # exported in the shell would otherwise let each spawned worker fan out to
    # all cores, so N workers x C BLAS threads thrashes the box. Spawned children
    # inherit this env and read it at numpy/BLAS import time.
    n = str(get('compute.blas_threads_per_worker', 1))
    for variable in (
        'OMP_NUM_THREADS',
        'OPENBLAS_NUM_THREADS',
        'MKL_NUM_THREADS',
        'NUMEXPR_NUM_THREADS',
    ):
        os.environ[variable] = n


def parallel_map(
    func: Callable,
    items: List[Any],
    max_workers: Optional[int] = None,
    desc: Optional[str] = None,
    show_progress: bool = True,
) -> List[Any]:
    if max_workers is None:
        max_workers = get('compute.default_workers', 4)
    _limit_worker_blas_threads()
    ctx = multiprocessing.get_context('spawn')
    results = [None] * len(items)
    window = max_workers * 2
    next_idx = 0
    pending = {}
    progress = tqdm(total=len(items), desc=desc) if show_progress else None
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
        while next_idx < len(items) and len(pending) < window:
            pending[executor.submit(func, items[next_idx])] = next_idx
            next_idx += 1
        while pending:
            done, _ = wait(set(pending), return_when=FIRST_COMPLETED)
            for future in done:
                idx = pending.pop(future)
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    print(f"Error processing item {idx}: {exc}")
                if progress:
                    progress.update(1)
                while next_idx < len(items) and len(pending) < window:
                    pending[executor.submit(func, items[next_idx])] = next_idx
                    next_idx += 1
    if progress:
        progress.close()
    return results


def parallel_process_symbols(
    func: Callable,
    symbols: Optional[List[str]] = None,
    table_name: Optional[str] = None,
    max_workers: Optional[int] = None,
    desc: Optional[str] = None,
) -> List[Tuple[str, Any]]:
    if symbols is None:
        if table_name is None:
            raise ValueError("Either symbols or table_name must be provided")
        symbols = get_table_symbols(table_name)
    if max_workers is None:
        max_workers = get('compute.default_workers', 4)
    _limit_worker_blas_threads()
    ctx = multiprocessing.get_context('spawn')
    results = []
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
        future_to_symbol = {
            executor.submit(func, symbol): symbol for symbol in symbols
        }
        futures = (
            tqdm(as_completed(future_to_symbol), total=len(symbols), desc=desc)
            if desc
            else as_completed(future_to_symbol)
        )
        for future in futures:
            symbol = future_to_symbol[future]
            try:
                results.append((symbol, future.result()))
            except Exception as exc:
                print(f"Error processing {symbol}: {exc}")
                results.append((symbol, None))
    return results


def get_parallel_executor(
    max_workers: Optional[int] = None,
    start_method: str = 'spawn',
):
    if max_workers is None:
        max_workers = get('compute.default_workers', 4)
    _limit_worker_blas_threads()
    return ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=multiprocessing.get_context(start_method),
    )


def get_parallel_pool(
    max_workers: Optional[int] = None,
    initializer: Optional[Callable] = None,
    initargs: Tuple = (),
):
    if max_workers is None:
        max_workers = get('compute.default_workers', 4)
    _limit_worker_blas_threads()
    return multiprocessing.get_context('spawn').Pool(
        processes=max_workers,
        initializer=initializer,
        initargs=initargs,
    )


def main():
    print("Parquet Store Viewer (Polars)")
    print("=" * 50)
    while True:
        print("\nOptions:")
        print("1. Show all tables")
        print("2. View data")
        print("3. Delete table")
        print("4. Remove symbol from all tables")
        print("5. Exit")
        choice = input("\nChoice: ").strip()
        if choice == '1':
            get_database_summary()
        elif choice == '2':
            table_name = input("Enter table name: ").strip()
            view_table_data(table_name)
        elif choice == '3':
            table_name = input("Enter table to delete: ").strip()
            delete_table(table_name)
        elif choice == '4':
            symbol = input("Enter symbol to remove: ").strip()
            remove_symbol_from_all_tables(symbol)
        elif choice == '5':
            break


if __name__ == '__main__':
    main()
