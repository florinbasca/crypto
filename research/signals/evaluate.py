"""
Signal Research: multi-horizon evaluation against forward residual returns.

For every signal in the generator universe and every configured horizon:
1. Compute the signal at FULL base-frequency resolution (transforms need the
   full grid), smooth + cross-sectionally z-score it.
2. Evaluate at NON-OVERLAPPING horizon stamps on the screening grid
   (config signals.screening_grid; stride = horizon / grid):
   - Rank IC per cross-section vs fwd_res_{horizon}. The targets from
     residual_returns are ALREADY forward sums over bars t+1..t+p - they are
     consumed directly with NO additional shift. (The previous pipeline
     double-shifted here and silently measured lag-2 IC. Do not reintroduce.)
     Overlapping stamps would autocorrelate the IC series and inflate
     t-stats ~sqrt(stride), making the downstream FDR anti-conservative.
   - A dollar-neutral leverage-1 backtest at the same stamps.
   The panel is restricted to the current candidate universe.
3. Persist compact DAILY aggregates (signal_daily_stats) used by the
   walk-forward selector, plus whole-period diagnostics (signal_metrics).
   No giant per-bar signal_values / equity_curves tables.

Usage:
    python research/signals/evaluate.py [signal_name] [--fresh] [--limit N] [--no-parallel]

    signal_name  one-signal dev mode: refresh and evaluate only that signal.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import logging
import os
import time
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl

from dbutil import (load_data, save_data, delete_table, delete_rows_where,
                    delete_rows_in, get_parallel_executor, get_table_columns)
from config import config as global_config, get, horizon_bars

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.WARNING,
                    format=global_config['logging']['format'],
                    datefmt=global_config['logging']['datefmt'])

SCREENING_GRID = os.environ.get('CRYPTO_SIGNAL_SCREENING_GRID',
                                get('signals.screening_grid', '1h'))
SMOOTHING_HALFLIFE = get('signals.smoothing_halflife', 3)
# A cross-section is valid only with enough names: a hard floor AND a fraction
# of the configured universe size, so quintiles/IC are never measured on a
# handful of assets.
MIN_ASSETS = max(get('signals.min_assets_per_timestamp', 10),
                 round(get('signals.min_universe_fraction', 0.0)
                       * get('universe.max_candidates', 130)))
COMPUTE_ON_FULL_HISTORY = get('signals.compute_on_full_history', True)
COST_BPS = get('portfolio.cost_bps', 5.0)
START_DATE = get('walk_forward.start_date')
END_DATE = get('walk_forward.end_date')

# IC term structure: forward-residual targets at a log-spaced lag grid (bars).
# Each signal's decay curve IC(tau) is measured here; the fitted half-life
# replaces fixed horizon buckets in the portfolio layer.
LAG_GRID = get('signals.decay_lag_grid', [1, 3, 6, 18, 36, 72, 144, 288])
LIQ_WINDOW = get('signals.liquidity_window_bars', 144)

# Stride (in screening-grid steps) per lag: stats must use non-overlapping
# forward windows, so the stride is the lag measured in grid bars (min 1).
_GRID_BARS = horizon_bars(SCREENING_GRID)  # screening grid length in base bars
_TARGET_CACHE_ENV = 'CRYPTO_SIGNAL_EVAL_TARGET_CACHE'


def _lag_stride(lag: int) -> int:
    # ceil so the forward windows never overlap even when lag is not a multiple
    # of the screening grid (overlapping windows inflate IC t-stats ~sqrt(stride)).
    return max(1, int(np.ceil(lag / _GRID_BARS)))


def lag_label(lag: int) -> str:
    """Label stored in the daily-stats 'horizon' column, e.g. '36b'."""
    return f'{lag}b'


def _lag_col(lag: int) -> str:
    return f'fwd_{lag}b'


def _raw_lag_col(lag: int) -> str:
    return f'fwd_raw_{lag}b'


def _nw_tstat(x: np.ndarray, lags='auto') -> float:
    """Newey-West HAC t-stat for the mean of a serially-correlated series."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return 0.0
    e = x - x.mean()
    var = float(e @ e) / n
    L = (int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
         if lags in ('auto', None) else int(lags))
    L = max(0, min(L, n - 1))
    for k in range(1, L + 1):
        w = 1.0 - k / (L + 1.0)
        var += 2.0 * w * float(e[k:] @ e[:-k]) / n
    if var <= 0:
        return 0.0
    return float(x.mean() / np.sqrt(var / n))


# =============================================================================
# Shared helpers (also used by research/portfolio/walk_forward.py)
# =============================================================================

def build_registry():
    """Signal registry {name: info} from the SPACES library (research/lib/spaces.py)."""
    from research.lib.spaces import build_registry_entries
    return build_registry_entries()


def signal_feature_columns(signal_def) -> List[str]:
    """Feature columns a space reads (for column-projected loading)."""
    return sorted(signal_def.columns)


def load_universe_membership() -> Optional[pd.DataFrame]:
    """Universe membership.

    Prefer a point-in-time table when available. Supported schemas:
    - universe_membership[symbol, start/end-ish date columns]
    - universe[symbol, start/end-ish date columns]

    If no dated schema exists, fall back to the current `universe` symbol list.
    That fallback is useful operationally but is not point-in-time research.
    """
    table = 'universe_membership' if get_table_columns('universe_membership') else 'universe'
    cols = get_table_columns(table)
    if not cols or 'symbol' not in cols:
        return None

    date_cols = [c for c in cols if c in (
        'timestamp', 'date', 'month', 'period', 'start_date', 'effective_date',
        'valid_from', 'end_date', 'valid_to',
    )]
    load_cols = ['symbol'] + date_cols
    mem = load_data(table, columns=load_cols)
    if mem is None or mem.empty:
        return None
    for c in date_cols:
        mem[c] = pd.to_datetime(mem[c])
    return mem.drop_duplicates().copy()


def universe_filter(df: pd.DataFrame, membership: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    Restrict a [timestamp, symbol, ...] panel to the current candidate
    universe. Missing history for newly listed names stays NaN; no monthly
    membership gate is applied.
    """
    if membership is None or df.empty:
        return df
    return df[universe_member_mask(df, membership)]


def universe_member_mask(df: pd.DataFrame,
                         membership: Optional[pd.DataFrame]) -> np.ndarray:
    """Boolean mask: row symbol is in the current candidate universe.

    All-True when membership is unavailable.
    """
    if membership is None or df.empty:
        return np.ones(len(df), dtype=bool)
    if not any(c in membership.columns for c in (
        'timestamp', 'date', 'month', 'period', 'start_date', 'effective_date',
        'valid_from',
    )):
        return df['symbol'].isin(set(membership['symbol'])).values

    m = membership.copy()
    start_col = next((c for c in ('valid_from', 'start_date', 'effective_date',
                                  'timestamp', 'date', 'month', 'period')
                      if c in m.columns), None)
    end_col = next((c for c in ('valid_to', 'end_date') if c in m.columns), None)
    if start_col is None:
        return df['symbol'].isin(set(m['symbol'])).values

    out = np.zeros(len(df), dtype=bool)
    right_cols = ['symbol', start_col] + ([end_col] if end_col else [])
    right_by_symbol = {
        sym: g[right_cols].sort_values(start_col)
        for sym, g in m.groupby('symbol', sort=False)
    }
    left = df[['timestamp', 'symbol']].copy()
    left['_row'] = np.arange(len(left))
    for sym, lgrp in left.groupby('symbol', sort=False):
        rgrp = right_by_symbol.get(sym)
        if rgrp is None or rgrp.empty:
            continue
        joined = pd.merge_asof(
            lgrp.sort_values('timestamp'),
            rgrp,
            left_on='timestamp',
            right_on=start_col,
            direction='backward',
        )
        ok = joined[start_col].notna()
        if end_col:
            ok &= joined[end_col].isna() | (joined['timestamp'] < joined[end_col])
        out[joined['_row'].to_numpy()] = ok.to_numpy()
    return out

def _smoothing_halflife_for(info: Dict) -> float:
    """Per-signal EWM smoothing halflife, set on the registry entry (0 = none)."""
    return float(info.get('smoothing_halflife', SMOOTHING_HALFLIFE) or 0.0)


def compute_signal_panel(signal_name: str, registry: Dict,
                         features: pd.DataFrame) -> pd.DataFrame:
    """
    Compute one signal at full resolution: raw space value -> EWM smoothing ->
    cross-sectional z-score, clipped at +-3. Direction applied so higher is
    always 'better'.

    Returns [timestamp, symbol, signal].
    """
    from research.lib.spaces import compute_space_raw
    info = registry[signal_name]
    raw = compute_space_raw(info['signal_def'], features)

    out = features[['timestamp', 'symbol']].copy()
    out['signal'] = raw * (info.get('direction', 1) or 1)

    # The current candidate-universe mask is applied AFTER full-history
    # transforms so smoothing, cross-sectional z-score, and evaluation only see
    # symbols we intentionally trade. Missing pre-listing history stays NaN.
    if '_is_member' in features.columns:
        out.loc[~features['_is_member'].to_numpy(), 'signal'] = np.nan

    halflife = _smoothing_halflife_for(info)
    if halflife and halflife > 0:
        out['signal'] = out.groupby('symbol')['signal'].transform(
            lambda x: x.ewm(halflife=halflife, min_periods=1).mean())

    # The final cross-sectional normalization is independent at each timestamp.
    # Evaluation only consumes screening-grid stamps, so normalize only those
    # rows after full-history time-series transforms have been computed.
    if _SCREEN_STAMPS is not None:
        out = out[out['timestamp'].isin(_SCREEN_STAMPS)].copy()

    g = out.groupby('timestamp')['signal']
    out['signal'] = (out['signal'] - g.transform('mean')) / (g.transform('std') + 1e-10)
    out['signal'] = out['signal'].clip(-3, 3)

    return out.dropna(subset=['signal'])


def rank_ic_per_timestamp(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Vectorized per-timestamp Spearman rank IC.

    df: [timestamp, signal, <target_col>]; returns [timestamp, ic, n].
    """
    if df.empty:
        return pd.DataFrame(columns=['timestamp', 'ic', 'n'])

    d = pl.DataFrame({
        'timestamp': df['timestamp'].astype('int64').to_numpy(),
        'signal': df['signal'].to_numpy(dtype=float),
        'target': df[target_col].to_numpy(dtype=float),
    })
    out = (
        d.with_columns(
            pl.col('signal').fill_nan(None).alias('signal'),
            pl.col('target').fill_nan(None).alias('target'),
        )
        .drop_nulls(['signal', 'target'])
        .with_columns(pl.len().over('timestamp').alias('n0'))
        .filter(pl.col('n0') >= MIN_ASSETS)
        .with_columns(
            pl.col('signal').rank(method='average').over('timestamp').alias('rs'),
            pl.col('target').rank(method='average').over('timestamp').alias('rr'),
        )
        .with_columns(
            (pl.col('rs') * pl.col('rr')).alias('rs_rr'),
            (pl.col('rs') * pl.col('rs')).alias('rs2'),
            (pl.col('rr') * pl.col('rr')).alias('rr2'),
        )
        .group_by('timestamp')
        .agg(
            pl.len().alias('n'),
            pl.col('rs').sum().alias('s_rs'),
            pl.col('rr').sum().alias('s_rr'),
            pl.col('rs_rr').sum().alias('s_rsrr'),
            pl.col('rs2').sum().alias('s_rs2'),
            pl.col('rr2').sum().alias('s_rr2'),
        )
        .with_columns(
            (
                (pl.col('s_rsrr') - pl.col('s_rs') * pl.col('s_rr') / pl.col('n'))
                / (
                    (
                        pl.col('s_rs2') - pl.col('s_rs') * pl.col('s_rs') / pl.col('n')
                    )
                    * (
                        pl.col('s_rr2') - pl.col('s_rr') * pl.col('s_rr') / pl.col('n')
                    )
                ).sqrt()
            ).alias('ic')
        )
        .select(
            pl.col('timestamp').cast(pl.Datetime('ns')).alias('timestamp'),
            'ic',
            'n',
        )
        .sort('timestamp')
    )
    if out.is_empty():
        return pd.DataFrame(columns=['timestamp', 'ic', 'n'])
    return out.to_pandas()


def quintile_spread_per_timestamp(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Q5-Q1 long-short spread per cross-section: mean target of the top signal
    quintile minus the bottom. Rank IC says 'monotone on average'; this says
    whether the alpha lives in the tradeable tails. Returns [timestamp, qspread].
    """
    if df.empty:
        return pd.DataFrame(columns=['timestamp', 'qspread'])

    d = pl.DataFrame({
        'timestamp': df['timestamp'].astype('int64').to_numpy(),
        'signal': df['signal'].to_numpy(dtype=float),
        'target': df[target_col].to_numpy(dtype=float),
    })
    out = (
        d.with_columns(
            pl.col('signal').fill_nan(None).alias('signal'),
            pl.col('target').fill_nan(None).alias('target'),
        )
        .drop_nulls(['signal', 'target'])
        .with_columns(
            pl.len().over('timestamp').alias('n0'),
            (
                pl.col('signal').rank(method='average').over('timestamp')
                / pl.len().over('timestamp')
            ).alias('pct'),
        )
        .filter(pl.col('n0') >= MIN_ASSETS)
        .group_by('timestamp')
        .agg(
            (
                pl.when(pl.col('pct') >= 0.8)
                .then(pl.col('target'))
                .otherwise(None)
                .mean()
                - pl.when(pl.col('pct') <= 0.2)
                .then(pl.col('target'))
                .otherwise(None)
                .mean()
            ).alias('qspread')
        )
        .drop_nulls('qspread')
        .select(
            pl.col('timestamp').cast(pl.Datetime('ns')).alias('timestamp'),
            'qspread',
        )
        .sort('timestamp')
    )
    if out.is_empty():
        return pd.DataFrame(columns=['timestamp', 'qspread'])
    return out.to_pandas()


def _empty_ic_df() -> pd.DataFrame:
    return pd.DataFrame(columns=['timestamp', 'ic', 'n'])


def _empty_qspread_df() -> pd.DataFrame:
    return pd.DataFrame(columns=['timestamp', 'qspread'])


def _empty_backtest_df() -> pd.DataFrame:
    return pd.DataFrame(columns=['timestamp', 'gross_return', 'net_return', 'turnover'])


def _empty_liq_bucket_df() -> pd.DataFrame:
    return pd.DataFrame(columns=['timestamp', 'liquidity_bucket', 'ic', 'n'])


def _backtest_from_arrays(ts_v: np.ndarray, symbols_v: np.ndarray,
                          signal_v: np.ndarray, target_v: np.ndarray) -> pd.DataFrame:
    if len(ts_v) == 0:
        return _empty_backtest_df()

    stamps, t_codes = np.unique(ts_v, return_inverse=True)
    counts = np.bincount(t_codes, minlength=len(stamps))
    keep_ts = counts >= MIN_ASSETS
    keep = keep_ts[t_codes]
    if not keep.any():
        return _empty_backtest_df()

    ts_v = ts_v[keep]
    signal_v = signal_v[keep]
    target_v = target_v[keep]
    symbols_v = symbols_v[keep]
    stamps, t_codes = np.unique(ts_v, return_inverse=True)
    counts = np.bincount(t_codes, minlength=len(stamps)).astype(float)

    sum_signal = np.bincount(t_codes, weights=signal_v, minlength=len(stamps))
    demeaned = signal_v - sum_signal[t_codes] / counts[t_codes]
    abs_sum = np.bincount(t_codes, weights=np.abs(demeaned), minlength=len(stamps))
    weights = np.divide(
        demeaned,
        abs_sum[t_codes],
        out=np.zeros_like(demeaned, dtype=float),
        where=abs_sum[t_codes] > 0,
    )
    gross = np.bincount(t_codes, weights=weights * target_v, minlength=len(stamps))

    s_codes, _ = pd.factorize(symbols_v, sort=False)
    wmat = np.zeros((len(stamps), int(s_codes.max()) + 1), dtype=float)
    wmat[t_codes, s_codes] = weights
    prev = np.vstack([np.zeros((1, wmat.shape[1]), dtype=float), wmat[:-1]])
    turnover = np.abs(wmat - prev).sum(axis=1)

    out = pd.DataFrame({
        'timestamp': pd.to_datetime(stamps),
        'gross_return': gross,
        'turnover': turnover,
    })
    out['net_return'] = out['gross_return'] - out['turnover'] * COST_BPS / 10000.0
    return out[['timestamp', 'gross_return', 'net_return', 'turnover']]


def lag_metrics(panel: pd.DataFrame, target_col: str,
                include_liquidity_buckets: bool = False):
    """
    Compute all per-timestamp diagnostics for one lag from one filtered panel.

    Returns (ics, liquid_ics, qspread, backtest) by default. With
    include_liquidity_buckets=True, appends a fifth frame with per-liquidity-
    bucket IC diagnostics. This intentionally processes one target column at a
    time: unpivoting several lags into one long table made the stride-1 case
    slower by multiplying the largest panel.
    """
    if panel.empty:
        base = (_empty_ic_df(), _empty_ic_df(), _empty_qspread_df(), _empty_backtest_df())
        return base + (_empty_liq_bucket_df(),) if include_liquidity_buckets else base

    ts = panel['timestamp'].to_numpy(dtype='datetime64[ns]')
    symbols = panel['symbol'].to_numpy()
    signal = panel['signal'].to_numpy(dtype=float)
    target = panel[target_col].to_numpy(dtype=float)
    liquid = panel['is_liquid'].to_numpy(dtype=bool)
    if include_liquidity_buckets and 'liquidity_bucket' in panel.columns:
        liq_bucket = panel['liquidity_bucket'].to_numpy(dtype=float)
    else:
        liq_bucket = np.full(len(panel), np.nan)
    valid = np.isfinite(signal) & np.isfinite(target)
    if not valid.any():
        base = (_empty_ic_df(), _empty_ic_df(), _empty_qspread_df(), _empty_backtest_df())
        return base + (_empty_liq_bucket_df(),) if include_liquidity_buckets else base

    ts_v = ts[valid]
    symbols_v = symbols[valid]
    signal_v = signal[valid]
    target_v = target[valid]
    liquid_v = liquid[valid]
    liq_bucket_v = liq_bucket[valid]

    d = pl.DataFrame({
        'timestamp': ts_v.astype('int64'),
        'signal': signal_v,
        'target': target_v,
        'is_liquid': liquid_v,
        'liquidity_bucket': liq_bucket_v,
    })

    ranked = (
        d.with_columns(pl.len().over('timestamp').alias('n0'))
        .filter(pl.col('n0') >= MIN_ASSETS)
        .with_columns(
            pl.col('signal').rank(method='average').over('timestamp').alias('rs'),
            pl.col('target').rank(method='average').over('timestamp').alias('rr'),
        )
        .with_columns(
            (pl.col('rs') / pl.col('n0')).alias('pct'),
            (pl.col('rs') * pl.col('rr')).alias('rs_rr'),
            (pl.col('rs') * pl.col('rs')).alias('rs2'),
            (pl.col('rr') * pl.col('rr')).alias('rr2'),
        )
    )
    if ranked.is_empty():
        ics = _empty_ic_df()
        qs = _empty_qspread_df()
    else:
        ics_pl = (
            ranked.group_by('timestamp')
            .agg(
                pl.len().alias('n'),
                pl.col('rs').sum().alias('s_rs'),
                pl.col('rr').sum().alias('s_rr'),
                pl.col('rs_rr').sum().alias('s_rsrr'),
                pl.col('rs2').sum().alias('s_rs2'),
                pl.col('rr2').sum().alias('s_rr2'),
            )
            .with_columns(
                (
                    (pl.col('s_rsrr') - pl.col('s_rs') * pl.col('s_rr') / pl.col('n'))
                    / (
                        (
                            pl.col('s_rs2') - pl.col('s_rs') * pl.col('s_rs') / pl.col('n')
                        )
                        * (
                            pl.col('s_rr2') - pl.col('s_rr') * pl.col('s_rr') / pl.col('n')
                        )
                    ).sqrt()
                ).alias('ic')
            )
            .select(
                pl.col('timestamp').cast(pl.Datetime('ns')).alias('timestamp'),
                'ic',
                'n',
            )
            .sort('timestamp')
        )
        ics = ics_pl.to_pandas()

        qs_pl = (
            ranked.group_by('timestamp')
            .agg(
                (
                    pl.when(pl.col('pct') >= 0.8)
                    .then(pl.col('target'))
                    .otherwise(None)
                    .mean()
                    - pl.when(pl.col('pct') <= 0.2)
                    .then(pl.col('target'))
                    .otherwise(None)
                    .mean()
                ).alias('qspread')
            )
            .drop_nulls('qspread')
            .select(
                pl.col('timestamp').cast(pl.Datetime('ns')).alias('timestamp'),
                'qspread',
            )
            .sort('timestamp')
        )
        qs = qs_pl.to_pandas() if not qs_pl.is_empty() else _empty_qspread_df()

    liq = (
        d.filter(pl.col('is_liquid'))
        .with_columns(pl.len().over('timestamp').alias('n0'))
        .filter(pl.col('n0') >= MIN_ASSETS)
        .with_columns(
            pl.col('signal').rank(method='average').over('timestamp').alias('rs'),
            pl.col('target').rank(method='average').over('timestamp').alias('rr'),
        )
        .with_columns(
            (pl.col('rs') * pl.col('rr')).alias('rs_rr'),
            (pl.col('rs') * pl.col('rs')).alias('rs2'),
            (pl.col('rr') * pl.col('rr')).alias('rr2'),
        )
    )
    if liq.is_empty():
        liq_ics = _empty_ic_df()
    else:
        liq_ics = (
            liq.group_by('timestamp')
            .agg(
                pl.len().alias('n'),
                pl.col('rs').sum().alias('s_rs'),
                pl.col('rr').sum().alias('s_rr'),
                pl.col('rs_rr').sum().alias('s_rsrr'),
                pl.col('rs2').sum().alias('s_rs2'),
                pl.col('rr2').sum().alias('s_rr2'),
            )
            .with_columns(
                (
                    (pl.col('s_rsrr') - pl.col('s_rs') * pl.col('s_rr') / pl.col('n'))
                    / (
                        (
                            pl.col('s_rs2') - pl.col('s_rs') * pl.col('s_rs') / pl.col('n')
                        )
                        * (
                            pl.col('s_rr2') - pl.col('s_rr') * pl.col('s_rr') / pl.col('n')
                        )
                    ).sqrt()
                ).alias('ic')
            )
            .select(
                pl.col('timestamp').cast(pl.Datetime('ns')).alias('timestamp'),
                'ic',
                'n',
            )
            .sort('timestamp')
            .to_pandas()
        )

    bt = _backtest_from_arrays(ts_v, symbols_v, signal_v, target_v)

    if not include_liquidity_buckets:
        return ics, liq_ics, qs, bt

    bucketed = (
        d.with_columns(pl.col('liquidity_bucket').fill_nan(None).alias('liquidity_bucket'))
        .drop_nulls(['liquidity_bucket'])
        .with_columns(pl.col('liquidity_bucket').cast(pl.Int32))
        .with_columns(pl.len().over(['timestamp', 'liquidity_bucket']).alias('n0'))
        .filter(pl.col('n0') >= max(3, MIN_ASSETS // 5))
        .with_columns(
            pl.col('signal').rank(method='average')
            .over(['timestamp', 'liquidity_bucket']).alias('rs'),
            pl.col('target').rank(method='average')
            .over(['timestamp', 'liquidity_bucket']).alias('rr'),
        )
        .with_columns(
            (pl.col('rs') * pl.col('rr')).alias('rs_rr'),
            (pl.col('rs') * pl.col('rs')).alias('rs2'),
            (pl.col('rr') * pl.col('rr')).alias('rr2'),
        )
    )
    if bucketed.is_empty():
        liq_buckets = _empty_liq_bucket_df()
    else:
        liq_buckets = (
            bucketed.group_by(['timestamp', 'liquidity_bucket'])
            .agg(
                pl.len().alias('n'),
                pl.col('rs').sum().alias('s_rs'),
                pl.col('rr').sum().alias('s_rr'),
                pl.col('rs_rr').sum().alias('s_rsrr'),
                pl.col('rs2').sum().alias('s_rs2'),
                pl.col('rr2').sum().alias('s_rr2'),
            )
            .with_columns(
                (
                    (pl.col('s_rsrr') - pl.col('s_rs') * pl.col('s_rr') / pl.col('n'))
                    / (
                        (
                            pl.col('s_rs2') - pl.col('s_rs') * pl.col('s_rs') / pl.col('n')
                        )
                        * (
                            pl.col('s_rr2') - pl.col('s_rr') * pl.col('s_rr') / pl.col('n')
                        )
                    ).sqrt()
                ).alias('ic')
            )
            .select(
                pl.col('timestamp').cast(pl.Datetime('ns')).alias('timestamp'),
                'liquidity_bucket',
                'ic',
                'n',
            )
            .sort(['timestamp', 'liquidity_bucket'])
            .to_pandas()
        )
    return ics, liq_ics, qs, bt, liq_buckets


def dollar_neutral_backtest(panel: pd.DataFrame, target_col: str,
                            stride_stamps: Optional[pd.DatetimeIndex]) -> pd.DataFrame:
    """
    Screening backtest: class-2 (dollar-neutral, leverage-1) weights at
    non-overlapping rebalance stamps; PnL = sum w * fwd target; costs on
    turnover at COST_BPS.

    Returns per-rebalance [timestamp, gross_return, net_return, turnover].
    """
    if panel.empty:
        return pd.DataFrame(columns=['timestamp', 'gross_return', 'net_return', 'turnover'])

    if stride_stamps is not None:
        panel = panel[panel['timestamp'].isin(stride_stamps)]
        if panel.empty:
            return pd.DataFrame(columns=['timestamp', 'gross_return', 'net_return', 'turnover'])

    ts = panel['timestamp'].to_numpy(dtype='datetime64[ns]')
    symbols = panel['symbol'].to_numpy()
    signal = panel['signal'].to_numpy(dtype=float)
    target = panel[target_col].to_numpy(dtype=float)
    valid = np.isfinite(signal) & np.isfinite(target)
    if not valid.any():
        return pd.DataFrame(columns=['timestamp', 'gross_return', 'net_return', 'turnover'])

    return _backtest_from_arrays(ts[valid], symbols[valid], signal[valid], target[valid])


# =============================================================================
# Worker state
# =============================================================================

_REGISTRY: Optional[Dict] = None
_TARGETS: Optional[pd.DataFrame] = None      # screening-grid targets
_SCREEN_STAMPS: Optional[set] = None
_GRID_STAMPS: Optional[np.ndarray] = None    # global sorted screening schedule
_MEMBERSHIP: Optional[pd.DataFrame] = None   # current universe symbols


def _build_targets() -> pd.DataFrame:
    """Build the screening-grid target/liquidity panel once per run/process."""
    # Lag-grid forward targets from single-bar residuals: fwd_Lb[t] = sum of
    # residuals over t+1..t+L (rolling(L).sum().shift(-L), full resolution),
    # then subset to the screening grid and evaluation window.
    res = load_data('residual_returns',
                    columns=['timestamp', 'symbol', 'residual_return', 'raw_return'])
    if res.empty:
        raise RuntimeError("residual_returns is empty - run risk_model/residual_returns.py")
    res['timestamp'] = pd.to_datetime(res['timestamp'])
    wide = res.pivot_table(index='timestamp', columns='symbol',
                           values='residual_return',
                           aggfunc='first').sort_index().astype(np.float32)
    raw_wide = res.pivot_table(index='timestamp', columns='symbol',
                               values='raw_return',
                               aggfunc='first').sort_index().astype(np.float32)
    raw_wide = raw_wide.reindex(index=wide.index, columns=wide.columns)
    del res

    sel = ((wide.index == wide.index.floor(SCREENING_GRID)) &
           (wide.index >= pd.Timestamp(START_DATE)) &
           (wide.index < pd.Timestamp(END_DATE)))

    frames = {}
    for lag in LAG_GRID:
        fwd = wide.rolling(lag, min_periods=lag).sum().shift(-lag)
        frames[_lag_col(lag)] = fwd[sel].stack(future_stack=True)
        fwd_raw = raw_wide.rolling(lag, min_periods=lag).sum().shift(-lag)
        frames[_raw_lag_col(lag)] = fwd_raw[sel].stack(future_stack=True)
    targets = pd.DataFrame(frames)
    targets = targets.dropna(how='all').reset_index()
    target_cols = []
    for lag in LAG_GRID:
        target_cols.extend([_lag_col(lag), _raw_lag_col(lag)])
    targets.columns = ['timestamp', 'symbol'] + target_cols

    # Liquid-half flag: top half by trailing dollar volume at each stamp.
    # A signal whose IC lives only in the illiquid tail dies on impact costs.
    try:
        px = load_data('prices', columns=['timestamp', 'symbol', 'quote_asset_volume'])
        px['timestamp'] = pd.to_datetime(px['timestamp'])
        qv = px.pivot_table(index='timestamp', columns='symbol',
                            values='quote_asset_volume',
                            aggfunc='first').sort_index()
        qv = qv.reindex(index=wide.index, columns=wide.columns)
        trail = qv.rolling(LIQ_WINDOW, min_periods=LIQ_WINDOW // 4).mean()
        liq = (trail[sel].rank(axis=1, pct=True) >= 0.5).stack(future_stack=True)
        liq = liq.rename('is_liquid').reset_index()
        liq.columns = ['timestamp', 'symbol', 'is_liquid']
        targets = targets.merge(liq, on=['timestamp', 'symbol'], how='left')
        targets['is_liquid'] = targets['is_liquid'].fillna(False).astype(bool)
        bucket = np.ceil(trail[sel].rank(axis=1, pct=True) * 10.0)
        bucket = bucket.clip(lower=1, upper=10).stack(future_stack=True)
        bucket = bucket.rename('liquidity_bucket').reset_index()
        bucket.columns = ['timestamp', 'symbol', 'liquidity_bucket']
        targets = targets.merge(bucket, on=['timestamp', 'symbol'], how='left')
        targets['liquidity_bucket'] = targets['liquidity_bucket'].astype('float64')
        del px, qv, trail
    except Exception as e:
        logging.warning(f"liquidity flag unavailable: {e}")
        targets['is_liquid'] = True
        targets['liquidity_bucket'] = np.nan
    del wide, raw_wide

    return targets


def _set_targets(targets: pd.DataFrame) -> None:
    global _TARGETS, _SCREEN_STAMPS, _GRID_STAMPS
    targets['timestamp'] = pd.to_datetime(targets['timestamp'])
    if 'is_liquid' in targets.columns:
        targets['is_liquid'] = targets['is_liquid'].fillna(False).astype(bool)
    if 'liquidity_bucket' not in targets.columns:
        targets['liquidity_bucket'] = np.nan
    _TARGETS = targets
    _SCREEN_STAMPS = set(targets['timestamp'].unique())
    # One global evaluation schedule so every signal is compared on identical
    # stamps (per-signal stamp sets would drift with each signal's coverage).
    _GRID_STAMPS = np.sort(targets['timestamp'].unique())


def _required_target_columns() -> set:
    return {_lag_col(l) for l in LAG_GRID} | {_raw_lag_col(l) for l in LAG_GRID}


def _init_state():
    """Lazy per-process initialization: registry + screening-grid targets."""
    global _REGISTRY, _MEMBERSHIP
    if _REGISTRY is not None:
        return

    _REGISTRY = build_registry()
    _MEMBERSHIP = load_universe_membership()
    if _MEMBERSHIP is None:
        logging.warning("universe unavailable - evaluating on the full "
                        "feature panel")

    cache_path = os.environ.get(_TARGET_CACHE_ENV)
    if cache_path and Path(cache_path).exists():
        cached = pd.read_parquet(cache_path)
        required = (_required_target_columns() | {'is_liquid', 'liquidity_bucket'})
        if required.issubset(cached.columns):
            _set_targets(cached)
        else:
            _set_targets(_build_targets())
    else:
        _set_targets(_build_targets())


def _target_cache_is_compatible(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        import pyarrow.parquet as pq
        cols = set(pq.ParquetFile(path).schema_arrow.names)
    except Exception:
        try:
            cols = set(pd.read_parquet(path).columns)
        except Exception:
            return False
    required = (_required_target_columns() | {'is_liquid', 'liquidity_bucket'})
    return required.issubset(cols)


def configure_screening_grid(grid: str) -> None:
    """Override screening grid for this process and spawned workers."""
    global SCREENING_GRID, _GRID_BARS, _REGISTRY, _TARGETS, _SCREEN_STAMPS, _GRID_STAMPS
    if not grid:
        return
    SCREENING_GRID = grid
    _GRID_BARS = horizon_bars(SCREENING_GRID)
    os.environ['CRYPTO_SIGNAL_SCREENING_GRID'] = grid
    os.environ.pop(_TARGET_CACHE_ENV, None)
    _REGISTRY = None
    _TARGETS = None
    _SCREEN_STAMPS = None
    _GRID_STAMPS = None


def _prepare_target_cache() -> Optional[Path]:
    """Materialize targets once so spawned workers do not rebuild them."""
    cache_env = os.environ.get(_TARGET_CACHE_ENV)
    if cache_env:
        cache_path = Path(cache_env)
        if _target_cache_is_compatible(cache_path):
            return cache_path
        os.environ.pop(_TARGET_CACHE_ENV, None)
    started = time.perf_counter()
    print("Building shared evaluation data: forward residual targets, "
          "screening timestamps, and liquid-half flags...", flush=True)
    _init_state()
    path = Path(os.environ.get('TMPDIR', '/tmp')) / (
        f"crypto_signal_eval_targets_{os.getpid()}_{int(time.time())}.parquet")
    _TARGETS.to_parquet(path, index=False)
    os.environ[_TARGET_CACHE_ENV] = str(path)
    print(f"Shared research state ready in "
          f"{time.perf_counter() - started:,.1f}s", flush=True)
    return path


def _load_features(columns: List[str]) -> pd.DataFrame:
    """Load + universe-filter the feature panel for a set of columns."""
    features = load_data('features', columns=['timestamp', 'symbol'] + list(columns))
    if features.empty:
        return features
    features['timestamp'] = pd.to_datetime(features['timestamp'])
    if COMPUTE_ON_FULL_HISTORY:
        features['_is_member'] = universe_member_mask(features, _MEMBERSHIP)
    else:
        features = universe_filter(features, _MEMBERSHIP)
    return features.sort_values(['symbol', 'timestamp']).reset_index(drop=True)


def evaluate_signal(signal_name: str,
                    features: pd.DataFrame) -> Tuple[pd.DataFrame, List[dict]]:
    """
    Evaluate one signal across all horizons on a pre-loaded feature panel
    (already universe-filtered and sorted; must contain the signal's columns).

    Returns (daily_stats_df, summary_rows). daily_stats has one row per
    (horizon, date): pooled IC moments + backtest aggregates.
    """
    _init_state()

    info = _REGISTRY.get(signal_name)
    if info is None:
        return pd.DataFrame(), [{'signal_name': signal_name, 'horizon': '',
                                 'error': 'not in registry'}]
    if features.empty:
        return pd.DataFrame(), [{'signal_name': signal_name, 'horizon': '',
                                 'error': 'features empty'}]

    panel = compute_signal_panel(signal_name, _REGISTRY, features)

    # Screening grid only, merged with targets
    panel = panel[panel['timestamp'].isin(_SCREEN_STAMPS)]
    panel = panel.merge(_TARGETS, on=['timestamp', 'symbol'], how='inner')
    if len(panel) < 1000:
        return pd.DataFrame(), [{'signal_name': signal_name, 'horizon': '',
                                 'error': 'insufficient overlap with targets'}]

    # Anchor strides to the GLOBAL screening schedule (not this signal's own
    # observed stamps) so every signal is compared on identical calendar stamps.
    grid_stamps = _GRID_STAMPS
    stamp_pos = pd.Series(np.arange(len(grid_stamps), dtype=np.int32),
                          index=grid_stamps)
    panel['_stamp_pos'] = panel['timestamp'].map(stamp_pos).astype(np.int32)

    daily_frames = []
    summaries = []

    for lag in LAG_GRID:
        tcol = _lag_col(lag)
        raw_tcol = _raw_lag_col(lag)
        if tcol not in panel.columns:
            continue

        # ICs and backtest both at NON-OVERLAPPING stamps: dense stamps share
        # most of the forward window, autocorrelate the IC series and inflate
        # the t-stat ~sqrt(stride) - the FDR gate would be anti-conservative.
        stride = _lag_stride(lag)
        h_panel = panel[panel['_stamp_pos'] % stride == 0]

        metric_cols = ['timestamp', 'symbol', 'signal', 'is_liquid',
                       'liquidity_bucket', tcol]
        if raw_tcol in h_panel.columns:
            metric_cols.append(raw_tcol)
        h_metric = h_panel[[c for c in metric_cols if c in h_panel.columns]]
        ics, liq_ics, qs, bt, liq_bucket_ics = lag_metrics(
            h_metric, tcol, include_liquidity_buckets=True)
        if ics.empty:
            continue
        ics['date'] = ics['timestamp'].dt.normalize()
        ics['ic_sq'] = ics['ic'] * ics['ic']
        ic_daily = ics.groupby('date').agg(
            ic_sum=('ic', 'sum'),
            ic_sumsq=('ic_sq', 'sum'),
            n_cs=('ic', 'size'),
        ).reset_index()

        raw_ics = raw_qs = raw_bt = pd.DataFrame()
        if raw_tcol in h_metric.columns:
            raw_ics, _, raw_qs, raw_bt = lag_metrics(h_metric, raw_tcol)
            if not raw_ics.empty:
                raw_ics['date'] = raw_ics['timestamp'].dt.normalize()
                raw_ics['ic_sq'] = raw_ics['ic'] * raw_ics['ic']
                raw_ic_daily = raw_ics.groupby('date').agg(
                    raw_ic_sum=('ic', 'sum'),
                    raw_ic_sumsq=('ic_sq', 'sum'),
                    raw_n_cs=('ic', 'size'),
                ).reset_index()
            else:
                raw_ic_daily = pd.DataFrame(columns=['date', 'raw_ic_sum',
                                                     'raw_ic_sumsq', 'raw_n_cs'])
        else:
            raw_ic_daily = pd.DataFrame(columns=['date', 'raw_ic_sum',
                                                 'raw_ic_sumsq', 'raw_n_cs'])

        # IC restricted to the liquid half (capacity check)
        if not liq_ics.empty:
            liq_ics['date'] = liq_ics['timestamp'].dt.normalize()
            liq_daily = liq_ics.groupby('date').agg(
                liq_ic_sum=('ic', 'sum'), n_liq=('ic', 'size')).reset_index()
        else:
            liq_daily = pd.DataFrame(columns=['date', 'liq_ic_sum', 'n_liq'])

        # Q5-Q1 tail spread (monotonicity check)
        if not qs.empty:
            qs['date'] = qs['timestamp'].dt.normalize()
            qs_daily = qs.groupby('date').agg(
                qs_sum=('qspread', 'sum'), n_qs=('qspread', 'size')).reset_index()
        else:
            qs_daily = pd.DataFrame(columns=['date', 'qs_sum', 'n_qs'])

        if not liq_bucket_ics.empty:
            liq_bucket_ics['date'] = liq_bucket_ics['timestamp'].dt.normalize()
            top = liq_bucket_ics[liq_bucket_ics['liquidity_bucket'] == 10]
            bot = liq_bucket_ics[liq_bucket_ics['liquidity_bucket'] == 1]
            top_daily = top.groupby('date').agg(
                liq_top_ic_sum=('ic', 'sum'),
                n_liq_top=('ic', 'size'),
            ).reset_index()
            bot_daily = bot.groupby('date').agg(
                liq_bottom_ic_sum=('ic', 'sum'),
                n_liq_bottom=('ic', 'size'),
            ).reset_index()
        else:
            top_daily = pd.DataFrame(columns=['date', 'liq_top_ic_sum', 'n_liq_top'])
            bot_daily = pd.DataFrame(columns=['date', 'liq_bottom_ic_sum', 'n_liq_bottom'])

        if not bt.empty:
            bt['date'] = bt['timestamp'].dt.normalize()
            bt_daily = bt.groupby('date').agg(
                ret_gross=('gross_return', 'sum'),
                ret_net=('net_return', 'sum'),
                turnover=('turnover', 'sum'),
                n_rebalances=('gross_return', 'size'),
            ).reset_index()
        else:
            bt_daily = pd.DataFrame(columns=['date', 'ret_gross', 'ret_net',
                                             'turnover', 'n_rebalances'])

        if not raw_bt.empty:
            raw_bt['date'] = raw_bt['timestamp'].dt.normalize()
            raw_bt_daily = raw_bt.groupby('date').agg(
                raw_ret_gross=('gross_return', 'sum'),
                raw_ret_net=('net_return', 'sum'),
            ).reset_index()
        else:
            raw_bt_daily = pd.DataFrame(columns=['date', 'raw_ret_gross',
                                                 'raw_ret_net'])

        daily = ic_daily.merge(liq_daily, on='date', how='outer') \
                        .merge(qs_daily, on='date', how='outer') \
                        .merge(top_daily, on='date', how='outer') \
                        .merge(bot_daily, on='date', how='outer') \
                        .merge(bt_daily, on='date', how='outer') \
                        .merge(raw_ic_daily, on='date', how='outer') \
                        .merge(raw_bt_daily, on='date', how='outer')
        daily['signal_name'] = signal_name
        daily['horizon'] = lag_label(lag)
        daily_frames.append(daily)

        # Whole-period diagnostics
        ic_all = ics['ic']
        n = len(ic_all)
        ic_mean = float(ic_all.mean())
        ic_std = float(ic_all.std())
        daily_ic = ic_daily['ic_sum'] / ic_daily['n_cs'].replace(0, np.nan)
        raw_ic_all = raw_ics['ic'] if isinstance(raw_ics, pd.DataFrame) and not raw_ics.empty else pd.Series(dtype=float)
        raw_ic_mean = float(raw_ic_all.mean()) if len(raw_ic_all) else np.nan
        raw_ic_std = float(raw_ic_all.std()) if len(raw_ic_all) else np.nan
        raw_daily_ic = (raw_ic_daily['raw_ic_sum'] /
                        raw_ic_daily['raw_n_cs'].replace(0, np.nan)
                        if len(raw_ic_daily) else pd.Series(dtype=float))
        liq_curve = {}
        if not liq_bucket_ics.empty:
            liq_curve = liq_bucket_ics.groupby('liquidity_bucket')['ic'].mean().to_dict()
        liq_bottom = float(liq_curve.get(1, np.nan))
        liq_top = float(liq_curve.get(10, np.nan))
        summaries.append({
            'signal_name': signal_name,
            'horizon': lag_label(lag),
            'ic_mean': ic_mean,
            'ic_std': ic_std,
            'ic_tstat': ic_mean / (ic_std / np.sqrt(n)) if ic_std > 0 else 0.0,
            'ic_tstat_hac': _nw_tstat(daily_ic.values, 'auto'),
            'icir': ic_mean / ic_std if ic_std > 0 else 0.0,
            'ic_liquid': float(liq_ics['ic'].mean()) if len(liq_ics) else np.nan,
            'ic_liq_bottom': liq_bottom,
            'ic_liq_top': liq_top,
            'ic_liq_top_minus_bottom': (
                liq_top - liq_bottom
                if np.isfinite(liq_top) and np.isfinite(liq_bottom) else np.nan
            ),
            'q_spread': float(qs['qspread'].mean()) if len(qs) else np.nan,
            'raw_ic_mean': raw_ic_mean,
            'raw_ic_tstat_hac': _nw_tstat(raw_daily_ic.values, 'auto'),
            'raw_q_spread': (
                float(raw_qs['qspread'].mean())
                if isinstance(raw_qs, pd.DataFrame) and len(raw_qs) else np.nan
            ),
            'n_cross_sections': n,
            'avg_n_assets': float(ics['n'].mean()) if len(ics) else 0.0,
            'avg_daily_net_ret': float(bt_daily['ret_net'].mean()) if len(bt_daily) else np.nan,
            'avg_daily_raw_net_ret': (
                float(raw_bt_daily['raw_ret_net'].mean())
                if len(raw_bt_daily) else np.nan
            ),
            'avg_turnover_per_rebalance': float(bt['turnover'].mean()) if len(bt) else np.nan,
            'error': '',
        })

    daily_df = pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame()
    return daily_df, summaries


def _error_result(signal_name: str, e: Exception):
    return signal_name, (pd.DataFrame(),
                         [{'signal_name': signal_name, 'horizon': '',
                           'error': str(e)[:200]}])


def _evaluate_group_worker(args: Tuple[Tuple[str, ...], List[str]]):
    """Evaluate all signals sharing one feature-column set on a single load.

    Signals cluster by their input columns (every lookback variant of a
    transform reads the same features), so loading per GROUP instead of per
    signal cuts feature-table reads ~30x.
    """
    columns, signal_names = args
    try:
        _init_state()
        features = _load_features(list(columns))
    except Exception as e:
        return [_error_result(s, e) for s in signal_names]

    results = []
    for name in signal_names:
        try:
            results.append((name, evaluate_signal(name, features)))
        except Exception as e:
            results.append(_error_result(name, e))
    return results


def _batch_signal_groups(
    groups: Dict[Tuple[str, ...], List[str]],
    max_columns: int,
    max_signals: int = 0,
) -> List[Tuple[Tuple[str, ...], List[str]]]:
    """Greedily pack compatible groups into bounded feature-column batches.

    The old exact-column grouping still scanned the 15M-row features table once
    per group (425 scans in the current universe). Packing groups that share
    columns reduces that to roughly a few dozen scans without loading the full
    167-column table into memory.
    """
    batches: List[Tuple[set, List[str]]] = []
    ordered = sorted(groups.items(),
                     key=lambda item: (-len(item[0]), -len(item[1]), item[0]))
    for columns, names in ordered:
        needed = set(columns)
        limit = max(max_columns, len(needed))
        best_idx = None
        best_score = None
        for idx, (batch_columns, batch_names) in enumerate(batches):
            union = batch_columns | needed
            if len(union) > limit:
                continue
            score = (len(batch_columns & needed), -len(union), -len(batch_names))
            if best_score is None or score > best_score:
                best_idx, best_score = idx, score
        if best_idx is None:
            batches.append((set(needed), list(names)))
        else:
            batches[best_idx][0].update(needed)
            batches[best_idx][1].extend(names)
    packed = []
    for columns, names in batches:
        names = sorted(names)
        chunk = max_signals if max_signals > 0 else len(names)
        for start in range(0, len(names), chunk):
            packed.append((tuple(sorted(columns)), names[start:start + chunk]))
    return packed


# =============================================================================
# Runner
# =============================================================================

def _processed_signals() -> set:
    try:
        existing = load_data('signal_metrics', columns=['signal_name'])
        if existing is not None and not existing.empty:
            return set(existing['signal_name'].unique())
    except Exception:
        pass
    return set()


def _flush(daily_buffer: list, summary_buffer: list):
    # Delete-then-append per signal_name so a re-flush *replaces* rather than
    # *adds*. This makes the writer crash-atomic: if a previous run died between
    # the two appends below (daily written, metrics not), the affected signals
    # are reprocessed on resume and their stale daily rows are removed here
    # instead of duplicating.
    if daily_buffer:
        df = pd.concat(daily_buffer, ignore_index=True)
        delete_rows_in('signal_daily_stats', 'signal_name', df['signal_name'].unique().tolist())
        save_data('signal_daily_stats', df, mode='append', datetime_columns=['date'])
        daily_buffer.clear()
    if summary_buffer:
        df = pd.DataFrame(summary_buffer)
        delete_rows_in('signal_metrics', 'signal_name', df['signal_name'].unique().tolist())
        save_data('signal_metrics', df, mode='append')
        summary_buffer.clear()


def run_all(parallel: bool = True, fresh_start: bool = False, limit: int = 0,
            signal: str = ''):
    from tqdm import tqdm

    registry = build_registry()
    all_signals = sorted(registry.keys())
    if signal:
        matches = [s for s in all_signals if s == signal]
        if not matches:
            matches = [s for s in all_signals if signal in s]
        if not matches:
            raise SystemExit(f"No signal matches: {signal}")
        all_signals = matches
    print(f"Signal universe: {len(all_signals)} signals x {len(LAG_GRID)} lags")

    if fresh_start and signal:
        n_daily = sum(delete_rows_where('signal_daily_stats', 'signal_name', s)
                      for s in all_signals)
        n_metrics = sum(delete_rows_where('signal_metrics', 'signal_name', s)
                        for s in all_signals)
        print(f"Fresh signal run: removed {n_daily:,} daily rows and "
              f"{n_metrics:,} metric rows for {len(all_signals)} signal(s)")
        signals = all_signals
    elif fresh_start:
        print("Fresh start: clearing signal_daily_stats and signal_metrics")
        delete_table('signal_daily_stats')
        delete_table('signal_metrics')
        signals = all_signals
    else:
        done = _processed_signals()
        signals = [s for s in all_signals if s not in done]
        if done:
            print(f"Skipping {len(done)} already-processed signals")

    if limit > 0:
        signals = signals[:limit]
    if not signals:
        print("Nothing to do")
        return

    # Group signals by their feature-column set: one load per group
    groups: Dict[Tuple[str, ...], List[str]] = {}
    for s in signals:
        key = tuple(signal_feature_columns(registry[s]['signal_def']))
        groups.setdefault(key, []).append(s)
    max_batch_columns = get('compute.signal_batch_max_columns', 8)
    max_batch_signals = get('compute.signal_batch_max_signals', 200)
    group_items = _batch_signal_groups(groups, max_batch_columns,
                                       max_batch_signals)
    # Start with smaller batches so visible progress arrives quickly, then let
    # longest-processing-first dominate the remaining makespan.
    group_items.sort(key=lambda item: len(item[1]))

    print(f"Evaluating {len(signals)} signals in {len(group_items)} "
          f"column batches from {len(groups)} groups "
          f"(max {max_batch_columns} columns and {max_batch_signals} "
          f"signals/batch, screening grid "
          f"{SCREENING_GRID}, cost {COST_BPS}bps)")

    daily_buffer, summary_buffer = [], []
    flush_every = 200
    n_ok = n_err = 0

    def _consume(group_results):
        nonlocal n_ok, n_err
        for _, (daily, summaries) in group_results:
            if not daily.empty:
                daily_buffer.append(daily)
            summary_buffer.extend(summaries)
            if any(s.get('error') for s in summaries):
                n_err += 1
            else:
                n_ok += 1
        if len(summary_buffer) >= flush_every * len(LAG_GRID):
            _flush(daily_buffer, summary_buffer)

    if parallel:
        max_workers = get('compute.signal_workers', 6)
        # Polars' runtime is NOT fork-safe: forking after the parent has used
        # Polars deadlocks the workers (they hang on the first Polars call in
        # _load_features). Default to 'spawn', where each worker starts a clean
        # interpreter and builds its own panel via _init_state(). Only 'fork'
        # benefits from a parent-side preload (copy-on-write sharing); under
        # 'spawn' that preload is wasted work, so skip it.
        start_method = get('compute.signal_start_method', 'spawn')
        if start_method == 'fork':
            print("Building shared evaluation data: forward residual targets, "
                  "screening timestamps, and liquid-half flags...", flush=True)
            init_started = time.perf_counter()
            _init_state()
            print(f"Shared research state ready in "
                  f"{time.perf_counter() - init_started:,.1f}s", flush=True)
        elif start_method == 'spawn':
            _prepare_target_cache()
        with get_parallel_executor(max_workers, start_method=start_method) as executor:
            from concurrent.futures import FIRST_COMPLETED, wait
            pending = {}
            next_item = 0
            while next_item < len(group_items) and len(pending) < max_workers:
                item = group_items[next_item]
                pending[executor.submit(_evaluate_group_worker, item)] = item
                next_item += 1
            with tqdm(total=len(signals), desc="Signals") as pbar:
                while pending:
                    done, _ = wait(set(pending), return_when=FIRST_COMPLETED)
                    for fut in done:
                        _, names = pending.pop(fut)
                        try:
                            group_results = fut.result()
                        except Exception as e:
                            group_results = [_error_result(n, e) for n in names]
                        _consume(group_results)
                        pbar.update(len(names))
                        if next_item < len(group_items):
                            item = group_items[next_item]
                            pending[executor.submit(
                                _evaluate_group_worker, item)] = item
                            next_item += 1
    else:
        with tqdm(total=len(signals), desc="Signals") as pbar:
            for item in group_items:
                _consume(_evaluate_group_worker(item))
                pbar.update(len(item[1]))
    # (flush threshold below counts summary rows: one per signal per lag)

    _flush(daily_buffer, summary_buffer)
    print(f"\nDone. OK: {n_ok}, errors: {n_err}")


def _list_signals(category: str = '', contains: str = '', limit: int = 100) -> None:
    """Print signals matching the filters (name, category, type, columns)."""
    registry = build_registry()
    rows = []
    for name, info in sorted(registry.items()):
        if category and info['category'] != category:
            continue
        if contains and contains not in name:
            continue
        sdef = info['signal_def']
        rows.append((name, info['category'], getattr(sdef, 'signal_type', info.get('kind', '')),
                     ','.join(signal_feature_columns(sdef))))
    for name, cat, stype, cols in rows[:limit]:
        print(f"{name:44s} {cat:18s} {stype:24s} {cols}")
    if len(rows) > limit:
        print(f"... {len(rows) - limit} more")
    print(f"{len(rows)} matching signals")


def _resolve_signal(name: str) -> str:
    """Resolve a signal name: exact match wins, else a unique substring match.
    Lists candidates and exits when the name is unknown or ambiguous."""
    registry = build_registry()
    if name in registry:
        return name
    matches = sorted(s for s in registry if name in s)
    if not matches:
        raise SystemExit(f"Unknown signal: {name!r} (use --list to browse)")
    if len(matches) > 1:
        print(f"{name!r} matches {len(matches)} signals - be specific:")
        for s in matches[:50]:
            print(f"  {s}")
        raise SystemExit(1)
    return matches[0]


def evaluate_one(name: str, save: bool = True, days: int = 0) -> None:
    """Evaluate a single signal and print compact horizon diagnostics.

    save=True (default) refreshes this signal's rows in signal_daily_stats /
    signal_metrics; save=False is the read-only dev loop (compute + print only).
    """
    _init_state()
    registry = build_registry()
    columns = signal_feature_columns(registry[name]['signal_def'])
    features = _load_features(columns)
    daily, summaries = evaluate_signal(name, features)

    shown = daily
    if days and not daily.empty:
        cutoff = pd.Timestamp(daily['date'].max()) - pd.Timedelta(days=days)
        shown = daily[pd.to_datetime(daily['date']) >= cutoff]

    print(f"\nSignal: {name}")
    print(f"Columns: {', '.join(columns) if columns else '(none)'}")
    print(f"Daily rows: {len(shown):,}")
    if summaries:
        summary = pd.DataFrame(summaries)
        cols = ['horizon', 'ic_mean', 'ic_tstat_hac', 'ic_tstat', 'icir',
                'ic_liquid', 'ic_liq_bottom', 'ic_liq_top',
                'q_spread', 'raw_ic_mean', 'raw_ic_tstat_hac',
                'raw_q_spread', 'n_cross_sections', 'avg_n_assets',
                'avg_daily_net_ret', 'avg_daily_raw_net_ret',
                'avg_turnover_per_rebalance', 'error']
        print(summary[[c for c in cols if c in summary.columns]].to_string(index=False))
    else:
        print("No summaries produced")

    if save:
        n_daily = delete_rows_where('signal_daily_stats', 'signal_name', name)
        n_metrics = delete_rows_where('signal_metrics', 'signal_name', name)
        _flush([daily] if not daily.empty else [], list(summaries))
        print(f"Saved (refreshed {n_daily:,} daily / {n_metrics:,} metric rows)")
    else:
        print("Not saved (--no-save): diagnostics only, tables untouched")


def main():
    parser = argparse.ArgumentParser(
        description='Signal evaluation - multi-horizon IC against forward residuals')
    parser.add_argument('signal', nargs='?',
                        help='Evaluate only this signal (exact or unique substring); '
                             'prints diagnostics and refreshes its rows. Omit to run all.')
    parser.add_argument('--no-save', action='store_true',
                        help='Single-signal dev loop: compute + print diagnostics, '
                             'do not write to signal_daily_stats / signal_metrics')
    parser.add_argument('--days', type=int, default=0,
                        help='Single-signal: summarize only the last N days of daily stats')
    parser.add_argument('--list', action='store_true',
                        help='List signals matching --category / --contains and exit')
    parser.add_argument('--category', default='', help='Filter --list by category')
    parser.add_argument('--contains', default='', help='Filter --list by name substring')
    parser.add_argument('--fresh', action='store_true',
                        help='Full run: clear all results first')
    parser.add_argument('--no-parallel', action='store_true', help='Full run: serial')
    parser.add_argument('--limit', type=int, default=0, help='Full run: first N signals only')
    parser.add_argument('--screening-grid', default='',
                        help='Override signals.screening_grid for this run')
    args = parser.parse_args()

    configure_screening_grid(args.screening_grid)

    if args.list or (not args.signal and (args.category or args.contains)):
        _list_signals(args.category, args.contains, args.limit or 100)
        return

    if args.signal:
        evaluate_one(_resolve_signal(args.signal), save=not args.no_save, days=args.days)
        return

    run_all(parallel=not args.no_parallel, fresh_start=args.fresh,
            limit=args.limit)


if __name__ == '__main__':
    main()
