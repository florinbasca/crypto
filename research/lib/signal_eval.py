"""
Signal scoring library: multi-horizon evaluation against forward residual
returns. No CLI, no persistence - the walk-forward calls score_registry() to
score the promoted disc_* signals in memory at startup (discovery is the only
signal source; the hand-curated library and the standalone evaluate.py
pipeline stage are retired).

Scoring per signal and horizon: compute at FULL base-frequency resolution,
smooth at a lag-matched halflife, cross-sectionally z-score, then rank-IC and
a dollar-neutral screening backtest at NON-OVERLAPPING stamps on the
screening grid. The fwd_* targets are ALREADY forward sums over t+1..t+p -
consumed with NO additional shift (a double shift here was a real historical
bug; do not reintroduce). Overlapping stamps would autocorrelate the IC
series and inflate t-stats ~sqrt(stride).
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl

from dbutil import load_data, get_table_columns
from config import config as global_config, get, horizon_bars

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.WARNING,
                    format=global_config['logging']['format'],
                    datefmt=global_config['logging']['datefmt'])

SCREENING_GRID = get('signals.screening_grid', '1h')
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
    """Signal registry {name: info}: every promoted discovery candidate as
    a disc_* entry (research/lib/discovered.py) plus any hand-written spaces
    (research/lib/spaces.py - currently empty by design). Discovered entries
    carry valid_from = their promotion date, which the selector honours."""
    from research.lib.spaces import build_registry_entries
    entries = build_registry_entries()
    from research.lib.discovered import load_discovered_entries
    entries.update(load_discovered_entries())
    return entries


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

    # merge_asof requires matching datetime units; pandas 3 mixes [s]/[us]/
    # [ns] depending on how each side was built, so pin both sides to ns.
    for c in [start_col] + ([end_col] if end_col else []):
        m[c] = pd.to_datetime(m[c]).astype('datetime64[ns]')

    out = np.zeros(len(df), dtype=bool)
    right_cols = ['symbol', start_col] + ([end_col] if end_col else [])
    right_by_symbol = {
        sym: g[right_cols].sort_values(start_col)
        for sym, g in m.groupby('symbol', sort=False)
    }
    left = df[['timestamp', 'symbol']].copy()
    left['timestamp'] = pd.to_datetime(left['timestamp']).astype('datetime64[ns]')
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


def smoothing_halflife_for_lag(lag: int, base_halflife: float = 0.0) -> float:
    """Effective EWM halflife (bars) for a signal scored at forward lag `lag`.

    Picks the smallest `signals.lag_smoothing` bucket with max_lag >= lag
    (lags beyond the last bound use the last bucket), floored by the
    per-space / global base halflife. Rationale: a slow holding lag tolerates
    a slow signal - measured on this panel, smoothing at halflife 36 vs 3 cuts
    signal turnover ~2.5-3x while keeping 75-85% of the IC, roughly doubling
    gross-per-turnover exactly where turnover dominates. Empty config ->
    base halflife only (legacy single-speed behaviour).
    """
    buckets = get('signals.lag_smoothing') or []
    base = float(base_halflife or 0.0)
    if not buckets:
        return base
    hl = float(buckets[-1][1])
    for max_lag, bucket_hl in buckets:
        if lag <= max_lag:
            hl = float(bucket_hl)
            break
    return max(base, hl)


def effective_halflife_for(info: Dict, lag: int) -> float:
    """Per-(signal, lag) smoothing halflife: lag bucket, floored by the
    space's own halflife. Used by evaluation AND the walk-forward composite
    recompute, so a signal selected at lag L is traded exactly as scored."""
    return smoothing_halflife_for_lag(lag, _smoothing_halflife_for(info))


def _raw_signal_frame(signal_name: str, registry: Dict,
                      features: pd.DataFrame) -> pd.DataFrame:
    """Raw space value x direction at FULL resolution, membership-masked -
    no smoothing, no normalization. Returns [timestamp, symbol, signal_raw].

    The candidate-universe mask is applied BEFORE the time-series transforms
    so smoothing and the cross-sectional z-score only ever see symbols we
    intentionally trade. Missing pre-listing history stays NaN.
    """
    from research.lib.spaces import compute_space_raw
    info = registry[signal_name]
    out = features[['timestamp', 'symbol']].copy()
    out['signal_raw'] = (compute_space_raw(info['signal_def'], features)
                         * (info.get('direction', 1) or 1))
    if '_is_member' in features.columns:
        out.loc[~features['_is_member'].to_numpy(), 'signal_raw'] = np.nan
    return out


def compute_signal_panel(signal_name: str, registry: Dict,
                         features: pd.DataFrame,
                         halflife: Optional[float] = None) -> pd.DataFrame:
    """
    Compute one signal at full resolution: raw space value -> EWM smoothing ->
    cross-sectional z-score, clipped at +-3. Direction applied so higher is
    always 'better'. `halflife` overrides the space's own smoothing halflife:
    the walk-forward passes the per-lag effective halflife of the SELECTED lag
    (effective_halflife_for), so the traded signal matches the scored one.

    Returns [timestamp, symbol, signal].
    """
    info = registry[signal_name]
    out = _raw_signal_frame(signal_name, registry, features).rename(
        columns={'signal_raw': 'signal'})

    hl = _smoothing_halflife_for(info) if halflife is None else float(halflife)
    if hl and hl > 0:
        out['signal'] = out.groupby('symbol')['signal'].transform(
            lambda x: x.ewm(halflife=hl, min_periods=1).mean())

    # The final cross-sectional normalization is independent at each timestamp.
    # Evaluation only consumes screening-grid stamps, so normalize only those
    # rows after full-history time-series transforms have been computed.
    if _SCREEN_STAMPS is not None:
        out = out[out['timestamp'].isin(_SCREEN_STAMPS)].copy()

    g = out.groupby('timestamp')['signal']
    out['signal'] = (out['signal'] - g.transform('mean')) / (g.transform('std') + 1e-10)
    out['signal'] = out['signal'].clip(-3, 3)

    return out.dropna(subset=['signal'])


def rank_ic_per_timestamp(df: pd.DataFrame, target_col: str,
                          min_assets: Optional[int] = None) -> pd.DataFrame:
    """
    Vectorized per-timestamp Spearman rank IC.

    df: [timestamp, signal, <target_col>]; returns [timestamp, ic, n].
    min_assets: cross-section size floor (None -> module MIN_ASSETS).
    """
    if df.empty:
        return pd.DataFrame(columns=['timestamp', 'ic', 'n'])
    min_assets = MIN_ASSETS if min_assets is None else int(min_assets)

    d = pl.DataFrame({
        # via datetime64[ns]: pandas 3 stores datetimes as [us]/[s], and a
        # bare astype(int64) would be misread by the pl.Datetime('ns') cast
        'timestamp': df['timestamp'].to_numpy(dtype='datetime64[ns]').astype('int64'),
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
        .filter(pl.col('n0') >= min_assets)
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
        'timestamp': df['timestamp'].to_numpy(dtype='datetime64[ns]').astype('int64'),
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
                          signal_v: np.ndarray, target_v: np.ndarray,
                          min_assets: Optional[int] = None,
                          cost_bps: Optional[float] = None) -> pd.DataFrame:
    if len(ts_v) == 0:
        return _empty_backtest_df()
    min_assets = MIN_ASSETS if min_assets is None else int(min_assets)
    cost_bps = COST_BPS if cost_bps is None else float(cost_bps)

    stamps, t_codes = np.unique(ts_v, return_inverse=True)
    counts = np.bincount(t_codes, minlength=len(stamps))
    keep_ts = counts >= min_assets
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
    out['net_return'] = out['gross_return'] - out['turnover'] * cost_bps / 10000.0
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
                            stride_stamps: Optional[pd.DatetimeIndex],
                            min_assets: Optional[int] = None,
                            cost_bps: Optional[float] = None) -> pd.DataFrame:
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

    return _backtest_from_arrays(ts[valid], symbols[valid], signal[valid],
                                 target[valid], min_assets=min_assets,
                                 cost_bps=cost_bps)


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

    _set_targets(_build_targets())


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

    # Per-lag smoothing: each lag is scored on the signal smoothed at ITS OWN
    # effective halflife (effective_halflife_for). One smoothed variant is
    # computed per UNIQUE halflife on the full-resolution raw frame, then all
    # variants are z-scored per screening stamp and merged with the targets in
    # a single pass.
    raw = _raw_signal_frame(signal_name, _REGISTRY, features)
    hl_by_lag = {lag: effective_halflife_for(info, lag) for lag in LAG_GRID}
    sig_col_by_lag = {lag: f'sig_h{hl_by_lag[lag]:g}' for lag in LAG_GRID}
    grouped = raw.groupby('symbol', sort=False)['signal_raw']
    for hl in sorted(set(hl_by_lag.values())):
        col = f'sig_h{hl:g}'
        if hl > 0:
            raw[col] = grouped.transform(
                lambda x, h=hl: x.ewm(halflife=h, min_periods=1).mean()
            ).astype(np.float32)
        else:
            raw[col] = raw['signal_raw'].astype(np.float32)
    panel = raw.drop(columns=['signal_raw'])
    del raw

    # Screening grid only; cross-sectional z-score each smoothed variant
    panel = panel[panel['timestamp'].isin(_SCREEN_STAMPS)].copy()
    sig_cols = sorted(set(sig_col_by_lag.values()))
    by_ts = panel.groupby('timestamp')
    for col in sig_cols:
        g = by_ts[col]
        panel[col] = ((panel[col] - g.transform('mean'))
                      / (g.transform('std') + 1e-10)).clip(-3, 3)
    panel = panel.dropna(subset=sig_cols, how='all')

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

        scol = sig_col_by_lag[lag]   # this lag's smoothing-matched variant
        metric_cols = ['timestamp', 'symbol', scol, 'is_liquid',
                       'liquidity_bucket', tcol]
        if raw_tcol in h_panel.columns:
            metric_cols.append(raw_tcol)
        h_metric = (h_panel[[c for c in metric_cols if c in h_panel.columns]]
                    .rename(columns={scol: 'signal'}))
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


# =============================================================================
# In-memory scoring for the walk-forward
# =============================================================================

def score_registry(registry: Optional[Dict] = None) -> pd.DataFrame:
    """Score every registry signal and return the daily-stats frame the
    walk-forward selector consumes. In memory, nothing persisted: the
    promoted disc_* set is small, so serial per-column-group evaluation is
    fine (one feature load per distinct column set)."""
    global _REGISTRY
    _init_state()
    if registry is not None:
        _REGISTRY = registry          # evaluate_signal reads the module state
    registry = _REGISTRY

    groups: Dict[tuple, list] = {}
    for name in sorted(registry):
        key = tuple(signal_feature_columns(registry[name]['signal_def']))
        groups.setdefault(key, []).append(name)

    frames = []
    for columns, names in sorted(groups.items()):
        features = _load_features(list(columns))
        for name in names:
            try:
                daily, summaries = evaluate_signal(name, features)
            except Exception as e:
                logging.warning(f"scoring failed for {name}: {e}")
                continue
            if not daily.empty:
                frames.append(daily)
            for row in summaries:
                if row.get('error'):
                    logging.warning(f"{name} [{row.get('horizon')}]: "
                                    f"{row['error']}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
