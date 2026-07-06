"""
DATA: the roll's data foundation and the compressed diagnostics.

1. Panel construction - features + residual/raw returns + forward residual
   targets (fwd_Lb[t] = sum of residuals over t+1..t+L, the repo's
   rolling(L).sum().shift(-L) convention) + liquid-half flag + daily factor
   betas, restricted to the point-in-time universe. Built ONCE per run; every
   candidate evaluation afterwards is column algebra on this cached panel.
2. Rolls and window slicing - train/select/OOS windows and the purge/embargo
   discipline (the last max-target-lag + embargo bars of TRAIN and SELECT are
   dropped so no forward target leaks across a boundary). Window slicing lives
   in exactly one place.
3. Diagnostics builder - per-feature stats on the TRAIN window, compressed
   for the proposer. This compressed view is the ONLY thing the LLM ever
   sees - never the data itself.

Everything except the loaders is pure DataFrame-in/DataFrame-out and runs on
synthetic panels in tests without a database.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from config import get, get_frequency_config, BASE_FREQUENCY
from research.signals.evaluate import _nw_tstat, rank_ic_per_timestamp

TARGET_PREFIX = 'fwd_'


def target_col(lag_bars: int) -> str:
    return f'{TARGET_PREFIX}{lag_bars}b'


def raw_target_col(lag_bars: int) -> str:
    return f'{TARGET_PREFIX}raw_{lag_bars}b'


# =============================================================================
# Rolls and window slicing
# =============================================================================

@dataclass(frozen=True)
class Roll:
    roll_id: int
    train_start: pd.Timestamp
    select_start: pd.Timestamp      # = train_end
    oos_start: pd.Timestamp         # = select_end
    oos_end: pd.Timestamp


def make_rolls(cfg: Optional[dict] = None) -> List[Roll]:
    """Train/select/OOS windows advancing by roll_step_months."""
    cfg = cfg or get('discovery', {})
    start = pd.Timestamp(cfg['start_date'])
    end = pd.Timestamp(cfg['end_date'])
    train = pd.DateOffset(months=int(cfg['train_months']))
    select = pd.DateOffset(months=int(cfg['select_months']))
    oos = pd.DateOffset(months=int(cfg['oos_months']))
    step = pd.DateOffset(months=int(cfg['roll_step_months']))

    rolls = []
    t0 = start
    rid = 0
    while True:
        select_start = t0 + train
        oos_start = select_start + select
        oos_end = oos_start + oos
        if oos_end > end:
            break
        rolls.append(Roll(rid, t0, select_start, oos_start, oos_end))
        t0 = t0 + step
        rid += 1
    return rolls


def purge_bars(cfg: Optional[dict] = None) -> int:
    """Bars dropped at the end of TRAIN and SELECT: longest target + embargo."""
    cfg = cfg or get('discovery', {})
    return int(max(cfg['horizon_lags_bars'])) + int(cfg['embargo_bars'])


def resolve_search_lags(cfg: Optional[dict] = None) -> List[int]:
    """Lags the search scores candidates at (discovery.search_lags_bars).

    'all' (default) -> the full horizon_lags_bars grid: each candidate is
    evaluated at every lag on TRAIN and pinned to its strongest one, so one
    run finds signals wherever on the speed spectrum they live. An explicit
    list restricts the search (must be a subset of horizon_lags_bars - the
    panel only builds targets for those)."""
    cfg = cfg or get('discovery', {})
    grid = [int(x) for x in cfg['horizon_lags_bars']]
    lags = cfg.get('search_lags_bars', 'all')
    if lags in ('all', None):
        return grid
    lags = [int(x) for x in lags]
    bad = [x for x in lags if x not in grid]
    if bad:
        raise ValueError(f"discovery.search_lags_bars {bad} not in "
                         f"horizon_lags_bars {grid}")
    return lags


def slice_window(panel: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp,
                 purge_end_bars: int = 0) -> pd.DataFrame:
    """Rows with start <= timestamp < end - purge_end_bars * bar."""
    bar = pd.Timedelta(get_frequency_config(BASE_FREQUENCY)['nanos'], unit='ns')
    eff_end = pd.Timestamp(end) - purge_end_bars * bar
    ts = panel['timestamp']
    return panel[(ts >= pd.Timestamp(start)) & (ts < eff_end)]


def strided_stamps(timestamps: pd.Series, stride: int) -> np.ndarray:
    """Every stride-th stamp of the window's GLOBAL sorted schedule, so every
    candidate in a roll is scored on identical calendar stamps. Stride = the
    target lag keeps forward windows non-overlapping (overlap autocorrelates
    the IC series and inflates t-stats ~sqrt(stride))."""
    stamps = np.sort(pd.unique(timestamps))
    return stamps[::max(1, int(stride))]


# =============================================================================
# Feature-column resolution (the bounded input space)
# =============================================================================

def resolve_family_columns(available: Sequence[str],
                           cfg: Optional[dict] = None) -> dict:
    """{family: [columns]} by matching config prefix patterns, capped per
    family. Deterministic: patterns and available columns are both ordered."""
    cfg = cfg or get('discovery', {})
    families = cfg['families']
    cap = int(cfg['max_features_per_family'])
    reserved = {'timestamp', 'symbol'}
    out = {}
    for family, patterns in families.items():
        cols = [c for c in available
                if c not in reserved and any(c.startswith(p) for p in patterns)]
        out[family] = sorted(cols)[:cap]
    return out


def all_family_columns(family_columns: dict) -> List[str]:
    return sorted({c for cols in family_columns.values() for c in cols})


# =============================================================================
# Panel construction (the only db-touching code in the discovery engine)
# =============================================================================

def build_panel(feature_cols: Sequence[str],
                cfg: Optional[dict] = None) -> pd.DataFrame:
    """One long panel [timestamp, symbol, features..., residual_return,
    raw_return, fwd_{L}b..., is_liquid, beta_*], universe-filtered, sorted by
    (symbol, timestamp)."""
    from dbutil import load_data
    from research.signals.evaluate import (load_universe_membership,
                                           universe_member_mask)

    cfg = cfg or get('discovery', {})
    lags = [int(x) for x in cfg['horizon_lags_bars']]
    start = pd.Timestamp(cfg['start_date'])
    end = pd.Timestamp(cfg['end_date'])

    res = load_data('residual_returns',
                    columns=['timestamp', 'symbol', 'residual_return',
                             'raw_return'])
    if res.empty:
        raise RuntimeError("residual_returns is empty - run "
                           "risk_model/residual_returns.py")
    res['timestamp'] = pd.to_datetime(res['timestamp'])
    res = res[(res['timestamp'] >= start) & (res['timestamp'] < end)]
    res = res.sort_values(['symbol', 'timestamp']).reset_index(drop=True)

    panel = attach_targets(res, lags)

    features = load_data('features',
                         columns=['timestamp', 'symbol'] + list(feature_cols))
    features['timestamp'] = pd.to_datetime(features['timestamp'])
    panel = panel.merge(features, on=['timestamp', 'symbol'], how='left')

    membership = load_universe_membership()
    if membership is not None:
        panel = panel[universe_member_mask(panel, membership)]
    else:
        logging.warning("universe membership unavailable - using full panel")

    panel = attach_liquidity(panel, cfg)
    panel = attach_betas(panel)
    return panel.sort_values(['symbol', 'timestamp']).reset_index(drop=True)


def attach_targets(res: pd.DataFrame, lags: Sequence[int]) -> pd.DataFrame:
    """Add forward residual targets for every lag to a [timestamp, symbol,
    residual_return, ...] frame, per symbol:

        fwd_Lb[t] = sum of residual_return over bars t+1..t+L

    (rolling(L).sum().shift(-L); a NaN gap poisons overlapping targets).
    Requires (symbol, timestamp)-sorted input.
    """
    out = res.copy()
    g_res = out.groupby('symbol')['residual_return']
    for lag in lags:
        out[target_col(lag)] = g_res.transform(
            lambda x, L=lag: x.rolling(L, min_periods=L).sum().shift(-L))
    return out


def attach_liquidity(panel: pd.DataFrame,
                     cfg: Optional[dict] = None) -> pd.DataFrame:
    """is_liquid = top half by trailing dollar volume at each stamp."""
    from dbutil import load_data
    cfg = cfg or get('discovery', {})
    window = int(cfg['liquidity_window_bars'])
    try:
        px = load_data('prices', columns=['timestamp', 'symbol',
                                          'quote_asset_volume'])
        px['timestamp'] = pd.to_datetime(px['timestamp'])
        qv = px.pivot_table(index='timestamp', columns='symbol',
                            values='quote_asset_volume',
                            aggfunc='first').sort_index()
        trail = qv.rolling(window, min_periods=window // 4).mean()
        liq = (trail.rank(axis=1, pct=True) >= 0.5).stack(future_stack=True)
        liq = liq.rename('is_liquid').reset_index()
        liq.columns = ['timestamp', 'symbol', 'is_liquid']
        panel = panel.merge(liq, on=['timestamp', 'symbol'], how='left')
        panel['is_liquid'] = panel['is_liquid'].fillna(False).astype(bool)
    except Exception as e:
        logging.warning(f"liquidity flag unavailable: {e}")
        panel = panel.copy()
        panel['is_liquid'] = True
    return panel


def attach_betas(panel: pd.DataFrame) -> pd.DataFrame:
    """Merge daily factor betas (factor_loadings) onto the bar panel as
    beta_<factor> columns. Loadings for date D are estimated from data
    strictly before D (residual_returns.py convention), so applying them to
    day D's bars is causal."""
    from dbutil import load_data, get_table_columns
    cols = get_table_columns('factor_loadings')
    beta_cols = [c for c in (cols or []) if c.startswith('beta_')]
    if not beta_cols:
        logging.warning("factor_loadings unavailable - dollar-only neutrality")
        return panel
    loadings = load_data('factor_loadings',
                         columns=['date', 'symbol'] + beta_cols)
    loadings['date'] = pd.to_datetime(loadings['date'])
    panel = panel.copy()
    panel['date'] = panel['timestamp'].dt.normalize()
    panel = panel.merge(loadings, on=['date', 'symbol'], how='left')
    return panel.drop(columns=['date'])


def beta_columns(panel: pd.DataFrame) -> List[str]:
    return [c for c in panel.columns if c.startswith('beta_')]


# =============================================================================
# Diagnostics builder (the proposer's entire view of the world)
# =============================================================================

def _feature_ic(df: pd.DataFrame, col: str, tcol: str,
                min_assets: int) -> dict:
    sub = df[['timestamp', col, tcol]].rename(columns={col: 'signal'})
    ics = rank_ic_per_timestamp(sub, tcol, min_assets=min_assets)
    if ics.empty:
        return {'ic': np.nan, 'tstat': 0.0, 'n': 0}
    daily = ics.set_index('timestamp')['ic'].groupby(
        lambda ts: ts.normalize()).mean()
    return {'ic': float(ics['ic'].mean()),
            'tstat': float(_nw_tstat(daily.values, 'auto')),
            'n': int(len(ics))}


def _binned_curve(df: pd.DataFrame, col: str, tcol: str, n_bins: int) -> list:
    """Mean forward target by cross-sectional decile of the feature - the
    nonlinearity detector (U-shapes, thresholds, sign flips)."""
    x = df[[col, tcol, 'timestamp']].dropna()
    if len(x) < n_bins * 10:
        return []
    pct = x.groupby('timestamp')[col].rank(pct=True)
    bins = np.minimum((pct * n_bins).astype(int), n_bins - 1)
    curve = x.groupby(bins)[tcol].mean()
    return [round(float(v), 6) for v in curve.reindex(range(n_bins)).values]


def _thirds_stability(df: pd.DataFrame, col: str, tcol: str,
                      min_assets: int) -> int:
    """How many time-thirds share the pooled IC sign (0-3)."""
    stamps = np.sort(df['timestamp'].unique())
    if len(stamps) < 9:
        return 0
    parts = np.array_split(stamps, 3)
    signs = []
    for part in parts:
        sub = df[df['timestamp'].isin(part)]
        signs.append(np.sign(_feature_ic(sub, col, tcol, min_assets)['ic']))
    pooled = np.sign(_feature_ic(df, col, tcol, min_assets)['ic'])
    return int(sum(1 for s in signs if s == pooled and s != 0))


def build_diagnostics(train_panel: pd.DataFrame,
                      family_columns: Dict[str, list],
                      tcol: str, lag_bars: int,
                      cfg: Optional[dict] = None) -> dict:
    """Compressed per-feature diagnostics on the TRAIN window."""
    cfg = cfg or get('discovery', {})
    diag_cfg = cfg['diagnostics']
    min_assets = int(cfg['min_assets_per_timestamp'])
    n_bins = int(diag_cfg['n_bins'])
    top_k = int(diag_cfg['top_per_family'])
    regime_cols = [c for c in diag_cfg['regime_columns']
                   if c in train_panel.columns]

    stamps = strided_stamps(train_panel['timestamp'], lag_bars)
    df = train_panel[train_panel['timestamp'].isin(stamps)]

    features: Dict[str, dict] = {}
    for family, cols in family_columns.items():
        for col in cols:
            if col not in df.columns or col in features:
                continue
            base = _feature_ic(df, col, tcol, min_assets)
            entry = {
                'family': family,
                'ic': round(base['ic'], 5) if np.isfinite(base['ic']) else None,
                'ic_tstat': round(base['tstat'], 2),
                'binned_fwd': _binned_curve(df, col, tcol, n_bins),
                'stable_thirds': _thirds_stability(df, col, tcol, min_assets),
                'regime_ic': {},
            }
            for rc in regime_cols:
                if rc == col:
                    continue
                med = df.groupby('timestamp')[rc].transform('median')
                hi = _feature_ic(df[df[rc] >= med], col, tcol,
                                 max(3, min_assets // 2))
                lo = _feature_ic(df[df[rc] < med], col, tcol,
                                 max(3, min_assets // 2))
                entry['regime_ic'][rc] = {
                    'high': round(hi['ic'], 5) if np.isfinite(hi['ic']) else None,
                    'low': round(lo['ic'], 5) if np.isfinite(lo['ic']) else None,
                }
            features[col] = entry

    top = {}
    for family, cols in family_columns.items():
        scored = [(c, abs(features[c]['ic_tstat'])) for c in cols
                  if c in features]
        scored.sort(key=lambda x: -x[1])
        top[family] = [c for c, _ in scored[:top_k]]

    return {'target': tcol, 'lag_bars': int(lag_bars),
            'features': features, 'top_by_family': top}
