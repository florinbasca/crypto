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
from research.lib.signal_eval import _nw_tstat, rank_ic_per_timestamp

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
    """Lags the search scores candidates at: the full horizon_lags_bars grid.
    Each candidate is evaluated at every lag (train AND select) - the
    per-lag profile is its alpha term structure; nothing is pinned."""
    cfg = cfg or get('discovery', {})
    return [int(x) for x in cfg['horizon_lags_bars']]


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
    from research.lib.signal_eval import (load_universe_membership,
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
# Feature descriptions (the proposer's data dictionary)
# =============================================================================
# What each column MEASURES, so the LLM can reason about the mechanism instead
# of guessing from the abbreviation. Point-in-time honest: describe the INPUT,
# never claim it predicts. Per-column where confirmed against
# risk_model/features.py; a prefix fallback covers the rest at family level.
# Keep in sync with the feature calculators.

FEATURE_DESCRIPTIONS = {
    # cross-sectional (per-timestamp rank/relative vs the universe)
    'cs_rel_volume': "own bar $volume vs its trailing average, relative to the universe median (volume surprise)",
    'cs_ret_rank_1h': "cross-sectional percentile rank of the trailing 1h return",
    'cs_ret_rank_1d': "cross-sectional percentile rank of the trailing 1d return",
    'cs_dispersion_1h': "universe-wide return dispersion over 1h (same for all coins; a regime conditioner)",
    'cs_breadth_sma': "fraction of the universe above its SMA (breadth; same for all coins)",
    'cs_funding_z': "coin's funding rate z-scored across the universe (relative crowding)",
    'cs_mcap_rank': "market-cap percentile rank across the universe (lagged 1 day)",
    'cs_cluster_rel_z': "residual cumulative return z-scored within the coin's trailing correlation cluster",
    # funding (perpetual funding rate)
    'fr_annualized': "perpetual funding rate, annualized (positive = longs pay shorts = long crowding)",
    'fr_rate': "current funding rate",
    'fr_rate_zscore': "funding rate vs its own recent history (z-score)",
    'fr_rate_change': "change in the funding rate",
    'fr_rate_ma_7d': "7-day moving-average funding rate",
    'fr_cumulative_24h': "funding paid/received over the last 24h (realized carry)",
    'fr_mean': "trailing mean funding rate",
    'fr_std': "trailing volatility of the funding rate",
    'fr_extreme_high': "flag: funding unusually high (crowded longs)",
    'fr_extreme_low': "flag: funding unusually low/negative (crowded shorts)",
    'fr_flip_age': "bars since funding last flipped sign",
    'fr_oi_crowding': "funding × open interest (size of the crowded carry position)",
    # open interest / positioning
    'oi_value': "open interest (notional outstanding)",
    'oi_change': "change in open interest",
    'oi_change_pct': "percent change in open interest",
    'oi_change_z': "open-interest change z-scored",
    'oi_change_zscore': "open-interest change z-scored",
    'oi_value_zscore': "open interest vs its own history (z-score)",
    'oi_relative': "open interest relative to the universe",
    'oi_price_divergence': "open interest rising while price falls (or vice-versa) — positioning/price divergence",
    'oi_flush': "flag: sharp open-interest drop (liquidation flush)",
    'oi_mean': "trailing mean open interest",
    'oi_std': "trailing volatility of open interest",
    # trader positioning (exchange long/short & taker flow)
    'pos_retail_ls': "retail long/short account ratio",
    'pos_retail_ls_zscore': "retail long/short ratio z-scored",
    'pos_toptrader_ls': "top-trader long/short position ratio",
    'pos_toptrader_ls_zscore': "top-trader long/short ratio z-scored",
    'pos_retail_vs_smart': "retail vs top-trader positioning gap (dumb-vs-smart money)",
    'pos_taker_ratio': "taker buy/sell volume ratio (aggressive flow direction)",
    'pos_taker_zscore': "taker buy/sell ratio z-scored",
    # market relationship / lead-lag to the market factor
    'mk_corr_market_1d': "trailing 1d correlation of the coin's returns with the market factor",
    'mk_beta_drift': "change in the coin's market beta",
    'mk_lag_response_gap': "how much the coin still has to catch up to a market move it lagged",
    'mk_market_ret_1h': "market factor return over 1h (same for all coins; regime)",
    'mk_market_vol_1d': "market factor volatility over 1d (same for all coins; regime)",
    'mk_market_move_z': "size of the current market move in std units (same for all coins)",
    'mk_lag_corr_short': "short-window lagged correlation to the market (coin leads/lags market)",
    'mk_lag_corr_long': "long-window lagged correlation to the market",
    # factor loadings (rolling betas + drift)
    'fl_beta_market': "rolling beta to the market factor",
    'fl_beta_size': "rolling beta to the size factor",
    'fl_beta_market_change_short': "short-horizon change in market beta",
    'fl_beta_market_change_long': "long-horizon change in market beta",
    'fl_beta_size_change_short': "short-horizon change in size beta",
    'fl_beta_size_change_long': "long-horizon change in size beta",
    'fl_r2_total': "how well the factor model explains this coin lately (R²; high = little idiosyncratic room)",
    # lead-lag vs the market leader (BTC)
    'll_leader_beta': "beta of the coin to the market leader (BTC)",
    'll_lag_corr': "correlation of the coin's return to the leader's lagged return (does BTC lead it?)",
    # intrabar microstructure
    'ib_rv_1h': "realized volatility over 1h from 1-min bars",
    'ib_rv_cc_ratio': "intrabar realized vol vs close-to-close vol (intrabar noise)",
    'ib_max_move_1m': "largest 1-min move in the bar, normalized",
    'ib_autocorr_1m': "1-min return autocorrelation within the bar (trending vs choppy)",
    'ib_vwap_dev': "close vs VWAP (where price settled within the bar)",
    'ib_zero_vol_share': "share of zero-volume minutes (illiquidity)",
    'ib_volume_herf_1h': "volume concentration across the hour (Herfindahl; bursty vs even)",
    # macro-beta (per-coin sensitivity to macro drivers)
    'mb_beta_rates': "coin's sensitivity (beta) to 2y rate changes",
    'mb_beta_dollar': "sensitivity to the US dollar index",
    'mb_beta_vix': "sensitivity to VIX (risk-off beta)",
    'mb_beta_stables': "sensitivity to stablecoin-supply flows",
    'mb_beta_dominance': "sensitivity to BTC dominance",
    'mb_event_vol_ratio': "how much the coin's volatility rises around macro events",
    'mb_event_volume_ratio': "how much the coin's volume rises on event days",
    'mb_event_drift': "the coin's typical residual drift in the 24h after an event",
}

# Family-level fallback by prefix, for columns not listed above.
_DESC_PREFIX = [
    ('res_', "residual-return dynamics: autocorrelation, volatility, and mean-reversion state of the coin's factor residual"),
    ('ou_', "Ornstein-Uhlenbeck mean-reversion fit on the residual price level (reversion speed, half-life, deviation)"),
    ('vr_', "volatility-regime feature (short vs long realized-vol ratios, breakouts, persistence)"),
    ('rb_', "range-based volatility estimator from OHLC (Parkinson / Garman-Klass / Rogers-Satchell)"),
    ('ib_', "intrabar microstructure feature from 1-min bars"),
    ('lq_', "liquidity/illiquidity feature (Amihud, price impact)"),
    ('vl_', "liquidity/volume feature"),
    ('ms_', "market-microstructure feature"),
    ('fr_', "perpetual funding-rate feature"),
    ('oi_', "open-interest / positioning feature"),
    ('pos_', "trader-positioning feature (retail vs top-trader, taker flow)"),
    ('cs_', "cross-sectional rank/relative feature vs the universe"),
    ('fl_', "rolling factor loading (beta) and its drift"),
    ('mk_', "relationship to the market factor (beta, lead-lag, market state)"),
    ('cap_', "market-cap / turnover feature"),
    ('sn_', "seasonality: hour-of-day or day-of-week residual pattern"),
    ('ll_', "lead-lag vs the market leader (BTC)"),
    ('ev_', "event timing (FOMC/CPI/NFP proximity) — identical for all coins; use as a gate"),
    ('mx_', "macro state (rates, dollar, VIX, liquidity, sentiment) — identical for all coins; use as a gate"),
    ('mb_', "per-coin sensitivity (beta) to a macro driver or event"),
]


def describe_column(col: str) -> str:
    """One-line description of what a feature column MEASURES (data dictionary
    for the LLM). Exact match first, then prefix fallback, then empty."""
    if col in FEATURE_DESCRIPTIONS:
        return FEATURE_DESCRIPTIONS[col]
    for prefix, desc in _DESC_PREFIX:
        if col.startswith(prefix):
            return desc
    return ""


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
                'desc': describe_column(col),
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
