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
from research.lib.signal_eval import _nw_tstat

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
    # order flow (directional aggressive-flow / informed-trading)
    'of_kyle_lambda': "price move per unit of net aggressive (taker) flow — Kyle's lambda, an impact/illiquidity measure",
    'of_signed_flow_return_corr': "does aggressive flow DRIVE price (high) or get ABSORBED by resting liquidity (low/negative)",
    'of_toxicity': "how one-sided the aggressive flow is (VPIN-style) — a proxy for informed trading / adverse selection",
    'of_flow_persistence': "autocorrelation of the taker imbalance — is aggressive flow trending or choppy",
    'of_flow_imbalance_1d': "net signed taker flow over ~1d, normalized by volume (persistent directional pressure)",
    'of_flow_accel': "recent taker imbalance minus its slow baseline (flow building or fading)",
    'ms_ofi_normalized': "order-flow imbalance (taker buy minus sell) over recent bars, normalized by volume",
    'ms_ofi_cumsum_short': "cumulative signed taker flow over a short window",
    'ms_ofi_cumsum_long': "cumulative signed taker flow over a long window",
    'ms_buy_ratio': "taker buy volume as a fraction of total volume (0.5 = balanced; >0.5 = buy-pressure)",
    'ms_buy_pressure_momentum': "change in taker buy-ratio (rising or falling buy pressure)",
    'ms_up_down_volume_ratio': "volume on up-bars vs down-bars (directional participation)",
    'ms_vol_return_correlation': "correlation of volume with |return| (is volume informative or noise)",
    'ms_trade_intensity': "number of trades vs its trailing average (activity spike)",
    'vl_taker_buy_ratio': "taker buy volume fraction (aggressive-buy share)",
    'vl_signed_volume_ratio': "signed (buy-minus-sell) volume as a fraction of total",
    # efficiency / diffusion (clean trend vs noisy overshoot)
    'ef_efficiency_ratio': "Kaufman efficiency: net move / summed absolute moves (high = clean directional trend, low = choppy/noisy)",
    'ef_variance_ratio': "Lo-MacKinlay variance ratio (>1 trending, <1 mean-reverting)",
    'ef_variance_ratio_long': "variance ratio over a longer window (trend vs reversion regime)",
    'ef_reversal_tendency': "how strongly recent moves have been reversing",
    'ef_info_processing_speed': "how fast the coin incorporates information (diffusion speed)",
    'ef_autocorr_lag1': "return autocorrelation at lag 1",
    'ef_autocorr_lag2': "return autocorrelation at lag 2",
    'ef_reversion_momentum_short': "short-window balance of mean-reversion vs momentum",
    'ef_reversion_momentum_medium': "medium-window balance of mean-reversion vs momentum",
    'ef_reversion_momentum_long': "long-window balance of mean-reversion vs momentum",
    # distribution shape
    'st_skew': "skewness of recent returns (crash-prone vs squeeze-prone)",
    'st_kurtosis': "kurtosis / fat-tailedness of recent returns (jump risk)",
    # tokenomics
    'cap_supply_inflation': "token supply inflation / dilution rate (unlock / supply-pressure proxy)",
    'cap_log_mcap': "log market cap (the size characteristic; the book neutralizes size, so best used as a small-vs-large gate)",
    # calendar (cross-sectionally constant — gate only)
    'tm_funding_window': "proximity to a perp funding settlement (identical for all coins at a bar; use ONLY as a gate)",
    # momentum quality
    'mq_momentum_strength': "strength of the current price momentum",
    'mq_momentum_acceleration': "is momentum accelerating or fading",
    'mq_vol_adjusted_momentum': "momentum scaled by volatility (risk-adjusted trend)",
    'mq_trend_consistency_short': "how consistent the short-window trend direction is",
    'mq_trend_consistency_medium': "how consistent the medium-window trend direction is",
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
    ('of_', "order-flow feature: directional aggressive (taker) flow, impact, or informed-trading proxy"),
    ('lq_', "liquidity/illiquidity feature (Amihud, price impact)"),
    ('vl_', "volume / taker-flow feature"),
    ('ms_', "market-microstructure feature (volume, trade size, order-flow imbalance)"),
    ('fr_', "perpetual funding-rate feature"),
    ('oi_', "open-interest / positioning feature"),
    ('pos_', "trader-positioning feature (retail vs top-trader, taker flow)"),
    ('cs_', "cross-sectional rank/relative feature vs the universe"),
    ('fl_', "rolling factor loading (beta) and its drift"),
    ('mk_', "relationship to the market factor (beta, lead-lag, market state)"),
    ('cap_', "market-cap / turnover feature"),
    ('ef_', "return-efficiency / diffusion feature (clean trend vs noisy overshoot)"),
    ('mq_', "momentum-quality feature (strength, acceleration, consistency)"),
    ('ma_', "price vs moving average (trend position)"),
    ('rs_', "RSI momentum oscillator (raw price)"),
    ('bb_', "Bollinger-band position/width (raw price)"),
    ('mc_', "MACD trend indicator (raw price)"),
    ('dm_', "ADX / DMI directional trend-strength (raw price)"),
    ('at_', "ATR volatility (raw price)"),
    ('ch_', "Chandelier trailing-stop distance (raw price)"),
    ('st_', "return distribution shape (skew / kurtosis)"),
    ('tm_', "calendar/time feature — identical for all coins; use as a gate"),
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

def _decile_nonlinearity(binned) -> float:
    """Spread of a feature's decile forward-return curve. Large for U-shapes,
    thresholds and sign-flips even when the monotonic alpha is ~0 - so these
    features are not hidden from the LLM by a t-stat-only ranking."""
    vals = [v for v in (binned or []) if v is not None and np.isfinite(v)]
    return float(max(vals) - min(vals)) if len(vals) >= 3 else 0.0


def _regime_spread(regime_ic) -> float:
    """Largest high-vs-low regime alpha gap: a feature that works only in
    one regime scores high here even with weak pooled alpha."""
    gaps = [abs(d['high'] - d['low']) for d in (regime_ic or {}).values()
            if d.get('high') is not None and d.get('low') is not None]
    return float(max(gaps)) if gaps else 0.0


def _rank01(values: Dict[str, float]) -> Dict[str, float]:
    """Percentile rank in [0, 1] within the group (ties averaged), so the
    blend combines components measured on different scales fairly."""
    if not values:
        return {}
    return pd.Series(values).rank(pct=True).to_dict()


def book_returns(df: pd.DataFrame, tcol: str, min_assets: int) -> pd.Series:
    """Per-stamp return of the gross-1 dollar-neutral book built from the
    signal cross-section: v_t = sum_i w_i * target_i with w = demeaned signal
    scaled to gross 1. Return units per bet."""
    d = df[['timestamp', 'signal', tcol]].dropna()
    if d.empty:
        return pd.Series(dtype=float)
    g = d.groupby('timestamp')
    n = g['signal'].transform('size')
    d = d[n >= min_assets]
    if d.empty:
        return pd.Series(dtype=float)
    g = d.groupby('timestamp')
    w = d['signal'] - g['signal'].transform('mean')
    gross = w.abs().groupby(d['timestamp']).transform('sum')
    w = w / gross.replace(0, np.nan)
    return (w * d[tcol]).groupby(d['timestamp']).sum(min_count=1).dropna()


def _feature_alpha(df: pd.DataFrame, col: str, tcol: str,
                   min_assets: int) -> dict:
    """Per-bet return of the raw column, measured EXACTLY like a candidate:
    per-stamp z-score + clip +-3 (the compiler pipeline for the identity
    expression ["col", x]), then demean/gross-1/dot with the target. Same
    currency as the reward, so the proposer is pointed at what scores."""
    sub = df[['timestamp', col, tcol]].rename(columns={col: 'signal'}).dropna()
    if sub.empty:
        return {'alpha': np.nan, 'tstat': 0.0, 'n': 0}
    g = sub.groupby('timestamp')['signal']
    sub['signal'] = ((sub['signal'] - g.transform('mean'))
                     / (g.transform('std') + 1e-10)).clip(-3, 3)
    bets = book_returns(sub, tcol, min_assets)
    if bets.empty:
        return {'alpha': np.nan, 'tstat': 0.0, 'n': 0}
    daily = bets.groupby(lambda ts: ts.normalize()).mean()
    return {'alpha': float(bets.mean()),
            'tstat': float(_nw_tstat(daily.values, 'auto')),
            'n': int(len(bets))}


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
    """How many time-thirds share the pooled alpha sign (0-3)."""
    stamps = np.sort(df['timestamp'].unique())
    if len(stamps) < 9:
        return 0
    parts = np.array_split(stamps, 3)
    signs = []
    for part in parts:
        sub = df[df['timestamp'].isin(part)]
        signs.append(np.sign(
            _feature_alpha(sub, col, tcol, min_assets)['alpha']))
    pooled = np.sign(_feature_alpha(df, col, tcol, min_assets)['alpha'])
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
            base = _feature_alpha(df, col, tcol, min_assets)
            entry = {
                'family': family,
                'desc': describe_column(col),
                'alpha_per_bet': (round(base['alpha'], 6)
                                  if np.isfinite(base['alpha']) else None),
                'alpha_tstat': round(base['tstat'], 2),
                'binned_fwd': _binned_curve(df, col, tcol, n_bins),
                'stable_thirds': _thirds_stability(df, col, tcol, min_assets),
                'regime_alpha': {},
            }
            for rc in regime_cols:
                if rc == col:
                    continue
                med = df.groupby('timestamp')[rc].transform('median')
                hi = _feature_alpha(df[df[rc] >= med], col, tcol,
                                    max(3, min_assets // 2))
                lo = _feature_alpha(df[df[rc] < med], col, tcol,
                                    max(3, min_assets // 2))
                entry['regime_alpha'][rc] = {
                    'high': (round(hi['alpha'], 6)
                             if np.isfinite(hi['alpha']) else None),
                    'low': (round(lo['alpha'], 6)
                            if np.isfinite(lo['alpha']) else None),
                }
            features[col] = entry

    # The LLM only gets full diagnostics for the top few features per family.
    # Rank them by a BLEND (each component rank-normalized within the family):
    # monotonic IC t-stat + decile-curve nonlinearity + regime spread +
    # stability - so U-shaped, threshold-only and regime-only primitives reach
    # the proposer instead of being hidden by a t-stat-only sort. A small
    # random quota keeps some lower-ranked features in rotation.
    blend = diag_cfg.get('top_blend', {'monotonic': 1.0, 'nonlinear': 0.6,
                                       'regime': 0.5, 'stability': 0.3})
    quota = int(diag_cfg.get('top_random_quota', 0))
    rng = np.random.default_rng(int(cfg['search']['seed']))
    top = {}
    for family, cols in family_columns.items():
        fam = [c for c in cols if c in features]
        if not fam:
            top[family] = []
            continue
        comps = {
            'monotonic': {c: abs(features[c]['alpha_tstat']) for c in fam},
            'nonlinear': {c: _decile_nonlinearity(features[c]['binned_fwd'])
                          for c in fam},
            'regime': {c: _regime_spread(features[c]['regime_alpha']) for c in fam},
            'stability': {c: float(features[c]['stable_thirds']) for c in fam},
        }
        score = {c: 0.0 for c in fam}
        for key, w in blend.items():
            for c, r in _rank01(comps.get(key, {})).items():
                score[c] += float(w) * r
        ranked = sorted(fam, key=lambda c: -score[c])
        keep = max(top_k - quota, 0)
        chosen, rest = ranked[:keep], ranked[keep:]
        if quota > 0 and rest:
            idx = rng.choice(len(rest), size=min(quota, len(rest)),
                             replace=False)
            chosen += [rest[int(i)] for i in idx]
        top[family] = chosen[:top_k]

    return {'target': tcol, 'lag_bars': int(lag_bars),
            'features': features, 'top_by_family': top}
