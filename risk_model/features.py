"""
Statistical Arbitrage Features for Residual Prediction (10-minute base panel).

One feature panel at config base_frequency. The prediction targets
(fwd_res_{horizon}) live in residual_returns; this file builds the
information set available at decision time t.

TIMING CONVENTION (unified): every feature may use data through bar t
INCLUSIVE. Bars are bar-end stamped, betas/factors are causal, and the
forward targets cover bars t+1..t+p - so close[t], volume[t], residual[t],
funding settled at t are all legitimately knowable at t with zero overlap
into the target window. (The legacy shift(1) convention cost one full bar of
staleness - material at a 10-minute horizon - and was removed deliberately.
The truncation test in tests/sanity_checks.py enforces causality instead.)

Feature groups:
  ef_  efficiency / mean reversion        vr_  volatility regime
  lq_  liquidity (dollar-volume Amihud)   ms_  microstructure (10min)
  ib_  intra-bar (from raw 1m data)       rb_  range-based vol estimators
  mq_  momentum quality                   st_  statistical moments
  px_  price/momentum                     vl_  volume
  ma_  price-vs-SMA (relative)            ch_  chandelier distances (relative)
  rs_/bb_/mc_/at_/dm_  technical          tm_  time encoding (sin/cos, funding)
  mk_  market context / BTC-style lead-lag
  res_ residual dynamics + spread state   ou_  OU on CUMULATIVE residual
  fl_  factor loadings (market/size)      cap_ market-cap derived
  cs_  cross-sectional (panel phase)      fr_/oi_/pos_  futures-derived
  sn_  per-symbol seasonality profiles    ll_  leader (BTC) lead-lag, slow
  ev_  macro-event timing (constant)      mx_  macro state (constant)
  mb_  per-name macro sensitivities (cross-sectional)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import ta

from dbutil import (load_data, save_data, get_table_symbols, delete_table,
                    parallel_map, ensure_symbol_index)
from config import config as global_config, get_frequency_config

# Memory-bound: each worker peaks ~1.4GB (full-history symbols)
MAX_WORKERS = global_config['compute']['feature_workers']
BASE_FREQUENCY = global_config['base_frequency']
FEATURE_CONFIG = global_config['features']
BARS_PER_DAY = get_frequency_config(BASE_FREQUENCY)['bars_per_day']
ANNUALIZER = np.sqrt(BARS_PER_DAY * 365)

logging.basicConfig(
    level=logging.WARNING,
    format=global_config['logging']['format'],
    datefmt=global_config['logging']['datefmt'],
)


def _nan_series(df: pd.DataFrame, names: List[str], fill=np.nan) -> Dict[str, pd.Series]:
    return {n: pd.Series([fill] * len(df), index=df.index) for n in names}


def _rolling_autocorr(x: pd.Series, window: int, lag: int) -> pd.Series:
    """Vectorized rolling autocorrelation via rolling corr with shifted self."""
    return x.rolling(window, min_periods=max(10, window // 2)).corr(x.shift(lag))


def _variance_ratio(returns: pd.Series, q: int, window: int) -> pd.Series:
    """
    True Lo-MacKinlay variance ratio: Var(q-bar return) / (q * Var(1-bar)).
    < 1 mean reversion, ~ 1 random walk, > 1 trending.
    """
    q_ret = returns.rolling(q, min_periods=q).sum()
    var_q = q_ret.rolling(window, min_periods=window // 2).var()
    var_1 = returns.rolling(window, min_periods=window // 2).var()
    return var_q / (q * var_1 + 1e-18)


# ============================================================================
# Mean Reversion & Efficiency
# ============================================================================

def calculate_efficiency_features(df: pd.DataFrame, config: Dict) -> Dict[str, pd.Series]:
    prefix = 'ef_'
    features = {}

    returns = df['close'].pct_change()
    close = df['close']

    features[f'{prefix}autocorr_lag1'] = _rolling_autocorr(returns, config['autocorr_window'], 1)
    features[f'{prefix}autocorr_lag2'] = _rolling_autocorr(returns, config['autocorr_window'], 2)

    reversal_indicator = (returns * returns.shift(1) < 0).astype(int)
    features[f'{prefix}reversal_tendency'] = reversal_indicator.rolling(window=config['reversal_window']).mean()

    # True Lo-MacKinlay variance ratios (the real mean-reversion diagnostic;
    # the previous implementation compared 1-bar variances over different
    # windows, which is a vol-regime ratio, not a VR)
    q_short, q_long = config['variance_ratio_q'][:2]
    vr_window = config['variance_ratio_window']
    features[f'{prefix}variance_ratio'] = _variance_ratio(returns, q_short, vr_window)
    features[f'{prefix}variance_ratio_long'] = _variance_ratio(returns, q_long, vr_window)

    window_names = ['short', 'medium', 'long']
    for i, window in enumerate(config['mean_reversion_windows'][:3]):
        price_ma = close.rolling(window=window, min_periods=window // 2).mean()
        deviation = (close - price_ma) / price_ma
        name = window_names[i]
        features[f'{prefix}mean_deviation_{name}'] = deviation
        features[f'{prefix}reversion_momentum_{name}'] = deviation - deviation.shift(config['momentum_short'])

    direction_sum = returns.rolling(window=config['autocorr_window']).sum().abs()
    volatility_sum = returns.abs().rolling(window=config['autocorr_window']).sum()
    features[f'{prefix}efficiency_ratio'] = direction_sum / (volatility_sum + 1e-10)

    volume = df['volume']
    features[f'{prefix}info_processing_speed'] = volume.rolling(
        window=config['reversal_window']).corr(returns.abs())

    return features


# ============================================================================
# Volatility Regime
# ============================================================================

def calculate_volatility_regime_features(df: pd.DataFrame, config: Dict) -> Dict[str, pd.Series]:
    prefix = 'vr_'
    features = {}

    returns = df['close'].pct_change()

    vol_short = returns.rolling(window=config['vol_short_window']).std()
    vol_long = returns.rolling(window=config['vol_long_window']).std()
    vol_extended = returns.rolling(window=config['vol_long_window'] * 2).std()

    features[f'{prefix}volatility_regime_short'] = vol_short / (vol_long + 1e-10)
    features[f'{prefix}volatility_regime_long'] = vol_long / (vol_extended + 1e-10)
    features[f'{prefix}vol_breakout_short'] = (vol_short > vol_long * 1.5).astype(int)
    features[f'{prefix}vol_breakout_long'] = (vol_long > vol_extended * 1.5).astype(int)

    # Persistence over windows long enough to be meaningful (the old version
    # used an 18-bar window of a heavily autocorrelated series - noise)
    half = max(2, config['vol_short_window'] // 2)
    features[f'{prefix}vol_persistence_short'] = vol_short.rolling(
        window=config['vol_medium_window']).corr(vol_short.shift(half))
    features[f'{prefix}vol_persistence_long'] = vol_short.rolling(
        window=config['vol_long_window']).corr(vol_short.shift(config['vol_short_window']))

    # Leverage-effect asymmetry: downside vs upside semi-vol. > 1 = down-moves
    # dominate the variance (positioning unwind / forced-selling profile)
    down = returns.where(returns < 0, 0.0)
    up = returns.where(returns > 0, 0.0)
    for label, w in [('short', config['vol_short_window']),
                     ('long', config['vol_long_window'])]:
        dvol = down.rolling(w, min_periods=w // 2).std()
        uvol = up.rolling(w, min_periods=w // 2).std()
        features[f'{prefix}semivol_ratio_{label}'] = dvol / (uvol + 1e-10)

    return features


# ============================================================================
# Liquidity (cross-sectionally comparable units)
# ============================================================================

def calculate_liquidity_features(df: pd.DataFrame, config: Dict) -> Dict[str, pd.Series]:
    prefix = 'lq_'
    features = {}

    # Amihud on DOLLAR volume: |ret| per dollar traded is comparable across
    # assets; the old base-unit version just ranked coin denominations
    dollar_vol = df['quote_asset_volume'] if 'quote_asset_volume' in df.columns else \
        df['volume'] * df['close']
    abs_ret = df['close'].pct_change().abs()
    amihud = abs_ret / (dollar_vol + 1e-10)
    features[f'{prefix}amihud_illiquidity'] = amihud.rolling(
        window=config['liquidity_window'], min_periods=2).mean() * 1e9

    vol_ma = dollar_vol.rolling(window=config['volume_ma_window'], min_periods=2).mean()
    normalized_volume = dollar_vol / (vol_ma + 1e-10)
    features[f'{prefix}price_impact_proxy'] = abs_ret / (normalized_volume + 1e-10)

    # Signed-impact asymmetry: Amihud on down bars vs up bars. > 1 = selling
    # moves price more than buying (thin bid side / unwind pressure)
    ret = df['close'].pct_change()
    w = config['liquidity_window']
    am_down = amihud.where(ret < 0).rolling(w, min_periods=2).mean()
    am_up = amihud.where(ret > 0).rolling(w, min_periods=2).mean()
    features[f'{prefix}amihud_asym'] = am_down / (am_up + 1e-18)

    return features


# ============================================================================
# Range-Based Estimators (Parkinson / Garman-Klass / Rogers-Satchell / CS)
# ============================================================================

def calculate_range_features(df: pd.DataFrame, config: Dict) -> Dict[str, pd.Series]:
    prefix = 'rb_'
    features = {}
    w = config['range_vol_window']

    o, h, l, c = df['open'], df['high'], df['low'], df['close']
    with np.errstate(divide='ignore', invalid='ignore'):
        log_hl = np.log(h / l)
        log_co = np.log(c / o)
        log_ho = np.log(h / o)
        log_lo = np.log(l / o)

    park_var = (log_hl ** 2) / (4 * np.log(2))
    features[f'{prefix}parkinson_vol'] = np.sqrt(
        park_var.rolling(w, min_periods=w // 2).mean()) * ANNUALIZER

    gk_var = 0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_co ** 2
    features[f'{prefix}garman_klass_vol'] = np.sqrt(
        gk_var.clip(lower=0).rolling(w, min_periods=w // 2).mean()) * ANNUALIZER

    rs_var = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
    features[f'{prefix}rogers_satchell_vol'] = np.sqrt(
        rs_var.clip(lower=0).rolling(w, min_periods=w // 2).mean()) * ANNUALIZER

    cc_vol = c.pct_change().rolling(w, min_periods=w // 2).std() * ANNUALIZER
    features[f'{prefix}gk_cc_ratio'] = features[f'{prefix}garman_klass_vol'] / (cc_vol + 1e-10)

    # Corwin-Schultz bid-ask spread estimator from consecutive-bar high/lows
    beta = log_hl ** 2 + (log_hl ** 2).shift(1)
    h2 = pd.concat([h, h.shift(1)], axis=1).max(axis=1)
    l2 = pd.concat([l, l.shift(1)], axis=1).min(axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        gamma = np.log(h2 / l2) ** 2
    denom = 3 - 2 * np.sqrt(2)
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / denom - np.sqrt(gamma / denom)
    spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    features[f'{prefix}cs_spread'] = spread.clip(lower=0).rolling(
        config['cs_spread_window'], min_periods=10).mean()

    return features


# ============================================================================
# Microstructure (10-minute bars)
# ============================================================================

def calculate_microstructure_features(df: pd.DataFrame, config: Dict) -> Dict[str, pd.Series]:
    prefix = 'ms_'
    features = {}

    has_trades = 'number_of_trades' in df.columns
    has_quote_vol = 'quote_asset_volume' in df.columns
    has_taker_quote = 'taker_buy_quote_asset_volume' in df.columns

    if not has_trades and not has_quote_vol:
        return features

    w_long = config['ms_long_window']
    w_short = config['ms_short_window']

    if has_trades:
        trades = df['number_of_trades'].clip(lower=1)
        trades_ma = trades.rolling(window=w_long, min_periods=w_short).mean()
        trades_std = trades.rolling(window=w_long, min_periods=w_short).std()
        features[f'{prefix}trade_intensity'] = trades / (trades_ma + 1e-10)
        features[f'{prefix}trade_intensity_zscore'] = (trades - trades_ma) / (trades_std + 1e-10)
        trades_ma_short = trades.rolling(window=w_short, min_periods=2).mean()
        features[f'{prefix}trade_intensity_momentum'] = trades_ma_short / (trades_ma + 1e-10) - 1

    if has_quote_vol:
        quote_vol = df['quote_asset_volume'].clip(lower=1e-10)
        quote_vol_ma = quote_vol.rolling(window=w_long, min_periods=w_short).mean()
        features[f'{prefix}dollar_volume_ratio'] = quote_vol / (quote_vol_ma + 1e-10)
        quote_vol_ma_short = quote_vol.rolling(window=w_short, min_periods=2).mean()
        features[f'{prefix}dollar_volume_momentum'] = quote_vol_ma_short / (quote_vol_ma + 1e-10) - 1

    if has_trades and has_quote_vol:
        quote_vol = df['quote_asset_volume'].clip(lower=1e-10)
        trades = df['number_of_trades'].clip(lower=1)
        avg_trade_size = quote_vol / trades
        ats_ma = avg_trade_size.rolling(window=w_long, min_periods=w_short).mean()
        ats_std = avg_trade_size.rolling(window=w_long, min_periods=w_short).std()
        features[f'{prefix}avg_trade_size'] = avg_trade_size
        features[f'{prefix}avg_trade_size_zscore'] = (avg_trade_size - ats_ma) / (ats_std + 1e-10)
        features[f'{prefix}large_trades_flag'] = (avg_trade_size > ats_ma * 1.5).astype(int)

    if has_quote_vol and has_taker_quote:
        quote_vol = df['quote_asset_volume'].clip(lower=1e-10)
        buy_ratio = df['taker_buy_quote_asset_volume'] / quote_vol
        features[f'{prefix}buy_ratio'] = buy_ratio
        br_short = buy_ratio.rolling(window=w_short, min_periods=2).mean()
        br_long = buy_ratio.rolling(window=w_long, min_periods=w_short).mean()
        features[f'{prefix}buy_pressure_momentum'] = br_short - br_long

        ofi = (buy_ratio - 0.5) * quote_vol
        features[f'{prefix}ofi_cumsum_short'] = ofi.rolling(window=config['vol_short_window']).sum()
        features[f'{prefix}ofi_cumsum_long'] = ofi.rolling(window=config['vol_medium_window']).sum()
        features[f'{prefix}ofi_normalized'] = (
            features[f'{prefix}ofi_cumsum_short'] /
            (quote_vol.rolling(window=config['vol_short_window']).sum() + 1e-10)
        )

    if has_quote_vol:
        quote_vol = df['quote_asset_volume']
        returns = df['close'].pct_change()
        features[f'{prefix}vol_return_correlation'] = quote_vol.rolling(window=w_long).corr(returns.abs())

        up_volume = (quote_vol * (returns > 0).astype(int)).rolling(window=w_long).sum()
        down_volume = (quote_vol * (returns < 0).astype(int)).rolling(window=w_long).sum()
        features[f'{prefix}up_down_volume_ratio'] = up_volume / (down_volume + 1e-10)

    return features


# ============================================================================
# Intra-Bar Features (raw 1-minute data)
# ============================================================================

IB_FEATURE_NAMES = [
    'ib_rv_1h', 'ib_rv_cc_ratio', 'ib_max_move_1m', 'ib_autocorr_1m',
    'ib_vwap_dev', 'ib_zero_vol_share', 'ib_volume_herf_1h',
]


def compute_intrabar_features(df_1m: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    Intra-bar features from raw 1-minute bars, indexed at base-frequency
    bar-end timestamps. All quantities at stamp t use 1m bars ENDING <= t.

    df_1m: [timestamp (bar-end), close, volume] sorted by time.
    """
    ib = config['intrabar']
    rule = get_frequency_config(BASE_FREQUENCY)['resample_rule']

    s = df_1m.sort_values('timestamp').reset_index(drop=True)
    ts = pd.to_datetime(s['timestamp'])
    close = s['close'].astype(float)
    vol = s['volume'].astype(float)

    with np.errstate(divide='ignore', invalid='ignore'):
        r1 = np.log(close / close.shift(1))
    bucket = ts.dt.ceil(rule)

    g = pd.DataFrame({
        'bucket': bucket, 'r2': r1 ** 2, 'absr': r1.abs(),
        'vol': vol, 'vol2': vol ** 2, 'pv': close * vol,
        'zero': (vol <= 0).astype(float), 'close': close, 'ts': ts,
    })

    # 1m-level rolling stats sampled at bucket ends
    w1m = ib['autocorr_window_1m']
    g['ac'] = _rolling_autocorr(pd.Series(r1), w1m, 1).values

    # Day-anchored VWAP (causal cumulative within UTC day)
    day = ts.dt.normalize()
    g['cum_pv'] = g.groupby(day.values)['pv'].cumsum()
    g['cum_vol'] = g.groupby(day.values)['vol'].cumsum()

    agg = g.groupby('bucket').agg(
        rv=('r2', 'sum'), max_abs=('absr', 'max'),
        vol_sum=('vol', 'sum'), vol2_sum=('vol2', 'sum'),
        zero_cnt=('zero', 'sum'), n=('r2', 'size'),
        last_close=('close', 'last'), last_cum_pv=('cum_pv', 'last'),
        last_cum_vol=('cum_vol', 'last'), last_ac=('ac', 'last'),
    )

    out = pd.DataFrame(index=agg.index)
    w_rv = ib['rv_window_bars']

    rv_1h = agg['rv'].rolling(w_rv, min_periods=w_rv).sum()
    out['ib_rv_1h'] = np.sqrt(rv_1h) * np.sqrt(BARS_PER_DAY / w_rv * 365)

    # Close-to-close realized variance over the same window (10min returns)
    with np.errstate(divide='ignore', invalid='ignore'):
        r10 = np.log(agg['last_close'] / agg['last_close'].shift(1))
    cc_var = (r10 ** 2).rolling(w_rv, min_periods=w_rv).sum()
    out['ib_rv_cc_ratio'] = np.sqrt(rv_1h / (cc_var + 1e-18))

    # Largest 1m move in the trailing hour, normalized by its trailing-1d level
    mm = agg['max_abs'].rolling(w_rv, min_periods=1).max()
    mm_norm = agg['max_abs'].rolling(ib['maxmove_norm_window_bars'], min_periods=12).mean()
    out['ib_max_move_1m'] = mm / (mm_norm + 1e-10)

    out['ib_autocorr_1m'] = agg['last_ac']

    out['ib_vwap_dev'] = agg['last_close'] / (agg['last_cum_pv'] /
                                              (agg['last_cum_vol'] + 1e-10) + 1e-10) - 1

    w_z = ib['zero_vol_window_bars']
    out['ib_zero_vol_share'] = (agg['zero_cnt'].rolling(w_z, min_periods=12).sum() /
                                agg['n'].rolling(w_z, min_periods=12).sum())

    # Volume concentration (Herfindahl of 1m volume shares, trailing hour)
    v_sum = agg['vol_sum'].rolling(w_rv, min_periods=w_rv).sum()
    v2_sum = agg['vol2_sum'].rolling(w_rv, min_periods=w_rv).sum()
    out['ib_volume_herf_1h'] = v2_sum / (v_sum ** 2 + 1e-18)

    out.index.name = 'timestamp'
    return out.reset_index()


# ============================================================================
# Momentum Quality
# ============================================================================

def calculate_momentum_quality_features(df: pd.DataFrame, config: Dict) -> Dict[str, pd.Series]:
    prefix = 'mq_'
    features = {}

    returns = df['close'].pct_change()
    sign = np.sign(returns)

    for name, window in [('short', config['momentum_short']), ('medium', config['momentum_medium'])]:
        mean_sign = sign.rolling(window=window, min_periods=max(2, window // 2)).mean()
        features[f'{prefix}trend_consistency_{name}'] = 0.5 * (1 + sign * mean_sign)

    momentum_mean = returns.rolling(window=config['momentum_long']).mean()
    momentum_std = returns.rolling(window=config['momentum_long']).std()
    features[f'{prefix}momentum_strength'] = momentum_mean.abs() / (momentum_std + 1e-10)

    momentum_current = returns.rolling(window=config['momentum_medium']).sum()
    features[f'{prefix}momentum_acceleration'] = momentum_current - momentum_current.shift(config['momentum_medium'])

    vol_adj = returns / (returns.rolling(window=config['momentum_short']).std() + 1e-10)
    features[f'{prefix}vol_adjusted_momentum'] = vol_adj.rolling(window=config['momentum_short']).mean()

    return features


# ============================================================================
# Statistical
# ============================================================================

def calculate_statistical_features(df: pd.DataFrame, config: Dict) -> Dict[str, pd.Series]:
    prefix = 'st_'
    returns = np.log(df['close'] / df['close'].shift(1))
    window = config['statistical_window']
    return {
        f'{prefix}skew': returns.rolling(window=window, min_periods=window // 2).skew(),
        f'{prefix}kurtosis': returns.rolling(window=window, min_periods=window // 2).kurt(),
    }


# ============================================================================
# Price / Momentum
# ============================================================================

def calculate_price_features(df: pd.DataFrame, config: Dict) -> Dict[str, pd.Series]:
    prefix = 'px_'
    features = {}

    close = df['close']
    returns = close.pct_change()

    features[f'{prefix}change_pct_2periods'] = close.pct_change(periods=2)
    features[f'{prefix}change_pct_8periods'] = close.pct_change(periods=8)
    features[f'{prefix}ret_1d'] = close.pct_change(periods=BARS_PER_DAY)
    features[f'{prefix}ret_3d'] = close.pct_change(periods=3 * BARS_PER_DAY)

    features[f'{prefix}volatility_4periods'] = returns.rolling(window=4).std() * ANNUALIZER
    features[f'{prefix}high_low_pct'] = (df['high'] - df['low']) / close

    return features


# ============================================================================
# Volume
# ============================================================================

def calculate_volume_features(df: pd.DataFrame, config: Dict) -> Dict[str, pd.Series]:
    prefix = 'vl_'
    features = {}

    volume = df['volume']
    close = df['close']
    volume_ma = volume.rolling(window=config['volume_ma_window'],
                               min_periods=config['volume_ma_window'] // 4).mean()

    volume_ratio = volume / (volume_ma + 1e-10)
    features[f'{prefix}volume_ratio'] = volume_ratio

    if 'taker_buy_base_asset_volume' in df.columns:
        features[f'{prefix}taker_buy_ratio'] = df['taker_buy_base_asset_volume'] / (volume + 1e-10)
    else:
        features[f'{prefix}taker_buy_ratio'] = pd.Series(0.5, index=df.index)

    # Unit-correct replacement for the old volume_price_trend (price x base
    # volume - cross-sectionally meaningless): direction times relative volume
    direction = np.sign(close.diff())
    features[f'{prefix}signed_volume_ratio'] = direction * volume_ratio

    return features


# ============================================================================
# Price-vs-SMA (relative)
# ============================================================================

def calculate_moving_averages(df: pd.DataFrame, config: Dict) -> Dict[str, pd.Series]:
    prefix = 'ma_'
    features = {}
    close = df['close']

    period_names = ['short', 'medium', 'long']
    for i, period in enumerate(config['sma_periods'][:3]):
        name = period_names[i]
        sma = close.rolling(window=period, min_periods=max(2, period // 2)).mean()
        features[f'{prefix}price_vs_sma_{name}'] = close / (sma + 1e-10) - 1

    return features


# ============================================================================
# Technical Indicators
# ============================================================================

def calculate_technical_indicators(df: pd.DataFrame, config: Dict) -> Dict[str, pd.Series]:
    features = {}

    close, high, low = df['close'], df['high'], df['low']

    # fillna=False everywhere: warmup periods stay NaN instead of fabricated
    rsi = ta.momentum.RSIIndicator(close=close, window=config['rsi_period'], fillna=False)
    features['rs_rsi'] = rsi.rsi()

    bb = ta.volatility.BollingerBands(
        close=close, window=config['bollinger_period'],
        window_dev=config['bollinger_std'], fillna=False)
    features['bb_position'] = bb.bollinger_pband()
    features['bb_width'] = bb.bollinger_wband()

    macd = ta.trend.MACD(
        close=close, window_fast=config['macd_fast'],
        window_slow=config['macd_slow'], window_sign=config['macd_signal'], fillna=False)
    features['mc_macd_histogram'] = macd.macd_diff()

    atr = ta.volatility.AverageTrueRange(
        high=high, low=low, close=close, window=config['atr_period'], fillna=False)
    atr_abs = atr.average_true_range()
    features['at_atr'] = atr_abs / (close + 1e-10)  # relative ATR

    dmi = ta.trend.ADXIndicator(
        high=high, low=low, close=close, window=config['dmi_period'], fillna=False)
    features['dm_di_plus'] = dmi.adx_pos()
    features['dm_di_minus'] = dmi.adx_neg()
    features['dm_adx'] = dmi.adx()

    period = config['chandelier_period']
    mult = config['chandelier_mult']
    highest_high = high.rolling(window=period, min_periods=period // 2).max()
    lowest_low = low.rolling(window=period, min_periods=period // 2).min()
    features['ch_long_exit_dist'] = (close - (highest_high - atr_abs * mult)) / (close + 1e-10)
    features['ch_short_exit_dist'] = ((lowest_low + atr_abs * mult) - close) / (close + 1e-10)

    return features


# ============================================================================
# Time Encoding
# ============================================================================

def calculate_time_features(df: pd.DataFrame, config: Dict) -> Dict[str, pd.Series]:
    """Cyclical encodings + funding-settlement proximity. These are
    cross-sectionally constant: useful as regime conditioners / interaction
    terms, never as direct cross-sectional signals."""
    prefix = 'tm_'
    timestamp = pd.to_datetime(df['timestamp'])
    frac_day = (timestamp.dt.hour * 60 + timestamp.dt.minute) / (24 * 60)
    dow = timestamp.dt.dayofweek

    funding_hours = config['funding_hours_utc']
    window_bars = config['funding_window_bars']
    bar_minutes = 24 * 60 // BARS_PER_DAY
    # Within `funding_window_bars` BEFORE a settlement hour
    minutes_of_day = timestamp.dt.hour * 60 + timestamp.dt.minute
    in_window = pd.Series(False, index=df.index)
    for fh in funding_hours:
        target = fh * 60
        dist = (target - minutes_of_day) % (24 * 60)
        in_window |= (dist > 0) & (dist <= window_bars * bar_minutes)

    return {
        f'{prefix}hour_sin': np.sin(2 * np.pi * frac_day),
        f'{prefix}hour_cos': np.cos(2 * np.pi * frac_day),
        f'{prefix}dow_sin': np.sin(2 * np.pi * dow / 7),
        f'{prefix}dow_cos': np.cos(2 * np.pi * dow / 7),
        f'{prefix}funding_window': in_window.astype(int),
    }


# ============================================================================
# Market Context / Lead-Lag (vs the market factor)
# ============================================================================

MK_FEATURE_NAMES = [
    'mk_corr_market_1d', 'mk_beta_drift', 'mk_lag_response_gap',
    'mk_market_ret_1h', 'mk_market_vol_1d', 'mk_market_move_z',
    'mk_lag_corr_short', 'mk_lag_corr_long',
]


def calculate_market_context_features(df: pd.DataFrame, factors: Optional[pd.DataFrame],
                                      loadings_df: pd.DataFrame,
                                      config: Dict) -> Dict[str, pd.Series]:
    """
    Symbol-vs-market dynamics plus market-state context.

    mk_lag_response_gap is the alt-lag feature: how much of the recent
    market move (beta-scaled) this symbol has NOT yet matched. Positive =
    underreacted = expected catch-up. Documented lead-lag effect in crypto.
    """
    features = _nan_series(df, MK_FEATURE_NAMES)
    if factors is None or factors.empty:
        return features

    mc = config['market_context']
    ts = pd.to_datetime(df['timestamp'])
    f_mkt = factors['market_factor'].reindex(ts).values
    f_mkt = pd.Series(f_mkt, index=df.index)

    returns = df['close'].pct_change()

    w = mc['corr_window']
    features['mk_corr_market_1d'] = returns.rolling(w, min_periods=w // 2).corr(f_mkt)

    # Fast beta vs the (daily) regression beta
    cov = returns.rolling(w, min_periods=w // 2).cov(f_mkt)
    var = f_mkt.rolling(w, min_periods=w // 2).var()
    fast_beta = cov / (var + 1e-18)

    beta_daily = pd.Series(np.nan, index=df.index)
    if not loadings_df.empty and 'beta_market' in loadings_df.columns:
        ld = loadings_df.copy()
        ld['date'] = pd.to_datetime(ld['date'])
        merged = pd.DataFrame({'date': ts.dt.normalize()}).merge(
            ld[['date', 'beta_market']], on='date', how='left')
        beta_daily = pd.Series(merged['beta_market'].values, index=df.index)

    features['mk_beta_drift'] = fast_beta - beta_daily

    # Lead-lag: beta-scaled recent market move minus own recent move
    p = mc['lag_response_bars']
    mkt_move = f_mkt.rolling(p, min_periods=p).sum()
    own_move = returns.rolling(p, min_periods=p).sum()
    features['mk_lag_response_gap'] = beta_daily * mkt_move - own_move

    # Speed of adjustment: corr of own return with the LAGGED market move.
    # High = slow adjuster (follows the market with delay) - the classic
    # cross-sectional crypto lead-lag sort
    f_mkt_lag = f_mkt.shift(1)
    features['mk_lag_corr_short'] = returns.rolling(
        w, min_periods=w // 2).corr(f_mkt_lag)
    wl = mc['lag_corr_window_long']
    features['mk_lag_corr_long'] = returns.rolling(
        wl, min_periods=wl // 2).corr(f_mkt_lag)

    # Market-state context (cross-sectionally constant - conditioners only)
    features['mk_market_ret_1h'] = f_mkt.rolling(6, min_periods=6).sum()
    mvol = f_mkt.rolling(mc['market_move_z_window'],
                         min_periods=mc['market_move_z_window'] // 2).std()
    features['mk_market_vol_1d'] = mvol * ANNUALIZER
    features['mk_market_move_z'] = mkt_move / (mvol * np.sqrt(p) + 1e-18)

    return features


# ============================================================================
# Market-Cap Derived
# ============================================================================

CAP_FEATURE_NAMES = ['cap_log_mcap', 'cap_turnover', 'cap_supply_inflation']


def calculate_cap_features(df: pd.DataFrame, mcap_series: Optional[pd.Series],
                           config: Dict) -> Dict[str, pd.Series]:
    """
    cap_turnover: trailing 1d dollar volume / market cap (liquidity-adjusted
    attention). cap_supply_inflation: trailing mcap growth minus price return
    ~ circulating-supply growth - a token-unlock detector.
    """
    features = _nan_series(df, CAP_FEATURE_NAMES)
    if mcap_series is None or mcap_series.dropna().empty:
        return features

    cap_cfg = config['cap']
    ts = pd.to_datetime(df['timestamp'])
    # mcap_series: daily, already lagged/ffilled by the loader
    mcap = mcap_series.reindex(ts.dt.normalize()).values
    mcap = pd.Series(mcap, index=df.index)

    features['cap_log_mcap'] = np.log(mcap.clip(lower=1.0))

    dollar_vol = df['quote_asset_volume'] if 'quote_asset_volume' in df.columns else \
        df['volume'] * df['close']
    w = cap_cfg['turnover_window_bars']
    features['cap_turnover'] = dollar_vol.rolling(w, min_periods=w // 4).sum() / (mcap + 1e-10)

    days = cap_cfg['supply_inflation_days']
    bars = days * BARS_PER_DAY
    with np.errstate(divide='ignore', invalid='ignore'):
        mcap_growth = np.log(mcap / mcap.shift(bars))
        price_growth = np.log(df['close'] / df['close'].shift(bars))
    features['cap_supply_inflation'] = mcap_growth - price_growth

    return features


# ============================================================================
# Per-Symbol Seasonality (trailing same-bucket residual stats)
# ============================================================================

SN_FEATURE_NAMES = ['sn_tod_res', 'sn_tod_vol_ratio', 'sn_dow_res']


def calculate_seasonality_features(df: pd.DataFrame, residual_series: pd.Series,
                                   config: Dict) -> Dict[str, pd.Series]:
    """Per-symbol time-of-day / day-of-week residual seasonality.

    Unlike the tm_ sin/cos encodings (identical across names, conditioners
    only), these are each name's OWN trailing seasonal profile, so they vary
    cross-sectionally and work as direct signals:

      sn_tod_res       trailing mean residual in the same HOUR-of-day bucket
                       over the past tod_days days
      sn_tod_vol_ratio same-hour trailing mean |residual| over the all-hours
                       trailing mean (this name's intraday vol profile)
      sn_dow_res       trailing mean of the FULL-day residual sum on the same
                       day-of-week over the past dow_weeks weeks, mapped onto
                       the current day's bars. Completed days only: the
                       same-weekday series is shifted one observation, so the
                       running (partial) current day never feeds its own value.

    All trailing through bar t - causal under the truncation test.
    """
    features = _nan_series(df, SN_FEATURE_NAMES)
    sn = config['seasonality']
    res = residual_series
    if res.notna().sum() < 100:
        return features

    ts = pd.to_datetime(df['timestamp'])
    bars_per_hour = max(1, BARS_PER_DAY // 24)
    hour = ts.dt.hour.values

    tod_w = int(sn['tod_days']) * bars_per_hour
    tod_min = int(sn['min_days']) * bars_per_hour
    by_hour = res.groupby(hour)
    features['sn_tod_res'] = by_hour.transform(
        lambda x: x.rolling(tod_w, min_periods=tod_min).mean())

    abs_res = res.abs()
    tod_vol = abs_res.groupby(hour).transform(
        lambda x: x.rolling(tod_w, min_periods=tod_min).mean())
    all_vol = abs_res.rolling(int(sn['tod_days']) * BARS_PER_DAY,
                              min_periods=int(sn['min_days']) * BARS_PER_DAY).mean()
    features['sn_tod_vol_ratio'] = tod_vol / (all_vol + 1e-12)

    # Day-of-week on completed days: daily sums grouped by weekday, rolling
    # mean of the last dow_weeks same-weekday days, shifted one same-weekday
    # observation (yesterday-and-back only), then mapped back to bars.
    date = ts.dt.normalize()
    daily = res.groupby(date.values).sum(min_count=1)
    dow = pd.Series(pd.DatetimeIndex(daily.index).dayofweek, index=daily.index)
    dow_mean = daily.groupby(dow.values).transform(
        lambda x: x.rolling(int(sn['dow_weeks']),
                            min_periods=int(sn['dow_min_weeks'])).mean().shift(1))
    features['sn_dow_res'] = pd.Series(date.map(dow_mean).values, index=df.index)

    return features


# ============================================================================
# Leader Lead-Lag (vs a single leader asset, slow horizons)
# ============================================================================

def leadlag_feature_names(config: Dict) -> List[str]:
    ll = config['lead_lag']
    return (['ll_leader_beta', 'll_lag_corr']
            + [f'll_leader_gap_{int(w)}b' for w in ll['gap_windows_bars']])


def calculate_leadlag_features(df: pd.DataFrame, symbol: str,
                               leader_close: Optional[pd.Series],
                               config: Dict) -> Dict[str, pd.Series]:
    """Lead-lag vs a single leader asset (config lead_lag.leader_symbol, BTC).

    Complements mk_* (market factor, 30min window) with SLOW leader dynamics:

      ll_leader_beta      7d rolling beta of own returns on leader returns
      ll_leader_gap_{w}b  beta-scaled leader move over w bars minus own move -
                          the share of the leader's multi-hour move this name
                          has NOT yet matched (positive = expected catch-up)
      ll_lag_corr         rolling corr of own bar return with the leader's
                          PRECEDING lag_bars move: how much of a delayed
                          follower this name is

    Leader bar t is knowable at t (bar-end stamps). The leader symbol itself
    returns NaN (self-reference is degenerate).
    """
    ll = config['lead_lag']
    features = _nan_series(df, leadlag_feature_names(config))
    if leader_close is None or symbol == ll['leader_symbol']:
        return features

    ts = pd.to_datetime(df['timestamp'])
    lead = pd.Series(leader_close.reindex(ts).values, index=df.index)
    lead_ret = lead.pct_change()
    own_ret = df['close'].pct_change()

    bw = int(ll['beta_window_bars'])
    cov = own_ret.rolling(bw, min_periods=bw // 2).cov(lead_ret)
    var = lead_ret.rolling(bw, min_periods=bw // 2).var()
    beta = cov / (var + 1e-18)
    features['ll_leader_beta'] = beta

    for w in ll['gap_windows_bars']:
        w = int(w)
        lead_move = lead_ret.rolling(w, min_periods=w).sum()
        own_move = own_ret.rolling(w, min_periods=w).sum()
        features[f'll_leader_gap_{w}b'] = beta * lead_move - own_move

    lead_lagged = lead_ret.rolling(int(ll['lag_bars']),
                                   min_periods=int(ll['lag_bars'])).sum().shift(1)
    cw = int(ll['lag_corr_window_bars'])
    features['ll_lag_corr'] = own_ret.rolling(cw, min_periods=cw // 2).corr(lead_lagged)

    return features


# ============================================================================
# Macro / event features (etl/macro.py tables). Three classes:
#   ev_  event timing (cross-sectionally CONSTANT: gate/interaction material)
#   mx_  macro state  (cross-sectionally CONSTANT: gate/interaction material)
#   mb_  per-name macro sensitivities (CROSS-SECTIONAL: direct signal material)
# macro_daily values are ALREADY availability-dated by the ETL (first UTC date
# each value is usable), so mapping a date's value onto that date's bars is
# causal. Event times are exact UTC datetimes from published schedules, so
# "hours until the next event" is legitimately knowable ahead of time.
# ============================================================================

EV_FEATURE_NAMES = [
    'ev_fomc_day', 'ev_cpi_day', 'ev_nfp_day',
    'ev_hours_to_event', 'ev_hours_since_event', 'ev_event_window',
]

MX_FEATURE_NAMES = [
    'mx_rates2y_chg_1d', 'mx_rates2y_chg_5d', 'mx_curve_2s10s',
    'mx_dollar_chg_5d', 'mx_vix_z', 'mx_vix_chg_1d', 'mx_breakeven_chg_5d',
    'mx_net_liquidity_chg_4w', 'mx_stables_flow_7d', 'mx_fear_greed',
    'mx_dominance_chg_7d', 'mx_event_shock',
]

MB_FEATURE_NAMES = [
    'mb_beta_rates', 'mb_beta_dollar', 'mb_beta_vix', 'mb_beta_stables',
    'mb_beta_dominance', 'mb_event_vol_ratio', 'mb_event_volume_ratio',
    'mb_event_drift',
]


def _daily_to_bars(daily: pd.Series, ts: pd.Series) -> pd.Series:
    """Map an availability-dated daily series onto bar timestamps (a date's
    value applies to all bars of that UTC date)."""
    return pd.Series(daily.reindex(ts.dt.normalize()).values, index=ts.index)


def calculate_event_features(df: pd.DataFrame, events: Optional[pd.DataFrame],
                             config: Dict) -> Dict[str, pd.Series]:
    """Event-timing features from the published macro calendar.

    ev_*_day: the bar's UTC date is an event date. ev_hours_to_event /
    ev_hours_since_event: hours to the next / since the last event of ANY
    type, clipped (pre-event positioning / post-event resolution regimes).
    ev_event_window: within +-event_window_bars of an exact event time.
    Constant across names by construction - gate material for the DSL."""
    features = _nan_series(df, EV_FEATURE_NAMES)
    if events is None or events.empty:
        return features
    mc = config['macro']
    ts = pd.to_datetime(df['timestamp'])
    dates = ts.dt.normalize()

    ev_times = np.sort(pd.to_datetime(events['event_time_utc']).values)
    for etype in ('fomc', 'cpi', 'nfp'):
        edates = set(pd.to_datetime(
            events.loc[events['event_type'] == etype, 'event_time_utc']
        ).dt.normalize())
        features[f'ev_{etype}_day'] = dates.isin(edates).astype(float)

    ts_v = ts.values
    nxt = np.searchsorted(ev_times, ts_v, side='left')
    prv = nxt - 1
    hours_clip = float(mc['hours_clip'])
    to_ev = np.full(len(df), hours_clip)
    ok = nxt < len(ev_times)
    to_ev[ok] = (ev_times[nxt[ok]] - ts_v[ok]) / np.timedelta64(1, 'h')
    since_ev = np.full(len(df), hours_clip)
    ok = prv >= 0
    since_ev[ok] = (ts_v[ok] - ev_times[prv[ok]]) / np.timedelta64(1, 'h')
    features['ev_hours_to_event'] = pd.Series(
        np.clip(to_ev, 0.0, hours_clip), index=df.index)
    features['ev_hours_since_event'] = pd.Series(
        np.clip(since_ev, 0.0, hours_clip), index=df.index)

    bar_h = 24.0 / BARS_PER_DAY
    win_h = float(mc['event_window_bars']) * bar_h
    features['ev_event_window'] = pd.Series(
        ((to_ev <= win_h) | (since_ev <= win_h)).astype(float), index=df.index)
    return features


def calculate_macro_state_features(df: pd.DataFrame,
                                   macro_daily: Optional[pd.DataFrame],
                                   dominance: Optional[pd.Series],
                                   events: Optional[pd.DataFrame],
                                   config: Dict) -> Dict[str, pd.Series]:
    """Macro-state conditioners (constant across names - gate material).

    Rate impulse, curve, dollar and inflation-expectation moves, VIX level/
    impulse, Fed net-liquidity impulse (balance sheet minus RRP - the classic
    crypto liquidity driver), stablecoin-supply flow (crypto dry powder),
    Fear&Greed, BTC-dominance rotation, and mx_event_shock = the |2Y move|
    printed by the latest CPI/FOMC day (the market's own measure of the
    surprise - no consensus data needed, fully point-in-time)."""
    features = _nan_series(df, MX_FEATURE_NAMES)
    mc = config['macro']
    ts = pd.to_datetime(df['timestamp'])
    if macro_daily is None or macro_daily.empty:
        m = pd.DataFrame()
    else:
        m = macro_daily.set_index('date').sort_index() \
            if 'date' in macro_daily.columns else macro_daily.sort_index()
        cal = pd.date_range(m.index.min(), ts.dt.normalize().max(), freq='D')
        m = m.reindex(cal).ffill(limit=int(mc['ffill_limit_days']))

    def col(name):
        return m[name] if name in m.columns else None

    r2 = col('rates2y')
    if r2 is not None:
        features['mx_rates2y_chg_1d'] = _daily_to_bars(r2.diff(1), ts)
        features['mx_rates2y_chg_5d'] = _daily_to_bars(r2.diff(5), ts)
    r10 = col('rates10y')
    if r2 is not None and r10 is not None:
        features['mx_curve_2s10s'] = _daily_to_bars(r10 - r2, ts)
    dol = col('dollar')
    if dol is not None:
        features['mx_dollar_chg_5d'] = _daily_to_bars(dol.pct_change(5), ts)
    vix = col('vix')
    if vix is not None:
        w = int(mc['vix_z_window_days'])
        z = (vix - vix.rolling(w, min_periods=w // 4).mean()) / \
            (vix.rolling(w, min_periods=w // 4).std() + 1e-10)
        features['mx_vix_z'] = _daily_to_bars(z, ts)
        features['mx_vix_chg_1d'] = _daily_to_bars(vix.diff(1), ts)
    be = col('breakeven10')
    if be is not None:
        features['mx_breakeven_chg_5d'] = _daily_to_bars(be.diff(5), ts)
    bs, rrp = col('fed_bs'), col('rrp')
    if bs is not None and rrp is not None:
        # WALCL is in $mn, RRPONTSYD in $bn -> net liquidity in $bn
        net = bs / 1000.0 - rrp
        features['mx_net_liquidity_chg_4w'] = _daily_to_bars(
            net.pct_change(28), ts)
    st = col('stables_mcap')
    if st is not None:
        features['mx_stables_flow_7d'] = _daily_to_bars(st.pct_change(7), ts)
    fg = col('fear_greed')
    if fg is not None:
        features['mx_fear_greed'] = _daily_to_bars(fg, ts)
    if dominance is not None and not dominance.dropna().empty:
        features['mx_dominance_chg_7d'] = _daily_to_bars(
            dominance.pct_change(7), ts)

    # Event shock: the 2Y move becomes observable one availability day after
    # the event day, so stamp |chg_1d| on event_date + 1.
    if r2 is not None and events is not None and not events.empty:
        shock_dates = set(pd.to_datetime(events['event_time_utc'])
                          .dt.normalize() + pd.Timedelta(days=1))
        chg = r2.diff(1).abs()
        shock = chg.where(chg.index.isin(shock_dates), 0.0)
        features['mx_event_shock'] = _daily_to_bars(shock, ts)
    return features


def calculate_macrobeta_features(df: pd.DataFrame, residual_series: pd.Series,
                                 macro_daily: Optional[pd.DataFrame],
                                 dominance: Optional[pd.Series],
                                 events: Optional[pd.DataFrame],
                                 config: Dict) -> Dict[str, pd.Series]:
    """Per-name macro sensitivities - the CROSS-SECTIONAL macro features.

    mb_beta_*: rolling beta of the name's daily residual to daily macro
    impulses (rates, dollar, VIX, stablecoin flow, BTC dominance) - which
    coins are rate-sensitive, dollar-sensitive, rotation-exposed. Completed
    days only (shift 1 day).
    mb_event_vol_ratio / mb_event_volume_ratio: the name's trailing average
    |residual| / relative volume on macro-event days vs its everyday
    baseline - the 'reacts to FOMC' profile, per coin.
    mb_event_drift: trailing mean of the name's signed residual over the 24h
    AFTER each event time - coins that systematically rally or fade after
    macro prints. Each event enters the trailing mean only once fully
    elapsed (available from event_time + event_response_hours)."""
    features = _nan_series(df, MB_FEATURE_NAMES)
    mc = config['macro']
    res = residual_series
    if res.notna().sum() < 100:
        return features
    ts = pd.to_datetime(df['timestamp'])
    dates = ts.dt.normalize()

    res_d = res.groupby(dates.values).sum(min_count=1)
    W, MINW = int(mc['beta_window_days']), int(mc['beta_min_days'])

    impulses = {}
    if macro_daily is not None and not macro_daily.empty:
        m = macro_daily.set_index('date').sort_index() \
            if 'date' in macro_daily.columns else macro_daily.sort_index()
        m = m.reindex(pd.date_range(m.index.min(), dates.max(), freq='D')) \
             .ffill(limit=int(mc['ffill_limit_days']))
        if 'rates2y' in m:
            impulses['mb_beta_rates'] = m['rates2y'].diff(1)
        if 'dollar' in m:
            impulses['mb_beta_dollar'] = m['dollar'].pct_change(1)
        if 'vix' in m:
            impulses['mb_beta_vix'] = m['vix'].diff(1)
        if 'stables_mcap' in m:
            impulses['mb_beta_stables'] = m['stables_mcap'].pct_change(1)
    if dominance is not None and not dominance.dropna().empty:
        impulses['mb_beta_dominance'] = dominance.pct_change(1)

    for name, x in impulses.items():
        x_al = x.reindex(res_d.index)
        cov = res_d.rolling(W, min_periods=MINW).cov(x_al)
        var = x_al.rolling(W, min_periods=MINW).var()
        beta = (cov / (var + 1e-18)).shift(1)     # completed days only
        features[name] = _daily_to_bars(beta, ts)

    if events is None or events.empty:
        return features

    K, MINE = int(mc['event_lookback_events']), int(mc['event_min_events'])
    ev_dates = pd.DatetimeIndex(
        pd.to_datetime(events['event_time_utc']).dt.normalize().unique()
    ).sort_values()

    def _event_ratio(daily_stat: pd.Series) -> pd.Series:
        """Trailing mean of the stat over the last K event days, divided by
        its everyday 60d baseline; completed days only (shift 1)."""
        on_events = daily_stat.reindex(ev_dates).dropna()
        trail = on_events.rolling(K, min_periods=MINE).mean()
        trail = trail.reindex(daily_stat.index).ffill()
        base = daily_stat.rolling(60, min_periods=20).mean()
        return (trail / (base + 1e-18)).shift(1)

    absres_d = res.abs().groupby(dates.values).sum(min_count=1)
    features['mb_event_vol_ratio'] = _daily_to_bars(_event_ratio(absres_d), ts)

    dollar_vol = df['quote_asset_volume'] if 'quote_asset_volume' in df.columns \
        else df['volume'] * df['close']
    dv_d = dollar_vol.groupby(dates.values).sum(min_count=1)
    relvol_d = dv_d / (dv_d.rolling(20, min_periods=10).mean() + 1e-10)
    features['mb_event_volume_ratio'] = _daily_to_bars(
        _event_ratio(relvol_d), ts)

    # Post-event drift: signed residual over (event_time, event_time + H];
    # each event's return becomes usable at event_time + H.
    H = np.timedelta64(int(float(mc['event_response_hours']) * 60), 'm')
    ts_v = ts.values
    cum = res.fillna(0.0).cumsum().values
    post_rets, avail = [], []
    for e in np.sort(pd.to_datetime(events['event_time_utc']).values):
        lo = np.searchsorted(ts_v, e, side='right')
        hi = np.searchsorted(ts_v, e + H, side='right')
        if hi <= lo or hi > len(ts_v):
            continue
        post_rets.append(cum[hi - 1] - (cum[lo - 1] if lo > 0 else 0.0))
        avail.append(e + H)
    if post_rets:
        trail = pd.Series(post_rets).rolling(K, min_periods=MINE).mean().values
        pos = np.searchsorted(np.array(avail), ts_v, side='right') - 1
        vals = np.full(len(df), np.nan)
        ok = pos >= 0
        vals[ok] = trail[pos[ok]]
        features['mb_event_drift'] = pd.Series(vals, index=df.index)
    return features


# ============================================================================
# Residual Dynamics
# ============================================================================

RES_FEATURE_NAMES = [
    'res_autocorr_lag1', 'res_autocorr_lag6', 'res_autocorr_lag36',
    'res_vol_short', 'res_vol_long', 'res_zscore', 'res_extreme_flag',
    'res_zscore_momentum_4', 'res_zscore_momentum_8', 'res_zscore_accel',
    'res_cumsum_short', 'res_cumsum_long', 'res_cumsum_xlong',
    'res_reversion_speed', 'res_sign_persistence', 'res_ac1_rolling',
    'res_mr_regime', 'res_mr_signal', 'res_mr_signal_scaled',
    'res_mr_regime_strength', 'res_mr_signal_weighted',
    'res_spread_drawdown', 'res_spread_runup', 'res_bars_since_extreme',
    'res_hurst',
]


def calculate_residual_features(df: pd.DataFrame, residual_series: pd.Series,
                                config: Dict) -> Dict[str, pd.Series]:
    """Features on single-bar residuals (knowable at t; targets start t+1)."""
    prefix = 'res_'
    features = _nan_series(df, RES_FEATURE_NAMES)
    res = residual_series

    if res.notna().sum() < 100:
        return features

    for lag in config['residual_autocorr_lags']:
        features[f'{prefix}autocorr_lag{lag}'] = _rolling_autocorr(res, config['autocorr_window'], lag)

    vol_windows = config['residual_vol_windows'][:2]
    features[f'{prefix}vol_short'] = res.rolling(window=vol_windows[0]).std()
    features[f'{prefix}vol_long'] = res.rolling(window=vol_windows[1]).std()

    window = config['residual_zscore_window']
    res_mean = res.rolling(window=window).mean()
    res_std = res.rolling(window=window).std()
    zscore = (res - res_mean) / (res_std + 1e-10)
    features[f'{prefix}zscore'] = zscore
    features[f'{prefix}extreme_flag'] = (zscore.abs() > config['residual_extreme_z']).astype(int)

    features[f'{prefix}zscore_momentum_4'] = zscore - zscore.shift(4)
    features[f'{prefix}zscore_momentum_8'] = zscore - zscore.shift(8)
    features[f'{prefix}zscore_accel'] = features[f'{prefix}zscore_momentum_4'] - \
        features[f'{prefix}zscore_momentum_4'].shift(4)

    features[f'{prefix}cumsum_short'] = res.rolling(window=vol_windows[0]).sum()
    features[f'{prefix}cumsum_long'] = res.rolling(window=vol_windows[1]).sum()
    features[f'{prefix}cumsum_xlong'] = res.rolling(window=config['residual_cumsum_xlong']).sum()
    features[f'{prefix}reversion_speed'] = zscore.diff().abs().rolling(window=8).mean()

    res_sign = (res > 0).astype(int)
    sign_changes = res_sign != res_sign.shift(1)
    sign_groups = sign_changes.cumsum()
    features[f'{prefix}sign_persistence'] = sign_groups.groupby(sign_groups).cumcount() + 1

    ac1 = _rolling_autocorr(res, config['residual_ac1_window'], 1)
    features[f'{prefix}ac1_rolling'] = ac1
    threshold = config['residual_mr_ac1_threshold']
    mr_regime = (ac1 < threshold).astype(int)
    features[f'{prefix}mr_regime'] = mr_regime
    features[f'{prefix}mr_signal'] = pd.Series(np.where(mr_regime == 1, -res, 0.0), index=df.index)
    features[f'{prefix}mr_signal_scaled'] = pd.Series(np.where(mr_regime == 1, -zscore, 0.0), index=df.index)
    features[f'{prefix}mr_regime_strength'] = pd.Series(np.where(mr_regime == 1, -ac1, 0.0), index=df.index)
    features[f'{prefix}mr_signal_weighted'] = (
        features[f'{prefix}mr_signal_scaled'] * features[f'{prefix}mr_regime_strength']
    )

    # Spread-state features on the cumulative residual
    spread = res.fillna(0).cumsum()
    leading_nan = res.isna() & (res.isna().cumprod() == 1)
    spread[leading_nan] = np.nan
    sw = config['residual_spread_window']
    roll_max = spread.rolling(sw, min_periods=sw // 2).max()
    roll_min = spread.rolling(sw, min_periods=sw // 2).min()
    vol_norm = features[f'{prefix}vol_long'] + 1e-10
    features[f'{prefix}spread_drawdown'] = (spread - roll_max) / vol_norm
    features[f'{prefix}spread_runup'] = (spread - roll_min) / vol_norm

    # Bars since the last |z| > threshold event (NaN before the first event)
    is_extreme = zscore.abs() > config['residual_extreme_z']
    event_groups = is_extreme.cumsum()
    bars_since = event_groups.groupby(event_groups).cumcount().astype(float)
    bars_since[event_groups == 0] = np.nan
    features[f'{prefix}bars_since_extreme'] = bars_since

    # Hurst exponent of the spread via the variance-ratio identity:
    # H = 0.5 + ln(VR(q)) / (2 ln q), VR on residual (spread increments)
    q = config['residual_hurst_q']
    vr = _variance_ratio(res, q, config['residual_spread_window'])
    with np.errstate(divide='ignore', invalid='ignore'):
        features[f'{prefix}hurst'] = (0.5 + np.log(vr) / (2 * np.log(q))).clip(0, 1)

    return features


# ============================================================================
# OU Process on CUMULATIVE residual (unchanged math)
# ============================================================================

OU_FEATURE_NAMES = [
    'ou_lambda_short', 'ou_lambda_long', 'ou_halflife_short', 'ou_halflife_long',
    'ou_zscore', 'ou_expected_change', 'ou_signal', 'ou_signal_scaled', 'ou_regime',
]


def _ou_params_rolling(x: np.ndarray, window: int, min_obs: int):
    """Rolling OU fit on a level series via dX = a + b*X (vectorized)."""
    n = len(x)
    lam = np.full(n, np.nan)
    mu = np.full(n, np.nan)
    sig = np.full(n, np.nan)

    X = x[:-1]
    Y = np.diff(x)
    if len(X) < min_obs:
        return lam, mu, sig

    s = pd.DataFrame({'X': X, 'Y': Y, 'XX': X * X, 'XY': X * Y, 'YY': Y * Y})
    roll = s.rolling(window, min_periods=min_obs).sum()
    cnt = s['X'].rolling(window, min_periods=min_obs).count()

    sx, sy, sxx, sxy, syy = (roll['X'].values, roll['Y'].values,
                             roll['XX'].values, roll['XY'].values, roll['YY'].values)
    c = cnt.values
    denom = c * sxx - sx * sx
    with np.errstate(invalid='ignore', divide='ignore'):
        b = (c * sxy - sx * sy) / denom
        a = (sy - b * sx) / c
        ss_res = syy - a * sy - b * sxy
        sigma = np.sqrt(np.maximum(ss_res / np.maximum(c - 2, 1), 0))
        lam_v = -b
        mu_v = np.where(np.abs(b) > 1e-12, -a / b, np.nan)

    valid = np.isfinite(lam_v) & (lam_v > 0) & (lam_v <= 2)
    lam[1:][valid] = lam_v[valid]
    mu[1:][valid] = mu_v[valid]
    sig[1:][valid] = sigma[valid]
    return lam, mu, sig


def calculate_ou_process_features(df: pd.DataFrame, residual_series: pd.Series,
                                  config: Dict) -> Dict[str, pd.Series]:
    features = _nan_series(df, OU_FEATURE_NAMES)

    ou_config = config.get('ou_process', {})
    short_window = ou_config.get('short_window', 144)
    long_window = ou_config.get('long_window', 1008)
    min_obs = ou_config.get('min_observations', 100)
    lambda_threshold = ou_config.get('lambda_threshold', 0.01)

    res = residual_series
    if res.notna().sum() < min_obs:
        return features

    spread = res.fillna(0).cumsum()
    spread[res.isna() & (res.isna().cumprod() == 1)] = np.nan
    x = spread.values.astype(float)

    lam_s, _, _ = _ou_params_rolling(x, short_window, min_obs)
    lam_l, mu_l, sig_l = _ou_params_rolling(x, long_window, min_obs)

    idx = df.index
    features['ou_lambda_short'] = pd.Series(lam_s, index=idx)
    features['ou_lambda_long'] = pd.Series(lam_l, index=idx)
    with np.errstate(invalid='ignore', divide='ignore'):
        features['ou_halflife_short'] = pd.Series(
            np.where(lam_s > 0, np.minimum(np.log(2) / lam_s, short_window), np.nan), index=idx)
        features['ou_halflife_long'] = pd.Series(
            np.where(lam_l > 0, np.minimum(np.log(2) / lam_l, long_window), np.nan), index=idx)

        eq_std = np.where(lam_l > 0, sig_l / np.sqrt(2 * lam_l + 1e-12), np.nan)
        deviation = x - mu_l
        zscore = deviation / (eq_std + 1e-12)

    features['ou_zscore'] = pd.Series(zscore, index=idx)
    features['ou_expected_change'] = pd.Series(lam_l * -deviation, index=idx)
    features['ou_signal'] = pd.Series(-np.sign(deviation) * np.abs(zscore), index=idx)
    features['ou_signal_scaled'] = features['ou_signal'] * pd.Series(lam_l, index=idx)
    features['ou_regime'] = pd.Series((lam_l > lambda_threshold).astype(float), index=idx)

    return features


# ============================================================================
# Factor Loading Features
# ============================================================================

FL_FEATURE_NAMES = [
    'fl_beta_market', 'fl_beta_size',
    'fl_beta_market_change_short', 'fl_beta_market_change_long',
    'fl_beta_size_change_short', 'fl_beta_size_change_long',
    'fl_r2_total',
]


def calculate_factor_loading_features(df: pd.DataFrame, loadings_df: pd.DataFrame,
                                      config: Dict) -> Dict[str, pd.Series]:
    """Beta levels/dynamics. Betas dated D are estimated from data before D."""
    features = _nan_series(df, FL_FEATURE_NAMES)

    if loadings_df.empty:
        return features

    loadings_df = loadings_df.copy()
    loadings_df['date'] = pd.to_datetime(loadings_df['date'])

    df_dates = pd.to_datetime(df['timestamp']).dt.normalize()
    merge_cols = ['date'] + [c for c in ['beta_market', 'beta_size', 'r_squared']
                             if c in loadings_df.columns]
    merged = pd.DataFrame({'date': df_dates}).merge(
        loadings_df[merge_cols], on='date', how='left')
    merged.index = df.index

    short_days = max(1, config['vol_short_window'] // BARS_PER_DAY + 1)
    long_days = max(2, config['vol_medium_window'] // BARS_PER_DAY + 1)

    if 'beta_market' in merged.columns:
        b = merged['beta_market']
        features['fl_beta_market'] = b
        features['fl_beta_market_change_short'] = b.diff(short_days * BARS_PER_DAY)
        features['fl_beta_market_change_long'] = b.diff(long_days * BARS_PER_DAY)
    if 'beta_size' in merged.columns:
        b = merged['beta_size']
        features['fl_beta_size'] = b
        features['fl_beta_size_change_short'] = b.diff(short_days * BARS_PER_DAY)
        features['fl_beta_size_change_long'] = b.diff(long_days * BARS_PER_DAY)
    if 'r_squared' in merged.columns:
        features['fl_r2_total'] = merged['r_squared']

    return features


# ============================================================================
# Cross-Sectional Phase (computed once in the main process, panel-wide)
# ============================================================================

CS_FEATURE_NAMES = [
    'cs_rel_volume', 'cs_ret_rank_1h', 'cs_ret_rank_1d',
    'cs_dispersion_1h', 'cs_breadth_sma', 'cs_funding_z', 'cs_mcap_rank',
    'cs_cluster_rel_z',
]


def _cluster_relative_z(close_index: pd.Index, cfg: Dict) -> pd.DataFrame:
    """
    Cluster-relative value: z-score of each name's trailing residual cumsum
    WITHIN its own residual-correlation cluster (e.g. the meme basket).
    Clusters re-estimated every recluster_days from residuals strictly
    before the estimation day - causal. Names outside any cluster get NaN.
    """
    from research.lib.portfolio_opt import residual_clusters

    res = load_data('residual_returns', columns=['timestamp', 'symbol',
                                                 'residual_return'])
    if res.empty:
        return pd.DataFrame(np.nan, index=close_index, columns=[])
    res['timestamp'] = pd.to_datetime(res['timestamp'])
    wide = res.pivot_table(index='timestamp', columns='symbol',
                           values='residual_return', aggfunc='first').sort_index()

    zw = cfg['z_window']
    cumsum = wide.rolling(zw, min_periods=zw // 2).sum()

    out = pd.DataFrame(np.nan, index=wide.index, columns=wide.columns)
    days = wide.index.normalize().unique().sort_values()
    recluster_dates = days[::cfg['recluster_days']]

    for i, d in enumerate(recluster_dates):
        hist = wide[(wide.index < d)].tail(cfg['corr_window_bars'])
        if len(hist) < cfg['corr_window_bars'] // 4:
            continue
        clusters = residual_clusters(hist,
                                     corr_threshold=cfg['corr_threshold'],
                                     min_cluster_size=cfg['min_cluster_size'])
        if not clusters:
            continue
        d_end = recluster_dates[i + 1] if i + 1 < len(recluster_dates) \
            else wide.index[-1] + pd.Timedelta(days=1)
        in_period = (cumsum.index >= d) & (cumsum.index < d_end)
        for members in clusters:
            cols = [m for m in members if m in cumsum.columns]
            if len(cols) < cfg['min_cluster_size']:
                continue
            block = cumsum.loc[in_period, cols]
            mu = block.mean(axis=1)
            sd = block.std(axis=1).replace(0, np.nan)
            out.loc[in_period, cols] = block.sub(mu, axis=0).div(sd, axis=0).clip(-4, 4)

    return out.reindex(close_index)


def compute_cross_sectional_features() -> Optional[pd.DataFrame]:
    """
    Panel features no single-symbol worker can compute. All causal (rolling /
    ranks over data through bar t). Returns a long frame
    [timestamp, symbol, cs_*] passed to workers as per-symbol slices.
    """
    cfg = FEATURE_CONFIG['cross_section']

    px = load_data('prices', columns=['timestamp', 'symbol', 'close', 'quote_asset_volume'])
    if px.empty:
        return None
    px['timestamp'] = pd.to_datetime(px['timestamp'])

    close = px.pivot_table(index='timestamp', columns='symbol', values='close',
                           aggfunc='last').sort_index()
    qvol = px.pivot_table(index='timestamp', columns='symbol', values='quote_asset_volume',
                          aggfunc='last').sort_index()
    rets = close.pct_change(fill_method=None)

    out = {}

    # Relative volume: own volume surprise vs the universe median surprise
    w = cfg['rel_volume_window']
    ratio = qvol / qvol.rolling(w, min_periods=w // 4).mean()
    out['cs_rel_volume'] = ratio.div(ratio.median(axis=1), axis=0)

    # Trailing-return cross-sectional percentile ranks (fill_method=None so
    # gaps yield NaN instead of pad-fabricated returns)
    w1, w2 = cfg['ret_rank_windows'][:2]
    out['cs_ret_rank_1h'] = close.pct_change(w1, fill_method=None).rank(axis=1, pct=True)
    out['cs_ret_rank_1d'] = close.pct_change(w2, fill_method=None).rank(axis=1, pct=True)

    # Market-state conditioners (same value across symbols)
    disp = rets.std(axis=1).rolling(cfg['dispersion_window'], min_periods=2).mean()
    sma = close.rolling(cfg['breadth_sma_window'],
                        min_periods=cfg['breadth_sma_window'] // 2).mean()
    breadth = (close > sma).mean(axis=1)
    out['cs_dispersion_1h'] = pd.DataFrame(
        np.tile(disp.values[:, None], (1, close.shape[1])), index=close.index, columns=close.columns)
    out['cs_breadth_sma'] = pd.DataFrame(
        np.tile(breadth.values[:, None], (1, close.shape[1])), index=close.index, columns=close.columns)

    # Cross-sectional funding z-score (relative crowding)
    fz = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    try:
        fr = load_data('funding_rates', columns=['timestamp', 'symbol', 'funding_rate'])
        if not fr.empty:
            fr['timestamp'] = pd.to_datetime(fr['timestamp'])
            fw = fr.pivot_table(index='timestamp', columns='symbol',
                                values='funding_rate', aggfunc='last').sort_index()
            fw = fw.reindex(close.index, method='ffill',
                            limit=cfg['funding_z_ffill_limit_bars'])
            mu = fw.mean(axis=1)
            sd = fw.std(axis=1).replace(0, np.nan)
            fz = fw.sub(mu, axis=0).div(sd, axis=0)
            fz = fz.reindex(columns=close.columns)
    except Exception as e:
        logging.warning(f"cs_funding_z unavailable: {e}")
    out['cs_funding_z'] = fz

    # Market-cap percentile rank (lagged 1 day, daily values applied intraday)
    mrank = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    try:
        mc = load_data('marketcap')
        if not mc.empty:
            mc['date'] = pd.to_datetime(mc['date'])
            mw = mc.pivot_table(index='date', columns='symbol', values='market_cap',
                                aggfunc='last').sort_index()
            mw = mw.reindex(pd.date_range(mw.index.min(), close.index.max().normalize(),
                                          freq='D')).ffill(limit=7).shift(1)
            ranks = mw.rank(axis=1, pct=True)
            mrank = ranks.reindex(close.index.normalize()).set_axis(close.index)
            mrank = mrank.reindex(columns=close.columns)
    except Exception as e:
        logging.warning(f"cs_mcap_rank unavailable: {e}")
    out['cs_mcap_rank'] = mrank

    # Cluster-relative value (residual cumsum z within own trailing cluster)
    crz = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    try:
        z = _cluster_relative_z(close.index, cfg['cluster_rel'])
        crz = z.reindex(columns=close.columns)
    except Exception as e:
        logging.warning(f"cs_cluster_rel_z unavailable: {e}")
    out['cs_cluster_rel_z'] = crz

    # Stack to long format
    frames = []
    for name, wide in out.items():
        s = wide.stack(future_stack=True).rename(name)
        frames.append(s)
    long_df = pd.concat(frames, axis=1).reset_index()
    long_df.columns = ['timestamp', 'symbol'] + CS_FEATURE_NAMES
    return long_df


# ============================================================================
# Main Feature Calculation
# ============================================================================

def calculate_all_features(df: pd.DataFrame, config: Dict, symbol: str,
                           residual_series: pd.Series,
                           loadings_df: pd.DataFrame,
                           factors_df: Optional[pd.DataFrame] = None,
                           intrabar_df: Optional[pd.DataFrame] = None,
                           mcap_series: Optional[pd.Series] = None,
                           xs_df: Optional[pd.DataFrame] = None,
                           leader_close: Optional[pd.Series] = None,
                           macro_daily: Optional[pd.DataFrame] = None,
                           macro_events: Optional[pd.DataFrame] = None,
                           dominance: Optional[pd.Series] = None) -> pd.DataFrame:
    feature_dfs = [df[['timestamp', 'symbol']].copy()]

    feature_dfs.append(pd.DataFrame(calculate_efficiency_features(df, config), index=df.index))
    feature_dfs.append(pd.DataFrame(calculate_volatility_regime_features(df, config), index=df.index))
    feature_dfs.append(pd.DataFrame(calculate_liquidity_features(df, config), index=df.index))
    feature_dfs.append(pd.DataFrame(calculate_range_features(df, config), index=df.index))
    feature_dfs.append(pd.DataFrame(calculate_microstructure_features(df, config), index=df.index))
    feature_dfs.append(pd.DataFrame(calculate_momentum_quality_features(df, config), index=df.index))
    feature_dfs.append(pd.DataFrame(calculate_statistical_features(df, config), index=df.index))
    feature_dfs.append(pd.DataFrame(calculate_price_features(df, config), index=df.index))
    feature_dfs.append(pd.DataFrame(calculate_volume_features(df, config), index=df.index))
    feature_dfs.append(pd.DataFrame(calculate_moving_averages(df, config), index=df.index))
    feature_dfs.append(pd.DataFrame(calculate_technical_indicators(df, config), index=df.index))
    feature_dfs.append(pd.DataFrame(calculate_time_features(df, config), index=df.index))
    feature_dfs.append(pd.DataFrame(
        calculate_market_context_features(df, factors_df, loadings_df, config), index=df.index))
    feature_dfs.append(pd.DataFrame(
        calculate_cap_features(df, mcap_series, config), index=df.index))
    feature_dfs.append(pd.DataFrame(
        calculate_residual_features(df, residual_series, config), index=df.index))
    feature_dfs.append(pd.DataFrame(
        calculate_ou_process_features(df, residual_series, config), index=df.index))
    feature_dfs.append(pd.DataFrame(
        calculate_factor_loading_features(df, loadings_df, config), index=df.index))
    feature_dfs.append(pd.DataFrame(
        calculate_seasonality_features(df, residual_series, config), index=df.index))
    feature_dfs.append(pd.DataFrame(
        calculate_leadlag_features(df, symbol, leader_close, config), index=df.index))
    feature_dfs.append(pd.DataFrame(
        calculate_event_features(df, macro_events, config), index=df.index))
    feature_dfs.append(pd.DataFrame(
        calculate_macro_state_features(df, macro_daily, dominance,
                                       macro_events, config), index=df.index))
    feature_dfs.append(pd.DataFrame(
        calculate_macrobeta_features(df, residual_series, macro_daily,
                                     dominance, macro_events, config),
        index=df.index))

    # Intra-bar features (precomputed on the 1m data, merged by timestamp)
    ib = pd.DataFrame(_nan_series(df, IB_FEATURE_NAMES), index=df.index)
    if intrabar_df is not None and not intrabar_df.empty:
        merged = df[['timestamp']].merge(intrabar_df, on='timestamp', how='left')
        merged.index = df.index
        for c in IB_FEATURE_NAMES:
            if c in merged.columns:
                ib[c] = merged[c]
    feature_dfs.append(ib)

    # Cross-sectional features (precomputed panel-wide, merged by timestamp)
    cs = pd.DataFrame(_nan_series(df, CS_FEATURE_NAMES), index=df.index)
    if xs_df is not None and not xs_df.empty:
        merged = df[['timestamp']].merge(xs_df, on='timestamp', how='left')
        merged.index = df.index
        for c in CS_FEATURE_NAMES:
            if c in merged.columns:
                cs[c] = merged[c]
    feature_dfs.append(cs)

    try:
        from risk_model.lib.features_futures import calculate_all_futures_features
        futures_features = calculate_all_futures_features(df, symbol, BASE_FREQUENCY)
        feature_dfs.append(futures_features)
    except Exception as e:
        logging.debug(f"Futures features not available for {symbol}: {e}")

    result = pd.concat(feature_dfs, axis=1)

    feature_cols = [c for c in result.columns if c not in ('timestamp', 'symbol')]
    result[feature_cols] = result[feature_cols].astype(np.float32)

    return result


# ============================================================================
# Per-Symbol Processing
# ============================================================================

def _load_mcap_series(symbol: str) -> Optional[pd.Series]:
    """Daily mcap, lagged 1 day and staleness-limited (matches factor model)."""
    try:
        mc = load_data('marketcap', filters={'symbol': symbol})
        if mc.empty:
            return None
        mc['date'] = pd.to_datetime(mc['date'])
        s = mc.set_index('date')['market_cap'].sort_index()
        full = pd.date_range(s.index.min(), pd.Timestamp.now().normalize(), freq='D')
        return s.reindex(full).ffill(limit=7).shift(1)
    except Exception:
        return None


_FACTORS_CACHE: Optional[pd.DataFrame] = None


def _load_factors() -> Optional[pd.DataFrame]:
    """Per-process cached risk_factors (small table)."""
    global _FACTORS_CACHE
    if _FACTORS_CACHE is None:
        f = load_data('risk_factors', columns=['timestamp', 'market_factor'])
        if f.empty:
            return None
        f['timestamp'] = pd.to_datetime(f['timestamp'])
        _FACTORS_CACHE = f.set_index('timestamp').sort_index()
    return _FACTORS_CACHE


_MACRO_CACHE: Optional[pd.DataFrame] = None
_EVENTS_CACHE: Optional[pd.DataFrame] = None
_DOMINANCE_CACHE: Optional[pd.Series] = None
_MACRO_LOADED = _EVENTS_LOADED = _DOMINANCE_LOADED = False


def _load_macro_daily() -> Optional[pd.DataFrame]:
    """macro_daily (availability-dated, etl/macro.py); None with a one-time
    warning when the table is absent - ev_/mx_/mb_ features degrade to NaN."""
    global _MACRO_CACHE, _MACRO_LOADED
    if not _MACRO_LOADED:
        _MACRO_LOADED = True
        try:
            df = load_data('macro_daily')
            if df is not None and not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                _MACRO_CACHE = df
            else:
                logging.warning("macro_daily unavailable - run etl/macro.py "
                                "(mx_/mb_ macro features will be NaN)")
        except Exception as e:
            logging.warning(f"macro_daily unavailable ({e})")
    return _MACRO_CACHE


def _load_macro_events() -> Optional[pd.DataFrame]:
    global _EVENTS_CACHE, _EVENTS_LOADED
    if not _EVENTS_LOADED:
        _EVENTS_LOADED = True
        try:
            df = load_data('macro_events')
            if df is not None and not df.empty:
                df['event_time_utc'] = pd.to_datetime(df['event_time_utc'])
                _EVENTS_CACHE = df
            else:
                logging.warning("macro_events unavailable - run etl/macro.py "
                                "(ev_/event features will be NaN)")
        except Exception as e:
            logging.warning(f"macro_events unavailable ({e})")
    return _EVENTS_CACHE


def _load_dominance() -> Optional[pd.Series]:
    """BTC dominance from the existing marketcap table (BTC mcap / total),
    daily, lagged one day like every other mcap-derived input."""
    global _DOMINANCE_CACHE, _DOMINANCE_LOADED
    if not _DOMINANCE_LOADED:
        _DOMINANCE_LOADED = True
        try:
            mc = load_data('marketcap')
            if mc is not None and not mc.empty:
                mc['date'] = pd.to_datetime(mc['date'])
                wide = mc.pivot_table(index='date', columns='symbol',
                                      values='market_cap', aggfunc='last')
                if 'BTC' in wide.columns:
                    dom = wide['BTC'] / wide.sum(axis=1)
                    full = pd.date_range(dom.index.min(), dom.index.max(),
                                         freq='D')
                    _DOMINANCE_CACHE = dom.reindex(full).ffill(limit=7).shift(1)
        except Exception as e:
            logging.warning(f"dominance unavailable ({e})")
    return _DOMINANCE_CACHE


_LEADER_CACHE: Optional[pd.Series] = None
_LEADER_LOADED = False


def _load_leader_close() -> Optional[pd.Series]:
    """Per-process cached leader close series (features.lead_lag.leader_symbol).
    None (with a one-time warning) when the leader has no price history."""
    global _LEADER_CACHE, _LEADER_LOADED
    if not _LEADER_LOADED:
        _LEADER_LOADED = True
        sym = FEATURE_CONFIG['lead_lag']['leader_symbol']
        px = load_data('prices', filters={'symbol': sym},
                       columns=['timestamp', 'close'])
        if px is None or px.empty:
            logging.warning(f"lead_lag: no prices for leader '{sym}' - "
                            "ll_ features will be NaN")
        else:
            px['timestamp'] = pd.to_datetime(px['timestamp'])
            _LEADER_CACHE = px.set_index('timestamp')['close'].sort_index()
    return _LEADER_CACHE


def process_symbol(symbol: str, xs_df: Optional[pd.DataFrame] = None) -> Optional[pd.DataFrame]:
    """Compute the full feature panel for one symbol."""
    try:
        prices_df = load_data('prices', filters={'symbol': symbol})
        if prices_df.empty:
            return None
        prices_df['timestamp'] = pd.to_datetime(prices_df['timestamp'])
        prices_df = prices_df.sort_values('timestamp').reset_index(drop=True)

        residuals_df = load_data('residual_returns', filters={'symbol': symbol},
                                 columns=['timestamp', 'symbol', 'residual_return'])
        if not residuals_df.empty:
            residuals_df['timestamp'] = pd.to_datetime(residuals_df['timestamp'])
            merged = prices_df[['timestamp']].merge(
                residuals_df[['timestamp', 'residual_return']], on='timestamp', how='left')
            residual_series = merged['residual_return']
            residual_series.index = prices_df.index
        else:
            residual_series = pd.Series(np.nan, index=prices_df.index)

        loadings_df = load_data('factor_loadings', filters={'symbol': symbol})
        factors_df = _load_factors()
        mcap_series = _load_mcap_series(symbol)

        intrabar_df = None
        if FEATURE_CONFIG['intrabar'].get('enabled', True):
            try:
                raw_1m = load_data('prices_raw', filters={'symbol': symbol},
                                   columns=['timestamp', 'close', 'volume'])
                if not raw_1m.empty:
                    raw_1m['timestamp'] = pd.to_datetime(raw_1m['timestamp'])
                    intrabar_df = compute_intrabar_features(raw_1m, FEATURE_CONFIG)
            except Exception as e:
                logging.warning(f"{symbol}: intrabar features failed: {e}")

        return calculate_all_features(prices_df, FEATURE_CONFIG, symbol,
                                      residual_series, loadings_df,
                                      factors_df=factors_df,
                                      intrabar_df=intrabar_df,
                                      mcap_series=mcap_series,
                                      xs_df=xs_df,
                                      leader_close=_load_leader_close(),
                                      macro_daily=_load_macro_daily(),
                                      macro_events=_load_macro_events(),
                                      dominance=_load_dominance())
    except Exception as e:
        logging.error(f"Error processing {symbol}: {e}")
        return None


def _process_and_save(args: Tuple) -> Tuple[str, int, Dict[str, float]]:
    """Worker: compute features for a symbol, append to the features table.

    Returns (symbol, n_rows, per-feature NaN share) for the coverage report.
    """
    symbol, xs_df = args
    features_df = process_symbol(symbol, xs_df=xs_df)
    if features_df is None or features_df.empty:
        return symbol, 0, {}
    nan_share = features_df.drop(columns=['timestamp', 'symbol']).isna().mean().to_dict()
    save_data('features', features_df, mode='append',
              datetime_columns=['timestamp'], use_file_lock=True)
    return symbol, len(features_df), nan_share


def main():
    print("=" * 60)
    print("Feature Generation")
    print(f"Base frequency: {BASE_FREQUENCY} | Workers: {MAX_WORKERS}")
    print("=" * 60)

    symbols = get_table_symbols('prices', use_universe_cache=False)
    if not symbols:
        print("No symbols found in prices table")
        return
    print(f"Processing {len(symbols)} symbols")

    # Per-symbol loads are full scans without these (~5GB read/symbol on prices_raw)
    for table in ('prices', 'prices_raw', 'residual_returns', 'futures_metrics'):
        ensure_symbol_index(table)

    print("Computing cross-sectional (panel) features...")
    xs_long = compute_cross_sectional_features()
    if xs_long is not None:
        xs_slices = {sym: g.drop(columns=['symbol']) for sym, g in xs_long.groupby('symbol')}
        print(f"  {len(CS_FEATURE_NAMES)} cs_ features for {len(xs_slices)} symbols")
    else:
        xs_slices = {}
        print("  prices unavailable - cs_ features will be NaN")
    del xs_long  # slices are copies; the 2.4GB long panel is dead weight from here

    delete_table('features')

    work_items = [(sym, xs_slices.get(sym)) for sym in symbols]
    results = parallel_map(_process_and_save, work_items, max_workers=MAX_WORKERS,
                           desc="Features", show_progress=True)

    total_rows = sum(r[1] for r in results if r)
    done = sum(1 for r in results if r and r[1] > 0)
    print(f"\nFeature generation complete: {done}/{len(symbols)} symbols, "
          f"{total_rows:,} rows")

    # NaN-coverage report: features that are mostly missing are dead weight
    # on the FDR gate in signal research
    nan_frames = [pd.Series(r[2]) for r in results if r and r[2]]
    if nan_frames:
        avg_nan = pd.concat(nan_frames, axis=1).mean(axis=1).sort_values(ascending=False)
        worst = avg_nan[avg_nan > 0.25]
        if not worst.empty:
            print(f"\nFeatures with >25% NaN (check data dependencies):")
            for name, share in worst.head(15).items():
                print(f"  {name}: {share * 100:.0f}%")


if __name__ == '__main__':
    main()
