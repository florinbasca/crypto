"""
Futures-Derived Features for Statistical Arbitrage.

Computes features from:
1. Funding rates (carry, crowding, funding-flip age)
2. Open interest (position buildup, liquidation flush)
3. Long/short ratios (sentiment, contrarian)

TIMING: panel stamps are bar-END. All series are resampled with
label='right', closed='right' so the value at stamp t comes from source
records stamped <= t - causally correct WITHOUT an extra lag. (The previous
shift(1) was compensating for a left-labeled resample leak; both the leak
and the compensating lag are gone.)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
from typing import Dict

import numpy as np
import pandas as pd

from dbutil import load_data
from config import get_frequency_config, config as global_config

# Library module: no basicConfig (it clobbers the entry point's logging config).

FUT_CFG = global_config['features'].get('futures', {})

FR_FEATURE_NAMES = [
    'fr_rate', 'fr_rate_zscore', 'fr_rate_ma_7d', 'fr_rate_change',
    'fr_cumulative_24h', 'fr_annualized', 'fr_extreme_high', 'fr_extreme_low',
    'fr_flip_age',
]
OI_FEATURE_NAMES = [
    'oi_value', 'oi_value_zscore', 'oi_change_pct', 'oi_change_zscore',
    'oi_relative', 'oi_price_divergence', 'oi_flush',
]
POS_FEATURE_NAMES = [
    'pos_retail_ls', 'pos_retail_ls_zscore', 'pos_toptrader_ls',
    'pos_toptrader_ls_zscore', 'pos_retail_vs_smart', 'pos_taker_ratio',
    'pos_taker_zscore',
]
def _nan_features(df: pd.DataFrame, names) -> Dict[str, pd.Series]:
    return {n: pd.Series([np.nan] * len(df), index=df.index) for n in names}


def _align_to_panel(df: pd.DataFrame, series: pd.Series, frequency: str,
                    how: str = 'last') -> pd.Series:
    """Right-labeled resample + merge onto the panel's bar-end timestamps."""
    res = series.resample(frequency, label='right', closed='right')
    res = res.last() if how == 'last' else res.sum()
    res = res.ffill()

    ts = pd.to_datetime(df['timestamp'])
    aligned = res.reindex(ts).values
    return pd.Series(aligned, index=df.index)


def calculate_funding_rate_features(df: pd.DataFrame, symbol: str,
                                    frequency: str) -> Dict[str, pd.Series]:
    features = _nan_features(df, FR_FEATURE_NAMES)

    try:
        funding_df = load_data('funding_rates', filters={'symbol': symbol})
        if funding_df is None or funding_df.empty:
            return features

        funding_df['timestamp'] = pd.to_datetime(funding_df['timestamp'])
        funding_df = funding_df.sort_values('timestamp').set_index('timestamp')

        bars_per_day = get_frequency_config(frequency)['bars_per_day']
        fr = _align_to_panel(df, funding_df['funding_rate'], frequency)

        features['fr_rate'] = fr

        window_7d = 7 * bars_per_day
        fr_mean = fr.rolling(window=window_7d, min_periods=bars_per_day).mean()
        fr_std = fr.rolling(window=window_7d, min_periods=bars_per_day).std()
        features['fr_rate_zscore'] = (fr - fr_mean) / (fr_std + 1e-10)
        features['fr_rate_ma_7d'] = fr_mean
        features['fr_rate_change'] = fr - fr.shift(bars_per_day)

        # Funding pays ~3x/day; the ffilled series repeats values, so use a
        # 24h window mean x 3 as the cumulative-day approximation
        features['fr_cumulative_24h'] = fr.rolling(window=bars_per_day, min_periods=1).mean() * 3
        features['fr_annualized'] = fr * 3 * 365
        features['fr_extreme_high'] = (fr > 0.001).astype(int)
        features['fr_extreme_low'] = (fr < -0.0001).astype(int)

        # Bars since the funding rate last flipped sign (regime age)
        sign = np.sign(fr).replace(0, np.nan).ffill()
        flips = (sign != sign.shift(1)) & sign.notna() & sign.shift(1).notna()
        groups = flips.cumsum()
        age = groups.groupby(groups).cumcount().astype(float)
        age[groups == 0] = np.nan
        cap = FUT_CFG.get('flip_age_cap_bars', 1008)
        features['fr_flip_age'] = age.clip(upper=cap)

    except Exception as e:
        logging.error(f"Error calculating funding rate features for {symbol}: {e}")

    return features


def calculate_open_interest_features(df: pd.DataFrame, symbol: str,
                                     frequency: str) -> Dict[str, pd.Series]:
    features = _nan_features(df, OI_FEATURE_NAMES)

    try:
        metrics_df = load_data('futures_metrics', filters={'symbol': symbol})
        if metrics_df is None or metrics_df.empty:
            return features

        metrics_df['timestamp'] = pd.to_datetime(metrics_df['timestamp'])
        metrics_df = metrics_df.sort_values('timestamp').set_index('timestamp')

        bars_per_day = get_frequency_config(frequency)['bars_per_day']
        window_7d = 7 * bars_per_day

        oi = _align_to_panel(df, metrics_df['open_interest_value'], frequency)

        features['oi_value'] = np.log1p(oi)

        oi_mean = oi.rolling(window=window_7d, min_periods=bars_per_day).mean()
        oi_std = oi.rolling(window=window_7d, min_periods=bars_per_day).std()
        features['oi_value_zscore'] = (oi - oi_mean) / (oi_std + 1e-10)

        oi_change = oi.pct_change(fill_method=None)
        features['oi_change_pct'] = oi_change

        ch_mean = oi_change.rolling(window=window_7d, min_periods=bars_per_day).mean()
        ch_std = oi_change.rolling(window=window_7d, min_periods=bars_per_day).std()
        oi_change_z = (oi_change - ch_mean) / (ch_std + 1e-10)
        features['oi_change_zscore'] = oi_change_z

        features['oi_relative'] = oi / (oi_mean + 1e-10)

        if 'close' in df.columns:
            price_change = df['close'].pct_change(fill_method=None)
            features['oi_price_divergence'] = (
                (np.sign(oi_change) != np.sign(price_change)).astype(int) * np.sign(oi_change)
            )

            # Liquidation flush: OI dropping fast WHILE price moves hard.
            # Forced deleveraging - a strongly mean-reverting state.
            zw = FUT_CFG.get('flush_z_window_bars', 144)
            ret_std = price_change.rolling(zw, min_periods=zw // 4).std()
            ret_z = price_change.abs() / (ret_std + 1e-10)
            features['oi_flush'] = (-oi_change_z).clip(lower=0) * ret_z

    except Exception as e:
        logging.error(f"Error calculating OI features for {symbol}: {e}")

    return features


def calculate_positioning_features(df: pd.DataFrame, symbol: str,
                                   frequency: str) -> Dict[str, pd.Series]:
    features = _nan_features(df, POS_FEATURE_NAMES)

    try:
        metrics_df = load_data('futures_metrics', filters={'symbol': symbol})
        if metrics_df is None or metrics_df.empty:
            return features

        metrics_df['timestamp'] = pd.to_datetime(metrics_df['timestamp'])
        metrics_df = metrics_df.sort_values('timestamp').set_index('timestamp')

        bars_per_day = get_frequency_config(frequency)['bars_per_day']
        window_7d = 7 * bars_per_day

        def z(s):
            mu = s.rolling(window=window_7d, min_periods=bars_per_day).mean()
            sd = s.rolling(window=window_7d, min_periods=bars_per_day).std()
            return (s - mu) / (sd + 1e-10)

        retail = toptrader = None
        if 'retail_ls_ratio' in metrics_df.columns:
            retail = _align_to_panel(df, metrics_df['retail_ls_ratio'], frequency)
            features['pos_retail_ls'] = retail
            features['pos_retail_ls_zscore'] = z(retail)

        if 'toptrader_ls_positions' in metrics_df.columns:
            toptrader = _align_to_panel(df, metrics_df['toptrader_ls_positions'], frequency)
            features['pos_toptrader_ls'] = toptrader
            features['pos_toptrader_ls_zscore'] = z(toptrader)

        if retail is not None and toptrader is not None:
            features['pos_retail_vs_smart'] = retail - toptrader

        if 'taker_buy_sell_ratio' in metrics_df.columns:
            taker = _align_to_panel(df, metrics_df['taker_buy_sell_ratio'], frequency)
            features['pos_taker_ratio'] = taker
            features['pos_taker_zscore'] = z(taker)

    except Exception as e:
        logging.error(f"Error calculating positioning features for {symbol}: {e}")

    return features


def calculate_all_futures_features(df: pd.DataFrame, symbol: str,
                                   frequency: str) -> pd.DataFrame:
    fr = calculate_funding_rate_features(df, symbol, frequency)
    oi = calculate_open_interest_features(df, symbol, frequency)
    pos = calculate_positioning_features(df, symbol, frequency)

    out = pd.DataFrame({**fr, **oi, **pos}, index=df.index)

    # Crowded AND levered: the liquidation-cascade precondition
    out['fr_oi_crowding'] = out['fr_rate_zscore'] * out['oi_value_zscore']

    return out
