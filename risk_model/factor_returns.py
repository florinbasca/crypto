"""
Factor Returns: Market (equal-weight) + Size / Momentum / Vol (rank-weighted spreads).

All factors are TRADABLE PORTFOLIO RETURNS in return units, computed per
base-frequency bar over the current candidate universe:

- market_factor[t]: equal-weight mean return of the universe names for the bar
  ending at t. (Bar-end timestamp convention: factor[t] covers (t-1bar, t].)

- size / momentum / vol factors: rank-weighted long/short spreads. Daily weights
  are proportional to the centered rank of a per-name CHARACTERISTIC among the
  universe members, long one tail / short the other, each side scaled to gross 1
  (every member used, no tercile boundary churn; dollar-neutral by construction).
  The characteristic is always computed from data STRICTLY BEFORE the bar's date
  (no look-ahead):
    * size     - lagged market cap            -> long small, short big
    * momentum - trailing cumulative return    -> long winners, short losers
                 (skips the most-recent skip_days)
    * vol      - trailing realized volatility   -> long low-vol, short high-vol
  Momentum/vol signs are arbitrary for neutralization (only the beta magnitude
  matters); the optimizer constrains beta to each factor near zero rather than
  trading it as standalone alpha.

Output: risk_factors [timestamp, market_factor, size_factor, momentum_factor,
  vol_factor, n_members, n_size, n_momentum, n_vol]

Timing convention (critical, see residual_returns.py): factor[t] is the
systematic return of the bar ENDING at t, exactly like asset returns r[t].
Forward-horizon hedging must therefore use factor[t+1..t+p] for the asset
return over (t, t+p] - never factor[t..t+p-1].
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging

import numpy as np
import pandas as pd

from config import config as global_config, get, get_frequency_config
from dbutil import load_data, save_data

logging.basicConfig(
    level=logging.INFO,
    format=global_config['logging']['format'],
    datefmt=global_config['logging']['datefmt'],
)

BASE_FREQ = global_config['base_frequency']
MARKET_MIN_MEMBERS = get('risk_model.market_min_members', 10)
MCAP_LAG_DAYS = get('risk_model.size.mcap_lag_days', 1)
MCAP_MAX_STALENESS_DAYS = get('risk_model.size.mcap_max_staleness_days', 7)
MOM_LOOKBACK_DAYS = get('risk_model.momentum.lookback_days', 30)
MOM_SKIP_DAYS = get('risk_model.momentum.skip_days', 2)
VOL_LOOKBACK_DAYS = get('risk_model.vol.lookback_days', 30)


def load_returns_wide() -> pd.DataFrame:
    """Load base-frequency close prices and compute single-bar returns (wide)."""
    px = load_data('prices', columns=['timestamp', 'symbol', 'close'])
    if px.empty:
        raise RuntimeError("prices table is empty - run etl/prices.py first")

    px['timestamp'] = pd.to_datetime(px['timestamp'])
    if getattr(px['timestamp'].dt, 'tz', None) is not None:
        px['timestamp'] = px['timestamp'].dt.tz_localize(None)

    close_wide = px.pivot_table(index='timestamp', columns='symbol', values='close',
                                aggfunc='last').sort_index()
    # Regular grid so gaps stay NaN instead of silently compounding across them
    full_index = pd.date_range(close_wide.index.min(), close_wide.index.max(), freq=BASE_FREQ)
    close_wide = close_wide.reindex(full_index)
    returns = close_wide.pct_change(fill_method=None)
    returns.index.name = 'timestamp'
    return returns


def load_membership_by_month() -> dict:
    """Compatibility wrapper: every month maps to the full current universe."""
    candidates = load_data('universe', columns=['symbol'])
    if candidates.empty:
        raise RuntimeError("universe is empty - run etl/universe.py first")
    members = set(candidates['symbol'])
    return {pd.Timestamp('1900-01-01'): members}


def membership_for_timestamp_index(index: pd.DatetimeIndex, by_month: dict) -> pd.Series:
    """Series mapping each month-start present in the index to its member set."""
    months_sorted = sorted(by_month.keys())
    result = {}
    for period in index.to_period('M').unique():
        month_start = period.to_timestamp()
        valid = [m for m in months_sorted if m <= month_start]
        result[period] = by_month[valid[-1]] if valid else set()
    return result


def load_mcap_daily() -> pd.DataFrame:
    """Daily market cap (wide, date x symbol), lagged and staleness-limited."""
    mc = load_data('marketcap')
    if mc.empty:
        logging.warning("marketcap table empty - size factor will be all-NaN")
        return pd.DataFrame()

    mc['date'] = pd.to_datetime(mc['date'])
    wide = mc.pivot_table(index='date', columns='symbol', values='market_cap',
                          aggfunc='last').sort_index()
    full_dates = pd.date_range(wide.index.min(), wide.index.max(), freq='D')
    wide = wide.reindex(full_dates).ffill(limit=MCAP_MAX_STALENESS_DAYS)
    # Lag so day D uses mcap known strictly before D
    wide = wide.shift(MCAP_LAG_DAYS)
    wide.index.name = 'date'
    return wide


def _daily_log_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """Per-day summed log returns (date x symbol) from bar returns."""
    log_ret = np.log1p(returns).replace([np.inf, -np.inf], np.nan)
    return log_ret.groupby(returns.index.normalize()).sum(min_count=1)


def compute_momentum_char(returns: pd.DataFrame) -> pd.DataFrame:
    """Trailing cumulative log return per name, STRICTLY before each date.

    Skips the most-recent MOM_SKIP_DAYS days (>=1 -> causal) to avoid
    short-term-reversal contamination. Index = date, columns = symbol.
    """
    daily = _daily_log_returns(returns)
    min_p = max(2, MOM_LOOKBACK_DAYS // 2)
    mom = daily.rolling(MOM_LOOKBACK_DAYS, min_periods=min_p).sum()
    return mom.shift(max(MOM_SKIP_DAYS, 1))


def compute_vol_char(returns: pd.DataFrame) -> pd.DataFrame:
    """Trailing realized volatility per name, STRICTLY before each date.

    sqrt of the trailing realized variance (sum of squared bar returns),
    shifted one day so the bar's own day is excluded. Index = date, cols = symbol.
    """
    daily_var = (returns ** 2).groupby(returns.index.normalize()).sum(min_count=1)
    min_p = max(2, VOL_LOOKBACK_DAYS // 2)
    rv = daily_var.rolling(VOL_LOOKBACK_DAYS, min_periods=min_p).sum()
    return np.sqrt(rv.shift(1))


def _daily_rank_weights(char_wide: pd.DataFrame, unique_dates: pd.DatetimeIndex,
                        membership: dict, low_is_long: bool = True) -> dict:
    """Daily rank weights from a per-name characteristic (date -> Series).

    Centered ordinal ranks (method='first': tie-proof), each side scaled to gross
    1. With low_is_long the low-characteristic tail gets positive (long) weight.
    A spread needs at least one name on each side; the pos/neg guard drops days
    that don't have it. Weights from char_wide are assumed already causal.
    """
    weight_map = {}
    if char_wide is None or char_wide.empty:
        return weight_map
    for d in unique_dates:
        members = membership.get(d.to_period('M'), set())
        if d not in char_wide.index or not members:
            continue
        row = char_wide.loc[d]
        row = row[row.index.isin(members)].dropna()
        row = row[np.isfinite(row)]
        if len(row) < 2:
            continue
        ranks = row.rank(method='first')
        centered = ranks - ranks.mean()
        w = -centered if low_is_long else centered
        pos_sum = w[w > 0].sum()
        neg_sum = -w[w < 0].sum()
        if pos_sum <= 0 or neg_sum <= 0:
            continue
        w[w > 0] = w[w > 0] / pos_sum
        w[w < 0] = w[w < 0] / neg_sum
        weight_map[d] = w
    return weight_map


def _factor_from_weights(returns: pd.DataFrame, dates: pd.DatetimeIndex,
                         weight_map: dict):
    """Apply daily long/short weights intraday -> (factor_return, n_members).

    Each side is renormalized per bar over names with a valid return so missing
    bars cannot break dollar neutrality. n counts the names with a valid return
    in EACH bar (diagnostic column, per-bar resolution).
    """
    factor = pd.Series(np.nan, index=returns.index)
    n = pd.Series(0, index=returns.index)
    if not weight_map:
        return factor, n
    date_values = dates.values
    for d, w in weight_map.items():
        in_day = date_values == d.to_datetime64()
        cols = [c for c in returns.columns if c in w.index]
        if not cols:
            continue
        r_day = returns.loc[in_day, cols]
        wv = w[cols]
        w_pos = wv.clip(lower=0)
        w_neg = (-wv).clip(lower=0)
        avail = r_day.notna()
        pos_norm = avail.mul(w_pos, axis=1).sum(axis=1)
        neg_norm = avail.mul(w_neg, axis=1).sum(axis=1)
        long_ret = r_day.mul(w_pos, axis=1).sum(axis=1) / pos_norm.replace(0, np.nan)
        short_ret = r_day.mul(w_neg, axis=1).sum(axis=1) / neg_norm.replace(0, np.nan)
        factor.loc[in_day] = (long_ret - short_ret).values
        n.loc[in_day] = avail.sum(axis=1).values
    return factor, n


def compute_factor_returns(returns: pd.DataFrame,
                           membership: dict,
                           mcap: pd.DataFrame) -> pd.DataFrame:
    """Compute market + characteristic (size/momentum/vol) factor returns per bar."""
    periods = returns.index.to_period('M')
    dates = returns.index.normalize()
    unique_dates = pd.DatetimeIndex(dates.unique())

    # Market factor: equal-weight mean member return per bar.
    member_mask = pd.DataFrame(False, index=returns.index, columns=returns.columns)
    for period, members in membership.items():
        in_month = periods == period
        cols = [c for c in returns.columns if c in members]
        if cols:
            member_mask.loc[in_month, cols] = True
    member_returns = returns.where(member_mask)
    n_members = member_returns.notna().sum(axis=1)
    market = member_returns.mean(axis=1).where(n_members >= MARKET_MIN_MEMBERS)

    out = pd.DataFrame({
        'timestamp': returns.index,
        'market_factor': market.values,
        'n_members': n_members.values,
    })

    # Characteristic factors: rank-weighted spreads, daily weights applied
    # intraday. (name, characteristic[date x symbol], low_is_long).
    char_specs = [
        ('size', mcap if not mcap.empty else None, True),   # long small mcap
        ('momentum', compute_momentum_char(returns), False),  # long winners
        ('vol', compute_vol_char(returns), True),             # long low vol
    ]
    for name, char_wide, low_is_long in char_specs:
        weight_map = _daily_rank_weights(char_wide, unique_dates, membership, low_is_long)
        factor, n = _factor_from_weights(returns, dates, weight_map)
        out[f'{name}_factor'] = factor.values
        out[f'n_{name}'] = n.values

    return out


def main():
    logging.info("Loading returns...")
    returns = load_returns_wide()
    logging.info(f"Returns panel: {returns.shape[0]:,} bars x {returns.shape[1]} symbols")

    membership = load_membership_by_month()
    by_period = membership_for_timestamp_index(returns.index, membership)

    mcap = load_mcap_daily()
    logging.info(f"Market cap coverage: {0 if mcap.empty else mcap.notna().any(axis=0).sum()} symbols")

    factors = compute_factor_returns(returns, by_period, mcap)

    n_total = len(factors)
    valid_mkt = factors['market_factor'].notna().sum()
    for col in [c for c in factors.columns if c.endswith('_factor')]:
        logging.info(f"{col}: {factors[col].notna().sum():,}/{n_total:,} valid bars, "
                     f"per-bar std {factors[col].std():.5f}")
    if 'size_factor' in factors and factors['size_factor'].notna().sum() < 0.5 * valid_mkt:
        logging.warning("size_factor coverage is low - check marketcap table / CoinGecko mappings")

    save_data('risk_factors', factors, mode='overwrite', datetime_columns=['timestamp'])
    logging.info(f"Saved {len(factors):,} rows to risk_factors")


if __name__ == '__main__':
    main()
