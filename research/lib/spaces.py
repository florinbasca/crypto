"""
Cross-sectional statistical-arbitrage SPACES.

A "space" is one economic hypothesis: relative cross-sectional displacement in
space S predicts idiosyncratic (residual) returns. evaluate.py z-scores the raw
value cross-sectionally; walk_forward selects which spaces have live edge each
training window. Outcome-agnostic - no returns are inspected here.

Add a space = add one `_S(...)` line. Sign is resolved from in-sample IC, so the
formula is direction-neutral.
"""
from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd

from config import get


@dataclass(frozen=True)
class SpaceDef:
    name: str
    columns: Tuple[str, ...]      # feature columns read (column-projected loading)
    theme: str                    # economic family (drives walk_forward family cap)
    rationale: str
    op: str = 'direct'            # direct | spread | ratio | diff
    lag: int = 0                  # for op='diff' (in BASE bars, not screening bars)
    halflife: Optional[float] = None  # per-space EWM smoothing; None -> global default

    @property
    def signal_type(self) -> str:
        return f'space_{self.op}'

    @property
    def category(self) -> str:
        return self.theme

    @property
    def direction(self) -> int:
        return 1


def _S(name, col, theme, rationale, op='direct', lag=0, col2=None, halflife=None):
    cols = (col,) if col2 is None else (col, col2)
    return SpaceDef(name=name, columns=cols, theme=theme, rationale=rationale,
                    op=op, lag=lag, halflife=halflife)


# =============================================================================
# The space library. Each line is one hypothesis. Grouped by economic theme.
# =============================================================================
SPACES = [
    # --- residual mean-reversion: the idiosyncratic-displacement core of statarb
    _S('residual_displacement', 'res_zscore', 'residual_reversion',
       'idiosyncratic price displaced from its mean reverts'),
    _S('ou_equilibrium', 'ou_zscore', 'residual_reversion',
       'Ornstein-Uhlenbeck distance from equilibrium reverts'),
    _S('ou_expected_reversion', 'ou_signal_scaled', 'residual_reversion',
       'OU model expected reversion, scaled by speed/confidence'),
    _S('residual_accel', 'res_zscore_accel', 'residual_reversion',
       'acceleration of residual displacement (over-extension)'),
    _S('ou_reversion_speed', 'ou_halflife_short', 'residual_reversion',
       'fast mean-reversion half-life = stronger reversion edge'),
    _S('xs_return_reversal', 'cs_ret_rank_1d', 'residual_reversion',
       'cross-sectional 1d return rank reverses (classic statarb)'),

    # --- efficiency: trending vs mean-reverting regime (the predictability space)
    _S('variance_ratio', 'ef_variance_ratio', 'efficiency',
       'deviation from random walk: <1 mean-reverts, >1 trends'),
    _S('short_autocorr', 'ef_autocorr_lag1', 'efficiency',
       'lag-1 return autocorrelation: over-reaction corrects'),
    _S('reversal_tendency', 'ef_reversal_tendency', 'efficiency',
       'measured tendency to reverse recent moves'),
    _S('reversion_momentum', 'ef_reversion_momentum_medium', 'efficiency',
       'medium-horizon reversion momentum'),

    # --- liquidity / microstructure cost: uncontested, slow, high information ratio
    _S('illiquidity', 'lq_amihud_illiquidity', 'liquidity',
       'Amihud illiquidity: harder-to-trade names carry mispricing'),
    _S('illiquidity_asym', 'lq_amihud_asym', 'liquidity',
       'asymmetric (up vs down) price impact'),
    _S('volume_concentration', 'ib_volume_herf_1h', 'liquidity',
       'intrabar volume Herfindahl: concentrated trading = information'),
    _S('intrabar_autocorr', 'ib_autocorr_1m', 'liquidity',
       'minute-return autocorrelation (microstructure persistence space)'),
    _S('bid_ask_spread', 'rb_cs_spread', 'liquidity',
       'Corwin-Schultz spread proxy: wider spread = costlier/mispriced'),
    _S('vwap_deviation', 'ib_vwap_dev', 'liquidity',
       'price vs intrabar VWAP: intraday pressure'),
    _S('zero_volume_share', 'ib_zero_vol_share', 'liquidity',
       'share of zero-volume bars: thin trading'),

    # --- order flow
    _S('buy_pressure', 'ms_buy_ratio', 'order_flow',
       'taker buy share: aggressive buying continues then exhausts'),
    _S('signed_volume', 'vl_signed_volume_ratio', 'order_flow',
       'order-flow imbalance (signed volume)'),
    _S('trade_intensity', 'ms_trade_intensity_zscore', 'order_flow',
       'abnormal trade count: attention/activity shock'),
    _S('avg_trade_size', 'ms_avg_trade_size_zscore', 'order_flow',
       'large average trade size: informed/whale flow'),
    _S('taker_imbalance', 'pos_taker_ratio', 'order_flow',
       'taker buy/sell imbalance'),
    _S('taker_imbalance_z', 'pos_taker_zscore', 'order_flow',
       'abnormal taker imbalance'),

    # --- positioning / crowding (contrarian)
    _S('retail_crowding', 'pos_retail_ls_zscore', 'positioning',
       'crowded retail long/short positioning unwinds'),
    _S('retail_vs_smart', 'pos_retail_vs_smart', 'positioning',
       'retail vs smart-money positioning divergence'),
    _S('toptrader_positioning', 'pos_toptrader_ls_zscore', 'positioning',
       'top-trader long/short positioning'),

    # --- funding (crypto-native carry / crowding)
    _S('funding_z', 'fr_rate_zscore', 'funding',
       'extreme funding = crowded carry, contrarian'),
    _S('funding_change', 'fr_rate_change', 'funding',
       'funding momentum / regime shift'),
    _S('funding_oi_crowding', 'fr_oi_crowding', 'funding',
       'funding x open-interest crowding stress'),
    _S('funding_flip_age', 'fr_flip_age', 'funding',
       'bars since funding flipped sign: regime maturity'),

    # --- open interest
    _S('oi_price_divergence', 'oi_price_divergence', 'open_interest',
       'OI rising without price = positioning that must unwind'),
    _S('oi_change', 'oi_change_zscore', 'open_interest',
       'abnormal OI change'),
    _S('oi_flush', 'oi_flush', 'open_interest',
       'OI flush / forced deleveraging'),
    _S('oi_relative', 'oi_relative', 'open_interest',
       'OI level relative to its own history'),

    # --- activity / volume
    _S('relative_volume', 'cs_rel_volume', 'volume',
       'cross-sectional relative volume surge'),
    _S('dollar_volume_trend', 'ms_dollar_volume_momentum', 'volume',
       'trend in dollar volume (the volume-space hypothesis)'),

    # --- volatility structure
    _S('semivol_asymmetry', 'vr_semivol_ratio_short', 'vol_structure',
       'downside vs upside realized-vol asymmetry'),
    _S('vol_breakout', 'vr_vol_breakout_short', 'vol_structure',
       'volatility breakout'),
    _S('vol_regime', 'vr_volatility_regime_short', 'vol_structure',
       'volatility regime level'),
    _S('realized_vol', 'ib_rv_1h', 'vol_structure',
       'realized volatility level (low-vol anomaly)'),
    _S('overnight_vol_ratio', 'ib_rv_cc_ratio', 'vol_structure',
       'intrabar vs close-close vol: jump / microstructure noise'),
    _S('return_skew', 'st_skew', 'vol_structure',
       'return skewness: lottery / crash-risk premium'),
    _S('return_kurtosis', 'st_kurtosis', 'vol_structure',
       'return kurtosis: tail-risk premium'),

    # --- cross-sectional / basket-relative context
    _S('cluster_relative_value', 'cs_cluster_rel_z', 'cross_sectional',
       'value relative to its correlation cluster (the statarb basket)'),
    _S('dispersion', 'cs_dispersion_1h', 'cross_sectional',
       'cross-sectional return dispersion regime'),

    # --- market-structure / lead-lag
    _S('lag_response', 'mk_lag_response_gap', 'market_structure',
       'lagged response to market moves: slow names catch up'),
    _S('beta_drift', 'mk_beta_drift', 'market_structure',
       'instability of market beta'),

    # --- price-trend (crowded but real; kept few)
    _S('vol_adjusted_momentum', 'mq_vol_adjusted_momentum', 'momentum',
       'risk-adjusted residual momentum'),
    _S('residual_momentum', 'res_zscore_momentum_8', 'momentum',
       'persistence of residual displacement'),
    _S('adx_trend', 'dm_adx', 'momentum',
       'ADX trend strength'),
    _S('price_vs_sma', 'ma_price_vs_sma_medium', 'momentum',
       'price distance from medium SMA'),
    _S('rsi', 'rs_rsi', 'momentum',
       'RSI overbought/oversold'),
    _S('bollinger_position', 'bb_position', 'momentum',
       'position within Bollinger band'),

    # --- fundamental (very uncontested in crypto)
    _S('supply_inflation', 'cap_supply_inflation', 'fundamental',
       'token supply inflation / unlock pressure'),
    _S('turnover', 'cap_turnover', 'fundamental',
       'volume / market-cap turnover'),
    _S('log_mcap', 'cap_log_mcap', 'fundamental',
       'log market cap: size premium'),
    _S('turnover_per_size', 'cap_turnover', 'fundamental',
       'turnover per unit (log) market cap', op='ratio', col2='cap_log_mcap'),

    # --- residual dynamics: accumulation, regime, term structure
    # Two conventions for the spread/ratio/diff ops below:
    #   * Build spread/ratio legs from RAW LEVEL columns, never from already
    #     z-scored ones (fr_rate_zscore, cs_funding_z, ...): evaluate.py applies
    #     the single cross-sectional z-score itself, so z(z_ts - z_xs) is muddled.
    #   * op='diff' lag is in BASE bars (full-resolution grid), not screening
    #     bars - diff lag 6 == 6 base bars regardless of screening_grid.
    _S('residual_momentum_fast', 'res_zscore_momentum_4', 'residual_reversion',
       'faster residual-displacement momentum (4-bar)'),
    _S('residual_cumsum_short', 'res_cumsum_short', 'residual_reversion',
       'short cumulative residual: accumulated idiosyncratic displacement'),
    _S('residual_cumsum_long', 'res_cumsum_long', 'residual_reversion',
       'long cumulative residual displacement'),
    _S('residual_cumsum_xlong', 'res_cumsum_xlong', 'residual_reversion',
       'extra-long cumulative residual displacement'),
    _S('residual_drawdown', 'res_spread_drawdown', 'residual_reversion',
       'residual spread drawdown: how far below its run-up high'),
    _S('residual_runup', 'res_spread_runup', 'residual_reversion',
       'residual spread run-up: stretched moves revert'),
    _S('residual_bars_since_extreme', 'res_bars_since_extreme', 'residual_reversion',
       'bars since last residual extreme: reversion-window maturity'),
    _S('residual_reversion_speed', 'res_reversion_speed', 'residual_reversion',
       'estimated residual reversion speed'),
    _S('residual_sign_persistence', 'res_sign_persistence', 'residual_reversion',
       'persistence of residual sign: trend-vs-flip of displacement'),
    _S('residual_hurst', 'res_hurst', 'residual_reversion',
       'Hurst exponent of residual: <0.5 mean-reverts'),
    _S('residual_autocorr', 'res_ac1_rolling', 'residual_reversion',
       'rolling lag-1 residual autocorrelation'),
    _S('residual_mr_signal_scaled', 'res_mr_signal_scaled', 'residual_reversion',
       'scaled mean-reversion signal (speed/confidence weighted)'),
    _S('residual_mr_signal_weighted', 'res_mr_signal_weighted', 'residual_reversion',
       'weighted mean-reversion signal'),
    _S('residual_mr_regime_strength', 'res_mr_regime_strength', 'residual_reversion',
       'strength of the mean-reversion regime'),
    _S('residual_cumsum_term', 'res_cumsum_short', 'residual_reversion',
       'short vs long cumulative residual: regime delta', op='spread',
       col2='res_cumsum_long'),
    _S('residual_cumsum_term_x', 'res_cumsum_short', 'residual_reversion',
       'short vs extra-long cumulative residual: regime delta', op='spread',
       col2='res_cumsum_xlong'),
    _S('residual_runup_vs_drawdown', 'res_spread_runup', 'residual_reversion',
       'run-up minus drawdown: net over-extension', op='spread',
       col2='res_spread_drawdown'),
    _S('residual_vol_term', 'res_vol_short', 'residual_reversion',
       'residual vol term structure (short/long)', op='ratio',
       col2='res_vol_long'),
    _S('residual_displacement_change_short', 'res_zscore', 'residual_reversion',
       'change in residual z over 6 base bars (displacement velocity)',
       op='diff', lag=6),
    _S('residual_displacement_change_long', 'res_zscore', 'residual_reversion',
       'change in residual z over 18 base bars', op='diff', lag=18),

    # --- Ornstein-Uhlenbeck dynamics
    _S('ou_expected_change', 'ou_expected_change', 'residual_reversion',
       'OU model expected next-step change'),
    _S('ou_reversion_rate_short', 'ou_lambda_short', 'residual_reversion',
       'fast OU mean-reversion rate'),
    _S('ou_reversion_rate_long', 'ou_lambda_long', 'residual_reversion',
       'slow OU mean-reversion rate'),
    _S('ou_halflife_long', 'ou_halflife_long', 'residual_reversion',
       'long-horizon OU half-life'),
    _S('ou_regime', 'ou_regime', 'residual_reversion',
       'OU regime classification'),
    _S('ou_signal_raw', 'ou_signal', 'residual_reversion',
       'raw OU reversion signal'),
    _S('ou_reversion_rate_term', 'ou_lambda_short', 'residual_reversion',
       'OU reversion-speed term structure (short/long)', op='ratio',
       col2='ou_lambda_long'),
    _S('ou_halflife_term', 'ou_halflife_short', 'residual_reversion',
       'OU half-life term structure (short/long)', op='ratio',
       col2='ou_halflife_long'),
    _S('ou_equilibrium_change', 'ou_zscore', 'residual_reversion',
       'change in OU z over 6 base bars', op='diff', lag=6),

    # --- factor loadings / beta instability
    _S('beta_market', 'fl_beta_market', 'factor_loading',
       'market-factor beta level'),
    _S('beta_size', 'fl_beta_size', 'factor_loading',
       'size-factor beta level'),
    _S('beta_market_drift_short', 'fl_beta_market_change_short', 'factor_loading',
       'short-window market-beta drift'),
    _S('beta_market_drift_long', 'fl_beta_market_change_long', 'factor_loading',
       'long-window market-beta drift'),
    _S('beta_size_drift_short', 'fl_beta_size_change_short', 'factor_loading',
       'short-window size-beta drift'),
    _S('beta_size_drift_long', 'fl_beta_size_change_long', 'factor_loading',
       'long-window size-beta drift'),
    _S('factor_r2', 'fl_r2_total', 'factor_loading',
       'total factor R2: how factor-explained a name is'),
    _S('beta_market_accel', 'fl_beta_market_change_short', 'factor_loading',
       'market-beta acceleration (short vs long drift)', op='spread',
       col2='fl_beta_market_change_long'),
    _S('beta_size_accel', 'fl_beta_size_change_short', 'factor_loading',
       'size-beta acceleration (short vs long drift)', op='spread',
       col2='fl_beta_size_change_long'),

    # --- additional market-structure / lead-lag
    _S('market_correlation', 'mk_corr_market_1d', 'market_structure',
       '1d correlation to the market factor'),
    _S('market_move', 'mk_market_move_z', 'market_structure',
       'standardized market move: beta-response context'),
    _S('market_vol', 'mk_market_vol_1d', 'market_structure',
       '1d market volatility regime'),
    _S('lag_corr_short', 'mk_lag_corr_short', 'market_structure',
       'short-window lagged correlation to market'),
    _S('lag_corr_long', 'mk_lag_corr_long', 'market_structure',
       'long-window lagged correlation to market'),
    _S('lag_corr_term', 'mk_lag_corr_short', 'market_structure',
       'lead-lag correlation term structure (short vs long)', op='spread',
       col2='mk_lag_corr_long'),
    _S('beta_drift_change', 'mk_beta_drift', 'market_structure',
       'change in beta drift over 6 base bars', op='diff', lag=6),

    # --- additional funding: raw-level relative value
    _S('funding_annualized', 'fr_annualized', 'funding',
       'annualized funding level: crowded carry'),
    _S('funding_cumulative_24h', 'fr_cumulative_24h', 'funding',
       '24h cumulative funding paid: carry stress'),
    _S('funding_extreme_high', 'fr_extreme_high', 'funding',
       'funding extreme-high flag: crowded longs'),
    _S('funding_extreme_low', 'fr_extreme_low', 'funding',
       'funding extreme-low flag: crowded shorts'),
    _S('funding_dev_from_mean', 'fr_rate', 'funding',
       'funding level vs its own 7d mean (level momentum)', op='spread',
       col2='fr_rate_ma_7d'),
    _S('funding_cumulative_vs_rate', 'fr_cumulative_24h', 'funding',
       'accumulated vs instantaneous funding: carry build-up', op='spread',
       col2='fr_rate'),
    _S('funding_change_short', 'fr_rate', 'funding',
       'change in funding level over 6 base bars', op='diff', lag=6),
    _S('funding_change_long', 'fr_rate', 'funding',
       'change in funding level over 18 base bars', op='diff', lag=18),

    # --- additional open interest
    _S('oi_level_z', 'oi_value_zscore', 'open_interest',
       'standardized open-interest level'),
    _S('oi_change_pct', 'oi_change_pct', 'open_interest',
       'percent change in open interest'),
    _S('oi_relative_change', 'oi_relative', 'open_interest',
       'change in relative OI over 6 base bars', op='diff', lag=6),

    # --- additional cross-sectional context + interactions
    _S('mcap_rank', 'cs_mcap_rank', 'cross_sectional',
       'cross-sectional market-cap rank (size context)'),
    _S('funding_cross_z', 'cs_funding_z', 'cross_sectional',
       'cross-sectional funding z: relative crowding'),
    _S('breadth', 'cs_breadth_sma', 'cross_sectional',
       'cross-sectional breadth regime'),
    _S('xs_return_reversal_fast', 'cs_ret_rank_1h', 'cross_sectional',
       '1h cross-sectional return rank (fast reversal)'),
    _S('relvol_per_size', 'cs_rel_volume', 'cross_sectional',
       'relative volume per unit size rank (liquidity interaction)',
       op='ratio', col2='cs_mcap_rank'),
    _S('xs_return_reversal_term', 'cs_ret_rank_1h', 'cross_sectional',
       '1h vs 1d return-rank reversal term structure', op='spread',
       col2='cs_ret_rank_1d'),

    # --- additional liquidity
    _S('max_intrabar_move', 'ib_max_move_1m', 'liquidity',
       'largest 1-minute intrabar move: jump / illiquidity'),
]


# Smoothing-halflife variants: many alphas differ mainly by speed. For a few
# strong base signals (order-flow, funding/OI, residual, market-structure) emit
# h0 / h12 / h36 variants; the global default (h3) is already covered by the
# base entry above, so it is not duplicated here. Appended to SPACES so they are
# ordinary members of the library.
_SMOOTHING_FAMILIES = [
    ('residual_displacement', 'res_zscore', 'residual_reversion', 'residual displacement'),
    ('funding_z', 'fr_rate_zscore', 'funding', 'funding z'),
    ('signed_volume', 'vl_signed_volume_ratio', 'order_flow', 'signed-volume imbalance'),
    ('oi_change', 'oi_change_zscore', 'open_interest', 'abnormal OI change'),
    ('lag_response', 'mk_lag_response_gap', 'market_structure', 'lagged market response'),
]
_SMOOTHING_HALFLIVES = (0.0, 12.0, 36.0)


def _smoothing_variants():
    out = []
    for base, col, theme, desc in _SMOOTHING_FAMILIES:
        for hl in _SMOOTHING_HALFLIVES:
            label = 'h0' if hl == 0 else f'h{int(hl)}'
            out.append(_S(f'{base}_{label}', col, theme,
                          f'{desc} smoothed at halflife {label}', halflife=hl))
    return out


SPACES = SPACES + _smoothing_variants()


def compute_space_raw(space: SpaceDef, features: pd.DataFrame) -> pd.Series:
    """Raw (pre-normalization) value of a space - a vectorized expression over
    feature columns. evaluate.py applies the cross-sectional z-score + neutrality.
    """
    c = space.columns
    if space.op == 'direct':
        return features[c[0]]
    if space.op == 'spread':
        return features[c[0]] - features[c[1]]
    if space.op == 'ratio':
        return features[c[0]] / (features[c[1]].abs() + 1e-9)
    if space.op == 'diff':
        return features[c[0]] - features.groupby('symbol', sort=False)[c[0]].shift(space.lag)
    raise ValueError(f"unknown space op: {space.op}")


def build_registry_entries(smoothing_halflife: Optional[float] = None) -> dict:
    """Registry entries {name: info} consumed by research.signals.evaluate."""
    if smoothing_halflife is None:
        smoothing_halflife = get('signals.spaces.smoothing_halflife',
                                 get('signals.smoothing_halflife', 3))
    entries = {}
    for sp in SPACES:
        # A per-space halflife (smoothing-speed variants) overrides the global
        # default; None falls back to it.
        hl = sp.halflife if sp.halflife is not None else smoothing_halflife
        entries[f'space_{sp.name}'] = {
            'signal_def': sp,
            'description': sp.rationale,
            'category': sp.theme,
            'direction': 1,
            'kind': 'space',
            'smoothing_halflife': hl,
            'family': sp.theme,
        }
    return entries
