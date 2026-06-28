"""
Walk-Forward Market-Neutral Portfolio.

Per rolling window (train_months -> test_days):
1. SELECT (training data only, from signal_daily_stats aggregates):
   - Pooled IC stats per (signal, horizon); Bonferroni correction across
     horizons and Benjamini-Yekutieli FDR across dependent signal variants.
   - Threshold filters (IC band, ICIR, Sharpe of daily net returns, turnover).
   - Composite ranking, family cap, greedy de-correlation on daily returns.
2. COMBINE: recompute selected signals at full resolution on the test window
   using the same full-history/current-universe convention as research, then
   IC/cost-weight.
3. ALPHA (Grinold): alpha_i[t] = sum_h IC_h * sigma_i * z_{i,h}[t] / sqrt(p_h)
   where IC_h is the pooled training IC of the horizon composite, sigma_i the
   per-asset single-bar residual vol, p_h the horizon length in bars.
4. OPTIMIZE (portfolio.weight_scheme): 'equal_weight' (default) sizes by the
   cross-sectional RANK of alpha - covariance-free - while 'mvo' uses the
   Ledoit-Wolf shrunk covariance of single-bar RESIDUAL returns (refreshed
   daily). Both impose the same equality constraints [dollar, market-beta,
   size-beta] = 0, per-name cap and gross leverage 1. The benchmark_scheme is
   run alongside each window as a monitored foil. Rank sizing is the default
   because the EW-vs-MVO walk-forward showed the covariance weighting was
   net-destructive on this low-breadth, negatively-skewed book.
5. BACKTEST at asset level on RAW forward returns (the honest test - the
   neutrality constraints, not the residual bookkeeping, must do the hedging),
   with per-side costs on turnover. Realized factor exposures are tracked and
   reported - this is the market-neutrality acceptance check.

Outputs: wf_portfolio_returns, wf_portfolio_windows, wf_portfolio_exposures.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import logging
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import norm

from dbutil import load_data, save_data, delete_table, delete_rows_where, _scan
from config import (config as global_config, get, horizon_col,
                    horizon_bars, BASE_FREQUENCY, BARS_PER_DAY)
from research.lib.portfolio_opt import (shrunk_covariance, solve_constrained_mvo,
                                              solve_equal_weight,
                                              benjamini_hochberg,
                                              benjamini_yekutieli,
                                              residual_clusters,
                                              cluster_penalty_matrix)
from research.signals.evaluate import (build_registry, compute_signal_panel,
                                      signal_feature_columns,
                                      load_universe_membership,
                                      universe_member_mask,
                                      LAG_GRID, SCREENING_GRID, lag_label)

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO,
                    format=global_config['logging']['format'],
                    datefmt=global_config['logging']['datefmt'])

WF = global_config['walk_forward']
PORT = global_config['portfolio']
FACTOR_NAMES = get('risk_model.factors', ['market', 'size'])
# Realized exposure to each factor is tracked as the neutrality acceptance check.
# market/size keep their legacy 'mkt_exposure'/'size_exposure' column names; any
# additional factors (momentum, vol, ...) are tracked generically as '<name>_exposure'.
EXTRA_EXPOSURE_FACTORS = [n for n in FACTOR_NAMES if n not in ('market', 'size')]
EXTRA_EXPOSURE_COLS = [f'{n}_exposure' for n in EXTRA_EXPOSURE_FACTORS]
IC_HAC_LAGS = WF.get('ic_hac_lags', 'auto')
PORTFOLIO_RETURNS_TABLE = 'wf_portfolio_returns'   # production-sizing book
# Benchmark (foil) sizing scheme run alongside the production book with the same
# alpha/selection/neutrality - isolates the value of the production risk model.
PORTFOLIO_RETURNS_BENCH_TABLE = 'wf_portfolio_returns_bench'
SIGNAL_ATTRIBUTION_TABLE = 'wf_portfolio_signal_attribution'
WEIGHTS_TABLE = 'wf_portfolio_weights'       # per-bar held weight per name
RISK_TABLE = 'wf_portfolio_risk'             # per-bar predicted risk / cov diagnostics
BACKTEST_TABLES = (PORTFOLIO_RETURNS_TABLE, PORTFOLIO_RETURNS_BENCH_TABLE,
                   'wf_portfolio_windows',
                   'wf_portfolio_exposures', SIGNAL_ATTRIBUTION_TABLE,
                   WEIGHTS_TABLE, RISK_TABLE)
# 'wf_portfolio_returns_ew' was the old name for the benchmark table (when MVO
# was production and EW the foil); cleaned up on reset so it can't go stale.
LEGACY_BACKTEST_TABLES = ('wf_portfolio_equity', 'wf_portfolio_returns_ew')


def _nw_tstat(x: np.ndarray, lags='auto') -> float:
    """Newey-West (Bartlett) HAC t-stat for the mean of a serially-correlated
    series. Daily cross-sectional ICs persist across days, so the iid t-stat
    mean/(std/sqrt(N)) is optimistic; this widens the SE for that persistence."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return 0.0
    e = x - x.mean()
    var = float(e @ e) / n                       # gamma_0
    L = (int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
         if lags in ('auto', None) else int(lags))
    L = max(0, min(L, n - 1))
    for k in range(1, L + 1):
        w = 1.0 - k / (L + 1.0)                   # Bartlett kernel
        var += 2.0 * w * float(e[k:] @ e[:-k]) / n
    if var <= 0:
        return 0.0
    return float(x.mean() / np.sqrt(var / n))


def grouped_nw_tstat(dd: pd.DataFrame, lags='auto') -> pd.Series:
    """Vectorized Newey-West HAC t-stat of the mean, per signal_name.

    Bit-for-bit equivalent to grouping `dd` by signal_name and applying
    `_nw_tstat` to each group's date-ordered `ic_day` series, but computed with
    grouped numpy ops instead of a per-signal Python apply (the relevance phase
    runs over the full candidate set every window, so this is the hot loop).

    `dd` needs columns ['signal_name', 'date', 'ic_day']; NaN ic_day rows are
    dropped exactly as `_nw_tstat` drops them per group. Returns a Series indexed
    by signal_name (signals with <3 finite obs or non-positive variance -> 0.0,
    matching `_nw_tstat`).
    """
    sub = dd[['signal_name', 'date', 'ic_day']]
    sub = sub[np.isfinite(sub['ic_day'].to_numpy())]
    if sub.empty:
        return pd.Series(dtype=float)
    # Contiguous, date-ordered groups (sort within group preserved by mergesort).
    sub = sub.sort_values(['signal_name', 'date'], kind='mergesort')
    x = sub['ic_day'].to_numpy(dtype=float)
    codes, uniq = pd.factorize(sub['signal_name'].to_numpy(), sort=False)
    G = len(uniq)
    n = np.bincount(codes, minlength=G).astype(float)
    mean = np.bincount(codes, weights=x, minlength=G) / n
    e = x - mean[codes]                                   # demeaned within group
    s0 = np.bincount(codes, weights=e * e, minlength=G)   # gamma_0 * n

    if lags in ('auto', None):
        L = np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)).astype(int)
    else:
        L = np.full(G, int(lags), dtype=int)
    L = np.clip(L, 0, (n - 1).astype(int))

    accum = np.zeros(G)
    for k in range(1, int(L.max()) + 1 if G else 1):
        same = codes[k:] == codes[:-k]                    # within-group pairs only
        prod = np.where(same, e[k:] * e[:-k], 0.0)
        sk = np.bincount(codes[k:], weights=prod, minlength=G)  # gamma_k * n
        w = np.where(L >= k, 1.0 - k / (L + 1.0), 0.0)    # Bartlett kernel
        accum += 2.0 * w * sk

    var = (s0 + accum) / n                                # long-run variance
    tstat = np.zeros(G)
    good = (n >= 3) & (var > 0)
    tstat[good] = mean[good] / np.sqrt(var[good] / n[good])
    return pd.Series(tstat, index=uniq)


def gp_trade_rate(cost_bps: float, gp: dict, legacy_halflife: float) -> float:
    """Per-bar trade rate: the fraction of the gap to the aim portfolio traded
    each bar (book_t = (1-rate)*book_{t-1} + rate*aim_t).

    Principled, cost-responsive replacement for the fixed-halflife EWM: the
    myopic optimal trade-toward-aim rate gamma/(gamma+lambda) that balances the
    penalty for sitting off the aim against quadratic trading cost. The cost-to-
    aversion ratio is scaled by the REAL trading cost (higher cost -> slower
    trading). Alpha decay is handled separately by the aim discount. Falls back
    to the legacy halflife rate when disabled or trade_urgency is unset.
    """
    urgency = gp.get('trade_urgency') if gp else None
    if not (gp and gp.get('enabled') and urgency is not None):
        return float(1.0 - np.exp(-np.log(2) / max(legacy_halflife, 1e-9)))
    ref = float(gp.get('ref_cost_bps', cost_bps))
    omega = float(urgency) * (ref / max(float(cost_bps), 1e-9))
    return float(omega / (1.0 + omega))


def liquidity_multipliers(adv: pd.Series, cfg: dict) -> Tuple[pd.Series, pd.Series]:
    """Per-name (cost_mult, speed_mult) from trailing dollar-volume (ADV).

    Cross-sectional vs the median ADV (scale-free), per the trading-speed/alpha tradeoff (
    Maximizes Your Alpha": illiquid names cost
    MORE to trade and should fill toward the aim SLOWER; liquid names cheaper /
    faster.

        cost_mult_i  = clip(1 + impact_coef * (adv_ref/adv_i - 1),
                            min_cost_mult, max_cost_mult)
        speed_mult_i = clip((adv_i/adv_ref) ** speed_exponent,
                            min_speed_mult, max_speed_mult)

    adv_ref is the cross-sectional median ADV. Names with missing/non-positive
    ADV get the neutral multiplier 1.0. Returns multipliers indexed like `adv`.
    """
    idx = adv.index
    a = pd.to_numeric(adv, errors='coerce')
    valid = a[a > 0]
    if valid.empty:
        ones = pd.Series(1.0, index=idx)
        return ones, ones.copy()
    adv_ref = float(valid.median())
    ratio = (a / adv_ref).where(a > 0)                  # adv_i / adv_ref
    inv = (adv_ref / a).where(a > 0)                    # adv_ref / adv_i
    cost_mult = (1.0 + float(cfg['impact_coef']) * (inv - 1.0)).clip(
        float(cfg['min_cost_mult']), float(cfg['max_cost_mult']))
    speed_mult = (ratio ** float(cfg['speed_exponent'])).clip(
        float(cfg['min_speed_mult']), float(cfg['max_speed_mult']))
    return cost_mult.reindex(idx).fillna(1.0), speed_mult.reindex(idx).fillna(1.0)

WARMUP_DAYS = int(get('signals.warmup_days', 10))   # feature warmup before test_start


@dataclass
class WindowResult:
    window_idx: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_end: pd.Timestamp
    selected: Dict[str, List[str]] = field(default_factory=dict)  # bucket -> signals
    weights: Dict[str, Dict[str, float]] = field(default_factory=dict)  # signed
    horizon_ic: Dict[str, float] = field(default_factory=dict)     # bucket -> |IC|
    bucket_h: Dict[str, float] = field(default_factory=dict)       # bucket -> half-life (bars)
    eff_breadth: Dict[str, float] = field(default_factory=dict)
    n_candidates: int = 0
    oos_returns: Optional[pd.DataFrame] = None  # production-sizing book
    oos_returns_bench: Optional[pd.DataFrame] = None  # benchmark-sizing foil
    oos_sharpe: float = np.nan
    oos_sharpe_bench: float = np.nan
    is_sharpe: float = np.nan
    avg_gross: float = np.nan
    avg_turnover: float = np.nan
    avg_abs_mkt_exposure: float = np.nan
    avg_abs_size_exposure: float = np.nan
    cost_to_alpha: float = np.nan   # half-alpha: costs / expected gross alpha
    net_to_gross: float = np.nan    # realized: sum(net) / sum(gross)
    train_oos_ic_rank_corr: float = np.nan
    train_oos_ic_sign_accuracy: float = np.nan
    selection_counts: Dict[str, int] = field(default_factory=dict)
    signal_attribution: List[dict] = field(default_factory=list)  # per-signal OOS edge
    oos_weights: Optional[pd.DataFrame] = None   # per-bar held weight per name
    oos_risk: Optional[pd.DataFrame] = None       # per-bar predicted risk / cov diagnostics


# =============================================================================
# Selection from daily aggregates
# =============================================================================

class SignalSelector:
    def __init__(self, daily_stats: pd.DataFrame,
                 signal_categories: Optional[Dict[str, str]] = None):
        ds = daily_stats.copy()
        ds['date'] = pd.to_datetime(ds['date'])
        self.daily = ds
        self.daily_by_horizon = {
            h: g.reset_index(drop=True)
            for h, g in ds.groupby('horizon', sort=False)
        }
        self.signal_categories = signal_categories or {}
        self.last_candidate_stats = pd.DataFrame()
        self.last_rets = pd.DataFrame()   # selected signals' training daily returns
        self.last_selection_counts: Dict[str, int] = {}

    def _daily_slice(self, horizon: str, start: pd.Timestamp,
                     end: pd.Timestamp,
                     signals: Optional[List[str]] = None) -> pd.DataFrame:
        d = self.daily_by_horizon.get(horizon)
        if d is None:
            return pd.DataFrame(columns=self.daily.columns)
        mask = (d['date'] >= start) & (d['date'] < end)
        if signals is not None:
            mask &= d['signal_name'].isin(signals)
        return d.loc[mask]

    @staticmethod
    def purged_end(lag: int, end: pd.Timestamp) -> pd.Timestamp:
        """Purged train end: drop the tail whose forward targets reach into
        the test period (an IC stamped on the last training day uses returns
        from the first test day - selection would peek across the boundary)."""
        purge_days = int(np.ceil(lag / BARS_PER_DAY))
        return end - pd.Timedelta(days=purge_days)

    def window_stats(self, horizon: str, start: pd.Timestamp,
                     end: pd.Timestamp) -> pd.DataFrame:
        """Pooled per-signal stats on [start, end) for one horizon.

        Direction is resolved here: 'sign' is the sign of the training IC.
        IC diagnostics and gross Sharpe are reported for the signal traded in
        that direction, so negative-IC signals can be selected as anti-signals.
        """
        d = self._daily_slice(horizon, start, end)
        if d.empty:
            return pd.DataFrame()

        g = d.groupby('signal_name')
        n = g['n_cs'].sum()
        ic_mean = g['ic_sum'].sum() / n
        ic_var = g['ic_sumsq'].sum() / n - ic_mean ** 2
        ic_std = np.sqrt(ic_var.clip(lower=0))
        # Newey-West HAC t-stat on the DAILY IC series (serially correlated
        # across days), not the iid pooled t over all cross-sections.
        dd = d.assign(ic_day=d['ic_sum'] / d['n_cs'].replace(0, np.nan))
        tstat = grouped_nw_tstat(dd, IC_HAC_LAGS).reindex(ic_mean.index).fillna(0.0)
        sign = np.where(ic_mean.values >= 0, 1.0, -1.0)

        ret = g['ret_gross'].mean()
        ret_std = g['ret_gross'].std()
        ret_f = -g['ret_gross'].mean()
        ret_f_std = g['ret_gross'].std()
        n_days = g['ret_net'].count()
        sharpe = (ret / ret_std.replace(0, np.nan)) * np.sqrt(365)
        sharpe_f = (ret_f / ret_f_std.replace(0, np.nan)) * np.sqrt(365)

        # Cost-aware (net) edge in the TRADED direction. The gross Sharpe above is
        # what the signal would earn if trading were free; the economic floor is
        # the Sharpe AFTER paying cost_bps/side on the signal's own rebalance
        # turnover. Many high-gross-IC short-horizon signals are net-NEGATIVE -
        # their few-bp edge never clears a round trip - so gating on gross alone
        # admits signals the book then loses money trading.
        cost = PORT['cost_bps'] / 10000.0
        sign_map = dict(zip(ic_mean.index, sign))
        net_row = (d['signal_name'].map(sign_map) * d['ret_gross']
                   - d['turnover'] * cost)
        gn = net_row.groupby(d['signal_name'])
        net_ret = gn.mean().reindex(ic_mean.index)
        net_std = gn.std().reindex(ic_mean.index)
        sharpe_net = (net_ret / net_std.replace(0, np.nan)) * np.sqrt(365)

        turnover = g['turnover'].sum() / g['n_rebalances'].sum().replace(0, np.nan)

        # Stability: IC sign agreement across window thirds. A pooled IC made
        # entirely in the first third is a decayed signal, not a live one.
        span = max((end - start).total_seconds(), 1.0)
        third = (((d['date'] - start).dt.total_seconds() / span) * 3
                 ).clip(upper=2.999).astype(int)
        g3 = d.groupby(['signal_name', third])
        ic3 = (g3['ic_sum'].sum() / g3['n_cs'].sum()).unstack()
        pooled_sign = pd.Series(sign, index=ic_mean.index)
        stable = ic3.apply(lambda col: np.sign(col) == pooled_sign).sum(axis=1)
        stable = stable.reindex(ic_mean.index).fillna(0)
        recent = (np.sign(ic3.get(2, pd.Series(index=ic3.index, dtype=float))) ==
                  pooled_sign).reindex(ic_mean.index).fillna(False)

        # Liquid-half IC and Q5-Q1 tail spread (columns exist for stats
        # produced by the current signals/evaluate; NaN-safe otherwise)
        if 'liq_ic_sum' in d.columns:
            ic_liq = g['liq_ic_sum'].sum() / g['n_liq'].sum().replace(0, np.nan)
        else:
            ic_liq = pd.Series(np.nan, index=ic_mean.index)
        if 'qs_sum' in d.columns:
            q_spread = g['qs_sum'].sum() / g['n_qs'].sum().replace(0, np.nan)
        else:
            q_spread = pd.Series(np.nan, index=ic_mean.index)

        out = pd.DataFrame({
            'signal_name': ic_mean.index,
            'ic_mean': ic_mean.values,
            'sign': sign,
            'ic_std': ic_std.values,
            'ic_tstat': tstat.values,
            'icir': (ic_mean / ic_std.replace(0, np.nan)).values,
            'sharpe': np.where(sign > 0, sharpe.values, sharpe_f.values),
            'sharpe_net': sharpe_net.values,
            'net_ret_mean': net_ret.values,
            'avg_turnover': turnover.values,
            'stable_thirds': stable.values,
            'recent_third_consistent': recent.values,
            'ic_liquid': ic_liq.reindex(ic_mean.index).values,
            'q_spread': q_spread.reindex(ic_mean.index).values,
            'n_obs': n.values,
            'n_days': n_days.values,
        })
        return out.dropna(subset=['ic_mean'])

    def daily_returns_matrix(self, horizon: str, signals: List[str],
                             start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        d = self._daily_slice(horizon, start, end, signals)
        if d.empty:
            return pd.DataFrame()
        return d.pivot_table(index='date', columns='signal_name',
                             values='ret_net', aggfunc='first')

    def signed_daily_returns(self, horizon: str, signal: str, sign: float,
                             start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
        """Daily gross return with the selected signal direction applied."""
        d = self._daily_slice(horizon, start, end, [signal])
        if d.empty:
            return pd.Series(dtype=float)
        d = d.set_index('date')
        return sign * d['ret_gross']

    def horizon_stats(self, start: pd.Timestamp,
                      end: pd.Timestamp) -> pd.DataFrame:
        """
        Select each signal's strongest directly observed forward-return lag.

        Cumulative-forward-return IC across horizons is not an exponential
        decay curve, so each signal is pinned to the lag with the largest
        absolute HAC t-stat, and `p_horizon` Bonferroni-adjusts that lag's
        p-value for the lag-grid search (consumed by the FDR pre-filter).
        """
        per_lag = {}
        for lag in LAG_GRID:
            st = self.window_stats(lag_label(lag), start,
                                   self.purged_end(lag, end))
            if not st.empty:
                per_lag[lag] = st.set_index('signal_name')
        if not per_lag:
            return pd.DataFrame()

        all_sigs = sorted(set().union(*[set(s.index) for s in per_lag.values()]))
        lags = np.array(sorted(per_lag.keys()), dtype=float)

        tstat = pd.DataFrame({lag: per_lag[lag]['ic_tstat'] for lag in lags},
                             index=all_sigs)

        rows = []
        min_valid = WF['horizon_selection'].get('min_valid_lags', 1)
        for sig in all_sigs:
            t_s = tstat.loc[sig]
            valid = t_s.replace([np.inf, -np.inf], np.nan).dropna()
            if len(valid) < min_valid:
                continue
            best_lag = float(valid.abs().idxmax())
            p_min = float(2 * norm.sf(abs(valid.loc[best_lag])))
            p_horizon = min(1.0, p_min * len(valid))
            best = per_lag[best_lag].loc[sig]
            rows.append({
                'signal_name': sig, 'ic0': best['ic_mean'],
                'holding_lag': best_lag, 'sign': best['sign'],
                'p_horizon': p_horizon, 'best_lag': best_lag,
                'ic_mean': best['ic_mean'], 'icir': best['icir'],
                'ic_tstat': best['ic_tstat'], 'sharpe': best['sharpe'],
                'sharpe_net': best['sharpe_net'],
                'net_ret_mean': best['net_ret_mean'],
                'avg_turnover': best['avg_turnover'],
                'stable_thirds': best['stable_thirds'],
                'recent_third_consistent': best['recent_third_consistent'],
                'ic_liquid': best['ic_liquid'], 'q_spread': best['q_spread'],
                'n_days': best['n_days'],
                'family': self.signal_categories.get(sig, sig),
            })
        return pd.DataFrame(rows)

    def select_decay(self, start: pd.Timestamp, end: pd.Timestamp
                     ) -> Tuple[Dict, pd.DataFrame, Dict]:
        """FDR pre-filter -> robustness gates -> composite rank -> family cap ->
        greedy de-correlation -> holding-lag buckets.

        Returns (buckets, selected stats, per-bucket effective breadth) where
        buckets = {label: [signals]} formed from observed holding-lag terciles.
        """
        stats = self.horizon_stats(start, end)
        self.last_candidate_stats = stats.copy()
        self.last_selection_counts = {'candidates': len(stats)}
        if stats.empty:
            return {}, stats, {}

        n_cand = len(stats)
        # FDR pre-filter: with hundreds of candidates a chunk clear any fixed
        # significance bar by luck. The procedure keeps an adaptive cutoff so the
        # expected false-discovery share among survivors stays <= fdr_alpha. A
        # loose alpha only sweeps out the clearly-spurious tail; the gates do the
        # real economic filtering after.
        #   * 'by' (Benjamini-Yekutieli) controls FDR under arbitrary dependence
        #     (alpha / H_m). The library carries dense families of correlated
        #     variants (lookback/term-structure/halflife twins of the same
        #     economic idea), where plain BH is anti-conservative - so BY is the
        #     default.
        #   * 'bh' (Benjamini-Hochberg) is the looser independent-tests variant.
        fdr_fn = (benjamini_yekutieli
                  if WF.get('fdr_method', 'by') == 'by' else benjamini_hochberg)
        discovered = fdr_fn(stats['p_horizon'].values, alpha=WF['fdr_alpha'])
        stats = stats[discovered]
        self.last_selection_counts['after_fdr'] = len(stats)
        if stats.empty:
            stats.attrs['n_candidates'] = n_cand
            return {}, stats, {}

        gates = (
            (stats['ic0'].abs() >= WF['min_ic']) &
            (stats['icir'].abs() >= WF['min_icir']) &
            (stats['sharpe'] >= WF['min_sharpe_threshold']) &
            (stats['avg_turnover'] <= WF['max_signal_turnover']) &
            (stats['stable_thirds'] >= WF['min_stable_thirds'])
        )
        # Cost-aware economic floor: the signal must clear its own trading cost at
        # its rebalance horizon. None -> off (gross-only behaviour); 0.0 -> require
        # the net edge to at least break even after cost_bps/side.
        if WF.get('min_net_sharpe_threshold') is not None:
            gates &= (stats['sharpe_net'] >= WF['min_net_sharpe_threshold'])
        if WF.get('min_holding_lag_bars', 0) > 0:
            gates &= (stats['holding_lag'] >= WF['min_holding_lag_bars'])
        if WF.get('require_recent_third', True):
            gates &= stats['recent_third_consistent']
        liq_ratio = WF.get('min_liquid_ic_ratio', 0.0)
        if liq_ratio > 0:
            ok_liq = (stats['ic_liquid'].abs() >=
                      liq_ratio * stats['ic_mean'].abs())
            # Pass-through when liquid IC is unavailable (NaN): a missing
            # liquid-half stat is a data gap, not a failed test. Auto-rejecting
            # it zeroed out early windows (W00-W03) where ic_liquid was empty.
            gates &= stats['ic_liquid'].isna() | ok_liq
        stats = stats[gates]
        self.last_selection_counts['after_gates'] = len(stats)
        if stats.empty:
            stats.attrs['n_candidates'] = n_cand
            return {}, stats, {}

        ranked = self._rank_candidates(stats)

        # De-correlation on each signal's daily returns at its own best lag.
        # Build the [date x signal] matrix in ONE pivot per distinct lag (signals
        # sharing a lag share dates) instead of one slice+pivot per signal, then
        # precompute the |correlation| matrix once for a single greedy prune
        # (FCBF's speed idea: rank once, sweep once).
        lag_of = dict(zip(stats['signal_name'], stats['best_lag']))
        sigs_by_lag: Dict[int, List[str]] = {}
        for sig in ranked:
            sigs_by_lag.setdefault(int(lag_of[sig]), []).append(sig)
        rets_cols = {}
        for lag, sigs in sigs_by_lag.items():
            mat = self.daily_returns_matrix(lag_label(lag), sigs, start,
                                            self.purged_end(lag, end))
            for sig in sigs:
                if sig in mat.columns:
                    rets_cols[sig] = mat[sig]
        rets = pd.DataFrame(rets_cols)
        self.last_rets = rets   # kept for the covariance-aware signal combination
        corr_abs = rets.corr().abs() if rets.shape[1] > 1 else pd.DataFrame()

        selected = []
        family_counts: Dict[str, int] = {}
        family_cap = WF['horizon_selection'].get('max_variants_per_family', 1)
        for sig in ranked:
            family = stats.loc[stats['signal_name'] == sig, 'family'].iloc[0]
            if family_counts.get(family, 0) >= family_cap:
                continue
            if not selected:
                selected.append(sig)
            else:
                cols = corr_abs.columns
                if (not corr_abs.empty and sig in cols
                        and all(s in cols for s in selected)):
                    max_corr = corr_abs.loc[selected, sig].max()
                    if max_corr > WF['max_correlation_threshold']:
                        continue
                selected.append(sig)
            family_counts[family] = family_counts.get(family, 0) + 1
            if len(selected) >= (WF['max_signals_per_window'] *
                                 WF['horizon_selection']['n_buckets']):
                break

        sel = stats[stats['signal_name'].isin(selected)].copy()
        sel.attrs['n_candidates'] = n_cand
        self.last_selection_counts['selected'] = len(sel)

        # Direct holding-lag terciles -> execution buckets.
        n_b = min(WF['horizon_selection']['n_buckets'], len(sel))
        labels = ['fast', 'mid', 'slow'][:n_b]
        if n_b >= 2:
            sel['bucket'] = pd.qcut(sel['holding_lag'].rank(method='first'),
                                    n_b, labels=labels).astype(str)
        else:
            sel['bucket'] = labels[0] if n_b else ''

        buckets = {lab: sel.loc[sel['bucket'] == lab, 'signal_name'].tolist()
                   for lab in labels}
        eff_n = {lab: self._effective_breadth(rets, sigs)
                 for lab, sigs in buckets.items() if sigs}
        return buckets, sel, eff_n

    def persistence_diagnostic(self, train_stats: pd.DataFrame,
                               start: pd.Timestamp,
                               end: pd.Timestamp) -> Tuple[float, float]:
        """Relate training IC to next-window IC at the same selected lag."""
        if train_stats.empty:
            return np.nan, np.nan
        rows = []
        for lag, tr in train_stats.groupby('best_lag'):
            lag = int(lag)
            oos = self.window_stats(lag_label(lag), start,
                                    self.purged_end(lag, end))
            if oos.empty:
                continue
            m = tr[['signal_name', 'ic_mean']].merge(
                oos[['signal_name', 'ic_mean']], on='signal_name',
                suffixes=('_train', '_oos'))
            rows.append(m)
        if not rows:
            return np.nan, np.nan
        d = pd.concat(rows, ignore_index=True).dropna()
        if len(d) < 3:
            return np.nan, np.nan
        rank_corr = d['ic_mean_train'].corr(d['ic_mean_oos'], method='spearman')
        sign_acc = (np.sign(d['ic_mean_train']) ==
                    np.sign(d['ic_mean_oos'])).mean()
        return float(rank_corr), float(sign_acc)

    @staticmethod
    def _effective_breadth(rets: pd.DataFrame, selected: List[str]) -> float:
        """(sum lambda)^2 / sum lambda^2 of the selected signals' daily-return
        correlation - what de-correlation actually bought, vs len(selected)."""
        cols = [s for s in selected if s in rets.columns]
        if len(cols) < 2:
            return float(len(selected))
        c = rets[cols].corr().fillna(0.0).values
        np.fill_diagonal(c, 1.0)
        ev = np.clip(np.linalg.eigvalsh(c), 0.0, None)
        denom = float((ev ** 2).sum())
        return float(ev.sum() ** 2 / denom) if denom > 0 else float(len(cols))

    @staticmethod
    def _rank_candidates(stats: pd.DataFrame) -> List[str]:
        cfg = WF.get('candidate_ranking', {})
        if not cfg.get('enabled', False):
            key = 'sharpe_net' if 'sharpe_net' in stats.columns else 'sharpe'
            return stats.sort_values(key, ascending=False)['signal_name'].tolist()

        weights = cfg.get('score_weights', {})
        scores = pd.Series(0.0, index=stats['signal_name'].values)
        # Longer directly selected holding lags are more implementation-friendly.
        col_map = {'sharpe': 'sharpe', 'sharpe_net': 'sharpe_net',
                   'icir': 'icir', 'ic_tstat': 'ic_tstat',
                   'inverse_turnover': 'avg_turnover',
                   'inverse_decay': 'holding_lag'}
        for metric, w in weights.items():
            col = col_map.get(metric)
            if col is None or col not in stats.columns:
                continue
            vals = stats[col].values.astype(float)
            if metric in ('icir', 'ic_tstat'):
                vals = np.abs(vals)  # direction is resolved via 'sign'
            if metric == 'inverse_turnover':
                vals = -vals         # lower turnover is better
            # The legacy inverse_decay key now means longer observed holding lag.
            std = np.nanstd(vals)
            if std > 1e-12:
                scores += w * (vals - np.nanmean(vals)) / std
        return scores.sort_values(ascending=False).index.tolist()

    @staticmethod
    def signal_weights(stats: pd.DataFrame, selected: List[str]) -> Dict[str, float]:
        """IC-strength weights with an explicit turnover penalty."""
        cfg = WF.get('signal_weighting', {})
        if not cfg.get('enabled', False) or not selected:
            w = 1.0 / max(len(selected), 1)
            return {s: w for s in selected}
        turn = dict(zip(stats['signal_name'], stats['avg_turnover']))
        ic = dict(zip(stats['signal_name'], stats['ic_mean'].abs()))
        raw = {}
        for s in selected:
            phi = turn.get(s, 0.5)
            if not np.isfinite(phi) or phi <= 0:
                phi = 0.5
            strength = ic.get(s, 0.0)
            adj = strength / (1.0 + phi * cfg['cost_factor'] /
                              cfg['risk_aversion'])
            raw[s] = max(adj, cfg['min_weight_ratio'] * max(ic.values()))
        total = sum(raw.values())
        return {s: v / total for s, v in raw.items()}

    @staticmethod
    def combination_weights(stats: pd.DataFrame, selected: List[str],
                            rets: pd.DataFrame) -> Dict[str, float]:
        """Covariance-aware signal combination (Grinold 'combining signals').

        Optimal combo weights w proportional to C^{-1} . IC, where C is the
        correlation of the signals' TRADED-DIRECTION daily returns and IC is the
        vector of |training IC|. Versus naive IC-weighting this down-weights
        redundant signals and up-weights unique ones - exploiting the
        diversification that flat IC-weighting throws away. C is shrunk toward the
        identity for stability; negative (anti-)weights are clipped to 0 so a
        noisy correlation can't flip a signal into being shorted. Returns POSITIVE
        per-signal weights summing to 1 (the caller applies each signal's sign).
        Falls back to signal_weights when disabled or returns are unavailable.
        """
        cfg = WF.get('signal_combination', {})
        cols = [s for s in selected if rets is not None and s in rets.columns]
        if not cfg.get('enabled', False) or len(cols) < 2:
            return SignalSelector.signal_weights(stats, selected)

        sign = dict(zip(stats['signal_name'], stats['sign']))
        ic = dict(zip(stats['signal_name'], stats['ic_mean'].abs()))
        signed = rets[cols].mul([sign.get(c, 1.0) for c in cols], axis=1)
        C = signed.corr().fillna(0.0).values
        np.fill_diagonal(C, 1.0)
        shrink = float(cfg.get('corr_shrink', 0.5))
        C = (1.0 - shrink) * C + shrink * np.eye(len(cols))
        ic_vec = np.array([max(ic.get(c, 0.0), 0.0) for c in cols])
        try:
            w = np.linalg.solve(C, ic_vec)
        except np.linalg.LinAlgError:
            w = ic_vec
        w = np.clip(w, 0.0, None)
        if w.sum() <= 1e-12:
            return SignalSelector.signal_weights(stats, selected)
        w = w / w.sum()
        out = {s: 0.0 for s in selected}
        out.update({c: float(wi) for c, wi in zip(cols, w)})
        return out


# =============================================================================
# Data context (loaded once)
# =============================================================================

class DataContext:
    def __init__(self, start: Optional[pd.Timestamp] = None,
                 end: Optional[pd.Timestamp] = None):
        start = pd.Timestamp(start) if start is not None else None
        end = pd.Timestamp(end) if end is not None else None

        def _time_filter(column: str) -> Dict[str, List[Tuple[str, pd.Timestamp]]]:
            cond = []
            if start is not None:
                cond.append(('>=', start))
            if end is not None:
                cond.append(('<', end))
            return {column: cond} if cond else {}

        def _wide_panel(table_name: str, value_col: str,
                        time_col: str = 'timestamp') -> pd.DataFrame:
            lf = _scan(table_name)
            if lf is None:
                return pd.DataFrame()
            if start is not None:
                lf = lf.filter(pl.col(time_col) >= start)
            if end is not None:
                lf = lf.filter(pl.col(time_col) < end)
            df = lf.select([time_col, 'symbol', value_col]).collect()
            if df.is_empty():
                return pd.DataFrame()
            wide = df.pivot(index=time_col, on='symbol', values=value_col,
                            aggregate_function='first').sort(time_col)
            out = wide.to_pandas().set_index(time_col)
            out.columns.name = 'symbol'
            return out

        if start is not None or end is not None:
            logging.info(f"Loading portfolio panels "
                         f"{start if start is not None else '-inf'} -> "
                         f"{end if end is not None else '+inf'}...")
        else:
            logging.info("Loading residual returns / loadings / universe...")
        self.res_wide = _wide_panel('residual_returns', 'residual_return')
        self.fwd_raw_wide = _wide_panel('residual_returns',
                                        horizon_col(BASE_FREQUENCY, 'raw'))

        self.loadings = load_data('factor_loadings', filters=_time_filter('date'))
        self.loadings['date'] = pd.to_datetime(self.loadings['date'])

        candidates = load_data('universe', columns=['symbol'])
        if candidates.empty:
            raise RuntimeError("universe table is empty - run etl/universe.py first")
        self.membership = set(candidates['symbol'])

        # Per-name dollar-volume panel (for liquidity-aware costs / trade speed).
        # quote_asset_volume is the bar's $ traded; ADV is a trailing mean of it.
        # Optional: if absent the backtest falls back to flat costs / scalar speed.
        self.dollar_vol_wide = None
        if PORT.get('liquidity_aware', {}).get('enabled'):
            try:
                pv = _wide_panel('prices', 'quote_asset_volume')
                if not pv.empty:
                    self.dollar_vol_wide = pv
            except Exception as e:  # missing column / table -> graceful fallback
                logging.warning(f"liquidity_aware: no dollar-volume panel ({e}); "
                                "falling back to flat costs / scalar trade speed")

        logging.info(f"Panels: {self.res_wide.shape[0]:,} bars x "
                     f"{self.res_wide.shape[1]} symbols")

    def members_at(self, ts: pd.Timestamp) -> set:
        return self.membership

    def betas_for_day(self, day: pd.Timestamp) -> pd.DataFrame:
        """Latest betas at or before `day` per symbol (estimated pre-`day`)."""
        d = self.loadings[self.loadings['date'] <= day]
        if d.empty:
            return pd.DataFrame()
        d = d.sort_values('date').groupby('symbol').tail(1).set_index('symbol')
        # Drop stale betas (> 5 days old)
        d = d[d['date'] >= day - pd.Timedelta(days=5)]
        return d[[f'beta_{n}' for n in FACTOR_NAMES]]


# =============================================================================
# Walk-forward engine
# =============================================================================

class WalkForwardPortfolio:
    def __init__(self):
        daily_stats = load_data('signal_daily_stats')
        if daily_stats.empty:
            raise RuntimeError("signal_daily_stats is empty - run research/signals/evaluate.py first")
        self.registry = build_registry()
        known = set(self.registry)
        before = len(daily_stats)
        daily_stats = daily_stats[daily_stats['signal_name'].isin(known)].copy()
        dropped = before - len(daily_stats)
        if dropped:
            logging.warning(f"Dropped {dropped:,} signal_daily_stats rows with "
                            "no current registry entry")
        categories = {name: info.get('family') or info.get('category', name)
                      for name, info in self.registry.items()}
        self.selector = SignalSelector(daily_stats, categories)
        self.ctx: Optional[DataContext] = None
        self._ctx_start: Optional[pd.Timestamp] = None
        self._ctx_end: Optional[pd.Timestamp] = None
        self.universe_members = load_universe_membership()
        self.windows: List[WindowResult] = []
        # Production sizing scheme + the benchmark foil run alongside it.
        # 'equal_weight' (rank) is the default production scheme; 'mvo' the foil.
        self.weight_scheme = get('portfolio.weight_scheme', 'equal_weight')
        self.benchmark_scheme = get('portfolio.benchmark_scheme', 'mvo')
        self.compare_benchmark = (bool(self.benchmark_scheme)
                                  and self.benchmark_scheme != self.weight_scheme)
        # Active scheme during a backtest pass (set by _backtest_scheme).
        self._weight_scheme = self.weight_scheme
        # Held book carried across contiguous windows (see _backtest_window),
        # one per scheme so production and benchmark evolve independent history.
        self._carry: Dict[str, pd.Series] = {}

    def _set_context_bounds(self, schedule: List[Tuple[pd.Timestamp,
                                                       pd.Timestamp,
                                                       pd.Timestamp]]) -> None:
        if not schedule:
            self._ctx_start = None
            self._ctx_end = None
            return
        liq_cfg = PORT.get('liquidity_aware', {})
        liq_days = (int(liq_cfg.get('adv_window_days', 0))
                    if liq_cfg.get('enabled') else 0)
        warmup_days = max(int(PORT['cov_window_days']),
                          int(PORT['residual_vol_window_days']),
                          liq_days,
                          5) + 1
        first_test_start = schedule[0][1]
        last_test_end = schedule[-1][2]
        self._ctx_start = first_test_start - pd.Timedelta(days=warmup_days)
        self._ctx_end = last_test_end

    def _ensure_context(self) -> None:
        if self.ctx is None:
            self.ctx = DataContext(self._ctx_start, self._ctx_end)

    # ---------------- alpha construction ----------------

    def composite_scores(self, selected: Dict[str, List[str]],
                         weights: Dict[str, Dict[str, float]],
                         feat_start: pd.Timestamp,
                         test_start: pd.Timestamp,
                         test_end: pd.Timestamp) -> Dict[str, pd.DataFrame]:
        """Per-horizon composite z-score panel (wide: timestamp x symbol) on the
        test window. Signals recomputed at full resolution with warmup."""
        all_sigs = sorted({s for sigs in selected.values() for s in sigs})
        if not all_sigs:
            return {}

        needed = sorted({c for s in all_sigs
                         for c in signal_feature_columns(self.registry[s]['signal_def'])})
        time_filter = {
            'timestamp': [('>=', pd.Timestamp(feat_start)),
                          ('<', pd.Timestamp(test_end))]
        }
        features = load_data('features', filters=time_filter,
                             columns=['timestamp', 'symbol'] + needed)
        features['timestamp'] = pd.to_datetime(features['timestamp'])
        features['_is_member'] = universe_member_mask(features, self.universe_members)
        features = features.sort_values(['symbol', 'timestamp']).reset_index(drop=True)
        if features.empty:
            return {}

        panels = {}
        for sig in all_sigs:
            p = compute_signal_panel(sig, self.registry, features)
            p = p[p['timestamp'] >= test_start]
            panels[sig] = p.pivot_table(index='timestamp', columns='symbol',
                                        values='signal', aggfunc='first')

        composites = {}
        for h, sigs in selected.items():
            if not sigs:
                continue
            acc = None
            for s in sigs:
                w = weights[h].get(s, 0.0)
                part = panels[s] * w
                acc = part if acc is None else acc.add(part, fill_value=0.0)
            # Re-standardize the composite cross-sectionally per bar
            mu = acc.mean(axis=1)
            sd = acc.std(axis=1).replace(0, np.nan)
            composites[h] = acc.sub(mu, axis=0).div(sd, axis=0).clip(-3, 3)
        return composites

    # ---------------- one window ----------------

    def run_window(self, idx: int, train_start: pd.Timestamp,
                   train_end: pd.Timestamp, test_end: pd.Timestamp,
                   prev: Optional[WindowResult]) -> Optional[WindowResult]:
        res = WindowResult(idx, train_start, train_end, test_end)

        buckets, sel, eff_breadth = self.selector.select_decay(train_start, train_end)
        n_candidates = int(sel.attrs.get('n_candidates', len(sel))) if not sel.empty else 0
        if not n_candidates:
            n_candidates = self.selector.last_selection_counts.get('candidates', 0)

        selected, weights, bucket_ic, bucket_h = {}, {}, {}, {}
        for lab, sigs in buckets.items():
            if not sigs:
                continue
            stats_b = sel[sel['signal_name'].isin(sigs)]
            selected[lab] = sigs
            # Fold the training-IC sign into the weight: the composite
            # trades each signal in its profitable direction. Covariance-aware
            # combination (Grinold) when enabled, else flat IC-weighting.
            w_pos = SignalSelector.combination_weights(
                stats_b, sigs, self.selector.last_rets)
            sign_map = dict(zip(stats_b['signal_name'], stats_b['sign']))
            weights[lab] = {s: w_pos[s] * sign_map.get(s, 1.0) for s in sigs}
            ic_map = dict(zip(stats_b['signal_name'], stats_b['ic_mean']))
            wsum = sum(abs(w) for w in weights[lab].values())
            # signed ic x signed weight = |ic| x positive weight
            bucket_ic[lab] = float(sum(ic_map.get(s, 0) * w
                                       for s, w in weights[lab].items()) / max(wsum, 1e-12))
            bucket_h[lab] = float(stats_b['holding_lag'].median())

        if not selected and WF.get('fallback_to_previous') and prev and prev.selected:
            selected, bucket_ic, bucket_h = (prev.selected, prev.horizon_ic,
                                             prev.bucket_h)
            weights = prev.weights or {b: {s: 1.0 / len(sigs) for s in sigs}
                                       for b, sigs in selected.items()}

        res.selected = selected
        res.weights = weights
        res.horizon_ic = bucket_ic
        res.bucket_h = bucket_h
        res.eff_breadth = eff_breadth
        res.n_candidates = n_candidates
        res.selection_counts = self.selector.last_selection_counts.copy()

        # Per-stage selection funnel for this window (stages skipped by an empty
        # early-return show as '-').
        c = res.selection_counts
        def _f(k):
            return c['after_' + k] if ('after_' + k) in c else (
                c[k] if k in c else '-')
        logging.info(
            f"W{idx:02d} {train_end.date()} selection funnel: "
            f"candidates {c.get('candidates', n_candidates)} "
            f"-> fdr {_f('fdr')} -> gates {_f('gates')} -> selected {_f('selected')}")
        (res.train_oos_ic_rank_corr,
         res.train_oos_ic_sign_accuracy) = self.selector.persistence_diagnostic(
            self.selector.last_candidate_stats, train_end, test_end)
        # Per-signal OOS attribution (Option A): trace a window's PnL to the
        # individual signals whose standalone edge held up or decayed/flipped OOS.
        res.signal_attribution = self._signal_oos_attribution(sel, train_end, test_end, idx)
        if not selected:
            return res

        feat_start = train_end - pd.Timedelta(days=WARMUP_DAYS)
        composites = self.composite_scores(selected, weights, feat_start,
                                           train_end, test_end)
        if not composites:
            return res

        self._ensure_context()
        bt_returns = self._backtest_scheme(self.weight_scheme, composites,
                                           bucket_ic, bucket_h, train_end, test_end)
        if bt_returns is None or bt_returns.empty:
            return res

        res.oos_returns = bt_returns[[
            'gross_return', 'net_return', 'net_return_lag1',
            'turnover', 'gross_exposure', 'net_exposure',
            'mkt_exposure', 'size_exposure', 'n_positions',
        ] + EXTRA_EXPOSURE_COLS]
        res.oos_weights = self._bt_weights
        res.oos_risk = self._bt_risk
        ppy = BARS_PER_DAY * 365
        std = bt_returns['net_return'].std()
        res.oos_sharpe = float(bt_returns['net_return'].mean() / std * np.sqrt(ppy)) if std > 0 else 0.0
        res.avg_gross = float(bt_returns['gross_exposure'].mean())
        res.avg_turnover = float(bt_returns['turnover'].mean())
        res.avg_abs_mkt_exposure = float(bt_returns['mkt_exposure'].abs().mean())
        res.avg_abs_size_exposure = float(bt_returns['size_exposure'].abs().mean())
        # Half-alpha diagnostic:
        # at the optimal size, costs+risk should consume ~half the gross alpha.
        # cost_to_alpha = sum(trade_cost)/sum(expected gross alpha); a value near
        # 0.5 is the sweet spot. >>0.5 -> over-trading / oversized; <<0.5 ->
        # likely undersized (leaving alpha on the table). net_to_gross is the
        # realized analogue (fraction of gross PnL kept after costs).
        sum_alpha = float(bt_returns['exp_alpha'].sum())
        sum_cost = float(bt_returns['trade_cost'].sum())
        sum_gross = float(bt_returns['gross_return'].sum())
        sum_net = float(bt_returns['net_return'].sum())
        res.cost_to_alpha = (sum_cost / sum_alpha) if sum_alpha > 1e-12 else np.nan
        res.net_to_gross = (sum_net / sum_gross) if abs(sum_gross) > 1e-12 else np.nan
        if np.isfinite(res.cost_to_alpha):
            logging.info(f"  half-alpha: cost/exp_alpha={res.cost_to_alpha:.2f} "
                         f"(~0.5 ideal), net/gross={res.net_to_gross:.2f}")

        # Benchmark (foil) sizing on the SAME composites/selection: only the
        # risk model differs, so production-minus-benchmark isolates what the
        # production risk model adds (or destroys).
        if self.compare_benchmark:
            bench = self._backtest_scheme(self.benchmark_scheme, composites,
                                          bucket_ic, bucket_h, train_end, test_end)
            if bench is not None and not bench.empty:
                res.oos_returns_bench = bench[[
                    'gross_return', 'net_return', 'net_return_lag1',
                    'turnover', 'gross_exposure', 'mkt_exposure', 'size_exposure',
                ]]
                std_b = bench['net_return'].std()
                res.oos_sharpe_bench = (float(bench['net_return'].mean() / std_b *
                                              np.sqrt(ppy)) if std_b > 0 else 0.0)
        return res

    def _backtest_scheme(self, scheme: str, composites: Dict[str, pd.DataFrame],
                         bucket_ic: Dict[str, float], bucket_h: Dict[str, float],
                         test_start: pd.Timestamp,
                         test_end: pd.Timestamp) -> Optional[pd.DataFrame]:
        """Run `_backtest_window` under a given weighting scheme, restoring the
        active scheme afterward. Each scheme carries its own position history."""
        prev = self._weight_scheme
        self._weight_scheme = scheme
        try:
            return self._backtest_window(composites, bucket_ic, bucket_h,
                                         test_start, test_end)
        finally:
            self._weight_scheme = prev

    def _signal_oos_attribution(self, sel: pd.DataFrame, test_start: pd.Timestamp,
                                test_end: pd.Timestamp, idx: int) -> List[dict]:
        """Per selected signal: did its standalone edge hold up out-of-sample?

        Compares the training IC that earned its selection against the realized
        OOS IC at the SAME lag/direction, plus the signal's own dollar-neutral
        OOS return - all in the traded direction (sign applied), so positive =
        still working, negative = decayed or flipped. This is what lets a bad
        window be traced to specific signals rather than 'the market'.
        IC/return are the signal's STANDALONE edge, not its MVO portfolio share.
        """
        if sel is None or sel.empty:
            return []
        oos_by_lag = {}
        for lag in {int(l) for l in sel['best_lag'].unique()}:
            st = self.selector.window_stats(lag_label(lag), test_start, test_end)
            oos_by_lag[lag] = st.set_index('signal_name') if not st.empty else None
        rows = []
        for _, rs in sel.iterrows():
            s = rs['signal_name']
            lag = int(rs['best_lag'])
            sign = float(rs['sign'])
            oos = oos_by_lag.get(lag)
            oos_ic_raw = (float(oos.loc[s, 'ic_mean'])
                          if oos is not None and s in oos.index else np.nan)
            stream = self.selector.signed_daily_returns(lag_label(lag), s, sign,
                                                        test_start, test_end)
            if len(stream):
                oos_ret = float((1 + stream).prod() - 1)
                oos_sharpe = (float(stream.mean() / stream.std() * np.sqrt(365))
                              if stream.std() > 0 else np.nan)
            else:
                oos_ret = oos_sharpe = np.nan
            rows.append({
                'window_idx': idx,
                'bucket': rs.get('bucket', ''),
                'signal_name': s,
                'family': rs.get('family', ''),
                'sign': sign,
                'holding_lag': lag,
                'train_ic': float(sign * rs['ic_mean']),         # traded-direction
                'oos_ic': float(sign * oos_ic_raw),              # traded-direction
                'oos_ret': oos_ret,                              # standalone, signed
                'oos_sharpe': oos_sharpe,
            })
        return rows

    # ---------------- MVO backtest ----------------

    def _backtest_window(self, composites: Dict[str, pd.DataFrame],
                         bucket_ic: Dict[str, float],
                         bucket_h: Dict[str, float],
                         test_start: pd.Timestamp,
                         test_end: pd.Timestamp) -> Optional[pd.DataFrame]:
        if self.ctx is None:
            self._ensure_context()
        # Per-window diagnostic side-channels (read by run_window after this call);
        # reset up front so a no-bar window can't inherit a prior window's data.
        self._bt_weights = None
        self._bt_risk = None
        idx = self.ctx.res_wide.index
        bars = idx[(idx >= test_start) & (idx < test_end)]
        if len(bars) == 0:
            return None

        cost_rate = PORT['cost_bps'] / 10000.0
        cap = PORT['max_position']
        gross_target = PORT['gross_leverage']
        # Per-bar trade rate toward the gross-1 aim (cost-responsive; PnL costs
        # unchanged). This is the SPEED at which the book fills toward the aim.
        gp = PORT.get('gp_trading', {})
        smooth_alpha = gp_trade_rate(PORT['cost_bps'], gp,
                                     PORT['weight_smoothing_halflife'])
        # Hard turnover budget -> max voluntary turnover per bar. The trade rate
        # is throttled below smooth_alpha whenever the aim is far enough that
        # trading at full speed would breach the annual budget.
        max_ann_to = PORT.get('max_annual_turnover')
        per_bar_to_budget = (max_ann_to / (BARS_PER_DAY * 365)
                             if max_ann_to else np.inf)
        cov_window = pd.Timedelta(days=PORT['cov_window_days'])
        vol_window = pd.Timedelta(days=PORT['residual_vol_window_days'])
        impl_lag = int(WF.get('implementation_lag_bars', 1))

        # Per-bucket per-bar alpha scale: (1 - ic_shrink) * IC_b / sqrt(h_b),
        # times the Garleanu-Pedersen aim-portfolio discount h/(h + 1/kappa):
        # alpha faster than the trade rate kappa can't be monetized through
        # costs. ic_shrink pulls the noisy realized IC toward 0 (Grinold & Kahn
        #) so IC swings don't whipsaw the book.
        gp_on = bool(gp.get('enabled', False))
        kappa = max(smooth_alpha, 1e-9)
        # The aim discount must use the rate at which the book ACTUALLY fills, not
        # the nominal trade rate. When the turnover budget binds, the book only
        # closes ~per_bar_budget/gross of the gap each bar - far below smooth_alpha
        # - so discounting at the nominal rate over-sizes alpha the book can never
        # capture. Cap kappa at the realized fill rate (no-op when turnover is
        # ample, since then per_bar_budget/gross >= smooth_alpha).
        if gp.get('discount_at_realized_rate', True) and np.isfinite(per_bar_to_budget):
            realized_rate = per_bar_to_budget / max(gross_target, 1e-9)
            kappa = max(min(kappa, realized_rate), 1e-9)
        ic_shrink = float(PORT.get('ic_shrink', 0.0))
        ic_keep = max(0.0, 1.0 - ic_shrink)
        # Two scales per bucket from the same (1 - ic_shrink) * IC_b / sqrt(h_b)
        # base:
        #  - ic_scale     applies the Garleanu-Pedersen aim discount h/(h+1/kappa)
        #    and drives the MVO/aim - alpha faster than the trade rate kappa can't
        #    be monetized, so the AIM is built from discounted alpha.
        #  - ic_scale_raw OMITS that discount. It is the raw expected edge used
        #    ONLY by the no-trade gate, which asks "is this signal worth a round
        #    trip?" - an economics question that must NOT depend on how fast the
        #    book trades. Folding the aim discount into the gate (the old code)
        #    let a slow trade_urgency veto every profitable signal, so no window
        #    produced OOS returns. ic_shrink stays in both (a conservative,
        #    shrunk edge estimate; Grinold & Kahn).
        ic_scale = {}
        ic_scale_raw = {}
        for b in composites.keys():
            h_b = max(bucket_h.get(b, 1.0), 1.0)
            raw = ic_keep * bucket_ic.get(b, 0.0) / np.sqrt(h_b)
            ic_scale_raw[b] = raw
            ic_scale[b] = raw * (h_b / (h_b + 1.0 / kappa)) if gp_on else raw

        cluster_cfg = PORT.get('cluster_penalty', {})

        # No-trade zone ("lazy trading"): a name whose per-bar expected residual
        # return |alpha_i| is below no_trade_band_mult * (per-name per-side cost)
        # is not worth trading on, so its alpha is zeroed before the MVO. The
        # per-name cost is cost_rate * cost_mult_i (liquidity-aware below).
        no_trade_mult = float(PORT.get('no_trade_band_mult', 0.0))

        # Liquidity-aware per-name cost / trade-speed multipliers (ADV-based),
        # refreshed daily alongside cov/betas. Neutral (1.0) when disabled or no
        # dollar-volume panel is available.
        liq_cfg = PORT.get('liquidity_aware', {})
        liq_on = bool(liq_cfg.get('enabled')) and self.ctx.dollar_vol_wide is not None
        adv_window = (pd.Timedelta(days=liq_cfg['adv_window_days'])
                      if liq_on else None)
        cost_mult = pd.Series(dtype=float)
        speed_mult = pd.Series(dtype=float)

        # Carry the held book across contiguous windows: retraining swaps the
        # alpha/selection, not the positions, so the book should NOT teleport to
        # cash and re-ramp gross from 0 each window. Names that are no longer
        # investable get liquidated by the per-bar `closed` logic below.
        w_history: List[Tuple[pd.Index, np.ndarray]] = []  # implementation lag
        day_cache_key = None
        cov = None
        cov_hist = pd.DataFrame()
        cov_assets = pd.Index([])
        betas = None
        sigma = None

        rows = []
        grid_bars = max(1, horizon_bars(SCREENING_GRID))
        cadence = {
            b: max(grid_bars, int(np.ceil(max(bucket_h.get(b, 1.0), 1.0) /
                                           grid_bars)) * grid_bars)
            for b in composites
        }
        held_scores: Dict[str, pd.Series] = {}
        bar_positions = idx.get_indexer(bars)
        fwd_index = self.ctx.fwd_raw_wide.index
        fwd_row_positions = fwd_index.get_indexer(bars)
        fwd_columns = self.ctx.fwd_raw_wide.columns
        fwd_values_all = self.ctx.fwd_raw_wide.to_numpy(copy=False)
        target_valid = False
        w_target = pd.Series(dtype=float)
        alpha = pd.Series(dtype=float)
        A = pd.DataFrame()
        target_index = pd.Index([])
        target_values = np.array([], dtype=float)
        carry = self._carry.get(self._weight_scheme, pd.Series(dtype=float))
        held_index = carry.index
        held_values = carry.values.astype(float, copy=True)
        alpha_values = np.array([], dtype=float)
        beta_values: Dict[str, np.ndarray] = {}
        cost_values = np.array([], dtype=float)
        speed_values = np.array([], dtype=float)
        fwd_col_positions = np.array([], dtype=int)
        neutralizer = None
        closed_abs_sum = 0.0
        closed_cost_sum = 0.0

        # Per-bar diagnostics to persist: held weights (one array per bar) and a
        # risk row. Cov-level stats (eigen-concentration, clusters) refresh with
        # the covariance; the predicted variance uses the optimized cov submatrix.
        ppy = BARS_PER_DAY * 365
        weight_rows: List[Tuple[pd.Timestamp, np.ndarray, np.ndarray]] = []
        risk_rows: List[dict] = []
        clusters: List = []
        cov_target_values = np.zeros((0, 0), dtype=float)
        cov_eig_top_share = np.nan
        cov_condition = np.nan
        n_clusters_day = 0
        max_cluster_day = 0

        for i, ts in enumerate(bars):
            day = ts.normalize()
            day_changed = day_cache_key != day
            if day_cache_key != day:
                day_cache_key = day
                hist = self.ctx.res_wide[(self.ctx.res_wide.index < day) &
                                         (self.ctx.res_wide.index >= day - cov_window)]
                members = self.ctx.members_at(ts)
                hist = hist[[c for c in hist.columns if c in members]]
                cov = None
                cov_hist = hist
                valid_counts = hist.notna().sum()
                cov_assets = valid_counts[
                    valid_counts >= PORT['cov_min_observations']
                ].index
                betas = self.ctx.betas_for_day(day)
                vol_hist = self.ctx.res_wide[(self.ctx.res_wide.index < day) &
                                             (self.ctx.res_wide.index >= day - vol_window)]
                sigma = vol_hist.std()
                # Trailing per-name ADV ($ volume) -> liquidity multipliers.
                # Strictly-past window (< day), so the multipliers are causal.
                if liq_on:
                    dv = self.ctx.dollar_vol_wide
                    dv_hist = dv[(dv.index < day) & (dv.index >= day - adv_window)]
                    adv = dv_hist.mean()
                    cost_mult, speed_mult = liquidity_multipliers(adv, liq_cfg)

            if len(cov_assets) == 0 or betas is None or betas.empty:
                continue

            pos = int(bar_positions[i])
            score_changed = False
            for h, comp in composites.items():
                if pos % cadence[h] == 0 and ts in comp.index:
                    held_scores[h] = comp.loc[ts]
                    score_changed = True

            if day_changed or score_changed or not target_valid:
                # Investable set: in covariance AND has betas
                assets = [a for a in cov_assets if a in betas.index]
                if len(assets) < PORT['min_assets']:
                    target_valid = False
                    continue

                # Alpha at this bar. `alpha` is the GP aim-discounted per-bar
                # Grinold alpha used by the MVO. `alpha_h` is the RAW (un-aim-
                # discounted) edge accumulated over each bucket's holding horizon
                # (raw per-bar contribution * h_b) - the expected residual return
                # captured before the position turns over, i.e. the horizon on
                # which it must out-earn a round-trip cost. It feeds the no-trade
                # gate only, and uses ic_scale_raw so trade speed never vetoes a
                # profitable signal.
                alpha = pd.Series(0.0, index=assets)
                alpha_h = pd.Series(0.0, index=assets)
                got_signal = False
                for h, score in held_scores.items():
                    z = score.reindex(assets)
                    if z.notna().sum() < PORT['min_assets']:
                        continue
                    sig_z = sigma.reindex(assets) * z.fillna(0.0)
                    alpha = alpha.add(ic_scale[h] * sig_z, fill_value=0.0)
                    alpha_h = alpha_h.add(
                        ic_scale_raw[h] * sig_z * max(bucket_h.get(h, 1.0), 1.0),
                        fill_value=0.0)
                    got_signal = True
                if not got_signal or alpha.abs().sum() < 1e-15:
                    target_valid = False
                    continue

                # No-trade zone ("lazy trading"): zero a name's alpha when its RAW
                # expected horizon edge |alpha_h_i| (no aim discount) can't clear
                # no_trade_mult * the per-name per-side cost (cost_rate *
                # cost_mult_i). The MVO then won't allocate fresh risk to names
                # whose signal can't pay for a round trip; any existing position is
                # unwound by the trade-rate step.
                if no_trade_mult > 0.0:
                    cm = (cost_mult.reindex(assets).fillna(1.0) if liq_on
                          else pd.Series(1.0, index=assets))
                    band = no_trade_mult * cost_rate * cm
                    alpha = alpha.where(alpha_h.abs() >= band, 0.0)
                    if alpha.abs().sum() < 1e-15:
                        target_valid = False
                        continue

                if cov is None:
                    cov = shrunk_covariance(cov_hist,
                                            PORT['cov_min_observations'],
                                            PORT['shrinkage'])
                    # Soft cluster-exposure penalty (same trailing window, causal)
                    if cluster_cfg.get('enabled') and cov is not None and not cov.empty:
                        clusters = residual_clusters(
                            cov_hist[cov.index],
                            corr_threshold=cluster_cfg['corr_threshold'],
                            min_cluster_size=cluster_cfg['min_cluster_size'])
                        if clusters:
                            cov = cluster_penalty_matrix(cov, clusters,
                                                         lam=cluster_cfg['lambda'])
                if cov is None or cov.empty:
                    target_valid = False
                    continue
                assets = [a for a in cov.index if a in alpha.index]
                if len(assets) < PORT['min_assets']:
                    target_valid = False
                    continue
                alpha = alpha.reindex(assets).fillna(0.0)

                # Constraints: dollar + factor-beta neutrality
                cons = {'dollar': pd.Series(1.0, index=assets)}
                for n in FACTOR_NAMES:
                    if n in PORT['neutrality']:
                        cons[n] = betas[f'beta_{n}'].reindex(assets).fillna(0.0)
                A = pd.DataFrame(cons)
                # Per-constraint neutrality bands, aligned to A's columns; each
                # exposure is held within +/-band rather than at exactly zero
                # (missing/0 -> exact). See portfolio.neutrality_band.
                band_cfg = PORT.get('neutrality_band', {})
                bands = np.array([band_cfg.get(c, 0.0) for c in A.columns],
                                 dtype=float)

                cov_a = cov.loc[assets, assets]
                # Risk diagnostics for this rebalance: eigen-concentration of the
                # optimized covariance, the active cluster count, and the cov
                # submatrix used for per-bar predicted variance below.
                cov_target_values = cov_a.values
                _eig = np.linalg.eigvalsh(cov_target_values)
                _eig = _eig[_eig > 0]
                if _eig.size:
                    cov_eig_top_share = float(_eig[-1] / _eig.sum())
                    cov_condition = float(_eig[-1] / _eig[0])
                else:
                    cov_eig_top_share = cov_condition = np.nan
                n_clusters_day = len(clusters) if clusters else 0
                max_cluster_day = max((len(c) for c in clusters), default=0)

                # Same alpha, neutrality (A), cap and gross for both schemes;
                # only the risk model differs. 'equal_weight' ignores cov_a and
                # weights by alpha rank (covariance-free benchmark).
                if self._weight_scheme == 'equal_weight':
                    w_target = solve_equal_weight(alpha, A,
                                                  max_position=cap,
                                                  gross_leverage=gross_target,
                                                  bands=bands)
                else:
                    w_target = solve_constrained_mvo(alpha, cov_a, A,
                                                     max_position=cap,
                                                     gross_leverage=gross_target,
                                                     bands=bands)
                target_index = pd.Index(w_target.index)
                target_values = w_target.values.astype(float, copy=False)
                alpha_values = alpha.reindex(target_index).fillna(0.0).values
                # Per-factor betas aligned to the held names (for the realized
                # exposure / neutrality acceptance check below).
                beta_values = {
                    n: betas[f'beta_{n}'].reindex(target_index).fillna(0.0).values
                    for n in FACTOR_NAMES if f'beta_{n}' in betas.columns
                }
                fwd_col_positions = fwd_columns.get_indexer(target_index)

                if liq_on:
                    cost_values = cost_mult.reindex(target_index).fillna(1.0).values
                    speed_values = speed_mult.reindex(target_index).fillna(1.0).values
                else:
                    cost_values = np.ones(len(target_index), dtype=float)
                    speed_values = np.ones(len(target_index), dtype=float)

                if len(held_index):
                    prev_pos = held_index.get_indexer(target_index)
                    w_prev_values = np.zeros(len(target_index), dtype=float)
                    valid_prev = prev_pos >= 0
                    if valid_prev.any():
                        w_prev_values[valid_prev] = held_values[prev_pos[valid_prev]]
                    closed_mask = ~held_index.isin(target_index)
                    if closed_mask.any():
                        closed_names = held_index[closed_mask]
                        closed_abs = np.abs(held_values[closed_mask])
                        closed_abs_sum = float(closed_abs.sum())
                        if liq_on:
                            closed_cm = cost_mult.reindex(closed_names).fillna(1.0).values
                            closed_cost_sum = float((closed_abs * closed_cm).sum())
                        else:
                            closed_cost_sum = closed_abs_sum
                    else:
                        closed_abs_sum = 0.0
                        closed_cost_sum = 0.0
                else:
                    w_prev_values = np.zeros(len(target_index), dtype=float)
                    closed_abs_sum = 0.0
                    closed_cost_sum = 0.0

                Av = A.reindex(target_index).values
                try:
                    neutralizer = np.linalg.solve(Av.T @ Av, Av.T)
                except np.linalg.LinAlgError:
                    neutralizer = None
                target_valid = True
            else:
                w_prev_values = held_values

            # Trade toward the aim at the GP rate, but never more than the
            # per-bar turnover budget: the voluntary trade is a_eff * |aim - held|,
            # so cap a_eff at budget / gap. Convex step preserves neutrality and
            # the cap; gross still floats below the aim. Then re-neutralize.
            gap = float(np.abs(target_values - w_prev_values).sum())
            a_eff = min(smooth_alpha, per_bar_to_budget / gap) if gap > 1e-12 else smooth_alpha
            # Liquidity-aware trade SPEED: liquid names fill toward the aim
            # faster, illiquid ones slower (impact persists -> trade slowly).
            # Per-name rate a_i = clip(a_eff * speed_mult_i, 0, 1); re-throttle
            # if the per-name trade breaches the turnover budget.
            if liq_on:
                a_vec = np.clip(a_eff * speed_values, 0.0, 1.0)
                trade = a_vec * (target_values - w_prev_values)
                vol_to = float(np.abs(trade).sum())
                if vol_to > per_bar_to_budget and vol_to > 1e-12:
                    trade = trade * (per_bar_to_budget / vol_to)
                w_new_values = w_prev_values + trade
            else:
                w_new_values = (1 - a_eff) * w_prev_values + a_eff * target_values
            cap_w = cap * gross_target * 0.999
            if neutralizer is not None:
                v = w_new_values
                for _ in range(3):
                    v = np.clip(v, -cap_w, cap_w)
                    v = v - Av @ (neutralizer @ v)
                w_new_values = v
            # Gross leverage is a soft CEILING, not a per-bar peg: only ever
            # scaled DOWN if it exceeds the gross target. The book floats below
            # the target and converges toward it as it tracks a stable aim -
            # gross 1 is the long-term destination, approached gradually, never
            # forced each bar.
            g = float(np.abs(w_new_values).sum())
            if g > 1e-12:
                w_new_values = w_new_values * min(1.0, gross_target / g)

            # Positions in symbols that left the investable set get closed
            traded = np.abs(w_new_values - w_prev_values)
            turnover = float(traded.sum() + closed_abs_sum)
            # Liquidity-aware $ cost: per-name turnover * cost_rate * cost_mult_i
            # (illiquid names cost more per unit traded). Flat when disabled.
            if liq_on:
                trade_cost = float((traded * cost_values).sum() +
                                   closed_cost_sum) * cost_rate
            else:
                trade_cost = turnover * cost_rate

            fwd_row = int(fwd_row_positions[i])
            if fwd_row < 0:
                continue
            fwd_values = np.zeros(len(target_index), dtype=float)
            valid_fwd_cols = fwd_col_positions >= 0
            if valid_fwd_cols.any():
                vals = fwd_values_all[fwd_row, fwd_col_positions[valid_fwd_cols]]
                fwd_values[valid_fwd_cols] = np.nan_to_num(vals, nan=0.0)
            gross_ret = float(w_new_values @ fwd_values)
            net_ret = gross_ret - trade_cost
            # Expected (model) per-bar gross alpha for the half-alpha diagnostic
            # (costs should consume ~half of it at the optimum).
            exp_alpha = float(alpha_values @ w_new_values)

            # Execution-fragility stress: same book decided impl_lag bars
            # earlier, earning this bar (costs ~identical, just shifted)
            if len(w_history) >= impl_lag:
                hist_index, hist_values = w_history[-impl_lag]
                hist_pos = hist_index.get_indexer(target_index)
                lag_values = np.zeros(len(target_index), dtype=float)
                valid_hist = hist_pos >= 0
                if valid_hist.any():
                    lag_values[valid_hist] = hist_values[hist_pos[valid_hist]]
                lag_ret = float(lag_values @ fwd_values) - trade_cost
            else:
                lag_ret = np.nan

            # Realized exposure to every neutralized factor (should be ~0).
            exposures = {n: float(w_new_values @ bv) for n, bv in beta_values.items()}

            row = {'timestamp': ts, 'gross_return': gross_ret,
                   'net_return': net_ret, 'net_return_lag1': lag_ret,
                   'turnover': float(turnover), 'trade_cost': trade_cost,
                   'exp_alpha': exp_alpha,
                   'gross_exposure': float(np.abs(w_new_values).sum()),
                   'net_exposure': float(w_new_values.sum()),
                   'mkt_exposure': exposures.get('market', np.nan),
                   'size_exposure': exposures.get('size', np.nan),
                   'n_positions': int((np.abs(w_new_values) > 1e-6).sum())}
            for n in EXTRA_EXPOSURE_FACTORS:
                row[f'{n}_exposure'] = exposures.get(n, np.nan)
            rows.append(row)

            # Per-name held weights (nonzero) + a per-bar risk row.
            nz = np.abs(w_new_values) > 1e-6
            if nz.any():
                weight_rows.append((ts, target_index[nz].to_numpy(),
                                    w_new_values[nz]))
            if cov_target_values.size:
                pred_var = float(w_new_values @ cov_target_values @ w_new_values)
            else:
                pred_var = np.nan
            n_capped = int((np.abs(w_new_values) >= cap_w - 1e-9).sum())
            risk_rows.append({
                'timestamp': ts,
                'pred_vol_ann': (np.sqrt(max(pred_var, 0.0)) * np.sqrt(ppy)
                                 if np.isfinite(pred_var) else np.nan),
                'gross': float(np.abs(w_new_values).sum()),
                'n_positions': int(nz.sum()),
                'n_at_cap': n_capped,
                'cov_eig_top_share': cov_eig_top_share,
                'cov_condition': cov_condition,
                'n_clusters': n_clusters_day,
                'max_cluster_size': max_cluster_day,
                'n_assets': int(len(target_index)),
            })
            held_index = target_index
            held_values = w_new_values
            w_history.append((held_index, held_values.copy()))
            if len(w_history) > impl_lag:
                w_history.pop(0)
            closed_abs_sum = 0.0
            closed_cost_sum = 0.0

        # Persist the final book so the next contiguous window starts from it
        # (per scheme - MVO and equal-weight keep independent position history).
        self._carry[self._weight_scheme] = pd.Series(held_values, index=held_index)
        if not rows:
            return None

        # Materialize per-bar weights (vectorized: repeat each bar's timestamp
        # across its held names) and the per-bar risk table for persistence.
        if weight_rows:
            counts = [len(s) for _, s, _ in weight_rows]
            self._bt_weights = pd.DataFrame({
                'timestamp': np.repeat(
                    np.array([t for t, _, _ in weight_rows], dtype='datetime64[ns]'),
                    counts),
                'symbol': np.concatenate([s for _, s, _ in weight_rows]),
                'weight': np.concatenate([w for _, _, w in weight_rows]),
            })
        if risk_rows:
            self._bt_risk = pd.DataFrame(risk_rows)
        return pd.DataFrame(rows).set_index('timestamp')

    # ---------------- driver ----------------

    def run(self, include_lockbox: bool = False) -> pd.DataFrame:
        from tqdm import tqdm

        full_start = pd.to_datetime(WF['start_date'])
        full_end = pd.to_datetime(WF['end_date'])
        lockbox = int(WF.get('lockbox_months', 0))
        if lockbox and not include_lockbox:
            full_end = full_end - pd.DateOffset(months=lockbox)
            logging.info(f"LOCKBOX: holding out the final {lockbox} months "
                         f"(test ends {full_end.date()}). Run --lockbox ONCE "
                         f"before any live decision.")
        train_months = WF['train_months']
        test_days = WF['test_days']

        schedule = []
        t0 = full_start
        while True:
            t1 = t0 + pd.DateOffset(months=train_months)
            t2 = t1 + pd.DateOffset(days=test_days)
            if t2 > full_end:
                break
            schedule.append((t0, t1, t2))
            t0 = t0 + pd.DateOffset(days=test_days)

        logging.info(f"Walk-forward: {len(schedule)} windows "
                     f"({train_months}mo train / {test_days}d test)")
        self._set_context_bounds(schedule)
        self.ctx = None

        # Progressive persistence: each window is checkpointed to the DB the
        # moment it finishes (via _save_window below), so a long or interrupted
        # run leaves queryable partial results instead of writing everything
        # only at the very end. Clear any prior run's rows up front so the
        # appended tables build a clean set.
        from datetime import datetime
        self._run_ts = datetime.now()
        self._cum_wealth = 1.0
        self._peak_wealth = 1.0
        self._carry = {}  # fresh run starts flat (per scheme)
        self._reset_backtest_tables()

        prev = None
        for i, (t0, t1, t2) in enumerate(tqdm(schedule, desc="Windows")):
            res = self.run_window(i, t0, t1, t2, prev)
            if res is not None:
                self.windows.append(res)
                self._save_window(res)
                if res.selected:
                    prev = res

        oos = [w.oos_returns for w in self.windows if w.oos_returns is not None]
        if not oos:
            logging.error("No OOS returns produced")
            return pd.DataFrame()

        returns = pd.concat(oos).sort_index()
        returns['cum_return'] = (1 + returns['net_return']).cumprod() - 1
        peak = (1 + returns['cum_return']).cummax()
        returns['drawdown'] = (1 + returns['cum_return']) / peak - 1
        return returns

    @staticmethod
    def _reset_backtest_tables() -> None:
        for table_name in BACKTEST_TABLES + LEGACY_BACKTEST_TABLES:
            delete_table(table_name)

    @staticmethod
    def _clear_window_outputs(window_idx: int) -> None:
        delete_rows_where('wf_portfolio_windows', 'window_idx', window_idx)
        delete_rows_where(PORTFOLIO_RETURNS_TABLE, 'window', window_idx)
        delete_rows_where(PORTFOLIO_RETURNS_BENCH_TABLE, 'window', window_idx)
        delete_rows_where('wf_portfolio_exposures', 'window', window_idx)
        delete_rows_where(SIGNAL_ATTRIBUTION_TABLE, 'window_idx', window_idx)
        delete_rows_where(WEIGHTS_TABLE, 'window', window_idx)
        delete_rows_where(RISK_TABLE, 'window', window_idx)

    def summary(self, returns: pd.DataFrame):
        ppy = BARS_PER_DAY * 365
        r = returns['net_return']
        ann_ret = r.mean() * ppy
        ann_vol = r.std() * np.sqrt(ppy)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0

        # Execution fragility: the same strategy traded one bar late
        rl = returns['net_return_lag1'].dropna()
        sharpe_lag = (rl.mean() / rl.std() * np.sqrt(ppy)) if rl.std() > 0 else 0.0

        traded = [w for w in self.windows if w.oos_returns is not None]
        avg_mkt = np.nanmean([w.avg_abs_mkt_exposure for w in traded]) if traded else np.nan
        avg_size = np.nanmean([w.avg_abs_size_exposure for w in traded]) if traded else np.nan
        avg_gross = np.nanmean([w.avg_gross for w in traded]) if traded else np.nan
        avg_to = np.nanmean([w.avg_turnover for w in traded]) if traded else np.nan

        print("\n" + "=" * 70)
        print("WALK-FORWARD MARKET-NEUTRAL PORTFOLIO")
        print(f"Sizing: {self.weight_scheme}" +
              (f"   (benchmark foil: {self.benchmark_scheme})"
               if self.compare_benchmark else ""))
        print("=" * 70)
        print(f"Annual return:      {ann_ret * 100:+.1f}%")
        print(f"Annual vol:         {ann_vol * 100:.1f}%")
        print(f"Sharpe:             {sharpe:.2f}")
        print(f"Sharpe (1-bar lag): {sharpe_lag:.2f}  (execution-fragility stress)")
        print(f"Max drawdown:       {returns['drawdown'].min() * 100:.1f}%")
        print(f"Windows traded:     {len(traded)}/{len(self.windows)}")
        print(f"Avg |mkt beta exp|: {avg_mkt:.4f}  (market-neutrality check, ~0)")
        print(f"Avg |size exp|:     {avg_size:.4f}  (size-neutrality check, ~0)")
        print(f"Avg gross:          {avg_gross:.4f}")
        print(f"Avg turnover/bar:   {avg_to:.4f}")

        # Benchmark (foil) sizing: same selection/alpha/neutrality, different
        # risk model. production-minus-benchmark = the value the production risk
        # model adds; <= 0 means the foil sizing is as good or better.
        bench_oos = [w.oos_returns_bench for w in self.windows
                     if w.oos_returns_bench is not None and not w.oos_returns_bench.empty]
        if bench_oos:
            b_r = pd.concat(bench_oos).sort_index()['net_return']
            b_ann = b_r.mean() * ppy
            b_vol = b_r.std() * np.sqrt(ppy)
            b_sharpe = b_ann / b_vol if b_vol > 0 else 0.0
            print(f"\nBenchmark sizing '{self.benchmark_scheme}', same selection/alpha:")
            print(f"  benchmark annual return: {b_ann * 100:+.1f}%   "
                  f"benchmark Sharpe: {b_sharpe:.2f}   "
                  f"(production '{self.weight_scheme}' Sharpe {sharpe:.2f}; "
                  f"production-minus-benchmark = {sharpe - b_sharpe:+.2f})")
        # Per-window detail as an aligned table. The f/m/s triplets are the
        # fast/mid/slow holding-lag buckets; top families are abbreviated to the
        # three largest (full breakdown lives in the notebook's attribution view).
        order = ['fast', 'mid', 'slow']
        row = ("  {win:<4}{period:<24}{nsig:>5}{bkt:>10}{hb:>12}{eff:>14}"
               "{sr:>8}{bchsr:>8}{rho:>8}{sign:>7}{mkt:>9}  {fam}")
        print(f"\nPer-window detail (f/m/s = fast/mid/slow buckets; "
              f"OOS_SR = production '{self.weight_scheme}', "
              f"BCH_SR = benchmark '{self.benchmark_scheme}'):")
        print(row.format(win='Win', period='OOS test period', nsig='#sig',
                         bkt='sigs', hb='h(bars)', eff='effN', sr='OOS_SR',
                         bchsr='BCH_SR', rho='IC_rho', sign='sign', mkt='|mkt|',
                         fam='top families'))
        for w in self.windows:
            tri = lambda d, f='{}': '/'.join(f.format(d[b]) if b in d else '-'
                                             for b in order)
            sel = {b: len(s) for b, s in w.selected.items()}
            fam = list(self._selected_family_counts(w).items())
            fam_str = ' '.join(f'{k}:{v}' for k, v in fam[:3])
            if len(fam) > 3:
                fam_str += f' (+{len(fam) - 3})'
            mkt = w.avg_abs_mkt_exposure if not np.isnan(w.avg_abs_mkt_exposure) else 0.0
            print(row.format(
                win=f'W{w.window_idx:02d}',
                period=f'{w.train_end.date()}-{w.test_end.date()}',
                nsig=sum(sel.values()),
                bkt=tri(sel),
                hb=tri({b: int(v) for b, v in w.bucket_h.items()}),
                eff=tri({b: v for b, v in w.eff_breadth.items()}, '{:.1f}'),
                sr=f'{w.oos_sharpe:.2f}',
                bchsr=(f'{w.oos_sharpe_bench:.2f}'
                       if np.isfinite(w.oos_sharpe_bench) else '-'),
                rho=f'{w.train_oos_ic_rank_corr:.2f}',
                sign=f'{w.train_oos_ic_sign_accuracy:.0%}',
                mkt=f'{mkt:.4f}',
                fam=fam_str))

    def _selected_family_counts(self, w: WindowResult) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for sigs in w.selected.values():
            for sig in sigs:
                cat = self.registry.get(sig, {}).get('category', 'unknown')
                out[cat] = out.get(cat, 0) + 1
        return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))

    def _window_row(self, w: WindowResult) -> dict:
        """One row of per-window selection / OOS diagnostics for wf_portfolio_windows."""
        family_counts = self._selected_family_counts(w)
        return {
            'window_idx': w.window_idx,
            'train_start': w.train_start, 'train_end': w.train_end,
            'test_end': w.test_end,
            'n_candidates': w.n_candidates,
            'n_after_fdr': w.selection_counts.get('after_fdr', 0),
            'n_after_gates': w.selection_counts.get('after_gates', 0),
            'n_selected': w.selection_counts.get('selected', 0),
            'selected': ';'.join(f"{b}:{','.join(s[:10])}" for b, s in w.selected.items()),
            'selected_families': ';'.join(f"{k}:{v}" for k, v in family_counts.items()),
            'horizon_ic': ';'.join(f"{b}:{ic:.4f}" for b, ic in w.horizon_ic.items()),
            'bucket_holding_lag': ';'.join(f"{b}:{v:.0f}" for b, v in w.bucket_h.items()),
            'eff_breadth': ';'.join(f"{b}:{v:.1f}" for b, v in w.eff_breadth.items()),
            'oos_sharpe': w.oos_sharpe,
            'train_oos_ic_rank_corr': w.train_oos_ic_rank_corr,
            'train_oos_ic_sign_accuracy': w.train_oos_ic_sign_accuracy,
            'avg_gross': w.avg_gross,
            'avg_turnover': w.avg_turnover,
            'avg_abs_mkt_exposure': w.avg_abs_mkt_exposure,
            'avg_abs_size_exposure': w.avg_abs_size_exposure,
            'cost_to_alpha': w.cost_to_alpha,
            'net_to_gross': w.net_to_gross,
            'run_timestamp': self._run_ts,
        }

    def _save_window(self, res: WindowResult):
        """Refresh one window's results in the DB the moment it finishes.

        Output tables are reset at run startup, and this method also deletes
        any existing rows for `res.window_idx` before appending. That makes each
        window checkpoint equivalent to a fresh write for that window."""
        self._clear_window_outputs(res.window_idx)

        # Window diagnostics row is written for every window, including
        # non-trading ones (n_selected == 0, no oos_returns).
        save_data('wf_portfolio_windows', pd.DataFrame([self._window_row(res)]),
                  mode='append',
                  datetime_columns=['train_start', 'train_end', 'test_end',
                                    'run_timestamp'])

        # Per-signal OOS attribution (written for every window with a selection).
        if res.signal_attribution:
            attr = pd.DataFrame(res.signal_attribution)
            attr['test_start'] = res.train_end
            attr['test_end'] = res.test_end
            attr['run_timestamp'] = self._run_ts
            save_data(SIGNAL_ATTRIBUTION_TABLE, attr, mode='append',
                      datetime_columns=['test_start', 'test_end', 'run_timestamp'])

        if res.oos_returns is None or res.oos_returns.empty:
            return

        chunk = res.oos_returns
        wealth = self._cum_wealth * (1.0 + chunk['net_return']).cumprod()
        running_peak = np.maximum(self._peak_wealth, wealth.cummax())
        eq = chunk.copy()
        eq['cum_return'] = wealth - 1.0
        eq['drawdown'] = wealth / running_peak - 1.0
        self._cum_wealth = float(wealth.iloc[-1])
        self._peak_wealth = float(running_peak.iloc[-1])
        eq = eq.reset_index().rename(columns={'index': 'timestamp'})
        eq['window'] = res.window_idx
        eq['run_timestamp'] = self._run_ts
        save_data(PORTFOLIO_RETURNS_TABLE, eq, mode='append',
                  datetime_columns=['timestamp', 'run_timestamp'])

        exp = (chunk[['mkt_exposure', 'size_exposure'] + EXTRA_EXPOSURE_COLS]
               .assign(window=res.window_idx)
               .reset_index().rename(columns={'index': 'timestamp'}))
        exp['run_timestamp'] = self._run_ts
        save_data('wf_portfolio_exposures', exp, mode='append',
                  datetime_columns=['timestamp', 'run_timestamp'])

        # Benchmark (foil) sizing returns: raw per-bar streams; the notebook
        # cumulates and de-overlaps them exactly as for the production book.
        if res.oos_returns_bench is not None and not res.oos_returns_bench.empty:
            bench = (res.oos_returns_bench.reset_index()
                     .rename(columns={'index': 'timestamp'}))
            bench['window'] = res.window_idx
            bench['run_timestamp'] = self._run_ts
            save_data(PORTFOLIO_RETURNS_BENCH_TABLE, bench, mode='append',
                      datetime_columns=['timestamp', 'run_timestamp'])

        # Per-bar held weights (enables exact position-cap audits + per-name PnL)
        # and per-bar predicted risk / covariance diagnostics.
        if res.oos_weights is not None and not res.oos_weights.empty:
            wts = res.oos_weights.copy()
            wts['window'] = res.window_idx
            wts['run_timestamp'] = self._run_ts
            save_data(WEIGHTS_TABLE, wts, mode='append',
                      datetime_columns=['timestamp', 'run_timestamp'])
        if res.oos_risk is not None and not res.oos_risk.empty:
            rk = res.oos_risk.copy()
            rk['window'] = res.window_idx
            rk['run_timestamp'] = self._run_ts
            save_data(RISK_TABLE, rk, mode='append',
                      datetime_columns=['timestamp', 'run_timestamp'])


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Walk-forward market-neutral portfolio')
    parser.add_argument('--lockbox', action='store_true',
                        help='Include the held-out lockbox months (run ONCE, '
                             'right before a live decision)')
    parser.add_argument('--no-benchmark', action='store_true',
                        help='Skip the benchmark (foil) sizing pass '
                             '(faster production-only run)')
    args = parser.parse_args()

    wf = WalkForwardPortfolio()
    if args.no_benchmark:
        wf.compare_benchmark = False
    # run() checkpoints each window to the DB (wf_portfolio_*) as it traverses.
    returns = wf.run(include_lockbox=args.lockbox)
    if returns.empty:
        print("No portfolio produced: no window had defensible tradable signals")
        return
    wf.summary(returns)


if __name__ == '__main__':
    main()
