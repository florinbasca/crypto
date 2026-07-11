"""
Walk-Forward Market-Neutral Portfolio - the per-month mirror of discovery.

One window per discovery roll. Roll r's signals were discovered on its train +
select months, which end exactly where its OOS month begins (with a purge so
forward targets cannot cross the boundary) - the traded month was never seen
by the discovery that produced its signals. Each OOS month trades exactly its
roll's promoted signals; no scoring pipeline, no re-selection.

1. COMBINE: each signal is recomputed on the OOS month and z-scored per bar;
   signals sharing a holding lag are averaged into one composite, weighted by
   their discovery train IC. A composite becomes an expected residual return
   per name by scaling with the name's residual vol and the bucket's IC,
   discounted for how much of it a slow book captures before it decays.
2. OPTIMIZE: mean-variance on a daily-refreshed shrunk covariance. The
   portfolio is optimized to be dollar- and factor-neutral within the
   configured bands (portfolio.neutrality_band), per-name cap, gross 1.
3. BACKTEST on raw forward returns, reported gross AND net of per-side
   trading costs, participation caps, the turnover budget, and perp funding.

Outputs: wf_portfolio_returns, wf_portfolio_windows, wf_portfolio_exposures.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import logging
import time
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
                                              residual_clusters,
                                              cluster_penalty_matrix)
from research.lib.signal_eval import (build_registry, compute_signal_panel,
                                      effective_halflife_for,
                                      signal_feature_columns,
                                      load_universe_membership,
                                      universe_member_mask, SCREENING_GRID)

warnings.filterwarnings('ignore')
# INFO to stdout; force=True so it wins even if a module configured first.
logging.basicConfig(level=logging.INFO,
                    stream=sys.stdout,
                    force=True,
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
PORTFOLIO_RETURNS_TABLE = 'wf_portfolio_returns'
SIGNAL_ATTRIBUTION_TABLE = 'wf_portfolio_signal_attribution'
WEIGHTS_TABLE = 'wf_portfolio_weights'       # per-bar held weight per name
RISK_TABLE = 'wf_portfolio_risk'             # per-bar predicted risk / cov diagnostics
BACKTEST_TABLES = (PORTFOLIO_RETURNS_TABLE,
                   'wf_portfolio_windows',
                   'wf_portfolio_exposures', SIGNAL_ATTRIBUTION_TABLE,
                   WEIGHTS_TABLE, RISK_TABLE)
# Retired output tables, cleaned up on reset so they can't go stale.
LEGACY_BACKTEST_TABLES = ('wf_portfolio_equity', 'wf_portfolio_returns_ew',
                          'wf_portfolio_returns_bench')


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


def effective_fill_rate() -> Tuple[float, float]:
    """(nominal per-bar trade rate, effective fill rate kappa).

    The nominal rate is the GP cost-responsive trade-toward-aim rate; kappa is
    the rate at which the book can ACTUALLY close the gap to the aim once the
    hard `max_annual_turnover` budget is accounted for (at realistic settings
    the budget binds long before the nominal rate). This one quantity drives
    the aim discount in `_backtest_window`.
    """
    gp = PORT.get('gp_trading', {})
    nominal = gp_trade_rate(PORT['cost_bps'], gp,
                            PORT['weight_smoothing_halflife'])
    kappa = max(nominal, 1e-9)
    max_ann_to = PORT.get('max_annual_turnover')
    per_bar_budget = (max_ann_to / (BARS_PER_DAY * 365)
                      if max_ann_to else np.inf)
    if gp.get('discount_at_realized_rate', True) and np.isfinite(per_bar_budget):
        realized = per_bar_budget / max(PORT['gross_leverage'], 1e-9)
        kappa = max(min(kappa, realized), 1e-9)
    return nominal, kappa


def edge_gross_multiplier(exp_edge: float, rt_cost: float,
                          edge_mult: float, min_mult: float = 0.0) -> float:
    """Gross scale for the aim book: clip(expected horizon edge per unit gross
    / (edge_mult x round-trip cost per unit gross), 0, 1). Full size only when
    the aim's expected edge covers edge_mult round trips; shrinks toward zero
    (instead of trading a full-gross book on alpha that cannot pay for
    itself). A multiplier below min_mult snaps to 0.0: deploying a sliver of
    gross on sub-cost edge still pays full RELATIVE costs (observed
    cost_to_alpha of 10^3-10^4 on 1e-5-gross books), so below the floor the
    aim is zeroed and the book unwinds instead of bleeding costs.
    rt_cost <= 0 (free execution) -> always 1."""
    if rt_cost <= 0:
        return 1.0
    m = float(np.clip(exp_edge / (edge_mult * rt_cost), 0.0, 1.0))
    return m if m >= min_mult else 0.0


def cost_holding_bars(lag_bars: float, turnover: Optional[float],
                      max_bars: Optional[float] = None) -> float:
    """Turnover-implied holding/persistence in bars: lag / per-rebalance
    turnover, capped at portfolio.cost_holding_max_bars.

    Cost is paid per unit TRADED, not per unit time: a bucket that rebalances
    a fraction `turnover` of its book per scoring lag holds a position for
    ~lag/turnover bars (low turnover <=> persistent z-scores, so the per-bar
    edge persists as long as the position does). turnover ~1 -> the scoring
    lag; turnover > 1 (clipped at 2) -> less; missing/invalid turnover ->
    the scoring lag."""
    if max_bars is None:
        max_bars = get('portfolio.cost_holding_max_bars', 1008)
    h = max(float(lag_bars), 1.0)
    if turnover is None or not np.isfinite(turnover) or turnover <= 0:
        return h
    cap = max(float(max_bars), h / 2.0)
    return float(min(h / min(float(turnover), 2.0), cap))




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


def participation_caps(avg_dollar_vol: np.ndarray, max_participation: float,
                       book_size_usd: float) -> np.ndarray:
    """Per-name per-bar trade cap in WEIGHT units from the trailing average
    bar $ volume: max |dw_i| = max_participation * avg_$vol_i / book_size_usd.
    Missing (NaN) or non-positive volume -> 0 (name not tradeable this bar)."""
    caps = (np.asarray(avg_dollar_vol, dtype=float)
            * (float(max_participation) / float(book_size_usd)))
    return np.where(np.isfinite(caps) & (caps > 0.0), caps, 0.0)


def clamp_to_participation(w_new: np.ndarray, w_prev: np.ndarray,
                           max_dw: np.ndarray) -> np.ndarray:
    """Clamp each name's trade to its participation cap: the new weight may
    not move more than max_dw from the held weight."""
    return w_prev + np.clip(w_new - w_prev, -max_dw, max_dw)

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
    bucket_to: Dict[str, float] = field(default_factory=dict)      # bucket -> per-rebalance turnover
    n_candidates: int = 0
    oos_returns: Optional[pd.DataFrame] = None
    oos_sharpe: float = np.nan
    is_sharpe: float = np.nan
    avg_gross: float = np.nan
    avg_turnover: float = np.nan
    avg_abs_mkt_exposure: float = np.nan
    avg_abs_size_exposure: float = np.nan
    cost_to_alpha: float = np.nan   # half-alpha: costs / expected gross alpha
    net_to_gross: float = np.nan    # realized: sum(net) / sum(gross)
    oos_weights: Optional[pd.DataFrame] = None   # per-bar held weight per name
    oos_risk: Optional[pd.DataFrame] = None       # per-bar predicted risk / cov diagnostics


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
                        time_col: str = 'timestamp',
                        pad_end: Optional[pd.Timedelta] = None) -> pd.DataFrame:
            lf = _scan(table_name)
            if lf is None:
                return pd.DataFrame()
            if start is not None:
                lf = lf.filter(pl.col(time_col) >= start)
            if end is not None:
                lf = lf.filter(pl.col(time_col) <
                               (end + pad_end if pad_end is not None else end))
            t0 = time.perf_counter()
            logging.info(f"  loading {table_name}.{value_col} wide...")
            df = lf.select([time_col, 'symbol', value_col]).collect()
            if df.is_empty():
                return pd.DataFrame()
            wide = df.pivot(index=time_col, on='symbol', values=value_col,
                            aggregate_function='first').sort(time_col)
            out = wide.to_pandas().set_index(time_col)
            out.columns.name = 'symbol'
            logging.info(f"  {table_name}.{value_col}: {out.shape[0]:,} bars x "
                         f"{out.shape[1]} symbols ({time.perf_counter()-t0:.0f}s)")
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

        # Point-in-time membership intervals when available (etl/universe.py
        # maintains universe_membership); the static current set is the
        # fallback. The initial cohort is seeded from the data start, so
        # behaviour only diverges once listing/delisting snapshots accrue.
        self.membership_intervals = None
        self._members_cache: Dict[pd.Timestamp, set] = {}
        mem = load_universe_membership()
        if (mem is not None and 'valid_from' in mem.columns
                and 'valid_to' in mem.columns):
            self.membership_intervals = mem[['symbol', 'valid_from', 'valid_to']]

        # First data bar per name (proxy for listing date), from the FULL
        # residual history - the context window is warmup-trimmed, so its own
        # first rows would fake every name as newly listed. Feeds the
        # walk_forward.min_listing_age_days survivorship sensitivity gate.
        self.first_bar = None
        if int(WF.get('min_listing_age_days', 0) or 0) > 0:
            lf = _scan('residual_returns')
            if lf is not None:
                fb = (lf.group_by('symbol')
                        .agg(pl.col('timestamp').min().alias('first_bar'))
                        .collect().to_pandas())
                self.first_bar = pd.to_datetime(
                    fb.set_index('symbol')['first_bar'])

        # Per-name dollar-volume panel (for liquidity-aware costs / trade speed
        # and the volume-participation trade cap).
        # quote_asset_volume is the bar's $ traded; ADV is a trailing mean of it.
        # Optional: if absent the backtest falls back to flat costs / scalar
        # speed and NO participation cap (warned loudly - the cap silently
        # vanishing would overstate capacity).
        self.dollar_vol_wide = None
        try:
            pv = _wide_panel('prices', 'quote_asset_volume')
            if not pv.empty:
                self.dollar_vol_wide = pv
        except Exception as e:  # missing column / table -> graceful fallback
            logging.warning(f"no dollar-volume panel ({e}); falling back to "
                            "flat costs / scalar trade speed / NO "
                            "volume-participation cap")

        # Funding-rate panel (settlement stamp x symbol) for the funding-PnL
        # accrual. Padded one bar past the context end: a settlement at exactly
        # test_end belongs to the last bar's forward interval (t, t+1bar].
        # Optional - missing table -> price PnL only, with a warning.
        self.funding_wide = None
        if PORT.get('funding_pnl'):
            try:
                fw = _wide_panel('funding_rates', 'funding_rate',
                                 pad_end=pd.Timedelta(BASE_FREQUENCY))
                if not fw.empty:
                    self.funding_wide = fw
                else:
                    logging.warning("funding_pnl enabled but funding_rates is "
                                    "empty - backtest will accrue NO funding")
            except Exception as e:
                logging.warning(f"funding_pnl: no funding panel ({e}); "
                                "backtest will accrue NO funding")

        logging.info(f"Panels: {self.res_wide.shape[0]:,} bars x "
                     f"{self.res_wide.shape[1]} symbols")

    def members_at(self, ts: pd.Timestamp) -> set:
        """Candidate set at `ts`: point-in-time interval lookup when the
        universe_membership table exists, else the static current set."""
        if self.membership_intervals is None:
            return self.membership
        day = ts.normalize()
        cached = self._members_cache.get(day)
        if cached is None:
            m = self.membership_intervals
            ok = (m['valid_from'] <= ts) & (m['valid_to'].isna() |
                                            (ts < m['valid_to']))
            cached = set(m.loc[ok, 'symbol'])
            self._members_cache[day] = cached
        return cached

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
        # Per-month mirror of discovery: no scoring pipeline. Each OOS month
        # trades ONLY the signals its roll promoted, sized from the metadata
        # discovery already measured (direction, lag, train IC, half-life,
        # turnover).
        self.registry = build_registry()
        if not self.registry:
            raise RuntimeError(
                "signal registry is empty - run "
                "research/signals/discovery.py first (promoted "
                "candidates are the only signal source)")
        self.month_meta = self._build_month_meta()
        if not self.month_meta:
            raise RuntimeError("no promotions found - run discovery first")
        self.ctx: Optional[DataContext] = None
        self._ctx_start: Optional[pd.Timestamp] = None
        self._ctx_end: Optional[pd.Timestamp] = None
        self.universe_members = load_universe_membership()
        self.windows: List[WindowResult] = []
        # Held book carried across contiguous windows (see _backtest_window).
        self._carry: pd.Series = pd.Series(dtype=float)

    def _build_month_meta(self) -> Dict[pd.Timestamp, List[dict]]:
        """{oos_start -> [{name, lag, ic, half_life, turnover}]} from the
        promotions table (one entry per signal per month it was promoted),
        joined with the ledger for that roll's train IC and turnover."""
        from research.signals.data import make_rolls
        tables = get('discovery.tables')
        promos = load_data(tables['promotions'])
        if promos is None or promos.empty:
            return {}
        led = load_data(tables['ledger'])
        led_key = {}
        if led is not None and not led.empty:
            for r in led.itertuples():
                led_key[(int(r.roll_id), r.cand_hash)] = (
                    float(getattr(r, 'train_ic_mean', np.nan) or np.nan),
                    float(getattr(r, 'turnover', np.nan) or np.nan))
        oos_by_roll = {r.roll_id: pd.Timestamp(r.oos_start)
                       for r in make_rolls(get('discovery'))}
        out: Dict[pd.Timestamp, List[dict]] = {}
        for _, r in promos.iterrows():
            name = f"disc_{r['family']}_{str(r['cand_hash'])[:10]}"
            oos = oos_by_roll.get(int(r['roll_id']))
            if oos is None or name not in self.registry:
                continue
            ic, to = led_key.get((int(r['roll_id']), r['cand_hash']),
                                 (np.nan, np.nan))
            hl = float(r.get('half_life_bars') or 0) or None
            out.setdefault(oos, []).append({
                'name': name,
                'lag': int(r.get('target_lag') or 0) or 6,
                'ic': ic, 'half_life': hl, 'turnover': to,
            })
        n = sum(len(v) for v in out.values())
        logging.info(f"per-month book: {n} promotions across {len(out)} "
                     f"OOS months")
        return out

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
        part_cfg = PORT.get('participation', {})
        part_days = int(np.ceil(int(part_cfg.get('volume_window_bars', 0))
                                / BARS_PER_DAY))
        warmup_days = max(int(PORT['cov_window_days']),
                          int(PORT['residual_vol_window_days']),
                          liq_days, part_days,
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
                         test_end: pd.Timestamp,
                         lag_of: Optional[Dict[str, int]] = None
                         ) -> Dict[str, pd.DataFrame]:
        """Per-horizon composite z-score panel (wide: timestamp x symbol) on the
        test window. Signals recomputed at full resolution with warmup, each
        smoothed at the per-lag effective halflife of its SELECTED lag
        (`lag_of`) so the traded signal matches the one that was scored."""
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
            hl = (effective_halflife_for(self.registry[sig], int(lag_of[sig]))
                  if lag_of and sig in lag_of else None)
            p = compute_signal_panel(sig, self.registry, features, halflife=hl)
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

        # This OOS month trades exactly its roll's promoted signals, sized
        # from discovery metadata. Buckets = distinct promoted lags; weights
        # within a bucket proportional to train IC (panels are already
        # direction-applied by compute_signal_panel).
        meta = self.month_meta.get(pd.Timestamp(train_end), [])
        selected, weights, bucket_ic, bucket_h = {}, {}, {}, {}
        bucket_to, bucket_hl = {}, {}
        by_lag: Dict[int, List[dict]] = {}
        for m in meta:
            by_lag.setdefault(m['lag'], []).append(m)
        for lag in sorted(by_lag):
            lab = f'{lag}b'
            ms = by_lag[lag]
            ics = np.array([m['ic'] if np.isfinite(m['ic']) else 0.0
                            for m in ms])
            ics = np.abs(ics)
            w = ics / ics.sum() if ics.sum() > 1e-12 else np.full(len(ms),
                                                                  1 / len(ms))
            selected[lab] = [m['name'] for m in ms]
            weights[lab] = {m['name']: float(wi) for m, wi in zip(ms, w)}
            bucket_ic[lab] = float(ics @ w)
            bucket_h[lab] = float(lag)
            tos = np.array([m['turnover'] if np.isfinite(m['turnover'])
                            else 1.0 for m in ms])
            bucket_to[lab] = float(tos @ w)
            hls = [(wi, m['half_life']) for m, wi in zip(ms, w)
                   if m['half_life']]
            bucket_hl[lab] = (float(sum(wi * h for wi, h in hls)
                                    / max(sum(wi for wi, _ in hls), 1e-12))
                              if hls else None)

        res.selected = selected
        res.weights = weights
        res.horizon_ic = bucket_ic
        res.bucket_h = bucket_h
        res.bucket_to = bucket_to
        res.n_candidates = len(meta)
        logging.info(f"W{idx:02d} OOS {train_end.date()}..{test_end.date()}: "
                     f"{len(meta)} promoted signals in "
                     f"{len(selected)} lag buckets")
        if not selected:
            logging.info(f"W{idx:02d}: no promotions this month - no new book "
                         f"(carried positions re-target next traded month)")
            return res

        feat_start = train_end - pd.Timedelta(days=WARMUP_DAYS)
        lag_of = {m['name']: m['lag'] for m in meta}
        composites = self.composite_scores(selected, weights, feat_start,
                                           train_end, test_end, lag_of=lag_of)
        if not composites:
            return res

        self._ensure_context()
        bt_returns = self._backtest_window(composites, bucket_ic, bucket_h,
                                           bucket_to, train_end, test_end,
                                           bucket_hl=bucket_hl)
        if bt_returns is None or bt_returns.empty:
            return res

        res.oos_returns = bt_returns[[
            'gross_return', 'net_return', 'net_return_lag1', 'funding_pnl',
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

        return res

    # ---------------- MVO backtest ----------------

    def _backtest_window(self, composites: Dict[str, pd.DataFrame],
                         bucket_ic: Dict[str, float],
                         bucket_h: Dict[str, float],
                         bucket_to: Dict[str, float],
                         test_start: pd.Timestamp,
                         test_end: pd.Timestamp,
                         bucket_hl: Optional[Dict[str, float]] = None
                         ) -> Optional[pd.DataFrame]:
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
        # Survivorship sensitivity gate (walk_forward.min_listing_age_days):
        # names whose first data bar is younger than this at the test day are
        # not investable. 0 -> off.
        min_age_days = int(WF.get('min_listing_age_days', 0) or 0)
        min_age = pd.Timedelta(days=min_age_days)

        # Per-bucket per-bar alpha scale: (1 - ic_shrink) * IC_b / sqrt(h_b),
        # times the Garleanu-Pedersen aim-portfolio discount h/(h + 1/kappa):
        # alpha faster than the trade rate kappa can't be monetized through
        # costs. ic_shrink pulls the noisy realized IC toward 0 (Grinold & Kahn
        #) so IC swings don't whipsaw the book.
        gp_on = bool(gp.get('enabled', False))
        # Effective fill rate kappa: the GP trade rate capped at what the
        # turnover budget allows.
        _, kappa = effective_fill_rate()
        ic_shrink = float(PORT.get('ic_shrink', 0.0))
        ic_keep = max(0.0, 1.0 - ic_shrink)
        # Turnover-implied holding/persistence per bucket (cost_holding_bars):
        # the bars a position actually lives - the scoring lag divided by the
        # bucket's per-rebalance turnover. Low turnover <=> slowly-moving
        # z-scores <=> the alpha PERSISTS that long, so this drives both the
        # cost amortization (alpha_h below) and the GP aim discount: a
        # slow-carry bucket whose ranking barely moves is fillable by a slow
        # book even though its SCORING lag is short (the lag-based discount
        # wrote off the OOS-proven funding sleeve as unfillable).
        # Turnover-implied persistence, CAPPED by the bucket's fitted alpha
        # half-life (from the discovery train profile) when one exists: low
        # turnover says the POSITIONS live long, the half-life says how long
        # the ALPHA does - the discount must honor the shorter of the two.
        hold_bars = {}
        for b in composites:
            h = cost_holding_bars(bucket_h.get(b, 1.0),
                                  (bucket_to or {}).get(b))
            hl = (bucket_hl or {}).get(b)
            if hl is not None and np.isfinite(hl) and hl > 0:
                h = min(h, float(hl))
            hold_bars[b] = h
        if composites:
            logging.info(
                "  holding/persistence (bars): %s",
                ", ".join(f"{b}:{hold_bars[b]:.0f} (to {(bucket_to or {}).get(b, float('nan')):.2f})"
                          for b in sorted(composites)))
        # Two scales per bucket from the same (1 - ic_shrink) * IC_b / sqrt(h_b)
        # base:
        #  - ic_scale     applies the Garleanu-Pedersen aim discount
        #    p/(p + 1/kappa) at the bucket's turnover-implied PERSISTENCE p
        #    (hold_bars) and drives the MVO/aim - alpha that reshuffles faster
        #    than the trade rate kappa can't be monetized, so the AIM is built
        #    from discounted alpha. Churny buckets: p = scoring lag (legacy).
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
            p_b = hold_bars[b]
            ic_scale[b] = raw * (p_b / (p_b + 1.0 / kappa)) if gp_on else raw
        # Observability: how hard the execution layer scales each bucket.
        if gp_on:
            disc = {b: hold_bars[b] / (hold_bars[b] + 1.0 / kappa)
                    for b in composites}
            logging.info(
                "  aim discounts (1/kappa=%.0f bars): %s",
                1.0 / kappa,
                ", ".join(f"{b}:{d:.3f}" for b, d in disc.items()))

        cluster_cfg = PORT.get('cluster_penalty', {})

        # No-trade zone ("lazy trading"): a name whose per-bar expected residual
        # return |alpha_i| is below no_trade_band_mult * (per-name per-side cost)
        # is not worth trading on, so its alpha is zeroed before the MVO. The
        # per-name cost is cost_rate * cost_mult_i (liquidity-aware below).
        no_trade_mult = float(PORT.get('no_trade_band_mult', 0.0))

        # Edge-scaled gross (see edge_gross_multiplier / config
        # portfolio.edge_scaled_gross): identity when costs are zero.
        esg_cfg = PORT.get('edge_scaled_gross', {})
        esg_on = bool(esg_cfg.get('enabled'))
        esg_mult = float(esg_cfg.get('edge_mult', 2.0))
        esg_min = float(esg_cfg.get('min_mult', 0.0))

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
        # Funding accrual: a settlement inside the bar's forward interval
        # (t, t+1bar] is paid by the book held over that interval (w_new at t),
        # so the funding row for bar t is looked up at stamp t + 1 bar. Columns
        # follow fwd_columns so fwd_col_positions indexes both panels. NaN
        # (no settlement at that stamp / no rate for that name) pays nothing.
        fund_on = self.ctx.funding_wide is not None
        funding_values_all = (
            self.ctx.funding_wide.reindex(
                index=bars + pd.Timedelta(BASE_FREQUENCY),
                columns=fwd_columns).to_numpy()
            if fund_on else None)
        # Volume-participation cap (portfolio.participation): per bar a name's
        # VOLUNTARY trade may not exceed max_participation x its trailing
        # volume_window_bars-bar average $ volume, converted to weight units by
        # book_size_usd. The trailing mean runs through bar t INCLUSIVE (bar-end
        # stamps: bar t is fully known when the trade earning (t, t+1] is
        # decided). Rows follow `bars`, columns follow fwd_columns so
        # fwd_col_positions indexes this panel too.
        part_cfg = PORT.get('participation', {})
        part_on = self.ctx.dollar_vol_wide is not None
        vol_ma_values_all = None
        part_rate = book_size = np.nan
        part_window_bars = 0
        if part_on:
            part_rate = float(part_cfg['max_participation'])
            book_size = float(part_cfg['book_size_usd'])
            part_window_bars = int(part_cfg['volume_window_bars'])
            vol_ma_values_all = (
                self.ctx.dollar_vol_wide
                .rolling(part_window_bars, min_periods=part_window_bars).mean()
                .reindex(index=bars, columns=fwd_columns).to_numpy())
        target_valid = False
        w_target = pd.Series(dtype=float)
        alpha = pd.Series(dtype=float)
        A = pd.DataFrame()
        target_index = pd.Index([])
        target_values = np.array([], dtype=float)
        carry = self._carry
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
                if min_age_days > 0 and self.ctx.first_bar is not None:
                    fb = self.ctx.first_bar.reindex(cov_assets)
                    old_enough = fb.notna() & (fb <= day - min_age)
                    cov_assets = cov_assets[old_enough.to_numpy()]
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
                # discounted) edge accumulated over each bucket's ACTUAL holding
                # period (raw per-bar contribution * hold_bars[b] - the scoring
                # lag divided by the bucket's per-rebalance turnover, see
                # cost_holding_bars) - the expected residual return captured
                # before the position turns over, i.e. the horizon on which it
                # must out-earn a round-trip cost. It feeds the no-trade band
                # and edge-scaled gross only, and uses ic_scale_raw so trade
                # speed never vetoes a profitable signal.
                alpha = pd.Series(0.0, index=assets)
                alpha_h = pd.Series(0.0, index=assets)
                got_signal = False
                for h, score in held_scores.items():
                    z = score.reindex(assets)
                    if z.notna().sum() < PORT['min_assets']:
                        continue
                    zf = z.fillna(0.0)
                    sig_z = sigma.reindex(assets) * zf
                    alpha = alpha.add(ic_scale[h] * sig_z, fill_value=0.0)
                    alpha_h = alpha_h.add(
                        ic_scale_raw[h] * sig_z * hold_bars[h],
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

                w_target = solve_constrained_mvo(alpha, cov_a, A,
                                                 max_position=cap,
                                                 gross_leverage=gross_target,
                                                 bands=bands)
                # Edge-scaled gross: shrink the aim when its expected horizon
                # edge cannot cover edge_mult round trips - never trade a
                # gross-1 book on alpha that can't pay for itself. Identity
                # multiplier when costs are zero.
                if esg_on:
                    g_t = float(np.abs(w_target.values).sum())
                    if g_t > 1e-12:
                        exp_edge = float(alpha_h.reindex(w_target.index)
                                         .fillna(0.0).values @ w_target.values) / g_t
                        cm_t = (cost_mult.reindex(w_target.index).fillna(1.0).values
                                if liq_on else np.ones(len(w_target)))
                        rt_cost = 2.0 * cost_rate * float(
                            (np.abs(w_target.values) * cm_t).sum()) / g_t
                        w_target = w_target * edge_gross_multiplier(
                            exp_edge, rt_cost, esg_mult, esg_min)
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
            # Which mechanism sets the trade speed this bar (observability):
            # True -> the hard turnover budget, False -> the GP rate itself.
            budget_bound = gap > 1e-12 and per_bar_to_budget / gap < smooth_alpha
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
            # Volume-participation cap: per-name |trade| <= max_dw this bar.
            # `want` (the trade the GP step asked for) is captured BEFORE the
            # clamp for the binding diagnostics below.
            n_vol_capped = 0
            vol_blocked = 0.0
            max_dw = None
            if part_on:
                vols = np.full(len(target_index), np.nan)
                vp_valid = fwd_col_positions >= 0
                if vp_valid.any():
                    vols[vp_valid] = vol_ma_values_all[
                        i, fwd_col_positions[vp_valid]]
                max_dw = participation_caps(vols, part_rate, book_size)
                want = np.abs(w_new_values - w_prev_values)
                n_vol_capped = int((want > max_dw + 1e-15).sum())
                vol_blocked = float(np.maximum(want - max_dw, 0.0).sum())
                w_new_values = clamp_to_participation(w_new_values,
                                                      w_prev_values, max_dw)
            cap_w = cap * gross_target * 0.999
            if neutralizer is not None:
                v = w_new_values
                for _ in range(3):
                    v = np.clip(v, -cap_w, cap_w)
                    if part_on:
                        v = clamp_to_participation(v, w_prev_values, max_dw)
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
            # Hard guarantee: the participation cap binds AFTER every other
            # adjustment (neutralize projection and gross scaling can both push
            # a trade back over it). The small neutrality slack this leaves is
            # absorbed by the neutrality bands and self-corrects next bar (the
            # cap re-centers on the newly held book); realized exposures below
            # remain the acceptance check.
            if part_on:
                w_new_values = clamp_to_participation(w_new_values,
                                                      w_prev_values, max_dw)

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
            # Perp funding exchanged inside this bar's forward interval:
            # longs PAY a positive rate, shorts receive -> pnl = -(w . rate).
            fund_values = np.zeros(len(target_index), dtype=float)
            if fund_on and valid_fwd_cols.any():
                fv = funding_values_all[i, fwd_col_positions[valid_fwd_cols]]
                fund_values[valid_fwd_cols] = np.nan_to_num(fv, nan=0.0)
            funding_pnl = -float(w_new_values @ fund_values)
            gross_ret = float(w_new_values @ fwd_values)
            net_ret = gross_ret - trade_cost + funding_pnl
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
                lag_ret = (float(lag_values @ fwd_values) - trade_cost
                           - float(lag_values @ fund_values))
            else:
                lag_ret = np.nan

            # Realized exposure to every neutralized factor (should be ~0).
            exposures = {n: float(w_new_values @ bv) for n, bv in beta_values.items()}

            row = {'timestamp': ts, 'gross_return': gross_ret,
                   'net_return': net_ret, 'net_return_lag1': lag_ret,
                   'funding_pnl': funding_pnl,
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
                'trade_rate': float(a_eff),
                'budget_bound': bool(budget_bound),
                'n_vol_capped': n_vol_capped,
                'vol_trade_blocked': vol_blocked,
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
        self._carry = pd.Series(held_values, index=held_index)
        if not rows:
            return None

        # Binding-constraint summary (production scheme): if the turnover budget
        # binds on ~all bars, the GP trade_urgency knob is inert and the budget
        # alone sets the book's speed - worth knowing before tuning either.
        if risk_rows:
            n_bound = sum(1 for r in risk_rows if r['budget_bound'])
            rates = np.array([r['trade_rate'] for r in risk_rows], dtype=float)
            logging.info(
                f"  trade speed: nominal {smooth_alpha:.4f}/bar, median "
                f"effective {float(np.median(rates)):.5f}/bar; turnover budget "
                f"binding on {n_bound / len(risk_rows):.0%} of bars")
            if part_on:
                n_pb = sum(1 for r in risk_rows if r['n_vol_capped'] > 0)
                blocked = float(np.mean([r['vol_trade_blocked']
                                         for r in risk_rows]))
                logging.info(
                    f"  participation cap ({part_rate:.0%} of "
                    f"{part_window_bars}-bar "
                    f"avg $vol, book ${book_size:,.0f}): binding for >=1 name "
                    f"on {n_pb / len(risk_rows):.0%} of bars, mean blocked "
                    f"trade {blocked:.4f} weight/bar")

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

    def run(self) -> pd.DataFrame:
        from tqdm import tqdm
        from research.signals.data import make_rolls

        # One window per discovery roll: train on that roll's train window,
        # trade its OOS month with the signals it promoted (out of sample).
        schedule = [(pd.Timestamp(r.train_start), pd.Timestamp(r.oos_start),
                     pd.Timestamp(r.oos_end)) for r in make_rolls(get('discovery'))]
        logging.info(f"Walk-forward: {len(schedule)} monthly windows "
                     f"(mirrors discovery rolls)")
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
        self._carry = pd.Series(dtype=float)  # fresh run starts flat
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
        if 'funding_pnl' in returns.columns:
            ann_funding = returns['funding_pnl'].mean() * ppy
            print(f"Funding PnL (ann.): {ann_funding * 100:+.1f}%  "
                  f"(perp funding on held positions; included in net)")

        # Per-window detail as an aligned table. Buckets are the distinct
        # selected lags ('3b:5' = 5 signals held at the 3-bar lag); top
        # families are abbreviated to the three largest (full breakdown lives
        # in the notebook's attribution view).
        row = ("  {win:<4}{period:<24}{nsig:>5}  {bkt:<20}"
               "{sr:>8}{mkt:>9}  {fam}")
        print("\nPer-window detail (buckets = promoted lags):")
        print(row.format(win='Win', period='OOS test period', nsig='#sig',
                         bkt='lag:sigs', sr='OOS_SR', mkt='|mkt|',
                         fam='top families'))
        for w in self.windows:
            labs = sorted(w.selected, key=lambda s: int(str(s).rstrip('b') or 0))
            bkt = ' '.join(f'{b}:{len(w.selected[b])}' for b in labs)
            fam = list(self._selected_family_counts(w).items())
            fam_str = ' '.join(f'{k}:{v}' for k, v in fam[:3])
            if len(fam) > 3:
                fam_str += f' (+{len(fam) - 3})'
            mkt = w.avg_abs_mkt_exposure if not np.isnan(w.avg_abs_mkt_exposure) else 0.0
            print(row.format(
                win=f'W{w.window_idx:02d}',
                period=f'{w.train_end.date()}-{w.test_end.date()}',
                nsig=sum(len(s) for s in w.selected.values()),
                bkt=bkt or '-',
                sr=f'{w.oos_sharpe:.2f}',
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
            'n_signals': w.n_candidates,
            'selected': ';'.join(f"{b}:{','.join(s[:10])}" for b, s in w.selected.items()),
            'selected_families': ';'.join(f"{k}:{v}" for k, v in family_counts.items()),
            'horizon_ic': ';'.join(f"{b}:{ic:.4f}" for b, ic in w.horizon_ic.items()),
            'bucket_holding_lag': ';'.join(f"{b}:{v:.0f}" for b, v in w.bucket_h.items()),
            'oos_sharpe': w.oos_sharpe,
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
    logging.info("Walk-forward starting (registry -> scoring -> panels)...")
    wf = WalkForwardPortfolio()
    # run() checkpoints each window to the DB (wf_portfolio_*) as it traverses.
    returns = wf.run()
    if returns.empty:
        print("No portfolio produced: no window had defensible tradable signals")
        return
    wf.summary(returns)


if __name__ == '__main__':
    main()
