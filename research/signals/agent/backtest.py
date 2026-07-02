"""
BACKTEST: promotion gates and the OOS portfolio.

Promotion (once per roll, applied to the search survivors):
  1. BY/BH FDR across the survivor set (select-window IC p-values)
  2. deflation haircut: |t| must clear deflation_mult x E[max |N(0,1)|] over
     the roll's TOTAL trial count from the ledger - the search-overfit tax,
     a property of the search, not of any one candidate
  3. N-consecutive-rolls persistence (by candidate hash, via the ledger)
  4. orthogonality vs the already-promoted book (incremental edge, greedy)
  5. per-roll and book-size caps

OOS portfolio: promoted signals combined equal-weight into one alpha, then
optimized per rebalance stamp - dollar + factor-beta neutral via the SAME
research/lib/portfolio_opt solvers as production. Each stamp is solved
INDEPENDENTLY (non-sequential by design): no turnover term in the objective,
costs applied ex-post from |dw|. That slightly overstates turnover versus a
turnover-aware optimizer - the price of keeping the backtest vectorizable.
PnL uses RAW returns (neutrality does the hedging), plus optional funding
accrual; realized factor exposures ~ 0 is the acceptance check.

Stitched OOS months across rolls = the discovery system's equity curve.
"""

import logging
import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config import get, get_frequency_config, BASE_FREQUENCY
from research.lib.portfolio_opt import (benjamini_hochberg,
                                        benjamini_yekutieli,
                                        shrunk_covariance,
                                        solve_constrained_mvo,
                                        solve_equal_weight)
from research.signals.agent.data import Roll, beta_columns, slice_window
from research.signals.agent.generation import Candidate, compile_candidate
from research.signals.agent.search import DiscoveryLedger, max_signal_correlation


# =============================================================================
# Promotion
# =============================================================================

def _tstat_pvalue(t: float) -> float:
    """Two-sided normal p-value for a t-stat (daily IC series is long enough
    that the normal approximation is fine)."""
    return math.erfc(abs(float(t)) / math.sqrt(2.0))


def expected_max_abs_normal(n_trials: int) -> float:
    """E[max |N(0,1)|] over n trials ~ sqrt(2 ln n) - the bar random noise
    reaches when the search tries n candidates against the same month."""
    n = max(int(n_trials), 1)
    if n == 1:
        return 0.8   # E|N(0,1)|
    return math.sqrt(2.0 * math.log(n))


def promote(survivors: List[dict], roll: Roll, ledger: DiscoveryLedger,
            book: List[dict], cfg: Optional[dict] = None) -> List[dict]:
    """Apply the promotion gates to a roll's survivors.

    survivors: run_search() output (with metrics_select + signal_select).
    book: currently promoted entries; entries carrying a 'signal_select'
          recompiled on THIS roll are used for the orthogonality gate.
    Returns the newly promoted survivor dicts (annotated with gate details).
    """
    cfg = cfg or get('discovery', {})
    promo = cfg['promotion']
    if not survivors:
        return []

    tstats = np.array([s['metrics_select']['ic_tstat'] for s in survivors])
    pvals = np.array([_tstat_pvalue(t) for t in tstats])
    fdr = (benjamini_yekutieli if promo['fdr_method'] == 'by'
           else benjamini_hochberg)
    fdr_mask = fdr(pvals, alpha=float(promo['fdr_alpha']))

    n_trials = ledger.n_trials(roll.roll_id)
    deflation_bar = (float(promo['deflation_mult'])
                     * expected_max_abs_normal(n_trials))

    book_signals = [b['signal_select'] for b in book
                    if b.get('signal_select') is not None]
    max_book_corr = float(promo['max_book_corr'])
    slots = min(int(promo['max_promoted_per_roll']),
                int(promo['max_book_size']) - len(book))

    promoted: List[dict] = []
    order = np.argsort(-np.abs(tstats))     # strongest evidence first
    for i in order:
        if len(promoted) >= max(slots, 0):
            break
        s = survivors[i]
        t = abs(tstats[i])
        gates = {
            'fdr': bool(fdr_mask[i]),
            'min_tstat': t >= float(promo['min_select_ic_tstat']),
            'deflation': (float(promo['deflation_mult']) <= 0
                          or t >= deflation_bar),
            'persistence': ledger.consecutive_survivals(
                s['candidate'].hash, roll.roll_id)
                >= int(promo['min_rolls_survived']),
            'orthogonal': max_signal_correlation(
                s['signal_select'],
                book_signals + [p['signal_select'] for p in promoted])
                <= max_book_corr,
        }
        if all(gates.values()):
            promoted.append({**s, 'roll_promoted': roll.roll_id,
                             'n_trials_at_promotion': n_trials})
        else:
            logging.debug(f"{s['candidate'].name} blocked by "
                          f"{[k for k, v in gates.items() if not v]}")

    ledger.mark_promoted(roll.roll_id,
                         [p['candidate'].hash for p in promoted])
    return promoted


# =============================================================================
# OOS portfolio backtest
# =============================================================================

def _combined_alpha(panel: pd.DataFrame, book: List[dict],
                    allowed_columns=None) -> pd.DataFrame:
    """Compile every book candidate on the panel (train history included for
    rolling warmup), apply its traded direction, and average into one alpha
    panel [timestamp, symbol, alpha]."""
    sigs = []
    for entry in book:
        sig = compile_candidate(entry['candidate'], panel, allowed_columns)
        sig = sig.rename(columns={'signal': entry['candidate'].hash})
        sig[entry['candidate'].hash] *= entry['direction']
        sigs.append(sig.set_index(['timestamp', 'symbol']))
    wide = pd.concat(sigs, axis=1)
    alpha = wide.mean(axis=1).rename('alpha').reset_index()
    return alpha


def backtest_oos(panel: pd.DataFrame, roll: Roll, book: List[dict],
                 cfg: Optional[dict] = None) -> dict:
    """Trade the promoted book through the OOS month.

    Returns {'daily': DataFrame[date, gross, cost, funding, net],
             'exposures': {constraint: mean |B'w|},
             'stamps': int, 'weights': last weight Series}.
    """
    cfg = cfg or get('discovery', {})
    bt_cfg = cfg['backtest']
    if not book:
        return {'daily': pd.DataFrame(
                    columns=['date', 'gross', 'cost', 'funding', 'net']),
                'exposures': {}, 'stamps': 0}

    cost_bps = bt_cfg['cost_bps']
    if cost_bps is None:
        cost_bps = get('portfolio.cost_bps')
    cost_rate = float(cost_bps) / 10000.0
    min_assets = int(bt_cfg['min_assets'])
    grid = bt_cfg['rebalance_grid']
    scheme = bt_cfg['weight_scheme']

    # Compile with full roll history (rolling warmup), trade only OOS stamps.
    roll_panel = slice_window(panel, roll.train_start, roll.oos_end, 0)
    roll_panel = roll_panel.reset_index(drop=True)
    alpha = _combined_alpha(roll_panel, book)

    oos = slice_window(roll_panel, roll.oos_start, roll.oos_end, 0)
    bars = np.sort(oos['timestamp'].unique())
    if len(bars) == 0:
        return {'daily': pd.DataFrame(
                    columns=['date', 'gross', 'cost', 'funding', 'net']),
                'exposures': {}, 'stamps': 0}
    stamp_index = pd.DatetimeIndex(bars)
    stamps = stamp_index[stamp_index == stamp_index.floor(grid)]

    # Wide raw returns over the OOS bars for interval PnL
    raw_wide = oos.pivot_table(index='timestamp', columns='symbol',
                               values='raw_return', aggfunc='first')
    raw_cum = raw_wide.fillna(0.0).cumsum()

    bcols = beta_columns(oos)
    alpha_oos = alpha[(alpha['timestamp'] >= roll.oos_start)
                      & (alpha['timestamp'] < roll.oos_end)]
    alpha_by_ts = dict(tuple(alpha_oos.groupby('timestamp')))
    panel_by_ts = dict(tuple(
        oos[['timestamp', 'symbol', 'residual_return'] + bcols]
        .groupby('timestamp')))

    cov = None
    if scheme == 'mvo':
        cov = _trailing_covariance(roll_panel, roll, bt_cfg)

    port_cfg = get('portfolio', {})
    max_position = float(port_cfg['max_position'])
    gross_leverage = float(port_cfg['gross_leverage'])

    weights: Dict[pd.Timestamp, pd.Series] = {}
    exposures: List[dict] = []
    for ts in stamps:
        a_t = alpha_by_ts.get(ts)
        p_t = panel_by_ts.get(ts)
        if a_t is None or p_t is None or len(a_t) < min_assets:
            continue
        a = a_t.set_index('symbol')['alpha']
        cons = pd.DataFrame({'dollar': 1.0}, index=a.index)
        for bc in bcols:
            cons[bc] = p_t.set_index('symbol')[bc].reindex(a.index).fillna(0.0)
        # Exact neutrality (band 0): each stamp solved INDEPENDENTLY.
        if scheme == 'mvo' and cov is not None:
            common = [s for s in a.index if s in cov.index]
            if len(common) < min_assets:
                continue
            w = solve_constrained_mvo(a[common], cov.loc[common, common],
                                      cons.loc[common],
                                      max_position=max_position,
                                      gross_leverage=gross_leverage)
        else:
            w = solve_equal_weight(a, cons, max_position=max_position,
                                   gross_leverage=gross_leverage)
        w = w[w != 0.0]
        if w.abs().sum() < 1e-12:
            continue
        weights[ts] = w
        exp = {'dollar': float(w.sum())}
        for bc in bcols:
            exp[bc] = float((w * cons[bc].reindex(w.index).fillna(0.0)).sum())
        exposures.append(exp)

    if not weights:
        return {'daily': pd.DataFrame(
                    columns=['date', 'gross', 'cost', 'funding', 'net']),
                'exposures': {}, 'stamps': 0}

    fund_wide = _load_funding(stamp_index, raw_wide.columns) \
        if bt_cfg['funding_pnl'] else None

    rows = []
    w_prev = pd.Series(dtype=float)
    stamp_list = sorted(weights)
    for i, ts in enumerate(stamp_list):
        w = weights[ts]
        ts_next = stamp_list[i + 1] if i + 1 < len(stamp_list) else stamp_index[-1]
        # PnL over the holding interval (ts, ts_next]: cumsum difference
        interval_ret = (raw_cum.loc[ts_next] - raw_cum.loc[ts]).reindex(
            w.index).fillna(0.0)
        gross = float((w * interval_ret).sum())
        union = w.index.union(w_prev.index)
        turnover = float((w.reindex(union).fillna(0.0)
                          - w_prev.reindex(union).fillna(0.0)).abs().sum())
        cost = turnover * cost_rate
        funding = 0.0
        if fund_wide is not None:
            in_interval = fund_wide.index[(fund_wide.index > ts)
                                          & (fund_wide.index <= ts_next)]
            for fts in in_interval:
                rates = fund_wide.loc[fts].reindex(w.index).fillna(0.0)
                funding += -float((w * rates).sum())   # longs PAY positive rates
        rows.append({'timestamp': ts, 'gross': gross, 'cost': cost,
                     'funding': funding, 'net': gross - cost + funding})
        w_prev = w

    per_stamp = pd.DataFrame(rows)
    per_stamp['date'] = per_stamp['timestamp'].dt.normalize()
    daily = per_stamp.groupby('date')[['gross', 'cost', 'funding',
                                       'net']].sum().reset_index()
    exp_df = pd.DataFrame(exposures)
    return {
        'daily': daily,
        'exposures': {c: float(exp_df[c].abs().mean()) for c in exp_df.columns},
        'stamps': len(stamp_list),
        'weights': weights[stamp_list[-1]],
    }


def _trailing_covariance(roll_panel: pd.DataFrame, roll: Roll,
                         bt_cfg: dict) -> Optional[pd.DataFrame]:
    """Ledoit-Wolf residual covariance from the window strictly before OOS
    (estimated once per roll: causal, and keeps the OOS loop cheap)."""
    bar_ns = get_frequency_config(BASE_FREQUENCY)['nanos']
    bars_per_day = int(24 * 3600 * 1e9 // bar_ns)
    window_bars = int(bt_cfg['cov_window_days']) * bars_per_day
    pre = roll_panel[roll_panel['timestamp'] < roll.oos_start]
    wide = pre.pivot_table(index='timestamp', columns='symbol',
                           values='residual_return', aggfunc='first')
    wide = wide.tail(window_bars)
    cov = shrunk_covariance(wide,
                            min_observations=int(bt_cfg['cov_min_observations']),
                            shrinkage=get('portfolio.shrinkage', 'ledoit_wolf'))
    return cov if not cov.empty else None


def _load_funding(stamp_index: pd.DatetimeIndex,
                  symbols) -> Optional[pd.DataFrame]:
    """Funding rates pivoted wide, restricted to the OOS bar range. None when
    the table is unavailable (backtest degrades to price PnL - costs)."""
    try:
        from dbutil import load_data, table_exists
        if not table_exists('funding_rates'):
            return None
        fr = load_data('funding_rates',
                       columns=['timestamp', 'symbol', 'funding_rate'])
        if fr is None or fr.empty:
            return None
        fr['timestamp'] = pd.to_datetime(fr['timestamp'])
        fr = fr[(fr['timestamp'] > stamp_index[0])
                & (fr['timestamp'] <= stamp_index[-1])]
        wide = fr.pivot_table(index='timestamp', columns='symbol',
                              values='funding_rate', aggfunc='first')
        return wide.reindex(columns=symbols)
    except Exception as e:
        logging.warning(f"funding accrual unavailable: {e}")
        return None


def stitch_oos(results: List[dict]) -> pd.DataFrame:
    """Stitch every roll's OOS daily PnL end-to-end: the equity curve."""
    frames = [r['daily'] for r in results if len(r.get('daily', [])) > 0]
    if not frames:
        return pd.DataFrame(columns=['date', 'gross', 'cost', 'funding',
                                     'net', 'cum_net'])
    out = pd.concat(frames, ignore_index=True).sort_values('date')
    out = out.groupby('date', as_index=False).sum()   # roll overlap safety
    out['cum_net'] = out['net'].cumsum()
    return out
