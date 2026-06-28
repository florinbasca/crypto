"""
signal_lab.py - KEEP/KILL scorecard for a candidate space.

Standalone IC says a space predicts; it does NOT say it adds anything new.
Reports both from the persisted `signal_daily_stats` (evaluate the space first):

    uv run python research/signals/evaluate.py space_<name>     # populate stats
    uv run python research/signals/signal_lab.py space_<name>   # scorecard

Prints: IC decay curve (HAC t-stat per horizon), tradeability (turnover +
liquid-half IC), incremental value (correlation of its PnL to the live book),
and a KEEP / REDUNDANT / KILL verdict. Reads persisted tables only.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import numpy as np
import pandas as pd

from dbutil import load_data
from config import get, BARS_PER_DAY
from research.portfolio.walk_forward import _nw_tstat

PPY = BARS_PER_DAY * 365
# Thresholds for the verdict (all from config so nothing is hardcoded here).
MIN_TSTAT = get('signal_lab.min_ic_tstat', 2.0)
MAX_BOOK_CORR = get('signal_lab.max_book_corr', 0.5)
MIN_LIQ_RATIO = get('signal_lab.min_liquid_ic_ratio',
                    get('walk_forward.min_liquid_ic_ratio', 0.3))


def _daily_ic(df: pd.DataFrame) -> pd.Series:
    """Per-date IC series for one horizon: ic_sum / n_cs."""
    d = df.assign(ic=df['ic_sum'] / df['n_cs'].replace(0, np.nan))
    return d.dropna(subset=['ic']).set_index('date')['ic'].sort_index()


def decay_curve(stats: pd.DataFrame) -> pd.DataFrame:
    """Mean IC, HAC t-stat, turnover, liquid-IC ratio per horizon."""
    rows = []
    for h, g in stats.groupby('horizon'):
        ic = _daily_ic(g)
        if ic.empty:
            continue
        n = g['n_cs'].sum()
        ic_mean = g['ic_sum'].sum() / n if n else np.nan
        liq = (g['liq_ic_sum'].sum() / g['n_liq'].sum()
               if g['n_liq'].sum() else np.nan)
        turn = (g['turnover'].sum() / g['n_rebalances'].sum()
                if g['n_rebalances'].sum() else np.nan)
        rows.append({
            'horizon': h,
            'bars': int(str(h).rstrip('b')) if str(h).rstrip('b').isdigit() else -1,
            'ic_mean': ic_mean,
            'ic_tstat': _nw_tstat(ic.values, 'auto'),
            'liq_ic_ratio': abs(liq) / abs(ic_mean) if ic_mean else np.nan,
            'avg_turnover': turn,
            'n_days': len(ic),
        })
    out = pd.DataFrame(rows).sort_values('bars')
    return out


def signal_daily_pnl(stats: pd.DataFrame, horizon: str) -> pd.Series:
    """The signal's own dollar-neutral daily net return at one horizon."""
    g = stats[stats['horizon'] == horizon]
    return g.dropna(subset=['ret_net']).set_index('date')['ret_net'].sort_index()


def portfolio_daily_pnl() -> pd.Series:
    """Live portfolio daily net return (from the last walk-forward run)."""
    r = load_data('wf_portfolio_returns')
    if r is None or r.empty:
        return pd.Series(dtype=float)
    r['timestamp'] = pd.to_datetime(r['timestamp'])
    if 'window' in r.columns:   # de-overlap window boundaries
        r = r.sort_values(['window', 'timestamp']).drop_duplicates('timestamp', keep='first')
    r = r.set_index('timestamp')['net_return']
    return r.groupby(r.index.normalize()).apply(lambda x: (1 + x).prod() - 1)


def sharpe(x: pd.Series) -> float:
    x = x.dropna()
    return float(x.mean() / x.std() * np.sqrt(365)) if len(x) > 2 and x.std() > 0 else np.nan


def incremental_value(sig_pnl: pd.Series, book_pnl: pd.Series, direction: float) -> dict:
    """Correlation of the signal's TRADED-direction PnL to the live book.

    Correlation is the robust, scale-free incremental-value metric (the signal's
    raw daily Sharpe is not comparable to the book's: high-horizon signals sum
    many intraday rebalances). Low |corr| with a real own-edge = diversifying.
    """
    sig = (sig_pnl * np.sign(direction)) if direction else sig_pnl
    j = pd.concat([sig.rename('sig'), book_pnl.rename('book')], axis=1).dropna()
    if len(j) < 30:
        return {'overlap_days': len(j)}
    return {
        'overlap_days': len(j),
        'corr_to_book': float(j['sig'].corr(j['book'])),
        'own_daily_mean_bps': float(j['sig'].mean() * 1e4),  # signed: should be >0
    }


def verdict(best: pd.Series, inc: dict) -> str:
    reasons, ok = [], True
    if abs(best['ic_tstat']) < MIN_TSTAT:
        ok = False; reasons.append(f"IC t-stat {best['ic_tstat']:+.1f} (|.|<{MIN_TSTAT})")
    if np.isfinite(best['liq_ic_ratio']) and best['liq_ic_ratio'] < MIN_LIQ_RATIO:
        ok = False; reasons.append(f"liquid-IC ratio {best['liq_ic_ratio']:.2f} (<{MIN_LIQ_RATIO})")
    corr = inc.get('corr_to_book')
    redundant = corr is not None and abs(corr) > MAX_BOOK_CORR
    if redundant:
        reasons.append(f"corr to book {corr:+.2f} (>{MAX_BOOK_CORR}: redundant)")
    if ok and not redundant:
        return "KEEP  - significant, tradeable, and diversifying (orthogonal to book)"
    if ok and redundant:
        return "REDUNDANT - real edge but already in the book: " + "; ".join(reasons)
    return "KILL  - " + "; ".join(reasons)


def lab(name: str) -> None:
    stats = load_data('signal_daily_stats')
    stats = stats[stats['signal_name'] == name].copy()
    if stats.empty:
        raise SystemExit(f"No signal_daily_stats for {name!r}. Run "
                         f"`evaluate.py {name}` first.")
    stats['date'] = pd.to_datetime(stats['date'])

    dc = decay_curve(stats)
    print(f"\n=== {name} : IC decay curve ===")
    print(dc[['horizon', 'ic_mean', 'ic_tstat', 'liq_ic_ratio',
              'avg_turnover', 'n_days']].round(4).to_string(index=False))
    best = dc.loc[dc['ic_tstat'].abs().idxmax()]
    print(f"\nbest horizon: {best['horizon']}  "
          f"(IC {best['ic_mean']:+.4f}, t {best['ic_tstat']:+.2f}, "
          f"turnover {best['avg_turnover']:.3f})")

    sig_pnl = signal_daily_pnl(stats, best['horizon'])
    inc = incremental_value(sig_pnl, portfolio_daily_pnl(), np.sign(best['ic_mean']))
    print(f"\n=== incremental value vs the live portfolio ===")
    if 'corr_to_book' not in inc:
        print(f"  insufficient overlap ({inc.get('overlap_days', 0)} days) - "
              f"is the portfolio backtest populated?")
    else:
        print(f"  overlap days        : {inc['overlap_days']}")
        print(f"  corr to book        : {inc['corr_to_book']:+.3f}   "
              f"(want |corr| < {MAX_BOOK_CORR}: low = adds diversification)")
        print(f"  own traded edge bps : {inc['own_daily_mean_bps']:+.2f}/day "
              f"(signed by IC; should be > 0)")

    print(f"\nVERDICT: {verdict(best, inc)}\n")


def main():
    ap = argparse.ArgumentParser(description="KEEP/KILL scorecard for a candidate signal")
    ap.add_argument('signal', help='signal_name (must be in signal_daily_stats)')
    lab(ap.parse_args().signal)


if __name__ == '__main__':
    main()
