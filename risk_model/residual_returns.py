"""
Residual Returns: market+size factor model, causal betas, aligned horizons.

Pipeline:
1. Betas: for each symbol, daily-refreshed exponentially-weighted OLS of
   single-bar returns on [market_factor, size_factor] using ONLY bars strictly
   before the day (no look-ahead). Window/halflife from config.
2. Single-bar residual: res[t] = r[t] - b_mkt(day(t)) * f_mkt[t]
                                  - b_size(day(t)) * f_size[t]
   (intercept used in estimation, NOT subtracted - alpha stays in the residual)
3. Forward targets per horizon p (in bars), CORRECTLY ALIGNED:
       fwd_res_h[t] = sum_{k=1..p} res[t+k]  =  res.rolling(p).sum().shift(-p)
   Bar-end convention: r[t]/f[t] cover (t-1bar, t], so the forward window
   (t, t+p] is bars t+1..t+p. Using factor[t..t+p-1] here was the critical
   misalignment bug in the previous PCA pipeline - do not reintroduce it.

Outputs:
- factor_loadings   [date, symbol, beta_market, beta_size, r_squared]
- residual_returns  [timestamp, symbol, raw_return, residual_return,
                     fwd_res_{h}, fwd_raw_{h} for each configured horizon]

Acceptance checks (printed, warn on failure):
- var(residual)/var(raw) <= 1 - min_variance_reduction
- corr(residual, raw) <= max_residual_raw_corr
A factor model that fails these is silently doing nothing - stop and debug.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
from concurrent.futures import as_completed

import numpy as np
import pandas as pd

from config import (config as global_config, get, get_frequency_config,
                    horizon_bars, horizon_col, HORIZONS)
from dbutil import load_data, save_data, delete_table, get_parallel_executor

logging.basicConfig(
    level=logging.INFO,
    format=global_config['logging']['format'],
    datefmt=global_config['logging']['datefmt'],
)

BASE_FREQ = global_config['base_frequency']
BARS_PER_DAY = get_frequency_config(BASE_FREQ)['bars_per_day']
FACTOR_NAMES = get('risk_model.factors', ['market', 'size'])
FACTOR_COLS = [f'{name}_factor' for name in FACTOR_NAMES]

BETA_WINDOW_DAYS = get('risk_model.beta.window_days', 30)
BETA_HALFLIFE_DAYS = get('risk_model.beta.halflife_days', 10)
BETA_MIN_OBS = get('risk_model.beta.min_observations', 1008)
MAX_WORKERS = get('compute.residual_workers', 4)

ACCEPT_MIN_VAR_REDUCTION = get('risk_model.acceptance.min_variance_reduction', 0.05)
ACCEPT_MAX_CORR = get('risk_model.acceptance.max_residual_raw_corr', 0.95)


def exponential_weights(n: int, halflife_bars: float) -> np.ndarray:
    """Exponential weights, most recent observation weighted highest."""
    alpha = np.log(2) / halflife_bars
    idx = np.arange(n)
    w = np.exp(-alpha * (n - 1 - idx))
    return w / w.sum()


def weighted_ols(y: np.ndarray, X: np.ndarray, w: np.ndarray, min_obs: int):
    """
    Weighted OLS with intercept. Returns (betas[k], r_squared) or (None, None).
    X must NOT include the intercept column.
    """
    # Drop any non-finite row (NaN or +/-inf) in the target or factors. inf
    # here (e.g. a divide-by-zero return that slipped through) would otherwise
    # produce inf/NaN coefficients and overflow warnings in the matmul below.
    valid = np.isfinite(y) & np.isfinite(X).all(axis=1)
    y, X, w = y[valid], X[valid], w[valid]
    n = len(y)
    if n < min_obs:
        return None, None

    w_sum = w.sum()
    if not np.isfinite(w_sum) or w_sum <= 0:
        return None, None
    w = w / w_sum * n
    sw = np.sqrt(w)
    Xi = np.column_stack([np.ones(n), X])
    # Inputs are finite (filtered above), so any fp flag raised here is numpy's
    # spurious matmul/SIMD flag-reporting, not a real numerical error. Ignore
    # those flags and instead verify the outputs explicitly.
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        try:
            coef, _, _, _ = np.linalg.lstsq(Xi * sw[:, None], y * sw, rcond=None)
        except np.linalg.LinAlgError:
            return None, None
        # An ill-conditioned/degenerate window can still yield non-finite betas;
        # treat those as "no estimate" rather than propagating inf downstream.
        if not np.all(np.isfinite(coef)):
            return None, None
        y_hat = Xi @ coef
        y_bar = np.average(y, weights=w)
        ss_res = np.sum(w * (y - y_hat) ** 2)
        ss_tot = np.sum(w * (y - y_bar) ** 2)

    r2 = float(np.clip(1 - ss_res / ss_tot, 0, 1)) if ss_tot > 1e-18 else 0.0
    return coef[1:], r2


def estimate_daily_betas(symbol: str,
                         asset_returns: pd.Series,
                         factors: pd.DataFrame) -> pd.DataFrame:
    """
    Daily-refreshed EWMA betas using bars STRICTLY BEFORE each day.

    asset_returns: Series indexed by timestamp (base frequency)
    factors: DataFrame indexed by timestamp with FACTOR_COLS
    """
    df = pd.DataFrame({'r': asset_returns}).join(factors[FACTOR_COLS], how='inner')
    df = df.dropna(subset=FACTOR_COLS)
    if df.empty:
        return pd.DataFrame()

    window_bars = BETA_WINDOW_DAYS * BARS_PER_DAY
    halflife_bars = BETA_HALFLIFE_DAYS * BARS_PER_DAY

    timestamps = df.index
    r = df['r'].values
    F = df[FACTOR_COLS].values

    # Positions where a new day starts
    days = timestamps.normalize()
    day_starts = np.flatnonzero(days != np.roll(days, 1))
    day_starts[0] = 0

    rows = []
    for pos in day_starts:
        day = days[pos]
        lo = max(0, pos - window_bars)
        if pos - lo < BETA_MIN_OBS:
            continue
        y_win = r[lo:pos]
        X_win = F[lo:pos]
        w = exponential_weights(pos - lo, halflife_bars)
        betas, r2 = weighted_ols(y_win, X_win, w, BETA_MIN_OBS)
        if betas is None:
            continue
        row = {'date': day, 'symbol': symbol, 'r_squared': r2}
        for name, b in zip(FACTOR_NAMES, betas):
            row[f'beta_{name}'] = float(b)
        rows.append(row)

    return pd.DataFrame(rows)


def compute_symbol_residuals(symbol: str,
                             asset_returns: pd.Series,
                             factors: pd.DataFrame,
                             betas: pd.DataFrame) -> pd.DataFrame:
    """Single-bar residuals + correctly aligned multi-horizon forward targets."""
    if betas.empty:
        return pd.DataFrame()

    df = pd.DataFrame({'raw_return': asset_returns}).join(factors[FACTOR_COLS], how='left')
    df['date'] = df.index.normalize()

    beta_cols = [f'beta_{name}' for name in FACTOR_NAMES]
    b = betas.set_index('date')[beta_cols]
    # Each bar uses the beta estimated at the start of its own day (strictly
    # past data). Days with no estimate produce NaN residuals - no fabrication.
    df = df.join(b, on='date')

    hedge = np.zeros(len(df))
    for name in FACTOR_NAMES:
        hedge = hedge + df[f'beta_{name}'].values * df[f'{name}_factor'].values
    df['residual_return'] = df['raw_return'] - hedge

    out = df[['raw_return', 'residual_return']].copy()

    # Forward targets: sum of the NEXT p bars -> rolling(p).sum().shift(-p).
    # min_periods=p so gaps yield NaN targets instead of partial sums.
    for h in HORIZONS:
        p = horizon_bars(h)
        out[horizon_col(h, 'res')] = (
            out['residual_return'].rolling(p, min_periods=p).sum().shift(-p)
        )
        out[horizon_col(h, 'raw')] = (
            out['raw_return'].rolling(p, min_periods=p).sum().shift(-p)
        )

    out = out.reset_index().rename(columns={'index': 'timestamp'})
    out['symbol'] = symbol
    return out


def _process_symbol(args):
    """Worker: betas + residuals for one symbol."""
    symbol, returns_series, factors = args
    betas = estimate_daily_betas(symbol, returns_series, factors)
    residuals = compute_symbol_residuals(symbol, returns_series, factors, betas)
    return betas, residuals


def run_acceptance_checks(residuals: pd.DataFrame) -> bool:
    """Verify the factor model is actually removing systematic variance."""
    ok = True
    sub = residuals.dropna(subset=['raw_return', 'residual_return'])
    var_ratio = sub['residual_return'].var() / sub['raw_return'].var()
    corr = sub['residual_return'].corr(sub['raw_return'])

    logging.info("=" * 70)
    logging.info("ACCEPTANCE CHECKS (single-bar residuals)")
    logging.info(f"  var(residual)/var(raw) = {var_ratio:.4f} "
                 f"(must be <= {1 - ACCEPT_MIN_VAR_REDUCTION:.2f})")
    logging.info(f"  corr(residual, raw)    = {corr:.4f} "
                 f"(must be <= {ACCEPT_MAX_CORR:.2f})")

    if var_ratio > 1 - ACCEPT_MIN_VAR_REDUCTION:
        logging.error("FAIL: factor model is not reducing variance - "
                      "residuals are effectively raw returns. Do NOT proceed.")
        ok = False
    if corr > ACCEPT_MAX_CORR:
        logging.error("FAIL: residuals are nearly identical to raw returns.")
        ok = False

    for h in HORIZONS:
        rc, cc = horizon_col(h, 'res'), horizon_col(h, 'raw')
        s = residuals.dropna(subset=[rc, cc])
        if len(s) > 1000:
            vr = s[rc].var() / s[cc].var()
            logging.info(f"  {h}: var(fwd_res)/var(fwd_raw) = {vr:.4f}")

    if ok:
        logging.info("PASS")
    logging.info("=" * 70)
    return ok


def main():
    logging.info("Loading prices and factors...")
    px = load_data('prices', columns=['timestamp', 'symbol', 'close'])
    if px.empty:
        raise RuntimeError("prices table is empty")
    px['timestamp'] = pd.to_datetime(px['timestamp'])

    factors = load_data('risk_factors')
    if factors.empty:
        raise RuntimeError("risk_factors is empty - run risk_model/factor_returns.py first")
    factors['timestamp'] = pd.to_datetime(factors['timestamp'])
    if getattr(factors['timestamp'].dt, 'tz', None) is not None:
        factors['timestamp'] = factors['timestamp'].dt.tz_localize(None)
    factors = factors.set_index('timestamp').sort_index()
    # Guard the hedge inputs too: a non-finite factor would make residuals inf.
    present_factor_cols = [c for c in FACTOR_COLS if c in factors.columns]
    factors[present_factor_cols] = factors[present_factor_cols].replace(
        [np.inf, -np.inf], np.nan
    )

    close_wide = px.pivot_table(index='timestamp', columns='symbol', values='close',
                                aggfunc='last').sort_index()
    full_index = pd.date_range(close_wide.index.min(), close_wide.index.max(), freq=BASE_FREQ)
    close_wide = close_wide.reindex(full_index)
    # Non-positive closes are invalid data (delisted/illiquid glitches). Masking
    # them to NaN avoids both the spurious -100% return into a zero and the
    # divide-by-zero that makes the *next* return +inf.
    close_wide = close_wide.where(close_wide > 0)
    returns_wide = close_wide.pct_change(fill_method=None)
    # Belt-and-suspenders: any remaining non-finite return -> NaN, so it can
    # never reach the regression or pollute acceptance-check variances.
    returns_wide = returns_wide.replace([np.inf, -np.inf], np.nan)

    symbols = list(returns_wide.columns)
    logging.info(f"Estimating betas + residuals for {len(symbols)} symbols "
                 f"({BETA_WINDOW_DAYS}d window, {BETA_HALFLIFE_DAYS}d halflife, daily refresh)")

    args_list = [(sym, returns_wide[sym], factors) for sym in symbols]

    all_betas, all_residuals = [], []
    with get_parallel_executor(MAX_WORKERS) as executor:
        futures = {executor.submit(_process_symbol, a): a[0] for a in args_list}
        from tqdm import tqdm
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Symbols"):
            sym = futures[fut]
            try:
                betas, residuals = fut.result()
                if not betas.empty:
                    all_betas.append(betas)
                if not residuals.empty:
                    all_residuals.append(residuals)
            except Exception as e:
                logging.error(f"{sym}: {e}")

    if not all_betas or not all_residuals:
        raise RuntimeError("No betas/residuals computed")

    betas_df = pd.concat(all_betas, ignore_index=True)
    residuals_df = pd.concat(all_residuals, ignore_index=True)

    # Drop rows with no residual at all (e.g. pre-history)
    residuals_df = residuals_df.dropna(subset=['residual_return'], how='all')

    logging.info(f"Betas: {len(betas_df):,} symbol-days | "
                 f"Residuals: {len(residuals_df):,} symbol-bars")
    for name in FACTOR_NAMES:
        col = f'beta_{name}'
        logging.info(f"  beta_{name}: mean={betas_df[col].mean():.3f} "
                     f"std={betas_df[col].std():.3f}")
    logging.info(f"  r_squared: mean={betas_df['r_squared'].mean():.3f}")

    passed = run_acceptance_checks(residuals_df)

    delete_table('factor_loadings')
    delete_table('residual_returns')
    save_data('factor_loadings', betas_df, mode='overwrite', datetime_columns=['date'])
    save_data('residual_returns', residuals_df, mode='overwrite', datetime_columns=['timestamp'])
    logging.info("Saved factor_loadings and residual_returns")

    if not passed:
        logging.error("Acceptance checks FAILED - inspect the factor model before "
                      "running features/signals.")
        sys.exit(1)


if __name__ == '__main__':
    main()
