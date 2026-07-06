"""
Synthetic-data sanity checks for the rebuilt pipeline. No database required.

Run: uv run python tests/sanity_checks.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

rng = np.random.default_rng(7)
FAILURES = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name} {detail}")
    if not cond:
        FAILURES.append(name)


# ---------------------------------------------------------------------------
# 1. Config frequency math
# ---------------------------------------------------------------------------
from config import get_frequency_config, horizon_bars, horizon_col, HORIZONS, BASE_FREQUENCY

check("freq: 10min bars/day", get_frequency_config('10min')['bars_per_day'] == 144)
check("freq: exchange 1m means one minute",
      get_frequency_config('1m')['bars_per_day'] == 1440)
check("freq: 1h bars/day", get_frequency_config('1h')['bars_per_day'] == 24)
check("freq: horizon bars 10min", horizon_bars('10min') == 1)
check("freq: horizon bars 1h", horizon_bars('1h') == 6)
check("freq: horizon bars 1d", horizon_bars('1d') == 144)
check("freq: horizon col", horizon_col('1h', 'res') == 'fwd_res_1h')

# ---------------------------------------------------------------------------
# 2. Forward-target alignment (the C1 fix): rolling(p).sum().shift(-p)
#    must equal sum of bars t+1 .. t+p
# ---------------------------------------------------------------------------
x = pd.Series(rng.normal(size=500))
for p in [1, 6, 144]:
    fast = x.rolling(p, min_periods=p).sum().shift(-p)
    slow = pd.Series([x.iloc[t + 1: t + p + 1].sum() if t + p < len(x) else np.nan
                      for t in range(len(x))])
    ok = np.allclose(fast.dropna().values, slow.dropna().values) and \
        fast.notna().sum() == slow.notna().sum()
    check(f"alignment: fwd sum p={p}", ok)

# NaN gaps must produce NaN targets, not partial sums
xg = x.copy()
xg.iloc[100] = np.nan
fast = xg.rolling(6, min_periods=6).sum().shift(-6)
check("alignment: NaN gap poisons overlapping targets",
      fast.iloc[94:100].isna().all())

# ---------------------------------------------------------------------------
# 3. Beta recovery + residual variance reduction on synthetic factor data
# ---------------------------------------------------------------------------
from risk_model.residual_returns import (estimate_daily_betas,
                                         compute_symbol_residuals, FACTOR_COLS)

n_days, bpd = 90, 144
n = n_days * bpd
idx = pd.date_range('2024-01-01', periods=n, freq='10min')
f_mkt = pd.Series(rng.normal(0, 0.002, n), index=idx)
f_size = pd.Series(rng.normal(0, 0.001, n), index=idx)
true_b = (1.3, -0.5)
eps = rng.normal(0, 0.001, n)
r = true_b[0] * f_mkt + true_b[1] * f_size + eps

factors = pd.DataFrame({'market_factor': f_mkt, 'size_factor': f_size}, index=idx)
# Any additional configured factors (momentum, vol, ...) enter as independent
# noise with zero true loading, so the known market/size betas must still be
# recovered regardless of how many factors the risk model defines.
for col in FACTOR_COLS:
    if col not in factors.columns:
        factors[col] = pd.Series(rng.normal(0, 0.001, n), index=idx)
betas = estimate_daily_betas('TEST', r, factors)

check("betas: produced daily rows", len(betas) > 30, f"({len(betas)} days)")
b_mkt_err = abs(betas['beta_market'].iloc[-20:].mean() - true_b[0])
b_size_err = abs(betas['beta_size'].iloc[-20:].mean() - true_b[1])
check("betas: market recovered", b_mkt_err < 0.05, f"(err {b_mkt_err:.4f})")
check("betas: size recovered", b_size_err < 0.08, f"(err {b_size_err:.4f})")
check("betas: causal (first beta date > data start)",
      pd.Timestamp(betas['date'].min()) > idx[0].normalize())

res = compute_symbol_residuals('TEST', r, factors, betas)
sub = res.dropna(subset=['residual_return'])
var_ratio = sub['residual_return'].var() / sub['raw_return'].var()
true_ratio = eps.var() / r.var()
check("residuals: variance reduced toward idiosyncratic share",
      abs(var_ratio - true_ratio) < 0.05, f"(got {var_ratio:.3f}, true {true_ratio:.3f})")

# Forward target columns exist and are aligned
fcol = horizon_col('1h', 'res')
got = res.set_index('timestamp')[fcol]
manual = res.set_index('timestamp')['residual_return']
t0 = got.index[5000]
expected = manual.iloc[5001:5007].sum()
check("residuals: fwd_res_1h = sum of next 6 bars",
      np.isclose(got.loc[t0], expected), f"(got {got.loc[t0]:.6f} vs {expected:.6f})")

# ---------------------------------------------------------------------------
# 4. MVO: constraints satisfied
# ---------------------------------------------------------------------------
from research.lib.portfolio_opt import (shrunk_covariance, solve_constrained_mvo,
                                              benjamini_hochberg,
                                              benjamini_yekutieli)

n_assets = 60
T = 2000
true_cov = np.diag(rng.uniform(0.5, 2.0, n_assets) * 1e-6)
rets = pd.DataFrame(rng.multivariate_normal(np.zeros(n_assets), true_cov, T),
                    columns=[f'A{i}' for i in range(n_assets)])
cov = shrunk_covariance(rets, min_observations=100)
check("cov: full universe kept", cov.shape == (n_assets, n_assets))
check("cov: positive definite", np.all(np.linalg.eigvalsh(cov.values) > 0))

alpha = pd.Series(rng.normal(0, 1e-4, n_assets), index=cov.index)
beta_m = pd.Series(rng.uniform(0.5, 1.5, n_assets), index=cov.index)
beta_s = pd.Series(rng.normal(0, 0.5, n_assets), index=cov.index)
A = pd.DataFrame({'dollar': 1.0, 'market': beta_m, 'size': beta_s})

w = solve_constrained_mvo(alpha, cov, A, max_position=0.05, gross_leverage=1.0)
check("mvo: dollar-neutral", abs(w.sum()) < 1e-6, f"(sum {w.sum():.2e})")
check("mvo: market-beta-neutral", abs((w * beta_m).sum()) < 1e-6,
      f"(exp {(w * beta_m).sum():.2e})")
check("mvo: size-beta-neutral", abs((w * beta_s).sum()) < 1e-6,
      f"(exp {(w * beta_s).sum():.2e})")
check("mvo: gross leverage 1", abs(w.abs().sum() - 1.0) < 1e-6,
      f"(gross {w.abs().sum():.6f})")
check("mvo: position cap respected", (w.abs() <= 0.05 + 1e-9).all(),
      f"(max |w| {w.abs().max():.4f})")
check("mvo: aligned with alpha", (w * alpha).sum() > 0)

# ---------------------------------------------------------------------------
# 5. Benjamini-Hochberg
# ---------------------------------------------------------------------------
# BH at alpha=.05, m=10: thresholds .005,.010,.015,... -> largest k with
# p_(k) <= k*alpha/m is k=2 (.001<=.005, .008<=.010, .039>.015)
p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.5, 0.99])
mask = benjamini_hochberg(p, alpha=0.05)
check("fdr: known example", mask.sum() == 2 and mask[:2].all(),
      f"(discoveries {mask.sum()})")
mask_null = benjamini_hochberg(rng.uniform(size=1000), alpha=0.05)
check("fdr: nulls mostly rejected", mask_null.mean() < 0.02,
      f"(rate {mask_null.mean():.3f})")
mask_nan = benjamini_hochberg(np.array([np.nan, 0.0001, np.nan]), alpha=0.05)
check("fdr: NaN handled", mask_nan.tolist() == [False, True, False])
by = benjamini_yekutieli(p, alpha=0.05)
check("fdr: BY no less conservative than BH",
      by.sum() <= mask.sum(), f"(BY {by.sum()} vs BH {mask.sum()})")

# ---------------------------------------------------------------------------
# 6. Rank IC vectorization matches scipy
# ---------------------------------------------------------------------------
import research.signals.evaluate as srmod
rank_ic_per_timestamp = srmod.rank_ic_per_timestamp
from scipy.stats import spearmanr

_orig_min_assets = srmod.MIN_ASSETS
srmod.MIN_ASSETS = 10

ts_idx = np.repeat(pd.date_range('2024-01-01', periods=30, freq='1h'), 40)
df_ic = pd.DataFrame({
    'timestamp': ts_idx,
    'signal': rng.normal(size=1200),
})
df_ic['tgt'] = 0.3 * df_ic['signal'] + rng.normal(size=1200)
ics = rank_ic_per_timestamp(df_ic, 'tgt')
manual = df_ic.groupby('timestamp').apply(
    lambda g: spearmanr(g['signal'], g['tgt'])[0], include_groups=False)
check("ic: matches scipy spearman", np.allclose(ics.set_index('timestamp')['ic'],
                                                manual, atol=1e-10))
srmod.MIN_ASSETS = _orig_min_assets

# ---------------------------------------------------------------------------
# 7. Signal registry builds against new feature names
# ---------------------------------------------------------------------------
from research.signals.evaluate import build_registry, compute_signal_panel, signal_feature_columns

registry = build_registry()
check("registry: spaces present", len(registry) >= 20, f"({len(registry)} spaces)")

space_cols = sorted({c for info in registry.values()
                     for c in signal_feature_columns(info['signal_def'])})
n_sym, n_ts = 12, 600
sym = np.repeat([f'S{i}' for i in range(n_sym)], n_ts)
tss = np.tile(pd.date_range('2024-01-01', periods=n_ts, freq='10min'), n_sym)
feat = pd.DataFrame({'timestamp': tss, 'symbol': sym})
for c in space_cols:
    feat[c] = rng.normal(size=len(feat))

errors = []
for s in registry:
    try:
        out = compute_signal_panel(s, registry, feat)
        if out.empty or out['signal'].abs().max() > 3.0001:
            errors.append((s, 'bad output'))
    except Exception as e:
        errors.append((s, str(e)[:60]))
check("signals: all spaces compute cleanly", len(errors) == 0, f"{errors[:3]}")

# ---------------------------------------------------------------------------
# 8. OU rolling fit recovers mean reversion on synthetic OU spread
# ---------------------------------------------------------------------------
from risk_model.features import _ou_params_rolling

lam_true, mu_true, sig_true = 0.05, 0.0, 0.01
xs = [0.0]
for _ in range(5000):
    xs.append(xs[-1] + lam_true * (mu_true - xs[-1]) + rng.normal(0, sig_true))
xs = np.array(xs)
lam, mu, sig = _ou_params_rolling(xs, window=1008, min_obs=100)
lam_est = np.nanmedian(lam[2000:])
check("ou: lambda recovered", abs(lam_est - lam_true) < 0.03, f"(est {lam_est:.4f})")

# ---------------------------------------------------------------------------
# 9. Factor returns: member masking + rank-weighted size (synthetic thresholds)
# ---------------------------------------------------------------------------
import risk_model.factor_returns as frmod

f_idx = pd.date_range('2024-01-01', periods=20 * 144, freq='10min')
f_syms = [f'S{i}' for i in range(12)]
f_rets = pd.DataFrame(rng.normal(0, 1e-3, (len(f_idx), 12)), index=f_idx, columns=f_syms)
f_rets[f_syms[:6]] += 1e-4   # small caps drift up
f_rets[f_syms[6:]] -= 1e-4   # big caps drift down

f_membership = {pd.Period('2024-01', 'M'): set(f_syms[:10])}  # S10/S11 not members
f_dates = pd.date_range('2023-12-25', '2024-01-25', freq='D')
f_mcap = pd.DataFrame({s: (1e8 * (1 + i * 0.01) if i < 6 else 1e10 * (1 + i * 0.01))
                       for i, s in enumerate(f_syms)}, index=f_dates)

f_out = frmod.compute_factor_returns(f_rets, f_membership, f_mcap)
mkt = f_out.set_index('timestamp')['market_factor'].dropna()
manual_mkt = f_rets[f_syms[:10]].mean(axis=1)
check("factors: market = EW of universe members only",
      np.allclose(mkt.values, manual_mkt.loc[mkt.index].values))

sf = f_out.set_index('timestamp')['size_factor'].dropna()
check("factors: size factor produced", len(sf) > 1000, f"({len(sf)} bars)")
check("factors: small-minus-big spread sign", sf.mean() > 1e-4,
      f"(mean {sf.mean() * 1e4:.2f} bp/bar)")

# Manual replication of rank weights among the 10 members (no NaNs in the
# synthetic panel, so per-bar renormalization equals the static weights)
row = f_mcap.iloc[-1][f_syms[:10]]
ranks = row.rank(method='first')
w = -(ranks - ranks.mean())
w[w > 0] = w[w > 0] / w[w > 0].sum()
w[w < 0] = w[w < 0] / (-w[w < 0]).sum()
manual_sf = (f_rets[f_syms[:10]] * w).sum(axis=1)
aligned = sf.reindex(manual_sf.index).dropna()
check("factors: size matches manual rank-weighted spread",
      np.allclose(aligned.values, manual_sf.loc[aligned.index].values))

f_mcap_tied = pd.DataFrame({s: 1e9 for s in f_syms}, index=f_dates)
sf_tied = frmod.compute_factor_returns(f_rets, f_membership, f_mcap_tied)['size_factor'].dropna()
check("factors: tied mcaps still produce a factor", len(sf_tied) > 1000)

# ---------------------------------------------------------------------------
# 10. Cluster detection + soft cluster penalty in MVO
# ---------------------------------------------------------------------------
from research.lib.portfolio_opt import residual_clusters, cluster_penalty_matrix

# Synthetic residuals: 2 planted clusters (0-7 memes, 8-15 ai) + 14 idio names
Tc, n_c = 4000, 30
common1 = rng.normal(0, 8e-4, Tc)
common2 = rng.normal(0, 8e-4, Tc)
R = rng.normal(0, 1e-3, (Tc, n_c))
R[:, :8] += common1[:, None]
R[:, 8:16] += common2[:, None]
cl_rets = pd.DataFrame(R, columns=[f'M{i}' for i in range(8)] +
                       [f'A{i}' for i in range(8)] + [f'I{i}' for i in range(14)])

clusters = residual_clusters(cl_rets, corr_threshold=0.30, min_cluster_size=3)
found_m = any(set(c) >= {'M0', 'M1', 'M2', 'M3'} for c in clusters)
found_a = any(set(c) >= {'A0', 'A1', 'A2', 'A3'} for c in clusters)
no_idio = all(not any(m.startswith('I') for m in c) for c in clusters)
check("clusters: planted clusters recovered", found_m and found_a,
      f"({len(clusters)} clusters)")
check("clusters: idio names not clustered", no_idio)

# Penalty reduces cluster exposure while constraints still hold
cl_cov = shrunk_covariance(cl_rets, min_observations=100)
cl_alpha = pd.Series(rng.normal(0, 1e-4, n_c), index=cl_cov.index)
cl_alpha[:8] = cl_alpha[:8].abs() + 1e-4   # alpha loves the meme cluster
cl_beta = pd.Series(rng.uniform(0.6, 1.4, n_c), index=cl_cov.index)
cl_A = pd.DataFrame({'dollar': 1.0, 'market': cl_beta})

w_plain = solve_constrained_mvo(cl_alpha, cl_cov, cl_A, max_position=0.10)
cov_pen = cluster_penalty_matrix(cl_cov, clusters, lam=1.0)
w_pen = solve_constrained_mvo(cl_alpha, cov_pen, cl_A, max_position=0.10)

meme_names = [c for c in cl_cov.index if c.startswith('M')]
exp_plain = abs(w_plain[meme_names].sum())
exp_pen = abs(w_pen[meme_names].sum())
check("clusters: penalty reduces cluster exposure", exp_pen < exp_plain,
      f"({exp_plain:.4f} -> {exp_pen:.4f})")
check("clusters: constraints still exact under penalty",
      abs(w_pen.sum()) < 1e-6 and abs((w_pen * cl_beta).sum()) < 1e-6 and
      abs(w_pen.abs().sum() - 1.0) < 1e-6)

# Degenerate constraints (collinear columns, e.g. all betas identical) must
# still be enforced exactly - the solver uses rank-revealing projections
cl_A_degen = pd.DataFrame({'dollar': 1.0, 'market': 1.0}, index=cl_cov.index)
w_degen = solve_constrained_mvo(cl_alpha, cl_cov, cl_A_degen, max_position=0.10)
check("mvo: collinear constraint columns handled",
      abs(w_degen.sum()) < 1e-6 and abs(w_degen.abs().sum() - 1.0) < 1e-6,
      f"(sum {w_degen.sum():.2e})")

# ---------------------------------------------------------------------------
# 11. Feature truncation test (gold-standard look-ahead detector):
#     feature values at time T must be IDENTICAL whether or not data after T
#     exists. With the shift(1) convention removed, this test is the
#     causality guarantee.
# ---------------------------------------------------------------------------
from risk_model.features import (calculate_all_features, compute_intrabar_features,
                                 FEATURE_CONFIG)
from config import BARS_PER_DAY as _BPD

def make_synth_panel(n_bars, seed=42):
    r = np.random.default_rng(seed)
    ts = pd.date_range('2024-01-01', periods=n_bars, freq='10min')
    close = 100 * np.exp(np.cumsum(r.normal(0, 1e-3, n_bars)))
    spread_hl = np.abs(r.normal(0, 2e-3, n_bars))
    df = pd.DataFrame({
        'timestamp': ts, 'symbol': 'SYN',
        'open': close * (1 + r.normal(0, 5e-4, n_bars)),
        'high': close * (1 + spread_hl),
        'low': close * (1 - spread_hl),
        'close': close,
        'volume': r.uniform(10, 100, n_bars),
        'quote_asset_volume': r.uniform(1e5, 1e6, n_bars),
        'number_of_trades': r.integers(50, 500, n_bars).astype('int64'),
        'taker_buy_base_asset_volume': r.uniform(5, 50, n_bars),
        'taker_buy_quote_asset_volume': r.uniform(5e4, 5e5, n_bars),
    })
    res = pd.Series(r.normal(0, 8e-4, n_bars), index=df.index)
    factors = pd.DataFrame({'market_factor': r.normal(0, 2e-3, n_bars)},
                           index=ts)
    dates = pd.date_range('2024-01-01', periods=n_bars // _BPD + 2, freq='D')
    loadings = pd.DataFrame({
        'date': dates, 'symbol': 'SYN',
        'beta_market': 1.0 + r.normal(0, 0.05, len(dates)),
        'beta_size': r.normal(0, 0.3, len(dates)),
        'r_squared': r.uniform(0.3, 0.7, len(dates)),
    })
    leader = pd.Series(100 * np.exp(np.cumsum(r.normal(0, 1e-3, n_bars))),
                       index=ts)
    return df, res, factors, loadings, leader

N_FULL, N_TRUNC = 2400, 1800
df_f, res_f, fac_f, ld_f, lead_f = make_synth_panel(N_FULL)
full = calculate_all_features(df_f, FEATURE_CONFIG, 'SYN', res_f, ld_f,
                              factors_df=fac_f, leader_close=lead_f)

df_t = df_f.iloc[:N_TRUNC].copy()
res_t = res_f.iloc[:N_TRUNC]
fac_t = fac_f.iloc[:N_TRUNC]
cutoff_ts = df_t['timestamp'].iloc[-1]
ld_t = ld_f[ld_f['date'] <= cutoff_ts]
trunc = calculate_all_features(df_t, FEATURE_CONFIG, 'SYN', res_t, ld_t,
                               factors_df=fac_t,
                               leader_close=lead_f.iloc[:N_TRUNC])

feat_cols = [c for c in full.columns if c not in ('timestamp', 'symbol')]
row_full = full.iloc[N_TRUNC - 1][feat_cols].astype(float).values
row_trunc = trunc.iloc[N_TRUNC - 1][feat_cols].astype(float).values
mismatch = [feat_cols[i] for i in range(len(feat_cols))
            if not (np.isclose(row_full[i], row_trunc[i], atol=1e-5, rtol=1e-4)
                    or (np.isnan(row_full[i]) and np.isnan(row_trunc[i])))]
check("features: truncation test (no look-ahead in any feature)",
      len(mismatch) == 0, f"leaking: {mismatch[:6]}")

# Same for intra-bar features computed from 1m data
n1m = 1800 * 10
r = np.random.default_rng(5)
ts1m = pd.date_range('2024-01-01 00:00:59.999', periods=n1m, freq='1min')
df1m = pd.DataFrame({
    'timestamp': ts1m,
    'close': 100 * np.exp(np.cumsum(r.normal(0, 3e-4, n1m))),
    'volume': np.where(r.random(n1m) < 0.02, 0.0, r.uniform(1, 10, n1m)),
})
ib_full = compute_intrabar_features(df1m, FEATURE_CONFIG).set_index('timestamp')
cut = 12000  # exact 10-min boundary (multiple of 10)
ib_trunc = compute_intrabar_features(df1m.iloc[:cut], FEATURE_CONFIG).set_index('timestamp')
common_ts = ib_trunc.index[-1]
a = ib_full.loc[common_ts].astype(float).values
b = ib_trunc.loc[common_ts].astype(float).values
ok = all(np.isclose(x, y, atol=1e-8) or (np.isnan(x) and np.isnan(y))
         for x, y in zip(a, b))
check("features: intra-bar truncation test", ok)

# ---------------------------------------------------------------------------
# 12. True variance ratio + Hurst recover known dynamics
# ---------------------------------------------------------------------------
from risk_model.features import _variance_ratio

rngv = np.random.default_rng(9)
n = 20000
# AR(1) with phi = -0.3 -> VR(6) well below 1
ar = np.zeros(n)
for i in range(1, n):
    ar[i] = -0.3 * ar[i - 1] + rngv.normal()
vr_ar = _variance_ratio(pd.Series(ar), q=6, window=2000).dropna().median()
# Random walk increments -> VR ~ 1
vr_rw = _variance_ratio(pd.Series(rngv.normal(size=n)), q=6, window=2000).dropna().median()
check("vr: AR(1) phi=-0.3 gives VR << 1", vr_ar < 0.75, f"(VR {vr_ar:.3f})")
check("vr: random walk gives VR ~ 1", 0.85 < vr_rw < 1.15, f"(VR {vr_rw:.3f})")

# Hurst via VR identity: mean-reverting spread -> H < 0.5
lam_t = 0.1
ou_x = [0.0]
for _ in range(n):
    ou_x.append(ou_x[-1] * (1 - lam_t) + rngv.normal(0, 0.01))
ou_inc = pd.Series(np.diff(ou_x))
vr_ou = _variance_ratio(ou_inc, q=36, window=2000)
hurst = (0.5 + np.log(vr_ou) / (2 * np.log(36))).dropna().median()
check("hurst: OU spread gives H < 0.5", hurst < 0.4, f"(H {hurst:.3f})")

# ---------------------------------------------------------------------------
# 13. Schema completeness: every feature the SPACES reference is produced
# ---------------------------------------------------------------------------
produced = set(full.columns) - {'timestamp', 'symbol'}
missing_feats = [c for c in space_cols if c not in produced]
check("schema: all space feature columns produced", len(missing_feats) == 0,
      f"missing: {missing_feats[:8]}")

# ---------------------------------------------------------------------------
# 14. Funding-PnL accrual: a settlement stamp maps to the bar whose forward
#     interval (t, t+1bar] contains it, and longs PAY positive rates
#     (mirrors the exact reindex expression in walk_forward._backtest_window)
# ---------------------------------------------------------------------------
bars = pd.date_range('2024-01-01 00:10', periods=144, freq=BASE_FREQUENCY)
fund_wide = pd.DataFrame({'AAA': [0.0010], 'BBB': [-0.0005]},
                         index=pd.DatetimeIndex([pd.Timestamp('2024-01-01 08:00')]))
fund_cols = pd.Index(['AAA', 'BBB'])
fund_all = fund_wide.reindex(index=bars + pd.Timedelta(BASE_FREQUENCY),
                             columns=fund_cols).to_numpy()
i_hit = list(bars).index(pd.Timestamp('2024-01-01 07:50'))
check("funding: settlement lands on the bar holding through it",
      np.isfinite(fund_all[i_hit]).all()
      and np.isnan(fund_all[i_hit + 1]).all()
      and np.isnan(fund_all[i_hit - 1]).all())
w_fund = np.array([0.5, -0.5])   # long AAA (pays +10bp/1x), short BBB (rate<0: short pays)
pnl_fund = -float(w_fund @ np.nan_to_num(fund_all[i_hit], nan=0.0))
check("funding: long pays positive rate, short pays negative rate",
      np.isclose(pnl_fund, -(0.5 * 0.0010) - (0.5 * 0.0005)),
      f"(pnl {pnl_fund:+.5f})")

# ---------------------------------------------------------------------------
# 15. Execution-derived selection speed floor (min_holding_lag_bars: 'auto')
# ---------------------------------------------------------------------------
import research.portfolio.walk_forward as wfmod

nominal_rate, kappa_eff = wfmod.effective_fill_rate()
gp_cfg = wfmod.PORT.get('gp_trading', {})
budget_ann = wfmod.PORT.get('max_annual_turnover')
per_bar_budget = budget_ann / (wfmod.BARS_PER_DAY * 365) if budget_ann else np.inf
exp_kappa = max(nominal_rate, 1e-9)
if gp_cfg.get('discount_at_realized_rate', True) and np.isfinite(per_bar_budget):
    exp_kappa = max(min(exp_kappa,
                        per_bar_budget / max(wfmod.PORT['gross_leverage'], 1e-9)),
                    1e-9)
check("fill rate: kappa = min(GP rate, budget-allowed rate)",
      np.isclose(kappa_eff, exp_kappa),
      f"(nominal {nominal_rate:.4f}, kappa {kappa_eff:.5f})")

lag_floor = wfmod.resolve_min_holding_lag()
mhl_cfg = wfmod.WF.get('min_holding_lag_bars')
if mhl_cfg == 'auto' and gp_cfg.get('enabled'):
    frac = float(wfmod.WF['min_monetizable_alpha_fraction'])
    disc = lambda h: h / (h + 1.0 / kappa_eff)
    check("speed floor: smallest lag whose aim discount clears the fraction",
          lag_floor >= 1 and disc(lag_floor) >= frac
          and (lag_floor == 1 or disc(lag_floor - 1) < frac),
          f"(floor {lag_floor} bars, discount {disc(lag_floor):.3f} >= {frac})")
else:
    check("speed floor: manual config passthrough",
          lag_floor == int(mhl_cfg or 0), f"(floor {lag_floor})")

# corr().values can be READ-ONLY under pandas copy-on-write; breadth /
# combination must take writable copies before fill_diagonal (regression:
# ValueError "underlying array is read-only" crashed select_decay).
rets_cow = pd.DataFrame(rng.normal(size=(60, 3)), columns=list('abc'),
                        index=pd.date_range('2024-01-01', periods=60))
eb_cow = wfmod.SignalSelector._effective_breadth(rets_cow, list('abc'))
check("selector: effective breadth CoW-safe", 1.0 <= eb_cow <= 3.0,
      f"(effN {eb_cow:.2f})")

# ---------------------------------------------------------------------------
# 16. Point-in-time universe membership: spell evolution + interval mask
# ---------------------------------------------------------------------------
from etl.universe import evolve_membership
from research.signals.evaluate import universe_member_mask

seed_date = pd.Timestamp('2023-01-01')
mem0, n_new0, _ = evolve_membership(None, {'AAA', 'BBB'}, '2025-06-01', seed_date)
check("membership: initial cohort seeded from the data start",
      n_new0 == 2 and (mem0['valid_from'] == seed_date).all()
      and mem0['valid_to'].isna().all())

mem1, n_new1, n_closed1 = evolve_membership(mem0, {'AAA', 'CCC'},
                                            '2025-07-01', seed_date)
check("membership: delisting closes / listing opens spells",
      n_new1 == 1 and n_closed1 == 1
      and mem1.loc[mem1['symbol'] == 'BBB', 'valid_to'].notna().all()
      and mem1.loc[mem1['symbol'] == 'AAA', 'valid_to'].isna().all())

mem2, n_new2, _ = evolve_membership(mem1, {'AAA', 'BBB', 'CCC'},
                                    '2025-09-01', seed_date)
check("membership: relisting opens a second spell",
      n_new2 == 1 and (mem2['symbol'] == 'BBB').sum() == 2)

mask_panel = pd.DataFrame({
    'timestamp': pd.to_datetime(['2025-06-15', '2025-08-01', '2025-10-01',
                                 '2025-06-15', '2024-01-01']),
    'symbol': ['BBB', 'BBB', 'BBB', 'CCC', 'AAA'],
})
mem_mask = universe_member_mask(mask_panel, mem2)
check("membership: interval mask (member / delisted gap / relisted)",
      list(mem_mask) == [True, False, True, False, True],
      f"(got {list(mem_mask)})")

# ---------------------------------------------------------------------------
# 16b. Per-lag smoothing halflife + edge-scaled gross helpers
# ---------------------------------------------------------------------------
from research.signals.evaluate import smoothing_halflife_for_lag
from research.portfolio.walk_forward import edge_gross_multiplier
from config import get as _cfg_get

_ls = _cfg_get('signals.lag_smoothing') or []
if _ls:
    check("lag smoothing: first bucket for fastest lag",
          smoothing_halflife_for_lag(1, 0.0) == float(_ls[0][1]))
    check("lag smoothing: beyond last bound uses last bucket",
          smoothing_halflife_for_lag(int(_ls[-1][0]) * 10, 0.0)
          == float(_ls[-1][1]))
    check("lag smoothing: bucket boundary is inclusive",
          smoothing_halflife_for_lag(int(_ls[0][0]), 0.0) == float(_ls[0][1]))
    check("lag smoothing: base halflife is a floor",
          smoothing_halflife_for_lag(1, 999.0) == 999.0)
else:
    check("lag smoothing: disabled -> base halflife",
          smoothing_halflife_for_lag(144, 3.0) == 3.0)

check("edge gross: free execution -> full gross",
      edge_gross_multiplier(0.0, 0.0, 2.0) == 1.0
      and edge_gross_multiplier(-1.0, 0.0, 2.0) == 1.0)
check("edge gross: edge covers edge_mult round trips -> full",
      edge_gross_multiplier(4e-4, 2e-4, 2.0) == 1.0)
check("edge gross: half coverage -> half gross",
      abs(edge_gross_multiplier(2e-4, 2e-4, 2.0) - 0.5) < 1e-12)
check("edge gross: negative edge -> zero gross",
      edge_gross_multiplier(-1e-4, 2e-4, 2.0) == 0.0)

# ---------------------------------------------------------------------------
# 17. Factor VIF: independent factors ~ 1, a collinear factor blows up
# ---------------------------------------------------------------------------
from risk_model.residual_returns import factor_vif, FACTOR_COLS as _VIF_COLS

rngf = np.random.default_rng(11)
n_vif = 5000
ind_factors = pd.DataFrame({c: rngf.normal(size=n_vif) for c in _VIF_COLS})
vif_ind = factor_vif(ind_factors)
check("vif: independent factors ~ 1",
      bool(vif_ind) and all(v < 1.2 for v in vif_ind.values()),
      f"({ {k: round(v, 2) for k, v in vif_ind.items()} })")

col_factors = ind_factors.copy()
col_factors[_VIF_COLS[1]] = (col_factors[_VIF_COLS[0]] * 0.98
                             + rngf.normal(0, 0.05, n_vif))
vif_col = factor_vif(col_factors)
check("vif: collinear factor detected", max(vif_col.values()) > 10,
      f"(max VIF {max(vif_col.values()):.1f})")

# ---------------------------------------------------------------------------
print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURES: {FAILURES}")
    sys.exit(1)
print("ALL SANITY CHECKS PASSED")
