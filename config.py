"""
Central Configuration for Crypto Trading System.

System-wide operational and model parameters live here.

Architecture (2026-06):
- Raw data: 1-minute Binance spot klines, from `data.start_date` onward.
- Base panel: 10-minute bars ('10min'). All features/signals/residuals on this grid.
- Horizons: multi-period forward residual targets at 10min / 1h / 1d.
- Universe: Hyperliquid-tradeable perps (no stablecoins); point-in-time
  membership spells recorded by etl/universe.py (pre-snapshot history seeded
  from the data start).
- Risk model: market (equal-weight) + size / momentum / vol (rank-weighted
  spreads). Residual[t] = r[t] - sum_f beta_f * factor_f[t], causal daily betas.
- Portfolio: rank/equal-weight sizing (Ledoit-Wolf MVO benchmark), dollar +
  factor-beta neutral, net of trading costs and perp funding accrual.
"""

import math
import os
from datetime import datetime, timedelta


def _total_ram_bytes():
    try:
        import psutil
        return int(psutil.virtual_memory().total)
    except Exception:
        try:
            return int(os.sysconf('SC_PHYS_PAGES') * os.sysconf('SC_PAGE_SIZE'))
        except (ValueError, OSError, AttributeError):
            return 16 * 1024 ** 3


def _is_wsl():
    """Running under WSL2? Its memory reclaim is poor: once the Linux VM fills
    RAM and starts swapping, the whole Windows host can freeze (swap thrash is
    not returned to the host), so we budget workers far more conservatively."""
    if os.environ.get('WSL_DISTRO_NAME') or os.environ.get('WSL_INTEROP'):
        return True
    try:
        with open('/proc/version') as fh:
            return 'microsoft' in fh.read().lower()
    except OSError:
        return False


_IS_WSL = _is_wsl()


# Machine-adaptive worker count: saturate cores (minus headroom) but cap the job
# at `ram_fraction` of RAM so the box never swaps. `mb_per_worker` is the peak
# RSS of one worker's panel; `env_var` lets the operator force a fixed count;
# `hard_cap` bounds the count regardless of how much RAM/cores the host has.
def _auto_workers(env_var, mb_per_worker=1900, ram_fraction=0.70,
                  core_headroom=2, hard_cap=None):
    env = os.environ.get(env_var)
    if env:
        return max(1, int(env))
    # On WSL2 a single swap-thrash episode freezes the Windows host, so leave a
    # much larger RAM and core margin than on a native box.
    if _IS_WSL:
        ram_fraction = min(ram_fraction, 0.50)
        core_headroom = max(core_headroom, 4)
    budget_mb = _total_ram_bytes() / (1024 * 1024) * ram_fraction
    workers_ram = max(1, int(budget_mb // mb_per_worker))
    workers = min((os.cpu_count() or 4) - core_headroom, workers_ram)
    if hard_cap:
        workers = min(workers, hard_cap)
    return max(1, workers)


# Peak RSS per signal worker is well above the steady feature panel: under
# 'spawn' each worker reads back the cached forward-target panel AND its
# column-projected feature batch, then groupby/Polars transforms copy both.
# Budget ~6GB/worker and hard-cap at 12 so the box never swaps. The 3.2GB
# estimate auto-scaled to ~7 workers (~22GB) on a 24-core/49GB host, which was
# heavier than wanted; 6GB picks ~4 here. Smaller batch sizes (see
# signal_batch_max_columns/signals) keep the actual per-worker footprint lower.
# Override with CRYPTO_SIGNAL_WORKERS.
def _auto_signal_workers():
    return _auto_workers('CRYPTO_SIGNAL_WORKERS', mb_per_worker=6000,
                         hard_cap=12)


# Feature generation also loads raw 1m history per symbol; peak RSS spikes well
# above the steady-state panel, so budget conservatively (~2.4GB/worker) and keep
# a hard ceiling for the intrabar memory spikes on full-history symbols.
def _auto_feature_workers():
    return min(8, _auto_workers('CRYPTO_FEATURE_WORKERS', mb_per_worker=2400))


_SIGNAL_WORKERS = _auto_signal_workers()
_FEATURE_WORKERS = _auto_feature_workers()


# =============================================================================
# Main Configuration Dictionary
# =============================================================================

config = {
    # Storage configuration. Each table is a Parquet dataset under this dir:
    # tables with a `symbol` column are stored one file per symbol
    # (db/<table>/<symbol>.parquet); other tables as a single file
    # (db/<table>.parquet). See dbutil.py.
    'database': {
        'data_dir': 'db',
    },

    # Time frequency configuration. The current feature taxonomy is calibrated
    # specifically for 10-minute bars; validate_config rejects other values.
    'base_frequency': '10min',          # Base panel frequency (pandas offset alias)
    'horizons': ['10min', '1h', '1d'],  # Multi-horizon forward residual targets

    # Raw data collection
    'data': {
        'start_date': '2023-01-01',     # Earliest date all ETL jobs should keep
        'history_years': 3,             # Legacy fallback if start_date is unset
        'raw_interval': '1m',           # Binance kline interval for prices_raw
        'quote_currencies': ['USDT', 'USDC'],  # Spot quote preference order
        'max_concurrent_downloads': 50,
        'max_concurrent_symbols': 8,    # Symbols downloaded/held in memory at once
    },

    # Universe construction
    'universe': {
        # Candidates = Hyperliquid perps (tradability constraint), mapped to
        # Binance spot symbols for historical data. No stablecoins/pegged assets.
        'hyperliquid_api': 'https://api.hyperliquid.xyz/info',
        'max_candidates': 130,          # Cap candidate list (by HL 24h notional volume)
        # Hyperliquid name -> Binance base symbol (k-prefix = 1000x contracts)
        'symbol_aliases': {
            'kPEPE': 'PEPE', 'kSHIB': 'SHIB', 'kBONK': 'BONK',
            'kFLOKI': 'FLOKI', 'kLUNC': 'LUNC', 'kNEIRO': 'NEIRO',
            'kDOGS': 'DOGS',
        },
        # Stablecoins / pegged assets excluded from the universe
        'stablecoin_blacklist': [
            'USDT', 'USDC', 'DAI', 'FDUSD', 'TUSD', 'USDP', 'PYUSD', 'USDE',
            'BUSD', 'USDD', 'FRAX', 'GUSD', 'LUSD', 'USD1', 'EUR', 'EURI',
            'AEUR', 'USTC', 'PAXG', 'XAUT', 'WBTC', 'WBETH', 'WSTETH',
        ],
        # Manual exclusions: recycled tickers where the Binance spot history
        # belongs to a DIFFERENT asset than the Hyperliquid perp and the
        # automated price-identity check cannot adjudicate (Binance delisted)
        'symbol_blacklist': [
            'LIT',   # Binance LITUSDT = Litentry (delisted 2025); HL LIT = Lighter
        ],
        # Identity check helper: drop candidates whose fresh Binance
        # close deviates from the same-time HL mark price by more than this
        # ratio. If price data is older than the snapshot by more than
        # identity_max_staleness_days, skip the automatic check rather than
        # mistaking ordinary market drift for a recycled ticker.
        'identity_max_price_ratio': 1.25,
        'identity_max_staleness_days': 2,
    },

    # Logging configuration
    'logging': {
        'format': '%(asctime)s - %(levelname)s - %(message)s',
        'datefmt': '%Y-%m-%d %H:%M:%S',
    },

    # Compute/parallelism settings
    'compute': {
        'default_workers': max(os.cpu_count() - 4, 1),
        # Memory-heavy (~1.4GB steady, higher peak on full-history symbols).
        # RAM-adaptive + hard-capped at 8; override with CRYPTO_FEATURE_WORKERS.
        'feature_workers': _FEATURE_WORKERS,
        'residual_workers': max(os.cpu_count() - 4, 1),
        # Auto-sized from host cores+RAM (see _auto_signal_resources): saturate
        # the CPU but cap total RAM use so the box never swaps. Override with the
        # CRYPTO_SIGNAL_WORKERS env var.
        'signal_workers': _SIGNAL_WORKERS,
        # Worker start method for signal evaluation. MUST be 'spawn' with the
        # Polars-backed loaders: Polars' runtime is not fork-safe, so 'fork'
        # deadlocks workers. 'spawn' makes each worker build its own panel.
        'signal_start_method': 'spawn',
        # Pack signal column-groups per worker load (bounds panel RSS, avoids
        # hundreds of full-table scans).
        'signal_batch_max_columns': 4,    # smaller batches -> lower per-worker RSS
        'signal_batch_max_signals': 100,  # (more batches, slightly slower)
        'blas_threads_per_worker': 1,    # OMP/BLAS threads per spawned worker
        # Polars threads. MUST stay 1: parallelism here is process-per-core, and
        # Polars' multithreaded runtime is not fork-safe (the signal evaluator
        # forks workers after the parent has used Polars -> deadlock). dbutil
        # exports this as POLARS_MAX_THREADS before importing polars.
        'polars_max_threads': 1,
    },

    # Risk model: market + size factor model
    'risk_model': {
        'factors': ['market', 'size', 'momentum', 'vol'],  # Factor names (column prefixes)
        'market_min_members': 10,          # Min members for a valid market factor bar
        'size': {
            # Rank-weighted small-minus-big: weights proportional to centered
            # mcap rank (long smalls / short bigs, each side scaled to 1).
            # Uses every member, no tercile boundary churn; still a tradable
            # portfolio return in return units.
            'mcap_lag_days': 1,            # Use mcap from D-1 for day D weights
            'mcap_max_staleness_days': 7,  # Max ffill of missing mcap
        },
        'momentum': {
            # Rank-weighted winners-minus-losers: daily weights from the trailing
            # cumulative return, long high-momentum / short low-momentum, each
            # side scaled to 1. Tradable portfolio return; same machinery as size.
            'lookback_days': 30,           # Trailing window for the momentum rank
            'skip_days': 2,                # Skip most-recent N days (strictly-past;
                                           # avoids short-term reversal contamination)
        },
        'vol': {
            # Rank-weighted low-minus-high realized vol. Daily weights from the
            # trailing realized vol, long low-vol / short high-vol (sign arbitrary
            # for neutralization). Tradable portfolio return.
            'lookback_days': 30,           # Trailing window for realized vol
        },
        'beta': {
            'window_days': 30,             # Rolling estimation window (calendar days)
            'halflife_days': 10,           # Exponential weighting half-life
            'min_observations': 1008,      # Min bars in window (= 7 days of 10min bars)
        },
        # Acceptance checks printed (and warned) after residual generation
        'acceptance': {
            'min_variance_reduction': 0.05,   # var(res)/var(raw) must be <= 1 - this
            'max_residual_raw_corr': 0.95,    # corr(res, raw) must be below this
            # Factor collinearity: variance-inflation factor of each factor's
            # returns vs the others. Collinear factors make betas unstable and
            # the hedge noisy. Warn-only (does not fail the build).
            'max_vif': 10.0,
        },
    },

    # Feature engineering windows (in BARS at base_frequency = 10min)
    # short ~ hours, medium ~ day, long ~ days
    # Timing convention: features may use data through bar t INCLUSIVE
    # (bar-end stamps; forward targets start at t+1, so no overlap).
    'features': {
        'mean_reversion_windows': [36, 144, 432],   # 6h, 1d, 3d
        'autocorr_window': 144,                     # 1d
        'reversal_window': 36,                      # 6h

        # True Lo-MacKinlay variance ratio: Var(q-bar ret) / (q * Var(1-bar ret))
        'variance_ratio_q': [6, 36],                # 1h, 6h aggregation
        'variance_ratio_window': 432,               # 3d estimation window

        'vol_short_window': 36,                     # 6h
        'vol_medium_window': 144,                   # 1d
        'vol_long_window': 432,                     # 3d

        'liquidity_window': 36,
        'volume_ma_window': 144,

        'momentum_short': 6,                        # 1h
        'momentum_medium': 36,                      # 6h
        'momentum_long': 144,                       # 1d

        'statistical_window': 144,

        'ms_short_window': 18,                      # 3h (microstructure short MA)
        'ms_long_window': 144,                      # 1d (microstructure long MA)

        'sma_periods': [36, 144, 432],
        'rsi_period': 84,                           # 14h
        'bollinger_period': 144,
        'bollinger_std': 2,
        'macd_fast': 72, 'macd_slow': 156, 'macd_signal': 54,
        'atr_period': 84,
        'dmi_period': 84,
        'chandelier_period': 132,
        'chandelier_mult': 3,

        # Range-based vol estimators (Parkinson / Garman-Klass / Rogers-
        # Satchell, Corwin-Schultz spread proxy) from 10min OHLC
        'range_vol_window': 144,                    # 1d
        'cs_spread_window': 144,                    # 1d smoothing of CS estimator

        # Intra-bar features from raw 1-minute data
        'intrabar': {
            'enabled': True,
            'rv_window_bars': 6,                    # 1h realized-vol window (10min bars)
            'autocorr_window_1m': 60,               # 1h of 1m returns
            'zero_vol_window_bars': 144,            # 1d staleness window
            'maxmove_norm_window_bars': 144,        # 1d normalization for wick size
        },

        # Market-context / lead-lag features (vs the market factor)
        'market_context': {
            'corr_window': 144,                     # 1d rolling corr to market
            'lag_response_bars': 3,                 # 30min market-move window
            'market_move_z_window': 144,            # z-norm window for market move
            'lag_corr_window_long': 1008,           # 7d corr with LAGGED market (lead-lag speed)
        },

        # Market-cap-derived features
        'cap': {
            'turnover_window_bars': 144,            # 1d dollar volume / mcap
            'supply_inflation_days': 30,            # mcap growth minus price return
        },

        # Cross-sectional (panel) features computed in the main process
        'cross_section': {
            'rel_volume_window': 144,
            'ret_rank_windows': [6, 144],           # 1h, 1d trailing-return CS rank
            'dispersion_window': 6,                 # 1h mean of CS std
            'breadth_sma_window': 144,
            'funding_z_ffill_limit_bars': 60,       # max 10h carry-forward
            # Cluster-relative value: z of own residual cumsum within its own
            # trailing-correlation cluster (meme basket relative value)
            'cluster_rel': {
                'corr_window_bars': 4320,           # 30d residual corr for clustering
                'recluster_days': 7,                # re-estimate clusters weekly
                'corr_threshold': 0.30,
                'min_cluster_size': 3,
                'z_window': 144,                    # 1d residual cumsum
            },
        },

        'residual_autocorr_lags': [1, 6, 36],
        'residual_vol_windows': [36, 144],
        'residual_zscore_window': 144,
        'residual_ac1_window': 432,                 # regime AC window (3d)
        'residual_mr_ac1_threshold': -0.05,         # mean-reversion regime cutoff
        'residual_cumsum_xlong': 432,               # 3d residual momentum
        'residual_spread_window': 432,              # 3d drawdown/run-up window
        'residual_extreme_z': 2.0,                  # |z| defining an extreme event
        'residual_hurst_q': 36,                     # aggregation for Hurst estimate

        # OU process fitted on CUMULATIVE residual (the spread level)
        'ou_process': {
            'short_window': 144,                    # 1d
            'long_window': 1008,                    # 7d
            'min_observations': 100,
            'lambda_threshold': 0.01,
        },

        # Time encoding
        'funding_hours_utc': [0, 8, 16],            # funding settlement hours
        'funding_window_bars': 3,                   # 30min pre-settlement flag

        # Futures-derived extras
        'futures': {
            'flip_age_cap_bars': 1008,              # cap bars-since-funding-flip (7d)
            'flush_z_window_bars': 144,             # OI-flush z-normalization window
        },
    },

    # Signal generation (lookbacks in BARS at base_frequency)
    # Signal research lab (signal_lab.py): KEEP/KILL thresholds for a candidate.
    'signal_lab': {
        'min_ic_tstat': 2.0,        # HAC IC t-stat a candidate must clear
        'max_book_corr': 0.5,       # |corr of its PnL to the live book| ceiling
        # min_liquid_ic_ratio defaults to walk_forward.min_liquid_ic_ratio
    },

    'signals': {
        'spaces': {'smoothing_halflife': 3},        # light EWM on each space's raw value
        'smoothing_halflife': 3,
        # Per-lag smoothing: a signal scored at forward lag L is smoothed at
        # the halflife of the smallest bucket with max_lag >= L (list of
        # [max_lag, halflife] in bars; lags beyond the last bound use the last
        # halflife; the per-space/global base halflife is a floor). Slow lags
        # tolerate slow signals: measured on this panel, halflife 36 vs 3 cuts
        # signal turnover ~2.5-3x while keeping 75-85% of the IC - so matching
        # smoothing to the holding lag roughly doubles gross-per-turnover
        # exactly where turnover matters. [] disables (single global halflife).
        'lag_smoothing': [[12, 3], [48, 12], [144, 36], [432, 108]],
        'warmup_days': 10,                          # feature warmup before a test window
        'screening_grid': '1h',                     # IC sampled here (computed at full res)
        'min_assets_per_timestamp': 10,
        'min_universe_fraction': 0.4,               # also >= this * universe.max_candidates names
        'compute_on_full_history': True,
        # Forward IC lags (bars). Deliberately FEW: every extra lag multiplies
        # the Bonferroni correction applied to every signal's best-lag p-value
        # and adds evaluation compute. Chosen from the measured decay of this
        # library (gross edge strong at <=24 bars, marginal at 48, ~zero beyond
        # - except funding, which lives at slow lags):
        #   3 = 30min, 6 = 1h  -> the fast reversal/order-flow/lead-lag core
        #   24 = 4h            -> the mid-speed body
        #   144 = 1d           -> the funding/carry sleeve
        # The portfolio pins each signal to its strongest lag of these four.
        'decay_lag_grid': [3, 6, 24, 144],
        'liquidity_window_bars': 144,
    },

    # Agentic signal discovery (research/signals/agent/): bounded-DSL search
    # over residual-predictive cross-sectional signals with a train/select/OOS
    # walk-forward. Design in research/signals/agent/agent.md. Everything here
    # is read via config.get('discovery.<...>') - never hardcoded.
    'discovery': {
        'start_date': '2023-08-01',      # First roll's train start
        'end_date': '2026-06-01',        # No roll's OOS end may exceed this
        'train_months': 5,               # Candidates generated/fit here
        'select_months': 1,              # Search reward scored here (only)
        'oos_months': 1,                 # Promoted book traded here (never searched)
        'roll_step_months': 1,
        # Purge at window boundaries: drop the last (max target lag + embargo)
        # bars of TRAIN before SELECT and of SELECT before OOS so no forward
        # target leaks across a boundary.
        'embargo_bars': 12,
        # Forward residual-return target lags (bars at base_frequency).
        # target_lag_bars is the primary lag the reward is computed on.
        'horizon_lags_bars': [6, 36, 144],
        'target_lag_bars': 36,
        'min_assets_per_timestamp': 10,
        'liquidity_window_bars': 144,    # trailing $vol window for the liquid-half flag
        # Input space: feature columns are resolved by matching these
        # per-family prefix patterns against the features table (bounded input
        # space - candidates can only reference resolved columns).
        'families': {
            'residual_shape':    ['res_', 'ou_'],
            'volatility_regime': ['vr_', 'rb_', 'ib_'],
            'liquidity':         ['lq_', 'vl_', 'ms_', 'cap_turnover'],
            'derivatives':       ['fr_', 'oi_', 'pos_'],
            'cross_sectional':   ['cs_'],
            'factor_context':    ['fl_', 'mk_'],
        },
        'max_features_per_family': 12,   # cap resolved columns per family
        # DSL bounds (hypothesis space)
        'dsl': {
            'windows': [6, 36, 144, 432],   # allowed rolling windows (bars)
            'max_depth': 4,                  # expression tree depth cap
            'max_conditions': 2,             # gates per candidate
            'max_nodes': 24,                 # total expression+condition nodes
        },
        # Evolutionary search (per roll). Budget = n_generations * batch_size.
        'search': {
            'seed': 7,
            'n_generations': 8,
            'batch_size': 16,
            'survivors': 12,                 # population carried between generations
            'mutation_prob': 0.6,            # mutate a parent vs sample fresh
            'diversity_max_corr': 0.8,       # survivor de-correlation ceiling
            'bandit_ucb_c': 1.0,             # family-bandit exploration constant
        },
        # Reward = sum_k weight_k * term_k / scale_k, SELECT window only.
        # Scales are FIXED constants (not batch-relative) so rewards are
        # comparable across generations, rolls and resumed runs.
        'reward': {
            'weights': {
                'ic_tstat': 1.0,
                'net_sharpe': 0.5,
                'liquid_ic_ratio': 0.25,
                'turnover': -0.25,
                'complexity': -0.15,
                'instability': -0.75,
                'similarity': -0.5,
            },
            'scales': {
                'ic_tstat': 2.0,
                'net_sharpe': 1.0,
                'liquid_ic_ratio': 1.0,
                'turnover': 1.0,
                'complexity': 10.0,
                'instability': 0.05,
                'similarity': 0.5,
            },
        },
        # Promotion gates, applied once per roll to the search survivors.
        'promotion': {
            'fdr_alpha': 0.10,
            'fdr_method': 'by',              # 'by' or 'bh'
            'min_select_ic_tstat': 2.0,
            # Search-overfit haircut: |t| must also clear deflation_mult x
            # E[max |N(0,1)| over n_trials] where n_trials comes from the
            # ledger (all candidates evaluated this roll). 0 disables.
            'deflation_mult': 1.0,
            'max_book_corr': 0.5,            # signal corr vs already-promoted book
            # A candidate must be a search survivor in this many CONSECUTIVE
            # rolls (by candidate hash) before it may be promoted.
            'min_rolls_survived': 2,
            'max_promoted_per_roll': 3,
            'max_book_size': 15,
        },
        # OOS portfolio built on promoted signals: dollar + factor neutral,
        # solved independently per rebalance stamp (non-sequential, ex-post
        # costs from |dw| - slightly overstates turnover vs a turnover-aware
        # optimizer; the price of keeping the backtest fully vectorizable).
        'backtest': {
            'rebalance_grid': '1h',
            'weight_scheme': 'equal_weight',  # 'equal_weight' (rank) or 'mvo'
            'cost_bps': None,                 # None -> portfolio.cost_bps
            'funding_pnl': True,              # accrue funding if table available
            'min_assets': 20,
            # mvo scheme only: trailing residual covariance
            'cov_window_days': 30,
            'cov_min_observations': 1008,
        },
        # ML ceiling probe (mode=ml): gradient boosting on ALL resolved
        # primitives, fit on TRAIN, scored on SELECT. Its IC estimates how much
        # predictability the feature set contains at all.
        'ml_probe': {
            'max_iter': 200,
            'max_depth': 3,
            'learning_rate': 0.05,
            'min_samples_leaf': 200,
            'l2_regularization': 1.0,
            'subsample_rows': 200000,        # cap training rows (recent-biased)
        },
        # LLM proposer (untrusted: sees compressed diagnostics only, emits DSL
        # JSON; everything it returns is re-validated and re-scored by code).
        # provider: 'anthropic' or 'gemini' (google-genai package). The API
        # key is read from the gitignored repo-root .env under the GENERIC
        # name below (key_name) - switching LLMs = change provider/model here
        # and swap the key value in .env; no code or variable renames.
        'llm': {
            'provider': 'gemini',
            'key_name': 'LLM_KEY',       # .env variable holding the API key
            'model': {
                'anthropic': 'claude-sonnet-4-6',
                'gemini': 'gemini-2.5-flash',
            },
            'max_tokens': 4096,
            'candidates_per_call': 8,
            # Gemini 2.5 thinking budget (tokens). 0 disables thinking so the
            # whole max_tokens budget goes to the JSON output (thoughts are
            # billed as output and can truncate the response mid-array).
            # None = provider default.
            'gemini_thinking_budget': 0,
            # $ per million tokens, per provider - used ONLY for the cost
            # estimate printed/persisted by run_discovery. Prices change, so
            # pin your provider's current rates here; None disables the
            # dollar estimate (token counts are always tracked).
            'price_per_mtok': {
                'anthropic': {'input': None, 'output': None},
                # gemini-2.5-flash paid tier (ai.google.dev/gemini-api/docs/
                # pricing, checked 2026-07): output price includes thinking
                # tokens. Update here if the model or Google's rates change.
                'gemini': {'input': 0.30, 'output': 2.50},
            },
        },
        'diagnostics': {
            'n_bins': 10,                    # binned forward-return deciles
            'regime_columns': ['res_vol_short', 'cs_rel_volume'],
            'top_per_family': 6,             # compressed view size
        },
        # Persisted tables
        'tables': {
            'ledger': 'discovery_ledger',
            'promotions': 'discovery_promotions',
            'oos_returns': 'discovery_oos_returns',
            'llm_usage': 'discovery_llm_usage',
        },
    },

    # Walk-forward configuration (no look-ahead)
    'walk_forward': {
        'start_date': '2023-08-01',      # First usable panel date (after warmup)
        'end_date': '2026-06-01',        # Last complete month of data
        'train_months': 6,
        'test_days': 30,
        # Training-window mode. 'expanding' (default): every window trains on
        # ALL data from start_date up to its train_end - this is the monthly
        # production retrain ("use everything I know so far") and it fixes the
        # power problem of short windows: by the last window the selector sees
        # ~2.5 years (~10x the daily-IC observations of a 6-month slice), so
        # honest signals clear the multiplicity-corrected t-stat bar instead of
        # dying to it. train_months then only sets the FIRST window's length.
        # 'rolling': legacy fixed 6-month lookback stepping forward monthly.
        'train_window': 'expanding',

        # Survivorship sensitivity gate: exclude names whose FIRST data bar is
        # within this many days of the test day. The universe is conditioned on
        # today's HL listing (historical listing dates are unavailable), and
        # newly listed names are both survivor-biased and pump-phase-prone;
        # re-running the backtest with e.g. 90 here bounds how much of the PnL
        # depends on them. 0 = off (default; production behaviour).
        'min_listing_age_days': 0,

        # Per-window signal selection (training data only).
        # IC t-stats use Newey-West HAC on the daily-IC series (cross-sectional
        # ICs are serially correlated across days; iid t-stats are optimistic);
        # the t-stat picks each signal's best lag and feeds candidate ranking.
        # 'auto' -> Bartlett lags = floor(4*(n_days/100)^(2/9)).
        'ic_hac_lags': 'auto',
        # FDR pre-filter on the per-signal IC p-values. Loose by design (only
        # sweeps out the clearly-spurious tail; the gates do the real filtering).
        # Raise alpha toward 1.0 to disable, lower toward 0.05 to tighten.
        # fdr_method: 'by' = Benjamini-Yekutieli (controls FDR under arbitrary
        # dependence; correct for the library's dense correlated-variant
        # families), 'bh' = Benjamini-Hochberg (looser, assumes independence).
        'fdr_alpha': 0.20,
        'fdr_method': 'by',
        # IC floor. A 22-window OOS sweep showed 0.01 maximizes the selected
        # set's IR proxy (mean OOS IC x sqrt(breadth)): it lifts OOS IC
        # 0.013 -> 0.014 (86% sign-correct) while keeping ~43 signals. Higher
        # floors raise per-signal IC but breadth collapses (0.03 -> ~7 signals).
        'min_ic': 0.01,
        'min_icir': 0.02,
        # Annualized Sharpe of the signal's own GROSS daily returns - the
        # standalone track-record floor a signal must clear to be selected.
        'min_sharpe_threshold': 0.3,
        # Cost-aware economic floor: annualized Sharpe of the signal's own daily
        # returns AFTER paying portfolio.cost_bps/side on its rebalance turnover,
        # traded in its selected direction. Gross IC/Sharpe ignore costs, so a
        # high-gross-IC short-horizon signal whose few-bp edge cannot clear a
        # round trip was being selected and then losing money in the book (the
        # 22-window run showed standalone net edge negative across every bucket
        # while gross looked like a +2 Sharpe). 0.0 = require net break-even;
        # None = disable (revert to gross-only selection).
        'min_net_sharpe_threshold': 0.0,
        'max_correlation_threshold': 0.50,
        'max_signal_turnover': 1.0,      # Max avg turnover per rebalance cycle
        # Minimum holding lag (bars) a signal may be selected at - the speed
        # match between signal decay and how fast the book actually trades.
        # 'auto' (default) DERIVES the floor from the execution layer: the
        # backtest discounts each bucket's alpha by h/(h + 1/kappa) at the
        # effective fill rate kappa (GP trade rate capped by the
        # max_annual_turnover budget), so 'auto' admits only lags that retain
        # at least min_monetizable_alpha_fraction of their alpha after that
        # discount (h >= f/(1-f) * 1/kappa). One knob set - trade_urgency plus
        # the turnover budget - moves selection AND execution together, so the
        # selector can no longer spend slots on fast signals the executor then
        # scales to ~1%. An integer keeps a manual floor in bars (0 = off).
        'min_holding_lag_bars': 'auto',
        # Fraction of a signal's alpha that must survive the aim discount for
        # its lag to be selectable under 'auto'. The floor moves with the
        # execution config: with zero-cost maker execution and no turnover
        # budget the fill rate kappa ~ 1/bar and the floor resolves to 1 bar
        # (everything selectable); with taker fees + a tight budget it climbs
        # into the multi-hour range automatically.
        'min_monetizable_alpha_fraction': 0.15,
        'max_signals_per_window': 15,    # TOTAL selected per window (all lags)
        # Do not keep trading a stale selection when the current window finds
        # no statistically defensible candidates.
        'fallback_to_previous': False,

        # Direct horizon selection. Forward cumulative-return IC is not an
        # exponential decay curve, so each signal is pinned to the lag with the
        # strongest HAC t-stat. Execution buckets are the DISTINCT SELECTED
        # LAGS themselves (each refreshes at its own cadence) - not terciles.
        'horizon_selection': {
            'min_valid_lags': 1,
            # Allow several decorrelated variants per family. At 1 this capped
            # the whole book at one signal per family, collapsing selection to a
            # handful regardless of how many survived the gates; greedy
            # de-correlation (max_correlation_threshold) still prevents redundant
            # near-duplicates from being kept.
            'max_variants_per_family': 4,
        },

        # Robustness gates (training window only)
        'min_stable_thirds': 2,
        'require_recent_third': True,    # latest third must retain pooled IC sign
        'min_liquid_ic_ratio': 0.3,      # |IC on liquid half| >= ratio * |IC|

        # Execution-fragility stress: also report PnL with weights applied
        # one bar late (decided at t-1, earn bar t)
        'implementation_lag_bars': 1,

        # Covariance-aware signal combination (Grinold): composite weights
        # w ~ C^{-1} . IC instead of flat IC-weighting, to exploit signal
        # diversification. corr_shrink pulls the signal-return correlation toward
        # the identity (1.0 = falls back to IC-weighting; 0.0 = full, overfits).
        # This is the ONLY signal-weighting system (the legacy turnover-
        # penalized 'signal_weighting' duplicate was removed); when it cannot
        # run (<2 signals with return history, or disabled) the fallback is
        # plain |IC|-proportional weights - no extra knobs.
        'signal_combination': {
            'enabled': True,
            'corr_shrink': 0.5,
        },

        # Standardized composite ranking of candidate signals
        'candidate_ranking': {
            'enabled': True,
            'score_weights': {
                # Rank on the COST-AWARE (net) Sharpe, not gross: prefer signals
                # whose edge survives their own trading cost under the family and
                # de-correlation caps.
                'sharpe_net': 0.30,
                'icir': 0.25,
                'ic_tstat': 0.20,
                'inverse_turnover': 0.15,
                'inverse_decay': 0.10,
            },
        },
    },

    # Portfolio construction: shrunk-covariance MVO, market-neutral
    'portfolio': {
        # Production position sizing. 'equal_weight' = covariance-free rank book
        # (dollar/factor-neutral, per-name capped, gross-1); 'mvo' = Ledoit-Wolf
        # shrunk-covariance MVO. The walk-forward EW-vs-MVO comparison found the
        # LW covariance weighting net-destructive on this low-breadth,
        # negatively-skewed (residual_reversion-heavy) book - rank sizing matched
        # the alpha's ordering without concentrating risk on its bad tails - so
        # rank sizing is the default. See research/portfolio/walk_forward.py.
        'weight_scheme': 'equal_weight',
        # Foil sizing scheme run alongside the production book each window for
        # monitoring (persisted to wf_portfolio_returns_bench). '' disables it.
        # Off by default: the EW-vs-MVO question is ANSWERED on this book (MVO
        # net Sharpe -1.23 vs +0.11 on identical alpha, 22-window run) - the
        # foil pass only doubled backtest runtime to re-answer it. Set to 'mvo'
        # to re-run the comparison after material risk-model changes.
        'benchmark_scheme': '',
        'cov_window_days': 30,               # Trailing window for residual covariance
        'cov_min_observations': 1008,        # Min bars (7d) for a valid covariance
        'shrinkage': 'ledoit_wolf',          # 'ledoit_wolf' or float in [0,1] (mvo only)
        'gross_leverage': 1.0,               # Sum |w| target
        'max_position': 0.05,                # Per-name cap (fraction of gross)
        'neutrality': ['dollar', 'market', 'size', 'momentum', 'vol'],  # Constrained exposures B'w
        # Neutrality BANDS: each exposure is held within +/- band rather than at
        # exactly zero. Bands give the optimizer slack to retain alpha and cut
        # turnover instead of fighting the position cap to hit exact zero. Units:
        # 'dollar' = net long-short as a fraction of gross; factor entries =
        # portfolio beta to that factor. A band of 0.0 reproduces exact neutrality.
        'neutrality_band': {
            'dollar':   0.05,   # net exposure <= 5% of gross
            'market':   0.10,   # |portfolio beta| <= 0.10
            'size':     0.10,
            'momentum': 0.10,
            'vol':      0.10,
        },
        'weight_smoothing_halflife': 6,      # legacy fixed EWM rate (fallback only)
        # HARD turnover budget (x gross per year); None -> uncapped. With
        # cost_bps = 0 (top-tier maker execution) there is no fee rationale for
        # a budget, and the measured alpha decays within hours - throttling the
        # fill to days destroyed transmission (the 100x/yr run deployed only
        # ~0.58 gross and had cost/alpha > 1 in most windows). Set a number to
        # model capacity- or operationally-constrained execution.
        'max_annual_turnover': None,
        # Garleanu-Pedersen multi-period trading toward the gross-1 aim. Two pieces:
        # 1) TRADE RATE (per bar): the myopic optimal gamma/(gamma+lambda) balance
        #    of off-aim penalty vs quadratic trade cost, made COST-RESPONSIVE:
        #        omega = trade_urgency * (ref_cost_bps / cost_bps);  rate = omega/(1+omega)
        #    Set LOW here to trade SLOWLY toward the target (~0.001 ~ 5-day
        #    halflife, ~15x/yr turnover); max_annual_turnover is the hard backstop.
        #    null -> fall back to the fixed halflife. PnL costs unchanged (linear).
        # 2) AIM DISCOUNT: bucket alpha scaled by h/(h + 1/rate) - alpha that decays
        #    faster than the (slow) trade rate can't be monetized, so it is
        #    downweighted; slow / persistent alpha is overweighted.
        'gp_trading': {
            'enabled': True,
            'trade_urgency': 0.8,          # gamma/lambda; LOW = slow trading, low turnover
            'ref_cost_bps': 5.0,             # cost at which trade_urgency is calibrated
            # Discount the aim at the rate the book ACTUALLY fills (capped by the
            # turnover budget), not the nominal trade rate. Keeps the aim from
            # over-sizing fast alpha the budget-throttled book can't capture.
            'discount_at_realized_rate': True,
        },
        'min_assets': 30,                    # Leaves room for caps + neutrality constraints
        # Soft cluster-exposure penalty: clusters from trailing residual
        # correlations (same window as the covariance, causal). Motivated by
        # the Marchenko-Pastur diagnostic: stable super-MP structure exists
        # (e.g. a meme-coin factor) beyond market+size. Sigma_eff =
        # Sigma + lambda * sum_k (sigma_k * 1_k)(sigma_k * 1_k)'.
        'cluster_penalty': {
            'enabled': True,
            'lambda': 1.0,                   # Penalty strength (1 = like doubling cluster var)
            'corr_threshold': 0.30,          # Merge names with residual corr above this
            'min_cluster_size': 3,           # Smaller groups are not penalized
        },
        # Costs: per-side bps applied to turnover. Set to 0.0 = pure-maker
        # execution at Hyperliquid's top volume tiers (verified 2026-07 against
        # the official fee docs: perp maker fee is 0.000% at tier 4+, >$500M
        # 14-day volume, with separate maker REBATES of -0.1..-0.3bp by maker
        # volume share; below tier 4 maker is 0.4-1.5bp and taker never drops
        # below 2.4bp). This models fees only - passive fills still carry
        # adverse-selection/miss risk that no bps number captures; the
        # implementation-lag stress is the closest proxy. Raise this to test
        # lower-tier maker (0.4-1.5) or taker (2.4-4.5) execution.
        'cost_bps': 0.0,
        # Accrue perp funding on held positions in the walk-forward backtest: at
        # each settlement stamp the book earns -sum(w_i * rate_i) (longs PAY a
        # positive rate, shorts receive). Rates come from the funding_rates table
        # (Binance USDT-perp, a proxy for Hyperliquid funding). Several signal
        # families tilt the book BY funding (short crowded-carry names), so this
        # term is correlated with the alpha and must not be ignored.
        # False -> price PnL minus trading costs only (legacy behaviour).
        'funding_pnl': True,
        'residual_vol_window_days': 10,      # Per-asset residual vol for Grinold alpha scaling
        # Grinold IC shrinkage: realized IC is noisy/non-stationary, so the alpha
        # scale uses (1 - ic_shrink) * IC. 0 = trust IC fully; 0.5 = halve it
        # (Grinold & Kahn). Applied per bucket.
        'ic_shrink': 0.1,
        # Edge-scaled gross: the aim book's gross is multiplied by
        # clip(expected horizon alpha per unit gross /
        #      (edge_mult x round-trip cost per unit gross), 0, 1),
        # so the book only deploys full size when the aim's expected edge
        # covers edge_mult round trips - it shrinks (to zero if need be)
        # instead of trading a gross-1 book on alpha that cannot pay for
        # itself (the old run had cost/alpha > 1 in 15/22 windows). With
        # cost_bps = 0 the multiplier is identically 1 (no cost to clear).
        'edge_scaled_gross': {
            'enabled': True,
            'edge_mult': 2.0,
        },
        # No-trade zone ("lazy trading"):
        # a name whose expected residual return OVER ITS HOLDING HORIZON
        # (per-bar Grinold alpha * horizon bars) is below no_trade_band_mult *
        # (per-name per-side cost) cannot pay for a round trip, so its alpha is
        # zeroed (the optimiser won't allocate fresh risk to it). Compared on the
        # holding horizon, not per-bar, so units match the round-trip cost.
        'no_trade_band_mult': 1.0,
        # Liquidity-aware costs and trade speed ("Trading Speed", "Multi-Period").
        # Per-name trailing dollar volume (ADV) makes illiquid names cost more to
        # trade and fill toward the aim more slowly; liquid names cheaper/faster.
        # Multipliers are cross-sectional vs the median ADV (scale-free).
        'liquidity_aware': {
            'enabled': True,
            'adv_window_days': 7,            # trailing window for per-name ADV ($ volume)
            'impact_coef': 0.5,              # cost_mult = 1 + impact_coef*(adv_ref/adv_i - 1)
            'min_cost_mult': 0.5,            # floor per-name cost multiplier (liquid names)
            'max_cost_mult': 3.0,            # cap per-name cost multiplier (illiquid names)
            'speed_exponent': 0.5,           # speed_mult = (adv_i/adv_ref)^exponent
            'min_speed_mult': 0.3,           # clip per-name speed multiplier (illiquid: slower)
            'max_speed_mult': 2.0,           # clip per-name speed multiplier (liquid: faster)
        },
    },

}


# =============================================================================
# Helper Functions
# =============================================================================

def load_env_key(name, path=None):
    """Read NAME=value from the gitignored .env file at the repo root.

    Secrets (API keys) live there, never in this file or in git. Lines are
    KEY=value; blank lines and '#' comments are ignored; surrounding quotes
    and whitespace are stripped. Returns the value or None.
    """
    env_path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    '.env')
    try:
        with open(env_path) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                k, v = line.split('=', 1)
                if k.strip() == name:
                    return v.strip().strip('"\'') or None
    except OSError:
        return None
    return None


def get(key, default=None):
    """Get a config value using dot notation, e.g. get('risk_model.beta.window_days')."""
    keys = key.split('.')
    value = config
    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            return default
    return value


def get_data_start_date() -> datetime:
    """Configured ETL start date, falling back to the legacy rolling window."""
    start_date = get('data.start_date')
    if start_date:
        if isinstance(start_date, datetime):
            return start_date.replace(hour=0, minute=0, second=0, microsecond=0)
        return datetime.strptime(str(start_date), '%Y-%m-%d')

    from dateutil.relativedelta import relativedelta
    history_years = get('data.history_years', 3)
    start = datetime.now() - relativedelta(years=history_years)
    return start.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def get_data_end_date(granularity: str = 'daily') -> datetime:
    """Latest complete ETL date for the requested source granularity."""
    now = datetime.now()
    if granularity == 'monthly':
        first_of_current_month = now.replace(day=1, hour=23, minute=59, second=59,
                                             microsecond=0)
        return first_of_current_month - timedelta(days=1)
    if granularity == 'daily':
        return (now - timedelta(days=1)).replace(hour=23, minute=59, second=59,
                                                microsecond=0)
    return now.replace(microsecond=0)


def get_frequency_config(freq: str):
    """
    Frequency metadata for a pandas offset alias ('10min', '1h', '1d', ...).

    Returns dict with:
    - bars_per_day: number of bars in 24h
    - resample_rule: pandas resample rule string
    - nanos: bar length in nanoseconds
    """
    from pandas.tseries.frequencies import to_offset

    # Exchange APIs use "1m" for one minute, while recent pandas versions
    # interpret "m" as month-end. Normalize numeric lowercase-minute aliases
    # only for local frequency arithmetic; keep the configured API string.
    pandas_freq = (
        f"{freq[:-1]}min"
        if freq.endswith('m') and freq[:-1].isdigit()
        else freq
    )
    offset = to_offset(pandas_freq)
    nanos = offset.nanos  # raises for non-fixed frequencies like 'M' - intended
    day_nanos = 24 * 3600 * 10 ** 9
    if day_nanos % nanos != 0:
        raise ValueError(f"Frequency {freq} does not evenly divide a day")
    return {
        'bars_per_day': day_nanos // nanos,
        'resample_rule': pandas_freq,
        'nanos': nanos,
    }


def horizon_bars(horizon: str, base: str = None) -> int:
    """Number of base-frequency bars in a horizon (e.g. '1d' at '10min' -> 144)."""
    base = base or config['base_frequency']
    h = get_frequency_config(horizon)['nanos']
    b = get_frequency_config(base)['nanos']
    if h % b != 0:
        raise ValueError(f"Horizon {horizon} is not a multiple of base {base}")
    return h // b


def horizon_col(horizon: str, kind: str = 'res') -> str:
    """Canonical column name for a forward-return target, e.g. fwd_res_1h."""
    return f"fwd_{kind}_{horizon}"


def validate_config() -> None:
    """Fail fast when coupled settings are internally inconsistent."""
    if config['base_frequency'] != '10min':
        raise ValueError(
            "This pipeline is calibrated for base_frequency='10min'; "
            "feature windows and bar-based research grids must be rescaled "
            "before changing it."
        )

    raw = get_frequency_config(config['data']['raw_interval'])
    base = get_frequency_config(config['base_frequency'])
    if raw['nanos'] > base['nanos'] or base['nanos'] % raw['nanos'] != 0:
        raise ValueError("data.raw_interval must evenly divide base_frequency")

    workers = config['compute']
    worker_keys = (
        'default_workers',
        'feature_workers',
        'residual_workers',
        'signal_workers',
        'blas_threads_per_worker',
    )
    if any(not isinstance(workers[key], int) or workers[key] < 1 for key in worker_keys):
        raise ValueError("all compute worker/thread settings must be positive integers")

    ranking_weights = config['walk_forward']['candidate_ranking']['score_weights']
    if not ranking_weights or any(weight < 0 for weight in ranking_weights.values()):
        raise ValueError("ranking score weights must be non-negative and non-empty")

    wf = config['walk_forward']
    if wf.get('train_window', 'expanding') not in ('expanding', 'rolling'):
        raise ValueError("walk_forward.train_window must be 'expanding' or 'rolling'")
    mhl = wf.get('min_holding_lag_bars', 0)
    if mhl == 'auto':
        frac = float(wf.get('min_monetizable_alpha_fraction', 0.0))
        if not 0.0 < frac < 1.0:
            raise ValueError(
                "walk_forward.min_monetizable_alpha_fraction must be in (0, 1) "
                "when min_holding_lag_bars is 'auto'")
    elif not isinstance(mhl, int) or mhl < 0:
        raise ValueError(
            "walk_forward.min_holding_lag_bars must be 'auto' or a "
            "non-negative integer")

    port = config['portfolio']
    if port.get('weight_scheme', 'equal_weight') not in ('equal_weight', 'mvo'):
        raise ValueError("portfolio.weight_scheme must be 'equal_weight' or 'mvo'")
    if port.get('benchmark_scheme', '') not in ('', 'equal_weight', 'mvo'):
        raise ValueError("portfolio.benchmark_scheme must be '', 'equal_weight' or 'mvo'")
    if port['gross_leverage'] <= 0 or not 0 < port['max_position'] <= 1:
        raise ValueError("portfolio leverage and position cap must be positive")
    min_positions = math.ceil(port['gross_leverage'] / port['max_position'])
    if port['min_assets'] <= min_positions:
        raise ValueError(
            "portfolio.min_assets must exceed "
            "ceil(gross_leverage / max_position) to leave room for neutrality"
        )
    esg = port.get('edge_scaled_gross', {})
    if esg.get('enabled') and float(esg.get('edge_mult', 0)) <= 0:
        raise ValueError("portfolio.edge_scaled_gross.edge_mult must be positive")
    lag_smoothing = config['signals'].get('lag_smoothing') or []
    if lag_smoothing:
        bounds = [b for b, _ in lag_smoothing]
        if bounds != sorted(bounds) or any(hl < 0 for _, hl in lag_smoothing):
            raise ValueError("signals.lag_smoothing needs ascending max_lag "
                             "bounds and non-negative halflives")


# =============================================================================
# Convenience Exports
# =============================================================================

BASE_FREQUENCY = config['base_frequency']
HORIZONS = config['horizons']
validate_config()
BARS_PER_DAY = get_frequency_config(BASE_FREQUENCY)['bars_per_day']
