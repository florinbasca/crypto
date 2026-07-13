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
- Signals: agentic discovery only (research/signals/); promoted DSL
  candidates are scored in-memory by the walk-forward.
- Portfolio: Ledoit-Wolf MVO, dollar + factor-beta neutral, net of a 5bps
  all-in cost model and perp funding accrual.
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
# Feature generation also loads raw 1m history per symbol; peak RSS spikes well
# above the steady-state panel, so budget conservatively (~2.4GB/worker) and keep
# a hard ceiling for the intrabar memory spikes on full-history symbols.
def _auto_feature_workers():
    return min(8, _auto_workers('CRYPTO_FEATURE_WORKERS', mb_per_worker=2400))


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
        'blas_threads_per_worker': 1,    # OMP/BLAS threads per spawned worker
        # Polars threads. MUST stay 1: parallelism here is process-per-core, and
        # Polars' multithreaded runtime is not fork-safe (the signal evaluator
        # forks workers after the parent has used Polars -> deadlock). dbutil
        # exports this as POLARS_MAX_THREADS before importing polars.
        'polars_max_threads': 1,
    },

    # Risk model: market + size factor model
    'risk_model': {
        'factors': ['market', 'size', 'momentum', 'vol', 'meme'],  # Factor names (column prefixes)
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
        'meme': {
            # Rank-weighted meme-minus-nonmeme. Meme-ness = trailing corr of a
            # name's MARKET-ADJUSTED daily returns with an anchor meme index
            # (fixed tiny seed of pre-sample canonical memes: point-in-time
            # safe, self-updating - new memes acquire high anchor corr within
            # weeks of listing, no list maintenance). Strictly-past, like all
            # characteristics. Sign arbitrary: the factor is HEDGED, not traded.
            'anchor_symbols': ['DOGE', 'SHIB', 'PEPE'],
            'corr_window_days': 60,        # trailing corr window
            'min_corr_days': 30,           # min overlap before non-NaN
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

        # Order-flow (of_) windows: signed aggressive-flow impact/toxicity.
        'order_flow': {
            'fast_window': 6,                       # 1h
            'short_window': 18,                     # 3h
            'long_window': 144,                     # 1d
        },

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

        # Per-symbol seasonality (sn_): trailing same-bucket RESIDUAL stats.
        # Unlike the tm_ sin/cos encodings (cross-sectionally constant), these
        # differ per name - each symbol's own time-of-day / day-of-week
        # residual profile - so they work as direct cross-sectional signals.
        # Slow by construction (profiles move over days/weeks), which is the
        # kind of alpha that survives realistic costs.
        'seasonality': {
            'tod_days': 20,                         # trailing days per hour-of-day bucket
            'min_days': 5,                          # min same-bucket days before non-NaN
            'dow_weeks': 8,                         # trailing same-weekday days (1/week)
            'dow_min_weeks': 2,                     # min same-weekday days before non-NaN
        },

        # Leader lead-lag (ll_): slow catch-up gaps vs a single leader asset
        # (BTC), complementing mk_* which uses the EW market factor at a 30min
        # window. ll_leader_gap_{w}b = rolling-beta-scaled leader move over w
        # bars minus own move - the share of the leader's multi-hour move this
        # name has NOT yet matched. The leader symbol itself gets NaN.
        'lead_lag': {
            'leader_symbol': 'BTC',
            'gap_windows_bars': [36, 144],          # 6h and 1d catch-up gaps
            'beta_window_bars': 1008,               # 7d rolling beta to the leader
            'lag_corr_window_bars': 1008,           # 7d corr with the LAGGED leader move
            'lag_bars': 6,                          # leader lead measured at 1h
        },

        # Macro/event features (ev_/mx_/mb_) from etl/macro.py tables.
        # ev_/mx_ are cross-sectionally constant (DSL gate material); mb_ are
        # per-name macro sensitivities (direct signal material).
        'macro': {
            'hours_clip': 168.0,             # cap on hours to/since event (1 week)
            'event_window_bars': 6,          # +-window flag around exact event times
            'ffill_limit_days': 10,          # max carry-forward of daily macro values
            'vix_z_window_days': 365,        # z window for the VIX level
            'beta_window_days': 90,          # rolling window for mb_beta_* (days)
            'beta_min_days': 45,
            'event_lookback_events': 12,     # trailing events in mb_event_* profiles
            'event_min_events': 4,
            'event_response_hours': 24,      # post-event drift horizon (mb_event_drift)
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
    'signals': {
        'spaces': {'smoothing_halflife': 3},        # light EWM on each space's raw value
        'smoothing_halflife': 3,
        # Bridge promoted discovery candidates into the signal registry as
        # disc_* entries (research/lib/discovered.py); each is selectable
        # only from its promotion date (valid_from). False = spaces only.
        'include_discovered': True,
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

    # Agentic signal discovery (research/signals/): bounded-DSL search
    # over residual-predictive cross-sectional signals with a train/select/OOS
    # walk-forward. Design in research/signals/signal.md. Everything here
    # is read via config.get('discovery.<...>') - never hardcoded.
    'discovery': {
        'start_date': '2023-08-01',      # First roll's train start
        'end_date': '2026-06-01',        # No roll's OOS end may exceed this
        'train_months': 5,               # Candidates generated/fit here
        'select_months': 1,              # Held out from the search; promotion tests it ONCE
        'oos_months': 1,                 # Promoted book traded here (never searched)
        'roll_step_months': 1,
        # Purge at window boundaries: drop the last (max target lag + embargo)
        # bars of TRAIN before SELECT and of SELECT before OOS so no forward
        # target leaks across a boundary.
        'embargo_bars': 12,
        # Forward target lags (bars): the grid the panel builds targets for.
        # NO PINNING: every candidate keeps its full IC profile across this
        # grid (train + select, per lag) - the profile IS the signal's alpha
        # term structure, and its fitted half-life drives the persistence
        # discount at the portfolio layer.
        'horizon_lags_bars': [6, 36, 72, 144],
        # Reference lag for the proposer's compressed diagnostics only.
        'target_lag_bars': 36,
        'min_assets_per_timestamp': 10,
        'liquidity_window_bars': 144,    # trailing $vol window for the liquid-half flag
        # Input space: feature columns are resolved by matching these
        # per-family prefix patterns against the features table (bounded input
        # space - candidates can only reference resolved columns).
        'families': {
            'residual_shape':    ['res_', 'ou_'],
            'volatility_regime': ['vr_', 'rb_', 'ib_'],
            # Return efficiency / diffusion: clean information diffusion vs
            # noisy overshoot - mechanism-rich, strong as standalone AND gate.
            'efficiency':        ['ef_'],
            # Distribution shape: skew/kurtosis - regime/gate primitives for
            # crashy/squeeze-prone names and post-shock reversion.
            'distribution_shape': ['st_'],
            # Tokenomics: supply-inflation/unlock pressure (economically
            # distinct from liquidity/size); log-mcap is the size gate.
            'tokenomics':        ['cap_supply_inflation', 'cap_log_mcap'],
            # Classic price TA (momentum quality, RSI, MACD, Bollinger,
            # ADX/DMI, ATR, Chandelier). Weak standalone on RESIDUAL returns -
            # best as GATES/interactions (prompt says so): "reversal only when
            # the trend is exhausted/choppy".
            'trend_state':       ['mq_', 'ma_', 'rs_', 'bb_', 'mc_', 'dm_',
                                  'at_', 'ch_'],
            # Liquidity/cost = how EXPENSIVE/illiquid a name is to trade.
            'liquidity':         ['lq_', 'vl_volume', 'ms_avg_trade',
                                  'ms_large_trades', 'ms_dollar_volume',
                                  'cap_turnover'],
            # Order flow = DIRECTIONAL aggressive-flow / informed-trading
            # signals (its own bandit arm, documented orthogonal breadth).
            # Split out of the crowded liquidity family so the OFI/toxicity/
            # signed-flow columns stop getting capped away.
            'order_flow':        ['of_', 'ms_ofi', 'ms_buy', 'ms_up_down',
                                  'ms_vol_return', 'ms_trade_intensity',
                                  'vl_taker', 'vl_signed'],
            'derivatives':       ['fr_', 'oi_', 'pos_'],
            # un_ = token-unlock calendar (etl/unlocks.py): a FORWARD-
            # knowable vesting schedule per name (cliff timing/size).
            'unlocks':           ['un_'],
            # dv_ = Electric Capital dev activity (30d-lagged);
            # ls_ = listing age (true first perp trade date).
            'dev_activity':      ['dv_'],
            'listing':           ['ls_'],
            'cross_sectional':   ['cs_'],
            'factor_context':    ['fl_', 'mk_'],
            'seasonality':       ['sn_'],
            'lead_lag':          ['ll_'],
            # ev_/mx_ are cross-sectionally constant: gate/interaction
            # ingredients (the proposer prompt says so); mb_ are per-name
            # macro sensitivities - direct cross-sectional material.
            'events':            ['ev_'],
            'macro':             ['mx_'],
            'macro_beta':        ['mb_'],
            # Calendar: perp funding-settlement proximity. Cross-sectionally
            # CONSTANT (same for all coins) -> gate-only, like ev_/mx_.
            'calendar':          ['tm_funding_window'],
        },
        'max_features_per_family': 16,   # cap resolved columns per family
        # (16 so the order_flow family holds both the existing ms_/vl_ flow
        # columns and the new of_ primitives after a feature rebuild; the LLM
        # still only sees diagnostics.top_per_family of them per call)
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
            'n_generations': 16,
            'batch_size': 32,
            'survivors': 12,                 # population carried between generations
            'mutation_prob': 0.6,            # mutate a parent vs sample fresh
            'diversity_max_corr': 0.8,       # survivor de-correlation ceiling (output)
            # Structural de-dup: reject a survivor whose AST (subtree) overlap
            # with a kept one exceeds this - catches clones that slip under the
            # output-correlation ceiling. 1.0 disables.
            'diversity_max_ast_sim': 0.6,
            # Frequent-subtree avoidance: how many over-mined building blocks to
            # show the LLM each generation (with a 'vary away' instruction).
            'overused_subtrees_shown': 6,
            'bandit_ucb_c': 1.0,             # family-bandit exploration constant
        },
        # Reward = sum_k weight_k * term_k / scale_k, TRAIN window ONLY.
        # The search (reward, survival, breeding, direction) never sees the
        # select window; promotion touches select exactly once per survivor.
        # Optimizing the same window promotion tests is a winner's curse the
        # deflation haircut cannot repair - hence the hard train/select split.
        # Scales are FIXED constants (not batch-relative) so rewards are
        # comparable across generations, rolls and resumed runs.
        # The performance term is the PER-BET RETURN (alpha in return
        # units: gross-1 dollar-neutral book from the signal, held for the
        # horizon) - money, not rank IC. Rank IC is recorded as a diagnostic
        # only: this crop proved a signal can order names correctly (rank IC
        # t~15 at 1d) while the large moves run against it (negative dollars).
        # alpha_tstat is the CAPTURE-WEIGHTED day-equivalent train t of that
        # per-bet return at the candidate's best lag:
        #   t / sqrt(stamps per day)  x  1/(1 + phi/kappa)
        # (phi = ln2 / effective persistence, kappa = the GP trade rate).
        # liquid_alpha_ratio is the liquid-half alpha vs the full book.
        # incremental is the train per-bet return the candidate ADDS to the
        # current survivor book. Scale is in return units (~1bp per bet).
        # complexity/instability/similarity are search hygiene; instability
        # is the std of per-bet alpha across train thirds (return units).
        'reward': {
            'weights': {
                'alpha_tstat': 1.0,
                'liquid_alpha_ratio': 0.25,
                'incremental': 0.5,
                'complexity': -0.15,
                'instability': -0.75,
                'similarity': -0.5,
            },
            'scales': {
                'alpha_tstat': 2.0,
                'liquid_alpha_ratio': 1.0,
                'incremental': 0.0001,
                'complexity': 10.0,
                'instability': 0.0005,
                'similarity': 0.5,
            },
        },
        # Promotion gates, applied once per roll to the search survivors.
        'promotion': {
            'fdr_alpha': 0.10,
            'fdr_method': 'by',              # 'by' or 'bh'
            # Minimum directed select t on the PER-BET RETURN (not rank IC)
            # at a promoted lag.
            'min_select_alpha_tstat': 2.0,
            # Select-window p-values use Student-t with (n_days - 1) dof: a
            # 1-month select gives ~30 daily alpha observations, where the
            # normal approximation is anti-conservative.
            # A survivor promotes if ANY lag of its profile clears the gates;
            # multiplicity is priced by the deflation haircut over the ACTUAL
            # looks at select: |t| must clear deflation_mult x
            # E[max |N(0,1)| over (n_survivors x n_lags)]. The search itself
            # never touches select (train-only reward), so trials that never
            # looked at select do not inflate the bar. 0 disables.
            'deflation_mult': 1.0,
            # Minimum daily alpha observations behind a select t-stat.
            'min_select_days': 20,
            # Fraction of profile lags whose train alpha agrees in sign with
            # the traded direction: a genuine alpha term structure has one
            # sign across horizons; a mixed profile is a red flag.
            'min_profile_sign_agreement': 0.75,
            'max_book_corr': 0.5,            # signal corr vs already-promoted book
            # Consecutive-roll survival required before promotion. 1 =
            # OFF: each month stands alone, promoted purely on its own
            # train + held-out select month, with no dependence on prior
            # months. (>1 would re-couple months as an anti-overfit filter.)
            'min_rolls_survived': 1,
            'max_book_size': 15,      # per roll (the book re-forms every roll)
            # Capture floor: minimum persistence weight 1/(1 + phi/kappa) a
            # candidate needs to promote. phi uses EFFECTIVE persistence
            # min(alpha half-life, position life 1/turnover_per_bar); kappa is
            # the GP fill rate the walk-forward trades at (~0.048/bar). 0.5 =
            # effective persistence >= ~15 bars, i.e. per-bar turnover <=
            # ~0.07 - this floor is the graded churn gate. 0 disables.
            # Duration-based, never a cost model.
            'min_capture': 0.5,
            # Turnover CEILING (backstop above the capture floor, which
            # already binds at ~0.07/bar): rejects a signal whose per-bar
            # churn (0.5*sum|dw| on the gross-1 signal) exceeds the cap.
            # Calibration: measured survivor churn is median ~0.14/bar with
            # only ~8% below 0.01 - the first pick (0.01) rejected every
            # statistically-passing survivor and left only sparse macro-gated
            # signals too thin to measure. 0.10 is a fails-open extremes
            # backstop; the capture floor does the graded work. None disables.
            'max_turnover': 0.10,
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
            # gemini-2.5-flash was RETIRED by Google mid-2026 (generate calls
            # 404 even though models.list still shows it). 3.1-flash-lite is
            # the price-equivalent replacement ($0.25/$1.50 vs the old
            # $0.30/$2.50), a generation newer, and accepts the same call
            # shape (JSON mode + thinking budget). The stronger 3.5-flash
            # works too but costs ~4x ($1.50/$9.00).
            'model': {
                'anthropic': 'claude-sonnet-4-6',
                'gemini': 'gemini-3.1-flash-lite',
            },
            # max_output_tokens must cover BOTH the reasoning budget below
            # AND the JSON (8 candidates ~= 2-3k tokens). 10240 leaves the
            # model room to think ~3k tokens and still emit a full batch;
            # the parser salvages the prefix if a batch is ever cut.
            'max_tokens': 10240,
            'candidates_per_call': 8,
            # Per-request timeout (seconds). A dropped connection (e.g. wifi
            # blip) then RAISES instead of hanging the whole run forever - the
            # proposer retries once, then degrades to an empty batch and the
            # search continues on parents. SDK auto-retries are disabled so
            # this is the only wait.
            'request_timeout_s': 120,
            # Concurrent per-family proposal calls within a generation. The
            # calls are independent (same parents/diagnostics snapshot), and
            # the sequential round-trips were the roll's entire wall-clock
            # (~17 families x 16 gens x 30-60s/call = hours). 1 = sequential;
            # lower it if the provider rate-limits.
            'parallel_requests': 8,
            # Gemini 2.5 thinking budget (tokens). The whole point of using an
            # LLM here is economic reasoning, so give it room to deliberate
            # before emitting JSON (weigh mechanisms, diversify the batch).
            # Counts against max_tokens above. 0 disables; None = provider
            # default.
            'gemini_thinking_budget': 3072,
            # $ per million tokens, per provider - used ONLY for the cost
            # estimate printed/persisted by discovery.py. Prices change, so
            # pin your provider's current rates here; None disables the
            # dollar estimate (token counts are always tracked).
            'price_per_mtok': {
                'anthropic': {'input': None, 'output': None},
                # gemini-3.1-flash-lite (checked 2026-07): output price
                # includes thinking tokens. Update here if the model or
                # Google's rates change.
                'gemini': {'input': 0.25, 'output': 1.50},
            },
        },
        'diagnostics': {
            'n_bins': 10,                    # binned forward-return deciles
            # Regime splits shown to the proposer (per-feature IC in the high
            # vs low half of each): name vol, crowding, cross-asset risk
            # appetite, and event proximity - the ev_/mx_ entries are what
            # prompt the LLM to write event/macro-GATED programs.
            'regime_columns': ['res_vol_short', 'cs_rel_volume',
                               'mx_vix_z', 'ev_hours_since_event'],
            'top_per_family': 6,             # compressed view size
            # Ranking blend for which features get full diagnostics (each term
            # rank-normalized within the family): monotonic alpha t-stat + decile
            # nonlinearity + regime spread + stability, so U-shaped/threshold/
            # regime-only features are not hidden by a t-stat-only sort.
            'top_blend': {'monotonic': 1.0, 'nonlinear': 0.6,
                          'regime': 0.5, 'stability': 0.3},
            'top_random_quota': 1,           # of top_per_family, reserved for exploration
        },
        # Persisted tables. Discovery is purely statistical: it emits
        # promotions (and its trial ledger); the walk-forward is the ONLY
        # money judge - there is no discovery-side PnL table.
        'tables': {
            'ledger': 'discovery_ledger',
            'promotions': 'discovery_promotions',
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
        # Economic floor: annualized Sharpe of the signal's own daily returns
        # AFTER the amortized per-side cost on its rebalance turnover, in the
        # traded direction. 0.0 = require net break-even. Also drives
        # candidate ranking and the composite combination weights.
        'min_net_sharpe_threshold': 0.0,
        'max_correlation_threshold': 0.50,
        # Minimum holding lag (bars) a signal may be selected at - the speed
        # match between signal decay and how fast the book actually trades.
        # Speed floor: 'auto' derives it from the execution layer (floor =
        # f/(1-f) * 1/kappa bars) and compares it against each signal's
        # TURNOVER-IMPLIED persistence (lag / per-rebalance turnover), so
        # selection and execution price signal speed identically. Integer =
        # manual floor in bars; 0 = off.
        'min_holding_lag_bars': 'auto',
        # f above: fraction of alpha that must survive the aim discount.
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

        # Covariance-aware signal combination: composite weights
        # w ~ C^{-1} . net_Sharpe (training, after amortized costs, clipped
        # at 0) - weight by measured after-cost value, de-correlated.
        # corr_shrink pulls C toward the identity for stability. Fallback
        # (<2 signals with return history, or disabled): plain
        # net-Sharpe-proportional weights.
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
        'cov_window_days': 30,               # Trailing window for residual covariance
        'cov_min_observations': 1008,        # Min bars (7d) for a valid covariance
        'shrinkage': 'ledoit_wolf',          # 'ledoit_wolf' or float in [0,1] (mvo only)
        'gross_leverage': 1.0,               # Sum |w| target
        # Per-name cap (fraction of gross). At 0.50 the neutrality
        # constraints and the volume-participation cap are the effective
        # diversification / capacity limits.
        'max_position': 0.50,
        'neutrality': ['dollar', 'market', 'size', 'momentum', 'vol', 'meme'],  # Constrained exposures B'w
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
            'meme':     0.10,
        },
        'weight_smoothing_halflife': 6,      # legacy fixed EWM rate (fallback only)
        # No global turnover budget: the volume-participation cap below is the
        # only hard fill constraint (a budget throttled the fill to ~3.6 days
        # on 1-day alpha and flipped a +52% OOS composite into a losing book).
        # Garleanu-Pedersen multi-period trading toward the gross-1 aim. Two pieces:
        # 1) TRADE RATE (per bar): the myopic optimal gamma/(gamma+lambda) balance
        #    of off-aim penalty vs quadratic trade cost, made COST-RESPONSIVE:
        #        omega = trade_urgency * (ref_cost_bps / cost_bps);  rate = omega/(1+omega)
        #    null -> fall back to the fixed halflife. PnL costs unchanged (linear).
        # 2) AIM DISCOUNT: bucket alpha scaled by h/(h + 1/rate) - alpha that decays
        #    faster than the (slow) trade rate can't be monetized, so it is
        #    downweighted; slow / persistent alpha is overweighted.
        'gp_trading': {
            'enabled': True,
            'trade_urgency': 0.05,           # ~4.8%/bar fill: positions build over hours
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
        # Per-side bps applied to turnover: the ALL-IN cost of trading slowly
        # (fees + fill-miss/adverse selection + drift while working orders).
        # Every net metric in the system - selection, reward, weighting,
        # deployment, backtest PnL - keys off this one number.
        'cost_bps': 5.0,
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
        # (Grinold & Kahn's recommendation, and SLS "Creatively MVO Your
        # Ranked Signals" - large IC swings should not whipsaw the book).
        # Applied per bucket.
        'ic_shrink': 0.5,
        # Edge-scaled gross: aim gross x clip(expected holding-period edge /
        # (edge_mult x round-trip cost), 0, 1); below min_mult the aim snaps
        # to ZERO (0.5 = deploy only at model break-even or better).
        'edge_scaled_gross': {
            'enabled': True,
            'edge_mult': 2.0,
            'min_mult': 0.5,
        },
        # No-trade zone ("lazy trading"):
        # a name whose expected residual return OVER ITS HOLDING HORIZON
        # (per-bar Grinold alpha * horizon bars) is below no_trade_band_mult *
        # (per-name per-side cost) cannot pay for a round trip, so its alpha is
        # zeroed (the optimiser won't allocate fresh risk to it). Compared on the
        # holding horizon, not per-bar, so units match the round-trip cost.
        'no_trade_band_mult': 1.0,
        # Cap on the turnover-implied holding period (bars): everywhere the
        # system asks how long a position lives / how long its alpha persists
        # it uses lag / per-rebalance turnover, capped here (7d).
        'cost_holding_max_bars': 1008,
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
        # Volume-participation cap: per bar, a name's trade may not exceed
        # max_participation x its trailing avg bar $ volume; book_size_usd
        # converts the $ cap to weight units. No volume history -> cap 0.
        'participation': {
            'book_size_usd': 1_000_000,      # notional gross book for $-based caps
            'max_participation': 0.10,       # max fraction of avg bar $ volume per bar
            'volume_window_bars': 10,        # trailing window for the average
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
    if port['gross_leverage'] <= 0 or not 0 < port['max_position'] <= 1:
        raise ValueError("portfolio leverage and position cap must be positive")
    min_positions = math.ceil(port['gross_leverage'] / port['max_position'])
    if port['min_assets'] <= min_positions:
        raise ValueError(
            "portfolio.min_assets must exceed "
            "ceil(gross_leverage / max_position) to leave room for neutrality"
        )
    esg = port.get('edge_scaled_gross', {})
    if esg.get('enabled'):
        if float(esg.get('edge_mult', 0)) <= 0:
            raise ValueError(
                "portfolio.edge_scaled_gross.edge_mult must be positive")
        if not 0.0 <= float(esg.get('min_mult', 0.0)) < 1.0:
            raise ValueError(
                "portfolio.edge_scaled_gross.min_mult must be in [0, 1)")
    part = port.get('participation', {})
    if float(part.get('book_size_usd', 0)) <= 0:
        raise ValueError(
            "portfolio.participation.book_size_usd must be positive")
    if not 0.0 < float(part.get('max_participation', 0)) <= 1.0:
        raise ValueError(
            "portfolio.participation.max_participation must be in (0, 1]")
    if int(part.get('volume_window_bars', 0)) < 1:
        raise ValueError(
            "portfolio.participation.volume_window_bars must be >= 1")
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
