"""
SEARCH: scoring, reward, the candidate ledger, and the budgeted evolutionary
loop that decides what gets tried next.

- evaluate_window(): compiled signal + window slice -> IC metrics dict,
  reusing the production math from research/lib/signal_eval.py (vectorized
  rank IC, Newey-West HAC t-stat on the daily IC series). Non-overlapping
  stamps (stride = target lag).
- compute_reward(): TRAIN-only metrics -> one scalar. The search (reward,
  survival, breeding, direction) NEVER sees the select window - promotion
  touches select exactly once per survivor, so a select t-stat is a
  measurement, not the maximum of a directed search on itself. Scales are
  FIXED config constants, never batch-relative: the same candidate must earn
  the same reward regardless of its batch-mates, or rewards stop being
  comparable across generations/rolls/resumes.
- NO PINNING: each candidate is evaluated at EVERY lag in
  discovery.horizon_lags_bars on train AND select; the per-lag profile is
  its alpha term structure. best_lag (day-equivalent t, so fast lags get no
  mechanical sqrt(stamps) advantage) picks the direction and the reward
  term; the WHOLE profile travels to promotion and the portfolio layer,
  where fit_half_life() turns it into the persistence discount.
- DiscoveryLedger: one row per (roll, candidate) evaluation - the debug
  surface, the dedup index, and the memory behind the N-consecutive-rolls
  persistence gate.
- run_search(): the per-roll evolutionary loop with a UCB bandit over
  candidate families.
- run_ml_probe(): gradient-boosting ceiling estimator on ALL resolved
  primitives, per search lag - where (if anywhere) does the feature set
  contain predictability at all?
"""

import json
import logging
import math
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from config import BARS_PER_DAY, get
from research.lib.signal_eval import _nw_tstat, rank_ic_per_timestamp
from research.signals.agent.data import (Roll, purge_bars, slice_window,
                                         strided_stamps, target_col)
from research.signals.agent.generation import (_to_list, ast_similarity,
                                               candidate_subtrees,
                                               Candidate, Proposer,
                                               ValidationError,
                                               compile_candidate, complexity,
                                               validate_candidate)

DAYS_PER_YEAR = 365


# =============================================================================
# Evaluator
# =============================================================================

def empty_metrics() -> dict:
    return {
        'ic_mean': np.nan, 'ic_tstat': 0.0, 'icir': np.nan,
        'liquid_ic_ratio': np.nan, 'target_dispersion': np.nan,
        'n_cross_sections': 0, 'n_days': 0,
    }


def _annualized_sharpe(daily_returns: np.ndarray) -> float:
    x = np.asarray(daily_returns, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 3 or x.std() == 0:
        return 0.0
    return float(x.mean() / x.std() * np.sqrt(DAYS_PER_YEAR))


def evaluate_window(signal: pd.DataFrame, window_panel: pd.DataFrame,
                    tcol: str, lag_bars: int,
                    min_assets: Optional[int] = None) -> dict:
    """Score one compiled signal on one window - IC statistics ONLY.

    Signals are judged purely on rank IC vs the forward residual target
    (signals are not tradeable objects: cost/PnL are properties of the
    portfolio layer, never of a signal). liquid_ic_ratio is itself an IC
    (liquid-half vs full cross-section - a capacity read).

    signal: [timestamp, symbol, signal] (from compile_candidate)
    window_panel: the window's slice of the roll panel (must contain tcol;
                  is_liquid optional). The inner join does the slicing.
    """
    cfg = get('discovery', {})
    if min_assets is None:
        min_assets = int(cfg['min_assets_per_timestamp'])

    cols = ['timestamp', 'symbol', tcol]
    if 'is_liquid' in window_panel.columns:
        cols.append('is_liquid')
    df = signal.merge(window_panel[cols], on=['timestamp', 'symbol'],
                      how='inner')
    if df.empty:
        return empty_metrics()

    stamps = strided_stamps(df['timestamp'], lag_bars)
    df = df[df['timestamp'].isin(stamps)]
    if df.empty:
        return empty_metrics()

    ics = rank_ic_per_timestamp(df[['timestamp', 'signal', tcol]], tcol,
                                min_assets=min_assets)
    if ics.empty:
        return empty_metrics()

    ic_mean = float(ics['ic'].mean())
    ic_std = float(ics['ic'].std())
    daily_ic = ics.set_index('timestamp')['ic'].groupby(
        lambda ts: ts.normalize()).mean()

    if 'is_liquid' in df.columns:
        liq_ics = rank_ic_per_timestamp(
            df[df['is_liquid']][['timestamp', 'signal', tcol]], tcol,
            min_assets=max(3, min_assets // 2))
        liq_ic = float(liq_ics['ic'].mean()) if not liq_ics.empty else np.nan
    else:
        liq_ic = np.nan
    # Signed ratio capped at 1: same-sign liquid IC of at least equal
    # magnitude scores 1; opposite sign goes negative (capacity red flag).
    if np.isfinite(liq_ic) and abs(ic_mean) > 1e-12:
        liquid_ic_ratio = float(np.clip(liq_ic / ic_mean, -1.0, 1.0))
    else:
        liquid_ic_ratio = np.nan

    # Mean per-stamp cross-sectional std of the forward target: with the
    # signal z-scored, alpha_k = ic_mean * target_dispersion is the Grinold
    # expected-return-per-bet at this horizon - the profile of these across
    # lags is the signal's alpha term structure (SLS "Multi-Period").
    disp = float(df.groupby('timestamp')[tcol].std().mean())

    return {
        'ic_mean': ic_mean,
        'ic_tstat': float(_nw_tstat(daily_ic.values, 'auto')),
        'icir': ic_mean / ic_std if ic_std > 0 else 0.0,
        'liquid_ic_ratio': liquid_ic_ratio,
        'target_dispersion': disp,
        'n_cross_sections': int(len(ics)),
        'n_days': int(len(daily_ic)),
    }


def flip_metrics(m: dict) -> dict:
    """Metrics of the sign-flipped signal, analytically: rank IC is exactly
    antisymmetric under negation (ic/tstat/icir flip sign; liquid_ic_ratio is
    a ratio of two flipped ICs, unchanged; dispersion/counts unchanged)."""
    out = dict(m)
    for k in ('ic_mean', 'ic_tstat', 'icir'):
        v = out.get(k)
        if v is not None and np.isfinite(v):
            out[k] = -v
    return out


def day_equivalent_tstat(m: dict, lag_bars: int) -> float:
    """Cross-lag-FAIR strength: t / sqrt(stamps per day). A raw t-stat grows
    ~sqrt(number of bets), handing 1h signals a mechanical ~sqrt(24) edge
    over 24h ones for the SAME per-bet IC; dividing by sqrt(stamps/day) puts
    every horizon on one bets-per-day-free scale (= ic_mean * sqrt(n_days) up
    to the daily-IC noise normalization)."""
    t = m.get('ic_tstat', 0.0)
    if t is None or not np.isfinite(t):
        return 0.0
    stamps_per_day = max(BARS_PER_DAY // max(int(lag_bars), 1), 1)
    return float(t) / math.sqrt(stamps_per_day)


def pooled_signal(signals: List[pd.DataFrame]) -> pd.DataFrame:
    """Average several [timestamp, symbol, signal] panels into one alpha panel
    (rank IC is scale-invariant, so a plain mean is enough)."""
    wide = pd.concat([s.set_index(['timestamp', 'symbol'])['signal']
                      for s in signals], axis=1)
    return wide.mean(axis=1).rename('signal').reset_index()


def alpha_term_structure(m_by_lag: Dict[int, dict]) -> Dict[int, float]:
    """Cumulative expected alpha per bet at each horizon: A(L) = ic_mean(L) *
    target_dispersion(L). This is the empirical alpha_k curve of SLS
    "Multi-Period Optimisation" (regression of cumulative forward returns on
    a unit-variance signal)."""
    out = {}
    for lag, m in m_by_lag.items():
        ic, disp = m.get('ic_mean'), m.get('target_dispersion')
        if (ic is not None and disp is not None
                and np.isfinite(ic) and np.isfinite(disp)):
            out[int(lag)] = float(ic) * float(disp)
    return out


def trade_rate_per_bar(cfg: Optional[dict] = None) -> float:
    """The portfolio layer's per-bar fill rate toward the aim - the SAME
    cost-responsive Garleanu-Pedersen rate walk_forward trades at:
    omega = trade_urgency * (ref_cost_bps / cost_bps); rate = omega/(1+omega).
    Fallback (gp_trading disabled): the legacy fixed smoothing halflife."""
    port = get('portfolio', {})
    gp = port.get('gp_trading', {})
    urgency = gp.get('trade_urgency')
    if gp.get('enabled', False) and urgency is not None:
        ref = float(gp.get('ref_cost_bps', port['cost_bps']) or port['cost_bps'])
        omega = float(urgency) * (ref / float(port['cost_bps']))
        return omega / (1.0 + omega)
    hl = float(port.get('weight_smoothing_halflife', 6) or 6)
    return 1.0 - math.exp(-math.log(2.0) / max(hl, 1e-9))


def persistence_weight(half_life_bars: float, rate_bar: float) -> float:
    """Garleanu-Pedersen capture fraction 1/(1 + phi/rate): phi =
    ln2/half-life is the signal's alpha decay rate, rate the book's per-bar
    fill rate. The fraction of a signal's IC a book trading at `rate` can
    actually be exposed to - alpha faster than the book's speed is
    discounted toward zero; persistent alpha keeps its weight. Duration,
    never bps. (SLS "Trading Multiple Forecasts Optimally": weight signals
    by how persistent they are, not just how accurate.)"""
    hl = max(float(half_life_bars), 1e-9)
    phi = math.log(2.0) / hl
    return 1.0 / (1.0 + phi / max(float(rate_bar), 1e-9))


# Candidate alpha half-lives (bars) for the deterministic grid fit below.
HALF_LIFE_GRID = [3, 6, 12, 24, 48, 96, 144, 288, 432, 720, 1008, 2016]


def fit_half_life(profile: Dict[int, float]) -> float:
    """Alpha half-life (bars) from the cumulative term structure A(L).

    Model: per-bar alpha a(t) = a0 * exp(-phi t), so A(L) = a0 (1 - e^{-phi
    L}) / phi. Fit by least squares over a fixed half-life grid (a0 solved
    analytically per grid point) - deterministic, 4 data points, no
    optimizer. Degenerate profiles (empty / non-positive everywhere) fall
    back to the SHORTEST grid half-life: unmeasurable persistence is priced
    as fast decay, never as free persistence."""
    lags = sorted(L for L, a in profile.items() if np.isfinite(a))
    if not lags or max(profile[L] for L in lags) <= 0:
        return float(HALF_LIFE_GRID[0])
    A = np.array([profile[L] for L in lags], dtype=float)
    Ls = np.array(lags, dtype=float)

    best_hl, best_sse = HALF_LIFE_GRID[0], np.inf
    for hl in HALF_LIFE_GRID:
        phi = math.log(2.0) / hl
        shape = (1.0 - np.exp(-phi * Ls)) / phi
        denom = float((shape ** 2).sum())
        if denom <= 0:
            continue
        a0 = float((A * shape).sum()) / denom
        if a0 <= 0:
            continue
        sse = float(((A - a0 * shape) ** 2).sum())
        if sse < best_sse:
            best_hl, best_sse = hl, sse
    return float(best_hl)


def signal_correlation(sig_a: pd.DataFrame, sig_b: pd.DataFrame) -> float:
    """Pearson correlation of two compiled signal panels on their common
    (timestamp, symbol) support. 0.0 when support is too thin."""
    m = sig_a.merge(sig_b, on=['timestamp', 'symbol'], suffixes=('_a', '_b'))
    if len(m) < 10:
        return 0.0
    a = m['signal_a'].values
    b = m['signal_b'].values
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 10 or a[ok].std() == 0 or b[ok].std() == 0:
        return 0.0
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def max_signal_correlation(signal: pd.DataFrame, others: list) -> float:
    """max |corr| of a signal vs a list of compiled signal panels."""
    if not others:
        return 0.0
    return max(abs(signal_correlation(signal, o)) for o in others)


# =============================================================================
# Reward
# =============================================================================

def reward_terms(train_metrics: dict, best_lag: int, half_life_bars: float,
                 instability: float, cand: Candidate,
                 similarity: float, incremental: float = 0.0) -> dict:
    """The raw (unscaled) reward terms - TRAIN window only. All finite.

    ic_tstat is the CAPTURE-WEIGHTED day-equivalent train t at the
    candidate's best per-bet lag: strength x the fraction of it a book
    trading at the portfolio rate can hold long enough to be exposed to
    (see persistence_weight - duration, never bps). A 6h-half-life signal
    outscores a 1h one of equal strength ~2.4x, so the search breeds toward
    persistence. instability is the std of the train-thirds ICs (temporal
    consistency inside train - select is never consulted). incremental is the
    IC the candidate ADDS to the current survivor book (marginal edge, not a
    redundant copy of the dominant signal - AlphaGen / Lucky-Factors idea),
    measured on train."""
    def _f(x, default=0.0):
        return float(x) if x is not None and np.isfinite(x) else default

    capture = persistence_weight(half_life_bars, trade_rate_per_bar())
    return {
        'ic_tstat': day_equivalent_tstat(train_metrics, best_lag) * capture,
        'liquid_ic_ratio': _f(train_metrics.get('liquid_ic_ratio')),
        'incremental': _f(incremental),
        'complexity': float(complexity(cand)),
        'instability': _f(instability),
        'similarity': _f(similarity),
    }


def compute_reward(train_metrics: dict, best_lag: int, half_life_bars: float,
                   instability: float, cand: Candidate, similarity: float,
                   incremental: float = 0.0,
                   reward_cfg: Optional[dict] = None) -> tuple:
    """reward = sum_k weight_k * term_k / scale_k. TRAIN window only, fixed
    scales from config. Returns (reward, terms)."""
    cfg = reward_cfg or get('discovery.reward', {})
    weights = cfg['weights']
    scales = cfg['scales']
    terms = reward_terms(train_metrics, best_lag, half_life_bars,
                         instability, cand, similarity, incremental)
    total = 0.0
    for key, w in weights.items():
        total += float(w) * terms[key] / float(scales[key])
    return float(total), terms


def train_thirds_instability(sig: pd.DataFrame, train: pd.DataFrame,
                             tcol: str, lag_bars: int,
                             min_assets: int) -> float:
    """Std of the signal's IC over three contiguous time-thirds of TRAIN -
    the search-hygiene consistency term (replaces the old train-vs-select
    gap, which leaked select into survival)."""
    stamps = np.sort(train['timestamp'].unique())
    if len(stamps) < 9:
        return 0.0
    ics = []
    for part in np.array_split(stamps, 3):
        sub = train[train['timestamp'].isin(part)]
        m = evaluate_window(sig, sub, tcol, lag_bars, min_assets)
        ic = m.get('ic_mean')
        ics.append(float(ic) if ic is not None and np.isfinite(ic) else 0.0)
    return float(np.std(ics))


# =============================================================================
# Ledger
# =============================================================================

class DiscoveryLedger:
    """One row per (roll, candidate) evaluation. table_name=None keeps it
    purely in memory (tests, dry runs); otherwise persisted via dbutil so the
    trial count and survivor history are honest across resumed runs."""

    def __init__(self, table_name: Optional[str] = None):
        self.table_name = table_name
        self._rows: List[dict] = []
        if table_name:
            self._load_existing()

    def _load_existing(self):
        from dbutil import load_data, table_exists
        if table_exists(self.table_name):
            df = load_data(self.table_name)
            if df is not None and not df.empty:
                self._rows = df.to_dict('records')

    # -- recording ----------------------------------------------------------

    def record(self, roll_id: int, generation: int, cand: Candidate,
               direction: int, train_metrics: dict, select_metrics: dict,
               reward: float, terms: dict,
               target_lag: Optional[int] = None,
               profile_json: Optional[str] = None,
               half_life_bars: Optional[float] = None) -> None:
        """target_lag = the BEST PER-BET train lag (display/sorting only -
        the signal is not pinned to it); profile_json = the full per-lag
        train+select metrics; half_life_bars = fitted alpha half-life."""
        row = {
            'roll_id': int(roll_id),
            'generation': int(generation),
            'cand_hash': cand.hash,
            'name': cand.name,
            'family': cand.family,
            'candidate_json': cand.to_json(),
            'direction': int(direction),
            'target_lag': int(target_lag) if target_lag is not None else -1,
            'half_life_bars': (float(half_life_bars)
                               if half_life_bars is not None else np.nan),
            'profile_json': profile_json or '',
            'reward': float(reward),
            'survivor': False,
            'promoted': False,
        }
        for k, v in select_metrics.items():
            row[f'select_{k}'] = v
        for k, v in train_metrics.items():
            row[f'train_{k}'] = v
        for k, v in terms.items():
            row[f'term_{k}'] = v
        self._rows.append(row)

    def _mark(self, roll_id: int, hashes, field: str) -> None:
        hashes = set(hashes)
        for row in self._rows:
            if row['roll_id'] == roll_id and row['cand_hash'] in hashes:
                row[field] = True

    def mark_survivors(self, roll_id: int, hashes) -> None:
        self._mark(roll_id, hashes, 'survivor')

    def mark_promoted(self, roll_id: int, hashes) -> None:
        self._mark(roll_id, hashes, 'promoted')

    # -- queries ------------------------------------------------------------

    def n_trials(self, roll_id: int) -> int:
        """Candidates evaluated this roll - the deflation denominator."""
        return sum(1 for r in self._rows if r['roll_id'] == roll_id)

    def seen_hashes(self, roll_id: Optional[int] = None) -> set:
        if roll_id is None:
            return {r['cand_hash'] for r in self._rows}
        return {r['cand_hash'] for r in self._rows if r['roll_id'] == roll_id}

    def survivor_hashes(self, roll_id: int) -> set:
        return {r['cand_hash'] for r in self._rows
                if r['roll_id'] == roll_id and r['survivor']}

    def survivor_candidates(self, roll_id: int) -> List[Candidate]:
        """Rebuild a roll's surviving Candidates from their stored JSON
        (used to seed the next roll's search on resumed runs)."""
        import json
        out, seen = [], set()
        for r in self._rows:
            if (r['roll_id'] == roll_id and r['survivor']
                    and r['cand_hash'] not in seen):
                seen.add(r['cand_hash'])
                out.append(Candidate.from_dict(json.loads(r['candidate_json'])))
        return out

    def consecutive_survivals(self, cand_hash: str, roll_id: int) -> int:
        """Consecutive rolls ending at roll_id in which cand_hash survived."""
        count = 0
        rid = roll_id
        while cand_hash in self.survivor_hashes(rid):
            count += 1
            rid -= 1
        return count

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self._rows)

    # -- persistence --------------------------------------------------------

    def flush(self) -> None:
        """Persist. Rewrites the whole table: survivor/promoted flags on old
        rows are updated in place, and the ledger is small (hundreds of rows
        per roll)."""
        if not self.table_name or not self._rows:
            return
        from dbutil import save_data
        save_data(self.table_name, self.to_frame(), mode='overwrite')


# =============================================================================
# Family bandit + evolutionary search controller
# =============================================================================

def allocate_batch(bandit: Dict[str, dict], families: List[str],
                   batch_size: int, ucb_c: float) -> Dict[str, int]:
    """UCB allocation of one generation's proposal slots across families.
    Untried families are drawn first (deterministically, in order)."""
    alloc = {f: 0 for f in families}
    n_total = sum(b['n'] for b in bandit.values())
    for _ in range(batch_size):
        untried = [f for f in families if bandit[f]['n'] + alloc[f] == 0]
        if untried:
            pick = untried[0]
        else:
            def ucb(f):
                n = bandit[f]['n'] + alloc[f]
                mean = bandit[f]['sum'] / max(bandit[f]['n'], 1)
                return mean + ucb_c * math.sqrt(
                    math.log(max(n_total + batch_size, 2)) / n)
            pick = max(families, key=ucb)
        alloc[pick] += 1
    return alloc


def select_survivors(population: List[dict], k: int, max_corr: float,
                     max_ast_sim: float = 1.0) -> List[dict]:
    """Best-first greedy de-duplication: keep the highest-reward candidates
    that are novel on TWO axes vs every already-kept one - OUTPUT correlation
    (train signal corr <= max_corr) AND STRUCTURE (AST similarity <=
    max_ast_sim). Correlation alone misses structural clones (same recipe,
    one column/window swapped, weakly correlated by luck); the AST check
    catches them. Train only - select stays unseen until promotion."""
    ranked = sorted(population, key=lambda s: -s['reward'])
    kept: List[dict] = []
    for cand in ranked:
        if len(kept) >= k:
            break
        corr_ok = max_signal_correlation(
            cand['signal_train'],
            [x['signal_train'] for x in kept]) <= max_corr
        ast_ok = max_ast_sim >= 1.0 or all(
            ast_similarity(cand['candidate'], x['candidate']) <= max_ast_sim
            for x in kept)
        if corr_ok and ast_ok:
            kept.append(cand)
    return kept


def run_search(panel: pd.DataFrame, roll: Roll,
               family_columns: Dict[str, list], proposer: Proposer,
               ledger: DiscoveryLedger,
               cfg: Optional[dict] = None,
               seed_candidates: Optional[List] = None) -> List[dict]:
    """One roll's budgeted propose -> compile -> evaluate -> keep-survivors
    loop. Returns the survivor list: dicts with candidate, direction, reward,
    metrics and the compiled SELECT-window signal. Never touches OOS.

    seed_candidates: the PREVIOUS roll's surviving Candidates. They are
    re-compiled and re-evaluated on THIS roll's windows (no leakage - they
    must re-earn survival), which is what makes the promotion gate's
    N-consecutive-rolls persistence measurable at all: without seeding, each
    roll starts empty and no hash can ever survive twice.
    """
    cfg = cfg or get('discovery', {})
    search_cfg = cfg['search']
    rng = np.random.default_rng(int(search_cfg['seed']) + roll.roll_id)
    # NO PINNING: every candidate is evaluated at EVERY search lag on train
    # and select - the per-lag profile is its alpha term structure. best_lag
    # (day-equivalent train t) only picks direction and the reward term.
    # target_lag_bars stays the reference lag for proposer diagnostics only.
    from research.signals.agent.data import resolve_search_lags
    search_lags = resolve_search_lags(cfg)
    diag_lag = int(cfg['target_lag_bars'])
    diag_tcol = target_col(diag_lag)
    min_assets = int(cfg['min_assets_per_timestamp'])

    pb = purge_bars(cfg)
    # Compile on train+select only: rolling warmup inside the roll, OOS unseen.
    roll_panel = slice_window(panel, roll.train_start, roll.oos_start, 0)
    roll_panel = roll_panel.reset_index(drop=True)
    train = slice_window(roll_panel, roll.train_start, roll.select_start, pb)
    select = slice_window(roll_panel, roll.select_start, roll.oos_start, pb)

    from research.signals.agent.data import (all_family_columns,
                                             build_diagnostics)
    allowed_cols = all_family_columns(family_columns)
    diagnostics = build_diagnostics(train, family_columns, diag_tcol,
                                    diag_lag, cfg)

    families = [f for f, cols in family_columns.items() if cols]
    bandit = {f: {'n': 0, 'sum': 0.0} for f in families}
    population: List[dict] = []
    seen = ledger.seen_hashes(roll.roll_id)
    # Frequent-subtree memory: how often each structural building block has been
    # tried this roll, so the LLM can be told which recipes are over-mined.
    from collections import Counter
    subtree_counts: "Counter" = Counter()
    # Incremental-contribution reward: the current survivor book (directed train
    # signals) and a per-lag cache of its pooled IC, refreshed each generation.
    incr_weight = float(cfg['reward']['weights'].get('incremental', 0.0))
    book_signals: List[pd.DataFrame] = []
    book_ic_cache: Dict[int, float] = {}

    def try_candidate(cand, gen: int) -> None:
        """Validate, compile, evaluate the FULL train+select profile, reward
        on train only, record."""
        if cand.hash in seen:
            return
        seen.add(cand.hash)
        try:
            validate_candidate(cand, allowed_cols, cfg['dsl'])
            sig = compile_candidate(cand, roll_panel)
        except (ValidationError, Exception) as e:
            logging.debug(f"candidate rejected: {cand.name}: {e}")
            return
        if sig.empty:
            return
        subtree_counts.update(candidate_subtrees(cand))

        # Train profile across the whole horizon grid. best_lag by the
        # DAY-EQUIVALENT t (per-bet-fair: no sqrt(stamps/day) advantage for
        # fast lags) picks direction and the reward term - nothing else.
        m_train_by_lag = {
            lag_i: evaluate_window(sig, train, target_col(lag_i), lag_i,
                                   min_assets)
            for lag_i in search_lags
        }
        best_lag = max(m_train_by_lag,
                       key=lambda l: abs(day_equivalent_tstat(
                           m_train_by_lag[l], l)))
        # Traded sign is fixed on TRAIN; the flip mirrors the whole profile
        # (rank IC is exactly antisymmetric under signal negation).
        direction = 1 if m_train_by_lag[best_lag]['ic_mean'] >= 0 else -1
        if direction < 0:
            sig = sig.assign(signal=-sig['signal'])
            m_train_by_lag = {l: flip_metrics(m)
                              for l, m in m_train_by_lag.items()}
        m_train = m_train_by_lag[best_lag]

        # Alpha term structure + half-life from TRAIN (drives the
        # persistence discount at the portfolio layer).
        half_life = fit_half_life(alpha_term_structure(m_train_by_lag))

        # Select profile: recorded for promotion to test ONCE - it feeds
        # nothing in this loop (no reward, no survival, no breeding).
        m_select_by_lag = {
            lag_i: evaluate_window(sig, select, target_col(lag_i), lag_i,
                                   min_assets)
            for lag_i in search_lags
        }
        m_select = m_select_by_lag[best_lag]
        sig_select = sig[sig['timestamp'] >= roll.select_start]
        sig_train = sig[sig['timestamp'] < roll.select_start]

        instability = train_thirds_instability(sig, train,
                                               target_col(best_lag),
                                               best_lag, min_assets)
        similarity = max_signal_correlation(
            sig_train, [s['signal_train'] for s in population])
        # Marginal contribution: how much train IC the candidate ADDS to the
        # current survivor book at its best lag (combined pooled IC minus the
        # book's own IC). 0 when the book is empty or the reward ignores it.
        incremental = 0.0
        if incr_weight != 0.0 and book_signals:
            tcol_b = target_col(best_lag)
            if best_lag not in book_ic_cache:
                bk = evaluate_window(pooled_signal(book_signals), train,
                                     tcol_b, best_lag, min_assets)['ic_mean']
                book_ic_cache[best_lag] = bk if np.isfinite(bk) else 0.0
            comb = evaluate_window(pooled_signal(book_signals + [sig_train]),
                                   train, tcol_b, best_lag,
                                   min_assets)['ic_mean']
            comb = comb if np.isfinite(comb) else 0.0
            incremental = comb - book_ic_cache[best_lag]
        rwd, terms = compute_reward(m_train, best_lag, half_life,
                                    instability, cand, similarity,
                                    incremental, cfg['reward'])
        profile = {
            str(l): {'train': {k: (None if v is None or not np.isfinite(v)
                                   else round(float(v), 8))
                               for k, v in m_train_by_lag[l].items()},
                     'select': {k: (None if v is None or not np.isfinite(v)
                                    else round(float(v), 8))
                                for k, v in m_select_by_lag[l].items()}}
            for l in search_lags
        }
        ledger.record(roll.roll_id, gen, cand, direction,
                      m_train, m_select, rwd, terms, target_lag=best_lag,
                      profile_json=json.dumps(profile),
                      half_life_bars=half_life)
        if cand.family in bandit:
            bandit[cand.family]['n'] += 1
            bandit[cand.family]['sum'] += rwd
        population.append({
            'candidate': cand, 'direction': direction,
            'target_lag': int(best_lag),
            'half_life_bars': float(half_life),
            'profile_train': m_train_by_lag,
            'profile_select': m_select_by_lag,
            'reward': rwd, 'metrics_train': m_train,
            'metrics_select': m_select,
            'signal_train': sig_train.reset_index(drop=True),
            'signal_select': sig_select.reset_index(drop=True),
        })

    # Generation -1: the previous roll's survivors re-earn their place on the
    # new windows before any fresh proposals are made.
    for cand in (seed_candidates or []):
        try_candidate(cand, -1)
    if seed_candidates:
        logging.info(f"roll {roll.roll_id}: seeded {len(population)} of "
                     f"{len(seed_candidates)} previous survivors")

    # Failure memory: recently-culled low-reward candidates, so the LLM can be
    # told what NOT to re-propose (worst first, capped).
    failures: List[dict] = []

    def _parent_scores(pop) -> Dict[str, dict]:
        return {s['candidate'].hash: {
            'reward': round(float(s['reward']), 3),
            'ic_tstat': round(day_equivalent_tstat(s['metrics_train'],
                                                   s['target_lag']), 2),
            'half_life_bars': int(s.get('half_life_bars', 0) or 0),
        } for s in pop}

    n_overused = int(search_cfg.get('overused_subtrees_shown', 6))
    max_ast_sim = float(search_cfg.get('diversity_max_ast_sim', 1.0))
    for gen in range(int(search_cfg['n_generations'])):
        alloc = allocate_batch(bandit, families,
                               int(search_cfg['batch_size']),
                               float(search_cfg['bandit_ucb_c']))
        parents = [s['candidate'] for s in population]
        parent_scores = _parent_scores(population)
        fail_hint = [{'expression': c.to_dict()['expression'],
                      'conditions': c.to_dict()['conditions'],
                      'reward': round(float(r), 3)}
                     for c, r in sorted(failures, key=lambda x: x[1])[:6]]
        # Refresh the survivor book the incremental-contribution reward scores
        # against (directed train signals from the last cull).
        book_signals[:] = [s['signal_train'] for s in population]
        book_ic_cache.clear()
        # Over-mined structural building blocks: tell the LLM to vary away.
        overused = [_to_list(st) for st, _ in subtree_counts.most_common(n_overused)]
        for family in families:
            n_fam = alloc[family]
            if n_fam <= 0:
                continue
            cands = proposer.propose(n_fam, family, diagnostics, parents,
                                     family_columns, rng,
                                     parent_scores=parent_scores,
                                     failures=fail_hint, overused=overused)
            for cand in cands:
                try_candidate(cand, gen)
        pre_cull = population
        population = select_survivors(population, int(search_cfg['survivors']),
                                      float(search_cfg['diversity_max_corr']),
                                      max_ast_sim)
        # Whatever was tried this gen but did not survive is a failure to
        # remember (low reward first is selected at prompt time).
        kept = {s['candidate'].hash for s in population}
        for s in pre_cull:
            if s['candidate'].hash not in kept:
                failures.append((s['candidate'], s['reward']))
        failures = sorted(failures, key=lambda x: x[1])[:50]   # keep worst 50
        logging.info(f"roll {roll.roll_id} gen {gen}: "
                     f"{ledger.n_trials(roll.roll_id)} trials, "
                     f"best reward "
                     f"{max((s['reward'] for s in population), default=0):.3f}")

    lag_mix: Dict[int, int] = {}
    for s in population:
        lag_mix[s['target_lag']] = lag_mix.get(s['target_lag'], 0) + 1
    logging.info(f"roll {roll.roll_id}: survivor lag mix "
                 f"{dict(sorted(lag_mix.items()))} (bars)")
    ledger.mark_survivors(roll.roll_id,
                          [s['candidate'].hash for s in population])
    return population


# =============================================================================
# ML ceiling probe
# =============================================================================

def run_ml_probe(panel: pd.DataFrame, roll: Roll, feature_cols: List[str],
                 cfg: Optional[dict] = None) -> dict:
    """Gradient boosting on ALL resolved primitives, fit on TRAIN, scored on
    SELECT, at EVERY search lag: the predictability ceiling of the feature
    set per horizon. If a lag's ceiling is ~0, the DSL search is digging in
    barren ground at that speed - this is the cheap map of where (if
    anywhere) alpha lives before the search spends its budget."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from research.signals.agent.data import resolve_search_lags

    cfg = cfg or get('discovery', {})
    ml_cfg = cfg['ml_probe']
    pb = purge_bars(cfg)

    roll_panel = slice_window(panel, roll.train_start, roll.oos_start, 0)
    train = slice_window(roll_panel, roll.train_start, roll.select_start, pb)
    select = slice_window(roll_panel, roll.select_start, roll.oos_start, pb)
    cols = [c for c in feature_cols if c in train.columns]

    metrics_by_lag: Dict[int, dict] = {}
    n_train_rows = 0
    degenerate_logged = False
    for lag in resolve_search_lags(cfg):
        tcol = target_col(lag)
        tr = train.dropna(subset=[tcol])
        tr = tr[tr[cols].notna().any(axis=1)]
        cap = int(ml_cfg['subsample_rows'])
        if len(tr) > cap:
            tr = tr.sort_values('timestamp').tail(cap)   # recent-biased
        if tr.empty:
            metrics_by_lag[lag] = empty_metrics()
            continue
        n_train_rows = max(n_train_rows, len(tr))

        # HistGradientBoosting's binner crashes on columns with < 2 distinct
        # non-NaN values (all-NaN or constant in this training slice - e.g.
        # futures columns before their data starts). Drop them per window;
        # they carry no information for the fit anyway.
        nun = tr[cols].nunique(dropna=True)
        use_cols = [c for c in cols if nun.get(c, 0) >= 2]
        if not use_cols:
            metrics_by_lag[lag] = empty_metrics()
            continue
        if len(use_cols) < len(cols) and not degenerate_logged:
            dropped = sorted(set(cols) - set(use_cols))
            logging.info(f"ml probe: dropped {len(dropped)} degenerate "
                         f"columns (all-NaN/constant in train): {dropped}")
            degenerate_logged = True

        model = HistGradientBoostingRegressor(
            max_iter=int(ml_cfg['max_iter']),
            max_depth=int(ml_cfg['max_depth']),
            learning_rate=float(ml_cfg['learning_rate']),
            min_samples_leaf=int(ml_cfg['min_samples_leaf']),
            l2_regularization=float(ml_cfg['l2_regularization']),
            random_state=int(cfg['search']['seed']),
        )
        model.fit(tr[use_cols].values, tr[tcol].values)

        sel = select[select[use_cols].notna().any(axis=1)]
        sig = sel[['timestamp', 'symbol']].copy()
        sig['signal'] = model.predict(sel[use_cols].values)
        g = sig.groupby('timestamp')['signal']
        sig['signal'] = ((sig['signal'] - g.transform('mean'))
                         / (g.transform('std') + 1e-10)).clip(-3, 3)
        metrics_by_lag[lag] = evaluate_window(
            sig.dropna(subset=['signal']), select, tcol, lag)

    return {'metrics_by_lag': metrics_by_lag, 'n_train_rows': n_train_rows}
