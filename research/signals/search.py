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
"""

import json
import logging
import math
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from config import BARS_PER_DAY, get
from research.lib.signal_eval import _nw_tstat, rank_ic_per_timestamp
from research.signals.data import (Roll, purge_bars, slice_window,
                                         strided_stamps, target_col)
from research.signals.generation import (_to_list, ast_similarity,
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
        'alpha_mean': np.nan, 'alpha_tstat': 0.0, 'alpha_ir': np.nan,
        'liquid_alpha_ratio': np.nan, 'rank_ic_mean': np.nan,
        'rank_ic_tstat': 0.0, 'target_dispersion': np.nan,
        'n_cross_sections': 0, 'n_days': 0,
    }


def _book_returns(df: pd.DataFrame, tcol: str, min_assets: int) -> pd.Series:
    """Per-stamp return of the gross-1 dollar-neutral book built from the
    signal cross-section: v_t = sum_i w_i * target_i with w = demeaned signal
    scaled to gross 1. Return units per bet - money, not correlation."""
    d = df[['timestamp', 'signal', tcol]].dropna()
    if d.empty:
        return pd.Series(dtype=float)
    g = d.groupby('timestamp')
    n = g['signal'].transform('size')
    d = d[n >= min_assets]
    if d.empty:
        return pd.Series(dtype=float)
    g = d.groupby('timestamp')
    w = d['signal'] - g['signal'].transform('mean')
    gross = w.abs().groupby(d['timestamp']).transform('sum')
    w = w / gross.replace(0, np.nan)
    return (w * d[tcol]).groupby(d['timestamp']).sum(min_count=1).dropna()


def evaluate_window(signal: pd.DataFrame, window_panel: pd.DataFrame,
                    tcol: str, lag_bars: int,
                    min_assets: Optional[int] = None) -> dict:
    """Score one compiled signal on one window in RETURN UNITS.

    alpha_mean is the mean per-bet return of the gross-1 dollar-neutral book
    built from the signal and held for the horizon - money per bet, not a
    correlation. Rank IC is kept as a DIAGNOSTIC only: a signal can order
    names correctly (high rank IC) while the large moves run against its
    positions (negative alpha) - rank IC never selects anything.
    liquid_alpha_ratio is the liquid-half alpha vs the full cross-section
    (a capacity read).

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

    bets = _book_returns(df, tcol, min_assets)
    if bets.empty:
        return empty_metrics()
    alpha_mean = float(bets.mean())
    alpha_std = float(bets.std())
    daily_alpha = bets.groupby(lambda ts: ts.normalize()).mean()

    if 'is_liquid' in df.columns:
        liq = _book_returns(df[df['is_liquid']], tcol,
                            max(3, min_assets // 2))
        liq_alpha = float(liq.mean()) if not liq.empty else np.nan
    else:
        liq_alpha = np.nan
    # Signed ratio capped at 1: same-sign liquid alpha of at least equal
    # magnitude scores 1; opposite sign goes negative (capacity red flag).
    if np.isfinite(liq_alpha) and abs(alpha_mean) > 1e-12:
        liquid_alpha_ratio = float(np.clip(liq_alpha / alpha_mean, -1.0, 1.0))
    else:
        liquid_alpha_ratio = np.nan

    # Diagnostic rank IC (ordering quality; never selects).
    ics = rank_ic_per_timestamp(df[['timestamp', 'signal', tcol]], tcol,
                                min_assets=min_assets)
    if not ics.empty:
        daily_ic = ics.set_index('timestamp')['ic'].groupby(
            lambda ts: ts.normalize()).mean()
        rank_ic_mean = float(ics['ic'].mean())
        rank_ic_tstat = float(_nw_tstat(daily_ic.values, 'auto'))
    else:
        rank_ic_mean, rank_ic_tstat = np.nan, 0.0

    disp = float(df.groupby('timestamp')[tcol].std().mean())

    return {
        'alpha_mean': alpha_mean,
        'alpha_tstat': float(_nw_tstat(daily_alpha.values, 'auto')),
        'alpha_ir': alpha_mean / alpha_std if alpha_std > 0 else 0.0,
        'liquid_alpha_ratio': liquid_alpha_ratio,
        'rank_ic_mean': rank_ic_mean,
        'rank_ic_tstat': rank_ic_tstat,
        'target_dispersion': disp,
        'n_cross_sections': int(len(bets)),
        'n_days': int(len(daily_alpha)),
    }


def flip_metrics(m: dict) -> dict:
    """Metrics of the sign-flipped signal, analytically: the per-bet return is
    exactly antisymmetric under negation (alpha/tstat/ir and rank IC flip
    sign; liquid_alpha_ratio is a ratio of two flipped alphas, unchanged;
    dispersion/counts unchanged)."""
    out = dict(m)
    for k in ('alpha_mean', 'alpha_tstat', 'alpha_ir',
              'rank_ic_mean', 'rank_ic_tstat'):
        v = out.get(k)
        if v is not None and np.isfinite(v):
            out[k] = -v
    return out


def day_equivalent_tstat(m: dict, lag_bars: int) -> float:
    """Cross-lag-FAIR strength: t / sqrt(stamps per day). A raw t-stat grows
    ~sqrt(number of bets), handing 1h signals a mechanical ~sqrt(24) edge
    over 24h ones for the SAME per-bet alpha; dividing by sqrt(stamps/day)
    puts every horizon on one bets-per-day-free scale."""
    t = m.get('alpha_tstat', 0.0)
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


def signal_turnover(sig: pd.DataFrame) -> float:
    """Mean per-bar one-sided turnover of a compiled signal, as a fraction of
    gross book: 0 = positions never change, 1 = the book is fully replaced
    each bar.

    Each timestamp's cross-section is normalised to gross 1 (the dollar-neutral
    book weights the signal implies), then turnover_t = 0.5 * sum_i |w_{i,t} -
    w_{i,t-1}| - the fraction of the book traded between consecutive bars,
    counting names entering/leaving the cross-section as full trades.

    This is a property of the SIGNAL ALONE - no portfolio, no assumed cost -
    and is the standalone 'is this even tradeable' diagnostic from SLS's
    'Modern Spirit of Statistical Arbitrage' (a 6.9% signal trades on its own;
    a 36.8% one does not). DIAGNOSTIC ONLY: never a reward or promotion term -
    real cost is a portfolio property, judged in the walk-forward."""
    if sig is None or sig.empty:
        return float('nan')
    w = sig.pivot_table(index='timestamp', columns='symbol',
                        values='signal', aggfunc='first').sort_index()
    gross = w.abs().sum(axis=1)
    w = w.div(gross.where(gross > 0), axis=0).fillna(0.0)   # gross-1 per bar
    dw = (w.diff().abs().sum(axis=1) * 0.5).iloc[1:]         # skip first bar
    return float(dw.mean()) if len(dw) else float('nan')


def alpha_term_structure(m_by_lag: Dict[int, dict]) -> Dict[int, float]:
    """Cumulative alpha per bet at each horizon: A(L) = alpha_mean(L),
    measured directly in return units (the book's held return per bet - no
    IC x dispersion proxy)."""
    out = {}
    for lag, m in m_by_lag.items():
        a = m.get('alpha_mean')
        if a is not None and np.isfinite(a):
            out[int(lag)] = float(a)
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
        rate = omega / (1.0 + omega)
    else:
        hl = float(port.get('weight_smoothing_halflife', 6) or 6)
        rate = 1.0 - math.exp(-math.log(2.0) / max(hl, 1e-9))
    return rate


def effective_persistence_bars(half_life_bars: float, lag_bars: int,
                               turnover: Optional[float]) -> float:
    """Persistence the capture weight prices: min(alpha half-life,
    turnover-implied position life). Turnover is PER BAR (fraction of the
    gross-1 signal replaced each bar), so position life = 1/turnover bars -
    how long until the signal has fully reshuffled itself. The half-life says
    how long the ALPHA lives; 1/turnover says how long the POSITIONS live;
    the discount honors the shorter. Turnover clipped to [1e-4, 2];
    missing/NaN turnover falls back to the half-life alone. (lag_bars kept
    for call-site stability; unused.)"""
    hl = max(float(half_life_bars), 1e-9)
    if turnover is None or not np.isfinite(turnover) or turnover <= 0:
        return hl
    return min(hl, 1.0 / min(max(float(turnover), 1e-4), 2.0))


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
                 similarity: float, incremental: float = 0.0,
                 turnover: Optional[float] = None) -> dict:
    """The raw (unscaled) reward terms - TRAIN window only. All finite.

    alpha_tstat is the CAPTURE-WEIGHTED day-equivalent train t of the
    candidate's PER-BET RETURN (not rank IC) at its best lag,
    discounted by the fraction of it a book trading at the portfolio rate
    can hold long enough to be exposed to (see persistence_weight - duration,
    never bps). instability is the std of the alpha across train thirds.
    incremental is the per-bet return the candidate ADDS to the current
    survivor book (marginal edge), measured on train."""
    def _f(x, default=0.0):
        return float(x) if x is not None and np.isfinite(x) else default

    # Capture prices the persistence the book can actually monetize:
    # min(alpha half-life, position life lag/turnover) at the GP fill rate.
    p_eff = effective_persistence_bars(half_life_bars, best_lag, turnover)
    capture = persistence_weight(p_eff, trade_rate_per_bar())
    return {
        'alpha_tstat': day_equivalent_tstat(train_metrics, best_lag) * capture,
        'liquid_alpha_ratio': _f(train_metrics.get('liquid_alpha_ratio')),
        'incremental': _f(incremental),
        'complexity': float(complexity(cand)),
        'instability': _f(instability),
        'similarity': _f(similarity),
    }


def compute_reward(train_metrics: dict, best_lag: int, half_life_bars: float,
                   instability: float, cand: Candidate, similarity: float,
                   incremental: float = 0.0,
                   reward_cfg: Optional[dict] = None,
                   turnover: Optional[float] = None) -> tuple:
    """reward = sum_k weight_k * term_k / scale_k. TRAIN window only, fixed
    scales from config. Returns (reward, terms)."""
    cfg = reward_cfg or get('discovery.reward', {})
    weights = cfg['weights']
    scales = cfg['scales']
    terms = reward_terms(train_metrics, best_lag, half_life_bars,
                         instability, cand, similarity, incremental,
                         turnover=turnover)
    total = 0.0
    for key, w in weights.items():
        total += float(w) * terms[key] / float(scales[key])
    return float(total), terms


def train_thirds_instability(sig: pd.DataFrame, train: pd.DataFrame,
                             tcol: str, lag_bars: int,
                             min_assets: int) -> float:
    """Std of the signal's per-bet alpha over three contiguous time-thirds of
    TRAIN - the search-hygiene consistency term (select is never consulted)."""
    stamps = np.sort(train['timestamp'].unique())
    if len(stamps) < 9:
        return 0.0
    alphas = []
    for part in np.array_split(stamps, 3):
        sub = train[train['timestamp'].isin(part)]
        m = evaluate_window(sig, sub, tcol, lag_bars, min_assets)
        a = m.get('alpha_mean')
        alphas.append(float(a) if a is not None and np.isfinite(a) else 0.0)
    return float(np.std(alphas))


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
               half_life_bars: Optional[float] = None,
               turnover: Optional[float] = None) -> None:
        """target_lag = the BEST PER-BET train lag (display/sorting only -
        the signal is not pinned to it); profile_json = the full per-lag
        train+select metrics; half_life_bars = fitted alpha half-life;
        turnover = mean per-bar book turnover of the train signal (DIAGNOSTIC
        ONLY - ledger column, never read by the walk-forward). Rows written by
        older code simply carry NaN here, so a resumed run mixes cleanly."""
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
            'turnover': (float(turnover) if turnover is not None
                         and np.isfinite(turnover) else np.nan),
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
    from research.signals.data import resolve_search_lags
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

    from research.signals.data import (all_family_columns,
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
        # (the per-bet return is exactly antisymmetric under signal negation).
        direction = 1 if m_train_by_lag[best_lag]['alpha_mean'] >= 0 else -1
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
        # Standalone tradeability diagnostic (train signal): what fraction of
        # the book this signal churns per bar. Ledger-only, never a reward or
        # promotion term - real cost belongs to the walk-forward.
        turnover = signal_turnover(sig_train)

        instability = train_thirds_instability(sig, train,
                                               target_col(best_lag),
                                               best_lag, min_assets)
        similarity = max_signal_correlation(
            sig_train, [s['signal_train'] for s in population])
        # Marginal contribution: the train per-bet return the candidate ADDS to
        # the current survivor book at its best lag (pooled alpha minus the
        # book's own). 0 when the book is empty or the reward ignores it.
        incremental = 0.0
        if incr_weight != 0.0 and book_signals:
            tcol_b = target_col(best_lag)
            if best_lag not in book_ic_cache:
                bk = evaluate_window(pooled_signal(book_signals), train,
                                     tcol_b, best_lag,
                                     min_assets)['alpha_mean']
                book_ic_cache[best_lag] = bk if np.isfinite(bk) else 0.0
            comb = evaluate_window(pooled_signal(book_signals + [sig_train]),
                                   train, tcol_b, best_lag,
                                   min_assets)['alpha_mean']
            comb = comb if np.isfinite(comb) else 0.0
            incremental = comb - book_ic_cache[best_lag]
        rwd, terms = compute_reward(m_train, best_lag, half_life,
                                    instability, cand, similarity,
                                    incremental, cfg['reward'],
                                    turnover=turnover)
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
                      half_life_bars=half_life, turnover=turnover)
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
            'turnover': turnover,
            'signal_train': sig_train.reset_index(drop=True),
            'signal_select': sig_select.reset_index(drop=True),
        })

    # Generation -1: the previous roll's survivors re-earn their place on the
    # new windows before any fresh proposals are made.
    for cand in (seed_candidates or []):
        try_candidate(cand, -1)
    if seed_candidates:
        logging.debug(f"roll {roll.roll_id}: seeded {len(population)} of "
                     f"{len(seed_candidates)} previous survivors")

    # Failure memory: recently-culled low-reward candidates, so the LLM can be
    # told what NOT to re-propose (worst first, capped).
    failures: List[dict] = []

    def _parent_scores(pop) -> Dict[str, dict]:
        return {s['candidate'].hash: {
            'reward': round(float(s['reward']), 3),
            'alpha_tstat': round(day_equivalent_tstat(s['metrics_train'],
                                                      s['target_lag']), 2),
            'half_life_bars': int(s.get('half_life_bars', 0) or 0),
        } for s in pop}

    n_overused = int(search_cfg.get('overused_subtrees_shown', 6))
    max_ast_sim = float(search_cfg.get('diversity_max_ast_sim', 1.0))
    # API proposal calls within a generation are independent (same
    # parents/diagnostics snapshot), so they run CONCURRENTLY - the LLM
    # round-trips are the roll's wall-clock, not the scoring. Sequential for
    # non-API proposers (RandomProposer: instant, and the shared rng is not
    # thread-safe). Scoring stays sequential: it mutates shared state and its
    # order must be deterministic.
    n_parallel = (int(cfg['llm'].get('parallel_requests', 1))
                  if getattr(proposer, 'provider', '') else 1)

    def propose_generation(alloc, parents, parent_scores, fail_hint,
                           overused) -> List:
        funded = [f for f in families if alloc[f] > 0]
        kw = dict(parent_scores=parent_scores, failures=fail_hint,
                  overused=overused)
        if n_parallel > 1 and len(funded) > 1:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(
                    max_workers=min(n_parallel, len(funded))) as ex:
                futs = {f: ex.submit(proposer.propose, alloc[f], f,
                                     diagnostics, parents, family_columns,
                                     rng, **kw)
                        for f in funded}
                for _ in tqdm(as_completed(futs.values()), total=len(futs),
                              desc='  propose (LLM)', unit='call',
                              leave=False):
                    pass
                # Collect in fixed family order: deterministic scoring order.
                return [(f, c) for f in funded for c in futs[f].result()]
        return [(f, c) for f in funded
                for c in proposer.propose(alloc[f], f, diagnostics, parents,
                                          family_columns, rng, **kw)]

    gen_bar = tqdm(range(int(search_cfg['n_generations'])),
                   desc=f"roll {roll.roll_id} search", unit='gen')
    with logging_redirect_tqdm():
        for gen in gen_bar:
            alloc = allocate_batch(bandit, families,
                                   int(search_cfg['batch_size']),
                                   float(search_cfg['bandit_ucb_c']))
            parents = [s['candidate'] for s in population]
            parent_scores = _parent_scores(population)
            fail_hint = [{'expression': c.to_dict()['expression'],
                          'conditions': c.to_dict()['conditions'],
                          'reward': round(float(r), 3)}
                         for c, r in sorted(failures, key=lambda x: x[1])[:6]]
            # Refresh the survivor book the incremental-contribution reward
            # scores against (directed train signals from the last cull).
            book_signals[:] = [s['signal_train'] for s in population]
            book_ic_cache.clear()
            # Over-mined structural building blocks: tell the LLM to vary away.
            overused = [_to_list(st)
                        for st, _ in subtree_counts.most_common(n_overused)]
            batch = propose_generation(alloc, parents, parent_scores,
                                       fail_hint, overused)
            for _, cand in tqdm(batch, desc='  score', unit='cand',
                                leave=False):
                try_candidate(cand, gen)
            pre_cull = population
            population = select_survivors(population,
                                          int(search_cfg['survivors']),
                                          float(search_cfg['diversity_max_corr']),
                                          max_ast_sim)
            # Whatever was tried this gen but did not survive is a failure to
            # remember (low reward first is selected at prompt time).
            kept = {s['candidate'].hash for s in population}
            for s in pre_cull:
                if s['candidate'].hash not in kept:
                    failures.append((s['candidate'], s['reward']))
            failures = sorted(failures, key=lambda x: x[1])[:50]  # worst 50
            best = max((s['reward'] for s in population), default=0)
            gen_bar.set_postfix(trials=ledger.n_trials(roll.roll_id),
                                best=f"{best:.3f}", pop=len(population))
            logging.debug(f"roll {roll.roll_id} gen {gen}: "
                         f"{ledger.n_trials(roll.roll_id)} trials, "
                         f"best reward {best:.3f}")
    gen_bar.close()

    lag_mix: Dict[int, int] = {}
    for s in population:
        lag_mix[s['target_lag']] = lag_mix.get(s['target_lag'], 0) + 1
    turns = [s['turnover'] for s in population
             if s.get('turnover') is not None and np.isfinite(s['turnover'])]
    turn_str = (f", turnover/bar {np.median(turns):.1%} median "
                f"[{min(turns):.1%}-{max(turns):.1%}]" if turns else "")
    logging.debug(f"roll {roll.roll_id}: survivor lag mix "
                 f"{dict(sorted(lag_mix.items()))} (bars){turn_str}")
    ledger.mark_survivors(roll.roll_id,
                          [s['candidate'].hash for s in population])
    return population

