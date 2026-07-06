"""
SEARCH: scoring, reward, the candidate ledger, and the budgeted evolutionary
loop that decides what gets tried next.

- evaluate_window(): compiled signal + window slice -> metrics dict, reusing
  the production math from research/signals/evaluate.py (vectorized rank IC,
  Newey-West HAC t-stat on the daily IC series, dollar-neutral screening
  backtest) so a discovered candidate's numbers mean the same thing as a
  production signal's. Non-overlapping stamps (stride = target lag).
- compute_reward(): SELECT-only metrics -> one scalar. Scales are FIXED config
  constants, never batch-relative: the same candidate must earn the same
  reward regardless of its batch-mates, or rewards stop being comparable
  across generations/rolls/resumes.
- DiscoveryLedger: one row per (roll, candidate) evaluation - the debug
  surface, the dedup index, the honest trial count behind the deflation
  haircut, and the memory behind the N-consecutive-rolls persistence gate.
- run_search(): the per-roll evolutionary loop with a UCB bandit over
  candidate families. MULTI-LAG: each candidate is evaluated at every lag in
  discovery.search_lags_bars on TRAIN and pinned to its strongest one there;
  traded sign AND lag are fixed on TRAIN (never SELECT), so the lag search
  adds no select-window multiplicity.
- run_ml_probe(): gradient-boosting ceiling estimator on ALL resolved
  primitives, per search lag - where (if anywhere) does the feature set
  contain predictability at all?
"""

import logging
import math
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from config import get
from research.signals.evaluate import (_nw_tstat, dollar_neutral_backtest,
                                       rank_ic_per_timestamp)
from research.signals.agent.data import (Roll, purge_bars, slice_window,
                                         strided_stamps, target_col)
from research.signals.agent.generation import (Candidate, Proposer,
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
        'liquid_ic_ratio': np.nan, 'net_sharpe': 0.0, 'gross_sharpe': 0.0,
        'turnover': np.nan, 'n_cross_sections': 0, 'n_days': 0,
    }


def _annualized_sharpe(daily_returns: np.ndarray) -> float:
    x = np.asarray(daily_returns, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 3 or x.std() == 0:
        return 0.0
    return float(x.mean() / x.std() * np.sqrt(DAYS_PER_YEAR))


def evaluate_window(signal: pd.DataFrame, window_panel: pd.DataFrame,
                    tcol: str, lag_bars: int,
                    min_assets: Optional[int] = None,
                    cost_bps: Optional[float] = None) -> dict:
    """Score one compiled signal on one window.

    signal: [timestamp, symbol, signal] (from compile_candidate)
    window_panel: the window's slice of the roll panel (must contain tcol;
                  is_liquid optional). The inner join does the slicing.
    """
    cfg = get('discovery', {})
    if min_assets is None:
        min_assets = int(cfg['min_assets_per_timestamp'])
    if cost_bps is None:
        cost_bps = cfg['backtest']['cost_bps']
        if cost_bps is None:
            cost_bps = get('portfolio.cost_bps')
    cost_bps = float(cost_bps)

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

    bt = dollar_neutral_backtest(df, tcol, None, min_assets=min_assets,
                                 cost_bps=cost_bps)
    if bt.empty:
        net_sharpe = gross_sharpe = 0.0
        turnover = np.nan
    else:
        daily = bt.set_index('timestamp').groupby(
            lambda ts: ts.normalize())[['gross_return', 'net_return']].sum()
        net_sharpe = _annualized_sharpe(daily['net_return'].values)
        gross_sharpe = _annualized_sharpe(daily['gross_return'].values)
        turnover = float(bt['turnover'].mean())

    return {
        'ic_mean': ic_mean,
        'ic_tstat': float(_nw_tstat(daily_ic.values, 'auto')),
        'icir': ic_mean / ic_std if ic_std > 0 else 0.0,
        'liquid_ic_ratio': liquid_ic_ratio,
        'net_sharpe': net_sharpe,
        'gross_sharpe': gross_sharpe,
        'turnover': turnover,
        'n_cross_sections': int(len(ics)),
        'n_days': int(len(daily_ic)),
    }


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

def reward_terms(train_metrics: dict, select_metrics: dict,
                 cand: Candidate, similarity: float) -> dict:
    """The raw (unscaled) reward terms. All finite."""
    def _f(x, default=0.0):
        return float(x) if x is not None and np.isfinite(x) else default

    return {
        'ic_tstat': _f(select_metrics.get('ic_tstat')),
        'net_sharpe': _f(select_metrics.get('net_sharpe')),
        'liquid_ic_ratio': _f(select_metrics.get('liquid_ic_ratio')),
        'turnover': _f(select_metrics.get('turnover')),
        'complexity': float(complexity(cand)),
        'instability': abs(_f(train_metrics.get('ic_mean'))
                           - _f(select_metrics.get('ic_mean'))),
        'similarity': _f(similarity),
    }


def compute_reward(train_metrics: dict, select_metrics: dict,
                   cand: Candidate, similarity: float,
                   reward_cfg: Optional[dict] = None) -> tuple:
    """reward = sum_k weight_k * term_k / scale_k. SELECT window only, fixed
    scales from config. Returns (reward, terms)."""
    cfg = reward_cfg or get('discovery.reward', {})
    weights = cfg['weights']
    scales = cfg['scales']
    terms = reward_terms(train_metrics, select_metrics, cand, similarity)
    total = 0.0
    for key, w in weights.items():
        total += float(w) * terms[key] / float(scales[key])
    return float(total), terms


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
               target_lag: Optional[int] = None) -> None:
        row = {
            'roll_id': int(roll_id),
            'generation': int(generation),
            'cand_hash': cand.hash,
            'name': cand.name,
            'family': cand.family,
            'candidate_json': cand.to_json(),
            'direction': int(direction),
            'target_lag': int(target_lag) if target_lag is not None else -1,
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


def select_survivors(population: List[dict], k: int,
                     max_corr: float) -> List[dict]:
    """Best-first greedy de-correlation: keep the highest-reward candidates
    whose SELECT-window signal correlates below max_corr with every already-
    kept one. Prevents the population collapsing to near-duplicates."""
    ranked = sorted(population, key=lambda s: -s['reward'])
    kept: List[dict] = []
    for cand in ranked:
        if len(kept) >= k:
            break
        if max_signal_correlation(cand['signal_select'],
                                  [x['signal_select'] for x in kept]) <= max_corr:
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
    # Multi-lag search: every candidate is evaluated at EVERY search lag on
    # TRAIN and pinned to the strongest one (|IC t-stat|), so one run finds
    # signals wherever on the speed spectrum they live. target_lag_bars stays
    # the reference lag for the proposer diagnostics only.
    from research.signals.agent.data import resolve_search_lags
    search_lags = resolve_search_lags(cfg)
    diag_lag = int(cfg['target_lag_bars'])
    diag_tcol = target_col(diag_lag)
    min_assets = int(cfg['min_assets_per_timestamp'])
    cost_bps = cfg['backtest']['cost_bps']
    if cost_bps is None:
        cost_bps = get('portfolio.cost_bps')

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

    def try_candidate(cand, gen: int) -> None:
        """Validate, compile, evaluate (train->select), reward, record."""
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

        # Best lag chosen on TRAIN only (strongest |IC t-stat| across the
        # search grid); SELECT is then scored ONCE, at that lag - the lag
        # search adds no select-window multiplicity (the FDR/deflation gates
        # keep operating on one select t-stat per candidate).
        m_by_lag = {
            lag_i: evaluate_window(sig, train, target_col(lag_i), lag_i,
                                   min_assets, cost_bps)
            for lag_i in search_lags
        }
        lag = max(m_by_lag, key=lambda l: abs(m_by_lag[l]['ic_tstat']))
        m_train = m_by_lag[lag]
        tcol = target_col(lag)
        # Traded sign is fixed on TRAIN, never on SELECT.
        direction = 1 if m_train['ic_mean'] >= 0 else -1
        if direction < 0:
            sig = sig.assign(signal=-sig['signal'])
            m_train = evaluate_window(sig, train, tcol, lag,
                                      min_assets, cost_bps)
        m_select = evaluate_window(sig, select, tcol, lag,
                                   min_assets, cost_bps)
        sig_select = sig[sig['timestamp'] >= roll.select_start]

        similarity = max_signal_correlation(
            sig_select, [s['signal_select'] for s in population])
        rwd, terms = compute_reward(m_train, m_select, cand,
                                    similarity, cfg['reward'])
        ledger.record(roll.roll_id, gen, cand, direction,
                      m_train, m_select, rwd, terms, target_lag=lag)
        if cand.family in bandit:
            bandit[cand.family]['n'] += 1
            bandit[cand.family]['sum'] += rwd
        population.append({
            'candidate': cand, 'direction': direction,
            'target_lag': int(lag),
            'reward': rwd, 'metrics_train': m_train,
            'metrics_select': m_select,
            'signal_select': sig_select.reset_index(drop=True),
        })

    # Generation -1: the previous roll's survivors re-earn their place on the
    # new windows before any fresh proposals are made.
    for cand in (seed_candidates or []):
        try_candidate(cand, -1)
    if seed_candidates:
        logging.info(f"roll {roll.roll_id}: seeded {len(population)} of "
                     f"{len(seed_candidates)} previous survivors")

    for gen in range(int(search_cfg['n_generations'])):
        alloc = allocate_batch(bandit, families,
                               int(search_cfg['batch_size']),
                               float(search_cfg['bandit_ucb_c']))
        parents = [s['candidate'] for s in population]
        for family in families:
            n_fam = alloc[family]
            if n_fam <= 0:
                continue
            cands = proposer.propose(n_fam, family, diagnostics, parents,
                                     family_columns, rng)
            for cand in cands:
                try_candidate(cand, gen)
        population = select_survivors(population, int(search_cfg['survivors']),
                                      float(search_cfg['diversity_max_corr']))
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
