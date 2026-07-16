"""
SEARCH: scoring, reward, the candidate ledger, and the budgeted evolutionary
loop that decides what gets tried next.

- response_curve() / fit_response_curve(): the ONE measurement instrument.
  A compiled signal + a window's residual matrix -> the gross-1 book's mean
  cumulative return bar-by-bar over a day (144 bars, entries every 6),
  fitted deterministically to a0 (edge at the peak), half-life, peak_k and
  rev_frac. try_candidate measures it TWICE per candidate: on the train
  window (feeds the reward and the direction) and on the test window (the
  ledger's verdict, read once at promotion). Same instrument everywhere -
  train and test are only ever compared in the same units.
- compute_reward(): TRAIN curve -> one scalar, TWO terms only (net economic
  rate at the curve's own optimal holding, max_k (A(k)-roundtrip)/k - the
  identical number promotion ranks by - minus similarity to the kept
  survivors). The search (reward, survival, breeding, direction) NEVER
  sees the test window. Scales are FIXED config constants, never
  batch-relative: the same candidate must earn the same reward regardless
  of its batch-mates, or rewards stop being comparable across
  generations/rolls/resumes.
- The traded SIGN is fitted from the candidate's POOLED train-curve a0
  across every roll it was measured in (pooled_train_direction - one
  5-month window is wrong ~26% of the time for a Sharpe-1 signal); a
  negative direction analytically negates the curve (_flip_curve). Both
  curves travel to promotion in profile_json; the portfolio layer consumes
  half-life capped at the peak.
- DiscoveryLedger: one row per (roll, candidate) evaluation - the debug
  surface, the dedup index, and the train-history store behind the pooled
  direction.
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
from research.signals.data import (Roll, purge_bars, slice_window,
                                         target_col)
from research.signals.generation import (_to_list, candidate_columns,
                                               candidate_subtrees,
                                               Candidate, Proposer,
                                               ValidationError,
                                               compile_candidate,
                                               validate_candidate)

DAYS_PER_YEAR = 365


# =============================================================================
# Evaluator
# =============================================================================

def empty_metrics() -> dict:
    """The ledger's flat metric columns when nothing is measurable: per-bet
    edge (curve a0), its t (a0/se), entry days."""
    return {'alpha_mean': np.nan, 'alpha_tstat': 0.0, 'n_days': 0}


def pooled_train_direction(months: List[dict]) -> int:
    """Traded sign from POOLED raw train measurements: the inverse-variance
    weighted mean of the per-window per-bet alphas (se = |mean/t| per
    window) decides the sign.

    Why pooled: one 5-month train window fits the wrong direction
    Phi(-SR*sqrt(train_years)) of the time - ~26% for a true Sharpe-1
    signal, and consecutive windows overlap 4 of 5 months so the error can
    persist for several rolls, each wrong-way month then correctly counting
    AGAINST the signal at promotion. Pooling every train window the
    candidate was ever measured on drops the error with candidate age
    (train-only: the select window is never consulted). Consecutive windows
    overlap, so this is an over-counted but unbiased sign estimate - fine
    for a direction, never usable as a t-stat. A candidate whose current
    window disagrees with its pooled history keeps the historical sign and
    pays for the disagreement in this roll's directed reward - a signal
    that flip-flops SHOULD rank low.

    Months without a measurable (mean, t) fall out; if nothing is
    measurable, the LAST month's raw sign decides (the current window -
    the single-window behavior)."""
    num = den = 0.0
    for m in months or []:
        mu, t = m.get('alpha_mean'), m.get('alpha_tstat')
        if mu is None or t is None:
            continue
        mu, t = float(mu), float(t)
        if not (np.isfinite(mu) and np.isfinite(t)) or t == 0.0:
            continue
        se = abs(mu / t)
        if not np.isfinite(se) or se <= 0:
            continue
        num += mu / se ** 2
        den += 1.0 / se ** 2
    if den > 0:
        return 1 if num / den >= 0 else -1
    for m in reversed(months or []):
        mu = m.get('alpha_mean')
        if mu is not None and np.isfinite(mu):
            return 1 if mu >= 0 else -1
    return 1


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


def effective_persistence_bars(half_life_bars: float,
                               turnover: Optional[float]) -> float:
    """Persistence the capture weight prices: min(alpha half-life,
    turnover-implied position life). Turnover is PER BAR (fraction of the
    gross-1 signal replaced each bar), so position life = 1/turnover bars -
    how long until the signal has fully reshuffled itself. The half-life says
    how long the ALPHA lives; 1/turnover says how long the POSITIONS live;
    the discount honors the shorter. Turnover clipped to [1e-4, 2];
    missing/NaN turnover falls back to the half-life alone."""
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


# =============================================================================
# Response curve (the verdict instrument): the book's cumulative return
# bar-by-bar after entry, averaged over an entry grid on the TEST window.
# =============================================================================

def response_curve(sig: pd.DataFrame, res_wide: pd.DataFrame,
                   horizon_bars: int, entry_stride: int,
                   min_assets: int,
                   sample_ks: Optional[list] = None) -> Optional[dict]:
    """Mean cumulative response curve of the gross-1 dollar-neutral book.

    For each entry stamp t on the grid (every entry_stride bars of sig's
    schedule, restricted to entries whose full horizon fits inside
    res_wide): weights w_t = demeaned signal scaled to gross 1; the path is
    A_e(k) = sum_{j=1..k} w_t . r_{t+j}. Returns the entry-mean curve plus
    per-entry outcomes at the horizon for robustness stats:

      {'A': (H,) mean cumulative curve,
       'entries': n, 'entry_days': distinct entry dates,
       'n_eff': entries deflated for path overlap (stride/H) - error bars
                must use this, adjacent paths share almost all their bars,
       'per_entry_at': {k: (n,) array of per-entry A_e(k)} at sparse ks}

    None when no entry has a full path (window too short) - the candidate
    is skipped. Missing residuals along a path (delistings) contribute
    zero (the book stops earning on names that vanish)."""
    if sig is None or sig.empty:
        return None
    w = sig.pivot_table(index='timestamp', columns='symbol',
                        values='signal', aggfunc='first')
    w = w.sub(w.mean(axis=1), axis=0)
    gross = w.abs().sum(axis=1)
    w = w.div(gross.where(gross > 0), axis=0)
    n_names = w.notna().sum(axis=1)

    idx = res_wide.index
    pos = idx.get_indexer(w.index)
    ok = (pos >= 0) & (pos + horizon_bars < len(idx)) & \
         (n_names.values >= min_assets)
    entry_rows = np.flatnonzero(ok)[::max(1, int(entry_stride))]
    if len(entry_rows) == 0:
        return None

    r_vals = res_wide.reindex(columns=w.columns).to_numpy(copy=False)
    paths = np.empty((len(entry_rows), horizon_bars))
    for e, row in enumerate(entry_rows):
        we = np.nan_to_num(w.values[row])
        blk = r_vals[pos[row] + 1: pos[row] + 1 + horizon_bars]
        paths[e] = np.cumsum(np.nan_to_num(blk) @ we)

    entry_ts = w.index[entry_rows]
    # Per-entry outcomes sampled on the SAME log-spaced grid as the stored
    # curve, so the median/se are always taken close to the true peak - a
    # coarse grid judged a 1-hour edge (peak ~bar 6) at bar 1, wrongly.
    ks = sorted({int(k) for k in
                 (sample_ks or (1, horizon_bars // 4, horizon_bars // 2,
                                horizon_bars))
                 if 1 <= int(k) <= horizon_bars})
    return {
        'A': paths.mean(axis=0),
        'entries': int(len(entry_rows)),
        'entry_days': int(pd.DatetimeIndex(entry_ts).normalize().nunique()),
        'n_eff': max(1.0, len(entry_rows) * entry_stride / horizon_bars),
        'per_entry_at': {k: paths[:, k - 1] for k in ks},
    }


def fit_response_curve(A: np.ndarray, n_eff: float,
                       per_entry_at: Optional[dict] = None) -> dict:
    """Deterministic anatomy of a response curve (no optimizer, grid fits
    only - same reproducibility contract as the rest of the search):

      a0        edge per bet if held to the curve's peak (A at peak)
      half_life exponential-decay fit (HALF_LIFE_GRID least squares) over
                the pre-peak curve - the REAL phi, replacing the saturated
                4-point artifact
      peak_k    bar where the smoothed curve tops out; holding beyond it is
                actively harmful when rev_frac is material
      rev_frac  fraction of the peak edge given back by the horizon end
      se_peak   error bar on a0 from per-entry spread, deflated by n_eff
                (overlapping paths are not independent observations)
      median_peak  median per-entry outcome at (nearest sampled k to) peak
    """
    H = len(A)
    smooth = pd.Series(A).rolling(5, center=True, min_periods=1).mean().values
    peak_k = int(np.argmax(smooth)) + 1
    a0 = float(smooth[peak_k - 1])
    a_end = float(smooth[-1])
    rev_frac = float(1.0 - a_end / a0) if a0 > 0 else 0.0

    ks = np.arange(1, min(peak_k, H) + 1, dtype=float)
    seg = A[:len(ks)]
    best_hl, best_sse = HALF_LIFE_GRID[0], np.inf
    for hl in HALF_LIFE_GRID:
        phi = math.log(2.0) / hl
        shape = 1.0 - np.exp(-phi * ks)
        denom = float((shape ** 2).sum())
        if denom <= 0:
            continue
        amp = float((seg * shape).sum()) / denom
        if amp <= 0:
            continue
        sse = float(((seg - amp * shape) ** 2).sum())
        if sse < best_sse:
            best_hl, best_sse = hl, sse

    se_peak = float('nan')
    median_peak = float('nan')
    if per_entry_at:
        k_near = min(per_entry_at, key=lambda k: abs(k - peak_k))
        vals = np.asarray(per_entry_at[k_near], dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) > 1:
            se_peak = float(np.std(vals, ddof=1) / math.sqrt(max(n_eff, 1.0)))
            median_peak = float(np.median(vals))
    return {'a0': a0, 'half_life': float(best_hl), 'peak_k': peak_k,
            'rev_frac': rev_frac, 'se_peak': se_peak,
            'median_peak': median_peak}


# Candidate alpha half-lives (bars) for the deterministic grid fit above.
HALF_LIFE_GRID = [3, 6, 12, 24, 48, 96, 144, 288, 432, 720, 1008, 2016]


def thirds_sign_consistent(per_entry_vals) -> bool:
    """Whole-window robustness (train-only, no test contact): split the
    chronological per-entry outcomes at the curve's peak into thirds and
    require every third's mean to carry the SAME sign. A formula whose
    entire train profit is one burst (train t 17, test ~0 - the classic
    overfit shape) fails; a formula that worked across the whole window
    passes with either sign (the direction flip handles which). Fails OPEN
    on unmeasurable input - never block on a missing diagnostic."""
    vals = np.asarray(per_entry_vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) < 3:
        return True
    means = [float(t.mean()) for t in np.array_split(vals, 3) if len(t)]
    if len(means) < 3:
        return True
    return all(m > 0 for m in means) or all(m < 0 for m in means)


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

def reward_terms(net_rate: float, similarity: float) -> dict:
    """The raw (unscaled) reward terms - TRAIN window only. All finite.

    TWO terms. net_rate is the TRAIN response curve's net economic rate at
    its own optimal holding - max_k (A(k) - roundtrip)/k - the IDENTICAL
    number promotion ranks by and the book earns, so the search breeds for
    exactly what gets judged (one instrument everywhere; the reward's copy
    is computed on the train window only). similarity (max train-signal
    correlation vs the kept survivors) de-duplicates the pool."""
    def _f(x, default=0.0):
        return float(x) if x is not None and np.isfinite(x) else default

    return {'net_rate': _f(net_rate), 'similarity': _f(similarity)}


def compute_reward(net_rate: float, similarity: float,
                   reward_cfg: Optional[dict] = None) -> tuple:
    """reward = sum_k weight_k * term_k / scale_k. TRAIN window only, fixed
    scales from config. Returns (reward, terms)."""
    cfg = reward_cfg or get('discovery.reward', {})
    weights = cfg['weights']
    scales = cfg['scales']
    terms = reward_terms(net_rate, similarity)
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
        # Provenance stamp merged into every recorded row (run_id, config
        # hash, data fingerprint - set by discovery.py). Config tuning across
        # runs spends the select window's honesty; the stamp makes every row
        # attributable to the exact run/config/data that produced it, so a
        # table mixing runs (--resume after a config change) is DETECTABLE
        # instead of silently blended.
        self.run_stamp: dict = {}
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
        """target_lag = the train curve's peak bar (display/sorting only);
        profile_json = {'curve': test curve, 'curve_train': train curve};
        half_life_bars = the train curve's fitted half-life; turnover =
        mean per-bar book turnover of the train signal (DIAGNOSTIC ONLY -
        ledger column, never read by the walk-forward). Rows written by
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
        row.update(self.run_stamp)
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
        """Candidates evaluated this roll (recorded on each promotion)."""
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
        out, seen = [], set()
        for r in self._rows:
            if (r['roll_id'] == roll_id and r['survivor']
                    and r['cand_hash'] not in seen):
                seen.add(r['cand_hash'])
                out.append(Candidate.from_dict(json.loads(r['candidate_json'])))
        return out

    def promoted_candidates(self, min_roll: int,
                            max_roll: int) -> List[Candidate]:
        """Distinct Candidates promoted in rolls min_roll..max_roll
        (inclusive) - the retention re-seed pool: recent book members stay
        under measurement even after missing a survivor cut (see
        discovery.py), so one bad train month never discards an accumulated
        evidence stream."""
        out, seen = [], set()
        for r in self._rows:
            if (min_roll <= r['roll_id'] <= max_roll and r.get('promoted')
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

    def train_history(self, cand_hash: str,
                      up_to_roll: Optional[int] = None) -> List[dict]:
        """RAW-SIGN train-curve measurements of cand_hash (a0 at its peak,
        its error bar, its entry days), one per roll it was evaluated in.
        Stored values are directed by that roll's fitted direction, so
        entries are un-flipped here. Feeds pooled_train_direction -
        train-only, the test window stays unspent."""
        out = []
        for r in self._rows:
            if r['cand_hash'] != cand_hash:
                continue
            if up_to_roll is not None and r['roll_id'] > up_to_roll:
                continue
            pj = r.get('profile_json')
            if not isinstance(pj, str) or not pj:
                continue
            try:
                prof = json.loads(pj)
            except (ValueError, TypeError):
                continue
            d = int(r.get('direction', 1) or 1)
            c = prof.get('curve_train')
            if not c or c.get('a0') is None:
                continue
            a0 = float(c['a0'])
            se = c.get('se_peak')
            t = (a0 / float(se)) if se and np.isfinite(se) and se > 0 \
                else 0.0
            out.append({'roll_id': int(r['roll_id']),
                        'alpha_mean': d * a0, 'alpha_tstat': d * t,
                        'n_days': int(c.get('entry_days', 0) or 0)})
        return sorted(out, key=lambda x: x['roll_id'])

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
                     max_per_column: int = 0) -> List[dict]:
    """Best-first greedy de-duplication: keep the highest-reward candidates
    whose OUTPUT (train signal correlation) stays <= max_corr vs every
    already-kept one. What a signal outputs is what the book trades - two
    builds that rank the coins the same way are one signal. Train only -
    select stays unseen until promotion.

    max_per_column (0 = off): at most this many survivors may lean on the
    same feature column (expression or gate). Gated variants of one idea
    fire on different days, so their outputs decorrelate and slip past the
    correlation guard - but they are still one mechanism, and a survivor
    pool of unlock-signals-in-different-costumes fakes diversity."""
    from collections import Counter
    ranked = sorted(population, key=lambda s: -s['reward'])
    kept: List[dict] = []
    col_counts: "Counter" = Counter()
    for cand in ranked:
        if len(kept) >= k:
            break
        cols = candidate_columns(cand['candidate'])
        if max_per_column and any(col_counts[c] >= max_per_column
                                  for c in cols):
            continue
        if max_signal_correlation(
                cand['signal_train'],
                [x['signal_train'] for x in kept]) <= max_corr:
            kept.append(cand)
            col_counts.update(cols)
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
    # target_lag_bars is the reference lag for the binned-forward-return
    # diagnostic only; every candidate is scored by its response curve.
    diag_lag = int(cfg['target_lag_bars'])
    diag_tcol = target_col(diag_lag)
    min_assets = int(cfg['min_assets_per_timestamp'])

    pb = purge_bars(cfg)
    # Compile on train+select only: rolling warmup inside the roll, OOS unseen.
    roll_panel = slice_window(panel, roll.train_start, roll.oos_start, 0)
    roll_panel = roll_panel.reset_index(drop=True)
    train = slice_window(roll_panel, roll.train_start, roll.select_start, pb)
    select = slice_window(roll_panel, roll.select_start, roll.oos_start, pb)
    # Coverage-floor denominator + threshold (see try_candidate).
    train_days_total = int(train['timestamp'].dt.normalize().nunique())
    min_train_coverage = float(search_cfg.get('min_train_coverage', 0.0))
    # Residual matrix for response curves (built once per roll; ends at
    # oos_start, so no path can touch OOS). The curve is computed on the
    # TEST window only - the verdict instrument - and never feeds the
    # reward/breeding loop.
    curve_cfg = cfg.get('curve') or {}
    res_wide = None
    if curve_cfg:
        res_wide = (roll_panel.pivot_table(
            index='timestamp', columns='symbol',
            values='residual_return', aggfunc='first').sort_index())

    from research.signals.data import (all_family_columns,
                                             build_diagnostics)
    # Feature coverage (the upstream junk check): a feature with at most
    # min_feature_nonnan non-NaN values over THIS roll's window is dropped
    # for the roll - never shown to the LLM, never compiled, never scored.
    # Dead inputs (unstarted series, unmapped names) produce no candidates.
    min_nonnan = int(cfg.get('min_feature_nonnan', 0) or 0)
    if min_nonnan > 0:
        present = [c for c in all_family_columns(family_columns)
                   if c in roll_panel.columns]
        counts = roll_panel[present].notna().sum()
        dead = set(counts[counts <= min_nonnan].index)
        dead |= {c for c in all_family_columns(family_columns)
                 if c not in roll_panel.columns}
        if dead:
            family_columns = {f: [c for c in cols if c not in dead]
                              for f, cols in family_columns.items()}
            logging.info(
                f"roll {roll.roll_id}: {len(dead)} features dropped for "
                f"coverage <= {min_nonnan} this window "
                f"({', '.join(sorted(dead)[:6])}{'...' if len(dead) > 6 else ''})")
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

    # Train-side residual matrix ends BEFORE the test window: a train
    # curve's paths must never read a test bar (the reward would see it).
    res_wide_train = (res_wide[res_wide.index < roll.select_start]
                      if res_wide is not None else None)
    H = int(curve_cfg.get('horizon_bars', 0) or 0)
    stride = int(curve_cfg.get('entry_stride_bars', 6) or 6)
    sample_ks = curve_cfg.get('sample_ks')
    rt_cost = (float(curve_cfg.get('roundtrip_mult', 2.0))
               * float(get('portfolio.cost_bps')) / 10000.0)

    def _curve_of(sig_part, matrix) -> Optional[dict]:
        """Compute + fit one response curve; None when the window can't
        host a full path (callers fall back / reject)."""
        rc = response_curve(sig_part, matrix, H, stride, min_assets,
                            sample_ks=sample_ks)
        if rc is None:
            return None
        fit = fit_response_curve(rc['A'], rc['n_eff'], rc['per_entry_at'])
        ks = [int(k) for k in (sample_ks or []) if int(k) <= len(rc['A'])]
        # Whole-window robustness at (nearest sampled k to) the peak; sign-
        # agnostic, so it survives the direction flip unchanged.
        consistent = True
        if rc['per_entry_at']:
            k_near = min(rc['per_entry_at'],
                         key=lambda k: abs(k - fit['peak_k']))
            consistent = thirds_sign_consistent(rc['per_entry_at'][k_near])
        return {**fit, 'entries': rc['entries'],
                'entry_days': rc['entry_days'],
                'n_eff': round(float(rc['n_eff']), 2), 'ks': ks,
                'thirds_consistent': bool(consistent),
                'A': [round(float(rc['A'][k - 1]), 8) for k in ks]}

    def _flip_curve(c: dict) -> dict:
        """A negated signal's curve is exactly the negated curve (the book
        return is linear in the weights)."""
        out = dict(c)
        out['A'] = [None if a is None else -a for a in c['A']]
        # the flipped curve's peak is the old trough; refit on the flipped
        # sampled path is overkill - flip a0/median and re-locate the peak
        # from the flipped samples (deterministic, coarse but honest).
        flipped = [(-a if a is not None else -np.inf) for a in c['A']]
        j = int(np.argmax(flipped))
        out['a0'] = float(flipped[j])
        out['peak_k'] = int(c['ks'][j]) if c.get('ks') else c.get('peak_k')
        for key in ('median_peak',):
            if c.get(key) is not None and np.isfinite(c[key]):
                out[key] = -c[key]
        a_end = flipped[-1] if flipped else 0.0
        out['rev_frac'] = (float(1.0 - a_end / out['a0'])
                           if out['a0'] > 0 else 0.0)
        return out

    def _curve_metrics(c: Optional[dict]) -> dict:
        """Flat ledger columns from a curve (keeps select_*/train_* column
        names meaningful): mean = a0, t = a0/se, n_days = entry days."""
        if not c:
            return empty_metrics()
        se = c.get('se_peak')
        t = (c['a0'] / se) if se and np.isfinite(se) and se > 0 else 0.0
        return {'alpha_mean': c['a0'], 'alpha_tstat': t,
                'n_days': c.get('entry_days', 0)}

    def try_candidate(cand, gen: int) -> None:
        """Validate, compile, measure the TRAIN and TEST response curves,
        reward on the train curve only, record. One instrument everywhere:
        the same curve that promotion judges is what the search breeds for
        (train-side copy - the reward never touches the test window)."""
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

        sig_train = sig[sig['timestamp'] < roll.select_start]
        curve_train_raw = _curve_of(sig_train, res_wide_train)
        if curve_train_raw is None:
            return

        # Coverage floor: gates that leave less than min_train_coverage of
        # the window's days with entries are lottery tickets promotion can
        # never accept - record with the penalty reward (bandit defunds the
        # pattern, failure memory warns the LLM), never into the population.
        cov = curve_train_raw['entry_days'] / max(train_days_total, 1)
        if min_train_coverage > 0 and cov < min_train_coverage:
            sparse = float(search_cfg['sparse_reward'])
            if cand.family in bandit:
                bandit[cand.family]['n'] += 1
                bandit[cand.family]['sum'] += sparse
            failures.append((cand, sparse))
            ledger.record(roll.roll_id, gen, cand, 1,
                          _curve_metrics(curve_train_raw), empty_metrics(),
                          sparse, {}, target_lag=curve_train_raw['peak_k'])
            logging.debug(f"{cand.name}: train coverage {cov:.0%} < "
                          f"{min_train_coverage:.0%} - rejected as sparse")
            return

        # Traded sign: pooled train-curve evidence - this window's a0 plus
        # every prior roll's (train-only; the test is never consulted).
        direction = pooled_train_direction(
            ledger.train_history(cand.hash, up_to_roll=roll.roll_id - 1)
            + [_curve_metrics(curve_train_raw)])
        if direction < 0:
            sig = sig.assign(signal=-sig['signal'])
            sig_train = sig[sig['timestamp'] < roll.select_start]
            curve_train = _flip_curve(curve_train_raw)
        else:
            curve_train = curve_train_raw
        sig_select = sig[sig['timestamp'] >= roll.select_start]

        # Standalone tradeability diagnostic (train signal): per-bar churn.
        turnover = signal_turnover(sig_train)

        # TEST curve: the verdict. Feeds nothing in this loop.
        curve = _curve_of(sig_select, res_wide)

        # Reward = the TRAIN curve's net economic rate at its own optimal
        # holding (the identical formula promotion ranks by) minus
        # similarity to the kept pool.
        rates = [(float(a) - rt_cost) / int(k)
                 for k, a in zip(curve_train['ks'], curve_train['A'])
                 if a is not None and int(k) > 0]
        net_rate = max(rates) if rates else float('-inf')
        similarity = max_signal_correlation(
            sig_train, [s['signal_train'] for s in population])
        rwd, terms = compute_reward(net_rate, similarity, cfg['reward'])

        half_life = float(curve_train['half_life'])
        m_train = _curve_metrics(curve_train)
        m_select = _curve_metrics(curve)
        profile = {}
        for key, c in (('curve', curve), ('curve_train', curve_train)):
            if c is not None:
                profile[key] = {k: (None if isinstance(v, float)
                                    and not np.isfinite(v) else v)
                                for k, v in c.items()}
        ledger.record(roll.roll_id, gen, cand, direction,
                      m_train, m_select, rwd, terms,
                      target_lag=curve_train['peak_k'],
                      profile_json=json.dumps(profile),
                      half_life_bars=half_life, turnover=turnover)
        if cand.family in bandit:
            bandit[cand.family]['n'] += 1
            bandit[cand.family]['sum'] += rwd
        population.append({
            'candidate': cand, 'direction': direction,
            'target_lag': int(curve_train['peak_k']),
            'half_life_bars': half_life,
            'profile_train': {}, 'profile_select': {},
            'curve': curve, 'curve_train': curve_train,
            'reward': rwd, 'metrics_train': m_train,
            'metrics_select': m_select,
            'turnover': turnover,
            'signal_train': sig_train.reset_index(drop=True),
            'signal_select': sig_select.reset_index(drop=True),
        })

    # Failure memory: recently-culled low-reward candidates (and coverage-
    # floor rejects), so the LLM can be told what NOT to re-propose (worst
    # first, capped). Defined before the seeding pass: try_candidate appends.
    failures: List[tuple] = []

    # Generation -1: the previous roll's survivors re-earn their place on the
    # new windows before any fresh proposals are made.
    for cand in (seed_candidates or []):
        try_candidate(cand, -1)
    if seed_candidates:
        logging.debug(f"roll {roll.roll_id}: seeded {len(population)} of "
                     f"{len(seed_candidates)} previous survivors")

    def _parent_scores(pop) -> Dict[str, dict]:
        return {s['candidate'].hash: {
            'reward': round(float(s['reward']), 3),
            'a0_per_bet': round(float(
                s['metrics_train'].get('alpha_mean', 0.0) or 0.0), 6),
            'peak_bars': int(s.get('target_lag', 0) or 0),
            'half_life_bars': int(s.get('half_life_bars', 0) or 0),
        } for s in pop}

    n_overused = int(search_cfg.get('overused_subtrees_shown', 6))
    overused_col_share = float(search_cfg.get('overused_column_share', 0.34))
    max_per_column = int(search_cfg.get('max_survivors_per_column', 0) or 0)
    require_thirds = bool(search_cfg.get('train_sign_thirds', True))
    # API proposal calls within a generation are independent (same
    # parents/diagnostics snapshot), so they run CONCURRENTLY - the LLM
    # round-trips are the roll's wall-clock, not the scoring. Sequential for
    # non-API proposers (RandomProposer: instant, and the shared rng is not
    # thread-safe). Scoring stays sequential: it mutates shared state and its
    # order must be deterministic.
    n_parallel = (int(cfg['llm'].get('parallel_requests', 1))
                  if getattr(proposer, 'provider', '') else 1)

    def propose_generation(alloc, parents, parent_scores, fail_hint,
                           overused, overused_cols) -> List:
        funded = [f for f in families if alloc[f] > 0]
        kw = dict(parent_scores=parent_scores, failures=fail_hint,
                  overused=overused, overused_columns=overused_cols)
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
            # Over-mined structural building blocks: tell the LLM to vary away.
            overused = [_to_list(st)
                        for st, _ in subtree_counts.most_common(n_overused)]
            # Idea-concentration hint: columns a large share of the current
            # survivor pool already leans on (expression + gates).
            col_counts = Counter()
            for s in population:
                col_counts.update(candidate_columns(s['candidate']))
            overused_cols = [c for c, cnt in col_counts.most_common()
                             if cnt / max(len(population), 1)
                             >= overused_col_share]
            batch = propose_generation(alloc, parents, parent_scores,
                                       fail_hint, overused, overused_cols)
            for _, cand in tqdm(batch, desc='  score', unit='cand',
                                leave=False):
                try_candidate(cand, gen)
            pre_cull = population
            # Within-train consistency cut (train-only): a formula whose
            # train profit lived in one burst never enters the survivor
            # pool - it breeds nothing and promotion never sees it.
            eligible = ([s for s in population
                         if s['curve_train'].get('thirds_consistent', True)]
                        if require_thirds else population)
            population = select_survivors(eligible,
                                          int(search_cfg['survivors']),
                                          float(search_cfg['diversity_max_corr']),
                                          max_per_column)
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

