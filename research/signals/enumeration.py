"""
INVENT's deterministic lane: sweep the pair-product template over EVERY
feature pair, screen cheaply on the train window, and hand the top N to the
normal search as seed candidates (they then get full-resolution curves,
dedup, and promotion exactly like any LLM proposal - MEASURE and CHOOSE are
untouched).

Why: the LLM samples ~250 formulas per roll from a pair space of ~16,000,
and converges to one template anyway (rank(A) x rank(B) + gate). This lane
measures the WHOLE template space systematically at ~zero token cost; the
LLM's budget stays pointed at structures enumeration can't reach.

The screen (train-only, purged - the test window is never touched):
  - hourly grid (every agg_bars-th bar) of cross-sectionally z-scored
    feature panels X_t (features x symbols) and demeaned forward residual
    steps Y_j[t] (the return over hourly step j after t);
  - the pair book's unnormalized response is a trilinear product, batched
    as one matrix identity per step: C_j = sum_t (X_t * Y_j[t]) X_t' gives
    ALL pairs' step-j responses at once (F x F);
  - per-bet scale from G = mean_t |X_t||X_t|' (the pair book's gross);
  - screen score = the same net-rate formula everything else uses,
    max_k (A(k) - roundtrip) / k, best of the two signs (direction is
    fitted downstream, the screen just must not miss inverted edges).

Approximation vs the real curve: no per-bar gross-1 renormalization and a
coarse entry/step grid - fine for ORDERING candidates; the survivors are
re-measured by the full instrument immediately after.
"""

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

from config import get
from research.signals.data import purge_bars, slice_window
from research.signals.generation import Candidate


def enumerate_candidates(panel: pd.DataFrame, roll, family_columns: Dict,
                         cfg: dict) -> List[Candidate]:
    e = cfg.get('enumeration') or {}
    if not e.get('enabled', False):
        return []
    agg = int(e.get('agg_bars', 6))
    H = int(e.get('horizon_steps', 24))
    top_n = int(e.get('top_n', 50))
    min_assets = int(cfg['min_assets_per_timestamp'])
    rt = (float(cfg['curve'].get('roundtrip_mult', 2.0))
          * float(get('portfolio.cost_bps')) / 10000.0)

    fam_of = {c: f for f, cs in family_columns.items() for c in cs}
    cols = sorted(c for c in fam_of if c in panel.columns)
    train = slice_window(panel, roll.train_start, roll.select_start,
                         purge_bars(cfg))
    if train.empty or not cols:
        return []

    res = train.pivot_table(index='timestamp', columns='symbol',
                            values='residual_return',
                            aggfunc='first').sort_index()
    T_all, syms = len(res.index), res.columns
    grid = np.arange(0, T_all - H * agg, agg)      # full forward path only
    if len(grid) < 50:
        return []

    # X: (T, F, N) z-scored features on the grid; rows with a thin
    # cross-section are zeroed (they contribute nothing to any pair).
    feats = np.empty((len(grid), len(cols), len(syms)), dtype=np.float32)
    for fi, c in enumerate(cols):
        w = train.pivot_table(index='timestamp', columns='symbol', values=c,
                              aggfunc='first').reindex(index=res.index,
                                                       columns=syms)
        v = w.to_numpy(dtype=np.float32)[grid]
        mu = np.nanmean(v, axis=1, keepdims=True)
        sd = np.nanstd(v, axis=1, keepdims=True)
        z = np.clip((v - mu) / np.where(sd > 0, sd, np.nan), -3, 3)
        n_ok = np.isfinite(z).sum(axis=1, keepdims=True)
        feats[:, fi, :] = np.where(np.isfinite(z) & (n_ok >= min_assets),
                                   z, 0.0)

    # Forward residual steps, cross-sectionally demeaned (dollar
    # neutrality: sum_i (w-mean(w)) r = sum_i w (r-mean(r))). The book
    # response is LINEAR in the weights, so one pass yields four template
    # families: pair products (the trilinear identity), and singles /
    # sums / differences (all derived from the single-feature responses).
    Cres = np.nan_to_num(res.to_numpy(dtype=np.float32)).cumsum(axis=0)
    F, T, N = len(cols), len(grid), len(syms)
    cum = np.zeros((H, F, F))                         # pair-product response
    s_cum = np.zeros((H, F))                          # single response
    for j in range(1, H + 1):
        Y = (Cres[grid + j * agg]
             - Cres[grid + (j - 1) * agg]).astype(np.float32)
        Y = Y - Y.mean(axis=1, keepdims=True)
        C = np.einsum('tfn,tn,tgn->fg', feats, Y, feats, optimize=True)
        s = np.einsum('tfn,tn->f', feats, Y, optimize=True)
        cum[j - 1] = C if j == 1 else cum[j - 2] + C
        s_cum[j - 1] = s if j == 1 else s_cum[j - 2] + s

    ks = (np.arange(1, H + 1) * agg)[:, None, None]

    def rate(A):
        """Best net rate over holdings and both signs (direction is fitted
        downstream; the screen must not miss inverted edges)."""
        while A.ndim < 3:
            A = A[..., None]
        return np.maximum((A - rt) / ks, (-A - rt) / ks).max(axis=0)

    # Per-bet scales (mean book gross): exact for pairs/singles, gaussian
    # approximation for sums/differences (zA -+ zB ~ N(0, 2(1 -+ rho))).
    G_pair = np.einsum('tfn,tgn->fg', np.abs(feats), np.abs(feats),
                       optimize=True) / T
    G_one = np.abs(feats).sum(axis=(0, 2)) / T
    rho = np.clip(np.einsum('tfn,tgn->fg', feats, feats,
                            optimize=True) / (T * N), -1.0, 1.0)
    scale = N * np.sqrt(2.0 / np.pi)
    iu = np.triu_indices(F, k=1)                      # each pair once
    scored = [
        ('x', rate(cum / np.maximum(G_pair * T, 1e-12))[iu]),
        ('minus', rate((s_cum[:, :, None] - s_cum[:, None, :])
                       / np.maximum(scale * np.sqrt(2 * (1 - rho)) * T,
                                    1e-12))[iu]),
        ('plus', rate((s_cum[:, :, None] + s_cum[:, None, :])
                      / np.maximum(scale * np.sqrt(2 * (1 + rho)) * T,
                                   1e-12))[iu]),
        ('one', rate(s_cum / np.maximum(G_one * T, 1e-12))[:, 0]),
    ]
    OPS = {'x': 'mul', 'minus': 'sub', 'plus': 'add'}
    ranked = sorted(
        [(sc[i], tpl, i) for tpl, sc in scored for i in range(len(sc))],
        key=lambda x: -x[0])[:top_n]
    out = []
    for _, tpl, i in ranked:
        if tpl == 'one':
            a = cols[i]
            expr = ('cs_zscore', ('col', a))
            name = f'enum_{a}'
        else:
            a, b = cols[iu[0][i]], cols[iu[1][i]]
            expr = (OPS[tpl], ('cs_zscore', ('col', a)),
                    ('cs_zscore', ('col', b)))
            name = f'enum_{a}_{tpl}_{b}'
        out.append(Candidate(name=name[:60], family=fam_of[a],
                             expression=expr))
    logging.info(f"roll {roll.roll_id}: enumeration screened {F:,} singles "
                 f"+ 3x{len(iu[0]):,} pair templates -> seeding "
                 f"top {len(out)}")
    return out
