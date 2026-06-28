"""
Portfolio Optimization: shrunk-covariance MVO with neutrality constraints.

Construction (per rebalance):
    maximize   alpha' w - (1/2) w' Sigma w
    subject to A' w = 0          (dollar / market-beta / size-beta neutrality)
               |w_i| <= max_pos
               sum |w_i| = gross

Sigma is the Ledoit-Wolf shrunk covariance of RESIDUAL returns (factors are
hedged by the neutrality constraints, so the residual covariance is the
relevant risk).

Solution: closed-form KKT projection of the unconstrained MVO direction onto
the null space of the equality constraints, then iterative box-clip +
re-projection, then gross normalization. Risk aversion drops out under the
gross-leverage normalization, so no gamma parameter is exposed.

This is deliberately solver-free: it runs at every 10-minute rebalance over a
3-year backtest (~150k solves), where a QP solver would dominate runtime. The
clip/re-project loop is the standard fast approximation; for ~100 names with
a 5% position cap it converges in a few iterations.
"""

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


def shrunk_covariance(returns: pd.DataFrame,
                      min_observations: int = 100,
                      shrinkage='ledoit_wolf') -> pd.DataFrame:
    """
    Shrunk covariance of a (time x asset) return panel.

    Assets with fewer than min_observations valid rows are dropped.
    NaNs are filled with 0 AFTER the validity filter (gap bars contribute
    no covariance rather than fabricated values).

    shrinkage: 'ledoit_wolf' or a float in [0, 1] mixing toward
               diag(sample variances).
    """
    valid_counts = returns.notna().sum()
    keep = valid_counts[valid_counts >= min_observations].index
    if len(keep) == 0:
        return pd.DataFrame()

    X = returns[keep].fillna(0.0).values
    X = X - X.mean(axis=0, keepdims=True)

    if shrinkage == 'ledoit_wolf':
        lw = LedoitWolf(assume_centered=True)
        lw.fit(X)
        cov = lw.covariance_
    else:
        delta = float(shrinkage)
        sample = (X.T @ X) / max(len(X) - 1, 1)
        target = np.diag(np.diag(sample))
        cov = (1 - delta) * sample + delta * target

    # Ridge floor for numerical invertibility
    eps = 1e-12 + 1e-6 * np.trace(cov) / len(keep)
    cov = cov + eps * np.eye(len(keep))

    return pd.DataFrame(cov, index=keep, columns=keep)


def solve_constrained_mvo(alpha: pd.Series,
                          cov: pd.DataFrame,
                          constraints: pd.DataFrame,
                          max_position: float = 0.05,
                          gross_leverage: float = 1.0,
                          n_iter: int = 10) -> pd.Series:
    """
    Solve the constrained MVO described in the module docstring.

    Args:
        alpha: expected (residual) returns per asset
        cov: covariance (assets x assets), must share index with alpha
        constraints: (assets x k) matrix A; solution satisfies A' w = 0.
            Typically columns = [ones, beta_market, beta_size].
        max_position: per-name cap as a fraction of gross leverage
        gross_leverage: target sum |w|
        n_iter: clip / re-project iterations

    Returns:
        Series of weights (0 for assets not in the covariance).
    """
    assets = [a for a in cov.index if a in alpha.index and not np.isnan(alpha[a])]
    if len(assets) == 0:
        return pd.Series(dtype=float)

    S = cov.loc[assets, assets].values
    a = alpha[assets].values.astype(float)
    A = constraints.loc[assets].values.astype(float)  # n x k

    # Drop degenerate constraint columns (e.g. all-NaN betas -> filled 0)
    A = np.nan_to_num(A, nan=0.0)
    col_norms = np.linalg.norm(A, axis=0)
    A = A[:, col_norms > 1e-12]

    try:
        S_inv_a = np.linalg.solve(S, a)
        if A.shape[1] > 0:
            S_inv_A = np.linalg.solve(S, A)              # n x k
            G = A.T @ S_inv_A                            # k x k
            # lstsq, not solve: constraint columns can be collinear (e.g.
            # the ones-vector and beta_market ~ 1 for every asset). The
            # minimum-norm multiplier still projects onto span(A) correctly.
            lam = np.linalg.lstsq(G, A.T @ S_inv_a, rcond=None)[0]
            w = S_inv_a - S_inv_A @ lam
        else:
            w = S_inv_a
    except np.linalg.LinAlgError:
        # Fall back to constraint-projected alpha direction
        w = a - A @ np.linalg.lstsq(A, a, rcond=None)[0] if A.shape[1] else a.copy()

    if np.abs(w).sum() < 1e-15:
        return pd.Series(0.0, index=cov.index)

    # Priority order: equality constraints (exact) > position cap > gross.
    # Clip slightly inside the cap so the final gross rescale cannot push
    # weights back over it.
    cap = max_position * gross_leverage
    cap_eff = cap * 0.999

    for _ in range(n_iter):
        gross = np.abs(w).sum()
        if gross < 1e-15:
            break
        w = w * (gross_leverage / gross)

        clipped = np.clip(w, -cap_eff, cap_eff)
        # Re-impose equality constraints: rank-revealing projection onto
        # col(A) via lstsq (robust to collinear constraint columns)
        if A.shape[1] > 0:
            correction = A @ np.linalg.lstsq(A, clipped, rcond=None)[0]
            w_new = clipped - correction
        else:
            w_new = clipped

        if np.max(np.abs(w_new - w)) < 1e-10:
            w = w_new
            break
        w = w_new

    gross = np.abs(w).sum()
    if gross > 1e-15:
        w = w * (gross_leverage / gross)

    out = pd.Series(0.0, index=cov.index)
    out[assets] = w
    return out


def solve_equal_weight(alpha: pd.Series,
                       constraints: pd.DataFrame,
                       max_position: float = 0.05,
                       gross_leverage: float = 1.0,
                       n_iter: int = 10) -> pd.Series:
    """Covariance-free benchmark book paired against `solve_constrained_mvo`.

    Weights come from the cross-sectional RANK of alpha (centered to zero sum),
    NOT from alpha magnitude or the covariance: a flat (identity) risk model.
    The same equality constraints (A' w = 0), per-name cap and gross target are
    then imposed by the identical clip/re-project loop, so the ONLY difference
    versus the MVO is the risk model. Running both isolates what the
    Ledoit-Wolf covariance weighting adds (or destroys). Rank weighting also
    strips the magnitude outliers a negatively-skewed signal feeds the MVO, so
    a large EW-minus-MVO gap points at cov-driven concentration on those tails.

    `constraints` is the same A passed to the MVO (columns = ones, beta_market,
    beta_size, ...). Returns weights indexed like the MVO solver (0 for assets
    not scored).
    """
    a = alpha.dropna()
    assets = list(a.index)
    if len(assets) == 0:
        return pd.Series(dtype=float)

    A = np.nan_to_num(constraints.loc[assets].values.astype(float), nan=0.0)
    A = A[:, np.linalg.norm(A, axis=0) > 1e-12]

    # Centered cross-sectional rank -> dollar-neutral, magnitude-free direction.
    ranks = pd.Series(a.values, index=assets).rank().values
    w = ranks - ranks.mean()
    if np.abs(w).sum() < 1e-15:
        return pd.Series(0.0, index=alpha.index)

    # Identical priority to the MVO: equality constraints (exact) > cap > gross.
    cap_eff = max_position * gross_leverage * 0.999
    for _ in range(n_iter):
        gross = np.abs(w).sum()
        if gross < 1e-15:
            break
        w = w * (gross_leverage / gross)
        clipped = np.clip(w, -cap_eff, cap_eff)
        if A.shape[1] > 0:
            w_new = clipped - A @ np.linalg.lstsq(A, clipped, rcond=None)[0]
        else:
            w_new = clipped
        if np.max(np.abs(w_new - w)) < 1e-10:
            w = w_new
            break
        w = w_new

    gross = np.abs(w).sum()
    if gross > 1e-15:
        w = w * (gross_leverage / gross)

    out = pd.Series(0.0, index=alpha.index)
    out[assets] = w
    return out


def residual_clusters(returns: pd.DataFrame,
                      corr_threshold: float = 0.30,
                      min_cluster_size: int = 3) -> list:
    """
    Cluster assets by trailing residual correlation (average-linkage
    hierarchical clustering on distance = 1 - corr, cut at
    1 - corr_threshold). Returns a list of clusters (lists of asset names),
    keeping only clusters of at least min_cluster_size.

    Motivated by the Marchenko-Pastur diagnostic: residuals of a market+size
    model retain stable nameable co-movement (e.g. memes). Estimated on the
    same trailing window as the covariance - causal and narrative-adaptive.
    """
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform

    valid = returns.notna().mean() > 0.5
    cols = list(returns.columns[valid])
    if len(cols) < min_cluster_size:
        return []

    X = returns[cols].fillna(0.0)
    std = X.std()
    cols = [c for c in cols if std[c] > 0]
    if len(cols) < min_cluster_size:
        return []
    C = np.corrcoef(X[cols].values, rowvar=False)
    C = np.nan_to_num(C, nan=0.0)
    np.fill_diagonal(C, 1.0)

    dist = squareform(1.0 - C, checks=False)
    Z = linkage(dist, method='average')
    labels = fcluster(Z, t=1.0 - corr_threshold, criterion='distance')

    clusters = []
    for lab in np.unique(labels):
        members = [cols[i] for i in np.flatnonzero(labels == lab)]
        if len(members) >= min_cluster_size:
            clusters.append(members)
    return clusters


def cluster_penalty_matrix(cov: pd.DataFrame, clusters: list,
                           lam: float = 1.0) -> pd.DataFrame:
    """
    Soft cluster-exposure penalty as a covariance augmentation:

        Sigma_eff = Sigma + lam * sum_k (s_k 1_k)(s_k 1_k)'

    where 1_k is the indicator of cluster k and s_k the average per-bar
    residual vol of its members - so a unit-gross cluster bet is penalized
    on the same scale as its members' actual risk. Keeps the MVO closed-form
    untouched: just pass Sigma_eff.
    """
    P = np.zeros((len(cov), len(cov)))
    idx = {a: i for i, a in enumerate(cov.index)}
    vols = np.sqrt(np.diag(cov.values))
    for members in clusters:
        in_cov = [m for m in members if m in idx]
        if len(in_cov) < 2:
            continue
        pos = [idx[m] for m in in_cov]
        s_k = float(np.mean(vols[pos]))
        v = np.zeros(len(cov))
        v[pos] = s_k
        P += np.outer(v, v)
    return cov + lam * pd.DataFrame(P, index=cov.index, columns=cov.columns)


def benjamini_hochberg(pvalues: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """
    Benjamini-Hochberg FDR control. Returns a boolean mask of discoveries.
    NaN p-values are never discoveries.
    """
    p = np.asarray(pvalues, dtype=float)
    mask = np.zeros(len(p), dtype=bool)
    valid = ~np.isnan(p)
    if valid.sum() == 0:
        return mask

    pv = p[valid]
    order = np.argsort(pv)
    m = len(pv)
    thresholds = alpha * (np.arange(1, m + 1) / m)
    below = pv[order] <= thresholds
    if below.any():
        k_max = np.max(np.flatnonzero(below))
        discovered = np.zeros(m, dtype=bool)
        discovered[order[:k_max + 1]] = True
        mask[np.flatnonzero(valid)] = discovered
    return mask


def benjamini_yekutieli(pvalues: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """
    Benjamini-Yekutieli FDR control under arbitrary dependence.

    Generated signal variants and horizon tests are strongly correlated, so
    ordinary BH can be anti-conservative. BY applies BH at alpha / H_m, where
    H_m is the m-th harmonic number.
    """
    p = np.asarray(pvalues, dtype=float)
    m = int(np.isfinite(p).sum())
    if m == 0:
        return np.zeros(len(p), dtype=bool)
    harmonic = float(np.sum(1.0 / np.arange(1, m + 1)))
    return benjamini_hochberg(p, alpha=alpha / harmonic)
