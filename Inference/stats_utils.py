"""
Inference/stats_utils.py
Created May 22, 2026

Reusable statistical utilities used across all metric modules.

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia/
"""

import numpy as np
from typing import List, Optional

N_BOOT    = 10000
N_PERM    = 1000
BOOT_SEED = 0



def bootstrap_proportion(
    values: np.ndarray,
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
) -> dict:
    """
    Percentile bootstrap 95% CI for the mean of a binary (0/1) array.

    Returns: {point, std, ci_lower, ci_upper, n}
        std       = standard deviation of the bootstrap distribution
                    (standard error of the proportion)
    Returns NaN entries when n == 0.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return {"point": np.nan, "std": np.nan, "ci_lower": np.nan,
                "ci_upper": np.nan, "n": 0}
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, n, size=(n_boot, n))
    boot_means = values[idx].mean(axis=1)
    return {
        "point":    float(values.mean()),
        "std":      float(boot_means.std(ddof=1)),
        "ci_lower": float(np.percentile(boot_means, 2.5)),
        "ci_upper": float(np.percentile(boot_means, 97.5)),
        "n":        int(n),
    }




def paired_bootstrap_diff(
    values_a: np.ndarray,
    values_b: np.ndarray,
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
) -> dict:
    """
    Paired bootstrap CI and two-sided p-value for:
        diff = mean(values_a) - mean(values_b)

    Both arrays must be aligned: same cases in the same order.

    p-value uses the shift-and-reflect method:
        - Bootstrap the paired difference to estimate sampling distribution.
        - Shift the bootstrap distribution to be centered at 0 (H0: diff = 0).
        - p = P(|shifted bootstrap diff| >= |observed diff|).

    Returns: {point, std, ci_lower, ci_upper, p_value, n}
    """
    values_a = np.asarray(values_a, dtype=float)
    values_b = np.asarray(values_b, dtype=float)
    n = len(values_a)
    assert len(values_b) == n, "Both arrays must have equal length."
    if n == 0:
        return {"point": np.nan, "std": np.nan, "ci_lower": np.nan,
                "ci_upper": np.nan, "p_value": np.nan, "n": 0}
    rng   = np.random.RandomState(seed)
    idx   = rng.randint(0, n, size=(n_boot, n))
    point = float(values_a.mean() - values_b.mean())
    boot_diffs = values_a[idx].mean(axis=1) - values_b[idx].mean(axis=1)
    ci_lower   = float(np.percentile(boot_diffs, 2.5))
    ci_upper   = float(np.percentile(boot_diffs, 97.5))
    std        = float(boot_diffs.std(ddof=1))
    centered = boot_diffs - point
    p_value  = float(np.mean(np.abs(centered) >= abs(point)))
    p_value  = max(p_value, 1.0 / n_boot)
    return {
        "point":    point,
        "std":      std,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value":  p_value,
        "n":        int(n),
    }



def permutation_test_2groups(
    values_a: np.ndarray,
    values_b: np.ndarray,
    n_perm: int = N_PERM,
    seed: int = BOOT_SEED,
) -> float:
    """
    Two-sided permutation test for H0: mean(values_a) == mean(values_b).

    Test statistic: |mean(a) - mean(b)|.
    Returns p-value. Returns NaN if either group has fewer than 2 observations.
    """
    values_a = np.asarray(values_a, dtype=float)
    values_b = np.asarray(values_b, dtype=float)
    if len(values_a) < 2 or len(values_b) < 2:
        return float("nan")
    obs    = abs(float(values_a.mean() - values_b.mean()))
    pooled = np.concatenate([values_a, values_b])
    n_a    = len(values_a)
    rng    = np.random.RandomState(seed)
    count  = 0
    for _ in range(n_perm):
        perm = rng.permutation(pooled)
        if abs(perm[:n_a].mean() - perm[n_a:].mean()) >= obs:
            count += 1
    return (count + 1) / (n_perm + 1)


def permutation_test_kgroups(
    groups: List[np.ndarray],
    n_perm: int = N_PERM,
    seed: int = BOOT_SEED,
) -> float:
    """
    Permutation F-test for H0: all group means are equal (k >= 2 groups).

    Uses the one-way ANOVA F-statistic as test statistic.
    Groups with fewer than 2 observations are dropped.
    Returns p-value. Returns NaN if fewer than 2 valid groups remain.
    """
    groups = [np.asarray(g, dtype=float) for g in groups if len(g) >= 2]
    if len(groups) < 2:
        return float("nan")
    sizes   = [len(g) for g in groups]
    pooled  = np.concatenate(groups)
    n_total = len(pooled)
    k       = len(groups)

    def _f(gs: List[np.ndarray]) -> float:
        grand      = np.concatenate(gs).mean()
        ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in gs)
        ss_within  = sum(((g - g.mean()) ** 2).sum() for g in gs)
        if ss_within == 0:
            return float("inf")
        return (ss_between / (k - 1)) / (ss_within / (n_total - k))

    obs   = _f(groups)
    rng   = np.random.RandomState(seed)
    count = 0
    for _ in range(n_perm):
        perm   = rng.permutation(pooled)
        start  = 0
        pgs    = []
        for s in sizes:
            pgs.append(perm[start:start + s])
            start += s
        if _f(pgs) >= obs:
            count += 1
    return (count + 1) / (n_perm + 1)



def bh_fdr(p_values: np.ndarray) -> np.ndarray:
    """
    Benjamini-Hochberg step-up FDR correction.
    Returns adjusted p-values (q-values). NaN inputs pass through as NaN.

    Reference: Benjamini & Hochberg (1995), J. Royal Stat. Soc. B, 57(1):289-300.
    """
    p_values = np.asarray(p_values, dtype=float)
    n = len(p_values)
    if n == 0:
        return np.array([])
    nan_mask  = np.isnan(p_values)
    valid_idx = np.where(~nan_mask)[0]
    if len(valid_idx) == 0:
        return p_values.copy()
    valid_p   = p_values[valid_idx]
    m         = len(valid_p)
    order     = np.argsort(valid_p)
    p_sorted  = valid_p[order]
    # BH adjustment: adj[i] = p_sorted[i] * m / (i+1)
    adj = p_sorted * m / (np.arange(m, dtype=float) + 1)
    # Enforce monotonicity via running minimum from the right
    for i in range(m - 2, -1, -1):
        adj[i] = min(adj[i], adj[i + 1])
    adj = np.minimum(adj, 1.0)
    # Map back to original order
    inv_order = np.empty(m, dtype=int)
    inv_order[order] = np.arange(m)
    result = p_values.copy()
    result[valid_idx] = adj[inv_order]
    return result
