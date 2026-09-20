"""
Inference/report_utils.py
Created on September 14, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

from decimal import ROUND_HALF_UP, Decimal
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from Inference.stats_utils import (
    N_BOOT, N_PERM, BOOT_SEED,
    bootstrap_proportion, paired_bootstrap_diff, bootstrap_two_sample_diff,
    permutation_test_2groups, permutation_test_kgroups, bh_fdr,
)


def fmt_pct(x: float) -> float:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return float("nan")
    return float(Decimal(repr(round(float(x) * 100.0, 8))).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def fmt_one(x: float) -> float:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return float("nan")
    return round(float(x), 1)


def keep_p(p: float) -> float:
    if p is None:
        return float("nan")
    return float(p)


def _pack(prefix: str, point, std, lo, hi, n, is_percent: bool) -> Dict:
    f = fmt_pct if is_percent else fmt_one
    return {
        f"{prefix}_mean":     f(point),
        f"{prefix}_std":      f(std),
        f"{prefix}_ci_low":   f(lo),
        f"{prefix}_ci_high":  f(hi),
        f"{prefix}_mean_raw": float(point) if point is not None and np.isfinite(point) else float("nan"),
        f"{prefix}_std_raw":  float(std)   if std   is not None and np.isfinite(std)   else float("nan"),
        f"{prefix}_ci_low_raw":  float(lo) if lo    is not None and np.isfinite(lo)    else float("nan"),
        f"{prefix}_ci_high_raw": float(hi) if hi    is not None and np.isfinite(hi)    else float("nan"),
        "n": int(n) if n is not None and not (isinstance(n, float) and np.isnan(n)) else 0,
    }


def flatten_metric_prefix(row: Dict, *id_keys: str) -> Dict:
    prefix = "".join(f"{row[k]}_" for k in id_keys
                     if row.get(k) not in (None, ""))
    if not prefix:
        return dict(row)
    return {(k[len(prefix):] if k.startswith(prefix) else k): v
            for k, v in row.items()}


def _pack_stat(prefix: str, estimate, n) -> Dict:
    def _r(v):
        return round(float(v), 6) if v is not None and np.isfinite(v) else float("nan")
    return {
        f"{prefix}_value":     _r(estimate),
        f"{prefix}_value_raw": float(estimate) if estimate is not None and np.isfinite(estimate) else float("nan"),
        "n": int(n) if n is not None and not (isinstance(n, float) and np.isnan(n)) else 0,
    }


def _bootstrap_diff_with_p(arrays: List[np.ndarray], stat_fn,
                           n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> Dict:
    arrays = [np.asarray(a, dtype=float) for a in arrays]
    n = len(arrays[0])
    if n == 0:
        return {"point": np.nan, "std": np.nan, "ci_lower": np.nan,
                "ci_upper": np.nan, "p_value": np.nan, "n": 0}
    point = float(stat_fn(*arrays))
    rng  = np.random.RandomState(seed)
    idx  = rng.randint(0, n, size=(n_boot, n))
    boot = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        resampled = [a[idx[b]] for a in arrays]
        try:
            boot[b] = stat_fn(*resampled)
        except Exception:
            boot[b] = np.nan
    finite = boot[np.isfinite(boot)]
    if finite.size == 0:
        return {"point": point, "std": np.nan, "ci_lower": np.nan,
                "ci_upper": np.nan, "p_value": np.nan, "n": int(n)}
    std = float(finite.std(ddof=1)) if finite.size > 1 else np.nan
    ci_lower = float(np.percentile(finite, 2.5))
    ci_upper = float(np.percentile(finite, 97.5))
    centered = finite - point
    p_value  = float(np.mean(np.abs(centered) >= abs(point)))
    p_value  = max(p_value, 1.0 / max(finite.size, 1))
    return {"point": point, "std": std, "ci_lower": ci_lower,
            "ci_upper": ci_upper, "p_value": p_value, "n": int(n)}


def _anova_f_stat(groups: List[np.ndarray]) -> float:
    groups = [np.asarray(g, dtype=float) for g in groups if len(g) >= 2]
    if len(groups) < 2:
        return float("nan")
    k = len(groups)
    pooled  = np.concatenate(groups)
    n_total = len(pooled)
    grand   = pooled.mean()
    ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ss_within  = sum(((g - g.mean()) ** 2).sum() for g in groups)
    if ss_within == 0:
        return float("inf")
    return (ss_between / (k - 1)) / (ss_within / (n_total - k))


def report_metric(
    values: np.ndarray,
    prefix: str = "value",
    is_percent: bool = True,
    cluster_ids: Optional[np.ndarray] = None,
    n_boot: int = N_BOOT,
) -> Dict:
    res = bootstrap_proportion(np.asarray(values, dtype=float),
                               cluster_ids=cluster_ids, n_boot=n_boot)
    return _pack(prefix, res["point"], res["std"], res["ci_lower"],
                 res["ci_upper"], res["n"], is_percent)


def report_paired_diff(
    a: np.ndarray,
    b: np.ndarray,
    prefix: str = "diff",
    is_percent: bool = True,
    cluster_ids: Optional[np.ndarray] = None,
    n_boot: int = N_BOOT,
) -> Dict:
    res = paired_bootstrap_diff(np.asarray(a, float), np.asarray(b, float),
                                cluster_ids=cluster_ids, n_boot=n_boot)
    out = _pack(prefix, res["point"], res["std"], res["ci_lower"],
                res["ci_upper"], res["n"], is_percent)
    out["p_raw"] = keep_p(res["p_value"])
    out["p_fdr"] = float("nan")
    return out


def report_permutation_2(
    a: np.ndarray,
    b: np.ndarray,
    prefix: str = "diff",
    is_percent: bool = True,
    n_boot: int = N_BOOT,
    n_perm: int = N_PERM,
) -> Dict:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    boot = bootstrap_two_sample_diff(a, b, n_boot=n_boot)
    p    = permutation_test_2groups(a, b, n_perm=n_perm)
    out  = _pack(prefix, boot["point"], boot["std"], boot["ci_lower"],
                boot["ci_upper"], boot["n_a"] + boot["n_b"], is_percent)
    f = fmt_pct if is_percent else fmt_one
    out[f"{prefix}_group_a_mean"] = f(np.nanmean(a)) if len(a) else float("nan")
    out[f"{prefix}_group_b_mean"] = f(np.nanmean(b)) if len(b) else float("nan")
    out["n_a"] = int(len(a))
    out["n_b"] = int(len(b))
    out["p_raw"] = keep_p(p)
    out["p_fdr"] = float("nan")
    return out


def report_permutation_k(
    groups: List[np.ndarray],
    prefix: str = "anova",
    n_perm: int = N_PERM,
) -> Dict:
    f_stat = _anova_f_stat(groups)
    p = permutation_test_kgroups(groups, n_perm=n_perm)
    k_valid = len([g for g in groups if len(g) >= 2])
    n_total = sum(len(g) for g in groups)
    out = _pack_stat(prefix, f_stat, n_total)
    out["k_groups"] = k_valid
    out["p_raw"] = keep_p(p)
    out["p_fdr"] = float("nan")
    return out


def report_spearman(x: np.ndarray, y: np.ndarray, prefix: str = "spearman") -> Dict:
    from scipy.stats import spearmanr
    x = np.asarray(x, float); y = np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 4:
        out = _pack_stat(prefix, np.nan, len(x))
        out["p_raw"] = float("nan"); out["p_fdr"] = float("nan")
        return out
    rho, p = spearmanr(x, y)
    out = _pack_stat(prefix, rho, len(x))
    out["p_raw"] = keep_p(p)
    out["p_fdr"] = float("nan")
    return out


def report_partial_spearman(
    x: np.ndarray, y: np.ndarray, z: np.ndarray,
    prefix: str = "partial_spearman",
) -> Dict:
    from scipy.stats import spearmanr
    from sklearn.linear_model import LinearRegression
    x = np.asarray(x, float); y = np.asarray(y, float); z = np.asarray(z, float)
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[mask], y[mask], z[mask]
    if len(x) < 5:
        out = _pack_stat(prefix, np.nan, len(x))
        out["p_raw"] = float("nan"); out["p_fdr"] = float("nan")
        return out

    def _resid(a, b):
        lr = LinearRegression().fit(b.reshape(-1, 1), a)
        return a - lr.predict(b.reshape(-1, 1))

    rx = _resid(x, z); ry = _resid(y, z)
    rho, p = spearmanr(rx, ry)
    out = _pack_stat(prefix, rho, len(x))
    out["p_raw"] = keep_p(p)
    out["p_fdr"] = float("nan")
    return out


def report_slope(x: np.ndarray, y: np.ndarray, prefix: str = "slope") -> Dict:
    from scipy.stats import linregress
    x = np.asarray(x, float); y = np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        out = _pack_stat(prefix, np.nan, len(x))
        out["p_raw"] = float("nan"); out["r_squared"] = float("nan")
        out["intercept_raw"] = float("nan"); out["p_fdr"] = float("nan")
        return out
    lr = linregress(x, y)
    out = _pack_stat(prefix, lr.slope, len(x))
    out["p_raw"] = keep_p(lr.pvalue)
    out["r_squared"] = round(float(lr.rvalue ** 2), 4)
    out["intercept_raw"] = float(lr.intercept)
    out["p_fdr"] = float("nan")
    return out


def report_mantel(
    d1: np.ndarray, d2: np.ndarray,
    prefix: str = "mantel", n_perm: int = N_PERM, seed: int = BOOT_SEED,
) -> Dict:
    from scipy.stats import spearmanr
    d1 = np.asarray(d1, float); d2 = np.asarray(d2, float)
    mask = np.isfinite(d1) & np.isfinite(d2)
    d1, d2 = d1[mask], d2[mask]
    if len(d1) < 4:
        out = _pack_stat(prefix, np.nan, len(d1))
        out["p_raw"] = float("nan"); out["p_fdr"] = float("nan")
        return out
    obs = float(spearmanr(d1, d2).correlation)
    rng = np.random.RandomState(seed)
    count = 0
    for _ in range(n_perm):
        if abs(float(spearmanr(d1, rng.permutation(d2)).correlation)) >= abs(obs):
            count += 1
    out = _pack_stat(prefix, obs, len(d1))
    out["p_raw"] = keep_p((count + 1) / (n_perm + 1))
    out["p_fdr"] = float("nan")
    return out


def report_auroc(
    y_true: np.ndarray, y_score: np.ndarray,
    prefix: str = "auroc", cluster_ids: Optional[np.ndarray] = None,
    n_boot: int = N_BOOT,
) -> Dict:
    from Inference.stats_utils import bootstrap_auroc
    res = bootstrap_auroc(np.asarray(y_true, float), np.asarray(y_score, float),
                          cluster_ids=cluster_ids, n_boot=n_boot)
    return _pack(prefix, res["point"], res["std"], res["ci_lower"],
                 res["ci_upper"], res["n"], is_percent=True)


def report_average_precision(
    y_true: np.ndarray, y_score: np.ndarray,
    prefix: str = "ap", cluster_ids: Optional[np.ndarray] = None,
    n_boot: int = N_BOOT,
) -> Dict:
    from Inference.stats_utils import bootstrap_average_precision
    res = bootstrap_average_precision(np.asarray(y_true, float), np.asarray(y_score, float),
                                      cluster_ids=cluster_ids, n_boot=n_boot)
    return _pack(prefix, res["point"], res["std"], res["ci_lower"],
                 res["ci_upper"], res["n"], is_percent=True)


def report_paired_auroc_diff(
    y_true: np.ndarray, score_a: np.ndarray, score_b: np.ndarray,
    prefix: str = "auroc_diff", cluster_ids: Optional[np.ndarray] = None,
    n_boot: int = N_BOOT,
) -> Dict:
    from Inference.stats_utils import bootstrap_auroc_diff
    y_true  = np.asarray(y_true, float)
    score_a = np.asarray(score_a, float)
    score_b = np.asarray(score_b, float)
    mask = np.isfinite(y_true) & np.isfinite(score_a) & np.isfinite(score_b)
    y_true, score_a, score_b = y_true[mask], score_a[mask], score_b[mask]
    cids = np.asarray(cluster_ids)[mask] if cluster_ids is not None else None
    n = len(y_true)
    if n < 2 or len(np.unique(y_true)) < 2:
        out = _pack(prefix, np.nan, np.nan, np.nan, np.nan, n, is_percent=True)
        out["p_raw"] = float("nan"); out["p_fdr"] = float("nan")
        return out

    res = bootstrap_auroc_diff(y_true, score_a, score_b, cluster_ids=cids, n_boot=n_boot)
    out = _pack(prefix, res["point"], res["std"], res["ci_lower"],
                res["ci_upper"], res["n"], is_percent=True)
    out["p_raw"] = keep_p(res["p_value"])
    out["p_fdr"] = float("nan")
    return out


def report_jaccard_overlap(
    mask_a: np.ndarray, mask_b: np.ndarray, prefix: str = "jaccard",
) -> Dict:
    from scipy.stats import hypergeom
    mask_a = np.asarray(mask_a, dtype=bool)
    mask_b = np.asarray(mask_b, dtype=bool)
    n = len(mask_a)
    k_a = int(mask_a.sum())
    k_b = int(mask_b.sum())
    k_obs = int((mask_a & mask_b).sum())
    union = k_a + k_b - k_obs
    jaccard_obs = k_obs / union if union > 0 else float("nan")

    if n == 0 or k_a == 0 or k_b == 0:
        out = _pack_stat(prefix, jaccard_obs, n)
        out[f"{prefix}_null_value"] = float("nan")
        out[f"{prefix}_excess"] = float("nan")
        out["p_raw"] = float("nan")
        out["p_fdr"] = float("nan")
        return out

    null_mean_intersection = k_a * k_b / n
    null_union = k_a + k_b - null_mean_intersection
    jaccard_null = null_mean_intersection / null_union if null_union > 0 else float("nan")
    p_upper = float(hypergeom.sf(k_obs - 1, n, k_a, k_b))

    out = _pack_stat(prefix, jaccard_obs, n)
    out[f"{prefix}_null_value"] = round(jaccard_null, 6) if np.isfinite(jaccard_null) else float("nan")
    out[f"{prefix}_excess"] = round(jaccard_obs - jaccard_null, 6) \
        if np.isfinite(jaccard_obs) and np.isfinite(jaccard_null) else float("nan")
    out["p_raw"] = keep_p(p_upper)
    out["p_fdr"] = float("nan")
    return out


def add_fdr(
    df: pd.DataFrame,
    family_cols: Optional[List[str]] = None,
    p_col: str = "p_raw",
    out_col: str = "p_fdr",
) -> pd.DataFrame:
    df = df.copy()
    if p_col not in df.columns:
        return df
    if family_cols:
        for _, idx in df.groupby(family_cols).groups.items():
            sub = df.loc[idx, p_col].values
            df.loc[idx, out_col] = bh_fdr(sub)
    else:
        df[out_col] = bh_fdr(df[p_col].values)
    df["significant_fdr05"] = df[out_col].apply(
        lambda q: bool(q < 0.05) if pd.notna(q) else False
    )
    return df
