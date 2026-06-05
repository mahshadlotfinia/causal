"""
Inference/subgroup_tests.py
Created May 22, 2026

Tests whether CGR and UAR differ significantly across demographic and
technical subgroups (gender, age, view).

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia/
"""

import numpy as np
import pandas as pd
from typing import Dict

from Inference.metrics import MetricsComputer
from Inference.stats_utils import (
    permutation_test_2groups,
    permutation_test_kgroups,
    bh_fdr,
    N_PERM,
)



def _cgr_outcomes_subgroup(df: pd.DataFrame) -> np.ndarray:
    return MetricsComputer._cgr_outcomes(df)


def _uar_outcomes_subgroup(df: pd.DataFrame) -> np.ndarray:
    return MetricsComputer._uar_outcomes(df)


def _age_bins(df: pd.DataFrame):
    """Returns list of arrays, one per non-empty age bin."""
    bins   = [0, 50, 70, 200]
    labels = ["<50", "50-70", ">70"]
    sub    = df[df["age"].notna()].copy()
    if len(sub) == 0:
        return [], []
    sub["age_group"] = pd.cut(sub["age"], bins=bins, labels=labels, right=False)
    groups = []
    used_labels = []
    for grp in labels:
        g = sub[sub["age_group"] == grp]
        if len(g) >= 2:
            groups.append(g)
            used_labels.append(grp)
    return groups, used_labels



def _run_tests_for_model(
    model_name: str,
    df: pd.DataFrame,
    n_perm: int = N_PERM,
) -> list:
    """
    Runs all 5 subgroup tests for one model.
    Returns a list of dicts (one per test), before FDR correction.
    """
    rows = []

    for g in ["M", "F"]:
        pass  # just to have consistent variable
    gm = _cgr_outcomes_subgroup(df[df["gender"] == "M"]) if "gender" in df.columns else np.array([])
    gf = _cgr_outcomes_subgroup(df[df["gender"] == "F"]) if "gender" in df.columns else np.array([])
    p_gender_cgr = permutation_test_2groups(gm, gf, n_perm=n_perm)
    rows.append({
        "model": model_name,
        "test":  "gender_CGR",
        "groups": f"M(n={len(gm)}),F(n={len(gf)})",
        "group_means": f"M={gm.mean():.3f},F={gf.mean():.3f}"
                       if len(gm) > 0 and len(gf) > 0 else "NA",
        "p_raw": p_gender_cgr,
    })

    um = _uar_outcomes_subgroup(df[df["gender"] == "M"]) if "gender" in df.columns else np.array([])
    uf = _uar_outcomes_subgroup(df[df["gender"] == "F"]) if "gender" in df.columns else np.array([])
    p_gender_uar = permutation_test_2groups(um, uf, n_perm=n_perm)
    rows.append({
        "model": model_name,
        "test":  "gender_UAR",
        "groups": f"M(n={len(um)}),F(n={len(uf)})",
        "group_means": f"M={um.mean():.3f},F={uf.mean():.3f}"
                       if len(um) > 0 and len(uf) > 0 else "NA",
        "p_raw": p_gender_uar,
    })

    vpa = _cgr_outcomes_subgroup(df[df["view"] == "PA"]) if "view" in df.columns else np.array([])
    vap = _cgr_outcomes_subgroup(df[df["view"] == "AP"]) if "view" in df.columns else np.array([])
    p_view_cgr = permutation_test_2groups(vpa, vap, n_perm=n_perm)
    rows.append({
        "model": model_name,
        "test":  "view_CGR",
        "groups": f"PA(n={len(vpa)}),AP(n={len(vap)})",
        "group_means": f"PA={vpa.mean():.3f},AP={vap.mean():.3f}"
                       if len(vpa) > 0 and len(vap) > 0 else "NA",
        "p_raw": p_view_cgr,
    })

    age_groups_df, age_labels = _age_bins(df) if "age" in df.columns else ([], [])
    if len(age_groups_df) >= 2:
        age_cgr_arrays = [_cgr_outcomes_subgroup(g) for g in age_groups_df]
        p_age_cgr = permutation_test_kgroups(age_cgr_arrays, n_perm=n_perm)
        age_means = ",".join(
            f"{lbl}={arr.mean():.3f}(n={len(arr)})"
            for lbl, arr in zip(age_labels, age_cgr_arrays)
        )
    else:
        p_age_cgr = float("nan")
        age_means = "NA"
    rows.append({
        "model": model_name,
        "test":  "age_CGR",
        "groups": ",".join(f"{l}(n={len(_cgr_outcomes_subgroup(g))})"
                           for l, g in zip(age_labels, age_groups_df))
                  if age_groups_df else "NA",
        "group_means": age_means,
        "p_raw": p_age_cgr,
    })

    if len(age_groups_df) >= 2:
        age_uar_arrays = [_uar_outcomes_subgroup(g) for g in age_groups_df]
        p_age_uar = permutation_test_kgroups(age_uar_arrays, n_perm=n_perm)
        age_uar_means = ",".join(
            f"{lbl}={arr.mean():.3f}(n={len(arr)})"
            for lbl, arr in zip(age_labels, age_uar_arrays)
        )
    else:
        p_age_uar = float("nan")
        age_uar_means = "NA"
    rows.append({
        "model": model_name,
        "test":  "age_UAR",
        "groups": ",".join(f"{l}(n={len(_uar_outcomes_subgroup(g))})"
                           for l, g in zip(age_labels, age_groups_df))
                  if age_groups_df else "NA",
        "group_means": age_uar_means,
        "p_raw": p_age_uar,
    })

    return rows



def compute_subgroup_tests(
    all_results: Dict[str, pd.DataFrame],
    manifest: pd.DataFrame,
    n_perm: int = N_PERM,
) -> pd.DataFrame:
    """
    Runs all subgroup tests for all models and applies BH FDR correction
    within each model across its 5 tests.

    Args:
        all_results : dict[model_name -> merged wide DataFrame]
        manifest    : probe set manifest (for joining demographic columns if
                      they are not already in the merged DataFrames)
        n_perm      : permutation resamples

    Returns pd.DataFrame with columns:
        model, test, groups, group_means, p_raw, p_adj, significant_fdr05
    """
    all_rows = []

    for model_name, df in all_results.items():
        rows = _run_tests_for_model(model_name, df, n_perm=n_perm)

        # Apply BH FDR within this model's test family
        p_raw = np.array([r["p_raw"] for r in rows])
        p_adj = bh_fdr(p_raw)

        for r, adj in zip(rows, p_adj):
            r["p_adj"] = float(adj) if not np.isnan(adj) else float("nan")
            r["significant_fdr05"] = bool(adj < 0.05) if not np.isnan(adj) else False
            all_rows.append(r)

    if not all_rows:
        return pd.DataFrame()

    out = pd.DataFrame(all_rows)
    out["p_raw"] = out["p_raw"].apply(
        lambda x: float(x) if not np.isnan(x) else float("nan")
    )

    # Print models with any significant subgroup effect after FDR correction
    sig = out[out["significant_fdr05"]]
    if len(sig) == 0:
        print("[SubgroupTests] No significant subgroup effects after BH FDR correction.")
    else:
        print("\n=== Significant Subgroup Effects (FDR q < 0.05) ===")
        print(sig[["model", "test", "group_means", "p_raw", "p_adj"]].to_string(index=False))

    return out.reset_index(drop=True)