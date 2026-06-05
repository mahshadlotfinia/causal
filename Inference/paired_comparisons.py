"""
Inference/paired_comparisons.py
Created May 22, 2026

Paired bootstrap comparisons for accuracy and UAR.

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia/
"""

import numpy as np
import pandas as pd
from typing import Dict

from Inference.stats_utils import paired_bootstrap_diff, bh_fdr, N_BOOT
from Inference.metrics import TEXT_ONLY_MODELS, VISION_ONLY_MODELS


def _outcome_series(df: pd.DataFrame, kind: str) -> pd.Series:
    """
    Binary outcome Series indexed by case_id.

    kind = 'accuracy': 1 if model correct, 0 otherwise
    kind = 'UAR'     : among correct cases, 1 if answer unchanged under swap
    """
    sub = df[
        df["pa_original"].isin([0, 1]) &
        df["ground_truth"].isin([0, 1])
    ].copy()

    if kind == "accuracy":
        sub["_out"] = (sub["pa_original"] == sub["ground_truth"]).astype(float)
        return sub.drop_duplicates("case_id").set_index("case_id")["_out"]

    if kind == "UAR":
        if "pa_swap" not in df.columns:
            return pd.Series(dtype=float)
        sub = sub[
            (sub["pa_original"] == sub["ground_truth"]) &
            sub["pa_swap"].isin([0, 1])
        ]
        sub["_out"] = (sub["pa_swap"] == sub["pa_original"]).astype(float)
        return sub.drop_duplicates("case_id").set_index("case_id")["_out"]

    return pd.Series(dtype=float)


def _paired_row(
    comparison_type: str,
    model_a: str,
    model_b: str,
    metric: str,
    series_a: pd.Series,
    series_b: pd.Series,
    n_boot: int,
) -> dict | None:
    """
    Paired bootstrap comparison between series_a and series_b on shared
    case_ids. Returns None if fewer than 10 shared cases.
    P-values stored at full float precision.
    """
    shared = series_a.index.intersection(series_b.index)
    if len(shared) < 10:
        return None
    a = series_a.loc[shared].values.astype(float)
    b = series_b.loc[shared].values.astype(float)
    res = paired_bootstrap_diff(a, b, n_boot=n_boot)
    return {
        "comparison_type": comparison_type,
        "model_a":         model_a,
        "model_b":         model_b,
        "metric":          metric,
        "value_a":         float(a.mean()),
        "value_b":         float(b.mean()),
        "diff":            res["point"],
        "diff_std":        res["std"],
        "diff_ci_lower":   res["ci_lower"],
        "diff_ci_upper":   res["ci_upper"],
        "p_value":         res["p_value"],
        "p_value_fdr":     float("nan"),
        "n_shared":        res["n"],
    }


def compute_paired_comparisons(
    all_results: Dict[str, pd.DataFrame],
    n_boot: int = N_BOOT,
) -> pd.DataFrame:
    """
    Args:
        all_results : dict[model_name -> merged wide DataFrame]
        n_boot      : bootstrap resamples

    Returns pd.DataFrame with one row per comparison, including both
    raw and FDR-adjusted p-values.
    """
    model_names = sorted(all_results.keys())
    text_only   = [m for m in model_names if m in TEXT_ONLY_MODELS]

    if not text_only:
        print("[PairedComparisons] No text-only models found. Skipping.")
        return pd.DataFrame()

    rows = []


    for baseline in text_only:
        baseline_acc = _outcome_series(all_results[baseline], "accuracy")
        baseline_uar = _outcome_series(all_results[baseline], "UAR")

        for model_name in model_names:
            # accuracy: all models including text-only (self-comparison = sanity check)
            s_acc = _outcome_series(all_results[model_name], "accuracy")
            row = _paired_row("vs_text_baseline", model_name, baseline,
                              "accuracy", s_acc, baseline_acc, n_boot)
            if row:
                rows.append(row)

            # UAR: skip vision-only (no pa_swap)
            if model_name in VISION_ONLY_MODELS:
                continue
            s_uar = _outcome_series(all_results[model_name], "UAR")
            if len(s_uar) == 0 or len(baseline_uar) == 0:
                continue
            row = _paired_row("vs_text_baseline", model_name, baseline,
                              "UAR", s_uar, baseline_uar, n_boot)
            if row:
                rows.append(row)


    for i, m_a in enumerate(model_names):
        for m_b in model_names[i + 1:]:
            s_a = _outcome_series(all_results[m_a], "accuracy")
            s_b = _outcome_series(all_results[m_b], "accuracy")
            row = _paired_row("pairwise_accuracy", m_a, m_b,
                              "accuracy", s_a, s_b, n_boot)
            if row:
                rows.append(row)

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)


    for (ctype, metric_name), grp in out.groupby(["comparison_type", "metric"]):
        adj = bh_fdr(grp["p_value"].values)
        out.loc[grp.index, "p_value_fdr"] = adj

    out = out.sort_values(
        ["comparison_type", "metric", "diff"], ascending=[True, True, False]
    ).reset_index(drop=True)

    summary = out[
        (out["comparison_type"] == "vs_text_baseline") &
        (out["metric"] == "accuracy")
    ][["model_a", "model_b", "diff", "diff_ci_lower", "diff_ci_upper",
       "p_value", "p_value_fdr"]]
    print("\n=== Accuracy vs. Text-Only Baseline ===")
    print(summary.to_string(index=False))

    return out
