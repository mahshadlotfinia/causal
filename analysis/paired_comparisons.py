"""
analysis/paired_comparisons.py
Created on September 14, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from analysis.compute_metrics import record_store_fingerprint, subgroup_masks
from analysis.metrics_core import (acc_outcomes, answered_pairs, balanced_accuracy, bootstrap_statistic, cgr_outcomes,
                                   change_outcomes, paired_diff_with_ci90, same_outcomes, tost_equivalent, wide_frame)
from config.serde import read_config
from data_loader.probe_dataset import load_manifest
from Inference.report_utils import _pack, add_fdr, fmt_pct, report_permutation_2, report_permutation_k
from Inference.resume_utils import MissingInput, append_status, ensure_dir, status_path, write_csv_atomic


ACC_SCOPES = {"mimic": ["pooled", "finding_presence", "image_necessary", "mimic_cxr", "ms_cxr", "rexerr"], "chexpert": ["pooled"]}
SWAP_SCOPE = {"mimic": "finding_presence", "chexpert": "pooled"}
PAIRWISE_SCOPE = {"mimic": "image_necessary", "chexpert": "pooled"}


def _frames(cfg: Dict, dataset: str) -> Dict[str, pd.DataFrame]:
    manifest = load_manifest(cfg, dataset)
    out = {}
    for model in cfg["model_order"]:
        try:
            out[model] = wide_frame(cfg, dataset, model, manifest=manifest, verbose=False)
        except MissingInput:
            continue
    return out


def _controls(cfg: Dict) -> List[str]:
    return [m for m in cfg["model_order"] if cfg["models"][m]["modality"] == "text_only"]


def _shared(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    m = a.merge(b, on="case_id", suffixes=("_a", "_b"))
    return m


def _diff_row(dataset: str, family: str, comparison: str, model_a: str, model_b: str, metric: str, scope: str,
              res: Dict, prefix: str, value_a: float, value_b: float, n_clusters: int) -> Dict:
    return {"dataset": dataset, "family": family, "comparison_type": comparison, "model_a": model_a, "model_b": model_b,
            "metric": metric, "scope": scope, "value_a_pct": fmt_pct(value_a), "value_b_pct": fmt_pct(value_b),
            "diff_pct": res[f"{prefix}_mean"], "diff_std_pct": res[f"{prefix}_std"], "diff_ci_low_pct": res[f"{prefix}_ci_low"],
            "diff_ci_high_pct": res[f"{prefix}_ci_high"], "diff_raw": res[f"{prefix}_mean_raw"], "diff_std_raw": res[f"{prefix}_std_raw"],
            "diff_ci_low_raw": res[f"{prefix}_ci_low_raw"], "diff_ci_high_raw": res[f"{prefix}_ci_high_raw"],
            "diff_ci90_low_raw": res.get(f"{prefix}_ci90_low_raw", np.nan), "diff_ci90_high_raw": res.get(f"{prefix}_ci90_high_raw", np.nan),
            "n_shared": res["n"], "n_clusters": int(n_clusters), "p_raw": res.get("p_raw", np.nan)}


def accuracy_contrast(dataset: str, family: str, comparison: str, model_a: str, model_b: str, scope: str,
                      fa: pd.DataFrame, fb: pd.DataFrame, min_shared: int) -> Optional[Dict]:
    a, b = acc_outcomes(fa, scope), acc_outcomes(fb, scope)
    m = _shared(a, b)
    if len(m) < min_shared:
        return None
    res = paired_diff_with_ci90(m["value_a"].values, m["value_b"].values, m["cluster_a"].values, "diff")
    return _diff_row(dataset, family, comparison, model_a, model_b, "accuracy", scope, res, "diff",
                     float(m["value_a"].mean()), float(m["value_b"].mean()), m["cluster_a"].nunique())


def _ba_diff(pa: np.ndarray, pb: np.ndarray, gt: np.ndarray) -> float:
    return balanced_accuracy(pa, gt) - balanced_accuracy(pb, gt)


def balanced_contrast(dataset: str, family: str, comparison: str, model_a: str, model_b: str, scope: str,
                      fa: pd.DataFrame, fb: pd.DataFrame, min_shared: int) -> Optional[Dict]:
    a, b = answered_pairs(fa, scope), answered_pairs(fb, scope)
    m = _shared(a, b)
    if len(m) < min_shared:
        return None
    res = bootstrap_statistic([m["pred_a"].values, m["pred_b"].values, m["gt_a"].values], m["cluster_a"].values, _ba_diff)
    packed = _pack("diff", res["point"], res["std"], res["ci_lower"], res["ci_upper"], res["n"], True)
    packed.update({"diff_ci90_low_raw": res["ci90_lower"], "diff_ci90_high_raw": res["ci90_upper"], "p_raw": res["p_value"]})
    return _diff_row(dataset, family, comparison, model_a, model_b, "balanced_accuracy", scope, packed, "diff",
                     balanced_accuracy(m["pred_a"].values, m["gt_a"].values), balanced_accuracy(m["pred_b"].values, m["gt_a"].values),
                     m["cluster_a"].nunique())


def swap_contrast(dataset: str, family: str, model_a: str, model_b: str, metric: str, scope: str, fa: pd.DataFrame,
                  fb: pd.DataFrame, min_shared: int) -> Optional[Dict]:
    if metric == "uar":
        a, b = same_outcomes(fa, "swap", scope, True), same_outcomes(fb, "swap", scope, True)
    else:
        a, b = change_outcomes(fa, "opposite_swap", scope, True), change_outcomes(fb, "opposite_swap", scope, True)
    m = _shared(a, b)
    if len(m) < min_shared:
        return None
    res = paired_diff_with_ci90(m["value_a"].values, m["value_b"].values, m["cluster_a"].values, "diff")
    return _diff_row(dataset, family, "vs_text_control", model_a, model_b, metric, scope, res, "diff",
                     float(m["value_a"].mean()), float(m["value_b"].mean()), m["cluster_a"].nunique())


def subgroup_tests(dataset: str, model: str, df: pd.DataFrame) -> List[Dict]:
    masks = subgroup_masks(df)
    out = []
    outcome_fns = {"cgr": cgr_outcomes, "uar": lambda d: same_outcomes(d, "swap", "pooled", True),
                   "ofr": lambda d: change_outcomes(d, "opposite_swap", "finding_presence", True)}
    for metric, fn in outcome_fns.items():
        for attr, levels in (("sex", ["F", "M"]), ("view", ["PA", "AP"])):
            groups = [fn(df[masks[f"{attr}={lv}"]])["value"].values for lv in levels]
            if any(len(g) < 2 for g in groups):
                continue
            res = report_permutation_2(groups[0], groups[1], prefix="diff")
            out.append({"dataset": dataset, "model": model, "metric": metric, "attribute": attr,
                        "groups": ",".join(f"{lv}(n={len(g)})" for lv, g in zip(levels, groups)),
                        "group_means_pct": ",".join(f"{lv}={fmt_pct(g.mean())}" for lv, g in zip(levels, groups)),
                        "diff_pct": res["diff_mean"], "diff_ci_low_pct": res["diff_ci_low"], "diff_ci_high_pct": res["diff_ci_high"],
                        "statistic": np.nan, "p_raw": res["p_raw"]})
        bands = [(name, fn(df[masks[name]])["value"].values) for name, _, _ in
                 (("age=<50", 0, 50), ("age=50-70", 50, 70), ("age=>70", 70, 200))]
        bands = [(n, g) for n, g in bands if len(g) >= 2]
        if len(bands) >= 2:
            res = report_permutation_k([g for _, g in bands], prefix="anova")
            out.append({"dataset": dataset, "model": model, "metric": metric, "attribute": "age",
                        "groups": ",".join(f"{n}(n={len(g)})" for n, g in bands),
                        "group_means_pct": ",".join(f"{n}={fmt_pct(g.mean())}" for n, g in bands),
                        "diff_pct": np.nan, "diff_ci_low_pct": np.nan, "diff_ci_high_pct": np.nan,
                        "statistic": res["anova_value_raw"], "p_raw": res["p_raw"]})
    return out


def main_paired_comparisons(cfg_path: str) -> None:
    from tqdm import tqdm
    cfg = read_config(cfg_path)["CausalAudit"]
    out_dir = ensure_dir(cfg["outputs"]["metrics_dir"])
    status = status_path(cfg, "paired_comparisons")
    min_shared = int(cfg["stats"]["min_cell_n"])
    margin = float(cfg["stats"]["equivalence_margin_points"])
    rows: List[Dict] = []
    sub_rows: List[Dict] = []
    for dataset in ("mimic", "chexpert"):
        frames = _frames(cfg, dataset)
        if not frames:
            raise MissingInput(f"no records for {dataset}; run main_import_delivered and the model runs first")
        controls = [c for c in _controls(cfg) if c in frames]
        systems = [m for m in cfg["model_order"] if m in frames]
        for control in controls:
            for scope in tqdm(ACC_SCOPES[dataset], desc=f"[paired] {dataset} against {control}", unit="scope"):
                for model in systems:
                    if model == control:
                        continue
                    fam = f"{dataset}|accuracy|{scope}|{control}"
                    r = accuracy_contrast(dataset, fam, "vs_text_control", model, control, scope, frames[model], frames[control], min_shared)
                    if r:
                        rows.append(r)
                    if scope != "ms_cxr":
                        fam = f"{dataset}|balanced_accuracy|{scope}|{control}"
                        r = balanced_contrast(dataset, fam, "vs_text_control", model, control, scope, frames[model], frames[control], min_shared)
                        if r:
                            rows.append(r)
            for metric in ("uar", "ofr"):
                for model in systems:
                    if model == control:
                        continue
                    r = swap_contrast(dataset, f"{dataset}|{metric}|{control}", model, control, metric, SWAP_SCOPE[dataset],
                                      frames[model], frames[control], min_shared)
                    if r:
                        rows.append(r)
        for i, ma in enumerate(systems):
            for mb in systems[i + 1:]:
                r = balanced_contrast(dataset, f"{dataset}|pairwise_balanced_accuracy|{PAIRWISE_SCOPE[dataset]}", "pairwise", ma, mb,
                                      PAIRWISE_SCOPE[dataset], frames[ma], frames[mb], min_shared)
                if r:
                    rows.append(r)
        for model in tqdm(systems, desc=f"[paired] {dataset} subgroup tests", unit="system"):
            sub_rows.extend(subgroup_tests(dataset, model, frames[model]))
        append_status(status, f"{dataset}: {len(rows)} contrasts, {len(sub_rows)} subgroup tests")
    if not rows:
        raise RuntimeError("[paired] no contrast was computed")
    fp = record_store_fingerprint(cfg)
    paired = add_fdr(pd.DataFrame(rows), family_cols=["family"])
    paired.insert(0, "record_store", fp)
    write_csv_atomic(paired, os.path.join(out_dir, "paired_comparisons.csv"))
    equiv = []
    for r in paired.to_dict("records"):
        t = tost_equivalent({"diff_ci90_low_raw": r["diff_ci90_low_raw"], "diff_ci90_high_raw": r["diff_ci90_high_raw"]}, margin, "diff")
        equiv.append({k: r[k] for k in ("dataset", "family", "comparison_type", "model_a", "model_b", "metric", "scope", "diff_pct",
                                         "diff_ci_low_pct", "diff_ci_high_pct", "n_shared", "p_raw", "p_fdr")} | t)
    write_csv_atomic(pd.DataFrame(equiv), os.path.join(out_dir, "equivalence_tests.csv"))
    if sub_rows:
        sub = add_fdr(pd.DataFrame(sub_rows), family_cols=["dataset", "model"])
        sub.insert(0, "record_store", fp)
        write_csv_atomic(sub, os.path.join(out_dir, "subgroup_tests.csv"))
    print(f"[paired] {len(paired)} contrasts, {sum(paired['significant_fdr05'])} significant after FDR, {len(equiv)} equivalence tests, "
          f"{len(sub_rows)} subgroup tests.", flush=True)
    show = paired[(paired["comparison_type"] == "vs_text_control") & (paired["metric"] == "balanced_accuracy") & (paired["scope"] == PAIRWISE_SCOPE["mimic"])]
    if len(show):
        print(show[["model_a", "model_b", "value_a_pct", "value_b_pct", "diff_pct", "diff_ci_low_pct", "diff_ci_high_pct", "p_fdr"]].to_string(index=False), flush=True)
    append_status(status, "done")
