"""
analysis/reanalyses.py
Created on September 14, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from analysis.compute_metrics import accuracy_family, record_store_fingerprint, swap_family
from analysis.metrics_core import (acc_outcomes, answered, answered_pairs, balanced_accuracy, category_submitted, category_v2,
                                   cgr_outcomes, change_outcomes, has_condition, is_outcomes, metric, paired_premium,
                                   same_outcomes, statistic_metric, wide_frame)
from config.serde import read_config
from data_loader.probe_dataset import load_manifest
from Inference.inference_runner import Unit, unit_path
from Inference.report_utils import fmt_pct, report_paired_diff
from Inference.resume_utils import MissingInput, append_status, ensure_dir, status_path, write_csv_atomic


def _systems(cfg: Dict, dataset: str, variant: str = "default", permissive: bool = False, suffix: str = "") -> Dict[str, pd.DataFrame]:
    manifest = load_manifest(cfg, dataset)
    out = {}
    for model in cfg["model_order"]:
        try:
            out[model] = wide_frame(cfg, dataset, model, variant=variant, suffix=suffix, manifest=manifest, permissive=permissive, verbose=False)
        except MissingInput:
            continue
    return out


def _flat(prefix: str, packed: Dict) -> Dict:
    return {"mean_pct": packed[f"{prefix}_mean"], "std_pct": packed[f"{prefix}_std"], "ci_low_pct": packed[f"{prefix}_ci_low"],
            "ci_high_pct": packed[f"{prefix}_ci_high"], "mean_raw": packed[f"{prefix}_mean_raw"], "n": packed["n"]}


def nonanswer_sensitivity(cfg: Dict, frames: Dict[str, pd.DataFrame]) -> List[Dict]:
    rows = []
    for model, df in frames.items():
        for scope in ("pooled", "finding_presence", "image_necessary", "mimic_cxr", "ms_cxr", "rexerr"):
            for mode, wrong in (("answered_only", False), ("nonanswer_wrong", True)):
                o = acc_outcomes(df, scope, "original", nonanswer_wrong=wrong)
                if len(o) == 0:
                    continue
                rows.append({"analysis": "nonanswer_sensitivity", "dataset": "mimic", "model": model, "scope": scope, "scoring": mode,
                             "metric": "accuracy", **_flat("m", metric(o, "m"))})
    return rows


def parser_sensitivity(cfg: Dict, strict: Dict[str, pd.DataFrame], permissive: Dict[str, pd.DataFrame]) -> List[Dict]:
    rows = []
    for model in strict:
        if model not in permissive:
            continue
        for name, frames in (("fixed", strict), ("permissive", permissive)):
            df = frames[model]
            rows.append({"analysis": "parser_sensitivity", "dataset": "mimic", "model": model, "parser": name, "metric": "accuracy",
                         "scope": "pooled", **_flat("m", metric(acc_outcomes(df, "pooled"), "m"))})
            rows.append({"analysis": "parser_sensitivity", "dataset": "mimic", "model": model, "parser": name, "metric": "answered_share",
                         "scope": "pooled", "mean_pct": fmt_pct(answered(df["pa_original"]).mean()), "n": int(len(df))})
            if has_condition(df, "swap"):
                rows.append({"analysis": "parser_sensitivity", "dataset": "mimic", "model": model, "parser": name, "metric": "uar",
                             "scope": "finding_presence", **_flat("m", metric(same_outcomes(df, "swap", "finding_presence"), "m"))})
            if has_condition(df, "opposite_swap"):
                rows.append({"analysis": "parser_sensitivity", "dataset": "mimic", "model": model, "parser": name, "metric": "ofr",
                             "scope": "finding_presence", **_flat("m", metric(change_outcomes(df, "opposite_swap", "finding_presence"), "m"))})
            if has_condition(df, "target_mask"):
                rows.append({"analysis": "parser_sensitivity", "dataset": "mimic", "model": model, "parser": name, "metric": "cgr",
                             "scope": "ms_cxr", **_flat("m", metric(cgr_outcomes(df), "m"))})
    return rows


def fallback_sensitivity(cfg: Dict, frames: Dict[str, pd.DataFrame]) -> List[Dict]:
    rows: List[Dict] = []
    for model, df in frames.items():
        sub = df[df["label_source"] != "value_3_fallback"]
        tmp: List[Dict] = []
        accuracy_family(tmp, sub, "mimic", model, "default", ["finding_presence", "mimic_cxr"])
        swap_family(tmp, sub, "mimic", model, "default", ["finding_presence", "mimic_cxr"])
        for r in tmp:
            r["analysis"] = "fallback_excluded"
        rows.extend(tmp)
    return rows


def common_subset(cfg: Dict, frames: Dict[str, pd.DataFrame], categories: pd.DataFrame) -> List[Dict]:
    users = categories[(categories["dataset"] == "mimic") & (categories["category_v2"] == "uses_image")]["model"].tolist()
    if not users:
        users = categories[(categories["dataset"] == "mimic") & (categories["category_submitted_corner"] == "uses_image")]["model"].tolist()
    users = [u for u in users if u in frames]
    if len(users) < 2:
        return []
    common: Optional[set] = None
    for m in users:
        df = frames[m]
        ok = (df["source"] == "ms_cxr") & (df["gt"] == 1) & (df["pa_original"] == 1)
        ids = set(df.loc[ok, "case_id"])
        common = ids if common is None else common & ids
    rows = []
    for m, df in frames.items():
        sub = df[df["case_id"].isin(common)]
        if not has_condition(sub, "target_mask"):
            continue
        rows.append({"analysis": "common_subset", "dataset": "mimic", "model": m, "metric": "cgr", "scope": "ms_cxr_common",
                     "systems_defining_subset": ",".join(users), **_flat("m", metric(cgr_outcomes(sub), "m"))})
        for mask in ("irrelevant_mask", "matched_mask"):
            if has_condition(sub, mask):
                rows.append({"analysis": "common_subset", "dataset": "mimic", "model": m, "metric": f"is_{mask}", "scope": "ms_cxr_common",
                             "systems_defining_subset": ",".join(users), **_flat("m", metric(is_outcomes(sub, mask), "m"))})
    return rows


def category_thresholds(cfg: Dict, frames: Dict[str, pd.DataFrame], dataset: str) -> List[Dict]:
    rows = []
    scope = "finding_presence" if dataset == "mimic" else "pooled"
    for model, df in frames.items():
        uar = metric(same_outcomes(df, "swap", scope), "m") if has_condition(df, "swap") else None
        ofr = metric(change_outcomes(df, "opposite_swap", scope), "m") if has_condition(df, "opposite_swap") else None
        ssp = paired_premium(df, "opposite_swap", "swap", scope, "ssp") if has_condition(df, "opposite_swap") else None
        if uar is not None and ofr is not None and ssp is not None:
            for n_min in (50, 100, 200):
                c2 = {**cfg, "stats": {**cfg["stats"], "min_informative_cases": n_min}}
                rows.append({"analysis": "category_v2_thresholds", "dataset": dataset, "model": model, "rule": "v2",
                             "min_informative_cases": n_min, "category": category_v2({"n": ofr["n"], "ofr_mean_raw": ofr["m_mean_raw"]},
                                                                                       {"n": uar["n"], "uar_mean_raw": uar["m_mean_raw"]},
                                                                                       {"ssp_ci_low_raw": ssp["ssp_ci_low_raw"], "ssp_mean_raw": ssp["ssp_mean_raw"]}, c2)})
        if dataset != "mimic" or not has_condition(df, "target_mask") or uar is None:
            continue
        cgr = metric(cgr_outcomes(df), "m")
        for mask in ("irrelevant_mask", "matched_mask"):
            if not has_condition(df, mask):
                continue
            for cond_name, correct_only in (("conditioned", True), ("unconditioned", False)):
                is_ = metric(is_outcomes(df, mask, "ms_cxr", correct_only), "m")
                for thr in cfg["stats"]["submitted_category_rule"]["sensitivity_thresholds"]:
                    rows.append({"analysis": "category_submitted_thresholds", "dataset": dataset, "model": model, "rule": "submitted",
                                 "mask": mask, "is_definition": cond_name, "is_threshold": thr,
                                 "category": category_submitted({"cgr_mean_raw": cgr["m_mean_raw"], "cgr_ci_low_raw": cgr["m_ci_low_raw"]},
                                                                {"uar_mean_raw": uar["m_mean_raw"]}, {"is_mean_raw": is_["m_mean_raw"]}, cfg,
                                                                unstable_below=thr, uses_min=thr)})
    return rows


def prompt_sensitivity(cfg: Dict) -> List[Dict]:
    rows = []
    manifest = load_manifest(cfg, "mimic")
    subset_path = os.path.join(cfg["data"]["subsets_dir"], "prompt_mimic_subset.csv")
    sub_ids = set(pd.read_csv(subset_path, low_memory=False)["case_id"].astype(str)) if os.path.exists(subset_path) else set()
    min_n = int(cfg["stats"]["min_cell_n"])
    for variant in ["default"] + list(cfg["prompt_sensitivity"]["variants"]):
        for model in cfg["model_order"]:
            try:
                df = wide_frame(cfg, "mimic", model, variant=variant, manifest=manifest, verbose=False)
            except MissingInput:
                continue
            ms = df[df["source"] == "ms_cxr"]
            base = {"analysis": "prompt_sensitivity", "dataset": "mimic", "model": model, "variant": variant}
            o = acc_outcomes(ms, "pooled")
            rows.append({**base, "metric": "accuracy", "scope": "ms_cxr", **_flat("m", metric(o, "m")), "interpretable": len(o) >= min_n})
            rows.append({**base, "metric": "answered_share", "scope": "ms_cxr", "mean_pct": fmt_pct(answered(ms["pa_original"]).mean()),
                         "n": int(ms["cls_original"].notna().sum()), "interpretable": True})
            if sub_ids:
                mm = df[df["case_id"].isin(sub_ids)]
                pairs = answered_pairs(mm, "pooled")
                rows.append({**base, "metric": "balanced_accuracy", "scope": "mimic_cxr_subset", **_flat("m", statistic_metric(pairs, balanced_accuracy, "m")),
                             "interpretable": len(pairs) >= min_n})
            if has_condition(ms, "target_mask"):
                o = cgr_outcomes(ms)
                rows.append({**base, "metric": "cgr", "scope": "ms_cxr", **_flat("m", metric(o, "m")), "interpretable": len(o) >= min_n})
            if has_condition(ms, "irrelevant_mask"):
                o = is_outcomes(ms, "irrelevant_mask")
                rows.append({**base, "metric": "is", "scope": "ms_cxr", **_flat("m", metric(o, "m")), "interpretable": len(o) >= min_n})
    return rows


def resolution_robustness(cfg: Dict, frames224: Dict[str, pd.DataFrame]) -> List[Dict]:
    rows = []
    manifest = load_manifest(cfg, "mimic")
    points = {}
    for model in cfg["model_order"]:
        try:
            df512 = wide_frame(cfg, "mimic512", model, manifest=manifest, verbose=False)
        except MissingInput:
            continue
        if not has_condition(df512, "target_mask") or model not in frames224:
            continue
        ids = set(df512.loc[answered(df512["pa_original"]), "case_id"])
        df224 = frames224[model][frames224[model]["case_id"].isin(ids)]
        o224, o512 = cgr_outcomes(df224), cgr_outcomes(df512)
        m224, m512 = metric(o224, "m"), metric(o512, "m")
        rows.append({"analysis": "resolution", "dataset": "mimic512", "model": model, "metric": "cgr_224", **_flat("m", m224)})
        rows.append({"analysis": "resolution", "dataset": "mimic512", "model": model, "metric": "cgr_512", **_flat("m", m512)})
        shared = o224.merge(o512, on="case_id", suffixes=("_224", "_512"))
        if len(shared) >= int(cfg["stats"]["min_cell_n"]):
            d = report_paired_diff(shared["value_512"].values, shared["value_224"].values, prefix="diff", cluster_ids=shared["cluster_224"].values)
            rows.append({"analysis": "resolution", "dataset": "mimic512", "model": model, "metric": "cgr_512_minus_224", **_flat("diff", d), "p_raw": d["p_raw"]})
        if np.isfinite(m224["m_mean_raw"]) and np.isfinite(m512["m_mean_raw"]):
            points[model] = (m224["m_mean_raw"], m512["m_mean_raw"])
    if len(points) >= 3:
        a = [v[0] for v in points.values()]
        b = [v[1] for v in points.values()]
        rho, p = spearmanr(a, b)
        rows.append({"analysis": "resolution", "dataset": "mimic512", "model": "all", "metric": "spearman_cgr_224_vs_512",
                     "value": float(rho), "p_raw": float(p), "n": len(points)})
    return rows


def transfer(cfg: Dict, categories: pd.DataFrame, metrics: pd.DataFrame) -> List[Dict]:
    rows = []
    cm = categories[categories["dataset"] == "mimic"].set_index("model")
    cc = categories[categories["dataset"] == "chexpert"].set_index("model")
    for model in sorted(set(cm.index) & set(cc.index)):
        rows.append({"analysis": "transfer", "model": model, "category_mimic": cm.at[model, "category_v2"], "category_chexpert": cc.at[model, "category_v2"],
                     "agree": cm.at[model, "category_v2"] == cc.at[model, "category_v2"]})
    for name, scope_m, scope_c in (("uar", "finding_presence", "pooled"), ("ofr", "finding_presence", "pooled"),
                                   ("balanced_accuracy", "finding_presence", "pooled"), ("accuracy", "finding_presence", "pooled")):
        a = metrics[(metrics["dataset"] == "mimic") & (metrics["metric"] == name) & (metrics["scope"] == scope_m) & (metrics["condition"].isin(["original", "swap", "opposite_swap"]))]
        b = metrics[(metrics["dataset"] == "chexpert") & (metrics["metric"] == name) & (metrics["scope"] == scope_c)]
        a = a[~a["model"].isin(["ALWAYS_YES", "ALWAYS_NO"])].set_index("model")["mean_raw"]
        b = b[~b["model"].isin(["ALWAYS_YES", "ALWAYS_NO"])].set_index("model")["mean_raw"]
        common = [m for m in a.index if m in b.index and np.isfinite(a[m]) and np.isfinite(b[m])]
        if len(common) >= 3:
            rho, p = spearmanr(a[common].values, b[common].values)
            rows.append({"analysis": "transfer", "model": "all", "metric": f"spearman_{name}", "value": float(rho), "p_raw": float(p), "n": len(common)})
    return rows


def text_only_decomposition(cfg: Dict, frames: Dict[str, pd.DataFrame], metrics: pd.DataFrame) -> List[Dict]:
    pooled = metrics[(metrics["dataset"] == "mimic") & (metrics["metric"] == "accuracy") & (metrics["scope"] == "pooled") & (metrics["condition"] == "original")]
    pooled = pooled[~pooled["model"].isin(["ALWAYS_YES", "ALWAYS_NO"])].set_index("model")["mean_raw"]
    multimodal = [m for m in pooled.index if cfg["models"].get(m, {}).get("modality") == "multimodal"]
    text = [m for m in pooled.index if cfg["models"].get(m, {}).get("modality") == "text_only"]
    if not multimodal or not text:
        return []
    best_mm, best_txt = pooled[multimodal].idxmax(), pooled[text].idxmax()
    rows = []
    for scope in ("pooled", "finding_presence", "image_necessary", "mimic_cxr", "ms_cxr", "rexerr", "rexerr_image_dependent", "rexerr_text_only", "rexerr_control"):
        a, b = acc_outcomes(frames[best_mm], scope), acc_outcomes(frames[best_txt], scope)
        m = a.merge(b, on="case_id", suffixes=("_a", "_b"))
        if len(m) < int(cfg["stats"]["min_cell_n"]):
            continue
        d = report_paired_diff(m["value_a"].values, m["value_b"].values, prefix="diff", cluster_ids=m["cluster_a"].values)
        rows.append({"analysis": "text_only_decomposition", "dataset": "mimic", "model_a": best_mm, "model_b": best_txt, "scope": scope,
                     "value_a_pct": fmt_pct(m["value_a"].mean()), "value_b_pct": fmt_pct(m["value_b"].mean()), **_flat("diff", d), "p_raw": d["p_raw"]})
    return rows


def sex_table(metrics: pd.DataFrame, subgroup: Optional[pd.DataFrame]) -> List[Dict]:
    rows = []
    sub = metrics[metrics["scope"].isin(["sex=F", "sex=M"])]
    for (dataset, model, name), g in sub.groupby(["dataset", "model", "metric"]):
        row = {"analysis": "sex_disaggregated", "dataset": dataset, "model": model, "metric": name}
        for _, r in g.iterrows():
            tag = r["scope"].split("=")[1]
            row[f"{tag}_mean_pct"], row[f"{tag}_ci_low_pct"], row[f"{tag}_ci_high_pct"], row[f"{tag}_n"] = r["mean_pct"], r["ci_low_pct"], r["ci_high_pct"], r["n"]
        if subgroup is not None:
            t = subgroup[(subgroup["dataset"] == dataset) & (subgroup["model"] == model) & (subgroup["metric"] == name) & (subgroup["attribute"] == "sex")]
            if len(t):
                row["p_raw"], row["p_fdr"] = float(t["p_raw"].iloc[0]), float(t["p_fdr"].iloc[0])
        rows.append(row)
    return rows


def budget_sensitivity(cfg: Dict, frames: Dict[str, pd.DataFrame]) -> List[Dict]:
    rows = []
    budget = int(cfg["generation"]["rerun_reasoning_max_new_tokens"])
    manifest = load_manifest(cfg, "mimic")
    for model, df in frames.items():
        spec = cfg["models"][model]
        if not spec["reasoning"]:
            continue
        u = Unit("mimic", model, "default", "original", suffix=f"__budget{budget}")
        if not os.path.exists(unit_path(cfg, u)):
            continue
        high = wide_frame(cfg, "mimic", model, suffix=f"__budget{budget}", manifest=manifest, verbose=False)
        merged = df.copy()
        for cond in [c[len("pa_"):] for c in high.columns if c.startswith("pa_")]:
            if f"pa_{cond}" not in merged.columns:
                continue
            fill = answered(high[f"pa_{cond}"]) & ~answered(merged[f"pa_{cond}"])
            merged.loc[fill, f"pa_{cond}"] = high.loc[fill, f"pa_{cond}"]
            merged.loc[fill, f"cls_{cond}"] = high.loc[fill, f"cls_{cond}"]
        for name, frame in (("primary", df), ("with_high_budget", merged)):
            rows.append({"analysis": "budget_sensitivity", "dataset": "mimic", "model": model, "budget": name, "metric": "accuracy", "scope": "pooled",
                         **_flat("m", metric(acc_outcomes(frame, "pooled"), "m"))})
            rows.append({"analysis": "budget_sensitivity", "dataset": "mimic", "model": model, "budget": name, "metric": "answered_share", "scope": "pooled",
                         "mean_pct": fmt_pct(answered(frame["pa_original"]).mean()), "n": int(len(frame))})
    return rows


def main_reanalyses(cfg_path: str) -> None:
    cfg = read_config(cfg_path)["CausalAudit"]
    out_dir = ensure_dir(cfg["outputs"]["analysis_dir"])
    status = status_path(cfg, "reanalyses")
    metrics_dir = cfg["outputs"]["metrics_dir"]
    metrics_path = os.path.join(metrics_dir, "system_metrics.csv")
    if not os.path.exists(metrics_path):
        raise MissingInput(f"{metrics_path} does not exist; run main_compute_metrics first")
    metrics = pd.read_csv(metrics_path, low_memory=False, float_precision="round_trip")
    categories = pd.read_csv(os.path.join(metrics_dir, "categories.csv"), low_memory=False, float_precision="round_trip")
    sub_path = os.path.join(metrics_dir, "subgroup_tests.csv")
    subgroup = pd.read_csv(sub_path, low_memory=False, float_precision="round_trip") if os.path.exists(sub_path) else None
    frames = _systems(cfg, "mimic")
    permissive = _systems(cfg, "mimic", permissive=True)
    chex = _systems(cfg, "chexpert")
    tables = {
        "nonanswer_sensitivity.csv": nonanswer_sensitivity(cfg, frames),
        "parser_sensitivity.csv": parser_sensitivity(cfg, frames, permissive),
        "fallback_sensitivity.csv": fallback_sensitivity(cfg, frames),
        "common_subset.csv": common_subset(cfg, frames, categories),
        "category_thresholds.csv": category_thresholds(cfg, frames, "mimic") + category_thresholds(cfg, chex, "chexpert"),
        "prompt_sensitivity.csv": prompt_sensitivity(cfg),
        "resolution.csv": resolution_robustness(cfg, frames),
        "transfer.csv": transfer(cfg, categories, metrics),
        "text_only_decomposition.csv": text_only_decomposition(cfg, frames, metrics),
        "sex_disaggregated.csv": sex_table(metrics, subgroup),
        "budget_sensitivity.csv": budget_sensitivity(cfg, frames),
    }
    fp = record_store_fingerprint(cfg)
    for name, rows in tables.items():
        path = os.path.join(out_dir, name)
        if not rows:
            print(f"[reanalyses] {name}: nothing to compute yet (its inputs are not on disk); no file written.", flush=True)
            if os.path.exists(path):
                os.remove(path)
            continue
        df = pd.DataFrame(rows)
        df.insert(0, "record_store", fp)
        write_csv_atomic(df, path)
        print(f"[reanalyses] {name}: {len(df)} rows", flush=True)
        append_status(status, f"{name}: {len(df)} rows")
    append_status(status, "done")
