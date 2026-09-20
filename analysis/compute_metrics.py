"""
analysis/compute_metrics.py
Created on September 14, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import hashlib
import os
from typing import Dict, List

import numpy as np
import pandas as pd

from analysis.metrics_core import (NON_ANSWER_CLASSES, REMOVAL_CONDITIONS, acc_outcomes, answered,
                                   answered_pairs, balanced_accuracy, brier, category_submitted, category_v2, cgr_outcomes,
                                   change_outcomes, confidence_informative, ece, f1_yes, has_condition, is_outcomes, metric,
                                   nonanswer_counts, paired_premium, removal_agreement_outcomes, same_outcomes, scope_mask,
                                   sens_outcomes, spec_outcomes, statistic_metric, swapped_acc_outcomes, wide_frame,
                                   wilson_interval, yes_outcomes)
from config.serde import read_config
from data_loader.probe_dataset import load_manifest
from Inference.report_utils import fmt_pct, report_auroc
from Inference.resume_utils import MissingInput, append_status, ensure_dir, status_path, write_csv_atomic, write_json_atomic


ACCURACY_SCOPES = {"mimic": ["pooled", "finding_presence", "image_necessary", "ms_cxr", "mimic_cxr", "rexerr",
                             "rexerr_image_dependent", "rexerr_text_only", "rexerr_control"],
                   "chexpert": ["pooled"]}
SWAP_SCOPES = {"mimic": ["finding_presence", "ms_cxr", "mimic_cxr", "pooled", "rexerr"], "chexpert": ["pooled"]}
AGE_BANDS = [("age=<50", 0, 50), ("age=50-70", 50, 70), ("age=>70", 70, 200)]


def _row(dataset: str, model: str, variant: str, scope: str, condition: str, name: str, packed: Dict, prefix: str,
         n_clusters: int, wilson=None) -> Dict:
    return {"dataset": dataset, "model": model, "variant": variant, "scope": scope, "condition": condition, "metric": name,
            "mean_pct": packed[f"{prefix}_mean"], "std_pct": packed[f"{prefix}_std"], "ci_low_pct": packed[f"{prefix}_ci_low"],
            "ci_high_pct": packed[f"{prefix}_ci_high"], "mean_raw": packed[f"{prefix}_mean_raw"], "std_raw": packed[f"{prefix}_std_raw"],
            "ci_low_raw": packed[f"{prefix}_ci_low_raw"], "ci_high_raw": packed[f"{prefix}_ci_high_raw"], "n": packed["n"],
            "n_clusters": int(n_clusters), "wilson_low_pct": fmt_pct(wilson[0]) if wilson else np.nan,
            "wilson_high_pct": fmt_pct(wilson[1]) if wilson else np.nan}


def _emit(rows: List[Dict], dataset: str, model: str, variant: str, scope: str, condition: str, name: str, o: pd.DataFrame,
          wilson: bool = False) -> Dict:
    packed = metric(o, "m")
    w = wilson_interval(int(o["value"].sum()), len(o)) if wilson and len(o) else None
    rows.append(_row(dataset, model, variant, scope, condition, name, packed, "m", o["cluster"].nunique() if len(o) else 0, w))
    return packed


def _emit_stat(rows: List[Dict], dataset: str, model: str, variant: str, scope: str, condition: str, name: str,
               pairs: pd.DataFrame, fn) -> Dict:
    packed = statistic_metric(pairs, fn, "m")
    rows.append(_row(dataset, model, variant, scope, condition, name, packed, "m", pairs["cluster"].nunique() if len(pairs) else 0))
    return packed


def subgroup_masks(df: pd.DataFrame) -> Dict[str, pd.Series]:
    out = {}
    for v in ("PA", "AP"):
        out[f"view={v}"] = df["view"] == v
    for g in ("F", "M"):
        out[f"sex={g}"] = df["gender"] == g
    age = pd.to_numeric(df["age"], errors="coerce")
    for name, lo, hi in AGE_BANDS:
        out[name] = (age >= lo) & (age < hi)
    return out


def accuracy_family(rows: List[Dict], df: pd.DataFrame, dataset: str, model: str, variant: str, scopes: List[str],
                    cond: str = "original") -> None:
    for scope in scopes:
        if not scope_mask(df, scope).any():
            continue
        _emit(rows, dataset, model, variant, scope, cond, "accuracy", acc_outcomes(df, scope, cond))
        _emit(rows, dataset, model, variant, scope, cond, "sensitivity", sens_outcomes(df, scope, cond))
        _emit(rows, dataset, model, variant, scope, cond, "specificity", spec_outcomes(df, scope, cond))
        _emit(rows, dataset, model, variant, scope, cond, "yes_rate", yes_outcomes(df, scope, cond))
        pairs = answered_pairs(df, scope, cond)
        _emit_stat(rows, dataset, model, variant, scope, cond, "balanced_accuracy", pairs, balanced_accuracy)
        _emit_stat(rows, dataset, model, variant, scope, cond, "f1", pairs, f1_yes)


def reference_rows(rows: List[Dict], df: pd.DataFrame, dataset: str, scopes: List[str]) -> None:
    ref = df.copy()
    for name, value in (("ALWAYS_YES", 1.0), ("ALWAYS_NO", 0.0)):
        ref["pa_original"] = value
        ref["cls_original"] = "yes" if value else "no"
        accuracy_family(rows, ref, dataset, name, "default", scopes)


def swap_family(rows: List[Dict], df: pd.DataFrame, dataset: str, model: str, variant: str, scopes: List[str]) -> Dict:
    packed: Dict[str, Dict] = {}
    for scope in scopes:
        if not scope_mask(df, scope).any():
            continue
        if has_condition(df, "swap"):
            packed[f"uar:{scope}"] = _emit(rows, dataset, model, variant, scope, "swap", "uar", same_outcomes(df, "swap", scope, True))
            _emit(rows, dataset, model, variant, scope, "swap", "change_rate_all", change_outcomes(df, "swap", scope, False))
            _emit(rows, dataset, model, variant, scope, "swap", "swapped_accuracy", swapped_acc_outcomes(df, "swap", scope))
        if has_condition(df, "opposite_swap") and scope != "rexerr":
            packed[f"ofr:{scope}"] = _emit(rows, dataset, model, variant, scope, "opposite_swap", "ofr", change_outcomes(df, "opposite_swap", scope, True))
            _emit(rows, dataset, model, variant, scope, "opposite_swap", "change_rate_all", change_outcomes(df, "opposite_swap", scope, False))
            _emit(rows, dataset, model, variant, scope, "opposite_swap", "swapped_accuracy", swapped_acc_outcomes(df, "opposite_swap", scope))
            ssp = paired_premium(df, "opposite_swap", "swap", scope, "ssp")
            packed[f"ssp:{scope}"] = ssp
            rows.append(_row(dataset, model, variant, scope, "opposite_swap", "ssp", ssp, "ssp", 0))
            rows[-1]["p_raw"] = ssp.get("p_raw", np.nan)
    return packed


def mask_family(rows: List[Dict], df: pd.DataFrame, dataset: str, model: str, variant: str) -> Dict:
    packed: Dict[str, Dict] = {}
    if not has_condition(df, "target_mask"):
        return packed
    packed["cgr"] = _emit(rows, dataset, model, variant, "ms_cxr", "target_mask", "cgr", cgr_outcomes(df))
    _emit(rows, dataset, model, variant, "ms_cxr", "target_mask", "change_rate_all", change_outcomes(df, "target_mask", "ms_cxr", False))
    for mask in ("irrelevant_mask", "matched_mask"):
        if not has_condition(df, mask):
            continue
        packed[f"is:{mask}"] = _emit(rows, dataset, model, variant, "ms_cxr", mask, "is", is_outcomes(df, mask, "ms_cxr", True))
        _emit(rows, dataset, model, variant, "ms_cxr", mask, "is_unconditioned", is_outcomes(df, mask, "ms_cxr", False))
        gsp = paired_premium(df, "target_mask", mask, "ms_cxr", "gsp", correct_only=True, affirmative_only=True)
        packed[f"gsp:{mask}"] = gsp
        rows.append(_row(dataset, model, variant, "ms_cxr", mask, "gsp", gsp, "gsp", 0))
        rows[-1]["p_raw"] = gsp.get("p_raw", np.nan)
    if has_condition(df, "matched_mask") and "matched_box_placement" in df.columns:
        for place in ("mirrored", "shifted", "corner_fallback"):
            sub = df[df["matched_box_placement"] == place]
            if len(sub):
                _emit(rows, dataset, model, variant, f"placement={place}", "matched_mask", "is", is_outcomes(sub, "matched_mask", "ms_cxr", True))
    return packed


def removal_family(rows: List[Dict], df: pd.DataFrame, dataset: str, model: str, variant: str) -> None:
    for cond in REMOVAL_CONDITIONS:
        if not has_condition(df, cond):
            continue
        for scope in ("pooled", "finding_presence", "image_necessary", "ms_cxr", "mimic_cxr", "rexerr"):
            if not scope_mask(df, scope).any():
                continue
            _emit(rows, dataset, model, variant, scope, cond, "accuracy", acc_outcomes(df, scope, cond))
            _emit(rows, dataset, model, variant, scope, cond, "prior_agreement", removal_agreement_outcomes(df, cond, scope))
            _emit(rows, dataset, model, variant, scope, cond, "yes_rate", yes_outcomes(df, scope, cond))


def per_finding_family(rows: List[Dict], df: pd.DataFrame, dataset: str, model: str, variant: str) -> None:
    for finding in sorted(df.loc[df["source"].isin(["ms_cxr", "mimic_cxr", "chexpert"]), "finding"].dropna().unique()):
        sub = df[df["finding"] == finding]
        scope = f"finding={finding}"
        if has_condition(sub, "swap"):
            _emit(rows, dataset, model, variant, scope, "swap", "uar", same_outcomes(sub, "swap", "finding_presence", True), wilson=True)
        if has_condition(sub, "opposite_swap"):
            _emit(rows, dataset, model, variant, scope, "opposite_swap", "ofr", change_outcomes(sub, "opposite_swap", "finding_presence", True), wilson=True)
        if has_condition(sub, "target_mask"):
            _emit(rows, dataset, model, variant, scope, "target_mask", "cgr", cgr_outcomes(sub), wilson=True)
        for mask in ("irrelevant_mask", "matched_mask"):
            if has_condition(sub, mask):
                _emit(rows, dataset, model, variant, scope, mask, "is", is_outcomes(sub, mask, "ms_cxr", True), wilson=True)
        _emit(rows, dataset, model, variant, scope, "original", "accuracy", acc_outcomes(sub, "finding_presence"), wilson=True)
    for state, gt in (("label_state=present", 1), ("label_state=absent", 0)):
        sub = df[(df["gt"] == gt) & df["source"].isin(["ms_cxr", "mimic_cxr", "chexpert"])]
        if has_condition(sub, "swap"):
            _emit(rows, dataset, model, variant, state, "swap", "uar", same_outcomes(sub, "swap", "finding_presence", True))
        if has_condition(sub, "opposite_swap"):
            _emit(rows, dataset, model, variant, state, "opposite_swap", "ofr", change_outcomes(sub, "opposite_swap", "finding_presence", True))


def subgroup_family(rows: List[Dict], df: pd.DataFrame, dataset: str, model: str, variant: str) -> None:
    for scope, mask in subgroup_masks(df).items():
        sub = df[mask]
        if len(sub) == 0:
            continue
        _emit(rows, dataset, model, variant, scope, "original", "accuracy", acc_outcomes(sub, "pooled"))
        if has_condition(sub, "swap"):
            _emit(rows, dataset, model, variant, scope, "swap", "uar", same_outcomes(sub, "swap", "pooled", True))
        if has_condition(sub, "opposite_swap"):
            _emit(rows, dataset, model, variant, scope, "opposite_swap", "ofr", change_outcomes(sub, "opposite_swap", "finding_presence", True))
        if has_condition(sub, "target_mask"):
            _emit(rows, dataset, model, variant, scope, "target_mask", "cgr", cgr_outcomes(sub))


def nonanswer_rows(df: pd.DataFrame, dataset: str, model: str, variant: str) -> List[Dict]:
    out = []
    for cond in [c[len("cls_"):] for c in df.columns if c.startswith("cls_")]:
        for scope in ("pooled", "ms_cxr", "mimic_cxr", "rexerr"):
            counts = nonanswer_counts(df, cond, scope)
            total = sum(counts.values())
            if not total:
                continue
            for cls in ("yes", "no") + NON_ANSWER_CLASSES:
                out.append({"dataset": dataset, "model": model, "variant": variant, "condition": cond, "scope": scope,
                            "answer_class": cls, "count": counts.get(cls, 0), "share_pct": fmt_pct(counts.get(cls, 0) / total),
                            "n": total})
            out.append({"dataset": dataset, "model": model, "variant": variant, "condition": cond, "scope": scope,
                        "answer_class": "answered", "count": counts.get("yes", 0) + counts.get("no", 0),
                        "share_pct": fmt_pct((counts.get("yes", 0) + counts.get("no", 0)) / total), "n": total})
    return out


def confidence_rows(df: pd.DataFrame, dataset: str, model: str, cfg: Dict) -> Dict:
    ok = answered(df["pa_original"]) & df["gt"].isin([0, 1]) & np.isfinite(df["conf_original"].astype(float))
    sub = df[ok]
    out = {"dataset": dataset, "model": model, "n_with_confidence": int(len(sub))}
    conf = sub["conf_original"].astype(float).values
    out["informative"] = bool(confidence_informative(conf, cfg)) if len(sub) else False
    if not out["informative"]:
        return out
    correct = sub["pa_original"] == sub["gt"]
    regimes = {"incorrect": sub[~correct]}
    if "pa_target_mask" in sub.columns:
        ms = (sub["source"] == "ms_cxr") & correct & answered(sub["pa_target_mask"])
        regimes["grounded_correct"] = sub[ms & (sub["pa_target_mask"] != sub["pa_original"])]
        regimes["ungrounded_correct"] = sub[ms & (sub["pa_target_mask"] == sub["pa_original"])]
    for name, r in regimes.items():
        c = r["conf_original"].astype(float).values
        out[f"conf_{name}_mean_pct"] = fmt_pct(float(c.mean())) if len(c) else np.nan
        out[f"conf_{name}_sd_pct"] = fmt_pct(float(c.std(ddof=1))) if len(c) > 1 else np.nan
        out[f"conf_{name}_n"] = int(len(c))
    au = report_auroc(sub["gt"].values, conf, prefix="auroc", cluster_ids=sub["cluster"].values)
    out.update({k: v for k, v in au.items() if k.startswith("auroc")})
    out["brier"] = round(brier(conf, sub["gt"].values.astype(float)), 4)
    out["ece"] = round(ece(conf, sub["pa_original"].values.astype(float), sub["gt"].values.astype(float), int(cfg["stats"]["ece_bins"])), 4)
    return out


def category_row(dataset: str, model: str, swap_packed: Dict, mask_packed: Dict, cfg: Dict) -> Dict:
    scope = "finding_presence" if dataset == "mimic" else "pooled"
    ofr, uar, ssp = swap_packed.get(f"ofr:{scope}"), swap_packed.get(f"uar:{scope}"), swap_packed.get(f"ssp:{scope}")
    out = {"dataset": dataset, "model": model, "scope": scope}
    for name, packed, prefix in (("ofr", ofr, "m"), ("uar", uar, "m"), ("ssp", ssp, "ssp")):
        if packed is None:
            out[f"{name}_pct"], out[f"{name}_ci_low_pct"], out[f"{name}_ci_high_pct"], out[f"{name}_n"] = np.nan, np.nan, np.nan, 0
            continue
        out[f"{name}_pct"] = packed[f"{prefix}_mean"]
        out[f"{name}_ci_low_pct"] = packed[f"{prefix}_ci_low"]
        out[f"{name}_ci_high_pct"] = packed[f"{prefix}_ci_high"]
        out[f"{name}_n"] = packed["n"]
    out["ssp_p_raw"] = ssp.get("p_raw", np.nan) if ssp else np.nan
    if ofr is not None and uar is not None and ssp is not None:
        out["category_v2"] = category_v2({"n": ofr["n"], "ofr_mean_raw": ofr["m_mean_raw"]}, {"n": uar["n"], "uar_mean_raw": uar["m_mean_raw"]},
                                         {"ssp_ci_low_raw": ssp["ssp_ci_low_raw"], "ssp_mean_raw": ssp["ssp_mean_raw"]}, cfg)
    else:
        out["category_v2"] = "not_computable"
    cgr = mask_packed.get("cgr")
    for mask in ("irrelevant_mask", "matched_mask"):
        is_ = mask_packed.get(f"is:{mask}")
        gsp = mask_packed.get(f"gsp:{mask}")
        tag = "corner" if mask == "irrelevant_mask" else "matched"
        out[f"is_{tag}_pct"] = is_["m_mean"] if is_ else np.nan
        out[f"is_{tag}_n"] = is_["n"] if is_ else 0
        out[f"gsp_{tag}_pct"] = gsp["gsp_mean"] if gsp else np.nan
        out[f"gsp_{tag}_ci_low_pct"] = gsp["gsp_ci_low"] if gsp else np.nan
        out[f"gsp_{tag}_ci_high_pct"] = gsp["gsp_ci_high"] if gsp else np.nan
        if cgr is not None and uar is not None and is_ is not None:
            out[f"category_submitted_{tag}"] = category_submitted(
                {"cgr_mean_raw": cgr["m_mean_raw"], "cgr_ci_low_raw": cgr["m_ci_low_raw"]}, {"uar_mean_raw": uar["m_mean_raw"]},
                {"is_mean_raw": is_["m_mean_raw"]}, cfg)
        else:
            out[f"category_submitted_{tag}"] = "not_computable"
    out["cgr_pct"] = cgr["m_mean"] if cgr else np.nan
    out["cgr_ci_low_pct"] = cgr["m_ci_low"] if cgr else np.nan
    out["cgr_ci_high_pct"] = cgr["m_ci_high"] if cgr else np.nan
    out["cgr_n"] = cgr["n"] if cgr else 0
    return out


def record_store_fingerprint(cfg: Dict) -> str:
    h = hashlib.blake2b(digest_size=8)
    root = cfg["runs"]["dir"]
    for dirpath, _, files in sorted(os.walk(root)):
        for f in sorted(files):
            if f.endswith(".jsonl"):
                st = os.stat(os.path.join(dirpath, f))
                h.update(f"{os.path.relpath(os.path.join(dirpath, f), root)}:{st.st_size}:{int(st.st_mtime)}".encode())
    return h.hexdigest()


def main_compute_metrics(cfg_path: str) -> None:
    from tqdm import tqdm
    cfg = read_config(cfg_path)["CausalAudit"]
    out_dir = ensure_dir(cfg["outputs"]["metrics_dir"])
    status = status_path(cfg, "compute_metrics")
    if not os.path.isdir(cfg["runs"]["dir"]):
        raise MissingInput(f"{cfg['runs']['dir']} does not exist; run main_import_delivered first")
    rows: List[Dict] = []
    nonanswers: List[Dict] = []
    confidences: List[Dict] = []
    categories: List[Dict] = []
    for dataset in ("mimic", "chexpert"):
        manifest = load_manifest(cfg, dataset)
        first = True
        for model in tqdm(cfg["model_order"], desc=f"[metrics] {dataset}", unit="system"):
            try:
                df = wide_frame(cfg, dataset, model, manifest=manifest)
            except MissingInput as e:
                print(f"[metrics] SKIP {dataset}/{model}: {e}", flush=True)
                continue
            if first:
                reference_rows(rows, df, dataset, ACCURACY_SCOPES[dataset])
                first = False
            accuracy_family(rows, df, dataset, model, "default", ACCURACY_SCOPES[dataset])
            swap_packed = swap_family(rows, df, dataset, model, "default", SWAP_SCOPES[dataset])
            mask_packed = mask_family(rows, df, dataset, model, "default")
            removal_family(rows, df, dataset, model, "default")
            per_finding_family(rows, df, dataset, model, "default")
            subgroup_family(rows, df, dataset, model, "default")
            nonanswers.extend(nonanswer_rows(df, dataset, model, "default"))
            confidences.append(confidence_rows(df, dataset, model, cfg))
            categories.append(category_row(dataset, model, swap_packed, mask_packed, cfg))
            append_status(status, f"{dataset}/{model}: {len(rows)} metric rows so far")
    fp = record_store_fingerprint(cfg)
    tables = {"system_metrics.csv": pd.DataFrame(rows), "nonanswer_rates.csv": pd.DataFrame(nonanswers),
              "confidence.csv": pd.DataFrame(confidences), "categories.csv": pd.DataFrame(categories)}
    for name, df in tables.items():
        if len(df) == 0:
            raise RuntimeError(f"[metrics] {name} came out empty; nothing was computed")
        df.insert(0, "record_store", fp)
        write_csv_atomic(df, os.path.join(out_dir, name))
        print(f"[metrics] {name}: {len(df)} rows", flush=True)
    write_json_atomic(os.path.join(out_dir, "build_info.json"), {"record_store": fp, "n_boot": cfg["stats"]["n_boot"],
                                                                 "cluster_by": cfg["stats"]["cluster_by"]})
    cats = tables["categories.csv"]
    print(cats[["dataset", "model", "ofr_pct", "uar_pct", "ssp_pct", "category_v2", "cgr_pct", "is_corner_pct", "category_submitted_corner"]].to_string(index=False), flush=True)
    append_status(status, "done")

