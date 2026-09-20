"""
analysis/metrics_core.py
Created on September 14, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import os
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from data_loader.build_utils import ground_truth
from data_loader.probe_dataset import input_key, load_manifest
from Inference.inference_runner import Unit, load_records, unit_path
from Inference.parser import parse_permissive
from Inference.report_utils import _pack, fmt_pct, report_metric, report_paired_diff
from Inference.resume_utils import MissingInput
from Inference.stats_utils import BOOT_SEED, N_BOOT, _cluster_bootstrap_means, _percentile_ci


NON_ANSWER_CLASSES = ("abstain", "truncated", "empty", "unparsed")
SWAP_CONDITIONS = ("swap", "opposite_swap")
MASK_CONDITIONS = ("target_mask", "irrelevant_mask", "matched_mask")
REMOVAL_CONDITIONS = ("no_image", "noise", "natural_image")
META_COLUMNS = ["case_id", "source", "finding", "label", "label_source", "view", "age", "gender", "subject_id", "dicom_id",
                "error_type", "rexerr_class", "image_necessary", "matched_box_placement", "swap_rebuilt"]


def conditions_of(cfg: Dict, dataset: str) -> List[str]:
    c = cfg["conditions"]
    if dataset == "chexpert":
        return list(c["chexpert"])
    if dataset == "mimic512":
        return list(c["resolution_512"])
    return list(c["mimic"])


def _strip_heads(key: str) -> str:
    return key.split("|heads:")[0] if isinstance(key, str) else key


def wide_frame(cfg: Dict, dataset: str, model: str, variant: str = "default", suffix: str = "",
               manifest: Optional[pd.DataFrame] = None, permissive: bool = False, verbose: bool = True) -> pd.DataFrame:
    spec = cfg["models"][model]
    modality = spec["modality"]
    if manifest is None:
        manifest = load_manifest(cfg, dataset)
    rows = manifest.sort_values("case_id").to_dict("records")
    out = pd.DataFrame({c: [r.get(c) for r in rows] for c in META_COLUMNS if c in manifest.columns})
    out["case_id"] = out["case_id"].astype(str)
    out["gt"] = [ground_truth(r) for r in rows]
    out["cluster"] = out["subject_id"].astype(str)
    if out["subject_id"].isna().any():
        raise ValueError(f"{dataset}: rows with no subject_id; it is the bootstrap cluster unit")
    present_any = False
    for cond in conditions_of(cfg, dataset):
        u = Unit(dataset, model, variant, cond, suffix)
        path = unit_path(cfg, u)
        if not os.path.exists(path):
            continue
        newest, multi, disagree = load_records(path)
        pa, cls, conf, stale = [], [], [], 0
        for r in rows:
            cid = str(r["case_id"])
            rec = newest.get(cid)
            if rec is None:
                pa.append(np.nan)
                cls.append(None)
                conf.append(np.nan)
                continue
            if _strip_heads(rec.get("input_key")) != input_key(cfg, dataset, cond, r, modality) or rec.get("prompt_version") != cfg["prompts"]["version"]:
                stale += 1
                pa.append(np.nan)
                cls.append(None)
                conf.append(np.nan)
                continue
            parsed = int(rec["parsed_answer"])
            if permissive and parsed == -1:
                parsed = parse_permissive(rec.get("raw_answer"), bool(spec["reasoning"]), cfg["parser"])
            pa.append(float(parsed))
            cls.append(rec.get("answer_class"))
            conf.append(np.nan if rec.get("confidence") is None else float(rec["confidence"]))
        out[f"pa_{cond}"] = pa
        out[f"cls_{cond}"] = cls
        out[f"conf_{cond}"] = conf
        present_any = True
        if verbose and (stale or disagree):
            print(f"[metrics] {u}: {stale} record(s) do not match the current manifest and are ignored; "
                  f"{disagree} case(s) with disagreeing duplicate records (newest used).", flush=True)
    if not present_any:
        raise MissingInput(f"no record for {dataset}/{model}/{variant}; run main_import_delivered and the model runs first")
    if modality == "text_only":
        for cond in conditions_of(cfg, dataset):
            if f"pa_{cond}" not in out.columns and cond != "original":
                out[f"pa_{cond}"] = out["pa_original"]
                out[f"cls_{cond}"] = out["cls_original"]
                out[f"conf_{cond}"] = out["conf_original"]
    return out


def answered(s: pd.Series) -> pd.Series:
    return s.isin([0.0, 1.0])


def has_condition(df: pd.DataFrame, cond: str) -> bool:
    return f"pa_{cond}" in df.columns and answered(df[f"pa_{cond}"]).any()


def scope_mask(df: pd.DataFrame, scope: str) -> pd.Series:
    if scope == "pooled":
        return pd.Series(True, index=df.index)
    if scope == "image_necessary":
        return df["image_necessary"].fillna(False).astype(bool)
    if scope == "finding_presence":
        return df["source"].isin(["ms_cxr", "mimic_cxr", "chexpert"])
    if scope.startswith("rexerr_"):
        return (df["source"] == "rexerr") & (df["rexerr_class"] == scope[len("rexerr_"):])
    return df["source"] == scope


def outcome_frame(df: pd.DataFrame, mask: pd.Series, value: pd.Series) -> pd.DataFrame:
    sub = df.loc[mask, ["case_id", "cluster"]].copy()
    sub["value"] = value.loc[mask].astype(float).values
    return sub.reset_index(drop=True)


def acc_outcomes(df: pd.DataFrame, scope: str = "pooled", cond: str = "original", nonanswer_wrong: bool = False) -> pd.DataFrame:
    pa = df[f"pa_{cond}"]
    ok = scope_mask(df, scope) & df["gt"].isin([0, 1])
    if nonanswer_wrong:
        ok = ok & df[f"cls_{cond}"].notna()
        return outcome_frame(df, ok, (pa == df["gt"]))
    ok = ok & answered(pa)
    return outcome_frame(df, ok, (pa == df["gt"]))


def sens_outcomes(df: pd.DataFrame, scope: str = "pooled", cond: str = "original") -> pd.DataFrame:
    pa = df[f"pa_{cond}"]
    ok = scope_mask(df, scope) & (df["gt"] == 1) & answered(pa)
    return outcome_frame(df, ok, (pa == 1))


def spec_outcomes(df: pd.DataFrame, scope: str = "pooled", cond: str = "original") -> pd.DataFrame:
    pa = df[f"pa_{cond}"]
    ok = scope_mask(df, scope) & (df["gt"] == 0) & answered(pa)
    return outcome_frame(df, ok, (pa == 0))


def yes_outcomes(df: pd.DataFrame, scope: str = "pooled", cond: str = "original") -> pd.DataFrame:
    pa = df[f"pa_{cond}"]
    ok = scope_mask(df, scope) & answered(pa)
    return outcome_frame(df, ok, (pa == 1))


def change_outcomes(df: pd.DataFrame, cond: str, scope: str = "pooled", correct_only: bool = True) -> pd.DataFrame:
    if f"pa_{cond}" not in df.columns:
        return outcome_frame(df, pd.Series(False, index=df.index), df["gt"] * 0.0)
    pa0, pa1 = df["pa_original"], df[f"pa_{cond}"]
    ok = scope_mask(df, scope) & answered(pa0) & answered(pa1) & df["gt"].isin([0, 1])
    if correct_only:
        ok = ok & (pa0 == df["gt"])
    return outcome_frame(df, ok, (pa1 != pa0))


def same_outcomes(df: pd.DataFrame, cond: str, scope: str = "pooled", correct_only: bool = True) -> pd.DataFrame:
    o = change_outcomes(df, cond, scope, correct_only)
    o["value"] = 1.0 - o["value"]
    return o


def swapped_acc_outcomes(df: pd.DataFrame, cond: str, scope: str = "pooled") -> pd.DataFrame:
    if f"pa_{cond}" not in df.columns:
        return outcome_frame(df, pd.Series(False, index=df.index), df["gt"] * 0.0)
    pa1 = df[f"pa_{cond}"]
    target = df["gt"] if cond == "swap" else 1 - df["gt"]
    ok = scope_mask(df, scope) & answered(pa1) & df["gt"].isin([0, 1])
    return outcome_frame(df, ok, (pa1 == target))


def cgr_outcomes(df: pd.DataFrame, scope: str = "ms_cxr") -> pd.DataFrame:
    if "pa_target_mask" not in df.columns:
        return outcome_frame(df, pd.Series(False, index=df.index), df["gt"] * 0.0)
    pa0, pa1 = df["pa_original"], df["pa_target_mask"]
    ok = scope_mask(df, scope) & (df["gt"] == 1) & (pa0 == 1) & answered(pa1)
    return outcome_frame(df, ok, (pa1 != pa0))


def is_outcomes(df: pd.DataFrame, mask_cond: str = "irrelevant_mask", scope: str = "ms_cxr", correct_only: bool = True) -> pd.DataFrame:
    return same_outcomes(df, mask_cond, scope, correct_only)


def removal_agreement_outcomes(df: pd.DataFrame, cond: str, scope: str = "pooled") -> pd.DataFrame:
    return same_outcomes(df, cond, scope, True)


def nonanswer_counts(df: pd.DataFrame, cond: str, scope: str = "pooled") -> Dict[str, int]:
    col = f"cls_{cond}"
    if col not in df.columns:
        return {}
    sub = df.loc[scope_mask(df, scope) & df[col].notna(), col]
    return {k: int(v) for k, v in sub.value_counts().items()}


def metric(o: pd.DataFrame, prefix: str = "metric") -> Dict:
    if len(o) == 0:
        return _pack(prefix, np.nan, np.nan, np.nan, np.nan, 0, True)
    return report_metric(o["value"].values, prefix=prefix, cluster_ids=o["cluster"].values)


def bootstrap_statistic(arrays: Sequence[np.ndarray], cluster_ids: np.ndarray, stat_fn: Callable,
                        n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> Dict:
    arrays = [np.asarray(a, dtype=float) for a in arrays]
    n = len(arrays[0])
    if n == 0:
        return {"point": np.nan, "std": np.nan, "ci_lower": np.nan, "ci_upper": np.nan, "p_value": np.nan, "n": 0}
    point = float(stat_fn(*arrays))
    cluster_ids = np.asarray(cluster_ids)
    unique_c, inverse = np.unique(cluster_ids, return_inverse=True)
    k = len(unique_c)
    order = np.argsort(inverse, kind="stable")
    bounds = np.searchsorted(inverse[order], np.arange(k + 1))
    rng = np.random.RandomState(seed)
    boot = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        draw = rng.randint(0, k, size=k)
        idx = np.concatenate([order[bounds[c]:bounds[c + 1]] for c in draw])
        try:
            boot[b] = stat_fn(*[a[idx] for a in arrays])
        except Exception:
            boot[b] = np.nan
    std, lo, hi = _percentile_ci(boot)
    finite = boot[np.isfinite(boot)]
    p = float(np.mean(np.abs(finite - point) >= abs(point))) if finite.size else np.nan
    p = max(p, 1.0 / n_boot) if np.isfinite(p) else np.nan
    lo90 = float(np.percentile(finite, 5)) if finite.size else np.nan
    hi90 = float(np.percentile(finite, 95)) if finite.size else np.nan
    return {"point": point, "std": std, "ci_lower": lo, "ci_upper": hi, "ci90_lower": lo90, "ci90_upper": hi90,
            "p_value": p, "n": int(n)}


def balanced_accuracy(pred: np.ndarray, gt: np.ndarray) -> float:
    pos, neg = gt == 1, gt == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return np.nan
    return 0.5 * (float((pred[pos] == 1).mean()) + float((pred[neg] == 0).mean()))


def f1_yes(pred: np.ndarray, gt: np.ndarray) -> float:
    tp = float(((pred == 1) & (gt == 1)).sum())
    fp = float(((pred == 1) & (gt == 0)).sum())
    fn = float(((pred == 0) & (gt == 1)).sum())
    return np.nan if (2 * tp + fp + fn) == 0 else 2 * tp / (2 * tp + fp + fn)


def answered_pairs(df: pd.DataFrame, scope: str = "pooled", cond: str = "original") -> pd.DataFrame:
    pa = df[f"pa_{cond}"]
    ok = scope_mask(df, scope) & answered(pa) & df["gt"].isin([0, 1])
    sub = df.loc[ok, ["case_id", "cluster", "gt"]].copy()
    sub["pred"] = pa.loc[ok].astype(float).values
    return sub.reset_index(drop=True)


def statistic_metric(pairs: pd.DataFrame, stat_fn: Callable, prefix: str) -> Dict:
    if len(pairs) == 0:
        return _pack(prefix, np.nan, np.nan, np.nan, np.nan, 0, True)
    res = bootstrap_statistic([pairs["pred"].values, pairs["gt"].values], pairs["cluster"].values, stat_fn)
    return _pack(prefix, res["point"], res["std"], res["ci_lower"], res["ci_upper"], res["n"], True)


def paired_premium(df: pd.DataFrame, cond_a: str, cond_b: str, scope: str, prefix: str, correct_only: bool = True,
                   affirmative_only: bool = False) -> Dict:
    if f"pa_{cond_a}" not in df.columns or f"pa_{cond_b}" not in df.columns:
        out = _pack(prefix, np.nan, np.nan, np.nan, np.nan, 0, True)
        out["p_raw"] = np.nan
        return out
    pa0, pa_a, pa_b = df["pa_original"], df[f"pa_{cond_a}"], df[f"pa_{cond_b}"]
    ok = scope_mask(df, scope) & answered(pa0) & answered(pa_a) & answered(pa_b) & df["gt"].isin([0, 1])
    if correct_only:
        ok = ok & (pa0 == df["gt"])
    if affirmative_only:
        ok = ok & (df["gt"] == 1) & (pa0 == 1)
    a = (pa_a != pa0).loc[ok].astype(float).values
    b = (pa_b != pa0).loc[ok].astype(float).values
    if len(a) == 0:
        out = _pack(prefix, np.nan, np.nan, np.nan, np.nan, 0, True)
        out["p_raw"] = np.nan
        return out
    return report_paired_diff(a, b, prefix=prefix, cluster_ids=df.loc[ok, "cluster"].values)


def category_v2(ofr: Dict, uar: Dict, ssp: Dict, cfg: Dict) -> str:
    rule = cfg["stats"]["category_rule"]
    n_min = int(cfg["stats"]["min_informative_cases"])
    lo = ssp.get("ssp_ci_low_raw", np.nan)
    if np.isfinite(lo) and lo > 0 and ssp.get("ssp_mean_raw", 0) > 0:
        return "uses_image"
    if (ofr["n"] >= n_min and uar["n"] >= n_min and ofr["ofr_mean_raw"] <= float(rule["ignores_image_ofr_max"]) / 100.0
            and uar["uar_mean_raw"] >= float(rule["ignores_image_uar_min"]) / 100.0):
        return "ignores_image"
    return "unstable"


def category_submitted(cgr: Dict, uar: Dict, is_: Dict, cfg: Dict, unstable_below: Optional[float] = None,
                       uses_min: Optional[float] = None) -> str:
    rule = cfg["stats"]["submitted_category_rule"]
    unstable_below = float(rule["unstable_is_below"]) if unstable_below is None else float(unstable_below)
    uses_min = float(rule["uses_image_is_min"]) if uses_min is None else float(uses_min)
    cgr_pt, uar_pt, is_pt = fmt_pct(cgr["cgr_mean_raw"]), fmt_pct(uar["uar_mean_raw"]), fmt_pct(is_["is_mean_raw"])
    if not all(np.isfinite(v) for v in (cgr_pt, uar_pt, is_pt)):
        return "unassigned"
    if cgr_pt == 0 and uar_pt == 100 and is_pt == 100:
        return "ignores_image"
    if is_pt < unstable_below:
        return "unstable"
    if cgr["cgr_ci_low_raw"] > 0 and is_pt >= uses_min:
        return "uses_image"
    return "unassigned"


def tost_equivalent(diff: Dict, margin_points: float, prefix: str) -> Dict:
    lo90, hi90 = diff.get(f"{prefix}_ci90_low_raw", np.nan), diff.get(f"{prefix}_ci90_high_raw", np.nan)
    m = margin_points / 100.0
    ok = np.isfinite(lo90) and np.isfinite(hi90) and lo90 > -m and hi90 < m
    return {"tost_margin_points": margin_points, "tost_ci90_low_pct": fmt_pct(lo90), "tost_ci90_high_pct": fmt_pct(hi90),
            "tost_equivalent": bool(ok)}


def paired_diff_with_ci90(a: np.ndarray, b: np.ndarray, cluster_ids: np.ndarray, prefix: str) -> Dict:
    out = report_paired_diff(a, b, prefix=prefix, cluster_ids=cluster_ids)
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if len(a):
        mean_a, mean_b = _cluster_bootstrap_means([a, b], np.asarray(cluster_ids), N_BOOT, BOOT_SEED)
        boot = mean_a - mean_b
        out[f"{prefix}_ci90_low_raw"] = float(np.percentile(boot, 5))
        out[f"{prefix}_ci90_high_raw"] = float(np.percentile(boot, 95))
    else:
        out[f"{prefix}_ci90_low_raw"] = np.nan
        out[f"{prefix}_ci90_high_raw"] = np.nan
    return out


def confidence_informative(conf: np.ndarray, cfg: Dict) -> bool:
    conf = conf[np.isfinite(conf)]
    if len(conf) == 0:
        return False
    return float(np.mean((conf == 0.0) | (conf == 1.0))) < float(cfg["stats"]["confidence_degenerate_share"])


def brier(conf: np.ndarray, gt: np.ndarray) -> float:
    p_correct = np.where(gt == 1, conf, 1.0 - conf)
    return float(((p_correct - 1.0) ** 2).mean())


def ece(conf: np.ndarray, pred: np.ndarray, gt: np.ndarray, n_bins: int) -> float:
    correct = (pred == gt).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for i in range(n_bins):
        m = (conf >= edges[i]) & ((conf < edges[i + 1]) if i < n_bins - 1 else (conf <= edges[i + 1]))
        if m.sum():
            total += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(total)


def wilson_interval(k: int, n: int) -> Tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    z = 1.959963984540054
    p = k / n
    denominator = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denominator
    half = z * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denominator
    return max(0.0, center - half), min(1.0, center + half)
