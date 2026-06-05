"""
reader_study/analyze_reader_studies.py
Created May 25, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia/
"""

import os
import shutil
from typing import Dict, Optional

import numpy as np
import pandas as pd




def _read_csv(path: str) -> pd.DataFrame:
    """Read a CSV with UTF-8, falling back to latin-1 for files saved by
    Mac/Windows editors that embed non-UTF-8 bytes (e.g. curly quotes)."""
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")

from config.serde import read_config
from Inference.stats_utils import (
    bootstrap_proportion,
    paired_bootstrap_diff,
    N_BOOT,
)



def _normalize_yes_no(s) -> int:
    """Convert reader-entered 'Yes'/'No' (any case) to int 1/0. Returns -1 on bad input."""
    if pd.isna(s):
        return -1
    t = str(s).strip().lower()
    if t in ("yes", "y", "1", "true"):
        return 1
    if t in ("no", "n", "0", "false"):
        return 0
    return -1


def _normalize_rating(s) -> str:
    """Normalize Task A ratings to canonical strings."""
    if pd.isna(s):
        return ""
    t = str(s).strip().lower()
    if t.startswith("acc"):  return "Accurate"
    if t.startswith("par"):  return "Partial"
    if t.startswith("inac"): return "Inaccurate"
    if t.startswith("can"):  return "Cannot tell"
    return str(s).strip()


def _ground_truth_for_cases(manifest: pd.DataFrame,
                              case_ids: pd.Series) -> pd.Series:
    """Look up ground_truth label from manifest for each case_id."""
    gt_map = manifest.set_index("case_id")["label"]
    return case_ids.map(gt_map)


def _boot_summary(values: np.ndarray, n_boot: int = N_BOOT,
                   prefix: str = "") -> dict:
    """Wraps bootstrap_proportion and prepends prefix to keys."""
    b = bootstrap_proportion(values, n_boot=n_boot)
    return {
        f"{prefix}point":    b["point"],
        f"{prefix}std":      b["std"],
        f"{prefix}ci_lower": b["ci_lower"],
        f"{prefix}ci_upper": b["ci_upper"],
        f"{prefix}n":        b["n"],
    }


def _wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple:
    """Wilson 95% CI for a binomial proportion."""
    if n == 0:
        return (np.nan, np.nan)
    from math import sqrt
    z = 1.959963984540054  # z(0.975)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return (max(0.0, center - half), min(1.0, center + half))



class TaskAAnalyzer:
    """
    Reads Task A ratings.
    Outputs distribution of ratings overall and per finding (with Wilson CI
    on the Accurate fraction).
    """

    def __init__(self, task_dir: str, manifest_csv: str):
        self.task_dir   = task_dir
        self.manifest   = _read_csv(manifest_csv)
        self.cases_csv  = os.path.join(task_dir, "cases.csv")
        self.mapping_csv = os.path.join(task_dir, "case_mapping.csv")

    def analyze(self) -> dict:
        if not os.path.exists(self.cases_csv):
            return {}
        ratings = _read_csv(self.cases_csv)
        ratings["rating_norm"] = ratings["rating"].map(_normalize_rating)
        ratings = ratings[ratings["rating_norm"] != ""]
        if len(ratings) == 0:
            print("[TaskA] No ratings entered yet.")
            return {}

        # Merge with case mapping to get true case_ids and findings.
        # cases.csv already contains a "finding" column; drop it first so the
        # merge doesn't produce finding_x / finding_y suffixes.
        mapping = _read_csv(self.mapping_csv)
        if "finding" in ratings.columns:
            ratings = ratings.drop(columns=["finding"])
        merged = ratings.merge(mapping[["display_id", "case_id", "finding"]],
                                 on="display_id", how="left")

        overall_counts = merged["rating_norm"].value_counts().to_dict()
        n_total = sum(overall_counts.values())
        accurate_frac = overall_counts.get("Accurate", 0) / n_total if n_total > 0 else float("nan")
        wl, wu = _wilson_ci(overall_counts.get("Accurate", 0), n_total)

        per_finding = []
        for f, grp in merged.groupby("finding"):
            n_f = len(grp)
            k_f = (grp["rating_norm"] == "Accurate").sum()
            lo, hi = _wilson_ci(k_f, n_f)
            per_finding.append({
                "finding":           f,
                "n":                 n_f,
                "n_accurate":        int(k_f),
                "frac_accurate":     k_f / n_f,
                "wilson_ci_lower":   lo,
                "wilson_ci_upper":   hi,
                "n_partial":         int((grp["rating_norm"] == "Partial").sum()),
                "n_inaccurate":      int((grp["rating_norm"] == "Inaccurate").sum()),
                "n_cannot_tell":     int((grp["rating_norm"] == "Cannot tell").sum()),
            })
        per_finding_df = pd.DataFrame(per_finding)

        return {
            "summary_row": {
                "task":                       "A_box_validation",
                "n_rated":                    n_total,
                "n_accurate":                 overall_counts.get("Accurate",   0),
                "n_partial":                  overall_counts.get("Partial",    0),
                "n_inaccurate":               overall_counts.get("Inaccurate", 0),
                "n_cannot_tell":              overall_counts.get("Cannot tell", 0),
                "frac_accurate":              accurate_frac,
                "frac_accurate_wilson_lower": wl,
                "frac_accurate_wilson_upper": wu,
            },
            "per_finding": per_finding_df,
        }



class HumanReadingAnalyzer:
    """
    Computes human metrics from a filled Task B or Task C cases.csv:
        accuracy, CGR (target_mask flips), irrelevant_stable (irrelevant
        mask does NOT flip), sensitivity, specificity.
    Each is reported with bootstrap mean, std, and 95% CI.
    Also produces per-finding breakdowns and confidence calibration.
    """

    def __init__(self, task_dir: str, manifest_csv: str,
                  reader_label: str = "reader1",
                  mapping_csv: str = None):
        self.task_dir     = task_dir
        self.manifest     = _read_csv(manifest_csv)
        self.reader_label = reader_label
        self.cases_csv    = os.path.join(task_dir, "cases.csv")
        # Allow an external mapping_csv (e.g. Task C reuses Task B's mapping)
        self.mapping_csv  = mapping_csv if mapping_csv is not None                             else os.path.join(task_dir, "case_mapping.csv")

    def _load_merged(self) -> Optional[pd.DataFrame]:
        if not os.path.exists(self.cases_csv) or not os.path.exists(self.mapping_csv):
            return None
        ratings = _read_csv(self.cases_csv)
        mapping = _read_csv(self.mapping_csv)
        # Identify shared non-key columns and drop from ratings so merging never
        # produces _x / _y suffixes (mapping is authoritative for all metadata).
        join_cols = ["display_id", "session"] if "session" in mapping.columns else ["display_id"]
        shared_meta = [c for c in mapping.columns
                       if c not in join_cols and c in ratings.columns]
        ratings_clean = ratings.drop(columns=shared_meta)
        merged = ratings_clean.merge(mapping, on=join_cols, how="left")
        # Some mapping CSVs may not carry 'session'; fall back to display_id only
        if "case_id" not in merged.columns or merged["case_id"].isna().all():
            ratings_clean2 = ratings.drop(columns=[c for c in mapping.columns
                                                     if c != "display_id" and c in ratings.columns])
            merged = ratings_clean2.merge(mapping, on="display_id", how="left")
        merged["answer_int"] = merged["answer"].map(_normalize_yes_no)
        merged = merged[merged["answer_int"].isin([0, 1])].copy()
        merged["confidence_num"] = pd.to_numeric(
            merged["confidence"], errors="coerce"
        )
        # Use finding from the merged mapping (always present)
        # Cases came in via MS-CXR positives, so ground_truth = 1
        merged["ground_truth"] = 1
        return merged

    def _outcomes_pivot(self, merged: pd.DataFrame) -> pd.DataFrame:
        """
        Reshape long->wide: one row per case_id with columns
            answer_original, answer_target_mask, answer_irrelevant_mask
            conf_original,   conf_target_mask,   conf_irrelevant_mask
        """
        pivot_ans = merged.pivot(index="case_id", columns="condition",
                                   values="answer_int").add_prefix("answer_")
        pivot_cnf = merged.pivot(index="case_id", columns="condition",
                                   values="confidence_num").add_prefix("conf_")
        pivot_fnd = merged.groupby("case_id")["finding"].first().rename("finding")
        out = pd.concat([pivot_ans, pivot_cnf, pivot_fnd], axis=1).reset_index()
        out["ground_truth"] = 1
        return out

    @staticmethod
    def _accuracy_outcomes(piv: pd.DataFrame) -> np.ndarray:
        sub = piv[piv["answer_original"].isin([0, 1])]
        if len(sub) == 0:
            return np.array([])
        return (sub["answer_original"] == sub["ground_truth"]).astype(float).values

    @staticmethod
    def _cgr_outcomes(piv: pd.DataFrame) -> np.ndarray:
        sub = piv[
            (piv["answer_original"] == 1) &
            (piv["answer_target_mask"].isin([0, 1]))
        ]
        if len(sub) == 0:
            return np.array([])
        return (sub["answer_target_mask"] != sub["answer_original"]).astype(float).values

    @staticmethod
    def _irr_outcomes(piv: pd.DataFrame) -> np.ndarray:
        sub = piv[
            (piv["answer_original"] == 1) &
            (piv["answer_irrelevant_mask"].isin([0, 1]))
        ]
        if len(sub) == 0:
            return np.array([])
        return (sub["answer_irrelevant_mask"] == sub["answer_original"]).astype(float).values

    @staticmethod
    def _sensitivity_outcomes(piv: pd.DataFrame) -> np.ndarray:
        sub = piv[piv["answer_original"].isin([0, 1])]
        if len(sub) == 0:
            return np.array([])
        return (sub["answer_original"] == 1).astype(float).values

    # Specificity: undefined here because all cases are positives. Reported as NaN.

    def _per_finding(self, piv: pd.DataFrame, kind: str) -> pd.DataFrame:
        rows = []
        for finding, grp in piv.groupby("finding"):
            if kind == "accuracy":
                vals = self._accuracy_outcomes(grp)
            elif kind == "CGR":
                vals = self._cgr_outcomes(grp)
            elif kind == "irrelevant_stable":
                vals = self._irr_outcomes(grp)
            else:
                continue
            if len(vals) == 0:
                continue
            b = bootstrap_proportion(vals)
            rows.append({
                "finding": finding, "metric": kind,
                "point": b["point"], "std": b["std"],
                "ci_lower": b["ci_lower"], "ci_upper": b["ci_upper"], "n": b["n"],
            })
        return pd.DataFrame(rows)

    def _confidence_calibration(self, piv: pd.DataFrame) -> dict:
        """Mean radiologist confidence on correct vs incorrect."""
        correct = piv[
            piv["answer_original"].isin([0, 1]) &
            (piv["answer_original"] == piv["ground_truth"])
        ]
        wrong = piv[
            piv["answer_original"].isin([0, 1]) &
            (piv["answer_original"] != piv["ground_truth"])
        ]
        return {
            "conf_correct_mean":     float(correct["conf_original"].mean()) if len(correct) else float("nan"),
            "conf_incorrect_mean":   float(wrong["conf_original"].mean()) if len(wrong) else float("nan"),
            "n_correct":             len(correct),
            "n_incorrect":           len(wrong),
        }

    def analyze(self) -> dict:
        merged = self._load_merged()
        if merged is None or len(merged) == 0:
            print(f"[{self.reader_label}] No ratings found in {self.task_dir}.")
            return {}

        piv = self._outcomes_pivot(merged)

        acc = _boot_summary(self._accuracy_outcomes(piv),    prefix="accuracy_")
        cgr = _boot_summary(self._cgr_outcomes(piv),         prefix="CGR_")
        irr = _boot_summary(self._irr_outcomes(piv),         prefix="irrelevant_stable_")
        sen = _boot_summary(self._sensitivity_outcomes(piv), prefix="sensitivity_")

        calib = self._confidence_calibration(piv)

        summary_row = {
            "reader":     self.reader_label,
            "n_cases":    len(piv),
            **acc, **cgr, **irr, **sen, **calib,
        }

        per_finding = pd.concat([
            self._per_finding(piv, "accuracy"),
            self._per_finding(piv, "CGR"),
            self._per_finding(piv, "irrelevant_stable"),
        ], ignore_index=True)
        per_finding["reader"] = self.reader_label

        return {
            "summary_row": summary_row,
            "per_finding": per_finding,
            "merged":      merged,
            "pivoted":     piv,
        }



class HumanVsModelComparator:
    """
    Paired bootstrap comparisons between the human reader (Reader 1) and
    every available model, on the SAME 80 cases used in Task B.

    Computes diff in:
        accuracy           (on original condition)
        CGR                (target_mask flip rate)
        irrelevant_stable  (irrelevant_mask non-flip rate)

    For each comparison: paired bootstrap mean, std, 95% CI, two-sided
    p-value. BH FDR correction within each (metric) family across models.
    """

    def __init__(self, human_pivoted: pd.DataFrame, results_dir: str,
                  manifest_csv: str):
        self.human   = human_pivoted
        self.results_dir = results_dir
        self.manifest    = _read_csv(manifest_csv)

    def _model_outcomes(self, model_name: str) -> Optional[dict]:
        """Returns per-case outcome Series for accuracy, CGR, irrelevant_stable."""
        out = {}
        for cond, fname in [("original", "original.csv"),
                              ("target_mask", "target_mask.csv"),
                              ("irrelevant_mask", "irrelevant_mask.csv")]:
            path = os.path.join(self.results_dir, model_name, fname)
            if not os.path.exists(path):
                return None
            df = _read_csv(path)
            df = df[df["parsed_answer"].isin([0, 1])]
            out[cond] = df.set_index("case_id")["parsed_answer"]
        return out

    def compare_all_models(self) -> pd.DataFrame:
        rows = []
        case_ids = list(self.human["case_id"])
        model_dirs = sorted([
            d for d in os.listdir(self.results_dir)
            if os.path.isdir(os.path.join(self.results_dir, d))
        ])

        human_idx = self.human.set_index("case_id")

        for model in model_dirs:
            mout = self._model_outcomes(model)
            if mout is None:
                continue

            for metric_name in ["accuracy", "CGR", "irrelevant_stable"]:
                human_vals, model_vals = [], []
                for cid in case_ids:
                    h = human_idx.loc[cid]
                    # Compute outcome per metric for this single case
                    if metric_name == "accuracy":
                        if not h["answer_original"] in (0, 1):
                            continue
                        h_v = int(h["answer_original"] == h["ground_truth"])
                        if cid not in mout["original"].index:
                            continue
                        m_pred = mout["original"].loc[cid]
                        m_v    = int(m_pred == h["ground_truth"])
                    elif metric_name == "CGR":
                        # Only eligible on positive cases the actor got right on original
                        if h["answer_original"] != 1 or not h["answer_target_mask"] in (0, 1):
                            continue
                        h_v = int(h["answer_target_mask"] != h["answer_original"])
                        if cid not in mout["original"].index or cid not in mout["target_mask"].index:
                            continue
                        m_orig = mout["original"].loc[cid]
                        m_tm   = mout["target_mask"].loc[cid]
                        if m_orig != 1:    # model not correct on original; skip for paired CGR
                            continue
                        m_v = int(m_tm != m_orig)
                    elif metric_name == "irrelevant_stable":
                        if h["answer_original"] != 1 or not h["answer_irrelevant_mask"] in (0, 1):
                            continue
                        h_v = int(h["answer_irrelevant_mask"] == h["answer_original"])
                        if cid not in mout["original"].index or cid not in mout["irrelevant_mask"].index:
                            continue
                        m_orig = mout["original"].loc[cid]
                        m_ir   = mout["irrelevant_mask"].loc[cid]
                        if m_orig != 1:
                            continue
                        m_v = int(m_ir == m_orig)
                    human_vals.append(h_v)
                    model_vals.append(m_v)

                if len(human_vals) < 10:
                    continue
                hv = np.asarray(human_vals, dtype=float)
                mv = np.asarray(model_vals, dtype=float)
                pb = paired_bootstrap_diff(hv, mv)
                rows.append({
                    "model":           model,
                    "metric":          metric_name,
                    "human_value":     float(hv.mean()),
                    "model_value":     float(mv.mean()),
                    "diff":            pb["point"],
                    "diff_std":        pb["std"],
                    "diff_ci_lower":   pb["ci_lower"],
                    "diff_ci_upper":   pb["ci_upper"],
                    "p_value":         pb["p_value"],
                    "n_paired":        pb["n"],
                    "p_value_fdr":     float("nan"),
                })

        out = pd.DataFrame(rows)
        if out.empty:
            return out
        from Inference.stats_utils import bh_fdr
        for metric, grp in out.groupby("metric"):
            adj = bh_fdr(grp["p_value"].values)
            out.loc[grp.index, "p_value_fdr"] = adj
        out = out.sort_values(["metric", "diff"], ascending=[True, False]).reset_index(drop=True)
        return out



class InterReaderAgreement:
    """
    Cohen's kappa (unweighted, for Yes/No) and quadratic-weighted kappa
    (for 1-5 confidence) between Reader 1 and Reader 2 on the shared
    Task B/C displays. Also percent agreement.

    Bootstrap mean, std, and 95% CI for kappa via case-level resampling.
    """

    def __init__(self, reader1_merged: pd.DataFrame,
                  reader2_merged: pd.DataFrame, n_boot: int = N_BOOT):
        self.r1 = reader1_merged
        self.r2 = reader2_merged
        self.n_boot = n_boot

    @staticmethod
    def _cohens_kappa(a: np.ndarray, b: np.ndarray) -> float:
        """Unweighted Cohen's kappa for binary labels (0/1)."""
        if len(a) == 0 or len(a) != len(b):
            return float("nan")
        n = len(a)
        agree   = float((a == b).mean())
        p_a1    = float((a == 1).mean()); p_a0 = 1 - p_a1
        p_b1    = float((b == 1).mean()); p_b0 = 1 - p_b1
        chance  = p_a0 * p_b0 + p_a1 * p_b1
        if chance >= 1.0:
            return float("nan")
        return (agree - chance) / (1.0 - chance)

    @staticmethod
    def _weighted_kappa_quadratic(a: np.ndarray, b: np.ndarray,
                                    categories: list) -> float:
        """Quadratic-weighted kappa on ordered categorical (e.g., 1..5)."""
        if len(a) == 0 or len(a) != len(b):
            return float("nan")
        K = len(categories)
        cat_to_idx = {c: i for i, c in enumerate(categories)}
        # Observed matrix
        O = np.zeros((K, K), dtype=float)
        for ai, bi in zip(a, b):
            if ai not in cat_to_idx or bi not in cat_to_idx:
                continue
            O[cat_to_idx[ai], cat_to_idx[bi]] += 1
        if O.sum() == 0:
            return float("nan")
        O = O / O.sum()
        # Marginals
        a_marg = O.sum(axis=1); b_marg = O.sum(axis=0)
        E = np.outer(a_marg, b_marg)
        # Quadratic weights
        W = np.zeros((K, K), dtype=float)
        for i in range(K):
            for j in range(K):
                W[i, j] = ((i - j) ** 2) / ((K - 1) ** 2)
        num = (W * O).sum(); den = (W * E).sum()
        if den == 0:
            return float("nan")
        return 1.0 - num / den

    def _bootstrap_kappa(self, a: np.ndarray, b: np.ndarray,
                          weighted: bool = False,
                          categories: list = None) -> dict:
        n = len(a)
        if n == 0:
            return {"point": float("nan"), "std": float("nan"),
                    "ci_lower": float("nan"), "ci_upper": float("nan"), "n": 0}
        if weighted:
            point = self._weighted_kappa_quadratic(a, b, categories)
        else:
            point = self._cohens_kappa(a, b)
        rng = np.random.RandomState(0)
        boots = []
        for _ in range(self.n_boot):
            idx = rng.randint(0, n, size=n)
            if weighted:
                k = self._weighted_kappa_quadratic(a[idx], b[idx], categories)
            else:
                k = self._cohens_kappa(a[idx], b[idx])
            if not np.isnan(k):
                boots.append(k)
        boots = np.array(boots) if boots else np.array([np.nan])
        return {
            "point":    float(point),
            "std":      float(np.nanstd(boots, ddof=1)),
            "ci_lower": float(np.nanpercentile(boots, 2.5)),
            "ci_upper": float(np.nanpercentile(boots, 97.5)),
            "n":        int(n),
        }

    def compute(self) -> dict:
        # Match on display_id (Task C images are aligned to Task B displays)
        m1 = self.r1[["display_id", "answer_int", "confidence_num"]].rename(
            columns={"answer_int": "ans_r1", "confidence_num": "conf_r1"}
        )
        m2 = self.r2[["display_id", "answer_int", "confidence_num"]].rename(
            columns={"answer_int": "ans_r2", "confidence_num": "conf_r2"}
        )
        merged = m1.merge(m2, on="display_id", how="inner")
        merged = merged[
            merged["ans_r1"].isin([0, 1]) & merged["ans_r2"].isin([0, 1])
        ]
        if len(merged) == 0:
            return {}

        a_ans = merged["ans_r1"].values.astype(int)
        b_ans = merged["ans_r2"].values.astype(int)
        kappa = self._bootstrap_kappa(a_ans, b_ans, weighted=False)

        agree     = float((a_ans == b_ans).mean())
        agree_b   = bootstrap_proportion((a_ans == b_ans).astype(float))

        weighted_kappa = {"point": float("nan"), "std": float("nan"),
                           "ci_lower": float("nan"), "ci_upper": float("nan"),
                           "n": 0}
        if merged["conf_r1"].notna().all() and merged["conf_r2"].notna().all():
            a_c = merged["conf_r1"].astype(int).values
            b_c = merged["conf_r2"].astype(int).values
            categories = sorted(set(np.concatenate([a_c, b_c])))
            if len(categories) >= 2:
                weighted_kappa = self._bootstrap_kappa(
                    a_c, b_c, weighted=True, categories=categories
                )

        return {
            "n_shared_displays":            len(merged),
            "percent_agreement":            agree,
            "percent_agreement_std":        agree_b["std"],
            "percent_agreement_ci_lower":   agree_b["ci_lower"],
            "percent_agreement_ci_upper":   agree_b["ci_upper"],
            "cohens_kappa":                 kappa["point"],
            "cohens_kappa_std":             kappa["std"],
            "cohens_kappa_ci_lower":        kappa["ci_lower"],
            "cohens_kappa_ci_upper":        kappa["ci_upper"],
            "weighted_kappa_confidence":           weighted_kappa["point"],
            "weighted_kappa_confidence_std":       weighted_kappa["std"],
            "weighted_kappa_confidence_ci_lower":  weighted_kappa["ci_lower"],
            "weighted_kappa_confidence_ci_upper":  weighted_kappa["ci_upper"],
        }



class TaskDAnalyzer:
    """
    Reads Task D ratings and computes failure-mode distribution per model.
    Decodes model_label -> source_model via the saved key file.
    Each category fraction is reported with Wilson 95% CI.
    """

    def __init__(self, task_dir: str):
        self.task_dir   = task_dir
        self.cases_csv  = os.path.join(task_dir, "cases.csv")
        self.key_csv    = os.path.join(task_dir, "model_label_key.csv")
        self.mapping_csv = os.path.join(task_dir, "case_mapping.csv")

    def analyze(self) -> pd.DataFrame:
        if not os.path.exists(self.cases_csv):
            return pd.DataFrame()
        ratings = _read_csv(self.cases_csv)
        if "error_category" not in ratings.columns or ratings["error_category"].isna().all():
            print("[TaskD] No error categories entered yet.")
            return pd.DataFrame()
        if os.path.exists(self.key_csv):
            key = _read_csv(self.key_csv)
        else:
            key = pd.DataFrame(columns=["model_label", "source_model"])
        merged = ratings.merge(key, on="model_label", how="left")

        rows = []
        for model, grp in merged.groupby("source_model"):
            counts = grp["error_category"].value_counts().to_dict()
            n = sum(counts.values())
            for cat in ["Ambiguous", "Image quality", "Plausible confounder",
                         "Clear model failure", "Other"]:
                k = counts.get(cat, 0)
                lo, hi = _wilson_ci(k, n)
                rows.append({
                    "model":          model,
                    "category":       cat,
                    "n":              n,
                    "count":          int(k),
                    "fraction":       k / n if n > 0 else float("nan"),
                    "wilson_lower":   lo,
                    "wilson_upper":   hi,
                })
            # Also: mean radiologist confidence per model
            rconf = pd.to_numeric(grp["radiologist_confidence"], errors="coerce")
            rows.append({
                "model":   model,
                "category": "mean_radiologist_confidence",
                "n":        int(rconf.notna().sum()),
                "count":    np.nan,
                "fraction": float(rconf.mean()) if rconf.notna().any() else float("nan"),
                "wilson_lower": float("nan"),
                "wilson_upper": float("nan"),
            })
        return pd.DataFrame(rows)



def analyze_all(cfg_path: str, reader_study_root: str,
                 reader1_subdir: str = "reader1_filled",
                 reader2_subdir: str = "reader2_filled"):
    """
    Args:
        cfg_path           : path to config.yaml
        reader_study_root  : root containing reader1_filled/, reader2_filled/
                             (each with task_A_box_validation/, etc.)
    """
    params       = read_config(cfg_path)
    cfg          = params["CausalAudit"]
    manifest_csv = cfg["manifest_csv"]
    results_dir  = cfg["results_dir"]
    out_dir      = os.path.join(reader_study_root, "analysis")
    used_dir     = os.path.join(out_dir, "already_used")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(used_dir, exist_ok=True)

    r1_root = os.path.join(reader_study_root, reader1_subdir)
    r2_root = os.path.join(reader_study_root, reader2_subdir)

    summary_rows  = []
    per_finding_all = []

    task_a = TaskAAnalyzer(
        os.path.join(r1_root, "task_A_box_validation"), manifest_csv
    ).analyze()
    if task_a:
        sr = task_a["summary_row"]
        sr["reader"] = "reader1"
        summary_rows.append(sr)
        task_a["per_finding"].to_csv(
            os.path.join(used_dir, "task_A_per_finding.csv"), index=False
        )

    r1_b = HumanReadingAnalyzer(
        os.path.join(r1_root, "task_B_finding_presence"),
        manifest_csv, reader_label="reader1"
    ).analyze()
    if r1_b:
        r1_b["summary_row"]["task"] = "B_finding_presence"
        summary_rows.append(r1_b["summary_row"])
        per_finding_all.append(r1_b["per_finding"])

    # Task C uses the same case mapping as Task B (shared display_ids and conditions)
    task_b_mapping = os.path.join(r1_root, "task_B_finding_presence", "case_mapping.csv")
    r2_c = HumanReadingAnalyzer(
        os.path.join(r2_root, "task_C_reader2_agreement"),
        manifest_csv, reader_label="reader2",
        mapping_csv=task_b_mapping,
    ).analyze()
    if r2_c:
        r2_c["summary_row"]["task"] = "C_finding_presence_reader2"
        summary_rows.append(r2_c["summary_row"])
        per_finding_all.append(r2_c["per_finding"])

    paired_rows = pd.DataFrame()
    if r1_b:
        comparator = HumanVsModelComparator(
            r1_b["pivoted"], results_dir, manifest_csv
        )
        paired_rows = comparator.compare_all_models()
        if not paired_rows.empty:
            paired_rows.to_csv(
                os.path.join(out_dir, "human_vs_model_paired.csv"), index=False
            )

    # -------- Inter-reader agreement --------
    agreement_row = {}
    if r1_b and r2_c:
        ag = InterReaderAgreement(r1_b["merged"], r2_c["merged"]).compute()
        if ag:
            agreement_row = {"task": "agreement_reader1_vs_reader2", **ag}
            summary_rows.append(agreement_row)

    # -------- Task D (Reader 2) --------
    task_d_df = TaskDAnalyzer(
        os.path.join(r2_root, "task_D_failure_taxonomy")
    ).analyze()
    if not task_d_df.empty:
        task_d_df.to_csv(
            os.path.join(out_dir, "task_D_failure_modes.csv"), index=False
        )

    if summary_rows:
        all_summary = pd.DataFrame(summary_rows)
        all_summary.to_csv(
            os.path.join(out_dir, "all_reader_metrics.csv"), index=False
        )
        print("\n=== Reader study summary ===")
        print(all_summary.to_string(index=False))

    if per_finding_all:
        pd.concat(per_finding_all, ignore_index=True).to_csv(
            os.path.join(used_dir, "per_finding_human.csv"), index=False
        )

    if task_a and "per_finding" in task_a:
        task_a["per_finding"].to_csv(
            os.path.join(used_dir, "task_A_per_finding.csv"), index=False
        )

    print(f"\nAll outputs saved to {out_dir}")






if __name__ == "__main__":
    CONFIG_PATH       = "/PATHcausal/config/config.yaml"
    READER_STUDY_ROOT = "/PATH/causal/reader_study"
    analyze_all(CONFIG_PATH, READER_STUDY_ROOT)