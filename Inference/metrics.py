"""
Inference/metrics.py
Created May 22, 2026

MetricsComputer: loads per-model result CSVs and computes the full metric suite.

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia/
"""

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import shutil as _shutil

from config.serde import read_config
from Inference.stats_utils import bootstrap_proportion, N_BOOT

import warnings
warnings.filterwarnings("ignore")




TEXT_ONLY_MODELS = {
    "MedGemma-27B-text",
    "DeepSeek-R1-7B",
}

VISION_ONLY_MODELS = {"RAD-DINO"}

# ReXErr error-type classification
REXERR_IMAGE_DEPENDENT = {
    "add_medical_device",
    "change_location",
    "change_name_of_device",
    "change_position_of_device",
    "change_severity",
    "false_negation",
    "false_prediction",
}
REXERR_TEXT_ONLY = {
    "add_typo",
    "change_to_homophone",
}
REXERR_CONTROL = {"not_applicable"}

_N_BINS_ECE = 10




def _auc_from_scores(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute AUROC via trapezoidal rule. No sklearn dependency."""
    pos_mask = labels == 1
    n_pos = pos_mask.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    idx    = np.argsort(scores)[::-1]
    labels_sorted = labels[idx]
    tpr = np.concatenate([[0], np.cumsum(labels_sorted == 1) / n_pos])
    fpr = np.concatenate([[0], np.cumsum(labels_sorted == 0) / n_neg])
    return float(np.trapz(tpr, fpr))


def _is_confidence_informative(conf: np.ndarray) -> bool:
    """
    Returns True if confidence values are not purely binary.
    Models lacking logprob access return conf = 0.0 or 1.0 based on the
    parsed answer; calibration metrics are meaningless for those.
    """
    binary_frac = float(np.mean((conf == 0.0) | (conf == 1.0)))
    return binary_frac < 0.90




class MetricsComputer:
    """
    Loads per-model result CSVs and computes the full metric suite.

    Args:
        cfg_path     : path to config.yaml
        dataset_type : 'mimic' (default) | 'chexpert'
        n_boot       : bootstrap resamples for all CIs (default 1000)
    """

    def __init__(
        self,
        cfg_path: str,
        dataset_type: str = "mimic",
        n_boot: int = N_BOOT,
    ):
        assert dataset_type in ("mimic", "chexpert")
        self.params       = read_config(cfg_path)
        self.dataset_type = dataset_type
        self.n_boot       = n_boot
        cfg = self.params["CausalAudit"]

        if dataset_type == "chexpert":
            self.results_dir  = cfg["chexpert_results_dir"]
            self.manifest_csv = cfg["chexpert_manifest_csv"]
        else:
            self.results_dir  = cfg["results_dir"]
            self.manifest_csv = cfg["manifest_csv"]

        self.manifest = pd.read_csv(self.manifest_csv)



    def _load_model_results(self, model_name: str) -> Optional[pd.DataFrame]:
        """
        Loads and merges the per-condition CSVs for one model.
        Returns a wide DataFrame indexed by case_id with columns:
            pa_{condition}, conf_{condition}
        plus case metadata joined from the manifest.
        """
        model_dir  = os.path.join(self.results_dir, model_name)
        conditions = ["original", "swap", "target_mask", "irrelevant_mask"]

        dfs = {}
        for cond in conditions:
            path = os.path.join(model_dir, f"{cond}.csv")
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path)
            df = df[["case_id", "parsed_answer", "confidence", "ground_truth"]].copy()
            df = df.rename(columns={
                "parsed_answer": f"pa_{cond}",
                "confidence":    f"conf_{cond}",
            })
            if cond != "original":
                df = df.drop(columns=["ground_truth"])
            dfs[cond] = df

        if "original" not in dfs:
            print(f"[MetricsComputer] No original.csv for {model_name}, skipping.")
            return None

        merged = dfs["original"]
        for cond in ["swap", "target_mask", "irrelevant_mask"]:
            if cond in dfs:
                merged = merged.merge(dfs[cond], on="case_id", how="left")

        manifest_cols = [
            c for c in [
                "case_id", "source", "finding", "label",
                "view", "age", "gender", "error_type", "error_present",
            ]
            if c in self.manifest.columns
        ]
        merged = merged.merge(
            self.manifest[manifest_cols], on="case_id", how="left"
        )
        merged["model_name"] = model_name
        return merged

    def _load_all_models(self) -> Dict[str, pd.DataFrame]:
        model_names = [
            d for d in os.listdir(self.results_dir)
            if os.path.isdir(os.path.join(self.results_dir, d))
        ]
        results = {}
        for m in sorted(model_names):
            df = self._load_model_results(m)
            if df is not None:
                results[m] = df
        return results



    @staticmethod
    def _cgr_outcomes(df: pd.DataFrame) -> np.ndarray:
        """
        Binary array for CGR.
          1 = answer flipped under target_mask (grounded)
          0 = did not flip

        Eligible cases: MS-CXR source, positive (GT=1), model predicted
        positive correctly (pa_original==1), target_mask result is valid.

        Restricting to positive cases is correct because MS-CXR bounding
        boxes mark the location of present findings; asking whether an
        absent-finding case (no box) flips under masking is undefined.
        """
        if "pa_target_mask" not in df.columns:
            return np.array([])
        sub = df[
            (df["source"] == "ms_cxr") &
            (df["ground_truth"] == 1) &
            (df["pa_original"] == 1) &
            (df["pa_target_mask"].isin([0, 1]))
        ]
        if len(sub) == 0:
            return np.array([])
        return (sub["pa_target_mask"] != sub["pa_original"]).astype(float).values

    @staticmethod
    def _uar_outcomes(df: pd.DataFrame) -> np.ndarray:
        """
        Binary array for UAR.
          1 = answer unchanged under swap (invariant)
          0 = answer changed

        Eligible cases: all cases where pa_original is valid, ground_truth
        is valid, model was correct, and pa_swap is valid.

        See module docstring for the correct interpretation of UAR.
        """
        if "pa_swap" not in df.columns:
            return np.array([])
        sub = df[
            (df["pa_original"].isin([0, 1])) &
            (df["ground_truth"].isin([0, 1])) &
            (df["pa_original"] == df["ground_truth"]) &
            (df["pa_swap"].isin([0, 1]))
        ]
        if len(sub) == 0:
            return np.array([])
        return (sub["pa_swap"] == sub["pa_original"]).astype(float).values

    @staticmethod
    def _irr_outcomes(df: pd.DataFrame) -> np.ndarray:
        """
        Binary array for irrelevant_mask stability (negative control).
          1 = answer unchanged under irrelevant-region mask
          0 = changed

        Eligible cases: MS-CXR source, correct predictions, irrelevant_mask
        result is valid.
        """
        if "pa_irrelevant_mask" not in df.columns:
            return np.array([])
        sub = df[
            (df["source"] == "ms_cxr") &
            (df["pa_original"].isin([0, 1])) &
            (df["ground_truth"].isin([0, 1])) &
            (df["pa_original"] == df["ground_truth"]) &
            (df["pa_irrelevant_mask"].isin([0, 1]))
        ]
        if len(sub) == 0:
            return np.array([])
        return (sub["pa_irrelevant_mask"] == sub["pa_original"]).astype(float).values

    @staticmethod
    def _accuracy_outcomes(df: pd.DataFrame) -> np.ndarray:
        """Binary array for accuracy (1=correct, 0=wrong) over parseable cases."""
        sub = df[
            (df["pa_original"].isin([0, 1])) &
            (df["ground_truth"].isin([0, 1]))
        ]
        if len(sub) == 0:
            return np.array([])
        return (sub["pa_original"] == sub["ground_truth"]).astype(float).values

    @staticmethod
    def _sensitivity_outcomes(df: pd.DataFrame) -> np.ndarray:
        """Binary array for sensitivity: TP / (TP + FN) = recall on positives."""
        sub = df[
            (df["pa_original"].isin([0, 1])) &
            (df["ground_truth"] == 1)
        ]
        if len(sub) == 0:
            return np.array([])
        return (sub["pa_original"] == 1).astype(float).values

    @staticmethod
    def _specificity_outcomes(df: pd.DataFrame) -> np.ndarray:
        """Binary array for specificity: TN / (TN + FP) = recall on negatives."""
        sub = df[
            (df["pa_original"].isin([0, 1])) &
            (df["ground_truth"] == 0)
        ]
        if len(sub) == 0:
            return np.array([])
        return (sub["pa_original"] == 0).astype(float).values



    @staticmethod
    def _cgr(df: pd.DataFrame) -> float:
        v = MetricsComputer._cgr_outcomes(df)
        return float(v.mean()) if len(v) > 0 else float("nan")

    @staticmethod
    def _uar(df: pd.DataFrame) -> float:
        v = MetricsComputer._uar_outcomes(df)
        return float(v.mean()) if len(v) > 0 else float("nan")

    @staticmethod
    def _accuracy(df: pd.DataFrame, source: str = None) -> float:
        """Point-estimate accuracy. Kept for backward compatibility."""
        sub = df.copy()
        if source:
            sub = sub[sub["source"] == source]
        v = MetricsComputer._accuracy_outcomes(sub)
        return float(v.mean()) if len(v) > 0 else float("nan")



    @staticmethod
    def _parse_rate(df: pd.DataFrame) -> float:
        """
        Fraction of all rows where pa_original is a valid answer (0 or 1).
        Rows with pa_original == -1 are parse failures.
        """
        if "pa_original" not in df.columns or len(df) == 0:
            return float("nan")
        return float((df["pa_original"].isin([0, 1])).mean())

    def _auroc(self, df: pd.DataFrame) -> dict:
        """
        AUROC from model confidence vs ground_truth.
        Returns NaN when confidence is non-informative (purely binary).
        Bootstrap CI via score resampling.
        """
        sub = df[
            (df["pa_original"].isin([0, 1])) &
            (df["ground_truth"].isin([0, 1])) &
            (df["conf_original"].notna())
        ]
        if len(sub) == 0:
            return {"point": float("nan"), "ci_lower": float("nan"),
                    "ci_upper": float("nan"), "n": 0}
        conf = sub["conf_original"].values.astype(float)
        gt   = sub["ground_truth"].values.astype(float)
        if not _is_confidence_informative(conf):
            return {"point": float("nan"), "ci_lower": float("nan"),
                    "ci_upper": float("nan"), "n": len(sub)}
        n = len(sub)
        point = _auc_from_scores(conf, gt)
        # Bootstrap CI
        rng = np.random.RandomState(0)
        idx = rng.randint(0, n, size=(self.n_boot, n))
        boot_aucs = np.array([
            _auc_from_scores(conf[idx[i]], gt[idx[i]])
            for i in range(self.n_boot)
        ])
        boot_aucs = boot_aucs[~np.isnan(boot_aucs)]
        if len(boot_aucs) == 0:
            return {"point": point, "ci_lower": float("nan"),
                    "ci_upper": float("nan"), "n": n}
        return {
            "point":    round(point, 4),
            "ci_lower": round(float(np.percentile(boot_aucs, 2.5)), 4),
            "ci_upper": round(float(np.percentile(boot_aucs, 97.5)), 4),
            "n":        n,
        }

    @staticmethod
    def _brier_score(df: pd.DataFrame) -> float:
        """
        Mean squared calibration error between P(correct) and correctness.
        For GT=1: P(correct) = conf_original.
        For GT=0: P(correct) = 1 - conf_original.
        Returns NaN when confidence is non-informative.
        """
        sub = df[
            (df["pa_original"].isin([0, 1])) &
            (df["ground_truth"].isin([0, 1])) &
            (df["conf_original"].notna())
        ]
        if len(sub) == 0:
            return float("nan")
        conf = sub["conf_original"].values.astype(float)
        gt   = sub["ground_truth"].values.astype(float)
        if not _is_confidence_informative(conf):
            return float("nan")
        # P(correct) per case
        p_correct = np.where(gt == 1, conf, 1.0 - conf)
        return float(((p_correct - 1.0) ** 2).mean())

    @staticmethod
    def _ece(df: pd.DataFrame, n_bins: int = _N_BINS_ECE) -> float:
        """
        Expected Calibration Error (equal-width bins).
        Confidence = P(Yes). Calibration is measured as |avg_conf - avg_acc|
        per bin, weighted by bin size.
        Returns NaN when confidence is non-informative.
        """
        sub = df[
            (df["pa_original"].isin([0, 1])) &
            (df["ground_truth"].isin([0, 1])) &
            (df["conf_original"].notna())
        ]
        if len(sub) == 0:
            return float("nan")
        conf    = sub["conf_original"].values.astype(float)
        correct = (sub["pa_original"] == sub["ground_truth"]).values.astype(float)
        if not _is_confidence_informative(conf):
            return float("nan")
        n   = len(conf)
        ece = 0.0
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        for i in range(n_bins):
            lo, hi  = bins[i], bins[i + 1]
            mask    = (conf >= lo) & (conf < hi)
            if i == n_bins - 1:
                mask = (conf >= lo) & (conf <= hi)
            if mask.sum() == 0:
                continue
            avg_conf = conf[mask].mean()
            avg_acc  = correct[mask].mean()
            ece += (mask.sum() / n) * abs(avg_conf - avg_acc)
        return float(ece)



    def _per_finding_cgr(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        CGR point estimate, CI, and n per finding.
        n = number of eligible cases for CGR (positive MS-CXR with valid mask).
        """
        rows = []
        for finding in sorted(df["finding"].dropna().unique()):
            sub  = df[df["finding"] == finding]
            vals = self._cgr_outcomes(sub)
            n    = len(vals)
            if n == 0:
                rows.append({"finding": finding, "CGR": float("nan"),
                              "CGR_ci_lower": float("nan"),
                              "CGR_ci_upper": float("nan"), "CGR_n": 0})
                continue
            boot = bootstrap_proportion(vals, n_boot=self.n_boot)
            rows.append({
                "finding":      finding,
                "CGR":          round(boot["point"], 4),
                "CGR_std":      round(boot["std"], 4),
                "CGR_ci_lower": round(boot["ci_lower"], 4),
                "CGR_ci_upper": round(boot["ci_upper"], 4),
                "CGR_n":        int(n),
            })
        return pd.DataFrame(rows)

    def _per_finding_uar(self, df: pd.DataFrame) -> pd.DataFrame:
        """UAR point estimate, CI, and n per finding."""
        rows = []
        for finding in sorted(df["finding"].dropna().unique()):
            sub  = df[df["finding"] == finding]
            vals = self._uar_outcomes(sub)
            n    = len(vals)
            if n == 0:
                rows.append({"finding": finding, "UAR": float("nan"),
                              "UAR_ci_lower": float("nan"),
                              "UAR_ci_upper": float("nan"), "UAR_n": 0})
                continue
            boot = bootstrap_proportion(vals, n_boot=self.n_boot)
            rows.append({
                "finding":      finding,
                "UAR":          round(boot["point"], 4),
                "UAR_std":      round(boot["std"], 4),
                "UAR_ci_lower": round(boot["ci_lower"], 4),
                "UAR_ci_upper": round(boot["ci_upper"], 4),
                "UAR_n":        int(n),
            })
        return pd.DataFrame(rows)



    def _cgr_by_view(self, df: pd.DataFrame) -> dict:
        out = {}
        for view in ["PA", "AP"]:
            sub  = df[df["view"] == view]
            vals = self._cgr_outcomes(sub)
            if len(vals) == 0:
                continue
            boot = bootstrap_proportion(vals, n_boot=self.n_boot)
            out[view] = {"CGR": boot["point"], "CGR_std": boot["std"],
                         "CGR_ci_lower": boot["ci_lower"],
                         "CGR_ci_upper": boot["ci_upper"], "n": boot["n"]}
        return out



    def _cgr_by_gender(self, df: pd.DataFrame) -> Dict[str, float]:
        return {
            g: self._cgr(df[df["gender"] == g])
            for g in ["M", "F"]
            if len(df[df["gender"] == g]) > 0
        }

    def _uar_by_gender(self, df: pd.DataFrame) -> Dict[str, float]:
        return {
            g: self._uar(df[df["gender"] == g])
            for g in ["M", "F"]
            if len(df[df["gender"] == g]) > 0
        }

    @staticmethod
    def _age_group(df: pd.DataFrame) -> pd.DataFrame:
        bins, labels = [0, 50, 70, 200], ["<50", "50-70", ">70"]
        sub = df[df["age"].notna()].copy()
        sub["age_group"] = pd.cut(sub["age"], bins=bins, labels=labels, right=False)
        return sub

    def _cgr_by_age(self, df: pd.DataFrame) -> Dict[str, float]:
        sub = self._age_group(df)
        return {
            grp: self._cgr(sub[sub["age_group"] == grp])
            for grp in ["<50", "50-70", ">70"]
            if (sub["age_group"] == grp).sum() > 0
        }

    def _uar_by_age(self, df: pd.DataFrame) -> Dict[str, float]:
        sub = self._age_group(df)
        return {
            grp: self._uar(sub[sub["age_group"] == grp])
            for grp in ["<50", "50-70", ">70"]
            if (sub["age_group"] == grp).sum() > 0
        }



    def _rexerr_breakdown(self, df: pd.DataFrame) -> dict:
        """
        For each ReXErr error type, compute the fraction of correct answers
        that change under full swap (swap sensitivity).

        High swap sensitivity = model actually uses the image to detect
        this type of error.

        Returns per-error-type values plus aggregated means for:
          - image_dependent: errors that require looking at the image
          - text_only:       errors detectable from text alone
          - control:         not_applicable (accurately described images)

        For image_dependent errors, high swap sensitivity is expected for
        genuinely grounded models and is a positive signal.
        For text_only errors, swap sensitivity should be low for all models
        (image swap is irrelevant). High swap sensitivity here indicates
        instability, not grounding.
        """
        if "pa_swap" not in df.columns:
            return {}
        sub = df[
            (df["source"] == "rexerr") &
            (df["pa_original"].isin([0, 1])) &
            (df["ground_truth"].isin([0, 1]))
        ]
        correct = sub[sub["pa_original"] == sub["ground_truth"]]

        per_type = {}
        for et in correct["error_type"].dropna().unique():
            grp     = correct[correct["error_type"] == et]
            changed = grp[
                grp["pa_swap"].isin([0, 1]) &
                (grp["pa_swap"] != grp["pa_original"])
            ]
            per_type[et] = {
                "swap_sens": len(changed) / len(grp) if len(grp) > 0 else float("nan"),
                "n":         len(grp),
            }

        def _mean_sens(types):
            vals = [
                per_type[et]["swap_sens"]
                for et in types
                if et in per_type and not np.isnan(per_type[et]["swap_sens"])
            ]
            return float(np.mean(vals)) if vals else float("nan")

        return {
            "per_type":                per_type,
            "image_dependent_mean":    _mean_sens(REXERR_IMAGE_DEPENDENT),
            "text_only_mean":          _mean_sens(REXERR_TEXT_ONLY),
            "control_mean":            _mean_sens(REXERR_CONTROL),
        }



    def _text_only_gap(self, all_results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        For each VLM, compute accuracy - best_text_only_accuracy.
        Uses the single best text-only model as the comparison baseline
        (MedGemma-27B-text in practice; whichever is highest).

        A negative gap means the VLM underperforms its text-only counterpart,
        indicating no net benefit from the visual modality.
        """
        text_only_accs = {
            m: float(np.mean(self._accuracy_outcomes(all_results[m])))
            for m in TEXT_ONLY_MODELS
            if m in all_results and len(self._accuracy_outcomes(all_results[m])) > 0
        }
        if not text_only_accs:
            return pd.DataFrame()
        best_text_name = max(text_only_accs, key=text_only_accs.get)
        best_text_acc  = text_only_accs[best_text_name]

        rows = []
        for m, df in all_results.items():
            if m in TEXT_ONLY_MODELS or m in VISION_ONLY_MODELS:
                continue
            vlm_acc = float(np.mean(self._accuracy_outcomes(df))) \
                if len(self._accuracy_outcomes(df)) > 0 else float("nan")
            rows.append({
                "model":                  m,
                "vlm_accuracy":           round(vlm_acc, 4),
                "best_text_only_model":   best_text_name,
                "best_text_only_accuracy": round(best_text_acc, 4),
                "gap":                    round(vlm_acc - best_text_acc, 4),
            })
        return pd.DataFrame(sorted(rows, key=lambda x: x["gap"]))




    def compute_all(self) -> Dict:
        """
        Computes all metrics for all models found in results_dir.

        Returns a dict with keys:
            summary_table        - pd.DataFrame, one row per model, all metrics + CIs
            per_finding_cgr      - dict[model_name -> pd.DataFrame]
            per_finding_uar      - dict[model_name -> pd.DataFrame]
            view_cgr             - dict[model_name -> dict]
            gender_cgr           - dict[model_name -> {M: float, F: float}]
            gender_uar           - dict[model_name -> {M: float, F: float}]
            age_cgr              - dict[model_name -> {<50: float, ...}]
            age_uar              - dict[model_name -> {<50: float, ...}]
            rexerr_breakdown     - dict[model_name -> breakdown dict]
            text_only_gap        - pd.DataFrame
            all_results          - dict[model_name -> merged DataFrame]
        """
        all_results = self._load_all_models()
        if not all_results:
            raise RuntimeError(
                f"No model result directories found in {self.results_dir}."
            )

        summary_rows       = []
        per_finding_cgr    = {}
        per_finding_uar    = {}
        view_cgr           = {}
        gender_cgr         = {}
        gender_uar         = {}
        age_cgr            = {}
        age_uar            = {}
        rexerr_breakdown   = {}

        for model_name, df in all_results.items():
            print(f"[MetricsComputer] Computing metrics for {model_name} ...")

            cgr_boot  = bootstrap_proportion(self._cgr_outcomes(df),   n_boot=self.n_boot)
            uar_boot  = bootstrap_proportion(self._uar_outcomes(df),   n_boot=self.n_boot)
            acc_boot  = bootstrap_proportion(self._accuracy_outcomes(df), n_boot=self.n_boot)
            irr_boot  = bootstrap_proportion(self._irr_outcomes(df),   n_boot=self.n_boot)
            sen_boot  = bootstrap_proportion(self._sensitivity_outcomes(df), n_boot=self.n_boot)
            spe_boot  = bootstrap_proportion(self._specificity_outcomes(df), n_boot=self.n_boot)

            auroc_res = self._auroc(df)
            brier     = self._brier_score(df)
            ece       = self._ece(df)
            parse_r   = self._parse_rate(df)

            summary_rows.append({
                "model":                       model_name,
                # accuracy
                "accuracy":                    round(acc_boot["point"],    4),
                "accuracy_std":                round(acc_boot["std"],      4),
                "accuracy_ci_lower":           round(acc_boot["ci_lower"], 4),
                "accuracy_ci_upper":           round(acc_boot["ci_upper"], 4),
                "accuracy_n":                  acc_boot["n"],
                # CGR
                "CGR":                         round(cgr_boot["point"],    4),
                "CGR_std":                     round(cgr_boot["std"],      4),
                "CGR_ci_lower":                round(cgr_boot["ci_lower"], 4),
                "CGR_ci_upper":                round(cgr_boot["ci_upper"], 4),
                "CGR_n":                       cgr_boot["n"],
                # UAR
                "UAR":                         round(uar_boot["point"],    4),
                "UAR_std":                     round(uar_boot["std"],      4),
                "UAR_ci_lower":                round(uar_boot["ci_lower"], 4),
                "UAR_ci_upper":                round(uar_boot["ci_upper"], 4),
                "UAR_n":                       uar_boot["n"],
                # irrelevant_stable
                "irrelevant_stable":           round(irr_boot["point"],    4),
                "irrelevant_stable_std":       round(irr_boot["std"],      4),
                "irrelevant_stable_ci_lower":  round(irr_boot["ci_lower"], 4),
                "irrelevant_stable_ci_upper":  round(irr_boot["ci_upper"], 4),
                "irrelevant_stable_n":         irr_boot["n"],
                # sensitivity / specificity
                "sensitivity":                 round(sen_boot["point"],    4),
                "sensitivity_std":             round(sen_boot["std"],      4),
                "sensitivity_ci_lower":        round(sen_boot["ci_lower"], 4),
                "sensitivity_ci_upper":        round(sen_boot["ci_upper"], 4),
                "specificity":                 round(spe_boot["point"],    4),
                "specificity_std":             round(spe_boot["std"],      4),
                "specificity_ci_lower":        round(spe_boot["ci_lower"], 4),
                "specificity_ci_upper":        round(spe_boot["ci_upper"], 4),
                # AUROC
                "AUROC":                       auroc_res["point"],
                "AUROC_ci_lower":              auroc_res["ci_lower"],
                "AUROC_ci_upper":              auroc_res["ci_upper"],
                # calibration
                "brier_score":                 round(brier, 4) if not np.isnan(brier) else float("nan"),
                "ECE":                         round(ece,   4) if not np.isnan(ece)   else float("nan"),
                # parse rate
                "parse_rate":                  round(parse_r, 4),
                # flags
                "is_text_only":                model_name in TEXT_ONLY_MODELS,
                "is_vision_only":              model_name in VISION_ONLY_MODELS,
            })

            per_finding_cgr[model_name]  = self._per_finding_cgr(df)
            per_finding_uar[model_name]  = self._per_finding_uar(df)
            view_cgr[model_name]         = self._cgr_by_view(df)
            gender_cgr[model_name]       = self._cgr_by_gender(df)
            gender_uar[model_name]       = self._uar_by_gender(df)
            age_cgr[model_name]          = self._cgr_by_age(df)
            age_uar[model_name]          = self._uar_by_age(df)
            rexerr_breakdown[model_name] = self._rexerr_breakdown(df)

        summary = (
            pd.DataFrame(summary_rows)
            .sort_values("CGR", ascending=False)
            .reset_index(drop=True)
        )

        return {
            "summary_table":     summary,
            "per_finding_cgr":   per_finding_cgr,
            "per_finding_uar":   per_finding_uar,
            "view_cgr":          view_cgr,
            "gender_cgr":        gender_cgr,
            "gender_uar":        gender_uar,
            "age_cgr":           age_cgr,
            "age_uar":           age_uar,
            "rexerr_breakdown":  rexerr_breakdown,
            "text_only_gap":     self._text_only_gap(all_results),
            "all_results":       all_results,
        }


    def save(self, output_dir: str):
        """
        Saves two files to output_dir/:
            all_metrics.csv        - single wide CSV, one row per model, everything
            paired_comparisons.csv - pairwise comparisons (multiple rows per model)

        All individual detail CSVs are written to output_dir/already_used/
        for reference but are not needed for analysis (content is in all_metrics).
        """
        os.makedirs(output_dir, exist_ok=True)
        _used = os.path.join(output_dir, "already_used")
        os.makedirs(_used, exist_ok=True)

        metrics = self.compute_all()
        all_results = metrics.pop("all_results")

        metrics["summary_table"].to_csv(os.path.join(_used, "summary_table.csv"), index=False)

        for model_name, df_cgr in metrics["per_finding_cgr"].items():
            df_cgr["model"] = model_name
        pd.concat(metrics["per_finding_cgr"].values(), ignore_index=True).to_csv(
            os.path.join(_used, "per_finding_cgr.csv"), index=False
        )
        for model_name, df_uar in metrics["per_finding_uar"].items():
            df_uar["model"] = model_name
        pd.concat(metrics["per_finding_uar"].values(), ignore_index=True).to_csv(
            os.path.join(_used, "per_finding_uar.csv"), index=False
        )

        view_rows = []
        for model_name, view_dict in metrics["view_cgr"].items():
            for view, stats in view_dict.items():
                view_rows.append({"model": model_name, "view": view, **stats})
        pd.DataFrame(view_rows).to_csv(os.path.join(_used, "view_cgr.csv"), index=False)

        for fname, data_dict in [
            ("gender_cgr.csv", metrics["gender_cgr"]),
            ("gender_uar.csv", metrics["gender_uar"]),
            ("age_cgr.csv",    metrics["age_cgr"]),
            ("age_uar.csv",    metrics["age_uar"]),
        ]:
            rows = [{"model": m, "subgroup": k, "value": v}
                    for m, d in data_dict.items() for k, v in d.items()]
            pd.DataFrame(rows).to_csv(os.path.join(_used, fname), index=False)

        if metrics["text_only_gap"] is not None and not metrics["text_only_gap"].empty:
            metrics["text_only_gap"].to_csv(os.path.join(_used, "text_only_gap.csv"), index=False)

        rexerr_rows = []
        for model_name, bd in metrics["rexerr_breakdown"].items():
            row = {"model": model_name,
                   "image_dependent_mean": bd.get("image_dependent_mean", float("nan")),
                   "text_only_mean":       bd.get("text_only_mean",       float("nan")),
                   "control_mean":         bd.get("control_mean",         float("nan"))}
            for et, stats in bd.get("per_type", {}).items():
                row[f"rexerr_{et}_swap_sens"] = stats["swap_sens"]
                row[f"rexerr_{et}_n"]         = stats["n"]
            rexerr_rows.append(row)
        pd.DataFrame(rexerr_rows).to_csv(os.path.join(_used, "rexerr_breakdown.csv"), index=False)

        from Inference.paired_comparisons import compute_paired_comparisons
        from Inference.subgroup_tests import compute_subgroup_tests

        paired_df = compute_paired_comparisons(all_results, n_boot=self.n_boot)
        paired_df.to_csv(os.path.join(output_dir, "paired_comparisons.csv"), index=False)

        subgroup_df = compute_subgroup_tests(all_results, self.manifest)
        subgroup_df.to_csv(os.path.join(_used, "subgroup_tests.csv"), index=False)

        supp = SupplementaryMetrics()
        supp_metrics = supp.compute_all(all_results)
        for fname, key in [
            ("confidence_calibration.csv", "confidence_calibration"),
            ("swap_consistency.csv",       "swap_consistency"),
            ("cgr_uar_by_polarity.csv",    "cgr_uar_by_polarity"),
        ]:
            supp_metrics[key].to_csv(os.path.join(_used, fname), index=False)
        # Combined supplementary file
        pd.concat([supp_metrics[k] for k in
                   ["confidence_calibration", "swap_consistency", "cgr_uar_by_polarity"]],
                  axis=1).loc[:, ~pd.concat(
                      [supp_metrics[k] for k in
                       ["confidence_calibration", "swap_consistency", "cgr_uar_by_polarity"]],
                      axis=1).columns.duplicated()
                  ].to_csv(os.path.join(_used, "supplementary_metrics_all.csv"), index=False)

        combined = self._build_combined(metrics, paired_df, supp_metrics, subgroup_df)
        combined.to_csv(os.path.join(output_dir, "all_metrics.csv"), index=False)

        print(f"[MetricsComputer] All outputs saved to {output_dir}")
        print("\n=== Summary Table ===")
        cols_to_show = ["model", "accuracy", "CGR", "UAR", "irrelevant_stable",
                        "parse_rate", "AUROC", "is_text_only", "is_vision_only"]
        cols_to_show = [c for c in cols_to_show if c in combined.columns]
        print(combined[cols_to_show].to_string(index=False))

        return metrics

    def _build_combined(
        self,
        metrics: Dict,
        paired_df: pd.DataFrame,
        supp_metrics: Dict = None,
        subgroup_df: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """
        Merges all metric tables into a single wide DataFrame (one row per model).
        Preserves backward-compatible column names where possible.
        """
        combined = metrics["summary_table"].copy()

        # Text-only gap
        if metrics["text_only_gap"] is not None and not metrics["text_only_gap"].empty:
            combined = combined.merge(
                metrics["text_only_gap"][["model", "gap", "best_text_only_accuracy",
                                          "best_text_only_model"]].rename(
                    columns={"gap": "text_only_gap"}
                ),
                on="model", how="left",
            )

        # Paired comparison results (vs_text_baseline rows only, wide format)
        if paired_df is not None and not paired_df.empty and "model_a" in paired_df.columns:
            vs_base = paired_df[
                (paired_df["comparison_type"] == "vs_text_baseline")
            ].copy()
            # Pivot: one set of columns per (metric, baseline) combination
            for metric_name in vs_base["metric"].unique():
                for baseline in vs_base["model_b"].unique():
                    sub = vs_base[
                        (vs_base["metric"] == metric_name) &
                        (vs_base["model_b"] == baseline)
                    ][["model_a", "diff", "diff_ci_lower", "diff_ci_upper", "p_value"]]
                    safe_base = baseline.replace("-", "_").replace(" ", "_")
                    sub = sub.rename(columns={
                        "model_a":       "model",
                        "diff":          f"{metric_name}_vs_{safe_base}_diff",
                        "diff_ci_lower": f"{metric_name}_vs_{safe_base}_ci_lower",
                        "diff_ci_upper": f"{metric_name}_vs_{safe_base}_ci_upper",
                        "p_value":       f"{metric_name}_vs_{safe_base}_p",
                    })
                    combined = combined.merge(sub, on="model", how="left")

        # Per-finding CGR (wide format, one column per finding)
        finding_cgr_wide_rows = {}
        for model_name, df_cgr in metrics["per_finding_cgr"].items():
            row = {}
            for _, r in df_cgr.iterrows():
                row[f"finding_cgr_{r['finding']}"] = r["CGR"]
                row[f"finding_cgr_{r['finding']}_n"] = r["CGR_n"]
            finding_cgr_wide_rows[model_name] = row
        if finding_cgr_wide_rows:
            combined = combined.merge(
                pd.DataFrame(finding_cgr_wide_rows).T.rename_axis("model").reset_index(),
                on="model", how="left",
            )

        # Per-finding UAR (wide format)
        finding_uar_wide_rows = {}
        for model_name, df_uar in metrics["per_finding_uar"].items():
            row = {}
            for _, r in df_uar.iterrows():
                row[f"finding_uar_{r['finding']}"] = r["UAR"]
                row[f"finding_uar_{r['finding']}_n"] = r["UAR_n"]
            finding_uar_wide_rows[model_name] = row
        if finding_uar_wide_rows:
            combined = combined.merge(
                pd.DataFrame(finding_uar_wide_rows).T.rename_axis("model").reset_index(),
                on="model", how="left",
            )

        # Subgroup breakdowns (wide format)
        for label, subgroup_dict in [
            ("gender_cgr", metrics["gender_cgr"]),
            ("gender_uar", metrics["gender_uar"]),
            ("age_cgr",    metrics["age_cgr"]),
            ("age_uar",    metrics["age_uar"]),
        ]:
            rows = {
                m: {f"{label}_{k}": v for k, v in d.items()}
                for m, d in subgroup_dict.items()
            }
            if rows:
                combined = combined.merge(
                    pd.DataFrame(rows).T.rename_axis("model").reset_index(),
                    on="model", how="left",
                )

        # ReXErr: aggregated means + per-type swap_sens and n
        rexerr_rows = {}
        for m, bd in metrics["rexerr_breakdown"].items():
            row = {
                "rexerr_image_dependent_swap_sens": bd.get("image_dependent_mean"),
                "rexerr_text_only_swap_sens":       bd.get("text_only_mean"),
                "rexerr_control_swap_sens":         bd.get("control_mean"),
            }
            for et, stats in bd.get("per_type", {}).items():
                row[f"rexerr_{et}_swap_sens"] = stats["swap_sens"]
                row[f"rexerr_{et}_n"]         = stats["n"]
            rexerr_rows[m] = row
        if rexerr_rows:
            combined = combined.merge(
                pd.DataFrame(rexerr_rows).T.rename_axis("model").reset_index(),
                on="model", how="left",
            )

        if supp_metrics is not None:
            for key, rename_map in [
                ("confidence_calibration", {}),
                ("swap_consistency",       {}),
                ("cgr_uar_by_polarity",    {}),
            ]:
                df_s = supp_metrics.get(key)
                if df_s is not None and not df_s.empty:
                    combined = combined.merge(df_s, on="model", how="left")

        if subgroup_df is not None and not subgroup_df.empty:
            for col_name, prefix in [
                ("p_raw",            "subgroup_praw_"),
                ("p_adj",            "subgroup_padj_"),
                ("significant_fdr05","subgroup_sig_"),
                ("group_means",      "subgroup_means_"),
            ]:
                if col_name not in subgroup_df.columns:
                    continue
                wide = subgroup_df.pivot(
                    index="model", columns="test", values=col_name
                ).rename(columns=lambda c: f"{prefix}{c}").reset_index()
                combined = combined.merge(wide, on="model", how="left")

        return combined.reset_index(drop=True)




def prompt_sensitivity_ci(df_raw: pd.DataFrame, n_boot: int = N_BOOT) -> dict:
    """
    Compute accuracy, 95% CI, and parse rate from a raw results CSV DataFrame.
    df_raw must have columns: parsed_answer, ground_truth.

    Returns: {accuracy, ci_lower, ci_upper, n, parse_rate}
    """
    if len(df_raw) == 0:
        return {"accuracy": float("nan"), "ci_lower": float("nan"),
                "ci_upper": float("nan"), "n": 0, "parse_rate": float("nan")}
    parse_rate = float((df_raw["parsed_answer"].isin([0, 1])).mean())
    valid = df_raw[
        df_raw["parsed_answer"].isin([0, 1]) &
        df_raw["ground_truth"].isin([0, 1])
    ]
    if len(valid) == 0:
        return {"accuracy": float("nan"), "ci_lower": float("nan"),
                "ci_upper": float("nan"), "n": 0, "parse_rate": parse_rate}
    outcomes = (valid["parsed_answer"] == valid["ground_truth"]).astype(float).values
    boot = bootstrap_proportion(outcomes, n_boot=n_boot)
    boot["parse_rate"] = round(parse_rate, 4)
    return boot



def compute_cgr_at_resolution(df_results: pd.DataFrame) -> float:
    """Convenience wrapper for resolution-check callers."""
    return MetricsComputer._cgr(df_results)


def spearman_rank_cgr(
    cgr_224: Dict[str, float],
    cgr_512: Dict[str, float],
) -> dict:
    """
    Spearman rank correlation between CGR at 224 px and 512 px across models.
    Interprets correlation of CGR rankings, not absolute values.
    With few models (n=6), even rho=1.0 should be reported with its p-value.
    """
    models = sorted(set(cgr_224.keys()) & set(cgr_512.keys()))
    if len(models) < 2:
        raise ValueError("Need at least 2 models for rank correlation.")
    v224 = [cgr_224[m] for m in models]
    v512 = [cgr_512[m] for m in models]
    rho, pval = spearmanr(v224, v512)
    print(
        f"[Resolution check] Spearman rho={rho:.4f}, p={pval:.4f} "
        f"(n={len(models)} models). Note: with n={len(models)}, "
        f"rho=1.0 corresponds to p≈0.003; interpret with caution."
    )
    return float(rho)



class SupplementaryMetrics:
    """
    Supplementary analyses on top of per-model merged DataFrames.
    All methods are static; call compute_all() for everything at once.
    """

    @staticmethod
    def confidence_calibration(df: pd.DataFrame) -> dict:
        """
        Compares mean confidence across three groups:
          - ungrounded_correct: correct AND answer unchanged under swap
          - grounded_correct:   correct AND answer flipped under target_mask
          - incorrect:          baseline reference

        High confidence on ungrounded_correct cases is a clinical safety concern:
        the model is sure of an answer it arrived at without using the image.
        """
        if "conf_original" not in df.columns or "pa_swap" not in df.columns:
            return {}
        base = df[
            df["pa_original"].isin([0, 1]) &
            df["ground_truth"].isin([0, 1]) &
            df["conf_original"].notna()
        ].copy()
        correct   = base[base["pa_original"] == base["ground_truth"]]
        incorrect = base[base["pa_original"] != base["ground_truth"]]
        ungrounded_correct = correct[
            correct["pa_swap"].isin([0, 1]) &
            (correct["pa_swap"] == correct["pa_original"])
        ]
        grounded_correct = None
        if "pa_target_mask" in df.columns:
            grounded_correct = correct[
                (correct["source"] == "ms_cxr") &
                correct["pa_target_mask"].isin([0, 1]) &
                (correct["pa_target_mask"] != correct["pa_original"])
            ]
        out = {
            "mean_conf_ungrounded_correct": float(
                ungrounded_correct["conf_original"].mean()
            ) if len(ungrounded_correct) > 0 else float("nan"),
            "mean_conf_incorrect": float(
                incorrect["conf_original"].mean()
            ) if len(incorrect) > 0 else float("nan"),
            "n_ungrounded_correct": len(ungrounded_correct),
            "n_incorrect":          len(incorrect),
        }
        if grounded_correct is not None:
            out["mean_conf_grounded_correct"] = float(
                grounded_correct["conf_original"].mean()
            ) if len(grounded_correct) > 0 else float("nan")
            out["n_grounded_correct"] = len(grounded_correct)
        return out



    @staticmethod
    def swap_consistency(df: pd.DataFrame) -> dict:
        """
        Among cases where the answer changed under swap, what fraction
        changed from correct to wrong vs from wrong to correct?
        Both fractions should be near 0.5 if swap is symmetric (balanced
        label-matched pairs). A significant imbalance would indicate
        directional label bias in the swap pool.
        """
        if "pa_swap" not in df.columns:
            return {}
        base = df[
            df["pa_original"].isin([0, 1]) &
            df["pa_swap"].isin([0, 1]) &
            df["ground_truth"].isin([0, 1])
        ]
        changed = base[base["pa_swap"] != base["pa_original"]]
        if len(changed) == 0:
            return {"swap_change_to_correct": float("nan"),
                    "swap_change_to_wrong": float("nan"), "n_changed": 0}
        swap_correct = float((changed["pa_swap"] == changed["ground_truth"]).mean())
        return {
            "swap_change_to_correct": swap_correct,
            "swap_change_to_wrong":   1.0 - swap_correct,
            "n_changed":              len(changed),
        }



    @staticmethod
    def cgr_uar_by_polarity(df: pd.DataFrame) -> dict:
        """
        CGR and UAR computed separately for positive (label=1) and
        negative (label=0) cases.
        """
        out = {}
        for polarity, label_val in [("positive", 1), ("negative", 0)]:
            sub = df[df["label"] == label_val]
            cgr_v = MetricsComputer._cgr_outcomes(sub)
            uar_v = MetricsComputer._uar_outcomes(sub)
            out[f"cgr_{polarity}"] = float(cgr_v.mean()) if len(cgr_v) > 0 else float("nan")
            out[f"uar_{polarity}"] = float(uar_v.mean()) if len(uar_v) > 0 else float("nan")
            out[f"n_{polarity}"]   = int(
                sub[sub["pa_original"].isin([0, 1]) &
                    sub["ground_truth"].isin([0, 1])].shape[0]
            )
        return out



    def compute_all(
        self,
        all_results: Dict[str, pd.DataFrame],
    ) -> Dict:
        calib_rows, swap_rows, polarity_rows = [], [], []
        for model_name, df in all_results.items():
            calib_rows.append({"model": model_name,
                                **self.confidence_calibration(df)})
            swap_rows.append({"model": model_name,
                               **self.swap_consistency(df)})
            polarity_rows.append({"model": model_name,
                                   **self.cgr_uar_by_polarity(df)})
        return {
            "confidence_calibration": pd.DataFrame(calib_rows),
            "swap_consistency":       pd.DataFrame(swap_rows),
            "cgr_uar_by_polarity":    pd.DataFrame(polarity_rows),
        }

    def save(
        self,
        all_results: Dict[str, pd.DataFrame],
        output_dir: str = None,
    ):
        os.makedirs(output_dir, exist_ok=True)
        m = self.compute_all(all_results)
        for fname, key in [
            ("confidence_calibration.csv", "confidence_calibration"),
            ("swap_consistency.csv",       "swap_consistency"),
            ("cgr_uar_by_polarity.csv",    "cgr_uar_by_polarity"),
        ]:
            m[key].to_csv(os.path.join(output_dir, fname), index=False)

        combined = m["confidence_calibration"].merge(
            m["swap_consistency"],     on="model", how="outer"
        ).merge(
            m["cgr_uar_by_polarity"],  on="model", how="outer"
        )
        combined.to_csv(
            os.path.join(output_dir, "supplementary_metrics_all.csv"), index=False
        )
        print(f"[SupplementaryMetrics] Saved to {output_dir}")
        print(combined.to_string(index=False))
