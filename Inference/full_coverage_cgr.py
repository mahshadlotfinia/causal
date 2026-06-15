"""
Inference/full_coverage_cgr.py
Created June 15, 2026

Sensitivity analysis: recompute the Causal Grounding Rate (CGR) on the subset
of MS-CXR cases whose target box was judged to FULLY COVER the finding in the
Task A box-validation, for every model and for both radiologist readers

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia/

"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config.serde import read_config
from Inference.metrics import MetricsComputer
from Inference.stats_utils import (
    bootstrap_proportion,
    paired_bootstrap_diff,
    bh_fdr,
    N_BOOT,
)

# Readers, named explicitly. Never collapsed into a single "human".
READERS = ["SZ", "LA"]
REFERENCE_READER = "SZ"

# Per-finding reporting threshold (Step 4)
MIN_N_PER_FINDING = 10


def _read_csv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")


def _normalize_rating(s) -> str:
    """Canonicalize a Task A rating label to one of the four values."""
    if pd.isna(s):
        return ""
    t = str(s).strip().lower()
    if t.startswith("acc"):
        return "accurate"
    if t.startswith("par"):
        return "partial"
    if t.startswith("inac"):
        return "inaccurate"
    if t.startswith("can"):
        return "cannot_tell"
    return t


def _normalize_yes_no(s) -> int:
    """Reader 'Yes'/'No' -> 1/0; -1 on unparseable."""
    if pd.isna(s):
        return -1
    t = str(s).strip().lower()
    if t in ("yes", "y", "1", "true"):
        return 1
    if t in ("no", "n", "0", "false"):
        return 0
    return -1


def load_task_a_per_case(reader_study_root: str) -> pd.DataFrame:
    """
    Surface the raw per-case Task A box-validation ratings, keyed by case_id,
    with the rating label and the reader who gave it.

    Looks for each reader's filled Task A under:
        {reader_study_root}/{reader_subdir}/task_A_box_validation/
            cases.csv         (reader-entered: display_id, rating, ...)
            case_mapping.csv   (display_id -> case_id, finding)

    Reader folders are matched by the reader code appearing in the subdir name
    (case-insensitive), e.g. 'reader1_filled_SZ' or 'sz'. If exactly two reader
    folders contain a Task A, they are assigned to SZ and LA by folder order
    only as a last resort; prefer encoding the reader code in the folder name.

    Returns a long DataFrame with columns:
        case_id, finding, reader, rating
    Only rated cases are included (blank ratings dropped).
    """
    rows = []
    if not os.path.isdir(reader_study_root):
        print(f"[full_cov] reader_study_root not found: {reader_study_root}")
        return pd.DataFrame(columns=["case_id", "finding", "reader", "rating"])

    subdirs = sorted(
        d for d in os.listdir(reader_study_root)
        if os.path.isdir(os.path.join(reader_study_root, d))
    )

    # Map each subdir to a reader code when possible
    def _infer_reader(subdir_name: str) -> Optional[str]:
        low = subdir_name.lower()
        for r in READERS:
            if r.lower() in low:
                return r
        return None

    unmatched = []
    for sub in subdirs:
        task_a_dir = os.path.join(reader_study_root, sub, "task_A_box_validation")
        cases_csv  = os.path.join(task_a_dir, "cases.csv")
        map_csv    = os.path.join(task_a_dir, "case_mapping.csv")
        if not (os.path.exists(cases_csv) and os.path.exists(map_csv)):
            continue
        reader = _infer_reader(sub)
        rec = (sub, cases_csv, map_csv, reader)
        if reader is None:
            unmatched.append(rec)
        else:
            rows.append(rec)

    # Last-resort assignment for unmatched reader folders, in folder order
    if unmatched:
        used = {r for (_, _, _, r) in rows if r is not None}
        free = [r for r in READERS if r not in used]
        for (sub, cases_csv, map_csv, _), r in zip(unmatched, free):
            print(f"[full_cov] reader code not in folder name '{sub}'; "
                  f"assigning to {r} by order.")
            rows.append((sub, cases_csv, map_csv, r))

    out = []
    for sub, cases_csv, map_csv, reader in rows:
        ratings = _read_csv(cases_csv)
        mapping = _read_csv(map_csv)
        if "rating" not in ratings.columns:
            print(f"[full_cov] no 'rating' column in {cases_csv}; skipping.")
            continue
        ratings = ratings.copy()
        ratings["rating_norm"] = ratings["rating"].map(_normalize_rating)
        ratings = ratings[ratings["rating_norm"] != ""]
        if "finding" in ratings.columns:
            ratings = ratings.drop(columns=["finding"])
        keep_map = [c for c in ["display_id", "case_id", "finding"] if c in mapping.columns]
        merged = ratings.merge(mapping[keep_map], on="display_id", how="left")
        for _, rr in merged.iterrows():
            out.append({
                "case_id": rr.get("case_id"),
                "finding": rr.get("finding"),
                "reader":  reader,
                "rating":  rr["rating_norm"],
            })

    df = pd.DataFrame(out, columns=["case_id", "finding", "reader", "rating"])
    df = df[df["case_id"].notna()].reset_index(drop=True)
    return df


def build_coverage_filters(task_a: pd.DataFrame) -> Dict[str, set]:
    """
    From per-case Task A ratings, build the full-coverage case-id sets.

    Primary:   full_coverage          = rating == 'accurate'
    Secondary: full_coverage_partial  = rating in {'accurate', 'partial'}

    If both readers rated the same case_id, the reference reader (SZ) rating
    decides the primary filter. Also build a stricter 'both accurate' set.

    Returns dict of sets:
        'accurate', 'accurate_or_partial', 'both_accurate'
    plus the reference-reader rating table used.
    """
    if task_a.empty:
        return {
            "accurate": set(), "accurate_or_partial": set(),
            "both_accurate": set(), "ref_table": pd.DataFrame(),
        }

    ref_rows = []
    for cid, grp in task_a.groupby("case_id"):
        ref = grp[grp["reader"] == REFERENCE_READER]
        chosen = ref.iloc[0] if len(ref) else grp.iloc[0]
        ref_rows.append({
            "case_id": cid,
            "finding": chosen.get("finding"),
            "rating":  chosen["rating"],
            "rating_source_reader": chosen["reader"],
        })
    ref_table = pd.DataFrame(ref_rows)

    accurate = set(ref_table.loc[ref_table["rating"] == "accurate", "case_id"])
    acc_or_partial = set(
        ref_table.loc[ref_table["rating"].isin(["accurate", "partial"]), "case_id"]
    )

    # Both-readers-accurate (only cases rated by both, both 'accurate')
    both_accurate = set()
    for cid, grp in task_a.groupby("case_id"):
        readers_here = set(grp["reader"])
        if {"SZ", "LA"}.issubset(readers_here):
            if all(grp.loc[grp["reader"] == r, "rating"].iloc[0] == "accurate"
                   for r in ["SZ", "LA"]):
                both_accurate.add(cid)

    return {
        "accurate": accurate,
        "accurate_or_partial": acc_or_partial,
        "both_accurate": both_accurate,
        "ref_table": ref_table,
    }


def _restrict(df: pd.DataFrame, case_ids: set) -> pd.DataFrame:
    return df[df["case_id"].isin(case_ids)].copy()


def _cgr_outcomes_model(df: pd.DataFrame) -> np.ndarray:
    """
    Per-case 0/1 grounding outcomes for a model wide-table, matching the EXACT
    eligibility and -1 handling of MetricsComputer._cgr:
        denominator = MS-CXR cases, parseable original matching ground_truth
                      (the "correct on original" set);
        numerator   = those whose target answer is parseable AND flips.
    An unparsed target answer (-1) stays in the denominator and counts as
    "not grounded" (outcome 0), identical to the existing CGR. The per-case
    outcome vector therefore has length == denominator, with mean == _cgr.
    """
    if "pa_target_mask" not in df.columns:
        return np.array([])
    sub = df[
        (df["source"] == "ms_cxr") &
        (df["pa_original"].isin([0, 1])) &
        (df["ground_truth"].isin([0, 1]))
    ].copy()
    correct = sub[sub["pa_original"] == sub["ground_truth"]]
    if len(correct) == 0:
        return np.array([])
    flipped = (
        correct["pa_target_mask"].isin([0, 1]) &
        (correct["pa_target_mask"] != correct["pa_original"])
    ).astype(float)
    return flipped.values


def _cgr_outcomes_reader(piv: pd.DataFrame) -> np.ndarray:
    """
    Per-case 0/1 grounding outcomes for a reader pivot-table with columns
    answer_original, answer_target_mask. Eligibility: answer_original == 1 and
    answer_target_mask parseable. 1 if the answer flips under target mask.
    Mirrors the reader CGR in the existing reader pipeline.
    """
    sub = piv[
        (piv["answer_original"] == 1) &
        (piv["answer_target_mask"].isin([0, 1]))
    ]
    if len(sub) == 0:
        return np.array([])
    return (sub["answer_target_mask"] != sub["answer_original"]).astype(float).values


def _summ(outcomes: np.ndarray) -> dict:
    """
    Proportion summary: mean, binomial SE sqrt(p(1-p)/n), Wilson 95% CI,
    and n. Consistent with the per-finding convention in the spec (Step 5).
    """
    n = len(outcomes)
    if n == 0:
        return {"cgr": float("nan"), "n": 0, "se": float("nan"),
                "ci_lower": float("nan"), "ci_upper": float("nan")}
    p = float(outcomes.mean())
    se = float(np.sqrt(p * (1.0 - p) / n)) if n > 0 else float("nan")
    lo, hi = _wilson_ci(int(round(p * n)), n)
    return {"cgr": p, "n": int(n), "se": se, "ci_lower": lo, "ci_upper": hi}


def _wilson_ci(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    from math import sqrt
    z = 1.959963984540054
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def load_reader_pivot(reader_study_root: str, reader: str) -> Optional[pd.DataFrame]:
    """
    Build the per-case reader answer pivot for the finding-presence task,
    with columns answer_original, answer_target_mask keyed by case_id.

    Searches reader folders for a Task B (reader1) or Task C (reader2) packet
    whose folder name contains the reader code. Uses the packet's case_mapping
    (display_id -> case_id, condition) to attach case_id and condition, then
    pivots answers wide.
    """
    if not os.path.isdir(reader_study_root):
        return None
    for sub in sorted(os.listdir(reader_study_root)):
        sub_path = os.path.join(reader_study_root, sub)
        if not os.path.isdir(sub_path):
            continue
        if reader.lower() not in sub.lower():
            continue
        for task_name in ["task_B_finding_presence", "task_C_reader2_agreement"]:
            task_dir = os.path.join(sub_path, task_name)
            cases_csv = os.path.join(task_dir, "cases.csv")
            map_csv   = os.path.join(task_dir, "case_mapping.csv")
            # Task C may reuse Task B's mapping; fall back to any sibling Task B mapping
            if os.path.exists(cases_csv) and not os.path.exists(map_csv):
                alt = os.path.join(sub_path, "task_B_finding_presence", "case_mapping.csv")
                map_csv = alt if os.path.exists(alt) else map_csv
            if not (os.path.exists(cases_csv) and os.path.exists(map_csv)):
                continue
            ratings = _read_csv(cases_csv)
            mapping = _read_csv(map_csv)
            join_cols = ["display_id", "session"] if (
                "session" in mapping.columns and "session" in ratings.columns
            ) else ["display_id"]
            shared = [c for c in mapping.columns
                      if c not in join_cols and c in ratings.columns]
            ratings_clean = ratings.drop(columns=shared)
            merged = ratings_clean.merge(mapping, on=join_cols, how="left")
            if "case_id" not in merged.columns or merged["case_id"].isna().all():
                merged = ratings.drop(
                    columns=[c for c in mapping.columns
                             if c != "display_id" and c in ratings.columns]
                ).merge(mapping, on="display_id", how="left")
            merged["answer_int"] = merged["answer"].map(_normalize_yes_no)
            merged = merged[merged["answer_int"].isin([0, 1])]
            if len(merged) == 0:
                continue
            piv = merged.pivot_table(
                index="case_id", columns="condition", values="answer_int",
                aggfunc="first"
            ).add_prefix("answer_")
            piv = piv.reset_index()
            if "finding" in mapping.columns:
                fmap = mapping.groupby("case_id")["finding"].first()
                piv["finding"] = piv["case_id"].map(fmap)
            # Ensure required columns exist
            for col in ["answer_original", "answer_target_mask"]:
                if col not in piv.columns:
                    piv[col] = np.nan
            return piv
    return None


class FullCoverageCGR:
    def __init__(self, cfg_path: str, reader_study_root: str,
                 dataset_type: str = "mimic", seed: int = 0):
        self.cfg_path = cfg_path
        self.params   = read_config(cfg_path)
        self.cfg      = self.params["CausalAudit"]
        self.reader_study_root = reader_study_root
        self.seed     = seed
        self.computer = MetricsComputer(cfg_path, dataset_type=dataset_type)
        self.out_dir  = os.path.join(self.cfg["results_dir"], "metrics", "full_coverage")
        os.makedirs(self.out_dir, exist_ok=True)

    def run(self):
        # ----- Task A per-case ratings and coverage filters -----
        task_a = load_task_a_per_case(self.reader_study_root)
        filters = build_coverage_filters(task_a)
        acc_set     = filters["accurate"]
        accpar_set  = filters["accurate_or_partial"]

        n_rated = task_a["case_id"].nunique() if not task_a.empty else 0
        print(f"[full_cov] Task A rated cases: {n_rated}")
        print(f"[full_cov] pass 'accurate': {len(acc_set)}")
        print(f"[full_cov] pass 'accurate_or_partial': {len(accpar_set)}")
        if not task_a.empty:
            ref = filters["ref_table"]
            per_finding_counts = (
                ref[ref["rating"] == "accurate"]
                .groupby("finding")["case_id"].nunique().to_dict()
            )
            print(f"[full_cov] full-coverage cases per finding: {per_finding_counts}")

        # ----- Model wide tables -----
        all_models = self.computer._load_all_models()   # {model: wide_df}

        # ----- Reader pivots -----
        reader_pivots = {}
        for r in READERS:
            piv = load_reader_pivot(self.reader_study_root, r)
            if piv is not None:
                reader_pivots[r] = piv
            else:
                print(f"[full_cov] no reader pivot found for {r}.")

        overall_rows      = []
        per_finding_rows  = []

        for model, df in all_models.items():
            cgr_all_outcomes = _cgr_outcomes_model(df)
            cgr_all = float(cgr_all_outcomes.mean()) if len(cgr_all_outcomes) else float("nan")
            n_all   = int(len(cgr_all_outcomes))

            fc_outcomes  = _cgr_outcomes_model(_restrict(df, acc_set))
            fcp_outcomes = _cgr_outcomes_model(_restrict(df, accpar_set))
            s  = _summ(fc_outcomes)
            sp = _summ(fcp_outcomes)

            overall_rows.append({
                "entity":               model,
                "entity_type":          "model",
                "cgr_all":              cgr_all,
                "n_all":                n_all,
                "cgr_fullcov":          s["cgr"],
                "n_fullcov":            s["n"],
                "cgr_fullcov_se":       s["se"],
                "cgr_fullcov_ci_lower": s["ci_lower"],
                "cgr_fullcov_ci_upper": s["ci_upper"],
                "cgr_fullcov_partial":  sp["cgr"],
                "n_fullcov_partial":    sp["n"],
            })

            # Per-finding (full-coverage, accurate)
            fc_df = _restrict(df, acc_set)
            for finding, grp in fc_df.groupby("finding"):
                out = _cgr_outcomes_model(grp)
                summ = _summ(out)
                per_finding_rows.append({
                    "entity":     model,
                    "entity_type": "model",
                    "finding":    finding,
                    "cgr_fullcov": summ["cgr"],
                    "n":          summ["n"],
                    "ci_lower":   summ["ci_lower"],
                    "ci_upper":   summ["ci_upper"],
                    "low_n":      bool(summ["n"] < MIN_N_PER_FINDING),
                })

        for reader, piv in reader_pivots.items():
            all_out = _cgr_outcomes_reader(piv)
            cgr_all = float(all_out.mean()) if len(all_out) else float("nan")
            n_all   = int(len(all_out))

            fc_piv  = piv[piv["case_id"].isin(acc_set)]
            fcp_piv = piv[piv["case_id"].isin(accpar_set)]
            s  = _summ(_cgr_outcomes_reader(fc_piv))
            sp = _summ(_cgr_outcomes_reader(fcp_piv))

            overall_rows.append({
                "entity":               reader,
                "entity_type":          "reader",
                "cgr_all":              cgr_all,
                "n_all":                n_all,
                "cgr_fullcov":          s["cgr"],
                "n_fullcov":            s["n"],
                "cgr_fullcov_se":       s["se"],
                "cgr_fullcov_ci_lower": s["ci_lower"],
                "cgr_fullcov_ci_upper": s["ci_upper"],
                "cgr_fullcov_partial":  sp["cgr"],
                "n_fullcov_partial":    sp["n"],
            })

            for finding, grp in fc_piv.groupby("finding"):
                out = _cgr_outcomes_reader(grp)
                summ = _summ(out)
                per_finding_rows.append({
                    "entity":     reader,
                    "entity_type": "reader",
                    "finding":    finding,
                    "cgr_fullcov": summ["cgr"],
                    "n":          summ["n"],
                    "ci_lower":   summ["ci_lower"],
                    "ci_upper":   summ["ci_upper"],
                    "low_n":      bool(summ["n"] < MIN_N_PER_FINDING),
                })

        overall_df     = pd.DataFrame(overall_rows)
        per_finding_df  = pd.DataFrame(per_finding_rows)

        paired_df = self._paired_human_vs_model(all_models, reader_pivots, acc_set)

        combined = self._combine(overall_df, per_finding_df, paired_df)
        combined_path = os.path.join(self.out_dir, "cgr_full_coverage_all.csv")
        combined.to_csv(combined_path, index=False)
        print(f"[full_cov] wrote {combined_path}")

    @staticmethod
    def _combine(overall_df: pd.DataFrame, per_finding_df: pd.DataFrame,
                 paired_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge the three result blocks into a single long CSV. A 'block' column
        marks the row type ('overall', 'per_finding', 'paired_diff'); columns
        not relevant to a given block are left blank for those rows.
        """
        parts = []

        if overall_df is not None and not overall_df.empty:
            o = overall_df.copy()
            o.insert(0, "block", "overall")
            parts.append(o)

        if per_finding_df is not None and not per_finding_df.empty:
            p = per_finding_df.copy()
            # Disambiguate the per-finding CGR/CI columns from the overall ones
            p = p.rename(columns={
                "cgr_fullcov": "perfinding_cgr_fullcov",
                "n":           "perfinding_n",
                "ci_lower":    "perfinding_ci_lower",
                "ci_upper":    "perfinding_ci_upper",
            })
            p.insert(0, "block", "per_finding")
            parts.append(p)

        if paired_df is not None and not paired_df.empty:
            d = paired_df.copy()
            # Map paired columns onto shared names where natural; keep the rest
            d = d.rename(columns={"model": "entity"})
            d["entity_type"] = "model"
            d.insert(0, "block", "paired_diff")
            parts.append(d)

        if not parts:
            return pd.DataFrame()

        combined = pd.concat(parts, ignore_index=True, sort=False)

        # Order columns: block + identity first, then everything else stable
        lead = [c for c in ["block", "entity", "entity_type", "finding",
                            "reference_reader"] if c in combined.columns]
        rest = [c for c in combined.columns if c not in lead]
        return combined[lead + rest]

    def _paired_human_vs_model(self, all_models: Dict[str, pd.DataFrame],
                                reader_pivots: Dict[str, pd.DataFrame],
                                acc_set: set) -> pd.DataFrame:
        """
        Paired-bootstrap CGR difference (model minus reader) on the shared
        full-coverage case set: cases that are eligible (correct-on-original,
        parseable) for BOTH sides AND rating == accurate. FDR-corrected within
        the CGR family. Reference reader SZ, plus LA when available.
        """
        rows = []
        for ref_reader in [REFERENCE_READER] + [r for r in READERS if r != REFERENCE_READER]:
            piv = reader_pivots.get(ref_reader)
            if piv is None:
                continue
            # Reader eligibility per case under full coverage
            r_elig = piv[
                piv["case_id"].isin(acc_set) &
                (piv["answer_original"] == 1) &
                (piv["answer_target_mask"].isin([0, 1]))
            ].copy()
            r_elig["reader_flip"] = (
                r_elig["answer_target_mask"] != r_elig["answer_original"]
            ).astype(float)
            reader_map = r_elig.set_index("case_id")["reader_flip"]

            for model, df in all_models.items():
                sub = df[
                    (df["case_id"].isin(acc_set)) &
                    (df["source"] == "ms_cxr") &
                    (df["pa_original"].isin([0, 1])) &
                    (df["ground_truth"].isin([0, 1]))
                ].copy()
                sub = sub[sub["pa_original"] == sub["ground_truth"]]
                if len(sub) == 0:
                    continue
                # Unparsed target (-1) stays in the denominator as a non-flip,
                # identical to the existing CGR definition.
                sub["model_flip"] = (
                    sub["pa_target_mask"].isin([0, 1]) &
                    (sub["pa_target_mask"] != sub["pa_original"])
                ).astype(float)
                model_map = sub.set_index("case_id")["model_flip"]

                shared = sorted(set(model_map.index) & set(reader_map.index))
                if len(shared) < 1:
                    continue
                m_vals = model_map.loc[shared].values.astype(float)
                r_vals = reader_map.loc[shared].values.astype(float)

                pb = paired_bootstrap_diff(m_vals, r_vals, n_boot=N_BOOT, seed=self.seed)
                rows.append({
                    "model":                   model,
                    "reference_reader":        ref_reader,
                    "reader_cgr":              float(r_vals.mean()),
                    "model_cgr":               float(m_vals.mean()),
                    "diff_model_minus_reader": pb["point"],
                    "diff_ci_lower":           pb["ci_lower"],
                    "diff_ci_upper":           pb["ci_upper"],
                    "p_value":                 pb["p_value"],
                    "p_value_fdr":             float("nan"),
                    "n_shared":                int(len(shared)),
                })

        out = pd.DataFrame(rows)
        if out.empty:
            return out
        # FDR within the CGR family, per reference reader
        for ref_reader, grp in out.groupby("reference_reader"):
            out.loc[grp.index, "p_value_fdr"] = bh_fdr(grp["p_value"].values)
        return out



def main_full_coverage_cgr(global_config_path: str, reader_study_root: str,
                            dataset_type: str = "mimic"):
    FullCoverageCGR(
        cfg_path=global_config_path,
        reader_study_root=reader_study_root,
        dataset_type=dataset_type,
        seed=0,
    ).run()



if __name__ == "__main__":
    CONFIG_PATH       = "/home/soroosh/Documents/Repositories/causal/config/config.yaml"
    READER_STUDY_ROOT = "/home/soroosh/Documents/Repositories_target_files/causal/reader_study"
    main_full_coverage_cgr(CONFIG_PATH, READER_STUDY_ROOT, dataset_type="mimic")
