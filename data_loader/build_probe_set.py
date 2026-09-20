"""
data_loader/build_probe_set.py
Created on September 14, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import os
from typing import Dict, Optional, Set

import numpy as np
import pandas as pd

from config.serde import read_config
from data_loader.build_utils import corner_box, read_csv_defensively
from Inference.resume_utils import append_status, ensure_dir, status_path, write_csv_atomic


JUNE_PARAMETERS = dict(target_resolution=224, min_box_pixels=50, mimic_per_finding=100, mimic_normals=100, mscxr_per_finding_cap=100,
                       rexerr_total=800, rexerr_image_dep_fraction=0.70, rexerr_text_only_fraction=0.15, rexerr_no_error_fraction=0.15,
                       chexpert_per_finding=100, chexpert_normals=100, random_seed=42)


class ProbeSetBuilder:

    FRONTAL_VIEWS = {"PA", "AP"}
    CHEXPERT_FINDINGS = ["atelectasis", "cardiomegaly", "consolidation", "edema", "enlarged_cardiomediastinum", "fracture", "lung_lesion",
                         "lung_opacity", "pleural_effusion", "pleural_other", "pneumonia", "pneumothorax", "support_devices"]
    MSCXR_TO_MIMIC = {"Atelectasis": "atelectasis", "Cardiomegaly": "cardiomegaly", "Consolidation": "consolidation", "Edema": "edema",
                      "Lung Opacity": "lung_opacity", "Pleural Effusion": "pleural_effusion", "Pneumonia": "pneumonia", "Pneumothorax": "pneumothorax"}
    IMAGE_DEPENDENT_ERROR_TYPES = {"Change location", "Change severity", "False prediction", "False negation", "Change position of device",
                                   "Add medical device", "Change name of device", "Change view"}
    TEXT_ONLY_ERROR_TYPES = {"Add typo", "Change to homophone", "Add repetition"}
    UNIFIED_COLS = ["case_id", "source", "image_path", "report_path", "dicom_id", "subject_id", "study_id", "view", "age", "gender",
                    "finding", "label", "label_text", "box_x", "box_y", "box_w", "box_h", "original_sentence", "error_sentence",
                    "error_type", "error_present"]

    def __init__(self, mimic_csv: str, mscxr_csv: str, rexerr_csv: str, chexpert_csv: Optional[str] = None,
                 target_resolution: int = 224, min_box_pixels: int = 50, mimic_per_finding: int = 100, mimic_normals: int = 100,
                 rexerr_total: int = 800, rexerr_image_dep_fraction: float = 0.70, rexerr_text_only_fraction: float = 0.15,
                 rexerr_no_error_fraction: float = 0.15, mscxr_per_finding_cap: int = 100, chexpert_per_finding: int = 100,
                 chexpert_normals: int = 100, random_seed: int = 42):
        assert abs(rexerr_image_dep_fraction + rexerr_text_only_fraction + rexerr_no_error_fraction - 1.0) < 1e-6
        self.mimic_csv, self.mscxr_csv, self.rexerr_csv, self.chexpert_csv = mimic_csv, mscxr_csv, rexerr_csv, chexpert_csv
        self.target_resolution, self.min_box_pixels = target_resolution, min_box_pixels
        self.mimic_per_finding, self.mimic_normals, self.mscxr_per_finding_cap = mimic_per_finding, mimic_normals, mscxr_per_finding_cap
        self.rexerr_total = rexerr_total
        self.rexerr_image_dep_fraction, self.rexerr_text_only_fraction = rexerr_image_dep_fraction, rexerr_text_only_fraction
        self.rexerr_no_error_fraction = rexerr_no_error_fraction
        self.chexpert_per_finding, self.chexpert_normals = chexpert_per_finding, chexpert_normals
        self.rng = np.random.default_rng(random_seed)
        self._mimic_full: Optional[pd.DataFrame] = None
        self._chexpert_full: Optional[pd.DataFrame] = None

    def _load_mimic(self) -> pd.DataFrame:
        if self._mimic_full is None:
            df = read_csv_defensively(self.mimic_csv)
            df["dicom_id"] = df["jpg_rel_path"].str.extract(r"([^/]+)\.jpg$", expand=False)
            df = df[df["age"].notna() & df["gender"].notna()].copy()
            self._mimic_full = df
        return self._mimic_full

    def _rs(self) -> int:
        return int(self.rng.integers(2 ** 31))

    def build_mscxr_subset(self) -> pd.DataFrame:
        df = read_csv_defensively(self.mscxr_csv).copy()
        target = self.target_resolution
        sx, sy = target / df["image_width"], target / df["image_height"]
        df["x_t"], df["y_t"], df["w_t"], df["h_t"] = df["x"] * sx, df["y"] * sy, df["w"] * sx, df["h"] * sy
        df = df[(df["w_t"] >= self.min_box_pixels) & (df["h_t"] >= self.min_box_pixels)].reset_index(drop=True)
        mimic = self._load_mimic()
        df = df.merge(mimic[["dicom_id", "subject_id", "study_id", "view", "report_rel_path", "age", "gender"]].drop_duplicates("dicom_id"),
                      on="dicom_id", how="left")
        df = df[df["view"].isin(self.FRONTAL_VIEWS)].reset_index(drop=True)
        df["finding"] = df["category_name"].map(self.MSCXR_TO_MIMIC)
        df = df[df["finding"].notna()].reset_index(drop=True)
        if self.mscxr_per_finding_cap is not None:
            chunks = []
            for finding_name in df["finding"].unique():
                sub = df[df["finding"] == finding_name]
                chunks.append(sub.sample(n=min(self.mscxr_per_finding_cap, len(sub)), random_state=self._rs()))
            df = pd.concat(chunks, ignore_index=True)
        df["case_id"] = [f"mscxr_{i:05d}" for i in range(len(df))]
        df["source"] = "ms_cxr"
        df["image_path"] = "mimic-cxr-jpg/" + df["path"]
        df["report_path"] = df["report_rel_path"]
        df["label"] = 1
        for c, src in (("box_x", "x_t"), ("box_y", "y_t"), ("box_w", "w_t"), ("box_h", "h_t")):
            df[c] = df[src].round().astype(int)
        for c in ["original_sentence", "error_sentence", "error_type", "error_present"]:
            df[c] = None
        return df[self.UNIFIED_COLS].copy()

    def build_mimic_subset(self, exclude_dicoms: Set[str]) -> pd.DataFrame:
        df = self._load_mimic().copy()
        df = df[(df["split"] == "test") & df["view"].isin(self.FRONTAL_VIEWS) & ~df["dicom_id"].isin(exclude_dicoms)]
        half = self.mimic_per_finding // 2
        rows, idx = [], 0
        for finding in self.CHEXPERT_FINDINGS:
            pos = df[df[finding] == 1].drop_duplicates("subject_id")
            neg_clean = df[df[finding] == 0].drop_duplicates("subject_id")
            if len(neg_clean) < half:
                neg = pd.concat([neg_clean, df[df[finding] == 3].drop_duplicates("subject_id")]).drop_duplicates("subject_id")
            else:
                neg = neg_clean
            for pool, label in ((pos, 1), (neg, 0)):
                n = min(half, len(pool))
                if n:
                    for _, r in pool.sample(n=n, random_state=self._rs()).iterrows():
                        rows.append(self._mimic_row(r, idx, finding, label))
                        idx += 1
        normals = df[df["no_finding"] == 1].drop_duplicates("subject_id")
        n_norm = min(self.mimic_normals, len(normals))
        if n_norm:
            for _, r in normals.sample(n=n_norm, random_state=self._rs()).iterrows():
                rows.append(self._mimic_row(r, idx, "no_finding", 1))
                idx += 1
        return pd.DataFrame(rows)[self.UNIFIED_COLS].copy()

    def _mimic_row(self, r, case_idx: int, finding: str, label: int) -> Dict:
        return {"case_id": f"mimic_{case_idx:05d}", "source": "mimic_cxr", "image_path": r["jpg_rel_path"], "report_path": r["report_rel_path"],
                "dicom_id": r["dicom_id"], "subject_id": r["subject_id"], "study_id": r["study_id"], "view": r["view"], "age": r.get("age"),
                "gender": r.get("gender"), "finding": finding, "label": label, "label_text": None, "box_x": None, "box_y": None, "box_w": None,
                "box_h": None, "original_sentence": None, "error_sentence": None, "error_type": None, "error_present": None}

    def build_rexerr_subset(self) -> pd.DataFrame:
        df = read_csv_defensively(self.rexerr_csv).copy()
        df = df[df["error_present"].isin([0.0, 1.0]) & df["original_sentence"].notna()].copy()
        df["dicom_id"] = df["dicom_id"].fillna("").astype(str)
        df["_dlist"] = df["dicom_id"].str.split(",")
        df = df.explode("_dlist")
        df["dicom_id"] = df["_dlist"].str.strip()
        df = df.drop(columns=["_dlist"])
        df = df[df["dicom_id"] != ""]
        mimic = self._load_mimic()
        merge_cols = ["dicom_id", "view", "jpg_rel_path", "report_rel_path", "age", "gender"] + self.CHEXPERT_FINDINGS
        df = df.merge(mimic[merge_cols].drop_duplicates("dicom_id"), on="dicom_id", how="inner")
        df = df[df["view"].isin(self.FRONTAL_VIEWS)]
        df = df.drop_duplicates(subset=["study_id", "original_sentence", "error_sentence"], keep="first").reset_index(drop=True)
        is_no_error = df["error_present"] == 0.0
        is_text_only = df["error_type"].isin(self.TEXT_ONLY_ERROR_TYPES) & (df["error_present"] == 1.0)
        is_image_dep = df["error_type"].isin(self.IMAGE_DEPENDENT_ERROR_TYPES) & (df["error_present"] == 1.0)
        n_img = int(round(self.rexerr_total * self.rexerr_image_dep_fraction))
        n_txt = int(round(self.rexerr_total * self.rexerr_text_only_fraction))
        n_no = self.rexerr_total - n_img - n_txt
        chunks = []
        for pool, budget in ((df[is_image_dep], n_img), (df[is_text_only], n_txt)):
            types = sorted(pool["error_type"].unique())
            if types:
                per = max(1, budget // len(types))
                for et in types:
                    sub = pool[pool["error_type"] == et]
                    k = min(per, len(sub))
                    if k:
                        chunks.append(sub.sample(n=k, random_state=self._rs()))
        if n_no > 0 and is_no_error.any():
            chunks.append(df[is_no_error].sample(n=min(n_no, int(is_no_error.sum())), random_state=self._rs()))
        out = pd.concat(chunks, ignore_index=True)
        out["case_id"] = [f"rexerr_{i:05d}" for i in range(len(out))]
        out["source"] = "rexerr"
        out["image_path"], out["report_path"] = out["jpg_rel_path"], out["report_rel_path"]

        def _infer_finding(row):
            for f in self.CHEXPERT_FINDINGS:
                if row.get(f) == 1:
                    return f
            return None

        out["finding"] = out.apply(_infer_finding, axis=1)
        out["label"] = out["error_present"].astype(int)
        out["label_text"] = out["error_sentence"]
        for c in ["box_x", "box_y", "box_w", "box_h"]:
            out[c] = None
        return out[self.UNIFIED_COLS].copy()

    def add_swap_candidates(self, manifest: pd.DataFrame) -> pd.DataFrame:
        pool = self._load_mimic()
        pool = pool[pool["view"].isin(self.FRONTAL_VIEWS)].copy()
        swap_paths = []
        for _, row in manifest.iterrows():
            finding, subj, label = row.get("finding"), row["subject_id"], row["label"]
            cand = None
            if finding in self.CHEXPERT_FINDINGS and pd.notna(label):
                cand = pool[(pool[finding] == label) & (pool["subject_id"] != subj)]
            elif finding == "no_finding":
                cand = pool[(pool["no_finding"] == 1) & (pool["subject_id"] != subj)]
            if cand is None or len(cand) == 0:
                cand = pool[pool["subject_id"] != subj]
            swap_paths.append(cand.sample(n=1, random_state=self._rs()).iloc[0]["jpg_rel_path"] if len(cand) else None)
        out = manifest.copy()
        out["swap_image_path"] = swap_paths
        return out

    def add_irrelevant_masks(self, manifest: pd.DataFrame) -> pd.DataFrame:
        cols = {c: [] for c in ("irrelevant_box_x", "irrelevant_box_y", "irrelevant_box_w", "irrelevant_box_h")}
        for _, row in manifest.iterrows():
            if pd.isna(row["box_x"]):
                for c in cols:
                    cols[c].append(None)
                continue
            cx, cy, bw, bh = corner_box(int(row["box_x"]), int(row["box_y"]), int(row["box_w"]), int(row["box_h"]), self.target_resolution)
            for c, v in zip(cols, (cx, cy, bw, bh)):
                cols[c].append(v)
        out = manifest.copy()
        for c, v in cols.items():
            out[c] = v
        return out

    def _load_chexpert(self) -> pd.DataFrame:
        if self._chexpert_full is None:
            df = read_csv_defensively(self.chexpert_csv)
            df["dicom_id"] = df["jpg_rel_path"].str.extract(r"([^/]+)\.jpg$", expand=False)
            df["study_id"] = df["jpg_rel_path"].str.extract(r"(patient\d+/study\d+)", expand=False)
            df["gender"] = df["gender"].map({"Female": "F", "Male": "M", "Unknown": None})
            df = df[df["age"].notna() & df["gender"].notna()].copy()
            self._chexpert_full = df
        return self._chexpert_full

    def build_chexpert_subset(self) -> pd.DataFrame:
        df = self._load_chexpert().copy()
        df = df[(df["split"] == "test") & (df["view"] == "Frontal")].copy()
        df["view"] = df["AP_PA"].fillna("AP")
        df["report_path"] = None
        half = self.chexpert_per_finding // 2
        rows, idx = [], 0
        for finding in self.CHEXPERT_FINDINGS:
            if finding not in df.columns:
                continue
            pos = df[df[finding] == 1].drop_duplicates("subject_id")
            neg = df[df[finding] == 0].drop_duplicates("subject_id")
            for pool, label in ((pos, 1), (neg, 0)):
                n = min(half, len(pool))
                if n:
                    for _, r in pool.sample(n=n, random_state=self._rs()).iterrows():
                        rows.append(self._chexpert_row(r, idx, finding, label))
                        idx += 1
        normals = df[df["no_finding"] == 1].drop_duplicates("subject_id")
        n_norm = min(self.chexpert_normals, len(normals))
        if n_norm:
            for _, r in normals.sample(n=n_norm, random_state=self._rs()).iterrows():
                rows.append(self._chexpert_row(r, idx, "no_finding", 1))
                idx += 1
        return pd.DataFrame(rows)[self.UNIFIED_COLS].copy()

    def _chexpert_row(self, r, case_idx: int, finding: str, label: int) -> Dict:
        return {"case_id": f"chexpert_{case_idx:05d}", "source": "chexpert", "image_path": str(r["jpg_rel_path"]), "report_path": r.get("report_path"),
                "dicom_id": r["dicom_id"], "subject_id": r["subject_id"], "study_id": r.get("study_id"), "view": r["view"], "age": r.get("age"),
                "gender": r.get("gender"), "finding": finding, "label": label, "label_text": None, "box_x": None, "box_y": None, "box_w": None,
                "box_h": None, "original_sentence": None, "error_sentence": None, "error_type": None, "error_present": None}

    def build_chexpert_manifest(self) -> pd.DataFrame:
        chexpert = self.build_chexpert_subset()
        pool = self._load_chexpert().copy()
        pool = pool[pool["view"] == "Frontal"].copy()
        swap_paths = []
        for _, row in chexpert.iterrows():
            finding, subj, label = row.get("finding"), row["subject_id"], row["label"]
            cand = None
            if finding in self.CHEXPERT_FINDINGS and pd.notna(label):
                cand = pool[(pool[finding] == label) & (pool["subject_id"] != subj)]
            elif finding == "no_finding":
                cand = pool[(pool["no_finding"] == 1) & (pool["subject_id"] != subj)]
            if cand is None or len(cand) == 0:
                cand = pool[pool["subject_id"] != subj]
            swap_paths.append(cand.sample(n=1, random_state=self._rs()).iloc[0]["jpg_rel_path"] if len(cand) else None)
        chexpert["swap_image_path"] = swap_paths
        for col in ["irrelevant_box_x", "irrelevant_box_y", "irrelevant_box_w", "irrelevant_box_h"]:
            chexpert[col] = None
        return chexpert

    def build(self) -> pd.DataFrame:
        mscxr = self.build_mscxr_subset()
        mimic = self.build_mimic_subset(set(mscxr["dicom_id"].dropna().unique()))
        rexerr = self.build_rexerr_subset()
        manifest = pd.concat([mscxr, mimic, rexerr], ignore_index=True)
        return self.add_irrelevant_masks(self.add_swap_candidates(manifest))


def _diff(frozen: pd.DataFrame, rebuilt: pd.DataFrame, name: str) -> Dict:
    f_ids = set(frozen["dicom_id"].dropna().astype(str))
    r_ids = set(rebuilt["dicom_id"].dropna().astype(str))
    return {"manifest": name, "frozen_rows": len(frozen), "rebuilt_rows": len(rebuilt),
            "frozen_by_source": frozen["source"].value_counts().to_dict(), "rebuilt_by_source": rebuilt["source"].value_counts().to_dict(),
            "shared_images": len(f_ids & r_ids), "images_only_in_frozen": len(f_ids - r_ids), "images_only_in_rebuilt": len(r_ids - f_ids)}


def main_check_frozen_manifests(cfg_path: str) -> None:
    cfg = read_config(cfg_path)["CausalAudit"]
    d = cfg["data"]
    out_dir = ensure_dir(os.path.join(os.path.dirname(d["mimic_manifest"]), "rebuild_check"))
    builder = ProbeSetBuilder(mimic_csv=d["mimic_master_csv"], mscxr_csv=d["mscxr_csv"], rexerr_csv=d["rexerr_csv"],
                              chexpert_csv=d["chexpert_master_csv"], **JUNE_PARAMETERS)
    rebuilt_mimic = builder.build()
    rebuilt_chex = builder.build_chexpert_manifest()
    write_csv_atomic(rebuilt_mimic, os.path.join(out_dir, "probe_set_manifest_rebuilt.csv"))
    write_csv_atomic(rebuilt_chex, os.path.join(out_dir, "chexpert_manifest_rebuilt.csv"))
    reports = [_diff(read_csv_defensively(d["frozen_mimic_manifest"]), rebuilt_mimic, "mimic"),
               _diff(read_csv_defensively(d["frozen_chexpert_manifest"]), rebuilt_chex, "chexpert")]
    for r in reports:
        print(f"[rebuild_check] {r}", flush=True)
    append_status(status_path(cfg, "rebuild_check"), f"{reports}")
