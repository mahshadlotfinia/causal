"""
data_loader/build_probe_set.py
Created May 22, 2026

Builds the unified manifest CSV combining MS-CXR, MIMIC-CXR, and ReXErr-v1
test subsets for the causal grounding audit of medical VLMs.

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia/
"""

import json
from pathlib import Path
from typing import Dict, Optional, Set

import numpy as np
import pandas as pd


class ProbeSetBuilder:
    FRONTAL_VIEWS = {"PA", "AP"}

    CHEXPERT_FINDINGS = [
        "atelectasis", "cardiomegaly", "consolidation", "edema",
        "enlarged_cardiomediastinum", "fracture", "lung_lesion",
        "lung_opacity", "pleural_effusion", "pleural_other",
        "pneumonia", "pneumothorax", "support_devices",
    ]

    MSCXR_TO_MIMIC = {
        "Atelectasis": "atelectasis",
        "Cardiomegaly": "cardiomegaly",
        "Consolidation": "consolidation",
        "Edema": "edema",
        "Lung Opacity": "lung_opacity",
        "Pleural Effusion": "pleural_effusion",
        "Pneumonia": "pneumonia",
        "Pneumothorax": "pneumothorax",
    }

    IMAGE_DEPENDENT_ERROR_TYPES = {
        "Change location",
        "Change severity",
        "False prediction",
        "False negation",
        "Change position of device",
        "Add medical device",
        "Change name of device",
        "Change view",
    }

    TEXT_ONLY_ERROR_TYPES = {
        "Add typo",
        "Change to homophone",
        "Add repetition",
    }

    UNIFIED_COLS = [
        "case_id", "source", "image_path", "report_path", "dicom_id",
        "subject_id", "study_id", "view", "age", "gender",
        "finding", "label", "label_text",
        "box_x", "box_y", "box_w", "box_h",
        "original_sentence", "error_sentence",
        "error_type", "error_present",
    ]

    def __init__(
        self,
        mimic_csv: str,
        mscxr_csv: str,
        rexerr_csv: str,
        chexpert_csv: str = None,
        chexpert_plus_csv: str = None,
        target_resolution: int = 224,
        min_box_pixels: int = 50,
        mimic_per_finding: int = 100,
        mimic_normals: int = 100,
        rexerr_total: int = 800,
        rexerr_image_dep_fraction: float = 0.70,
        rexerr_text_only_fraction: float = 0.15,
        rexerr_no_error_fraction: float = 0.15,
        mscxr_per_finding_cap: int = 100,
        chexpert_per_finding: int = 100,
        chexpert_normals: int = 100,
        random_seed: int = 42,
    ):
        assert abs(
            rexerr_image_dep_fraction
            + rexerr_text_only_fraction
            + rexerr_no_error_fraction
            - 1.0
        ) < 1e-6, "ReXErr fractions must sum to 1.0"

        self.mimic_csv = mimic_csv
        self.mscxr_csv = mscxr_csv
        self.rexerr_csv = rexerr_csv
        self.chexpert_csv = chexpert_csv
        self.chexpert_plus_csv = chexpert_plus_csv

        self.target_resolution = target_resolution
        self.min_box_pixels = min_box_pixels

        self.mimic_per_finding = mimic_per_finding
        self.mimic_normals = mimic_normals
        self.mscxr_per_finding_cap = mscxr_per_finding_cap

        self.rexerr_total = rexerr_total
        self.rexerr_image_dep_fraction = rexerr_image_dep_fraction
        self.rexerr_text_only_fraction = rexerr_text_only_fraction
        self.rexerr_no_error_fraction = rexerr_no_error_fraction

        self.chexpert_per_finding = chexpert_per_finding
        self.chexpert_normals = chexpert_normals

        self.rng = np.random.default_rng(random_seed)
        self._mimic_full: Optional[pd.DataFrame] = None
        self._chexpert_full: Optional[pd.DataFrame] = None


    def _load_mimic(self) -> pd.DataFrame:
        if self._mimic_full is None:
            df = pd.read_csv(self.mimic_csv)
            df["dicom_id"] = df["jpg_rel_path"].str.extract(
                r"([^/]+)\.jpg$", expand=False
            )
            before = len(df)
            df = df[df["age"].notna() & df["gender"].notna()].copy()
            print(f"[_load_mimic] Dropped {before - len(df)} rows with missing age/gender. Remaining: {len(df)}")
            self._mimic_full = df
        return self._mimic_full

    def load_mscxr(self) -> pd.DataFrame:
        return pd.read_csv(self.mscxr_csv)

    def load_rexerr(self) -> pd.DataFrame:
        return pd.read_csv(self.rexerr_csv)

    def _rs(self) -> int:
        """Random integer for sklearn / pandas random_state slot."""
        return int(self.rng.integers(2**31))


    def build_mscxr_subset(self) -> pd.DataFrame:
        df = self.load_mscxr().copy()
        target = self.target_resolution

        # Scale boxes to target resolution
        sx = target / df["image_width"]
        sy = target / df["image_height"]
        df["x_t"] = df["x"] * sx
        df["y_t"] = df["y"] * sy
        df["w_t"] = df["w"] * sx
        df["h_t"] = df["h"] * sy

        # Filter: each side >= min_box_pixels at target resolution
        keep = (df["w_t"] >= self.min_box_pixels) & (
            df["h_t"] >= self.min_box_pixels
        )
        df = df[keep].reset_index(drop=True)

        # Join with MIMIC for view, demographics, and report path; frontal only
        mimic = self._load_mimic()
        df = df.merge(
            mimic[["dicom_id", "subject_id", "study_id", "view", "report_rel_path", "age", "gender"]]
            .drop_duplicates(subset=["dicom_id"]),
            on="dicom_id",
            how="left",
        )
        df = df[df["view"].isin(self.FRONTAL_VIEWS)].reset_index(drop=True)

        # Standardize finding name to CheXpert vocabulary
        df["finding"] = df["category_name"].map(self.MSCXR_TO_MIMIC)
        df = df[df["finding"].notna()].reset_index(drop=True)

        # Cap per finding to avoid cardiomegaly (and any future dominant finding)
        # dominating MS-CXR analyses
        if self.mscxr_per_finding_cap is not None:
            capped_chunks = []
            for finding_name in df["finding"].unique():
                sub = df[df["finding"] == finding_name]
                n = min(self.mscxr_per_finding_cap, len(sub))
                capped_chunks.append(sub.sample(n=n, random_state=self._rs()))
            df = pd.concat(capped_chunks, ignore_index=True)

        df["case_id"] = [f"mscxr_{i:05d}" for i in range(len(df))]
        df["source"] = "ms_cxr"
        df["image_path"] = "mimic-cxr-jpg/" + df["path"]
        df["report_path"] = df["report_rel_path"]
        df["label"] = 1  # MS-CXR cases are always positive for the queried finding
        df["box_x"] = df["x_t"].round().astype(int)
        df["box_y"] = df["y_t"].round().astype(int)
        df["box_w"] = df["w_t"].round().astype(int)
        df["box_h"] = df["h_t"].round().astype(int)

        for c in ["original_sentence", "error_sentence", "error_type", "error_present"]:
            df[c] = None

        return df[self.UNIFIED_COLS].copy()


    def build_mimic_subset(self, exclude_dicoms: Set[str]) -> pd.DataFrame:
        df = self._load_mimic().copy()
        df = df[df["split"] == "test"]
        df = df[df["view"].isin(self.FRONTAL_VIEWS)]
        df = df[~df["dicom_id"].isin(exclude_dicoms)]

        half = self.mimic_per_finding // 2
        rows = []
        idx = 0

        for finding in self.CHEXPERT_FINDINGS:
            # Positives: clean label=1 only
            pos = (
                df[df[finding] == 1]
                .drop_duplicates(subset=["subject_id"])
            )
            # Negatives: prefer explicit 0; fall back to "not mentioned" (3)
            # when 0 is insufficient (e.g., edema has no label=0 in test set)
            neg_clean = df[df[finding] == 0].drop_duplicates(subset=["subject_id"])
            if len(neg_clean) < half:
                neg_fallback = df[df[finding] == 3].drop_duplicates(
                    subset=["subject_id"]
                )
                neg = pd.concat([neg_clean, neg_fallback]).drop_duplicates(
                    subset=["subject_id"]
                )
                if len(neg_clean) < half:
                    print(
                        f"[WARN] {finding}: only {len(neg_clean)} label=0 negatives; "
                        f"supplemented with {min(len(neg_fallback), half - len(neg_clean))} "
                        f"label=3 (not mentioned) cases."
                    )
            else:
                neg = neg_clean

            n_pos = min(half, len(pos))
            n_neg = min(half, len(neg))

            if n_pos < half:
                print(f"[WARN] {finding}: requested {half} positives, found {n_pos}.")
            if n_neg < half:
                print(f"[WARN] {finding}: requested {half} negatives, found {n_neg}.")

            if n_pos:
                for _, r in pos.sample(n=n_pos, random_state=self._rs()).iterrows():
                    rows.append(self._mimic_row(r, idx, finding, 1))
                    idx += 1
            if n_neg:
                for _, r in neg.sample(n=n_neg, random_state=self._rs()).iterrows():
                    rows.append(self._mimic_row(r, idx, finding, 0))
                    idx += 1

        # Normals pool
        normals = df[df["no_finding"] == 1].drop_duplicates(subset=["subject_id"])
        n_norm = min(self.mimic_normals, len(normals))
        if n_norm:
            for _, r in normals.sample(n=n_norm, random_state=self._rs()).iterrows():
                rows.append(self._mimic_row(r, idx, "no_finding", 1))
                idx += 1

        return pd.DataFrame(rows)[self.UNIFIED_COLS].copy()

    def _mimic_row(self, r, case_idx: int, finding: str, label: int) -> Dict:
        return {
            "case_id": f"mimic_{case_idx:05d}",
            "source": "mimic_cxr",
            "image_path": r["jpg_rel_path"],
            "report_path": r["report_rel_path"],
            "dicom_id": r["dicom_id"],
            "subject_id": r["subject_id"],
            "study_id": r["study_id"],
            "view": r["view"],
            "age": r.get("age", None),
            "gender": r.get("gender", None),
            "finding": finding,
            "label": label,
            "label_text": None,
            "box_x": None, "box_y": None, "box_w": None, "box_h": None,
            "original_sentence": None, "error_sentence": None,
            "error_type": None, "error_present": None,
        }


    def build_rexerr_subset(self) -> pd.DataFrame:
        df = self.load_rexerr().copy()

        # Drop ambiguous error labels (2.0)
        df = df[df["error_present"].isin([0.0, 1.0])].copy()

        # Rows with null original_sentence are unusable (no ground truth claim)
        df = df[df["original_sentence"].notna()].copy()

        # dicom_id may be comma-separated; explode and keep frontals only
        df["dicom_id"] = df["dicom_id"].fillna("").astype(str)
        df["_dlist"] = df["dicom_id"].str.split(",")
        df = df.explode("_dlist")
        df["dicom_id"] = df["_dlist"].str.strip()
        df = df.drop(columns=["_dlist"])
        df = df[df["dicom_id"] != ""]

        mimic = self._load_mimic()
        merge_cols = ["dicom_id", "view", "jpg_rel_path", "report_rel_path", "age", "gender"] + self.CHEXPERT_FINDINGS
        df = df.merge(
            mimic[merge_cols].drop_duplicates(subset=["dicom_id"]),
            on="dicom_id",
            how="inner",
        )
        df = df[df["view"].isin(self.FRONTAL_VIEWS)]

        # One frontal image per ReXErr sentence-pair
        df = df.drop_duplicates(
            subset=["study_id", "original_sentence", "error_sentence"],
            keep="first",
        ).reset_index(drop=True)

        # Bucket the rows
        is_no_error = df["error_present"] == 0.0
        is_text_only = (
            df["error_type"].isin(self.TEXT_ONLY_ERROR_TYPES)
            & (df["error_present"] == 1.0)
        )
        is_image_dep = (
            df["error_type"].isin(self.IMAGE_DEPENDENT_ERROR_TYPES)
            & (df["error_present"] == 1.0)
        )

        df_img = df[is_image_dep]
        df_txt = df[is_text_only]
        df_no = df[is_no_error]

        n_img = int(round(self.rexerr_total * self.rexerr_image_dep_fraction))
        n_txt = int(round(self.rexerr_total * self.rexerr_text_only_fraction))
        n_no = self.rexerr_total - n_img - n_txt

        chunks = []
        # Image-dependent: stratify by error_type
        img_types = sorted(set(df_img["error_type"].unique()))
        if img_types:
            per = max(1, n_img // len(img_types))
            for et in img_types:
                sub = df_img[df_img["error_type"] == et]
                k = min(per, len(sub))
                if k:
                    chunks.append(sub.sample(n=k, random_state=self._rs()))

        # Text-only: stratify by error_type
        txt_types = sorted(set(df_txt["error_type"].unique()))
        if txt_types:
            per = max(1, n_txt // len(txt_types))
            for et in txt_types:
                sub = df_txt[df_txt["error_type"] == et]
                k = min(per, len(sub))
                if k:
                    chunks.append(sub.sample(n=k, random_state=self._rs()))

        # No-error controls (random sample)
        if n_no > 0 and len(df_no) > 0:
            k = min(n_no, len(df_no))
            chunks.append(df_no.sample(n=k, random_state=self._rs()))

        if not chunks:
            return pd.DataFrame(columns=self.UNIFIED_COLS)

        out = pd.concat(chunks, ignore_index=True)
        out["case_id"] = [f"rexerr_{i:05d}" for i in range(len(out))]
        out["source"] = "rexerr"
        out["image_path"] = out["jpg_rel_path"]
        out["report_path"] = out["report_rel_path"]
        out["age"] = out["age"]
        out["gender"] = out["gender"]

        # Infer a single canonical finding from CheXpert labels of the
        # underlying image. Pick the first positive finding if any,
        # otherwise None. This is used downstream by the swap candidate
        # routine to pick a label-matched swap when possible.
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
        mimic = self._load_mimic()
        pool = mimic[mimic["view"].isin(self.FRONTAL_VIEWS)].copy()

        swap_paths = []
        for _, row in manifest.iterrows():
            finding = row.get("finding")
            subj = row["subject_id"]
            label = row["label"]

            cand = None
            if finding in self.CHEXPERT_FINDINGS and pd.notna(label):
                # Label-matched on the queried finding
                cand = pool[
                    (pool[finding] == label) & (pool["subject_id"] != subj)
                ]
            elif finding == "no_finding":
                cand = pool[
                    (pool["no_finding"] == 1) & (pool["subject_id"] != subj)
                ]

            # Fallback: any different-patient frontal
            if cand is None or len(cand) == 0:
                cand = pool[pool["subject_id"] != subj]

            if len(cand) > 0:
                pick = cand.sample(n=1, random_state=self._rs()).iloc[0]
                swap_paths.append(pick["jpg_rel_path"])
            else:
                swap_paths.append(None)

        out = manifest.copy()
        out["swap_image_path"] = swap_paths
        return out


    def add_irrelevant_masks(self, manifest: pd.DataFrame) -> pd.DataFrame:
        target = self.target_resolution
        ix, iy, iw, ih = [], [], [], []

        for _, row in manifest.iterrows():
            if pd.isna(row["box_x"]):
                ix.append(None); iy.append(None)
                iw.append(None); ih.append(None)
                continue

            bx = int(row["box_x"]); by = int(row["box_y"])
            bw = int(row["box_w"]); bh = int(row["box_h"])
            bcx, bcy = bx + bw / 2.0, by + bh / 2.0

            corners = [
                (0, 0),
                (target - bw, 0),
                (0, target - bh),
                (target - bw, target - bh),
            ]
            cx, cy = max(
                corners,
                key=lambda c: (c[0] + bw / 2.0 - bcx) ** 2
                + (c[1] + bh / 2.0 - bcy) ** 2,
            )
            cx = int(max(0, min(target - bw, cx)))
            cy = int(max(0, min(target - bh, cy)))

            ix.append(cx); iy.append(cy); iw.append(bw); ih.append(bh)

        out = manifest.copy()
        out["irrelevant_box_x"] = ix
        out["irrelevant_box_y"] = iy
        out["irrelevant_box_w"] = iw
        out["irrelevant_box_h"] = ih
        return out


    def _load_chexpert(self) -> pd.DataFrame:
        if self._chexpert_full is None:
            assert self.chexpert_csv, "chexpert_csv path not set."
            df = pd.read_csv(self.chexpert_csv)

            # Derive dicom_id from filename (last path component without extension)
            df["dicom_id"] = df["jpg_rel_path"].str.extract(
                r"([^/]+)\.jpg$", expand=False
            )
            # Derive study_id from path (e.g. CheXpert-v1.0/train/patient00001/study1/...)
            df["study_id"] = df["jpg_rel_path"].str.extract(
                r"(patient\d+/study\d+)", expand=False
            )
            # Normalize gender to F/M to match MIMIC convention
            df["gender"] = df["gender"].map(
                {"Female": "F", "Male": "M", "Unknown": None}
            )
            # Drop rows without age or gender (same policy as MIMIC)
            before = len(df)
            df = df[df["age"].notna() & df["gender"].notna()].copy()
            print(
                f"[_load_chexpert] Dropped {before - len(df)} rows with "
                f"missing age/gender. Remaining: {len(df)}"
            )
            self._chexpert_full = df
        return self._chexpert_full


    def build_chexpert_subset(self) -> pd.DataFrame:
        """
        Builds a balanced CheXpert test-split probe subset following the same
        design as build_mimic_subset.

        Key differences vs MIMIC:
        - No bounding boxes: target_mask and irrelevant_mask conditions are
          not applicable. Only original and swap conditions apply.
        - view column is 'Frontal'/'Lateral'; filter on 'Frontal'.
          AP/PA distinction comes from the AP_PA column.
        - Report text joined from CheXpert Plus via jpg_rel_path.
        - Label encoding: 0=negative, 1=positive, 2=uncertain.
          No label=3 (not-mentioned) exists in CheXpert; fallback for
          negatives is not needed but handled gracefully.
        """
        df = self._load_chexpert().copy()
        df = df[df["split"] == "test"]
        df = df[df["view"] == "Frontal"].copy()

        # Use AP_PA as the stored view (matches MIMIC PA/AP convention)
        df["view"] = df["AP_PA"].fillna("AP")

        # Join reports from CheXpert Plus if provided
        if self.chexpert_plus_csv:
            plus = pd.read_csv(
                self.chexpert_plus_csv,
                usecols=["jpg_rel_path", "report"],
            )
            df = df.merge(plus, on="jpg_rel_path", how="left")
            df = df.rename(columns={"report": "report_path"})
        else:
            df["report_path"] = None

        half = self.chexpert_per_finding // 2
        rows = []
        idx = 0

        for finding in self.CHEXPERT_FINDINGS:
            if finding not in df.columns:
                continue

            pos = df[df[finding] == 1].drop_duplicates(subset=["subject_id"])
            neg_clean = df[df[finding] == 0].drop_duplicates(subset=["subject_id"])

            # CheXpert has no label=3; if label=0 is insufficient, warn and
            # use what is available (no fallback)
            if len(neg_clean) < half:
                print(
                    f"[WARN] chexpert {finding}: only {len(neg_clean)} "
                    f"label=0 negatives available (requested {half})."
                )
            neg = neg_clean

            n_pos = min(half, len(pos))
            n_neg = min(half, len(neg))

            if n_pos < half:
                print(f"[WARN] chexpert {finding}: requested {half} positives, found {n_pos}.")
            if n_neg < half:
                print(f"[WARN] chexpert {finding}: requested {half} negatives, found {n_neg}.")

            if n_pos:
                for _, r in pos.sample(n=n_pos, random_state=self._rs()).iterrows():
                    rows.append(self._chexpert_row(r, idx, finding, 1))
                    idx += 1
            if n_neg:
                for _, r in neg.sample(n=n_neg, random_state=self._rs()).iterrows():
                    rows.append(self._chexpert_row(r, idx, finding, 0))
                    idx += 1

        # Normals pool
        if "no_finding" in df.columns:
            normals = df[df["no_finding"] == 1].drop_duplicates(subset=["subject_id"])
            n_norm = min(self.chexpert_normals, len(normals))
            if n_norm:
                for _, r in normals.sample(n=n_norm, random_state=self._rs()).iterrows():
                    rows.append(self._chexpert_row(r, idx, "no_finding", 1))
                    idx += 1

        return pd.DataFrame(rows)[self.UNIFIED_COLS].copy()

    def _chexpert_row(self, r, case_idx: int, finding: str, label: int) -> Dict:
        return {
            "case_id": f"chexpert_{case_idx:05d}",
            "source": "chexpert",
            "image_path": str(r["jpg_rel_path"]),
            "report_path": r.get("report_path", None),
            "dicom_id": r["dicom_id"],
            "subject_id": r["subject_id"],
            "study_id": r.get("study_id", None),
            "view": r["view"],
            "age": r.get("age", None),
            "gender": r.get("gender", None),
            "finding": finding,
            "label": label,
            "label_text": None,
            "box_x": None, "box_y": None, "box_w": None, "box_h": None,
            "original_sentence": None, "error_sentence": None,
            "error_type": None, "error_present": None,
        }


    def build_chexpert_manifest(
        self, output_csv: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Builds a standalone CheXpert probe manifest (not merged with MIMIC).
        Swap candidates are drawn from within the CheXpert pool.

        Use this manifest separately from the MIMIC-based probe_set_manifest.csv.
        The two manifests feed the same inference harness; results are compared
        in the generalization analysis (Step 5 secondary analysis).
        """
        chexpert = self.build_chexpert_subset()

        # Swap candidates from CheXpert pool (different patient, same finding/label)
        pool = self._load_chexpert().copy()
        pool = pool[pool["view"] == "Frontal"].copy()
        pool["view"] = pool["AP_PA"].fillna("AP")

        swap_paths = []
        for _, row in chexpert.iterrows():
            finding = row.get("finding")
            subj    = row["subject_id"]
            label   = row["label"]

            cand = None
            if finding in self.CHEXPERT_FINDINGS and pd.notna(label):
                cand = pool[
                    (pool[finding] == label) & (pool["subject_id"] != subj)
                ]
            elif finding == "no_finding":
                cand = pool[
                    (pool["no_finding"] == 1) & (pool["subject_id"] != subj)
                ]

            if cand is None or len(cand) == 0:
                cand = pool[pool["subject_id"] != subj]

            if len(cand) > 0:
                pick = cand.sample(n=1, random_state=self._rs()).iloc[0]
                swap_paths.append(pick["jpg_rel_path"])
            else:
                swap_paths.append(None)

        chexpert["swap_image_path"] = swap_paths
        # No boxes → irrelevant box columns are all None
        for col in ["irrelevant_box_x", "irrelevant_box_y",
                    "irrelevant_box_w", "irrelevant_box_h"]:
            chexpert[col] = None

        if output_csv:
            chexpert.to_csv(output_csv, index=False)
            print(f"[build_chexpert_manifest] Saved {len(chexpert)} cases to {output_csv}")

        return chexpert


    def build(self, output_csv: Optional[str] = None) -> pd.DataFrame:
        mscxr = self.build_mscxr_subset()
        mscxr_dicoms = set(mscxr["dicom_id"].dropna().unique())

        mimic = self.build_mimic_subset(mscxr_dicoms)
        rexerr = self.build_rexerr_subset()

        manifest = pd.concat([mscxr, mimic, rexerr], ignore_index=True)
        manifest = self.add_swap_candidates(manifest)
        manifest = self.add_irrelevant_masks(manifest)

        if output_csv:
            manifest.to_csv(output_csv, index=False)
        return manifest

    def summary(self, manifest: pd.DataFrame) -> Dict:
        return {
            "total": int(len(manifest)),
            "by_source": manifest["source"].value_counts().to_dict(),
            "with_swap": int(manifest["swap_image_path"].notna().sum()),
            "with_target_box": int(manifest["box_x"].notna().sum()),
            "mscxr_by_finding": (
                manifest[manifest["source"] == "ms_cxr"]["finding"]
                .value_counts().to_dict()
            ),
            "mimic_by_finding_label": (
                manifest[manifest["source"] == "mimic_cxr"]
                .groupby(["finding", "label"]).size().to_dict()
            ),
            "rexerr_by_error_type": (
                manifest[manifest["source"] == "rexerr"]["error_type"]
                .value_counts(dropna=False).to_dict()
            ),
            "unique_subjects": int(manifest["subject_id"].nunique()),
        }


if __name__ == "__main__":
    builder = ProbeSetBuilder(
        mimic_csv="/home/soroosh/Documents/datasets/XRay/MIMIC/mimic_master_list.csv",
        mscxr_csv="/0/soroosh/Documents/datasets/XRay/MS-CXR/MS_CXR_Local_Alignment_v1_1_0.csv",
        rexerr_csv="/home/soroosh/Documents/datasets/XRay/ReXErr-v1/Sentence_level/ReXErr-sentence-level_test.csv",
        chexpert_csv="/home/soroosh/Documents/datasets/XRay/CheXpert-v1.0/chexpert_master_list_20percenttest.csv",
        chexpert_plus_csv="/0home/soroosh/Documents/datasets/XRay/CheXpert-v1.0/chexpert_plus.csv",
        target_resolution=224,
        min_box_pixels=50,
        mimic_per_finding=100,
        mimic_normals=100,
        mscxr_per_finding_cap=100,
        rexerr_total=800,
        rexerr_image_dep_fraction=0.70,
        rexerr_text_only_fraction=0.15,
        rexerr_no_error_fraction=0.15,
        chexpert_per_finding=100,
        chexpert_normals=100,
        random_seed=42,
    )

    # Build main MIMIC-based manifest (MS-CXR + MIMIC + ReXErr)
    # manifest = builder.build(output_csv="probe_set_manifest.csv")

    # Build separate CheXpert generalization manifest
    builder.build_chexpert_manifest(output_csv="chexpert_manifest.csv")



