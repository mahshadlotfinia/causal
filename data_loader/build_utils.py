"""
data_loader/build_utils.py
Created on September 14, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


FRONTAL_VIEWS = ("PA", "AP")
FINDING_PRESENCE_SOURCES = ("ms_cxr", "mimic_cxr", "chexpert")


def read_csv_defensively(path: str, usecols: Optional[Sequence[str]] = None,
                         dtype=None) -> pd.DataFrame:
    kwargs = {}
    if usecols is not None:
        kwargs["usecols"] = list(usecols)
    if dtype is not None:
        kwargs["dtype"] = dtype
    try:
        return pd.read_csv(path, encoding="utf-8", low_memory=False, float_precision="round_trip", **kwargs)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1", low_memory=False, float_precision="round_trip", **kwargs)


def dicom_id_from_path(rel_path: pd.Series) -> pd.Series:
    return rel_path.astype(str).str.extract(r"([^/]+)\.jpg$", expand=False)


def load_mimic_master(cfg: Dict, usecols: Optional[Sequence[str]] = None) -> pd.DataFrame:
    cols = None if usecols is None else sorted(set(usecols) | {"jpg_rel_path"})
    df = read_csv_defensively(cfg["data"]["mimic_master_csv"], usecols=cols)
    df["dicom_id"] = dicom_id_from_path(df["jpg_rel_path"])
    return df


def load_chexpert_master(cfg: Dict, usecols: Optional[Sequence[str]] = None) -> pd.DataFrame:
    cols = None if usecols is None else sorted(set(usecols) | {"jpg_rel_path"})
    df = read_csv_defensively(cfg["data"]["chexpert_master_csv"], usecols=cols)
    df["dicom_id"] = dicom_id_from_path(df["jpg_rel_path"])
    df["study_id"] = df["jpg_rel_path"].astype(str).str.extract(r"(patient\d+/study\d+)", expand=False)
    if "gender" in df.columns:
        df["gender"] = df["gender"].map({"Female": "F", "Male": "M", "F": "F", "M": "M"})
    return df


def resolve_image_path(cfg: Dict, dataset: str, rel_path: str, resolution: int) -> str:
    folder = cfg["data"]["resolution_folders"][str(int(resolution))]
    rel = str(rel_path)
    if dataset == "chexpert":
        rel = rel.replace("CheXpert-v1.0/", f"CheXpert-v1.0/{folder}/", 1)
        return os.path.join(cfg["data"]["chexpert_image_root"], rel)
    rel = rel.replace("files/", f"{folder}/", 1)
    return os.path.join(cfg["data"]["mimic_image_root"], rel)


def load_rgb(path: str):
    from PIL import Image
    return Image.open(path).convert("RGB")


def apply_mask(img, x: int, y: int, w: int, h: int):
    from PIL import ImageDraw
    img = img.copy()
    ImageDraw.Draw(img).rectangle([x, y, x + w, y + h], fill=(0, 0, 0))
    return img


def scale_box(x: float, y: float, w: float, h: float, resolution: int,
              box_space: int) -> Tuple[int, int, int, int]:
    s = float(resolution) / float(box_space)
    return int(x * s), int(y * s), int(w * s), int(h * s)


def corner_box(bx: int, by: int, bw: int, bh: int, size: int) -> Tuple[int, int, int, int]:
    bcx, bcy = bx + bw / 2.0, by + bh / 2.0
    corners = [(0, 0), (size - bw, 0), (0, size - bh), (size - bw, size - bh)]
    cx, cy = max(corners, key=lambda c: (c[0] + bw / 2.0 - bcx) ** 2 + (c[1] + bh / 2.0 - bcy) ** 2)
    cx = int(max(0, min(size - bw, cx)))
    cy = int(max(0, min(size - bh, cy)))
    return cx, cy, bw, bh


def _overlaps(ax, ay, aw, ah, bx, by, bw, bh) -> bool:
    return not (ax + aw <= bx or bx + bw <= ax or ay + ah <= by or by + bh <= ay)


def matched_box(bx: int, by: int, bw: int, bh: int, size: int) -> Tuple[int, int, int, int, str]:
    mx = int(max(0, min(size - bw, int(round(size - (bx + bw))))))
    my = int(max(0, min(size - bh, by)))
    if not _overlaps(mx, my, bw, bh, bx, by, bw, bh):
        return mx, my, bw, bh, "mirrored"
    vertical = [(abs(cy - my), cy) for cy in range(0, size - bh + 1) if not _overlaps(mx, cy, bw, bh, bx, by, bw, bh)]
    if vertical:
        return mx, int(min(vertical)[1]), bw, bh, "shifted"
    target_left = (bx + bw / 2.0) < size / 2.0
    hx = bx + bw if target_left else bx - bw
    hx = int(max(0, min(size - bw, hx)))
    if not _overlaps(hx, my, bw, bh, bx, by, bw, bh):
        return hx, my, bw, bh, "shifted"
    cx, cy, cw, ch = corner_box(bx, by, bw, bh, size)
    return cx, cy, cw, ch, "corner_fallback"


def display_name(finding: str, cfg: Dict) -> str:
    names = cfg["prompts"]["display_names"]
    return names.get(finding, str(finding).replace("_", " "))


def pixel_digest(path: str) -> str:
    import hashlib
    return hashlib.blake2b(np.asarray(load_rgb(path), dtype=np.uint8).tobytes(), digest_size=8).hexdigest()


def build_prompt(row: Dict, variant: str, cfg: Dict) -> str:
    source = str(row.get("source", ""))
    if source in FINDING_PRESENCE_SOURCES:
        display = display_name(str(row.get("finding") or ""), cfg)
        if variant == "brief":
            return f"Is {display} present? Yes or No."
        if variant == "clinical":
            return (f"You are a radiologist reviewing a chest X-ray. Is {display} present? "
                    f"Answer with a single word: Yes or No.")
        return f"Is {display} present in this chest X-ray? Answer with a single word: Yes or No."
    if source == "rexerr":
        sentence = str(row.get("error_sentence") or "")
        if variant == "brief":
            return f"Is this sentence accurate for this X-ray? \"{sentence}\" Yes or No."
        if variant == "clinical":
            return (f"You are a radiologist. Does the following sentence accurately describe findings "
                    f"in this chest X-ray?\nSentence: \"{sentence}\"\nAnswer with a single word: Yes or No.")
        return (f"Does the following sentence accurately describe the findings visible in this chest X-ray?\n"
                f"Sentence: \"{sentence}\"\nAnswer with a single word: Yes or No.")
    raise ValueError(f"unknown source {source!r} for case {row.get('case_id')}")


def ground_truth(row: Dict) -> int:
    source = str(row.get("source", ""))
    if source in FINDING_PRESENCE_SOURCES:
        return int(row["label"])
    if source == "rexerr":
        return 1 - int(float(row["error_present"]))
    raise ValueError(f"unknown source {source!r}")


def rexerr_class(error_type, error_present, cfg: Dict) -> Optional[str]:
    if error_type is None or (isinstance(error_type, float) and np.isnan(error_type)):
        return None
    if int(float(error_present)) == 0 or error_type == cfg["data"]["rexerr_control_type"]:
        return "control"
    if error_type in cfg["data"]["rexerr_text_only_types"]:
        return "text_only"
    if error_type in cfg["data"]["rexerr_image_dependent_types"]:
        return "image_dependent"
    raise ValueError(f"unknown ReXErr error type {error_type!r}")


def absent_pool(df: pd.DataFrame, finding: str, cfg: Dict) -> pd.DataFrame:
    d = cfg["data"]
    zero = df[df[finding] == d["absent_code"]]
    if zero["subject_id"].nunique() >= int(d["min_absent_patients"]):
        return zero
    return df[df[finding].isin([d["absent_code"], d["fallback_absent_code"]])]


def present_pool(df: pd.DataFrame, finding: str, cfg: Dict) -> pd.DataFrame:
    return df[df[finding] == cfg["data"]["present_code"]]


def abnormal_pool(df: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    cols = cfg["data"]["pathology_findings"]
    return df[(df[cols] == cfg["data"]["present_code"]).any(axis=1)]


def normal_pool(df: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    return df[df["no_finding"] == cfg["data"]["present_code"]]


def allocate_counts(strata: List[str], n_total: int, rng: np.random.Generator) -> Dict[str, int]:
    base, extra = divmod(n_total, len(strata))
    counts = {s: base for s in strata}
    for s in rng.choice(strata, size=extra, replace=False):
        counts[str(s)] += 1
    return counts


def stratified_sample(df: pd.DataFrame, stratum_col: str, n_total: int, seed: int,
                      counts: Optional[Dict[str, int]] = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = df.sort_values("case_id").reset_index(drop=True)
    strata = sorted(df[stratum_col].astype(str).unique())
    if counts is None:
        counts = allocate_counts(strata, n_total, rng)
    parts = []
    for s in strata:
        sub = df[df[stratum_col].astype(str) == s]
        k = min(int(counts.get(s, 0)), len(sub))
        parts.append(sub.iloc[np.sort(rng.choice(len(sub), size=k, replace=False))])
    return pd.concat(parts, ignore_index=True)
