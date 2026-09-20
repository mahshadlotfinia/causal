"""
data_loader/probe_dataset.py
Created on September 14, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import os
from typing import Dict, List, Optional, Tuple

import pandas as pd

from data_loader.build_utils import (FINDING_PRESENCE_SOURCES, apply_mask, build_prompt, ground_truth, load_rgb,
                                     pixel_digest, read_csv_defensively, resolve_image_path, scale_box)
from Inference.resume_utils import MissingInput, read_build_params


MASK_CONDITIONS = {"target_mask": ("box_x", "box_y", "box_w", "box_h"),
                   "irrelevant_mask": ("irrelevant_box_x", "irrelevant_box_y", "irrelevant_box_w", "irrelevant_box_h"),
                   "matched_mask": ("matched_box_x", "matched_box_y", "matched_box_w", "matched_box_h")}
SWAP_COLUMNS = {"swap": "swap_image_path", "opposite_swap": "opposite_swap_image_path"}
ASSET_CONDITIONS = {"noise": "noise_image", "natural_image": "natural_image"}
_ASSET_CACHE: Dict[str, object] = {}
_ASSET_DIGEST: Dict[str, str] = {}
KNOWN_CONDITIONS = {"original", "no_image"} | set(MASK_CONDITIONS) | set(SWAP_COLUMNS) | set(ASSET_CONDITIONS)


def _check_condition(condition: str) -> None:
    if condition not in KNOWN_CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}; known: {sorted(KNOWN_CONDITIONS)}")


def load_manifest(cfg: Dict, dataset: str) -> pd.DataFrame:
    key = "chexpert_manifest" if dataset == "chexpert" else "mimic_manifest"
    path = cfg["data"][key]
    if not os.path.exists(path):
        stage = "main_extend_manifest_chexpert" if dataset == "chexpert" else "main_extend_manifest_mimic"
        raise MissingInput(f"{path} does not exist; run {stage} first")
    return read_csv_defensively(path)


def resolution_of(cfg: Dict, dataset: str) -> int:
    return int(cfg["resolution_probe"]["resolution"]) if dataset == "mimic512" else int(cfg["data"]["resolution"])


def image_dataset(dataset: str) -> str:
    return "chexpert" if dataset == "chexpert" else "mimic"


def cases_for(cfg: Dict, dataset: str, condition: str, modality: str, manifest: pd.DataFrame,
              subset_ids: Optional[set] = None) -> List[Dict]:
    _check_condition(condition)
    df = manifest
    if subset_ids is not None:
        df = df[df["case_id"].astype(str).isin(set(map(str, subset_ids)))]
    if condition in MASK_CONDITIONS:
        df = df[df[MASK_CONDITIONS[condition][0]].notna()]
    elif condition in SWAP_COLUMNS:
        df = df[df[SWAP_COLUMNS[condition]].notna()]
    if modality == "vision_only" or condition == "opposite_swap":
        df = df[df["source"].isin(FINDING_PRESENCE_SOURCES)]
    if modality == "vision_only" and condition == "no_image":
        return []
    return df.sort_values("case_id").to_dict("records")


def _box(row: Dict, cols: Tuple[str, str, str, str], resolution: int, box_space: int) -> Tuple[int, int, int, int]:
    return scale_box(*(float(row[c]) for c in cols), resolution=resolution, box_space=box_space)


def _asset_digest(path: str) -> str:
    if path not in _ASSET_DIGEST:
        if not os.path.exists(path):
            return "absent"
        recorded = (read_build_params(path) or {}).get("pixel_digest")
        _ASSET_DIGEST[path] = str(recorded) if recorded else pixel_digest(path)
    return _ASSET_DIGEST[path]


def input_key(cfg: Dict, dataset: str, condition: str, row: Dict, modality: str) -> str:
    _check_condition(condition)
    if modality == "text_only" or condition == "no_image":
        return "none"
    res = resolution_of(cfg, dataset)
    if condition in ASSET_CONDITIONS:
        path = cfg["assets"][ASSET_CONDITIONS[condition]]
        return f"asset:{os.path.basename(path)}:{_asset_digest(path)}|res:{res}"
    if condition in SWAP_COLUMNS:
        return f"img:{row[SWAP_COLUMNS[condition]]}|res:{res}"
    key = f"img:{row['image_path']}|res:{res}"
    if condition in MASK_CONDITIONS:
        x, y, w, h = _box(row, MASK_CONDITIONS[condition], res, int(cfg["data"]["box_space"]))
        key += f"|box:{x},{y},{w},{h}"
    return key


def _asset(cfg: Dict, condition: str, resolution: int):
    path = cfg["assets"][ASSET_CONDITIONS[condition]]
    cache_key = f"{path}:{resolution}"
    if cache_key not in _ASSET_CACHE:
        if not os.path.exists(path):
            raise MissingInput(f"{path} does not exist; supply it (see main_build_assets)")
        from PIL import Image
        _ASSET_CACHE[cache_key] = load_rgb(path).resize((resolution, resolution), Image.BILINEAR)
    return _ASSET_CACHE[cache_key]


def make_input(cfg: Dict, dataset: str, condition: str, row: Dict, modality: str, variant: str):
    _check_condition(condition)
    prompt = build_prompt(row, variant, cfg)
    if modality == "text_only" or condition == "no_image":
        return None, prompt
    res = resolution_of(cfg, dataset)
    if condition in ASSET_CONDITIONS:
        return _asset(cfg, condition, res), prompt
    rel = row[SWAP_COLUMNS[condition]] if condition in SWAP_COLUMNS else row["image_path"]
    image = load_rgb(resolve_image_path(cfg, image_dataset(dataset), rel, res))
    if condition in MASK_CONDITIONS:
        x, y, w, h = _box(row, MASK_CONDITIONS[condition], res, int(cfg["data"]["box_space"]))
        image = apply_mask(image, x, y, w, h)
    return image, prompt


def case_ground_truth(row: Dict) -> int:
    return ground_truth(row)
