"""
data_loader/extend_manifest.py
Created on September 14, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import os
from typing import Dict, Optional

import numpy as np
import pandas as pd

from config.serde import read_config
from data_loader.build_utils import (FRONTAL_VIEWS, abnormal_pool, absent_pool, load_chexpert_master,
                                     load_mimic_master, matched_box, normal_pool, present_pool,
                                     read_csv_defensively, rexerr_class)
from Inference.resume_utils import (append_status, check_build_params, claim_unit, fingerprint_file, release_claim,
                                    status_path, write_build_params, write_csv_atomic)


NEW_COLUMNS = ["label_source", "label_value_raw", "swap_label_matched", "swap_rebuilt",
               "opposite_swap_image_path", "opposite_swap_label_value", "matched_box_x", "matched_box_y",
               "matched_box_w", "matched_box_h", "matched_box_placement", "image_necessary", "rexerr_class"]


def _build_params(cfg: Dict, frozen_csv: str, master_csv: str) -> Dict:
    d = cfg["data"]
    return {"seed": int(cfg["seed"]), "present_code": d["present_code"], "absent_code": d["absent_code"],
            "fallback_absent_code": d["fallback_absent_code"], "min_absent_patients": d["min_absent_patients"],
            "box_space": d["box_space"], "frozen_manifest": fingerprint_file(frozen_csv),
            "master_list": fingerprint_file(master_csv), "columns": NEW_COLUMNS}


def _pick(pool: pd.DataFrame, subject, rng: np.random.Generator) -> Optional[pd.Series]:
    cand = pool[pool["subject_id"] != subject]
    if len(cand) == 0:
        return None
    return cand.iloc[int(rng.integers(len(cand)))]


def _label_state(row: pd.Series) -> str:
    return "normal" if row["finding"] == "no_finding" else ("present" if int(row["label"]) == 1 else "absent")


def _same_label_pool(pool: pd.DataFrame, row: pd.Series, cfg: Dict) -> pd.DataFrame:
    state = _label_state(row)
    if state == "normal":
        return normal_pool(pool, cfg)
    if state == "present":
        return present_pool(pool, row["finding"], cfg)
    return absent_pool(pool, row["finding"], cfg)


def _opposite_pool(pool: pd.DataFrame, row: pd.Series, cfg: Dict) -> pd.DataFrame:
    state = _label_state(row)
    if state == "normal":
        return abnormal_pool(pool, cfg)
    if state == "present":
        return absent_pool(pool, row["finding"], cfg)
    return present_pool(pool, row["finding"], cfg)


def _value_of(master_by_path: pd.DataFrame, path: str, finding: str):
    key = str(path)
    if key not in master_by_path.index or finding not in master_by_path.columns:
        return np.nan
    return master_by_path.at[key, finding]


def _extend(cfg: Dict, frozen: pd.DataFrame, master: pd.DataFrame, pool: pd.DataFrame,
            dataset: str) -> pd.DataFrame:
    d = cfg["data"]
    rng = np.random.default_rng(int(cfg["seed"]))
    master_by_path = master.drop_duplicates("jpg_rel_path").set_index("jpg_rel_path")
    findings = d["all_findings"]
    out = frozen.copy()
    for c in NEW_COLUMNS:
        out[c] = pd.Series([None] * len(out), index=out.index, dtype="object")
    out = out.sort_values("case_id").reset_index(drop=True)
    presence = out["source"].isin(["ms_cxr", "mimic_cxr", "chexpert"])
    mscxr_dicoms = set(out.loc[out["source"] == "ms_cxr", "dicom_id"].dropna().astype(str))
    opp_pool = pool[~pool["dicom_id"].isin(mscxr_dicoms)]
    for i in out.index[presence]:
        row = out.loc[i]
        finding = str(row["finding"])
        raw = _value_of(master_by_path, row["image_path"], finding) if finding in findings else np.nan
        out.at[i, "label_value_raw"] = raw
        if finding == "no_finding":
            out.at[i, "label"] = 0
            out.at[i, "label_source"] = "normal_study"
        elif int(row["label"]) == 1:
            out.at[i, "label_source"] = "value_1"
        elif not pd.isna(raw) and int(raw) == int(d["fallback_absent_code"]):
            out.at[i, "label_source"] = "value_3_fallback"
        else:
            out.at[i, "label_source"] = "value_0"
        swap_value = _value_of(master_by_path, row["swap_image_path"], finding)
        state = _label_state(row)
        if state == "normal":
            matched = (not pd.isna(swap_value)) and int(swap_value) == int(d["present_code"])
        elif state == "present":
            matched = (not pd.isna(swap_value)) and int(swap_value) == int(d["present_code"])
        else:
            matched = (not pd.isna(swap_value)) and int(swap_value) in (int(d["absent_code"]), int(d["fallback_absent_code"]))
        out.at[i, "swap_label_matched"] = bool(matched)
        out.at[i, "swap_rebuilt"] = False
        if out.at[i, "label_source"] == "value_3_fallback" and not matched:
            pick = _pick(_same_label_pool(pool, row, cfg), row["subject_id"], rng)
            if pick is None:
                raise ValueError(f"no same-label counterpart for {row['case_id']}")
            out.at[i, "swap_image_path"] = pick["jpg_rel_path"]
            out.at[i, "swap_label_matched"] = True
            out.at[i, "swap_rebuilt"] = True
        pick = _pick(_opposite_pool(opp_pool, row, cfg), row["subject_id"], rng)
        if pick is None:
            raise ValueError(f"no opposite-label counterpart for {row['case_id']}")
        out.at[i, "opposite_swap_image_path"] = pick["jpg_rel_path"]
        out.at[i, "opposite_swap_label_value"] = "abnormal" if state == "normal" else str(int(pick[finding]))
    size = int(d["box_space"])
    for i in out.index[out["box_x"].notna()]:
        r = out.loc[i]
        mx, my, mw, mh, place = matched_box(int(r["box_x"]), int(r["box_y"]), int(r["box_w"]), int(r["box_h"]), size)
        out.at[i, "matched_box_x"], out.at[i, "matched_box_y"] = mx, my
        out.at[i, "matched_box_w"], out.at[i, "matched_box_h"] = mw, mh
        out.at[i, "matched_box_placement"] = place
    if dataset == "mimic":
        out["rexerr_class"] = [rexerr_class(et, ep, cfg) if s == "rexerr" else None
                               for et, ep, s in zip(out["error_type"], out["error_present"], out["source"])]
        out["image_necessary"] = (out["source"] == "mimic_cxr") | out["rexerr_class"].isin(["image_dependent", "control"])
        out.loc[out["source"] == "rexerr", "label_source"] = "rexerr"
    else:
        out["image_necessary"] = True
    for c in ("label_value_raw", "matched_box_x", "matched_box_y", "matched_box_w", "matched_box_h"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    for c in ("swap_label_matched", "swap_rebuilt"):
        out[c] = out[c].map(lambda v: None if v is None else bool(v))
    return out


def _report(out: pd.DataFrame, dataset: str) -> None:
    print(f"[extend_manifest] {dataset}: {len(out)} rows; label_source "
          f"{out['label_source'].value_counts(dropna=False).to_dict()}", flush=True)
    print(f"[extend_manifest] {dataset}: swaps rebuilt {int(out['swap_rebuilt'].fillna(False).astype(bool).sum())}, "
          f"same-label swaps not label-matched {int((out['swap_label_matched'] == False).sum())}, "
          f"opposite swaps {int(out['opposite_swap_image_path'].notna().sum())}", flush=True)
    if "matched_box_placement" in out.columns and out["matched_box_placement"].notna().any():
        print(f"[extend_manifest] matched mask placement {out['matched_box_placement'].value_counts().to_dict()}", flush=True)


def _write(cfg: Dict, out: pd.DataFrame, path: str, params: Dict, dataset: str) -> None:
    write_csv_atomic(out, path)
    write_build_params(path, params)
    _report(out, dataset)
    append_status(status_path(cfg, "extend_manifest"), f"{dataset}: wrote {len(out)} rows to {path}")


def main_extend_manifest_mimic(cfg_path: str) -> None:
    cfg = read_config(cfg_path)["CausalAudit"]
    d = cfg["data"]
    path = d["mimic_manifest"]
    params = _build_params(cfg, d["frozen_mimic_manifest"], d["mimic_master_csv"])
    if os.path.exists(path) and check_build_params(path, params, "extend_manifest"):
        print(f"[extend_manifest] mimic: {path} is current, skipping.", flush=True)
        return
    if not claim_unit(os.path.dirname(path), "extend_mimic"):
        print("[extend_manifest] mimic: claimed by a live job, leaving it alone.", flush=True)
        return
    try:
        frozen = read_csv_defensively(d["frozen_mimic_manifest"])
        master = load_mimic_master(cfg, usecols=["jpg_rel_path", "subject_id", "split", "view"] + d["all_findings"])
        pool = master[master["view"].isin(FRONTAL_VIEWS)].reset_index(drop=True)
        out = _extend(cfg, frozen, master, pool, "mimic")
        _write(cfg, out, path, params, "mimic")
    finally:
        release_claim(os.path.dirname(path), "extend_mimic")


def main_extend_manifest_chexpert(cfg_path: str) -> None:
    cfg = read_config(cfg_path)["CausalAudit"]
    d = cfg["data"]
    path = d["chexpert_manifest"]
    params = _build_params(cfg, d["frozen_chexpert_manifest"], d["chexpert_master_csv"])
    if os.path.exists(path) and check_build_params(path, params, "extend_manifest"):
        print(f"[extend_manifest] chexpert: {path} is current, skipping.", flush=True)
        return
    if not claim_unit(os.path.dirname(path), "extend_chexpert"):
        print("[extend_manifest] chexpert: claimed by a live job, leaving it alone.", flush=True)
        return
    try:
        frozen = read_csv_defensively(d["frozen_chexpert_manifest"])
        master = load_chexpert_master(cfg, usecols=["jpg_rel_path", "subject_id", "split", "view"] + d["all_findings"])
        pool = master[master["view"] == "Frontal"].reset_index(drop=True)
        out = _extend(cfg, frozen, master, pool, "chexpert")
        _write(cfg, out, path, params, "chexpert")
    finally:
        release_claim(os.path.dirname(path), "extend_chexpert")
