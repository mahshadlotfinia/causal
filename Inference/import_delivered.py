"""
Inference/import_delivered.py
Created on September 14, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import os
import time
from typing import Dict, List, Optional, Tuple

import pandas as pd

from config.serde import read_config
from data_loader.build_utils import ground_truth, read_csv_defensively
from data_loader.probe_dataset import input_key, load_manifest
from Inference.inference_runner import Unit, load_records, max_tokens_for, unit_dir, unit_path
from Inference.parser import classify, parse_fixed
from Inference.resume_utils import (MissingInput, append_jsonl, append_status, claim_unit, ensure_dir, release_claim,
                                    status_path)


FAILURE_PREFIXES = ("Connection error", "Error code:", "Request timed out", "APIConnectionError", "APITimeoutError")


def _is_failed_call(raw: str) -> bool:
    return raw.strip().startswith(FAILURE_PREFIXES)


def _delivered_files(cfg: Dict) -> List[Tuple[str, Unit]]:
    r = cfg["runs"]
    out: List[Tuple[str, Unit]] = []
    for model_key in cfg["model_order"]:
        if cfg["models"][model_key]["backend"] == "raddino":
            continue
        for cond in ("original", "swap", "target_mask", "irrelevant_mask"):
            out.append((os.path.join(r["delivered_mimic_dir"], model_key, f"{cond}.csv"), Unit("mimic", model_key, "default", cond)))
        for cond in ("original", "swap"):
            out.append((os.path.join(r["delivered_chexpert_dir"], model_key, f"{cond}.csv"), Unit("chexpert", model_key, "default", cond)))
        for cond in ("original", "target_mask"):
            out.append((os.path.join(r["delivered_resolution_dir"], model_key, f"{cond}.csv"), Unit("mimic512", model_key, "default", cond)))
        for variant in cfg["prompt_sensitivity"]["variants"]:
            out.append((os.path.join(r["delivered_prompt_dir"], variant, model_key, "original.csv"), Unit("mimic", model_key, variant, "original")))
    return out


def _frozen_rows(cfg: Dict, dataset: str) -> Dict[str, Dict]:
    key = "frozen_chexpert_manifest" if dataset == "chexpert" else "frozen_mimic_manifest"
    df = read_csv_defensively(cfg["data"][key])
    return {str(r["case_id"]): r for r in df.to_dict("records")}


def _v2_rows(cfg: Dict, dataset: str) -> Dict[str, Dict]:
    df = load_manifest(cfg, dataset)
    return {str(r["case_id"]): r for r in df.to_dict("records")}


def _import_file(cfg: Dict, path: str, u: Unit, frozen: Dict[str, Dict], v2: Dict[str, Dict],
                 status: str) -> Optional[Dict[str, int]]:
    spec = cfg["models"][u.model]
    modality = spec["modality"]
    reasoning = bool(spec["reasoning"])
    df = read_csv_defensively(path)
    keys = {}
    for cid in df["case_id"].astype(str):
        if cid not in frozen:
            raise ValueError(f"{path}: case {cid} is not in the frozen manifest")
        keys[cid] = input_key(cfg, u.dataset, u.condition, frozen[cid], modality)
    newest, _, _ = load_records(unit_path(cfg, u))
    importable = [cid for cid, raw in zip(df["case_id"].astype(str), df["raw_answer"].fillna("").astype(str))
                  if not _is_failed_call(raw)]
    already = sum(1 for cid in importable if cid in newest and newest[cid].get("source_run") == "delivered"
                  and newest[cid].get("input_key") == keys[cid])
    if already == len(importable):
        return None
    ensure_dir(unit_dir(cfg, u))
    if not claim_unit(unit_dir(cfg, u), u.name, float(cfg["runs"]["claim_stale_after_s"])):
        print(f"[import] {u}: claimed by a live job, leaving it alone.", flush=True)
        return None
    counts = {"rows": len(df), "imported": 0, "failed_calls": 0, "parse_disagreements": 0}
    classes: Dict[str, int] = {}
    max_tokens = max_tokens_for(cfg, spec)
    version = cfg["prompts"]["version"]
    for row in df.to_dict("records"):
        cid = str(row["case_id"])
        raw = "" if pd.isna(row["raw_answer"]) else str(row["raw_answer"])
        if _is_failed_call(raw):
            counts["failed_calls"] += 1
            continue
        if cid in newest and newest[cid].get("source_run") == "delivered" and newest[cid].get("input_key") == keys[cid]:
            continue
        parsed = parse_fixed(raw, reasoning, cfg["parser"])
        if parsed != int(row["parsed_answer"]):
            counts["parse_disagreements"] += 1
        cls = classify(raw, parsed, reasoning, None, cfg["parser"])
        classes[cls] = classes.get(cls, 0) + 1
        conf = None
        if not reasoning and not pd.isna(row["confidence"]):
            conf = float(row["confidence"])
        rec = {"case_id": cid, "input_key": keys[cid], "prompt_version": version,
               "parser_version": str(cfg["parser"]["version"]), "ground_truth": int(ground_truth(v2[cid])),
               "max_tokens": max_tokens, "source_run": "delivered", "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "raw_answer": raw, "parsed_answer": int(parsed), "answer_class": cls, "confidence": conf, "finish_reason": None,
               "prompt_tokens": None, "completion_tokens": None, "elapsed_s": None}
        append_jsonl(unit_path(cfg, u), rec)
        counts["imported"] += 1
    counts["classes"] = classes
    release_claim(unit_dir(cfg, u), u.name)
    append_status(status, f"{u}: imported {counts['imported']} of {counts['rows']} rows, {counts['failed_calls']} failed call(s) left for the rerun")
    return counts


def main_import_delivered(cfg_path: str) -> None:
    cfg = read_config(cfg_path)["CausalAudit"]
    status = status_path(cfg, "import_delivered")
    frozen = {"mimic": _frozen_rows(cfg, "mimic"), "chexpert": _frozen_rows(cfg, "chexpert")}
    frozen["mimic512"] = frozen["mimic"]
    v2 = {"mimic": _v2_rows(cfg, "mimic"), "chexpert": _v2_rows(cfg, "chexpert")}
    v2["mimic512"] = v2["mimic"]
    n_files, n_new, n_skipped, n_failed = 0, 0, 0, 0
    from tqdm import tqdm
    files = _delivered_files(cfg)
    for path, u in tqdm(files, desc="[import] delivered files", unit="file"):
        if not os.path.exists(path):
            print(f"[import] MISSING {path}: no delivered file for {u}; nothing imported.", flush=True)
            continue
        n_files += 1
        counts = _import_file(cfg, path, u, frozen[u.dataset], v2[u.dataset], status)
        if counts is None:
            n_skipped += 1
            continue
        n_new += counts["imported"]
        n_failed += counts["failed_calls"]
        print(f"[import] {u}: {counts['imported']} imported of {counts['rows']}, {counts['failed_calls']} failed call(s) skipped, "
              f"{counts['parse_disagreements']} parse disagreement(s), classes {counts['classes']}", flush=True)
    if n_files == 0:
        raise MissingInput(f"no delivered file found under {cfg['runs']['delivered_mimic_dir']}; check the machine block of config.yaml")
    print(f"[import] done. {n_files} file(s) seen, {n_skipped} already imported, {n_new} record(s) written, "
          f"{n_failed} failed call(s) left for the run stage.", flush=True)

