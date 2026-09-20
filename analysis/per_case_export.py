"""
analysis/per_case_export.py
Created on September 14, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import os
from typing import Dict, List

import numpy as np
import pandas as pd

from analysis.compute_metrics import record_store_fingerprint
from analysis.metrics_core import META_COLUMNS, wide_frame
from config.serde import read_config
from data_loader.probe_dataset import load_manifest
from Inference.resume_utils import MissingInput, append_status, ensure_dir, status_path, write_csv_atomic


def _wide_export(cfg: Dict, dataset: str) -> pd.DataFrame:
    manifest = load_manifest(cfg, dataset)
    base = None
    for model in cfg["model_order"]:
        try:
            df = wide_frame(cfg, dataset, model, manifest=manifest, verbose=False)
        except MissingInput:
            continue
        keep = [c for c in df.columns if c.startswith(("pa_", "cls_", "conf_"))]
        block = df[["case_id"] + keep].rename(columns={c: f"{model}__{c}" for c in keep})
        if base is None:
            base = df[[c for c in META_COLUMNS if c in df.columns] + ["gt"]].copy()
        base = base.merge(block, on="case_id", how="left")
    if base is None:
        raise MissingInput(f"no records for {dataset}; run main_import_delivered and the model runs first")
    answer_cols = [c for c in base.columns if "__pa_" in c]
    return base[base[answer_cols].notna().any(axis=1)].reset_index(drop=True)


def _long_export(cfg: Dict, dataset: str) -> pd.DataFrame:
    manifest = load_manifest(cfg, dataset)
    rows: List[pd.DataFrame] = []
    variants = ["default"] + list(cfg["prompt_sensitivity"]["variants"])
    for model in cfg["model_order"]:
        for variant in variants:
            try:
                df = wide_frame(cfg, dataset, model, variant=variant, manifest=manifest, verbose=False)
            except MissingInput:
                continue
            for cond in [c[len("pa_"):] for c in df.columns if c.startswith("pa_")]:
                sub = df[df[f"cls_{cond}"].notna()]
                if len(sub) == 0:
                    continue
                rows.append(pd.DataFrame({"case_id": sub["case_id"].values, "source": sub["source"].values, "model": model,
                                          "variant": variant, "condition": cond, "ground_truth": sub["gt"].values,
                                          "parsed_answer": sub[f"pa_{cond}"].astype(int).values, "answer_class": sub[f"cls_{cond}"].values,
                                          "confidence": sub[f"conf_{cond}"].values}))
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def main_per_case_export(cfg_path: str) -> None:
    cfg = read_config(cfg_path)["CausalAudit"]
    out_dir = ensure_dir(cfg["outputs"]["per_case_dir"])
    status = status_path(cfg, "per_case_export")
    fp = record_store_fingerprint(cfg)
    for dataset in ("mimic", "chexpert", "mimic512"):
        try:
            wide = _wide_export(cfg, dataset)
        except MissingInput as e:
            print(f"[per_case] SKIP {dataset}: {e}", flush=True)
            continue
        wide.insert(0, "record_store", fp)
        write_csv_atomic(wide, os.path.join(out_dir, f"{dataset}_cases_wide.csv"))
        long = _long_export(cfg, dataset)
        if len(long):
            long.insert(0, "record_store", fp)
            write_csv_atomic(long, os.path.join(out_dir, f"{dataset}_answers_long.csv"))
        n_answers = int(np.sum([c.startswith("pa_") or "__pa_" in c for c in wide.columns]))
        print(f"[per_case] {dataset}: wide {wide.shape[0]} cases x {n_answers} answer columns; long {len(long)} rows", flush=True)
        append_status(status, f"{dataset}: wide {wide.shape}, long {len(long)}")
    append_status(status, "done")
