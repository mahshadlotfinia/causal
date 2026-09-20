"""
Inference/reparse_records.py
Created on September 15, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import os
from typing import Dict, List

import pandas as pd

from config.serde import read_config
from Inference.inference_runner import Unit, unit_dir
from Inference.parser import classify, parse_fixed
from Inference.resume_utils import (MissingInput, append_status, claim_unit, read_jsonl, release_claim, status_path,
                                    write_csv_atomic, write_jsonl_atomic)


def _units(cfg: Dict) -> List[Unit]:
    root = cfg["runs"]["dir"]
    if not os.path.isdir(root):
        raise MissingInput(f"{root} does not exist; run main_import_delivered first")
    out = []
    for dataset in sorted(os.listdir(root)):
        for model in sorted(os.listdir(os.path.join(root, dataset))):
            for variant in sorted(os.listdir(os.path.join(root, dataset, model))):
                vdir = os.path.join(root, dataset, model, variant)
                for name in sorted(os.listdir(vdir)):
                    if name.endswith(".jsonl"):
                        out.append(Unit(dataset, model, variant, name[:-len(".jsonl")]))
    return out


def _reparsed(records: List[Dict], reasoning: bool, pcfg: Dict, version: str) -> List[Dict]:
    out = []
    for rec in records:
        new = dict(rec)
        parsed = parse_fixed(rec.get("raw_answer"), reasoning, pcfg)
        new["parsed_answer"] = int(parsed)
        new["answer_class"] = classify(rec.get("raw_answer"), parsed, reasoning, rec.get("finish_reason"), pcfg)
        new["parser_version"] = version
        out.append(new)
    return out


def main_reparse_records(cfg_path: str) -> None:
    cfg = read_config(cfg_path)["CausalAudit"]
    pcfg = cfg["parser"]
    version = str(pcfg["version"])
    status = status_path(cfg, "reparse_records")
    from tqdm import tqdm
    units = _units(cfg)
    n_units, n_changed, moves = 0, 0, {}
    for u in tqdm(units, desc="[reparse] units", unit="unit"):
        if u.model not in cfg["models"]:
            print(f"[reparse] SKIP {u}: {u.model} is not in the panel.", flush=True)
            continue
        path = os.path.join(unit_dir(cfg, u), f"{u.name}.jsonl")
        records = read_jsonl(path, "reparse_records")
        if not records:
            continue
        reasoning = bool(cfg["models"][u.model]["reasoning"])
        fresh = _reparsed(records, reasoning, pcfg, version)
        changed = [(a, b) for a, b in zip(records, fresh)
                   if a.get("parsed_answer") != b["parsed_answer"] or a.get("answer_class") != b["answer_class"]
                   or a.get("parser_version") != version]
        if not changed:
            continue
        if not claim_unit(unit_dir(cfg, u), u.name, float(cfg["runs"]["claim_stale_after_s"])):
            print(f"[reparse] {u}: claimed by a live job, leaving it alone.", flush=True)
            continue
        try:
            write_jsonl_atomic(path, fresh)
        finally:
            release_claim(unit_dir(cfg, u), u.name)
        parses = [(a, b) for a, b in changed if a.get("answer_class") != b["answer_class"]]
        for a, b in parses:
            moves[f'{a.get("answer_class")} -> {b["answer_class"]}'] = moves.get(f'{a.get("answer_class")} -> {b["answer_class"]}', 0) + 1
        n_units += 1
        n_changed += len(parses)
        if parses:
            print(f"[reparse] {u}: {len(parses)} of {len(records)} record(s) re-parsed.", flush=True)
        append_status(status, f"{u}: {len(parses)} re-parsed of {len(records)}")
    print(f"[reparse] done. {len(units)} unit(s) read, {n_units} rewritten, {n_changed} record(s) changed class, moves {moves}.", flush=True)
    if n_changed:
        write_csv_atomic(pd.DataFrame(sorted(moves.items()), columns=["move", "records"]), os.path.join(cfg["outputs"]["metrics_dir"], "reparse_moves.csv"))
