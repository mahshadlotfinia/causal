"""
Inference/inference_runner.py
Created on September 14, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd

from config.serde import read_config
from data_loader.probe_dataset import case_ground_truth, cases_for, input_key, load_manifest, make_input
from Inference.model_wrappers import build_model
from Inference.resume_utils import (MissingInput, append_jsonl, append_status, claim_unit, ensure_dir, heartbeat_claim,
                                    read_jsonl, release_claim, status_path)


@dataclass(frozen=True)
class Unit:
    dataset: str
    model: str
    variant: str
    condition: str
    suffix: str = ""

    @property
    def name(self) -> str:
        return f"{self.condition}{self.suffix}"

    def __str__(self) -> str:
        return f"{self.dataset}/{self.model}/{self.variant}/{self.name}"


def unit_dir(cfg: Dict, u: Unit) -> str:
    return os.path.join(cfg["runs"]["dir"], u.dataset, u.model, u.variant)


def unit_path(cfg: Dict, u: Unit) -> str:
    return os.path.join(unit_dir(cfg, u), f"{u.name}.jsonl")


def load_records(path: str) -> Tuple[Dict[str, Dict], int, int]:
    newest: Dict[str, Dict] = {}
    answers: Dict[str, Set[int]] = {}
    counts: Dict[str, int] = {}
    for rec in read_jsonl(path, "inference_runner"):
        cid = str(rec["case_id"])
        newest[cid] = rec
        answers.setdefault(cid, set()).add(int(rec["parsed_answer"]))
        counts[cid] = counts.get(cid, 0) + 1
    multi = sum(1 for n in counts.values() if n > 1)
    disagree = sum(1 for cid, n in counts.items() if n > 1 and len(answers[cid]) > 1)
    return newest, multi, disagree


def max_tokens_for(cfg: Dict, spec: Dict, rerun: bool = False, variant: str = "default", condition: str = "original") -> int:
    g = cfg["generation"]
    if spec["reasoning"]:
        return int(g["rerun_reasoning_max_new_tokens"] if rerun else g["reasoning_max_new_tokens"])
    return max(int(g["max_new_tokens"]), int(g["variant_max_new_tokens"].get(variant, 0)),
               int(g["condition_max_new_tokens"].get(condition, 0)))


def _subset_ids(cfg: Dict, name: Optional[str]) -> Optional[Set[str]]:
    if name is None:
        return None
    path = os.path.join(cfg["data"]["subsets_dir"], f"{name}.csv")
    if not os.path.exists(path):
        raise MissingInput(f"{path} does not exist; run main_build_assets first")
    return set(pd.read_csv(path, low_memory=False)["case_id"].astype(str))


def unit_plan(cfg: Dict, model_key: str) -> List[Tuple[Unit, Optional[str]]]:
    spec = cfg["models"][model_key]
    mod = spec["modality"]
    c = cfg["conditions"]
    plan: List[Tuple[Unit, Optional[str]]] = []
    if mod == "text_only":
        mimic_conditions, chex_conditions = ["original"], ["original"]
    elif mod == "vision_only":
        mimic_conditions = [x for x in c["mimic"] if x != "no_image"]
        chex_conditions = list(c["chexpert"])
    else:
        mimic_conditions, chex_conditions = list(c["mimic"]), list(c["chexpert"])
    for cond in mimic_conditions:
        plan.append((Unit("mimic", model_key, "default", cond), None))
    for cond in chex_conditions:
        plan.append((Unit("chexpert", model_key, "default", cond), None))
    if mod != "vision_only":
        for variant in cfg["prompt_sensitivity"]["variants"]:
            plan.append((Unit("mimic", model_key, variant, "original"), "prompt_original"))
            if mod == "multimodal":
                for cond in cfg["prompt_sensitivity"]["mscxr_conditions"]:
                    if cond != "original":
                        plan.append((Unit("mimic", model_key, variant, cond), "prompt_mscxr"))
    return plan


def _cases_of(cfg: Dict, u: Unit, modality: str, manifest: pd.DataFrame, subset: Optional[str]) -> List[Dict]:
    if subset == "prompt_original":
        ids = set(manifest.loc[manifest["source"] == "ms_cxr", "case_id"].astype(str)) | _subset_ids(cfg, "prompt_mimic_subset")
    elif subset == "prompt_mscxr":
        ids = set(manifest.loc[manifest["source"] == "ms_cxr", "case_id"].astype(str))
    else:
        ids = None
    return cases_for(cfg, u.dataset, u.condition, modality, manifest, ids)


def run_unit(cfg: Dict, u: Unit, model, manifest: pd.DataFrame, cases: List[Dict], max_tokens: int,
             status_file: str, only_cases: Optional[Set[str]] = None, force: bool = False,
             extra: Optional[Dict] = None) -> Dict[str, int]:
    from tqdm import tqdm
    path = unit_path(cfg, u)
    ensure_dir(unit_dir(cfg, u))
    modality = model.modality
    version = cfg["prompts"]["version"]
    suffix = f"|heads:{model.fingerprint}" if modality == "vision_only" else ""
    def _todo():
        newest, multi, disagree = load_records(path)
        if multi:
            print(f"[run] {u}: {multi} case(s) carry more than one record, {disagree} disagree on the answer (the newest record is used).", flush=True)
        out = []
        for row in cases:
            cid = str(row["case_id"])
            if only_cases is not None and cid not in only_cases:
                continue
            key = input_key(cfg, u.dataset, u.condition, row, modality) + suffix
            rec = newest.get(cid)
            budget_ok = rec is not None and (rec.get("max_tokens") is None or rec.get("answer_class") != "truncated"
                                             or int(rec["max_tokens"]) >= int(max_tokens))
            if not force and budget_ok and rec.get("input_key") == key and rec.get("prompt_version") == version:
                continue
            out.append((row, key))
        return out

    todo = _todo()
    counts = {"cases": len(cases), "done_before": len(cases) - len(todo), "run": 0, "failed": 0}
    print(f"[run] {u}: {len(cases)} case(s), {counts['done_before']} already done, {len(todo)} to run.", flush=True)
    if not todo:
        return counts
    if not claim_unit(unit_dir(cfg, u), u.name, float(cfg["runs"]["claim_stale_after_s"])):
        print(f"[run] {u}: claimed by a live job, leaving it alone.", flush=True)
        counts["claimed_elsewhere"] = 1
        return counts
    todo = _todo()
    if not todo:
        release_claim(unit_dir(cfg, u), u.name)
        counts["done_before"] = len(cases)
        return counts
    serving = cfg["serving"]
    if modality == "vision_only":
        wave = int(cfg["raddino"]["inference_batch_size"])
    else:
        wave = int(serving["concurrency"]) * 4
    consecutive_failures = 0
    limit = int(serving["max_consecutive_failures"])
    t_unit = time.time()
    try:
        bar = tqdm(total=len(todo), desc=f"[run] {u}", unit="case")
        with ThreadPoolExecutor(max_workers=8) as loader:
            for start in range(0, len(todo), wave):
                chunk = todo[start:start + wave]
                inputs = list(loader.map(lambda rk: make_input(cfg, u.dataset, u.condition, rk[0], modality, u.variant), chunk))
                items = [{"image": img, "prompt": prompt, "finding": str(row.get("finding") or "")}
                         for (row, _), (img, prompt) in zip(chunk, inputs)]
                results = model.answer_batch(items, max_tokens)
                for (row, key), res in zip(chunk, results):
                    if res is None:
                        counts["failed"] += 1
                        consecutive_failures += 1
                        if consecutive_failures >= limit:
                            raise RuntimeError(f"[run] {u}: {limit} consecutive failed calls; the server is down or the model is not loaded. Stopping.")
                        continue
                    consecutive_failures = 0
                    rec = {"case_id": str(row["case_id"]), "input_key": key, "prompt_version": version,
                           "parser_version": str(cfg["parser"]["version"]),
                           "ground_truth": int(case_ground_truth(row)), "max_tokens": int(max_tokens),
                           "source_run": "r1", "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
                    rec.update(res)
                    if extra:
                        rec.update(extra)
                    append_jsonl(path, rec)
                    counts["run"] += 1
                bar.update(len(chunk))
                heartbeat_claim(unit_dir(cfg, u), u.name)
                append_status(status_file, f"{u}: {counts['run']} written, {counts['failed']} failed, {bar.n}/{len(todo)}")
        bar.close()
    finally:
        release_claim(unit_dir(cfg, u), u.name)
    rate = counts["run"] / max(1e-9, time.time() - t_unit)
    print(f"[run] {u}: wrote {counts['run']} record(s), {counts['failed']} failed call(s), {rate * 60:.1f} cases/min.", flush=True)
    return counts


def _manifests(cfg: Dict, datasets: Iterable[str]) -> Dict[str, pd.DataFrame]:
    return {d: load_manifest(cfg, d) for d in datasets}


def main_run(cfg_path: str, model_key: str) -> None:
    cfg = read_config(cfg_path)["CausalAudit"]
    if model_key not in cfg["models"]:
        raise KeyError(f"unknown model {model_key}; known: {list(cfg['models'])}")
    plan = unit_plan(cfg, model_key)
    manifests = _manifests(cfg, sorted({u.dataset for u, _ in plan}))
    status = status_path(cfg, f"run_{model_key}")
    models = {}
    total = {"run": 0, "failed": 0, "skipped": 0}
    for u, subset in plan:
        if u.dataset not in models:
            models[u.dataset] = build_model(model_key, cfg, u.dataset)
        model = models[u.dataset]
        try:
            cases = _cases_of(cfg, u, model.modality, manifests[u.dataset], subset)
            counts = run_unit(cfg, u, model, manifests[u.dataset], cases, max_tokens_for(cfg, model.spec, variant=u.variant, condition=u.condition), status)
        except MissingInput as e:
            print(f"[run] SKIP {u}: {e}", flush=True)
            total["skipped"] += 1
            continue
        total["run"] += counts["run"]
        total["failed"] += counts["failed"]
    print(f"[run] {model_key}: done. {total['run']} record(s) written, {total['failed']} failed call(s), {total['skipped']} unit(s) skipped.", flush=True)


def main_run_raddino(cfg_path: str) -> None:
    main_run(cfg_path, "RAD-DINO")


def _rerun_units(cfg: Dict, model_key: str) -> List[Unit]:
    units = []
    for dataset in ("mimic", "chexpert", "mimic512"):
        base = os.path.join(cfg["runs"]["dir"], dataset, model_key)
        if not os.path.isdir(base):
            continue
        for variant in sorted(os.listdir(base)):
            vdir = os.path.join(base, variant)
            for name in sorted(os.listdir(vdir)):
                if name.endswith(".jsonl") and "__" not in name:
                    units.append(Unit(dataset, model_key, variant, name[:-len(".jsonl")]))
    return units


def main_rerun_failed(cfg_path: str, model_key: str) -> None:
    cfg = read_config(cfg_path)["CausalAudit"]
    spec = cfg["models"][model_key]
    if spec["backend"] == "raddino":
        print("[rerun] RAD-DINO has no failed calls to rerun.", flush=True)
        return
    manifests: Dict[str, pd.DataFrame] = {}
    status = status_path(cfg, f"rerun_{model_key}")
    model = build_model(model_key, cfg, "mimic")
    attempts = int(cfg["runs"]["empty_rerun_attempts"])
    n_units = 0
    for u in _rerun_units(cfg, model_key):
        if u.dataset not in manifests:
            manifests[u.dataset] = load_manifest(cfg, u.dataset)
        budget = max_tokens_for(cfg, spec, rerun=True, variant=u.variant, condition=u.condition)
        newest, _, _ = load_records(unit_path(cfg, u))
        empties = {cid for cid, r in newest.items() if r.get("answer_class") == "empty"
                   and int(r.get("rerun_count", 0)) < attempts and (not spec["reasoning"] or int(r.get("max_tokens", 0)) < budget)}
        truncated = {cid for cid, r in newest.items() if r.get("answer_class") == "truncated"} if spec["reasoning"] else set()
        if not empties and not truncated:
            continue
        n_units += 1
        cases = cases_for(cfg, u.dataset, u.condition, model.modality, manifests[u.dataset], None)
        if empties:
            print(f"[rerun] {u}: {len(empties)} empty output(s) rerun at {budget} tokens.", flush=True)
            run_unit(cfg, u, model, manifests[u.dataset], cases, budget, status, only_cases=empties, force=True,
                     extra={"rerun_count": attempts})
        if truncated:
            sens = Unit(u.dataset, u.model, u.variant, u.condition, suffix=f"__budget{budget}")
            print(f"[rerun] {sens}: {len(truncated)} truncated trace(s) rerun at {budget} tokens for the sensitivity analysis.", flush=True)
            run_unit(cfg, sens, model, manifests[u.dataset], cases, budget, status, only_cases=truncated)
    print(f"[rerun] {model_key}: done. {n_units} unit(s) had something to rerun.", flush=True)
