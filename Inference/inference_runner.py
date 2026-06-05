"""
Inference/inference_runner.py
Created May 22, 2026

InferenceRunner: orchestrates inference for one model over all applicable
conditions and saves results to per-condition CSVs.

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia/
"""

import os

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.serde import read_config
from data_loader.probe_set_data_loader import (
    ProbeSetDataset,
    probe_collate_fn,
    CONDITIONS as MIMIC_CONDITIONS,
    PROMPT_VARIANTS,
)
from data_loader.chexpert_data_loader import (
    CheXpertDataset,
    chexpert_collate_fn,
    CHEXPERT_CONDITIONS,
)
from Inference.model_wrappers import load_model_wrapper

import warnings
warnings.filterwarnings("ignore")


class InferenceRunner:
    """
    Args:
        cfg_path       : path to config.yaml
        model_name     : key from MODEL_REGISTRY
        dataset_type   : 'mimic' | 'chexpert'
        prompt_variant : 'default' | 'brief' | 'clinical'
                         When not 'default', results go to a separate subdir.
        manifest_csv   : optional override (used by resolution check)
        results_dir    : optional override (used by resolution check)
        image_root     : optional override (used by resolution check)
        resolution     : optional override (used by resolution check)
    """

    DATASET_MAP = {
        "mimic":    (ProbeSetDataset,  probe_collate_fn,    MIMIC_CONDITIONS,    "results_dir"),
        "chexpert": (CheXpertDataset,  chexpert_collate_fn, CHEXPERT_CONDITIONS, "chexpert_results_dir"),
    }

    def __init__(
        self,
        cfg_path: str,
        model_name: str,
        dataset_type: str = "mimic",
        prompt_variant: str = "default",
        manifest_csv: str = None,
        results_dir: str = None,
        image_root: str = None,
        resolution: int = None,
    ):
        assert dataset_type in self.DATASET_MAP
        assert prompt_variant in PROMPT_VARIANTS

        self.params         = read_config(cfg_path)
        self.cfg_path       = cfg_path
        self.model_name     = model_name
        self.dataset_type   = dataset_type
        self.prompt_variant = prompt_variant
        self.manifest_csv   = manifest_csv
        self.image_root     = image_root
        self.resolution     = resolution

        cfg = self.params["CausalAudit"]
        dataset_cls, collate_fn, valid_conditions, results_key = self.DATASET_MAP[dataset_type]

        self.dataset_cls      = dataset_cls
        self.collate_fn       = collate_fn
        self.valid_conditions = valid_conditions
        self.num_workers      = int(cfg.get("num_workers", 4))

        # Results dir: use override, then config key, then subdirectory for
        # non-default prompt variants
        base_results = results_dir or cfg[results_key]
        if prompt_variant != "default":
            base_results = os.path.join(
                base_results, "prompt_sensitivity", prompt_variant
            )
        self.results_dir = base_results

        self.model_out_dir = os.path.join(self.results_dir, model_name)
        os.makedirs(self.model_out_dir, exist_ok=True)

        self.wrapper  = load_model_wrapper(model_name, self.params, dataset_type=dataset_type)
        self.modality = self.wrapper.modality

    def run(self, conditions: list = None):
        conditions = conditions or self.valid_conditions
        for condition in conditions:
            if condition not in self.valid_conditions:
                print(
                    f"[InferenceRunner] '{condition}' not supported for "
                    f"dataset_type='{self.dataset_type}'. Skipping."
                )
                continue
            self._run_condition(condition)

    def _run_condition(self, condition: str):
        out_path = os.path.join(self.model_out_dir, f"{condition}.csv")

        # Build dataset with all overrides
        if self.dataset_type == "mimic":
            dataset = ProbeSetDataset(
                cfg_path=self.cfg_path,
                condition=condition,
                modality=self.modality,
                prompt_variant=self.prompt_variant,
                manifest_csv=self.manifest_csv,
                image_root=self.image_root,
                resolution=self.resolution,
            )
        else:
            dataset = CheXpertDataset(
                cfg_path=self.cfg_path,
                condition=condition,
                modality=self.modality,
            )

        if len(dataset) == 0:
            print(f"[InferenceRunner] {self.model_name} | {condition}: 0 cases, skipping.")
            return

        done_ids = self._load_done_ids(out_path)
        if done_ids:
            print(
                f"[InferenceRunner] {self.model_name} | {condition}: "
                f"resuming, {len(done_ids)} done."
            )

        loader = DataLoader(
            dataset=dataset, batch_size=1, shuffle=False,
            num_workers=self.num_workers, collate_fn=self.collate_fn,
            pin_memory=False,
        )

        rows = []
        for batch in tqdm(
            loader,
            desc=f"{self.model_name} | {self.dataset_type} | {self.prompt_variant} | {condition}",
            unit="batch",
        ):
            for i in range(len(batch["case_ids"])):
                case_id = batch["case_ids"][i]
                if case_id in done_ids:
                    continue

                try:
                    raw_text, parsed_answer, confidence = self.wrapper.predict(
                        image=batch["images"][i],
                        prompt=batch["prompts"][i],
                        finding=batch["findings"][i],
                    )
                except Exception as e:
                    print(f"[InferenceRunner] Error on {case_id}: {e}")
                    raw_text, parsed_answer, confidence = str(e), -1, 0.5

                rows.append({
                    "case_id":       case_id,
                    "source":        batch["sources"][i],
                    "model_name":    self.model_name,
                    "condition":     condition,
                    "prompt_variant": self.prompt_variant,
                    "raw_answer":    raw_text,
                    "parsed_answer": parsed_answer,
                    "confidence":    round(confidence, 6),
                    "ground_truth":  int(batch["ground_truths"][i].item()),
                })

                if len(rows) % 100 == 0:
                    self._append_rows(rows, out_path)
                    rows = []

        if rows:
            self._append_rows(rows, out_path)

        total = pd.read_csv(out_path).shape[0] if os.path.exists(out_path) else 0
        print(f"[InferenceRunner] {self.model_name} | {condition}: done. Rows: {total}")

    def _load_done_ids(self, out_path: str) -> set:
        if not os.path.exists(out_path):
            return set()
        try:
            return set(pd.read_csv(out_path)["case_id"].astype(str).tolist())
        except Exception:
            return set()

    def _append_rows(self, rows: list, out_path: str):
        df_new = pd.DataFrame(rows)
        if os.path.exists(out_path):
            df_new.to_csv(out_path, mode="a", header=False, index=False)
        else:
            df_new.to_csv(out_path, mode="w", header=True, index=False)