"""
data_loader/chexpert_data_loader.py
Created May 23, 2026

CheXpert data loader for the causal grounding audit generalization analysis.

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia/
"""

import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from config.serde import read_config

CHEXPERT_CONDITIONS = ["original", "swap"]
MODALITIES = ["multimodal", "text_only", "vision_only"]

FINDING_TO_DISPLAY = {
    "atelectasis": "atelectasis",
    "cardiomegaly": "cardiomegaly",
    "consolidation": "consolidation",
    "edema": "pulmonary edema",
    "enlarged_cardiomediastinum": "enlarged cardiomediastinum",
    "fracture": "rib fracture",
    "lung_lesion": "lung lesion",
    "lung_opacity": "lung opacity",
    "no_finding": "any acute abnormality",
    "pleural_effusion": "pleural effusion",
    "pleural_other": "pleural abnormality",
    "pneumonia": "pneumonia",
    "pneumothorax": "pneumothorax",
    "support_devices": "support device",
}



def _resolve_chexpert_path(image_root: str, rel_path: str) -> str:
    return os.path.join(
        image_root,
        str(rel_path).replace("CheXpert-v1.0/", "CheXpert-v1.0/preprocessed224/")
    )


def _load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")



def _build_chexpert_prompt(row: dict) -> str:
    finding = row.get("finding") or ""
    display = FINDING_TO_DISPLAY.get(finding, finding.replace("_", " "))
    return (
        f"Is {display} present in this chest X-ray? "
        f"Answer with a single word: Yes or No."
    )


def _compute_ground_truth(row: dict) -> int:
    """Ground truth: label=1 -> Yes (1), label=0 -> No (0)."""
    return int(row.get("label", -1))



class CheXpertDataset(Dataset):
    """
    Loads the CheXpert probe manifest and serves one item per (case, condition).

    Args:
        cfg_path  : path to config.yaml
        condition : 'original' | 'swap'
        modality  : 'multimodal' | 'text_only' | 'vision_only'
    """

    def __init__(
        self,
        cfg_path: str,
        condition: str,
        modality: str = "multimodal",
    ):
        assert condition in CHEXPERT_CONDITIONS, (
            f"CheXpertDataset only supports conditions: {CHEXPERT_CONDITIONS}. "
            f"Got '{condition}'."
        )
        assert modality in MODALITIES, f"modality must be one of {MODALITIES}"

        self.params    = read_config(cfg_path)
        self.condition = condition
        self.modality  = modality

        cfg = self.params["CausalAudit"]
        self.image_root = cfg["chexpert_image_root"]

        df = pd.read_csv(cfg["chexpert_manifest_csv"])

        # RAD-DINO requires a known finding to select the right probe
        if modality == "vision_only":
            df = df[df["finding"].notna()].copy()

        self.records = df.reset_index(drop=True).to_dict("records")
        print(
            f"[CheXpertDataset] condition={condition} | modality={modality} "
            f"| cases={len(self.records)}"
        )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        row = self.records[idx]

        image = None
        if self.modality != "text_only":
            rel   = row["swap_image_path"] if self.condition == "swap" else row["image_path"]
            image = _load_image(_resolve_chexpert_path(self.image_root, rel))

        return {
            "case_id":      str(row["case_id"]),
            "source":       str(row["source"]),
            "condition":    self.condition,
            "image":        image,
            "prompt":       _build_chexpert_prompt(row),
            "ground_truth": _compute_ground_truth(row),
            "finding":      str(row.get("finding") or ""),
            "label":        int(row.get("label", -1)),
        }




def chexpert_collate_fn(batch: list) -> dict:
    return {
        "case_ids":     [b["case_id"] for b in batch],
        "sources":      [b["source"] for b in batch],
        "conditions":   [b["condition"] for b in batch],
        "images":       [b["image"] for b in batch],
        "prompts":      [b["prompt"] for b in batch],
        "ground_truths": torch.tensor(
            [b["ground_truth"] for b in batch], dtype=torch.long
        ),
        "findings":     [b["finding"] for b in batch],
        "labels":       torch.tensor([b["label"] for b in batch], dtype=torch.long),
    }