"""
data_loader/probe_set_data_loader.py
Created May 22, 2026

Probe set data loader for the causal grounding audit of medical VLMs.

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia/
"""

import os

import pandas as pd
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

from config.serde import read_config


CONDITIONS = ["original", "swap", "target_mask", "irrelevant_mask"]
MODALITIES = ["multimodal", "text_only", "vision_only"]
PROMPT_VARIANTS = ["default", "brief", "clinical"]

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



def _resolve_image_path(image_root: str, rel_path: str, resolution: int = 224) -> str:
    rel = str(rel_path)
    if resolution == 512:
        rel = rel.replace("files/", "preprocessed/")
    else:
        rel = rel.replace("files/", "preprocessed224/")
    return os.path.join(image_root, rel)


def _load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def _apply_mask(img: Image.Image, x: int, y: int, w: int, h: int) -> Image.Image:
    img = img.copy()
    draw = ImageDraw.Draw(img)
    draw.rectangle([x, y, x + w, y + h], fill=(0, 0, 0))
    return img



def _build_prompt(row: dict, modality: str, variant: str = "default") -> str:
    source = str(row.get("source", ""))

    if source in ("mimic_cxr", "ms_cxr"):
        finding = row.get("finding") or ""
        display = FINDING_TO_DISPLAY.get(finding, finding.replace("_", " "))
        if variant == "brief":
            return f"Is {display} present? Yes or No."
        if variant == "clinical":
            return (
                f"You are a radiologist reviewing a chest X-ray. "
                f"Is {display} present? Answer with a single word: Yes or No."
            )
        # default
        return (
            f"Is {display} present in this chest X-ray? "
            f"Answer with a single word: Yes or No."
        )

    if source == "rexerr":
        sentence = str(row.get("error_sentence") or "")
        if variant == "brief":
            return (
                f"Is this sentence accurate for this X-ray? "
                f"\"{sentence}\" Yes or No."
            )
        if variant == "clinical":
            return (
                f"You are a radiologist. Does the following sentence accurately "
                f"describe findings in this chest X-ray?\n"
                f"Sentence: \"{sentence}\"\n"
                f"Answer with a single word: Yes or No."
            )
        # default
        return (
            f"Does the following sentence accurately describe the findings "
            f"visible in this chest X-ray?\n"
            f"Sentence: \"{sentence}\"\n"
            f"Answer with a single word: Yes or No."
        )

    return "Answer with a single word: Yes or No."


def _compute_ground_truth(row: dict) -> int:
    source = str(row.get("source", ""))
    if source in ("mimic_cxr", "ms_cxr"):
        return int(row.get("label", -1))
    if source == "rexerr":
        return 1 - int(row.get("error_present", 0))
    return -1



class ProbeSetDataset(Dataset):
    """
    Args:
        cfg_path       : path to config.yaml
        condition      : one of CONDITIONS
        modality       : one of MODALITIES
        prompt_variant : one of PROMPT_VARIANTS (default: 'default')
        manifest_csv   : optional override (used by resolution check)
        image_root     : optional override (used by resolution check)
        resolution     : image resolution for path resolution (default 224)
    """

    def __init__(
        self,
        cfg_path: str,
        condition: str,
        modality: str = "multimodal",
        prompt_variant: str = "default",
        manifest_csv: str = None,
        image_root: str = None,
        resolution: int = None,
    ):
        assert condition in CONDITIONS, f"condition must be one of {CONDITIONS}"
        assert modality in MODALITIES, f"modality must be one of {MODALITIES}"
        assert prompt_variant in PROMPT_VARIANTS, (
            f"prompt_variant must be one of {PROMPT_VARIANTS}"
        )

        self.params         = read_config(cfg_path)
        self.condition      = condition
        self.modality       = modality
        self.prompt_variant = prompt_variant

        cfg = self.params["CausalAudit"]
        self.image_root = image_root or cfg["image_root"]
        self.resolution = resolution or int(cfg.get("target_resolution", 224))

        df = pd.read_csv(manifest_csv or cfg["manifest_csv"])

        if condition == "target_mask":
            df = df[df["box_x"].notna()].copy()
        elif condition == "irrelevant_mask":
            df = df[df["irrelevant_box_x"].notna()].copy()

        if modality == "vision_only":
            df = df[df["finding"].notna()].copy()

        self.records = df.reset_index(drop=True).to_dict("records")
        print(
            f"[ProbeSetDataset] condition={condition} | modality={modality} "
            f"| variant={prompt_variant} | resolution={self.resolution} "
            f"| cases={len(self.records)}"
        )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        row = self.records[idx]

        image = None
        if self.modality != "text_only":
            rel = row["swap_image_path"] if self.condition == "swap" else row["image_path"]
            image = _load_image(
                _resolve_image_path(self.image_root, rel, self.resolution)
            )
            # Boxes are stored in 224-space; scale to actual loaded resolution
            scale = self.resolution / 224
            if self.condition == "target_mask":
                image = _apply_mask(
                    image,
                    int(row["box_x"] * scale), int(row["box_y"] * scale),
                    int(row["box_w"] * scale), int(row["box_h"] * scale),
                )
            elif self.condition == "irrelevant_mask":
                image = _apply_mask(
                    image,
                    int(row["irrelevant_box_x"] * scale), int(row["irrelevant_box_y"] * scale),
                    int(row["irrelevant_box_w"] * scale), int(row["irrelevant_box_h"] * scale),
                )

        return {
            "case_id":      str(row["case_id"]),
            "source":       str(row["source"]),
            "condition":    self.condition,
            "image":        image,
            "prompt":       _build_prompt(row, self.modality, self.prompt_variant),
            "ground_truth": _compute_ground_truth(row),
            "finding":      str(row.get("finding") or ""),
            "label":        int(row.get("label", -1)),
        }



def probe_collate_fn(batch: list) -> dict:
    return {
        "case_ids":      [b["case_id"] for b in batch],
        "sources":       [b["source"] for b in batch],
        "conditions":    [b["condition"] for b in batch],
        "images":        [b["image"] for b in batch],
        "prompts":       [b["prompt"] for b in batch],
        "ground_truths": torch.tensor(
            [b["ground_truth"] for b in batch], dtype=torch.long
        ),
        "findings":      [b["finding"] for b in batch],
        "labels":        torch.tensor([b["label"] for b in batch], dtype=torch.long),
    }