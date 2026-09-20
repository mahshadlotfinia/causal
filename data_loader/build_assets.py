"""
data_loader/build_assets.py
Created on September 14, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import os
from typing import Dict

import numpy as np
import pandas as pd

from config.serde import read_config
from data_loader.build_utils import load_rgb, pixel_digest, read_csv_defensively, resolve_image_path, stratified_sample
from Inference.resume_utils import (MissingInput, append_status, check_build_params, claim_unit, ensure_dir, fingerprint_file,
                                    read_build_params, release_claim, status_path, write_build_params, write_csv_atomic)


def _manifest(cfg: Dict) -> pd.DataFrame:
    path = cfg["data"]["mimic_manifest"]
    if not os.path.exists(path):
        raise MissingInput(f"{path} does not exist; run main_extend_manifest_mimic first")
    return read_csv_defensively(path)


def _build_noise_image(cfg: Dict, manifest: pd.DataFrame) -> None:
    from PIL import Image
    from tqdm import tqdm
    a = cfg["assets"]
    path = a["noise_image"]
    params = {"seed": int(a["noise_seed"]), "resolution": int(cfg["data"]["resolution"]),
              "manifest": fingerprint_file(cfg["data"]["mimic_manifest"])}
    if os.path.exists(path) and check_build_params(path, params, "build_assets"):
        print(f"[build_assets] noise image is current: {path}", flush=True)
        return
    res = int(cfg["data"]["resolution"])
    total, total_sq, n = 0.0, 0.0, 0
    for rel in tqdm(manifest["image_path"].tolist(), desc="[build_assets] image statistics", unit="img"):
        arr = np.asarray(load_rgb(resolve_image_path(cfg, "mimic", rel, res)).convert("L"), dtype=np.float64)
        total += arr.sum()
        total_sq += (arr ** 2).sum()
        n += arr.size
    mean = total / n
    std = float(np.sqrt(max(total_sq / n - mean ** 2, 0.0)))
    rng = np.random.default_rng(int(a["noise_seed"]))
    noise = np.clip(rng.normal(mean, std, size=(res, res)), 0, 255).astype(np.uint8)
    ensure_dir(os.path.dirname(path))
    tmp = path + ".tmp.png"
    Image.fromarray(noise, mode="L").convert("RGB").save(tmp, format="PNG")
    os.replace(tmp, path)
    params.update({"pixel_mean": float(mean), "pixel_std": std, "pixel_digest": pixel_digest(path)})
    write_build_params(path, params)
    print(f"[build_assets] noise image written: mean {mean:.2f}, SD {std:.2f}, {path}", flush=True)


def _check_natural_image(cfg: Dict) -> None:
    a = cfg["assets"]
    path, prov = a["natural_image"], a["natural_image_provenance"]
    if os.path.exists(path) and os.path.exists(prov):
        from PIL import Image
        im = Image.open(path)
        recorded = (read_build_params(path) or {}).get("pixel_digest")
        if not recorded:
            write_build_params(path, {"pixel_digest": pixel_digest(path), "size": os.path.getsize(path)})
        print(f"[build_assets] natural image present: {path} ({im.size[0]}x{im.size[1]}, {im.mode}); provenance recorded.", flush=True)
        return
    print(f"[build_assets] the natural image is MISSING. Put a public-domain photograph of a cat at {path} "
          f"and its source URL and license in {prov}. The natural_image units skip until then.", flush=True)


def _build_subset(cfg: Dict, manifest: pd.DataFrame) -> None:
    subsets_dir = ensure_dir(cfg["data"]["subsets_dir"])
    mimic = manifest[manifest["source"] == "mimic_cxr"].copy()
    mimic["stratum"] = mimic["finding"].astype(str) + "__" + mimic["label"].astype(int).astype(str)
    name = "prompt_mimic_subset.csv"
    path = os.path.join(subsets_dir, name)
    params = {"seed": int(cfg["prompt_sensitivity"]["subset_seed"]), "n_mimic": int(cfg["prompt_sensitivity"]["mimic_subset_n"]),
              "manifest": fingerprint_file(cfg["data"]["mimic_manifest"])}
    if os.path.exists(path) and check_build_params(path, params, "build_assets"):
        print(f"[build_assets] {name} is current.", flush=True)
        return
    out = stratified_sample(mimic, "stratum", params["n_mimic"], params["seed"])[["case_id", "source", "finding", "label"]]
    write_csv_atomic(out, path)
    write_build_params(path, params)
    print(f"[build_assets] {name}: {len(out)} cases ({out['source'].value_counts().to_dict()})", flush=True)


def main_build_assets(cfg_path: str) -> None:
    cfg = read_config(cfg_path)["CausalAudit"]
    manifest = _manifest(cfg)
    subsets_dir = ensure_dir(cfg["data"]["subsets_dir"])
    if not claim_unit(subsets_dir, "build_assets"):
        print("[build_assets] claimed by a live job, leaving it alone.", flush=True)
        return
    try:
        _build_noise_image(cfg, manifest)
        _check_natural_image(cfg)
        _build_subset(cfg, manifest)
    finally:
        release_claim(subsets_dir, "build_assets")
    append_status(status_path(cfg, "build_assets"), "assets and subsets checked")
