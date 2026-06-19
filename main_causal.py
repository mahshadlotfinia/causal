"""
main_causal.py
Created May 22, 2026

Main governing functions for the causal grounding audit of medical VLMs

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia/
"""

import os
import pickle
import shutil

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoProcessor
from PIL import Image

from config.serde import read_config
from Inference.model_wrappers import MODEL_REGISTRY
from Inference.inference_runner import InferenceRunner
from Inference.metrics import MetricsComputer, SupplementaryMetrics, spearman_rank_cgr

import warnings
warnings.filterwarnings("ignore")


CHEXPERT_FINDINGS = [
    "atelectasis", "cardiomegaly", "consolidation", "edema",
    "enlarged_cardiomediastinum", "fracture", "lung_lesion",
    "lung_opacity", "no_finding", "pleural_effusion", "pleural_other",
    "pneumonia", "pneumothorax", "support_devices",
]




def main_setup_raddino_probe(global_config_path: str):
    """
    Extracts RAD-DINO CLS features for the MIMIC train+valid splits and
    trains one LogisticRegression probe per finding. Saves everything to
    raddino_probes.pkl. Run once before any RAD-DINO inference.
    """

    params = read_config(global_config_path)
    cfg    = params["CausalAudit"]

    mimic_csv     = cfg["mimic_master_csv"]
    image_root    = cfg["image_root"]
    features_path = cfg["raddino_features_path"]
    probe_dir     = cfg["raddino_probe_dir"]
    hf_id         = MODEL_REGISTRY["RAD-DINO"]["hf_id"]
    batch_size    = int(cfg.get("raddino_feature_batch_size", 64))
    train_splits  = cfg.get("raddino_train_splits", ["train", "valid"])

    os.makedirs(probe_dir, exist_ok=True)

    df = pd.read_csv(mimic_csv)
    df = df[df["view"].isin(["PA", "AP"])]
    df = df[df["split"].isin(train_splits)].reset_index(drop=True)

    def _resolve(rel):
        return os.path.join(
            image_root, str(rel).replace("/files/", "/preprocessed224/")
        )

    image_paths = df["jpg_rel_path"].apply(_resolve).tolist()
    print(f"[setup_raddino] {len(image_paths)} frontal images for feature extraction.")

    if os.path.exists(features_path):
        print(f"[setup_raddino] Found cached features, skipping extraction.")
        data     = np.load(features_path, allow_pickle=True)
        features = data["features"]
        df       = df.iloc[data["row_indices"].tolist()].reset_index(drop=True)
    else:
        device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        processor = AutoProcessor.from_pretrained(hf_id)
        model     = AutoModel.from_pretrained(
            hf_id, torch_dtype=torch.float16
        ).to(device)
        model.eval()

        all_features, valid_indices = [], []
        for start in tqdm(range(0, len(image_paths), batch_size), desc="RAD-DINO features"):
            batch_paths = image_paths[start: start + batch_size]
            imgs, idxs = [], []
            for j, p in enumerate(batch_paths):
                try:
                    imgs.append(Image.open(p).convert("RGB"))
                    idxs.append(start + j)
                except Exception:
                    continue
            if not imgs:
                continue
            inputs = processor(images=imgs, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                out = model(**inputs)
            cls = out.last_hidden_state[:, 0, :].float().cpu().numpy()
            all_features.append(cls)
            valid_indices.extend(idxs)

        features = np.concatenate(all_features, axis=0)
        np.savez_compressed(
            features_path,
            features=features,
            row_indices=np.array(valid_indices),
        )
        df = df.iloc[valid_indices].reset_index(drop=True)
        print(f"[setup_raddino] Features saved. Shape: {features.shape}")

    scaler           = StandardScaler()
    features_scaled  = scaler.fit_transform(features)

    probes = {}
    for finding in CHEXPERT_FINDINGS:
        if finding not in df.columns:
            continue
        labels = df[finding].copy()
        mask   = labels.isin([0, 1])
        X, y   = features_scaled[mask.values], labels[mask].astype(int).values
        if len(np.unique(y)) < 2:
            print(f"[setup_raddino] {finding}: single class, skipping.")
            continue
        clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs",
                                  n_jobs=-1, random_state=42)
        clf.fit(X, y)
        probes[finding] = clf
        print(f"[setup_raddino] {finding}: {len(y)} samples, pos={y.mean():.3f}")

    probe_file = os.path.join(probe_dir, "raddino_probes.pkl")
    with open(probe_file, "wb") as f:
        pickle.dump({"probes": probes, "scaler": scaler}, f)
    print(f"[setup_raddino] {len(probes)} probes saved to {probe_file}.")




def main_setup_raddino_probe_chexpert(global_config_path: str):
    """
    Same as main_setup_raddino_probe but uses CheXpert train+valid splits.
    Saves probes to raddino_chexpert_probe_dir. Run once before CheXpert
    RAD-DINO inference.
    """

    params = read_config(global_config_path)
    cfg    = params["CausalAudit"]

    chexpert_csv  = cfg["chexpert_master_csv"]
    image_root    = cfg["chexpert_image_root"]
    features_path = cfg["raddino_chexpert_features_path"]
    probe_dir     = cfg["raddino_chexpert_probe_dir"]
    hf_id         = MODEL_REGISTRY["RAD-DINO"]["hf_id"]
    batch_size    = int(cfg.get("raddino_feature_batch_size", 64))
    train_splits  = cfg.get("raddino_train_splits", ["train", "valid"])

    os.makedirs(probe_dir, exist_ok=True)

    df = pd.read_csv(chexpert_csv)
    df = df[df["view"] == "Frontal"]
    df = df[df["split"].isin(train_splits)].reset_index(drop=True)

    def _resolve(rel):
        return os.path.join(
            image_root, str(rel).replace("CheXpert-v1.0/", "CheXpert-v1.0/preprocessed224/")
        )

    image_paths = df["jpg_rel_path"].apply(_resolve).tolist()
    print(f"[setup_raddino_chexpert] {len(image_paths)} frontal images.")

    if os.path.exists(features_path):
        print(f"[setup_raddino_chexpert] Found cached features, skipping extraction.")
        data     = np.load(features_path, allow_pickle=True)
        features = data["features"]
        df       = df.iloc[data["row_indices"].tolist()].reset_index(drop=True)
    else:
        device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        processor = AutoProcessor.from_pretrained(hf_id)
        model     = AutoModel.from_pretrained(
            hf_id, torch_dtype=torch.float16
        ).to(device)
        model.eval()

        all_features, valid_indices = [], []
        for start in tqdm(range(0, len(image_paths), batch_size), desc="RAD-DINO features (CheXpert)"):
            batch_paths = image_paths[start: start + batch_size]
            imgs, idxs = [], []
            for j, p in enumerate(batch_paths):
                try:
                    imgs.append(Image.open(p).convert("RGB"))
                    idxs.append(start + j)
                except Exception:
                    continue
            if not imgs:
                continue
            inputs = processor(images=imgs, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                out = model(**inputs)
            cls = out.last_hidden_state[:, 0, :].float().cpu().numpy()
            all_features.append(cls)
            valid_indices.extend(idxs)

        features = np.concatenate(all_features, axis=0)
        np.savez_compressed(
            features_path,
            features=features,
            row_indices=np.array(valid_indices),
        )
        df = df.iloc[valid_indices].reset_index(drop=True)
        print(f"[setup_raddino_chexpert] Features saved. Shape: {features.shape}")

    scaler          = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    probes = {}
    for finding in CHEXPERT_FINDINGS:
        if finding not in df.columns:
            continue
        labels = df[finding].copy()
        mask   = labels.isin([0, 1])
        X, y   = features_scaled[mask.values], labels[mask].astype(int).values
        if len(np.unique(y)) < 2:
            print(f"[setup_raddino_chexpert] {finding}: single class, skipping.")
            continue
        clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs",
                                  n_jobs=-1, random_state=42)
        clf.fit(X, y)
        probes[finding] = clf
        print(f"[setup_raddino_chexpert] {finding}: {len(y)} samples, pos={y.mean():.3f}")

    probe_file = os.path.join(probe_dir, "raddino_probes.pkl")
    with open(probe_file, "wb") as f:
        pickle.dump({"probes": probes, "scaler": scaler}, f)
    print(f"[setup_raddino_chexpert] {len(probes)} probes saved to {probe_file}.")



def main_run_baseline(global_config_path: str, model_name: str):
    """Run inference on the original unperturbed image for one model (MIMIC)."""
    runner = InferenceRunner(cfg_path=global_config_path, model_name=model_name, dataset_type="mimic")
    runner.run(conditions=["original"])



def main_run_perturbed(global_config_path: str, model_name: str):
    """Run swap + target_mask + irrelevant_mask for one model (MIMIC)."""
    runner = InferenceRunner(cfg_path=global_config_path, model_name=model_name, dataset_type="mimic")
    runner.run(conditions=["swap", "target_mask", "irrelevant_mask"])



def main_run_model(
    global_config_path: str,
    model_name: str,
    conditions: list = None,
):
    """
    Run all four MIMIC conditions (or a specified subset) for a single model.

    Args:
        model_name : must match a key in MODEL_REGISTRY
        conditions : list of conditions, or None for all four
    """
    runner = InferenceRunner(cfg_path=global_config_path, model_name=model_name, dataset_type="mimic")
    runner.run(conditions=conditions)



def main_compute_metrics(global_config_path: str, output_dir: str = None):
    """Loads all per-model MIMIC result CSVs and computes the full metric suite."""
    params = read_config(global_config_path)
    cfg    = params["CausalAudit"]
    if output_dir is None:
        output_dir = os.path.join(cfg["results_dir"], "metrics")
    computer = MetricsComputer(global_config_path, dataset_type="mimic")
    computer.save(output_dir)



def main_resolution_setup(
    global_config_path: str,
    n_cases: int = 100,
    seed: int = 42,
):
    """
    Creates the 100-case manifest and temporary config for the 512-resolution
    sensitivity check. Run this once before running per-model resolution checks.
    Outputs: {results_dir}/resolution_check/manifest_512.csv
             {results_dir}/resolution_check/config_512.yaml
    """
    import yaml

    params = read_config(global_config_path)
    cfg    = params["CausalAudit"]

    check_dir = os.path.join(cfg["results_dir"], "resolution_check")
    os.makedirs(check_dir, exist_ok=True)

    manifest = pd.read_csv(cfg["manifest_csv"])
    ms_cases = manifest[
        (manifest["source"] == "ms_cxr") & manifest["box_x"].notna()
    ].copy()
    sampled = ms_cases.sample(n=min(n_cases, len(ms_cases)), random_state=seed)
    tmp_manifest = os.path.join(check_dir, "manifest_512.csv")
    sampled.to_csv(tmp_manifest, index=False)
    print(f"[resolution_setup] Saved {len(sampled)} cases to {tmp_manifest}")

    tmp_cfg = params.copy()
    tmp_cfg["CausalAudit"] = dict(cfg)
    tmp_cfg["CausalAudit"]["manifest_csv"]      = tmp_manifest
    tmp_cfg["CausalAudit"]["target_resolution"] = 512
    tmp_cfg["CausalAudit"]["results_dir"]       = os.path.join(check_dir, "results_512")
    os.makedirs(tmp_cfg["CausalAudit"]["results_dir"], exist_ok=True)

    tmp_cfg_path = os.path.join(check_dir, "config_512.yaml")
    with open(tmp_cfg_path, "w") as f:
        yaml.dump(tmp_cfg, f)
    print(f"[resolution_setup] Config written to {tmp_cfg_path}")



def main_resolution_run_model(global_config_path: str, model_name: str):
    """
    Runs original + target_mask at 512 resolution for one model.
    Run main_resolution_setup() once before calling this.
    """
    params    = read_config(global_config_path)
    cfg       = params["CausalAudit"]
    check_dir = os.path.join(cfg["results_dir"], "resolution_check")
    tmp_cfg   = os.path.join(check_dir, "config_512.yaml")

    if not os.path.exists(tmp_cfg):
        raise FileNotFoundError(
            f"config_512.yaml not found at {tmp_cfg}. "
            f"Run main_resolution_setup() first."
        )

    runner = InferenceRunner(
        cfg_path=tmp_cfg,
        model_name=model_name,
        dataset_type="mimic",
        manifest_csv=os.path.join(check_dir, "manifest_512.csv"),
        results_dir=os.path.join(check_dir, "results_512"),
        image_root=cfg["image_root"],
        resolution=512,
    )
    runner.run(conditions=["original", "target_mask"])



def main_resolution_finalize(global_config_path: str):
    """
    After all models have run at 512, compute Spearman rank correlation
    between CGR@224 and CGR@512. Saves comparison CSV.
    """
    params    = read_config(global_config_path)
    cfg       = params["CausalAudit"]
    check_dir = os.path.join(cfg["results_dir"], "resolution_check")
    tmp_cfg   = os.path.join(check_dir, "config_512.yaml")

    computer_224 = MetricsComputer(global_config_path, dataset_type="mimic")
    computer_512 = MetricsComputer(
        tmp_cfg,
        dataset_type="mimic",
    )
    # Override the manifest and results_dir for the 512 computer
    computer_512.results_dir  = os.path.join(check_dir, "results_512")
    computer_512.manifest     = pd.read_csv(os.path.join(check_dir, "manifest_512.csv"))

    all_224 = computer_224._load_all_models()
    all_512 = computer_512._load_all_models()

    cgr_224 = {m: MetricsComputer._cgr(df) for m, df in all_224.items()}
    cgr_512 = {m: MetricsComputer._cgr(df) for m, df in all_512.items()}
    cgr_224 = {m: v for m, v in cgr_224.items() if not np.isnan(v)}
    cgr_512 = {m: v for m, v in cgr_512.items() if not np.isnan(v)}

    rho = spearman_rank_cgr(cgr_224, cgr_512)

    from Inference.stats_utils import bootstrap_proportion, paired_bootstrap_diff

    models_in_both = sorted(set(cgr_224) & set(cgr_512))

    rows = []
    for m in models_in_both:
        v224 = MetricsComputer._cgr_outcomes(all_224[m])
        v512 = MetricsComputer._cgr_outcomes(all_512[m])
        b224 = bootstrap_proportion(v224)
        b512 = bootstrap_proportion(v512)

        # Paired bootstrap on shared case_ids
        df224 = all_224[m][["case_id"]].copy()
        df512 = all_512[m][["case_id"]].copy()
        shared = set(df224["case_id"]) & set(df512["case_id"])
        paired_p = float("nan")
        if len(shared) >= 10:
            a = (all_224[m].set_index("case_id")
                 .loc[list(shared)]["pa_original"]
                 .isin([0,1]).astype(float).values)
            b = (all_512[m].set_index("case_id")
                 .loc[list(shared)]["pa_original"]
                 .isin([0,1]).astype(float).values)
            # Use CGR outcomes aligned by shared case_ids
            def _cgr_from_ids(df, ids):
                sub = df[
                    (df["case_id"].isin(ids)) &
                    (df["source"] == "ms_cxr") &
                    (df["ground_truth"] == 1) &
                    (df["pa_original"] == 1) &
                    (df["pa_target_mask"].isin([0, 1]))
                ] if "pa_target_mask" in df.columns else df.iloc[:0]
                return (sub["pa_target_mask"] != sub["pa_original"]).astype(float).values
            a_cgr = _cgr_from_ids(all_224[m], shared)
            b_cgr = _cgr_from_ids(all_512[m], shared)
            if len(a_cgr) > 5 and len(a_cgr) == len(b_cgr):
                pd_res = paired_bootstrap_diff(a_cgr, b_cgr)
                paired_p = pd_res["p_value"]

        rows.append({
            "model":          m,
            "CGR_224":        round(b224["point"],    4),
            "CGR_224_ci_lower": round(b224["ci_lower"], 4),
            "CGR_224_ci_upper": round(b224["ci_upper"], 4),
            "CGR_224_n":      b224["n"],
            "CGR_512":        round(b512["point"],    4),
            "CGR_512_ci_lower": round(b512["ci_lower"], 4),
            "CGR_512_ci_upper": round(b512["ci_upper"], 4),
            "CGR_512_n":      b512["n"],
            "paired_p_224vs512": round(paired_p, 4) if not np.isnan(paired_p) else float("nan"),
        })

    comparison = pd.DataFrame(rows)
    comparison["rank_224"] = comparison["CGR_224"].rank(ascending=False).astype(int)
    comparison["rank_512"] = comparison["CGR_512"].rank(ascending=False).astype(int)
    comparison["spearman_rho"] = rho

    out_path = os.path.join(check_dir, "resolution_check_results.csv")
    comparison.to_csv(out_path, index=False)
    print(f"\n[resolution_finalize] Results saved to {out_path}")
    print(comparison.to_string(index=False))
    verdict = "PASS" if rho >= 0.9 else "WARN"
    print(f"\n[resolution_finalize] {verdict}: Spearman rho={rho:.4f}")



def main_prompt_sensitivity_run_model(
    global_config_path: str,
    model_name: str,
    n_cases: int = 100,
    seed: int = 42,
):
    """
    Runs the 'brief' and 'clinical' prompt variants for one model on a
    100-case MS-CXR subset (original condition only). The 'default' variant
    is already covered by the main MIMIC inference run.

    Results go to {results_dir}/prompt_sensitivity/{variant}/{model_name}/original.csv
    """
    params = read_config(global_config_path)
    cfg    = params["CausalAudit"]

    # Sample a fixed subset (same seed = same 100 cases for all models)
    manifest = pd.read_csv(cfg["manifest_csv"])
    ms_cases = manifest[manifest["source"] == "ms_cxr"].copy()
    sampled  = ms_cases.sample(n=min(n_cases, len(ms_cases)), random_state=seed)
    subset_path = os.path.join(cfg["results_dir"], "prompt_sensitivity", "manifest_subset.csv")
    os.makedirs(os.path.dirname(subset_path), exist_ok=True)
    sampled.to_csv(subset_path, index=False)

    for variant in ["brief", "clinical"]:
        runner = InferenceRunner(
            cfg_path=global_config_path,
            model_name=model_name,
            dataset_type="mimic",
            prompt_variant=variant,
            manifest_csv=subset_path,
        )
        runner.run(conditions=["original"])



def main_prompt_sensitivity_metrics(global_config_path: str):
    """
    Loads results for all three prompt variants and computes accuracy and CGR
    per model per variant. Saves a comparison CSV and prints a summary.
    """
    params     = read_config(global_config_path)
    cfg        = params["CausalAudit"]
    sens_dir   = os.path.join(cfg["results_dir"], "prompt_sensitivity")
    subset_csv = os.path.join(sens_dir, "manifest_subset.csv")
    manifest   = pd.read_csv(subset_csv)
    out_dir    = os.path.join(sens_dir, "metrics")
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for variant in ["default", "brief", "clinical"]:
        if variant == "default":
            results_dir = cfg["results_dir"]
        else:
            results_dir = os.path.join(cfg["results_dir"], "prompt_sensitivity", variant)

        model_dirs = [
            d for d in os.listdir(results_dir)
            if os.path.isdir(os.path.join(results_dir, d))
            and os.path.exists(os.path.join(results_dir, d, "original.csv"))
        ] if os.path.exists(results_dir) else []

        for model_name in model_dirs:
            csv_path = os.path.join(results_dir, model_name, "original.csv")
            df_raw = pd.read_csv(csv_path)

            # For default variant, restrict to the same 100-case subset
            if variant == "default":
                subset_ids = set(manifest["case_id"].astype(str))
                df_raw = df_raw[df_raw["case_id"].astype(str).isin(subset_ids)]

            # Join ground truth and metadata from subset manifest
            _meta_cols = ["case_id", "source", "finding", "label", "box_x"]
            if "pa_target_mask" in manifest.columns:
                _meta_cols.append("pa_target_mask")
            df_raw = df_raw.merge(
                manifest[_meta_cols].drop_duplicates("case_id"),
                on="case_id", how="left",
            )

            valid = df_raw[
                df_raw["parsed_answer"].isin([0, 1]) &
                df_raw["ground_truth"].isin([0, 1])
            ]
            acc = (valid["parsed_answer"] == valid["ground_truth"]).mean() \
                if len(valid) > 0 else float("nan")

            from Inference.metrics import prompt_sensitivity_ci
            ci = prompt_sensitivity_ci(df_raw)
            rows.append({
                "model":       model_name,
                "variant":     variant,
                "accuracy":    ci["point"],
                "ci_lower":    ci["ci_lower"],
                "ci_upper":    ci["ci_upper"],
                "n_cases":     ci["n"],
                "parse_rate":  ci["parse_rate"],
            })

    comparison = pd.DataFrame(rows)
    out_path = os.path.join(out_dir, "prompt_sensitivity_comparison.csv")
    comparison.to_csv(out_path, index=False)
    print("\n=== Prompt Sensitivity Comparison ===")
    print(comparison.pivot(index="model", columns="variant", values="accuracy").to_string())



def main_supplementary_metrics(
    global_config_path: str,
    dataset_type: str = "mimic",
    output_dir: str = None,
):
    """
    Computes confidence calibration, swap consistency, CGR by label polarity,
    and a sample of ungrounded correct cases for qualitative review.

    Args:
        dataset_type : 'mimic' | 'chexpert'
        output_dir   : defaults to {results_dir}/supplementary
    """
    params = read_config(global_config_path)
    cfg    = params["CausalAudit"]

    if output_dir is None:
        base = cfg["results_dir"] if dataset_type == "mimic" else cfg["chexpert_results_dir"]
        output_dir = os.path.join(base, "metrics")

    computer    = MetricsComputer(global_config_path, dataset_type=dataset_type)
    all_results = computer._load_all_models()
    manifest    = computer.manifest

    supp = SupplementaryMetrics()
    supp.save(all_results, manifest, output_dir)



def main_resolution_check(
    global_config_path: str,
    n_cases: int = 100,
    seed: int = 42,
):
    """
    Step 6 of the execution plan.

    Samples n_cases from the MS-CXR subset of the manifest (these are the
    cases with target boxes, making CGR computable), re-runs all 11 models
    at 448x448, computes CGR at 448, then reports Spearman rank correlation
    vs the 224 CGR values already on disk.

    Results are saved to {results_dir}/resolution_check/.

    Args:
        n_cases : number of MS-CXR cases to sample (default 100)
        seed    : random seed for case sampling
    """

    params = read_config(global_config_path)
    cfg    = params["CausalAudit"]

    results_dir   = cfg["results_dir"]
    manifest_csv  = cfg["manifest_csv"]
    check_dir     = os.path.join(results_dir, "resolution_check")
    os.makedirs(check_dir, exist_ok=True)

    # Sample n_cases from MS-CXR (cases with target boxes)
    manifest = pd.read_csv(manifest_csv)
    ms_cases = manifest[
        (manifest["source"] == "ms_cxr") & manifest["box_x"].notna()
    ].copy()
    sampled  = ms_cases.sample(n=min(n_cases, len(ms_cases)), random_state=seed)

    # Write a temporary manifest containing only the sampled cases
    tmp_manifest = os.path.join(check_dir, "resolution_check_manifest.csv")
    sampled.to_csv(tmp_manifest, index=False)

    # Build a temporary config pointing to the sampled manifest and 448 resolution
    import yaml
    tmp_cfg = params.copy()
    tmp_cfg["CausalAudit"] = dict(cfg)
    tmp_cfg["CausalAudit"]["manifest_csv"]       = tmp_manifest
    tmp_cfg["CausalAudit"]["target_resolution"]  = 448
    tmp_cfg["CausalAudit"]["results_dir"]        = os.path.join(check_dir, "results_448")

    tmp_cfg_path = os.path.join(check_dir, "config_448.yaml")
    with open(tmp_cfg_path, "w") as f:
        yaml.dump(tmp_cfg, f)

    os.makedirs(tmp_cfg["CausalAudit"]["results_dir"], exist_ok=True)

    # Run original + target_mask conditions at 448 for all models
    model_list = list(MODEL_REGISTRY.keys())
    for model_name in model_list:
        print(f"\n[resolution_check] Running {model_name} at 448x448 ...")
        try:
            runner = InferenceRunner(cfg_path=tmp_cfg_path, model_name=model_name)
            runner.run(conditions=["original", "target_mask"])
            del runner
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"[resolution_check] ERROR on {model_name}: {e}")
            torch.cuda.empty_cache()

    # Compute CGR at 448 per model
    computer_224 = MetricsComputer(global_config_path)
    computer_448 = MetricsComputer(tmp_cfg_path)

    all_224 = computer_224._load_all_models()
    all_448 = computer_448._load_all_models()

    cgr_224 = {m: MetricsComputer._cgr(df) for m, df in all_224.items()}
    cgr_448 = {m: MetricsComputer._cgr(df) for m, df in all_448.items()}

    # Filter out NaN (models without MS-CXR target_mask results)
    cgr_224 = {m: v for m, v in cgr_224.items() if not np.isnan(v)}
    cgr_448 = {m: v for m, v in cgr_448.items() if not np.isnan(v)}

    rho = spearman_rank_cgr(cgr_224, cgr_448)

    models_in_both = sorted(set(cgr_224) & set(cgr_448))
    comparison = pd.DataFrame({
        "model":   models_in_both,
        "CGR_224": [cgr_224[m] for m in models_in_both],
        "CGR_448": [cgr_448[m] for m in models_in_both],
    })
    comparison["rank_224"] = comparison["CGR_224"].rank(ascending=False).astype(int)
    comparison["rank_448"] = comparison["CGR_448"].rank(ascending=False).astype(int)
    comparison["spearman_rho"] = rho

    out_path = os.path.join(check_dir, "resolution_check_results.csv")
    comparison.to_csv(out_path, index=False)
    print(f"\n[resolution_check] Results saved to {out_path}")
    print(comparison.to_string(index=False))

    if rho >= 0.9:
        print(f"\n[resolution_check] PASS: rho={rho:.4f} >= 0.90. "
              f"Model ranking is stable across resolutions.")
    else:
        print(f"\n[resolution_check] WARN: rho={rho:.4f} < 0.90. "
              f"Model ranking may be resolution-sensitive. Investigate.")



def main_run_chexpert_model(global_config_path: str, model_name: str):
    """
    Run original + swap conditions on the CheXpert manifest for one model.
    Results are saved to chexpert_results_dir/{model_name}/.
    """
    runner = InferenceRunner(
        cfg_path=global_config_path,
        model_name=model_name,
        dataset_type="chexpert",
    )
    runner.run(conditions=["original", "swap"])



def main_compute_chexpert_metrics(global_config_path: str, output_dir: str = None):
    """
    Computes metrics from CheXpert inference results and produces a
    cross-dataset UAR/accuracy comparison table (MIMIC vs CheXpert).
    CGR and AFS will be NaN (no bounding boxes in CheXpert).
    """
    params = read_config(global_config_path)
    cfg    = params["CausalAudit"]

    if output_dir is None:
        output_dir = os.path.join(cfg["chexpert_results_dir"], "metrics")

    chexpert_computer = MetricsComputer(global_config_path, dataset_type="chexpert")
    chexpert_computer.save(output_dir)

    mimic_computer = MetricsComputer(global_config_path, dataset_type="mimic")
    mimic_all      = mimic_computer._load_all_models()
    chexpert_all   = chexpert_computer._load_all_models()

    rows = []
    for model_name in sorted(set(mimic_all) & set(chexpert_all)):
        rows.append({
            "model":        model_name,
            "UAR_mimic":    round(MetricsComputer._uar(mimic_all[model_name]), 4),
            "UAR_chexpert": round(MetricsComputer._uar(chexpert_all[model_name]), 4),
            "acc_mimic":    round(MetricsComputer._accuracy(mimic_all[model_name]), 4),
            "acc_chexpert": round(MetricsComputer._accuracy(chexpert_all[model_name]), 4),
        })

    comparison = pd.DataFrame(rows).sort_values("UAR_mimic", ascending=False)
    comparison.to_csv(os.path.join(output_dir, "cross_dataset_comparison.csv"), index=False)

    print("\n=== Cross-Dataset UAR Comparison (MIMIC vs CheXpert) ===")
    print(comparison.to_string(index=False))










if __name__ == "__main__":
    global_config_path = "/PATH/Repositories/causal/config/config.yaml"

    # ==========================================================
    # STAGE 0: one-time setups (run each once, then keep commented)
    # ==========================================================
    # main_setup_raddino_probe(global_config_path)           # MIMIC RAD-DINO probes
    # main_setup_raddino_probe_chexpert(global_config_path)  # CheXpert RAD-DINO probes
    # main_resolution_setup(global_config_path)              # 512 resolution manifest + config

    # ==========================================================
    # STAGE 1-3: MIMIC inference (run one model at a time)
    # ==========================================================
    # main_run_model(global_config_path, "MedGemma-1.5-4B")
    # main_run_model(global_config_path, "Gemma-4-26B")
    # main_run_model(global_config_path, "Qwen3-VL-32B")
    # main_run_model(global_config_path, "RAD-DINO")
    # main_run_model(global_config_path, "MedGemma-27B-text")
    # main_run_model(global_config_path, "Mistral-Small-4-119B")
    # main_run_model(global_config_path, "GPT-5")
    # main_run_model(global_config_path, "DeepSeek-R1-7B")
    # main_run_model(global_config_path, "LLaVA-Med-7B")

    # ---- Or run baseline and perturbed separately ----
    # main_run_baseline(global_config_path, "LLaVA-Med-7B")
    # main_run_perturbed(global_config_path, "LLaVA-Med-7B")

    # ==========================================================
    # STAGE 4: MIMIC metrics
    # ==========================================================
    main_compute_metrics(global_config_path)

    # ==========================================================
    # STAGE 5: resolution sensitivity check (512 vs 224)
    # Run 5b one model at a time, then finalize with 5c
    # ==========================================================
    # main_resolution_run_model(global_config_path, "MedGemma-1.5-4B")   # 5b
    # main_resolution_run_model(global_config_path, "Gemma-4-26B")        # 5b
    # main_resolution_run_model(global_config_path, "Qwen3-VL-32B")       # 5b
    # main_resolution_run_model(global_config_path, "Mistral-Small-4-119B") # 5b
    # main_resolution_run_model(global_config_path, "RAD-DINO")           # 5b
    # main_resolution_run_model(global_config_path, "DeepSeek-R1-7B")      # 5b
    # main_resolution_run_model(global_config_path, "GPT-5")      # 5b
    # main_resolution_run_model(global_config_path, "MedGemma-27B-text")      # 5b
    # main_resolution_run_model(global_config_path, "LLaVA-Med-7B")      # 5b
    main_resolution_finalize(global_config_path)                         # 5c

    # ==========================================================
    # STAGE 6: CheXpert inference (run one model at a time)
    # ==========================================================
    # main_run_chexpert_model(global_config_path, "MedGemma-1.5-4B")
    # main_run_chexpert_model(global_config_path, "Gemma-4-26B")
    # main_run_chexpert_model(global_config_path, "Qwen3-VL-32B")
    # main_run_chexpert_model(global_config_path, "Mistral-Small-4-119B")
    # main_run_chexpert_model(global_config_path, "MedGemma-27B-text")
    # main_run_chexpert_model(global_config_path, "RAD-DINO")
    # main_run_chexpert_model(global_config_path, "GPT-5")
    # main_run_chexpert_model(global_config_path, "DeepSeek-R1-7B")
    # main_run_chexpert_model(global_config_path, "DeepSeek-R1-70B")
    # main_run_chexpert_model(global_config_path, "LLaVA-Med-7B")

    # ==========================================================
    # STAGE 7: CheXpert metrics + cross-dataset comparison
    # ==========================================================
    main_compute_chexpert_metrics(global_config_path)

    # ==========================================================
    # STAGE 8: prompt sensitivity (run one model at a time)
    # ==========================================================
    # main_prompt_sensitivity_run_model(global_config_path, "MedGemma-1.5-4B")
    # main_prompt_sensitivity_run_model(global_config_path, "Gemma-4-26B")
    # main_prompt_sensitivity_run_model(global_config_path, "Qwen3-VL-32B")
    # main_prompt_sensitivity_run_model(global_config_path, "Mistral-Small-4-119B")
    # main_prompt_sensitivity_run_model(global_config_path, "RAD-DINO")
    # main_prompt_sensitivity_run_model(global_config_path, "LLaVA-Med-7B")
    # main_prompt_sensitivity_run_model(global_config_path, "MedGemma-27B-text")
    # main_prompt_sensitivity_run_model(global_config_path, "DeepSeek-R1-7B")
    # main_prompt_sensitivity_run_model(global_config_path, "GPT-5")

    # ==========================================================
    # STAGE 9: prompt sensitivity metrics
    # ==========================================================
    main_prompt_sensitivity_metrics(global_config_path)

    # ==========================================================
    # STAGE 10: supplementary metrics (MIMIC and CheXpert)
    # ==========================================================
    main_supplementary_metrics(global_config_path, dataset_type="mimic")
    main_supplementary_metrics(global_config_path, dataset_type="chexpert")
