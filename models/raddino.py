"""
models/raddino.py
Created on September 14, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import hashlib
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config.serde import read_config
from data_loader.build_utils import (FRONTAL_VIEWS, abnormal_pool, absent_pool, load_chexpert_master, load_mimic_master,
                                     load_rgb, normal_pool, present_pool, resolve_image_path)
from Inference.resume_utils import (MissingInput, append_status, check_build_params, claim_unit, ensure_dir, fingerprint_file,
                                    read_build_params, release_claim, status_path, write_build_params, write_npz_atomic)
from Inference.serving_utils import require_packages


EMBED_DIM = 768


def _training_rows(cfg: Dict, dataset: str) -> pd.DataFrame:
    d = cfg["data"]
    cols = ["jpg_rel_path", "subject_id", "split", "view"] + d["all_findings"]
    if dataset == "chexpert":
        df = load_chexpert_master(cfg, usecols=cols)
        df = df[df["view"] == "Frontal"]
    else:
        df = load_mimic_master(cfg, usecols=cols)
        df = df[df["view"].isin(FRONTAL_VIEWS)]
    df = df[df["split"].isin(cfg["raddino"]["train_splits"])]
    return df.sort_values("jpg_rel_path").reset_index(drop=True)


def _master_csv(cfg: Dict, dataset: str) -> str:
    return cfg["data"]["chexpert_master_csv"] if dataset == "chexpert" else cfg["data"]["mimic_master_csv"]


def _cache_params(cfg: Dict, dataset: str) -> Dict:
    r = cfg["raddino"]
    return {"hf_id": r["hf_id"], "train_splits": list(r["train_splits"]), "source_resolution": int(cfg["data"]["resolution"]),
            "master_list": fingerprint_file(_master_csv(cfg, dataset)), "embed_dim": EMBED_DIM}


def cache_path(cfg: Dict, dataset: str) -> str:
    return os.path.join(cfg["raddino"]["cache_dir"], dataset, f"{dataset}_features.npz")


def heads_path(cfg: Dict, dataset: str) -> str:
    return os.path.join(cfg["raddino"]["heads_dir"], f"{dataset}_heads.npz")


class Backbone:

    def __init__(self, cfg: Dict):
        import torch
        from transformers import AutoModel, AutoProcessor
        hf_id = cfg["raddino"]["hf_id"]
        try:
            self.processor = AutoProcessor.from_pretrained(hf_id, local_files_only=True)
            self.model = AutoModel.from_pretrained(hf_id, local_files_only=True)
        except Exception:
            self.processor = AutoProcessor.from_pretrained(hf_id)
            self.model = AutoModel.from_pretrained(hf_id)
        want = cfg["raddino"]["device"]
        self.device = torch.device(want if (want == "cpu" or torch.cuda.is_available()) else "cpu")
        self.model = self.model.to(self.device).eval()
        if self.device.type == "cuda":
            self.model = self.model.half()
        print(f"[raddino] backbone {hf_id} on {self.device}", flush=True)

    def embed(self, images: List) -> np.ndarray:
        import torch
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        if self.device.type == "cuda":
            inputs = {k: (v.half() if v.dtype == torch.float32 else v) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model(**inputs)
        cls = out.last_hidden_state[:, 0, :].float().cpu().numpy()
        if cls.shape[1] != EMBED_DIM:
            raise ValueError(f"RAD-DINO returned width {cls.shape[1]}, declared {EMBED_DIM}")
        return cls

    def embed_paths(self, paths: List[str], batch_size: int, desc: str) -> np.ndarray:
        import gc
        import torch
        from tqdm import tqdm
        out = np.empty((len(paths), EMBED_DIM), dtype=np.float32)
        i, bs = 0, max(1, int(batch_size))
        bar = tqdm(total=len(paths), desc=desc, unit="img")
        while i < len(paths):
            chunk = paths[i:i + bs]
            oom = False
            try:
                out[i:i + len(chunk)] = self.embed([load_rgb(p) for p in chunk])
            except torch.OutOfMemoryError:
                oom = True
            if oom:
                gc.collect()
                torch.cuda.empty_cache()
                if bs == 1:
                    raise RuntimeError("RAD-DINO out of memory at batch size 1")
                bs = max(1, bs // 2)
                print(f"[raddino] out of memory, batch size now {bs}", flush=True)
                continue
            i += len(chunk)
            bar.update(len(chunk))
        bar.close()
        return out


def _alignment_ok(cfg: Dict, backbone: Backbone, features: np.ndarray, keys: List[str], paths: List[str]) -> Tuple[bool, float]:
    r = cfg["raddino"]
    n = min(int(r["alignment_sample"]), len(paths))
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(len(paths), size=n, replace=False))
    fresh = backbone.embed_paths([paths[i] for i in idx], int(r["feature_batch_size"]), "[raddino] alignment sample")
    cached = features[idx]
    cos = (fresh * cached).sum(1) / (np.linalg.norm(fresh, axis=1) * np.linalg.norm(cached, axis=1) + 1e-12)
    share = float((cos >= float(r["alignment_min_cosine"])).mean())
    return share >= 0.95, share


def main_raddino_cache(cfg_path: str, dataset: str) -> None:
    cfg = read_config(cfg_path)["CausalAudit"]
    require_packages(["torch", "transformers", "PIL"], f"raddino_cache {dataset}")
    r = cfg["raddino"]
    path = cache_path(cfg, dataset)
    params = _cache_params(cfg, dataset)
    rows = _training_rows(cfg, dataset)
    keys = rows["jpg_rel_path"].astype(str).tolist()
    status = status_path(cfg, "raddino_cache")
    if os.path.exists(path) and check_build_params(path, params, "raddino_cache"):
        with np.load(path, allow_pickle=False) as z:
            same = z["case_key"].shape[0] == len(keys) and np.array_equal(z["case_key"].astype(str), np.array(keys))
        if same:
            print(f"[raddino_cache] {dataset}: cache is current ({len(keys)} rows), skipping.", flush=True)
            return
        print(f"[raddino_cache] {dataset}: the cached rows differ from the current master list, rebuilding.", flush=True)
    paths = [resolve_image_path(cfg, dataset, k, int(cfg["data"]["resolution"])) for k in keys]
    if not os.path.exists(paths[0]):
        raise MissingInput(f"the first image does not resolve: {paths[0]}; check the machine block of config.yaml")
    print(f"[raddino_cache] {dataset}: {len(keys)} images; the merged cache holds "
          f"{len(keys) * EMBED_DIM * 4 / 1024 ** 3:.2f} GiB of float32.", flush=True)
    backbone = Backbone(cfg)
    legacy = r["legacy_chexpert_features"] if dataset == "chexpert" else r["legacy_mimic_features"]
    features = None
    if os.path.exists(legacy):
        with np.load(legacy, allow_pickle=True) as z:
            old = z["features"]
            if old.shape[0] == len(keys):
                ok, share = _alignment_ok(cfg, backbone, old, keys, paths)
                print(f"[raddino_cache] {dataset}: legacy cache has the right row count; alignment sample agrees on "
                      f"{share:.0%} of rows, {'adopted' if ok else 'rejected'}.", flush=True)
                if ok:
                    features = old.astype(np.float32)
                    params["adopted_legacy_cache"] = fingerprint_file(legacy)
                    params["alignment_share"] = share
            else:
                print(f"[raddino_cache] {dataset}: legacy cache has {old.shape[0]} rows against {len(keys)} current, rebuilding.", flush=True)
    if features is None:
        shard_dir = ensure_dir(os.path.join(os.path.dirname(path), "shards"))
        shard = int(r["shard_size"])
        n_shards = (len(keys) + shard - 1) // shard
        def _shard_done(sp, lo, hi):
            if not os.path.exists(sp):
                return False
            with np.load(sp, allow_pickle=False) as z:
                return z["features"].shape == (hi - lo, EMBED_DIM) and np.array_equal(z["case_key"].astype(str), np.array(keys[lo:hi]))

        for s in range(n_shards):
            sp = os.path.join(shard_dir, f"shard_{s:04d}.npz")
            lo, hi = s * shard, min(len(keys), (s + 1) * shard)
            if _shard_done(sp, lo, hi):
                continue
            if not claim_unit(shard_dir, f"shard_{s:04d}"):
                print(f"[raddino_cache] {dataset}: shard {s + 1} is claimed by a live job, skipping it.", flush=True)
                continue
            try:
                if _shard_done(sp, lo, hi):
                    continue
                feats = backbone.embed_paths(paths[lo:hi], int(r["feature_batch_size"]), f"[raddino_cache] {dataset} shard {s + 1}/{n_shards}")
                if not np.isfinite(feats).all():
                    raise RuntimeError(f"non-finite features in shard {s}")
                write_npz_atomic(sp, features=feats, case_key=np.array(keys[lo:hi]))
                append_status(status, f"{dataset}: shard {s + 1}/{n_shards} written")
            finally:
                release_claim(shard_dir, f"shard_{s:04d}")
        missing = [s for s in range(n_shards) if not _shard_done(os.path.join(shard_dir, f"shard_{s:04d}.npz"), s * shard, min(len(keys), (s + 1) * shard))]
        if missing:
            print(f"[raddino_cache] {dataset}: {len(missing)} shard(s) are still owned by another job; rerun this line once they finish.", flush=True)
            return
        features = np.empty((len(keys), EMBED_DIM), dtype=np.float32)
        for s in range(n_shards):
            lo, hi = s * shard, min(len(keys), (s + 1) * shard)
            with np.load(os.path.join(shard_dir, f"shard_{s:04d}.npz"), allow_pickle=False) as z:
                features[lo:hi] = z["features"]
    write_npz_atomic(path, features=features, case_key=np.array(keys))
    write_build_params(path, params)
    append_status(status, f"{dataset}: merged cache written, {len(keys)} rows")
    print(f"[raddino_cache] {dataset}: wrote {path}", flush=True)


def _head_labels(rows: pd.DataFrame, finding: str, cfg: Dict) -> Tuple[np.ndarray, np.ndarray, str]:
    if finding == "no_finding":
        pos, neg = abnormal_pool(rows, cfg), normal_pool(rows, cfg)
        rule = "positive: any pathology finding equal to 1; negative: no_finding equal to 1"
    else:
        pos, neg = present_pool(rows, finding, cfg), absent_pool(rows, finding, cfg)
        codes = sorted(set(neg[finding].astype(int).tolist()))
        rule = f"positive: value 1; negative: value(s) {codes}"
    idx = np.concatenate([pos.index.values, neg.index.values])
    y = np.concatenate([np.ones(len(pos), dtype=np.int8), np.zeros(len(neg), dtype=np.int8)])
    return idx, y, rule


def main_raddino_heads(cfg_path: str) -> None:
    require_packages(["sklearn"], "raddino_heads")
    from sklearn.linear_model import LogisticRegression
    cfg = read_config(cfg_path)["CausalAudit"]
    r = cfg["raddino"]
    for dataset in ("mimic", "chexpert"):
        cache = cache_path(cfg, dataset)
        if not os.path.exists(cache):
            raise MissingInput(f"{cache} does not exist; run main_raddino_cache for {dataset} first")
        out = heads_path(cfg, dataset)
        params = {"cache": fingerprint_file(cache), "C": float(r["regularization_C"]), "max_iter": int(r["max_iter"]),
                  "fit_seed": int(r["fit_seed"]), "findings": list(cfg["data"]["all_findings"])}
        if os.path.exists(out) and check_build_params(out, params, "raddino_heads"):
            print(f"[raddino_heads] {dataset}: heads are current, skipping.", flush=True)
            continue
        if not claim_unit(ensure_dir(r["heads_dir"]), f"heads_{dataset}"):
            print(f"[raddino_heads] {dataset}: claimed by a live job, leaving it alone.", flush=True)
            continue
        try:
            _fit_heads(cfg, dataset, cache, out, params, LogisticRegression)
        finally:
            release_claim(r["heads_dir"], f"heads_{dataset}")


def _fit_heads(cfg: Dict, dataset: str, cache: str, out: str, params: Dict, LogisticRegression) -> None:
    r = cfg["raddino"]
    rows = _training_rows(cfg, dataset)
    with np.load(cache, allow_pickle=False) as z:
        keys, features = z["case_key"].astype(str), z["features"]
    if not np.array_equal(keys, rows["jpg_rel_path"].astype(str).values):
        raise RuntimeError(f"{cache} rows do not match the current master list; rerun main_raddino_cache")
    mean = features.mean(0)
    scale = features.std(0) + 1e-8
    x_all = (features - mean) / scale
    arrays: Dict[str, np.ndarray] = {"scaler_mean": mean.astype(np.float32), "scaler_scale": scale.astype(np.float32)}
    rules = {}
    from tqdm import tqdm
    for finding in tqdm(cfg["data"]["all_findings"], desc=f"[raddino_heads] {dataset}", unit="head"):
        idx, y, rule = _head_labels(rows, finding, cfg)
        if len(np.unique(y)) < 2 or len(y) == 0:
            print(f"[raddino_heads] {dataset}: {finding} has a single class, no head.", flush=True)
            continue
        clf = LogisticRegression(C=float(r["regularization_C"]), max_iter=int(r["max_iter"]), solver="lbfgs",
                                 random_state=int(r["fit_seed"]))
        clf.fit(x_all[idx], y)
        arrays[f"coef_{finding}"] = clf.coef_[0].astype(np.float32)
        arrays[f"intercept_{finding}"] = np.array([float(clf.intercept_[0])], dtype=np.float32)
        arrays[f"n_{finding}"] = np.array([len(y)])
        arrays[f"pos_rate_{finding}"] = np.array([float(y.mean())])
        rules[finding] = rule
        print(f"[raddino_heads] {dataset}: {finding}: n {len(y)}, positive share {y.mean():.3f}, "
              f"training accuracy {clf.score(x_all[idx], y):.3f}", flush=True)
    write_npz_atomic(out, **arrays)
    params["label_rules"] = rules
    params["heads"] = sorted(rules)
    write_build_params(out, params)
    append_status(status_path(cfg, "raddino_heads"), f"{dataset}: {len(rules)} heads written")

class RadDinoAnswerer:

    def __init__(self, cfg: Dict, dataset: str):
        path = heads_path(cfg, dataset)
        if not os.path.exists(path):
            raise MissingInput(f"{path} does not exist; run main_raddino_heads first")
        with np.load(path, allow_pickle=False) as z:
            self.arrays = {k: z[k] for k in z.files}
        self.heads = sorted(k[len("coef_"):] for k in self.arrays if k.startswith("coef_"))
        digest = hashlib.blake2b(digest_size=8)
        for k in sorted(self.arrays):
            digest.update(k.encode())
            digest.update(np.ascontiguousarray(self.arrays[k]).tobytes())
        self.fingerprint = digest.hexdigest()
        params = read_build_params(path) or {}
        self.label_rules = params.get("label_rules", {})
        self.cfg = cfg
        self.backbone: Optional[Backbone] = None

    def probabilities(self, images: List, findings: List[str]) -> List[Optional[float]]:
        if self.backbone is None:
            self.backbone = Backbone(self.cfg)
        feats = self.backbone.embed(images)
        x = (feats - self.arrays["scaler_mean"]) / self.arrays["scaler_scale"]
        out: List[Optional[float]] = []
        for i, f in enumerate(findings):
            if f not in self.heads:
                out.append(None)
                continue
            z = float(x[i] @ self.arrays[f"coef_{f}"] + self.arrays[f"intercept_{f}"][0])
            out.append(float(1.0 / (1.0 + np.exp(-z))))
        return out

