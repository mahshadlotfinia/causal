"""
reader_study/prepare_reader_studies.py
Created May 25, 2026

Prepares the four radiologist tasks (A, B/C, D) as zippable folders.

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia/
"""

import os
import random
import shutil
from typing import List, Optional

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from config.serde import read_config




RANDOM_SEED = 20260526


def _set_seed(seed: int = RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)


def _resolve_image_path(image_root: str, rel_path: str, resolution: int = 512) -> str:
    """Mirrors probe_set_data_loader._resolve_image_path for both resolutions."""
    rel = str(rel_path)
    if resolution == 512:
        rel = rel.replace("files/", "preprocessed/")
    else:
        rel = rel.replace("files/", "preprocessed224/")
    return os.path.join(image_root, rel)


def _scale_box_to_512(box: tuple) -> tuple:
    """Boxes in the manifest are in 224-space. Scale by 512/224."""
    s = 512.0 / 224.0
    return tuple(int(round(v * s)) for v in box)


def _draw_red_box(img: Image.Image, x: int, y: int, w: int, h: int,
                   line_width: int = 3) -> Image.Image:
    img = img.copy()
    draw = ImageDraw.Draw(img)
    draw.rectangle([x, y, x + w, y + h], outline=(255, 0, 0), width=line_width)
    return img


def _apply_black_mask(img: Image.Image, x: int, y: int, w: int, h: int) -> Image.Image:
    img = img.copy()
    draw = ImageDraw.Draw(img)
    draw.rectangle([x, y, x + w, y + h], fill=(0, 0, 0))
    return img


FINDING_TO_DISPLAY = {
    "atelectasis": "atelectasis",
    "cardiomegaly": "cardiomegaly",
    "consolidation": "consolidation",
    "edema": "pulmonary edema",
    "enlarged_cardiomediastinum": "enlarged cardiomediastinum",
    "fracture": "rib fracture",
    "lung_lesion": "lung lesion",
    "lung_opacity": "lung opacity",
    "pleural_effusion": "pleural effusion",
    "pleural_other": "pleural abnormality",
    "pneumonia": "pneumonia",
    "pneumothorax": "pneumothorax",
    "support_devices": "support device",
}


def _question_for(finding: str) -> str:
    display = FINDING_TO_DISPLAY.get(finding, finding.replace("_", " "))
    return f"Is {display} present in this chest X-ray?"




class TaskAPreparer:
    """
    Selects 120 MS-CXR positive cases stratified across findings (cap 15
    per finding), renders each image at 512x512 with the bounding box
    drawn in red, and produces a CSV for the radiologist to rate each box.
    """

    N_CASES        = 120
    CAP_PER_FINDING = 15

    def __init__(self, cfg_path: str, output_root: str):
        self.cfg_path    = cfg_path
        self.params      = read_config(cfg_path)
        self.cfg         = self.params["CausalAudit"]
        self.image_root  = self.cfg["image_root"]
        self.manifest    = pd.read_csv(self.cfg["manifest_csv"])
        self.output_dir  = os.path.join(output_root, "task_A_box_validation")
        self.images_dir  = os.path.join(self.output_dir, "images")

    def select_cases(self) -> pd.DataFrame:
        """MS-CXR positives with valid boxes, stratified by finding."""
        df = self.manifest[
            (self.manifest["source"] == "ms_cxr") &
            (self.manifest["label"] == 1) &
            (self.manifest["box_x"].notna())
        ].copy()
        chosen = []
        for finding, grp in df.groupby("finding"):
            grp = grp.sample(min(len(grp), self.CAP_PER_FINDING),
                              random_state=RANDOM_SEED)
            chosen.append(grp)
        all_chosen = pd.concat(chosen, ignore_index=True)
        if len(all_chosen) > self.N_CASES:
            all_chosen = all_chosen.sample(self.N_CASES, random_state=RANDOM_SEED)
        all_chosen = all_chosen.sample(frac=1, random_state=RANDOM_SEED) \
                                 .reset_index(drop=True)
        all_chosen["display_id"] = [f"A_{i+1:03d}" for i in range(len(all_chosen))]
        return all_chosen

    def render_images(self, cases: pd.DataFrame):
        os.makedirs(self.images_dir, exist_ok=True)
        for _, row in cases.iterrows():
            src = _resolve_image_path(self.image_root, row["image_path"], 512)
            img = Image.open(src).convert("RGB")
            bx, by, bw, bh = _scale_box_to_512(
                (row["box_x"], row["box_y"], row["box_w"], row["box_h"])
            )
            out = _draw_red_box(img, bx, by, bw, bh)
            out.save(os.path.join(self.images_dir, f"{row['display_id']}.png"))

    def prepare(self):
        _set_seed()
        cases = self.select_cases()
        self.render_images(cases)

        reader_csv = cases[["display_id", "finding"]].copy()
        reader_csv["finding_display"] = reader_csv["finding"].map(
            lambda f: FINDING_TO_DISPLAY.get(f, f.replace("_", " "))
        )
        reader_csv["rating"]  = ""
        reader_csv["comment"] = ""
        reader_csv.to_csv(os.path.join(self.output_dir, "cases.csv"), index=False)

        # Internal mapping for analysis (not for the reader)
        cases[["display_id", "case_id", "finding", "box_x", "box_y",
                "box_w", "box_h"]].to_csv(
            os.path.join(self.output_dir, "case_mapping.csv"), index=False
        )

        print(f"[TaskA] Prepared {len(cases)} cases in {self.output_dir}")




class TaskBPreparer:
    """
    Selects 80 MS-CXR positive cases stratified by finding (cap 12/finding)
    and by difficulty (40 "easy", 40 "hard" by top-VLM agreement on
    original-condition predictions).
    For each case, renders three image variants: original, target_mask,
    irrelevant_mask. Total: 240 PNG files with anonymized display IDs.
    Generates a session schedule with at least 5 days between repeats
    of the same underlying case.
    """

    N_CASES         = 80
    CAP_PER_FINDING = 12
    N_SESSIONS      = 3
    TOP_VLMS        = ["GPT-5", "Gemma-4-26B", "Qwen3-VL-32B", "MedGemma-1.5-4B"]

    def __init__(self, cfg_path: str, output_root: str, task_name: str = "task_B_finding_presence"):
        self.cfg_path    = cfg_path
        self.params      = read_config(cfg_path)
        self.cfg         = self.params["CausalAudit"]
        self.image_root  = self.cfg["image_root"]
        self.results_dir = self.cfg["results_dir"]
        self.manifest    = pd.read_csv(self.cfg["manifest_csv"])
        self.output_dir  = os.path.join(output_root, task_name)
        self.images_dir  = os.path.join(self.output_dir, "images")

    def _correctness_on_original(self, model_name: str) -> Optional[pd.Series]:
        path = os.path.join(self.results_dir, model_name, "original.csv")
        if not os.path.exists(path):
            print(f"[TaskB] Missing {path}; skipping {model_name} for difficulty stratification.")
            return None
        df = pd.read_csv(path)
        df = df[df["parsed_answer"].isin([0, 1])]
        df["correct"] = (df["parsed_answer"] == df["ground_truth"]).astype(int)
        return df.set_index("case_id")["correct"]

    def select_cases(self) -> pd.DataFrame:
        # Eligible pool: MS-CXR positives with both target and irrelevant boxes
        pool = self.manifest[
            (self.manifest["source"] == "ms_cxr") &
            (self.manifest["label"] == 1) &
            (self.manifest["box_x"].notna()) &
            (self.manifest["irrelevant_box_x"].notna())
        ].copy()

        # Compute number-of-top-VLMs-correct per case
        correctness_dict = {}
        for m in self.TOP_VLMS:
            s = self._correctness_on_original(m)
            if s is not None:
                correctness_dict[m] = s
        if not correctness_dict:
            raise RuntimeError("No top-VLM results found for difficulty stratification.")
        correctness_df = pd.DataFrame(correctness_dict)
        # Cases present in pool but missing some models get NaN -> treat as 0
        pool = pool.set_index("case_id")
        for m in correctness_df.columns:
            pool[f"correct_{m}"] = correctness_df[m]
        pool = pool.reset_index()
        correctness_cols = [c for c in pool.columns if c.startswith("correct_")]
        pool["n_correct_top"] = pool[correctness_cols].fillna(0).sum(axis=1)
        n_models_available = len(correctness_cols)

        # Difficulty bins (using available top models)
        easy_threshold = 0.75 * n_models_available  # ≥ ~3 of 4
        hard_threshold = 0.25 * n_models_available  # ≤ ~1 of 4
        pool["difficulty"] = "medium"
        pool.loc[pool["n_correct_top"] >= easy_threshold, "difficulty"] = "easy"
        pool.loc[pool["n_correct_top"] <= hard_threshold, "difficulty"] = "hard"

        # Stratified sample: 40 easy + 40 hard, capped per finding
        def _stratified_subset(diff_label: str, n_target: int) -> pd.DataFrame:
            cand = pool[pool["difficulty"] == diff_label]
            chosen = []
            remaining = n_target
            findings = sorted(cand["finding"].dropna().unique())
            # Round-robin sampling within finding caps, until we hit n_target
            cand_by_finding = {f: cand[cand["finding"] == f].sample(
                                 frac=1, random_state=RANDOM_SEED).copy()
                                for f in findings}
            taken_per_finding = {f: 0 for f in findings}
            while remaining > 0 and any(
                len(cand_by_finding[f]) > taken_per_finding[f] and
                taken_per_finding[f] < self.CAP_PER_FINDING
                for f in findings
            ):
                for f in findings:
                    if remaining == 0:
                        break
                    if (taken_per_finding[f] < self.CAP_PER_FINDING and
                        taken_per_finding[f] < len(cand_by_finding[f])):
                        chosen.append(cand_by_finding[f].iloc[taken_per_finding[f]])
                        taken_per_finding[f] += 1
                        remaining -= 1
            return pd.DataFrame(chosen) if chosen else pd.DataFrame()

        easy = _stratified_subset("easy", self.N_CASES // 2)
        hard = _stratified_subset("hard", self.N_CASES // 2)

        # If easy or hard came up short, fill from medium
        chosen = pd.concat([easy, hard], ignore_index=True)
        shortfall = self.N_CASES - len(chosen)
        if shortfall > 0:
            print(f"[TaskB] Easy+hard yielded {len(chosen)}; filling {shortfall} from medium.")
            medium_pool = pool[
                (pool["difficulty"] == "medium") &
                (~pool["case_id"].isin(chosen["case_id"]))
            ].sample(min(shortfall, len(pool)), random_state=RANDOM_SEED)
            chosen = pd.concat([chosen, medium_pool], ignore_index=True)

        chosen = chosen.head(self.N_CASES).reset_index(drop=True)
        chosen["internal_case_idx"] = chosen.index
        return chosen

    def _build_displays(self, cases: pd.DataFrame) -> pd.DataFrame:
        """Expand 80 cases into 240 displays with anonymized IDs."""
        rng = np.random.RandomState(RANDOM_SEED)
        rows = []
        for _, c in cases.iterrows():
            for cond in ["original", "target_mask", "irrelevant_mask"]:
                rows.append({"case_id": c["case_id"], "condition": cond,
                              "finding": c["finding"], "internal_case_idx": c["internal_case_idx"]})
        displays = pd.DataFrame(rows)
        # Shuffle then assign anonymized display_ids
        displays = displays.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
        displays["display_id"] = [f"B_{i+1:03d}" for i in range(len(displays))]
        return displays

    def _assign_sessions(self, displays: pd.DataFrame) -> pd.DataFrame:
        """
        Each case appears 3 times (one per condition). Assign one of each
        case's 3 displays to each session, then randomize the within-session
        display order.
        """
        rng = np.random.RandomState(RANDOM_SEED + 1)
        sessions = np.zeros(len(displays), dtype=int)
        for case_id, grp in displays.groupby("case_id"):
            idxs = list(grp.index)
            assert len(idxs) == 3, f"Case {case_id} does not have 3 conditions"
            order = list(range(1, 4))
            rng.shuffle(order)
            for i, idx in zip(order, idxs):
                sessions[idx] = i
        displays["session"] = sessions
        # Final reorder: sort by session, then random within session, and renumber display_id
        rng2 = np.random.RandomState(RANDOM_SEED + 2)
        sorted_rows = []
        for s in [1, 2, 3]:
            block = displays[displays["session"] == s].sample(
                frac=1, random_state=rng2.randint(0, 1_000_000)
            )
            sorted_rows.append(block)
        displays = pd.concat(sorted_rows, ignore_index=True)
        displays["display_id"] = [f"B_{i+1:03d}" for i in range(len(displays))]
        return displays

    def render_images(self, displays: pd.DataFrame, cases: pd.DataFrame):
        os.makedirs(self.images_dir, exist_ok=True)
        cases_indexed = cases.set_index("case_id")
        for _, d in displays.iterrows():
            row = cases_indexed.loc[d["case_id"]]
            src = _resolve_image_path(self.image_root, row["image_path"], 512)
            img = Image.open(src).convert("RGB")

            if d["condition"] == "original":
                out = img
            elif d["condition"] == "target_mask":
                bx, by, bw, bh = _scale_box_to_512(
                    (row["box_x"], row["box_y"], row["box_w"], row["box_h"])
                )
                out = _apply_black_mask(img, bx, by, bw, bh)
            elif d["condition"] == "irrelevant_mask":
                bx, by, bw, bh = _scale_box_to_512(
                    (row["irrelevant_box_x"], row["irrelevant_box_y"],
                      row["irrelevant_box_w"], row["irrelevant_box_h"])
                )
                out = _apply_black_mask(img, bx, by, bw, bh)
            else:
                raise ValueError(f"Unknown condition {d['condition']}")

            out.save(os.path.join(self.images_dir, f"{d['display_id']}.png"))

    def prepare(self):
        _set_seed()
        cases    = self.select_cases()
        displays = self._build_displays(cases)
        displays = self._assign_sessions(displays)
        self.render_images(displays, cases)

        # Reader-facing CSV
        reader_csv = displays[["display_id", "session", "finding"]].copy()
        reader_csv["question"]   = reader_csv["finding"].map(_question_for)
        reader_csv = reader_csv.drop(columns=["finding"])
        reader_csv["answer"]     = ""   # Yes / No
        reader_csv["confidence"] = ""   # 1 - 5
        reader_csv["comment"]    = ""
        reader_csv = reader_csv.sort_values(["session", "display_id"]).reset_index(drop=True)
        reader_csv.to_csv(os.path.join(self.output_dir, "cases.csv"), index=False)

        # Internal mapping (NOT given to the reader)
        displays[["display_id", "session", "case_id", "condition",
                   "finding"]].to_csv(
            os.path.join(self.output_dir, "case_mapping.csv"), index=False
        )
        displays[["display_id", "session"]].to_csv(
            os.path.join(self.output_dir, "session_assignment.csv"), index=False
        )

        # Selection diagnostics (which cases came in via easy/hard/medium)
        if "difficulty" in cases.columns:
            cases[["case_id", "finding", "difficulty", "n_correct_top"]].to_csv(
                os.path.join(self.output_dir, "case_selection_diagnostics.csv"),
                index=False
            )

        print(f"[TaskB] Prepared {len(cases)} cases, {len(displays)} displays in {self.output_dir}")



class TaskDPreparer:
    """
    Selects confident-wrong predictions from GPT-5, Gemma-4-26B, and
    RAD-DINO on the MIMIC original.csv. Confidence threshold defined per
    model via the top quartile of its confidence-on-wrong cases (handles
    the different confidence distributions: API logprob vs RAD-DINO probe).
    Final count: 17/17/16 (= 50). Model identity is masked to A/B/C.
    """

    N_TOTAL = 50
    MODELS_TASK_D = ["GPT-5", "Gemma-4-26B", "RAD-DINO"]
    CAP_PER_FINDING = 8

    def __init__(self, cfg_path: str, output_root: str):
        self.cfg_path    = cfg_path
        self.params      = read_config(cfg_path)
        self.cfg         = self.params["CausalAudit"]
        self.image_root  = self.cfg["image_root"]
        self.results_dir = self.cfg["results_dir"]
        self.manifest    = pd.read_csv(self.cfg["manifest_csv"])
        self.output_dir  = os.path.join(output_root, "task_D_failure_taxonomy")
        self.images_dir  = os.path.join(self.output_dir, "images")

    def _load_model_originals(self, model_name: str) -> Optional[pd.DataFrame]:
        path = os.path.join(self.results_dir, model_name, "original.csv")
        if not os.path.exists(path):
            print(f"[TaskD] Missing {path}; cannot include {model_name}.")
            return None
        df = pd.read_csv(path)
        df = df[df["parsed_answer"].isin([0, 1])]
        df["wrong"] = (df["parsed_answer"] != df["ground_truth"]).astype(int)
        return df

    def _top_quartile_wrong(self, df: pd.DataFrame, k: int) -> pd.DataFrame:
        """
        Select k confident-wrong cases for one model.
        Confidence threshold = 75th percentile of confidence over wrong
        cases (or directional equivalent if confidence is degenerate).
        """
        wrong = df[df["wrong"] == 1].copy()
        if len(wrong) == 0:
            return pd.DataFrame()

        # If confidence is binary (degenerate), we cannot subselect by confidence
        binary_frac = float(((wrong["confidence"] == 0.0) |
                              (wrong["confidence"] == 1.0)).mean())
        if binary_frac > 0.5:
            # All wrong cases treated equally; sample k randomly
            return wrong.sample(min(k, len(wrong)), random_state=RANDOM_SEED)

        # Otherwise, for "confidence" we want the answer-confidence not the P(Yes).
        # parsed_answer=1 -> conf used as-is; parsed_answer=0 -> use 1-conf.
        wrong["answer_conf"] = np.where(
            wrong["parsed_answer"] == 1,
            wrong["confidence"], 1.0 - wrong["confidence"]
        )
        threshold = wrong["answer_conf"].quantile(0.75)
        candidates = wrong[wrong["answer_conf"] >= threshold]
        if len(candidates) < k:
            candidates = wrong.nlargest(k, "answer_conf")
        else:
            candidates = candidates.sample(min(k, len(candidates)),
                                            random_state=RANDOM_SEED)
        return candidates

    def select_cases(self) -> pd.DataFrame:
        # Quota: 17 + 17 + 16
        quotas = {self.MODELS_TASK_D[0]: 17,
                  self.MODELS_TASK_D[1]: 17,
                  self.MODELS_TASK_D[2]: 16}

        all_chosen = []
        for model in self.MODELS_TASK_D:
            df = self._load_model_originals(model)
            if df is None:
                continue
            picked = self._top_quartile_wrong(df, quotas[model])
            picked["source_model"] = model
            all_chosen.append(picked)
        if not all_chosen:
            raise RuntimeError("[TaskD] No model results found.")
        chosen = pd.concat(all_chosen, ignore_index=True)

        # Light per-finding cap to avoid one finding dominating
        chosen["finding_from_case"] = chosen["case_id"].map(
            self.manifest.set_index("case_id")["finding"]
        )
        capped = []
        per_finding_count = {}
        chosen = chosen.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
        for _, r in chosen.iterrows():
            f = r["finding_from_case"]
            if pd.isna(f):
                capped.append(r)
                continue
            if per_finding_count.get(f, 0) < self.CAP_PER_FINDING:
                capped.append(r)
                per_finding_count[f] = per_finding_count.get(f, 0) + 1
        chosen = pd.DataFrame(capped).head(self.N_TOTAL).reset_index(drop=True)

        # Map source_model -> hidden model_label A/B/C (randomized per run)
        unique_models = sorted(chosen["source_model"].unique())
        rng = np.random.RandomState(RANDOM_SEED + 3)
        labels = list("ABC")[:len(unique_models)]
        rng.shuffle(labels)
        model_to_label = dict(zip(unique_models, labels))
        chosen["model_label"] = chosen["source_model"].map(model_to_label)

        # Final randomization for presentation order
        chosen = chosen.sample(frac=1, random_state=RANDOM_SEED + 4).reset_index(drop=True)
        chosen["display_id"] = [f"D_{i+1:03d}" for i in range(len(chosen))]
        return chosen

    def render_images(self, cases: pd.DataFrame):
        os.makedirs(self.images_dir, exist_ok=True)
        manifest_indexed = self.manifest.set_index("case_id")
        for _, row in cases.iterrows():
            cid = row["case_id"]
            if cid not in manifest_indexed.index:
                print(f"[TaskD] Warning: {cid} not in manifest; skipping.")
                continue
            mrow = manifest_indexed.loc[cid]
            src = _resolve_image_path(self.image_root, mrow["image_path"], 512)
            img = Image.open(src).convert("RGB")
            img.save(os.path.join(self.images_dir, f"{row['display_id']}.png"))

    def prepare(self):
        _set_seed()
        cases = self.select_cases()
        self.render_images(cases)

        reader_rows = []
        for _, r in cases.iterrows():
            finding = r["finding_from_case"]
            if pd.isna(finding):
                question = "Is the named finding present in this chest X-ray?"
                gt_str = ""
            else:
                question = _question_for(finding)
                gt_str   = "Yes" if int(r["ground_truth"]) == 1 else "No"
            model_answer = "Yes" if int(r["parsed_answer"]) == 1 else "No"
            reader_rows.append({
                "display_id":             r["display_id"],
                "question":               question,
                "ground_truth":           gt_str,
                "model_label":            r["model_label"],
                "model_answer":           model_answer,
                "model_confidence":       round(float(r["confidence"]), 3),
                "error_category":         "",   # Ambiguous / Image quality / Plausible confounder / Clear model failure / Other
                "radiologist_confidence": "",   # 1 - 5
                "comment":                "",
            })
        pd.DataFrame(reader_rows).to_csv(
            os.path.join(self.output_dir, "cases.csv"), index=False
        )

        # Internal mapping (NOT given to the reader)
        cases[["display_id", "case_id", "source_model", "model_label",
                "finding_from_case", "ground_truth", "parsed_answer",
                "confidence"]].to_csv(
            os.path.join(self.output_dir, "case_mapping.csv"), index=False
        )

        # Also save the model_label -> source_model decoding for the analysis stage
        pd.DataFrame([
            {"model_label": lbl, "source_model": src}
            for src, lbl in {r["source_model"]: r["model_label"]
                              for _, r in cases.iterrows()}.items()
        ]).to_csv(os.path.join(self.output_dir, "model_label_key.csv"), index=False)

        print(f"[TaskD] Prepared {len(cases)} cases in {self.output_dir}")




def prepare_all_reader_studies(cfg_path: str, output_root: str):
    """
    Runs the three preparers and prints final folder summary.
    Reader 2's Task C uses the same image set as Task B; no separate
    preparation needed (just a separate empty cases.csv copy).
    """
    os.makedirs(output_root, exist_ok=True)

    TaskAPreparer(cfg_path, output_root).prepare()
    TaskBPreparer(cfg_path, output_root, task_name="task_B_finding_presence").prepare()
    TaskDPreparer(cfg_path, output_root).prepare()

    # Task C = Task B images, separate empty rating sheet for Reader 2.
    # We do NOT recompute or copy images (Reader 2 receives the same images
    # but gets a separate, empty cases.csv to fill in independently).
    task_b_dir = os.path.join(output_root, "task_B_finding_presence")
    task_c_dir = os.path.join(output_root, "task_C_reader2_agreement")
    os.makedirs(task_c_dir, exist_ok=True)
    # Symlink images to avoid duplication; if symlinks unsupported, copy.
    images_src = os.path.join(task_b_dir, "images")
    images_dst = os.path.join(task_c_dir, "images")
    if not os.path.exists(images_dst):
        try:
            os.symlink(os.path.abspath(images_src), images_dst)
        except (OSError, NotImplementedError):
            shutil.copytree(images_src, images_dst)
    # Build empty reader CSV for Reader 2 (same display_ids, blank fields)
    cases_b = pd.read_csv(os.path.join(task_b_dir, "cases.csv"))
    cases_c = cases_b.copy()
    cases_c["answer"]     = ""
    cases_c["confidence"] = ""
    cases_c["comment"]    = ""
    cases_c.to_csv(os.path.join(task_c_dir, "cases.csv"), index=False)
    # Copy session assignment so Reader 2 follows the same session schedule
    shutil.copy(os.path.join(task_b_dir, "session_assignment.csv"),
                 os.path.join(task_c_dir, "session_assignment.csv"))
    print(f"[TaskC] Linked Task B images and created empty rating sheet in {task_c_dir}")

    print("\n=== Reader study preparation complete ===")
    print(f"Output root: {output_root}")
    print("Folders:")
    for d in sorted(os.listdir(output_root)):
        full = os.path.join(output_root, d)
        if os.path.isdir(full):
            img_dir = os.path.join(full, "images")
            n_images = len(os.listdir(img_dir)) if os.path.isdir(img_dir) else 0
            print(f"  {d}: {n_images} images")



if __name__ == "__main__":
    CONFIG_PATH = "/PATH/causal/config/config.yaml"
    OUTPUT_ROOT = "/PATH/Repositories_target_files/causal/reader_study"
    prepare_all_reader_studies(CONFIG_PATH, OUTPUT_ROOT)
