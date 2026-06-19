# Vision-language models for chest radiography do not always need the image

## Overview

This is the official repository of the paper **Vision-language models for chest radiography do not always need the image**.

Preprint version: [https://arxiv.org/abs/2606.17710](https://arxiv.org/abs/2606.17710).

Medical vision-language models (VLMs) are increasingly reported to match or exceed clinician accuracy on chest X-ray question answering. Accuracy alone, however, does not establish that a model reaches its answer by reading the image. This repository implements a causal audit that separates being correct from looking at the image. We probe nine models with controlled image interventions and measure whether a correct answer actually depends on the visual evidence it claims to use.

The audit is built around three image interventions applied to the same finding-presence question, and a small family of metrics derived from how a model's answer responds to those interventions:

- **Causal Grounding Rate (CGR)**: among cases a model gets right on the original image, the fraction whose answer flips when the bounding box of the named finding is masked. High CGR means the answer causally depended on the relevant region.
- **Answer Invariance Rate (UAR)**: the fraction of correct answers that stay unchanged when the image is swapped for a different patient with the same label. UAR is interpreted jointly with CGR and accuracy, never alone.
- **Irrelevant-mask stability**: a negative control measuring whether masking an unrelated region leaves the answer intact. Low values flag answer instability rather than grounding.

The pipeline runs the full panel of models across two datasets (MIMIC-based probe set and CheXpert), under four conditions (original, swap, target-region mask, irrelevant-region mask), and produces bootstrap confidence intervals, paired model comparisons with multiplicity control, subgroup tests, and robustness checks for image resolution and prompt phrasing. It also prepares and analyzes a blinded radiologist reader study that establishes a human grounding baseline.

## Key features

- **Causal intervention design**: four conditions per case (original, swap, target mask, irrelevant mask) that isolate image dependence from text priors.
- **Grounding metrics with uncertainty**: CGR, UAR, and irrelevant-mask stability, plus accuracy, sensitivity, specificity, AUROC, Brier score, and ECE, each reported as bootstrap mean, standard deviation, and 95% confidence interval (10,000 resamples).
- **Rigorous comparisons**: paired bootstrap differences between every model and the text-only baselines, all-pairs accuracy comparisons, and Benjamini-Hochberg FDR correction within each comparison family.
- **Subgroup analysis**: permutation tests for grounding differences across sex, view (PA vs AP), and age band, FDR-corrected per model.
- **Robustness checks**: resolution sensitivity (224 vs 512 px) with rank-stability analysis, and prompt sensitivity across phrasing variants with parse-rate tracking.
- **Reader study tooling**: scripts that build blinded, randomized reader packets (box validation, finding presence under interventions, failure-mode taxonomy) and analyze the returned ratings, including inter-reader agreement.

## Model panel

The audit covers medical VLMs, general-purpose VLMs, a frontier API model, text-only language models as baselines, and a vision-only reference probe:

| Model | Identifier | Type |
|---|---|---|
| LLaVA-Med-7B | microsoft/llava-med-v1.5-mistral-7b | Medical VLM |
| MedGemma-1.5-4B | google/medgemma-1.5-4b-it | Medical VLM |
| MedGemma-27B-text | google/medgemma-27b-it | Medical, text-only |
| Gemma-4-26B | google/gemma-4-26B-A4B-it | General VLM |
| Qwen3-VL-32B | Qwen/Qwen3-VL-32B-Instruct | General VLM |
| Mistral-Small-4-119B | mistralai/Mistral-Small-4-119B-2603-NVFP4 | General VLM |
| GPT-5 | gpt-5 (Azure OpenAI Service) | Frontier VLM |
| DeepSeek-R1-7B | deepseek-ai/DeepSeek-R1-Distill-Qwen-7B | Text-only reasoning baseline |
| RAD-DINO | microsoft/rad-dino | Vision-only reference (linear probe) |

Text-only and vision-only models serve as bounds: a text-only model cannot use the image at all, so its accuracy marks the level reachable from priors alone, while the vision-only probe marks what a purely visual representation achieves.

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/<your-org>/causal-grounding-cxr.git
cd causal-grounding-cxr
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .
```

The pipeline uses PyTorch, Hugging Face Transformers, scikit-learn, NumPy, SciPy, pandas, and Pillow. Open-weight models are loaded through Transformers; GPT-5 is accessed through the Azure OpenAI Service.

### 2. Configuration

All paths and run options live in a single YAML file under `config/config.yaml`. The relevant keys are:

```yaml
CausalAudit:
  manifest_csv:          /path/to/probe_set_manifest.csv
  image_root:            /path/to/MIMIC
  results_dir:           /path/to/results
  chexpert_manifest_csv: /path/to/chexpert_manifest.csv
  chexpert_results_dir:  /path/to/results_chexpert
  global_config_path:    /path/to/config/config.yaml
```

To evaluate GPT-5, set the Azure OpenAI keys in `config.yaml` (`azure_openai_endpoint`, `azure_openai_api_key`, `azure_openai_api_version`, `azure_openai_deployment`).

### 3. Data access

MIMIC-CXR, MS-CXR, ReXErr, and CheXpert are credentialed datasets and are not redistributed here. Obtain access through their respective providers (PhysioNet for MIMIC-CXR, MS-CXR, and ReXErr; the Stanford ML Group for CheXpert), then point `image_root` and the manifest paths to your local copies. The probe-set builder consumes the master lists and produces the stratified manifest used by every downstream step.

## Pipeline

The audit runs as a sequence of stages, all exposed through `main_causal.py`. Each stage reads from the shared config and writes intermediate results that the next stage consumes, so runs can be interrupted and resumed.

### Build the probe set

```bash
python -m data_loader.build_probe_set
```

This assembles the MIMIC-based manifest from MS-CXR (cases with finding bounding boxes), MIMIC-CXR (general finding-presence cases), and ReXErr (report-error cases), records the target and irrelevant mask boxes, and pairs each positive case with a same-label swap image.

### Set up the vision-only reference

```bash
python -m main_causal setup_raddino_probe            # MIMIC
python -m main_causal setup_raddino_probe_chexpert   # CheXpert
```

Fits the RAD-DINO linear probe on frozen features. Note that on CheXpert this probe is evaluated in-distribution and is therefore not directly comparable to the zero-shot VLMs; the paper treats it accordingly.

### Run inference

```bash
python -m main_causal run_model --model "Gemma-4-26B"          # all conditions, MIMIC
python -m main_causal run_chexpert_model --model "Gemma-4-26B" # CheXpert
```

Each model is queried under every applicable condition. Outputs are per-condition CSVs (`original.csv`, `swap.csv`, `target_mask.csv`, `irrelevant_mask.csv`) holding the parsed answer, the raw response, the confidence, and the ground-truth label per case.

### Compute metrics

```bash
python -m main_causal compute_metrics            # MIMIC
python -m main_causal compute_chexpert_metrics   # CheXpert
```

This produces a single wide `all_metrics.csv` per dataset containing every primary and supplementary metric with bootstrap mean, standard deviation, and 95% CI, alongside `paired_comparisons.csv` for the model-versus-baseline and all-pairs comparisons. Redundant per-table breakdowns are written to an `already_used/` subfolder for provenance.

### Robustness checks

```bash
python -m main_causal resolution_run_model --model "Gemma-4-26B"
python -m main_causal resolution_finalize
python -m main_causal prompt_sensitivity_run_model --model "Gemma-4-26B"
python -m main_causal prompt_sensitivity_metrics
```

The resolution check re-runs the masking conditions at 512 px and reports CGR at both resolutions with a paired comparison and rank stability. The prompt-sensitivity check repeats inference under alternative phrasings and tracks both accuracy and parse rate, which exposes models whose outputs become unparseable under certain prompts.


## File overview

- `data_loader/build_probe_set.py` – Assembles the stratified MIMIC probe-set manifest and mask/swap metadata.
- `data_loader/probe_set_data_loader.py` – Loads images and per-condition inputs for the MIMIC probe set.
- `data_loader/chexpert_data_loader.py` – Loads the CheXpert generalization set.
- `Inference/model_wrappers.py` – Uniform interface over the open-weight, API, and probe models.
- `Inference/inference_runner.py` – Runs a model across all conditions and persists per-condition outputs.
- `Inference/metrics.py` – Computes all primary and supplementary metrics into the wide metrics table.
- `Inference/stats_utils.py` – Bootstrap, paired bootstrap, permutation tests, and FDR correction.
- `Inference/paired_comparisons.py` – Model-versus-baseline and all-pairs comparisons with FDR.
- `Inference/subgroup_tests.py` – Permutation tests for grounding across sex, view, and age band.
- `main_causal.py` – Stage orchestrator for setup, inference, metrics, and robustness checks.
- `reader_study/prepare_reader_studies.py` – Builds blinded, randomized reader packets.
- `reader_study/analyze_reader_studies.py` – Analyzes returned ratings and inter-reader agreement.
- `config/config.yaml` – Central configuration for paths and run options.

## Citation

If you use this repository, please cite our paper:

```bibtex
@misc{causalgroundingcxr2026,
  title         = {Vision-language models for chest radiography do not always need the image},
  author        = {Lotfinia, Mahshad and others},
  year          = {2026},
  eprint        = {2606.17710},
  archivePrefix = {arXiv},
  primaryClass  = {cs},
  doi           = {10.48550/arXiv.2606.17710},
  url           = {https://arxiv.org/abs/2606.17710}
}
```


## License

MIT License. See `LICENSE` for details.