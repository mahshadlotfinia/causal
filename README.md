# Vision-language models for chest radiography do not always need the image

Code for the study "Vision-language models for chest radiography do not always need the image" ([arXiv:2606.17710](https://arxiv.org/abs/2606.17710)).

## The audit

A vision-language model can answer a question about a chest radiograph without using the radiograph. The name of the finding in the question, and how often that finding co-occurs with other words in the training corpus, often decide the answer on their own. Accuracy on a benchmark does not separate a model that reads the image from one that does not, because both can be right for different reasons.

This pipeline separates them by changing the image and measuring what happens to the answer. Each case pairs one radiograph with one yes-or-no question about a finding. Every system answers that question under the original radiograph and under a fixed set of interventions. The radiograph is swapped for another patient's with the same label, swapped for one with the opposite label, occluded at the radiologist-marked region, occluded at a region of the same size elsewhere, and replaced by nothing, by a fixed Gaussian-noise image, or by a photograph. For each system the pipeline computes how often a correct answer changes under each intervention. It repeats the comparison on a second dataset, varies the prompt phrasing and the input resolution, and compares the systems with radiologists who read the same displays.

The panel is open weight throughout: general-purpose and medical multimodal models, text-only language models that receive no image and serve as controls, and a vision-only reference, one logistic-regression head per finding on frozen RAD-DINO features.

## Layout

- `main_causal.py` is the controller. Each stage is a `main_*` function with a lazy import, so a stage loads only the packages it needs. Call the stage you want and pass it the path to the configuration file.
- `config/config.yaml` sets the machine block, the roots, the credentials, and every key the code reads. Moving to another machine takes two edits: the machine block at the top of the configuration, and `GLOBAL_CONFIG_PATH` in the controller.
- `data_loader/` builds the question manifests and the intervention assets, and serves the cases to the models.
- `models/` caches the frozen image features and fits the per-finding heads of the vision-only reference.
- `Inference/` contains the model wrappers, the resumable runner, the answer parser, and the shared statistics.
- `analysis/` computes the metrics, the paired comparisons, the equivalence tests, and the sensitivity analyses.

## Installing and running

1. The model runs and the vision-only reference need PyTorch, transformers, and scikit-learn. The analyses and the figures need NumPy, SciPy, pandas, statsmodels, and Matplotlib.
2. Set the machine block and `GLOBAL_CONFIG_PATH`, and put the serving credentials either in the configuration or in the environment variables the configuration names.
3. Call the stages in dependency order. The manifests and the intervention assets come first, then the feature cache and the heads, then the model runs, then the metrics, the comparisons, the reanalyses, and the aggregation. Each stage prints what it depends on when it cannot find an input.
4. Every stage saves its work continuously and skips a unit it has already finished. Before it reuses an artifact it compares the build parameters stored inside that artifact against the current ones, so an interruption costs one unit of work and a changed parameter forces a named rebuild.
5. The feature cache and the heads need a GPU. The served systems need the serving system with the named model loaded. The analysis stages need only a CPU.

## Data

The study uses MIMIC-CXR, MS-CXR, ReXErr-v1, MIMIC-IV, and CheXpert. Each is released under its own access terms, and none of them is redistributed here. The frozen manifests that pin the exact cases are also not in this repository, because they contain identifiers and report text from credentialed datasets. Obtain each dataset through the route named in the paper's data availability statement, then build the manifests with the `data_loader` stages.

## Checks

`tests/test_fixture_pipeline.py` builds a small synthetic corpus, runs every CPU stage on it with a stub model and stub readers, and asserts that a second run of each stage reproduces its output byte for byte.

## Citation

```bibtex
@article{lotfinia2026causal,
  title   = {Vision-language models for chest radiography do not always need the image},
  author  = {Lotfinia, Mahshad and Ziegelmayer, Sebastian and Adams, Lisa and Nguyen, Tri-Thien and Truhn, Daniel and Maier, Andreas and Tayebi Arasteh, Soroosh},
  journal = {arXiv preprint arXiv:2606.17710},
  year    = {2026}
}
```

## License

MIT. See `LICENSE`.
