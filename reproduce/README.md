# Reproducing the paper

This directory holds the code and data behind the paper's experiments. `experiments/` is the paper code
(including the reference DirBucket implementation, `watermark_dirbucket.py`), `scripts/` has one bash
pipeline per experiment, and `results/` ships the summaries of every run. **Run all commands from `reproduce/`.**

## Setup

```bash
pip install -e "..[repro]"     # the rentarag library + plotting / transformers extras
export OPENAI_API_KEY=...      # only needed to re-run experiments (answers are generated with gpt-4o-mini)
```

Every script accepts `DRY_RUN=1` (print the exact commands, run nothing), `PYTHON=...` (interpreter) and
`RUN_ROOT=...` (where re-runs write; default `results/rerun/`). Each script's header lists its inputs,
outputs, cost and further knobs.

## 1. Main figures from the shipped results (no API, minutes)

```bash
scripts/19_main_figures.sh
```

| Paper result (Sec. 5) | Output (under `results/figs/`) |
|---|---|
| Watermark comparison, `fig:wm-frontier-z-q2d` | `watermark_compare_global_tuned_mT20_kgwn3/fig_wm_frontier_z_q2d_tikz.tex` |
| Key dependence, `fig:null-calibration-single` | `watermark_compare_global_tuned_mT20_kgwn3/fig_null_calibration_panel_a_only_tikz.tex` |
| Evidence accumulation, `fig:evidence-accumulation-main` | `watermark_compare_global_tuned_mT20_kgwn3_rag160/fig_evidence_accumulation_main_tikz.tex`; the headline numbers are printed as `first_q_median_target_z_ge_4` (10 answers) and `q_all_detected` (23 answers) |

The `.tex` files are the TikZ sources that the paper includes. The post-answer laundering table
(`tab:answer-variation`) is re-scored in step 2.

## 2. Re-run the main experiments

Download the released corpora, trial logs and laundering cache (~52 MB), then run the audits:

```bash
export ARTIFACTS_BASE_URL=<artifact URL>   # until the default URL in scripts/download_artifacts.sh is set
scripts/download_artifacts.sh corpora trial_logs_main laundering_cache

DRY_RUN=1 scripts/10_main_benchmark.sh     # inspect the commands first
scripts/10_main_benchmark.sh               # watermark comparison + key dependence: ~20k answers, ~$20
scripts/11_rag160.sh                       # evidence accumulation: 9.6k answers, ~$10
scripts/20_laundering.sh                   # laundering table: re-scores the cached answers, no API, CPU-hours
RESULTS_ROOT=results/rerun scripts/19_main_figures.sh   # figures from your runs
```

Detection embeds every answer window with `all-mpnet-base-v2`, so the audits run much faster on a GPU.
Answers are generated with sampled LLM calls, so every re-run is a fresh set of audits.

## 3. Build the benchmark from scratch

```bash
scripts/00_get_farad.sh               # FARAD benchmark (from WARD)
scripts/01_build_manifest.sh          # shared manifest: 400 documents, 6 providers
scripts/02_build_dirbucket.sh         # watermark the provider corpora with DirBucket (API, ~$10)
scripts/03_build_baseline_corpora.sh  # KGW / SemStamp baseline corpora (GPU + API)
```

## Additional experiments

`python experiments/paper_appendix_tables.py all` prints the appendix tables from the shipped results.
`scripts/19_main_figures.sh` also writes the appendix figures and tables built from the main runs
(`fig:null-calibration`, `tab:postselection-null`, `tab:baseline-diagnostics`, `fig:benchmark_difficulty`,
`fig:style-mixing-robustness-main`, `tab:utility-fidelity`); the trial-log based ones need the
`trial_logs_*` bundles. One script per remaining experiment:

| Script | Paper item |
|---|---|
| `12_mixstress.sh` | Provider-mixing stress run, `fig:style-mixing-robustness-main` |
| `13_nonwm_baselines.sh` | Non-watermark heuristics, `fig:nonwm-critical-failure` |
| `14_oracle_baselines.sh` | Oracle-aligned baselines, `tab:baseline-diagnostics` |
| `15_direction_ablation.sh` | Naive vs orthogonalised directions, `tab:ortho-ablation` |
| `21_laundering_utility.sh` | Utility of laundered answers, `tab:laundering-utility` |
| `22_llm_judge_utility.sh` | LLM-judge utility evaluation, `app:llm-judge` |
| `23_cost_ablation.sh` | Cost–strength ablation, `tab:cost-strength` |
| `24_sensitivity.sh` | Detector hyperparameter sensitivity, `tab:sensitivity` |
| `25_tmin.sh` | Support-gate calibration, `tab:tmin-crossbenchmark` |
| `26_provider_scaling.sh` | Marketplace size, `tab:marketplace-scaling` |
| `27_model_robustness.sh` | Encoders and answer generators, `tab:model-robustness` |
| `30_e1_multidomain.sh` | Real-domain providers, `tab:multidomain` |
| `31_e2_coverage.sh` | Selective watermarking, `tab:coverage` |
| `32_e3_corpus_evolution.sh` | Corpus evolution, `tab:corpus-evolution` |
| `33_e5_open_weight_paraphraser.sh` | Open-weight paraphraser, `tab:paraphrase-model` |

[`ARTIFACTS.md`](ARTIFACTS.md) lists the downloadable bundles, their licences and a map of `results/`.
