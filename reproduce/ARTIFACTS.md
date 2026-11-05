# Result data and downloadable artifacts

`results/` (tracked in git) holds the compact JSON summaries of the runs behind the paper's tables and figures.
The watermarked corpora, per-answer trial logs and LLM caches are larger and are distributed as separate
bundles. Run everything from `reproduce/`.

## Downloading the bundles

```bash
export ARTIFACTS_BASE_URL=<artifact URL>                   # until the default URL is set
scripts/download_artifacts.sh --list                       # show bundles
scripts/download_artifacts.sh corpora trial_logs_main      # selected bundles
scripts/download_artifacts.sh                              # everything (~112 MB download)
```

The script checks every tarball against an embedded SHA-256 and extracts it into `reproduce/`, so the
files land at `work/...` and `results/...` next to the tracked summaries. `ARTIFACTS_CACHE` sets the
download directory and `KEEP_TARBALLS=1` keeps the tarballs. To unpack a bundle by hand, run
`tar -xzf rentarag_<bundle>.tar.gz -C .` from `reproduce/`.

> **TODO (authors):** after uploading the bundles, set `DEFAULT_BASE_URL` in
> `scripts/download_artifacts.sh`, add the link here, and confirm the upstream data terms below.

| Bundle | Download | Unpacked | Contents | Needed by |
|---|---:|---:|---|---|
| `corpora` | 5.1 MB | 12.9 MB | `work/farad_shared_manifest_p6_d400_seed13.jsonl`, `work/farad_dirbucket_naive/` (DirBucket corpus, provider keys, retrieval embeddings), `work/farad_kgw_ward/`, `work/farad_semstamp_k8_relaxed/`, `work/farad_dirbucket_ortho/` (provider keys and embeddings) | Re-running the audits (`10`–`14`), laundering, sensitivity, T_min, cost, model robustness, LLM judge, coverage and staleness builders |
| `corpora_ext` | 5.8 MB | 10.3 MB | `work/coverage/c{25,50,75}/`, `work/staleness/` (evolved corpora + `evolved_rewrites.json`), `work/e5_sub50_manifest.jsonl`, `work/e5_llama8b/`, `work/e5_ref_gpt4omini/` | Coverage audits, staleness replay, paraphraser comparison |
| `trial_logs_main` | 43.0 MB | 190.1 MB | `*.trials*.jsonl` of the primary matrix (all schemes), the rag160 run and the mixing-stress run | Trial-log figures, laundering, sensitivity, T_min, staleness, LLM judges, model robustness |
| `trial_logs_baselines` | 34.8 MB | 158.9 MB | `*.trials*.jsonl` of `watermark_compare_nonwm_natural_full`, `..._nonwm_natural_pilot`, `..._oracle_aligned_baselines` | `fig:nonwm-critical-failure` |
| `trial_logs_coverage` | 5.0 MB | 23.2 MB | `results/coverage/c*/dirbucket_naive/*.trials.jsonl` | Re-scoring the coverage audits |
| `laundering_cache` | 4.0 MB | 10.4 MB | `results/laundering_attack{,_seeds45}/answer_cache/` | `scripts/20_laundering.sh` (API-free re-score), `scripts/21_laundering_utility.sh` |
| `appendix_aux` | 13.8 MB | 58.6 MB | `results/watermark_cost_ablation/candidates_debug.jsonl`, re-watermarked mini-corpora in `results/provider_scaling/m{12,20}/`, `results/model_robustness/part_a_minilm/`, `results/model_robustness_gtr/part_a_minilm/` | Cost ablation and LLM judge offline, re-auditing provider scaling / model robustness |

### Licences

- **Code:** see the repository `LICENSE`. It does not cover the data below.
- **FARAD-derived data:** every bundle, the text in `results/llm_judge_*/` and
  `results/laundering_attack/utility_judge_results.json`, and the test fixture
  `tests/fixtures/farad_dirbucket_naive_sample.jsonl`. The source documents come from FARAD, released with
  WARD (<https://github.com/eth-sri/ward>, MIT-licensed repository), which builds on RepLiQA
  (`ServiceNow/repliqa`, CC-BY-4.0 on Hugging Face). Paraphrases and answers were generated with
  gpt-4o-mini; the E5 corpus was also generated with Llama-3.1-8B-Instruct (Llama 3.1 Community Licence).
  Please cite WARD and RepLiQA when you use these files.
- **Watermark keys:** the `provider_keys.json` files are research keys (seed 13; seeds 42 and 43 for the
  mini-corpora in `appendix_aux`). They are public, so never reuse them in a real deployment.
- **Not distributed:**
  - The E1 real-domain corpora and trial logs. They contain text derived from Hugging Face datasets that
    list no licence (`guychuk/cyber-threat-intelligence-reports`,
    `joelniklaus/legal_case_document_summarization`) and are available on request. Their numeric
    summaries are in `results/multidomain_full/`, and `scripts/30_e1_multidomain.sh` rebuilds the corpus
    from the upstream datasets.
  - The human-study materials.
  - The raw FARAD data: get it from WARD (`scripts/00_get_farad.sh`).

## What is in `results/`

Per-run JSONs are stored compactly (re-serialised without indentation). Trial logs, answer caches and
corpora come from the bundles above. Run cards (`paper_rag_multi_run_card.json`) record each run's
commands with paths relative to `reproduce/`; `submission_repro_dirbucket/` in the E1/E2 cards
corresponds to `experiments/`.

| `results/` subdirectory | Paper item | Contents |
|---|---|---|
| `watermark_compare_global_tuned_mT20_kgwn3/` | `fig:wm-frontier-z-q2d`, `fig:null-calibration-single`, `fig:null-calibration`, `tab:postselection-null`, `tab:baseline-diagnostics`, `fig:benchmark_difficulty`, `fig:style-mixing-robustness-main`, `tab:utility-fidelity`, `tab:tmin-crossbenchmark` (FARAD rows) | Primary matrix: `{dirbucket_naive,kgw_ward,semstamp}/rag_multi_t{0-5}_seed{01-05}.json` (30 audits per scheme), `dirbucket_naive/sanity_{coop,unrel}_t*_seed{01-20}.json` (null audits), `style_sweep_seed*.json` |
| `watermark_compare_global_tuned_mT20_kgwn3_rag160/` | `fig:evidence-accumulation-main` | DirBucket, 10 seeds × 6 targets × 160 answers |
| `watermark_compare_global_mixstress_k9_u5/` | `fig:style-mixing-robustness-main` | DirBucket with top-9 retrieval from ≥ 5 providers; target 0: seeds 1–5, targets 1–5: seeds 2, 4, 5 |
| `watermark_compare_nonwm_natural_full/` | `fig:nonwm-critical-failure` | Similarity / overlap / entailment heuristics, 5 seeds |
| `watermark_compare_nonwm_natural_pilot/` | `app:nonwm-calibration` (in-text TPRs) | Same protocol, seed 1 |
| `watermark_compare_oracle_aligned_baselines/` | `tab:baseline-diagnostics` (oracle row), `app:baseline-adaptations` | Oracle-aligned DirBucket / KGW / SemStamp and heuristics |
| `direction_compare/{naive,ortho}/` | `tab:ortho-ablation` | Naive vs orthogonalised directions |
| `laundering_attack_seeds45/` | `tab:answer-variation` | Seeds 4–5, all six conditions |
| `laundering_attack/` | `tab:answer-variation`, `tab:laundering-utility` | `laundering_results.json`: Neutral×2, seeds 1–3 (`scripts/20_laundering.sh` scores all conditions of seeds 1–3 from `laundering_cache`); `utility_judge_results.json`: GPT-4o judgement of laundered answers |
| `llm_judge_gpt4o/` | `app:llm-judge` | GPT-4o judge: 200 sentence pairs, 90 answer pairs |
| `llm_judge_utility/` | – | gpt-4o-mini judge run (100 sentence + 60 answer pairs) |
| `watermark_cost_ablation/` | `tab:cost-strength` | `ablation_results.json`, `ABLATION_NOTE.md` |
| `sensitivity_sweep/` | `tab:sensitivity`, `app:sensitivity` | `sensitivity_results.json` (180 combinations), `sensitivity_extended.json` (324), `tmin_sweep_dense.json`, `SENSITIVITY_NOTE.md` |
| `provider_scaling/m{12,20}/` | `tab:marketplace-scaling` | `summary.json` per marketplace size (m=12: 4 target audits, T_min 12; m=20: 5 target audits, T_min 25) |
| `model_robustness/` | `tab:model-robustness` | `part_a_minilm/summary_a.json` (MiniLM encoder), `part_b_gpt4o/summary_b.json` (gpt-3.5-turbo generator), `part_c_qwen/summary_c.json` (Qwen2.5-1.5B generator), `robustness_results.json` (4 target audits per model) |
| `model_robustness_gtr/` | `tab:model-robustness` (GTR-T5 row) | `summary_gtr.json` |
| `multidomain_full/` | `tab:multidomain`, `app:multidomain`, `tab:tmin-crossbenchmark` (real-domain rows) | `paper_rag_multi_summary.json`, `e1_domain_analysis.json`, 30 run JSONs |
| `coverage/` | `tab:coverage` | `coverage_full_summary.json`, `c{25,50,75}/` summaries, run cards and run JSONs |
| `staleness_full/` | `tab:corpus-evolution` | `staleness_full_summary.json` and `f{00,10,25,50}/` per-seed replays |
| `e5_paraphraser/compare.json` | `tab:paraphrase-model` | Llama-3.1-8B vs gpt-4o-mini paraphraser |
| `e6_tmin/summary_cameraready.json` | `tab:tmin-crossbenchmark` | T_min sweep over the 30 FARAD and 30 real-domain audits |
| `figs/watermark_compare_global_tuned_mT20_kgwn3/` | Figure sources | `fig_wm_frontier_z_q2d_tikz.tex`, `fig_null_calibration_panel_a_only_tikz.tex`, `fig_null_calibration_tikz.tex`, `fig_style_mixing_robustness_main_tikz.tex`, `fig_benchmark_task_difficulty.pdf`, `dirbucket_utility_extra_stats.json` |
| `figs/watermark_compare_global_tuned_mT20_kgwn3_rag160/` | `fig:evidence-accumulation-main` | `fig_evidence_accumulation_main_tikz.tex` |
| `figs/watermark_compare_nonwm_natural_full/` | `fig:nonwm-critical-failure` | `fig_nonwm_critical_failure.pdf` |

`fig:pipeline-provider` and `fig:pipeline-auditor` are diagrams. `fig:utility-snippets` is a qualitative
example; `aggregate_watermark_compare_and_plot.py` writes comparable examples as
`fig_utility_snippets.{png,pdf}`.

The scripts in `scripts/` write re-runs under `results/rerun/` (`RUN_ROOT`). `scripts/19_main_figures.sh`
and the `experiments/*.py` entry points called with their default paths write into `results/`; `git diff
results/` compares the output with the shipped files and `git checkout -- results/` restores them.
