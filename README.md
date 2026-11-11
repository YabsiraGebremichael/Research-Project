# Learning When Not to Be Confident: Time-Series Foundation Models in the M6 Competition

Code and reproducibility workflow for evaluating pretrained time-series foundation models on the M6 individual forecasting task, analysing their probabilistic behaviour and testing model-specific uncertainty interventions.

BSc Honours Research Project, School of Computer Science and Applied Mathematics,
University of the Witwatersrand.

Author: Yabsira Gebremichael (2661262). Supervisor: Dr. Hairong Bau.

---

## Overview

The M6 individual forecasting task requires probability forecasts over five relative-return groups for 100 assets. This project evaluates Chronos-T5 Base, Financial Chronos and Chronos-2 against a naive forecast, feature-based classifiers and adapted competition methods. It also examines confidence and calibration before testing sampling interventions for the token-based models and low-rank adaptation for Chronos-2.

The repository is organised as a complete research workflow. It acquires and prepares market data, constructs forecast contexts and benchmark datasets, runs the required GPU notebooks, converts model outputs into M6 probabilities, scores the twelve competition rounds and produces the diagnostic analyses, statistical comparisons, tables and figures used in the study.

## Models and experiments

| Group | Model or experiment | Where it lives |
|---|---|---|
| Foundation model | Chronos-T5 Base (200M, `amazon/chronos-t5-base`) | [`TSFM Inference and Sampling/`](TSFM%20Inference%20and%20Sampling/) |
| Foundation model | Financial Chronos (46.2M, `FinText/Chronos_Small_2021_Global`) | [`TSFM Inference and Sampling/`](TSFM%20Inference%20and%20Sampling/) |
| Foundation model | Chronos-2 (120M, `amazon/chronos-2`) | [`TSFM Inference and Sampling/`](TSFM%20Inference%20and%20Sampling/) |
| Benchmark | Naive equal-probability forecast | [`scripts/evaluate_m6_rps.py`](scripts/evaluate_m6_rps.py) |
| Feature baseline | Random Forest | [`Feature Baselines/`](Feature%20Baselines/) |
| Feature baseline | LightGBM | [`Feature Baselines/`](Feature%20Baselines/) |
| Feature baseline | Softmax regression | [`Feature Baselines/softmax_feature_baseline.py`](Feature%20Baselines/softmax_feature_baseline.py) |
| Competition entry | FinQBoost (adapted reimplementation) | [`FinQ Boost Reimplementation/`](FinQ%20Boost%20Reimplementation/) |
| Competition entry | Quasi-Average (adapted reimplementation) | [`Quasi-Average Reimplementation DRE-as-Stock/`](Quasi-Average%20Reimplementation%20DRE-as-Stock/) |
| Intervention | Inference-time temperature and top-k sampling | [`TSFM Uncertainty Experiment/`](TSFM%20Uncertainty%20Experiment/) |
| Intervention | Ordinary Chronos-2 LoRA | [`TSFM Uncertainty Experiment/`](TSFM%20Uncertainty%20Experiment/) |
| Intervention | Uncertainty-regularised Chronos-2 LoRA | [`TSFM Uncertainty Experiment/`](TSFM%20Uncertainty%20Experiment/) |

The final settings used for every model are encoded directly in the scripts and notebooks that run it: the search spaces and selection rules in the feature-baseline tuning script, the sampling grids and selected temperature and top-k values in the intervention notebooks, and the LoRA rank, alpha, loss and penalty weights in the fine-tuning notebooks. The complete experimental settings, including the reasoning behind them, are reported in the research report, which is not part of this repository.

## Data sources

Two historical panels support the work, with a longer third panel used only for Chronos-2 fine-tuning. The names Dataset A and Dataset D are retained throughout the code.

| Name | Source | Range | Role |
|---|---|---|---|
| Dataset A | Yahoo Finance, with EODHD recovering DRE, RE and WRK | 2020-01-02 to 2023-02-03, 807 weekdays | Adjusted closes behind the 512-weekday foundation-model contexts |
| Dataset D | EODHD | 2009-01-02 to 2023-02-03, 3,676 weekdays | Shared panel for the five data-driven benchmarks |
| Extended EODHD history | EODHD | 2000-01-03 to 2023-02-03, 6,025 weekdays | Chronos-2 LoRA fine-tuning only |

The feature-based baselines read daily log returns, close, adjusted close and volume. Processed open, high and low panels are not required and are not written by `preprocess_dataset_d.py`, although those fields are still read from the raw file and validated during preprocessing.

## What is and is not included

**Included.** All source code is in the repository: 35 Python scripts and 18 notebooks covering data acquisition, preprocessing, context construction, model inference, fine-tuning, scoring, diagnostics, statistics and visualisation. The genuine static inputs are included too:

- [`Data/raw/m6_official/assets_m6.csv`](Data/raw/m6_official/assets_m6.csv), the official M6 asset and price file published by the competition organisers. It is the tracked official outcome file and the sole source of realised returns and ground-truth quintiles. It is not provider data.
- [`Data/metadata/m6_asset_metadata.csv`](Data/metadata/m6_asset_metadata.csv), the 100-asset universe in official order with sector and asset-class labels.
- [`Competition Visuals/system_overview/`](Competition%20Visuals/system_overview/), the pipeline diagram in SVG and PNG.

**Not included.** Yahoo Finance and EODHD provider data are not distributed. Neither are the panels, contexts and benchmark datasets derived row by row from them, which the pipeline scripts generate locally.

Model outputs, reconstructed scenarios, LoRA adapters, predictions, result tables, statistical outputs, figures and tables are also not distributed. The workflow below documents how to regenerate them from the stated inputs and external services.

Users remain responsible for complying with the terms of the relevant data providers.

### Provider credentials

The EODHD acquisition scripts read a token from a git-ignored env file in the repository root:

```
# proj.env
EODHD_API_KEY=your_token_here
```

`EODHD_API_TOKEN` is accepted as an alternative name. The scripts never print, log or embed the token, and they abort before any request if an env file is tracked by git or is not covered by [`.gitignore`](.gitignore).

**`proj.env` and the EODHD API key must never be committed, pushed, pasted into a notebook or otherwise exposed.** `.gitignore` covers `.env`, `*.env` and `proj.env`, but that is a safety net, not a substitute for checking `git status` before committing.

## Pipeline

![Flow diagram linking market-data preparation, forecasting methods, M6 probability construction, evaluation, diagnostics and model-specific interventions.](Competition%20Visuals/system_overview/m6-system-overview.png)

The diagram maps the repository workflow from data preparation to final analysis. Local scripts build the datasets and forecast contexts, GPU notebooks produce forecasts and adapted models, and the scoring and analysis modules generate M6 probabilities, diagnostics, statistical comparisons and report-ready outputs.

## Repository structure

Everything shown below is in the repository. Directories created by a full run are listed separately under [Outputs produced by a full run](#outputs-produced-by-a-full-run).

```
.
├── scripts/
│   ├── evaluate_m6_rps.py                shared M6 scorer and ground truth
│   ├── exploratory_evaluation.py         cross-family metrics and the shared ECE definition
│   └── pipeline/
│       ├── eodhd_client.py               shared EODHD client and token handling
│       ├── download_dataset_a.py         Yahoo adjusted closes
│       ├── repair_dataset_a_eodhd.py     EODHD recovery for DRE, RE and WRK
│       ├── preprocess_dataset_a.py       weekday calendar and log returns
│       ├── download_dataset_d_eodhd.py   EODHD OHLCV
│       ├── preprocess_dataset_d.py       Dataset D weekday panels
│       ├── build_rolling_origins.py      the twelve 512-weekday contexts and the round schedule
│       └── build_validation_contexts.py  the six pre-M6 validation contexts
├── Data/
│   ├── metadata/m6_asset_metadata.csv    asset universe, official order, sector, asset class
│   └── raw/m6_official/assets_m6.csv     official M6 asset and price file
├── TSFM Inference and Sampling/
│   ├── inference_notebooks/              5 zero-shot GPU notebooks
│   ├── scoring/                          4 scenario-count scoring and aggregation scripts
│   └── chronos2_sampling_comparison/     2 Chronos-2 scenario-reconstruction methods
├── TSFM Uncertainty Experiment/
│   ├── finetuning_notebooks/             3 Chronos-2 LoRA training notebooks
│   ├── inference_notebooks/              7 sampling-validation and intervention notebooks
│   └── analysis/                         4 validation, M6 scoring and figure scripts
├── Feature Baselines/                    dataset build, tuning, evaluation, softmax baseline
├── FinQ Boost Reimplementation/          extended download, FinQBoost, seed sensitivity
├── Quasi-Average Reimplementation DRE-as-Stock/
│   └── quasi_average.py                  DRE-as-Stock reimplementation
├── Horizon Experiment/
│   ├── horizon_experiment.py             multi-horizon RPS
│   ├── horizon_diagnostics.py            per-asset horizon diagnostics
│   ├── horizon_visualisations.ipynb      working diagnostic figures
│   └── results/.gitkeep                  placeholder for the generated results
├── TSFM Calibration Experiment/          probability-shrinkage sweep
├── Statistical Analysis/                 paired tests, forest plot, ablation table, unit tests
├── Competition Visuals/
│   ├── competition_visuals.ipynb         cross-model tables and figures
│   ├── horizon_visualisations.ipynb      report-ready horizon figures
│   ├── system_overview/                  hand-authored pipeline diagram (SVG and PNG)
│   └── tables/.gitkeep                   placeholder for the generated tables
├── requirements.txt
├── .gitignore
└── README.md
```

The two `horizon_visualisations.ipynb` files are both retained because they serve different purposes, and neither is a copy of the other. `Horizon Experiment/horizon_visualisations.ipynb` is the working notebook: it reads `Horizon Experiment/results/` and writes exploratory figures into `Horizon Experiment/visualisations/`, including diagnostics that were never published. `Competition Visuals/horizon_visualisations.ipynb` is the report version: it reads the same results and writes the final horizon figures and tables into `Competition Visuals/`. Both only draw figures; every value they plot is computed by `Horizon Experiment/horizon_diagnostics.py`.

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.10 | The project was developed and run on 3.10.11. |
| EODHD API key | Stored in `proj.env` in the repository root. Needed for the Dataset A repair, Dataset D and the extended panel. |
| Network access | Yahoo Finance is reached through `yfinance`; no key is required for it. |
| Google Colab or an equivalent GPU | All 15 inference and fine-tuning notebooks assume a CUDA GPU. They are not runnable in the CPU environment. |
| Google Drive | The notebooks read contexts from, and write outputs to, `MyDrive/HonoursResearch/`. An equivalent mount works if the path variables at the top of each notebook are changed. |
| `amazon/chronos-t5-base` | Downloaded from Hugging Face at notebook runtime. |
| `FinText/Chronos_Small_2021_Global` | Downloaded from Hugging Face at notebook runtime. |
| `amazon/chronos-2` | Downloaded from Hugging Face at notebook runtime. |
| GPU packages | PyTorch, Transformers, Chronos Forecasting and PEFT are installed inside the Colab session by the notebooks themselves, not through `requirements.txt`. |

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

This installs the CPU stack only: pandas, numpy, scipy, scikit-learn, LightGBM, XGBoost, Hyperopt, matplotlib, requests, python-dotenv, yfinance and the Jupyter execution packages. It is enough for every script in the repository; no script imports PyTorch, Transformers or PEFT. The GPU stack is listed commented out at the end of `requirements.txt` for reference only.

## Reproduction

One workflow, run in order. Unless a stage says otherwise, commands run from the repository root and paths are exactly as they appear in the repository. Stages marked **GPU** run in Colab; stages marked **manual** are file transfers between the local repository and Google Drive.

The ordering matters in one place in particular: stage 4 writes `Data/metadata/m6_round_schedule.csv`, which sixteen downstream scripts read. Nothing after stage 4 will run until that file exists.

### Stage 1: credentials and environment

Create the virtual environment as above, then create `proj.env` in the repository root with your EODHD key. Nothing else is needed to start.

### Stage 2: Dataset A acquisition and preprocessing

Builds the adjusted-close panel behind the foundation-model contexts.

```bash
python scripts/pipeline/download_dataset_a.py
python scripts/pipeline/repair_dataset_a_eodhd.py
python scripts/pipeline/preprocess_dataset_a.py
```

Requires `Data/raw/m6_official/assets_m6.csv` (in the repository) for the official asset universe and order, network access, and the EODHD key for the repair step. Produces `Data/raw/yahoo/`, `Data/raw/recovered_sources/eodhd/`, the two `Data/processed/dataset_a_*.csv` panels, a preprocessing summary under `Data/metadata/diagnostics/` and markdown reports under `reports/`.

The repair step writes `dataset_a_adjusted_close_repaired.csv` alongside the original download and leaves the original untouched. Every later stage reads the repaired copy.

### Stage 3: Dataset D acquisition and preprocessing

Builds the shared benchmark panel.

```bash
python scripts/pipeline/download_dataset_d_eodhd.py
python scripts/pipeline/preprocess_dataset_d.py
```

Requires the EODHD key. Produces `Data/raw/dataset_d_eodhd/`, the four `Data/processed/dataset_d/` panels and a preprocessing summary under `Data/metadata/diagnostics/`.

### Stage 4: forecast contexts and the round schedule

Slices the fixed 512-weekday contexts and writes the official round schedule that the rest of the workflow depends on.

```bash
python scripts/pipeline/build_rolling_origins.py
python scripts/pipeline/build_validation_contexts.py
```

Requires the Dataset A processed panels from stage 2. The first script writes `Data/contexts/round_01_context.csv` through `round_12_context.csv` and **`Data/metadata/m6_round_schedule.csv`**. The second writes the six pre-M6 validation contexts and their manifest into `Data/validation_contexts/`, using the schedule as a guard so no validation target overlaps the first scored M6 round.

`build_validation_contexts.py` refuses to overwrite an existing context file. On a rerun, delete the files it names first.

### Stage 5: feature baselines

Builds the supervised rows, tunes the models and evaluates them on the twelve official rounds.

```bash
python "Feature Baselines/build_feature_baseline_dataset.py"
python "Feature Baselines/tune_feature_baselines.py"
python "Feature Baselines/evaluate_feature_baselines_m6.py"
python "Feature Baselines/softmax_feature_baseline.py"
```

Requires the Dataset D panels and the round schedule. Produces `Data/processed/feature_baseline/`, `Data/metadata/feature_baseline_origin_schedule.csv`, `Feature Baselines/tuning/` and `Feature Baselines/results/`. Tuning is the slowest CPU step in the repository.

### Stage 6: competition reimplementations

```bash
python "FinQ Boost Reimplementation/finqboost.py"
python "Quasi-Average Reimplementation DRE-as-Stock/quasi_average.py"
```

Both default to Dataset D and require the round schedule. These default settings reproduce the reported competition results. The optional `--panel extended` variant requires the extended panel generated in stage 14. `quasi_average.py` derives its own adjusted-close panel into `Quasi-Average Reimplementation DRE-as-Stock/data/`. FinQBoost writes its predictions and RPS into its own folder; Quasi-Average writes into `results/`.

### Stage 7: upload the contexts to Google Drive (manual)

The GPU notebooks read contexts from Drive, not from the repository. Copy:

| From | To |
|---|---|
| `Data/contexts/` | `MyDrive/HonoursResearch/contexts` |
| `Data/contexts/` | `MyDrive/HonoursResearch/Round_1_Context` |
| `Data/validation_contexts/` | `MyDrive/HonoursResearch/validation_contexts` |

The twelve round contexts are needed under two names. `Round_1_Context` is a historical folder name used by five notebooks in `TSFM Uncertainty Experiment/inference_notebooks/`; despite the name it holds the contexts for all twelve rounds. Either keep the name, or change `DRIVE_CONTEXT_DIR` at the top of those notebooks.

### Stage 8: zero-shot foundation-model inference (GPU)

Run these notebooks in this order. All five read the twelve shared round contexts and write raw forecasts to `MyDrive/HonoursResearch/outputs/`.

1. `TSFM Inference and Sampling/inference_notebooks/chronos_t5_base_100_samples.ipynb`: 100 sampled trajectories per asset.
2. `TSFM Inference and Sampling/inference_notebooks/chronos_t5_base_scenarios_300_500.ipynb`: the 300 and 500-scenario sensitivity runs.
3. `TSFM Inference and Sampling/inference_notebooks/financial_chronos_100_samples.ipynb`
4. `TSFM Inference and Sampling/inference_notebooks/financial_chronos_scenarios_300_500_1000.ipynb`: 300, 500 and 1000 scenarios.
5. `TSFM Inference and Sampling/inference_notebooks/chronos2_quantile_inference.ipynb`: 21 quantile levels per asset and day, not trajectories.

Seeds, context length, horizon and asset order are fixed inside the notebooks and should not be changed if the reported values are to be reproduced.

### Stage 9: download the zero-shot outputs (manual)

Copy the Drive outputs into the repository:

| From | To |
|---|---|
| `MyDrive/HonoursResearch/outputs/chronos_t5_base/` | `TSFM Inference and Sampling/inference_outputs/` |
| `MyDrive/HonoursResearch/outputs/financial_chronos/` | `TSFM Inference and Sampling/inference_outputs/` |
| `MyDrive/HonoursResearch/outputs/chronos2/quantiles/` | `TSFM Inference and Sampling/inference_outputs/` |

`inference_outputs/` holds one subdirectory per model, named as each notebook names it. `scripts/evaluate_m6_rps.py` discovers the models by scanning those subdirectories, so the names must match what the notebooks wrote.

### Stage 10: scenario reconstruction and zero-shot scoring

Chronos-2 returns marginal quantiles rather than joint trajectories, so scenarios have to be reconstructed before cross-sectional four-week returns can be ranked. Run the copula method first: the independent-sampling ablation imports its quantile interpolation and scoring so the two cannot drift apart.

```bash
python "TSFM Inference and Sampling/chronos2_sampling_comparison/evaluate_spatiotemporal_copula.py"
python "TSFM Inference and Sampling/chronos2_sampling_comparison/evaluate_independent_sampling.py"
python scripts/evaluate_m6_rps.py
python scripts/exploratory_evaluation.py
python "TSFM Inference and Sampling/scoring/score_chronos_t5_base_scenarios.py"
python "TSFM Inference and Sampling/scoring/score_financial_chronos_scenarios.py"
python "TSFM Inference and Sampling/scoring/score_chronos2_scenarios.py"
```

`evaluate_m6_rps.py` builds the ground truth from the official asset file and writes `Evaluation Results/`. `exploratory_evaluation.py` must follow it, because it reads those files; it adds accuracy, macro ROC-AUC and expected calibration error, and it is also where `expected_calibration_error` and `ECE_BINS_PRIMARY` live, which the horizon diagnostics and the competition visuals both import. The three scoring scripts write the per-model scenario-count tables into `TSFM Inference and Sampling/results/` and, for Chronos-2, reconstructed scenarios into `scenario_outputs/`.

### Stage 11: horizon experiment and diagnostics

```bash
python "Horizon Experiment/horizon_experiment.py"
python "Horizon Experiment/horizon_diagnostics.py"
```

The first truncates the same saved 500-scenario trajectories to 1, 5, 10, 15 and 20 weekdays and writes `horizon_rps_by_round.csv` and `horizon_rps_summary.csv`. The second reads that summary and is the single source of truth for the per-asset detail, the model-by-horizon diagnostics and the per-round Spearman correlations. Both write into `Horizon Experiment/results/`. Later stages depend on this: the intervention scoring reads the zero-shot baselines from here, and the competition visuals read the 20-day, 500-scenario rows as the headline TSFM results.

### Stage 12: probability-shrinkage analysis

```bash
python "TSFM Calibration Experiment/run_calibration_experiment.py"
```

Shrinks each model's existing 500-scenario probabilities towards the uniform forecast for 21 values of lambda and rescores the same twelve rounds. Pure arithmetic on saved scenarios; no inference is rerun. Writes `TSFM Calibration Experiment/results/`, `tables/` and `figures/`.

### Stage 13: sampling-parameter interventions for the token-based models

First the validation sweep (GPU), over the six pre-M6 origins only:

1. `TSFM Uncertainty Experiment/inference_notebooks/chronos_t5_temperature_validation.ipynb`
2. `TSFM Uncertainty Experiment/inference_notebooks/financial_chronos_temperature_validation.ipynb`

Each writes 108 NPZ files, being 6 origins by 18 temperature and top-k combinations. Download them into `TSFM Uncertainty Experiment/model_outputs/<model>/validation/`, then select the winning configuration locally:

```bash
python "TSFM Uncertainty Experiment/analysis/score_validation.py" --model chronos_t5_base
python "TSFM Uncertainty Experiment/analysis/score_validation.py" --model financial_chronos
```

This writes `results/<model>/validation/` and `visuals/<model>/validation/`, including the `selected_config.json` that records the winner. No M6 outcome is read at this point.

Then the holdout runs (GPU), using the selected settings over the twelve official rounds:

3. `TSFM Uncertainty Experiment/inference_notebooks/chronos_t5_base_temperature_m6.ipynb`
4. `TSFM Uncertainty Experiment/inference_notebooks/financial_chronos_temperature_m6.ipynb`

Download their outputs into `TSFM Uncertainty Experiment/model_outputs/` as well.

### Stage 14: Chronos-2 low-rank adaptation

The extended panel is only needed here. Smoke-test the download before committing to all 100 assets:

```bash
python "FinQ Boost Reimplementation/download_extended_eodhd.py" --limit 3
```

Inspect `TSFM Uncertainty Experiment/data/extended_eodhd_status.csv`, then run the full download:

```bash
python "FinQ Boost Reimplementation/download_extended_eodhd.py"
```

Upload `TSFM Uncertainty Experiment/data/extended_eodhd_ohlcv_2000_2023.csv` to `MyDrive/HonoursResearch/datasets/`, then train and run the adapters (GPU), in this order:

1. `TSFM Uncertainty Experiment/finetuning_notebooks/chronos2_lora_training.ipynb`: ordinary LoRA, the control.
2. `TSFM Uncertainty Experiment/finetuning_notebooks/chronos2_uncertainty_lora_training.ipynb`: uncertainty-regularised, beta 0.5, 1 and 2.
3. `TSFM Uncertainty Experiment/finetuning_notebooks/chronos2_uncertainty_lora_training_extended.ipynb`: the aggressive betas 4, 8 and 16.
4. `TSFM Uncertainty Experiment/inference_notebooks/chronos2_lora_m6.ipynb`
5. `TSFM Uncertainty Experiment/inference_notebooks/chronos2_uncertainty_lora_m6.ipynb`
6. `TSFM Uncertainty Experiment/inference_notebooks/chronos2_uncertainty_lora_m6_extended.ipynb`

Each training notebook writes its adapter and `training_metadata.json` to `MyDrive/HonoursResearch/checkpoints/`; each inference notebook writes twelve quantile NPZ files per configuration. Download the adapters and metadata into `TSFM Uncertainty Experiment/artifacts/` and the quantile files into `TSFM Uncertainty Experiment/model_outputs/`.

### Stage 15: intervention scoring

```bash
python "TSFM Uncertainty Experiment/analysis/score_m6.py"
python "TSFM Uncertainty Experiment/analysis/score_uncertainty_m6.py"
```

Requires the downloaded intervention outputs from stages 13 and 14, and the zero-shot baselines from stage 11. Writes `TSFM Uncertainty Experiment/results/` and `visuals/`, including the per-beta tables and the combined comparison.

### Stage 16: paired statistical analysis

```bash
python "Statistical Analysis/paired_rps_analysis.py"
python "Statistical Analysis/test_paired_rps.py"
python "Statistical Analysis/paired_rps_visualisation.py"
python "Statistical Analysis/build_sampling_ablation_table.py"
```

`paired_rps_analysis.py` takes the twelve rounds as the statistical units and writes six files into `Statistical Analysis/results/`: the paired comparisons with BCa intervals and Holm-adjusted p-values, the canonical round RPS series, a model summary, an input manifest, the analysis configuration and a verification report. It overwrites its own outputs in place, so it needs no preparation.

`test_paired_rps.py` must run after it, because part of what the suite checks is the consistency of those written outputs.

`paired_rps_visualisation.py` and `build_sampling_ablation_table.py` then write the forest plot and the sampling ablation table into `Competition Visuals/`.

### Stage 17: tables and figures

Run each notebook from within its own directory, which is where its relative paths resolve.

1. `Competition Visuals/competition_visuals.ipynb`: the cross-model tables and figures.
2. `Competition Visuals/horizon_visualisations.ipynb`: the report-ready horizon figures.
3. `Horizon Experiment/horizon_visualisations.ipynb`: optional, the working diagnostic version.

## Outputs produced by a full run

None of the following are distributed. Each is created by the stage shown.

| Generated location | Created by |
|---|---|
| `Data/raw/`, `Data/processed/`, `Data/contexts/`, `Data/validation_contexts/` | stages 2 to 4 |
| `Data/metadata/m6_round_schedule.csv`, `feature_baseline_origin_schedule.csv`, `diagnostics/` | stages 2 to 5 |
| `logs/`, `reports/` | the pipeline and evaluation scripts throughout |
| `Feature Baselines/tuning/`, `Feature Baselines/results/` | stage 5 |
| `FinQ Boost Reimplementation/finqboost_m6_*.csv` | stage 6 |
| `Quasi-Average Reimplementation DRE-as-Stock/data/`, `results/` | stage 6 |
| `TSFM Inference and Sampling/inference_outputs/` | stages 8 and 9 |
| `TSFM Inference and Sampling/scenario_outputs/`, `results/` | stage 10 |
| `TSFM Inference and Sampling/chronos2_sampling_comparison/results/` | stage 10 |
| `Evaluation Results/` | stage 10 |
| `Horizon Experiment/results/`, `visualisations/` | stages 11 and 17 |
| `TSFM Calibration Experiment/results/`, `tables/`, `figures/` | stage 12 |
| `TSFM Uncertainty Experiment/data/`, `model_outputs/`, `artifacts/` | stages 13 and 14 |
| `TSFM Uncertainty Experiment/results/`, `visuals/` | stages 13 and 15 |
| `Statistical Analysis/results/` | stage 16 |
| `Competition Visuals/figures/`, `Competition Visuals/tables/` | stages 16 and 17 |

### Reference results

The following values are the final results reported by the study. They are included here as reference values for checking a complete reproduction.

Overall M6 RPS across the twelve rounds, with the naive equal-probability forecast at exactly 0.160000:

| Model | Overall RPS | Rounds beating naive |
|---|---|---|
| FinQBoost | 0.156703 | 12/12 |
| Random Forest | 0.157053 | 9/12 |
| Quasi-Average | 0.157234 | 9/12 |
| Softmax regression | 0.157355 | 9/12 |
| LightGBM | 0.157432 | 8/12 |
| Naive | 0.160000 | 0/12 |
| Chronos-2 | 0.165927 | 3/12 |
| Financial Chronos | 0.177947 | 2/12 |
| Chronos-T5 Base | 0.226920 | 0/12 |

After the architecture-specific interventions, Financial Chronos reaches 0.158526 and Chronos-2 with uncertainty-regularised LoRA reaches 0.160503, against 0.162997 for the ordinary-LoRA control.

## Tests and verification

The statistical analysis has a unit-test suite. Run it from the repository root:

```bash
python "Statistical Analysis/test_paired_rps.py"
```

It checks the bootstrap and permutation implementations, the Holm correction, the sign convention and the consistency of the written outputs, and prints a pass count.

**`Statistical Analysis/paired_rps_analysis.py` must have run first.** Part of the suite verifies the generated files in `Statistical Analysis/results/`, so on a fresh clone the test has nothing to check and will fail until stage 16 has produced them.

This is a focused test of the statistical layer. There is no automated end-to-end test of the full pipeline.

## Optional diagnostics

These three scripts are not required to reproduce the primary reported results. They run after the stages they depend on and can be skipped entirely.

| Script | What it adds |
|---|---|
| `TSFM Inference and Sampling/scoring/sampling_comparison.py` | Stacks the three per-model scenario-count tables from stage 10 into one combined table. Nothing is rescored. |
| `TSFM Uncertainty Experiment/analysis/plot_validation_vs_m6_coverage.py` | Plots internal-validation against final-M6 interval coverage and width by beta, from the stage 14 adapter metadata and the stage 15 summary. |
| `FinQ Boost Reimplementation/seed_sensitivity.py` | Characterises how much of the FinQBoost RPS is model stochasticity. The headline result uses seed 29 only and nothing here is averaged into it. |

## Acknowledgements

- The M6 Financial Forecasting Competition and its organisers, for the task definition, the asset universe and the official outcome data.
- Amazon Science for the Chronos and Chronos-2 checkpoints, and FinText for the financially pretrained Chronos checkpoint.
- Yahoo Finance and EODHD as the market-data providers.
- The released FinQBoost and Quasi-Average M6 entries, which the reimplementations here adapt and credit in their source files.

## Licence

No licence has been assigned to this repository. All rights are reserved by the author pending that decision. Third-party data and model checkpoints remain subject to the terms of their respective providers.
