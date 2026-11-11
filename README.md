# Learning When Not to Be Confident: Time-Series Foundation Models in the M6 Competition

Evaluating whether pretrained time-series foundation models can transfer zero-shot to the
M6 financial forecasting competition, and whether their forecast uncertainty can be improved
through each architecture's own mechanisms.

BSc Honours Research Project, School of Computer Science and Applied Mathematics,
University of the Witwatersrand.

Author: Yabisira Gebremichael (2661262). Supervisor: Dr Hairong Bau.

---

## Overview

The M6 Competition's individual forecasting task asked entrants to submit, for each of
100 assets (50 S&P 500 stocks and 50 ETFs), a probability distribution over five
cross-sectional return quintiles. The competition ran for twelve four-week rounds from
March 2022 to February 2023, and submissions were scored with the Ranked Probability
Score (RPS). The naive equal-probability forecast, 0.2 in every quintile, scores exactly
0.160000.

Foundation models do not emit quintile probabilities directly, so each model's native
output is converted into one. The 20-trading-day horizon is the competition's own. Chronos-T5
Base and Financial Chronos produce sampled return trajectories, which are cumulated and ranked
cross-sectionally; Chronos-2 produces quantiles, which are turned into scenarios by independent
percentile sampling before the same ranking is applied. The resulting quintile frequencies are
the submitted probabilities.

This repository contains the full comparison: three foundation models evaluated zero-shot,
a naive benchmark, three feature-based classifiers, two reimplemented M6 competition entries,
and the adaptation experiments that follow from the diagnosis.

## Project aims

1. How competitively do the evaluated foundation models perform zero-shot, relative to the
   naive, feature-based and competition-derived benchmarks?
2. What performance differences appear between the multivariate and univariate
   general-domain models, and between the financially pretrained and architecturally
   matched general-domain univariate models?
3. How do confidence and calibration relate to foundation-model performance, and can
   architecture-specific uncertainty interventions improve RPS?

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

Selected hyperparameters are not restated here. They are recorded in the tracked outputs:
`Feature Baselines/tuning/`, the configuration columns of
`FinQ Boost Reimplementation/finqboost_m6_rps.csv`, the per-model
`selected_config.json` under `TSFM Uncertainty Experiment/results/*/validation/`, and the
`training_metadata.json` beside each LoRA adapter.

## Data sources and repository data policy

Two historical panels support the work. The names Dataset A and Dataset D are retained
throughout the code.

| Name | Source | Range | Role |
|---|---|---|---|
| Dataset A | Yahoo Finance, with EODHD recovering DRE, RE and WRK | 2020-01-02 to 2023-02-03, 807 weekdays | Adjusted closes behind the 512-weekday foundation-model contexts |
| Dataset D | EODHD | 2009-01-02 to 2023-02-03, 3,676 weekdays | Shared panel for the five data-driven benchmarks |
| Extended EODHD history | EODHD | 2000-01-03 to 2023-02-03, 6,025 weekdays | Chronos-2 LoRA fine-tuning only |

The extended history lives at
`TSFM Uncertainty Experiment/data/extended_eodhd_ohlcv_2000_2023.csv` once generated.

### What is and is not included

Yahoo Finance and EODHD source data, and every panel derived row by row from them, are
**deliberately not tracked**. That covers `Data/raw/yahoo/`, `Data/raw/dataset_d_eodhd/`,
`Data/processed/`, `Data/contexts/`, the validation context CSVs, the extended EODHD panel
and the derived Quasi-Average panel. These paths are listed in
[`.gitignore`](.gitignore) and are created locally when you run the pipeline.

Every script needed to download, validate, preprocess and build contexts from that data
**is** included, so a user with provider access can regenerate all of it.

Model outputs, scenario outputs, LoRA adapters, predictions, RPS tables, statistical
outputs and the inputs to the visuals **are** tracked. That is deliberate: it lets the
evaluation, statistics and figures be reproduced and checked without redistributing
restricted provider data.

[`Data/raw/m6_official/assets_m6.csv`](Data/raw/m6_official/assets_m6.csv) is tracked. It is
the official asset file published by the M6 organisers and is the sole source of realised
outcomes and ground-truth quintiles. It is not Yahoo or EODHD provider data.

Users remain responsible for complying with the terms of the relevant data providers.

### Feature inputs

The feature-based baselines read daily log returns, close, adjusted close and volume.
Processed open, high and low panels are not required and are no longer written by
`preprocess_dataset_d.py`, although those fields are still read from the raw file and
validated during preprocessing.

## Repository structure

Only the main tracked locations are shown. Paths marked *(local, ignored)* are generated
on your machine and are not in the repository.

```
.
├── scripts/
│   ├── evaluate_m6_rps.py              shared M6 scorer and ground truth
│   ├── exploratory_evaluation.py       cross-family metrics, ECE definition
│   └── pipeline/
│       ├── eodhd_client.py             shared EODHD client and token handling
│       ├── download_dataset_a.py       Yahoo adjusted closes
│       ├── repair_dataset_a_eodhd.py   EODHD recovery for DRE, RE, WRK
│       ├── preprocess_dataset_a.py     weekday calendar and log returns
│       ├── download_dataset_d_eodhd.py EODHD OHLCV
│       ├── preprocess_dataset_d.py     Dataset D weekday panels
│       ├── build_rolling_origins.py    the twelve 512-day M6 contexts
│       └── build_validation_contexts.py  the six pre-M6 validation contexts
├── Data/
│   ├── metadata/                       schedules, asset identifiers, diagnostics
│   ├── raw/m6_official/assets_m6.csv   official M6 asset file
│   ├── raw/, processed/, contexts/     (local, ignored)
│   └── validation_contexts/            manifest tracked, contexts (local, ignored)
├── TSFM Inference and Sampling/
│   ├── inference_notebooks/            Colab/GPU foundation-model inference
│   ├── scenario_outputs/               saved trajectories and quantiles
│   ├── scoring/                        trajectory and quantile scoring
│   └── chronos2_sampling_comparison/   independent sampling and copula check
├── TSFM Uncertainty Experiment/
│   ├── finetuning_notebooks/           Colab/GPU LoRA training
│   ├── inference_notebooks/            Colab/GPU intervention inference
│   ├── analysis/                       validation and M6 scoring
│   ├── artifacts/                      LoRA adapters and training metadata
│   ├── model_outputs/, results/, visuals/
│   └── data/                           extended_eodhd_ohlcv_2000_2023.csv (local, ignored)
├── Feature Baselines/                  dataset build, tuning, evaluation, results
├── FinQ Boost Reimplementation/        FinQBoost script and results
├── Quasi-Average Reimplementation DRE-as-Stock/
├── Horizon Experiment/                 multi-horizon RPS and diagnostics
├── TSFM Calibration Experiment/        probability-shrinkage sweep
├── Statistical Analysis/               paired tests, forest plot, unit tests
├── Evaluation Results/                 ground truth and shared evaluation tables
├── Competition Visuals/                figures, tables, system_overview/
└── requirements.txt
```

## Installation

The project was developed and run on Python 3.10.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

This installs the CPU stack only: pandas, numpy, scipy, scikit-learn, LightGBM, XGBoost,
Hyperopt, matplotlib, yfinance and the Jupyter execution packages. It is enough for every
step except foundation-model inference and LoRA training.

The GPU stack is **not** installed by `requirements.txt`. PyTorch, Transformers, Chronos
Forecasting and PEFT are listed commented out at the end of the file because the notebooks
install them inside the Colab session: PyTorch and Transformers provide the model runtime,
Chronos Forecasting loads the Chronos and Chronos-2 checkpoints, and PEFT supplies the LoRA
adapters.

### Provider credentials

The EODHD acquisition scripts read a token from a git-ignored env file in the repository
root:

```
# proj.env
EODHD_API_KEY=your_token_here
```

`EODHD_API_TOKEN` is accepted as an alternative name. The scripts never print, log or embed
the token, and they abort before any request if an env file is tracked by git or is not
covered by `.gitignore`. Do not commit this file.

## Reproducibility levels

The three levels differ in how much of the pipeline is rerun and what each one
requires. Level 1 reads the saved results and needs nothing beyond the repository.
Level 2 rebuilds the statistics, tables and figures from the model and scenario
outputs that ship with the repository, so it needs Python but no provider data and no
GPU. Level 3 rebuilds everything from the raw provider downloads onwards and needs an
EODHD token, network access and a GPU. Each level includes the work of the ones above it,
so start at Level 1 and go further only as needed.

### Level 1: inspect the included results

No provider data and no model runs are needed. The results shipped with the repository are:

- `Evaluation Results/` for the ground-truth quintiles and the shared evaluation tables
- `Feature Baselines/results/` and `Feature Baselines/tuning/`
- `FinQ Boost Reimplementation/finqboost_m6_rps.csv` and its predictions
- `Quasi-Average Reimplementation DRE-as-Stock/results/`
- `Horizon Experiment/results/` and `TSFM Calibration Experiment/results/`
- `TSFM Uncertainty Experiment/results/` for the sampling and LoRA interventions
- `Statistical Analysis/results/` for the paired comparisons
- `Competition Visuals/figures/`, `tables/` and `report_ready_tables/`

The quickest entry points are
[`Competition Visuals/tables/headline_model_comparison.csv`](Competition%20Visuals/tables/headline_model_comparison.csv),
[`Statistical Analysis/results/paired_comparisons.csv`](Statistical%20Analysis/results/paired_comparisons.csv)
and the notebook [`Competition Visuals/competition_visuals.ipynb`](Competition%20Visuals/competition_visuals.ipynb),
which carries its stored outputs.

### Level 2: reproduce the downstream evaluation

Keep the included model and scenario outputs, and regenerate the statistics, tables and
figures from them. **Do not delete any model or scenario outputs for this level.** All
commands run from the repository root.

```bash
python "Statistical Analysis/paired_rps_analysis.py"
python "Statistical Analysis/test_paired_rps.py"
```

`paired_rps_analysis.py` overwrites the six files in `Statistical Analysis/results/` in
place, so it needs no preparation.

Two scripts refuse to overwrite an existing target and stop with a message naming the
file. Remove exactly these before rerunning them:

| Script | Remove first |
|---|---|
| `Statistical Analysis/paired_rps_visualisation.py` | `Competition Visuals/figures/paired_rps_effects_forest.png`, `Competition Visuals/tables/paired_rps_primary_statistics.csv`, `Competition Visuals/tables/paired_rps_primary_statistics.png` |
| `Statistical Analysis/build_sampling_ablation_table.py` | `Competition Visuals/tables/sampling_factorial_ablation_compact.{png,csv,tex}` |

```bash
rm "Competition Visuals/figures/paired_rps_effects_forest.png" \
   "Competition Visuals/tables/paired_rps_primary_statistics.csv" \
   "Competition Visuals/tables/paired_rps_primary_statistics.png"
python "Statistical Analysis/paired_rps_visualisation.py"
```

The other audited output-generating scripts create their own output directory and overwrite
their result files in place, so repeated runs are safe.

The competition figures and tables are regenerated by running
`Competition Visuals/competition_visuals.ipynb` from within its own directory, which is
where its relative paths resolve.

### Level 3: full reproduction

This level needs provider access, a GPU for the foundation-model steps, and considerable
time. The order below is the one the code supports.

**1. Credentials.** Create `proj.env` as described above.

**2. Acquire and prepare the data.** All from the repository root:

```bash
python scripts/pipeline/download_dataset_a.py
python scripts/pipeline/repair_dataset_a_eodhd.py
python scripts/pipeline/preprocess_dataset_a.py
python scripts/pipeline/download_dataset_d_eodhd.py
python scripts/pipeline/preprocess_dataset_d.py
python scripts/pipeline/build_rolling_origins.py
python scripts/pipeline/build_validation_contexts.py
python "FinQ Boost Reimplementation/download_extended_eodhd.py"
```

Dataset A and Dataset D acquisition require network access and, for EODHD, a valid token.
`build_validation_contexts.py` reads `Data/processed/`, which is not shipped, and writes the six
`Data/validation_contexts/validation_*_context.csv` files, which are gitignored. It therefore
belongs to this level rather than Level 2, and it refuses to overwrite an existing context file,
so remove any stale ones before rerunning it.
`download_extended_eodhd.py` has never been run against the live API, so treat its first
run as a smoke test with `--limit 3` and inspect the status file before fetching all 100
assets.

**3. Build the feature datasets.**

```bash
python "Feature Baselines/build_feature_baseline_dataset.py"
python "Feature Baselines/tune_feature_baselines.py"
python "Feature Baselines/evaluate_feature_baselines_m6.py"
python "Feature Baselines/softmax_feature_baseline.py"
```

Tuning is the slowest CPU step in the repository.

**4. Run the competition reimplementations.**

```bash
python "FinQ Boost Reimplementation/finqboost.py"
python "Quasi-Average Reimplementation DRE-as-Stock/quasi_average.py"
```

Both default to Dataset D. Pass `--panel extended` to use the 2000-start panel instead.

**5. Foundation-model inference (GPU, Colab).** Run the notebooks under
`TSFM Inference and Sampling/inference_notebooks/` for the zero-shot trajectories and
Chronos-2 quantiles, then the notebooks under `TSFM Uncertainty Experiment/` for the
temperature and top-k grids and for LoRA training and intervention inference. These are the
expensive steps and they are not runnable from the CPU environment above.

**Before a clean full rerun, empty the additive output folders.** These accumulate rather
than overwrite, so a partial rerun at a different sample count will otherwise leave earlier
artifacts beside the new ones:

- `TSFM Inference and Sampling/scenario_outputs/`
- `TSFM Uncertainty Experiment/model_outputs/`

**6. Score and evaluate.**

```bash
python "TSFM Inference and Sampling/scoring/score_chronos2_scenarios.py"
python "TSFM Inference and Sampling/scoring/score_chronos_t5_base_scenarios.py"
python "TSFM Inference and Sampling/scoring/score_financial_chronos_scenarios.py"
python "Horizon Experiment/horizon_experiment.py"
python "TSFM Calibration Experiment/run_calibration_experiment.py"
python "TSFM Uncertainty Experiment/analysis/score_validation.py" --model financial_chronos
python "TSFM Uncertainty Experiment/analysis/score_validation.py" --model chronos_t5_base
python "TSFM Uncertainty Experiment/analysis/score_m6.py"
python "TSFM Uncertainty Experiment/analysis/score_uncertainty_m6.py"
```

**7. Statistics and visuals.** Follow Level 2 above.

These commands are documented from the current code. They were not executed end to end
while this README was prepared.

### Google Drive and Colab note

Two inference notebooks read their contexts from a Google Drive folder named
`Round_1_Context`. The name is historical: despite it, the folder holds the contexts for
**all twelve** M6 rounds. It is a Drive path, not a description of the data. Either keep the
folder name for compatibility, or change the path variable at the top of those notebooks to
match your own Drive layout.

## Pipeline

![Diagram of the M6 evaluation pipeline: market data are prepared once, passed to the zero-shot foundation models and the benchmark methods, converted into M6 quintile probabilities and scored with RPS across twelve rounds, after which diagnostic analyses motivate the architecture-specific interventions.](Competition%20Visuals/system_overview/m6-system-overview.png)

Market data are prepared once and shared by the foundation models and the benchmarks. Their
forecasts are converted into M6 quintile probabilities and scored over the twelve rounds;
diagnostic analyses then motivate the architecture-specific interventions, which are
re-evaluated through the same procedure.

## Main outputs

| Category | Location |
|---|---|
| Ground truth and shared evaluation tables | `Evaluation Results/` |
| Per-model M6 predictions and RPS | each model's own `results/` folder |
| Multi-horizon RPS and return diagnostics | `Horizon Experiment/results/` |
| Probability-shrinkage sweep | `TSFM Calibration Experiment/results/` |
| Sampling and LoRA intervention results | `TSFM Uncertainty Experiment/results/` |
| Paired comparisons, intervals and Holm-adjusted p-values | `Statistical Analysis/results/` |
| Figures and report-ready tables | `Competition Visuals/` |

Overall M6 RPS across the twelve rounds, from
[`Competition Visuals/tables/headline_model_comparison.csv`](Competition%20Visuals/tables/headline_model_comparison.csv):

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

After the architecture-specific interventions, Financial Chronos reaches 0.158526 and
Chronos-2 with uncertainty-regularised LoRA reaches 0.160503. The full comparison, including
the ordinary-LoRA control, is in
[`Competition Visuals/report_ready_tables/table_c_final_intervention_comparison.csv`](Competition%20Visuals/report_ready_tables/table_c_final_intervention_comparison.csv).

## Tests and verification

The statistical analysis has a unit-test suite. Run it from the repository root:

```bash
python "Statistical Analysis/test_paired_rps.py"
```

It checks the bootstrap and permutation implementations, the Holm correction, the sign
convention and the consistency of the written outputs, and prints a pass count.

This is a focused test of the statistical layer. There is no automated end-to-end test of
the full pipeline: foundation-model inference and LoRA training need a GPU and provider
data, so those stages are verified by their own recorded diagnostics rather than by a test
runner.

## Data availability and repository size

Restricted source data and the panels derived row by row from it are not in the repository,
but the code that downloads, validates and builds them is. Generated outputs are kept so
that the evaluation, statistics and figures can be checked independently.

That choice makes the repository large: roughly 446 MB of model outputs, 243 MB of scenario
outputs and 42 MB of LoRA adapters, about 752 MB in total. Expect a clone of that order.

## Acknowledgements

- The M6 Financial Forecasting Competition and its organisers, for the task definition, the
  asset universe and the official outcome data.
- Amazon Science for the Chronos and Chronos-2 checkpoints, and FinText for the financially
  pretrained Chronos checkpoint.
- Yahoo Finance and EODHD as the market-data providers.
- The released FinQBoost and Quasi-Average M6 entries, which the reimplementations here
  adapt and credit in their source files.

## Licence

No licence has been assigned to this repository. All rights are reserved by the author
pending that decision. Third-party data and model checkpoints remain subject to the terms of
their respective providers.
