"""Additional evaluation metrics for the completed M6 experiments.

Everything here is computed from retained artifacts only: no model is rerun or
retrained, and no completed experimental output is modified. For every model
family with saved 12-round 5-class probabilities it reports RPS (using the
project scorer), accuracy, macro one-vs-rest ROC-AUC and Expected Calibration
Error, and cross-checks each recomputed RPS against the saved value.

It is also the home of expected_calibration_error and ECE_BINS_PRIMARY, which
the horizon diagnostics and the competition visuals both import.

Inputs:
    Evaluation Results/m6_ground_truth_quintiles.csv            official truth
    Evaluation Results/predicted_quintile_probabilities_*.csv   two TSFMs
    Feature Baselines/results/{feature_baseline,softmax}_m6_predictions.csv
    TSFM Inference and Sampling/chronos2_sampling_comparison/results/*.csv

Outputs:
    Evaluation Results/exploratory_model_metrics.csv
    logs/exploratory_evaluation.log

Run:  python scripts/exploratory_evaluation.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from evaluate_m6_rps import RANK_COLUMNS, rps_scores  # noqa: E402

EVAL_DIR = PROJECT_ROOT / "Evaluation Results"
INFERENCE_DIR = (PROJECT_ROOT / "TSFM Inference and Sampling" / "inference_outputs")
FEATURE_BASELINE_DIR = PROJECT_ROOT / "Feature Baselines" / "results"
CHRONOS2_COMPARISON_DIR = (PROJECT_ROOT / "TSFM Inference and Sampling"
                           / "chronos2_sampling_comparison" / "results")
GROUND_TRUTH = EVAL_DIR / "m6_ground_truth_quintiles.csv"
OUT_METRICS = EVAL_DIR / "exploratory_model_metrics.csv"
LOG_PATH = PROJECT_ROOT / "logs" / "exploratory_evaluation.log"

N_QUINTILES = 5
N_ROUNDS = 12
N_ASSETS = 100
NAIVE_RPS = 0.16

# Expected Calibration Error: the standard top-label (confidence) formulation of
# Guo et al. (2017), which is also exactly what the reference implementation
# uses (Calculate_metrics.py::calc_ece, uniform binning, default M=5 bins,
# true label taken as argmax of the truth vector). M=5 is reported as the
# primary figure for comparability with that source; 10 and 15 bins are carried
# alongside because ECE is known to be sensitive to bin count and reporting a
# single binning would hide that.
ECE_BINS_PRIMARY = 5
ECE_BINS_SENSITIVITY = (10, 15)

# Ordering is deliberate: feature-based classifiers, then TSFMs, then benchmark.
MODEL_ORDER = [
    "RF / no-KNN", "RF / KNN", "LightGBM / no-KNN", "Softmax / KNN",
    "Chronos-2 120M (copula)", "Chronos-2 120M (independent)",
    "Financial Chronos 46M", "Chronos-T5 Base 200M",
    "Naive (0.2 each)",
]
FAMILY = {
    "RF / no-KNN": "Feature-based", "RF / KNN": "Feature-based",
    "LightGBM / no-KNN": "Feature-based", "Softmax / KNN": "Feature-based",
    "Chronos-2 120M (copula)": "TSFM", "Chronos-2 120M (independent)": "TSFM",
    "Financial Chronos 46M": "TSFM", "Chronos-T5 Base 200M": "TSFM",
    "Naive (0.2 each)": "Benchmark",
}

logger = logging.getLogger("exploratory_evaluation")


class ArtifactError(RuntimeError):
    """A required retained artifact is missing or malformed."""


# --------------------------------------------------------------------------- #
# Ground truth and saved probability forecasts
# --------------------------------------------------------------------------- #

def load_ground_truth() -> pd.DataFrame:
    """The official M6 truth already produced by the project's evaluator."""
    if not GROUND_TRUTH.is_file():
        raise ArtifactError(f"Missing {GROUND_TRUTH}")
    truth = pd.read_csv(GROUND_TRUTH)
    if len(truth) != N_ROUNDS * N_ASSETS:
        raise ArtifactError(f"Ground truth has {len(truth)} rows, expected 1200.")
    membership = truth[RANK_COLUMNS].to_numpy(dtype=float)
    if not np.allclose(membership.sum(axis=1), 1.0):
        raise ArtifactError("Ground-truth memberships do not sum to 1.")
    truth["true_class"] = membership.argmax(axis=1)
    return truth


def _standardise(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    out = frame[["round", "symbol"] + RANK_COLUMNS].copy()
    out["model"] = label
    return out


def load_probability_forecasts(truth: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Every complete 12-round probability set, aligned to the truth's row order.

    The incomplete LightGBM/KNN experiment is deliberately absent: it was stopped
    at round 11 and has no 12-round forecast, so it must not enter any aggregate.
    """
    forecasts: dict[str, pd.DataFrame] = {}

    feature = pd.read_csv(FEATURE_BASELINE_DIR / "feature_baseline_m6_predictions.csv")
    for (model, variant), block in feature.groupby(["model", "variant"]):
        label = {("rf", "no_knn"): "RF / no-KNN", ("rf", "knn"): "RF / KNN",
                 ("lgbm", "no_knn"): "LightGBM / no-KNN"}[(model, variant)]
        forecasts[label] = _standardise(block, label)

    softmax = pd.read_csv(FEATURE_BASELINE_DIR / "softmax_m6_predictions.csv")
    forecasts["Softmax / KNN"] = _standardise(softmax, "Softmax / KNN")

    forecasts["Chronos-T5 Base 200M"] = _standardise(
        pd.read_csv(EVAL_DIR / "predicted_quintile_probabilities_Chronos_T5_Base_200M.csv"),
        "Chronos-T5 Base 200M")
    forecasts["Financial Chronos 46M"] = _standardise(
        pd.read_csv(EVAL_DIR / "predicted_quintile_probabilities_"
                               "Financial_Chronos_Small_46M_2021_Global.csv"),
        "Financial Chronos 46M")

    chronos2 = CHRONOS2_COMPARISON_DIR
    forecasts["Chronos-2 120M (independent)"] = _standardise(
        pd.read_csv(chronos2 / "chronos2_independent_sampling_quintile_probabilities.csv"),
        "Chronos-2 120M (independent)")
    forecasts["Chronos-2 120M (copula)"] = _standardise(
        pd.read_csv(chronos2 / "chronos2_spatiotemporal_copula_quintile_probabilities.csv"),
        "Chronos-2 120M (copula)")

    naive = truth[["round", "symbol"]].copy()
    for column in RANK_COLUMNS:
        naive[column] = 1.0 / N_QUINTILES
    naive["model"] = "Naive (0.2 each)"
    forecasts["Naive (0.2 each)"] = naive

    key = truth[["round", "symbol"]]
    for label, frame in forecasts.items():
        merged = key.merge(frame, on=["round", "symbol"], how="left")
        if merged[RANK_COLUMNS].isna().any().any():
            raise ArtifactError(f"{label}: missing forecasts for some asset-rounds.")
        if len(merged) != len(truth):
            raise ArtifactError(f"{label}: {len(merged)} rows, expected {len(truth)}.")
        probabilities = merged[RANK_COLUMNS].to_numpy(dtype=float)
        if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
            raise ArtifactError(f"{label}: probability rows do not sum to 1.")
        forecasts[label] = merged
    return forecasts


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def expected_calibration_error(probabilities: np.ndarray, true_class: np.ndarray,
                               n_bins: int) -> float:
    """Top-label ECE with uniform binning (Guo et al., 2017).

    Confidence is max_k p_k, the probability assigned to the predicted class;
    accuracy is whether that predicted class is the realised quintile. The
    weighted mean absolute gap between the two, over equal-width confidence
    bins, is the ECE. Lower is better; 0 means perfect top-label calibration.
    """
    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    correct = (predicted == true_class).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        in_bin = (confidence > lower) & (confidence <= upper)
        weight = in_bin.mean()
        if weight > 0:
            ece += abs(confidence[in_bin].mean() - correct[in_bin].mean()) * weight
    return float(ece)


def reliability_curve(probabilities: np.ndarray, true_class: np.ndarray,
                      n_bins: int = ECE_BINS_PRIMARY) -> pd.DataFrame:
    """Per-bin confidence vs accuracy, for a reliability diagram."""
    confidence = probabilities.max(axis=1)
    correct = (probabilities.argmax(axis=1) == true_class).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        in_bin = (confidence > lower) & (confidence <= upper)
        rows.append({"bin_lower": lower, "bin_upper": upper,
                     "bin_centre": (lower + upper) / 2,
                     "n": int(in_bin.sum()),
                     "mean_confidence": float(confidence[in_bin].mean()) if in_bin.any() else np.nan,
                     "accuracy": float(correct[in_bin].mean()) if in_bin.any() else np.nan})
    return pd.DataFrame(rows)


def one_vs_rest_auc(probabilities: np.ndarray, true_class: np.ndarray
                    ) -> tuple[float, list[float]]:
    """Full macro one-vs-rest ROC-AUC and the five per-quintile AUCs.

    FULL ROC-AUC is used, not partial AUC: this project has no independently
    motivated false-positive-rate cutoff that would justify evaluating only part
    of the curve. (The reference implementation uses max_fpr=0.25; that choice
    is not adopted here.)
    """
    per_class = []
    for quintile in range(N_QUINTILES):
        positive = (true_class == quintile).astype(int)
        if positive.min() == positive.max():
            per_class.append(float("nan"))
            continue
        per_class.append(float(roc_auc_score(positive, probabilities[:, quintile])))
    return float(np.nanmean(per_class)), per_class


def evaluate_model(label: str, forecast: pd.DataFrame, truth: pd.DataFrame) -> dict:
    probabilities = forecast[RANK_COLUMNS].to_numpy(dtype=float)
    membership = truth[RANK_COLUMNS].to_numpy(dtype=float)
    true_class = truth["true_class"].to_numpy()

    rps = float(rps_scores(membership, probabilities).mean())
    predicted = probabilities.argmax(axis=1)
    accuracy = float((predicted == true_class).mean())
    macro_auc, per_class_auc = one_vs_rest_auc(probabilities, true_class)

    record = {
        "model": label, "family": FAMILY[label], "n_observations": len(forecast),
        "rps": round(rps, 8), "rps_vs_naive": round(rps - NAIVE_RPS, 8),
        "accuracy": round(accuracy, 6),
        "accuracy_vs_random": round(accuracy - 1 / N_QUINTILES, 6),
        "macro_roc_auc": round(macro_auc, 6),
        **{f"roc_auc_q{q + 1}": round(per_class_auc[q], 6) for q in range(N_QUINTILES)},
        f"ece_{ECE_BINS_PRIMARY}bin": round(
            expected_calibration_error(probabilities, true_class, ECE_BINS_PRIMARY), 6),
        "mean_confidence": round(float(probabilities.max(axis=1).mean()), 6),
        "mean_max_minus_accuracy": round(
            float(probabilities.max(axis=1).mean()) - accuracy, 6),
    }
    for bins in ECE_BINS_SENSITIVITY:
        record[f"ece_{bins}bin"] = round(
            expected_calibration_error(probabilities, true_class, bins), 6)
    return record


def per_round_metrics(forecasts: dict[str, pd.DataFrame], truth: pd.DataFrame
                      ) -> pd.DataFrame:
    """Long-format RPS and accuracy by round; used by the notebook figures."""
    rows = []
    for label in MODEL_ORDER:
        forecast = forecasts[label]
        for round_number in range(1, N_ROUNDS + 1):
            mask = (truth["round"] == round_number).to_numpy()
            probabilities = forecast.loc[mask, RANK_COLUMNS].to_numpy(dtype=float)
            membership = truth.loc[mask, RANK_COLUMNS].to_numpy(dtype=float)
            true_class = truth.loc[mask, "true_class"].to_numpy()
            rows.append({
                "model": label, "family": FAMILY[label], "round": round_number,
                "rps": float(rps_scores(membership, probabilities).mean()),
                "accuracy": float((probabilities.argmax(axis=1) == true_class).mean()),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# TSFM return RMSE
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def verify_rps_reproduces_saved(metrics: pd.DataFrame) -> list[str]:
    """Cross-check the recomputed RPS against every saved RPS in the project."""
    saved = {"Naive (0.2 each)": NAIVE_RPS}
    comparison = pd.read_csv(EVAL_DIR / "sampled_tsfm_rps_by_model.csv")
    lookup = dict(zip(comparison.iloc[:, 0], comparison["Overall Mean RPS"]))
    saved["Chronos-T5 Base 200M"] = lookup["Chronos T5 Base 200M"]
    saved["Financial Chronos 46M"] = lookup["Financial Chronos Small 46M 2021 Global"]
    for label, path in (("Chronos-2 120M (independent)",
                         "chronos2_independent_sampling_rps.csv"),
                        ("Chronos-2 120M (copula)",
                         "chronos2_spatiotemporal_copula_rps.csv")):
        frame = pd.read_csv(CHRONOS2_COMPARISON_DIR / path)
        saved[label] = float(frame.loc[frame["scope"] == "overall", "mean_RPS"].iloc[0])
    feature = pd.read_csv(FEATURE_BASELINE_DIR / "feature_baseline_m6_rps.csv")
    feature = feature[feature["status"] == "complete"]
    for record in feature.itertuples():
        label = {("rf", "no_knn"): "RF / no-KNN", ("rf", "knn"): "RF / KNN",
                 ("lgbm", "no_knn"): "LightGBM / no-KNN"}[(record.model, record.variant)]
        saved[label] = float(record.mean_m6_rps)
    softmax = pd.read_csv(FEATURE_BASELINE_DIR / "softmax_m6_rps.csv").iloc[0]
    saved["Softmax / KNN"] = float(softmax["mean_m6_rps"])

    checks = []
    for label, expected in saved.items():
        got = float(metrics.loc[metrics["model"] == label, "rps"].iloc[0])
        agree = np.isclose(got, expected, atol=1e-6)
        checks.append(f"{label}: recomputed {got:.6f} vs saved {expected:.6f} "
                      f"-> {'MATCH' if agree else 'MISMATCH'}")
        if not agree:
            raise ArtifactError(f"{label}: recomputed RPS {got} != saved {expected}")
    return checks


def run() -> pd.DataFrame:
    truth = load_ground_truth()
    forecasts = load_probability_forecasts(truth)
    logger.info("Loaded %d complete 12-round probability sets (LGBM/KNN excluded: "
                "stopped at round 11, no 12-round forecast).", len(forecasts))

    metrics = pd.DataFrame([evaluate_model(label, forecasts[label], truth)
                            for label in MODEL_ORDER])
    for line in verify_rps_reproduces_saved(metrics):
        logger.info("RPS cross-check | %s", line)

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUT_METRICS, index=False)
    logger.info("Wrote %s (%d models)", OUT_METRICS, len(metrics))

    return metrics


def main() -> int:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8"),
                  logging.StreamHandler(sys.stdout)])
    try:
        run()
    except ArtifactError as exc:
        logger.error("Artifact problem: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
