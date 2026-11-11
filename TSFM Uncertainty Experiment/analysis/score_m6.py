"""Score the three uncertainty interventions on the twelve official M6 rounds.

Each model keeps the post-processing its zero-shot run used, so the only
difference from the original result is the intervention itself. Financial
Chronos and Chronos-T5 Base are scored directly from saved 500-sample
trajectories; Chronos-2 LoRA quantiles become 1,000 independent-percentile
scenarios (seed 42) of which the first 500 are used.

Every numerical step is imported rather than reimplemented: evaluate_m6_rps.py
for returns, ranking, tie handling, quintile probabilities and RPS,
exploratory_evaluation.py for ECE, and the Chronos-2 comparison module for the
quantile-to-scenario conversion. Zero-shot baselines are read from the Horizon
Experiment h=20 results.

Outputs results/<model>/m6/, results/combined_m6/ and visuals/combined_m6/.

Usage: python "TSFM Uncertainty Experiment/analysis/score_m6.py"
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXPERIMENT_DIR = HERE.parent
REPO_ROOT = EXPERIMENT_DIR.parent
TSFM_DIR = REPO_ROOT / "TSFM Inference and Sampling"
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(TSFM_DIR / "chronos2_sampling_comparison"))

from evaluate_m6_rps import (  # noqa: E402
    RANK_COLUMNS, build_ground_truth, load_prices, quintile_probabilities,
    rps_scores,
)
from exploratory_evaluation import (  # noqa: E402
    ECE_BINS_PRIMARY, expected_calibration_error,
)
from evaluate_independent_sampling import (  # noqa: E402
    N_SCENARIOS, RANDOM_SEED, generate_independent_scenarios,
)
from evaluate_spatiotemporal_copula import marginal_curves  # noqa: E402

SCHEDULE_PATH = REPO_ROOT / "Data" / "metadata" / "m6_round_schedule.csv"
HORIZON_BY_ROUND = REPO_ROOT / "Horizon Experiment" / "results" / "horizon_rps_by_round.csv"
HORIZON_DIAGNOSTICS = REPO_ROOT / "Horizon Experiment" / "results" / "horizon_diagnostics.csv"

MODEL_OUTPUTS = EXPERIMENT_DIR / "model_outputs"
RESULTS_DIR = EXPERIMENT_DIR / "results"
COMBINED_RESULTS = RESULTS_DIR / "combined_m6"
COMBINED_VISUALS = EXPERIMENT_DIR / "visuals" / "combined_m6"

N_ASSETS = 100
N_SAMPLES = 500
HORIZON = 20
NAIVE_RPS = 0.16

# Each model: where its files are, how they are named, which post-processing
# applies, and which Horizon row holds its authoritative zero-shot original.
MODELS = {
    "financial_chronos": {
        "display": "Financial Chronos",
        "intervention": "sampling calibration",
        "label": "temperature 5.0, top-k full",
        "directory": MODEL_OUTPUTS / "financial_chronos" / "m6" / "temp5.0_topkfull",
        "pattern": "financial_chronos_round{:02d}_temp5.0_topkfull_samples500.npz",
        "kind": "samples",
        "original": "Financial Chronos 46M",
        "colour": "#2ca02c",
    },
    "chronos_t5_base": {
        "display": "Chronos-T5 Base",
        "intervention": "sampling calibration",
        "label": "temperature 1.5, top-k full",
        "directory": MODEL_OUTPUTS / "chronos_t5_base" / "m6" / "temp1.5_topkfull",
        "pattern": "chronos_t5_base_round{:02d}_temp1.5_topkfull_samples500.npz",
        "kind": "samples",
        "original": "Chronos-T5 Base 200M",
        "colour": "#d62728",
    },
    "chronos2_lora": {
        "display": "Chronos-2",
        "intervention": "LoRA fine-tuning",
        "label": "LoRA rank 8, alpha 16",
        "directory": MODEL_OUTPUTS / "chronos2_lora" / "m6",
        "pattern": "chronos2_lora_round{:02d}_quantiles.npz",
        "kind": "quantiles",
        "original": "Chronos-2 120M (independent sampling)",
        "colour": "#1f77b4",
    },
}


def load_round_probabilities(config: dict, round_number: int,
                             symbols_expected: list[str]) -> tuple[np.ndarray, list[str]]:
    """Return one round's (100, 5) M6 quintile probabilities and its asset order.

    Sampled models are scored straight from their trajectories. Chronos-2 is
    reconstructed from its quantiles with the same independent-percentile
    sampling, seed and prefix rule as the published zero-shot N=500 result.
    """
    path = config["directory"] / config["pattern"].format(round_number)
    with np.load(path, allow_pickle=False) as data:
        symbols = [str(s) for s in data["asset_symbols"]]
        if config["kind"] == "samples":
            scenarios = data["forecast_samples"].astype(np.float64)
        else:
            forecasts = data["quantile_forecasts"].astype(np.float64)
            levels = data["quantile_levels"].astype(np.float64)

    if config["kind"] == "quantiles":
        # 1,000 scenarios drawn once, first 500 used: the published N=500 path.
        scenarios = generate_independent_scenarios(
            marginal_curves(forecasts), levels, N_SCENARIOS, RANDOM_SEED)
        if scenarios.shape != (N_ASSETS, N_SCENARIOS, HORIZON):
            raise ValueError(f"round {round_number}: scenarios {scenarios.shape}")
        scenarios = scenarios[:, :N_SAMPLES, :]

    if scenarios.shape != (N_ASSETS, N_SAMPLES, HORIZON):
        raise ValueError(f"{path.name}: shape {scenarios.shape}")
    if not np.isfinite(scenarios).all():
        raise ValueError(f"{path.name}: non-finite values")
    if symbols != symbols_expected:
        raise ValueError(f"{path.name}: asset order differs from the ground truth")

    probabilities = quintile_probabilities(scenarios)
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-9):
        raise ValueError(f"{path.name}: probabilities do not sum to 1")
    return probabilities, symbols


def score_model(key: str, config: dict, ground_truth: pd.DataFrame) -> dict:
    """Score one model over all 12 rounds and write its per-model result files."""
    round_rows, probability_rows = [], []
    all_probabilities, all_actual = [], []

    for round_number in range(1, 13):
        truth = ground_truth[ground_truth["round"] == round_number].sort_values("symbol")
        symbols = list(truth["symbol"])
        probabilities, _ = load_round_probabilities(config, round_number, symbols)
        actual = truth[RANK_COLUMNS].to_numpy(dtype=float)

        scores = rps_scores(actual, probabilities)
        all_probabilities.append(probabilities)
        all_actual.append(actual)

        round_rows.append({"model": config["display"], "round": round_number,
                           "mean_rps": float(scores.mean())})
        for i, symbol in enumerate(symbols):
            probability_rows.append({
                "model": config["display"], "round": round_number, "symbol": symbol,
                **{c: float(probabilities[i, q]) for q, c in enumerate(RANK_COLUMNS)},
                "RPS": float(scores[i]),
            })
        print(f"   round {round_number:02d}: RPS {scores.mean():.6f}")

    probabilities = np.vstack(all_probabilities)
    actual = np.vstack(all_actual)
    true_class = actual.argmax(axis=1)
    safe = np.where(probabilities > 0, probabilities, 1.0)

    by_round = pd.DataFrame(round_rows)
    summary = {
        "model_key": key,
        "model": config["display"],
        "intervention": config["intervention"],
        "configuration": config["label"],
        "overall_rps": float(by_round["mean_rps"].mean()),
        "rps_std_across_rounds": float(by_round["mean_rps"].std()),
        "accuracy": float((probabilities.argmax(axis=1) == true_class).mean()),
        "mean_confidence": float(probabilities.max(axis=1).mean()),
        "mean_entropy": float((-(safe * np.log(safe)).sum(axis=1)).mean()),
        f"ece_{ECE_BINS_PRIMARY}bin": expected_calibration_error(
            probabilities, true_class, ECE_BINS_PRIMARY),
        "beats_naive": bool(by_round["mean_rps"].mean() < NAIVE_RPS),
    }

    out = RESULTS_DIR / key / "m6"
    out.mkdir(parents=True, exist_ok=True)
    by_round.round(8).to_csv(out / "m6_rps_by_round.csv", index=False)
    pd.DataFrame(probability_rows).round(8).to_csv(
        out / "m6_predicted_quintile_probabilities.csv", index=False)
    pd.DataFrame([summary]).round(8).to_csv(out / "m6_summary.csv", index=False)
    return {"summary": summary, "by_round": by_round}


def load_originals() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Authoritative zero-shot M6 results, from the Horizon Experiment at h=20."""
    by_round = pd.read_csv(HORIZON_BY_ROUND)
    by_round = by_round[by_round["horizon_days"] == HORIZON]
    diagnostics = pd.read_csv(HORIZON_DIAGNOSTICS)
    diagnostics = diagnostics[diagnostics["horizon_days"] == HORIZON]
    return by_round, diagnostics


def build_comparison(scored: dict, originals: pd.DataFrame,
                     diagnostics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One row per model comparing its original with its intervention."""
    ece_column = next(c for c in diagnostics.columns if c.startswith("ece_"))
    rows, round_rows = [], []

    for key, config in MODELS.items():
        new = scored[key]["summary"]
        original_rounds = (originals[originals["model"] == config["original"]]
                           .sort_values("round"))
        original_diagnostics = (diagnostics[diagnostics["model"] == config["original"]]
                                .iloc[0])
        original_rps = float(original_rounds["mean_rps"].mean())

        merged = (original_rounds[["round", "mean_rps"]]
                  .rename(columns={"mean_rps": "original_rps"})
                  .merge(scored[key]["by_round"][["round", "mean_rps"]]
                         .rename(columns={"mean_rps": "intervention_rps"}), on="round"))
        merged["change"] = merged["intervention_rps"] - merged["original_rps"]
        merged.insert(0, "intervention", config["intervention"])
        merged.insert(0, "model", config["display"])
        round_rows.append(merged)

        rows.append({
            "model": config["display"],
            "intervention": config["intervention"],
            "configuration": config["label"],
            "original_rps": round(original_rps, 6),
            "intervention_rps": round(new["overall_rps"], 6),
            "rps_change": round(new["overall_rps"] - original_rps, 6),
            "rounds_improved": int((merged["change"] < 0).sum()),
            "original_beats_naive": bool(original_rps < NAIVE_RPS),
            "intervention_beats_naive": bool(new["overall_rps"] < NAIVE_RPS),
            "original_accuracy": round(float(original_diagnostics["accuracy"]), 6),
            "intervention_accuracy": round(new["accuracy"], 6),
            "original_confidence": round(float(original_diagnostics["mean_confidence"]), 6),
            "intervention_confidence": round(new["mean_confidence"], 6),
            "original_entropy": round(float(original_diagnostics["mean_entropy"]), 6),
            "intervention_entropy": round(new["mean_entropy"], 6),
            "original_ece": round(float(original_diagnostics[ece_column]), 6),
            "intervention_ece": round(new[f"ece_{ECE_BINS_PRIMARY}bin"], 6),
        })

    return pd.DataFrame(rows), pd.concat(round_rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Visuals
# ---------------------------------------------------------------------------

def plot_overall_rps(comparison: pd.DataFrame) -> None:
    """Original against intervention for each model, with the naive benchmark."""
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    fig.patch.set_facecolor("white")
    x = np.arange(len(comparison))
    width = 0.36

    ax.bar(x - width / 2, comparison["original_rps"], width, label="original (zero-shot)",
           color="#9e9e9e")
    ax.bar(x + width / 2, comparison["intervention_rps"], width,
           color=[MODELS[k]["colour"] for k in MODELS])
    ax.axhline(NAIVE_RPS, color="#000000", linestyle="--", linewidth=1.2, zorder=3)


    for i, row in comparison.iterrows():
        ax.text(i - width / 2, row["original_rps"] + 0.0015, f"{row['original_rps']:.4f}",
                ha="center", fontsize=8.5, color="#555555")
        ax.text(i + width / 2, row["intervention_rps"] + 0.0015,
                f"{row['intervention_rps']:.4f}", ha="center", fontsize=8.5,
                fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['model']}\n{r['intervention']}"
                        for _, r in comparison.iterrows()], fontsize=9.5)
    ax.set_ylabel("Overall M6 RPS (lower is better)")
    ax.set_title("M6 holdout: effect of each uncertainty intervention\n"
                 "12 official rounds, 500 scenarios", fontsize=11.5, pad=12)
    # headroom so the legend clears the tallest bar and its value label
    ax.set_ylim(0.14, max(comparison["original_rps"]) * 1.17)
    # Intervention bars carry each model's own colour, so the legend uses a
    # neutral swatch rather than implying one colour means "after".
    handles = [Patch(facecolor="#9e9e9e", label="original (zero-shot)"),
               Patch(facecolor="#555555", label="after intervention (model colour)"),
               Line2D([], [], color="#000000", linestyle="--", linewidth=1.2,
                      label="naive equal-probability benchmark (0.160)")]
    ax.legend(handles=handles, frameon=False, fontsize=9.5, loc="upper left")
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(COMBINED_VISUALS / "m6_overall_rps_comparison.png", dpi=200,
                facecolor="white")
    plt.close(fig)


def plot_round_changes(by_round: pd.DataFrame) -> None:
    """Per-round RPS change, to show whether each intervention helped consistently."""
    fig, ax = plt.subplots(figsize=(10.2, 5.2))
    fig.patch.set_facecolor("white")
    rounds = sorted(by_round["round"].unique())
    width = 0.26

    for offset, (key, config) in zip((-width, 0.0, width), MODELS.items()):
        block = (by_round[by_round["model"] == config["display"]]
                 .sort_values("round"))
        ax.bar(np.array(rounds) + offset, block["change"], width,
               label=f"{config['display']} ({config['intervention']})",
               color=config["colour"])

    ax.axhline(0.0, color="#000000", linewidth=1.0)
    ax.set_xticks(rounds)
    ax.set_xlabel("M6 round")
    ax.set_ylabel("RPS change (intervention minus original)\nnegative = improved")
    ax.set_title("M6 holdout: per-round effect of each intervention",
                 fontsize=11.5, pad=12)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(COMBINED_VISUALS / "m6_per_round_rps_change.png", dpi=200,
                facecolor="white")
    plt.close(fig)


def plot_confidence_accuracy(comparison: pd.DataFrame) -> None:
    """Confidence against accuracy, with the move from original to intervention.

    A point on the diagonal is perfectly calibrated at the top label. Points to
    the right of it are overconfident, and the arrow shows how far each
    intervention moved the model.
    """
    fig, ax = plt.subplots(figsize=(8.0, 5.6))
    fig.patch.set_facecolor("white")

    lo, hi = 0.21, 0.56
    y_lo, y_hi = 0.205, 0.285
    ax.plot([lo, y_hi], [lo, y_hi], color="#000000", linestyle="--", linewidth=1.1,
            zorder=1)
    ax.annotate("perfect calibration\n(confidence = accuracy)", xy=(0.2275, 0.2275),
                xytext=(0.238, 0.212), fontsize=8.5, color="#444444",
                arrowprops=dict(arrowstyle="-", color="#888888", linewidth=0.9))

    handles = []
    for key, config in MODELS.items():
        row = comparison[comparison["model"] == config["display"]].iloc[0]
        ax.annotate("", xy=(row["intervention_confidence"], row["intervention_accuracy"]),
                    xytext=(row["original_confidence"], row["original_accuracy"]),
                    arrowprops=dict(arrowstyle="-|>", color=config["colour"],
                                    linewidth=1.8, shrinkA=7, shrinkB=8), zorder=2)
        ax.scatter(row["original_confidence"], row["original_accuracy"], s=95,
                   facecolor="white", edgecolor=config["colour"], linewidth=1.9, zorder=3)
        ax.scatter(row["intervention_confidence"], row["intervention_accuracy"], s=115,
                   color=config["colour"], zorder=3)
        handles.append(Line2D([], [], marker="o", linestyle="",
                              markerfacecolor=config["colour"],
                              markeredgecolor=config["colour"], markersize=9,
                              label=f"{config['display']} ({config['intervention']})"))

    handles += [Line2D([], [], marker="o", linestyle="", markerfacecolor="white",
                       markeredgecolor="#555555", markeredgewidth=1.9, markersize=9,
                       label="open = original (zero-shot)"),
                Line2D([], [], marker="o", linestyle="", markerfacecolor="#555555",
                       markeredgecolor="#555555", markersize=9,
                       label="filled = after intervention")]

    ax.set_xlim(lo, hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel("Mean confidence (mean maximum predicted probability)")
    ax.set_ylabel("Accuracy (top-label hit rate)")
    ax.set_title("M6 holdout: overconfidence before and after each intervention\n"
                 "points to the right of the diagonal are overconfident",
                 fontsize=11.5, pad=12)
    ax.legend(handles=handles, frameon=False, fontsize=8.8, loc="upper right")
    ax.grid(True, alpha=0.25, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(COMBINED_VISUALS / "m6_confidence_vs_accuracy.png", dpi=200,
                facecolor="white")
    plt.close(fig)


def main() -> int:
    schedule = pd.read_csv(SCHEDULE_PATH)
    ground_truth = build_ground_truth(load_prices(), schedule)
    print(f"Ground truth built for {ground_truth['round'].nunique()} rounds "
          f"from the official M6 price file\n")

    scored = {}
    for key, config in MODELS.items():
        print(f"{config['display']} ({config['intervention']}, {config['label']})")
        scored[key] = score_model(key, config, ground_truth)
        print(f"   overall {scored[key]['summary']['overall_rps']:.6f}\n")

    originals, diagnostics = load_originals()
    comparison, by_round = build_comparison(scored, originals, diagnostics)

    COMBINED_RESULTS.mkdir(parents=True, exist_ok=True)
    COMBINED_VISUALS.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(COMBINED_RESULTS / "combined_m6_summary.csv", index=False)
    by_round.round(8).to_csv(COMBINED_RESULTS / "combined_m6_rps_by_round.csv", index=False)

    plot_overall_rps(comparison)
    plot_round_changes(by_round)
    plot_confidence_accuracy(comparison)

    print("M6 holdout comparison")
    print(comparison[["model", "intervention", "original_rps", "intervention_rps",
                      "rps_change", "rounds_improved",
                      "intervention_beats_naive"]].to_string(index=False))
    print(f"\nWrote {COMBINED_RESULTS.relative_to(REPO_ROOT)} and "
          f"{COMBINED_VISUALS.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
