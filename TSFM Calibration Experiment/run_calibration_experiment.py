"""Post-hoc calibration sensitivity analysis for the three TSFMs.

Each model's existing 500-scenario quintile probabilities are shrunk towards the
uniform forecast, p_lambda = (1 - lambda) * p + lambda * (0.2, ..., 0.2), and
rescored over the same 12 M6 rounds for 21 values of lambda. Lambda 0 is the
original model, lambda 1 is the naive equal-probability benchmark, and
intermediate values reduce confidence while preserving the ordering of each
row's five probabilities.

This is a sensitivity analysis on the evaluation rounds themselves, not a tuned
out-of-sample calibration model. No model inference is rerun: the saved
scenarios are read in place and the sweep is pure arithmetic.

Outputs:
    results/calibration_sweep.csv          one row per model and lambda
    results/calibration_rps_by_round.csv   one row per model, lambda and round
    tables/calibration_summary.csv         one row per model
    tables/calibration_summary_table.png   the rendered summary table
    figures/calibration_rps_vs_lambda.png  the main figure (also .pdf)

    python "TSFM Calibration Experiment/run_calibration_experiment.py"
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from evaluate_m6_rps import (  # noqa: E402
    RANK_COLUMNS, build_ground_truth, load_prices, quintile_probabilities,
    rps_scores,
)
from exploratory_evaluation import (  # noqa: E402
    ECE_BINS_PRIMARY, expected_calibration_error,
)

SCHEDULE_PATH = REPO_ROOT / "Data" / "metadata" / "m6_round_schedule.csv"
SWEEP_CSV = HERE / "results" / "calibration_sweep.csv"
BY_ROUND_CSV = HERE / "results" / "calibration_rps_by_round.csv"
SUMMARY_CSV = HERE / "tables" / "calibration_summary.csv"
SUMMARY_PNG = HERE / "tables" / "calibration_summary_table.png"
FIGURE_STEM = HERE / "figures" / "calibration_rps_vs_lambda"

N_SCENARIOS = 500
N_ROUNDS = 12
N_ASSETS = 100
N_QUINTILES = 5
NAIVE_RPS = 0.16
UNIFORM = 1.0 / N_QUINTILES

# 0.00, 0.05, ..., 1.00. Built from integers so the grid points are exact.
LAMBDAS = np.arange(0, 21) / 20.0

# The same common 500-scenario sources the horizon experiment reads.
SOURCES = [
    ("Chronos-T5 Base 200M",
     "TSFM Inference and Sampling/scenario_outputs/chronos_t5_base/chronos_base_round{:02d}_samples500.npz",
     "forecast_samples"),
    ("Financial Chronos 46M",
     "TSFM Inference and Sampling/scenario_outputs/financial_chronos/financial_chronos_round{:02d}_samples500.npz",
     "forecast_samples"),
    ("Chronos-2 120M (independent sampling)",
     "TSFM Inference and Sampling/scenario_outputs/chronos2/chronos_2_round{:02d}_scenarios.npz",
     "scenarios"),
]

# Established 500-scenario results that lambda = 0 must reproduce.
BASELINE_RPS = {
    "Chronos-T5 Base 200M": 0.226920,
    "Financial Chronos 46M": 0.177947,
    "Chronos-2 120M (independent sampling)": 0.165927,
}
TOLERANCE = 1e-6

COLOURS = {"Chronos-2 120M (independent sampling)": "#1f77b4",
           "Financial Chronos 46M": "#2ca02c",
           "Chronos-T5 Base 200M": "#d62728"}
SHORT = {"Chronos-2 120M (independent sampling)": "Chronos-2",
         "Financial Chronos 46M": "Financial Chronos",
         "Chronos-T5 Base 200M": "Chronos-T5 Base"}


def collect_forecasts(template: str, key: str, ground_truth: pd.DataFrame
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read one model's 12 saved rounds and convert them to M6 probabilities.

    Returns the stacked (1200, 5) probabilities, the matching (1200, 5) realised
    quintile memberships, and the (1200,) round labels. The probability
    construction is the project's own quintile_probabilities, so these are the
    same numbers that produce the published 500-scenario result.
    """
    probabilities, actual, rounds = [], [], []

    for round_number in range(1, N_ROUNDS + 1):
        path = REPO_ROOT / template.format(round_number)
        with np.load(path, allow_pickle=False) as data:
            # Chronos-2 stores 1,000 scenarios; the first 500 are the common set.
            samples = data[key][:, :N_SCENARIOS, :].astype(np.float64)
            symbols = [str(s) for s in data["asset_symbols"]]
        if samples.shape != (N_ASSETS, N_SCENARIOS, 20):
            raise ValueError(f"{path.name}: shape {samples.shape}")
        if not np.isfinite(samples).all():
            raise ValueError(f"{path.name}: non-finite returns")

        block = quintile_probabilities(samples)

        # The NPZ asset order and the ground truth order need not agree.
        truth = (ground_truth[ground_truth["round"] == round_number]
                 .set_index("symbol").loc[symbols, RANK_COLUMNS].to_numpy(dtype=float))

        probabilities.append(block)
        actual.append(truth)
        rounds.append(np.full(N_ASSETS, round_number))

    return (np.vstack(probabilities), np.vstack(actual), np.concatenate(rounds))


def shrink(probabilities: np.ndarray, lam: float) -> np.ndarray:
    """Move every row a fraction lambda of the way towards the uniform forecast."""
    return (1.0 - lam) * probabilities + lam * UNIFORM


def summarise(model: str, sweep: pd.DataFrame) -> dict[str, object]:
    """One summary row: where this model's RPS curve bottoms out on the grid."""
    block = sweep[sweep["model"] == model]
    original = float(block[block["lambda"] == 0.0]["overall_rps"].iloc[0])
    best = block.loc[block["overall_rps"].idxmin()]
    return {
        "model": model,
        "original_rps": round(original, 6),
        "original_ece": round(float(block[block["lambda"] == 0.0]["ece"].iloc[0]), 6),
        "best_observed_lambda": round(float(best["lambda"]), 2),
        "best_observed_rps": round(float(best["overall_rps"]), 6),
        "rps_improvement": round(original - float(best["overall_rps"]), 6),
        "difference_from_naive_at_best": round(float(best["overall_rps"]) - NAIVE_RPS, 6),
        # strictly better than the benchmark somewhere other than lambda = 1
        "beats_naive_at_any_lambda": bool(
            (block[block["lambda"] < 1.0]["overall_rps"] < NAIVE_RPS - 1e-12).any()),
    }


def main() -> int:
    # Realised quintiles come from the official price file, exactly as the
    # headline scorer builds them.
    prices = load_prices()
    schedule = pd.read_csv(SCHEDULE_PATH)
    ground_truth = build_ground_truth(prices, schedule)

    sweep_rows, round_rows = [], []

    for label, template, key in SOURCES:
        probabilities, actual, rounds = collect_forecasts(template, key, ground_truth)
        if probabilities.shape != (N_ROUNDS * N_ASSETS, N_QUINTILES):
            raise ValueError(f"{label}: {probabilities.shape} forecast rows")

        # ECE compares the top-label prediction against the realised quintile.
        true_class = actual.argmax(axis=1)

        for lam in LAMBDAS:
            shrunk = shrink(probabilities, lam)
            if not np.allclose(shrunk.sum(axis=1), 1.0, atol=1e-12):
                raise ValueError(f"{label} lambda={lam}: rows do not sum to 1")

            scores = rps_scores(actual, shrunk)
            by_round = pd.Series(scores).groupby(rounds).mean()
            if len(by_round) != N_ROUNDS:
                raise ValueError(f"{label} lambda={lam}: {len(by_round)} rounds")

            # M6 reports the mean of the 12 round means.
            overall = float(by_round.mean())

            safe = np.where(shrunk > 0, shrunk, 1.0)
            sweep_rows.append({
                "model": label,
                "lambda": round(float(lam), 2),
                "overall_rps": round(overall, 8),
                "mean_max_probability": round(float(shrunk.max(axis=1).mean()), 6),
                "mean_entropy": round(float((-(safe * np.log(safe)).sum(axis=1)).mean()), 6),
                f"ece_{ECE_BINS_PRIMARY}bin": round(
                    expected_calibration_error(shrunk, true_class, ECE_BINS_PRIMARY), 6),
            })
            for round_number, value in by_round.items():
                round_rows.append({"model": label, "lambda": round(float(lam), 2),
                                   "round": int(round_number),
                                   "rps": round(float(value), 8)})

        start = sweep_rows[-len(LAMBDAS)]["overall_rps"]
        end = sweep_rows[-1]["overall_rps"]
        print(f"{label}\n  lambda 0.00 {start:.6f}   lambda 1.00 {end:.6f}")

    sweep = pd.DataFrame(sweep_rows).rename(columns={f"ece_{ECE_BINS_PRIMARY}bin": "ece"})
    by_round = pd.DataFrame(round_rows)

    # --- sanity checks before anything is written -------------------------
    for model, expected in BASELINE_RPS.items():
        got = float(sweep[(sweep["model"] == model) & (sweep["lambda"] == 0.0)]["overall_rps"].iloc[0])
        if abs(got - expected) > TOLERANCE:
            raise ValueError(f"{model}: lambda 0 gives {got:.6f}, expected {expected:.6f}")
        naive = float(sweep[(sweep["model"] == model) & (sweep["lambda"] == 1.0)]["overall_rps"].iloc[0])
        if abs(naive - NAIVE_RPS) > 1e-9:
            raise ValueError(f"{model}: lambda 1 gives {naive:.8f}, expected {NAIVE_RPS}")
    print("\nlambda 0 reproduces every published 500-scenario result; "
          "lambda 1 reproduces the naive benchmark")

    SWEEP_CSV.parent.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(SWEEP_CSV, index=False)
    by_round.to_csv(BY_ROUND_CSV, index=False)

    summary = pd.DataFrame([summarise(label, sweep) for label, _, _ in SOURCES])
    summary = summary.sort_values("best_observed_rps").reset_index(drop=True)
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_CSV, index=False)

    build_figure(sweep, summary)
    build_summary_table(summary)

    print()
    for row in summary.itertuples():
        print(f"  {SHORT[row.model]:<18} original {row.original_rps:.6f} -> "
              f"{row.best_observed_rps:.6f} at lambda {row.best_observed_lambda:.2f} "
              f"({row.rps_improvement:+.6f}), "
              f"{'beats' if row.beats_naive_at_any_lambda else 'never beats'} naive")
    print(f"\nWrote {SWEEP_CSV.relative_to(REPO_ROOT)}, "
          f"{BY_ROUND_CSV.relative_to(REPO_ROOT)}, {SUMMARY_CSV.relative_to(REPO_ROOT)}")
    return 0


def build_figure(sweep: pd.DataFrame, summary: pd.DataFrame) -> None:
    """The main figure: overall M6 RPS against the shrinkage weight lambda."""
    FIGURE_STEM.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.axhline(NAIVE_RPS, color="#555555", linestyle="--", linewidth=1.1, zorder=1)
    # sits in the empty band below the benchmark, clear of every curve
    ax.text(0.02, NAIVE_RPS - 0.0019, "Naive equal-probability benchmark, RPS = 0.160",
            fontsize=8.5, color="#555555", va="center")

    best = summary.set_index("model")
    for label, _, _ in SOURCES:
        block = sweep[sweep["model"] == label].sort_values("lambda")
        ax.plot(block["lambda"], block["overall_rps"], color=COLOURS[label],
                linewidth=1.9, label=SHORT[label], zorder=3)
        # a small open marker at the lowest point on the grid
        ax.plot(best.loc[label, "best_observed_lambda"],
                best.loc[label, "best_observed_rps"],
                marker="o", markersize=6.5, markerfacecolor="white",
                markeredgecolor=COLOURS[label], markeredgewidth=1.6, zorder=4)

    # The minima sit fractions of a thousandth below the benchmark, which is
    # invisible at the scale of the main axes. The inset shows how small the
    # sub-benchmark region actually is, so the dip is neither missed nor
    # mistaken for a material improvement.
    inset = ax.inset_axes([0.44, 0.30, 0.52, 0.34])
    inset.set_facecolor("white")
    inset.axhline(NAIVE_RPS, color="#555555", linestyle="--", linewidth=1.0)
    for label, _, _ in SOURCES:
        block = sweep[sweep["model"] == label].sort_values("lambda")
        inset.plot(block["lambda"], block["overall_rps"], color=COLOURS[label],
                   linewidth=1.5)
        inset.plot(best.loc[label, "best_observed_lambda"],
                   best.loc[label, "best_observed_rps"],
                   marker="o", markersize=5, markerfacecolor="white",
                   markeredgecolor=COLOURS[label], markeredgewidth=1.4)
    inset.set_xlim(0.60, 1.005)
    inset.set_ylim(0.15925, 0.16075)
    inset.set_yticks([0.1595, 0.1600, 0.1605])
    inset.tick_params(labelsize=7.5)
    inset.set_title("detail: $\\lambda$ = 0.60 to 1.00", fontsize=8, pad=4)
    inset.grid(True, alpha=0.25, linewidth=0.5)

    ax.set_xlabel("Shrinkage weight $\\lambda$   (0 = original model, 1 = uniform)")
    ax.set_ylabel("Overall M6 RPS (lower is better)")
    ax.set_title("Reducing TSFM forecast confidence: M6 RPS under uniform "
                 "probability shrinkage", fontsize=11, pad=12)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.1566, None)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, loc="upper right", fontsize=9.5)

    fig.tight_layout()
    for suffix in (".png", ".pdf"):
        fig.savefig(FIGURE_STEM.with_suffix(suffix), dpi=200, facecolor="white")
    plt.close(fig)


def build_summary_table(summary: pd.DataFrame) -> None:
    """Render the summary CSV as a clean table image beside the figure."""
    table = pd.DataFrame({
        "Model": [SHORT[m] for m in summary["model"]],
        "Original RPS": summary["original_rps"].map("{:.6f}".format),
        "Best observed $\\lambda$": summary["best_observed_lambda"].map("{:.2f}".format),
        "Best observed RPS": summary["best_observed_rps"].map("{:.6f}".format),
        "RPS improvement": summary["rps_improvement"].map("{:+.6f}".format),
        "Difference vs naive": summary["difference_from_naive_at_best"].map("{:+.6f}".format),
    })

    fig, ax = plt.subplots(figsize=(10.6, 1.9))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.axis("off")

    rendered = ax.table(cellText=table.to_numpy(), colLabels=table.columns,
                        cellLoc="center", loc="center")
    rendered.auto_set_font_size(False)
    rendered.set_fontsize(9)
    rendered.scale(1, 1.45)
    for (row, _), cell in rendered.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if row == 0:
            cell.set_facecolor("#ececec")
            cell.set_text_props(weight="bold")
        else:
            cell.set_facecolor("white")

    ax.set_title("Calibration sensitivity on the M6 evaluation rounds",
                 fontsize=11, pad=14)
    fig.tight_layout()
    fig.savefig(SUMMARY_PNG, dpi=200, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
