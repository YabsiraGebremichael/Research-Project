"""Score the uncertainty-regularised Chronos-2 LoRA betas on the twelve M6 rounds.

Every numerical step is imported from the project's own modules, so these
results sit on the same footing as the existing Chronos-2 numbers:
evaluate_m6_rps.py for returns, ranking, tie handling, quintile probabilities
and RPS, exploratory_evaluation.py for ECE, and the Chronos-2 comparison module
for the quantile-to-scenario conversion with the same seed and prefix rule as
the published N=500 result.

Quantile crossings: a small number of adjacent quantile pairs are out of order,
and interpolating an inverse CDF needs a non-decreasing curve, so each
asset-day vector is sorted in memory before scenario generation. The NPZ files
are opened read-only. Both the raw and rearranged pipelines are scored with
identical settings and the difference is reported rather than used to choose
between them.

Baselines are read from existing artifacts: zero-shot Chronos-2 from the
Horizon Experiment at h=20, ordinary LoRA from results/chronos2_lora/m6/.

Outputs results/chronos2_uncertainty_lora/ (per-beta tables, the crossing and
raw-versus-rearranged diagnostics, combined_summary.csv) and visuals/.

Usage: python "TSFM Uncertainty Experiment/analysis/score_uncertainty_m6.py"
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

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
RETURNS_PATH = REPO_ROOT / "Data" / "processed" / "dataset_a_daily_log_returns.csv"
HORIZON_DIAGNOSTICS = REPO_ROOT / "Horizon Experiment" / "results" / "horizon_diagnostics.csv"
HORIZON_BY_ROUND = REPO_ROOT / "Horizon Experiment" / "results" / "horizon_rps_by_round.csv"
ORDINARY_LORA_DIR = EXPERIMENT_DIR / "results" / "chronos2_lora" / "m6"

MODEL_OUTPUTS = EXPERIMENT_DIR / "model_outputs" / "chronos2_uncertainty_lora"
RESULTS_DIR = EXPERIMENT_DIR / "results" / "chronos2_uncertainty_lora"
VISUALS_DIR = EXPERIMENT_DIR / "visuals" / "chronos2_uncertainty_lora"

BETAS = ["0.5", "1.0", "2.0", "4.0", "8.0", "16.0"]
N_ASSETS, N_SAMPLES, HORIZON, N_QUANTILES = 100, 500, 20, 21
NAIVE_RPS = 0.16
ZERO_SHOT_LABEL = "Chronos-2 120M (independent sampling)"

# The two intervals the training objective targeted, for coverage reporting.
INTERVALS = {(0.10, 0.90): 0.95, (0.20, 0.80): 0.75}

# light to dark as the widening pressure rises
BETA_COLOURS = {0.5: "#c6dbef", 1.0: "#9ecae1", 2.0: "#6baed6",
                4.0: "#4292c6", 8.0: "#2171b5", 16.0: "#084594"}


def npz_path(beta: str, round_number: int) -> Path:
    return (MODEL_OUTPUTS / f"beta{beta}" / "m6"
            / f"chronos2_uncertainty_lora_beta{beta}_round{round_number:02d}_quantiles.npz")


def load_raw(beta: str, round_number: int):
    """Read one saved forecast. The file is opened read-only and never written."""
    with np.load(npz_path(beta, round_number), allow_pickle=False) as data:
        return (data["quantile_forecasts"].astype(np.float64),
                data["quantile_levels"].astype(np.float64),
                [str(s) for s in data["asset_symbols"]])


def crossing_report(forecasts: np.ndarray) -> dict:
    """Count and size the adjacent quantile-order violations in one round."""
    gaps = np.diff(forecasts, axis=2)
    violations = gaps < 0
    magnitudes = -gaps[violations]
    affected = violations.any(axis=2)
    pairs = np.unique(np.argwhere(violations)[:, 2]) if violations.any() else np.array([])
    return {
        "crossings": int(violations.sum()),
        "comparisons": int(violations.size),
        "asset_days_affected": int(affected.sum()),
        "asset_days_total": int(affected.size),
        "max_magnitude": float(magnitudes.max()) if magnitudes.size else 0.0,
        "mean_magnitude": float(magnitudes.mean()) if magnitudes.size else 0.0,
        "median_magnitude": float(np.median(magnitudes)) if magnitudes.size else 0.0,
        "level_pairs": ";".join(str(int(p)) for p in pairs),
    }


def probabilities_from_quantiles(forecasts: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Quantiles to M6 quintile probabilities, the published Chronos-2 path.

    1,000 independent-percentile scenarios are drawn once at seed 42 and the
    first 500 are used, exactly as score_chronos2_scenarios.py does at N=500.
    """
    scenarios = generate_independent_scenarios(
        marginal_curves(forecasts), levels, N_SCENARIOS, RANDOM_SEED)
    if scenarios.shape != (N_ASSETS, N_SCENARIOS, HORIZON):
        raise ValueError(f"scenarios {scenarios.shape}")
    probabilities = quintile_probabilities(scenarios[:, :N_SAMPLES, :])
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-9):
        raise ValueError("probabilities do not sum to 1")
    return probabilities


def interval_diagnostics(forecasts: np.ndarray, levels: np.ndarray,
                         realised: np.ndarray) -> dict:
    """Hard interval coverage and width against the realised daily returns."""
    index = {float(l): int(np.flatnonzero(np.isclose(levels, l, atol=1e-9))[0])
             for l in (0.10, 0.20, 0.80, 0.90)}
    valid = np.isfinite(realised)
    out = {}
    for (lo, hi), target in INTERVALS.items():
        low, high = forecasts[:, :, index[lo]], forecasts[:, :, index[hi]]
        covered = (realised >= low) & (realised <= high) & valid
        key = f"Q{lo:.2f}-Q{hi:.2f}"
        out[f"{key}_coverage"] = float(covered.sum() / valid.sum())
        out[f"{key}_mean_width"] = float(np.mean((high - low)[valid]))
        out[f"{key}_target"] = target
        out[f"{key}_nominal"] = round(hi - lo, 2)
    return out


def score_beta(beta: str, ground_truth: pd.DataFrame, realised_daily: pd.DataFrame,
               forecast_windows: dict, rearrange: bool) -> dict:
    """Score one beta over all 12 rounds, raw or rearranged."""
    round_rows, probability_rows = [], []
    all_probabilities, all_actual, interval_rows = [], [], []

    for round_number in range(1, 13):
        forecasts, levels, symbols = load_raw(beta, round_number)
        if rearrange:
            # Restore the non-decreasing order an inverse CDF requires. This is
            # quantile rearrangement: the same values, sorted, nothing replaced.
            forecasts = np.sort(forecasts, axis=-1)

        truth = ground_truth[ground_truth["round"] == round_number].set_index("symbol")
        truth = truth.loc[symbols]
        actual = truth[RANK_COLUMNS].to_numpy(dtype=float)

        probabilities = probabilities_from_quantiles(forecasts, levels)
        scores = rps_scores(actual, probabilities)

        all_probabilities.append(probabilities)
        all_actual.append(actual)
        round_rows.append({"beta": float(beta), "round": round_number,
                           "mean_rps": float(scores.mean())})

        if rearrange:
            for i, symbol in enumerate(symbols):
                probability_rows.append({
                    "beta": float(beta), "round": round_number, "symbol": symbol,
                    **{c: float(probabilities[i, q]) for q, c in enumerate(RANK_COLUMNS)},
                    "RPS": float(scores[i]),
                })
            # the 20 forecast weekdays, not the origin-to-end span the
            # ground truth carries, which is 21 days long
            start, end = forecast_windows[round_number]
            dates = pd.bdate_range(start, end)
            assert len(dates) == HORIZON, f"round {round_number}: {len(dates)} dates"
            realised = realised_daily.reindex(index=dates, columns=symbols).to_numpy(float).T
            assert realised.shape == (N_ASSETS, HORIZON)
            interval_rows.append(interval_diagnostics(forecasts, levels, realised))

    by_round = pd.DataFrame(round_rows)
    probabilities = np.vstack(all_probabilities)
    actual = np.vstack(all_actual)
    true_class = actual.argmax(axis=1)
    safe = np.where(probabilities > 0, probabilities, 1.0)

    summary = {
        "beta": float(beta),
        "overall_rps": float(by_round["mean_rps"].mean()),
        "rps_std_across_rounds": float(by_round["mean_rps"].std()),
        "accuracy": float((probabilities.argmax(axis=1) == true_class).mean()),
        "mean_confidence": float(probabilities.max(axis=1).mean()),
        "mean_entropy": float((-(safe * np.log(safe)).sum(axis=1)).mean()),
        f"ece_{ECE_BINS_PRIMARY}bin": expected_calibration_error(
            probabilities, true_class, ECE_BINS_PRIMARY),
    }
    if interval_rows:
        for key in interval_rows[0]:
            summary[key] = float(np.mean([r[key] for r in interval_rows]))
    return {"summary": summary, "by_round": by_round,
            "probabilities": pd.DataFrame(probability_rows)}


# ---------------------------------------------------------------------------
# Visuals
# ---------------------------------------------------------------------------

def plot_rps_progression(table: pd.DataFrame) -> None:
    """Zero-shot, ordinary LoRA, then the three betas, against the benchmark."""
    fig, ax = plt.subplots(figsize=(11.0, 5.4))
    fig.patch.set_facecolor("white")
    labels = table["label"].tolist()
    x = np.arange(len(labels))
    colours = ["#9e9e9e", "#1f77b4"] + [BETA_COLOURS[float(b)] for b in BETAS]

    ax.bar(x, table["overall_rps"], 0.66, color=colours)
    ax.axhline(NAIVE_RPS, color="#000000", linestyle="--", linewidth=1.2)
    for i, value in enumerate(table["overall_rps"]):
        ax.text(i, value + 0.0005, f"{value:.4f}", ha="center", fontsize=8.5,
                fontweight="bold" if i >= 2 else "normal")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Overall M6 RPS (lower is better)")
    ax.set_title("Chronos-2 on the 12 M6 rounds: zero-shot, ordinary LoRA,\n"
                 "and uncertainty-regularised LoRA across beta",
                 fontsize=11.5, pad=12)
    ax.set_ylim(0.150, max(table["overall_rps"]) * 1.06)
    ax.legend(handles=[Line2D([], [], color="#000000", linestyle="--", linewidth=1.2,
                              label="naive equal-probability benchmark (0.160)")],
              frameon=False, fontsize=9, loc="upper left")
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(VISUALS_DIR / "uncertainty_rps_progression.png", dpi=200,
                facecolor="white")
    plt.close(fig)


def plot_width_vs_rps(table: pd.DataFrame) -> None:
    """Does wider actually mean better? Width and coverage against RPS."""
    rows = table[table["beta"].notna()].sort_values("beta")
    key = "Q0.10-Q0.90"
    betas = rows["beta"].to_numpy()
    widths = rows[f"{key}_mean_width"].to_numpy()
    coverage = rows[f"{key}_coverage"].to_numpy()
    rps = rows["overall_rps"].to_numpy()
    target = float(rows[f"{key}_target"].iloc[0])
    nominal = float(rows[f"{key}_nominal"].iloc[0])

    fig, (left, right) = plt.subplots(1, 2, figsize=(11.4, 4.9))
    fig.patch.set_facecolor("white")

    # left: how much widening each beta actually achieved
    left.plot(betas, coverage, marker="o", color="#08519c", linewidth=1.8,
              label="achieved coverage")
    left.axhline(target, color="#d62728", linestyle="--", linewidth=1.1,
                 label=f"training target ({target:.0%})")
    left.axhline(nominal, color="#555555", linestyle=":", linewidth=1.1,
                 label=f"nominal level ({nominal:.0%})")
    for b, c, w in zip(betas, coverage, widths):
        left.annotate(f"width {w:.3f}", (b, c), textcoords="offset points",
                      xytext=(0, 9), ha="center", fontsize=8.5, color="#333333")
    left.set_xlabel("beta (widening pressure)")
    left.set_ylabel(f"{key} empirical coverage")
    left.set_xscale("log", base=2)
    left.set_xticks(betas)
    left.set_xticklabels([f"{b:g}" for b in betas])
    left.set_title("Widening achieved against the training target", fontsize=10.5)
    left.legend(frameon=False, fontsize=8.5, loc="center right")

    # right: did the widening buy anything on M6?
    right.axhline(NAIVE_RPS, color="#000000", linestyle="--", linewidth=1.1)
    right.plot(widths, rps, marker="o", color="#08519c", linewidth=1.8, zorder=3)
    for b, w, r in zip(betas, widths, rps):
        right.annotate(f"beta {b:.1f}", (w, r), textcoords="offset points",
                       xytext=(9, -4), fontsize=8.5)
    right.set_xlabel(f"mean {key} interval width")
    right.set_ylabel("Overall M6 RPS (lower is better)")
    right.set_title("Wider intervals against M6 performance", fontsize=10.5)
    right.text(0.02, 0.06, "naive 0.160", transform=right.transAxes,
               fontsize=8.5, color="#444444")

    for ax in (left, right):
        ax.grid(True, alpha=0.25, linewidth=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.suptitle("Uncertainty regularisation: how far the intervals widened, "
                 "and what it bought", fontsize=11.5)
    fig.tight_layout()
    fig.savefig(VISUALS_DIR / "uncertainty_width_vs_rps.png", dpi=200,
                facecolor="white")
    plt.close(fig)


def plot_confidence_accuracy(table: pd.DataFrame) -> None:
    """Does the intervention move calibration without destroying accuracy?"""
    fig, ax = plt.subplots(figsize=(8.0, 5.6))
    fig.patch.set_facecolor("white")
    lo, hi = 0.21, 0.34
    ax.plot([lo, hi], [lo, hi], color="#000000", linestyle="--", linewidth=1.1)
    ax.annotate("perfect calibration\n(confidence = accuracy)", xy=(0.243, 0.243),
                xytext=(0.252, 0.222), fontsize=8.5, color="#444444",
                arrowprops=dict(arrowstyle="-", color="#888888", linewidth=0.9))

    markers = {"zero-shot": ("o", "#9e9e9e"), "ordinary LoRA": ("s", "#1f77b4")}
    handles = []
    for row in table.itertuples():
        if pd.isna(row.beta):
            marker, colour = markers[row.label.split(" (")[0]] if row.label.split(" (")[0] in markers else ("o", "#9e9e9e")
        else:
            marker, colour = "^", BETA_COLOURS[float(row.beta)]
        ax.scatter(row.mean_confidence, row.accuracy, s=130, marker=marker,
                   color=colour, zorder=3)
        handles.append(Line2D([], [], marker=marker, linestyle="", markersize=10,
                              color=colour, label=row.label))

    ax.set_xlim(lo, hi)
    ax.set_ylim(0.215, 0.285)
    ax.set_xlabel("Mean confidence (mean maximum predicted probability)")
    ax.set_ylabel("Accuracy (top-label hit rate)")
    ax.set_title("Chronos-2 variants on M6: confidence against accuracy\n"
                 "points to the right of the diagonal are overconfident",
                 fontsize=11.5, pad=12)
    ax.legend(handles=handles, frameon=False, fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.25, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(VISUALS_DIR / "uncertainty_confidence_accuracy.png", dpi=200,
                facecolor="white")
    plt.close(fig)


def main() -> int:
    schedule = pd.read_csv(SCHEDULE_PATH)
    prices = load_prices()
    ground_truth = build_ground_truth(prices, schedule)
    realised_daily = pd.read_csv(RETURNS_PATH, parse_dates=["date"]).set_index("date")
    forecast_windows = {int(r.round): (r.forecast_start_date, r.forecast_end_date)
                        for r in schedule.itertuples()}

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    VISUALS_DIR.mkdir(parents=True, exist_ok=True)

    # --- crossing diagnostic on the untouched raw forecasts ----------------
    crossing_rows = []
    for beta in BETAS:
        for round_number in range(1, 13):
            forecasts, _, _ = load_raw(beta, round_number)
            crossing_rows.append({"beta": float(beta), "round": round_number,
                                  **crossing_report(forecasts)})
    crossings = pd.DataFrame(crossing_rows)
    crossings.to_csv(RESULTS_DIR / "crossing_diagnostic.csv", index=False)
    total = int(crossings["crossings"].sum())
    print(f"Crossing diagnostic: {total} adjacent violations across "
          f"{int(crossings['comparisons'].sum()):,} comparisons "
          f"({total / crossings['comparisons'].sum():.4%})")
    print(f"  asset-day forecasts affected: {int(crossings['asset_days_affected'].sum())}"
          f"/{int(crossings['asset_days_total'].sum()):,}")
    print(f"  largest magnitude: {crossings['max_magnitude'].max():.3e}\n")

    # --- score raw and rearranged, identical settings and seed -------------
    scored_raw, scored_fixed = {}, {}
    for beta in BETAS:
        scored_raw[beta] = score_beta(beta, ground_truth, realised_daily,
                                      forecast_windows, rearrange=False)
        scored_fixed[beta] = score_beta(beta, ground_truth, realised_daily,
                                        forecast_windows, rearrange=True)
        r = scored_raw[beta]["summary"]["overall_rps"]
        f = scored_fixed[beta]["summary"]["overall_rps"]
        print(f"beta {beta}: raw {r:.6f}  rearranged {f:.6f}  diff {f - r:+.3e}")

    sensitivity = []
    for beta in BETAS:
        merged = (scored_raw[beta]["by_round"][["round", "mean_rps"]]
                  .rename(columns={"mean_rps": "raw_rps"})
                  .merge(scored_fixed[beta]["by_round"][["round", "mean_rps"]]
                         .rename(columns={"mean_rps": "rearranged_rps"}), on="round"))
        merged.insert(0, "beta", float(beta))
        merged["difference"] = merged["rearranged_rps"] - merged["raw_rps"]
        sensitivity.append(merged)
    sensitivity = pd.concat(sensitivity, ignore_index=True)
    sensitivity.round(10).to_csv(RESULTS_DIR / "raw_vs_rearranged.csv", index=False)

    # --- per-beta result files, from the rearranged pipeline ---------------
    for beta in BETAS:
        out = RESULTS_DIR / f"beta{beta}" / "m6"
        out.mkdir(parents=True, exist_ok=True)
        scored_fixed[beta]["by_round"].round(8).to_csv(
            out / "m6_rps_by_round.csv", index=False)
        scored_fixed[beta]["probabilities"].round(8).to_csv(
            out / "m6_predicted_quintile_probabilities.csv", index=False)
        pd.DataFrame([scored_fixed[beta]["summary"]]).round(8).to_csv(
            out / "m6_summary.csv", index=False)

    # --- combined table with both baselines --------------------------------
    diagnostics = pd.read_csv(HORIZON_DIAGNOSTICS)
    diagnostics = diagnostics[(diagnostics["horizon_days"] == 20)
                              & (diagnostics["model"] == ZERO_SHOT_LABEL)].iloc[0]
    by_round_zs = pd.read_csv(HORIZON_BY_ROUND)
    by_round_zs = by_round_zs[(by_round_zs["horizon_days"] == 20)
                              & (by_round_zs["model"] == ZERO_SHOT_LABEL)]
    ece_column = next(c for c in pd.read_csv(HORIZON_DIAGNOSTICS).columns
                      if c.startswith("ece_"))
    ordinary = pd.read_csv(ORDINARY_LORA_DIR / "m6_summary.csv").iloc[0]
    ordinary_rounds = pd.read_csv(ORDINARY_LORA_DIR / "m6_rps_by_round.csv")

    zero_shot_rps = float(by_round_zs["mean_rps"].mean())
    ordinary_rps = float(ordinary["overall_rps"])

    rows = [
        {"label": "zero-shot", "beta": np.nan, "overall_rps": zero_shot_rps,
         "accuracy": float(diagnostics["accuracy"]),
         "mean_confidence": float(diagnostics["mean_confidence"]),
         "mean_entropy": float(diagnostics["mean_entropy"]),
         f"ece_{ECE_BINS_PRIMARY}bin": float(diagnostics[ece_column])},
        {"label": "ordinary LoRA", "beta": np.nan, "overall_rps": ordinary_rps,
         "accuracy": float(ordinary["accuracy"]),
         "mean_confidence": float(ordinary["mean_confidence"]),
         "mean_entropy": float(ordinary["mean_entropy"]),
         f"ece_{ECE_BINS_PRIMARY}bin": float(ordinary["ece_5bin"])},
    ]
    for beta in BETAS:
        summary = dict(scored_fixed[beta]["summary"])
        summary["label"] = f"uncertainty LoRA\nbeta {beta}"
        rows.append(summary)
    table = pd.DataFrame(rows)

    # changes against both baselines, and rounds improved against each
    for beta in BETAS:
        mask = table["beta"] == float(beta)
        beta_rounds = scored_fixed[beta]["by_round"].set_index("round")["mean_rps"]
        table.loc[mask, "change_vs_zero_shot"] = (
            table.loc[mask, "overall_rps"] - zero_shot_rps)
        table.loc[mask, "change_vs_ordinary_lora"] = (
            table.loc[mask, "overall_rps"] - ordinary_rps)
        table.loc[mask, "difference_vs_naive"] = (
            table.loc[mask, "overall_rps"] - NAIVE_RPS)
        zs = by_round_zs.set_index("round")["mean_rps"]
        ol = ordinary_rounds.set_index("round")["mean_rps"]
        table.loc[mask, "rounds_improved_vs_zero_shot"] = int((beta_rounds < zs).sum())
        table.loc[mask, "rounds_improved_vs_ordinary_lora"] = int((beta_rounds < ol).sum())
        table.loc[mask, "beats_naive"] = bool(
            table.loc[mask, "overall_rps"].iloc[0] < NAIVE_RPS)
    table.round(8).to_csv(RESULTS_DIR / "combined_summary.csv", index=False)

    plot_rps_progression(table)
    plot_width_vs_rps(table)
    plot_confidence_accuracy(table)

    print("\nCombined M6 results")
    show = ["label", "overall_rps", "change_vs_zero_shot",
            "change_vs_ordinary_lora", "difference_vs_naive",
            "rounds_improved_vs_zero_shot", "accuracy", "mean_confidence",
            "mean_entropy", f"ece_{ECE_BINS_PRIMARY}bin"]
    printable = table.copy()
    printable["label"] = printable["label"].str.replace("\n", " ", regex=False)
    print(printable[show].to_string(index=False))

    best = table[table["beta"].notna()].sort_values("overall_rps").iloc[0]
    print(f"\nBest observed beta on these 12 M6 rounds: {best['beta']:.1f} "
          f"at {best['overall_rps']:.6f}")
    print("  The beta grid was evaluated directly on M6, so this is the best")
    print("  observed value on the evaluated grid, not a validated optimum.")
    print(f"\nWrote {RESULTS_DIR.relative_to(REPO_ROOT)} and "
          f"{VISUALS_DIR.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
