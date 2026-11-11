"""Score one model's temperature and top-k validation grid.

Reads the 108 saved validation forecasts (18 sampling configurations across 6
pre-M6 origins), converts each to M6 quintile probabilities, scores them with
the project's own RPS, ranks the configurations by mean validation RPS and
records the winner. Selection uses the validation origins only: no M6 outcome
is read.

Scoring reuses evaluate_m6_rps.py unchanged, and ECE reuses the definition in
exploratory_evaluation.py.

Inputs  model_outputs/<model>/validation/temp*_topk*/*.npz,
        Data/validation_contexts/validation_context_manifest.csv,
        Data/processed/dataset_a_adjusted_close_weekday.csv
Outputs results/<model>/validation/ and visuals/<model>/validation/

Usage: python "TSFM Uncertainty Experiment/analysis/score_validation.py"
       --model {financial_chronos,chronos_t5_base}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXPERIMENT_DIR = HERE.parent
REPO_ROOT = EXPERIMENT_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from evaluate_m6_rps import (  # noqa: E402
    price_on_or_before, quintile_probabilities, rank_to_quintiles, load_prices,
    rps_scores, sampled_four_week_returns,
)
from exploratory_evaluation import (  # noqa: E402
    ECE_BINS_PRIMARY, expected_calibration_error,
)

MANIFEST = REPO_ROOT / "Data" / "validation_contexts" / "validation_context_manifest.csv"
DATASET_A_PRICES = (REPO_ROOT / "Data" / "processed"
                    / "dataset_a_adjusted_close_weekday.csv")
SCHEDULE_PATH = REPO_ROOT / "Data" / "metadata" / "m6_round_schedule.csv"
# The two models differ only in their checkpoint, file prefix and display name.
MODELS = {
    "financial_chronos": {
        "model_id": "FinText/Chronos_Small_2021_Global",
        "display_name": "Financial Chronos",
    },
    "chronos_t5_base": {
        "model_id": "amazon/chronos-t5-base",
        "display_name": "Chronos-T5 Base",
    },
}

# Set by main() once the model is chosen.
MODEL_KEY = MODEL_ID = DISPLAY_NAME = OUTPUT_PREFIX = ""
VALIDATION_DIR = RESULTS_DIR = VISUALS_DIR = Path()


def configure(model_key: str) -> None:
    """Point the module's paths and labels at one model."""
    global MODEL_KEY, MODEL_ID, DISPLAY_NAME, OUTPUT_PREFIX
    global VALIDATION_DIR, RESULTS_DIR, VISUALS_DIR
    MODEL_KEY = model_key
    MODEL_ID = MODELS[model_key]["model_id"]
    DISPLAY_NAME = MODELS[model_key]["display_name"]
    OUTPUT_PREFIX = model_key
    VALIDATION_DIR = EXPERIMENT_DIR / "model_outputs" / model_key / "validation"
    RESULTS_DIR = EXPERIMENT_DIR / "results" / model_key / "validation"
    VISUALS_DIR = EXPERIMENT_DIR / "visuals" / model_key / "validation"

TEMPERATURES = [1.0, 1.5, 2.0, 3.0, 3.5, 5.0]
TOP_K_LABELS = ["50", "500", "full"]
DEFAULT_CONFIG = (1.0, "50")          # stock Chronos sampling settings

N_ASSETS = 100
N_SAMPLES = 500
HORIZON = 20
NAIVE_RPS = 0.16

COLOURS = {"selected": "#2ca02c", "default": "#777777"}


def samples_path(temperature: float, top_k: str, validation_id: int) -> Path:
    """Location of one saved forecast, matching the notebook's naming."""
    folder = VALIDATION_DIR / f"temp{temperature:.1f}_topk{top_k}"
    return folder / (f"{OUTPUT_PREFIX}_validation{validation_id:02d}"
                     f"_temp{temperature:.1f}_topk{top_k}_samples{N_SAMPLES}.npz")


def load_validation_prices() -> pd.DataFrame:
    """Dataset A adjusted closes, the price source for the validation targets.

    The official M6 price file only begins on 2022-01-31, so it cannot reach the
    validation origins, the earliest of which is 2021-12-31. Dataset A is the
    project's own weekday adjusted-close panel and covers the whole window.
    check_target_source_agreement below confirms the two agree on the M6 rounds.
    """
    panel = pd.read_csv(DATASET_A_PRICES, parse_dates=["date"]).set_index("date")
    if panel.shape[1] != N_ASSETS:
        raise ValueError(f"{DATASET_A_PRICES}: {panel.shape[1]} symbols")
    return panel.sort_index()


def check_target_source_agreement(dataset_a: pd.DataFrame) -> None:
    """Confirm Dataset A reproduces the official M6 realised quintiles.

    The two price sources use different conventions and their raw returns differ
    slightly, but the M6 target is the quintile an asset lands in, not the return
    itself. This asserts the quintile memberships match on all 12 scored rounds
    before Dataset A is trusted for the validation origins.
    """
    official = load_prices()
    schedule = pd.read_csv(SCHEDULE_PATH)
    symbols = sorted(official.columns)
    for record in schedule.itertuples():
        origin, end = record.origin_date, record.forecast_end_date
        official_returns = ((price_on_or_before(official, end)
                             - price_on_or_before(official, origin))
                            / price_on_or_before(official, origin)).reindex(symbols)
        dataset_returns = ((price_on_or_before(dataset_a, end)
                            - price_on_or_before(dataset_a, origin))
                           / price_on_or_before(dataset_a, origin)).reindex(symbols)
        if not np.array_equal(rank_to_quintiles(official_returns.to_numpy()),
                              rank_to_quintiles(dataset_returns.to_numpy())):
            raise ValueError(
                f"Round {record.round}: Dataset A and the official price file "
                "disagree on the realised quintiles; the validation target "
                "source cannot be trusted."
            )
    print(f"Dataset A reproduces the official realised quintiles on all "
          f"{len(schedule)} M6 rounds")


def realised_quintiles(prices: pd.DataFrame, origin: str, end: str,
                       symbols: list[str]) -> np.ndarray:
    """Quintile membership of the realised return from origin to end.

    Uses the same on-or-before price rule as the M6 ground truth, so the target
    is built the same way and is independent of any model output.
    """
    open_price = price_on_or_before(prices, origin)
    close_price = price_on_or_before(prices, end)
    returns = ((close_price - open_price) / open_price).reindex(symbols)
    if returns.isna().any():
        missing = sorted(returns.index[returns.isna()])
        raise ValueError(f"{origin}..{end}: missing realised returns for {missing}")
    return rank_to_quintiles(returns.to_numpy())


def score_one(path: Path, actual: np.ndarray) -> dict[str, float]:
    """Post-process one saved forecast and return its origin-level metrics."""
    with np.load(path, allow_pickle=False) as data:
        samples = data["forecast_samples"].astype(np.float64)
        symbols = [str(s) for s in data["asset_symbols"]]
        metadata = json.loads(str(data["run_metadata"]))

    if samples.shape != (N_ASSETS, N_SAMPLES, HORIZON):
        raise ValueError(f"{path.name}: shape {samples.shape}")
    if not np.isfinite(samples).all():
        raise ValueError(f"{path.name}: non-finite forecast values")

    # Steps 1 to 4: daily log returns -> 20-day cumulative -> per-scenario
    # cross-sectional ranking -> averaged quintile probabilities.
    probabilities = quintile_probabilities(samples)
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-9):
        raise ValueError(f"{path.name}: probabilities do not sum to 1")

    # Steps 5 and 6: realised quintiles and the project's RPS.
    scores = rps_scores(actual, probabilities)

    # Supporting diagnostics, using the project's existing definitions.
    confidence = probabilities.max(axis=1)
    true_class = actual.argmax(axis=1)
    safe = np.where(probabilities > 0, probabilities, 1.0)
    four_week = sampled_four_week_returns(samples)

    return {
        "rps": float(scores.mean()),
        "accuracy": float((probabilities.argmax(axis=1) == true_class).mean()),
        "mean_max_probability": float(confidence.mean()),
        "mean_entropy": float((-(safe * np.log(safe)).sum(axis=1)).mean()),
        "ece": expected_calibration_error(probabilities, true_class, ECE_BINS_PRIMARY),
        # mean across assets of the spread of 20-day returns across samples,
        # the same quantity the shared scorer calls within_asset_sample_spread
        "cumulative_return_sample_spread": float(np.mean(np.std(four_week, axis=1))),
        "resolved_top_k": int(metadata["resolved_top_k"]),
    }


def build_summary(by_origin: pd.DataFrame) -> pd.DataFrame:
    """One row per configuration, ranked by mean validation RPS.

    Also carries the paired per-origin comparison against the default sampling
    settings: the same six origins scored both ways, so the difference is
    matched rather than an average of two independent means.
    """
    # Explicit names: a blanket "mean_" prefix would give mean_mean_entropy.
    metric_names = {
        "rps": "mean_rps",
        "accuracy": "mean_accuracy",
        "mean_max_probability": "mean_confidence",
        "mean_entropy": "mean_entropy",
        "ece": "mean_ece",
        "cumulative_return_sample_spread": "mean_sample_spread",
    }
    summary = (by_origin.groupby(["temperature", "top_k"], as_index=False)
               [list(metric_names)].mean().rename(columns=metric_names))
    spread = (by_origin.groupby(["temperature", "top_k"], as_index=False)["rps"]
              .agg(rps_std="std", rps_min="min", rps_max="max"))
    summary = summary.merge(spread, on=["temperature", "top_k"])

    # Paired difference against the default, origin by origin.
    default = (by_origin[(by_origin["temperature"] == DEFAULT_CONFIG[0])
                         & (by_origin["top_k"] == DEFAULT_CONFIG[1])]
               .set_index("validation_id")["rps"])
    paired = []
    for (temperature, top_k), block in by_origin.groupby(["temperature", "top_k"]):
        diff = block.set_index("validation_id")["rps"] - default
        paired.append({
            "temperature": temperature, "top_k": top_k,
            "mean_diff_vs_default": float(diff.mean()),
            "origins_better_than_default": int((diff < 0).sum()),
            "beats_naive": bool(block["rps"].mean() < NAIVE_RPS),
        })
    summary = summary.merge(pd.DataFrame(paired), on=["temperature", "top_k"])
    summary = summary.sort_values("mean_rps").reset_index(drop=True)
    summary.insert(0, "rank", summary.index + 1)
    return summary


def heatmap(summary: pd.DataFrame) -> None:
    """Mean validation RPS for every temperature and top-k combination."""
    grid = summary.pivot(index="temperature", columns="top_k", values="mean_rps")
    grid = grid.loc[TEMPERATURES, TOP_K_LABELS]

    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    fig.patch.set_facecolor("white")
    image = ax.imshow(grid.to_numpy(), cmap="viridis_r", aspect="auto")

    best = summary.iloc[0]
    for i, temperature in enumerate(TEMPERATURES):
        for j, top_k in enumerate(TOP_K_LABELS):
            value = grid.loc[temperature, top_k]
            is_best = (temperature == best["temperature"] and top_k == best["top_k"])
            ax.text(j, i, f"{value:.5f}", ha="center", va="center", fontsize=9,
                    color="white" if value > grid.to_numpy().mean() else "black",
                    fontweight="bold" if is_best else "normal")
            if is_best:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                           edgecolor="#d62728", linewidth=2.5))

    ax.set_xticks(range(len(TOP_K_LABELS)))
    ax.set_xticklabels(["top-k 50", "top-k 500", "full vocabulary"])
    ax.set_yticks(range(len(TEMPERATURES)))
    ax.set_yticklabels([f"{t:.1f}" for t in TEMPERATURES])
    ax.set_xlabel("top-k")
    ax.set_ylabel("temperature")
    ax.set_title(f"{DISPLAY_NAME}: mean validation RPS\n"
                 "six pre-M6 origins, lower is better", fontsize=11, pad=12)
    fig.colorbar(image, ax=ax, label="mean validation RPS", shrink=0.85)
    fig.tight_layout()
    fig.savefig(VISUALS_DIR / "validation_rps_heatmap.png", dpi=200, facecolor="white")
    plt.close(fig)


def per_origin_plot(by_origin: pd.DataFrame, selected: tuple[float, str]) -> None:
    """Selected configuration against the default, origin by origin."""
    def block(config):
        return (by_origin[(by_origin["temperature"] == config[0])
                          & (by_origin["top_k"] == config[1])]
                .sort_values("validation_id"))

    chosen, default = block(selected), block(DEFAULT_CONFIG)
    ids = chosen["validation_id"].to_numpy()
    width = 0.38

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    fig.patch.set_facecolor("white")
    ax.bar(ids - width / 2, default["rps"], width, label=
           f"default (temp {DEFAULT_CONFIG[0]:.1f}, top-k {DEFAULT_CONFIG[1]})",
           color=COLOURS["default"])
    ax.bar(ids + width / 2, chosen["rps"], width, label=
           f"selected (temp {selected[0]:.1f}, top-k {selected[1]})",
           color=COLOURS["selected"])
    ax.axhline(NAIVE_RPS, color="#d62728", linestyle="--", linewidth=1.1)
    ax.text(0.6, NAIVE_RPS, " naive 0.160", color="#d62728", fontsize=8.5, va="bottom")

    ax.set_xticks(ids)
    ax.set_xticklabels([f"{i}\n{o}" for i, o in
                        zip(ids, chosen["forecast_origin"])], fontsize=8.5)
    ax.set_xlabel("validation origin")
    ax.set_ylabel("RPS (lower is better)")
    ax.set_title(f"{DISPLAY_NAME}: selected sampling settings against the default",
                 fontsize=11, pad=12)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(VISUALS_DIR / "selected_vs_default_by_origin.png", dpi=200,
                facecolor="white")
    plt.close(fig)


def uncertainty_plot(summary: pd.DataFrame) -> None:
    """How RPS moves with forecast confidence across the grid."""
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    fig.patch.set_facecolor("white")

    markers = {"50": "o", "500": "s", "full": "^"}
    for top_k in TOP_K_LABELS:
        block = summary[summary["top_k"] == top_k].sort_values("temperature")
        ax.plot(block["mean_confidence"], block["mean_rps"],
                marker=markers[top_k], markersize=6, linewidth=1.4,
                label=f"top-k {top_k}")
        for row in block.itertuples():
            ax.annotate(f"{row.temperature:.1f}",
                        (row.mean_confidence, row.mean_rps),
                        textcoords="offset points", xytext=(6, 3), fontsize=7.5,
                        color="#555555")

    ax.axhline(NAIVE_RPS, color="#d62728", linestyle="--", linewidth=1.1)
    ax.text(ax.get_xlim()[0], NAIVE_RPS, " naive 0.160", color="#d62728",
            fontsize=8.5, va="bottom")
    ax.set_xlabel("mean maximum predicted probability (confidence)")
    ax.set_ylabel("mean validation RPS (lower is better)")
    ax.set_title(f"{DISPLAY_NAME}: confidence against RPS across the sampling grid\n"
                 "point labels are temperature", fontsize=11, pad=12)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(VISUALS_DIR / "confidence_vs_rps.png", dpi=200, facecolor="white")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=sorted(MODELS),
                        help="which model's validation grid to score")
    model_key = parser.parse_args().model
    configure(model_key)
    print(f"Scoring {DISPLAY_NAME} ({MODEL_ID})")

    manifest = pd.read_csv(MANIFEST).set_index("validation_id")
    prices = load_validation_prices()
    check_target_source_agreement(prices)

    # Realised targets, built once per origin and shared by all 18 configurations.
    with np.load(samples_path(*DEFAULT_CONFIG, 1), allow_pickle=False) as data:
        symbols = [str(s) for s in data["asset_symbols"]]

    targets = {}
    for validation_id, row in manifest.iterrows():
        targets[validation_id] = realised_quintiles(
            prices, row["context_end"], row["target_end"], symbols)
    print(f"Built realised quintiles for {len(targets)} validation origins")

    rows = []
    for temperature in TEMPERATURES:
        for top_k in TOP_K_LABELS:
            for validation_id in manifest.index:
                path = samples_path(temperature, top_k, validation_id)
                metrics = score_one(path, targets[validation_id])
                rows.append({
                    "temperature": temperature, "top_k": top_k,
                    "validation_id": int(validation_id),
                    "forecast_origin": manifest.loc[validation_id, "context_end"],
                    "target_start": manifest.loc[validation_id, "target_start"],
                    "target_end": manifest.loc[validation_id, "target_end"],
                    **metrics,
                })
            block = [r for r in rows if r["temperature"] == temperature
                     and r["top_k"] == top_k]
            print(f"  temp {temperature:.1f} top-k {top_k:<4}: "
                  f"mean RPS {np.mean([r['rps'] for r in block]):.6f}")

    by_origin = pd.DataFrame(rows)
    if len(by_origin) != len(TEMPERATURES) * len(TOP_K_LABELS) * len(manifest):
        raise ValueError(f"Expected 108 scored rows, got {len(by_origin)}")

    summary = build_summary(by_origin)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    VISUALS_DIR.mkdir(parents=True, exist_ok=True)
    by_origin.round(8).to_csv(RESULTS_DIR / "validation_by_origin.csv", index=False)
    summary.round(8).to_csv(RESULTS_DIR / "validation_summary.csv", index=False)

    # --- selection: lowest mean validation RPS -----------------------------
    best = summary.iloc[0]
    runner_up = summary.iloc[1]
    default = summary[(summary["temperature"] == DEFAULT_CONFIG[0])
                      & (summary["top_k"] == DEFAULT_CONFIG[1])].iloc[0]
    selected = (float(best["temperature"]), str(best["top_k"]))

    selected_config = {
        "model_key": MODEL_KEY,
        "model_id": MODEL_ID,
        "temperature": float(best["temperature"]),
        "top_k_label": str(best["top_k"]),
        "resolved_top_k": int(by_origin[
            (by_origin["temperature"] == best["temperature"])
            & (by_origin["top_k"] == best["top_k"])]["resolved_top_k"].iloc[0]),
        "top_p": 1.0,
        "num_samples": N_SAMPLES,
        "prediction_length": HORIZON,
        "context_length": 512,
        "random_seed": 42,
        "selection_metric": "mean RPS across the six pre-M6 validation origins",
        "selection_value": round(float(best["mean_rps"]), 8),
        "rps_std_across_origins": round(float(best["rps_std"]), 8),
        "default_config": {"temperature": DEFAULT_CONFIG[0],
                           "top_k_label": DEFAULT_CONFIG[1],
                           "mean_rps": round(float(default["mean_rps"]), 8)},
        "improvement_vs_default": round(float(default["mean_rps"] - best["mean_rps"]), 8),
        "origins_better_than_default": int(best["origins_better_than_default"]),
        "runner_up": {"temperature": float(runner_up["temperature"]),
                      "top_k_label": str(runner_up["top_k"]),
                      "mean_rps": round(float(runner_up["mean_rps"]), 8)},
        "winning_margin": round(float(runner_up["mean_rps"] - best["mean_rps"]), 8),
        "m6_outcomes_used_in_selection": False,
    }
    with (RESULTS_DIR / "selected_config.json").open("w", encoding="utf-8") as fh:
        json.dump(selected_config, fh, indent=2)
        fh.write("\n")

    heatmap(summary)
    per_origin_plot(by_origin, selected)
    uncertainty_plot(summary)

    # --- report ------------------------------------------------------------
    print("\nTop 5 configurations by mean validation RPS:")
    print(summary.head(5)[["rank", "temperature", "top_k", "mean_rps", "rps_std",
                           "mean_diff_vs_default", "origins_better_than_default"]]
          .to_string(index=False))
    print(f"\nSelected: temperature {selected[0]:.1f}, top-k {selected[1]} "
          f"(resolved {selected_config['resolved_top_k']})")
    print(f"  mean validation RPS {best['mean_rps']:.6f} "
          f"vs default {default['mean_rps']:.6f} "
          f"(improvement {selected_config['improvement_vs_default']:+.6f})")
    print(f"  better than default on {best['origins_better_than_default']}/6 origins")
    print(f"  winning margin over runner-up: {selected_config['winning_margin']:.6f}")
    print(f"\nWrote {RESULTS_DIR.relative_to(REPO_ROOT)} and "
          f"{VISUALS_DIR.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
