"""Score the Chronos-2 scenario-count runs (100/300/500/1000 scenarios).

Chronos-2 is the one model whose sampled scenarios were never saved, so they are
regenerated here from its saved 21-quantile forecasts using the same
independent-percentile sampling, interpolation and seed as the completed
Chronos-2 evaluation. The 1,000 scenarios are drawn once per round and saved,
then N = 100/300/500 are scored from the first N of that same set, so the only
thing changing between counts is how many scenarios are used.

No model inference is rerun.

Input:  inference_outputs/Chronos_2_120M/*_quantiles.npz
Output: scenario_outputs/chronos2/*.npz and results/chronos2_rps_by_sample_count.csv

Run:  python "TSFM Inference and Sampling/scoring/score_chronos2_scenarios.py"
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCORING_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR_ROOT = SCORING_DIR.parent
REPO_ROOT = EXPERIMENT_DIR_ROOT.parent
sys.path.insert(0, str(EXPERIMENT_DIR_ROOT / "chronos2_sampling_comparison"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from evaluate_independent_sampling import (  # noqa: E402
    N_SCENARIOS, RANDOM_SEED, generate_independent_scenarios,
)
from evaluate_spatiotemporal_copula import (  # noqa: E402
    HORIZON, N_ASSETS, discover_quantile_files, load_quantile_forecasts,
    marginal_curves,
)
from evaluate_m6_rps import (  # noqa: E402
    EXPECTED_ROUNDS, RANK_COLUMNS, build_ground_truth, load_prices,
    quintile_probabilities, rps_scores,
)

SCHEDULE_PATH = REPO_ROOT / "Data" / "metadata" / "m6_round_schedule.csv"
CONTEXT_PATH = REPO_ROOT / "Data" / "contexts" / "round_01_context.csv"

SCENARIO_DIR = EXPERIMENT_DIR_ROOT / "scenario_outputs" / "chronos2"
RESULTS_CSV = EXPERIMENT_DIR_ROOT / "results" / "chronos2_rps_by_sample_count.csv"

SAMPLE_COUNTS = [100, 300, 500, 1000]
BASELINE_SAMPLES = 100
NAIVE_RPS = 0.16
# The published Chronos-2 independent-sampling result. Regenerating the
# scenarios must reproduce it, otherwise the regeneration is wrong.
EXPECTED_1000_OVERALL = 0.165559


def score_all_counts() -> dict[int, dict[int, float]]:
    """Regenerate and save the scenarios, and score every count as {count: {round: RPS}}."""
    schedule = pd.read_csv(SCHEDULE_PATH)
    ground_truth = build_ground_truth(load_prices(), schedule)
    official_order = list(pd.read_csv(CONTEXT_PATH, nrows=0).columns[1:])
    SCENARIO_DIR.mkdir(parents=True, exist_ok=True)

    by_config: dict[int, dict[int, float]] = {n: {} for n in SAMPLE_COUNTS}
    for round_number, path in discover_quantile_files().items():
        # Rebuild scenarios from the saved quantiles rather than rerunning the model.
        forecasts, levels, symbols, dates = load_quantile_forecasts(
            path, round_number, schedule, official_order)
        scenarios = generate_independent_scenarios(
            marginal_curves(forecasts), levels, N_SCENARIOS, RANDOM_SEED)
        if scenarios.shape != (N_ASSETS, N_SCENARIOS, HORIZON):
            raise ValueError(f"round {round_number}: scenarios {scenarios.shape}")
        if not np.isfinite(scenarios).all():
            raise ValueError(f"round {round_number}: non-finite scenario returns")

        # Saved so later work can reuse them without regenerating.
        np.savez_compressed(
            SCENARIO_DIR / f"chronos_2_round{round_number:02d}_scenarios.npz",
            scenarios=scenarios, asset_symbols=symbols, forecast_dates=dates,
            quantile_levels=levels, seed=RANDOM_SEED)

        truth = ground_truth[ground_truth["round"] == round_number].set_index("symbol").loc[symbols]
        actual = truth[RANK_COLUMNS].to_numpy(dtype=float)

        # Each count uses the first N scenarios of the same draw.
        for n in SAMPLE_COUNTS:
            probabilities = quintile_probabilities(scenarios[:, :n, :])
            if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-9):
                raise ValueError(f"round {round_number}, N={n}: probabilities do not sum to 1")
            by_config[n][round_number] = float(rps_scores(actual, probabilities).mean())
        print(f"  round {round_number:02d}: "
              + "  ".join(f"N={n} {by_config[n][round_number]:.6f}" for n in SAMPLE_COUNTS))
    return by_config


def comparison_table(by_config: dict[int, dict[int, float]],
                     baseline_overall: float) -> pd.DataFrame:
    """One row per scenario count, matching the other models' result format.

    Values are stored to 8 decimals rather than the 6 used by the other two
    models: Chronos-2's scores are computed here rather than read from an
    already-rounded file, and rounding to 6 would shift the combined table in
    the 6th decimal.
    """
    rows = []
    for num_samples in sorted(by_config):
        round_rps = by_config[num_samples]
        overall = float(np.mean([round_rps[r] for r in EXPECTED_ROUNDS]))
        row = {"Sample Count": num_samples}
        row.update({f"Round {r} RPS": round(round_rps[r], 8) for r in EXPECTED_ROUNDS})
        row["Overall RPS"] = round(overall, 8)
        row["Difference vs 100 Samples"] = round(overall - baseline_overall, 6)
        row["Difference vs Naive 0.160000"] = round(overall - NAIVE_RPS, 6)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    print(f"Regenerating {N_SCENARIOS} Chronos-2 scenarios per round from the saved "
          f"quantiles (seed {RANDOM_SEED}), then scoring N = {SAMPLE_COUNTS}")
    by_config = score_all_counts()

    # Build and save the comparison table, using N=100 as the baseline column.
    baseline_overall = float(np.mean(list(by_config[BASELINE_SAMPLES].values())))
    table = comparison_table(by_config, baseline_overall)
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULTS_CSV, index=False)

    # The regenerated 1,000-scenario run must reproduce the published result.
    achieved = float(table.loc[table["Sample Count"] == 1000, "Overall RPS"].iloc[0])
    if abs(achieved - EXPECTED_1000_OVERALL) > 1e-6:
        raise ValueError(
            f"1,000-scenario overall RPS {achieved:.6f} does not reproduce the published "
            f"Chronos-2 result {EXPECTED_1000_OVERALL}. Investigate before using these numbers."
        )

    overall_by_config = dict(zip(table["Sample Count"], table["Overall RPS"]))
    best = min(overall_by_config, key=overall_by_config.get)
    print("\nOverall RPS (lower is better):")
    for num_samples in sorted(overall_by_config):
        mark = "  <-- best" if num_samples == best else ""
        print(f"  {num_samples:>4} samples: {overall_by_config[num_samples]:.6f} "
              f"({overall_by_config[num_samples] - baseline_overall:+.6f} vs 100){mark}")
    print(f"\n1,000 scenarios reproduces the published {EXPECTED_1000_OVERALL}")

    size = sum(p.stat().st_size for p in SCENARIO_DIR.glob("*.npz")) / 2**20
    print(f"Wrote {RESULTS_CSV.relative_to(REPO_ROOT)}")
    print(f"Wrote {len(list(SCENARIO_DIR.glob('*.npz')))} scenario files to "
          f"{SCENARIO_DIR.relative_to(REPO_ROOT)} ({size:.0f} MB)")


if __name__ == "__main__":
    main()
