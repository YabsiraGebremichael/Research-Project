"""Score the Chronos-T5 Base scenario-count runs (300/500 trajectories).

The Chronos-T5 Base counterpart of score_financial_chronos_scenarios.py, which
it imports its discovery, validation and scoring helpers from so the two models
are scored identically. Only 300 and 500 trajectories were run for this model.

Input:  scenario_outputs/chronos_t5_base/*_round<NN>_samples<N>.npz
Output: results/chronos_t5_base_rps_by_sample_count.csv

Run:  python "TSFM Inference and Sampling/scoring/score_chronos_t5_base_scenarios.py"
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCORING_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR_ROOT = SCORING_DIR.parent
REPO_ROOT = EXPERIMENT_DIR_ROOT.parent
sys.path.insert(0, str(SCORING_DIR))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from score_financial_chronos_scenarios import (  # noqa: E402
    BASELINE_SAMPLES, CONTEXT_PATH, MAIN_ROUND_RPS, SCHEDULE_PATH,
    comparison_table, discover_experiment_files, score_configuration, validate_file,
)
from evaluate_m6_rps import (  # noqa: E402
    EXPECTED_ASSETS, EXPECTED_HORIZON, EXPECTED_ROUNDS, build_ground_truth, load_prices,
)

MODEL_LABEL = "Chronos T5 Base 200M"
# Cross-check only: the primary 100-sample Base result in Evaluation Results/.
EXPECTED_BASELINE_OVERALL = 0.226899

SCENARIO_DIR = EXPERIMENT_DIR_ROOT / "scenario_outputs" / "chronos_t5_base"
SAMPLE_COUNTS = [300, 500]
RESULTS_CSV = EXPERIMENT_DIR_ROOT / "results" / "chronos_t5_base_rps_by_sample_count.csv"


def load_baseline_round_rps() -> dict[int, float]:
    """Read the existing 100-trajectory Base per-round results from the primary outputs."""
    if not MAIN_ROUND_RPS.is_file():
        raise FileNotFoundError(
            f"Baseline results not found at {MAIN_ROUND_RPS}. Run scripts/evaluate_m6_rps.py first."
        )
    frame = pd.read_csv(MAIN_ROUND_RPS)
    baseline = frame[frame["model"] == MODEL_LABEL]
    if sorted(baseline["round"]) != EXPECTED_ROUNDS:
        raise ValueError(f"Baseline is missing rounds: found {sorted(baseline['round'])}")
    return dict(zip(baseline["round"], baseline["mean_RPS"]))


def main() -> None:
    # Same ground truth, schedule and asset ordering as the primary evaluation.
    schedule = pd.read_csv(SCHEDULE_PATH)
    ground_truth = build_ground_truth(load_prices(), schedule)
    official_order = list(pd.read_csv(CONTEXT_PATH, nrows=0).columns[1:])

    # Locate the scenario files and confirm the 300/500 set is complete.
    files = discover_experiment_files(SCENARIO_DIR, SAMPLE_COUNTS)
    n_files = sum(len(rounds) for rounds in files.values())
    print(f"Found {n_files} scenario NPZ files in {SCENARIO_DIR.relative_to(REPO_ROOT)}")

    # Validate each file before it is scored.
    for num_samples, rounds in files.items():
        for round_number, path in rounds.items():
            validate_file(path, round_number, num_samples, schedule, official_order)
        print(f"  {num_samples:>4} samples: 12/12 rounds valid "
              f"({EXPECTED_ASSETS}, {num_samples}, {EXPECTED_HORIZON}), finite, "
              "official asset order, dates match the round schedule")

    # Reuse the 100-trajectory baseline, and check it still matches the recorded
    # primary result so a stale or wrong baseline cannot slip into the comparison.
    by_config = {BASELINE_SAMPLES: load_baseline_round_rps()}
    baseline_overall = float(np.mean(list(by_config[BASELINE_SAMPLES].values())))
    if not np.isclose(baseline_overall, EXPECTED_BASELINE_OVERALL, atol=5e-6):
        raise ValueError(
            f"Baseline overall RPS {baseline_overall:.6f} does not match the recorded "
            f"primary Base result {EXPECTED_BASELINE_OVERALL}"
        )
    print(f"\nBaseline: {BASELINE_SAMPLES} samples = {baseline_overall:.6f}, read from "
          f"{MAIN_ROUND_RPS.relative_to(REPO_ROOT)}")

    # Score the larger configurations.
    for num_samples, rounds in files.items():
        by_config[num_samples] = score_configuration(rounds, ground_truth)
        print(f"  scored {num_samples} samples: overall "
              f"{np.mean(list(by_config[num_samples].values())):.6f}")

    # Build and save the comparison table.
    table = comparison_table(by_config, baseline_overall)
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULTS_CSV, index=False)

    # Report each configuration against the 100-sample baseline.
    overall_by_config = dict(zip(table["Sample Count"], table["Overall RPS"]))
    best = min(overall_by_config, key=overall_by_config.get)
    print("\nOverall RPS (lower is better):")
    for num_samples in sorted(overall_by_config):
        mark = "  <-- best" if num_samples == best else ""
        print(f"  {num_samples:>4} samples: {overall_by_config[num_samples]:.6f} "
              f"({overall_by_config[num_samples] - baseline_overall:+.6f} vs 100){mark}")
    print(f"\nWrote {RESULTS_CSV.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
