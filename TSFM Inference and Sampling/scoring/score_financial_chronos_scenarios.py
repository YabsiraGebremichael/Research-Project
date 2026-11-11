"""Score the Financial Chronos scenario-count runs (300/500/1000 trajectories).

Answers one question: does drawing more sampled trajectories change the M6 RPS?
Each configuration is scored against the same ground truth with the same
methodology as the primary evaluation, and compared with the existing
100-trajectory result read from Evaluation Results/.

Everything methodological is imported from scripts/evaluate_m6_rps.py, so the
only difference between configurations is the number of trajectories.

Input:  scenario_outputs/financial_chronos/*_round<NN>_samples<N>.npz
Output: results/financial_chronos_rps_by_sample_count.csv

Run:  python "TSFM Inference and Sampling/scoring/score_financial_chronos_scenarios.py"
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

SCORING_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR_ROOT = SCORING_DIR.parent
REPO_ROOT = EXPERIMENT_DIR_ROOT.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from evaluate_m6_rps import (  # noqa: E402
    EXPECTED_ASSETS, EXPECTED_HORIZON, EXPECTED_ROUNDS, RANK_COLUMNS,
    build_ground_truth, load_prices, load_round_samples, quintile_probabilities,
    rps_scores,
)

SCHEDULE_PATH = REPO_ROOT / "Data" / "metadata" / "m6_round_schedule.csv"
CONTEXT_PATH = REPO_ROOT / "Data" / "contexts" / "round_01_context.csv"
MAIN_ROUND_RPS = REPO_ROOT / "Evaluation Results" / "rps_by_round.csv"
MAIN_MODEL_LABEL = "Financial Chronos Small 46M 2021 Global"

BASELINE_SAMPLES = 100
EXPECTED_SAMPLE_COUNTS = [300, 500, 1000]
NAIVE_RPS = 0.16

EXPERIMENT_DIR = EXPERIMENT_DIR_ROOT / "scenario_outputs" / "financial_chronos"
RESULTS_CSV = EXPERIMENT_DIR_ROOT / "results" / "financial_chronos_rps_by_sample_count.csv"

FILENAME_RE = re.compile(r"round(\d{2})_samples(\d+)\.npz$", re.IGNORECASE)


def discover_experiment_files(root: Path = EXPERIMENT_DIR,
                              expected_counts: list[int] | None = None,
                              ) -> dict[int, dict[int, Path]]:
    """Find the scenario NPZs under ``root`` as {num_samples: {round: path}}.

    Filenames are parsed rather than assumed, and the set of files is checked to
    be complete before anything is scored, so a half-copied experiment fails
    here instead of silently producing a partial result. ``expected_counts``
    lets the Chronos-T5 Base script reuse this with its own [300, 500].
    """
    expected_counts = list(EXPECTED_SAMPLE_COUNTS if expected_counts is None
                           else expected_counts)
    found: dict[int, dict[int, list[Path]]] = defaultdict(lambda: defaultdict(list))
    for path in sorted(root.rglob("*.npz")):
        match = FILENAME_RE.search(path.name)
        if match:
            found[int(match.group(2))][int(match.group(1))].append(path)

    if not found:
        raise FileNotFoundError(
            f"No experiment NPZ files found under {root}. Expected files named like "
            "'..._round01_samples300.npz'."
        )

    # Collect every problem first so one run reports all of them.
    problems = []
    if sorted(found) != expected_counts:
        problems.append(f"expected sample counts {expected_counts}, found {sorted(found)}")
    for num_samples in sorted(found):
        rounds = found[num_samples]
        duplicates = {r: [p.name for p in ps] for r, ps in rounds.items() if len(ps) > 1}
        if duplicates:
            problems.append(f"{num_samples} samples: duplicate files for round(s) {duplicates}")
        missing = [r for r in EXPECTED_ROUNDS if r not in rounds]
        if missing:
            problems.append(f"{num_samples} samples: missing round(s) {missing}")
        unexpected = [r for r in rounds if r not in EXPECTED_ROUNDS]
        if unexpected:
            problems.append(f"{num_samples} samples: unexpected round(s) {unexpected}")

    total = sum(len(ps) for rounds in found.values() for ps in rounds.values())
    expected_total = len(expected_counts) * len(EXPECTED_ROUNDS)
    if total != expected_total:
        problems.append(f"expected {expected_total} NPZ files, found {total}")

    if problems:
        raise FileNotFoundError(
            "Scenario experiment files are incomplete:\n  - " + "\n  - ".join(problems))

    return {n: {r: ps[0] for r, ps in sorted(rounds.items())} for n, rounds in sorted(found.items())}


def validate_file(path: Path, round_number: int, num_samples: int, schedule: pd.DataFrame,
                  official_order: list[str]) -> None:
    """Check one NPZ's shape, asset ordering, forecast dates and finiteness."""
    samples, symbols, dates = load_round_samples(path)

    expected_shape = (EXPECTED_ASSETS, num_samples, EXPECTED_HORIZON)
    if samples.shape != expected_shape:
        raise ValueError(f"{path.name}: shape {samples.shape} != {expected_shape}")
    if not np.isfinite(samples).all():
        raise ValueError(f"{path.name}: contains non-finite forecast values")
    if symbols != official_order:
        raise ValueError(f"{path.name}: asset_symbols differ from the official M6 order")

    row = schedule.set_index("round").loc[round_number]
    if len(dates) != EXPECTED_HORIZON:
        raise ValueError(f"{path.name}: {len(dates)} forecast dates, expected {EXPECTED_HORIZON}")
    if dates[0] != row.forecast_start_date or dates[-1] != row.forecast_end_date:
        raise ValueError(
            f"{path.name}: forecast dates {dates[0]}..{dates[-1]} do not match round "
            f"{round_number} ({row.forecast_start_date}..{row.forecast_end_date})"
        )


def score_configuration(files: dict[int, Path], ground_truth: pd.DataFrame) -> dict[int, float]:
    """Score one scenario count, returning {round: mean RPS}.

    Uses the imported evaluator functions unchanged, which is what makes the
    result directly comparable with the primary 100-trajectory evaluation.
    """
    round_rps = {}
    for round_number, path in files.items():
        samples, symbols, _ = load_round_samples(path)
        probabilities = quintile_probabilities(samples)
        if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-9):
            raise ValueError(f"{path.name}: quintile probabilities do not sum to 1")

        # Align the ground truth to this NPZ's asset ordering before scoring.
        truth = ground_truth[ground_truth["round"] == round_number].set_index("symbol").loc[symbols]
        actual = truth[RANK_COLUMNS].to_numpy(dtype=float)
        scores = rps_scores(actual, probabilities)
        if not np.isfinite(scores).all():
            raise ValueError(f"{path.name}: non-finite RPS values")
        round_rps[round_number] = float(scores.mean())
    return round_rps


def load_baseline_round_rps() -> dict[int, float]:
    """Read the existing 100-trajectory per-round results from the primary outputs."""
    if not MAIN_ROUND_RPS.is_file():
        raise FileNotFoundError(
            f"Baseline results not found at {MAIN_ROUND_RPS}. Run scripts/evaluate_m6_rps.py first."
        )
    frame = pd.read_csv(MAIN_ROUND_RPS)
    baseline = frame[frame["model"] == MAIN_MODEL_LABEL]
    if sorted(baseline["round"]) != EXPECTED_ROUNDS:
        raise ValueError(f"Baseline is missing rounds: found {sorted(baseline['round'])}")
    return dict(zip(baseline["round"], baseline["mean_RPS"]))


def comparison_table(by_config: dict[int, dict[int, float]],
                     baseline_overall: float) -> pd.DataFrame:
    """One row per scenario count: per-round RPS, overall, and the two differences."""
    rows = []
    for num_samples in sorted(by_config):
        round_rps = by_config[num_samples]
        overall = float(np.mean([round_rps[r] for r in EXPECTED_ROUNDS]))
        row = {"Sample Count": num_samples}
        row.update({f"Round {r} RPS": round(round_rps[r], 6) for r in EXPECTED_ROUNDS})
        row["Overall RPS"] = round(overall, 6)
        row["Difference vs 100 Samples"] = round(overall - baseline_overall, 6)
        row["Difference vs Naive 0.160000"] = round(overall - NAIVE_RPS, 6)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    # The ground truth and schedule are the same ones the primary evaluation uses.
    schedule = pd.read_csv(SCHEDULE_PATH)
    ground_truth = build_ground_truth(load_prices(), schedule)

    # The NPZs store assets in the official M6 order, taken from a round context file.
    official_order = list(pd.read_csv(CONTEXT_PATH, nrows=0).columns[1:])

    # Locate the scenario files and confirm the set is complete before scoring.
    files = discover_experiment_files()
    n_files = sum(len(rounds) for rounds in files.values())
    print(f"Found {n_files} scenario NPZ files in "
          f"{EXPERIMENT_DIR.relative_to(REPO_ROOT)}")

    # Validate every file individually, so a bad NPZ is caught before it is scored.
    for num_samples, rounds in files.items():
        for round_number, path in rounds.items():
            validate_file(path, round_number, num_samples, schedule, official_order)
        print(f"  {num_samples:>4} samples: 12/12 rounds valid "
              f"({EXPECTED_ASSETS}, {num_samples}, {EXPECTED_HORIZON}), finite, "
              "official asset order, dates match the round schedule")

    # The 100-trajectory baseline is reused from the primary evaluation rather
    # than rescored, then each larger configuration is scored here.
    by_config = {BASELINE_SAMPLES: load_baseline_round_rps()}
    print(f"\nBaseline: {BASELINE_SAMPLES} samples read from "
          f"{MAIN_ROUND_RPS.relative_to(REPO_ROOT)}")
    for num_samples, rounds in files.items():
        by_config[num_samples] = score_configuration(rounds, ground_truth)
        print(f"  scored {num_samples} samples: overall "
              f"{np.mean(list(by_config[num_samples].values())):.6f}")

    # Build and save the comparison table.
    baseline_overall = float(np.mean(list(by_config[BASELINE_SAMPLES].values())))
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
