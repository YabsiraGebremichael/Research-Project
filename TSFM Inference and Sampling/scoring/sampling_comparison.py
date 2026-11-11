"""Combine the three models' scenario-count results into one comparison table.

Each model is scored by its own script in this folder, which writes a per-model
CSV into results/. This script only reads those three CSVs and stacks them into
a single table, so the question "does the number of sampled scenarios change the
M6 RPS?" can be read across all three models at once.

Nothing is rescored here and no model inference is rerun.

Input:  results/{chronos2,financial_chronos,chronos_t5_base}_rps_by_sample_count.csv
Output: results/all_models_rps_by_sample_count.csv

Run:  python "TSFM Inference and Sampling/scoring/sampling_comparison.py"
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SCORING_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR_ROOT = SCORING_DIR.parent
REPO_ROOT = EXPERIMENT_DIR_ROOT.parent

RESULTS_DIR = EXPERIMENT_DIR_ROOT / "results"
OUT_CSV = RESULTS_DIR / "all_models_rps_by_sample_count.csv"

# Display label -> the per-model CSV written by that model's scoring script.
MODEL_RESULTS = {
    "Chronos-2 120M (independent sampling)": "chronos2_rps_by_sample_count.csv",
    "Chronos-T5 Base 200M": "chronos_t5_base_rps_by_sample_count.csv",
    "Financial Chronos 46M": "financial_chronos_rps_by_sample_count.csv",
}
N_ROUNDS = 12
NAIVE_RPS = 0.16


def model_rows(path: Path, model: str) -> list[dict]:
    """Read one model's scenario-count CSV into the combined table's row format."""
    frame = pd.read_csv(path)
    rows = []
    for _, record in frame.iterrows():
        values = [float(record[f"Round {i} RPS"]) for i in range(1, N_ROUNDS + 1)]
        # The stored per-round values should still average to the stored overall.
        if not np.isclose(np.mean(values), record["Overall RPS"], atol=5e-7):
            raise ValueError(f"{path.name}: round values do not average to the overall RPS")
        rows.append({
            "model": model,
            "n_samples": int(record["Sample Count"]),
            **{f"round_{i:02d}_rps": round(v, 8) for i, v in enumerate(values, 1)},
            "mean_m6_rps": round(float(record["Overall RPS"]), 8),
            "source": path.name,
        })
    return rows


def main() -> None:
    # Read every model's result, failing clearly if one has not been scored yet.
    rows = []
    for model, filename in MODEL_RESULTS.items():
        path = RESULTS_DIR / filename
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} not found. Run that model's scoring script in this folder first.")
        added = model_rows(path, model)
        print(f"  {model}: N = {[r['n_samples'] for r in added]}")
        rows += added

    # Stack into one table, sorted so each model's counts read in order.
    results = pd.DataFrame(rows).sort_values(["model", "n_samples"])
    results["difference_vs_naive"] = (results["mean_m6_rps"] - NAIVE_RPS).round(8)
    results.to_csv(OUT_CSV, index=False)

    print("\nOverall M6 RPS by scenario count")
    for model, block in results.groupby("model"):
        print(f"  {model}")
        for record in block.itertuples():
            print(f"    N={record.n_samples:>5}  {record.mean_m6_rps:.6f}")
    print(f"\nWrote {OUT_CSV.relative_to(REPO_ROOT)} ({len(results)} rows)")


if __name__ == "__main__":
    main()
