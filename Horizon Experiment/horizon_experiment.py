"""TSFM forecast-horizon experiment: RPS at 1, 5, 10, 15 and 20 weekdays.

Only the 20-day horizon is the official M6 specification; 1, 5, 10 and 15 days
are additional experiments outside it. 15 was added later as an intermediate
diagnostic point between 10 and 20.

All three models are read from their existing saved outputs at a common 500
scenarios. The same 500 full 20-day trajectories are truncated to the first h
days for every horizon, so the horizons are nested rather than separate runs.
No model inference is rerun.

    python "Horizon Experiment/horizon_experiment.py"
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from evaluate_m6_rps import (  # noqa: E402
    RANK_COLUMNS, build_ground_truth, load_prices, price_on_or_before,
    quintile_probabilities, rank_to_quintiles, rps_scores,
)

SCHEDULE_PATH = REPO_ROOT / "Data" / "metadata" / "m6_round_schedule.csv"
BY_ROUND_CSV = HERE / "results" / "horizon_rps_by_round.csv"
SUMMARY_CSV = HERE / "results" / "horizon_rps_summary.csv"

HORIZONS = (1, 5, 10, 15, 20)
N_SCENARIOS = 500

# (label, path template, array key) - all read in place, nothing is copied.
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

# Existing 500-scenario 20-day results, from the sampling experiment.
VALIDATION_20D = {
    "Chronos-T5 Base 200M": 0.226920,
    "Financial Chronos 46M": 0.177947,
    "Chronos-2 120M (independent sampling)": 0.165927,
}
TOLERANCE = 1e-6


def realised(prices, origin: str, end: str, symbols):
    """Quintile memberships of the realised return from origin to end.

    Also returns the number of distinct returns: at short horizons a
    non-trading end date can leave most assets tied at exactly 0, which makes
    the target itself near-degenerate.
    """
    open_price = price_on_or_before(prices, origin)
    close_price = price_on_or_before(prices, end)
    returns = ((close_price - open_price) / open_price).reindex(symbols)
    if returns.isna().any():
        raise ValueError(f"{origin}..{end}: missing realised returns")
    return rank_to_quintiles(returns.to_numpy()), int(returns.nunique())


def main() -> int:
    prices = load_prices()
    schedule = pd.read_csv(SCHEDULE_PATH)
    official_truth = build_ground_truth(prices, schedule)
    origins = dict(zip(schedule["round"], schedule["origin_date"]))
    official_ends = dict(zip(schedule["round"], schedule["forecast_end_date"]))

    rows = []
    for label, template, key in SOURCES:
        print(f"{label}")
        for round_number in range(1, 13):
            path = REPO_ROOT / template.format(round_number)
            with np.load(path, allow_pickle=False) as data:
                samples = data[key][:, :N_SCENARIOS, :].astype(np.float64)
                symbols = [str(s) for s in data["asset_symbols"]]
                dates = [str(d) for d in data["forecast_dates"]]
            if samples.shape != (100, N_SCENARIOS, 20):
                raise ValueError(f"{path.name}: shape {samples.shape}")
            if not np.isfinite(samples).all():
                raise ValueError(f"{path.name}: non-finite returns")

            origin = origins[round_number]
            if dates[-1] != official_ends[round_number]:
                raise ValueError(f"round {round_number}: day 20 is {dates[-1]}, "
                                 f"expected {official_ends[round_number]}")

            for horizon in HORIZONS:
                end = dates[horizon - 1]
                if not pd.Timestamp(origin) < pd.Timestamp(end) <= pd.Timestamp(dates[-1]):
                    raise ValueError(f"round {round_number} h={horizon}: bad end {end}")

                actual, distinct = realised(prices, origin, end, symbols)
                if horizon == 20:
                    # ties this horizon back to the official ground truth
                    reference = (official_truth[official_truth["round"] == round_number]
                                 .set_index("symbol").loc[symbols, RANK_COLUMNS].to_numpy())
                    if not np.allclose(actual, reference, atol=1e-12):
                        raise ValueError(f"round {round_number}: 20-day realised "
                                         "quintiles differ from the official ground truth")

                probabilities = quintile_probabilities(samples[:, :, :horizon])
                if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-9):
                    raise ValueError(f"round {round_number} h={horizon}: probabilities")

                uniform = np.full_like(probabilities, 0.2)
                naive = float(rps_scores(actual, uniform).mean())
                rows.append({
                    "model": label,
                    "horizon_days": horizon,
                    "official_m6_horizon": horizon == 20,
                    "round": round_number,
                    "origin_date": origin,
                    "horizon_end_date": end,
                    "n_scenarios": N_SCENARIOS,
                    "mean_rps": round(float(rps_scores(actual, probabilities).mean()), 8),
                    "naive_rps": round(naive, 8),
                    "distinct_realised_returns": distinct,
                    # heavy ties in the target pull the naive benchmark off 0.16
                    "degenerate_ground_truth": abs(naive - 0.16) > 1e-6,
                })
            print(f"  round {round_number:02d}: "
                  + "  ".join(f"h={h} {rows[-len(HORIZONS) + i]['mean_rps']:.6f}"
                              for i, h in enumerate(HORIZONS)))

    by_round = pd.DataFrame(rows)
    summary = (by_round.groupby(["model", "horizon_days", "official_m6_horizon"],
                                as_index=False)[["mean_rps", "naive_rps"]].mean()
               .round(8).sort_values(["model", "horizon_days"]))
    summary = summary.rename(columns={"mean_rps": "overall_mean_rps",
                                      "naive_rps": "naive_mean_rps"})
    summary["difference_vs_naive"] = (summary["overall_mean_rps"]
                                      - summary["naive_mean_rps"]).round(8)
    clean = (by_round[~by_round["degenerate_ground_truth"]]
             .groupby(["model", "horizon_days"])
             .agg(overall_mean_rps_excl_degenerate=("mean_rps", "mean"),
                  rounds_used=("round", "size")).round(8).reset_index())
    summary = summary.merge(clean, on=["model", "horizon_days"], how="left")

    BY_ROUND_CSV.parent.mkdir(parents=True, exist_ok=True)
    by_round.to_csv(BY_ROUND_CSV, index=False)
    summary.to_csv(SUMMARY_CSV, index=False)

    print("\nOverall mean RPS across the 12 rounds")
    print(summary.pivot(index="model", columns="horizon_days",
                        values="overall_mean_rps").round(6).to_string())

    degenerate = by_round.loc[by_round["degenerate_ground_truth"],
                              ["horizon_days", "round", "horizon_end_date",
                               "distinct_realised_returns", "naive_rps"]].drop_duplicates()
    if not degenerate.empty:
        print("\nNear-degenerate targets (heavy ties; naive benchmark is not 0.16):")
        print(degenerate.to_string(index=False))
        print("Excluding them:")
        print(summary.pivot(index="model", columns="horizon_days",
                            values="overall_mean_rps_excl_degenerate")
              .round(6).to_string())

    print("\n20-day validation against the existing 500-scenario results")
    failed = False
    for label, expected in VALIDATION_20D.items():
        achieved = float(summary.loc[(summary["model"] == label)
                                     & (summary["horizon_days"] == 20),
                                     "overall_mean_rps"].iloc[0])
        delta = achieved - expected
        ok = abs(delta) <= TOLERANCE
        failed |= not ok
        print(f"  {label:40s} {achieved:.6f} vs {expected:.6f} "
              f"({delta:+.2e}) {'ok' if ok else 'MISMATCH'}")
    if failed:
        print("STOP: the 20-day horizon does not reproduce the existing 500-scenario "
              "results. Nothing was changed; investigate before using these numbers.")
        return 1

    print(f"\nwrote {BY_ROUND_CSV.relative_to(REPO_ROOT)} ({len(by_round)} rows)")
    print(f"wrote {SUMMARY_CSV.relative_to(REPO_ROOT)} ({len(summary)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
