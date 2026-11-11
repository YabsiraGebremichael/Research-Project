"""Per-asset horizon diagnostics: confidence, entropy, accuracy, ECE, Spearman.

This is the single source of truth for the horizon diagnostics: it writes the
per-asset detail (including the derived confidence, entropy and
predicted-quintile columns), the model x horizon diagnostics and the per-round
Spearman correlations. `horizon_visualisations.ipynb` only reads those CSVs.

Rows already present in the detail CSV are left alone, so a run only computes
horizons that are missing.

No model inference is rerun; the same saved 500 scenarios are read in place.

    python "Horizon Experiment/horizon_diagnostics.py"
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from evaluate_m6_rps import (  # noqa: E402
    RANK_COLUMNS, load_prices, price_on_or_before, quintile_probabilities,
    rank_to_quintiles, sampled_four_week_returns,
)
from exploratory_evaluation import (  # noqa: E402
    ECE_BINS_PRIMARY, expected_calibration_error,
)
from horizon_experiment import HORIZONS, N_SCENARIOS, SOURCES  # noqa: E402

RESULTS = HERE / "results"
DETAIL_CSV = RESULTS / "horizon_forecast_detail.csv"
DIAGNOSTICS_CSV = RESULTS / "horizon_diagnostics.csv"
SPEARMAN_CSV = RESULTS / "horizon_spearman_by_round.csv"

DEGENERATE = (4, 1)   # (round, horizon): 2022-05-30 is US Memorial Day


def build_detail(horizons) -> pd.DataFrame:
    prices = load_prices()
    schedule = pd.read_csv(REPO_ROOT / "Data" / "metadata" / "m6_round_schedule.csv")
    origins = dict(zip(schedule["round"], schedule["origin_date"]))

    records = []
    for label, template, key in SOURCES:
        for rnd in range(1, 13):
            with np.load(REPO_ROOT / template.format(rnd), allow_pickle=False) as data:
                samples = data[key][:, :N_SCENARIOS, :].astype(np.float64)
                symbols = [str(s) for s in data["asset_symbols"]]
                dates = [str(d) for d in data["forecast_dates"]]
            open_price = price_on_or_before(prices, origins[rnd])
            for h in horizons:
                realised = ((price_on_or_before(prices, dates[h - 1]) - open_price)
                            / open_price).reindex(symbols).to_numpy()
                membership = rank_to_quintiles(realised)
                probabilities = quintile_probabilities(samples[:, :, :h])
                predicted = np.median(sampled_four_week_returns(samples[:, :, :h]), axis=1)
                for i, symbol in enumerate(symbols):
                    records.append({
                        "model": label, "horizon_days": h, "round": rnd,
                        "symbol": symbol,
                        **{c: probabilities[i, q] for q, c in enumerate(RANK_COLUMNS)},
                        "median_predicted_return": predicted[i],
                        "realised_return": realised[i],
                        "realised_quintile": int(np.argmax(membership[i])) + 1,
                        "realised_membership_max": float(membership[i].max()),
                    })
        print(f"  {label}: done")
    return pd.DataFrame(records)


def main() -> int:
    detail = pd.read_csv(DETAIL_CSV)
    existing = sorted(detail["horizon_days"].unique())
    missing = [h for h in HORIZONS if h not in existing]
    print(f"detail has horizons {existing}; missing {missing}")

    if missing:
        before = len(detail)
        detail = pd.concat([detail, build_detail(missing)], ignore_index=True)
        detail = detail.sort_values(["model", "horizon_days", "round", "symbol"])
        if len(detail) != before + len(missing) * 3 * 12 * 100:
            raise ValueError(f"unexpected row count {len(detail)}")
        print(f"appended {len(detail) - before} rows -> {len(detail)} total")

    probs = detail[RANK_COLUMNS].to_numpy()
    if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-9):
        raise ValueError("probability rows do not sum to 1")
    # Derived per-asset columns are saved so the notebook can plot straight from
    # this file without recomputing anything.
    detail["confidence"] = probs.max(axis=1)
    detail["entropy"] = -(probs * np.log(np.clip(probs, 1e-12, None))).sum(axis=1)
    detail["predicted_quintile"] = probs.argmax(axis=1) + 1
    detail.to_csv(DETAIL_CSV, index=False)

    clean = detail[~((detail["round"] == DEGENERATE[0])
                     & (detail["horizon_days"] == DEGENERATE[1]))]

    rows = []
    for (model, h), block in clean.groupby(["model", "horizon_days"]):
        p = block[RANK_COLUMNS].to_numpy()
        true_class = block["realised_quintile"].to_numpy() - 1
        correct = (block["predicted_quintile"].to_numpy() - 1 == true_class)
        top = block["predicted_quintile"] == 5
        rows.append({
            "model": model, "horizon_days": h, "official_m6_horizon": h == 20,
            "n_observations": len(block),
            "mean_confidence": block["confidence"].mean(),
            "mean_entropy": block["entropy"].mean(),
            "accuracy": float(correct.mean()),
            "top_quintile_precision": float(
                (block.loc[top, "realised_quintile"] == 5).mean()) if top.any() else np.nan,
            "n_predicted_top_quintile": int(top.sum()),
            f"ece_{ECE_BINS_PRIMARY}bin": expected_calibration_error(
                p, true_class, ECE_BINS_PRIMARY),
            "confidence_minus_accuracy": float(block["confidence"].mean() - correct.mean()),
        })
    diagnostics = pd.DataFrame(rows).round(6)

    spear = []
    for (model, h, rnd), block in clean.groupby(["model", "horizon_days", "round"]):
        rho, pval = spearmanr(block["median_predicted_return"], block["realised_return"])
        spear.append({"model": model, "horizon_days": h, "round": rnd,
                      "spearman_rho": round(float(rho), 6),
                      "p_value": round(float(pval), 6)})
    spearman_by_round = pd.DataFrame(spear)

    summary = pd.read_csv(RESULTS / "horizon_rps_summary.csv")
    diagnostics = diagnostics.merge(
        spearman_by_round.groupby(["model", "horizon_days"], as_index=False)["spearman_rho"]
        .mean().rename(columns={"spearman_rho": "mean_spearman_rho"}),
        on=["model", "horizon_days"]).merge(
        summary[["model", "horizon_days", "overall_mean_rps",
                 "overall_mean_rps_excl_degenerate"]],
        on=["model", "horizon_days"])

    diagnostics.to_csv(DIAGNOSTICS_CSV, index=False)
    spearman_by_round.to_csv(SPEARMAN_CSV, index=False)

    for column in ["overall_mean_rps_excl_degenerate", "mean_confidence", "mean_entropy",
                   "accuracy", f"ece_{ECE_BINS_PRIMARY}bin", "mean_spearman_rho"]:
        print(f"\n{column}")
        print(diagnostics.pivot(index="model", columns="horizon_days",
                                values=column).round(4).to_string())

    print(f"\nwrote {DIAGNOSTICS_CSV.name} ({len(diagnostics)} rows) and "
          f"{SPEARMAN_CSV.name} ({len(spearman_by_round)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
