"""M6 quintile post-processing and RPS evaluation.

Each time-series model was run ahead of time and its sampled log-return
trajectories saved as NPZ files. This script turns those trajectories into M6
quintile probability forecasts, rebuilds the realised quintiles from the
official price file, and scores both with the Ranked Probability Score.

Models are discovered automatically from
``TSFM Inference and Sampling/inference_outputs/<model>/*_round??_samples.npz``,
so adding a model folder is enough to include it.

Run:  python scripts/evaluate_m6_rps.py
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = REPO_ROOT / "Evaluation Results"
# Saved TSFM inference outputs, one folder per model.
INFERENCE_DIR = REPO_ROOT / "TSFM Inference and Sampling" / "inference_outputs"
PRICE_PATH = REPO_ROOT / "Data" / "raw" / "m6_official" / "assets_m6.csv"
SCHEDULE_PATH = REPO_ROOT / "Data" / "metadata" / "m6_round_schedule.csv"

# The M6 competition: 100 assets, 12 four-week rounds, five return quintiles.
N_QUINTILES = 5
EXPECTED_ROUNDS = list(range(1, 13))
EXPECTED_ASSETS = 100
EXPECTED_SAMPLES = 100
EXPECTED_HORIZON = 20

NAIVE_LABEL = "Naive equal-probability benchmark"
RANK_COLUMNS = [f"Rank{i}" for i in range(1, N_QUINTILES + 1)]


@dataclass
class ModelArtifacts:
    """Where one model's saved forecasts live on disk."""

    key: str                 # directory name, e.g. Chronos_T5_Base_200M
    label: str               # human label used in result tables
    directory: Path          # the model's inference-output folder
    files: dict[int, Path]   # round number -> NPZ path


def discover_models(inference_dir: Path = INFERENCE_DIR) -> list[ModelArtifacts]:
    """Find every model folder holding per-round sample NPZ files."""
    pattern = re.compile(r"round(\d{2})_samples\.npz$", re.IGNORECASE)
    models = []
    for model_dir in sorted(p for p in inference_dir.glob("*") if p.is_dir()):
        files = {}
        for path in sorted(model_dir.glob("*.npz")):
            match = pattern.search(path.name)
            if match:
                files[int(match.group(1))] = path
        if not files:
            continue
        models.append(ModelArtifacts(
            key=model_dir.name,
            label=model_dir.name.replace("_", " "),
            directory=model_dir,
            files=dict(sorted(files.items())),
        ))
    return models


def rank_to_quintiles(values: np.ndarray, n_quintiles: int = N_QUINTILES) -> np.ndarray:
    """Cross-sectional returns -> (n_assets, 5) quintile membership summing to 1.

    Assets are ranked lowest-to-highest. An asset with a unique return falls in
    exactly one quintile and gets a one-hot row; a block of tied assets shares
    the rank slots it occupies and takes the average membership of those slots.

    Tie handling differs slightly from the official M6 scorer (in how the
    quintile boundaries are derived, and in how a tied block spanning a boundary
    is split), but this was checked against the actual data and makes no
    difference to the reported results.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError("rank_to_quintiles expects a 1-D array of returns")
    if np.isnan(values).any():
        raise ValueError("rank_to_quintiles received NaN returns")
    n = values.size

    # "min" ranking means tied assets all share the lowest slot of their block.
    positions = pd.Series(values).rank(method="min").to_numpy()

    # Map each 1-based rank slot to its quintile: q covers int(n*q/5)+1 .. int(n*(q+1)/5).
    edges = [int(n * q / n_quintiles) for q in range(n_quintiles + 1)]
    slot_quintile = np.empty(n + 1, dtype=int)
    for q in range(n_quintiles):
        slot_quintile[edges[q] + 1: edges[q + 1] + 1] = q

    membership = np.zeros((n, n_quintiles), dtype=float)
    for position in np.unique(positions):
        tied = np.flatnonzero(positions == position)
        slots = np.arange(int(position), int(position) + tied.size)
        membership[tied] = np.bincount(
            slot_quintile[slots], minlength=n_quintiles) / tied.size
    return membership


def load_prices(path: Path = PRICE_PATH) -> pd.DataFrame:
    """Load the official M6 price file into a date x symbol matrix."""
    raw = pd.read_csv(path)
    expected = {"symbol", "date", "price"}
    if not expected.issubset(raw.columns):
        raise ValueError(f"{path} must contain columns {sorted(expected)}")
    raw["date"] = pd.to_datetime(raw["date"], format="%Y/%m/%d")
    return raw.pivot(index="date", columns="symbol", values="price").sort_index()


def price_on_or_before(prices: pd.DataFrame, date: str) -> pd.Series:
    """Most recent price at or before ``date``, following the official rule.

    Forward filling over the whole history matters for an asset that stops
    trading mid-competition (DRE, acquired 2022-10-03): it keeps its last traded
    price instead of dropping out of the universe. No other filling is applied.
    """
    window = prices.loc[prices.index <= pd.Timestamp(date)]
    if window.empty:
        raise ValueError(f"No prices at or before {date}")
    return window.ffill().iloc[-1]


def build_ground_truth(prices: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    """Realised four-week returns and the quintile each asset actually landed in.

    This is built straight from the official price file rather than from any
    model output, so it is the same target for every model.
    """
    rows = []
    for record in schedule.itertuples():
        start, end = record.origin_date, record.forecast_end_date
        open_price = price_on_or_before(prices, start)
        close_price = price_on_or_before(prices, end)
        returns = ((close_price - open_price) / open_price).sort_index()
        if returns.isna().any():
            missing = sorted(returns.index[returns.isna()])
            raise ValueError(f"Round {record.round}: missing realised returns for {missing}")

        membership = rank_to_quintiles(returns.to_numpy())
        positions = pd.Series(returns.to_numpy()).rank(method="min").to_numpy()
        n_tied = int(len(positions) - len(np.unique(positions)))

        for i, symbol in enumerate(returns.index):
            rows.append({
                "round": int(record.round),
                "symbol": symbol,
                "start_date": start,
                "end_date": end,
                "actual_return": float(returns.iloc[i]),
                **{col: float(membership[i, q]) for q, col in enumerate(RANK_COLUMNS)},
                "tied_assets_in_round": n_tied,
            })
    return pd.DataFrame(rows)


def sampled_four_week_returns(samples: np.ndarray) -> np.ndarray:
    """(assets, samples, 20) daily log returns -> (assets, samples) simple returns.

    Daily log returns add up over the 20-day horizon, so the four-week simple
    return is ``exp(sum) - 1``.
    """
    if samples.ndim != 3:
        raise ValueError(f"Expected a 3-D sample array, got shape {samples.shape}")
    return np.expm1(samples.astype(np.float64).sum(axis=2))


def quintile_probabilities(samples: np.ndarray) -> np.ndarray:
    """(assets, samples, 20) -> (assets, 5) M6 quintile probabilities.

    Each sampled future is a complete scenario, so it is ranked across all 100
    assets on its own and converted to quintile membership. Averaging those
    memberships over the samples gives the probability of each quintile.
    """
    four_week = sampled_four_week_returns(samples)
    n_assets, n_samples = four_week.shape
    totals = np.zeros((n_assets, N_QUINTILES), dtype=float)
    for s in range(n_samples):
        totals += rank_to_quintiles(four_week[:, s])
    return totals / n_samples


def rps_scores(actual: np.ndarray, forecast: np.ndarray) -> np.ndarray:
    """Per-asset RPS: mean squared error between the cumulative quintile vectors.

    Working on cumulative probabilities is what makes RPS reward forecasts that
    are close to the right quintile, not just exactly right.
    """
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    if actual.shape != forecast.shape:
        raise ValueError(f"Shape mismatch: actual {actual.shape} vs forecast {forecast.shape}")
    diff = np.cumsum(actual, axis=-1) - np.cumsum(forecast, axis=-1)
    return np.mean(diff ** 2, axis=-1)


def load_round_samples(path: Path) -> tuple[np.ndarray, list[str], list[str]]:
    """Read one round's saved trajectories, asset symbols and forecast dates."""
    with np.load(path, allow_pickle=False) as data:
        missing = {"forecast_samples", "asset_symbols", "forecast_dates"} - set(data.files)
        if missing:
            raise ValueError(f"{path.name} is missing array(s): {sorted(missing)}")
        return (data["forecast_samples"],
                [str(s) for s in data["asset_symbols"]],
                [str(d) for d in data["forecast_dates"]])


def evaluate_model(model: ModelArtifacts, ground_truth: pd.DataFrame,
                   schedule: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Score one model over all 12 rounds.

    Returns per-asset quintile probabilities, per-asset RPS, and a per-round
    dispersion diagnostic.
    """
    prob_rows, rps_rows, diag_rows = [], [], []
    schedule_by_round = schedule.set_index("round")

    for round_number in EXPECTED_ROUNDS:
        if round_number not in model.files:
            raise FileNotFoundError(f"{model.key}: no NPZ for round {round_number}")
        path = model.files[round_number]
        samples, symbols, dates = load_round_samples(path)

        # Guard against a mismatched or corrupted NPZ before it reaches the scoring.
        if samples.shape != (EXPECTED_ASSETS, EXPECTED_SAMPLES, EXPECTED_HORIZON):
            raise ValueError(f"{path.name}: shape {samples.shape} != "
                             f"({EXPECTED_ASSETS}, {EXPECTED_SAMPLES}, {EXPECTED_HORIZON})")
        if not np.isfinite(samples).all():
            raise ValueError(f"{path.name}: contains non-finite forecast values")
        expected = schedule_by_round.loc[round_number]
        if dates[0] != expected.forecast_start_date or dates[-1] != expected.forecast_end_date:
            raise ValueError(
                f"{path.name}: forecast dates {dates[0]}..{dates[-1]} do not match round "
                f"{round_number} ({expected.forecast_start_date}..{expected.forecast_end_date})"
            )

        truth_round = ground_truth[ground_truth["round"] == round_number]
        if sorted(symbols) != sorted(truth_round["symbol"]):
            raise ValueError(f"{path.name}: asset universe differs from the ground truth")

        probabilities = quintile_probabilities(samples)
        if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-9):
            raise ValueError(f"{path.name}: quintile probabilities do not sum to 1")
        if probabilities.min() < -1e-12 or probabilities.max() > 1 + 1e-12:
            raise ValueError(f"{path.name}: quintile probabilities outside [0, 1]")

        # The NPZ and the ground truth may list assets in different orders.
        truth_aligned = truth_round.set_index("symbol").loc[symbols]
        actual = truth_aligned[RANK_COLUMNS].to_numpy(dtype=float)
        scores = rps_scores(actual, probabilities)

        index = {"model": model.label, "round": round_number, "symbol": symbols}
        prob_rows.append(pd.DataFrame({**index, **dict(zip(RANK_COLUMNS, probabilities.T))}))
        rps_rows.append(pd.DataFrame({**index, "RPS": scores}))

        # Diagnostic only; does not affect the RPS results. When the spread of median returns
        # across assets is large relative to the spread within each asset's own
        # samples, the ranking barely changes between samples and the resulting
        # probabilities come out close to one-hot.
        four_week = sampled_four_week_returns(samples)
        cross_asset = float(np.std(np.median(four_week, axis=1)))
        within_asset = float(np.mean(np.std(four_week, axis=1)))
        safe = np.where(probabilities > 0, probabilities, 1.0)
        diag_rows.append({
            "model": model.label,
            "round": round_number,
            "mean_max_probability": float(probabilities.max(axis=1).mean()),
            "mean_probability_entropy": float((-(safe * np.log(safe)).sum(axis=1)).mean()),
            "max_possible_entropy": float(np.log(N_QUINTILES)),
            "cross_asset_median_spread": cross_asset,
            "within_asset_sample_spread": within_asset,
            "dispersion_ratio": cross_asset / within_asset if within_asset else np.nan,
        })

    return (pd.concat(prob_rows, ignore_index=True),
            pd.concat(rps_rows, ignore_index=True),
            pd.DataFrame(diag_rows))


def naive_benchmark_rps(ground_truth: pd.DataFrame) -> pd.DataFrame:
    """Score the flat [0.2]*5 forecast, the benchmark every model is judged against."""
    actual = ground_truth[RANK_COLUMNS].to_numpy(dtype=float)
    flat = np.full_like(actual, 1.0 / N_QUINTILES)
    return pd.DataFrame({
        "model": NAIVE_LABEL,
        "round": ground_truth["round"].to_numpy(),
        "symbol": ground_truth["symbol"].to_numpy(),
        "RPS": rps_scores(actual, flat),
    })


def main() -> None:
    # Everything this script produces lands in one folder.
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    # Fail early and clearly if the two hand-maintained inputs are missing.
    for required in (PRICE_PATH, SCHEDULE_PATH):
        if not required.is_file():
            raise FileNotFoundError(f"Required input not found: {required}")

    # The schedule defines the 12 competition rounds and their forecast windows.
    schedule = pd.read_csv(SCHEDULE_PATH)
    if sorted(schedule["round"]) != EXPECTED_ROUNDS:
        raise ValueError(f"Round schedule does not contain rounds 1-12: {list(schedule['round'])}")

    # Each model is a folder of saved forecasts; a model is only scored if all
    # 12 rounds are present, so a partial run cannot produce a partial result.
    models = discover_models()
    if not models:
        raise FileNotFoundError(f"No model folders with sample NPZs found under {INFERENCE_DIR}")
    for model in models:
        if sorted(model.files) != EXPECTED_ROUNDS:
            raise FileNotFoundError(
                f"{model.key}: expected 12 round NPZ files, found rounds {sorted(model.files)}")
    print(f"Discovered {len(models)} model(s):")
    for model in models:
        print(f"  {model.key}: {len(model.files)} rounds in {model.directory.relative_to(REPO_ROOT)}")

    # Build the realised outcome once from the official prices, so every model is
    # scored against exactly the same target.
    prices = load_prices()
    ground_truth = build_ground_truth(prices, schedule)
    if not np.allclose(ground_truth[RANK_COLUMNS].sum(axis=1), 1.0, atol=1e-12):
        raise ValueError("Ground-truth quintile rows do not sum to 1")
    if len(ground_truth) != len(EXPECTED_ROUNDS) * EXPECTED_ASSETS:
        raise ValueError(f"Ground truth has {len(ground_truth)} rows, expected 1200")
    ground_truth.to_csv(EVAL_DIR / "m6_ground_truth_quintiles.csv", index=False)

    # Score each model in turn, saving its own outputs and collecting the scores
    # for the cross-model tables built further down.
    all_rps, all_diag = [], []
    for model in models:
        probabilities, rps_frame, diag_frame = evaluate_model(model, ground_truth, schedule)
        if len(rps_frame) != len(EXPECTED_ROUNDS) * EXPECTED_ASSETS:
            raise ValueError(f"{model.key}: expected 1200 asset-round scores, got {len(rps_frame)}")
        if rps_frame["RPS"].isna().any():
            raise ValueError(f"{model.key}: RPS contains missing values")

        probabilities.to_csv(
            EVAL_DIR / f"predicted_quintile_probabilities_{model.key}.csv", index=False)
        rps_frame.to_csv(EVAL_DIR / f"rps_by_asset_{model.key}.csv", index=False)
        all_rps.append(rps_frame)
        all_diag.append(diag_frame)

    # The naive forecast is the benchmark every model is compared against, and it
    # goes through the same scoring path rather than being assumed to equal 0.16.
    all_rps.append(naive_benchmark_rps(ground_truth))

    # Aggregate the 1,200 asset-level scores per model up to round means, then to
    # a single overall score. M6 reports the mean of the 12 round means.
    rps_all = pd.concat(all_rps, ignore_index=True)
    by_round = (rps_all.groupby(["model", "round"], as_index=False)["RPS"]
                .mean().rename(columns={"RPS": "mean_RPS"}))
    overall = (by_round.groupby("model", as_index=False)["mean_RPS"]
               .mean().rename(columns={"mean_RPS": "overall_mean_RPS"}))

    # Rounds are equally sized, so the mean of round means must also equal the
    # mean of all 1,200 scores. A mismatch would mean rounds went missing.
    flat_mean = rps_all.groupby("model")["RPS"].mean()
    for row in overall.itertuples():
        if not np.isclose(row.overall_mean_RPS, flat_mean[row.model], atol=1e-12):
            raise ValueError(f"{row.model}: mean of round means != mean of asset scores")

    by_round.to_csv(EVAL_DIR / "rps_by_round.csv", index=False)

    # Headline table: one row per model, one column per round, best model first.
    comparison = by_round.pivot(index="model", columns="round", values="mean_RPS")
    comparison.columns = [f"Round {c}" for c in comparison.columns]
    comparison = comparison.merge(
        overall.set_index("model"), left_index=True, right_index=True
    ).sort_values("overall_mean_RPS")
    comparison = comparison.rename(columns={"overall_mean_RPS": "Overall Mean RPS"})
    comparison.index.name = "Model (lower RPS is better)"
    comparison.to_csv(EVAL_DIR / "sampled_tsfm_rps_by_model.csv")

    # The same numbers transposed, which is the easier shape for plotting by round.
    long_comparison = by_round.pivot(index="round", columns="model", values="mean_RPS")
    long_comparison.index.name = "Round"
    long_comparison.to_csv(EVAL_DIR / "sampled_tsfm_rps_by_round_wide.csv")

    # Supplementary confidence/dispersion diagnostics, kept separate from the scores.
    pd.concat(all_diag, ignore_index=True).to_csv(
        EVAL_DIR / "forecast_dispersion_diagnostics.csv", index=False)

    # Finally, print the headline number for each model, best first.
    print("\nOverall mean RPS (lower is better):")
    for row in overall.sort_values("overall_mean_RPS").itertuples():
        print(f"  {row.model:<45} {row.overall_mean_RPS:.6f}")
    print(f"\nResults written to {EVAL_DIR.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
