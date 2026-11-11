"""Adapted Python reimplementation of Perez Michaus's FinQBoost M6 entry.

The model-defining methodology is preserved unchanged and untuned: the quintile
label with trailing two-weekday smoothing, the released ten-feature set,
Friday-only sampling, the XGBoost settings (depth 3, eta 0.02, 230 rounds,
subsample 0.5, colsample 0.8, seed 29) and the probability post-processing.
Each asset is paired with each candidate quintile, so the model is a binary
learner over (asset-date, quintile) pairs whose five outputs are normalised per
asset and shifted to a cross-sectional mean of 0.20.

Causality: the feature panel is built once and each origin selects from it,
which is numerically identical to recomputing on a truncated panel because
every feature uses trailing windows, backward shifts or per-date
cross-sections. A Friday may train at an origin only if its label date, twenty
weekdays later, falls on or before that origin.

The panel source is selectable: dataset_d (default) or extended.

Usage: python "FinQ Boost Reimplementation/finqboost.py" [--panel extended]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from evaluate_m6_rps import (  # noqa: E402
    RANK_COLUMNS,
    build_ground_truth,
    load_prices,
    rps_scores,
)

EXTENDED_OHLCV = (PROJECT_ROOT / "TSFM Uncertainty Experiment" / "data"
                  / "extended_eodhd_ohlcv_2000_2023.csv")
DATASET_D_OHLCV = (PROJECT_ROOT / "Data" / "raw" / "dataset_d_eodhd"
                   / "dataset_d_ohlcv.csv")
# Dataset D is the snapshot the feature baselines use; the extended file is a
# different EODHD snapshot, so truncating it does not reproduce Dataset D.
PANEL_SOURCES = {"dataset_d": DATASET_D_OHLCV, "extended": EXTENDED_OHLCV}

OFFICIAL_M6 = PROJECT_ROOT / "Data" / "raw" / "m6_official" / "assets_m6.csv"
M6_SCHEDULE = PROJECT_ROOT / "Data" / "metadata" / "m6_round_schedule.csv"

OUT_PREDICTIONS = HERE / "finqboost_m6_predictions.csv"
OUT_RPS = HERE / "finqboost_m6_rps.csv"
LOG_PATH = HERE / "finqboost_run.log"


def output_paths(panel_start: str | None) -> tuple[Path, Path, Path]:
    """Output paths for a run. `panel_start=None` keeps the original names."""
    if panel_start is None:
        return OUT_PREDICTIONS, OUT_RPS, LOG_PATH
    tag = f"_from{pd.Timestamp(panel_start).year}"
    return (HERE / f"finqboost_m6_predictions{tag}.csv",
            HERE / f"finqboost_m6_rps{tag}.csv",
            HERE / f"finqboost_run{tag}.log")

# ---- Miguel's fixed, model-defining constants ------------------------------ #
LAG_WEEKDAYS = 20            # 5*4: the four-week label horizon, in weekday rows
N_QUINTILES = 5
TARGET_SOFTEN_NDAYS = 2      # trailing mean applied to the one-hot label
RSI_N = 80
ROLLING_MONTHS = range(1, 16)     # i * 20 weekdays, i = 1..15
SEED = 29
XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "max_depth": 3,
    "eta": 0.02,
    "gamma": 0.0,
    "min_child_weight": 1,
    "max_delta_step": 0,
    "colsample_bytree": 0.8,
    "subsample": 0.5,
    "num_parallel_tree": 1,
}
NUM_BOOST_ROUND = 230

# The released final feature selection, verbatim from `keep_features`.
SELECTED_FEATURES = [
    "rollsd_pctlrank1d_windown4wmonths_3",
    "rollsd_pctlrank1d_windown4wmonths_4",
    "rollsd_pctlrank1d_windown4wmonths_5",
    "rollsd_pctlrank1d_prev_windown4wmonths_9",
    "rollsd_pctlrank1d_windown4wmonths_2",
    "rollsd_pctlrank1d_windown4wmonths_10",
    "rollsd_pctlrank1d_prev_windown4wmonths_15",
    "rollsd_pctlrank1d_windown4wmonths_13",
    "rollsd_pctlrank1d_prev_windown4wmonths_6",
    "ratio_RSI_vs_mean_RSI_close_by_date_n_80",
]
MODEL_INPUTS = SELECTED_FEATURES + ["smooth_quintile"]

NAIVE_RPS = 0.16
N_ROUNDS = 12
N_ASSETS = 100
PUBLISHED_FINQBOOST_RPS = 0.15648

logger = logging.getLogger("finqboost")


class ImplementationError(RuntimeError):
    """A correctness expectation was violated; the run stops rather than adapt."""


# --------------------------------------------------------------------------- #
# Panel
# --------------------------------------------------------------------------- #

def build_panel(path: Path = EXTENDED_OHLCV,
                panel_start: str | None = None) -> pd.DataFrame:
    """Wide adjusted-close panel on the project's shared Mon-Fri calendar.

    Forward-filled after each asset's genuine inception and never backfilled,
    which is both this project's established convention and exactly what
    Miguel's `nafill(locf)` on an all-weekday reindex does.

    DRE receives THIS PROJECT'S treatment, not Miguel's: its genuine history is
    retained and its last traded price (2022-10-03) is simply carried forward by
    the same forward fill that every other asset gets. Miguel's `freeze_DRE`,
    which overwrites DRE's ENTIRE history with a constant for origins from
    2022-09-01, is deliberately NOT implemented.
    """
    if not path.is_file():
        raise ImplementationError(f"EODHD panel source not found: {path}")
    raw = pd.read_csv(path, parse_dates=["date"])
    official = list(dict.fromkeys(
        pd.read_csv(OFFICIAL_M6)["symbol"].astype(str).str.strip()))
    wide = raw.pivot(index="date", columns="symbol", values="adjusted_close")
    missing = [s for s in official if s not in wide.columns]
    if missing:
        raise ImplementationError(f"Extended data lacks assets: {missing}")
    wide = wide.reindex(columns=official).sort_index()
    if panel_start is not None:
        # Miguel's `dt <- dt[index >= as.Date("2007-01-01")]`: the raw dated rows
        # are filtered BEFORE the weekday reindex, so the calendar starts here too.
        wide = wide.loc[pd.Timestamp(panel_start):]
        if wide.empty:
            raise ImplementationError(f"No data on or after {panel_start}.")
    calendar = pd.date_range(wide.index.min(), wide.index.max(), freq="B")
    panel = wide.reindex(calendar).ffill()
    panel.index.name = "date"
    panel.columns.name = None

    for symbol in panel.columns:
        first = panel[symbol].first_valid_index()
        if panel[symbol].loc[:first].iloc[:-1].notna().any():
            raise ImplementationError(f"{symbol}: value created before inception.")
        if panel[symbol].loc[first:].isna().any():
            raise ImplementationError(f"{symbol}: gap after inception.")
    logger.info("Panel: %d weekdays x %d assets, %s .. %s",
                panel.shape[0], panel.shape[1],
                panel.index.min().date(), panel.index.max().date())
    return panel


# --------------------------------------------------------------------------- #
# R-equivalent primitives
# --------------------------------------------------------------------------- #

def cross_sectional_percentile(frame: pd.DataFrame) -> pd.DataFrame:
    """R's `ecdf(x)(x)` applied per date across assets.

    ecdf is built on the non-NA values and evaluated at every point, so
    F(x) = #{observed <= x} / n_observed, with NA in -> NA out. The pandas
    equivalent is rank(method="max", pct=True), which counts values <= x and
    divides by the non-NA count. Verified against R in validation/.
    """
    return frame.rank(axis=1, method="max", pct=True)


def percentile_to_quintile(percentile: pd.DataFrame) -> pd.DataFrame:
    """R's `cut(pct, seq(0, 1, length.out = 6))`, labels 1..5.

    R's cut defaults to right-closed, left-open intervals and does NOT include
    the lower bound, so a percentile of exactly 0 yields NA. With 100 assets the
    smallest percentile is 0.01, so that edge is unreachable here - asserted
    below rather than assumed.
    """
    values = percentile.to_numpy()
    observed = values[np.isfinite(values)]
    if observed.size and observed.min() <= 0.0:
        raise ImplementationError("A percentile of exactly 0 occurred; R's cut() "
                                  "would return NA there and the edge case must "
                                  "be handled explicitly.")
    edges = np.linspace(0.0, 1.0, N_QUINTILES + 1)
    binned = np.full(values.shape, np.nan)
    finite = np.isfinite(values)
    binned[finite] = np.digitize(values[finite], edges[1:-1], right=True) + 1.0
    return pd.DataFrame(binned, index=percentile.index, columns=percentile.columns)


def wilder_rsi(prices: pd.DataFrame, n: int = RSI_N) -> pd.DataFrame:
    """TTR::RSI(price, n) with its default Wilder smoothing.

    TTR's default maType is EMA with wilder = TRUE, i.e. alpha = 1/n seeded by
    the simple mean of the first n changes - NOT the simple-moving-average RSI
    that most quick implementations produce. Reproduced explicitly and validated
    against TTR in validation/.
    """
    out = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    for symbol in prices.columns:
        series = prices[symbol]
        first = series.first_valid_index()
        if first is None:
            continue
        values = series.loc[first:].to_numpy(dtype=float)
        if len(values) <= n:
            continue
        change = np.diff(values)
        gain = np.where(change > 0, change, 0.0)
        loss = np.where(change < 0, -change, 0.0)

        up = np.full(len(change), np.nan)
        down = np.full(len(change), np.nan)
        up[n - 1] = gain[:n].mean()
        down[n - 1] = loss[:n].mean()
        for i in range(n, len(change)):
            up[i] = up[i - 1] + (gain[i] - up[i - 1]) / n
            down[i] = down[i - 1] + (loss[i] - down[i - 1]) / n

        with np.errstate(divide="ignore", invalid="ignore"):
            rsi = 100.0 - 100.0 / (1.0 + up / down)
        rsi = np.where(down == 0, 100.0, rsi)          # TTR: all-gain -> 100
        aligned = np.concatenate([[np.nan], rsi])       # diff drops one row
        out.loc[first:, symbol] = aligned
    return out


# --------------------------------------------------------------------------- #
# Features and labels
# --------------------------------------------------------------------------- #

def build_feature_panel(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Every selected feature, wide (dates x assets). All strictly causal."""
    features: dict[str, pd.DataFrame] = {}

    returns_1d = panel / panel.shift(1) - 1.0
    percentile_1d = cross_sectional_percentile(returns_1d)

    for i in ROLLING_MONTHS:
        width = i * LAG_WEEKDAYS
        rolling = percentile_1d.rolling(width, min_periods=width).std(ddof=1)
        features[f"rollsd_pctlrank1d_windown4wmonths_{i}"] = rolling
        features[f"rollsd_pctlrank1d_prev_windown4wmonths_{i}"] = rolling.shift(
            LAG_WEEKDAYS)

    rsi = wilder_rsi(panel, RSI_N)
    mean_rsi = rsi.mean(axis=1, skipna=True)
    features["ratio_RSI_vs_mean_RSI_close_by_date_n_80"] = rsi.div(mean_rsi, axis=0)
    return {name: features[name] for name in SELECTED_FEATURES}


def build_soft_targets(panel: pd.DataFrame) -> tuple[dict[int, pd.DataFrame],
                                                     pd.DataFrame]:
    """The 2-weekday-smoothed one-hot quintile label, and the label date.

    The smoothing is a TRAILING mean over two consecutive WEEKDAYS of the daily
    panel - not two Fridays - applied before any Friday sampling, exactly as in
    the source. A soft target therefore takes values in {0, 0.5, 1}.

    `label_date` records, for each row, the date whose price closes the forward
    window (t + 20 weekdays). Because the smoothing also reaches one row back,
    the information a smoothed target depends on extends to the LATER of that
    row's and the previous row's label date - handled by the caller.
    """
    forward = panel.shift(-LAG_WEEKDAYS) / panel - 1.0
    quintile = percentile_to_quintile(cross_sectional_percentile(forward))

    soft: dict[int, pd.DataFrame] = {}
    for k in range(1, N_QUINTILES + 1):
        one_hot = (quintile == k).astype(float).where(quintile.notna())
        soft[k] = one_hot.rolling(TARGET_SOFTEN_NDAYS,
                                  min_periods=TARGET_SOFTEN_NDAYS).mean()

    positions = np.arange(len(panel)) + LAG_WEEKDAYS
    label_dates = np.where(positions < len(panel),
                           panel.index.to_numpy()[np.clip(positions, 0,
                                                          len(panel) - 1)],
                           np.datetime64("NaT"))
    label_date = pd.Series(label_dates, index=panel.index, name="label_date")
    return soft, label_date


def assemble_long(panel: pd.DataFrame) -> pd.DataFrame:
    """Fridays only, five rows per (date, asset), features + soft target."""
    features = build_feature_panel(panel)
    soft, label_date = build_soft_targets(panel)

    fridays = panel.index[panel.index.dayofweek == 4]
    # A smoothed target averages this row and the previous WEEKDAY row, so the
    # information it depends on closes at the later of the two label dates.
    previous_label = label_date.shift(1)
    effective = pd.concat([label_date, previous_label], axis=1).max(axis=1)

    blocks = []
    for k in range(1, N_QUINTILES + 1):
        block = pd.DataFrame({"target": soft[k].loc[fridays].stack(future_stack=True)})
        block["smooth_quintile"] = k
        blocks.append(block)
    long = pd.concat(blocks)
    long.index.names = ["date", "prod"]
    long = long.reset_index()

    for name, frame in features.items():
        long[name] = frame.loc[fridays].stack(future_stack=True).reindex(
            pd.MultiIndex.from_arrays([long["date"], long["prod"]])).to_numpy()

    long["label_available_from"] = effective.reindex(long["date"]).to_numpy()
    logger.info("Long frame: %d rows (%d Fridays x %d assets x %d quintiles)",
                len(long), len(fridays), panel.shape[1], N_QUINTILES)
    return long


# --------------------------------------------------------------------------- #
# Causal split, model, post-processing
# --------------------------------------------------------------------------- #

def causal_split(long: pd.DataFrame, origin: pd.Timestamp
                 ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Training rows whose label was already observable at `origin`, and the
    prediction cross-section at `origin`."""
    train = long[(long["label_available_from"] <= origin)
                 & long["target"].notna()
                 & (long["date"] < origin)]
    predict = long[long["date"] == origin]
    if train.empty or predict.empty:
        raise ImplementationError(f"Empty split at {origin.date()}.")
    if (train["label_available_from"] > origin).any():
        raise ImplementationError(f"Leakage at {origin.date()}: a label that was "
                                  "not yet observable entered training.")
    if len(predict) != N_ASSETS * N_QUINTILES:
        raise ImplementationError(f"{origin.date()}: prediction block has "
                                  f"{len(predict)} rows, expected "
                                  f"{N_ASSETS * N_QUINTILES}.")
    return train, predict


def fit_predict(train: pd.DataFrame, predict: pd.DataFrame,
                seed: int = SEED) -> np.ndarray:
    """Miguel's XGBoost configuration, unchanged and untuned."""
    dtrain = xgb.DMatrix(train[MODEL_INPUTS].to_numpy(dtype=float),
                         label=train["target"].to_numpy(dtype=float),
                         feature_names=MODEL_INPUTS)
    dpredict = xgb.DMatrix(predict[MODEL_INPUTS].to_numpy(dtype=float),
                           feature_names=MODEL_INPUTS)
    booster = xgb.train({**XGB_PARAMS, "seed": seed, "nthread": -1},
                        dtrain, num_boost_round=NUM_BOOST_ROUND)
    # XGBoost returns float32; R's predict() returns doubles, and the 5-decimal
    # post-processing below is sensitive at the 1e-7 level, so widen here.
    return booster.predict(dpredict).astype(np.float64)


def postprocess(predict: pd.DataFrame, raw: np.ndarray) -> pd.DataFrame:
    """Miguel's post-processing chain, in his order.

    1. `add_to_one_scale`: divide each asset's five predictions by their sum.
    2. `postprocess_sub`: shift each Rank column so its cross-sectional mean is
       exactly 0.20. This is an UNBOUNDED additive shift - the same rule that
       produced a negative probability elsewhere in this project - so validity is
       asserted rather than assumed.
    3. Round to 5 decimals and absorb the residual into Rank3.
    """
    wide = predict.assign(pred=raw).pivot(index="prod", columns="smooth_quintile",
                                          values="pred")
    wide = wide.reindex(columns=range(1, N_QUINTILES + 1))
    wide.columns = RANK_COLUMNS

    probabilities = wide.div(wide.sum(axis=1), axis=0)                    # step 1
    probabilities = probabilities + (0.2 - probabilities.mean(axis=0))    # step 2

    rounded = probabilities.round(5)                                      # step 3
    residual = rounded.sum(axis=1) - 1.0
    rounded["Rank3"] = (rounded["Rank3"] - residual).round(5)

    values = rounded.to_numpy()
    if not np.isfinite(values).all():
        raise ImplementationError("Non-finite probability after post-processing.")
    if values.min() < 0.0:
        raise ImplementationError(
            f"The released 0.20 column shift produced a negative probability "
            f"(min {values.min():.6f}). Stopping rather than clipping.")
    if values.max() > 1.0:
        raise ImplementationError(
            f"Probability above 1 after post-processing (max {values.max():.6f}).")
    # Every entry is a decimal multiple of 1e-5 after step 3, and such values
    # cannot sum to exactly 1.0 in binary floating point. The meaningful test is
    # therefore that the row sums to 1 AT THE 5-DECIMAL PRECISION THE SOURCE
    # EMITS - asserted exactly - with a loose float guard alongside it.
    scaled = np.round(values.sum(axis=1) * 1e5)
    if not np.all(scaled == 1e5):
        raise ImplementationError(
            f"Rows do not sum to 1 at 5-decimal precision "
            f"(worst {np.abs(scaled - 1e5).max() / 1e5:.3g}).")
    error = float(np.abs(values.sum(axis=1) - 1.0).max())
    if error > 1e-6:
        raise ImplementationError(f"Row-sum float error {error:.3g} exceeds what "
                                  "5-decimal rounding can explain.")
    return rounded


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #

def run(seed: int = SEED, write: bool = True,
        panel_start: str | None = None,
        panel_source: str = "dataset_d") -> pd.DataFrame:
    panel = build_panel(path=PANEL_SOURCES[panel_source],
                        panel_start=panel_start)
    long = assemble_long(panel)
    schedule = pd.read_csv(M6_SCHEDULE, parse_dates=["origin_date",
                                                     "forecast_end_date"])

    blocks, audit = [], []
    for record in schedule.itertuples():
        origin = pd.Timestamp(record.origin_date)
        train, predict = causal_split(long, origin)
        probabilities = postprocess(predict, fit_predict(train, predict, seed))

        block = probabilities.reset_index().rename(columns={"prod": "symbol"})
        block.insert(0, "origin_date", origin.date().isoformat())
        block.insert(0, "round", int(record.round))
        block.insert(0, "model", "FinQBoost (Python reimplementation)")
        blocks.append(block)
        audit.append({"round": int(record.round),
                      "origin": origin.date().isoformat(),
                      "train_rows": len(train),
                      "train_fridays": int(train["date"].nunique()),
                      "last_train_friday": str(train["date"].max().date()),
                      "min_prob": float(probabilities.to_numpy().min()),
                      "max_prob": float(probabilities.to_numpy().max())})
        logger.info("Round %2d %s: %d train rows (%d Fridays, last %s), "
                    "prob range [%.5f, %.5f]", record.round, origin.date(),
                    len(train), audit[-1]["train_fridays"],
                    audit[-1]["last_train_friday"], audit[-1]["min_prob"],
                    audit[-1]["max_prob"])

    predictions = pd.concat(blocks, ignore_index=True)

    logger.info("Forecasts complete; loading the official M6 ground truth.")
    truth = build_ground_truth(load_prices(), pd.read_csv(M6_SCHEDULE))
    per_round = {}
    for round_number, truth_block in truth.groupby("round"):
        block = predictions[predictions["round"] == round_number]
        aligned = block.set_index("symbol").reindex(truth_block["symbol"])
        if aligned[RANK_COLUMNS].isna().any().any():
            raise ImplementationError(f"Round {round_number}: missing forecasts.")
        per_round[int(round_number)] = float(rps_scores(
            truth_block[RANK_COLUMNS].to_numpy(dtype=float),
            aligned[RANK_COLUMNS].to_numpy(dtype=float)).mean())

    overall = float(np.mean(list(per_round.values())))
    beating = sum(1 for v in per_round.values() if v < NAIVE_RPS)
    logger.info("Overall M6 RPS %.6f (naive %.4f, difference %+.6f); %d/12 rounds "
                "beat naive", overall, NAIVE_RPS, overall - NAIVE_RPS, beating)

    audit_frame = pd.DataFrame(audit)
    results = pd.DataFrame([{
        "model": "FinQBoost (Python reimplementation)",
        "seed": seed,
        **{f"round_{i:02d}_rps": round(per_round[i], 8) for i in range(1, 13)},
        "mean_m6_rps": round(overall, 8),
        "naive_rps": NAIVE_RPS,
        "difference_vs_naive": round(overall - NAIVE_RPS, 8),
        "rounds_beating_naive": beating,
        "published_finqboost_rps": PUBLISHED_FINQBOOST_RPS,
        "difference_vs_published": round(overall - PUBLISHED_FINQBOOST_RPS, 8),
        "train_rows_min": int(audit_frame["train_rows"].min()),
        "train_rows_max": int(audit_frame["train_rows"].max()),
        "train_fridays_min": int(audit_frame["train_fridays"].min()),
        "train_fridays_max": int(audit_frame["train_fridays"].max()),
        "min_probability": round(float(audit_frame["min_prob"].min()), 8),
        "max_probability": round(float(audit_frame["max_prob"].max()), 8),
        "features": json.dumps(SELECTED_FEATURES),
        "xgb_params": json.dumps({**XGB_PARAMS, "num_boost_round": NUM_BOOST_ROUND,
                                  "seed": seed}, sort_keys=True),
        "xgboost_version": xgb.__version__,
        "panel_source": panel_source,
        "panel_start": str(panel.index.min().date()),
        "evaluated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }])

    if write:
        predictions_path, rps_path, _ = output_paths(panel_start)
        predictions.to_csv(predictions_path, index=False)
        results.to_csv(rps_path, index=False)
        logger.info("Wrote %s and %s", predictions_path.name, rps_path.name)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", default="dataset_d",
                        choices=sorted(PANEL_SOURCES),
                        help="adjusted-close source. dataset_d is the shared "
                             "2009 EODHD snapshot used by the feature "
                             "baselines; extended is the 2000 EODHD snapshot.")
    parser.add_argument("--panel-start", default=None,
                        help="training-history cutoff within the chosen panel, "
                             "e.g. 2007-01-01. Omit to use all of it.")
    args = parser.parse_args()
    _, _, log_path = output_paths(args.panel_start)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(log_path, mode="w", encoding="utf-8"),
                  logging.StreamHandler(sys.stdout)])
    try:
        run(panel_start=args.panel_start, panel_source=args.panel)
    except ImplementationError as exc:
        logger.error("STOPPED: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
