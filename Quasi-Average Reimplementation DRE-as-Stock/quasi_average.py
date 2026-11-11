"""Vilar's quasi-average M6 forecast, with DRE treated as a normal Stock.

Identical to `Quasi-Average Reimplementation/quasi_average.py` in every respect
except one: DRE stays inside the Stock group and receives the ordinary Stock
type component, instead of Vilar's separate uniform-vector rule. Nothing else
was changed to compensate.

Fixed released specification, unchanged: lag 20; cross-sectional quintiles of the
20-day log return (1 = worst); type distribution over the trailing 50 periods
with ETFs, Stocks and VXX as groups; temporal distribution 0.2/0.2/0.6 over the
trailing 5/10/400 periods; mixed forecast 0.5 * type + 0.5 * temporal,
first-differenced to ordinary probabilities.

Input is Dataset D's EODHD adjusted closes on the shared Mon-Fri
calendar. Scored at the 12 official M6 origins with the project's own evaluator.

    python "Quasi-Average Reimplementation DRE-as-Stock/quasi_average.py"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from evaluate_m6_rps import (  # noqa: E402
    RANK_COLUMNS, build_ground_truth, load_prices, rps_scores,
)

EXTENDED_OHLCV = (PROJECT_ROOT / "TSFM Uncertainty Experiment" / "data"
                  / "extended_eodhd_ohlcv_2000_2023.csv")
DATASET_D_OHLCV = (PROJECT_ROOT / "Data" / "raw" / "dataset_d_eodhd"
                   / "dataset_d_ohlcv.csv")
# Dataset D is the snapshot the feature baselines use; the extended file is a
# different EODHD snapshot, so truncating it does not reproduce Dataset D.
PANEL_SOURCES = {"dataset_d": DATASET_D_OHLCV, "extended": EXTENDED_OHLCV}

ASSET_METADATA = PROJECT_ROOT / "Data" / "metadata" / "m6_asset_metadata.csv"
OFFICIAL_M6 = PROJECT_ROOT / "Data" / "raw" / "m6_official" / "assets_m6.csv"
M6_SCHEDULE = PROJECT_ROOT / "Data" / "metadata" / "m6_round_schedule.csv"

PANEL_CSV = HERE / "data" / "adjusted_close_panel.csv"
OUT_PREDICTIONS = HERE / "results" / "quasi_average_m6_predictions.csv"
OUT_RPS = HERE / "results" / "quasi_average_m6_rps.csv"

LAG = 20
N_QUINTILES = 5
TYPE_PERIODS = 50
TEMPORAL = ((5, 0.2), (10, 0.2), (400, 0.6))
TEMPORAL_WEIGHT = 0.5                      # the released mixed method's av
UNIFORM = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
NAIVE_RPS = 0.16


def build_panel(source: Path = DATASET_D_OHLCV) -> pd.DataFrame:
    """Extended EODHD adjusted closes on the shared Mon-Fri calendar.

    Forward-filled after each asset's genuine inception, never backfilled - the
    project's standard treatment. The derived panel is saved for the notebook.
    """
    raw = pd.read_csv(source, parse_dates=["date"])
    official = list(dict.fromkeys(
        pd.read_csv(OFFICIAL_M6)["symbol"].astype(str).str.strip()))
    wide = raw.pivot(index="date", columns="symbol",
                     values="adjusted_close").reindex(columns=official).sort_index()
    panel = wide.reindex(pd.date_range(wide.index.min(), wide.index.max(),
                                       freq="B")).ffill()
    panel.index.name = "date"
    panel.columns.name = None

    PANEL_CSV.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(PANEL_CSV)
    return panel


def quintile_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Daily cross-sectional quintiles of the 20-day log return, 1 = worst.

    Includes the released leading trim, which is what fixes where the
    400-period window begins.
    """
    trim = -LAG * ((len(panel) - 1) // LAG - 1) - 1
    returns = np.log(panel).diff(LAG)[trim:]
    return 1 + returns.apply(
        lambda row: pd.qcut(row, N_QUINTILES, duplicates="drop", labels=False),
        axis=1)


def q_dists(column: pd.Series) -> np.ndarray:
    """Cumulative quintile distribution; uniform when the asset has no history."""
    values = column.dropna().astype(int)
    if len(values) == 0:
        return UNIFORM.copy()
    counts = np.array([(values == q).sum() for q in range(1, N_QUINTILES + 1)])
    return counts.cumsum() / counts.sum()


def mixed_forecast(quintiles: pd.DataFrame, groups: dict[str, list[str]],
                   end: int) -> pd.DataFrame:
    """0.5 * type + 0.5 * temporal, as ordinary probabilities.

    `end` is the exclusive stop of the information window, so the last row used
    is `quintiles.index[end - 1]` and nothing dated later is read.
    """
    def window(periods: int) -> pd.DataFrame:
        return quintiles.iloc[max(0, end - periods * LAG):end]

    type_window = window(TYPE_PERIODS)
    type_vectors = {name: np.stack([q_dists(type_window[s]) for s in members]).mean(axis=0)
                    for name, members in groups.items()}
    # DRE is an ordinary Stock here, so it is covered by the Stock membership
    # below and takes the Stock vector like any other stock.
    membership = {**{s: "etfs" for s in groups["etfs"]},
                  **{s: "stocks" for s in groups["stocks"]}, "VXX": "vxx"}
    type_block = np.stack([type_vectors[membership[s]] for s in quintiles.columns],
                          axis=1)

    temporal_block = np.zeros((N_QUINTILES, quintiles.shape[1]))
    for periods, weight in TEMPORAL:
        block = window(periods)
        temporal_block += weight * np.stack(
            [q_dists(block[s]) for s in quintiles.columns], axis=1)

    cumulative = ((1 - TEMPORAL_WEIGHT) * type_block
                  + TEMPORAL_WEIGHT * temporal_block)
    return pd.DataFrame(np.diff(cumulative, axis=0, prepend=0.0),
                        index=RANK_COLUMNS, columns=quintiles.columns)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", default="dataset_d",
                        choices=sorted(PANEL_SOURCES),
                        help="adjusted-close source. dataset_d is the shared "
                             "2009 EODHD snapshot used by the feature "
                             "baselines; extended is the 2000 EODHD snapshot.")
    args = parser.parse_args()

    panel = build_panel(PANEL_SOURCES[args.panel])
    quintiles = quintile_panel(panel)
    print(f"panel {panel.shape[0]} weekdays x {panel.shape[1]} assets, "
          f"{panel.index.min().date()} .. {panel.index.max().date()}")
    print(f"quintile panel {quintiles.shape[0]} rows "
          f"(from {quintiles.index.min().date()})")

    metadata = pd.read_csv(ASSET_METADATA)
    etfs = [s for s in metadata.loc[metadata["asset_type"] == "ETF", "symbol"]
            if s != "VXX"]
    stocks = list(metadata.loc[metadata["asset_type"] == "Stock", "symbol"])
    groups = {"etfs": etfs, "stocks": stocks, "vxx": ["VXX"]}
    assert "DRE" in stocks and len(stocks) == 50, "DRE must be an ordinary Stock"
    assert len(etfs) + len(stocks) + 1 == 100

    schedule = pd.read_csv(M6_SCHEDULE, parse_dates=["origin_date"])
    blocks = []
    for record in schedule.itertuples():
        origin = pd.Timestamp(record.origin_date)
        end = int(quintiles.index.get_loc(origin)) + 1
        assert quintiles.index[end - 1] == origin, "information window misaligned"

        forecast = mixed_forecast(quintiles, groups, end)
        probabilities = forecast.to_numpy().T
        assert np.isfinite(probabilities).all()
        assert probabilities.min() >= -1e-12
        assert np.abs(probabilities.sum(axis=1) - 1).max() < 1e-9

        block = pd.DataFrame(probabilities, columns=RANK_COLUMNS)
        block.insert(0, "symbol", quintiles.columns)
        block.insert(0, "origin_date", origin.date().isoformat())
        block.insert(0, "round", int(record.round))
        blocks.append(block)
        print(f"  round {record.round:2d} {origin.date()}: {end} information rows")

    predictions = pd.concat(blocks, ignore_index=True)

    truth = build_ground_truth(load_prices(), pd.read_csv(M6_SCHEDULE))
    per_round = {}
    for round_number, truth_block in truth.groupby("round"):
        aligned = (predictions[predictions["round"] == round_number]
                   .set_index("symbol").reindex(truth_block["symbol"]))
        assert not aligned[RANK_COLUMNS].isna().any().any()
        per_round[int(round_number)] = float(rps_scores(
            truth_block[RANK_COLUMNS].to_numpy(dtype=float),
            aligned[RANK_COLUMNS].to_numpy(dtype=float)).mean())

    overall = float(np.mean(list(per_round.values())))
    beating = sum(1 for v in per_round.values() if v < NAIVE_RPS)

    results = pd.DataFrame([{
        "model": "Vilar quasi-average (DRE as ordinary Stock)",
        **{f"round_{i:02d}_rps": round(per_round[i], 8) for i in range(1, 13)},
        "mean_m6_rps": round(overall, 8),
        "naive_rps": NAIVE_RPS,
        "difference_vs_naive": round(overall - NAIVE_RPS, 8),
        "rounds_beating_naive": beating,
        "panel_source": args.panel,
        "panel_start": str(panel.index.min().date()),
        "min_probability": round(float(predictions[RANK_COLUMNS].to_numpy().min()), 8),
        "max_probability": round(float(predictions[RANK_COLUMNS].to_numpy().max()), 8),
    }])

    OUT_PREDICTIONS.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(OUT_PREDICTIONS, index=False)
    results.to_csv(OUT_RPS, index=False)

    print(f"\noverall M6 RPS {overall:.6f} (naive {NAIVE_RPS}, "
          f"difference {overall - NAIVE_RPS:+.6f}); {beating}/12 rounds beat naive")
    print(f"wrote {OUT_PREDICTIONS.relative_to(PROJECT_ROOT)}")
    print(f"wrote {OUT_RPS.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
