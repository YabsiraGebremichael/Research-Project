"""Build the six pre-M6 validation contexts for the sampling-parameter experiment.

These contexts are used to select Chronos sampling settings (temperature, top-k)
before anything is evaluated on the 12 scored M6 rounds. Their forecast targets
all fall in early 2022 and finish before the first M6 target begins, so the M6
rounds stay untouched.

Construction is identical to the M6 contexts in build_rolling_origins.py: the
same Dataset A log-return panel, the same shared weekday calendar, the same
512-row window ending on and including the origin, the same official asset
order, and the same string-form copy so values are reproduced byte for byte.
The only difference is which origins are sliced. The six files are shared by
both models and every sampling configuration.

Existing files are never overwritten.

Inputs:
    Data/processed/dataset_a_daily_log_returns.csv
    Data/raw/m6_official/assets_m6.csv          (official asset order)
    Data/metadata/m6_round_schedule.csv         (first M6 target, for the guard)

Outputs:
    Data/validation_contexts/validation_01_context.csv .. validation_06_context.csv
    Data/validation_contexts/validation_context_manifest.csv

    python scripts/pipeline/build_validation_contexts.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RETURNS_CSV = PROJECT_ROOT / "Data" / "processed" / "dataset_a_daily_log_returns.csv"
OFFICIAL_M6_CSV = PROJECT_ROOT / "Data" / "raw" / "m6_official" / "assets_m6.csv"
SCHEDULE_CSV = PROJECT_ROOT / "Data" / "metadata" / "m6_round_schedule.csv"

OUT_DIR = PROJECT_ROOT / "Data" / "validation_contexts"
MANIFEST_CSV = OUT_DIR / "validation_context_manifest.csv"

CONTEXT_STEPS = 512
FORECAST_STEPS = 20
EXPECTED_ASSETS = 100
SOURCE_DATASET = "Dataset A daily log returns (Data/processed/dataset_a_daily_log_returns.csv)"

# Weekly Friday origins through January 2022. The 20-weekday targets overlap
# each other on purpose: this is a small validation grid, not a set of
# independent rounds.
ORIGINS = [
    "2021-12-31", "2022-01-07", "2022-01-14",
    "2022-01-21", "2022-01-28", "2022-02-04",
]


class ValidationError(RuntimeError):
    """Raised when the data or calendar cannot support a trustworthy context."""


def load_official_order() -> list[str]:
    """The 100 official M6 symbols in first-occurrence order."""
    frame = pd.read_csv(OFFICIAL_M6_CSV)
    symbols = list(dict.fromkeys(frame["symbol"].astype(str).str.strip()))
    if len(symbols) != EXPECTED_ASSETS:
        raise ValidationError(f"Official order has {len(symbols)} symbols.")
    return symbols


def load_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the parsed return panel and its string-form copy.

    The string copy is what the context files are written from, so the saved
    values are byte-identical to the preprocessed panel rather than passing
    through a float round trip.
    """
    if not RETURNS_CSV.is_file():
        raise ValidationError(f"Missing preprocessed input: {RETURNS_CSV}")
    returns = pd.read_csv(RETURNS_CSV, parse_dates=["date"]).set_index("date")
    returns_str = pd.read_csv(RETURNS_CSV, dtype=str)

    official = load_official_order()
    if list(returns.columns) != official:
        raise ValidationError("Return panel columns are not in official order.")
    if returns.index.duplicated().any() or not returns.index.is_monotonic_increasing:
        raise ValidationError("Return panel dates are duplicated or unsorted.")
    return returns, returns_str


def slice_for(index: pd.DatetimeIndex, origin: str) -> dict[str, object]:
    """Positional context and target slices for one validation origin."""
    stamp = pd.Timestamp(origin)
    if stamp not in index:
        raise ValidationError(f"Origin {origin} is not a row in the weekday calendar.")

    pos = index.get_loc(stamp)
    if pos + 1 < CONTEXT_STEPS:
        raise ValidationError(
            f"Origin {origin}: only {pos + 1} weekday rows precede it, "
            f"{CONTEXT_STEPS} required."
        )
    context_index = index[pos - CONTEXT_STEPS + 1: pos + 1]
    target_index = index[pos + 1: pos + 1 + FORECAST_STEPS]

    if len(target_index) != FORECAST_STEPS:
        raise ValidationError(
            f"Origin {origin}: only {len(target_index)} weekday rows follow it, "
            f"{FORECAST_STEPS} required."
        )
    # The context must end exactly on the origin and never look past it.
    if context_index[-1] != stamp or (context_index > stamp).any():
        raise ValidationError(f"Origin {origin}: context leaks past the origin.")

    return {
        "origin": stamp,
        "context_positions": (pos - CONTEXT_STEPS + 1, pos + 1),
        "context_index": context_index,
        "target_index": target_index,
    }


def missing_pattern(block: pd.DataFrame) -> tuple[dict[str, int], dict[str, int]]:
    """Split each asset's missing values into leading and interior counts.

    Leading gaps are genuine later inceptions (CARR, OGN) and are preserved
    rather than filled, exactly as in the M6 contexts. Interior gaps would mean
    the panel had changed and are treated as a failure.
    """
    leading, interior = {}, {}
    for symbol in block.columns:
        values = block[symbol].to_numpy(dtype=float)
        n_lead = int(np.argmax(~np.isnan(values))) if np.isnan(values[0]) else 0
        if n_lead:
            leading[symbol] = n_lead
        after = np.isnan(values[n_lead:]).sum()
        if after:
            interior[symbol] = int(after)
    return leading, interior


def main() -> int:
    returns, returns_str = load_panel()
    index = returns.index
    official = list(returns.columns)

    # Guard: every scored target must land before the first M6 forecast target.
    first_m6_target = pd.Timestamp(pd.read_csv(SCHEDULE_CSV)["forecast_start_date"].iloc[0])
    print(f"Panel {index[0].date()} .. {index[-1].date()}, {len(index)} weekday rows")
    print(f"First M6 forecast target: {first_m6_target.date()}\n")

    slices = [slice_for(index, origin) for origin in ORIGINS]

    rows = []
    for number, slc in enumerate(slices, start=1):
        context_index = slc["context_index"]
        target_index = slc["target_index"]
        origin = slc["origin"]

        if target_index[-1] >= first_m6_target:
            raise ValidationError(
                f"Origin {origin.date()}: target ends {target_index[-1].date()}, "
                f"which is not before the first M6 target {first_m6_target.date()}."
            )
        if target_index[0].year != 2022 or target_index[-1].year != 2022:
            raise ValidationError(
                f"Origin {origin.date()}: target {target_index[0].date()}.."
                f"{target_index[-1].date()} is not wholly within 2022."
            )

        # Leading gaps are expected; anything interior means the panel changed.
        block = returns.loc[context_index]
        leading, interior = missing_pattern(block)
        if interior:
            raise ValidationError(
                f"Origin {origin.date()}: interior missing values {interior}")

        path = OUT_DIR / f"validation_{number:02d}_context.csv"
        if path.exists():
            raise ValidationError(f"{path} already exists; refusing to overwrite.")

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        start, stop = slc["context_positions"]
        returns_str.iloc[start:stop].to_csv(path, index=False)

        rows.append({
            "validation_id": number,
            "forecast_origin": origin.date().isoformat(),
            "context_start": context_index[0].date().isoformat(),
            "context_end": context_index[-1].date().isoformat(),
            "target_start": target_index[0].date().isoformat(),
            "target_end": target_index[-1].date().isoformat(),
            "context_filename": path.name,
            "context_rows": CONTEXT_STEPS,
            "context_columns": len(official) + 1,
            "shape": f"{CONTEXT_STEPS} x {len(official)}",
            "n_assets": len(official),
            "target_weekdays": FORECAST_STEPS,
            # recorded so the inference notebooks can assert the same pattern
            "leading_missing": "; ".join(f"{k}={v}" for k, v in sorted(leading.items())) or "none",
            "interior_missing": "none",
            "source_dataset": SOURCE_DATASET,
            "validation_status": "pass",
        })
        print(f"  {path.name}: context {context_index[0].date()} .. {origin.date()}, "
              f"target {target_index[0].date()} .. {target_index[-1].date()}, "
              f"leading NaN {leading or 'none'}")

    manifest = pd.DataFrame(rows)
    manifest.to_csv(MANIFEST_CSV, index=False)

    # --- reload every saved file and re-check it independently -------------
    for row, slc in zip(rows, slices):
        saved = pd.read_csv(OUT_DIR / row["context_filename"], parse_dates=["date"])
        if saved.shape != (CONTEXT_STEPS, len(official) + 1):
            raise ValidationError(f"{row['context_filename']}: shape {saved.shape}")
        if list(saved.columns) != ["date"] + official:
            raise ValidationError(f"{row['context_filename']}: asset order changed")
        if saved["date"].iloc[-1] != slc["origin"]:
            raise ValidationError(f"{row['context_filename']}: does not end on its origin")
        if not saved["date"].is_monotonic_increasing or saved["date"].duplicated().any():
            raise ValidationError(f"{row['context_filename']}: dates unsorted or duplicated")
        _, interior = missing_pattern(saved.set_index("date"))
        if interior:
            raise ValidationError(f"{row['context_filename']}: interior NaNs {interior}")
    print(f"\nReloaded and re-checked all {len(rows)} files")
    print(f"Wrote {MANIFEST_CSV.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
