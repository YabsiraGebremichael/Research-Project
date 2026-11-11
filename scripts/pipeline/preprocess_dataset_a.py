"""Preprocess Dataset A onto the shared weekday calendar and build log returns.

Transformations, and only these: reindex the 100-asset adjusted-close panel onto
a Monday-to-Friday calendar (2020-01-02 .. 2023-02-03, 807 dates); forward-fill
each asset only after its first genuine observation, so holidays and closures
become zero return; and take daily log returns. DRE's final genuine price
(2022-10-03) carries forward under that same rule, which reproduces the official
post-acquisition treatment and is verified explicitly.

No backward filling, interpolation, smoothing, normalisation, clipping or
outlier removal of any kind.

Input   Data/raw/yahoo/dataset_a_adjusted_close_repaired.csv
Outputs Data/processed/dataset_a_adjusted_close_weekday.csv
        Data/processed/dataset_a_daily_log_returns.csv
        Data/metadata/diagnostics/dataset_a_preprocessing_summary.csv
        reports/dataset_a_preprocessing_report.md

Usage: python scripts/pipeline/preprocess_dataset_a.py
"""

from __future__ import annotations

import hashlib
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_REPAIRED = PROJECT_ROOT / "Data" / "raw" / "yahoo" / "dataset_a_adjusted_close_repaired.csv"
OFFICIAL_M6_CSV = PROJECT_ROOT / "Data" / "raw" / "m6_official" / "assets_m6.csv"

OUT_PRICES = PROJECT_ROOT / "Data" / "processed" / "dataset_a_adjusted_close_weekday.csv"
OUT_RETURNS = PROJECT_ROOT / "Data" / "processed" / "dataset_a_daily_log_returns.csv"
OUT_SUMMARY = PROJECT_ROOT / "Data" / "metadata" / "diagnostics" / "dataset_a_preprocessing_summary.csv"
OUT_REPORT = PROJECT_ROOT / "reports" / "dataset_a_preprocessing_report.md"

START_DATE = pd.Timestamp("2020-01-02")
END_DATE = pd.Timestamp("2023-02-03")
EXPECTED_ASSETS = 100
EXPECTED_WEEKDAYS = 807

DRE_LAST_GENUINE = pd.Timestamp("2022-10-03")
UNUSUAL_GAP_WEEKDAYS = 5           # internal gaps longer than this are reported
FIRST_ORIGIN = pd.Timestamp("2022-03-04")
REQUIRED_CONTEXT_RETURNS = 512

M6_FRIDAY_ANCHORS = [pd.Timestamp(d) for d in (
    "2022-03-04", "2022-04-01", "2022-04-29", "2022-05-27", "2022-06-24",
    "2022-07-22", "2022-08-19", "2022-09-16", "2022-10-14", "2022-11-11",
    "2022-12-09", "2023-01-06", "2023-02-03",
)]

logger = logging.getLogger("preprocess_dataset_a")


class ValidationError(RuntimeError):
    """A structural expectation about the input or output was violated."""


# --------------------------------------------------------------------------- #
# Input loading and validation
# --------------------------------------------------------------------------- #

def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_official_order(csv_path: Path = OFFICIAL_M6_CSV) -> list[str]:
    """Official 100-asset order; the prices in that file are never used here."""
    df = pd.read_csv(csv_path)
    symbols = list(dict.fromkeys(df["symbol"].astype(str).str.strip()))
    if len(symbols) != EXPECTED_ASSETS:
        raise ValidationError(
            f"Official file {csv_path} yields {len(symbols)} unique symbols, "
            f"expected {EXPECTED_ASSETS}."
        )
    return symbols


def load_and_validate_raw(
    csv_path: Path = RAW_REPAIRED,
    official: list[str] | None = None,
) -> pd.DataFrame:
    """Load the repaired raw panel and enforce every structural expectation."""
    if not csv_path.is_file():
        raise ValidationError(f"Input not found: {csv_path}")
    official = official or load_official_order()

    df = pd.read_csv(csv_path)
    if df.columns[0] != "date":
        raise ValidationError(f"First column must be 'date', got {df.columns[0]!r}.")
    asset_cols = list(df.columns[1:])
    if len(df.columns) != EXPECTED_ASSETS + 1:
        raise ValidationError(
            f"Expected {EXPECTED_ASSETS + 1} columns, found {len(df.columns)}."
        )
    if asset_cols != official:
        raise ValidationError(
            "Asset columns do not match the official first-occurrence order."
        )

    dates = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="raise")
    if dates.duplicated().any():
        raise ValidationError("Input contains duplicate dates.")
    if not dates.is_monotonic_increasing:
        raise ValidationError("Input dates are not sorted ascending.")
    if dates.iloc[0] != START_DATE:
        raise ValidationError(f"First date is {dates.iloc[0].date()}, "
                              f"expected {START_DATE.date()}.")
    if dates.iloc[-1] != END_DATE:
        raise ValidationError(f"Final date is {dates.iloc[-1].date()}, "
                              f"expected {END_DATE.date()}.")

    panel = df.set_index(pd.DatetimeIndex(dates, name="date"))[asset_cols].astype(float)
    if (panel <= 0).any().any():
        bad = [c for c in asset_cols if (panel[c] <= 0).any()]
        raise ValidationError(f"Non-positive adjusted-close values in: {bad}.")

    logger.info("Raw input validated: %d rows x %d assets, %s .. %s",
                panel.shape[0], panel.shape[1],
                panel.index.min().date(), panel.index.max().date())
    return panel


# --------------------------------------------------------------------------- #
# Calendar and forward-filling
# --------------------------------------------------------------------------- #

def weekday_calendar(start: pd.Timestamp = START_DATE,
                     end: pd.Timestamp = END_DATE) -> pd.DatetimeIndex:
    """Shared Monday-to-Friday calendar (public holidays included)."""
    cal = pd.date_range(start, end, freq="B", name="date")
    if len(cal) != EXPECTED_WEEKDAYS:
        raise ValidationError(
            f"Weekday calendar has {len(cal)} dates, expected {EXPECTED_WEEKDAYS}."
        )
    return cal


def internal_gaps_before_filling(series: pd.Series) -> int:
    """Longest run of consecutive missing weekdays strictly inside the
    first..last genuine-observation span (leading/trailing gaps excluded)."""
    valid = series.dropna()
    if valid.empty:
        return 0
    inner = series.loc[valid.index.min():valid.index.max()]
    is_na = inner.isna().to_numpy()
    longest = run = 0
    for missing in is_na:
        run = run + 1 if missing else 0
        longest = max(longest, run)
    return int(longest)


def forward_fill_after_inception(panel: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill each asset only after its first genuine observation.

    pandas ffill never fills before the first valid value, which is exactly
    the required rule; leading (pre-inception) missing values are preserved.
    """
    return panel.ffill()


# --------------------------------------------------------------------------- #
# Returns
# --------------------------------------------------------------------------- #

def daily_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """log_return[t] = log(price[t] / price[t-1]); NaN where either is missing."""
    returns = np.log(prices).diff()
    if np.isinf(returns.to_numpy()).any():
        raise ValidationError("Infinite values found in log returns.")
    return returns


# --------------------------------------------------------------------------- #
# Per-asset summary
# --------------------------------------------------------------------------- #

def build_summary(
    raw_on_calendar: pd.DataFrame,
    processed: pd.DataFrame,
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """One row per asset (official order): counts and dates only."""
    rows = []
    for order, symbol in enumerate(raw_on_calendar.columns, start=1):
        raw_s = raw_on_calendar[symbol]
        proc_s = processed[symbol]
        ret_s = returns[symbol]
        genuine = raw_s.dropna()
        first_gen = genuine.index.min()
        last_gen = genuine.index.max()
        first_proc = proc_s.first_valid_index()
        first_ret = ret_s.first_valid_index()
        leading_missing = int((raw_s.index < first_gen).sum())
        filled = int(proc_s.notna().sum() - genuine.shape[0])
        gap = internal_gaps_before_filling(raw_s)

        notes: list[str] = []
        if symbol == "DRE":
            notes.append("Final genuine price (2022-10-03) carried forward "
                         "after acquisition by PLD; zero subsequent returns.")
        if symbol in ("CARR", "OGN"):
            notes.append("Spin-off with genuine later inception; leading "
                         "missing values preserved.")
        if symbol == "RE":
            notes.append("Recovered from EODHD identifier EG.US; canonical "
                         "column remains RE.")
        if symbol == "WRK":
            notes.append("Recovered from EODHD delisted identifier WRK.US.")
        if gap > UNUSUAL_GAP_WEEKDAYS:
            notes.append(f"Internal gap of {gap} consecutive missing weekdays "
                         "before filling.")

        rows.append({
            "official_order": order,
            "symbol": symbol,
            "first_genuine_price_date": first_gen.date().isoformat(),
            "last_genuine_price_date": last_gen.date().isoformat(),
            "first_processed_price_date": first_proc.date().isoformat(),
            "first_valid_log_return_date": first_ret.date().isoformat(),
            "genuine_price_observation_count": int(genuine.shape[0]),
            "forward_filled_price_count": filled,
            "leading_missing_price_count": leading_missing,
            "remaining_missing_price_count": int(proc_s.isna().sum()),
            "valid_log_return_count": int(ret_s.notna().sum()),
            "missing_log_return_count": int(ret_s.isna().sum()),
            "longest_internal_missing_weekday_gap_before_filling": gap,
            "notes": " ".join(notes),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Output validation
# --------------------------------------------------------------------------- #

def validate_outputs(
    processed: pd.DataFrame,
    returns: pd.DataFrame,
    raw_on_calendar: pd.DataFrame,
) -> dict[str, object]:
    """Enforce every output rule for this step; return facts for the report."""
    facts: dict[str, object] = {}

    idx = processed.index
    if len(idx) != EXPECTED_WEEKDAYS or idx.min() != START_DATE or idx.max() != END_DATE:
        raise ValidationError("Processed calendar shape or bounds are wrong.")
    if idx.duplicated().any() or not idx.is_monotonic_increasing:
        raise ValidationError("Processed dates are duplicated or unsorted.")
    if (idx.dayofweek > 4).any():
        raise ValidationError("Weekend dates present in the processed calendar.")
    if not returns.index.equals(idx) or list(returns.columns) != list(processed.columns):
        raise ValidationError("Return panel index/columns differ from prices.")

    # Forward-fill must never create values before an asset's inception,
    # and after inception the processed panel must be complete.
    for symbol in processed.columns:
        first_gen = raw_on_calendar[symbol].first_valid_index()
        if processed[symbol].loc[:first_gen].iloc[:-1].notna().any():
            raise ValidationError(f"{symbol}: value created before inception.")
        if processed[symbol].loc[first_gen:].isna().any():
            raise ValidationError(f"{symbol}: missing price after inception.")

    # DRE treatment.
    dre_raw = raw_on_calendar["DRE"].dropna()
    if dre_raw.index.max() != DRE_LAST_GENUINE:
        raise ValidationError(
            f"DRE's final genuine observation is {dre_raw.index.max().date()}, "
            f"expected {DRE_LAST_GENUINE.date()}."
        )
    dre_final_price = float(dre_raw.iloc[-1])
    if not dre_final_price > 0:
        raise ValidationError("DRE's final genuine price is not positive.")
    dre_after = processed["DRE"].loc[DRE_LAST_GENUINE:]
    if not (dre_after == dre_final_price).all():
        raise ValidationError("DRE price is not constant after 2022-10-03.")
    dre_ret_after = returns["DRE"].loc[DRE_LAST_GENUINE + pd.Timedelta(days=1):]
    if not (dre_ret_after == 0.0).all():
        raise ValidationError("DRE has a non-zero return after 2022-10-03.")
    facts["dre_final_price"] = dre_final_price
    facts["dre_post_rows"] = int(dre_after.shape[0] - 1)

    # No infinities anywhere.
    if np.isinf(returns.to_numpy()).any() or np.isinf(processed.to_numpy()).any():
        raise ValidationError("Infinite values present in outputs.")

    # M6 Friday anchors.
    missing_anchors = [d.date().isoformat() for d in M6_FRIDAY_ANCHORS
                       if d not in idx]
    if missing_anchors:
        raise ValidationError(f"Missing M6 anchor dates: {missing_anchors}.")

    facts["dates_added"] = int(len(idx) - raw_on_calendar.dropna(how="all").shape[0])
    return facts


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #

def build_report(
    raw_panel: pd.DataFrame,
    raw_on_calendar: pd.DataFrame,
    processed: pd.DataFrame,
    returns: pd.DataFrame,
    summary: pd.DataFrame,
    facts: dict[str, object],
    input_hash_unchanged: bool,
) -> str:
    idx = processed.index
    added = len(idx) - raw_panel.shape[0]
    total_filled = int(summary["forward_filled_price_count"].sum())
    long_gaps = summary.loc[
        summary["longest_internal_missing_weekday_gap_before_filling"]
        > UNUSUAL_GAP_WEEKDAYS,
        ["symbol", "longest_internal_missing_weekday_gap_before_filling"],
    ]

    returns_by_origin = returns.loc[:FIRST_ORIGIN].notna().sum()
    short_assets = returns_by_origin[returns_by_origin < REQUIRED_CONTEXT_RETURNS]

    carr_first = summary.loc[summary["symbol"] == "CARR",
                             "first_genuine_price_date"].iloc[0]
    ogn_first = summary.loc[summary["symbol"] == "OGN",
                            "first_genuine_price_date"].iloc[0]

    fill_lines = [
        f"| {r.symbol} | {r.forward_filled_price_count} | "
        f"{r.longest_internal_missing_weekday_gap_before_filling} |"
        for r in summary.itertuples(index=False)
        if r.forward_filled_price_count > 0
    ]
    ret_lines = [
        f"| {r.symbol} | {r.valid_log_return_count} | "
        f"{r.missing_log_return_count} |"
        for r in summary.itertuples(index=False)
    ]

    lines = [
        "# Dataset A Preprocessing Report",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "",
        "## 1. Input",
        "",
        f"- Input: `Data/raw/yahoo/dataset_a_adjusted_close_repaired.csv`",
        f"- Shape: {raw_panel.shape[0]} rows x {raw_panel.shape[1]} assets "
        f"(+ date column), {raw_panel.index.min().date()} to "
        f"{raw_panel.index.max().date()}",
        "- Structural checks passed: `date` first column, 100 asset columns "
        "in official first-occurrence order, unique sorted dates, no "
        "non-positive prices.",
        f"- Raw file unmodified (SHA-256 verified before/after): "
        f"{'CONFIRMED' if input_hash_unchanged else 'FAILED'}",
        "- The official M6 file was used only to confirm asset names and "
        "order; its prices were not used.",
        "",
        "## 2. Weekday alignment",
        "",
        "- A single shared Monday-to-Friday calendar is used so that every "
        "asset — trading on US or London exchanges with different holidays — "
        "shares one index, which later stages require for cross-sectional "
        "work. Individual exchange calendars are deliberately not used.",
        f"- Output range: {idx.min().date()} to {idx.max().date()}; "
        f"{len(idx)} weekday rows (Saturdays/Sundays excluded: confirmed).",
        f"- Dates added by reindexing (shared holidays absent from the raw "
        f"union calendar): {added}",
        "",
        "## 3. Forward-filling",
        "",
        "- Forward-filling copies the most recently observed price of the "
        "same asset onto later missing weekdays. On a day a market is closed "
        "the holder's position cannot change value, so a carried-forward "
        "price (and hence a zero log return) is the faithful representation.",
        f"- Total cells forward-filled across all assets: {total_filled}. "
        "Per-asset counts (assets with at least one filled cell):",
        "",
        "| symbol | forward-filled cells | longest internal gap (weekdays, "
        "pre-fill) |",
        "|---|---|---|",
        *fill_lines,
        "",
        ("- Unusually long internal gaps (> "
         f"{UNUSUAL_GAP_WEEKDAYS} consecutive weekdays) before filling: "
         + (", ".join(f"{r.symbol} ({r.longest_internal_missing_weekday_gap_before_filling})"
                      for r in long_gaps.itertuples(index=False))
            if not long_gaps.empty else "none found.")),
        "- No backward-filling, interpolation, zero-substitution or "
        "cross-asset copying occurred; leading pre-inception values were "
        "never filled.",
        "",
        "## 4. Special assets",
        "",
        f"- DRE: final genuine observation {DRE_LAST_GENUINE.date()} at price "
        f"{facts['dre_final_price']:.4f} (positive, non-null: confirmed). The "
        f"processed price is constant over the {facts['dre_post_rows']} "
        "subsequent weekdays through 2023-02-03, and every DRE log return "
        "after 2022-10-03 equals exactly zero — the official M6 zero-return "
        "treatment. No zero prices, no PLD prices, no jump at the "
        "acquisition date.",
        f"- CARR: first genuine date {carr_first} unchanged; all earlier "
        "weekdays remain missing (no backfill).",
        f"- OGN: first genuine date {ogn_first} unchanged; all earlier "
        "weekdays remain missing (no backfill).",
        "- RE remains the canonical column (repair provider identifier was "
        "EG.US). WRK remains the canonical column (provider identifier "
        "WRK.US, delisted list).",
        "",
        "## 5. Returns",
        "",
        "- Formula: `log_return[t] = log(price[t] / price[t-1])`, implemented "
        "as `np.log(processed_prices).diff()`, per asset, on the shared "
        "weekday index in official column order.",
        "- A return is missing exactly where the current or preceding "
        "processed price is missing (pre-inception periods and each asset's "
        "first row); remaining missing returns were NOT replaced with zero.",
        "- No scaling, averaging, smoothing, clipping, winsorising or "
        "normalisation was applied; no statistics were fitted.",
        "- Infinite values: none (checked; prices are strictly positive).",
        "",
        "Valid / missing return counts per asset:",
        "",
        "| symbol | valid returns | missing returns |",
        "|---|---|---|",
        *ret_lines,
        "",
        "## 6. Rolling-origin readiness (verification only)",
        "",
        "- All 13 M6 Friday anchor dates exist in both processed panels: "
        + ", ".join(d.date().isoformat() for d in M6_FRIDAY_ANCHORS),
        "",
        f"- Valid historical log returns available on or before the first "
        f"origin ({FIRST_ORIGIN.date()}): min "
        f"{int(returns_by_origin.min())}, max {int(returns_by_origin.max())} "
        "across assets.",
        f"- Assets with fewer than {REQUIRED_CONTEXT_RETURNS} returns at the "
        "first origin: "
        + (", ".join(f"{s} ({int(n)})" for s, n in short_assets.items())
           if not short_assets.empty else "none"),
        "- Rolling-origin model contexts were NOT created or padded in this "
        "stage.",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = frame.copy()
    out.insert(0, "date", out.index.strftime("%Y-%m-%d"))
    out.to_csv(path, index=False)
    logger.info("Wrote %s (%d rows x %d columns)", path, out.shape[0], out.shape[1])


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        input_hash_before = sha256_of(RAW_REPAIRED)
        official = load_official_order()
        raw_panel = load_and_validate_raw(RAW_REPAIRED, official)

        calendar = weekday_calendar()
        raw_on_calendar = raw_panel.reindex(calendar)
        processed = forward_fill_after_inception(raw_on_calendar)
        returns = daily_log_returns(processed)
        summary = build_summary(raw_on_calendar, processed, returns)

        facts = validate_outputs(processed, returns, raw_on_calendar)
        input_hash_unchanged = sha256_of(RAW_REPAIRED) == input_hash_before
        if not input_hash_unchanged:
            raise ValidationError("Raw input file changed during processing.")

        write_csv(processed, OUT_PRICES)
        write_csv(returns, OUT_RETURNS)
        OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(OUT_SUMMARY, index=False)
        logger.info("Wrote %s (%d assets)", OUT_SUMMARY, summary.shape[0])

        report = build_report(raw_panel, raw_on_calendar, processed, returns,
                              summary, facts, input_hash_unchanged)
        OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
        OUT_REPORT.write_text(report, encoding="utf-8")
        logger.info("Wrote %s", OUT_REPORT)
    except ValidationError as exc:
        logger.error("Validation failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
