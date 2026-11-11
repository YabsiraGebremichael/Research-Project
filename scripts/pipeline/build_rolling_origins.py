"""Build and validate the 12 M6 rolling-origin model contexts.

Slices the fixed 512-weekday context ending on each Friday forecast origin and
checks it against the official round schedule. Values are copied unchanged from
the preprocessed panel: nothing is transformed, no missing value is filled, and
no observation after an origin can enter its context.

Inputs  Data/processed/dataset_a_daily_log_returns.csv (context source)
        Data/processed/dataset_a_adjusted_close_weekday.csv (anchor checks)
Outputs Data/contexts/round_01_context.csv .. round_12_context.csv
        Data/metadata/m6_round_schedule.csv
        reports/rolling_origin_construction_report.md

Usage: python scripts/pipeline/build_rolling_origins.py
"""

from __future__ import annotations

import hashlib
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RETURNS_CSV = PROJECT_ROOT / "Data" / "processed" / "dataset_a_daily_log_returns.csv"
PRICES_CSV = PROJECT_ROOT / "Data" / "processed" / "dataset_a_adjusted_close_weekday.csv"
OFFICIAL_M6_CSV = PROJECT_ROOT / "Data" / "raw" / "m6_official" / "assets_m6.csv"

OUT_DIR = PROJECT_ROOT / "Data" / "contexts"
OUT_SCHEDULE = PROJECT_ROOT / "Data" / "metadata" / "m6_round_schedule.csv"
OUT_REPORT = PROJECT_ROOT / "reports" / "rolling_origin_construction_report.md"

CONTEXT_STEPS = 512
FORECAST_STEPS = 20
EXPECTED_ASSETS = 100
DRE_LAST_GENUINE = pd.Timestamp("2022-10-03")

ANCHORS = [pd.Timestamp(d) for d in (
    "2022-03-04", "2022-04-01", "2022-04-29", "2022-05-27", "2022-06-24",
    "2022-07-22", "2022-08-19", "2022-09-16", "2022-10-14", "2022-11-11",
    "2022-12-09", "2023-01-06", "2023-02-03",
)]

SUBMISSION_DEADLINES = [pd.Timestamp(d) for d in (
    "2022-03-06", "2022-04-03", "2022-05-01", "2022-05-29", "2022-06-26",
    "2022-07-24", "2022-08-21", "2022-09-18", "2022-10-16", "2022-11-13",
    "2022-12-11", "2023-01-08",
)]

# Expected (context_start, context_end, forecast_start, forecast_end) per round.
EXPECTED_RANGES = [
    ("2020-03-19", "2022-03-04", "2022-03-07", "2022-04-01"),
    ("2020-04-16", "2022-04-01", "2022-04-04", "2022-04-29"),
    ("2020-05-14", "2022-04-29", "2022-05-02", "2022-05-27"),
    ("2020-06-11", "2022-05-27", "2022-05-30", "2022-06-24"),
    ("2020-07-09", "2022-06-24", "2022-06-27", "2022-07-22"),
    ("2020-08-06", "2022-07-22", "2022-07-25", "2022-08-19"),
    ("2020-09-03", "2022-08-19", "2022-08-22", "2022-09-16"),
    ("2020-10-01", "2022-09-16", "2022-09-19", "2022-10-14"),
    ("2020-10-29", "2022-10-14", "2022-10-17", "2022-11-11"),
    ("2020-11-26", "2022-11-11", "2022-11-14", "2022-12-09"),
    ("2020-12-24", "2022-12-09", "2022-12-12", "2023-01-06"),
    ("2021-01-21", "2023-01-06", "2023-01-09", "2023-02-03"),
]

logger = logging.getLogger("build_rolling_origins")


class ValidationError(RuntimeError):
    """A structural expectation about the schedule or a context failed."""


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_official_order() -> list[str]:
    df = pd.read_csv(OFFICIAL_M6_CSV)
    symbols = list(dict.fromkeys(df["symbol"].astype(str).str.strip()))
    if len(symbols) != EXPECTED_ASSETS:
        raise ValidationError(f"Official order has {len(symbols)} symbols.")
    return symbols


def load_panels() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (returns_parsed, returns_raw_strings, prices_parsed)."""
    for path in (RETURNS_CSV, PRICES_CSV):
        if not path.is_file():
            raise ValidationError(f"Missing preprocessed input: {path}")
    official = load_official_order()

    returns = pd.read_csv(RETURNS_CSV, parse_dates=["date"]).set_index("date")
    prices = pd.read_csv(PRICES_CSV, parse_dates=["date"]).set_index("date")
    # String-form copy: context files are written from this so preprocessed values
    # are reproduced byte-for-byte, untouched by float round-tripping.
    returns_str = pd.read_csv(RETURNS_CSV, dtype=str)

    for name, panel in (("returns", returns), ("prices", prices)):
        if list(panel.columns) != official:
            raise ValidationError(f"{name} panel columns not in official order.")
        if panel.index.duplicated().any() or not panel.index.is_monotonic_increasing:
            raise ValidationError(f"{name} panel dates duplicated or unsorted.")
    if not returns.index.equals(prices.index):
        raise ValidationError("Return and price panels have different calendars.")
    logger.info("Loaded preprocessed panels: %d weekday rows x %d assets.",
                returns.shape[0], returns.shape[1])
    return returns, returns_str, prices


# --------------------------------------------------------------------------- #
# Schedule validation
# --------------------------------------------------------------------------- #

def validate_schedule(index: pd.DatetimeIndex) -> None:
    """Anchors exist, are 20 weekday rows apart, and rounds chain correctly."""
    for anchor in ANCHORS:
        if anchor not in index:
            raise ValidationError(f"Anchor {anchor.date()} missing from panels.")
    positions = [index.get_loc(a) for a in ANCHORS]
    for r, (p0, p1) in enumerate(zip(positions, positions[1:]), start=1):
        if p1 - p0 != FORECAST_STEPS:
            raise ValidationError(
                f"Round {r}: forecast period holds {p1 - p0} weekday rows, "
                f"expected {FORECAST_STEPS}."
            )
    if len(SUBMISSION_DEADLINES) != len(ANCHORS) - 1:
        raise ValidationError("Need exactly 12 submission deadlines.")
    for r, (origin, deadline) in enumerate(zip(ANCHORS, SUBMISSION_DEADLINES), 1):
        if (deadline - origin).days != 2 or deadline.dayofweek != 6:
            raise ValidationError(
                f"Round {r}: deadline {deadline.date()} is not the Sunday "
                f"after origin {origin.date()}."
            )
    logger.info("Round schedule validated: 13 anchors, 12 rounds, "
                "%d-weekday forecast periods, non-overlapping and chained.",
                FORECAST_STEPS)


# --------------------------------------------------------------------------- #
# Context construction
# --------------------------------------------------------------------------- #

def round_slices(index: pd.DatetimeIndex, round_no: int) -> dict[str, object]:
    """Positional context/forecast slices for one round, fully validated."""
    origin = ANCHORS[round_no - 1]
    eval_end = ANCHORS[round_no]
    pos = index.get_loc(origin)
    if pos + 1 < CONTEXT_STEPS:
        raise ValidationError(
            f"Round {round_no}: only {pos + 1} rows precede the origin; "
            f"{CONTEXT_STEPS} required."
        )
    ctx_index = index[pos - CONTEXT_STEPS + 1: pos + 1]
    fc_index = index[pos + 1: index.get_loc(eval_end) + 1]

    exp_cs, exp_ce, exp_fs, exp_fe = (pd.Timestamp(d)
                                      for d in EXPECTED_RANGES[round_no - 1])
    actual = (ctx_index[0], ctx_index[-1], fc_index[0], fc_index[-1])
    if actual != (exp_cs, exp_ce, exp_fs, exp_fe):
        raise ValidationError(
            f"Round {round_no}: computed ranges "
            f"{[d.date().isoformat() for d in actual]} differ from the "
            f"expected official ranges {EXPECTED_RANGES[round_no - 1]}."
        )
    if len(ctx_index) != CONTEXT_STEPS or len(fc_index) != FORECAST_STEPS:
        raise ValidationError(f"Round {round_no}: slice lengths wrong.")
    if ctx_index[-1] != origin or (ctx_index > origin).any():
        raise ValidationError(f"Round {round_no}: context leaks past origin.")
    return {
        "round": round_no,
        "origin": origin,
        "eval_end": eval_end,
        "context_positions": (pos - CONTEXT_STEPS + 1, pos + 1),
        "context_index": ctx_index,
        "forecast_index": fc_index,
    }


def write_context(returns_str: pd.DataFrame, slc: dict[str, object]) -> Path:
    """Write one context file from the string-form panel (values untouched)."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    start, stop = slc["context_positions"]
    out = returns_str.iloc[start:stop]
    path = OUT_DIR / f"round_{slc['round']:02d}_context.csv"
    out.to_csv(path, index=False)
    return path


# --------------------------------------------------------------------------- #
# Availability accounting
# --------------------------------------------------------------------------- #

def availability_rows(returns: pd.DataFrame,
                      slc: dict[str, object]) -> list[dict[str, object]]:
    """Per-asset validity accounting for one round's context."""
    ctx = returns.loc[slc["context_index"]]
    origin = slc["origin"]
    rows = []
    for symbol in ctx.columns:
        s = ctx[symbol]
        valid = s.dropna()
        first_valid = valid.index.min() if not valid.empty else None
        leading = int((s.index < first_valid).sum()) if first_valid is not None else len(s)

        notes: list[str] = []
        if symbol in ("CARR", "OGN") and len(valid) < CONTEXT_STEPS:
            notes.append("Genuine later inception; leading missing values "
                         "preserved, no padding.")
        if symbol == "DRE" and origin > DRE_LAST_GENUINE:
            notes.append("Context includes post-acquisition period; returns "
                         "after 2022-10-03 are exactly zero (the preprocessing "
                         "treatment, unchanged).")

        rows.append({
            "round": slc["round"],
            "symbol": symbol,
            "context_start_date": s.index[0].date().isoformat(),
            "origin_date": origin.date().isoformat(),
            "total_context_rows": int(len(s)),
            "valid_return_count": int(valid.shape[0]),
            "missing_return_count": int(s.isna().sum()),
            "leading_missing_count": leading,
            "first_valid_return_date": (first_valid.date().isoformat()
                                        if first_valid is not None else ""),
            "last_valid_return_date": (valid.index.max().date().isoformat()
                                       if not valid.empty else ""),
            "origin_return_is_valid": bool(pd.notna(s.loc[origin])),
            "notes": " ".join(notes),
        })
    return rows


# --------------------------------------------------------------------------- #
# Cross-round validation
# --------------------------------------------------------------------------- #

def validate_contexts(
    returns: pd.DataFrame,
    slices: list[dict[str, object]],
    availability: pd.DataFrame,
) -> None:
    index = returns.index

    for prev, curr in zip(slices, slices[1:]):
        p0 = index.get_loc(prev["origin"])
        p1 = index.get_loc(curr["origin"])
        if p1 - p0 != FORECAST_STEPS:
            raise ValidationError("Origins do not advance by 20 weekdays.")
        overlap = prev["context_index"].intersection(curr["context_index"])
        if len(overlap) != CONTEXT_STEPS - FORECAST_STEPS:
            raise ValidationError("Consecutive contexts do not overlap by "
                                  f"{CONTEXT_STEPS - FORECAST_STEPS} rows.")
        if len(prev["forecast_index"].intersection(curr["forecast_index"])) != 0:
            raise ValidationError("Forecast periods overlap.")
        if prev["eval_end"] != curr["origin"]:
            raise ValidationError("Evaluation end does not chain to next origin.")

    # DRE zero returns wherever a context crosses the acquisition date.
    for slc in slices:
        after = [d for d in slc["context_index"] if d > DRE_LAST_GENUINE]
        if after:
            vals = returns.loc[after, "DRE"]
            if not (vals == 0.0).all():
                raise ValidationError(
                    f"Round {slc['round']}: DRE returns after "
                    f"{DRE_LAST_GENUINE.date()} are not all zero."
                )

    # Expected special behaviour.
    def valid_count(rnd: int, symbol: str) -> int:
        row = availability[(availability["round"] == rnd)
                           & (availability["symbol"] == symbol)]
        return int(row["valid_return_count"].iloc[0])

    if valid_count(1, "CARR") != CONTEXT_STEPS - 1:
        raise ValidationError("CARR should have 511 valid returns in round 1.")
    for rnd in range(2, 13):
        if valid_count(rnd, "CARR") != CONTEXT_STEPS:
            raise ValidationError(f"CARR should be complete in round {rnd}.")
    for rnd in range(1, 13):
        if valid_count(rnd, "OGN") >= CONTEXT_STEPS:
            raise ValidationError("OGN should be short in every round.")

    short = availability[(availability["valid_return_count"] < CONTEXT_STEPS)
                         & (~availability["symbol"].isin(["CARR", "OGN"]))]
    if not short.empty:
        raise ValidationError(
            "Unexpected short histories: "
            + ", ".join(f"{r.symbol} (round {r.round})"
                        for r in short.itertuples(index=False))
        )
    logger.info("Cross-round validation passed: movement, overlap, DRE zeros "
                "and expected CARR/OGN behaviour all confirmed.")


# --------------------------------------------------------------------------- #
# Metadata and report
# --------------------------------------------------------------------------- #

def build_schedule(slices: list[dict[str, object]]) -> pd.DataFrame:
    rows = []
    for slc in slices:
        r = slc["round"]
        rows.append({
            "round": r,
            "submission_deadline": SUBMISSION_DEADLINES[r - 1].date().isoformat(),
            "origin_date": slc["origin"].date().isoformat(),
            "context_start_date": slc["context_index"][0].date().isoformat(),
            "context_end_date": slc["context_index"][-1].date().isoformat(),
            "forecast_start_date": slc["forecast_index"][0].date().isoformat(),
            "forecast_end_date": slc["forecast_index"][-1].date().isoformat(),
            "context_steps": CONTEXT_STEPS,
            "forecast_steps": FORECAST_STEPS,
            "context_file": f"Data/contexts/round_{r:02d}_context.csv",
        })
    return pd.DataFrame(rows)


def build_report(
    schedule: pd.DataFrame,
    availability: pd.DataFrame,
    inputs_unchanged: bool,
) -> str:
    carr = availability[availability["symbol"] == "CARR"]
    ogn = availability[availability["symbol"] == "OGN"]
    dre_rounds = [int(r) for r in schedule.loc[
        pd.to_datetime(schedule["context_end_date"]) > DRE_LAST_GENUINE, "round"
    ]]

    sched_lines = [
        f"| {r.round} | {r.context_start_date} → {r.context_end_date} | "
        f"{r.forecast_start_date} → {r.forecast_end_date} | "
        f"{r.submission_deadline} |"
        for r in schedule.itertuples(index=False)
    ]
    carr_line = ", ".join(f"R{int(r.round)}: {int(r.valid_return_count)}"
                          for r in carr.itertuples(index=False))
    ogn_line = ", ".join(f"R{int(r.round)}: {int(r.valid_return_count)}"
                         for r in ogn.itertuples(index=False))

    lines = [
        "# Rolling-Origin Construction Report",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "",
        "## 1. Rolling-origin design",
        "",
        "- A forecast origin is the last date whose information a model may "
        "use: the Friday close ending an M6 round. Each context ends on that "
        "Friday because submissions were due the following Sunday, when the "
        "Friday close was already public — so the origin-date return is "
        "legitimately known and included.",
        "- The following 20 weekdays form the evaluation period being "
        "forecast; they are excluded from model input entirely, since "
        "including any of them would leak the future being predicted.",
        "- Each context is a fixed window of 512 shared weekday rows — 512 "
        "time positions on the common calendar, not 512 non-null values per "
        "asset.",
        "",
        "## 2. Context validation",
        "",
        "- 12 contexts were created, one per scored M6 round; every context "
        "has exactly 512 date rows and 100 asset columns (101 columns with "
        "`date`), official order, ascending unique dates.",
        "- No context contains any date after its origin (verified "
        "positionally for every round).",
        "- Context values are copied byte-for-byte from the preprocessed return "
        "file (written from an unparsed string copy) — no transformation, "
        "rounding or recalculation. Input files verified unchanged "
        f"by SHA-256: {'CONFIRMED' if inputs_unchanged else 'FAILED'}.",
        "",
        "| round | context (512 weekdays) | forecast (20 weekdays) | "
        "submission deadline |",
        "|---|---|---|---|",
        *sched_lines,
        "",
        "## 3. Context movement",
        "",
        "- Each origin lies exactly 20 weekday rows after the previous one, "
        "and each round's evaluation-end anchor is the next round's origin "
        "(verified for all 12 rounds).",
        "- Consecutive contexts therefore share 492 dates: each round drops "
        "the oldest 20 dates and appends the 20 newest observed dates "
        "(verified). Contexts overlap by design — they are histories — while "
        "the 12 evaluation periods are strictly disjoint (verified: no "
        "shared dates between consecutive forecast windows).",
        "",
        "## 4. Missing history",
        "",
        f"- CARR valid context returns by round: {carr_line}.",
        f"- OGN valid context returns by round: {ogn_line}.",
        "- No other asset-round pair has fewer than 512 valid returns "
        "(verified).",
        "- Leading missing values were preserved exactly as produced by "
        "preprocessing: no zero padding, backward-filling, window extension or "
        "fabricated history anywhere. Model-specific input preparation for "
        "short histories is deferred to the model-wrapper stage.",
        "",
        "## 5. DRE",
        "",
        f"- Round contexts including the post-acquisition period (context "
        f"end after {DRE_LAST_GENUINE.date()}): rounds "
        f"{', '.join(str(r) for r in dre_rounds)}.",
        "- In every such context, DRE's returns after 2022-10-03 are exactly "
        "zero, the competition treatment applied in preprocessing, copied "
        "unchanged; nothing "
        "was recalculated and no PLD data was used.",
        "- Documented future modelling rule (not implemented here): when "
        "later stages generate forecasts for rounds after the acquisition, "
        "DRE's competition return is known to remain zero, so forecast "
        "generation must respect that known-zero treatment where applicable "
        "rather than modelling DRE's stale series as if it were still "
        "trading.",
        "",
        "## 6. Readiness",
        "",
        "- All 12 rolling-origin contexts passed validation and are ready "
        "for the later model-wrapper stage.",
        "- Unresolved issues: none. Known accommodations for later stages: "
        "CARR (round 1: 511 returns) and OGN (all rounds: fewer than 512) "
        "require shorter model contexts; DRE requires the known-zero rule "
        "above.",
        "- No model inference, realised targets, quintile probabilities or "
        "RPS were produced, and no descriptive statistics of the returns "
        "were calculated.",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        hashes_before = {p: sha256_of(p) for p in (RETURNS_CSV, PRICES_CSV)}
        returns, returns_str, prices = load_panels()
        validate_schedule(returns.index)

        slices = [round_slices(returns.index, r) for r in range(1, 13)]

        # Per-asset context accounting. Not an output: it exists so the
        # cross-round checks below can assert the expected CARR/OGN gaps
        # and catch any other unexpectedly short history.
        availability = pd.DataFrame(
            [row for slc in slices for row in availability_rows(returns, slc)]
        )
        if len(availability) != 1200:
            raise ValidationError(
                f"Availability table has {len(availability)} rows, expected 1200."
            )
        validate_contexts(returns, slices, availability)

        for slc in slices:
            path = write_context(returns_str, slc)
            logger.info("Round %02d: context %s → %s written to %s",
                        slc["round"],
                        slc["context_index"][0].date(),
                        slc["context_index"][-1].date(), path.name)

        schedule = build_schedule(slices)
        OUT_SCHEDULE.parent.mkdir(parents=True, exist_ok=True)
        schedule.to_csv(OUT_SCHEDULE, index=False)
        logger.info("Wrote %s", OUT_SCHEDULE.name)

        inputs_unchanged = all(sha256_of(p) == h for p, h in hashes_before.items())
        if not inputs_unchanged:
            raise ValidationError("An input file changed while contexts were being built.")

        OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
        OUT_REPORT.write_text(
            build_report(schedule, availability, inputs_unchanged),
            encoding="utf-8",
        )
        logger.info("Wrote %s", OUT_REPORT)
    except ValidationError as exc:
        logger.error("Validation failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
