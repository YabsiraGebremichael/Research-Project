"""Download Dataset A: raw Yahoo Finance adjusted closes for the 100 M6 assets.

Reads the official universe from Data/raw/m6_official/assets_m6.csv in released
order, bulk-downloads daily adjusted closes with yfinance and retries any absent
or all-null symbol individually. The raw file is written with no filling,
interpolation or calendar alignment.

Outputs Data/raw/yahoo/dataset_a_adjusted_close.csv
        reports/dataset_a_download_report.md

Usage: python scripts/pipeline/download_dataset_a.py [--force]
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OFFICIAL_CSV = PROJECT_ROOT / "Data" / "raw" / "m6_official" / "assets_m6.csv"

OUTPUT_CSV = PROJECT_ROOT / "Data" / "raw" / "yahoo" / "dataset_a_adjusted_close.csv"
REPORT_MD = PROJECT_ROOT / "reports" / "dataset_a_download_report.md"

START_DATE = "2020-01-01"
END_DATE = "2023-02-04"  # exclusive: last included calendar date is 2023-02-03
LAST_INCLUDED_DATE = "2023-02-03"
EXPECTED_ASSET_COUNT = 100

# Sufficiency check: at least 513 non-null observations on or before this date
# (needed later for one 512-observation context plus one forecast origin).
SUFFICIENCY_CUTOFF = "2022-03-04"
SUFFICIENCY_MIN_OBS = 513

META_TICKER_CHANGE_DATE = "2022-06-09"  # FB -> META
DRE_LAST_TRADING_DATE = "2022-10-03"    # acquired by PLD

logger = logging.getLogger("download_dataset_a")


# --------------------------------------------------------------------------- #
# Result bookkeeping
# --------------------------------------------------------------------------- #

@dataclass
class DownloadResult:
    """Everything the report needs to know about how the download went."""

    symbols: list[str]
    prices: pd.DataFrame  # date index, one column per official symbol
    retried_symbols: list[str] = field(default_factory=list)  # attempted individually
    failed_symbols: list[str] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Official asset universe
# --------------------------------------------------------------------------- #

def locate_official_csv() -> Path:
    """Return the official M6 asset CSV, failing loudly if it is absent."""
    if not OFFICIAL_CSV.is_file():
        raise FileNotFoundError(f"Official M6 CSV not found at {OFFICIAL_CSV}")
    return OFFICIAL_CSV


def load_official_symbols(csv_path: Path) -> list[str]:
    """Read the official universe, preserving first-occurrence symbol order."""
    df = pd.read_csv(csv_path)
    required = {"symbol", "date", "price"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{csv_path} lacks required columns {sorted(missing)}; "
            f"found {list(df.columns)}"
        )
    symbols = list(dict.fromkeys(df["symbol"].astype(str).str.strip()))
    if len(symbols) != EXPECTED_ASSET_COUNT:
        raise ValueError(
            f"Expected exactly {EXPECTED_ASSET_COUNT} unique symbols in "
            f"{csv_path}, found {len(symbols)}"
        )
    logger.info("Loaded %d official symbols from %s", len(symbols), csv_path)
    return symbols


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #

def extract_adj_close(raw: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    """Pull the ``Adj Close`` field out of a yfinance result.

    Handles the MultiIndex column layouts yfinance can return
    (``(field, ticker)`` or ``(ticker, field)``) and the flat single-ticker
    layout. Never substitutes ``Close`` for ``Adj Close``.
    """
    if raw.empty:
        return pd.DataFrame(index=pd.DatetimeIndex([], name="date"))

    if isinstance(raw.columns, pd.MultiIndex):
        level0 = raw.columns.get_level_values(0)
        level1 = raw.columns.get_level_values(1)
        if "Adj Close" in level0:
            adj = raw.xs("Adj Close", axis=1, level=0)
        elif "Adj Close" in level1:
            adj = raw.xs("Adj Close", axis=1, level=1)
        else:
            raise KeyError(
                "'Adj Close' not present in yfinance MultiIndex columns; "
                f"fields seen: {sorted(set(level0) | set(level1))}. "
                "Refusing to substitute 'Close'."
            )
        if isinstance(adj, pd.Series):
            adj = adj.to_frame(name=symbols[0] if len(symbols) == 1 else adj.name)
    else:
        if "Adj Close" not in raw.columns:
            raise KeyError(
                "'Adj Close' not present in yfinance flat columns "
                f"{list(raw.columns)}. Refusing to substitute 'Close'."
            )
        if len(symbols) != 1:
            raise ValueError(
                "Flat (non-MultiIndex) columns returned for a multi-symbol "
                "request; cannot attribute 'Adj Close' to a single symbol."
            )
        adj = raw[["Adj Close"]].rename(columns={"Adj Close": symbols[0]})

    adj = adj.copy()
    adj.index = pd.to_datetime(adj.index).tz_localize(None).normalize()
    adj.index.name = "date"
    return adj


def download_bulk(symbols: list[str]) -> pd.DataFrame:
    """Bulk-download all symbols in one yfinance call."""
    logger.info("Bulk download of %d symbols (%s to %s, exclusive) ...",
                len(symbols), START_DATE, END_DATE)
    raw = yf.download(
        tickers=symbols,
        start=START_DATE,
        end=END_DATE,
        interval="1d",
        auto_adjust=False,
        actions=False,
        progress=False,
        threads=True,
        group_by="column",
    )
    return extract_adj_close(raw, symbols)


def download_single(symbol: str) -> Optional[pd.Series]:
    """Download one symbol; return its Adj Close series or None on failure."""
    try:
        raw = yf.download(
            tickers=symbol,
            start=START_DATE,
            end=END_DATE,
            interval="1d",
            auto_adjust=False,
            actions=False,
            progress=False,
            threads=False,
        )
    except Exception as exc:  # network / ticker errors must not kill the run
        logger.warning("Individual download raised for %s: %s", symbol, exc)
        return None
    if raw.empty:
        logger.warning("Individual download returned no rows for %s", symbol)
        return None
    adj = extract_adj_close(raw, [symbol])
    series = adj.iloc[:, 0]
    if series.dropna().empty:
        logger.warning("Individual download returned only nulls for %s", symbol)
        return None
    series.name = symbol
    return series


def download_dataset(symbols: list[str]) -> DownloadResult:
    """Bulk download, then retry absent/all-null symbols individually.

    Failed symbols are kept as all-null columns so no asset is silently removed.
    """
    bulk = download_bulk(symbols)

    needs_retry = [
        s for s in symbols
        if s not in bulk.columns or bulk[s].dropna().empty
    ]
    if needs_retry:
        logger.info("Retrying %d symbol(s) individually: %s",
                    len(needs_retry), ", ".join(needs_retry))

    failed: list[str] = []
    errors: dict[str, str] = {}
    retry_series: list[pd.Series] = []

    for symbol in needs_retry:
        series = download_single(symbol)
        if series is None:
            failed.append(symbol)
            errors[symbol] = "no data or only null values from Yahoo Finance"
        else:
            retry_series.append(series)

    frames = [bulk.drop(columns=needs_retry, errors="ignore")]
    if retry_series:
        frames.append(pd.concat(retry_series, axis=1))
    prices = pd.concat(frames, axis=1)

    # Enforce the official universe and order; failed assets stay as null columns.
    prices = prices.reindex(columns=symbols)
    prices = prices.sort_index()
    prices = prices[~prices.index.duplicated(keep="first")]
    prices = prices.loc[prices.index <= pd.Timestamp(LAST_INCLUDED_DATE)]

    return DownloadResult(
        symbols=symbols,
        prices=prices,
        retried_symbols=needs_retry,
        failed_symbols=failed,
        errors=errors,
    )


# --------------------------------------------------------------------------- #
# Validation of an existing Dataset A file
# --------------------------------------------------------------------------- #

def existing_dataset_is_valid(path: Path, symbols: list[str]) -> bool:
    """Check whether an existing Dataset A file can be reused (no --force)."""
    if not path.is_file():
        return False
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        logger.warning("Existing dataset unreadable (%s); re-downloading.", exc)
        return False
    expected_cols = ["date"] + symbols
    if list(df.columns) != expected_cols:
        logger.warning("Existing dataset columns do not match the official "
                       "order; re-downloading.")
        return False
    dates = pd.to_datetime(df["date"], errors="coerce")
    if dates.isna().any():
        logger.warning("Existing dataset has unparseable dates; re-downloading.")
        return False
    if dates.duplicated().any():
        logger.warning("Existing dataset has duplicate dates; re-downloading.")
        return False
    if (dates > pd.Timestamp(LAST_INCLUDED_DATE)).any():
        logger.warning("Existing dataset has dates after %s; re-downloading.",
                       LAST_INCLUDED_DATE)
        return False
    logger.info("Existing Dataset A at %s passed validation; reusing it. "
                "Use --force to re-download.", path)
    return True


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #

def per_asset_summary(prices: pd.DataFrame) -> pd.DataFrame:
    """First/last available date and non-null counts for every asset."""
    rows = []
    cutoff = pd.Timestamp(SUFFICIENCY_CUTOFF)
    for symbol in prices.columns:
        series = prices[symbol].dropna()
        n_by_cutoff = int(series.loc[series.index <= cutoff].shape[0])
        rows.append({
            "symbol": symbol,
            "first_date": series.index.min().date().isoformat() if not series.empty else "—",
            "last_date": series.index.max().date().isoformat() if not series.empty else "—",
            "non_null_obs": int(series.shape[0]),
            "non_null_obs_by_cutoff": n_by_cutoff,
            "sufficient": n_by_cutoff >= SUFFICIENCY_MIN_OBS,
        })
    return pd.DataFrame(rows)


def build_report(result: DownloadResult, output_csv: Path) -> str:
    """Assemble the markdown validation report."""
    prices = result.prices
    summary = per_asset_summary(prices)

    n_success = int((summary["non_null_obs"] > 0).sum())
    insufficient = summary.loc[~summary["sufficient"], "symbol"].tolist()
    structurally_valid = list(prices.columns) == result.symbols and len(prices.columns) == EXPECTED_ASSET_COUNT
    data_complete = not result.failed_symbols and not insufficient

    # META findings
    meta = prices["META"].dropna() if "META" in prices.columns else pd.Series(dtype=float)
    if meta.empty:
        meta_finding = "META returned no data from Yahoo Finance."
    else:
        pre_change = meta.loc[meta.index < pd.Timestamp(META_TICKER_CHANGE_DATE)]
        meta_finding = (
            f"META series spans {meta.index.min().date()} to "
            f"{meta.index.max().date()} with {len(meta)} non-null observations. "
            f"It contains {len(pre_change)} observations before the FB→META "
            f"ticker change on {META_TICKER_CHANGE_DATE}, so Yahoo's META "
            f"series {'does' if len(pre_change) > 0 else 'does NOT'} include "
            f"the pre-change (FB-era) history. The asset is stored as a single "
            f"continuous `META` column; no separate FB column was created and "
            f"no artificial values were inserted at the ticker-change date."
        )

    # DRE findings
    dre = prices["DRE"].dropna() if "DRE" in prices.columns else pd.Series(dtype=float)
    if dre.empty:
        dre_finding = (
            "DRE returned no data from Yahoo Finance. Dataset A is therefore "
            "INCOMPLETE for DRE; no values were fabricated."
        )
    else:
        dre_finding = (
            f"DRE series spans {dre.index.min().date()} to "
            f"{dre.index.max().date()} with {len(dre)} non-null observations. "
            f"DRE stopped updating on {DRE_LAST_TRADING_DATE} after its "
            f"acquisition by PLD; its final available date in this download is "
            f"{dre.index.max().date()}. The raw Yahoo output is preserved: no "
            f"no-price-change rule was applied at this stage."
        )

    non_null_any = prices.dropna(how="all")
    earliest = non_null_any.index.min().date().isoformat() if not non_null_any.empty else "—"
    latest = non_null_any.index.max().date().isoformat() if not non_null_any.empty else "—"

    lines = [
        "# Dataset A Download Report",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"Output file: `{output_csv.relative_to(PROJECT_ROOT).as_posix()}`",
        "",
        "## Download summary",
        "",
        f"- Requested date range: {START_DATE} to {END_DATE} (end exclusive; "
        f"last included calendar date {LAST_INCLUDED_DATE})",
        f"- Actual earliest returned date: {earliest}",
        f"- Actual latest returned date: {latest}",
        f"- Official assets: {len(result.symbols)}",
        f"- Successfully downloaded (≥1 non-null observation): {n_success}",
        f"- Required individual retries: "
        + (", ".join(result.retried_symbols) if result.retried_symbols else "none"),
        f"- Failed / only-null assets: "
        + (", ".join(result.failed_symbols) if result.failed_symbols else "none"),
        "",
        "## Status",
        "",
        f"- **Structurally valid**: {'YES' if structurally_valid else 'NO'} — "
        f"the output contains `date` plus all {EXPECTED_ASSET_COUNT} official "
        f"asset columns in official order.",
        f"- **Data complete**: {'YES' if data_complete else 'NO'} — "
        + ("every asset has data and sufficient history."
           if data_complete else
           "one or more assets failed or lack sufficient usable history "
           "(see below). Structural validity alone does not make Dataset A "
           "complete."),
        "",
        "## Integrity confirmations",
        "",
        "- No missing values were filled, interpolated, forward-/backward-filled "
        "or replaced with zeros; gaps caused by differing exchange holidays "
        "remain missing.",
        "- No official asset was silently removed: all "
        f"{EXPECTED_ASSET_COUNT} assets are present as columns, including any "
        "that failed to download (kept as all-null columns).",
        "- No returns were calculated, no standardisation was applied, no "
        "shared business-day calendar was constructed, and the official M6 "
        "price file was not used to fill Yahoo values.",
        "",
        "## META findings",
        "",
        meta_finding,
        "",
        "## DRE findings",
        "",
        dre_finding,
        "",
        f"> **Note for preprocessing**: to reproduce the official competition "
        f"treatment, preprocessing must carry DRE's final available price forward "
        f"after {DRE_LAST_TRADING_DATE} (zero subsequent return). This was "
        f"deliberately NOT applied in this raw-download stage.",
        "",
        "## Ticker-related errors",
        "",
    ]
    if result.errors:
        lines += [f"- `{s}`: {msg}" for s, msg in result.errors.items()]
    else:
        lines.append("- None.")

    lines += [
        "",
        f"## Assets with fewer than {SUFFICIENCY_MIN_OBS} non-null observations "
        f"on or before {SUFFICIENCY_CUTOFF}",
        "",
    ]
    if insufficient:
        lines += [f"- `{s}` "
                  f"({int(summary.loc[summary['symbol'] == s, 'non_null_obs_by_cutoff'].iloc[0])} "
                  f"observations)" for s in insufficient]
    else:
        lines.append(f"- None. Every asset has at least {SUFFICIENCY_MIN_OBS} "
                     f"non-null observations on or before {SUFFICIENCY_CUTOFF}.")

    lines += [
        "",
        "## Per-asset availability",
        "",
        "| symbol | first date | last date | non-null obs | non-null obs "
        f"≤ {SUFFICIENCY_CUTOFF} | ≥ {SUFFICIENCY_MIN_OBS} by cutoff |",
        "|---|---|---|---|---|---|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.symbol} | {row.first_date} | {row.last_date} | "
            f"{row.non_null_obs} | {row.non_null_obs_by_cutoff} | "
            f"{'yes' if row.sufficient else 'NO'} |"
        )
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #

def write_dataset(prices: pd.DataFrame, path: Path) -> None:
    """Write Dataset A with a leading `date` column."""
    path.parent.mkdir(parents=True, exist_ok=True)
    out = prices.copy()
    out.insert(0, "date", out.index.strftime("%Y-%m-%d"))
    out.to_csv(path, index=False)
    logger.info("Wrote Dataset A: %s (%d rows x %d columns)",
                path, out.shape[0], out.shape[1])


def write_report(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    logger.info("Wrote report: %s", path)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                        help="re-download even if a valid Dataset A file exists")
    args = parser.parse_args()

    try:
        csv_path = locate_official_csv()
        symbols = load_official_symbols(csv_path)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        logger.error("Cannot establish the official asset universe: %s", exc)
        return 1

    if not args.force and existing_dataset_is_valid(OUTPUT_CSV, symbols):
        print(f"Existing Dataset A at {OUTPUT_CSV} is valid; nothing to do. "
              f"Use --force to re-download.")
        return 0

    try:
        result = download_dataset(symbols)
    except Exception as exc:
        logger.error("Download failed: %s", exc)
        return 1

    if result.prices.empty:
        logger.error("Download produced no rows at all; refusing to write an "
                     "empty Dataset A.")
        return 1

    write_dataset(result.prices, OUTPUT_CSV)
    write_report(build_report(result, OUTPUT_CSV), REPORT_MD)

    if result.failed_symbols:
        logger.warning("Dataset A is INCOMPLETE: failed assets: %s",
                       ", ".join(result.failed_symbols))
    return 0


if __name__ == "__main__":
    sys.exit(main())
