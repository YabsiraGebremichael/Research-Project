"""Download the extended EODHD history used for Chronos-2 fine-tuning.

Daily OHLCV for the 100 official M6 assets from 2000-01-01, or each asset's
earliest genuine observation, through 2023-02-03. Genuine inception is
preserved: nothing is backfilled, padded or proxied. Dataset D is never written
to, but each asset is compared against it and any material discrepancy is
reported.

The token is loaded from the git-ignored env file and never printed, logged or
written to an output; a secret-safety check aborts if an env file is tracked.

The network path is unverified: this script has not been run against the live
API, so treat the first real run as a smoke test with --limit 3.

Writes extended_eodhd_ohlcv_2000_2023.csv and extended_eodhd_status.csv into
TSFM Uncertainty Experiment/data/.

Usage: python "FinQ Boost Reimplementation/download_extended_eodhd.py"
       [--dry-run] [--limit N]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "pipeline"))

from eodhd_client import (  # noqa: E402
    EodhdClient,
    identifier_candidates,
    load_token,
    official_asset_order,
    verify_git_secret_safety,
    verify_subscription,
)

OUT_DIR = PROJECT_ROOT / "TSFM Uncertainty Experiment" / "data"
OUT_CSV = OUT_DIR / "extended_eodhd_ohlcv_2000_2023.csv"
STATUS_CSV = OUT_DIR / "extended_eodhd_status.csv"

DATASET_D_CSV = PROJECT_ROOT / "Data" / "raw" / "dataset_d_eodhd" / "dataset_d_ohlcv.csv"

FROM_DATE = "2000-01-01"
TO_DATE = "2023-02-03"          # EODHD `to` is inclusive; M6 evaluation end
FIELDS = ["open", "high", "low", "close", "adjusted_close", "volume"]

# Consistency thresholds for the Dataset D overlap. Same provider and same
# identifiers, so agreement should be near-exact; anything looser is reported.
MAX_MEDIAN_REL_DIFF = 1e-6
MAX_WORST_REL_DIFF = 1e-3

logger = logging.getLogger("download_extended_eodhd")


class AcquisitionError(RuntimeError):
    """A safety or structural expectation was violated."""


# --------------------------------------------------------------------------- #
# Safety
# --------------------------------------------------------------------------- #

def guard_output_paths() -> None:
    """Refuse to run if the outputs could collide with existing project data."""
    for path in (OUT_CSV, STATUS_CSV):
        resolved = path.resolve()
        if not str(resolved).startswith(str(HERE.resolve())):
            raise AcquisitionError(f"Output {resolved} escapes {HERE}.")
        if "dataset_d" in resolved.name or "dataset_d_eodhd" in str(resolved.parent):
            raise AcquisitionError(f"Output {resolved} collides with Dataset D.")
    if not DATASET_D_CSV.is_file():
        raise AcquisitionError(f"Dataset D reference missing: {DATASET_D_CSV}")
    logger.info("Output guard passed: writes confined to %s; Dataset D read-only.",
                HERE.name)


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #

def fetch_asset(client: EodhdClient, symbol: str
                ) -> tuple[pd.DataFrame, str, list[str]]:
    """One asset's raw OHLCV over the extended range.

    Mirrors the Dataset D fetch exactly, including the identifier candidate
    order, so the same securities are retrieved under the same codes. An empty
    frame means every candidate failed; no substitute is ever used.
    """
    tried: list[str] = []
    for candidate in identifier_candidates(symbol):
        tried.append(candidate)
        records = client.get(f"eod/{candidate}",
                             **{"from": FROM_DATE, "to": TO_DATE,
                                "period": "d", "order": "a"})
        if isinstance(records, list) and records:
            frame = pd.DataFrame(records)
            missing = [f for f in FIELDS if f not in frame.columns]
            if missing:
                raise AcquisitionError(f"{candidate}: response lacks {missing}")
            frame = frame[["date"] + FIELDS].copy()
            frame["date"] = pd.to_datetime(frame["date"])
            frame.insert(0, "symbol", symbol)
            frame.insert(1, "eodhd_identifier", candidate)
            return frame.sort_values("date").drop_duplicates("date"), candidate, tried
    return pd.DataFrame(), "", tried


# --------------------------------------------------------------------------- #
# Consistency against Dataset D
# --------------------------------------------------------------------------- #

def compare_with_dataset_d(symbol: str, frame: pd.DataFrame,
                           dataset_d: pd.DataFrame) -> dict[str, object]:
    """Compare the overlapping history with Dataset D, field by field.

    Same provider, same identifier, so the expectation is near-exact equality.
    Reports the median and worst relative difference on adjusted_close and
    close, and the number of overlapping dates. Dataset D never supplies a value
    to the new download.
    """
    reference = dataset_d[dataset_d["symbol"] == symbol].set_index("date")
    if reference.empty:
        return {"overlap_obs": 0, "verdict": "no Dataset D reference",
                "median_rel_diff_adjclose": None, "worst_rel_diff_adjclose": None,
                "worst_rel_diff_close": None, "overlap_first": None,
                "overlap_last": None}

    current = frame.set_index("date")
    common = current.index.intersection(reference.index)
    if len(common) == 0:
        return {"overlap_obs": 0, "verdict": "NO OVERLAP - investigate",
                "median_rel_diff_adjclose": None, "worst_rel_diff_adjclose": None,
                "worst_rel_diff_close": None, "overlap_first": None,
                "overlap_last": None}

    def relative(field: str) -> np.ndarray:
        a = current.loc[common, field].astype(float).to_numpy()
        b = reference.loc[common, field].astype(float).to_numpy()
        scale = np.maximum(np.abs(b), 1e-12)
        return np.abs(a - b) / scale

    adj = relative("adjusted_close")
    close = relative("close")
    median_adj = float(np.median(adj))
    worst_adj = float(adj.max())
    worst_close = float(close.max())

    # Dataset D's own known KR adjusted-close break is expected to reappear
    # identically; it is a provider artifact, not a download discrepancy.
    if median_adj <= MAX_MEDIAN_REL_DIFF and worst_adj <= MAX_WORST_REL_DIFF:
        verdict = "consistent"
    elif median_adj <= MAX_MEDIAN_REL_DIFF:
        verdict = f"consistent except {int((adj > MAX_WORST_REL_DIFF).sum())} day(s)"
    else:
        verdict = "MATERIAL DISCREPANCY - investigate"

    return {
        "overlap_obs": int(len(common)),
        "overlap_first": str(min(common).date()),
        "overlap_last": str(max(common).date()),
        "median_rel_diff_adjclose": round(median_adj, 12),
        "worst_rel_diff_adjclose": round(worst_adj, 8),
        "worst_rel_diff_close": round(worst_close, 8),
        "verdict": verdict,
    }


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #

def load_progress() -> tuple[pd.DataFrame, set[str]]:
    if not (OUT_CSV.is_file() and STATUS_CSV.is_file()):
        return pd.DataFrame(), set()
    status = pd.read_csv(STATUS_CSV)
    done = set(status.loc[status["status"] == "ok", "symbol"])
    existing = pd.read_csv(OUT_CSV, parse_dates=["date"])
    return existing[existing["symbol"].isin(done)], done


def write_outputs(blocks: list[pd.DataFrame], rows: list[dict]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if rows:
        pd.DataFrame(rows).to_csv(STATUS_CSV, index=False)
    if blocks:
        table = pd.concat(blocks, ignore_index=True).sort_values(["symbol", "date"])
        table.to_csv(OUT_CSV, index=False)


def summarise(symbol: str, frame: pd.DataFrame, identifier: str,
              tried: list[str], status: str, note: str,
              comparison: dict[str, object]) -> dict[str, object]:
    return {
        "symbol": symbol,
        "eodhd_identifier": identifier,
        "identifiers_tried": " | ".join(tried),
        "status": status,
        "n_observations": 0 if frame.empty else len(frame),
        "first_date": None if frame.empty else str(frame["date"].min().date()),
        "last_date": None if frame.empty else str(frame["date"].max().date()),
        **comparison,
        "note": note,
    }


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true",
                        help="re-download every asset, ignoring saved progress")
    parser.add_argument("--limit", type=int, default=None,
                        help="only process the first N assets (smoke testing)")
    parser.add_argument("--dry-run", action="store_true",
                        help="run every non-network check and exit without "
                             "contacting EODHD or writing any output")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    symbols = official_asset_order()
    if len(symbols) != 100:
        raise AcquisitionError(f"Expected 100 official assets, found {len(symbols)}")
    if args.limit:
        symbols = symbols[: args.limit]

    guard_output_paths()
    verify_git_secret_safety()

    dataset_d = pd.read_csv(DATASET_D_CSV, parse_dates=["date"])
    print(f"Dataset D reference loaded read-only: {len(dataset_d)} rows, "
          f"{dataset_d['symbol'].nunique()} assets, "
          f"{dataset_d['date'].min().date()} .. {dataset_d['date'].max().date()}")

    if args.dry_run:
        print(f"\nDRY RUN - no network call, no file written.")
        print(f"  would request : {FROM_DATE} .. {TO_DATE}")
        print(f"  fields        : {', '.join(FIELDS)}")
        print(f"  assets        : {len(symbols)}")
        print(f"  outputs       : {OUT_CSV.relative_to(PROJECT_ROOT)}")
        print(f"                  {STATUS_CSV.relative_to(PROJECT_ROOT)}")
        sample = symbols[0]
        print(f"  identifier candidates for {sample}: {identifier_candidates(sample)}")
        print(f"  identifier candidates for RE     : {identifier_candidates('RE')}")
        print(f"  identifier candidates for SEGA.L : {identifier_candidates('SEGA.L')}")
        block = dataset_d[dataset_d["symbol"] == sample]
        print(f"  self-comparison of {sample} against Dataset D (must be exact): "
              f"{compare_with_dataset_d(sample, block, dataset_d)['verdict']}")
        return 0

    try:
        token = load_token()
    except RuntimeError as exc:
        # The message names the expected files and never contains the token.
        raise AcquisitionError(str(exc)) from None
    client = EodhdClient(token)
    subscription = verify_subscription(client, from_date=FROM_DATE)
    print(f"\nSubscription OK (type={subscription.get('subscription_type')}, "
          f"requests remaining today="
          f"{subscription.get('requests_remaining_today', 'n/a')})")
    print(f"Requesting {FROM_DATE} .. {TO_DATE} for {len(symbols)} assets\n")

    existing, done = (pd.DataFrame(), set()) if args.refresh else load_progress()
    if done:
        print(f"Resuming: {len(done)} asset(s) already retrieved.\n")

    blocks = [existing] if not existing.empty else []
    rows: list[dict] = []
    stopped = False

    for i, symbol in enumerate(symbols, 1):
        if symbol in done:
            block = existing[existing["symbol"] == symbol]
            rows.append(summarise(symbol, block, block["eodhd_identifier"].iloc[0],
                                  [], "ok", "reused saved download",
                                  compare_with_dataset_d(symbol, block, dataset_d)))
            continue
        try:
            frame, identifier, tried = fetch_asset(client, symbol)
        except Exception as exc:                       # already redacted upstream
            print(f"\nSTOPPED at {symbol}: {exc}")
            print("Assets already retrieved are saved; re-run to resume.")
            stopped = True
            break

        if frame.empty:
            rows.append(summarise(symbol, frame, "", tried, "failed",
                                  f"no data from any candidate ({', '.join(tried)})",
                                  compare_with_dataset_d(symbol, frame, dataset_d)))
            print(f"  [{i:3d}/{len(symbols)}] {symbol:<8} FAILED - tried {tried}")
            continue

        comparison = compare_with_dataset_d(symbol, frame, dataset_d)
        blocks.append(frame)
        rows.append(summarise(symbol, frame, identifier, tried, "ok", "", comparison))
        print(f"  [{i:3d}/{len(symbols)}] {symbol:<8} {identifier:<10} "
              f"{len(frame):>5} rows  {frame['date'].min().date()} .. "
              f"{frame['date'].max().date()}  vs Dataset D: {comparison['verdict']}")
        write_outputs(blocks, rows)                    # persist after each asset

    write_outputs(blocks, rows)

    status = pd.DataFrame(rows)
    ok = status[status["status"] == "ok"]
    print("\n" + "=" * 70)
    print(f"Retrieved {len(ok)}/{len(symbols)} assets"
          + (" (STOPPED EARLY)" if stopped else ""))
    if not ok.empty:
        print(f"Earliest observation across assets: {ok['first_date'].min()}")
        print(f"Assets starting in 2000          : "
              f"{int((ok['first_date'] <= '2000-12-31').sum())}")
        bad = ok[~ok["verdict"].astype(str).str.startswith("consistent")]
        print(f"Dataset D consistency            : "
              f"{len(ok) - len(bad)}/{len(ok)} consistent")
        if not bad.empty:
            print("  assets needing investigation:")
            for record in bad.itertuples():
                print(f"    {record.symbol}: {record.verdict} "
                      f"(median rel diff {record.median_rel_diff_adjclose})")
    print(f"\nWrote {OUT_CSV.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {STATUS_CSV.relative_to(PROJECT_ROOT)}")
    return 1 if stopped else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AcquisitionError as exc:
        logging.getLogger("download_extended_eodhd").error("Aborted: %s", exc)
        sys.exit(1)
