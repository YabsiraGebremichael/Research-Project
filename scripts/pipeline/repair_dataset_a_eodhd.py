"""Repair Dataset A from EODHD for the three assets Yahoo cannot serve.

DRE (acquired, last traded 2022-10-03), RE (renamed EG) and WRK (merged) have no
Yahoo history. This resolves their EODHD identifiers, downloads daily adjusted
closes for 2020-01-01..2023-02-03, validates each candidate's returns against the
official M6 price file over the overlap, and writes a repaired copy alongside the
original, which is never modified.

The EODHD token is read from the local env file and is never printed, logged or
embedded in a recorded URL.

Output Data/raw/yahoo/dataset_a_adjusted_close_repaired.csv

Usage: python scripts/pipeline/repair_dataset_a_eodhd.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import requests
from dotenv import load_dotenv

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ENV_FILES = (PROJECT_ROOT / "proj.env", PROJECT_ROOT / ".env")
TOKEN_ENV_NAMES = ("EODHD_API_TOKEN", "EODHD_API_KEY")

ORIGINAL_DATASET = PROJECT_ROOT / "Data" / "raw" / "yahoo" / "dataset_a_adjusted_close.csv"
REPAIRED_DATASET = PROJECT_ROOT / "Data" / "raw" / "yahoo" / "dataset_a_adjusted_close_repaired.csv"
OFFICIAL_M6_CSV = PROJECT_ROOT / "Data" / "raw" / "m6_official" / "assets_m6.csv"
RAW_SAVE_DIR = PROJECT_ROOT / "Data" / "raw" / "recovered_sources" / "eodhd"
REPORT_MD = PROJECT_ROOT / "reports" / "dataset_a_repair_report.md"

API_BASE = "https://eodhd.com/api"
FROM_DATE = "2020-01-01"
TO_DATE = "2023-02-03"  # EODHD `to` is inclusive

MIN_REMAINING_REQUESTS = 50

# Acceptance thresholds for overlap validation.
MIN_OVERLAP_OBS = 60
MIN_RETURN_CORR = 0.99
MAX_MEDIAN_ABS_RETURN_DIFF = 0.002
LATEST_ACCEPTABLE_FIRST_DATE = "2020-01-10"
MAJOR_DISAGREEMENT_THRESHOLD = 0.01  # absolute daily-return difference
RATIO_JUMP_THRESHOLD = 0.01          # day-on-day jump in price-level ratio

logger = logging.getLogger("repair_dataset_a_eodhd")


@dataclass
class TargetSpec:
    """One M6 asset to repair."""

    canonical: str                 # Dataset A column name
    company_keywords: list[str]    # uppercase substrings identifying the company
    code_candidates: list[str]     # ticker codes worth inspecting
    search_queries: list[str]
    # Official series is only comparable up to this date (DRE goes flat after
    # the competition froze its price); None = full official range.
    official_cutoff: Optional[str] = None
    # The recovered series must reach at least this date to be usable.
    required_last_date: str = TO_DATE


TARGETS = [
    TargetSpec(
        canonical="DRE",
        company_keywords=["DUKE REALTY"],
        code_candidates=["DRE"],
        search_queries=["DRE", "Duke Realty Corporation"],
        official_cutoff="2022-10-03",
        required_last_date="2022-09-26",  # last trading week before acquisition
    ),
    TargetSpec(
        canonical="RE",
        company_keywords=["EVEREST RE", "EVEREST GROUP"],
        code_candidates=["RE", "EG"],
        search_queries=["RE", "Everest Re Group", "EG", "Everest Group"],
    ),
    TargetSpec(
        canonical="WRK",
        company_keywords=["WESTROCK"],
        code_candidates=["WRK"],
        search_queries=["WRK", "WestRock Company"],
    ),
]


@dataclass
class Candidate:
    """A possible EODHD identifier for a target asset."""

    code: str
    exchange: str
    name: str
    status: str                          # 'active list', 'delisted list', 'search'
    identifier: str = ""                 # e.g. 'DRE.US'
    first_date: Optional[str] = None
    last_date: Optional[str] = None
    n_obs: int = 0
    has_adjusted_close: bool = False
    download_error: Optional[str] = None
    validation: dict[str, Any] = field(default_factory=dict)
    accepted: bool = False
    decision: str = ""
    records: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.identifier:
            self.identifier = f"{self.code}.{self.exchange}"


# --------------------------------------------------------------------------- #
# Token handling and API access
# --------------------------------------------------------------------------- #

class EodhdClient:
    """Thin EODHD client that keeps the token out of logs and errors."""

    def __init__(self, token: str) -> None:
        self._token = token
        self._session = requests.Session()

    def _redact(self, text: str) -> str:
        return text.replace(self._token, "***REDACTED***")

    def get(self, path: str, **params: Any) -> Any:
        """GET an API path; the token never appears in logs or exceptions."""
        params = {**params, "api_token": self._token, "fmt": "json"}
        url = f"{API_BASE}/{path}"
        logger.info("GET /%s", path)
        try:
            resp = self._session.get(url, params=params, timeout=60)
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Request to /{path} failed: {self._redact(str(exc))}"
            ) from None
        if resp.status_code != 200:
            body = self._redact(resp.text[:300])
            raise RuntimeError(
                f"EODHD returned HTTP {resp.status_code} for /{path}: {body}"
            )
        try:
            return resp.json()
        except ValueError:
            raise RuntimeError(
                f"EODHD returned non-JSON for /{path}: "
                f"{self._redact(resp.text[:200])}"
            ) from None

    def token_in(self, text: str) -> bool:
        return self._token in text


def load_token() -> str:
    """Load the EODHD token from local env files without exposing it."""
    for env_file in ENV_FILES:
        if env_file.is_file():
            load_dotenv(env_file, override=False)
    for name in TOKEN_ENV_NAMES:
        token = os.getenv(name)
        if token and token.strip():
            logger.info("Loaded EODHD token from environment variable %s "
                        "(value not shown).", name)
            return token.strip()
    raise RuntimeError(
        "EODHD API token not found. Define EODHD_API_TOKEN (or EODHD_API_KEY) "
        f"in one of: {', '.join(str(p) for p in ENV_FILES)}. "
        "The token value is never printed."
    )


def verify_git_secret_safety() -> None:
    """Fail if an env file is tracked by git; warn if .gitignore misses it."""
    import subprocess

    tracked = subprocess.run(
        ["git", "ls-files"], cwd=PROJECT_ROOT, capture_output=True, text=True,
    ).stdout.splitlines()
    for env_file in ENV_FILES:
        rel = env_file.name
        if rel in tracked:
            raise RuntimeError(
                f"SECURITY: {rel} is tracked by git. Remove it from the index "
                "(git rm --cached) and rotate the token before continuing."
            )
    gitignore = PROJECT_ROOT / ".gitignore"
    patterns = gitignore.read_text(encoding="utf-8").splitlines() if gitignore.is_file() else []
    for env_file in ENV_FILES:
        if env_file.is_file() and not any(
            p.strip() in (env_file.name, "*.env", ".env") for p in patterns
        ):
            raise RuntimeError(
                f"SECURITY: {env_file.name} exists but is not covered by "
                ".gitignore. Add it before running."
            )
    logger.info("Secret-safety checks passed: env files untracked and ignored.")


# --------------------------------------------------------------------------- #
# Subscription verification
# --------------------------------------------------------------------------- #

def verify_subscription(client: EodhdClient) -> dict[str, Any]:
    """Validate the token and record only non-personal account facts."""
    user = client.get("user")
    if not isinstance(user, dict):
        raise RuntimeError("Unexpected /user response shape; cannot verify "
                           "subscription.")
    safe = {
        "token_valid": True,
        "subscription_type": user.get("subscriptionType"),
        "daily_rate_limit": user.get("dailyRateLimit"),
        "api_requests_used_today": user.get("apiRequests"),
    }
    limit = safe["daily_rate_limit"]
    used = safe["api_requests_used_today"]
    if isinstance(limit, (int, float)) and isinstance(used, (int, float)):
        remaining = limit - used
        safe["requests_remaining_today"] = remaining
        if remaining < MIN_REMAINING_REQUESTS:
            raise RuntimeError(
                f"Only {remaining} EODHD requests remain today; at least "
                f"{MIN_REMAINING_REQUESTS} are needed for the repair."
            )
    # Probe the EOD historical endpoint with a cheap, well-known symbol.
    probe = client.get("eod/AAPL.US", **{"from": "2023-01-30", "to": "2023-02-03",
                                         "period": "d", "order": "a"})
    safe["eod_endpoint_accessible"] = bool(probe)
    safe["eod_probe_has_adjusted_close"] = bool(
        probe and isinstance(probe, list) and "adjusted_close" in probe[0]
    )
    if not safe["eod_probe_has_adjusted_close"]:
        raise RuntimeError("EOD endpoint probe did not return adjusted_close; "
                           "subscription may lack historical EOD access.")
    logger.info("Subscription verified: type=%s, daily limit=%s, used today=%s",
                safe["subscription_type"], limit, used)
    return safe


# --------------------------------------------------------------------------- #
# Symbol discovery
# --------------------------------------------------------------------------- #

def fetch_symbol_lists(client: EodhdClient) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (active, delisted) US exchange symbol lists as DataFrames."""
    active = pd.DataFrame(client.get("exchange-symbol-list/US"))
    delisted = pd.DataFrame(client.get("exchange-symbol-list/US", delisted=1))
    logger.info("US symbol lists: %d active, %d delisted rows",
                len(active), len(delisted))
    return active, delisted


def discover_candidates(
    client: EodhdClient,
    target: TargetSpec,
    active: pd.DataFrame,
    delisted: pd.DataFrame,
) -> list[Candidate]:
    """Find candidate identifiers via exchange lists and symbol search."""
    found: dict[str, Candidate] = {}

    def matches(code: str, name: str) -> bool:
        code_u, name_u = str(code).upper(), str(name).upper()
        code_hit = any(
            code_u == c or code_u.startswith(f"{c}-") or code_u.startswith(f"{c}_")
            for c in target.code_candidates
        )
        name_hit = any(k in name_u for k in target.company_keywords)
        return code_hit or name_hit

    for frame, status in ((active, "active list"), (delisted, "delisted list")):
        if frame.empty:
            continue
        for row in frame.itertuples(index=False):
            code = str(getattr(row, "Code", ""))
            name = str(getattr(row, "Name", ""))
            if matches(code, name):
                key = f"{code}.US"
                if key not in found:
                    found[key] = Candidate(code=code, exchange="US",
                                           name=name, status=status)

    for query in target.search_queries:
        try:
            results = client.get(f"search/{query}")
        except RuntimeError as exc:
            logger.warning("Search '%s' failed: %s", query, exc)
            continue
        for item in results or []:
            code = str(item.get("Code", ""))
            exchange = str(item.get("Exchange", ""))
            name = str(item.get("Name", ""))
            if exchange != "US" or not matches(code, name):
                continue
            key = f"{code}.US"
            if key not in found:
                found[key] = Candidate(code=code, exchange="US",
                                       name=name, status="search")

    # Only keep candidates whose NAME identifies the company, or whose code is
    # an exact expected ticker; a bare substring code hit is not evidence.
    kept = [
        c for c in found.values()
        if any(k in c.name.upper() for k in target.company_keywords)
        or c.code.upper() in {x.upper() for x in target.code_candidates}
    ]
    logger.info("%s: %d candidate identifier(s): %s",
                target.canonical, len(kept),
                ", ".join(f"{c.identifier} ({c.name!r}, {c.status})" for c in kept)
                or "none")
    return kept


# --------------------------------------------------------------------------- #
# Download and validation
# --------------------------------------------------------------------------- #

def download_candidate(client: EodhdClient, cand: Candidate) -> None:
    """Fetch the candidate's EOD history and note adjusted_close availability."""
    try:
        records = client.get(
            f"eod/{cand.identifier}",
            **{"from": FROM_DATE, "to": TO_DATE, "period": "d", "order": "a"},
        )
    except RuntimeError as exc:
        cand.download_error = str(exc)
        logger.warning("Download failed for %s: %s", cand.identifier, exc)
        return
    if not records:
        cand.download_error = "empty response"
        logger.warning("No rows returned for %s", cand.identifier)
        return
    cand.records = records
    cand.n_obs = len(records)
    cand.first_date = records[0].get("date")
    cand.last_date = records[-1].get("date")
    cand.has_adjusted_close = all("adjusted_close" in r for r in records)
    logger.info("%s: %d rows, %s to %s, adjusted_close=%s",
                cand.identifier, cand.n_obs, cand.first_date, cand.last_date,
                cand.has_adjusted_close)


def candidate_series(cand: Candidate) -> pd.Series:
    """Adjusted-close series for a downloaded candidate."""
    if not cand.records:
        return pd.Series(dtype=float)
    df = pd.DataFrame(cand.records)
    if "adjusted_close" not in df.columns:
        return pd.Series(dtype=float)
    s = pd.Series(
        pd.to_numeric(df["adjusted_close"], errors="coerce").values,
        index=pd.to_datetime(df["date"]),
        name=cand.identifier,
    ).dropna().sort_index()
    return s[~s.index.duplicated(keep="first")]


def load_official_series(symbol: str, cutoff: Optional[str]) -> pd.Series:
    """Official M6 price series for one symbol (validation only)."""
    df = pd.read_csv(OFFICIAL_M6_CSV)
    df = df[df["symbol"] == symbol].copy()
    if df.empty:
        raise RuntimeError(f"Official M6 file has no rows for {symbol}")
    s = pd.Series(
        pd.to_numeric(df["price"], errors="coerce").values,
        index=pd.to_datetime(df["date"], format="%Y/%m/%d"),
        name=f"{symbol}_official",
    ).dropna().sort_index()
    if cutoff:
        s = s.loc[s.index <= pd.Timestamp(cutoff)]
    return s


def validate_overlap(cand: Candidate, official: pd.Series) -> dict[str, Any]:
    """Compare candidate vs official returns over their common dates."""
    recovered = candidate_series(cand)
    common = recovered.index.intersection(official.index)
    stats: dict[str, Any] = {"overlap_obs": int(len(common))}
    if len(common) < 2:
        stats["error"] = "insufficient overlap with the official series"
        return stats

    rec = recovered.loc[common]
    off = official.loc[common]
    rec_ret = rec.pct_change().dropna()
    off_ret = off.pct_change().dropna()
    diff = (rec_ret - off_ret).abs()

    ratio = rec / off
    ratio_jumps = ratio.pct_change().abs()
    jump_dates = ratio_jumps[ratio_jumps > RATIO_JUMP_THRESHOLD].index

    stats.update({
        "first_overlap": common.min().date().isoformat(),
        "last_overlap": common.max().date().isoformat(),
        "return_correlation": round(float(rec_ret.corr(off_ret)), 6),
        "median_abs_return_diff": round(float(diff.median()), 6),
        "max_abs_return_diff": round(float(diff.max()), 6),
        "major_disagreement_dates": [
            d.date().isoformat()
            for d in diff[diff > MAJOR_DISAGREEMENT_THRESHOLD].index
        ],
        "ratio_jump_dates": [d.date().isoformat() for d in jump_dates],
    })
    return stats


def decide(cand: Candidate, target: TargetSpec) -> None:
    """Apply the acceptance criteria and record the reasoning."""
    reasons: list[str] = []
    if cand.download_error:
        cand.decision = f"rejected: download failed ({cand.download_error})"
        return
    if not any(k in cand.name.upper() for k in target.company_keywords):
        reasons.append(
            f"company name {cand.name!r} does not identify "
            f"{'/'.join(target.company_keywords)}"
        )
    if not cand.has_adjusted_close:
        reasons.append("adjusted_close field missing")
    if cand.first_date is None or cand.first_date > LATEST_ACCEPTABLE_FIRST_DATE:
        reasons.append(f"history starts too late ({cand.first_date})")
    if cand.last_date is None or cand.last_date < target.required_last_date:
        reasons.append(
            f"history ends too early ({cand.last_date} < "
            f"{target.required_last_date})"
        )
    v = cand.validation
    if "error" in v:
        reasons.append(v["error"])
    else:
        if v.get("overlap_obs", 0) < MIN_OVERLAP_OBS:
            reasons.append(f"only {v.get('overlap_obs')} overlapping observations")
        if v.get("return_correlation", 0) < MIN_RETURN_CORR:
            reasons.append(
                f"return correlation {v.get('return_correlation')} below "
                f"{MIN_RETURN_CORR}"
            )
        if v.get("median_abs_return_diff", 1) > MAX_MEDIAN_ABS_RETURN_DIFF:
            reasons.append(
                f"median absolute return difference "
                f"{v.get('median_abs_return_diff')} above "
                f"{MAX_MEDIAN_ABS_RETURN_DIFF}"
            )
    if reasons:
        cand.decision = "rejected: " + "; ".join(reasons)
    else:
        cand.accepted = True
        cand.decision = (
            "accepted: company identity, coverage and return agreement with "
            "the official M6 series all confirmed"
        )


def repair_target(
    client: EodhdClient,
    target: TargetSpec,
    active: pd.DataFrame,
    delisted: pd.DataFrame,
) -> tuple[list[Candidate], Optional[Candidate]]:
    """Discover, download, validate and select the candidate for one target."""
    official = load_official_series(target.canonical, target.official_cutoff)
    candidates = discover_candidates(client, target, active, delisted)
    for cand in candidates:
        download_candidate(client, cand)
        if cand.records:
            cand.validation = validate_overlap(cand, official)
        decide(cand, target)
        logger.info("%s -> %s: %s", target.canonical, cand.identifier,
                    cand.decision)

    accepted = [c for c in candidates if c.accepted]
    if not accepted:
        return candidates, None
    # Prefer the identifier matching the canonical M6 ticker, then best
    # return correlation.
    accepted.sort(
        key=lambda c: (
            c.code.upper() != target.canonical.upper(),
            -(c.validation.get("return_correlation") or 0),
        )
    )
    chosen = accepted[0]
    for other in accepted[1:]:
        other.accepted = False
        other.decision += (
            " (passed validation but a better-matching identifier "
            f"{chosen.identifier} was preferred)"
        )
    return candidates, chosen


# --------------------------------------------------------------------------- #
# Outputs
# --------------------------------------------------------------------------- #

def save_raw_records(client: EodhdClient, cand: Candidate,
                     canonical: str) -> Path:
    """Save the unchanged accepted API records, after a token-leak check."""
    RAW_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_SAVE_DIR / f"{canonical.lower()}_eodhd_raw.json"
    payload = {
        "canonical_m6_symbol": canonical,
        "eodhd_identifier": cand.identifier,
        "company_name": cand.name,
        "endpoint": f"{API_BASE}/eod/{cand.identifier}",
        "parameters": {"from": FROM_DATE, "to": TO_DATE, "period": "d",
                       "order": "a", "fmt": "json"},
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "records": cand.records,
    }
    text = json.dumps(payload, indent=2)
    if client.token_in(text) or client.token_in(str(path)):
        raise RuntimeError("SECURITY: API token detected in content about to "
                           "be saved; aborting write.")
    path.write_text(text, encoding="utf-8")
    logger.info("Saved raw EODHD records for %s -> %s", canonical, path)
    return path


def build_repaired_dataset(
    chosen: dict[str, Candidate],
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Insert recovered series into a copy of the original Dataset A.

    The original CSV is loaded as strings so the 97 untouched Yahoo columns
    are preserved byte-for-byte (re-parsing floats can perturb the last bit).
    Only the repaired columns are replaced.
    """
    original = pd.read_csv(ORIGINAL_DATASET, dtype=str)
    dates = pd.to_datetime(original["date"])
    assert not dates.duplicated().any()
    assert dates.is_monotonic_increasing
    assert dates.max() <= pd.Timestamp(TO_DATE)

    dropped: dict[str, list[str]] = {}
    for symbol, cand in chosen.items():
        series = candidate_series(cand)
        outside = series.index.difference(pd.DatetimeIndex(dates))
        if len(outside) > 0:
            dropped[symbol] = [d.date().isoformat() for d in outside]
            logger.warning(
                "%s: %d recovered date(s) not in the original Dataset A "
                "calendar were left out to keep the 97 Yahoo series unchanged: "
                "%s", symbol, len(outside), dropped[symbol][:10])
        aligned = dates.map(series)  # NaN where the date has no recovered value
        original[symbol] = aligned.map(
            lambda v: "" if pd.isna(v) else repr(float(v))
        )

    return original.set_index(dates.rename("date_index")), dropped


def write_repaired(repaired: pd.DataFrame) -> None:
    REPAIRED_DATASET.parent.mkdir(parents=True, exist_ok=True)
    repaired.to_csv(REPAIRED_DATASET, index=False)
    logger.info("Wrote repaired Dataset A: %s (%d rows x %d columns)",
                REPAIRED_DATASET, repaired.shape[0], repaired.shape[1])


def format_validation(v: dict[str, Any]) -> str:
    if not v:
        return "not validated (no data)"
    if "error" in v:
        return v["error"]
    major = ", ".join(v["major_disagreement_dates"]) or "none"
    jumps = ", ".join(v["ratio_jump_dates"]) or "none"
    return (
        f"{v['overlap_obs']} overlapping observations "
        f"({v['first_overlap']} to {v['last_overlap']}); "
        f"return correlation {v['return_correlation']}; "
        f"median |Δreturn| {v['median_abs_return_diff']}; "
        f"max |Δreturn| {v['max_abs_return_diff']}; "
        f"major disagreement dates: {major}; "
        f"price-ratio jump dates: {jumps}"
    )


def write_report(
    subscription: dict[str, Any],
    results: dict[str, tuple[list[Candidate], Optional[Candidate]]],
    repaired: Optional[pd.DataFrame],
    dropped: dict[str, list[str]],
) -> None:
    all_resolved = all(chosen is not None for _, chosen in results.values())
    dre_chosen = results["DRE"][1]
    dre_last = dre_chosen.last_date if dre_chosen else "unresolved"

    lines = [
        "# Dataset A Repair Report (EODHD)",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "",
        "## EODHD subscription verification",
        "",
        f"- Token validation: {'SUCCEEDED' if subscription.get('token_valid') else 'FAILED'}",
        f"- Subscription type: {subscription.get('subscription_type')}",
        f"- Daily request limit: {subscription.get('daily_rate_limit')} "
        f"(used today at time of run: {subscription.get('api_requests_used_today')})",
        f"- Historical EOD endpoint accessible with `adjusted_close`: "
        f"{'YES' if subscription.get('eod_probe_has_adjusted_close') else 'NO'}",
        "- No personal account fields (name, email, payment details) were "
        "saved, and the API token appears nowhere in this repository.",
        "",
        "## Requested download window",
        "",
        f"- from = {FROM_DATE}, to = {TO_DATE} (inclusive), period = d, "
        "order = a, fmt = json, field = `adjusted_close` (never `close`, "
        "`open`, an acquirer's series, or official M6 values).",
        "",
    ]

    for symbol, (candidates, chosen) in results.items():
        lines += [f"## {symbol}", ""]
        if not candidates:
            lines += ["- No candidate identifiers were discovered.", ""]
        for cand in candidates:
            v = format_validation(cand.validation)
            lines += [
                f"### `{cand.identifier}` — {cand.name}",
                "",
                f"- Source: {cand.status}",
                f"- Returned range: {cand.first_date} to {cand.last_date} "
                f"({cand.n_obs} rows); adjusted_close available: "
                f"{'yes' if cand.has_adjusted_close else 'no'}",
                f"- Overlap validation: {v}",
                f"- Decision: {cand.decision}",
                "",
            ]
        if chosen:
            lines += [
                f"**Accepted identifier for {symbol}: `{chosen.identifier}`** "
                f"({chosen.name}); stored under canonical Dataset A column "
                f"`{symbol}`.",
                "",
            ]
        else:
            lines += [f"**{symbol} remains UNRESOLVED.**", ""]

    lines += [
        "## Special-handling confirmations",
        "",
        f"- DRE's final genuine available date in the recovered data: "
        f"{dre_last}. Later dates are left missing; the price was not set to "
        "zero and was NOT carried forward (preprocessing does that, matching "
        "the competition's zero-return treatment).",
        "- PLD (the acquirer) was NOT used for DRE.",
        "- SW (Smurfit WestRock) was NOT automatically substituted for WRK; "
        "only the original WestRock Company security was acceptable.",
        "- RE remains the canonical Dataset A column; no separate EG column "
        "was created. The provider identifier actually used is documented "
        "above.",
        "- CARR and OGN were not touched: their shorter histories are genuine "
        "(2020 and 2021 spin-offs), and later model inputs must use the "
        "available history rather than fabricating exactly 512 observations.",
        "- The official M6 file was used ONLY for overlap validation, never "
        "to construct pre-2022 history or fill values.",
        "- The original `dataset_a_adjusted_close.csv` was not modified; the "
        "97 successful Yahoo series were copied into the repaired file "
        "unchanged.",
        "- The EODHD API token was loaded from the local env file and was not "
        "printed, logged, saved, or embedded in any output.",
        "",
        "## Repaired dataset",
        "",
    ]
    if repaired is not None:
        lines += [
            f"- File: `Data/raw/yahoo/dataset_a_adjusted_close_repaired.csv`",
            f"- Shape: {repaired.shape[0]} rows x {repaired.shape[1]} "
            "columns (`date` + 100 official assets, official order)",
            f"- Dates: {repaired.index.min().date()} to "
            f"{repaired.index.max().date()}, ascending, no duplicates, none "
            f"after {TO_DATE}.",
        ]
        if dropped:
            for sym, dates in dropped.items():
                lines.append(
                    f"- {sym}: {len(dates)} recovered date(s) outside the "
                    f"original calendar were excluded: {', '.join(dates)}"
                )
        else:
            lines.append("- All recovered observations fell on dates already "
                         "present in the original Dataset A calendar.")
    else:
        lines.append("- Not written: one or more assets unresolved.")

    lines += [
        "",
        "## Verdict",
        "",
        f"- All three assets resolved: {'YES' if all_resolved else 'NO'}",
        "- Dataset A ready for preprocessing: "
        + ("YES" if all_resolved else
           "NO: do not begin preprocessing until the unresolved assets above are "
           "repaired"),
        "",
    ]
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote repair report: %s", REPORT_MD)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if not ORIGINAL_DATASET.is_file():
        logger.error("Original Dataset A not found at %s; run "
                     "scripts/pipeline/download_dataset_a.py first.", ORIGINAL_DATASET)
        return 1
    if not OFFICIAL_M6_CSV.is_file():
        logger.error("Official M6 validation file not found at %s.",
                     OFFICIAL_M6_CSV)
        return 1

    try:
        verify_git_secret_safety()
        token = load_token()
        client = EodhdClient(token)
        subscription = verify_subscription(client)
    except RuntimeError as exc:
        logger.error("%s", exc)
        return 1

    active, delisted = fetch_symbol_lists(client)

    results: dict[str, tuple[list[Candidate], Optional[Candidate]]] = {}
    chosen: dict[str, Candidate] = {}
    for target in TARGETS:
        candidates, best = repair_target(client, target, active, delisted)
        results[target.canonical] = (candidates, best)
        if best:
            chosen[target.canonical] = best
            save_raw_records(client, best, target.canonical)
        else:
            logger.warning("%s remains unresolved.", target.canonical)

    repaired: Optional[pd.DataFrame] = None
    dropped: dict[str, list[str]] = {}
    if len(chosen) == len(TARGETS):
        repaired, dropped = build_repaired_dataset(chosen)
        write_repaired(repaired)
    else:
        logger.warning("Repaired dataset NOT written: unresolved assets: %s",
                       ", ".join(t.canonical for t in TARGETS
                                 if t.canonical not in chosen))

    write_report(subscription, results, repaired, dropped)
    return 0 if repaired is not None else 1


if __name__ == "__main__":
    sys.exit(main())
