"""Shared EODHD client used by the data-acquisition scripts.

Provides the HTTP client, token loading, secret-safety and subscription checks,
ticker identifier resolution, the official 100-asset order and the comparison
thresholds used when a download is validated against Dataset A.

Imported by scripts/pipeline/download_dataset_d_eodhd.py and by
FinQ Boost Reimplementation/download_extended_eodhd.py. Not run directly.

The EODHD token is read from the local env file and is never printed, logged or
written to any recorded URL.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]

logger = logging.getLogger("eodhd_client")


ENV_FILES = (PROJECT_ROOT / "proj.env", PROJECT_ROOT / ".env")
TOKEN_ENV_NAMES = ("EODHD_API_TOKEN", "EODHD_API_KEY")
OFFICIAL_ORDER_SOURCE = (PROJECT_ROOT / "Data" / "raw" / "m6_official"
                         / "assets_m6.csv")
DATASET_A = PROJECT_ROOT / "Data" / "raw" / "yahoo" / "dataset_a_adjusted_close_repaired.csv"
API_BASE = "https://eodhd.com/api"
CHECK_FROM_DATE = "2012-03-05"   # probe window for the subscription check only
CHECK_TO_DATE = "2022-03-04"     # EODHD `to` is inclusive
MIN_REMAINING_REQUESTS = 120    # 100 assets + verification headroom
NETWORK_RETRIES = 4
RETRY_BACKOFF_SECONDS = 15
MIN_OVERLAP_OBS = 60
MIN_RETURN_CORR = 0.98
MAX_MEDIAN_ABS_RETURN_DIFF = 0.003
VALIDATED_IDENTIFIERS = {
    "DRE": "DRE.US",
    "RE": "EG.US",
    "WRK": "WRK.US",
}


def identifier_candidates(symbol: str) -> list[str]:
    """EODHD identifiers to try for a canonical M6 ticker, best first."""
    if symbol in VALIDATED_IDENTIFIERS:
        primary = VALIDATED_IDENTIFIERS[symbol]
        return [primary] + [c for c in (f"{symbol}.US",) if c != primary]
    if symbol.endswith(".L"):                 # London-listed ETFs
        stem = symbol[:-2]
        return [f"{stem}.LSE", f"{stem}.L", f"{stem}.XETRA"]
    return [f"{symbol}.US"]


class EodhdClient:
    """Thin EODHD client that keeps the token out of logs and errors."""

    def __init__(self, token: str) -> None:
        self._token = token
        self._session = requests.Session()

    def _redact(self, text: str) -> str:
        return text.replace(self._token, "***REDACTED***")

    def get(self, path: str, **params: Any) -> Any:
        params = {**params, "api_token": self._token, "fmt": "json"}
        url = f"{API_BASE}/{path}"
        logger.info("GET /%s", path)
        # Transient network errors are retried a few times; a persistent failure
        # still raises so the caller can stop safely with progress preserved.
        last_error: Optional[str] = None
        for attempt in range(1, NETWORK_RETRIES + 1):
            try:
                resp = self._session.get(url, params=params, timeout=60)
                break
            except requests.RequestException as exc:
                last_error = self._redact(str(exc))
                if attempt < NETWORK_RETRIES:
                    wait = RETRY_BACKOFF_SECONDS * attempt
                    logger.info("  network error on /%s (attempt %d/%d); "
                                "retrying in %ds", path, attempt, NETWORK_RETRIES, wait)
                    time.sleep(wait)
        else:
            raise RuntimeError(
                f"Request to /{path} failed after {NETWORK_RETRIES} attempts: "
                f"{last_error}"
            ) from None
        if resp.status_code == 404:
            return None
        if resp.status_code == 429:
            raise RuntimeError(
                f"EODHD rate limit hit on /{path}; stopping so completed "
                "downloads are preserved. Re-run later to resume."
            )
        if resp.status_code != 200:
            raise RuntimeError(
                f"EODHD returned HTTP {resp.status_code} for /{path}: "
                f"{self._redact(resp.text[:300])}"
            )
        try:
            return resp.json()
        except ValueError:
            raise RuntimeError(
                f"EODHD returned non-JSON for /{path}: "
                f"{self._redact(resp.text[:200])}"
            ) from None


def load_token() -> str:
    for env_file in ENV_FILES:
        if env_file.is_file():
            load_dotenv(env_file, override=False)
    for name in TOKEN_ENV_NAMES:
        token = os.getenv(name)
        if token and token.strip():
            logger.info("Loaded EODHD token from %s (value not shown).", name)
            return token.strip()
    raise RuntimeError(
        "EODHD API token not found. Define EODHD_API_TOKEN (or EODHD_API_KEY) "
        f"in one of: {', '.join(str(p) for p in ENV_FILES)}. "
        "The token value is never printed."
    )


def verify_git_secret_safety() -> None:
    """Fail if an env file is tracked by git; require .gitignore coverage."""
    import subprocess

    tracked = subprocess.run(["git", "ls-files"], cwd=PROJECT_ROOT,
                             capture_output=True, text=True).stdout.splitlines()
    for env_file in ENV_FILES:
        if env_file.name in tracked:
            raise RuntimeError(
                f"SECURITY: {env_file.name} is tracked by git. Remove it from "
                "the index and rotate the token before continuing."
            )
    gitignore = PROJECT_ROOT / ".gitignore"
    patterns = (gitignore.read_text(encoding="utf-8").splitlines()
                if gitignore.is_file() else [])
    for env_file in ENV_FILES:
        if env_file.is_file() and not any(
            p.strip() in (env_file.name, "*.env", ".env") for p in patterns
        ):
            raise RuntimeError(
                f"SECURITY: {env_file.name} exists but is not covered by "
                ".gitignore. Add it before running."
            )
    logger.info("Secret-safety checks passed: env files untracked and ignored.")


def verify_subscription(client: EodhdClient, from_date: str = CHECK_FROM_DATE,
                        probe_to: Optional[str] = None) -> dict[str, Any]:
    """Confirm the token works and the historical EOD endpoint is available.

    ``from_date``/``probe_to`` default to this dataset's own range; a companion
    acquisition (Dataset D) passes its earlier start date so the probe checks
    that history actually reaches that far back.
    """
    probe_to = probe_to or (pd.Timestamp(from_date) + pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    user = client.get("user")
    if not isinstance(user, dict):
        raise RuntimeError("Unexpected /user response; cannot verify subscription.")
    safe = {
        "token_valid": True,
        "subscription_type": user.get("subscriptionType"),
        "daily_rate_limit": user.get("dailyRateLimit"),
        "api_requests_used_today": user.get("apiRequests"),
    }
    limit, used = safe["daily_rate_limit"], safe["api_requests_used_today"]
    if isinstance(limit, (int, float)) and isinstance(used, (int, float)):
        remaining = limit - used
        safe["requests_remaining_today"] = remaining
        if remaining < MIN_REMAINING_REQUESTS:
            raise RuntimeError(
                f"Only {remaining} EODHD requests remain today; at least "
                f"{MIN_REMAINING_REQUESTS} are needed for the 100 assets. "
                "Re-run when the quota resets; completed downloads are kept."
            )

    # Probe the deep-history endpoint at the actual start of the requested range.
    probe = client.get("eod/AAPL.US", **{"from": from_date, "to": probe_to,
                                         "period": "d", "order": "a"})
    if not probe or not isinstance(probe, list):
        raise RuntimeError(
            "The EOD endpoint returned no data for a known symbol at "
            f"{from_date}; the subscription may not include history that far "
            "back. Nothing was downloaded."
        )
    if "adjusted_close" not in probe[0]:
        raise RuntimeError("EOD probe did not return adjusted_close; the "
                           "subscription may lack historical EOD access.")
    safe["eod_endpoint_accessible"] = True
    safe["eod_history_reaches_from_date"] = True
    logger.info("Subscription verified: type=%s, daily limit=%s, used today=%s",
                safe["subscription_type"], limit, used)
    return safe


def official_asset_order() -> list[str]:
    """The 100 official M6 symbols in their released order."""
    symbols = pd.read_csv(OFFICIAL_ORDER_SOURCE)["symbol"]
    return list(dict.fromkeys(symbols.astype(str).str.strip()))
