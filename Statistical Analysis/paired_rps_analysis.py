"""Paired round-level statistical analysis of M6 RPS across the 12 official rounds.

Loads the authoritative per-round RPS series identified in the existing-results
audit, then for each predefined comparison computes the mean paired difference,
a 95% BCa bootstrap interval, an exact two-sided sign-flip permutation p-value
and a Holm-adjusted p-value within its family.

The 12 rounds are the statistical units. Assets within a round are
cross-sectionally dependent, so the 1,200 asset-rounds are not independent
observations and are deliberately not used here.

Sign convention throughout: difference = candidate - reference, so a NEGATIVE
difference favours the candidate.

Run from the repository root:
    python "Statistical Analysis/paired_rps_analysis.py"
"""
from __future__ import annotations

import hashlib
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "results"

N_ROUNDS = 12
N_BOOTSTRAP = 100_000
SEED = 42
ALPHA = 0.05
TIE_TOL = 1e-12
NAIVE_RPS = 0.160000

# --------------------------------------------------------------------------
# Input specification: every configuration is loaded by explicit path and
# filter, never by glob, because several superseded variants differ from the
# authoritative result only in the fourth decimal place.
# --------------------------------------------------------------------------
HORIZON = "Horizon Experiment/results/horizon_rps_by_round.csv"
FEATURE = "Feature Baselines/results/feature_baseline_m6_rps.csv"
SOFTMAX = "Feature Baselines/results/softmax_m6_rps.csv"
FINQBOOST = "FinQ Boost Reimplementation/finqboost_m6_rps.csv"
QUASI = ("Quasi-Average Reimplementation DRE-as-Stock/results/"
         "quasi_average_m6_rps.csv")
UNC = "TSFM Uncertainty Experiment/results"

# key -> (display label, source path, filter description, expected overall RPS)
CONFIGS = {
    "t5_zero":   ("Chronos-T5 Base zero-shot", HORIZON,
                  "horizon_days==20 & n_scenarios==500 & "
                  "model=='Chronos-T5 Base 200M'", 0.226920),
    "fc_zero":   ("Financial Chronos zero-shot", HORIZON,
                  "horizon_days==20 & n_scenarios==500 & "
                  "model=='Financial Chronos 46M'", 0.177947),
    "c2_zero":   ("Chronos-2 zero-shot", HORIZON,
                  "horizon_days==20 & n_scenarios==500 & "
                  "model=='Chronos-2 120M (independent sampling)'", 0.165927),
    "naive":     ("Naive", "(constant 0.2 forecast)",
                  "0.160000 in every round by construction", 0.160000),
    "rf":        ("Random Forest", FEATURE, "model=='rf' & variant=='no_knn'",
                  0.157053),
    "lgbm":      ("LightGBM", FEATURE, "model=='lgbm' & variant=='no_knn'",
                  0.157432),
    "softmax":   ("Softmax Regression", SOFTMAX, "variant=='knn'", 0.157355),
    "finqboost": ("FinQBoost", FINQBOOST,
                  "seed==29 & panel_start=='2009-01-02'", 0.156703),
    "quasi":     ("Quasi-Average", QUASI, "single row, DRE-as-Stock", 0.157234),
    "fc_samp":   ("Financial Chronos sampling", f"{UNC}/financial_chronos/m6/"
                  "m6_rps_by_round.csv", "all rows", 0.158526),
    "t5_samp":   ("Chronos-T5 Base sampling", f"{UNC}/chronos_t5_base/m6/"
                  "m6_rps_by_round.csv", "all rows", 0.160949),
    "c2_lora":   ("Chronos-2 ordinary LoRA", f"{UNC}/chronos2_lora/m6/"
                  "m6_rps_by_round.csv", "all rows", 0.162997),
}
for _b, _exp in [("0.5", 0.160772), ("1.0", 0.160640), ("2.0", 0.160503),
                 ("4.0", 0.160510), ("8.0", 0.160532), ("16.0", 0.160611)]:
    CONFIGS[f"beta{_b}"] = (
        f"Uncertainty LoRA beta {float(_b):g}",
        f"{UNC}/chronos2_uncertainty_lora/beta{_b}/m6/m6_rps_by_round.csv",
        "all rows", _exp)

# --------------------------------------------------------------------------
# Comparison families. Holm correction is applied WITHIN each family.
# --------------------------------------------------------------------------
FAMILIES = {
    "F1 Core zero-shot": [
        ("t5_zero", "naive"), ("fc_zero", "naive"), ("c2_zero", "naive"),
        ("c2_zero", "t5_zero"), ("fc_zero", "t5_zero"),
    ],
    "F2 Benchmarks vs naive": [
        ("rf", "naive"), ("lgbm", "naive"), ("softmax", "naive"),
        ("finqboost", "naive"), ("quasi", "naive"),
    ],
    "F3 Sampling interventions": [
        ("fc_samp", "fc_zero"), ("t5_samp", "t5_zero"),
        ("fc_samp", "naive"), ("t5_samp", "naive"),
    ],
    "F4 Chronos-2 adaptation": [
        ("c2_lora", "c2_zero"),
        ("beta0.5", "c2_lora"), ("beta1.0", "c2_lora"), ("beta2.0", "c2_lora"),
        ("beta4.0", "c2_lora"), ("beta8.0", "c2_lora"), ("beta16.0", "c2_lora"),
    ],
}


# ==========================================================================
# Statistics
# ==========================================================================
def exact_sign_flip_p(diff: np.ndarray) -> float:
    """Exact two-sided paired sign-flip permutation p-value.

    Enumerates all 2**n sign assignments, so the p-value is exact and no
    Monte Carlo plus-one correction is applied. Under the null of symmetry
    about zero, each sign assignment is equally likely.
    """
    n = len(diff)
    signs = np.array(list(product([1.0, -1.0], repeat=n)))   # (2**n, n)
    means = signs @ diff / n
    observed = abs(float(np.mean(diff)))
    return float(np.mean(np.abs(means) >= observed - TIE_TOL))


def bca_interval(diff: np.ndarray, n_boot: int, seed: int,
                 alpha: float = ALPHA) -> tuple[float, float]:
    """95% BCa bootstrap interval for the mean paired difference.

    Rounds are resampled as whole pairs: one index draw selects the same round
    for both candidate and reference, so pairing is preserved by construction.
    """
    from scipy.stats import norm

    n = len(diff)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))      # resample ROUND INDICES
    boot = diff[idx].mean(axis=1)
    theta = float(np.mean(diff))

    # bias correction from the proportion of bootstrap means below the estimate
    prop = float(np.mean(boot < theta))
    if prop <= 0 or prop >= 1:                       # degenerate, fall back
        return float(np.percentile(boot, 100 * alpha / 2)), \
               float(np.percentile(boot, 100 * (1 - alpha / 2)))
    z0 = norm.ppf(prop)

    # acceleration from the jackknife over rounds
    jack = np.array([np.mean(np.delete(diff, i)) for i in range(n)])
    jm = jack.mean()
    num = float(np.sum((jm - jack) ** 3))
    den = 6.0 * float(np.sum((jm - jack) ** 2)) ** 1.5
    a = num / den if den != 0 else 0.0

    zl, zu = norm.ppf(alpha / 2), norm.ppf(1 - alpha / 2)
    lo_adj = z0 + (z0 + zl) / (1 - a * (z0 + zl))
    hi_adj = z0 + (z0 + zu) / (1 - a * (z0 + zu))
    lo_pct = 100 * float(norm.cdf(lo_adj))
    hi_pct = 100 * float(norm.cdf(hi_adj))
    return float(np.percentile(boot, lo_pct)), float(np.percentile(boot, hi_pct))


def holm(pvals: list[float], alpha: float = ALPHA) -> tuple[list[float], list[bool]]:
    """Holm-Bonferroni step-down adjustment within one family."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * pvals[i]
        running = max(running, val)              # enforce monotonicity
        adj[i] = min(1.0, running)
    return adj, [adj[i] <= alpha for i in range(m)]


# ==========================================================================
# Loading
# ==========================================================================
def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_round_series() -> tuple[pd.DataFrame, list[dict]]:
    """Return a 12-row wide frame of per-round RPS, plus the input manifest."""
    series: dict[str, np.ndarray] = {}
    manifest: list[dict] = []
    seen_files: dict[str, str] = {}

    def note(path_rel: str) -> str:
        if path_rel not in seen_files:
            seen_files[path_rel] = sha256(ROOT / path_rel)
        return seen_files[path_rel]

    # long format, one row per round
    hz = pd.read_csv(ROOT / HORIZON)
    hz20 = hz[(hz.horizon_days == 20) & (hz.n_scenarios == 500)]
    for key, model in [("t5_zero", "Chronos-T5 Base 200M"),
                       ("fc_zero", "Financial Chronos 46M"),
                       ("c2_zero", "Chronos-2 120M (independent sampling)")]:
        g = hz20[hz20.model == model].sort_values("round")
        assert len(g) == N_ROUNDS, f"{key}: {len(g)} rows"
        series[key] = g.mean_rps.to_numpy(float)

    # wide format, round_01_rps .. round_12_rps on a single row
    cols = [f"round_{i:02d}_rps" for i in range(1, 13)]
    fb = pd.read_csv(ROOT / FEATURE)
    for key, model in [("rf", "rf"), ("lgbm", "lgbm")]:
        r = fb[(fb.model == model) & (fb.variant == "no_knn")]
        assert len(r) == 1, f"{key}: {len(r)} rows"
        series[key] = r.iloc[0][cols].to_numpy(float)

    sm = pd.read_csv(ROOT / SOFTMAX)
    r = sm[sm.variant == "knn"]
    assert len(r) == 1
    series["softmax"] = r.iloc[0][cols].to_numpy(float)

    fq = pd.read_csv(ROOT / FINQBOOST)
    r = fq[(fq.seed == 29) & (fq.panel_start == "2009-01-02")]
    assert len(r) == 1
    series["finqboost"] = r.iloc[0][cols].to_numpy(float)

    qa = pd.read_csv(ROOT / QUASI)
    assert len(qa) == 1
    series["quasi"] = qa.iloc[0][cols].to_numpy(float)

    # long format interventions and betas
    for key in list(CONFIGS):
        if key in series or key == "naive":
            continue
        rel = CONFIGS[key][1]
        d = pd.read_csv(ROOT / rel).sort_values("round")
        assert len(d) == N_ROUNDS, f"{key}: {len(d)} rows"
        assert d["round"].tolist() == list(range(1, 13)), f"{key}: round ids"
        series[key] = d.mean_rps.to_numpy(float)

    series["naive"] = np.full(N_ROUNDS, NAIVE_RPS)

    for key, (label, rel, filt, exp) in CONFIGS.items():
        digest = "n/a (constant)" if key == "naive" else note(rel)
        manifest.append(dict(config_key=key, label=label, source_path=rel,
                             filter=filt, sha256=digest,
                             expected_overall_rps=exp,
                             calculated_mean=float(series[key].mean())))

    frame = pd.DataFrame(series)
    frame.insert(0, "round", range(1, N_ROUNDS + 1))
    return frame, manifest


# ==========================================================================
# Validation
# ==========================================================================
def validate(frame: pd.DataFrame, manifest: list[dict], log: list[str]) -> None:
    def check(ok: bool, msg: str) -> None:
        log.append(f"  [{'PASS' if ok else 'FAIL'}] {msg}")
        if not ok:
            raise SystemExit(f"VALIDATION FAILED: {msg}")

    log.append("INPUT AND ALIGNMENT CHECKS")
    check(frame["round"].tolist() == list(range(1, 13)),
          "exactly 12 unique rounds, ordered 1..12")
    check(frame["round"].is_unique, "round identifiers unique")
    check(len(frame) == N_ROUNDS, f"frame has {N_ROUNDS} rows")

    values = frame.drop(columns="round")
    check(not values.isna().any().any(), "no missing values in any series")
    check(np.isfinite(values.to_numpy(float)).all(), "all values finite")
    check(bool((values.to_numpy(float) > 0).all()), "all RPS values positive")
    check(len(values.columns) == len(CONFIGS),
          f"all {len(CONFIGS)} configurations loaded")

    check(bool(np.all(frame["naive"].to_numpy() == NAIVE_RPS)),
          "naive RPS is exactly 0.160000 in every round")

    log.append("OVERALL MEAN REPRODUCTION (tolerance 1e-6)")
    for row in manifest:
        diff = abs(row["calculated_mean"] - row["expected_overall_rps"])
        check(diff < 1e-6,
              f"{row['label']}: mean {row['calculated_mean']:.8f} vs expected "
              f"{row['expected_overall_rps']:.6f} (diff {diff:.2e})")


def self_tests(log: list[str]) -> None:
    """Unit checks of the statistical machinery against known answers."""
    def check(ok: bool, msg: str) -> None:
        log.append(f"  [{'PASS' if ok else 'FAIL'}] {msg}")
        if not ok:
            raise SystemExit(f"SELF-TEST FAILED: {msg}")

    log.append("EXACT SIGN-FLIP PERMUTATION TEST")
    # all differences identical and non-zero: only the all-same-sign
    # assignments reach the observed |mean|, so p = 2/2**n
    d = np.full(12, -0.001)
    check(abs(exact_sign_flip_p(d) - 2 / 4096) < 1e-15,
          f"12 identical non-zero differences give p = 2/4096 "
          f"(got {exact_sign_flip_p(d):.8f})")
    # all differences zero: every assignment ties the observed mean, p = 1
    check(exact_sign_flip_p(np.zeros(12)) == 1.0,
          "all-zero differences give p = 1")
    # perfectly symmetric differences: observed mean is 0, p = 1
    check(exact_sign_flip_p(np.array([1.0, -1.0, 2.0, -2.0])) == 1.0,
          "symmetric differences give p = 1")
    # enumeration size
    check(len(list(product([1, -1], repeat=12))) == 4096,
          "enumeration covers all 4,096 sign assignments")
    # small hand-checkable case: n=3, d=[1,1,1] -> p = 2/8
    check(abs(exact_sign_flip_p(np.ones(3)) - 0.25) < 1e-15,
          "n=3 all-ones gives p = 2/8 = 0.25")

    log.append("HOLM CORRECTION")
    adj, rej = holm([0.01, 0.04, 0.03])
    # sorted 0.01,0.03,0.04 -> 3*0.01=0.03, 2*0.03=0.06, 1*0.04=0.06 (monotone)
    check(np.allclose(adj, [0.03, 0.06, 0.06]),
          f"known case [0.01,0.04,0.03] -> [0.03,0.06,0.06] (got {adj})")
    check(rej == [True, False, False], "rejection at alpha 0.05 matches")
    adj2, _ = holm([0.001, 0.002])
    check(np.allclose(adj2, [0.002, 0.002]),
          "monotonicity enforced: [0.001,0.002] -> [0.002,0.002]")
    adj3, _ = holm([0.5])
    check(np.allclose(adj3, [0.5]), "single-comparison family is unchanged")
    check(all(a <= 1.0 for a in holm([0.6, 0.7, 0.8])[0]),
          "adjusted p-values are capped at 1.0")

    log.append("BOOTSTRAP STRUCTURE")
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, N_ROUNDS, size=(1000, N_ROUNDS))
    check(idx.shape == (1000, N_ROUNDS),
          "each resample draws 12 round indices")
    check(int(idx.min()) >= 0 and int(idx.max()) <= N_ROUNDS - 1,
          "resampled indices stay within the 12 rounds")
    # pairing: resampling the DIFFERENCE vector is identical to resampling
    # candidate and reference with the same index draw
    cand = np.arange(12, dtype=float)
    ref = np.arange(12, dtype=float) * 0.5
    take = rng.integers(0, 12, size=12)
    check(np.allclose((cand - ref)[take], cand[take] - ref[take]),
          "bootstrap preserves round pairing (candidate and reference "
          "resampled together)")
    # determinism of the interval for a fixed seed
    d = np.linspace(-0.01, 0.002, 12)
    check(bca_interval(d, 5000, SEED) == bca_interval(d, 5000, SEED),
          "BCa interval is deterministic for a fixed seed")
    lo, hi = bca_interval(d, 5000, SEED)
    check(lo < float(np.mean(d)) < hi,
          "BCa interval brackets the observed mean difference")


# ==========================================================================
# Main
# ==========================================================================
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    log: list[str] = ["PAIRED ROUND-LEVEL RPS ANALYSIS - VERIFICATION REPORT", ""]

    frame, manifest = load_round_series()
    validate(frame, manifest, log)
    log.append("")
    self_tests(log)
    log.append("")

    rows = []
    for family, pairs in FAMILIES.items():
        fam_rows = []
        for cand, ref in pairs:
            c = frame[cand].to_numpy(float)
            r = frame[ref].to_numpy(float)
            d = c - r                                   # candidate - reference
            lo, hi = bca_interval(d, N_BOOTSTRAP, SEED)
            p = exact_sign_flip_p(d)
            fam_rows.append(dict(
                family=family,
                candidate=CONFIGS[cand][0], reference=CONFIGS[ref][0],
                comparison=f"{CONFIGS[cand][0]} vs {CONFIGS[ref][0]}",
                candidate_mean_rps=float(c.mean()),
                reference_mean_rps=float(r.mean()),
                mean_paired_difference=float(d.mean()),
                ci95_low=lo, ci95_high=hi,
                rounds_improved=int(np.sum(d < -TIE_TOL)),
                rounds_worsened=int(np.sum(d > TIE_TOL)),
                rounds_tied=int(np.sum(np.abs(d) <= TIE_TOL)),
                p_raw=p,
                sign_convention="negative difference favours the candidate",
            ))
        adj, rej = holm([r_["p_raw"] for r_ in fam_rows])
        for r_, a, k in zip(fam_rows, adj, rej):
            r_["p_holm"] = a
            r_["reject_at_0.05"] = bool(k)
            r_["family_size"] = len(fam_rows)
        rows.extend(fam_rows)

    comparisons = pd.DataFrame(rows)
    assert len(comparisons) == 21, f"expected 21 comparisons, got {len(comparisons)}"

    summary = pd.DataFrame([
        dict(config_key=k, model=CONFIGS[k][0],
             mean_rps=float(frame[k].mean()),
             sd_across_rounds=float(frame[k].std(ddof=1)),
             min_round_rps=float(frame[k].min()),
             max_round_rps=float(frame[k].max()),
             rounds_below_naive=int(np.sum(frame[k].to_numpy() < NAIVE_RPS - TIE_TOL)))
        for k in CONFIGS
    ]).sort_values("mean_rps").reset_index(drop=True)

    config = dict(
        analysis="paired round-level RPS comparison on the 12 official M6 rounds",
        statistical_unit="M6 round (n=12); asset-rounds are NOT independent",
        sign_convention="difference = candidate - reference; negative favours candidate",
        n_rounds=N_ROUNDS, n_bootstrap=N_BOOTSTRAP, seed=SEED, alpha=ALPHA,
        tie_tolerance=TIE_TOL,
        ci_method="BCa bootstrap, rounds resampled as whole pairs, "
                  "comparison-level and NOT multiplicity-adjusted",
        permutation_test="exact two-sided paired sign-flip over all 2**12 = 4096 "
                         "sign assignments; no Monte Carlo plus-one correction",
        multiplicity="Holm-Bonferroni within each predefined family",
        families={k: [f"{CONFIGS[a][0]} vs {CONFIGS[b][0]}" for a, b in v]
                  for k, v in FAMILIES.items()},
        n_comparisons=len(comparisons),
    )

    frame.to_csv(OUT / "canonical_round_rps.csv", index=False)
    comparisons.to_csv(OUT / "paired_comparisons.csv", index=False)
    summary.to_csv(OUT / "model_rps_summary.csv", index=False)
    pd.DataFrame(manifest).to_csv(OUT / "input_manifest.csv", index=False)
    (OUT / "analysis_config.json").write_text(json.dumps(config, indent=2),
                                              encoding="utf-8")

    log.append("COMPARISON COUNT")
    log.append(f"  [PASS] 21 comparisons across 4 families "
               f"({', '.join(str(len(v)) for v in FAMILIES.values())})")
    log.append("")
    log.append("RESULTS BY FAMILY")
    for family in FAMILIES:
        log.append(f"  {family}")
        for _, r_ in comparisons[comparisons.family == family].iterrows():
            log.append(f"    {r_.comparison:<52} "
                       f"d={r_.mean_paired_difference:+.6f} "
                       f"CI[{r_.ci95_low:+.6f},{r_.ci95_high:+.6f}] "
                       f"p={r_.p_raw:.6f} holm={r_.p_holm:.6f} "
                       f"{'REJECT' if r_['reject_at_0.05'] else 'retain H0'} "
                       f"({r_.rounds_improved}/12)")
    (OUT / "verification_report.txt").write_text("\n".join(log) + "\n",
                                                 encoding="utf-8")

    print("\n".join(log))
    print(f"\nwrote 6 files to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
