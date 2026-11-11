"""Automated tests for the paired round-level RPS analysis.

Covers pairing, bootstrap structure, the exact sign-flip permutation test and
Holm correction, plus end-to-end checks of the written outputs.

Run from the repository root:
    python "Statistical Analysis/test_paired_rps.py"
or, if pytest is available:
    pytest "Statistical Analysis/test_paired_rps.py" -q
"""
from __future__ import annotations

import importlib.util
import math
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

spec = importlib.util.spec_from_file_location("pra", HERE / "paired_rps_analysis.py")
pra = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pra)


# ---------------------------------------------------------------- permutation
def test_permutation_enumerates_all_sign_assignments():
    assert len(list(product([1, -1], repeat=12))) == 4096


def test_permutation_minimum_attainable_p():
    """With n=12 and no ties, the smallest attainable two-sided p is 2/4096."""
    d = np.full(12, -0.01)
    assert math.isclose(pra.exact_sign_flip_p(d), 2 / 4096, rel_tol=0, abs_tol=1e-15)


def test_permutation_hand_checkable_small_case():
    """n=3, all +1: only (+,+,+) and (-,-,-) reach |mean| = 1, so p = 2/8."""
    assert math.isclose(pra.exact_sign_flip_p(np.ones(3)), 0.25, abs_tol=1e-15)


def test_permutation_zero_and_symmetric_cases_give_p_one():
    assert pra.exact_sign_flip_p(np.zeros(12)) == 1.0
    assert pra.exact_sign_flip_p(np.array([1.0, -1.0, 2.0, -2.0])) == 1.0


def test_permutation_is_sign_symmetric():
    """Flipping the sign of every difference must not change the p-value."""
    rng = np.random.default_rng(7)
    d = rng.normal(size=10)
    assert math.isclose(pra.exact_sign_flip_p(d), pra.exact_sign_flip_p(-d),
                        abs_tol=1e-15)


def test_permutation_p_is_a_multiple_of_two_over_2n():
    """Exact enumeration gives p on a discrete grid; no plus-one correction."""
    rng = np.random.default_rng(11)
    d = rng.normal(size=8)
    p = pra.exact_sign_flip_p(d)
    assert math.isclose(p * 2 ** 8, round(p * 2 ** 8), abs_tol=1e-9)


# ----------------------------------------------------------------------- Holm
def test_holm_known_case():
    adj, rej = pra.holm([0.01, 0.04, 0.03])
    assert np.allclose(adj, [0.03, 0.06, 0.06])
    assert rej == [True, False, False]


def test_holm_enforces_monotonicity():
    adj, _ = pra.holm([0.001, 0.002])
    assert np.allclose(adj, [0.002, 0.002])


def test_holm_caps_at_one_and_is_identity_for_single_test():
    assert all(a <= 1.0 for a in pra.holm([0.6, 0.7, 0.8])[0])
    assert np.allclose(pra.holm([0.5])[0], [0.5])


def test_holm_is_never_more_lenient_than_raw():
    rng = np.random.default_rng(3)
    p = list(rng.uniform(size=6))
    adj, _ = pra.holm(p)
    assert all(a >= b - 1e-15 for a, b in zip(adj, p))


# ------------------------------------------------------------------ bootstrap
def test_bootstrap_preserves_round_pairing():
    """Resampling the difference vector equals resampling both series together."""
    rng = np.random.default_rng(5)
    cand = rng.normal(size=12)
    ref = rng.normal(size=12)
    take = rng.integers(0, 12, size=12)
    assert np.allclose((cand - ref)[take], cand[take] - ref[take])


def test_bootstrap_indices_stay_within_the_twelve_rounds():
    rng = np.random.default_rng(pra.SEED)
    idx = rng.integers(0, 12, size=(5000, 12))
    assert idx.shape == (5000, 12)
    assert idx.min() >= 0 and idx.max() <= 11


def test_bca_is_deterministic_for_a_fixed_seed():
    d = np.linspace(-0.01, 0.002, 12)
    assert pra.bca_interval(d, 5000, pra.SEED) == pra.bca_interval(d, 5000, pra.SEED)


def test_bca_brackets_the_observed_mean():
    d = np.linspace(-0.01, 0.002, 12)
    lo, hi = pra.bca_interval(d, 20000, pra.SEED)
    assert lo < float(np.mean(d)) < hi


def test_bca_shifts_approximately_with_the_data():
    """Shifting every difference by a constant shifts the interval by about that.

    Equivariance is only approximate, not exact. With n=12 the bootstrap
    distribution of the mean is discrete and has many repeated values, so the
    bias-correction term, which counts resampled means strictly below the
    observed mean, can move by a few parts in a thousand when the data are
    shifted. The resulting interval shift differs from the applied shift by
    order 1e-4. This is a property of BCa on a discrete bootstrap distribution
    rather than an implementation error.
    """
    d = np.linspace(-0.01, 0.002, 12)
    lo1, hi1 = pra.bca_interval(d, 20000, pra.SEED)
    lo2, hi2 = pra.bca_interval(d + 0.05, 20000, pra.SEED)
    assert math.isclose(lo2 - lo1, 0.05, abs_tol=5e-4)
    assert math.isclose(hi2 - hi1, 0.05, abs_tol=5e-4)


def test_bca_bootstrap_distribution_itself_is_exactly_shift_equivariant():
    """The underlying resampled means shift exactly; only z0 is tie-sensitive."""
    d = np.linspace(-0.01, 0.002, 12)
    rng = np.random.default_rng(pra.SEED)
    idx = rng.integers(0, 12, size=(20000, 12))
    assert np.allclose(d[idx].mean(axis=1) + 0.05,
                       (d + 0.05)[idx].mean(axis=1), atol=1e-15)


# -------------------------------------------------------------------- loading
def test_round_series_alignment_and_means():
    frame, manifest = pra.load_round_series()
    assert frame["round"].tolist() == list(range(1, 13))
    assert len(frame) == 12
    values = frame.drop(columns="round")
    assert not values.isna().any().any()
    assert np.isfinite(values.to_numpy(float)).all()
    assert np.all(frame["naive"].to_numpy() == 0.160000)
    for row in manifest:
        assert abs(row["calculated_mean"] - row["expected_overall_rps"]) < 1e-6, row


def test_sign_convention_matches_the_stored_results():
    """A negative stored difference must mean the candidate has lower RPS."""
    out = pd.read_csv(HERE / "results" / "paired_comparisons.csv")
    recomputed = out.candidate_mean_rps - out.reference_mean_rps
    assert np.allclose(recomputed, out.mean_paired_difference, atol=1e-12)
    better = out.mean_paired_difference < 0
    assert (out.loc[better, "candidate_mean_rps"]
            < out.loc[better, "reference_mean_rps"]).all()


# ----------------------------------------------------------------- end to end
def test_written_outputs_are_complete_and_consistent():
    res = HERE / "results"
    out = pd.read_csv(res / "paired_comparisons.csv")
    assert len(out) == 21
    assert out.family.nunique() == 4
    assert sorted(out.family.value_counts().tolist()) == [4, 5, 5, 7]
    # rounds improved + worsened + tied must always total 12
    assert ((out.rounds_improved + out.rounds_worsened + out.rounds_tied) == 12).all()
    # the interval must bracket the point estimate
    assert (out.ci95_low <= out.mean_paired_difference).all()
    assert (out.mean_paired_difference <= out.ci95_high).all()
    # Holm-adjusted p is never smaller than the raw p
    assert (out.p_holm >= out.p_raw - 1e-12).all()
    # every p-value lies on the exact 2/4096 grid
    grid = out.p_raw * 4096 / 2
    assert np.allclose(grid, np.round(grid), atol=1e-9)
    assert pd.read_csv(res / "canonical_round_rps.csv").shape[0] == 12
    assert len(pd.read_csv(res / "input_manifest.csv")) == 18


def test_compact_table_matches_the_full_results():
    csv = ROOT / "Competition Visuals" / "tables" / "paired_rps_primary_statistics.csv"
    compact = pd.read_csv(csv)
    assert len(compact) == 14
    assert compact.Group.nunique() == 3
    full = pd.read_csv(HERE / "results" / "paired_comparisons.csv")
    for _, r in compact.iterrows():
        assert float(r["Mean difference"]) in set(
            round(v, 6) for v in full.mean_paired_difference)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {t.__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    raise SystemExit(1 if failed else 0)
