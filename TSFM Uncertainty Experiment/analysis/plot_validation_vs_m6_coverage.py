"""Plot internal-validation against final-M6 interval coverage and width by beta.

Reads the recovered per-beta training_metadata.json files for the internal
validation figures and the scored combined_summary.csv for the final M6 figures,
then writes one two-panel figure. No loss values are plotted, because beta
changes the objective and losses are not comparable across beta.

Run from the repository root:
    python "TSFM Uncertainty Experiment/analysis/plot_validation_vs_m6_coverage.py"
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / "TSFM Uncertainty Experiment"
METADATA_DIR = (ROOT / "TSFM Uncertainty Experiment" / "artifacts"
                / "uncertainty_regularised")
SUMMARY = EXPERIMENT / "results" / "chronos2_uncertainty_lora" / "combined_summary.csv"
VISUALS_DIR = EXPERIMENT / "visuals" / "chronos2_uncertainty_lora"

BETAS = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
INTERVALS = ["Q0.10-Q0.90", "Q0.20-Q0.80"]
COLOURS = {"Q0.10-Q0.90": "#08519c", "Q0.20-Q0.80": "#e6550d"}


def load_internal_validation() -> pd.DataFrame:
    """Per-beta internal-validation coverage and width from the recovered metadata."""
    rows = []
    for beta in BETAS:
        path = METADATA_DIR / f"beta{beta:.1f}" / "training_metadata.json"
        meta = json.loads(path.read_text(encoding="utf-8"))
        # the metadata records beta as a number; fail loudly if the folder lies
        assert float(meta["beta"]) == beta, f"{path} records beta {meta['beta']}"
        coverage = meta["validation_coverage"]
        row = {"beta": beta, "valid_observations": coverage["valid_observations"]}
        for interval in INTERVALS:
            row[f"{interval}_coverage"] = coverage[interval]["empirical_coverage"]
            row[f"{interval}_mean_width"] = coverage[interval]["mean_width"]
            row[f"{interval}_target"] = coverage[interval]["target"]
        rows.append(row)
    table = pd.DataFrame(rows)
    assert (table["valid_observations"] == 2000).all(), "expected 2,000 validation obs"
    return table


def load_final_m6() -> pd.DataFrame:
    """Per-beta final-M6 coverage and width from the scored summary."""
    table = pd.read_csv(SUMMARY)
    table = table[table["beta"].notna()].sort_values("beta").reset_index(drop=True)
    assert list(table["beta"]) == BETAS, f"unexpected betas {list(table['beta'])}"
    return table


def main() -> None:
    internal = load_internal_validation()
    m6 = load_final_m6()

    # the two targets are shared across betas; read them once and check they agree
    targets = {}
    for interval in INTERVALS:
        values = set(internal[f"{interval}_target"]) | set(m6[f"{interval}_target"])
        assert len(values) == 1, f"{interval} target disagrees: {values}"
        targets[interval] = values.pop()

    fig, (left, right) = plt.subplots(1, 2, figsize=(11.8, 5.0))
    fig.patch.set_facecolor("white")

    # ---- left panel: coverage, internal validation against final M6 ----
    for interval in INTERVALS:
        colour = COLOURS[interval]
        left.plot(internal["beta"], internal[f"{interval}_coverage"],
                  marker="o", markersize=6, linewidth=2.0, color=colour,
                  label=f"{interval}  internal validation")
        left.plot(m6["beta"], m6[f"{interval}_coverage"],
                  marker="s", markersize=5, linewidth=1.8, linestyle="--",
                  color=colour, alpha=0.75,
                  label=f"{interval}  final M6")
        left.axhline(targets[interval], color=colour, linestyle=":", linewidth=1.2,
                     alpha=0.9)
        left.annotate(f"training target {targets[interval]:.0%}",
                      (BETAS[0], targets[interval]), textcoords="offset points",
                      xytext=(2, 4), fontsize=8.5, color=colour)

    left.set_xlabel("beta (widening pressure, log scale)")
    left.set_ylabel("Empirical interval coverage")
    # the 75% target is met on BOTH bases; only the 95% target separates them
    left.set_title("Both bases reach the 75% target; only the validation\n"
                   "window reaches the 95% target", fontsize=10.5)
    left.legend(frameon=False, fontsize=8.2, loc="lower right")

    # ---- right panel: mean interval width on the same two bases ----
    for interval in INTERVALS:
        colour = COLOURS[interval]
        right.plot(internal["beta"], internal[f"{interval}_mean_width"],
                   marker="o", markersize=6, linewidth=2.0, color=colour,
                   label=f"{interval}  internal validation")
        right.plot(m6["beta"], m6[f"{interval}_mean_width"],
                   marker="s", markersize=5, linewidth=1.8, linestyle="--",
                   color=colour, alpha=0.75,
                   label=f"{interval}  final M6")

    right.set_xlabel("beta (widening pressure, log scale)")
    right.set_ylabel("Mean interval width (daily log return)")
    right.set_title("Validation intervals are wider at every beta, and the\n"
                    "gap to the M6 rounds grows with beta", fontsize=10.5)
    right.legend(frameon=False, fontsize=8.2, loc="upper left")

    for ax in (left, right):
        ax.set_xscale("log", base=2)
        ax.set_xticks(BETAS)
        ax.set_xticklabels([f"{b:g}" for b in BETAS])
        ax.grid(True, alpha=0.25, linewidth=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    fig.suptitle("Chronos-2 uncertainty-regularised LoRA: internal validation "
                 "(1 origin, 2,000 observations, solid)\nagainst final M6 "
                 "evaluation (12 rounds, 24,000 observations, dashed)",
                 fontsize=11.5)
    fig.text(0.5, 0.030,
             "Internal validation is the single 2022-02-07 to 2022-03-04 window "
             "used during training;",
             ha="center", fontsize=8.2, color="#444444")
    fig.text(0.5, 0.009,
             "the M6 figures span 2022-03 to 2023-02. Different periods, so "
             "not interchangeable coverage estimates.",
             ha="center", fontsize=8.2, color="#444444")

    fig.tight_layout(rect=(0, 0.062, 1, 1))
    VISUALS_DIR.mkdir(parents=True, exist_ok=True)
    out = VISUALS_DIR / "uncertainty_validation_vs_m6_coverage.png"
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)

    print(f"wrote {out.relative_to(ROOT)}")
    print("\ninternal validation against final M6, by beta:")
    for interval in INTERVALS:
        print(f"  {interval}  target {targets[interval]:.0%}")
        for beta in BETAS:
            iv = internal.loc[internal["beta"] == beta].iloc[0]
            fm = m6.loc[m6["beta"] == beta].iloc[0]
            print(f"    beta {beta:>4}: validation {iv[f'{interval}_coverage']:.4f} "
                  f"(w {iv[f'{interval}_mean_width']:.5f})   "
                  f"M6 {fm[f'{interval}_coverage']:.4f} "
                  f"(w {fm[f'{interval}_mean_width']:.5f})")


if __name__ == "__main__":
    main()
