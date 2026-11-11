"""Report-oriented forest plot and compact statistics table for the paired analysis.

Reads Statistical Analysis/results/paired_comparisons.csv and produces:
  Competition Visuals/figures/paired_rps_effects_forest.png
  Competition Visuals/tables/paired_rps_primary_statistics.{csv,png}

Both outputs cover the same 14 comparisons. The benchmark-versus-naive family
and the two sampling-versus-naive comparisons stay in the complete 21-comparison
CSV but are deliberately excluded here, to avoid duplicating the existing
headline comparison figure and overcrowding the plot.

Existing files are never overwritten: the script stops if a target exists.

Run from the repository root:
    python "Statistical Analysis/paired_rps_visualisation.py"
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path(__file__).resolve().parent / "results" / "paired_comparisons.csv"
FIG = ROOT / "Competition Visuals" / "figures" / "paired_rps_effects_forest.png"
TBL_CSV = ROOT / "Competition Visuals" / "tables" / "paired_rps_primary_statistics.csv"
TBL_PNG = ROOT / "Competition Visuals" / "tables" / "paired_rps_primary_statistics.png"

NAVY, ORANGE, GREY = "#08519c", "#e6550d", "#555555"

# (comparison string in the CSV, short label for the figure, group)
PANEL_A = [
    ("Chronos-T5 Base zero-shot vs Naive",
     "Chronos-T5 Base  vs  naive", "core"),
    ("Financial Chronos zero-shot vs Naive",
     "Financial Chronos  vs  naive", "core"),
    ("Chronos-2 zero-shot vs Naive",
     "Chronos-2  vs  naive", "core"),
    ("Chronos-2 zero-shot vs Chronos-T5 Base zero-shot",
     "Chronos-2  vs  Chronos-T5 Base", "core"),
    ("Financial Chronos zero-shot vs Chronos-T5 Base zero-shot",
     "Financial Chronos  vs  Chronos-T5 Base", "core"),
    ("Financial Chronos sampling vs Financial Chronos zero-shot",
     "FC + sampling  vs  FC zero-shot", "adapt"),
    ("Chronos-T5 Base sampling vs Chronos-T5 Base zero-shot",
     "T5 + sampling  vs  T5 zero-shot", "adapt"),
    ("Chronos-2 ordinary LoRA vs Chronos-2 zero-shot",
     "C2 ordinary LoRA  vs  C2 zero-shot", "adapt"),
]
PANEL_B = [
    (f"Uncertainty LoRA beta {b:g} vs Chronos-2 ordinary LoRA",
     f"beta {b:g}  vs  ordinary LoRA", "beta")
    for b in (0.5, 1, 2, 4, 8, 16)
]


def guard(path: Path) -> None:
    if path.exists():
        raise SystemExit(f"STOPPING: {path.relative_to(ROOT)} already exists; "
                         "refusing to overwrite.")


def rows_for(spec, table: pd.DataFrame) -> pd.DataFrame:
    out = []
    for comparison, label, group in spec:
        m = table[table.comparison == comparison]
        assert len(m) == 1, f"{comparison!r} matched {len(m)} rows"
        r = m.iloc[0].to_dict()
        r["label"] = label
        r["group"] = group
        out.append(r)
    return pd.DataFrame(out)


def draw_panel(ax, data: pd.DataFrame, title: str) -> None:
    y = range(len(data))[::-1]                 # first row at the top
    ax.axvline(0, color="#111111", linewidth=1.3, zorder=1)
    for yi, (_, r) in zip(y, data.iterrows()):
        favours_candidate = r.mean_paired_difference < 0
        colour = NAVY if favours_candidate else ORANGE
        survives = bool(r["reject_at_0.05"])
        ax.plot([r.ci95_low, r.ci95_high], [yi, yi], color=colour,
                linewidth=2.0, solid_capstyle="round", zorder=2)
        ax.plot([r.mean_paired_difference], [yi], marker="o" if survives else "o",
                markersize=8 if survives else 7,
                markerfacecolor=colour if survives else "white",
                markeredgecolor=colour, markeredgewidth=1.8, zorder=3)
    ax.set_yticks(list(y))
    ax.set_yticklabels(data.label.tolist(), fontsize=9)
    ax.set_title(title, fontsize=10.5, pad=8)
    ax.grid(True, axis="x", alpha=0.25, linewidth=0.6)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.7, len(data) - 0.3)


def main() -> None:
    for p in (FIG, TBL_CSV, TBL_PNG):
        guard(p)

    table = pd.read_csv(RESULTS)
    a = rows_for(PANEL_A, table)
    b = rows_for(PANEL_B, table)

    # ---------------- forest plot ----------------
    fig, (axa, axb) = plt.subplots(
        1, 2, figsize=(12.4, 5.2), gridspec_kw={"width_ratios": [1.28, 1.0]})
    fig.patch.set_facecolor("white")

    draw_panel(axa, a, "A. Zero-shot comparisons, sampling interventions\n"
                       "and ordinary LoRA")
    draw_panel(axb, b, "B. Uncertainty-regularised LoRA\n"
                       "against ordinary LoRA")
    # panel B spans a far narrower range; thin the ticks so they stay legible
    axb.xaxis.set_major_locator(MaxNLocator(nbins=5))

    for ax in (axa, axb):
        ax.set_xlabel("Paired mean RPS difference (candidate $-$ reference)\n"
                      "negative favours the candidate", fontsize=9.5)

    handles = [
        Line2D([], [], color=NAVY, marker="o", markersize=8, linewidth=2.0,
               markerfacecolor=NAVY,
               label="candidate better; survives Holm correction"),
        Line2D([], [], color=NAVY, marker="o", markersize=7, linewidth=2.0,
               markerfacecolor="white", markeredgewidth=1.8,
               label="candidate better; does not survive Holm"),
        Line2D([], [], color=ORANGE, marker="o", markersize=8, linewidth=2.0,
               markerfacecolor=ORANGE,
               label="candidate worse; survives Holm correction"),
        Line2D([], [], color=ORANGE, marker="o", markersize=7, linewidth=2.0,
               markerfacecolor="white", markeredgewidth=1.8,
               label="candidate worse; does not survive Holm"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               fontsize=8.6, bbox_to_anchor=(0.5, -0.015))

    fig.suptitle("Paired round-level RPS differences across the 12 official M6 "
                 "rounds, with 95% BCa bootstrap intervals", fontsize=11.5)
    fig.text(0.5, 0.075,
             "Filled markers survive Holm correction within their predefined "
             "family at alpha 0.05. Intervals are unadjusted comparison-level "
             "95% intervals, not multiplicity-adjusted.",
             ha="center", fontsize=8.2, color="#444444")
    fig.tight_layout(rect=(0, 0.125, 1, 0.94))
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"wrote {FIG.relative_to(ROOT)}")

    # ---------------- compact report table ----------------
    GROUP_TITLE = {"core": "Core zero-shot comparisons",
                   "adapt": "Sampling and ordinary-LoRA comparisons",
                   "beta": "Uncertainty-regularised LoRA comparisons"}
    compact = pd.concat([a, b], ignore_index=True)
    out = pd.DataFrame({
        "Group": [GROUP_TITLE[g] for g in compact.group],
        "Comparison": compact.label.str.replace("  ", " ", regex=False),
        "Mean difference": compact.mean_paired_difference.map(lambda v: f"{v:+.6f}"),
        "95% CI": [f"[{lo:+.6f}, {hi:+.6f}]"
                   for lo, hi in zip(compact.ci95_low, compact.ci95_high)],
        "p (exact)": compact.p_raw.map(lambda v: f"{v:.4f}"),
        "p (Holm)": compact.p_holm.map(lambda v: f"{v:.4f}"),
        "Rounds improved": compact.rounds_improved.map(lambda v: f"{v}/12"),
    })
    TBL_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(TBL_CSV, index=False)
    print(f"wrote {TBL_CSV.relative_to(ROOT)}")

    # rendered version: group titles are inserted as their own banner rows so
    # they can never drift out of alignment with the rows they label
    display_cols = list(out.columns[1:])
    cells, kinds = [], []
    current = None
    for _, r in out.iterrows():
        if r.Group != current:
            current = r.Group
            cells.append([current.upper()] + [""] * (len(display_cols) - 1))
            kinds.append("group")
        cells.append([r[c] for c in display_cols])
        kinds.append("data")

    fig, ax = plt.subplots(figsize=(13.2, 6.4))
    fig.patch.set_facecolor("white")
    ax.axis("off")
    tbl = ax.table(cellText=cells, colLabels=display_cols, cellLoc="center",
                   loc="upper center",
                   colWidths=[0.30, 0.15, 0.26, 0.10, 0.10, 0.13])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.5)

    stripe = 0
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#d8dde3")
        cell.set_linewidth(0.6)
        if row == 0:                                   # header
            cell.set_facecolor(NAVY)
            cell.set_text_props(color="white", fontweight="bold")
            continue
        kind = kinds[row - 1]
        if kind == "group":
            cell.set_facecolor("#dce6f2")
            cell.set_edgecolor("#dce6f2")
            if col == 0:
                cell.set_text_props(color=NAVY, fontweight="bold", ha="left")
                cell._text.set_x(0.02)
                cell._text.set_clip_on(False)   # banner may overflow its cell
            stripe = 0
        else:
            cell.set_facecolor("#f6f8fb" if stripe % 2 else "white")
            if col == 0:
                cell.set_text_props(ha="left")
                cell._text.set_x(0.03)
            if col == len(display_cols) - 1:
                stripe += 1

    ax.set_title("Paired round-level RPS statistics, 12 official M6 rounds",
                 fontsize=12, pad=16, loc="left", x=0.012)
    fig.text(0.012, 0.045,
             "Mean difference = candidate minus reference; NEGATIVE values "
             "favour the candidate. Intervals are 95% BCa bootstrap intervals "
             "over the 12 paired rounds and are unadjusted\ncomparison-level "
             "intervals. p (exact) is a two-sided paired sign-flip permutation "
             "p-value enumerating all 4,096 sign assignments. p (Holm) applies "
             "Holm correction within each\npredefined family. The complete "
             "21-comparison results are in "
             "Statistical Analysis/results/paired_comparisons.csv.",
             fontsize=8.0, color="#444444", va="bottom")
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    fig.savefig(TBL_PNG, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {TBL_PNG.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
