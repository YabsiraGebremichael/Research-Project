"""Build the compact temperature x top-k validation ablation table.

Reads only the stored validation-grid results and the saved selected-configuration
artifacts, then writes a two-panel table as PNG, CSV and a standalone LaTeX
snippet. No inference is run and no experimental output is altered.

Run from the repository root:
    python "Statistical Analysis/build_sampling_ablation_table.py"
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "Competition Visuals" / "tables"
STEM = "sampling_factorial_ablation_compact"

MODELS = [("financial_chronos", "Financial Chronos (46M)"),
          ("chronos_t5_base", "Chronos-T5 Base (200M)")]
TEMPERATURES = [1.0, 1.5, 2.0, 3.0, 3.5, 5.0]
TOPK = ["50", "500", "full"]
TOPK_HEADER = {"50": "top-k 50", "500": "top-k 500", "full": "full vocab."}
DECIMALS = 5          # differences of interest appear in the 4th and 5th place


def load(model_key: str) -> tuple[pd.DataFrame, tuple[float, str], float]:
    """Return the 6x3 RPS grid, the selected cell and its recorded RPS."""
    base = ROOT / "TSFM Uncertainty Experiment" / "results" / model_key / "validation"
    grid = pd.read_csv(base / "validation_summary.csv")
    assert len(grid) == 18, f"{model_key}: expected 18 rows, got {len(grid)}"
    assert sorted(grid.temperature.unique()) == TEMPERATURES, f"{model_key}: temperatures"
    assert sorted(grid.top_k.astype(str).unique()) == sorted(TOPK), f"{model_key}: top_k"

    pivot = grid.pivot(index="temperature", columns="top_k",
                       values="mean_rps").loc[TEMPERATURES, TOPK]
    assert pivot.notna().all().all(), f"{model_key}: grid has gaps"

    cfg = json.loads((base / "selected_config.json").read_text(encoding="utf-8"))
    selected = (float(cfg["temperature"]), str(cfg["top_k_label"]))
    return pivot, selected, float(cfg["selection_value"])


def verify(model_key: str, pivot: pd.DataFrame, selected, selection_value) -> list[str]:
    """Check every grid value, the selected cell and the marginal means."""
    log = []
    base = ROOT / "TSFM Uncertainty Experiment" / "results" / model_key / "validation"
    raw = pd.read_csv(base / "validation_summary.csv")

    # 1. all 18 cells match the source rows exactly
    n = 0
    for _, r in raw.iterrows():
        stored = pivot.loc[float(r.temperature), str(r.top_k)]
        assert abs(stored - r.mean_rps) < 1e-12, f"{model_key}: cell mismatch"
        n += 1
    log.append(f"  [PASS] {model_key}: all {n} grid cells match validation_summary.csv exactly")

    # 2. the selected cell is the one the artifact names, and its value agrees
    t, k = selected
    cell = pivot.loc[t, k]
    assert abs(cell - selection_value) < 1e-8, (
        f"{model_key}: selected cell {cell} != selection_value {selection_value}")
    log.append(f"  [PASS] {model_key}: selected cell (T={t:g}, {k}) = {cell:.8f} "
               f"matches selected_config.json selection_value")

    # 3. the selected cell is in fact the grid minimum (the stated selection rule)
    flat = pivot.stack()
    assert abs(flat.min() - cell) < 1e-12, f"{model_key}: selected cell is not the minimum"
    log.append(f"  [PASS] {model_key}: selected cell is the grid minimum, consistent "
               f"with the lowest-mean-RPS selection rule")

    # 4. marginal means recomputed independently from the raw rows
    for k2 in TOPK:
        expected = raw[raw.top_k.astype(str) == k2].mean_rps.mean()
        assert abs(pivot[k2].mean() - expected) < 1e-12, f"{model_key}: col margin {k2}"
    for t2 in TEMPERATURES:
        expected = raw[raw.temperature == t2].mean_rps.mean()
        assert abs(pivot.loc[t2].mean() - expected) < 1e-12, f"{model_key}: row margin {t2}"
    log.append(f"  [PASS] {model_key}: all 6 row and 3 column marginal means reproduce "
               f"the raw grid means")

    # 5. grand mean consistent both ways
    assert abs(pivot.mean().mean() - raw.mean_rps.mean()) < 1e-12
    log.append(f"  [PASS] {model_key}: grand mean agrees across both margins")
    return log


def main() -> None:
    for ext in ("png", "csv", "tex"):
        p = TABLES / f"{STEM}.{ext}"
        if p.exists():
            raise SystemExit(f"STOPPING: {p.relative_to(ROOT)} already exists.")

    panels, checks = [], []
    for key, label in MODELS:
        pivot, selected, value = load(key)
        checks += verify(key, pivot, selected, value)
        panels.append((label, pivot, selected))

    # ---------------------------------------------------------------- CSV
    rows = []
    for label, pivot, selected in panels:
        for t in TEMPERATURES:
            rows.append({
                "panel": label, "temperature": f"{t:g}",
                **{TOPK_HEADER[k]: round(float(pivot.loc[t, k]), 8) for k in TOPK},
                "marginal mean across top-k": round(float(pivot.loc[t].mean()), 8),
                "selected in this row": TOPK_HEADER[selected[1]] if selected[0] == t else "",
            })
        rows.append({
            "panel": label, "temperature": "marginal mean across temperatures",
            **{TOPK_HEADER[k]: round(float(pivot[k].mean()), 8) for k in TOPK},
            "marginal mean across top-k": round(float(pivot.mean().mean()), 8),
            "selected in this row": "",
        })
    csv_frame = pd.DataFrame(rows)
    csv_frame.to_csv(TABLES / f"{STEM}.csv", index=False)

    # ---------------------------------------------------------------- PNG
    # grayscale-safe: selection is marked by a bold value, an asterisk and a
    # heavy border, never by colour alone
    fig, ax = plt.subplots(figsize=(9.2, 5.9))
    fig.patch.set_facecolor("white")
    ax.axis("off")

    header = ["Temperature"] + [TOPK_HEADER[k] for k in TOPK] + ["Marginal mean\nacross top-k"]
    cells, kinds = [], []
    for label, pivot, selected in panels:
        cells.append([label] + [""] * 4)
        kinds.append(("panel", None))
        for t in TEMPERATURES:
            row = [f"{t:g}"]
            for k in TOPK:
                v = f"{pivot.loc[t, k]:.{DECIMALS}f}"
                row.append(f"{v}*" if (t, k) == selected else v)
            row.append(f"{pivot.loc[t].mean():.{DECIMALS}f}")
            cells.append(row)
            kinds.append(("data", t))
        cells.append(["Marginal mean\nacross temperatures"]
                     + [f"{pivot[k].mean():.{DECIMALS}f}" for k in TOPK]
                     + [f"{pivot.mean().mean():.{DECIMALS}f}"])
        kinds.append(("margin", None))

    tbl = ax.table(cellText=cells, colLabels=header, cellLoc="center",
                   loc="upper center", colWidths=[0.235, 0.17, 0.17, 0.17, 0.20])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.2)
    tbl.scale(1, 1.75)

    sel_by_panel = {i: s for i, (_, _, s) in enumerate(panels)}
    panel_index, stripe = -1, 0
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#c8ced6")
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_facecolor("#1a1a1a")
            cell.set_text_props(color="white", fontweight="bold")
            cell._text.set_clip_on(False)   # two-line header may overflow
            cell._text.set_fontsize(8.8)
            continue
        kind, t = kinds[row - 1]
        if kind == "panel":
            cell.set_facecolor("#d9d9d9")
            cell.set_edgecolor("#d9d9d9")
            if col == 0:
                cell.set_text_props(fontweight="bold", ha="left", color="#1a1a1a")
                cell._text.set_x(0.02)
                cell._text.set_clip_on(False)
            stripe = 0
        elif kind == "margin":
            cell.set_facecolor("#ededed")
            cell.set_text_props(fontweight="bold")
            if col == 0:
                cell.set_text_props(fontweight="bold", ha="left", fontsize=8.2)
                cell._text.set_x(0.03)
        else:
            cell.set_facecolor("#f7f7f7" if stripe % 2 else "white")
            if col == len(header) - 1:
                cell.set_facecolor("#ededed")
                stripe += 1
            if col == 0:
                cell.set_text_props(ha="left")
                cell._text.set_x(0.06)

    # heavy border on the two selected cells, found by position not by assumption
    for row, (kind, t) in enumerate(kinds, start=1):
        if kind != "data":
            continue
        panel_index = 0 if row <= len(TEMPERATURES) + 1 else 1
        sel_t, sel_k = sel_by_panel[panel_index]
        if t == sel_t:
            col = TOPK.index(sel_k) + 1
            c = tbl[row, col]
            c.set_linewidth(2.4)
            c.set_edgecolor("#1a1a1a")
            c.set_text_props(fontweight="bold")

    ax.set_title("Temperature x top-k validation ablation: mean RPS over the six "
                 "pre-M6 origins",
                 fontsize=11.5, pad=14, loc="left", x=0.012)
    fig.text(0.012, 0.030,
             "LOWER RPS IS BETTER. Each cell is the mean Ranked Probability Score "
             "across the same six pre-M6 validation origins, with every other\n"
             "setting held fixed (500 samples, context 512, horizon 20, seed 42). "
             "The marginal mean across top-k averages a temperature over the three\n"
             "top-k settings; the marginal mean across temperatures averages a "
             "top-k setting over the six temperatures. Comparing the two margins\n"
             "shows which factor dominates. The bold, asterisked and heavily "
             "bordered cell in each panel is the configuration selected by the\n"
             "lowest-mean-RPS rule, read from that model's selected_config.json.\n"
             "Rounding note: for Chronos-T5 Base the selected cell beats "
             "the T=5.0 full-vocabulary cell by 0.0000069, a near-tie that "
             "five decimals cannot show.",
             fontsize=8.0, color="#333333", va="bottom", linespacing=1.5)
    fig.tight_layout(rect=(0, 0.155, 1, 1))
    fig.savefig(TABLES / f"{STEM}.png", dpi=200, facecolor="white",
                bbox_inches="tight")
    plt.close(fig)

    # ---------------------------------------------------------------- LaTeX
    B = chr(92)
    lines = [
        "% Compact temperature x top-k validation ablation table.",
        "% Standalone snippet: paste into the report where required.",
        "% Requires booktabs. Generated from the stored validation grids.",
        B + "begin{table}[t]", B + "centering", B + "small",
        B + "caption{Temperature and top-$k$ validation ablation. Each cell is "
        "the mean RPS over the same six pre-M6 origins; lower is better. "
        "Marginal means average a row over the three top-$k$ settings and a "
        "column over the six temperatures. Bold marks the configuration "
        "selected by the lowest-mean-RPS rule. For Chronos-T5 Base that "
        "selection is a near-tie, ahead of the $T=5.0$ full-vocabulary cell "
        "by 0.0000069.}",
        B + "label{tab:sampling-ablation}",
        B + "begin{tabular}{lrrrr}", B + "toprule",
        "Temperature & top-$k$ 50 & top-$k$ 500 & full vocab. & Marginal mean "
        + B * 2, B + "midrule",
    ]
    for i, (label, pivot, selected) in enumerate(panels):
        if i:
            lines.append(B + "midrule")
        lines.append(B + "multicolumn{5}{l}{" + B + "textit{" + label + "}} " + B * 2)
        for t in TEMPERATURES:
            vals = []
            for k in TOPK:
                v = f"{pivot.loc[t, k]:.{DECIMALS}f}"
                vals.append(B + "textbf{" + v + "}" if (t, k) == selected else v)
            lines.append(f"{t:g} & " + " & ".join(vals)
                         + f" & {pivot.loc[t].mean():.{DECIMALS}f} " + B * 2)
        lines.append(B + "cmidrule(lr){1-5}")
        lines.append("Marginal mean & "
                     + " & ".join(f"{pivot[k].mean():.{DECIMALS}f}" for k in TOPK)
                     + f" & {pivot.mean().mean():.{DECIMALS}f} " + B * 2)
    lines += [B + "bottomrule", B + "end{tabular}", B + "end{table}", ""]
    (TABLES / f"{STEM}.tex").write_text("\n".join(lines), encoding="utf-8")

    print("VERIFICATION")
    print("\n".join(checks))
    print(f"\nwrote {STEM}.{{png,csv,tex}} to {TABLES.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
