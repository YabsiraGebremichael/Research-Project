"""Optional seed-sensitivity experiment for the FinQBoost reimplementation.

DELIBERATELY SEPARATE from `finqboost.py`. The headline result is seed 29 and
seed 29 only - nothing here is averaged into it. This exists solely to
characterise how much of the reported RPS is model stochasticity, which matters
because `subsample=0.5` and `colsample_bytree=0.8` make training random and
because R and Python seed their RNGs differently, so the R original and this
port cannot agree bit-for-bit by construction.

Run it only when you want that spread; it is not part of the main pipeline.

Output:
    seed_sensitivity.csv

Usage:
    python "FinQ Boost Reimplementation/seed_sensitivity.py"
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("finqboost", HERE / "finqboost.py")
fq = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fq)

OUT = HERE / "seed_sensitivity.csv"
SEEDS = (29, 1, 7, 42, 123)          # 29 first: the primary, reported result


def main() -> int:
    logging.basicConfig(level=logging.WARNING)
    rows = []
    for seed in SEEDS:
        result = fq.run(seed=seed, write=False).iloc[0]
        rows.append({"seed": seed,
                     "mean_m6_rps": float(result["mean_m6_rps"]),
                     "difference_vs_naive": float(result["difference_vs_naive"]),
                     "rounds_beating_naive": int(result["rounds_beating_naive"]),
                     "is_primary": seed == fq.SEED})
        print(f"  seed {seed:>4}: RPS {rows[-1]['mean_m6_rps']:.6f}  "
              f"vs naive {rows[-1]['difference_vs_naive']:+.6f}  "
              f"{rows[-1]['rounds_beating_naive']}/12 beat naive"
              f"{'   <-- PRIMARY' if seed == fq.SEED else ''}")

    frame = pd.DataFrame(rows)
    values = frame["mean_m6_rps"].to_numpy()
    print(f"\nspread across {len(SEEDS)} seeds: min {values.min():.6f}  "
          f"max {values.max():.6f}  range {values.max() - values.min():.6f}  "
          f"sd {values.std(ddof=1):.6f}")
    print(f"primary (seed {fq.SEED}) remains the reported result: "
          f"{frame.loc[frame['is_primary'], 'mean_m6_rps'].iloc[0]:.6f}")
    frame.to_csv(OUT, index=False)
    print(f"wrote {OUT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
