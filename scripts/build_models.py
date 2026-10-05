"""Fit every crop's yield model once and write the serving artifacts.

Run this after the analysis table changes (new data pull, new features, a join
fix). It is slow -- minutes -- because it cross-validates each crop honestly; that
is exactly why it is a build step and not something a web request does.

    python scripts/build_models.py                 # all crops
    python scripts/build_models.py corn_grain_yield soybean_yield

Outputs to data/models/: one joblib per crop plus MANIFEST.json, which records the
build time, the analysis table's content hash, the seed, and the gate threshold.
The web app reads these; `strata.main` still cross-validates live, because it is
the research report and its claim is that the numbers come from a real run.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from strata.artifacts import BUILD_CROPS, MODEL_DIR, build  # noqa: E402
from strata.metrics_log import append_run  # noqa: E402
from strata.yield_model import YieldModel  # noqa: E402


def main(argv: list[str]) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    crops = argv[1:] or BUILD_CROPS
    table = YieldModel.load_table()
    unknown = sorted(set(crops) - set(table["crop"].unique()))
    if unknown:
        print(f"Unknown crop(s): {unknown}\n"
              f"Available: {sorted(table['crop'].unique())}", file=sys.stderr)
        return 2

    print(f"Building {len(crops)} model(s) from {len(table)} analysis rows.")
    print("This cross-validates each crop honestly, so it takes a few minutes.\n")
    artifacts = build(crops=crops, table=table)

    print(f"\n{'crop':<22}{'LCO R2':>9}{'MAE':>10}  {'unit':<12}{'decision':>10}")
    print("-" * 65)
    for crop in crops:
        f = artifacts.forecasts[crop]
        print(f"{crop:<22}{f.forecast_r2:>9.3f}{f.forecast_mae:>10.2f}  "
              f"{f.unit.lower():<12}{('SHIP' if f.shipped else 'WITHHOLD'):>10}")

    shipped = artifacts.shipped_crops()
    print(f"\nGate: LCO forecast R^2 >= {YieldModel.PRODUCT_MIN_FORECAST_R2:.2f}")
    print(f"{len(shipped)} of {len(crops)} crops ship: {', '.join(shipped)}")
    print(f"\nArtifacts written to {MODEL_DIR}")

    # One line in docs/metrics.jsonl per build, so the technical report can plot
    # each crop's honest skill against the change that produced it.
    append_run("build_models", {
        "table_hash": artifacts.manifest.table_hash,
        "table_rows": artifacts.manifest.table_rows,
        "gate_threshold": artifacts.manifest.gate_threshold,
        "crops": {c: {"lco_r2": round(f.forecast_r2, 4),
                      "lco_mae": round(f.forecast_mae, 4),
                      "unit": f.unit, "shipped": f.shipped,
                      "n_rows": f.n_rows, "n_counties": f.n_counties}
                  for c, f in artifacts.forecasts.items() if c in crops},
    })
    print("Metrics appended to docs/metrics.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
