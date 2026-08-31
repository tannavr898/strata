"""artifacts.py — fit the models once, serve them everywhere.

WHY THIS EXISTS
---------------
Cross-validating the yield model is expensive: a full `python -m strata.main` run
takes about four minutes, and almost all of it is cross-validation. A web request
cannot do that. Neither, really, should three separate entry points -- before this
module, `main.py` cross-validated corn once for its report and again for the
product gate, and `report.py` did it a third time to draw the same number.

So the expensive work moves to a build step. `build()` fits and evaluates every
crop ONCE, writes the fitted pipelines and their metrics to `data/models/`, and
`load()` reads them back in milliseconds. The web app, the dashboard, and any
future surface all read the same artifacts, which means they cannot disagree with
each other -- the stale-number bug this project already hit once (main.py quoting
R^2 0.55 while the run printed 0.61) becomes structurally impossible.

WHAT IS AND IS NOT CACHED
-------------------------
Cached: the fitted per-crop pipelines and their honest metrics. Serving a
prediction from these is legitimate -- it is the same fitted model, just not refit
per request.

NOT cached, and deliberately so: `python -m strata.main` still cross-validates
live. It is the research report, and its whole claim is that the numbers come from
a real run rather than a stored file. Speed is not what it is for.

HONESTY
-------
Every artifact carries a MANIFEST recording when it was built, from which analysis
table (by content hash), with which seed and gate threshold. A surface that shows a
prediction can therefore always say what produced it and how old it is, and a stale
artifact can be detected rather than silently trusted: `Artifacts.is_stale()`
compares the recorded table hash against the table on disk.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd

from strata.yield_model import ProductForecast, YieldModel

logger = logging.getLogger("strata.artifacts")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PROCESSED = _REPO_ROOT / "data" / "processed"
MODEL_DIR = _REPO_ROOT / "data" / "models"

_TABLE = _PROCESSED / "analysis_county_year.csv"
_MANIFEST = "MANIFEST.json"

#: Crops the product builds models for. Every one is evaluated and recorded; the
#: gate then decides which may actually be SHOWN. Research reports all of them --
#: withholding is a display decision, not a reason to skip the measurement.
BUILD_CROPS = [
    "corn_grain_yield",
    "soybean_yield",
    "corn_silage_yield",
    "winter_wheat_yield",
    "hay_yield",
    "barley_yield",
    "alfalfa_hay_yield",
    "oats_yield",
]


def _table_hash(path: Path = _TABLE) -> str:
    """Content hash of the analysis table, so staleness is detectable."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


@dataclass(frozen=True)
class Manifest:
    built_at: str
    table_hash: str
    table_rows: int
    seed: int
    gate_threshold: float
    crops: list[str]
    strata_features: list[str]


class Artifacts:
    """Loaded models + metrics. Cheap to construct, safe to hold for a process."""

    def __init__(self, manifest: Manifest, forecasts: dict[str, ProductForecast],
                 models: dict[str, object]):
        self.manifest = manifest
        self.forecasts = forecasts
        self._models = models

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, model_dir: Path = MODEL_DIR) -> "Artifacts":
        model_dir = Path(model_dir)
        manifest_path = model_dir / _MANIFEST
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"No model artifacts at {model_dir}. Build them first:\n"
                f"    python scripts/build_models.py"
            )
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = Manifest(**raw["manifest"])
        forecasts = {c: ProductForecast(**f) for c, f in raw["forecasts"].items()}
        models = {
            crop: joblib.load(model_dir / f"yield_{crop}.joblib")
            for crop in manifest.crops
            if (model_dir / f"yield_{crop}.joblib").exists()
        }
        logger.info("Loaded %d models built %s (table %s)",
                    len(models), manifest.built_at, manifest.table_hash)
        return cls(manifest, forecasts, models)

    def is_stale(self, table_path: Path = _TABLE) -> bool:
        """True if the analysis table changed since these artifacts were built."""
        table_path = Path(table_path)
        if not table_path.exists():
            return False        # cannot tell; do not cry wolf
        return _table_hash(table_path) != self.manifest.table_hash

    # ------------------------------------------------------------------ serving
    def shipped_crops(self) -> list[str]:
        """Crops the gate allows a forecast for, best skill first."""
        return sorted(
            (c for c, f in self.forecasts.items() if f.shipped),
            key=lambda c: self.forecasts[c].forecast_r2,
            reverse=True,
        )

    def predict(self, crop: str, features: pd.DataFrame) -> Optional[float]:
        """Predict one yield, or None if the gate withholds this crop.

        Returning None rather than a number is the point: a caller cannot
        accidentally display a forecast the product decided not to offer. Pair it
        with `forecasts[crop].message` to tell the user WHY there is no number.
        """
        forecast = self.forecasts.get(crop)
        if forecast is None:
            raise KeyError(f"No model artifact for crop={crop!r}. "
                           f"Available: {sorted(self.forecasts)}")
        if not forecast.shipped:
            logger.info("Withholding %s prediction: %s", crop, forecast.message)
            return None
        return float(self._models[crop].predict(features)[0])


def latest_conditions(county_ansi: str, crop: str,
                      table: Optional[pd.DataFrame] = None) -> Optional[pd.DataFrame]:
    """The most recent observed county-year row for a crop, as a prediction input.

    A forecast needs that season's weather and NDVI, which for a future season do
    not exist yet. The prototype therefore predicts at the LATEST OBSERVED
    conditions -- currently 2023, where Daymet coverage ends. Any surface showing
    the result must say so; it is a "what this county yields under recent
    conditions" number, not a forecast of an unobserved future season.

    Returns None when the county has no rows for this crop (thin NASS series, e.g.
    barley in most counties), so callers can say "no data" instead of guessing.
    """
    if table is None:
        table = YieldModel.load_table()
    rows = table[(table["crop"] == crop) & (table["county_ansi"] == county_ansi)]
    rows = rows.dropna(subset=[YieldModel.TARGET]).sort_values("year")
    if rows.empty:
        return None
    return rows.iloc[[-1]]


# --------------------------------------------------------------------------- #
# Building
# --------------------------------------------------------------------------- #
def build(crops: Optional[list[str]] = None, model_dir: Path = MODEL_DIR,
          table: Optional[pd.DataFrame] = None) -> Artifacts:
    """Fit and evaluate every crop once, then write the artifacts to disk.

    Slow by design (minutes) -- that is exactly why it is a build step and not
    something a request does.
    """
    crops = list(crops or BUILD_CROPS)
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    if table is None:
        table = YieldModel.load_table()

    forecasts: dict[str, ProductForecast] = {}
    models: dict[str, object] = {}

    for crop in crops:
        logger.info("Building %s ...", crop)
        # ONE cross-validation per crop: evaluate_for_product now returns the MAE
        # and dataset shape alongside the gate decision, so nothing downstream has
        # to run this again.
        forecast = YieldModel.evaluate_for_product(crop, table)
        forecasts[crop] = forecast

        # Fit on ALL rows for serving. The honest skill estimate is the
        # leave-counties-out number above, measured on held-out counties; refitting
        # on everything afterwards is standard and gives the served model the most
        # data. The two must never be confused, which is why the metric travels
        # WITH the model in the same artifact.
        X, y, _ = YieldModel.prepare_dataset(crop, table)
        model = YieldModel(include_trend=True).fit(X, y)
        models[crop] = model.pipeline
        # compress=3 matters more than it looks: a 400-tree forest pickles to ~46 MB
        # raw, and eight crops came to 230 MB. Compression takes that to a fraction
        # of it at a negligible load cost. The MODEL is unchanged -- same trees, same
        # predictions -- only the bytes on disk shrink.
        joblib.dump(model.pipeline, model_dir / f"yield_{crop}.joblib", compress=3)

        verdict = "SHIP" if forecast.shipped else "WITHHOLD"
        logger.info("  %s: LCO R^2 %.3f  MAE %.2f %s  -> %s",
                    crop, forecast.forecast_r2, forecast.forecast_mae,
                    forecast.unit, verdict)

    manifest = Manifest(
        built_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        table_hash=_table_hash(),
        table_rows=len(table),
        seed=42,
        gate_threshold=YieldModel.PRODUCT_MIN_FORECAST_R2,
        crops=crops,
        strata_features=list(YieldModel.FEATURES),
    )
    (model_dir / _MANIFEST).write_text(
        json.dumps(
            {"manifest": asdict(manifest),
             "forecasts": {c: asdict(f) for c, f in forecasts.items()}},
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("Wrote %d models + manifest to %s", len(models), model_dir)
    return Artifacts(manifest, forecasts, models)
