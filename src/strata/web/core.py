"""core.py — shared state, templates, and the domain helpers the routes use.

Split out of `app.py` so the read routes (`views.py`) and the write routes
(`edit.py`) can share this without importing each other. `app.py` is now just
assembly: create the app, load state, mount static files, include the routers.

THE HONESTY STATES
------------------
These are the reason this app is not a generic dashboard, and the part most
likely to be lost in a redesign. Each is a case where the truthful answer is not
a number:

    WITHHELD       the crop's model does not clear the product gate
    NEEDS LAB TEST Mehlich-3 P/K absent, so no rate can be quantified
    SYNTHETIC      seeded or demo data, never a finding
    UNCALIBRATED   a raw probe reading with no probe->lab mapping behind it
    COUNTY-SCALE   the prediction is county context, not a field-level claim

`prediction_for` below is where the first one is enforced: it returns a dict with
a `state`, never a bare number, so "we withhold this" is exactly as renderable as
a forecast and a template cannot accidentally print past it.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from fastapi import Request
from fastapi.templating import Jinja2Templates

from strata.artifacts import Artifacts, latest_conditions
from strata.recommender import OptimizationEngine, SoilTest
from strata.yield_model import YieldModel

logger = logging.getLogger("strata.web")

HERE = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=HERE / "templates")

#: Loaded once at startup. None means the build step has not been run; every page
#: then degrades to "no models built" rather than crashing, because a prototype
#: someone just cloned should tell them what to run next.
STATE: dict = {"artifacts": None, "table": None, "error": None}


def load_state() -> None:
    """Load models and the analysis table once, at process start.

    Cross-validating a crop takes minutes, so this cannot happen per request --
    see artifacts.py. A missing build is a warning, not a crash: the app still
    serves and tells you which script to run.
    """
    try:
        STATE["artifacts"] = Artifacts.load()
        STATE["table"] = YieldModel.load_table()
        if STATE["artifacts"].is_stale():
            logger.warning("Model artifacts are STALE: the analysis table changed "
                           "since they were built. Re-run scripts/build_models.py")
    except FileNotFoundError as exc:
        STATE["error"] = str(exc)
        logger.warning("Starting without models: %s", exc)


# --------------------------------------------------------------------------- #
# Display
# --------------------------------------------------------------------------- #
#: Internal crop keys are NASS series names. They are correct and must stay the
#: keys, but "corn_grain_yield" should never reach a farmer's screen -- and it does
#: reach one, because the recommender embeds the crop key in its rule text. Rather
#: than rewrite the audited rule engine's strings, the display layer substitutes
#: labels on the way out. Display-only: nothing upstream sees these.
CROP_LABELS = {
    "corn_grain_yield": "corn",
    "corn_silage_yield": "corn silage",
    "soybean_yield": "soybeans",
    "winter_wheat_yield": "winter wheat",
    "barley_yield": "barley",
    "oats_yield": "oats",
    "hay_yield": "hay",
    "alfalfa_hay_yield": "alfalfa",
}


def crop_label(text: Optional[str]) -> str:
    """Replace any crop key inside a string with its farmer-facing label.

    LONGEST KEY FIRST, which is not optional: "hay_yield" is a substring of
    "alfalfa_hay_yield", so replacing the short one first turns alfalfa into
    "alfalfa_hay" and leaves the long key unmatched forever.
    """
    if not text:
        return ""
    out = str(text)
    for key in sorted(CROP_LABELS, key=len, reverse=True):
        out = out.replace(key, CROP_LABELS[key])
    return out


templates.env.filters["crop"] = crop_label


def base_context(request: Request, details: bool) -> dict:
    artifacts = STATE["artifacts"]
    return {
        "request": request,
        "details": details,
        "artifacts": artifacts,
        "manifest": artifacts.manifest if artifacts else None,
        "stale": artifacts.is_stale() if artifacts else False,
        "startup_error": STATE["error"],
    }


def not_found(request: Request, details: bool, message: str):
    ctx = base_context(request, details)
    ctx["message"] = message
    return templates.TemplateResponse(request, "notfound.html", ctx, status_code=404)


# --------------------------------------------------------------------------- #
# Domain
# --------------------------------------------------------------------------- #
def soiltest_from_lab(lab, crop: str, county_name: Optional[str]) -> SoilTest:
    """Turn a stored lab_test row into the recommender's input.

    Missing columns stay None rather than becoming defaults. That is what makes
    the engine return 'obtain a soil test' instead of a fabricated rate, so a
    zero-filling convenience here would silently disable the product's central
    honesty guarantee.
    """
    return SoilTest(
        crop=crop,
        ph=lab["ph"], buffer_ph=lab["buffer_ph"],
        mehlich3_p_ppm=lab["mehlich3_p_ppm"], mehlich3_k_ppm=lab["mehlich3_k_ppm"],
        cec=lab["cec"], om_pct=lab["om_pct"], yield_goal=lab["yield_goal"],
        county_name=county_name, note=lab["notes"],
    )


def prediction_for(field) -> dict:
    """The yield answer for a field, including the case where there isn't one.

    Always returns a dict with a `state` the template switches on, never a bare
    number, so "we withhold this" is as renderable as a forecast.
    """
    artifacts, table = STATE["artifacts"], STATE["table"]
    crop = field["crop"]

    if artifacts is None:
        return {"state": "no_models", "crop": crop}

    forecast = artifacts.forecasts.get(crop)
    if forecast is None:
        return {"state": "no_model_for_crop", "crop": crop}

    if not forecast.shipped:
        return {"state": "withheld", "crop": crop, "forecast": forecast,
                "message": forecast.message}

    row = latest_conditions(field["county_ansi"], crop, table)
    if row is None:
        return {"state": "no_county_data", "crop": crop, "forecast": forecast}

    features = row[YieldModel.FEATURES + [YieldModel.TREND_FEATURE]]
    value = artifacts.predict(crop, features)
    return {
        "state": "ok",
        "crop": crop,
        "value": value,
        "unit": forecast.unit.lower(),
        "mae": forecast.forecast_mae,
        "conditions_year": int(row["year"].iloc[0]),
        "forecast": forecast,
    }


def recommendations_for(field, lab, prediction: dict) -> list:
    """Cited advice for a field, given a prediction already computed by the caller.

    `prediction` is passed in rather than recomputed: the route needs it anyway to
    render the yield card, and computing it twice per request is the same
    duplication artifacts.py exists to remove -- just cheaper.
    """
    if lab is None:
        return []
    soiltest = soiltest_from_lab(lab, field["crop"], field["county_name"])
    # The module handoff, GATED exactly as main.py gates it: a model yield scales
    # the maintenance rate ONLY when the crop cleared the product gate. Otherwise
    # the engine falls back to the farmer's stated goal, or declines to state a
    # rate at all.
    expected = prediction["value"] if prediction["state"] == "ok" else None
    return OptimizationEngine().recommend(soiltest, expected_yield=expected)


def sparkline(points: list[dict], width: int = 640, height: int = 180) -> dict:
    """Lay out a time series as SVG coordinates.

    Server-side rather than a charting library: it is a dozen lines, adds no
    dependency and no build step, and renders identically everywhere. Swap it for
    a real chart library when the app needs interaction, not before.
    """
    values = [p["raw"] for p in points]
    if len(values) < 2:
        return {"empty": True}
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0
    pad = 24
    step = (width - 2 * pad) / (len(values) - 1)
    coords = [
        (pad + i * step, height - pad - (v - lo) / span * (height - 2 * pad))
        for i, v in enumerate(values)
    ]
    return {
        "empty": False,
        "path": " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}"
                         for i, (x, y) in enumerate(coords)),
        "dots": [{"x": round(x, 1), "y": round(y, 1), "v": v}
                 for (x, y), v in zip(coords, values)],
        "lo": lo, "hi": hi, "width": width, "height": height,
    }


# --------------------------------------------------------------------------- #
# Reference data for the field form
# --------------------------------------------------------------------------- #
def pa_counties() -> dict[str, str]:
    """{county_ansi: 'Chester'} for every PA county Strata has soil data for.

    Read from the SSURGO table rather than hardcoded, so the dropdown can only
    ever offer counties the pipeline can actually back with data. Cached: it is
    67 rows that change only when the data is re-pulled.
    """
    cached = STATE.get("counties")
    if cached is not None:
        return cached

    import pandas as pd

    path = Path(__file__).resolve().parents[3] / "data" / "processed" / "ssurgo_pa_soil.csv"
    if not path.exists():
        STATE["counties"] = {}
        return {}
    df = pd.read_csv(path, dtype={"county_ansi": "string"})
    counties = {
        str(r.county_ansi): str(r.areaname).split(" County")[0]
        for r in df.itertuples()
    }
    STATE["counties"] = dict(sorted(counties.items(), key=lambda kv: kv[1]))
    return STATE["counties"]


def modelled_crops() -> list[str]:
    """Crops with a built model, best skill first.

    Every built crop is offered, including the ones the gate withholds. Hiding
    them would be the wrong kindness: someone who grows oats should be able to
    add their oat field and see the honest "we can't forecast this" answer, along
    with the fertilizer advice that does not depend on a yield model at all.
    """
    artifacts = STATE["artifacts"]
    if artifacts is None:
        return []
    return sorted(artifacts.forecasts,
                  key=lambda c: artifacts.forecasts[c].forecast_r2, reverse=True)
