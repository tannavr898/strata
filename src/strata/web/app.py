"""app.py — the Strata web app.

    python scripts/run_app.py          # http://127.0.0.1:8000

Farmer-facing, with a details toggle that doubles as the prototyping view. It
reads three things and computes nothing expensive:

    data/models/     fitted models + honest metrics   (scripts/build_models.py)
    data/strata.db   fields, sites, readings, lab tests (scripts/seed_db.py)
    data/processed/  the county-year analysis table

Models are loaded ONCE at startup, not per request -- cross-validating a crop
takes minutes, which is why `artifacts.py` exists at all.

THE HONESTY STATES
------------------
The states below are the reason this app is not a generic dashboard, and they are
the part most likely to be lost in a redesign. Each one is a case where the
truthful answer is not a number:

    WITHHELD       the crop's model does not clear the product gate
    NEEDS LAB TEST Mehlich-3 P/K absent, so no rate can be quantified
    SYNTHETIC      seeded or demo data, never a finding
    UNCALIBRATED   a raw probe reading with no probe->lab mapping behind it
    COUNTY-SCALE   the prediction is county context, not a field-level claim

That last one is the easiest to lose and the most important to keep. The moment a
number appears under a heading that names someone's field, the UI has implicitly
made a field-level claim the model does not support (METHODS.md 12.2). So the
caveat is attached to the prediction component itself rather than left to a
footnote someone can delete.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from strata import db
from strata.artifacts import Artifacts, latest_conditions
from strata.probe import CHANNELS
from strata.recommender import OptimizationEngine, SoilTest
from strata.yield_model import YieldModel

logger = logging.getLogger("strata.web")

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[2]

#: Loaded once at startup. None means the build step has not been run; every page
#: then degrades to "no models built" rather than crashing, because a prototype
#: someone just cloned should tell them what to run next.
STATE: dict = {"artifacts": None, "table": None, "error": None}


def _load_state() -> None:
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


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _load_state()
    yield


app = FastAPI(title="Strata", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=_HERE / "static"), name="static")
templates = Jinja2Templates(directory=_HERE / "templates")


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


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _soiltest_from_lab(lab, crop: str, county_name: Optional[str]) -> SoilTest:
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


def _prediction_for(field) -> dict:
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


def _recommendations_for(field, lab, prediction: dict) -> list:
    """Cited advice for a field, given a prediction already computed by the caller.

    `prediction` is passed in rather than recomputed: the route needs it anyway to
    render the yield card, and computing it twice per request is the same
    duplication artifacts.py exists to remove -- just cheaper.
    """
    if lab is None:
        return []
    soiltest = _soiltest_from_lab(lab, field["crop"], field["county_name"])
    # The module handoff, GATED exactly as main.py gates it: a model yield scales
    # the maintenance rate ONLY when the crop cleared the product gate. Otherwise
    # the engine falls back to the farmer's stated goal, or declines to state a
    # rate at all.
    expected = prediction["value"] if prediction["state"] == "ok" else None
    return OptimizationEngine().recommend(soiltest, expected_yield=expected)


def _base_context(request: Request, details: bool) -> dict:
    artifacts = STATE["artifacts"]
    return {
        "request": request,
        "details": details,
        "artifacts": artifacts,
        "manifest": artifacts.manifest if artifacts else None,
        "stale": artifacts.is_stale() if artifacts else False,
        "startup_error": STATE["error"],
    }


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/", response_class=HTMLResponse)
def index(request: Request, details: bool = False):
    conn = db.init_db()
    fields = db.get_fields(conn)
    conn.close()

    if len(fields) == 1:
        return field_view(request, fields[0]["id"], details)

    ctx = _base_context(request, details)
    ctx["fields"] = fields
    return templates.TemplateResponse(request, "index.html", ctx)


@app.get("/field/{field_id}", response_class=HTMLResponse)
def field_view(request: Request, field_id: int, details: bool = False):
    conn = db.init_db()
    field = db.get_field(conn, field_id)
    if field is None:
        conn.close()
        ctx = _base_context(request, details)
        ctx["message"] = f"No field with id {field_id}."
        return templates.TemplateResponse(request, "notfound.html", ctx, status_code=404)

    lab = db.latest_lab_test(conn, field_id)
    sites = db.sites_for_field(conn, field_id)
    channels = db.reading_summary(conn, field_id)
    conn.close()

    prediction = _prediction_for(field)
    ctx = _base_context(request, details)
    ctx.update({
        "field": field,
        "lab": lab,
        "sites": sites,
        "channels": channels,
        "channel_meta": CHANNELS,
        "prediction": prediction,
        "recommendations": _recommendations_for(field, lab, prediction),
    })
    return templates.TemplateResponse(request, "field.html", ctx)


@app.get("/field/{field_id}/site/{site_id}", response_class=HTMLResponse)
def site_view(request: Request, field_id: int, site_id: int,
              channel: str = "soil_moisture", details: bool = False):
    conn = db.init_db()
    field = db.get_field(conn, field_id)
    readings = db.readings_for_site(conn, site_id, channel)
    site = next((s for s in db.sites_for_field(conn, field_id) if s["id"] == site_id), None)
    channels = [r["channel"] for r in db.reading_summary(conn, field_id)]
    conn.close()

    if field is None or site is None:
        ctx = _base_context(request, details)
        ctx["message"] = "No such site."
        return templates.TemplateResponse(request, "notfound.html", ctx, status_code=404)

    points = [
        {"t": r["taken_at"], "raw": r["raw_value"],
         "calibrated": r["calibrated_value"], "source": r["source"]}
        for r in readings
    ]
    ctx = _base_context(request, details)
    ctx.update({
        "field": field, "site": site, "channel": channel,
        "channel_meta": CHANNELS.get(channel),
        "channels": channels, "points": points,
        "sparkline": _sparkline(points),
        "any_synthetic": any(p["source"] == "synthetic" for p in points),
        "any_uncalibrated": any(p["calibrated"] is None for p in points),
    })
    return templates.TemplateResponse(request, "site.html", ctx)


def _sparkline(points: list[dict], width: int = 640, height: int = 180) -> dict:
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


@app.get("/models", response_class=HTMLResponse)
def models_view(request: Request, details: bool = True):
    """The model card: every crop's honest skill and its gate decision.

    Deliberately shows WITHHELD crops too. A product surface that only listed what
    it ships would hide the most interesting thing about the system, which is that
    it declines to answer when it cannot.
    """
    ctx = _base_context(request, details)
    artifacts = STATE["artifacts"]
    ctx["forecasts"] = (
        sorted(artifacts.forecasts.values(), key=lambda f: f.forecast_r2, reverse=True)
        if artifacts else []
    )
    ctx["threshold"] = YieldModel.PRODUCT_MIN_FORECAST_R2
    return templates.TemplateResponse(request, "models.html", ctx)
