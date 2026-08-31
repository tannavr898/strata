"""views.py — read-only routes.

Everything here answers a question about a field. Nothing here changes state;
writes live in `edit.py`.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from strata import db
from strata.probe import CHANNELS
from strata.web.core import (
    STATE, base_context, not_found, prediction_for, recommendations_for,
    sparkline, templates,
)
from strata.yield_model import YieldModel

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
def index(request: Request, details: bool = False):
    conn = db.init_db()
    fields = db.get_fields(conn)
    conn.close()

    # One field is the common case for a single farm, so skip the list entirely
    # rather than making someone tap through a page with one item on it.
    if len(fields) == 1:
        return field_view(request, fields[0]["id"], details)

    ctx = base_context(request, details)
    ctx["fields"] = fields
    return templates.TemplateResponse(request, "index.html", ctx)


@router.get("/field/{field_id}", response_class=HTMLResponse)
def field_view(request: Request, field_id: int, details: bool = False,
               error: Optional[str] = None):
    conn = db.init_db()
    field = db.get_field(conn, field_id)
    if field is None:
        conn.close()
        return not_found(request, details, f"No field with id {field_id}.")

    lab = db.latest_lab_test(conn, field_id)
    sites = db.sites_for_field(conn, field_id)
    channels = db.reading_summary(conn, field_id)
    conn.close()

    prediction = prediction_for(field)
    ctx = base_context(request, details)
    ctx.update({
        "field": field,
        "lab": lab,
        "sites": sites,
        "channels": channels,
        "channel_meta": CHANNELS,
        "prediction": prediction,
        "recommendations": recommendations_for(field, lab, prediction),
        # Site creation redirects back here, so a validation failure has to be
        # able to surface on this page rather than vanishing.
        "error": error,
    })
    return templates.TemplateResponse(request, "field.html", ctx)


@router.get("/field/{field_id}/site/{site_id}", response_class=HTMLResponse)
def site_view(request: Request, field_id: int, site_id: int,
              channel: Optional[str] = None, details: bool = False):
    conn = db.init_db()
    field = db.get_field(conn, field_id)
    site = next((s for s in db.sites_for_field(conn, field_id) if s["id"] == site_id), None)
    channels = [r["channel"] for r in db.reading_summary(conn, field_id)]

    # Default to a channel this SITE actually has, not a hardcoded favourite. A
    # fixed default showed an empty plot and then described soil moisture on a
    # site that only ever recorded pH -- an empty state that looked like a bug and
    # named the wrong measurement.
    available = [r["channel"] for r in conn.execute(
        "SELECT DISTINCT channel FROM reading WHERE site_id = ? ORDER BY channel",
        (site_id,),
    ).fetchall()]
    if channel is None or channel not in available:
        channel = "soil_moisture" if "soil_moisture" in available else (
            available[0] if available else "soil_moisture")

    readings = db.readings_for_site(conn, site_id, channel)
    conn.close()

    if field is None or site is None:
        return not_found(request, details, "No such site.")

    points = [
        {"t": r["taken_at"], "raw": r["raw_value"],
         "calibrated": r["calibrated_value"], "source": r["source"]}
        for r in readings
    ]
    ctx = base_context(request, details)
    ctx.update({
        "field": field, "site": site, "channel": channel,
        "channel_meta": CHANNELS.get(channel),
        "channels": channels, "points": points,
        "sparkline": sparkline(points),
        "any_synthetic": any(p["source"] == "synthetic" for p in points),
        "any_uncalibrated": any(p["calibrated"] is None for p in points),
    })
    return templates.TemplateResponse(request, "site.html", ctx)


@router.get("/models", response_class=HTMLResponse)
def models_view(request: Request, details: bool = True):
    """The model card: every crop's honest skill and its gate decision.

    Deliberately shows WITHHELD crops too. A product surface that only listed what
    it ships would hide the most interesting thing about the system, which is that
    it declines to answer when it cannot.
    """
    ctx = base_context(request, details)
    artifacts = STATE["artifacts"]
    ctx["forecasts"] = (
        sorted(artifacts.forecasts.values(), key=lambda f: f.forecast_r2, reverse=True)
        if artifacts else []
    )
    ctx["threshold"] = YieldModel.PRODUCT_MIN_FORECAST_R2
    return templates.TemplateResponse(request, "models.html", ctx)
