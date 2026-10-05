"""views.py — read-only routes.

Everything here answers a question about a field. Nothing here changes state;
writes live in `edit.py`.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from strata import db
from strata.data_ingest import load_source_config
from strata.probe import CHANNELS
from strata.web.core import (
    STATE, base_context, not_found, prediction_for, recommendations_for,
    sparkline, templates,
)
from strata.web.charts import field_map, season_chart
from strata.web.status import alerts_for, soil_status
from strata.yield_model import YieldModel

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
def overview(request: Request, details: bool = False):
    """The farm, not a field.

    Two things at once, which is what was asked for: a passive summary of every
    field, and an attention list headlining what actually needs doing. The
    attention list deliberately mixes "you should act" with "we cannot tell you",
    because a dashboard listing only actions would quietly imply an unmeasured
    field is a fine one.
    """
    conn = db.init_db()
    fields = db.get_fields(conn)

    cards = []
    alerts = []
    for field in fields:
        lab = db.latest_lab_test(conn, field["id"])
        prediction = prediction_for(field)
        recs = recommendations_for(field, lab, prediction)
        status = soil_status(lab, recs)
        summary = db.reading_summary(conn, field["id"])
        uncalibrated = sum(r["n_uncalibrated"] for r in summary)
        n_readings = sum(r["n"] for r in summary)

        cards.append({
            "field": field, "status": status, "prediction": prediction,
            "lab": lab, "n_readings": n_readings,
            "n_sites": len(db.sites_for_field(conn, field["id"])),
            "last_reading": max((r["last_at"] for r in summary), default=None),
        })
        alerts.extend(alerts_for(field, status, prediction, uncalibrated))
    conn.close()

    # Worst first: a runoff concern outranks a deficiency, which outranks a thing
    # we simply have not measured.
    rank = {"env": 0, "low": 1, "unknown": 2, "slate": 3}
    alerts.sort(key=lambda a: rank.get(a.tone, 9))

    ctx = base_context(request, details)
    ctx.update({"cards": cards, "alerts": alerts, "fields": fields})
    return templates.TemplateResponse(request, "overview.html", ctx)


@router.get("/fields", response_class=HTMLResponse)
def fields_index(request: Request, details: bool = False):
    conn = db.init_db()
    fields = db.get_fields(conn)
    conn.close()
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


@router.get("/field/{field_id}/season", response_class=HTMLResponse)
def season_view(request: Request, field_id: int, channel: Optional[str] = None,
                year: Optional[str] = None, details: bool = False):
    """One channel across every spot in a field, over one growing season.

    This is the farmer-facing view of the thing the whole project argues for: a
    measurement that varies within a field AND within a season. The static county
    soil snapshot has neither axis, which is why the yield model hits a ceiling
    (TECHNICAL_REPORT §7). Here both axes are visible at once -- spots differ, and
    each spot moves through the year.

    The model's own phenology windows are drawn behind the readings, so the grower
    and the yield model are visibly reading the same calendar.
    """
    conn = db.init_db()
    field = db.get_field(conn, field_id)
    if field is None:
        conn.close()
        return not_found(request, details, f"No field with id {field_id}.")

    channels = [r["channel"] for r in db.reading_summary(conn, field_id)]
    years = db.reading_years(conn, field_id)
    if channel is None or channel not in channels:
        channel = "soil_moisture" if "soil_moisture" in channels else (
            channels[0] if channels else "soil_moisture")
    if year is None or year not in years:
        year = years[0] if years else None

    rows = db.season_readings(conn, field_id, channel, year) if year else []
    conn.close()

    # Group by site, preserving first-seen order so a site keeps its colour slot
    # for as long as it has data -- colour follows the entity, never its rank.
    by_site: dict[str, list] = {}
    for r in rows:
        by_site.setdefault(r["site_label"], []).append({
            "taken_at": r["taken_at"],
            # Plot the lab-equivalent value when a calibration exists, the raw
            # reading otherwise, and remember which so the UI can say so.
            "value": r["calibrated_value"] if r["calibrated_value"] is not None
                     else r["raw_value"],
            "uncalibrated": r["calibrated_value"] is None,
            "synthetic": r["source"] == "synthetic",
        })
    series_data = [{"label": k, "points": v} for k, v in by_site.items()]

    cfg = load_source_config()
    daymet = cfg.get("daymet", {})
    chart = season_chart(
        series_data,
        windows=daymet.get("season_windows", []),
        season=(daymet.get("season_start_yday", 91), daymet.get("season_end_yday", 304)),
    )

    meta = CHANNELS.get(channel)
    ctx = base_context(request, details)
    ctx.update({
        "field": field, "channel": channel, "channel_meta": meta,
        "channels": channels, "years": years, "year": year,
        "chart": chart, "rows": rows,
        "any_uncalibrated": any(r["calibrated_value"] is None for r in rows),
        "any_synthetic": any(r["source"] == "synthetic" for r in rows),
    })
    return templates.TemplateResponse(request, "season.html", ctx)


@router.get("/field/{field_id}/map", response_class=HTMLResponse)
def map_view(request: Request, field_id: int, channel: Optional[str] = None,
             details: bool = False):
    """Where the sampling spots actually are, shaded by a chosen measurement.

    A schematic, not a basemap: no tiles, no API key, no script. Coordinates are
    projected to real metres (with the cos-latitude correction), so the spacing is
    true and a surface layer could be added later without redoing the geometry.

    It is explicitly NOT an interpolated surface. Shading three points and
    smoothing between them would be drawing rather than measuring, which is why
    the page states how many more spots a believable gradient would need.
    """
    conn = db.init_db()
    field = db.get_field(conn, field_id)
    if field is None:
        conn.close()
        return not_found(request, details, f"No field with id {field_id}.")

    sites = db.sites_for_field(conn, field_id)
    channels = [r["channel"] for r in db.reading_summary(conn, field_id)]
    if channel is None or channel not in channels:
        channel = "soil_moisture" if "soil_moisture" in channels else (
            channels[0] if channels else None)

    # The most recent reading of this channel at each spot: a map shows the field
    # as it stands now, where the season view shows how it got there.
    latest: dict[int, dict] = {}
    if channel:
        for r in db.season_readings(conn, field_id, channel):
            latest[r["site_id"]] = {
                "value": r["calibrated_value"] if r["calibrated_value"] is not None
                         else r["raw_value"],
                "when": r["taken_at"][:10],
                "uncalibrated": r["calibrated_value"] is None,
            }
    counts = {
        row["site_id"]: row["n"] for row in conn.execute(
            """SELECT s.id AS site_id, COUNT(r.id) AS n FROM site s
                 LEFT JOIN reading r ON r.site_id = s.id
                WHERE s.field_id = ? GROUP BY s.id""", (field_id,)).fetchall()
    }
    conn.close()

    fmap = field_map([{
        "label": s["label"], "lat": s["lat"], "lon": s["lon"],
        "value": latest.get(s["id"], {}).get("value"),
        "reading_count": counts.get(s["id"], 0),
    } for s in sites])

    ctx = base_context(request, details)
    ctx.update({
        "field": field, "map": fmap, "sites": sites,
        "channel": channel, "channels": channels,
        "channel_meta": CHANNELS.get(channel) if channel else None,
        "latest": latest,
        "any_uncalibrated": any(v["uncalibrated"] for v in latest.values()),
    })
    return templates.TemplateResponse(request, "map.html", ctx)
