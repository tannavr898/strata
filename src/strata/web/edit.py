"""edit.py — routes that change state.

Plain HTML forms, POST then redirect (POST/Redirect/GET), no JavaScript. A
refresh after submitting therefore cannot re-submit, and the whole app keeps
working in a field browser on a bad connection.

Validation lives in `forms.py`, where the reasoning behind each rule is written
down. The one to remember: a blank input stays None and never becomes 0.0,
because "not measured" and "measured as zero" lead to opposite recommendations.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Optional
from urllib.parse import quote

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from strata import db
from strata.calibration import ProbeCalibration
from strata.probe import CHANNELS
from strata.web import forms
from strata.web.core import (
    base_context, modelled_crops, not_found, pa_counties, templates,
)

logger = logging.getLogger("strata.web.edit")

router = APIRouter()


def _back(path: str, details: bool, **params) -> RedirectResponse:
    """Redirect after a POST, preserving the details toggle.

    303 specifically: it turns the POST into a GET, so the browser's reload
    button cannot resubmit the form.
    """
    query = {k: str(v) for k, v in params.items() if v is not None}
    if details:
        query["details"] = "true"
    suffix = "?" + "&".join(f"{k}={quote(v)}" for k, v in query.items()) if query else ""
    return RedirectResponse(path + suffix, status_code=303)


# --------------------------------------------------------------------------- #
# Soil test — the form that unlocks quantified P/K and lime advice
# --------------------------------------------------------------------------- #
@router.get("/field/{field_id}/soil-test", response_class=HTMLResponse)
def soil_test_form(request: Request, field_id: int, details: bool = False,
                   error: Optional[str] = None):
    conn = db.init_db()
    field = db.get_field(conn, field_id)
    previous = db.latest_lab_test(conn, field_id) if field else None
    conn.close()
    if field is None:
        return not_found(request, details, f"No field with id {field_id}.")

    ctx = base_context(request, details)
    ctx.update({"field": field, "previous": previous, "error": error,
                "measurements": forms.SOIL_TEST_FIELDS,
                "today": date.today().isoformat()})
    return templates.TemplateResponse(request, "soil_test.html", ctx)


@router.post("/field/{field_id}/soil-test")
async def soil_test_submit(request: Request, field_id: int, details: bool = False):
    form = await request.form()
    try:
        values = forms.parse_soil_test(form)
    except forms.ValidationError as exc:
        return _back(f"/field/{field_id}/soil-test", details, error=str(exc))

    conn = db.init_db()
    if db.get_field(conn, field_id) is None:
        conn.close()
        return not_found(request, details, f"No field with id {field_id}.")
    with conn:
        conn.execute(
            """INSERT INTO lab_test (field_id, sampled_at, lab_name, ph, buffer_ph,
                                     mehlich3_p_ppm, mehlich3_k_ppm, cec, om_pct,
                                     yield_goal, notes)
               VALUES (:field_id, :sampled_at, :lab_name, :ph, :buffer_ph,
                       :mehlich3_p_ppm, :mehlich3_k_ppm, :cec, :om_pct,
                       :yield_goal, :notes)""",
            {"field_id": field_id, **values},
        )
    conn.close()
    logger.info("Recorded soil test for field %d sampled %s",
                field_id, values["sampled_at"])
    return _back(f"/field/{field_id}", details)


# --------------------------------------------------------------------------- #
# Reading import — the path that works before the device firmware does
# --------------------------------------------------------------------------- #
@router.get("/field/{field_id}/import", response_class=HTMLResponse)
def import_form(request: Request, field_id: int, details: bool = False,
                error: Optional[str] = None, added: Optional[int] = None,
                skipped: Optional[int] = None, unknown: Optional[str] = None):
    conn = db.init_db()
    field = db.get_field(conn, field_id)
    sites = db.sites_for_field(conn, field_id) if field else []
    conn.close()
    if field is None:
        return not_found(request, details, f"No field with id {field_id}.")

    ctx = base_context(request, details)
    ctx.update({"field": field, "sites": sites, "error": error,
                "added": added, "skipped": skipped, "unknown": unknown,
                "channels": CHANNELS})
    return templates.TemplateResponse(request, "import.html", ctx)


@router.post("/field/{field_id}/import")
async def import_submit(request: Request, field_id: int, details: bool = False):
    form = await request.form()

    # An uploaded file wins over pasted text when both are present, because
    # choosing a file is the more deliberate action.
    text = str(form.get("csv_text") or "")
    upload = form.get("csv_file")
    if hasattr(upload, "read"):
        raw = await upload.read()
        if raw:
            text = raw.decode("utf-8", errors="replace")

    default_site = str(form.get("default_site") or "").strip() or None
    try:
        parsed = forms.parse_readings_csv(text, default_site=default_site)
    except forms.ValidationError as exc:
        return _back(f"/field/{field_id}/import", details, error=str(exc))

    conn = db.init_db()
    if db.get_field(conn, field_id) is None:
        conn.close()
        return not_found(request, details, f"No field with id {field_id}.")

    # Sites are created on demand: a device dump names its own spots, and making
    # someone pre-register every label before importing would just mean
    # abandoning the import halfway through.
    sites = {s["label"].strip().lower(): s["id"] for s in db.sites_for_field(conn, field_id)}
    added = 0
    with conn:
        for label, taken_at, channel, value in parsed.rows:
            key = label.strip().lower()
            if key not in sites:
                sites[key] = conn.execute(
                    "INSERT INTO site (field_id, label) VALUES (?, ?)",
                    (field_id, label.strip()),
                ).lastrowid
            # INSERT OR IGNORE against the (site, time, channel) unique index, so
            # re-importing the same file is idempotent instead of duplicating a
            # whole season of readings.
            added += conn.execute(
                """INSERT OR IGNORE INTO reading
                       (site_id, taken_at, channel, raw_value, source)
                   VALUES (?, ?, ?, ?, 'import')""",
                (sites[key], taken_at, channel, value),
            ).rowcount
    conn.close()

    logger.info("Imported %d readings into field %d (%d rows skipped)",
                added, field_id, len(parsed.skipped))
    return _back(f"/field/{field_id}/import", details,
                 added=added, skipped=len(parsed.skipped),
                 unknown=", ".join(sorted(parsed.unknown_channels)) or None)


# --------------------------------------------------------------------------- #
# Calibration — the probe->lab seam, with a UI
# --------------------------------------------------------------------------- #
@router.get("/calibration", response_class=HTMLResponse)
def calibration_view(request: Request, details: bool = False,
                     error: Optional[str] = None, fitted: Optional[str] = None):
    conn = db.init_db()
    fits = conn.execute(
        "SELECT * FROM calibration ORDER BY fitted_at DESC"
    ).fetchall()
    uncalibrated = conn.execute(
        """SELECT channel, COUNT(*) AS n FROM reading
            WHERE calibrated_value IS NULL GROUP BY channel ORDER BY channel"""
    ).fetchall()
    conn.close()

    ctx = base_context(request, details)
    ctx.update({
        "fits": fits,
        "uncalibrated": uncalibrated,
        "error": error,
        "fitted": fitted,
        # Only channels with a lab scale can be calibrated ONTO one. Moisture,
        # temperature, nitrogen and CO2 have no laboratory equivalent (see
        # probe.py), so offering a calibration box for them would invite someone
        # to invent a mapping to a scale that does not exist.
        "calibratable": {n: c for n, c in CHANNELS.items() if c.lab_scale},
        "not_calibratable": {n: c for n, c in CHANNELS.items() if not c.lab_scale},
    })
    return templates.TemplateResponse(request, "calibration.html", ctx)


@router.post("/calibration")
async def calibration_submit(request: Request, details: bool = False):
    form = await request.form()
    channel = str(form.get("channel") or "").strip()

    meta = CHANNELS.get(channel)
    if meta is None or not meta.lab_scale:
        return _back("/calibration", details,
                     error=f"{channel!r} has no laboratory scale to calibrate onto.")

    try:
        probe, lab = forms.parse_pairs(str(form.get("pairs") or ""))
        fit = ProbeCalibration().fit(channel, probe=probe, lab=lab,
                                     units=meta.units, synthetic=False)
    except (forms.ValidationError, ValueError) as exc:
        return _back("/calibration", details, error=str(exc))

    conn = db.init_db()
    with conn:
        calibration_id = conn.execute(
            """INSERT INTO calibration (channel, slope, intercept, r2, rmse,
                                        mean_bias, n_pairs, synthetic, notes)
               VALUES (?,?,?,?,?,?,?,0,?)""",
            (channel, fit.slope, fit.intercept, fit.r2, fit.rmse, fit.mean_bias,
             fit.n, forms.clean_text(form.get("notes"))),
        ).lastrowid
        # Apply to readings of this channel that have no calibrated value yet.
        # Already-calibrated readings are left alone: silently re-deriving them
        # under a new fit would rewrite numbers a farmer may already have acted
        # on, with no record that they changed.
        applied = conn.execute(
            """UPDATE reading
                  SET calibrated_value = ? * raw_value + ?, calibration_id = ?
                WHERE channel = ? AND calibrated_value IS NULL""",
            (fit.slope, fit.intercept, calibration_id, channel),
        ).rowcount
    conn.close()

    logger.info("Calibrated %s (R2 %.3f, n=%d); applied to %d readings",
                channel, fit.r2, fit.n, applied)
    return _back("/calibration", details,
                 fitted=f"{channel}: R2 {fit.r2:.3f} from {fit.n} pairs, "
                        f"applied to {applied} readings.")


# --------------------------------------------------------------------------- #
# Fields and sites — so the app can be used without writing SQL
# --------------------------------------------------------------------------- #
@router.get("/fields/new", response_class=HTMLResponse)
def field_new_form(request: Request, details: bool = False,
                   error: Optional[str] = None):
    ctx = base_context(request, details)
    ctx.update({"field": None, "counties": pa_counties(),
                "crops": modelled_crops(), "error": error})
    return templates.TemplateResponse(request, "field_form.html", ctx)


@router.get("/field/{field_id}/edit", response_class=HTMLResponse)
def field_edit_form(request: Request, field_id: int, details: bool = False,
                    error: Optional[str] = None):
    conn = db.init_db()
    field = db.get_field(conn, field_id)
    conn.close()
    if field is None:
        return not_found(request, details, f"No field with id {field_id}.")

    ctx = base_context(request, details)
    ctx.update({"field": field, "counties": pa_counties(),
                "crops": modelled_crops(), "error": error})
    return templates.TemplateResponse(request, "field_form.html", ctx)


@router.post("/fields/new")
async def field_create(request: Request, details: bool = False):
    form = await request.form()
    try:
        values = forms.parse_field(form, pa_counties(), modelled_crops())
    except forms.ValidationError as exc:
        return _back("/fields/new", details, error=str(exc))

    conn = db.init_db()
    with conn:
        field_id = conn.execute(
            """INSERT INTO field (name, county_ansi, county_name, crop, acres, notes)
               VALUES (:name, :county_ansi, :county_name, :crop, :acres, :notes)""",
            values,
        ).lastrowid
    conn.close()
    logger.info("Created field %d (%s, %s)", field_id, values["name"], values["crop"])
    return _back(f"/field/{field_id}", details)


@router.post("/field/{field_id}/edit")
async def field_update(request: Request, field_id: int, details: bool = False):
    form = await request.form()
    try:
        values = forms.parse_field(form, pa_counties(), modelled_crops())
    except forms.ValidationError as exc:
        return _back(f"/field/{field_id}/edit", details, error=str(exc))

    conn = db.init_db()
    if db.get_field(conn, field_id) is None:
        conn.close()
        return not_found(request, details, f"No field with id {field_id}.")
    with conn:
        conn.execute(
            """UPDATE field SET name = :name, county_ansi = :county_ansi,
                                county_name = :county_name, crop = :crop,
                                acres = :acres, notes = :notes
                WHERE id = :id""",
            {**values, "id": field_id},
        )
    conn.close()
    return _back(f"/field/{field_id}", details)


@router.post("/field/{field_id}/sites")
async def site_create(request: Request, field_id: int, details: bool = False):
    form = await request.form()
    try:
        values = forms.parse_site(form)
    except forms.ValidationError as exc:
        return _back(f"/field/{field_id}", details, error=str(exc))

    conn = db.init_db()
    if db.get_field(conn, field_id) is None:
        conn.close()
        return not_found(request, details, f"No field with id {field_id}.")
    with conn:
        conn.execute(
            """INSERT INTO site (field_id, label, lat, lon, notes)
               VALUES (:field_id, :label, :lat, :lon, :notes)""",
            {"field_id": field_id, **values},
        )
    conn.close()
    return _back(f"/field/{field_id}", details)
