"""app.py — the Strata web app.

    python scripts/run_app.py          # http://127.0.0.1:8000

Farmer-facing, with a details toggle that doubles as the prototyping view. It
reads three things and computes nothing expensive:

    data/models/     fitted models + honest metrics     (scripts/build_models.py)
    data/strata.db   fields, sites, readings, lab tests (scripts/seed_db.py)
    data/processed/  the county-year analysis table

Models load ONCE at startup, never per request -- cross-validating a crop takes
minutes, which is why `artifacts.py` exists at all.

This module is assembly only. The pieces:

    core.py    shared state, templates, and the domain helpers
    views.py   read routes
    edit.py    write routes (forms, POST/redirect)
    forms.py   input validation, and the reasoning behind each rule
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from strata.web import edit, views
from strata.web.core import (  # noqa: F401  (re-exported for tests and callers)
    CROP_LABELS, HERE, STATE, crop_label, load_state, templates,
)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    load_state()
    yield


app = FastAPI(title="Strata", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
app.include_router(views.router)
app.include_router(edit.router)
