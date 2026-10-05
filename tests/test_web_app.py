"""The app layer: routing, and the honesty states surviving contact with a UI.

The interesting tests here are not "does the page load". They are: does a withheld
crop stay withheld once there is a template that could just as easily print a
number, and does the county-scale caveat still ride along with the prediction.
A UI is where carefully-built refusals quietly turn into confident answers.

Every test runs against a temporary database, so it never touches the developer's
data/strata.db.
"""

from __future__ import annotations

import html
import re

import pytest

from strata import db

fastapi = pytest.importorskip("fastapi", reason="web extras not installed")
from fastapi.testclient import TestClient  # noqa: E402

from strata.web import app as webapp  # noqa: E402


def visible_text(markup: str) -> str:
    """The text a reader actually sees, tags and entities resolved."""
    markup = re.sub(r"<(script|style|svg).*?</\1>", " ", markup, flags=re.S)
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", markup))).strip()


@pytest.fixture
def client(tmp_path, monkeypatch):
    """A client backed by a throwaway DB holding both demo fields."""
    test_db = tmp_path / "test.db"
    conn = db.init_db(test_db)
    with conn:
        corn = conn.execute(
            """INSERT INTO field (name, county_ansi, county_name, crop, acres)
               VALUES ('North Field', '029', 'Chester', 'corn_grain_yield', 42.0)"""
        ).lastrowid
        conn.execute(
            """INSERT INTO lab_test (field_id, sampled_at, ph, buffer_ph,
                                     mehlich3_p_ppm, mehlich3_k_ppm, cec, om_pct)
               VALUES (?, '2026-04-12', 5.72, 6.50, 22.0, 85.0, 10.5, 2.4)""",
            (corn,),
        )
        site = conn.execute(
            "INSERT INTO site (field_id, label, lat, lon) VALUES (?,'Upper',39.96,-75.63)",
            (corn,),
        ).lastrowid
        conn.executemany(
            """INSERT INTO reading (site_id, taken_at, channel, raw_value, source)
               VALUES (?, ?, 'soil_moisture', ?, 'synthetic')""",
            [(site, f"2026-05-{d:02d}T09:00:00", 20.0 + d * 0.4) for d in (4, 11, 18, 25)],
        )
        # A second spot, so the season view is tested with more than one series:
        # legend, distinct colour slots and direct labels only exist above one.
        swale = conn.execute(
            "INSERT INTO site (field_id, label, lat, lon) VALUES (?,'Low swale',39.96,-75.62)",
            (corn,),
        ).lastrowid
        conn.executemany(
            """INSERT INTO reading (site_id, taken_at, channel, raw_value, source)
               VALUES (?, ?, 'soil_moisture', ?, 'synthetic')""",
            [(swale, f"2026-05-{d:02d}T09:15:00", 27.0 + d * 0.3) for d in (4, 11, 18, 25)],
        )
        # No lab test on this one, and oats is below the product gate.
        conn.execute(
            """INSERT INTO field (name, county_ansi, county_name, crop, acres)
               VALUES ('Back Lot', '029', 'Chester', 'oats_yield', 11.0)"""
        )
    conn.close()

    # Patch db.init_db itself rather than a route module's reference to it. Every
    # route calls db.init_db() with no argument, so redirecting the default here
    # covers views, edit, and anything added later -- and cannot go stale when the
    # routes are reorganised, which patching per-module references already did once.
    real_init = db.init_db
    monkeypatch.setattr(db, "DB_PATH", test_db)
    monkeypatch.setattr(db, "init_db", lambda *a, **k: real_init(test_db))

    with TestClient(webapp.app) as c:
        yield c


# --------------------------------------------------------------------------- #
# Routing
# --------------------------------------------------------------------------- #
def test_every_page_renders(client):
    for url in ["/", "/models", "/field/1", "/field/2", "/field/1/site/1"]:
        assert client.get(url).status_code == 200, f"{url} did not render"


def test_missing_field_is_404_not_a_crash(client):
    assert client.get("/field/9999").status_code == 404


def test_unknown_channel_falls_back_to_a_real_one(client):
    """An unrecognised ?channel= used to render an empty plot captioned with a
    measurement the site had no data for. It now falls back to a channel the site
    actually has, so a stale bookmark or a typo shows real data instead of a page
    that looks broken."""
    r = client.get("/field/1/site/1?channel=not_a_channel")
    assert r.status_code == 200
    page = visible_text(r.text)
    assert "Not enough readings" not in page
    assert "SEN0193" in page, "did not fall back to the site's soil-moisture data"


# --------------------------------------------------------------------------- #
# The honesty states
# --------------------------------------------------------------------------- #
def test_withheld_crop_shows_no_number(client):
    """Oats is below the gate. The page must explain, not estimate."""
    page = visible_text(client.get("/field/2").text)
    assert "No estimate for this crop" in page
    assert "bu / acre" not in page.split("What to add")[0], (
        "a yield figure appeared for a crop the product gate withholds"
    )


def test_field_without_lab_test_refuses_to_advise(client):
    page = visible_text(client.get("/field/2").text)
    assert "No soil test on file" in page
    for word in ("Apply lime", "lb P2O5", "lb K2O"):
        assert word not in page, f"advised {word!r} with no soil test on file"


def test_prediction_carries_the_county_scale_caveat(client):
    """The easiest caveat to lose in a redesign, and the one that matters most:
    a number under a field's name is implicitly a field-level claim (METHODS 12.2)."""
    page = visible_text(client.get("/field/1").text)
    assert "county-scale estimate" in page
    assert "Your field can differ from the county" in page


def test_shipped_crop_shows_a_number_with_its_error(client):
    page = visible_text(client.get("/field/1").text)
    assert "bu / acre" in page
    assert "Typically within" in page, "a point estimate was shown with no error range"


def test_synthetic_readings_are_marked(client):
    page = visible_text(client.get("/field/1/site/1").text)
    assert "synthetic data" in page


def test_uncalibrated_readings_are_marked(client):
    """Raw sensor values with no probe->lab fit must not look lab-comparable."""
    page = visible_text(client.get("/field/1").text)
    assert "uncalibrated" in page


def test_model_card_shows_withheld_crops_too(client):
    """A surface listing only what it ships would hide the most interesting
    property of the system: that it declines when it cannot answer."""
    page = visible_text(client.get("/models").text)
    assert "withheld" in page
    assert "oats" in page


# --------------------------------------------------------------------------- #
# Display layer
# --------------------------------------------------------------------------- #
def test_internal_crop_keys_never_reach_the_page(client):
    for url in ["/field/1", "/field/2", "/models"]:
        page = visible_text(client.get(url).text)
        leaked = [k for k in webapp.CROP_LABELS if k in page]
        assert not leaked, f"{url} showed raw NASS crop keys: {leaked}"


def test_crop_label_prefers_the_longest_key():
    """'hay_yield' is a substring of 'alfalfa_hay_yield'; replacing the short key
    first would render alfalfa as 'alfalfa_hay' and never match the long one."""
    assert webapp.crop_label("alfalfa_hay_yield") == "alfalfa"
    assert webapp.crop_label("hay_yield") == "hay"
    assert webapp.crop_label("corn_silage_yield") == "corn silage"


def test_provenance_footer_names_the_build(client):
    """Every page says which models produced it and when, so a stale build is
    visible rather than silently trusted."""
    page = visible_text(client.get("/field/1").text)
    # The build stamp now appears twice: compactly in the sidebar foot, and in
    # full in the page footer. Both must name the build, so a stale one is
    # visible wherever you happen to be looking.
    assert "gate" in page.lower()
    assert "county-year rows" in page, "the footer no longer names the training data"
    assert "table " in page, "the footer no longer names the analysis table hash"


# --------------------------------------------------------------------------- #
# Season view
# --------------------------------------------------------------------------- #
def test_season_view_renders_with_a_legend_and_a_table(client):
    """The categorical series palette fails the 3:1 contrast check in light mode,
    which obliges both visible labels and a table view — identity can never rest
    on colour alone."""
    r = client.get("/field/1/season")
    assert r.status_code == 200
    assert 'class="legend"' in r.text, "no legend above a single series"
    assert "serieslabel" in r.text, "series are not directly labelled"
    assert 'class="card tableview"' in r.text, "no table view beside the chart"


def test_season_view_draws_the_models_own_phenology_windows(client):
    """The bands are the yield model's sub-season feature windows, so the grower
    and the model are visibly reading the same calendar."""
    markup = client.get("/field/1/season").text
    for band in ("Establishment", "Vegetative", "Silking", "Grain fill"):
        assert f">{band}</text>" in markup, f"the {band} window is not drawn"
    assert 'class="band band-jul"' in markup, "the silking window is not distinguished"


def test_season_view_marks_synthetic_and_uncalibrated(client):
    page = visible_text(client.get("/field/1/season").text)
    assert "synthetic data" in page
    assert "uncalibrated" in page


def test_season_view_falls_back_for_an_unknown_channel(client):
    r = client.get("/field/1/season?channel=not_a_channel")
    assert r.status_code == 200
    assert "Not enough readings" not in visible_text(r.text)


def test_season_view_on_a_field_with_no_readings(client):
    r = client.get("/field/2/season")
    assert r.status_code == 200
    assert "No readings yet" in visible_text(r.text)
