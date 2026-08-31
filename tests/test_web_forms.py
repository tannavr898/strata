"""Input validation, and the write routes that use it.

The single most important test in this file is
`test_blank_measurement_is_missing_not_zero`. Everything the recommender promises
about refusing to guess collapses if a form helpfully turns an empty phosphorus
box into 0.0 — that is not missing data, it is a severely deficient soil, and the
engine would confidently recommend a large phosphate application on a number
nobody measured.
"""

from __future__ import annotations

import html
import re

import pytest

from strata import db

pytest.importorskip("fastapi", reason="web extras not installed")
from fastapi.testclient import TestClient  # noqa: E402

from strata.web import forms  # noqa: E402
from strata.web import app as webapp  # noqa: E402


def visible_text(markup: str) -> str:
    markup = re.sub(r"<(script|style|svg).*?</\1>", " ", markup, flags=re.S)
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", markup))).strip()


@pytest.fixture
def client(tmp_path, monkeypatch):
    test_db = tmp_path / "forms.db"
    conn = db.init_db(test_db)
    with conn:
        conn.execute(
            """INSERT INTO field (name, county_ansi, county_name, crop, acres)
               VALUES ('North Field', '029', 'Chester', 'corn_grain_yield', 42.0)"""
        )
    conn.close()

    real_init = db.init_db
    monkeypatch.setattr(db, "DB_PATH", test_db)
    monkeypatch.setattr(db, "init_db", lambda *a, **k: real_init(test_db))
    with TestClient(webapp.app) as c:
        c.test_db = test_db
        yield c


def _rows(client, sql, params=()):
    conn = db.init_db(client.test_db)
    out = conn.execute(sql, params).fetchall()
    conn.close()
    return out


# --------------------------------------------------------------------------- #
# optional_float — the honesty rule
# --------------------------------------------------------------------------- #
def test_blank_measurement_is_missing_not_zero():
    """An empty box means "not measured", which the engine answers with 'obtain a
    soil test'. Coercing it to 0.0 would instead assert a severely deficient soil
    and produce a confident fertilizer rate from a number nobody measured."""
    assert forms.optional_float("", "P") is None
    assert forms.optional_float("   ", "P") is None
    assert forms.optional_float(None, "P") is None
    # A REAL zero is still a real measurement and must survive.
    assert forms.optional_float("0", "P") == 0.0


def test_non_numeric_input_is_rejected_with_the_field_name():
    with pytest.raises(forms.ValidationError, match="Soil pH"):
        forms.optional_float("about six", "Soil pH")


def test_impossible_values_are_rejected():
    """Ranges reject the impossible, not the unusual: a pH of 87 is a typo or a
    unit error, and storing it would propagate a wrong lime rate to a real field."""
    with pytest.raises(forms.ValidationError, match="above"):
        forms.optional_float("87", "Soil pH", 2.0, 11.0)
    with pytest.raises(forms.ValidationError, match="below"):
        forms.optional_float("-3", "Mehlich-3 P", 0.0, 1000.0)
    # Unusual but possible soils must still go through -- they need advice most.
    assert forms.optional_float("3.9", "Soil pH", 2.0, 11.0) == 3.9


def test_soil_test_requires_a_date():
    with pytest.raises(forms.ValidationError, match="required"):
        forms.parse_soil_test({"ph": "6.2"})


def test_soil_test_requires_at_least_one_measurement():
    with pytest.raises(forms.ValidationError, match="at least one measurement"):
        forms.parse_soil_test({"sampled_at": "2026-04-12", "lab_name": "AASL"})


# --------------------------------------------------------------------------- #
# CSV parsing
# --------------------------------------------------------------------------- #
WIDE = ("site,taken_at,soil_moisture,soil_temp\n"
        "Upper bench,2026-06-01 09:00,24.1,17.2\n"
        "Low swale,2026-06-01 09:15,29.6,16.8\n")

LONG = ("site,taken_at,channel,raw_value\n"
        "Upper bench,2026-06-01 09:00,soil_moisture,24.1\n"
        "Upper bench,2026-06-01 09:00,soil_temp,17.2\n")


def test_wide_and_long_csv_produce_the_same_readings():
    """Both shapes are plausible exports, so the parser detects rather than
    demands one. They must agree, or the shape silently changes the data."""
    wide = forms.parse_readings_csv(WIDE)
    long_ = forms.parse_readings_csv(LONG)
    assert wide.n_rows == 4
    assert set(long_.rows) <= set(wide.rows)


def test_unknown_channels_are_reported_not_guessed():
    """probe.CHANNELS is the registry of what this hardware measures. A column
    outside it is a firmware change or a typo -- both worth surfacing."""
    csv = ("site,taken_at,channel,raw_value\n"
           "A,2026-06-01,soil_moisture,20\n"
           "A,2026-06-01,electrical_conductivity,0.4\n")
    result = forms.parse_readings_csv(csv)
    assert result.unknown_channels == {"electrical_conductivity"}
    assert result.n_rows == 1
    assert any("electrical_conductivity" in s for s in result.skipped)


def test_unreadable_rows_are_counted_not_silently_dropped():
    csv = ("site,taken_at,channel,raw_value\n"
           "A,2026-06-01,soil_moisture,20\n"
           "A,not-a-date,soil_temp,18\n"
           "A,2026-06-02,soil_temp,not-a-number\n")
    result = forms.parse_readings_csv(csv)
    assert result.n_rows == 1
    assert len(result.skipped) == 2


def test_missing_timestamp_column_is_a_clear_error():
    with pytest.raises(forms.ValidationError, match="timestamp"):
        forms.parse_readings_csv("site,soil_moisture\nA,20\n")


def test_default_site_covers_a_file_with_no_site_column():
    result = forms.parse_readings_csv(
        "taken_at,soil_moisture\n2026-06-01 09:00,24.1\n", default_site="Upper bench")
    assert result.rows[0][0] == "Upper bench"


def test_csv_with_no_site_and_no_default_is_refused():
    with pytest.raises(forms.ValidationError, match="site"):
        forms.parse_readings_csv("taken_at,soil_moisture\n2026-06-01 09:00,24.1\n")


# --------------------------------------------------------------------------- #
# Calibration pairs
# --------------------------------------------------------------------------- #
def test_two_pairs_are_refused():
    """Two points fit a line perfectly and report R² = 1.0, which would look like
    an excellent calibration while saying nothing at all."""
    with pytest.raises(forms.ValidationError, match="at least 3"):
        forms.parse_pairs("5.4,5.6\n6.0,6.2")


def test_pairs_tolerate_a_header_row():
    probe, lab = forms.parse_pairs("probe,lab\n5.4,5.6\n6.0,6.2\n6.5,6.7")
    assert len(probe) == 3 and len(lab) == 3


# --------------------------------------------------------------------------- #
# Write routes
# --------------------------------------------------------------------------- #
def test_soil_test_submission_unlocks_cited_advice(client):
    before = visible_text(client.get("/field/1").text)
    assert "No soil test on file" in before

    r = client.post("/field/1/soil-test", data={
        "sampled_at": "2026-04-12", "lab_name": "Penn State AASL",
        "ph": "5.72", "buffer_ph": "6.50",
        "mehlich3_p_ppm": "22", "mehlich3_k_ppm": "85",
    }, follow_redirects=False)
    assert r.status_code == 303, "POST must redirect so a refresh cannot resubmit"

    after = visible_text(client.get("/field/1").text)
    assert "lb P2O5" in after
    assert "tons ag lime" in after


def test_blank_pk_still_refuses_after_submission(client):
    """The end-to-end version of the rule: a form submitted with P and K left
    empty must produce a refusal, not a zero-based deficiency."""
    client.post("/field/1/soil-test",
                data={"sampled_at": "2026-04-12", "ph": "6.10"})
    page = visible_text(client.get("/field/1").text)
    assert "not provided" in page
    assert "lb P2O5" not in page, "advised phosphate from an unmeasured value"

    stored = _rows(client, "SELECT mehlich3_p_ppm FROM lab_test")[0]
    assert stored["mehlich3_p_ppm"] is None, "blank P was stored as a number"


def test_invalid_submission_returns_the_form_with_the_reason(client):
    r = client.post("/field/1/soil-test",
                    data={"sampled_at": "2026-04-12", "ph": "87"},
                    follow_redirects=True)
    assert "plausible maximum" in visible_text(r.text)
    assert not _rows(client, "SELECT * FROM lab_test"), "invalid test was stored"


def test_import_is_idempotent(client):
    first = client.post("/field/1/import", data={"csv_text": WIDE},
                        follow_redirects=False)
    assert "added=4" in first.headers["location"]

    again = client.post("/field/1/import", data={"csv_text": WIDE},
                        follow_redirects=False)
    assert "added=0" in again.headers["location"], (
        "re-importing the same file duplicated readings"
    )
    assert len(_rows(client, "SELECT * FROM reading")) == 4


def test_import_creates_sites_named_in_the_file(client):
    client.post("/field/1/import", data={"csv_text": WIDE})
    labels = {r["label"] for r in _rows(client, "SELECT label FROM site")}
    assert labels == {"Upper bench", "Low swale"}


def test_calibration_applies_to_existing_readings(client):
    client.post("/field/1/import", data={
        "csv_text": "site,taken_at,channel,raw_value\n"
                    "A,2026-06-01,ph,5.61\nA,2026-06-08,ph,5.58\n"})
    assert all(r["calibrated_value"] is None
               for r in _rows(client, "SELECT * FROM reading"))

    client.post("/calibration", data={
        "channel": "ph", "pairs": "5.61,5.72\n5.58,5.69\n5.70,5.81"})

    rows = _rows(client, "SELECT * FROM reading")
    assert all(r["calibrated_value"] is not None for r in rows)
    assert all(r["calibration_id"] is not None for r in rows)


def test_calibration_never_overwrites_an_existing_calibrated_value(client):
    """Re-deriving numbers a farmer may already have acted on, with no record
    that they changed, is worse than leaving an older fit in place."""
    client.post("/field/1/import", data={
        "csv_text": "site,taken_at,channel,raw_value\nA,2026-06-01,ph,5.61\n"})
    client.post("/calibration", data={
        "channel": "ph", "pairs": "5.61,5.72\n5.58,5.69\n5.70,5.81"})
    first = _rows(client, "SELECT calibrated_value FROM reading")[0][0]

    client.post("/calibration", data={
        "channel": "ph", "pairs": "5.61,9.90\n5.58,9.80\n5.70,9.99"})
    second = _rows(client, "SELECT calibrated_value FROM reading")[0][0]
    assert first == second


def test_channel_without_a_lab_scale_cannot_be_calibrated(client):
    """CO2 has no AASL equivalent to map onto. Accepting a fit for it would
    manufacture a laboratory scale that does not exist."""
    r = client.post("/calibration", data={
        "channel": "co2", "pairs": "800,810\n900,915\n1000,1010"},
        follow_redirects=True)
    assert "no laboratory scale" in visible_text(r.text)
    assert not _rows(client, "SELECT * FROM calibration")


def test_calibration_page_explains_what_cannot_be_calibrated(client):
    page = visible_text(client.get("/calibration").text)
    assert "What cannot be calibrated" in page
    for channel in ("co2", "nitrogen", "soil moisture"):
        assert channel in page


def test_site_page_defaults_to_a_channel_that_site_actually_has(client):
    """A hardcoded default showed an empty plot and then described soil moisture
    on a site that only ever recorded pH -- an empty state that looked like a bug
    and named the wrong measurement."""
    client.post("/field/1/import", data={
        "csv_text": "site,taken_at,channel,raw_value\n"
                    "A,2026-06-01,ph,5.61\nA,2026-06-08,ph,5.58\nA,2026-06-15,ph,5.63\n"})
    site_id = _rows(client, "SELECT id FROM site")[0]["id"]

    page = visible_text(client.get(f"/field/1/site/{site_id}").text)
    assert "Not enough readings" not in page
    assert "pH probe" in page, "did not fall back to the channel this site has"
    assert "SEN0193" not in page, "described a measurement the site has no data for"


def test_site_page_honours_an_explicit_channel(client):
    client.post("/field/1/import", data={"csv_text": WIDE})
    site_id = _rows(client, "SELECT id FROM site ORDER BY id")[0]["id"]
    page = visible_text(client.get(f"/field/1/site/{site_id}?channel=soil_temp").text)
    assert "DS18B20" in page
