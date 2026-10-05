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


# --------------------------------------------------------------------------- #
# Field and site management
# --------------------------------------------------------------------------- #
def test_create_a_field_from_the_ui(client):
    r = client.post("/fields/new", data={
        "name": "South Forty", "county_ansi": "029",
        "crop": "soybean_yield", "acres": "38"}, follow_redirects=False)
    assert r.status_code == 303
    names = {f["name"] for f in _rows(client, "SELECT name FROM field")}
    assert "South Forty" in names


def test_field_county_is_validated_against_real_counties(client):
    """An unrecognised county would produce a field with no soil or weather data
    behind it, which surfaces later as a puzzling empty state instead of an error
    at the point of entry."""
    r = client.post("/fields/new", data={
        "name": "Nowhere", "county_ansi": "999", "crop": "soybean_yield"},
        follow_redirects=True)
    assert "Pennsylvania county" in visible_text(r.text)
    assert len(_rows(client, "SELECT * FROM field")) == 1, "invalid field was created"


def test_field_crop_is_validated_against_built_models(client):
    r = client.post("/fields/new", data={
        "name": "Rye Patch", "county_ansi": "029", "crop": "rye_yield"},
        follow_redirects=True)
    assert "not a crop Strata models" in visible_text(r.text)


def test_withheld_crops_are_still_offered_when_adding_a_field(client):
    """Someone who grows oats must be able to add their oat field and see the
    honest 'we can't forecast this' answer, plus the fertilizer advice that does
    not depend on a yield model at all. Hiding the crop would be the wrong
    kindness -- it would look like Strata does not know what oats are."""
    page = visible_text(client.get("/fields/new").text)
    assert "oats" in page


def test_edit_a_field(client):
    client.post("/field/1/edit", data={
        "name": "North Field (renamed)", "county_ansi": "029",
        "crop": "corn_grain_yield", "acres": "44"})
    field = _rows(client, "SELECT * FROM field WHERE id = 1")[0]
    assert field["name"] == "North Field (renamed)"
    assert field["acres"] == 44.0


def test_add_a_site_with_coordinates(client):
    client.post("/field/1/sites", data={
        "label": "Upper bench", "lat": "39.964", "lon": "-75.63"})
    site = _rows(client, "SELECT * FROM site")[0]
    assert site["label"] == "Upper bench"
    assert site["lat"] == pytest.approx(39.964)


def test_a_site_may_omit_coordinates(client):
    """Someone adding spots at the kitchen table does not have them yet, and
    refusing over a missing lat/lon just sends them back to a spreadsheet."""
    client.post("/field/1/sites", data={"label": "Back corner"})
    site = _rows(client, "SELECT * FROM site")[0]
    assert site["lat"] is None


def test_site_validation_error_surfaces_on_the_field_page(client):
    r = client.post("/field/1/sites", data={"label": ""}, follow_redirects=True)
    assert "Give the spot a name" in visible_text(r.text)
    assert not _rows(client, "SELECT * FROM site")


# --------------------------------------------------------------------------- #
# Deletion
#
# These are the paths that destroy data, so the assertions are mostly about
# what must SURVIVE: a scoped delete that takes more than it was asked for is
# far worse than one that fails.
# --------------------------------------------------------------------------- #
def test_deleting_one_channel_leaves_the_others(client):
    client.post("/field/1/import", data={"csv_text": WIDE})
    assert len(_rows(client, "SELECT * FROM reading")) == 4

    client.post("/field/1/readings/delete", data={"channel": "soil_temp"})
    left = _rows(client, "SELECT * FROM reading")
    assert len(left) == 2
    assert {r["channel"] for r in left} == {"soil_moisture"}


def test_deleting_one_day_leaves_the_others(client):
    client.post("/field/1/import", data={
        "csv_text": "site,taken_at,channel,raw_value\n"
                    "A,2026-06-01,soil_moisture,20\n"
                    "A,2026-06-08,soil_moisture,21\n"})
    client.post("/field/1/readings/delete", data={"day": "2026-06-01"})
    left = _rows(client, "SELECT * FROM reading")
    assert len(left) == 1
    assert left[0]["taken_at"].startswith("2026-06-08")


def test_a_scoped_delete_never_touches_another_field(client):
    """The SQL joins through `site` to scope by field. A missing clause there
    would silently delete another field's season."""
    client.post("/fields/new", data={"name": "Other", "county_ansi": "029",
                                     "crop": "soybean_yield"})
    other = [f for f in _rows(client, "SELECT * FROM field") if f["name"] == "Other"][0]
    client.post("/field/1/import", data={"csv_text": WIDE})
    client.post(f"/field/{other['id']}/import", data={"csv_text": WIDE})
    assert len(_rows(client, "SELECT * FROM reading")) == 8

    client.post("/field/1/readings/delete", data={})
    survivors = _rows(client, "SELECT * FROM reading")
    assert len(survivors) == 4, "a scoped delete crossed into another field"


def test_deleting_a_site_takes_its_readings_with_it(client):
    """A reading whose spot is gone has no location, and an unlocatable
    measurement is not worth keeping."""
    client.post("/field/1/import", data={"csv_text": WIDE})
    site = _rows(client, "SELECT * FROM site ORDER BY id")[0]
    other_count = len(_rows(client, "SELECT * FROM reading WHERE site_id != ?",
                            (site["id"],)))

    client.post(f"/field/1/site/{site['id']}/delete")
    assert not _rows(client, "SELECT * FROM site WHERE id = ?", (site["id"],))
    assert len(_rows(client, "SELECT * FROM reading")) == other_count


def test_deleting_a_soil_test_restores_the_refusal(client):
    """Removing the only soil test must put the engine back to 'obtain a soil
    test', not leave the last advice cached on the page."""
    client.post("/field/1/soil-test", data={
        "sampled_at": "2026-04-12", "ph": "5.72", "buffer_ph": "6.50",
        "mehlich3_p_ppm": "22", "mehlich3_k_ppm": "85"})
    assert "lb P2O5" in visible_text(client.get("/field/1").text)

    test_id = _rows(client, "SELECT * FROM lab_test")[0]["id"]
    client.post(f"/field/1/soil-test/{test_id}/delete")
    page = visible_text(client.get("/field/1").text)
    assert "No soil test on file" in page
    assert "lb P2O5" not in page


def test_deleting_a_field_requires_its_name_typed_back(client):
    r = client.post("/field/1/delete", data={"confirm": "wrong"},
                    follow_redirects=True)
    assert "Type the field name exactly" in visible_text(r.text)
    assert _rows(client, "SELECT * FROM field WHERE id = 1"), "field deleted without confirmation"


def test_deleting_a_field_removes_everything_under_it(client):
    client.post("/field/1/import", data={"csv_text": WIDE})
    client.post("/field/1/soil-test", data={"sampled_at": "2026-04-12", "ph": "6.1"})

    client.post("/field/1/delete", data={"confirm": "North Field"})
    assert not _rows(client, "SELECT * FROM field WHERE id = 1")
    assert not _rows(client, "SELECT * FROM site WHERE field_id = 1")
    assert not _rows(client, "SELECT * FROM lab_test WHERE field_id = 1")
    assert not _rows(client, "SELECT * FROM reading"), "orphaned readings survived"


def test_deleting_something_that_does_not_exist_is_a_404(client):
    assert client.post("/field/1/soil-test/999/delete").status_code == 404
    assert client.post("/field/1/site/999/delete").status_code == 404


def test_deletes_are_not_reachable_by_get(client):
    """A GET that destroys data can be triggered by a crawler, a prefetch, or a
    pasted link."""
    for url in ["/field/1/delete", "/field/1/readings/delete",
                "/field/1/site/1/delete"]:
        assert client.get(url).status_code in (404, 405), f"{url} responds to GET"
