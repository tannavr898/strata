"""The Overview's soil summary.

These tests exist to stop a dashboard widget from quietly undoing the project's
central claim. The temptation on an overview screen is a single "soil health"
number, and the whole point of `status.py` is that no such number is defensible:
`recommender.CITATIONS` covers pH, Mehlich-3 P and Mehlich-3 K, and nothing in
Penn State's publications supplies weights for combining them with organic matter
or biology.

So the assertions here are mostly about what must NOT appear: no score, no
averaging an unmeasured value away against a good one, no judgement on a property
that has no citation.
"""

from __future__ import annotations

import pytest

from strata.recommender import OptimizationEngine, SoilTest
from strata.web.status import alerts_for, soil_status


def _lab(**over):
    """A lab_test row as sqlite3.Row-like mapping access."""
    base = {"ph": None, "buffer_ph": None, "mehlich3_p_ppm": None,
            "mehlich3_k_ppm": None, "cec": None, "om_pct": None,
            "yield_goal": None, "notes": None, "sampled_at": "2026-04-12"}
    base.update(over)
    return base


def _status(crop="corn_grain_yield", **over):
    lab = _lab(**over)
    test = SoilTest(
        crop=crop, ph=lab["ph"], buffer_ph=lab["buffer_ph"],
        mehlich3_p_ppm=lab["mehlich3_p_ppm"], mehlich3_k_ppm=lab["mehlich3_k_ppm"],
        cec=lab["cec"], om_pct=lab["om_pct"],
    )
    recs = OptimizationEngine().recommend(test)
    return soil_status(lab, recs)


# --------------------------------------------------------------------------- #
# What must not exist
# --------------------------------------------------------------------------- #
def test_status_exposes_no_composite_score():
    """A 0-100 soil health number would need weights no publication supplies.
    A circular scorer was deleted from this project once already."""
    s = _status(ph=5.72, buffer_ph=6.5, mehlich3_p_ppm=22, mehlich3_k_ppm=85)
    for banned in ("score", "points", "grade", "rating", "index", "percent"):
        assert not hasattr(s, banned), f"SoilStatus grew a {banned!r} field"


def test_unruled_properties_never_enter_the_judgement():
    """CEC and organic matter have no citation in `recommender.CITATIONS`, so they
    are context, not evidence. Organic matter is the classic soil-health measure,
    which makes it the most tempting one to fold in without a source."""
    s = _status(ph=6.5, mehlich3_p_ppm=60, mehlich3_k_ppm=150,
                cec=10.5, om_pct=2.4)
    chip_keys = {c.key for c in s.chips}
    assert chip_keys <= {"pH", "P", "K"}, f"a chip appeared for an unruled measure: {chip_keys}"
    labels = {label for label, _ in s.context}
    assert labels == {"CEC", "Organic matter"}


def test_every_chip_carries_its_citation():
    """A chip is a compressed finding. If it cannot name the rule behind it, it is
    an assertion rather than a citation."""
    s = _status(ph=5.72, buffer_ph=6.5, mehlich3_p_ppm=22, mehlich3_k_ppm=85)
    assert s.chips
    for chip in s.chips:
        assert chip.citation, f"{chip.key} has no citation"
        assert "Penn State" in chip.citation


# --------------------------------------------------------------------------- #
# "Not measured" is its own state
# --------------------------------------------------------------------------- #
def test_no_lab_test_is_unmeasured_not_healthy():
    s = soil_status(None, [])
    assert s.state == "unmeasured"
    assert s.needs_attention, "an unassessed field was treated as needing nothing"


def test_partial_measurement_does_not_read_as_all_clear():
    """pH fine, P and K never measured. The field is not 'ok' -- two thirds of the
    judgement is missing, and a green card would say otherwise."""
    s = _status(ph=6.5)
    assert s.state == "partial"
    assert s.n_unknown == 2
    assert "2" not in s.headline or "3" in s.headline


def test_unknown_findings_do_not_average_against_good_ones():
    s = _status(ph=6.5)
    assert s.n_ruled == 1
    assert s.n_unknown == 2
    assert s.state != "ok", "unmeasured values were absorbed into an all-clear"


def test_all_measures_good_is_ok():
    s = _status(ph=6.5, mehlich3_p_ppm=60, mehlich3_k_ppm=150)
    assert s.state == "ok"
    assert not s.needs_attention


def test_a_deficiency_demands_attention():
    s = _status(ph=5.72, buffer_ph=6.5, mehlich3_p_ppm=22, mehlich3_k_ppm=85)
    assert s.state == "attention"
    assert s.n_action >= 1


def test_excessive_phosphorus_outranks_a_deficiency():
    """Excess P is a runoff and P-Index concern for the Chesapeake watershed, not
    just a yield question, so it must headline over a shortfall elsewhere."""
    s = _status(ph=6.5, mehlich3_p_ppm=250, mehlich3_k_ppm=85)
    assert s.state == "attention"
    assert "excessive" in s.headline.lower()


# --------------------------------------------------------------------------- #
# The attention list
# --------------------------------------------------------------------------- #
def _field(name="North Field", crop="corn_grain_yield"):
    return {"id": 1, "name": name, "crop": crop, "county_name": "Chester"}


def test_an_unmeasured_field_produces_an_alert():
    """A dashboard listing only actions would imply an unmeasured field is fine."""
    s = soil_status(None, [])
    alerts = alerts_for(_field(), s, {"state": "ok"})
    assert any("no soil test" in a.title.lower() for a in alerts)
    assert all(a.href for a in alerts), "an alert with nowhere to go"


def test_a_withheld_crop_produces_an_explaining_alert():
    s = _status(ph=6.5, mehlich3_p_ppm=60, mehlich3_k_ppm=150)
    alerts = alerts_for(_field(crop="oats_yield"), s, {"state": "withheld"})
    assert any("no yield estimate" in a.title.lower() for a in alerts)


def test_uncalibrated_readings_are_surfaced():
    s = _status(ph=6.5, mehlich3_p_ppm=60, mehlich3_k_ppm=150)
    alerts = alerts_for(_field(), s, {"state": "ok"}, uncalibrated=252)
    assert any("uncalibrated" in a.title for a in alerts)


def test_a_clean_field_raises_nothing():
    s = _status(ph=6.5, mehlich3_p_ppm=60, mehlich3_k_ppm=150)
    assert alerts_for(_field(), s, {"state": "ok"}) == []


def test_ph_alert_does_not_say_to_add_ph():
    """pH is corrected by liming, not by adding pH. The generic nutrient phrasing
    reads as nonsense on the one measure that is not an input."""
    s = _status(ph=5.72, buffer_ph=6.5, mehlich3_p_ppm=60, mehlich3_k_ppm=150)
    alerts = alerts_for(_field(), s, {"state": "ok"})
    ph_alerts = [a for a in alerts if "pH" in a.title]
    assert ph_alerts
    for a in ph_alerts:
        assert "added pH" not in a.detail
        assert "lime" in a.detail.lower()


def test_one_field_gets_one_deficiency_alert_not_three():
    """Three near-identical 'below optimum' rows for the same field read as three
    problems when it is one soil test with three findings — and the repetition
    buries the genuinely different items further down the list."""
    s = _status(ph=5.72, buffer_ph=6.5, mehlich3_p_ppm=22, mehlich3_k_ppm=85)
    alerts = alerts_for(_field(), s, {"state": "ok"})
    low = [a for a in alerts if a.tone == "low"]
    assert len(low) == 1, f"expected one grouped deficiency alert, got {len(low)}"
    assert "pH, P and K" in low[0].title
    for measure in ("pH 5.72", "P 22", "K 85"):
        assert measure in low[0].detail, "the grouped alert dropped a measured value"


def test_excess_phosphorus_stays_a_separate_alert():
    """Excess P carries the opposite instruction to a shortfall — apply none.
    Folding it in with deficiencies would blur two opposite actions."""
    s = _status(ph=5.72, buffer_ph=6.5, mehlich3_p_ppm=250, mehlich3_k_ppm=85)
    alerts = alerts_for(_field(), s, {"state": "ok"})
    env = [a for a in alerts if a.tone == "env"]
    assert len(env) == 1
    assert "excessive" in env[0].title
    assert "Apply none" in env[0].detail


def test_unmeasured_measures_are_grouped_too():
    s = _status(ph=6.5)
    alerts = alerts_for(_field(), s, {"state": "ok"})
    unknown = [a for a in alerts if a.tone == "unknown"]
    assert len(unknown) == 1
    assert "P and K not measured" in unknown[0].title
