"""The recommender's refusal behavior.

Refusing to answer is this project's central design claim: a tool that confidently
guesses a fertilizer rate is worse than no tool, because a wrong rate costs a
farmer money and pollutes the Chesapeake watershed. That claim is only true if the
refusals actually hold, so they are tested like any other requirement.

These tests need no data files -- the engine is a pure cited rule table.
"""

from __future__ import annotations

import pytest

from strata.recommender import OptimizationEngine, SoilTest


@pytest.fixture
def engine():
    return OptimizationEngine()


def _by_category(recs, needle):
    matches = [r for r in recs if needle.lower() in r.category.lower()]
    assert matches, f"no {needle!r} recommendation was produced"
    return matches[0]


# --------------------------------------------------------------------------- #
# Input honesty: SSURGO carries pH/CEC/OM but NOT Mehlich-3 P/K or buffer pH.
# --------------------------------------------------------------------------- #
def test_refuses_to_interpret_phosphorus_without_a_lab_test(engine):
    ssurgo_only = SoilTest(crop="corn_grain_yield", ph=5.72, cec=10.5, om_pct=2.4)
    rec = _by_category(engine.recommend(ssurgo_only), "Phosphate")
    assert rec.severity == "unknown"
    assert rec.amount is None, f"invented a P rate with no soil test: {rec.amount!r}"


def test_refuses_to_interpret_potassium_without_a_lab_test(engine):
    ssurgo_only = SoilTest(crop="corn_grain_yield", ph=5.72, cec=10.5, om_pct=2.4)
    rec = _by_category(engine.recommend(ssurgo_only), "Potash")
    assert rec.severity == "unknown"
    assert rec.amount is None, f"invented a K rate with no soil test: {rec.amount!r}"


def test_a_predicted_yield_never_unlocks_a_missing_soil_test(engine):
    """The yield->reco handoff scales a rate; it must not manufacture the input.

    This is the sharp edge of the module composition: passing a confident model
    number in must not make the engine forget it has no P/K measurement.
    """
    ssurgo_only = SoilTest(crop="corn_grain_yield", ph=5.72, cec=10.5, om_pct=2.4)
    recs = engine.recommend(ssurgo_only, expected_yield=179.0)
    for needle in ("Phosphate", "Potash"):
        rec = _by_category(recs, needle)
        assert rec.severity == "unknown"
        assert rec.amount is None, (
            f"a predicted yield unlocked a {needle} rate that has no measurement "
            f"behind it: {rec.amount!r}"
        )


def test_no_lime_rate_without_buffer_ph(engine):
    """Water pH says lime IS needed; only buffer pH says HOW MUCH."""
    no_buffer = SoilTest(crop="corn_grain_yield", ph=5.72, cec=10.5, om_pct=2.4)
    rec = _by_category(engine.recommend(no_buffer), "Lime")
    assert rec.severity == "deficiency", "acidic soil should still flag a lime need"
    assert rec.amount is None, f"invented a lime tonnage from water pH alone: {rec.amount!r}"


def test_lime_rate_appears_once_buffer_ph_is_supplied(engine):
    with_buffer = SoilTest(crop="corn_grain_yield", ph=5.72, buffer_ph=6.50,
                           cec=10.5, om_pct=2.4)
    rec = _by_category(engine.recommend(with_buffer), "Lime")
    assert rec.amount is not None
    assert "ESTIMATE" in rec.amount, (
        "a transcribed buffer-table rate must be labeled an estimate to confirm "
        "against the current AASL report"
    )


# --------------------------------------------------------------------------- #
# Provenance: a model estimate must never be passed off as a measurement.
# --------------------------------------------------------------------------- #
def test_rate_names_the_model_as_its_basis(engine):
    measured = SoilTest(crop="corn_grain_yield", ph=6.2, mehlich3_p_ppm=22.0,
                        mehlich3_k_ppm=85.0, cec=10.5, om_pct=2.4)
    rec = _by_category(engine.recommend(measured, expected_yield=179.0), "Phosphate")
    assert rec.amount is not None
    assert "YieldModel" in rec.amount, (
        f"the removal basis must name itself as a model prediction: {rec.amount!r}"
    )


def test_farmers_stated_goal_is_used_when_no_prediction_is_offered(engine):
    goal = SoilTest(crop="corn_grain_yield", ph=6.2, mehlich3_p_ppm=22.0,
                    mehlich3_k_ppm=85.0, cec=10.5, om_pct=2.4, yield_goal=160.0)
    rec = _by_category(engine.recommend(goal), "Phosphate")
    assert rec.amount is not None
    assert "YieldModel" not in rec.amount, (
        "with no prediction supplied the basis is the farmer's goal, not the model"
    )


def test_no_rate_when_there_is_no_yield_basis_at_all(engine):
    """No prediction and no stated goal -> no fabricated removal rate."""
    bare = SoilTest(crop="corn_grain_yield", ph=6.2, mehlich3_p_ppm=22.0,
                    mehlich3_k_ppm=85.0, cec=10.5, om_pct=2.4)
    rec = _by_category(engine.recommend(bare), "Phosphate")
    assert rec.severity == "deficiency", "22 ppm M3-P is below optimum and must be flagged"
    assert rec.amount is None or "removal" not in rec.amount.lower(), (
        f"produced a crop-removal rate with no yield basis: {rec.amount!r}"
    )


# --------------------------------------------------------------------------- #
# Every recommendation is auditable.
# --------------------------------------------------------------------------- #
def test_every_recommendation_carries_a_citation(engine):
    full = SoilTest(crop="corn_grain_yield", ph=5.72, buffer_ph=6.50,
                    mehlich3_p_ppm=22.0, mehlich3_k_ppm=85.0, cec=10.5, om_pct=2.4)
    recs = engine.recommend(full, expected_yield=179.0)
    assert recs, "no recommendations produced for a fully specified soil test"
    for rec in recs:
        assert rec.citation is not None, f"{rec.category} has no citation"
        assert getattr(rec.citation, "source", ""), f"{rec.category} citation has no source"


def test_excessive_phosphorus_is_flagged_as_environmental(engine):
    """>200 ppm M3-P is a P-Index / runoff concern, not just 'no P needed'."""
    loaded = SoilTest(crop="corn_grain_yield", ph=6.2, mehlich3_p_ppm=250.0,
                      mehlich3_k_ppm=140.0, cec=10.5, om_pct=2.4)
    rec = _by_category(engine.recommend(loaded), "Phosphate")
    assert rec.severity == "environmental", (
        f"250 ppm M3-P should raise an environmental flag, got {rec.severity!r}"
    )
