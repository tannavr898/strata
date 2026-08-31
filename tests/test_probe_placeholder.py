"""The probe seam is a PLACEHOLDER, and these tests keep it honest as one.

Two jobs:
  1. Guard the channel registry's routing rules, which are real design decisions
     (a channel with no lab scale must never reach a cited AASL threshold).
  2. Assert the unimplemented functions still REFUSE. A placeholder that quietly
     grows a guessed implementation is exactly how invented data enters a project
     that promises it never does. When one is genuinely implemented, delete its
     test here and replace it with a real one -- that deletion is the signal.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from strata.probe import (
    CHANNELS,
    RECOMMENDER_CHANNELS,
    ProbeReading,
    parse_device_output,
    readings_to_features,
    readings_to_soiltest,
)


# --------------------------------------------------------------------------- #
# Registry rules
# --------------------------------------------------------------------------- #
def test_registry_covers_the_hardware_that_was_actually_bought():
    assert set(CHANNELS) == {
        "soil_moisture", "soil_temp", "ph", "nitrogen", "phosphorus", "potassium", "co2",
    }


def test_every_channel_declares_a_valid_destination():
    for name, ch in CHANNELS.items():
        assert ch.destination in {"recommender", "yield_model", "both"}, (
            f"{name} has an unrecognized destination {ch.destination!r}"
        )


def test_channels_without_a_lab_scale_never_reach_the_rule_engine():
    """The core routing rule. The recommender compares values against Penn State
    LAB thresholds; feeding it a raw sensor number would compare two different
    measurements and produce a cited-looking recommendation that is not real."""
    offenders = [
        name for name, ch in CHANNELS.items()
        if ch.lab_scale is None and ch.destination in ("recommender", "both")
    ]
    assert not offenders, (
        f"channels routed to the rule engine with no lab scale to compare against: "
        f"{offenders}"
    )


def test_co2_is_never_a_recommendation_input():
    """CO2 is the differentiator, which makes it the most tempting to overclaim.
    No AASL threshold exists for soil respiration, so it can be a feature and a
    trend shown to the farmer, never a cited recommendation."""
    assert CHANNELS["co2"].lab_scale is None
    assert CHANNELS["co2"].destination == "yield_model"
    assert "co2" not in RECOMMENDER_CHANNELS


def test_nitrogen_is_not_routed_to_the_recommender():
    """Penn State's corn N guidance is the PSNT nitrate test, which this sensor
    does not perform. There is no rule for this channel to answer to."""
    assert CHANNELS["nitrogen"].lab_scale is None
    assert "nitrogen" not in RECOMMENDER_CHANNELS


def test_npk_p_and_k_carry_their_mehlich3_caveat():
    """These two claim a Mehlich-3 lab scale they have not yet earned. The caveat
    must stay attached until paired lab samples validate it."""
    for name in ("phosphorus", "potassium"):
        ch = CHANNELS[name]
        assert ch.lab_scale and ch.lab_scale.startswith("mehlich3")
        assert "CAUTION" in ch.note or "caveat" in ch.note, (
            f"{name} claims a Mehlich-3 lab scale with no caveat recorded"
        )


def test_ph_maps_to_water_ph_not_buffer_ph():
    """No field probe performs a buffer-pH titration, so the lime RATE stays
    unavailable even with a calibrated probe. Encoding that here stops a future
    implementation from quietly wiring pH into the rate rule."""
    assert CHANNELS["ph"].lab_scale == "water pH"
    assert "buffer" not in (CHANNELS["ph"].lab_scale or "")


# --------------------------------------------------------------------------- #
# The reading record
# --------------------------------------------------------------------------- #
def test_raw_and_calibrated_values_are_kept_separate():
    r = ProbeReading(site_id="north-field-1", taken_at=datetime(2026, 6, 15, 9, 30),
                     values={"ph": 5.4, "soil_moisture": 22.1})
    assert r.calibrated == {}, "a fresh reading must carry no calibrated values"
    r.calibrated["ph"] = 5.41
    assert r.values["ph"] == 5.4, "calibrating must not mutate the raw reading"


def test_unknown_channels_are_reported():
    r = ProbeReading(site_id="s1", taken_at=datetime(2026, 6, 15),
                     values={"ph": 5.4, "electrical_conductivity": 0.8})
    assert r.unknown_channels() == ["electrical_conductivity"]


# --------------------------------------------------------------------------- #
# The placeholders must stay placeholders until real data exists
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "fn, args",
    [
        (parse_device_output, ("some raw payload",)),
        (readings_to_soiltest, ([], "corn_grain_yield", None)),
        (readings_to_features, ([],)),
    ],
    ids=["parse_device_output", "readings_to_soiltest", "readings_to_features"],
)
def test_placeholder_refuses_rather_than_guessing(fn, args):
    with pytest.raises(NotImplementedError, match="placeholder"):
        fn(*args)


def test_probe_module_is_not_wired_into_the_pipeline():
    """main.py must keep running end to end without any probe code. The product's
    zero-hardware-dependency claim is load-bearing for the grant application."""
    import ast
    from pathlib import Path

    main_src = Path(__file__).resolve().parents[1] / "src" / "strata" / "main.py"
    tree = ast.parse(main_src.read_text(encoding="utf-8"))

    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    assert not any(m.endswith("probe") for m in imported), (
        f"main.py imports the probe placeholder ({sorted(imported)}) -- Strata "
        f"must run end to end with no hardware"
    )
