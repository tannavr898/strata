"""The probe -> lab calibration seam.

No hardware data exists yet, so these tests exercise the CONTRACT: the mapping is
recovered correctly, refuses to run on too few pairs, refuses to apply an unfitted
property, and -- critically -- carries the `synthetic` flag through, because the
project rule is that synthetic output is always labeled and never reported as a
finding. A silent loss of that flag is how fake numbers become real claims.
"""

from __future__ import annotations

import numpy as np
import pytest

from strata.calibration import ProbeCalibration


@pytest.fixture
def known_pairs():
    """A probe with a known gain and offset: probe = 0.9*lab + 0.5.

    Inverting exactly, the fit should recover lab = (1/0.9)*probe - 0.5/0.9.
    """
    lab = np.linspace(5.0, 7.5, 25)
    probe = 0.9 * lab + 0.5
    return probe, lab


def test_recovers_a_known_linear_mapping(known_pairs):
    probe, lab = known_pairs
    fit = ProbeCalibration().fit("ph", probe=probe, lab=lab, units="pH", synthetic=True)
    assert fit.slope == pytest.approx(1 / 0.9, rel=1e-6)
    assert fit.intercept == pytest.approx(-0.5 / 0.9, rel=1e-6)
    assert fit.r2 == pytest.approx(1.0, abs=1e-9)


def test_apply_round_trips_to_the_lab_value(known_pairs):
    probe, lab = known_pairs
    cal = ProbeCalibration()
    cal.fit("ph", probe=probe, lab=lab, synthetic=True)
    for p, expected in zip(probe, lab):
        assert cal.apply("ph", float(p)) == pytest.approx(float(expected), abs=1e-9)


def test_mean_bias_reports_the_uncorrected_offset(known_pairs):
    """mean_bias is what the uncalibrated sensor was off by -- the thing the
    calibration exists to remove -- so it is measured BEFORE correction."""
    probe, lab = known_pairs
    fit = ProbeCalibration().fit("ph", probe=probe, lab=lab, synthetic=True)
    assert fit.mean_bias == pytest.approx(float((probe - lab).mean()))


def test_synthetic_flag_is_carried_through(known_pairs):
    probe, lab = known_pairs
    cal = ProbeCalibration()
    assert cal.fit("ph", probe=probe, lab=lab, synthetic=True).synthetic is True
    assert cal.fit("k", probe=probe, lab=lab, synthetic=False).synthetic is False


def test_refuses_too_few_pairs():
    with pytest.raises(ValueError, match=r">=3 paired samples"):
        ProbeCalibration().fit("ph", probe=[6.0, 6.5], lab=[6.1, 6.6])


def test_refuses_mismatched_arrays():
    with pytest.raises(ValueError):
        ProbeCalibration().fit("ph", probe=[6.0, 6.5, 7.0], lab=[6.1, 6.6])


def test_refuses_to_apply_an_unfitted_property():
    cal = ProbeCalibration()
    cal.fit("ph", probe=[5.0, 6.0, 7.0], lab=[5.1, 6.1, 7.1])
    with pytest.raises(KeyError, match="mehlich3_p"):
        cal.apply("mehlich3_p", 30.0)


def test_fits_are_independent_per_property(known_pairs):
    """One instance holds several fits; they must not overwrite each other."""
    probe, lab = known_pairs
    cal = ProbeCalibration()
    cal.fit("ph", probe=probe, lab=lab)
    cal.fit("mehlich3_k", probe=probe * 10, lab=lab * 10)
    assert set(cal.fits) == {"ph", "mehlich3_k"}
    assert cal.fits["ph"].slope != pytest.approx(cal.fits["mehlich3_k"].slope * 10)
