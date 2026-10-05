"""calibration.py — the clean seam for a future cheap soil probe.

WHAT this module does
---------------------
Defines how a cheap field sensor's raw reading (pH, P, K, ...) gets mapped onto
the LAB ground-truth scale the rest of Strata speaks in. A probe is only useful if
its numbers can be trusted like a lab's; this module fits a per-property
calibration from paired (probe, lab) measurements, reports how good that mapping
is (slope, R^2, RMSE, residual bias), and applies it to convert a raw reading into
a lab-equivalent value the `recommender` can consume.

WHY it is a separate, optional layer (the whole-product design rule)
--------------------------------------------------------------------
Strata has ZERO dependency on any sensor existing. Yield regression, soil
classification, and the agronomy recommender all run today on lab/public data. A
cheap probe, when one exists, plugs in HERE and nowhere else: calibrate the probe
against a handful of lab samples, then feed calibrated readings into the same
recommender. Nothing upstream needs to change. This file defines that contract now
so the hardware can land later without a rewrite.

WHAT a probe adds that the current feature matrix cannot (the ceiling it lifts)
-------------------------------------------------------------------------------
The yield model's honest ceiling (LCO R^2 ~0.5 / forward ~0.35) is not a tuning
problem — it is a feature-information limit. SSURGO soil is a single STATIC snapshot
per county: it has no time axis, so it explains only the ~20% between-county share
of corn yield variance and NOTHING of the ~80% that varies year to year (see the
README variance decomposition). A cheap probe read at a field in-season is the one
thing that changes that: a TIME-VARYING soil measurement, a feature that does not
exist anywhere in the matrix today. Calibrated onto the lab scale here, it becomes
the first input that can speak to the within-county, year-to-year variance the
static SSURGO layer structurally cannot. That — not merely finer spatial resolution
— is why this seam is where genuinely new predictive signal would enter.

HONESTY: there is no real paired probe data yet
-----------------------------------------------
The probe hardware now exists (see `probe.CHANNELS`), but no paired (probe, lab)
samples have been collected, so there is NO real calibration dataset to fit. The demo
below fits the calibration on a clearly-labeled SYNTHETIC paired sample (a known
gain/offset/noise applied to real lab values) purely to exercise the interface.
Per the project rule, synthetic output is labeled as such and is NEVER reported as
a finding — it shows the seam works, not that a real probe is accurate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

logger = logging.getLogger("strata.calibration")

_SEED = 42


@dataclass(frozen=True)
class CalibrationFit:
    """The fitted probe->lab mapping for ONE soil property, plus its quality.

    lab_equivalent = slope * probe_reading + intercept

    `r2`/`rmse` say how well the probe tracks the lab after correction; `mean_bias`
    is the raw average offset (probe - lab) BEFORE correction, i.e. how far off the
    uncalibrated sensor was — the thing calibration exists to remove.
    """
    property_name: str
    units: str
    n: int
    slope: float
    intercept: float
    r2: float
    rmse: float
    mean_bias: float
    synthetic: bool = False   # True if fit on synthetic pairs (never a real finding)


class ProbeCalibration:
    """Fit and apply per-property cheap-probe -> lab calibrations.

    One instance can hold several fits (pH, Mehlich-3 P, Mehlich-3 K, ...). The
    fit is a deliberately simple ordinary least squares line: with the few paired
    samples a farmer would realistically collect, a 1-parameter-per-axis linear
    correction is honest; anything fancier would overfit a tiny calibration set.
    """

    def __init__(self) -> None:
        self.fits: dict[str, CalibrationFit] = {}

    def fit(
        self,
        property_name: str,
        probe: Sequence[float],
        lab: Sequence[float],
        units: str = "",
        synthetic: bool = False,
    ) -> CalibrationFit:
        """Least-squares fit lab ~ slope*probe + intercept for one property."""
        p = np.asarray(probe, dtype=float)
        l = np.asarray(lab, dtype=float)
        if p.shape != l.shape or p.ndim != 1:
            raise ValueError("probe and lab must be 1-D arrays of equal length.")
        if len(p) < 3:
            raise ValueError(
                f"Need >=3 paired samples to calibrate {property_name!r}; got {len(p)}."
            )

        slope, intercept = np.polyfit(p, l, 1)
        pred = slope * p + intercept
        ss_res = float(((l - pred) ** 2).sum())
        ss_tot = float(((l - l.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        rmse = float(np.sqrt(((l - pred) ** 2).mean()))
        mean_bias = float((p - l).mean())

        fit = CalibrationFit(
            property_name=property_name, units=units, n=len(p),
            slope=float(slope), intercept=float(intercept),
            r2=r2, rmse=rmse, mean_bias=mean_bias, synthetic=synthetic,
        )
        self.fits[property_name] = fit
        logger.info(
            "Calibrated %s: lab = %.3f*probe + %.3f  (R2=%.3f, RMSE=%.3f, raw bias=%+.3f, n=%d%s)",
            property_name, fit.slope, fit.intercept, fit.r2, fit.rmse, fit.mean_bias,
            fit.n, ", SYNTHETIC" if synthetic else "",
        )
        return fit

    def apply(self, property_name: str, probe_reading: float) -> float:
        """Convert one raw probe reading into its lab-equivalent value."""
        if property_name not in self.fits:
            raise KeyError(
                f"No calibration fitted for {property_name!r}; call fit() first."
            )
        f = self.fits[property_name]
        return f.slope * probe_reading + f.intercept


def _print_fit(fit: CalibrationFit) -> None:
    flag = "  [SYNTHETIC -- interface demo only, not a finding]" if fit.synthetic else ""
    print(f"\n{fit.property_name} ({fit.units}){flag}")
    print(f"  mapping     : lab = {fit.slope:.3f} * probe + {fit.intercept:.3f}")
    print(f"  fit quality : R^2 {fit.r2:.3f}   RMSE {fit.rmse:.3f} {fit.units}")
    print(f"  raw offset  : {fit.mean_bias:+.3f} {fit.units} (uncalibrated probe - lab)")
    print(f"  samples     : n = {fit.n}")


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    print("#" * 72)
    print("# SYNTHETIC calibration demo. No real paired probe/lab data exists yet.")
    print("# These pairs are fabricated to exercise the interface, NOT a result.")
    print("#" * 72)

    rng = np.random.default_rng(_SEED)
    cal = ProbeCalibration()

    # For each property: take plausible lab values, then SIMULATE a cheap probe as a
    # biased, noisy version of the lab (known gain/offset/noise). Calibration should
    # recover a line that removes the bias — that is all this demonstrates.
    specs = {
        # property        lab range          probe gain  probe offset  noise sd  units
        "ph":            (np.linspace(5.2, 7.0, 40), 0.92,  0.45, 0.10, ""),
        "mehlich3_p_ppm":(np.linspace(8,   80,  40), 1.15, -3.0,  4.0,  "ppm"),
        "mehlich3_k_ppm":(np.linspace(50, 220,  40), 0.88,  12.0, 8.0,  "ppm"),
    }
    for name, (lab, gain, offset, sd, units) in specs.items():
        probe = gain * lab + offset + rng.normal(0, sd, size=lab.shape)
        fit = cal.fit(name, probe=probe, lab=lab, units=units, synthetic=True)
        _print_fit(fit)

    # Show the seam in action: a raw probe pH reading -> lab-equivalent the
    # recommender would actually use.
    raw = 5.40
    print(f"\nExample: raw probe pH {raw:.2f} -> lab-equivalent "
          f"{cal.apply('ph', raw):.2f} (then fed to recommender).")

    print("\nThe contract: calibrate a probe against a few lab samples here, then "
          "feed\ncalibrated readings into strata.recommender. Nothing upstream "
          "changes.")
