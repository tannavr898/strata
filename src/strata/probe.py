"""probe.py — PLACEHOLDER seam for field probe readings.

STATUS: contract only. Nothing here is wired into the running pipeline yet, and
`python -m strata.main` does not import it. It exists so the sensor->software
mapping is written down and reviewable NOW, while the device's actual output
format is still unknown.

WHY A PLACEHOLDER AND NOT AN IMPLEMENTATION
-------------------------------------------
The sensor suite is final (the team has the hardware), so the CHANNELS below are
settled. What is NOT settled is what the device emits: field names, units, sample
cadence, how a spot reading is delimited from a logging session, and how location
is attached. Writing a parser against a guessed format would be inventing data,
which this project does not do. The three conversion functions therefore raise
NotImplementedError with the specific question each one is blocked on. Fill them
in once a real reading exists; the surrounding contract should not need to change.

WHERE THIS FITS
---------------
    device -> ProbeReading -> ProbeCalibration -> SoilTest -> recommender
                           \\
                            -> time-varying features    -> yield_model

The FIRST path is a refinement: it replaces a coarse county SSURGO pH with a real
field measurement in rules the recommender already applies today.

The SECOND path is the one that matters scientifically. The yield model's ceiling
(METHODS.md 7) exists because SSURGO is a single static snapshot per county, so it
can only address corn's ~20% between-county variance and none of the ~80% that
varies year to year. Free gridded soil proxies did NOT break it -- the water
balance and ERA5-Land soil moisture both added ~0, because they are reanalyses
derived from the same precip/temp the weather windows already carry. A probe reads
ACTUAL field soil water and biology, independent of county-mean meteorology. That
independence is the whole argument, and this is where it enters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


# --------------------------------------------------------------------------- #
# Channel registry — settled, because the hardware is bought.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Channel:
    """One measurable quantity the device reports.

    `lab_scale` is the honest part. A channel that maps onto a lab-equivalent value
    can be calibrated by `ProbeCalibration` and consumed by the recommender's cited
    AASL thresholds. A channel with lab_scale=None has no AASL rule to answer to; it
    is a direct physical measurement whose value is the TIME SERIES, and it feeds
    the yield model rather than the rule engine. Never route a lab_scale=None
    channel into a threshold rule -- that would compare a raw sensor number against
    a lab-defined cutoff.
    """
    name: str
    sensor: str
    units: str
    lab_scale: Optional[str]   # the lab measurement this calibrates onto, if any
    destination: str           # "recommender" | "yield_model" | "both"
    note: str = ""


CHANNELS: dict[str, Channel] = {
    "soil_moisture": Channel(
        name="soil_moisture", sensor="SEN0193 capacitive", units="volumetric %",
        lab_scale=None, destination="yield_model",
        note="The headline time-varying signal. Capacitive output is soil-specific "
             "and drifts with texture and salinity, so it needs a per-soil "
             "reference (gravimetric oven-dry samples), NOT a lab equivalence.",
    ),
    "soil_temp": Channel(
        name="soil_temp", sensor="DS18B20", units="degC",
        lab_scale=None, destination="yield_model",
        note="Root-zone heat. Factory-calibrated and stable; also the covariate "
             "several other channels must be corrected against (pH and CO2 are "
             "both temperature-sensitive).",
    ),
    "ph": Channel(
        name="ph", sensor="pH probe", units="pH",
        lab_scale="water pH", destination="both",
        note="Calibrates against standard buffer solutions (pH 4.01/6.86/9.18). "
             "Maps onto SSURGO/lab WATER pH -- NOT buffer pH, so it refines the "
             "lime TRIGGER but still cannot supply a lime RATE.",
    ),
    "nitrogen": Channel(
        name="nitrogen", sensor="NPK sensor", units="mg/kg (device scale)",
        lab_scale=None, destination="yield_model",
        note="No AASL rule to answer to: Penn State's corn N guidance is the PSNT "
             "(pre-sidedress nitrate) test, which this sensor does not perform. "
             "Treated as a time-varying feature, not a recommendation input.",
    ),
    "phosphorus": Channel(
        name="phosphorus", sensor="NPK sensor", units="mg/kg (device scale)",
        lab_scale="mehlich3_p_ppm", destination="both",
        note="CAUTION: the recommender's thresholds are specifically MEHLICH-3 "
             "extractable P. A conductivity-style NPK probe does not perform a "
             "Mehlich-3 extraction, so this is an EMPIRICAL correlation onto that "
             "scale, not the same measurement. It must earn its lab_scale claim "
             "with real paired samples and an honest R2, or be demoted to "
             "yield_model-only like nitrogen.",
    ),
    "potassium": Channel(
        name="potassium", sensor="NPK sensor", units="mg/kg (device scale)",
        lab_scale="mehlich3_k_ppm", destination="both",
        note="Same caveat as phosphorus.",
    ),
    "co2": Channel(
        name="co2", sensor="SCD40", units="ppm",
        lab_scale=None, destination="yield_model",
        note="Soil respiration as a proxy for microbial activity -- the channel "
             "commercial probes skip. No AASL threshold exists for it, so it can "
             "only ever be a yield-model feature and a trend shown to the farmer, "
             "never a cited recommendation.",
    ),
}

#: Channels that may legitimately reach the cited rule engine, once each has a
#: real calibration fitted. Everything else is yield-model / display only.
RECOMMENDER_CHANNELS = tuple(
    c.name for c in CHANNELS.values() if c.destination in ("recommender", "both")
)


# --------------------------------------------------------------------------- #
# The reading record
# --------------------------------------------------------------------------- #
@dataclass
class ProbeReading:
    """One observation from the device: one place, one moment, N channels.

    Both device modes produce these. A handheld spot check is a single reading; a
    planted logging session is many readings sharing a `site_id`. Nothing here
    assumes which mode produced it -- that distinction lives in how many readings
    share a site and how far apart their timestamps are.
    """
    site_id: str                       # stable id for a spot in a field
    taken_at: datetime
    values: dict[str, float] = field(default_factory=dict)   # channel name -> RAW reading
    lat: Optional[float] = None
    lon: Optional[float] = None
    device_id: Optional[str] = None
    # Set once ProbeCalibration has mapped a raw value onto its lab scale. Kept
    # SEPARATE from `values` so a calibrated number can never be mistaken for a raw
    # one, or a raw one silently used where a lab-equivalent was required.
    calibrated: dict[str, float] = field(default_factory=dict)

    def unknown_channels(self) -> list[str]:
        """Channel names not in the registry — a device firmware change, or a typo."""
        return sorted(set(self.values) - set(CHANNELS))


# --------------------------------------------------------------------------- #
# Conversions — deliberately unimplemented
# --------------------------------------------------------------------------- #
def parse_device_output(raw) -> list[ProbeReading]:
    """Turn whatever the device emits into ProbeReadings.

    BLOCKED ON: one real capture from the assembled device. Specifically needed --
      * the serialization (CSV line? JSON? BLE characteristic payload?)
      * field names and units per channel, and whether units are per-device
      * the sample cadence when logging, and how a session is delimited
      * whether location rides with each reading or is attached by the phone at
        sync time (phone-at-sync is acceptable), and its precision
      * the device clock's timezone and whether it survives a power cycle
    """
    raise NotImplementedError(
        "parse_device_output is a placeholder: no device output format is known "
        "yet. Capture one real reading, record its shape here, then implement."
    )


def readings_to_soiltest(readings, crop: str, calibration):
    """Collapse readings at one site into a SoilTest for the cited rule engine.

    BLOCKED ON: two decisions that need real data, not a guess --
      * how to reduce a logging session to the single value a threshold rule wants
        (mean over the session? the most recent? a specific depth or time of day?).
        The AASL rules were written against a lab result from a composite core, so
        whatever is chosen has to be defensible as standing in for that.
      * whether `phosphorus`/`potassium` earn their Mehlich-3 lab_scale claim at
        all -- see the CAUTION on those channels. Until paired lab samples say yes
        with an honest R2, they must be excluded here, and the recommender will
        correctly keep returning 'obtain a soil test' for P and K.

    Note what does NOT change: a calibrated pH refines the lime TRIGGER, but buffer
    pH is a separate laboratory measurement no field probe performs, so the engine
    will still decline to state a lime RATE. That refusal is intended.
    """
    raise NotImplementedError(
        "readings_to_soiltest is a placeholder: the session-to-single-value "
        "reduction and the P/K lab-scale validation both need real paired data."
    )


def readings_to_features(readings):
    """Aggregate readings into time-varying features for the yield model.

    This is the function that would actually break the ceiling, and it is the
    hardest one to do honestly. BLOCKED ON a decision the data has to settle:

    The yield model is trained on COUNTY-YEAR rows against a NASS county yield.
    Probe readings are FIELD-scale. Attaching a field measurement to a county-mean
    target is a granularity mismatch in the opposite direction from the usual one,
    and it cannot be resolved by aggregation alone -- one probe in one field is not
    a county mean, and pretending otherwise would fabricate representativeness.

    The two candidate resolutions, neither of which should be picked before there
    is data to judge them on:
      1. Keep the county-year model as the public backbone and use probe features
         only as a documented field-level ADJUSTMENT on top of its prediction.
      2. Move to a field-year model, which needs field-level yield truth (farmer
         reported or combine monitor) that Strata does not currently have.

    Whichever is chosen must be stated in METHODS.md before any number derived
    from it is reported.
    """
    raise NotImplementedError(
        "readings_to_features is a placeholder: the field-vs-county granularity "
        "question must be settled with real readings before this can be honest."
    )
