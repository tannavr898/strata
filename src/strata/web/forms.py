"""forms.py — parsing and validating what people type in.

Separated from the routes because the rules here are the product's honesty
guarantees expressed as input handling, not incidental plumbing.

THE RULE THAT MATTERS MOST
--------------------------
**A blank field stays None. It never becomes 0.0.**

The recommender's whole design is that a missing measurement produces "obtain a
soil test" rather than a number. A form that helpfully coerces an empty
phosphorus box to 0.0 would hand the engine a *measurement of zero* -- which is
not missing data, it is a severely deficient soil, and the engine would
confidently recommend a large phosphate application on the strength of a number
nobody measured. `optional_float` exists to make that impossible, and a test
asserts it.

The same reasoning applies to out-of-range values. A pH of 87 is a typo or a unit
error, and silently storing it would propagate a wrong lime recommendation to a
real field. Ranges below are deliberately WIDE -- they reject the impossible, not
the merely unusual, because a soil that is genuinely odd is exactly the soil
someone most needs advice about.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field as dc_field
from datetime import datetime
from typing import Iterable, Optional

from strata.probe import CHANNELS


class ValidationError(ValueError):
    """A problem with what the user entered, phrased for the user."""


# --------------------------------------------------------------------------- #
# Scalars
# --------------------------------------------------------------------------- #
def optional_float(raw: Optional[str], label: str,
                   lo: Optional[float] = None,
                   hi: Optional[float] = None) -> Optional[float]:
    """Parse a form value that is allowed to be absent.

    Returns None for empty input -- NOT 0.0. See the module docstring: the
    difference between "not measured" and "measured as zero" is the difference
    between the engine refusing to advise and the engine advising wrongly.
    """
    if raw is None:
        return None
    text = str(raw).strip()
    if text == "":
        return None
    try:
        value = float(text)
    except ValueError:
        raise ValidationError(f"{label}: {text!r} is not a number.") from None
    if lo is not None and value < lo:
        raise ValidationError(f"{label}: {value:g} is below the plausible minimum ({lo:g}).")
    if hi is not None and value > hi:
        raise ValidationError(f"{label}: {value:g} is above the plausible maximum ({hi:g}).")
    return value


def required_date(raw: Optional[str], label: str) -> str:
    """An ISO date (YYYY-MM-DD). Required, because an undated soil test cannot be
    ordered against other tests and soil chemistry changes after every liming."""
    text = (raw or "").strip()
    if not text:
        raise ValidationError(f"{label} is required.")
    try:
        return datetime.strptime(text, "%Y-%m-%d").date().isoformat()
    except ValueError:
        raise ValidationError(f"{label}: {text!r} is not a date (use YYYY-MM-DD).") from None


def clean_text(raw: Optional[str], limit: int = 500) -> Optional[str]:
    text = (raw or "").strip()
    return text[:limit] if text else None


# --------------------------------------------------------------------------- #
# Soil test
# --------------------------------------------------------------------------- #
#: (form name, label, min, max). Ranges reject the impossible, not the unusual --
#: a genuinely strange soil is the one most in need of advice.
SOIL_TEST_FIELDS = [
    ("ph",             "Soil pH",           2.0,   11.0),
    ("buffer_ph",      "Buffer pH",         4.0,   8.0),
    ("mehlich3_p_ppm", "Mehlich-3 P (ppm)", 0.0,   1000.0),
    ("mehlich3_k_ppm", "Mehlich-3 K (ppm)", 0.0,   2000.0),
    ("cec",            "CEC (meq/100g)",    0.0,   100.0),
    ("om_pct",         "Organic matter (%)", 0.0,  100.0),
    ("yield_goal",     "Yield goal",        0.0,   10000.0),
]


def parse_soil_test(form) -> dict:
    """Validate a soil-test submission into column values.

    Every measurement is optional. That is not laziness -- a farmer typing in a
    partial report should get advice on what they DID measure and an explicit
    "obtain a soil test" on the rest, which is exactly what the engine does when
    handed None.
    """
    parsed = {
        "sampled_at": required_date(form.get("sampled_at"), "Sample date"),
        "lab_name": clean_text(form.get("lab_name"), 120),
        "notes": clean_text(form.get("notes")),
    }
    for name, label, lo, hi in SOIL_TEST_FIELDS:
        parsed[name] = optional_float(form.get(name), label, lo, hi)

    if all(parsed[n] is None for n, *_ in SOIL_TEST_FIELDS):
        raise ValidationError(
            "Enter at least one measurement. A soil test with no values would "
            "not change any recommendation."
        )
    return parsed


# --------------------------------------------------------------------------- #
# Reading import
# --------------------------------------------------------------------------- #
@dataclass
class ImportResult:
    """What an import did, including what it refused and why.

    Refusals are counted and explained rather than silently dropped: an import
    that says "312 readings added" while having discarded 40 rows it did not
    understand is how bad data enters a system that promises it does not.
    """
    rows: list[tuple] = dc_field(default_factory=list)   # (site_label, taken_at, channel, value)
    skipped: list[str] = dc_field(default_factory=list)
    unknown_channels: set[str] = dc_field(default_factory=set)

    @property
    def n_rows(self) -> int:
        return len(self.rows)


#: Column names accepted for each concept, so a device dump does not have to be
#: renamed by hand before import. Lowercased and stripped before matching.
_SITE_KEYS = {"site", "site_label", "label", "location", "spot"}
_TIME_KEYS = {"taken_at", "timestamp", "time", "datetime", "date"}
_CHANNEL_KEYS = {"channel", "measurement", "sensor", "metric"}
_VALUE_KEYS = {"raw_value", "value", "reading", "raw"}

_TIME_FORMATS = [
    "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M",
    "%Y-%m-%d %H:%M", "%Y-%m-%d", "%m/%d/%Y %H:%M", "%m/%d/%Y",
]


def _parse_time(raw: str) -> Optional[str]:
    text = raw.strip().replace("Z", "")
    for fmt in _TIME_FORMATS:
        try:
            return datetime.strptime(text, fmt).isoformat(timespec="seconds")
        except ValueError:
            continue
    return None


def _match(header: Iterable[str], keys: set[str]) -> Optional[str]:
    for column in header:
        if column.strip().lower() in keys:
            return column
    return None


def parse_readings_csv(text: str, default_site: Optional[str] = None) -> ImportResult:
    """Parse a CSV of probe readings in either long or wide form.

    LONG   site,taken_at,channel,raw_value        one row per measurement
    WIDE   site,taken_at,soil_moisture,soil_temp  one row per visit

    Both are supported because both are plausible: the database is long, but a
    device or a spreadsheet will more naturally emit wide. The shape is detected
    from the header rather than configured, since a person exporting a file
    should not have to know which one Strata prefers.

    Unknown channel names are collected, not guessed at. `probe.CHANNELS` is the
    registry of what this hardware measures; a column outside it means either a
    firmware change worth noticing or a typo worth fixing, and quietly importing
    it under a made-up name would hide both.
    """
    result = ImportResult()
    text = text.strip()
    if not text:
        raise ValidationError("Nothing to import — paste some CSV or choose a file.")

    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise ValidationError("Could not read a header row from that CSV.")

    header = [h for h in reader.fieldnames if h]
    site_col = _match(header, _SITE_KEYS)
    time_col = _match(header, _TIME_KEYS)
    channel_col = _match(header, _CHANNEL_KEYS)
    value_col = _match(header, _VALUE_KEYS)

    if time_col is None:
        raise ValidationError(
            f"No timestamp column found. Expected one of: "
            f"{', '.join(sorted(_TIME_KEYS))}. Found: {', '.join(header)}."
        )
    if site_col is None and default_site is None:
        raise ValidationError(
            f"No site column found and no default site chosen. Expected one of: "
            f"{', '.join(sorted(_SITE_KEYS))}."
        )

    is_long = channel_col is not None and value_col is not None
    wide_channels: list[str] = []
    if not is_long:
        wide_channels = [
            c for c in header
            if c not in {site_col, time_col} and c.strip().lower() not in _SITE_KEYS | _TIME_KEYS
        ]
        if not wide_channels:
            raise ValidationError(
                "No measurement columns found. Provide either channel/value "
                "columns, or one column per measurement."
            )

    for line_no, row in enumerate(reader, start=2):
        site = (row.get(site_col) or "").strip() if site_col else (default_site or "")
        site = site or (default_site or "")
        if not site:
            result.skipped.append(f"line {line_no}: no site")
            continue

        taken_at = _parse_time(row.get(time_col) or "")
        if taken_at is None:
            result.skipped.append(f"line {line_no}: unreadable timestamp "
                                  f"{(row.get(time_col) or '').strip()!r}")
            continue

        pairs = ([(row.get(channel_col) or "", row.get(value_col) or "")] if is_long
                 else [(c, row.get(c) or "") for c in wide_channels])

        for channel_raw, value_raw in pairs:
            channel = channel_raw.strip().lower().replace(" ", "_").replace("-", "_")
            value_text = str(value_raw).strip()
            if not channel or value_text == "":
                continue
            if channel not in CHANNELS:
                result.unknown_channels.add(channel)
                result.skipped.append(f"line {line_no}: unknown measurement {channel!r}")
                continue
            try:
                value = float(value_text)
            except ValueError:
                result.skipped.append(f"line {line_no}: {channel} value "
                                      f"{value_text!r} is not a number")
                continue
            result.rows.append((site, taken_at, channel, value))

    if not result.rows:
        detail = f" Skipped: {'; '.join(result.skipped[:5])}" if result.skipped else ""
        raise ValidationError(f"No usable readings found in that CSV.{detail}")
    return result


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #
def parse_pairs(text: str) -> tuple[list[float], list[float]]:
    """Parse pasted `probe,lab` pairs, one per line.

    Blank lines and a header row are tolerated, because people paste from
    spreadsheets. At least three pairs are required, which `ProbeCalibration.fit`
    also enforces -- a line through two points has no residual and would report a
    perfect R² that means nothing.
    """
    probe: list[float] = []
    lab: list[float] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.replace("\t", ",").split(",") if p.strip()]
        if len(parts) < 2:
            raise ValidationError(f"Line {line_no}: expected two numbers "
                                  f"(probe reading, lab value).")
        try:
            p, l = float(parts[0]), float(parts[1])
        except ValueError:
            if line_no == 1:
                continue        # a header row
            raise ValidationError(f"Line {line_no}: {line!r} is not two numbers.") from None
        probe.append(p)
        lab.append(l)

    if len(probe) < 3:
        raise ValidationError(
            f"Need at least 3 paired samples to fit a calibration; got {len(probe)}. "
            f"Two points always fit a line perfectly, which would report a "
            f"meaningless R²."
        )
    return probe, lab
