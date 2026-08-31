"""Join and coverage invariants for the county-year analysis table.

These exist because of a real defect (METHODS.md 12.1): `SsurgoSoilLoader` used to
derive county FIPS by string-slicing the survey-area symbol ('PA029' -> '029'),
which silently fabricated codes like '605' for the five PA JOINT two-county survey
areas and dropped the 10 real counties they cover. Nothing failed loudly -- the
model just quietly trained on 56 counties instead of 66 for weeks.

Every assertion below would have caught that at the moment it was introduced.
"""

from __future__ import annotations

import pytest

# The 10 counties that vanished under the areasymbol-slicing bug. They are the
# regression canary: if any of them goes missing again, the join broke again.
JOINT_SURVEY_COUNTIES = {
    "015",  # Bradford   (PA610 with Sullivan)
    "023",  # Cameron    (PA607 with Elk)
    "047",  # Elk        (PA607 with Cameron)
    "053",  # Forest     (PA609 with Warren)
    "059",  # Greene     (PA611 with Washington)
    "067",  # Juniata    (PA605 with Mifflin)
    "087",  # Mifflin    (PA605 with Juniata)
    "113",  # Sullivan   (PA610 with Bradford)
    "123",  # Warren     (PA609 with Forest)
    "125",  # Washington (PA611 with Greene)
}

# PA county FIPS are the odd numbers 001..133. Anything outside this set is
# fabricated -- which is exactly what the sliced survey-area codes were.
VALID_PA_COUNTY_FIPS = {f"{n:03d}" for n in range(1, 134, 2)}


def test_ssurgo_covers_all_67_pa_counties(ssurgo_soil):
    assert len(ssurgo_soil) == 67, (
        f"expected one soil row per PA county, got {len(ssurgo_soil)}"
    )
    assert ssurgo_soil["county_ansi"].nunique() == 67


def test_no_fabricated_county_fips(ssurgo_soil):
    """The bug's signature: survey-area codes ('605') masquerading as county FIPS."""
    bad = set(ssurgo_soil["county_ansi"].dropna()) - VALID_PA_COUNTY_FIPS
    assert not bad, f"county_ansi values that are not real PA county FIPS: {sorted(bad)}"


def test_joint_survey_counties_are_all_present(ssurgo_soil):
    """The 10 counties in the 5 two-county survey areas must each get their own row."""
    present = set(ssurgo_soil["county_ansi"].dropna())
    missing = JOINT_SURVEY_COUNTIES - present
    assert not missing, (
        f"counties from joint survey areas are missing from the soil table: "
        f"{sorted(missing)} -- the laoverlap county mapping regressed"
    )


def test_joint_survey_areas_emit_one_row_per_county(ssurgo_soil):
    """A joint area is shared by exactly 2 counties, each carrying that area's summary."""
    shared = (ssurgo_soil.groupby("areasymbol")["county_ansi"]
              .nunique().loc[lambda s: s > 1])
    assert set(shared.index) == {"PA605", "PA607", "PA609", "PA610", "PA611"}, (
        f"unexpected set of multi-county survey areas: {sorted(shared.index)}"
    )
    assert (shared == 2).all(), f"a joint survey area covered != 2 counties: {shared.to_dict()}"


def test_soil_join_drops_no_real_county(analysis_table, nass_yields):
    """Every real NASS county must survive the soil join.

    The NASS 'OTHER (COMBINED) COUNTIES' pseudo-records carry no county_ansi at all
    and are dropped one step earlier -- a separate population, never conflated with
    a join failure.
    """
    real = set(nass_yields["county_ansi"].dropna()) & VALID_PA_COUNTY_FIPS
    survived = set(analysis_table["county_ansi"].dropna())
    lost = real - survived
    assert not lost, f"real NASS counties lost in the soil join: {sorted(lost)}"


def test_analysis_table_trains_on_66_counties(analysis_table):
    n = analysis_table["county_ansi"].nunique()
    assert n == 66, (
        f"expected 66 counties in the modeling table (67 PA counties minus the one "
        f"NASS reports no yields for), got {n}"
    )


def test_no_duplicate_county_year_crop_rows(analysis_table):
    key = ["crop", "county_ansi", "year"]
    dupes = analysis_table.duplicated(subset=key).sum()
    assert dupes == 0, f"{dupes} duplicated {key} rows -- a join fanned out"


def test_soil_columns_are_static_within_a_county(analysis_table):
    """SSURGO is one snapshot per county: it MUST NOT vary by year.

    This is the mechanical fact the whole ceiling argument rests on (METHODS.md 7).
    If a soil column ever varies within a county, either the join fanned out or a
    time-varying source got mislabeled as soil, and the ceiling analysis is void.
    """
    soil_cols = ["topsoil_ph", "topsoil_om_pct", "topsoil_cec", "topsoil_clay_pct"]
    varying = [
        c for c in soil_cols
        if (analysis_table.groupby("county_ansi")[c].nunique(dropna=True) > 1).any()
    ]
    assert not varying, f"static SSURGO columns vary within a county: {varying}"
