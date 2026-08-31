"""assemble.py — build the county-year-crop analysis table.

WHAT this module does
---------------------
It merges the three independently-pulled, tidy data layers into the single flat
table the yield model trains on. One output row = one **(county, year, crop)**
observation: a yield TARGET sitting next to the static soil features for that
county and the growing-season weather features for that exact year.

    one row = (county, year, crop, YIELD) + [soil features] + [that year's weather]

WHY it is its own module (not buried in a loader)
-------------------------------------------------
`data_ingest.py` pulls each source on its own terms — different grains, different
endpoints, different caveats:

    NASS yields    grain = county x year x crop   (the prediction TARGET)
    SSURGO soil    grain = county (static)        (slow-changing context)
    Daymet weather grain = county x year          (year-to-year driver)

Reconciling those grains is a modeling decision with consequences (every row it
drops is a county-year-crop we will NOT train on). Per the project rules that
reconciliation, and every dropped row, is logged here step by step rather than
hidden inside a SQL join or a one-liner merge.

Join discipline
---------------
* INNER joins on purpose. A training row is only usable if it has a yield target
  AND soil AND weather; anything missing a piece is dropped, and we report how
  many and why:
    - NASS "other (combined) counties" pseudo-FIPS  -> no SSURGO match -> dropped
    - yield years after Daymet's coverage (2024-2025) -> no weather    -> dropped
    - NASS-suppressed yields (NaN target)             -> nothing to learn -> dropped
* Identifiers (county_ansi, county_name, areasymbol) ride along for traceability
  but are NEVER model features — the model must generalize from soil+weather, not
  memorize county identity. `yield_model.py` selects the feature columns
  explicitly; this module just keeps provenance attached.
* Granularity mismatch (stated, not masked): one static county soil summary and
  one county-centroid weather point describe a whole county, while yield is a
  county aggregate too — so the units agree at the COUNTY level. This is coarse
  context, NOT a field-level claim; field resolution is the future probe layer.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger("strata.assemble")

# __file__ = src/strata/assemble.py -> two parents up is the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_PROCESSED = _REPO_ROOT / "data" / "processed"

# County FIPS is a 3-digit zero-padded code ("001"). It is the universal join key,
# so it MUST be read as a string everywhere — parsed as an int, "001" becomes 1
# and silently fails to match across files. We pin its dtype on every read.
_KEY_DTYPE = {"county_ansi": "string"}

# The soil columns carried onto each row (county-static). dominant_taxorder and
# dominant_drainage are categorical; the rest are numeric topsoil (0-25cm) means.
_SOIL_FEATURES = [
    "topsoil_sand_pct",
    "topsoil_silt_pct",
    "topsoil_clay_pct",
    "topsoil_ph",
    "topsoil_om_pct",
    "topsoil_cec",
    "dominant_taxorder",
    "dominant_drainage",
]

# The weather columns carried onto each row (that year's growing season).
# The first five are the full-season (Apr-Oct) aggregates; the rest are the WATER
# and HEAT-STRESS stats within sub-season phenology windows (see daymet.season_windows
# in config/sources.yaml). Only per-window PRECIP + HEAT-DAYS are carried: an A/B on
# leave-counties-out AND forward-in-time splits showed these capture the within-year
# timing signal (corn forward-forecast R^2 0.28 -> 0.39), while the per-window
# mean-temp/GDD columns are collinear with the season means and only added
# forward-generalization noise (full 20-column set fell back to forward R^2 0.28).
# The Daymet loader still COMPUTES all five stats per window into daymet_pa_weather.csv;
# this is the model's deliberate feature selection. Kept in sync with WEATHER_NUMERIC
# in yield_model.py (assemble writes them; the model reads them).
_WEATHER_FEATURES = [
    "gdd_base10",
    "precip_total_mm",
    "tmax_mean_c",
    "tmin_mean_c",
    "heat_days_30c",
    # per-window precip (mm) + heat-stress days: Apr-May, June, July (silking), Aug-Oct
    "precip_aprmay_mm", "heat_days_aprmay",
    "precip_jun_mm", "heat_days_jun",
    "precip_jul_mm", "heat_days_jul",
    "precip_augoct_mm", "heat_days_augoct",
]

# The drought columns carried onto each row. Climatic water balance (precip minus
# Hargreaves PET, mm) for the season + the same four phenology windows -- a
# moisture-deficit signal that combines rain AND evaporative demand, which neither
# precip nor temperature alone carries. These live in the SAME daymet_pa_weather.csv
# (derived there from daily temps + latitude), so they are selected in the weather
# join below. Kept in sync with DROUGHT_NUMERIC in yield_model.py.
_DROUGHT_FEATURES = [
    "waterbal_season_mm",
    "waterbal_aprmay_mm",
    "waterbal_jun_mm",
    "waterbal_jul_mm",
    "waterbal_augoct_mm",
]

# The soil-moisture columns carried onto each row (that year's growing season).
# ERA5-Land root-zone volumetric soil moisture (m3/m3), season mean + the same four
# phenology-window means as the weather features. This is the TIME-VARYING soil
# signal the static SSURGO snapshot cannot supply -- the free gridded preview of the
# probe's moisture channel. Joined on (county_ansi, year) like weather. Kept in sync
# with SOILMOIST_NUMERIC in yield_model.py.
_SOILMOIST_FEATURES = [
    "soilmoist_root_season",
    "soilmoist_root_aprmay",
    "soilmoist_root_jun",
    "soilmoist_root_jul",
    "soilmoist_root_augoct",
]

# The NDVI columns carried onto each row. Cropland-masked MODIS NDVI, county-year,
# TIME-GATED to <= July (Apr-May / Jun / Jul) so late-season greenness -- which is
# almost the harvested yield itself -- cannot leak into the forecast. Unlike the
# gridded weather/soil-moisture reanalyses, NDVI is a direct satellite OBSERVATION
# of the crop (vigor, management, pest stress), so it can carry information the
# meteorology does not. Joined on (county_ansi, year). Kept in sync with
# NDVI_NUMERIC in yield_model.py.
_NDVI_FEATURES = [
    "ndvi_aprmay",
    "ndvi_jun",
    "ndvi_jul",
]


def _read_processed(name: str) -> pd.DataFrame:
    """Read one processed CSV with county_ansi pinned to string.

    These files are the deterministic outputs of `data_ingest.py`. Reading them
    here (instead of re-pulling) keeps the join offline and fast; they are fully
    reproducible from the provenance in the README.
    """
    path = _PROCESSED / name
    if not path.exists():
        raise FileNotFoundError(
            f"Expected {path} — run the matching loader in data_ingest.py first."
        )
    df = pd.read_csv(path, dtype=_KEY_DTYPE)
    logger.info("Read %s: %d rows x %d cols", name, len(df), df.shape[1])
    return df


def build_analysis_table(
    nass: Optional[pd.DataFrame] = None,
    soil: Optional[pd.DataFrame] = None,
    weather: Optional[pd.DataFrame] = None,
    soilmoist: Optional[pd.DataFrame] = None,
    ndvi: Optional[pd.DataFrame] = None,
    drop_suppressed: bool = True,
    include_soilmoist: bool = True,
    include_ndvi: bool = True,
) -> pd.DataFrame:
    """Join NASS yields + SSURGO soil + Daymet weather into one analysis table.

    Parameters
    ----------
    nass, soil, weather:
        Optionally pass in already-loaded frames (used by tests / the orchestrator).
        If omitted, the corresponding processed CSV is read from data/processed/.
    drop_suppressed:
        If True (default), rows whose yield target is NaN (NASS-suppressed) are
        removed — the model cannot learn from a missing answer. Set False to keep
        them for inspection.

    Returns
    -------
    pd.DataFrame
        One row per (county_ansi, year, crop) with target + soil + weather, sorted
        deterministically. Identifier columns are kept for traceability.

    Every step logs rows-in / rows-out / rows-dropped so the shrinkage is auditable.
    """
    if nass is None:
        nass = _read_processed("nass_pa_yields.csv")
    if soil is None:
        soil = _read_processed("ssurgo_pa_soil.csv")
    if weather is None:
        weather = _read_processed("daymet_pa_weather.csv")
    if include_soilmoist and soilmoist is None:
        soilmoist = _read_processed("openmeteo_pa_soilmoisture.csv")
    if include_ndvi and ndvi is None:
        ndvi = _read_processed("ndvi_pa.csv")

    # ------------------------------------------------------------------ NASS prep
    # "query_name" is our crop label; rename to the clearer "crop". Keep only the
    # columns we need downstream (target + keys + unit, since units differ by crop).
    base = nass.rename(columns={"query_name": "crop"})[
        ["crop", "county_ansi", "county_name", "year", "unit_desc", "yield_value"]
    ].copy()
    n_start = len(base)
    logger.info("NASS yield rows (all crops, all counties): %d", n_start)

    # Drop rows with no usable county FIPS — they can never join to soil/weather.
    no_fips = base["county_ansi"].isna()
    if no_fips.any():
        logger.info("  dropping %d rows with missing county_ansi", int(no_fips.sum()))
        base = base[~no_fips]

    # ----------------------------------------------------- join 1: static soil
    # Inner join on county FIPS ONLY (not on year): the SSURGO soil summary is a
    # single STATIC snapshot per county with no time axis, so every year of a county
    # is stamped with an identical soil row. Consequence for the yield model
    # (quantified in the README): these features have ZERO within-county variance and
    # can only address the ~20% between-county share of corn yield variance -- never
    # the ~80% that varies year to year. That within-county signal has to come from
    # weather + the time trend; a future within-field, within-season probe reading
    # (the calibration seam) is the only way to add a time-varying soil feature here.
    # This is also where NASS pseudo-counties ("other combined counties") fall out:
    # they have no SSURGO survey area.
    soil_cols = ["county_ansi", "areasymbol"] + _SOIL_FEATURES
    before = len(base)
    real_fips = set(soil["county_ansi"])
    matched = base["county_ansi"].isin(real_fips)
    dropped_counties = sorted(set(base.loc[~matched, "county_ansi"].dropna()))
    merged = base.merge(soil[soil_cols], on="county_ansi", how="inner")
    logger.info(
        "Join 1 (+SSURGO soil): %d -> %d rows  (dropped %d; %d unmatched FIPS: %s)",
        before,
        len(merged),
        before - len(merged),
        len(dropped_counties),
        dropped_counties,
    )

    # ------------------------------------------------- join 2: that year's weather
    # Inner join on (FIPS, year). This is where post-coverage years (2024-2025,
    # past Daymet v4 R1) fall out — no weather row exists for them.
    weather_cols = ["county_ansi", "year"] + _WEATHER_FEATURES + _DROUGHT_FEATURES
    before = len(merged)
    weather_max_year = int(weather["year"].max())
    merged = merged.merge(weather[weather_cols], on=["county_ansi", "year"], how="inner")
    dropped = before - len(merged)
    logger.info(
        "Join 2 (+Daymet weather): %d -> %d rows  (dropped %d; weather coverage ends %d)",
        before,
        len(merged),
        dropped,
        weather_max_year,
    )

    # -------------------------------------------- join 3: that year's soil moisture
    # Inner join on (FIPS, year), exactly like weather. ERA5-Land soil moisture shares
    # Daymet's 2000-2023 x 67-county coverage (same centroids), so this should drop
    # nothing; we log it to verify rather than assume. This is the first TIME-VARYING
    # soil feature in the table -- the axis the static SSURGO snapshot cannot supply.
    if include_soilmoist:
        sm_cols = ["county_ansi", "year"] + _SOILMOIST_FEATURES
        before = len(merged)
        merged = merged.merge(soilmoist[sm_cols], on=["county_ansi", "year"], how="inner")
        logger.info(
            "Join 3 (+ERA5-Land soil moisture): %d -> %d rows  (dropped %d)",
            before, len(merged), before - len(merged),
        )
    else:
        logger.info("Join 3 (soil moisture) SKIPPED (include_soilmoist=False).")

    # ---------------------------------------------- join 4: that year's NDVI (satellite)
    # Inner join on (FIPS, year). Cropland-masked MODIS NDVI, time-gated to <= July.
    # MOD13Q1 covers 2000+ so it aligns with the panel; logged to verify no drops.
    if include_ndvi:
        ndvi_cols = ["county_ansi", "year"] + _NDVI_FEATURES
        before = len(merged)
        merged = merged.merge(ndvi[ndvi_cols], on=["county_ansi", "year"], how="inner")
        logger.info(
            "Join 4 (+MODIS NDVI): %d -> %d rows  (dropped %d)",
            before, len(merged), before - len(merged),
        )
    else:
        logger.info("Join 4 (NDVI) SKIPPED (include_ndvi=False).")

    # ----------------------------------------------- drop suppressed (NaN) targets
    if drop_suppressed:
        before = len(merged)
        suppressed = merged["yield_value"].isna()
        merged = merged[~suppressed]
        logger.info(
            "Drop suppressed targets: %d -> %d rows  (dropped %d NaN yields)",
            before,
            len(merged),
            int(suppressed.sum()),
        )

    # Deterministic ordering so the saved table is byte-stable across runs.
    merged = merged.sort_values(
        ["crop", "county_ansi", "year"], kind="mergesort"
    ).reset_index(drop=True)

    logger.info(
        "Analysis table: %d rows  (%.1f%% of the %d starting NASS rows survived the joins)",
        len(merged),
        100.0 * len(merged) / n_start if n_start else 0.0,
        n_start,
    )
    return merged


def _report(df: pd.DataFrame) -> None:
    """Print a human-readable summary of the assembled analysis table."""
    print("\n=== Strata analysis table (county-year-crop) ===")
    print(f"rows: {len(df)}   columns: {df.shape[1]}")
    print(f"counties: {df['county_ansi'].nunique()}   years: "
          f"{int(df['year'].min())}-{int(df['year'].max())}")

    print("\nrows per crop (these are the per-crop training set sizes):")
    hdr = f"  {'crop':<20}{'rows':>6}{'counties':>10}{'years':>14}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for crop, g in df.groupby("crop"):
        yrs = f"{int(g['year'].min())}-{int(g['year'].max())}"
        print(f"  {crop:<20}{len(g):>6}{g['county_ansi'].nunique():>10}{yrs:>14}")

    # Feature completeness — by construction this should be fully populated, but we
    # verify rather than assume (council rule: show, don't claim).
    feat_cols = [c for c in (_SOIL_FEATURES + _WEATHER_FEATURES + _DROUGHT_FEATURES
                             + _SOILMOIST_FEATURES + _NDVI_FEATURES) if c in df.columns]
    miss = df[feat_cols].isna().mean() * 100
    worst = miss[miss > 0]
    if worst.empty:
        print("\nfeature completeness: all soil + weather features fully populated.")
    else:
        print("\nfeature columns with missingness:")
        for c, pct in worst.items():
            print(f"  {c:<22}{pct:5.1f}% missing")

    chester = df[df["county_name"].str.upper() == "CHESTER"]
    print(f"\nChester County training rows: {len(chester)}")
    for crop, g in chester.groupby("crop"):
        print(f"  {crop:<20}{len(g):>4}  ({int(g['year'].min())}-{int(g['year'].max())})")


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    table = build_analysis_table()
    _report(table)

    out = _PROCESSED / "analysis_county_year.csv"
    table.to_csv(out, index=False)
    print(f"\nWrote {len(table)} rows -> {out}")
    print("\nSample (Chester County corn grain, most recent years):")
    sample = table[(table["crop"] == "corn_grain_yield")
                   & (table["county_name"].str.upper() == "CHESTER")].tail(5)
    print(sample.to_string(index=False))
