"""Feature-list contracts and leakage guards.

`assemble.py` writes feature columns and `yield_model.py` selects them, and the
two are today kept in sync by a hand-maintained comment in each file ("Kept in
sync with ..."). A comment cannot fail. These tests can.

The leakage guards encode design rules stated in METHODS.md that are invisible at
runtime -- nothing crashes if `county_ansi` becomes a feature or a post-harvest
NDVI window sneaks in; the scores just quietly get better and stop being true.
"""

from __future__ import annotations

import pytest

from strata import assemble
from strata.soil_classifier import SoilClassifier
from strata.yield_model import YieldModel


# --------------------------------------------------------------------------- #
# assemble.py <-> yield_model.py must agree
# --------------------------------------------------------------------------- #
def test_model_weather_features_are_all_written_by_assemble():
    written = set(assemble._WEATHER_FEATURES)
    selected = set(YieldModel.WEATHER_NUMERIC)
    orphans = selected - written
    assert not orphans, (
        f"YieldModel selects weather features assemble.py never writes: "
        f"{sorted(orphans)} -- the two lists drifted apart"
    )


def test_model_ndvi_features_are_all_written_by_assemble():
    orphans = set(YieldModel.NDVI_NUMERIC) - set(assemble._NDVI_FEATURES)
    assert not orphans, f"YieldModel selects NDVI features assemble.py never writes: {sorted(orphans)}"


def test_every_model_feature_exists_in_the_analysis_table(analysis_table):
    missing = [c for c in YieldModel.FEATURES if c not in analysis_table.columns]
    assert not missing, (
        f"features the model asks for are absent from analysis_county_year.csv: "
        f"{missing} -- rebuild it with `python -m strata.assemble`"
    )


def test_target_and_group_keys_exist(analysis_table):
    for col in (YieldModel.TARGET, YieldModel.GROUP, YieldModel.TREND_FEATURE):
        assert col in analysis_table.columns, f"missing required column {col!r}"


def _numeric_columns(model: YieldModel) -> list[str]:
    """The numeric columns the fitted pipeline will actually consume.

    Read off the ColumnTransformer rather than a convenience accessor, so the test
    asserts on what the model really does, not on a parallel description of it.
    """
    pre = model.pipeline.named_steps["pre"]
    (name, _transformer, columns), = [t for t in pre.transformers if t[0] == "num"]
    return list(columns)


# --------------------------------------------------------------------------- #
# Leakage guards
# --------------------------------------------------------------------------- #
def test_county_identity_is_never_a_feature():
    """County identity is the grouping key for leave-counties-out CV. If it also
    becomes a feature, LCO stops measuring generalization to a NEW county."""
    banned = {"county_ansi", "county_name", "fips", "areasymbol", "areaname"}
    leaked = banned & set(YieldModel.FEATURES)
    assert not leaked, f"county identity leaked into the feature matrix: {sorted(leaked)}"


def test_no_per_county_target_statistic_is_a_feature():
    suspicious = [f for f in YieldModel.FEATURES
                  if "yield" in f.lower() or f == YieldModel.TARGET]
    assert not suspicious, f"target-derived features present: {suspicious}"


def test_ndvi_is_time_gated_to_july():
    """Late-season greenness is almost the harvested yield itself. Only windows
    ending on or before July may be features (METHODS.md 3.2, source 8)."""
    allowed_suffixes = {"aprmay", "jun", "jul"}
    for feat in YieldModel.NDVI_NUMERIC:
        suffix = feat.rsplit("_", 1)[-1]
        assert suffix in allowed_suffixes, (
            f"{feat!r} is an NDVI window past July -- post-silking greenness leaks "
            f"the harvest into the forecast"
        )


def test_trend_term_is_excluded_in_attribution_mode():
    """Attribution mode answers 'how much do soil+weather ALONE explain?'. If the
    year trend survives into it, the conservative number is no longer conservative."""
    assert YieldModel.TREND_FEATURE not in _numeric_columns(YieldModel(include_trend=False))
    assert YieldModel.TREND_FEATURE in _numeric_columns(YieldModel(include_trend=True))


def test_soil_classifier_excludes_coordinates():
    """Predicting soil order from lat/lon is geography, not soil science."""
    leaked = {"lat", "lon", "latitude", "longitude"} & set(SoilClassifier.FEATURES)
    assert not leaked, f"coordinates leaked into the soil classifier: {sorted(leaked)}"


def test_soil_classifier_excludes_taxonomy_labels():
    """Finer taxonomy (great group, suborder) trivially determines the order."""
    leaked = [f for f in SoilClassifier.FEATURES
              if "tax" in f.lower() or "group" in f.lower() or "order" in f.lower()]
    assert not leaked, f"taxonomy labels leaked into the classifier features: {leaked}"


# --------------------------------------------------------------------------- #
# The product quality gate
# --------------------------------------------------------------------------- #
def test_product_gate_threshold_is_a_named_constant():
    assert 0.0 < YieldModel.PRODUCT_MIN_FORECAST_R2 < 1.0
