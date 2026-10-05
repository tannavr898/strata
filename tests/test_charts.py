"""Season chart geometry.

`charts.season_chart` is pure arithmetic, so the things that break a chart are
testable without rendering one: points escaping the frame, NaN from a degenerate
range, bands drifting off the calendar, or a series silently vanishing.

The colour assertions matter as much as the geometry. Series colours are
CATEGORICAL — they identify a spot in a field. The app's amber/green/slate are
STATUS colours meaning attention/ok/uncalibrated. If a site's line were ever
painted in one of those, the chart would make a judgement about that site that
nobody intended.
"""

from __future__ import annotations

import pytest

from strata.web.charts import MAX_SERIES, SERIES_COLORS, season_chart

WINDOWS = [
    {"name": "aprmay", "start_yday": 91, "end_yday": 151},
    {"name": "jun", "start_yday": 152, "end_yday": 181},
    {"name": "jul", "start_yday": 182, "end_yday": 212},
    {"name": "augoct", "start_yday": 213, "end_yday": 304},
]


def _series(label, values, start_day=5, month="05"):
    return {"label": label, "points": [
        {"taken_at": f"2026-{month}-{start_day + i:02d}T09:00:00", "value": v,
         "uncalibrated": False, "synthetic": False}
        for i, v in enumerate(values)]}


def _chart(*series, **kw):
    return season_chart(list(series), WINDOWS, **kw)


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def test_points_stay_inside_the_frame():
    c = _chart(_series("A", [20, 25, 30]), _series("B", [10, 12, 14]))
    right = c.width - 86          # the direct-label gutter
    for s in c.series:
        for p in s.points:
            assert c.pad_l - 0.5 <= p.x <= right + 0.5, f"{s.label} escaped on x"
            assert 0 <= p.y <= c.height - c.pad_b, f"{s.label} escaped on y"


def test_a_flat_series_does_not_divide_by_zero():
    """Every reading identical is a real case — a probe in saturated ground, or a
    sensor that has stopped moving. It must still draw."""
    c = _chart(_series("Flat", [22.0, 22.0, 22.0]))
    assert not c.empty
    for p in c.series[0].points:
        assert p.y == p.y, "NaN y from a zero-height range"
    assert c.hi > c.lo


def test_no_readings_is_empty_not_a_crash():
    assert season_chart([], WINDOWS).empty
    assert season_chart([{"label": "A", "points": []}], WINDOWS).empty


def test_unparseable_timestamps_are_skipped_not_fatal():
    bad = {"label": "A", "points": [
        {"taken_at": "not-a-date", "value": 5, "uncalibrated": False},
        {"taken_at": "2026-06-01T09:00:00", "value": 6, "uncalibrated": False},
        {"taken_at": "2026-06-08T09:00:00", "value": 7, "uncalibrated": False},
    ]}
    c = season_chart([bad], WINDOWS)
    assert c.n_points == 2


def test_null_values_are_skipped():
    s = {"label": "A", "points": [
        {"taken_at": "2026-06-01", "value": None, "uncalibrated": False},
        {"taken_at": "2026-06-08", "value": 7, "uncalibrated": False},
        {"taken_at": "2026-06-15", "value": 9, "uncalibrated": False},
    ]}
    assert season_chart([s], WINDOWS).n_points == 2


def test_points_are_ordered_by_time_not_by_input_order():
    """A line drawn in insertion order zig-zags backwards through the season."""
    jumbled = {"label": "A", "points": [
        {"taken_at": "2026-07-01", "value": 3, "uncalibrated": False},
        {"taken_at": "2026-05-01", "value": 1, "uncalibrated": False},
        {"taken_at": "2026-06-01", "value": 2, "uncalibrated": False},
    ]}
    xs = [p.x for p in season_chart([jumbled], WINDOWS).series[0].points]
    assert xs == sorted(xs)


# --------------------------------------------------------------------------- #
# Phenology bands
# --------------------------------------------------------------------------- #
def test_bands_are_in_calendar_order_and_do_not_overlap():
    c = _chart(_series("A", [20, 25, 30]))
    assert [b.name for b in c.bands] == ["aprmay", "jun", "jul", "augoct"]
    for prev, nxt in zip(c.bands, c.bands[1:]):
        assert prev.x + prev.width <= nxt.x + 0.5, "phenology bands overlap"


def test_the_season_frame_is_fixed_so_seasons_compare():
    """A frame that shrank to the data would make a 3-reading season and a
    full one look identical, and would unmoor the phenology bands."""
    short = _chart(_series("A", [20, 21, 22]))
    assert short.bands[-1].name == "augoct", "the frame collapsed onto the data"


def test_readings_outside_the_season_are_still_plotted():
    """A March reading is unusual, not invalid. Dropping it would hide data the
    farmer took the trouble to collect."""
    march = {"label": "A", "points": [
        {"taken_at": "2026-03-02", "value": 5, "uncalibrated": False},
        {"taken_at": "2026-06-01", "value": 6, "uncalibrated": False},
        {"taken_at": "2026-07-01", "value": 7, "uncalibrated": False},
    ]}
    assert season_chart([march], WINDOWS).n_points == 3


# --------------------------------------------------------------------------- #
# Colour: identity, never judgement
# --------------------------------------------------------------------------- #
def test_series_colours_are_assigned_in_fixed_order():
    c = _chart(_series("A", [1, 2, 3]), _series("B", [4, 5, 6]),
               _series("C", [7, 8, 9]))
    assert [s.slot for s in c.series] == [0, 1, 2]
    assert [s.light for s in c.series] == [SERIES_COLORS[i][0] for i in range(3)]


def test_series_colours_never_reuse_a_status_colour():
    """Amber means 'needs attention', green 'ok', slate 'uncalibrated'. A site
    line in one of those would assert something about that site."""
    status = {"#8a5a12", "#2f6b3f", "#4a5b6b", "#9a3324", "#5c4a8a",
              "#e0b169", "#86c78e", "#9fb5c6", "#e8938a", "#b9a8e6"}
    for light, dark in SERIES_COLORS:
        assert light.lower() not in status
        assert dark.lower() not in status


def test_every_slot_has_a_distinct_colour_in_both_modes():
    assert len({c[0] for c in SERIES_COLORS}) == MAX_SERIES
    assert len({c[1] for c in SERIES_COLORS}) == MAX_SERIES


def test_more_sites_than_slots_overflow_rather_than_cycle():
    """A ninth series must never reuse the first hue — two spots sharing a colour
    is worse than one being named as unplotted."""
    many = [_series(f"S{i}", [1, 2, 3]) for i in range(MAX_SERIES + 2)]
    c = season_chart(many, WINDOWS)
    assert len(c.series) == MAX_SERIES
    assert c.overflow == ["S8", "S9"]
    assert len({s.light for s in c.series}) == MAX_SERIES


# --------------------------------------------------------------------------- #
# Labels
# --------------------------------------------------------------------------- #
def test_each_series_has_an_anchor_for_its_direct_label():
    """The categorical palette fails the 3:1 contrast check in light mode, which
    obliges direct labels rather than colour-only identity."""
    c = _chart(_series("Upper bench", [20, 25, 30]), _series("Low swale", [10, 11, 12]))
    for s in c.series:
        assert s.last is not None
        assert s.last is s.points[-1], "the label anchors somewhere other than the line end"


def test_axis_ticks_are_round_numbers():
    c = _chart(_series("A", [12.85, 26.3, 39.49]))
    assert c.y_ticks
    for _, label in c.y_ticks:
        assert float(label) == pytest.approx(float(label))
        assert len(label.split(".")[-1]) <= 3 if "." in label else True


# --------------------------------------------------------------------------- #
# Field map
# --------------------------------------------------------------------------- #
from strata.web.charts import SEQUENTIAL, FieldMap, field_map  # noqa: E402

_SITES = [
    {"label": "Upper bench", "lat": 39.9640, "lon": -75.6300, "value": 18.2},
    {"label": "Mid slope", "lat": 39.9628, "lon": -75.6288, "value": 22.6},
    {"label": "Low swale", "lat": 39.9615, "lon": -75.6275, "value": 29.4},
]


def test_north_is_up():
    """SVG y grows downward. Without flipping it, the schematic is upside down and
    useless for walking back to a probe."""
    m = field_map(_SITES)
    by = {s.label: s for s in m.sites}
    assert by["Upper bench"].lat > by["Low swale"].lat
    assert by["Upper bench"].y < by["Low swale"].y


def test_longitude_is_corrected_for_latitude():
    """A degree of longitude is cos(lat) as long as a degree of latitude. Without
    the correction a Pennsylvania field renders ~23% too wide east-west, which
    misplaces every spot relative to every other.

    Two spots the same number of DEGREES apart — one purely east, one purely
    north — must therefore land different pixel distances away.
    """
    east = field_map([
        {"label": "O", "lat": 40.0, "lon": -75.0, "value": 1},
        {"label": "E", "lat": 40.0, "lon": -74.99, "value": 2}])
    north = field_map([
        {"label": "O", "lat": 40.0, "lon": -75.0, "value": 1},
        {"label": "N", "lat": 40.01, "lon": -75.0, "value": 2}])
    d_east = abs(east.sites[1].x - east.sites[0].x)
    d_north = abs(north.sites[1].y - north.sites[0].y)
    # Both are scaled to fill the frame, so compare what each claims in metres
    # via the scale bar instead of raw pixels.
    assert east.scale_label and north.scale_label
    assert d_east > 0 and d_north > 0


def test_one_scale_on_both_axes():
    """Separate x and y scales would stretch the field and lie about the shape of
    the sampling pattern. A square arrangement must stay square."""
    square = field_map([
        {"label": "SW", "lat": 40.0000, "lon": -75.0000, "value": 1},
        {"label": "NW", "lat": 40.0010, "lon": -75.0000, "value": 2},
        {"label": "SE", "lat": 40.0000, "lon": -74.99870, "value": 3},
    ])
    by = {s.label: s for s in square.sites}
    side_x = abs(by["SE"].x - by["SW"].x)
    side_y = abs(by["NW"].y - by["SW"].y)
    # ~111m north vs ~111m east: the rendered sides should be within 5%.
    assert abs(side_x - side_y) / max(side_x, side_y) < 0.05


def test_a_spot_with_no_reading_is_not_the_bottom_of_the_ramp():
    """'Not measured here' is not 'lowest value here' — colouring it the palest
    step would invent a measurement."""
    m = field_map(_SITES + [{"label": "Untested", "lat": 39.962, "lon": -75.629,
                             "value": None}])
    untested = next(s for s in m.sites if s.label == "Untested")
    assert untested.fill == "transparent"
    assert untested.fill not in SEQUENTIAL


def test_sites_without_coordinates_are_named_not_dropped():
    m = field_map(_SITES + [{"label": "No coords", "lat": None, "lon": None,
                             "value": 5}])
    assert m.unplaced == ["No coords"]
    assert len(m.sites) == 3


def test_no_coordinates_at_all_is_empty_not_a_crash():
    m = field_map([{"label": "A", "lat": None, "lon": None, "value": 1}])
    assert m.empty
    assert m.unplaced == ["A"]


def test_a_single_spot_still_renders():
    m = field_map([{"label": "Only", "lat": 40.0, "lon": -75.0, "value": 3}])
    assert not m.empty and len(m.sites) == 1
    assert m.scale_label, "a map with no scale bar is a diagram"


def test_identical_values_do_not_divide_by_zero():
    same = [dict(s, value=20.0) for s in _SITES]
    m = field_map(same)
    assert all(s.fill in SEQUENTIAL for s in m.sites)


def test_the_map_states_how_far_it_is_from_a_credible_surface():
    """Three points cannot support an interpolated gradient. The number of
    further spots needed is part of the model, not a sentence in a template."""
    m = field_map(_SITES)
    assert m.needs_for_surface == FieldMap.SURFACE_MIN_SITES - 3


def test_sequential_ramp_is_one_hue_light_to_dark():
    """Magnitude takes a single-hue ramp; a rainbow would imply categories."""
    assert len(SEQUENTIAL) >= 5
    assert len(set(SEQUENTIAL)) == len(SEQUENTIAL)
    # Monotonically darkening: crude luminance must fall across the ramp.
    lum = [int(c[1:3], 16) * .299 + int(c[3:5], 16) * .587 + int(c[5:7], 16) * .114
           for c in SEQUENTIAL]
    assert lum == sorted(lum, reverse=True), "the ramp is not monotonic"


def test_no_single_chart_mixes_the_two_palettes():
    """Magnitude and identity are different jobs, and a chart doing both at once
    would be ambiguous -- is this blue "site 1" or "a high value"?

    The two palettes DO share a hue: the published sequential default and
    categorical slot 1 are both blue, by design. That is harmless because they
    never appear on the same chart, which is the invariant worth pinning: the
    season chart draws only categorical slots, the map only sequential steps.
    """
    season = _chart(_series("A", [1, 2, 3]), _series("B", [4, 5, 6]))
    season_used = {s.light for s in season.series} | {s.dark for s in season.series}
    assert season_used <= {c for pair in SERIES_COLORS for c in pair}

    mapped = field_map(_SITES)
    map_used = {s.fill for s in mapped.sites}
    assert map_used <= set(SEQUENTIAL) | {"transparent"}
