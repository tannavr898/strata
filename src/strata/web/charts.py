"""charts.py — server-rendered chart geometry.

Pure functions: they take data and return SVG-ready coordinates. Nothing here
touches a request, a template, or the database, so the geometry is testable
without rendering anything.

WHY SERVER-RENDERED SVG
-----------------------
The app has no build step and no npm, and a chart library would bring both. The
charts needed here are a dozen lines of arithmetic, and computing the coordinates
in Python keeps them inspectable and unit-testable in a way a client-side library
would not. Hover is handled by SVG `<title>`, which every browser renders as a
native tooltip with no script, and every chart ships a table view beside it so the
underlying numbers are always reachable.

COLOR
-----
Series colors are CATEGORICAL — they say *which site*, not *how good*. They are
therefore kept strictly separate from the app's status palette (amber for
attention, green for ok, slate for uncalibrated). Painting a site's line amber
would read as a judgement about that site, which is exactly the kind of accidental
claim this project avoids elsewhere.

The eight slots below are a published, validated categorical order. Against this
app's own surfaces they pass the lightness band, chroma floor, colorblind
separation and normal-vision floors in both modes (worst adjacent CVD ΔE 9.1
light / 8.4 dark; normal-vision 19.6 / 19.3). Light mode fails the 3:1 contrast
check, which is not dismissable: it obliges **direct labels and a table view**,
both of which the season template ships. Slots are assigned in fixed order and
never cycled — past eight sites, the extras fold into a neutral "other" rather
than reusing a hue that already means something.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from datetime import date, datetime
from typing import Optional, Sequence

#: Validated categorical order. Index = series slot, never reordered, never cycled.
SERIES_COLORS = [
    ("#2a78d6", "#3987e5"),   # blue
    ("#eb6834", "#d95926"),   # orange
    ("#1baf7a", "#199e70"),   # aqua
    ("#eda100", "#c98500"),   # yellow
    ("#e87ba4", "#d55181"),   # magenta
    ("#008300", "#008300"),   # green
    ("#4a3aa7", "#9085e9"),   # violet
    ("#e34948", "#e66767"),   # red
]
MAX_SERIES = len(SERIES_COLORS)


@dataclass(frozen=True)
class Point:
    x: float
    y: float
    value: float
    when: str
    uncalibrated: bool
    synthetic: bool


@dataclass(frozen=True)
class Series:
    label: str
    slot: int
    path: str
    points: list[Point]
    last: Optional[Point]        # anchor for the direct label
    light: str
    dark: str


@dataclass(frozen=True)
class Band:
    """A phenology window drawn behind the data.

    These are the *model's own* sub-season feature windows (see
    `config/sources.yaml` → `daymet.season_windows`). Drawing them behind a
    farmer's probe readings is the one place the app shows the grower and the
    yield model looking at the same calendar: July is the silking window the
    README identifies as deciding the corn year.
    """
    name: str
    label: str
    x: float
    width: float


@dataclass(frozen=True)
class SeasonChart:
    empty: bool
    width: int = 0
    height: int = 0
    pad_l: int = 0
    pad_b: int = 0
    series: list[Series] = dc_field(default_factory=list)
    bands: list[Band] = dc_field(default_factory=list)
    y_ticks: list[tuple[float, str]] = dc_field(default_factory=list)
    x_ticks: list[tuple[float, str]] = dc_field(default_factory=list)
    lo: float = 0.0
    hi: float = 0.0
    n_points: int = 0
    overflow: list[str] = dc_field(default_factory=list)   # sites beyond MAX_SERIES


_WINDOW_LABELS = {
    "aprmay": "Establishment",
    "jun": "Vegetative",
    "jul": "Silking",
    "augoct": "Grain fill",
}


def _yday(stamp: str) -> Optional[int]:
    """Day-of-year from an ISO timestamp, or None if unparseable."""
    try:
        return datetime.fromisoformat(stamp).timetuple().tm_yday
    except (ValueError, TypeError):
        try:
            return date.fromisoformat(stamp[:10]).timetuple().tm_yday
        except (ValueError, TypeError):
            return None


def _nice_ticks(lo: float, hi: float, target: int = 4) -> list[float]:
    """A few round values spanning the range.

    Deliberately simple: picks a 1/2/5×10ⁿ step so axis labels read as round
    numbers rather than artefacts of the data's min and max.
    """
    span = hi - lo
    if span <= 0:
        return [lo]
    raw = span / target
    mag = 10 ** (len(str(int(abs(raw)))) - 1) if abs(raw) >= 1 else 10 ** -2
    for mult in (1, 2, 2.5, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000):
        step = mult * mag
        if span / step <= target + 2:
            break
    first = (int(lo / step) + (1 if lo > 0 and lo % step else 0)) * step
    ticks, v = [], first
    while v <= hi + 1e-9:
        ticks.append(round(v, 6))
        v += step
    return ticks or [lo, hi]


def season_chart(
    series_data: Sequence[dict],
    windows: Sequence[dict],
    season: tuple[int, int] = (91, 304),
    width: int = 720,
    height: int = 300,
) -> SeasonChart:
    """Lay out one channel's readings for every site across a growing season.

    `series_data` is [{"label": str, "points": [{"taken_at", "value",
    "uncalibrated", "synthetic"}]}]. `windows` are the phenology windows from
    `config/sources.yaml`.

    The x domain is the growing season UNIONED with the data's own range, so a
    reading taken outside Apr–Oct is still plotted rather than silently dropped,
    while the bands stay anchored to the real calendar.
    """
    pad_l, pad_r, pad_t, pad_b = 46, 86, 22, 34   # pad_r leaves room for direct labels

    prepared = []
    values: list[float] = []
    ydays: list[int] = []
    for s in series_data:
        pts = []
        for p in s["points"]:
            yd = _yday(p["taken_at"])
            if yd is None or p["value"] is None:
                continue
            pts.append((yd, float(p["value"]), p))
            ydays.append(yd)
            values.append(float(p["value"]))
        if pts:
            prepared.append((s["label"], sorted(pts, key=lambda t: t[0])))

    if not prepared or not values:
        return SeasonChart(empty=True, width=width, height=height)

    overflow = [label for label, _ in prepared[MAX_SERIES:]]
    prepared = prepared[:MAX_SERIES]

    x0 = min(season[0], min(ydays))
    x1 = max(season[1], max(ydays))
    lo, hi = min(values), max(values)
    if hi == lo:                       # a flat series still needs a drawable band
        lo, hi = lo - 1, hi + 1
    pad_v = (hi - lo) * 0.08
    lo, hi = lo - pad_v, hi + pad_v

    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    def sx(yd: float) -> float:
        return pad_l + (yd - x0) / (x1 - x0) * plot_w

    def sy(v: float) -> float:
        return pad_t + (hi - v) / (hi - lo) * plot_h

    out_series: list[Series] = []
    n_points = 0
    for slot, (label, pts) in enumerate(prepared):
        points = [
            Point(x=round(sx(yd), 1), y=round(sy(v), 1), value=v,
                  when=raw["taken_at"][:10],
                  uncalibrated=bool(raw.get("uncalibrated")),
                  synthetic=raw.get("synthetic", False))
            for yd, v, raw in pts
        ]
        n_points += len(points)
        path = " ".join(
            f"{'M' if i == 0 else 'L'}{p.x},{p.y}" for i, p in enumerate(points)
        )
        light, dark = SERIES_COLORS[slot]
        out_series.append(Series(
            label=label, slot=slot, path=path, points=points,
            last=points[-1] if points else None, light=light, dark=dark,
        ))

    bands = []
    for w in windows:
        bx0, bx1 = max(w["start_yday"], x0), min(w["end_yday"], x1)
        if bx1 <= bx0:
            continue
        bands.append(Band(
            name=w["name"],
            label=_WINDOW_LABELS.get(w["name"], w["name"]),
            x=round(sx(bx0), 1), width=round(sx(bx1) - sx(bx0), 1),
        ))

    y_ticks = [(round(sy(v), 1), f"{v:g}") for v in _nice_ticks(lo, hi)]
    x_ticks = []
    for yd, label in ((91, "Apr"), (121, "May"), (152, "Jun"), (182, "Jul"),
                      (213, "Aug"), (244, "Sep"), (274, "Oct")):
        if x0 <= yd <= x1:
            x_ticks.append((round(sx(yd), 1), label))

    return SeasonChart(
        empty=False, width=width, height=height, pad_l=pad_l, pad_b=pad_b,
        series=out_series, bands=bands, y_ticks=y_ticks, x_ticks=x_ticks,
        lo=lo, hi=hi, n_points=n_points, overflow=overflow,
    )


# --------------------------------------------------------------------------- #
# Field map
# --------------------------------------------------------------------------- #
#: Sequential ramp for MAGNITUDE, one hue light->dark. Kept separate from the
#: categorical series order above, which encodes identity: a map shades a spot by
#: how much of something is there, which is a different question from which spot
#: it is. Light steps mean "low" and are allowed to recede toward the surface.
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281"]


@dataclass(frozen=True)
class MapSite:
    label: str
    x: float
    y: float
    lat: float
    lon: float
    value: Optional[float]
    fill: str            # sequential step, or a neutral when there is no value
    reading_count: int


@dataclass(frozen=True)
class FieldMap:
    """A schematic of a field's sampling spots in real coordinate space.

    SCHEMATIC, NOT A BASEMAP. There are no tiles and no field boundary -- only the
    spots, placed where they actually are relative to one another, with a scale
    bar so the distances are readable. That is deliberate: it needs no network, no
    API key and no JavaScript, and it is honest about being a sketch of sampling
    geometry rather than a picture of the farm.

    IT IS NOT AN INTERPOLATED SURFACE, and must not be mistaken for one. Shading
    three points and smoothing between them would be drawing, not measuring: a
    credible gradient over a 40-acre field wants on the order of 10-20
    well-distributed samples plus uncertainty that grows with distance from each
    one. The coordinates here are projected properly so that a surface layer can
    be added on top later without redoing the geometry -- see `needs_for_surface`.
    """
    empty: bool
    sites: list[MapSite] = dc_field(default_factory=list)
    width: int = 0
    height: int = 0
    scale_label: str = ""       # e.g. "100 m"
    scale_px: float = 0.0
    lo: Optional[float] = None
    hi: Optional[float] = None
    unplaced: list[str] = dc_field(default_factory=list)   # sites with no coordinates

    #: How many well-distributed spots a believable interpolated surface needs.
    #: Stated in the type rather than a comment so the UI can quote one number.
    SURFACE_MIN_SITES = 10

    @property
    def needs_for_surface(self) -> int:
        return max(0, self.SURFACE_MIN_SITES - len(self.sites))


_EARTH_M_PER_DEG = 111_320.0


def field_map(sites: Sequence[dict], width: int = 560, height: int = 380,
              pad: int = 46) -> FieldMap:
    """Project a field's sites onto a square-metre-accurate schematic.

    `sites` is [{"label", "lat", "lon", "value", "reading_count"}]; `value` may be
    None, which renders as a neutral marker rather than the bottom of the ramp --
    "not measured here" is not "lowest reading here".

    Longitude is compressed by cos(latitude) before scaling. Without that a field
    in Pennsylvania renders about 23% too wide east-west, which would misplace
    every spot relative to every other and make the schematic worse than useless
    for walking back to a probe.
    """
    placed = [s for s in sites if s.get("lat") is not None and s.get("lon") is not None]
    unplaced = [s["label"] for s in sites
                if s.get("lat") is None or s.get("lon") is None]
    if not placed:
        return FieldMap(empty=True, width=width, height=height, unplaced=unplaced)

    lats = [float(s["lat"]) for s in placed]
    lons = [float(s["lon"]) for s in placed]
    mid_lat = sum(lats) / len(lats)
    import math
    lon_scale = math.cos(math.radians(mid_lat))

    # Metres east/north of the south-west corner of the bounding box.
    xs_m = [(lon - min(lons)) * _EARTH_M_PER_DEG * lon_scale for lon in lons]
    ys_m = [(lat - min(lats)) * _EARTH_M_PER_DEG for lat in lats]
    span_x = max(xs_m) - min(xs_m)
    span_y = max(ys_m) - min(ys_m)
    # A single spot, or spots in a perfect line, still needs a drawable extent.
    span = max(span_x, span_y, 20.0)

    plot_w, plot_h = width - 2 * pad, height - 2 * pad
    # ONE scale for both axes: separate x and y scales would stretch the field and
    # silently lie about the shape of the sampling pattern.
    px_per_m = min(plot_w, plot_h) / span
    off_x = pad + (plot_w - span_x * px_per_m) / 2
    off_y = pad + (plot_h - span_y * px_per_m) / 2

    values = [s["value"] for s in placed if s.get("value") is not None]
    lo = min(values) if values else None
    hi = max(values) if values else None

    out = []
    for s in placed:
        value = s.get("value")
        if value is None or lo is None or hi == lo:
            fill = SEQUENTIAL[3] if value is not None else "transparent"
        else:
            idx = int((value - lo) / (hi - lo) * (len(SEQUENTIAL) - 1))
            fill = SEQUENTIAL[max(0, min(idx, len(SEQUENTIAL) - 1))]
        x_m = (float(s["lon"]) - min(lons)) * _EARTH_M_PER_DEG * lon_scale
        y_m = (float(s["lat"]) - min(lats)) * _EARTH_M_PER_DEG
        out.append(MapSite(
            label=s["label"],
            x=round(off_x + x_m * px_per_m, 1),
            # SVG y grows downward; north must be up.
            y=round(height - (off_y + y_m * px_per_m), 1),
            lat=float(s["lat"]), lon=float(s["lon"]),
            value=value, fill=fill,
            reading_count=s.get("reading_count", 0),
        ))

    # A round scale bar: pick the largest of 10/25/50/100/250/500 m that fits.
    scale_px, scale_label = 0.0, ""
    for metres in (500, 250, 100, 50, 25, 10):
        if metres * px_per_m <= plot_w * 0.45:
            scale_px, scale_label = metres * px_per_m, f"{metres} m"
            break

    return FieldMap(
        empty=False, sites=out, width=width, height=height,
        scale_label=scale_label, scale_px=round(scale_px, 1),
        lo=lo, hi=hi, unplaced=unplaced,
    )
