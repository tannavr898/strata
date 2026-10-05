"""Regenerate every data-driven figure in docs/figures/.

    python docs/scripts/make_figures.py          # use the project venv (needs matplotlib, pandas)

Sources, and what happens when one is missing:

    docs/metrics.jsonl                   always present (committed) -> gate + history charts
    data/processed/analysis_county_year.csv   gitignored -> variance + join charts, SKIPPED if absent
    data/processed/nass_pa_yields.csv         gitignored -> join funnel, SKIPPED if absent

Nothing here recomputes a model: history comes only from what evaluation runs
logged (plus rows explicitly marked `"source": "backfill"`, drawn hollow), so a
chart can never show a number no run produced. Rerun this after any evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
METRICS = ROOT / "docs" / "metrics.jsonl"
FIGS = ROOT / "docs" / "figures"
PROCESSED = ROOT / "data" / "processed"

# Reference palette (dataviz skill, light mode): categorical slots in fixed order.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GREY = "#a3a29d"
INK, INK_2, INK_3 = "#0b0b0b", "#52514e", "#8a8984"
SURFACE, GRID = "#fcfcfb", "#e6e5e1"

CROP_ORDER = ["corn_grain_yield", "soybean_yield", "corn_silage_yield", "hay_yield",
              "barley_yield", "winter_wheat_yield", "alfalfa_hay_yield", "oats_yield"]
CROP_LABEL = {"corn_grain_yield": "Corn grain", "soybean_yield": "Soybean",
              "corn_silage_yield": "Corn silage", "hay_yield": "Hay",
              "barley_yield": "Barley", "winter_wheat_yield": "Winter wheat",
              "alfalfa_hay_yield": "Alfalfa hay", "oats_yield": "Oats"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.labelsize": 10, "xtick.color": INK_3, "ytick.color": INK_3,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "legend.frameon": False,
    "legend.fontsize": 9,
})


def _save(fig, name: str) -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    out = FIGS / name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.relative_to(ROOT)}")


def _skip(name: str, why: str) -> None:
    print(f"  SKIPPED {name}: {why}")


def load_metrics() -> list[dict]:
    if not METRICS.exists():
        return []
    rows = [json.loads(line) for line in METRICS.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    return sorted(rows, key=lambda r: r["timestamp"])


def _version_label(r: dict) -> str:
    """Short two-line tick label: MM-DD, then a tag (`short` on backfilled rows,
    otherwise the run kind and commit). The long `label` belongs in the report."""
    tag = r.get("short") or (f"{r['kind']}\n@{r['commit']}" if r.get("commit") else r["kind"])
    return f"{r['timestamp'][5:10]}\n{tag}"


def _title(ax, title: str, subtitle: str) -> None:
    ax.set_title(title, loc="left", pad=22)
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=8.5, color=INK_3)


# --------------------------------------------------------------------------- #
# 1. Current gate: every crop's honest skill against the ship threshold
# --------------------------------------------------------------------------- #
def fig_gate(rows: list[dict]) -> None:
    builds = [r for r in rows if r["kind"] == "build_models" and "crops" in r["metrics"]]
    if not builds:
        return _skip("gate_latest.png", "no build_models entry in metrics.jsonl")
    latest = builds[-1]
    crops = latest["metrics"]["crops"]
    gate = latest["metrics"].get("gate_threshold", 0.30)
    order = sorted(crops, key=lambda c: crops[c]["lco_r2"])

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ys = range(len(order))
    vals = [crops[c]["lco_r2"] for c in order]
    colors = [BLUE if crops[c]["shipped"] else GREY for c in order]
    ax.barh(list(ys), vals, color=colors, height=0.62, zorder=2)
    ax.axvline(gate, color=INK_2, lw=1.2, ls="--", zorder=3)
    ax.text(gate, -0.75, f" gate {gate:.2f}", color=INK_2, fontsize=9, va="center")
    for y, c in zip(ys, order):
        f = crops[c]
        verdict = "ships" if f["shipped"] else "withheld"
        mae = f" · ±{f['lco_mae']:.2f} {f.get('unit', '').lower()}" if "lco_mae" in f else ""
        ax.text(f["lco_r2"] + 0.01, y, f"{f['lco_r2']:.2f}  {verdict}{mae}",
                va="center", fontsize=8.5, color=INK_2, zorder=4,
                bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1})
    ax.set_yticks(list(ys), [CROP_LABEL.get(c, c) for c in order])
    ax.set_xlim(0, 0.95)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Leave-counties-out forecast R²  (unseen counties)")
    src = f"{latest['timestamp'][:10]}, commit {latest.get('commit') or 'n/a'}"
    ax.set_ylim(-1.1, len(order) - 0.5)
    _title(ax, "Product gate: which crops get a yield forecast",
           f"Latest build ({src}). Grey bars are withheld: below the gate a "
           "forecast is no better than the county mean.")
    _save(fig, "gate_latest.png")


# --------------------------------------------------------------------------- #
# 2. Headline crop over versions: LCO and forward-in-time R²
# --------------------------------------------------------------------------- #
def fig_corn_history(rows: list[dict]) -> None:
    pts = []
    for r in rows:
        m = r["metrics"]
        head = m.get("yield_headline") or {}
        lco = head.get("lco_r2", (m.get("crops", {}).get("corn_grain_yield") or {}).get("lco_r2"))
        fwd = head.get("forward_r2")
        if lco is None and fwd is None:
            continue
        pts.append((r, lco, fwd))
    if not pts:
        return _skip("corn_history.png", "no corn metrics logged")

    fig, ax = plt.subplots(figsize=(9, 4.4))
    xs = list(range(len(pts)))
    for key, color, name in [(1, BLUE, "Leave-counties-out R²"), (2, ORANGE, "Forward-in-time R² (train ≤2017)")]:
        series = [(x, p[key], p[0]["source"]) for x, p in zip(xs, pts) if p[key] is not None]
        if not series:
            continue
        ax.plot([s[0] for s in series], [s[1] for s in series], color=color, lw=2, zorder=2)
        for x, v, source in series:
            hollow = source == "backfill"
            ax.plot(x, v, "o", ms=8, zorder=3, color=color,
                    markerfacecolor=SURFACE if hollow else color, markeredgewidth=2)
        lx, lv, _ = series[-1]
        ax.annotate(f"{name}\n{lv:.2f}", (lx, lv), xytext=(10, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=INK_2)
    ax.set_xticks(xs, [_version_label(p[0]) for p in pts], fontsize=8)
    ax.set_xlim(-0.4, len(pts) - 0.4 + 1.3)
    ax.set_ylim(0, 0.8)
    ax.set_ylabel("R²")
    _title(ax, "Corn grain yield model, across versions",
           "Hollow markers are backfilled from earlier docs; filled markers were "
           "logged by an evaluation run.")
    _save(fig, "corn_history.png")


# --------------------------------------------------------------------------- #
# 3. Every crop over versions (small multiples, one series each)
# --------------------------------------------------------------------------- #
def fig_crop_history(rows: list[dict]) -> None:
    versions = [r for r in rows if "crops" in r["metrics"]]
    if not versions:
        return _skip("crop_history.png", "no per-crop metrics logged")
    fig, axes = plt.subplots(2, 4, figsize=(11, 5), sharey=True)
    for ax, crop in zip(axes.flat, CROP_ORDER):
        pts = [(i, v["metrics"]["crops"][crop]["lco_r2"], v["source"])
               for i, v in enumerate(versions) if crop in v["metrics"]["crops"]]
        ax.axhline(0.30, color=INK_3, lw=1, ls="--", zorder=1)
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=BLUE, lw=2, zorder=2)
            for x, v, src in pts:
                ax.plot(x, v, "o", ms=6, color=BLUE, zorder=3, markeredgewidth=1.6,
                        markerfacecolor=SURFACE if src == "backfill" else BLUE)
            ax.text(pts[-1][0], pts[-1][1] + 0.05, f"{pts[-1][1]:.2f}", ha="center",
                    fontsize=8, color=INK_2)
        ax.set_title(CROP_LABEL[crop], fontsize=10)
        ax.set_xticks(range(len(versions)), [v["timestamp"][5:10] for v in versions],
                      fontsize=7, rotation=45)
        ax.set_ylim(0, 0.8)
    axes[0, 0].set_ylabel("LCO forecast R²")
    axes[1, 0].set_ylabel("LCO forecast R²")
    fig.suptitle("Per-crop honest skill across versions  (dashed = 0.30 product gate)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout()
    _save(fig, "crop_history.png")


# --------------------------------------------------------------------------- #
# 4. Variance decomposition: why static soil hits a ceiling
# --------------------------------------------------------------------------- #
def fig_variance(table) -> None:
    if table is None:
        return _skip("variance_decomposition.png", "analysis_county_year.csv absent")
    shares = []
    for crop in CROP_ORDER:
        c = table[table["crop"] == crop].dropna(subset=["yield_value"])
        if c.empty:
            continue
        g = c["yield_value"].mean()
        cm = c.groupby("county_ansi")["yield_value"].transform("mean")
        sst = ((c["yield_value"] - g) ** 2).sum()
        shares.append((crop, 100 * ((cm - g) ** 2).sum() / sst))
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ys = list(range(len(shares)))[::-1]
    between = [s for _, s in shares]
    within = [100 - s for s in between]
    ax.barh(ys, between, color=BLUE, height=0.6, label="Between counties (static soil can reach)")
    ax.barh(ys, within, left=[b + 0.6 for b in between], color=ORANGE, height=0.6,
            label="Within county, year to year (static soil cannot)")
    for y, b in zip(ys, between):
        ax.text(b / 2, y, f"{b:.0f}%", ha="center", va="center", color="white", fontsize=8.5)
        ax.text(b + (100 - b) / 2, y, f"{100 - b:.0f}%", ha="center", va="center",
                color="white", fontsize=8.5)
    ax.set_yticks(ys, [CROP_LABEL[c] for c, _ in shares])
    ax.set_xlim(0, 101)
    ax.set_xlabel("Share of yield variance (%)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=1)
    ax.set_title("Where yield variance lives", loc="left")
    _save(fig, "variance_decomposition.png")


# --------------------------------------------------------------------------- #
# 5. Join funnel: rows surviving each assemble.py join
# --------------------------------------------------------------------------- #
def fig_join_funnel(pd) -> None:
    need = ["nass_pa_yields.csv", "ssurgo_pa_soil.csv", "daymet_pa_weather.csv",
            "openmeteo_pa_soilmoisture.csv", "ndvi_pa.csv"]
    if not all((PROCESSED / n).exists() for n in need):
        return _skip("join_funnel.png", "processed CSVs absent")
    read = lambda n: pd.read_csv(PROCESSED / n, dtype={"county_ansi": "string"}, low_memory=False)
    nass = read("nass_pa_yields.csv")
    steps = [("NASS yield rows", len(nass))]
    base = nass.dropna(subset=["county_ansi"])
    steps.append(("with a county FIPS", len(base)))
    base = base[base["county_ansi"].isin(set(read("ssurgo_pa_soil.csv")["county_ansi"]))]
    steps.append(("+ SSURGO soil", len(base)))
    for label, name in [("+ Daymet weather", "daymet_pa_weather.csv"),
                        ("+ ERA5 soil moisture", "openmeteo_pa_soilmoisture.csv"),
                        ("+ MODIS NDVI", "ndvi_pa.csv")]:
        keys = read(name)[["county_ansi", "year"]].drop_duplicates()
        base = base.merge(keys, on=["county_ansi", "year"], how="inner")
        steps.append((label, len(base)))
    base = base.dropna(subset=["yield_value"])
    steps.append(("non-missing yield", len(base)))

    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ys = list(range(len(steps)))[::-1]
    ax.barh(ys, [n for _, n in steps], color=BLUE, height=0.6)
    for y, (_, n) in zip(ys, steps):
        ax.text(n + 60, y, f"{n:,}", va="center", fontsize=8.5, color=INK_2)
    ax.set_yticks(ys, [s for s, _ in steps])
    ax.set_xlim(0, steps[0][1] * 1.15)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Rows (county × year × crop)")
    ax.set_title("assemble.py join funnel", loc="left")
    _save(fig, "join_funnel.png")


def main() -> int:
    print("make_figures:")
    rows = load_metrics()
    fig_gate(rows)
    fig_corn_history(rows)
    fig_crop_history(rows)
    try:
        import pandas as pd
    except ImportError:
        pd = None
    table_path = PROCESSED / "analysis_county_year.csv"
    table = (pd.read_csv(table_path, dtype={"county_ansi": "string"}, low_memory=False)
             if pd is not None and table_path.exists() else None)
    fig_variance(table)
    if pd is not None:
        fig_join_funnel(pd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
