"""report.py — render the Strata diagnostic dashboard to a PNG.

A picture of what the modules prove, for people who read charts faster than logs.
Runs offline on data/processed/*.csv (same data as main.py) and writes a 4-panel
figure to data/processed/strata_dashboard.png:

  A. Yield forecast, predicted vs actual (leave-counties-out, forecast mode).
  B. The accuracy ladder: baseline -> attribution -> forecast -> forward-in-time,
     including the negative-R^2 "no trend" forward bar that motivates the trend.
  C. Soil classifier confusion matrix (diagonal = correct).
  D. Calibration seam (SYNTHETIC probe->lab fit; labeled, never a finding).

Run:  python -m strata.report   (use Agg backend; no display needed)
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")  # file-only backend; works headless
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, mean_absolute_error, r2_score
from sklearn.model_selection import (GroupKFold, KFold, StratifiedKFold,
                                     cross_val_predict, cross_val_score)
from sklearn.dummy import DummyRegressor

from strata.calibration import ProbeCalibration
from strata.soil_classifier import SoilClassifier
from strata.yield_model import YieldModel

logger = logging.getLogger("strata.report")
_SEED = 42
_REPO_ROOT = Path(__file__).resolve().parents[2]
_OUT = _REPO_ROOT / "data" / "processed" / "strata_dashboard.png"
_GREEN, _BLUE, _GREY, _RED = "#2a7", "#7ac", "#bbb", "#c63"


def _panel_yield_scatter(ax, X, y, groups) -> None:
    """A. Forecast-mode predicted vs actual, leave-counties-out (honest)."""
    model = YieldModel(include_trend=True)
    gkf = GroupKFold(n_splits=5)
    pred = cross_val_predict(model.pipeline, X, y, cv=gkf, groups=groups, n_jobs=-1)
    ax.scatter(y, pred, s=10, alpha=0.35, color=_GREEN)
    lo, hi = y.min(), y.max()
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="perfect prediction")
    ax.set_xlabel("ACTUAL corn yield (bu/acre)")
    ax.set_ylabel("PREDICTED yield (bu/acre)")
    ax.set_title(f"A. Yield forecast, leave-counties-out\n"
                 f"R^2={r2_score(y, pred):.2f}, MAE={mean_absolute_error(y, pred):.1f} bu "
                 f"(forecast mode, new county)", fontsize=10)
    ax.legend(fontsize=8)


def _panel_accuracy_ladder(ax, X, y, groups) -> None:
    """B. Baseline -> attribution -> forecast -> forward, the honest ladder."""
    kf = KFold(5, shuffle=True, random_state=_SEED)
    base = cross_val_score(DummyRegressor(strategy="mean"), X, y, cv=kf,
                           scoring="r2").mean()
    attrib = YieldModel(include_trend=False).cross_validate_by_county(X, y, groups)["r2_mean"]
    fc_model = YieldModel(include_trend=True)
    fcst = fc_model.cross_validate_by_county(X, y, groups)["r2_mean"]
    fwd = fc_model.forward_forecast(X, y, 2017)["r2"]
    fwd_naive = YieldModel(include_trend=False).forward_forecast(X, y, 2017)["r2"]

    labels = ["mean\nbaseline", "attribution\nLCO\n(soil+wx)",
              "forecast\nLCO\n(+trend)", "forecast\nforward\n(+trend)",
              "forward\nNO trend"]
    vals = [base, attrib, fcst, fwd, fwd_naive]
    colors = [_GREY, _BLUE, _GREEN, _GREEN, _RED]
    bars = ax.bar(labels, vals, color=colors)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("R^2  (higher = better)")
    ax.set_title("B. Accuracy ladder: the trend term is the biggest lift\n"
                 "(NDVI now lifts the no-trend forward forecast above the mean)", fontsize=10)
    for b, v in zip(bars, vals):
        off = 0.02 if v >= 0 else -0.06
        ax.text(b.get_x() + b.get_width() / 2, v + off, f"{v:.2f}",
                ha="center", fontsize=9)


def _panel_confusion(ax) -> None:
    """C. Soil classifier confusion matrix (normalized; diagonal = correct)."""
    Xc, yc = SoilClassifier.prepare_dataset()
    labels = sorted(yc.unique())
    skf = StratifiedKFold(5, shuffle=True, random_state=_SEED)
    pred = cross_val_predict(SoilClassifier().pipeline, Xc, yc, cv=skf, n_jobs=-1)
    cm = confusion_matrix(yc, pred, labels=labels, normalize="true")
    ax.imshow(cm, cmap="Greens", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([l[:5] for l in labels], rotation=45, fontsize=8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels([l[:5] for l in labels], fontsize=8)
    ax.set_xlabel("predicted order")
    ax.set_ylabel("true order")
    ax.set_title("C. Soil classifier confusion\n(diagonal = correct; soil IS "
                 "readable from numbers)", fontsize=10)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j] > 0.02:
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                        fontsize=7, color="white" if cm[i, j] > 0.5 else "black")


def _panel_calibration(ax) -> None:
    """D. Synthetic probe->lab calibration seam (labeled; never a finding)."""
    rng = np.random.default_rng(_SEED)
    lab = np.linspace(5.2, 7.0, 40)
    probe = 0.92 * lab + 0.45 + rng.normal(0, 0.10, size=lab.shape)
    cal = ProbeCalibration()
    fit = cal.fit("ph", probe, lab, units="", synthetic=True)
    ax.scatter(probe, lab, s=18, alpha=0.6, color=_RED, label="synthetic probe vs lab")
    xs = np.linspace(probe.min(), probe.max(), 50)
    ax.plot(xs, fit.slope * xs + fit.intercept, "k-", lw=1.5,
            label=f"fit: lab={fit.slope:.2f}*probe+{fit.intercept:.2f}")
    ax.plot(xs, xs, "b:", lw=1, label="uncalibrated (y=x)")
    ax.set_xlabel("cheap probe pH reading")
    ax.set_ylabel("lab pH (truth)")
    ax.set_title(f"D. Calibration seam [SYNTHETIC]\nR^2={fit.r2:.2f} -- corrects probe "
                 "bias to lab scale", fontsize=10)
    ax.legend(fontsize=7)


def build_dashboard(out: Path | str = _OUT, crop: str = "corn_grain_yield") -> Path:
    """Render and save the 4-panel dashboard; returns the output path."""
    warnings.filterwarnings("ignore", category=UserWarning)
    X, y, groups = YieldModel.prepare_dataset(crop)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle("Strata -- offline diagnostic dashboard (real PA data, no hardware)",
                 fontsize=14, fontweight="bold")
    _panel_yield_scatter(axes[0, 0], X, y, groups)
    _panel_accuracy_ladder(axes[0, 1], X, y, groups)
    _panel_confusion(axes[1, 0])
    _panel_calibration(axes[1, 1])
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    path = build_dashboard()
    print(f"Saved dashboard -> {path}")
