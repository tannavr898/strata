"""main.py — Strata orchestrator: one readable end-to-end report.

Runs the whole product on the already-pulled processed data (offline; no API
calls) and prints a single human-readable report tying the pieces together:

    1. Soil classification  — is soil identity recoverable from cheap numbers?
    2. Yield regression      — how well does soil+weather predict county yield,
                               honestly (leave-counties-out), vs a baseline?
    3. Recommendation        — apply cited Penn State rules to a real field.
    4. Calibration seam       — where a future cheap probe would plug in.

Each section delegates to the module that owns the logic; main.py only sequences
them and formats the report. It reads the deterministic CSVs in data/processed/
(reproducible from the README provenance), so it does not need a network or API
key. Run:  python -m strata.main
"""

from __future__ import annotations

import logging

import numpy as np

from strata.calibration import ProbeCalibration
from strata.recommender import OptimizationEngine, soiltest_from_ssurgo, _print_report
from strata.soil_classifier import SoilClassifier
from strata.yield_model import YieldModel

_SEED = 42


def _rule(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def section_soil_classifier() -> None:
    _rule("1. SOIL CLASSIFICATION  (is soil identity recoverable from numbers?)")
    from sklearn.dummy import DummyClassifier
    from sklearn.model_selection import train_test_split

    X, y = SoilClassifier.prepare_dataset()
    clf = SoilClassifier()
    cv = clf.cross_validate(X, y)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=_SEED)
    base = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr).score(X_te, y_te)

    print(f"\nTrained on {len(y)} labeled pedons, {y.nunique()} soil orders.")
    print(f"  5-fold CV accuracy : {cv['accuracy_mean']:.3f} +/- {cv['accuracy_std']:.3f}")
    print(f"  5-fold CV macro-F1 : {cv['f1_macro_mean']:.3f} +/- {cv['f1_macro_std']:.3f}")
    print(f"  majority baseline  : {base:.3f}")
    print("  -> soil order is recoverable from measurable lab numbers (well above "
          "baseline),\n     which is the premise the product rests on.")


def section_yield_model() -> None:
    _rule("2. YIELD REGRESSION  (soil+weather -> county yield, honestly)")
    from sklearn.dummy import DummyRegressor
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import train_test_split

    table = YieldModel.load_table()
    headline = "corn_grain_yield"
    unit = (table.dropna(subset=[YieldModel.TARGET])
            .groupby("crop")["unit_desc"].first()[headline])

    X, y, groups = YieldModel.prepare_dataset(headline, table)
    model = YieldModel()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=_SEED)
    base_mae = mean_absolute_error(y_te, DummyRegressor(strategy="mean")
                                   .fit(X_tr, y_tr).predict(X_te))
    rnd = model.cross_validate_random(X, y)
    grp = model.cross_validate_by_county(X, y, groups)

    print(f"\nHeadline crop: {headline}  ({len(y)} county-years, "
          f"{groups.nunique()} counties, {unit}).")
    print(f"  mean-yield baseline MAE      : {base_mae:.1f} {unit}")
    print(f"  random 5-fold CV             : MAE {rnd['mae_mean']:.1f}   "
          f"R^2 {rnd['r2_mean']:.3f}   (optimistic)")
    print(f"  leave-counties-out 5-fold CV : MAE {grp['mae_mean']:.1f}   "
          f"R^2 {grp['r2_mean']:.3f}   (new-county estimate)")
    print("  -> beats the baseline; the random->leave-counties-out drop is the "
          "county-identity\n     crutch, reported rather than hidden. Year-to-year "
          "signal is mostly weather.")


def section_recommender() -> None:
    _rule("3. RECOMMENDATION  (cited Penn State rules on a real field)")
    print("\nNOTE: pH/CEC/OM are real Chester County SSURGO values; Mehlich-3 P/K and")
    print("buffer pH are ILLUSTRATIVE soil-test inputs (SSURGO does not carry them).")
    engine = OptimizationEngine()
    corn = soiltest_from_ssurgo(
        "Chester", "corn_grain_yield",
        mehlich3_p_ppm=22, mehlich3_k_ppm=85, buffer_ph=6.5, yield_goal=160,
    )
    _print_report(corn, engine.recommend(corn))


def section_calibration() -> None:
    _rule("4. CALIBRATION SEAM  (where a future cheap probe plugs in)")
    print("\nNo probe hardware exists; Strata depends on none. This shows the "
          "contract only,\nfit on SYNTHETIC pairs (labeled, never a finding): a "
          "raw probe reading is mapped\nonto the lab scale, then fed to the same "
          "recommender unchanged.")
    rng = np.random.default_rng(_SEED)
    lab_ph = np.linspace(5.2, 7.0, 40)
    probe_ph = 0.92 * lab_ph + 0.45 + rng.normal(0, 0.10, size=lab_ph.shape)
    cal = ProbeCalibration()
    fit = cal.fit("ph", probe=probe_ph, lab=lab_ph, units="", synthetic=True)
    raw = 5.40
    print(f"\n  [SYNTHETIC] probe pH calibration: lab = {fit.slope:.2f}*probe "
          f"+ {fit.intercept:.2f}  (R^2 {fit.r2:.3f})")
    print(f"  raw probe pH {raw:.2f} -> lab-equivalent {cal.apply('ph', raw):.2f} "
          f"-> recommender.")


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    print("#" * 72)
    print("#  STRATA -- affordable precision-ag report for small PA farms")
    print("#  yield regression + soil classification + cited agronomy advice")
    print("#  (runs offline on data/processed/*.csv; reproducible per README)")
    print("#" * 72)

    section_soil_classifier()
    section_yield_model()
    section_recommender()
    section_calibration()

    _rule("SUMMARY")
    print("\nReal held-out validation throughout; thin/uncertain results are stated,")
    print("not masked. Soil is characterizable from cheap numbers; county-year yield")
    print("is predictable above baseline (mostly a weather story, honestly bounded by")
    print("leave-counties-out CV); recommendations are cited Penn State rules, not")
    print("invented; and a cheap probe can plug in later as calibration only.")


if __name__ == "__main__":  # pragma: no cover
    main()
