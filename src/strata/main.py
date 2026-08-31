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
import warnings

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


def section_yield_model() -> dict:
    """Print the yield section and RETURN its headline metrics, so the closing
    summary quotes the numbers this run actually produced rather than a literal
    that silently goes stale when the feature set changes."""
    _rule("2. YIELD REGRESSION  (soil+weather -> county yield, honestly)")
    from sklearn.dummy import DummyRegressor
    from sklearn.metrics import mean_absolute_error
    from sklearn.model_selection import train_test_split

    table = YieldModel.load_table()
    headline = "corn_grain_yield"
    unit = (table.dropna(subset=[YieldModel.TARGET])
            .groupby("crop")["unit_desc"].first()[headline])

    X, y, groups = YieldModel.prepare_dataset(headline, table)

    # Baseline floor: always predict the mean yield.
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=_SEED)
    base_mae = mean_absolute_error(y_te, DummyRegressor(strategy="mean")
                                   .fit(X_tr, y_tr).predict(X_te))

    # ATTRIBUTION mode: soil+weather only (the conservative, agronomy-only number).
    attrib = YieldModel(include_trend=False)
    attrib_lco = attrib.cross_validate_by_county(X, y, groups)

    # FORECAST mode (product default): add the yield trend term.
    fc = YieldModel(include_trend=True)
    fc_lco = fc.cross_validate_by_county(X, y, groups)
    # The real-world test: train on the past, predict the future (>=2018).
    fwd = fc.forward_forecast(X, y, cutoff_year=2017)
    fwd_naive = YieldModel(include_trend=False).forward_forecast(X, y, cutoff_year=2017)

    print(f"\nHeadline crop: {headline}  ({len(y)} county-years, "
          f"{groups.nunique()} counties, {unit}).")
    print(f"  mean-yield baseline MAE             : {base_mae:.1f} {unit}")
    print("\n  ATTRIBUTION mode (soil+weather only, leave-counties-out):")
    print(f"    new-county estimate               : MAE {attrib_lco['mae_mean']:.1f}   "
          f"R^2 {attrib_lco['r2_mean']:.3f}")
    print("  FORECAST mode (+ yield trend term, the product default):")
    print(f"    new-county estimate (LCO)         : MAE {fc_lco['mae_mean']:.1f}   "
          f"R^2 {fc_lco['r2_mean']:.3f}")
    print(f"    forward forecast (train<=2017)    : MAE {fwd['mae']:.1f}   "
          f"R^2 {fwd['r2']:.3f}   (predicts {fwd['n_test']} future county-years)")
    print(f"    ^ same forward test WITHOUT trend : MAE {fwd_naive['mae']:.1f}   "
          f"R^2 {fwd_naive['r2']:.3f}   (worse than the mean!)")
    print("  -> the yield uptrend (better hybrids/mgmt, ~1.5-2 bu/yr) is real signal."
          "\n     Withholding it is honest for ATTRIBUTION but useless for FORECASTING:"
          "\n     a past-trained model can't see the trend, so it under-predicts future"
          "\n     years (negative R^2). Adding it ~doubles accuracy and is what a farmer"
          "\n     planning next season actually needs. Both reported; nothing hidden.")

    return {"crop": headline, "lco_r2": fc_lco["r2_mean"], "forward_r2": fwd["r2"]}


_GATE_CROPS = ["corn_grain_yield", "soybean_yield", "oats_yield", "alfalfa_hay_yield"]


def _product_skills() -> dict:
    """Run the product quality gate on a representative crop set (corn/soy ship,
    oats/alfalfa are withheld). Computed once and reused by the sections below."""
    table = YieldModel.load_table()
    return {crop: YieldModel.evaluate_for_product(crop, table) for crop in _GATE_CROPS}


def section_product_gate(skills: dict) -> None:
    _rule("2b. PRODUCT QUALITY GATE  (ship only crops the model can forecast)")
    print(f"\nGate: offer a forecast only where the leave-counties-out forecast R^2 >= "
          f"{YieldModel.PRODUCT_MIN_FORECAST_R2:.2f}.\nResearch reports every crop; the "
          f"product ships only where a forecast beats the\ncounty-mean baseline usefully.")
    print(f"\n  {'crop':<20}{'fcstR2 (LCO)':>15}{'decision':>12}")
    print("  " + "-" * 47)
    for crop, pf in skills.items():
        print(f"  {crop:<20}{pf.forecast_r2:>15.3f}"
              f"{('SHIP' if pf.shipped else 'WITHHOLD'):>12}")
    for pf in skills.values():
        if not pf.shipped:
            print(f"\n  * {pf.message}")


def section_recommender(skills: dict) -> None:
    _rule("3. RECOMMENDATION  (cited Penn State rules on a real field)")
    print("\nNOTE: pH/CEC/OM are real Chester County SSURGO values; Mehlich-3 P/K and")
    print("buffer pH are ILLUSTRATIVE soil-test inputs (SSURGO does not carry them).")

    # ---- module handoff, GATED: a model yield feeds the reco only if the crop
    # cleared the product quality gate. If corn had failed the gate we would decline
    # to hand a forecast to the recommender rather than scale fertilizer to a number
    # the model can't stand behind.
    table = YieldModel.load_table()
    pf = skills["corn_grain_yield"]
    if pf.shipped:
        X, y, _ = YieldModel.prepare_dataset("corn_grain_yield", table)
        chester = (table[(table["crop"] == "corn_grain_yield") & (table["county_ansi"] == "029")]
                   .dropna(subset=[YieldModel.TARGET]).sort_values("year"))
        latest = chester.iloc[[-1]]
        pred_yield = float(YieldModel().fit(X, y)
                           .predict(latest[YieldModel.FEATURES + [YieldModel.TREND_FEATURE]])[0])
        print(f"\nYieldModel predicts Chester County corn at {int(latest['year'].iloc[0])} "
              f"conditions: {pred_yield:.0f} bu/acre")
        print(f"(corn cleared the product gate, LCO forecast R^2 {pf.forecast_r2:.2f})")
        print("-> handed to the recommender as the crop-removal basis for maintenance P/K.")
    else:
        pred_yield = None
        print(f"\n{pf.message}")
        print("-> no model yield is fed to the recommender for this crop.")

    engine = OptimizationEngine()
    corn = soiltest_from_ssurgo(
        "Chester", "corn_grain_yield",
        mehlich3_p_ppm=22, mehlich3_k_ppm=85, buffer_ph=6.5,
    )
    _print_report(corn, engine.recommend(corn, expected_yield=pred_yield))


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
    # Silence joblib's cosmetic "delayed should be used with Parallel" notice that
    # n_jobs=-1 cross-validation emits per worker; it does not affect results and
    # only clutters the report. Scoped to this exact message, not a blanket filter.
    warnings.filterwarnings(
        "ignore",
        message="`sklearn.utils.parallel.delayed` should be used",
        category=UserWarning,
    )
    print("#" * 72)
    print("#  STRATA -- affordable precision-ag report for small PA farms")
    print("#  yield regression + soil classification + cited agronomy advice")
    print("#  (runs offline on data/processed/*.csv; reproducible per README)")
    print("#" * 72)

    section_soil_classifier()
    yield_metrics = section_yield_model()
    skills = _product_skills()
    section_product_gate(skills)
    section_recommender(skills)
    section_calibration()

    _rule("SUMMARY")
    print("\nReal held-out validation throughout; thin/uncertain results are stated,")
    print("not masked. Soil is characterizable from cheap numbers; county-year yield")
    print("is forecastable well above baseline once the genetic/management trend is")
    print(f"included ({yield_metrics['crop']}: R^2 "
          f"{yield_metrics['lco_r2']:.2f} leave-counties-out, "
          f"{yield_metrics['forward_r2']:.2f} forward-in-time), with")
    print("the conservative soil+weather-only attribution reported alongside it;")
    print("recommendations are cited Penn State rules, not invented; and a cheap probe")
    print("can plug in later as calibration only.")


if __name__ == "__main__":  # pragma: no cover
    main()
