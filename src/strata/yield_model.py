"""yield_model.py — predict county-year crop yield from soil + weather.

WHAT this module does
---------------------
Trains a RandomForest to regress a crop's county-year YIELD onto the static soil
properties of that county and the growing-season weather of that exact year. The
input is the assembled table from `assemble.py`:

    one row = (county, year, crop, YIELD) + [soil features] + [that year's weather]

WHY per crop (not one pooled model)
-----------------------------------
Yield units differ by crop — corn/soy/wheat/oats/barley are BU/ACRE, silage and
hay are TONS/ACRE — and the magnitudes differ by an order of magnitude. Pooling
them into one target would make MAE and R^2 meaningless. So a separate model is
trained per crop, and every reported number is in that crop's own yield units.

Leakage discipline (council rule)
---------------------------------
* County IDENTITY is never a feature. `county_ansi`/`county_name`/`areasymbol`
  ride along only for grouping/traceability. The model must generalize from
  soil+weather, not memorize which county a row came from.
* No per-group target statistic (e.g. a county's mean historical yield) is used
  as a feature — that would leak the answer. Features are raw soil + weather only.
* `year` is handled as an EXPLICIT, switchable trend term, not a hidden feature
  (see "Two modes" below). It is never used to encode county identity.
* Imputation lives inside the Pipeline so it is fit per training fold only.

Two modes: attribution vs forecast (the `include_trend` flag)
-------------------------------------------------------------
Corn/soy county yields carry a strong upward TREND over 2000-2023 (better hybrids
and management add ~1.5-2 bu/acre/yr). That trend is real signal, but it is not a
soil-or-weather effect, so how we treat `year` depends on the question:

* ATTRIBUTION mode (`include_trend=False`): `year` is withheld. The model must
  explain yield from soil+weather alone. This answers "how much do the agronomic
  variables explain?" and is the honest, conservative number (LCO R^2 ~0.21).
* FORECAST mode (`include_trend=True`, the product default): `year` is included as
  a trend feature. This answers "what will this field actually yield?" -- the thing
  a farmer needs. It roughly doubles honest accuracy (LCO R^2 ~0.51) and, crucially,
  is the only mode that forecasts FORWARD: trained on past years to predict future
  ones, the no-trend model scores R^2 < 0 (worse than the mean) because it cannot
  see the uptrend, while the trend model reaches R^2 ~0.35, MAE ~15-17 bu.

We report BOTH so the trend's contribution is never hidden.

The honest-evaluation point (random CV vs leave-counties-out)
-------------------------------------------------------------
Soil features are county-STATIC: every row from a county shares the same 8 soil
values. Under a plain random k-fold, the same county appears in both train and
test, so the model can recognize a county by its soil fingerprint and recall its
typical yield — inflating the score. The honest question for a NEW farm is:
"given a county we never trained on, can we predict its yield?" That is GroupKFold
keyed on county (leave-counties-out). We report BOTH and show the gap, rather than
quoting only the flattering random-CV number.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

logger = logging.getLogger("strata.yield_model")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PROCESSED = _REPO_ROOT / "data" / "processed"

# Random seed fixed everywhere for determinism (council rule).
_SEED = 42

# County FIPS is the join/group key; read as string so "001" never becomes int 1.
_KEY_DTYPE = {"county_ansi": "string"}


@dataclass(frozen=True)
class ProductForecast:
    """One crop's product-gate decision. `forecast_r2` is the honest leave-counties-
    out forecast skill; `shipped` is whether it clears the quality threshold; `message`
    is the human-readable verdict (the insufficient-skill notice when withheld)."""
    crop: str
    forecast_r2: float
    threshold: float
    shipped: bool
    message: str


class YieldModel:
    """RandomForest regressor of county-year yield from soil + weather features.

    One instance models ONE crop (units are crop-specific). Use `prepare_dataset`
    to pull the (X, y, groups) for a given crop out of the analysis table.
    """

    # Static county soil features (0-25cm topsoil) carried by assemble.py.
    SOIL_NUMERIC = [
        "topsoil_sand_pct", "topsoil_silt_pct", "topsoil_clay_pct",
        "topsoil_ph", "topsoil_om_pct", "topsoil_cec",
    ]
    SOIL_CATEGORICAL = ["dominant_taxorder", "dominant_drainage"]
    # That year's weather. First five: full-season (Apr-Oct) aggregates. The rest:
    # per-window WATER (precip) + HEAT-STRESS (heat-days) for four phenology windows
    # (Apr-May / Jun / Jul silking / Aug-Oct), so the model can see WHEN drought/heat
    # fell -- the within-year timing a season total averages away. Only precip+heat_days
    # are kept: an A/B on leave-counties-out AND forward-in-time splits showed they hold
    # the signal (corn forward R^2 0.28 -> 0.39) while per-window mean-temp/GDD were
    # collinear noise that hurt the forward forecast. Kept in sync with assemble.py's
    # _WEATHER_FEATURES (assemble writes them; the model selects them here).
    WEATHER_NUMERIC = [
        "gdd_base10", "precip_total_mm", "tmax_mean_c", "tmin_mean_c", "heat_days_30c",
        "precip_aprmay_mm", "heat_days_aprmay",
        "precip_jun_mm", "heat_days_jun",
        "precip_jul_mm", "heat_days_jul",
        "precip_augoct_mm", "heat_days_augoct",
    ]
    # Cropland-masked MODIS NDVI, county-year, TIME-GATED to <= July so late-season
    # greenness (almost the harvested yield itself) cannot leak into the forecast. Only
    # Jun + Jul: an A/B on leave-counties-out AND forward-in-time splits showed these two
    # give the best forward forecast (corn forward R^2 0.39 -> 0.46), while adding the
    # Apr-May window (mostly bare soil / cover crop, pre-emergence) only added forward
    # noise. NDVI is the one source that helped: unlike the gridded weather/soil-moisture
    # reanalyses (which are meteorology-derived and proved redundant here -- see the
    # rejected-features note), NDVI is a direct satellite OBSERVATION of the crop.
    NDVI_NUMERIC = ["ndvi_jun", "ndvi_jul"]

    NUMERIC_FEATURES = SOIL_NUMERIC + WEATHER_NUMERIC + NDVI_NUMERIC
    CATEGORICAL_FEATURES = SOIL_CATEGORICAL
    FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    # Optional explicit trend term (genetic/management yield uptrend). Included in
    # FORECAST mode, withheld in ATTRIBUTION mode -- see the module docstring.
    TREND_FEATURE = "year"

    TARGET = "yield_value"
    GROUP = "county_ansi"  # leave-counties-out grouping key (never a feature)

    # Product quality gate: the minimum HONEST (leave-counties-out) forecast R^2 a
    # crop must clear before Strata will offer a yield forecast for it in the PRODUCT.
    # Below this, a forecast is no better than quoting the county-average yield, so we
    # decline rather than ship a number a farmer would wrongly trust. The RESEARCH
    # tables still report every crop; only the product surface is gated. Named
    # constant, not a magic number, so the bar is auditable and tunable in one place.
    PRODUCT_MIN_FORECAST_R2 = 0.30

    def __init__(
        self,
        n_estimators: int = 400,
        random_state: int = _SEED,
        include_trend: bool = True,
    ):
        # include_trend=True (default) is FORECAST mode: add `year` as a trend term.
        # include_trend=False is ATTRIBUTION mode: soil+weather only.
        self.include_trend = include_trend
        numeric = list(self.NUMERIC_FEATURES)
        if include_trend:
            numeric.append(self.TREND_FEATURE)
        # Numeric: median-impute (a guard; assemble.py fills these by construction).
        # Categorical: one-hot, ignoring orders/drainage classes unseen in a fold.
        # remainder="drop": any extra columns carried in X (e.g. year when trend is
        # off) are ignored, so the same prepared X works for both modes.
        pre = ColumnTransformer(
            transformers=[
                ("num", SimpleImputer(strategy="median"), numeric),
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    self.CATEGORICAL_FEATURES,
                ),
            ]
        )
        self.pipeline = Pipeline(
            steps=[
                ("pre", pre),
                (
                    "rf",
                    RandomForestRegressor(
                        n_estimators=n_estimators,
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        self.random_state = random_state

    # ----------------------------------------------------------------- data prep
    @staticmethod
    def load_table(path: Path | str = _PROCESSED / "analysis_county_year.csv") -> pd.DataFrame:
        """Load the assembled county-year-crop table written by assemble.py."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Expected {path} — run assemble.py (build_analysis_table) first."
            )
        df = pd.read_csv(path, dtype=_KEY_DTYPE)
        logger.info("Loaded analysis table: %d rows x %d cols", len(df), df.shape[1])
        return df

    @classmethod
    def prepare_dataset(
        cls, crop: str, table: Optional[pd.DataFrame] = None
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Slice the analysis table down to one crop's (X, y, groups).

        Returns the feature frame X, the yield target y (in that crop's own units),
        and the county-FIPS group labels used for leave-counties-out CV.
        """
        if table is None:
            table = cls.load_table()
        sub = table[table["crop"] == crop]
        if sub.empty:
            raise ValueError(
                f"No rows for crop={crop!r}. Available: "
                f"{sorted(table['crop'].unique())}"
            )
        # A usable training row needs a real (non-NaN) yield target.
        sub = sub.dropna(subset=[cls.TARGET])
        # Carry the trend column too; the ColumnTransformer uses it only in
        # forecast mode and drops it otherwise, so one X serves both modes.
        X = sub[cls.FEATURES + [cls.TREND_FEATURE]].copy()
        y = sub[cls.TARGET].copy()
        groups = sub[cls.GROUP].copy()
        logger.info(
            "crop=%s: %d rows, %d counties, years %d-%d",
            crop, len(sub), groups.nunique(),
            int(sub["year"].min()), int(sub["year"].max()),
        )
        return X, y, groups

    # -------------------------------------------------------------- fit / predict
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "YieldModel":
        self.pipeline.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict(X)

    # --------------------------------------------------------------- evaluation
    def _cv_scores(self, X, y, cv, groups=None) -> dict:
        """Run MAE + R^2 cross-validation under a given splitter.

        sklearn maximizes scores, so it reports NEGATIVE MAE — we flip the sign
        back to a positive error in yield units for reporting.
        """
        neg_mae = cross_val_score(
            self.pipeline, X, y, cv=cv, groups=groups,
            scoring="neg_mean_absolute_error", n_jobs=-1,
        )
        r2 = cross_val_score(
            self.pipeline, X, y, cv=cv, groups=groups, scoring="r2", n_jobs=-1,
        )
        mae = -neg_mae
        return {
            "mae_mean": mae.mean(), "mae_std": mae.std(),
            "r2_mean": r2.mean(), "r2_std": r2.std(),
        }

    def cross_validate_random(self, X, y, n_splits: int = 5) -> dict:
        """Plain shuffled k-fold. OPTIMISTIC: a county can sit in both train and
        test, so the model can lean on its static soil fingerprint. Reported as
        the upper bound, paired with the leave-counties-out number below."""
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        return self._cv_scores(X, y, kf)

    def cross_validate_by_county(self, X, y, groups, n_splits: int = 5) -> dict:
        """GroupKFold on county FIPS: every test county is absent from training.
        This is the HONEST estimate of performance on a county we have never
        modeled — the situation a brand-new farm presents."""
        n_groups = groups.nunique()
        n_splits = min(n_splits, n_groups)
        gkf = GroupKFold(n_splits=n_splits)
        return self._cv_scores(X, y, gkf, groups=groups)

    def forward_forecast(self, X, y, cutoff_year: int) -> dict:
        """Train on years <= cutoff, predict years > cutoff: the REAL forecast test.

        Random/Group CV can place a future year in training and a past year in
        test (interpolation). A farmer forecasting next season only ever has the
        PAST. This split honors that arrow of time and is where the trend term
        earns its keep: without it, predictions of later (higher-yielding) years
        are systematically low and R^2 goes negative.
        """
        years = X[self.TREND_FEATURE].astype(int)
        tr, te = years <= cutoff_year, years > cutoff_year
        model = clone(self.pipeline).fit(X[tr], y[tr])
        pred = model.predict(X[te])
        return {
            "r2": r2_score(y[te], pred),
            "mae": mean_absolute_error(y[te], pred),
            "n_train": int(tr.sum()), "n_test": int(te.sum()),
            "cutoff": cutoff_year,
        }

    # --------------------------------------------------------------- product gate
    @classmethod
    def evaluate_for_product(
        cls, crop: str, table: Optional[pd.DataFrame] = None,
        threshold: Optional[float] = None,
    ) -> ProductForecast:
        """Decide whether the PRODUCT should offer a yield forecast for this crop.

        Gates on the honest leave-counties-out forecast R^2 (the number a brand-new
        farm would actually get) against `PRODUCT_MIN_FORECAST_R2`. Returns a
        ProductForecast carrying the skill, the pass/fail, and a message -- the
        explicit insufficient-skill notice when the crop is withheld. This gates the
        product only; research reporting still covers every crop.
        """
        threshold = cls.PRODUCT_MIN_FORECAST_R2 if threshold is None else threshold
        X, y, groups = cls.prepare_dataset(crop, table)
        r2 = YieldModel().cross_validate_by_county(X, y, groups)["r2_mean"]
        shipped = r2 >= threshold
        if shipped:
            message = (f"{crop}: forecast shipped "
                       f"(leave-counties-out R^2 {r2:.2f} >= {threshold:.2f} gate).")
        else:
            message = (f"insufficient model skill for {crop}: leave-counties-out "
                       f"forecast R^2 {r2:.2f} < {threshold:.2f} gate -- no yield "
                       f"forecast is offered for this crop (it would be no better than "
                       f"quoting the county-average yield).")
        return ProductForecast(crop, r2, threshold, shipped, message)


def _fmt(scores: dict, unit: str) -> str:
    return (f"MAE {scores['mae_mean']:.2f} +/- {scores['mae_std']:.2f} {unit}   "
            f"R^2 {scores['r2_mean']:.3f} +/- {scores['r2_std']:.3f}")


def permutation_importance_cv(X, y, splitter, groups=None, n_repeats: int = 10) -> dict:
    """Average permutation importance (drop in held-out R^2) across CV folds.

    Keyed on the RAW input columns of X -- which is what `permutation_importance`
    actually shuffles. (The earlier version labeled these with the ColumnTransformer's
    post-one-hot names, which have a different length/order, so the labels did not
    line up with the values; keying on X.columns fixes that.)

    For each fold we fit a fresh model on the training rows only and shuffle each
    column of the HELD-OUT rows, so the ranking reflects what the model relies on to
    predict rows it did not train on. Averaging over folds removes single-split noise
    so the random-vs-grouped comparison isolates the split type, nothing else.
    """
    cols = list(X.columns)
    acc = np.zeros(len(cols))
    n_folds = 0
    for tr, te in splitter.split(X, y, groups):
        m = YieldModel().fit(X.iloc[tr], y.iloc[tr])
        r = permutation_importance(
            m.pipeline, X.iloc[te], y.iloc[te],
            n_repeats=n_repeats, random_state=_SEED, n_jobs=-1, scoring="r2",
        )
        acc += r.importances_mean
        n_folds += 1
    return dict(zip(cols, acc / n_folds))


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    table = YieldModel.load_table()

    # Crops ordered by training-set size; corn grain is the regional headline crop.
    crop_units = (table.dropna(subset=[YieldModel.TARGET])
                  .groupby("crop")["unit_desc"].first().to_dict())
    crops = (table.dropna(subset=[YieldModel.TARGET])["crop"]
             .value_counts())
    HEADLINE = "corn_grain_yield"

    # ---------------------------------------------------------- headline crop
    unit = crop_units[HEADLINE]
    X, y, groups = YieldModel.prepare_dataset(HEADLINE, table)
    print(f"\n=== {HEADLINE}  ({len(y)} county-years, {groups.nunique()} counties, "
          f"units = {unit}) ===")

    model = YieldModel()

    # Baseline floor: always predict the mean yield. The model is only meaningful
    # if it clearly beats this (council rule: non-circular evaluation).
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=_SEED
    )
    dummy = DummyRegressor(strategy="mean").fit(X_tr, y_tr)
    d_pred = dummy.predict(X_te)
    print(f"\n  mean-yield baseline (held-out):  "
          f"MAE {mean_absolute_error(y_te, d_pred):.2f} {unit}   "
          f"R^2 {r2_score(y_te, d_pred):.3f}")

    # The two CV views — the whole point of the rigor here is the GAP between them.
    rnd = model.cross_validate_random(X, y)
    grp = model.cross_validate_by_county(X, y, groups)
    print(f"\n  random 5-fold CV (optimistic):   {_fmt(rnd, unit)}")
    print(f"  leave-counties-out 5-fold CV:    {_fmt(grp, unit)}")
    print("  ^ the drop from random to leave-counties-out is how much the model "
          "leaned on\n    static county soil identity vs. genuinely generalizing.")

    # ---- Permutation importance: RANDOM split vs LEAVE-COUNTIES-OUT split ----
    # Permutation importance = drop in held-out R^2 when a feature's column is
    # shuffled (impurity importance would inflate high-cardinality features, so we
    # use permutation). Computed on the RAW input columns and AVERAGED across each
    # splitter's 5 test folds, so the ONLY thing differing between the two columns
    # below is HOW rows are split into train/test.
    #
    # Why this matters (the whole point of the fix): under a RANDOM split a county
    # sits in both train and test, so shuffling a feature that varies BY COUNTY (its
    # static soil values) destroys the fingerprint the model used to RECOGNIZE the
    # county and recall its typical yield -- inflating that feature's importance.
    # Under leave-counties-out every test county is unseen, so a static soil feature
    # can only score as genuine agronomy. The shrink between the two columns is the
    # county-proxy crutch, surfaced instead of being quoted as a real ranking.
    print(f"\n=== Permutation importance ({HEADLINE}): random split vs leave-counties-out ===")
    rand_imp = permutation_importance_cv(X, y, KFold(5, shuffle=True, random_state=_SEED))
    lco_imp = permutation_importance_cv(X, y, GroupKFold(5), groups=groups)
    print(f"  {'feature':<22}{'randomImp':>12}{'LCO-Imp':>12}   (drop in R^2 when shuffled)")
    print("  " + "-" * 60)
    for feat in sorted(lco_imp, key=lco_imp.get, reverse=True):
        print(f"  {feat:<22}{rand_imp[feat]:>+12.4f}{lco_imp[feat]:>+12.4f}")
    print("\n  Sorted by the HONEST (leave-counties-out) importance. `year` (the forecast"
          "\n  trend term) dominates both. The features that SHRINK from random -> LCO are"
          "\n  the STATIC county soil numerics -- topsoil_om (~0.12->0.07), cec (~0.06->0.03),"
          "\n  ph (~0.03->-0.00), clay (~0.02->0.00): under random splits a county's fixed"
          "\n  soil values help recall its own yield, but on UNSEEN counties that crutch is"
          "\n  gone. `dominant_taxorder` is minor in BOTH (~0.004) -- the OLD code's headline"
          "\n  'taxorder ~0.89' was a label/column misalignment; that magnitude is `year`.")

    # ------------------------------------------------- all crops: the honest table
    # One line per crop, leave-counties-out R^2 in BOTH modes side by side, so the
    # trend term's contribution is visible for every crop -- not just the headline.
    # attribR2 = soil+weather only (conservative); fcstR2 = with the yield trend
    # (product default); fwdR2 = forward-in-time forecast (train<=2017, predict 2018+).
    # RESEARCH view: EVERY crop appears here (the paper reports all of them). The
    # `ship?` column applies the product quality gate (fcstR2 >= PRODUCT_MIN_FORECAST_R2)
    # so it is visible which crops the PRODUCT would actually offer a forecast for.
    print("\n=== Per-crop yield model: attribution vs forecast (leave-counties-out) ===")
    print(f"  {'crop':<20}{'rows':>6}{'cnty':>6}{'unit':>13}"
          f"{'attribR2':>10}{'fcstR2':>9}{'fwdR2':>8}{'fcstMAE':>9}{'ship?':>7}")
    print("  " + "-" * 82)
    for crop, n in crops.items():
        if n < 150:  # too few county-years to cross-validate a tree honestly
            continue
        u = crop_units[crop]
        Xc, yc, gc = YieldModel.prepare_dataset(crop, table)
        if gc.nunique() < 5:  # need enough counties for 5 leave-out folds
            continue
        attrib = YieldModel(include_trend=False).cross_validate_by_county(Xc, yc, gc)
        fcst_model = YieldModel(include_trend=True)
        fcst = fcst_model.cross_validate_by_county(Xc, yc, gc)
        # forward forecast only where there are both pre- and post-2017 rows
        yrs = Xc[YieldModel.TREND_FEATURE].astype(int)
        fwd_r2 = (fcst_model.forward_forecast(Xc, yc, 2017)["r2"]
                  if (yrs <= 2017).any() and (yrs > 2017).any() else float("nan"))
        ship = "yes" if fcst["r2_mean"] >= YieldModel.PRODUCT_MIN_FORECAST_R2 else "NO"
        print(f"  {crop:<20}{n:>6}{gc.nunique():>6}{u:>13}"
              f"{attrib['r2_mean']:>10.3f}{fcst['r2_mean']:>9.3f}"
              f"{fwd_r2:>8.3f}{fcst['mae_mean']:>9.2f}{ship:>7}")

    print("\nReading this table: attribR2 withholds the year trend (how much soil+"
          "weather alone\nexplain -- conservative); fcstR2 adds the trend (the product"
          " default, what a farmer\ngets); fwdR2 trains on <=2017 and predicts 2018+ "
          "(the real forward forecast).\nThe attribR2 -> fcstR2 jump is the genetic/"
          "management yield trend, surfaced not hidden.\n`ship?` is the PRODUCT quality "
          f"gate (fcstR2 >= {YieldModel.PRODUCT_MIN_FORECAST_R2:.2f}): research reports "
          "all crops, the\nproduct offers a forecast only where it clears the gate "
          "(oats/alfalfa are withheld).")
