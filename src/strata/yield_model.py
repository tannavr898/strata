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
* `year` is NOT a feature. It would let the model fit the genetic/management yield
  uptrend (better hybrids over time) instead of the soil+weather signal the
  product rests on. Weather already carries each year's actual conditions.
* Imputation lives inside the Pipeline so it is fit per training fold only.

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
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
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
    # That year's growing-season (Apr-Oct) weather features.
    WEATHER_NUMERIC = [
        "gdd_base10", "precip_total_mm", "tmax_mean_c", "tmin_mean_c", "heat_days_30c",
    ]

    NUMERIC_FEATURES = SOIL_NUMERIC + WEATHER_NUMERIC
    CATEGORICAL_FEATURES = SOIL_CATEGORICAL
    FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    TARGET = "yield_value"
    GROUP = "county_ansi"  # leave-counties-out grouping key (never a feature)

    def __init__(self, n_estimators: int = 400, random_state: int = _SEED):
        # Numeric: median-impute (a guard; assemble.py fills these by construction).
        # Categorical: one-hot, ignoring orders/drainage classes unseen in a fold.
        pre = ColumnTransformer(
            transformers=[
                ("num", SimpleImputer(strategy="median"), self.NUMERIC_FEATURES),
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
        X = sub[cls.FEATURES].copy()
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


def _fmt(scores: dict, unit: str) -> str:
    return (f"MAE {scores['mae_mean']:.2f} +/- {scores['mae_std']:.2f} {unit}   "
            f"R^2 {scores['r2_mean']:.3f} +/- {scores['r2_std']:.3f}")


def _feature_names(model: YieldModel) -> list[str]:
    """Recover readable feature names after the ColumnTransformer (one-hot expands
    the two categoricals into one column per observed level)."""
    return list(model.pipeline.named_steps["pre"].get_feature_names_out())


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

    # Held-out split for permutation importance (out-of-sample feature ranking).
    model.fit(X_tr, y_tr)
    print(f"\n=== Permutation importance ({HEADLINE}, held-out test, top 10) ===")
    print("  (drop in R^2 when each feature is shuffled; impurity importance "
          "would inflate\n   high-cardinality features, so we use permutation)")
    perm = permutation_importance(
        model.pipeline, X_te, y_te, n_repeats=10, random_state=_SEED, n_jobs=-1,
        scoring="r2",
    )
    names = _feature_names(model)
    order = np.argsort(perm.importances_mean)[::-1][:10]
    for idx in order:
        print(f"  {names[idx]:<28}{perm.importances_mean[idx]:+.4f} "
              f"+/- {perm.importances_std[idx]:.4f}")

    # ------------------------------------------------- all crops: the honest table
    # One line per crop: the random vs leave-counties-out R^2 side by side, so the
    # generalization gap is visible for every crop, not just the headline.
    print("\n=== Per-crop yield model (sorted by training rows) ===")
    print(f"  {'crop':<20}{'rows':>6}{'cnty':>6}{'unit':>13}"
          f"{'randR2':>9}{'LCO-R2':>9}{'LCO-MAE':>10}")
    print("  " + "-" * 73)
    for crop, n in crops.items():
        if n < 150:  # too few county-years to cross-validate a tree honestly
            continue
        u = crop_units[crop]
        Xc, yc, gc = YieldModel.prepare_dataset(crop, table)
        if gc.nunique() < 5:  # need enough counties for 5 leave-out folds
            continue
        m = YieldModel()
        r = m.cross_validate_random(Xc, yc)
        g = m.cross_validate_by_county(Xc, yc, gc)
        print(f"  {crop:<20}{n:>6}{gc.nunique():>6}{u:>13}"
              f"{r['r2_mean']:>9.3f}{g['r2_mean']:>9.3f}{g['mae_mean']:>10.2f}")

    print("\nReading this table: randR2 is the optimistic random-CV R^2; LCO-R2 / "
          "LCO-MAE\nare the leave-counties-out (new-county) estimates -- the numbers "
          "an unseen farm\nwould actually get. The gap between them is the county-"
          "identity crutch, surfaced.")
