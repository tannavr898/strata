"""soil_classifier.py — predict Soil Taxonomy ORDER from measurable lab properties.

WHAT this module does
---------------------
Trains a RandomForest to classify a soil pedon's taxonomic ORDER (Alfisols,
Ultisols, Inceptisols, ...) from its laboratory-measured physical and chemical
properties — texture, pH, CEC, organic carbon, and how those vary with depth.

WHY it exists (the "rigor" deliverable)
---------------------------------------
This model is NOT the product's recommendation engine. Its job is to demonstrate,
on REAL held-out lab data, that soil identity is recoverable from the same kinds
of cheap, measurable numbers a farmer (or a future soil probe) could obtain. If a
model can name the soil order from sand/silt/clay/pH/CEC/organic-C, then those
numbers genuinely carry soil information — which is the premise the whole product
rests on.

Leakage discipline (council rule)
---------------------------------
* Features are ONLY measured lab properties. `lat`/`lon` are dropped — predicting
  soil order from coordinates would memorize geography, not learn soil science.
* `tax_greatgroup` / `tax_subgroup` are dropped — they are FINER labels of the
  same taxonomy as the target, so using them would leak the answer.
* The train/test split is stratified, and imputation lives INSIDE a Pipeline so
  the imputer is fit on the training fold only (test statistics never leak in).

Granularity note
----------------
Input lab data is layer-level (one row per soil horizon). Soil order is a
WHOLE-PROFILE property, so `engineer_pedon_features` aggregates each pedon's
horizons into one feature row — including depth-contrast features (e.g. clay
increase with depth) that encode the diagnostic subsurface horizons taxonomy
actually keys on.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

logger = logging.getLogger("strata.soil_classifier")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PROCESSED = _REPO_ROOT / "data" / "processed"

# Surface horizon depth (cm) used for "topsoil" features. Matches the 0-25cm
# convention SSURGO aggregation uses elsewhere, so the two soil views are aligned.
_SURFACE_CM = 25.0

# Random seed fixed everywhere for determinism (council rule).
_SEED = 42


class SoilClassifier:
    """RandomForest classifier of Soil Taxonomy order from pedon lab features."""

    # The engineered per-pedon feature columns. Grouped by what they capture:
    #   surf_*  : depth-weighted topsoil (0-25cm) — the epipedon (mollic/histic?).
    #   prof_*  : depth-weighted whole-profile means — overall soil chemistry/texture.
    #   diagnostic: depth CONTRAST features that flag subsurface diagnostic horizons
    #               (clay increase -> argillic; oc_max -> histic; ph spread -> leaching).
    FEATURES = [
        "surf_sand", "surf_silt", "surf_clay", "surf_ph", "surf_cec", "surf_oc",
        "prof_sand", "prof_silt", "prof_clay", "prof_ph", "prof_cec", "prof_oc",
        "clay_max", "clay_increase", "ph_min", "ph_max", "oc_max",
        "solum_depth", "n_horizons",
    ]
    TARGET = "tax_order"

    # Orders too rare to learn OR validate honestly (pedon counts: aridisols 19,
    # vertisols 5, oxisols 1 in this NE/Mid-Atlantic region). Dropped, not faked —
    # a 1-pedon class cannot appear in both a train and a test fold. Decision made
    # HERE (visible), not hidden in ingest.
    RARE_ORDERS = {"aridisols", "vertisols", "oxisols"}

    def __init__(self, n_estimators: int = 400, random_state: int = _SEED):
        # Median imputation handles the documented lab missingness (e.g. CEC ~37%).
        # It sits in the Pipeline so it is fit per training fold (no test leakage).
        self.pipeline = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="median")),
                (
                    "rf",
                    RandomForestClassifier(
                        n_estimators=n_estimators,
                        random_state=random_state,
                        # Balance class weights so big classes (alfisols) don't
                        # swamp the rare-but-valid ones (histosols, mollisols).
                        class_weight="balanced",
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        self.random_state = random_state
        self.classes_: Optional[np.ndarray] = None

    # ----------------------------------------------------------------- data prep
    @staticmethod
    def load_pedons(path: Path | str = _PROCESSED / "kssl_pedons.csv") -> pd.DataFrame:
        """Load the layer-level KSSL lab table written by data_ingest.py."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Expected {path} — run KSSLPedonLoader in data_ingest.py first."
            )
        df = pd.read_csv(path)
        logger.info("Loaded KSSL layers: %d rows, %d pedons",
                    len(df), df["pedon_key"].nunique())
        return df

    @staticmethod
    def _wmean(values: pd.Series, weights: pd.Series) -> float:
        """Weighted mean that ignores NaN values and zero/negative weights.

        Returns NaN if no horizon contributes (later imputed inside the Pipeline).
        """
        mask = values.notna() & (weights > 0)
        if not mask.any():
            return np.nan
        w = weights[mask]
        return float((values[mask] * w).sum() / w.sum())

    @classmethod
    def engineer_pedon_features(cls, layers: pd.DataFrame) -> pd.DataFrame:
        """Aggregate layer-level horizons into one feature row per pedon.

        Each pedon's horizons are collapsed using depth as the weight, plus a few
        depth-contrast features that encode the diagnostic SUBSURFACE horizons soil
        taxonomy keys on. Pedons with no valid horizon geometry are dropped.
        """
        df = layers.copy()

        # Valid horizon geometry is required to weight by depth.
        df = df.dropna(subset=["hzn_top", "hzn_bot"])
        df = df[df["hzn_bot"] > df["hzn_top"]]
        df["thick"] = df["hzn_bot"] - df["hzn_top"]

        # Overlap thickness of each horizon with the 0-25cm surface window.
        df["surf_thick"] = (df["hzn_bot"].clip(upper=_SURFACE_CM) - df["hzn_top"]).clip(lower=0)

        texture_chem = ["sand_pct", "silt_pct", "clay_pct", "ph_h2o", "cec", "organic_c_pct"]

        rows = []
        for pedon_key, g in df.groupby("pedon_key"):
            # Subsoil = horizons starting at/below the surface window; used to detect
            # a clay increase (argillic horizon) relative to the topsoil.
            subsoil = g[g["hzn_top"] >= _SURFACE_CM]
            surf_clay = cls._wmean(g["clay_pct"], g["surf_thick"])
            max_subsoil_clay = subsoil["clay_pct"].max() if not subsoil.empty else np.nan

            rows.append({
                "pedon_key": pedon_key,
                # taxonomy label — first non-null order for the pedon (constant per pedon).
                cls.TARGET: g["tax_order"].dropna().iloc[0] if g["tax_order"].notna().any() else np.nan,
                # surface (0-25cm) depth-weighted features
                "surf_sand": cls._wmean(g["sand_pct"], g["surf_thick"]),
                "surf_silt": cls._wmean(g["silt_pct"], g["surf_thick"]),
                "surf_clay": surf_clay,
                "surf_ph":   cls._wmean(g["ph_h2o"], g["surf_thick"]),
                "surf_cec":  cls._wmean(g["cec"], g["surf_thick"]),
                "surf_oc":   cls._wmean(g["organic_c_pct"], g["surf_thick"]),
                # whole-profile depth-weighted features
                "prof_sand": cls._wmean(g["sand_pct"], g["thick"]),
                "prof_silt": cls._wmean(g["silt_pct"], g["thick"]),
                "prof_clay": cls._wmean(g["clay_pct"], g["thick"]),
                "prof_ph":   cls._wmean(g["ph_h2o"], g["thick"]),
                "prof_cec":  cls._wmean(g["cec"], g["thick"]),
                "prof_oc":   cls._wmean(g["organic_c_pct"], g["thick"]),
                # depth-contrast / diagnostic features
                "clay_max": g["clay_pct"].max(),
                "clay_increase": (max_subsoil_clay - surf_clay)
                                 if pd.notna(max_subsoil_clay) and pd.notna(surf_clay) else np.nan,
                "ph_min": g["ph_h2o"].min(),
                "ph_max": g["ph_h2o"].max(),
                "oc_max": g["organic_c_pct"].max(),
                "solum_depth": g["hzn_bot"].max(),
                "n_horizons": len(g),
            })

        pedons = pd.DataFrame(rows)
        logger.info("Engineered %d pedon feature rows from %d horizons",
                    len(pedons), len(df))
        return pedons

    @classmethod
    def prepare_dataset(cls, layers: Optional[pd.DataFrame] = None
                        ) -> tuple[pd.DataFrame, pd.Series]:
        """Produce the (X, y) modeling matrix: engineer features, drop unlabeled
        and ultra-rare classes. Returns feature frame X and label series y.
        """
        if layers is None:
            layers = cls.load_pedons()
        pedons = cls.engineer_pedon_features(layers)

        # Keep only pedons that carry an order label.
        labeled = pedons.dropna(subset=[cls.TARGET]).copy()
        logger.info("Labeled pedons: %d", len(labeled))

        # Drop ultra-rare orders that cannot be learned or validated.
        before = len(labeled)
        labeled = labeled[~labeled[cls.TARGET].isin(cls.RARE_ORDERS)]
        logger.info("Dropped %d pedons in rare orders %s; %d remain",
                    before - len(labeled), sorted(cls.RARE_ORDERS), len(labeled))

        X = labeled[cls.FEATURES]
        y = labeled[cls.TARGET]
        return X, y

    # -------------------------------------------------------------- fit / predict
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SoilClassifier":
        self.pipeline.fit(X, y)
        self.classes_ = self.pipeline.named_steps["rf"].classes_
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict(X)

    def cross_validate(self, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> dict:
        """Stratified k-fold CV for a robust, non-circular performance estimate.

        Reports accuracy AND macro-F1 — macro-F1 weights every class equally, so a
        model that only nails the big classes can't hide behind raw accuracy.
        """
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        acc = cross_val_score(self.pipeline, X, y, cv=skf, scoring="accuracy", n_jobs=-1)
        f1 = cross_val_score(self.pipeline, X, y, cv=skf, scoring="f1_macro", n_jobs=-1)
        return {
            "accuracy_mean": acc.mean(), "accuracy_std": acc.std(),
            "f1_macro_mean": f1.mean(), "f1_macro_std": f1.std(),
        }


def _print_class_distribution(y: pd.Series) -> None:
    print("\n=== Soil ORDER class distribution (labeled pedons) ===")
    counts = y.value_counts()
    for order, n in counts.items():
        print(f"  {order:<14}{n:>6}  ({100*n/len(y):4.1f}%)")
    print(f"  {'TOTAL':<14}{len(y):>6}")


def _print_confusion(y_true, y_pred, labels) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    short = [l[:5] for l in labels]  # truncate names for a readable grid
    print("\n=== Confusion matrix (rows = true, cols = predicted) ===")
    print(f"{'':<14}" + "".join(f"{s:>7}" for s in short))
    for i, lab in enumerate(labels):
        print(f"{lab:<14}" + "".join(f"{cm[i, j]:>7}" for j in range(len(labels))))


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    X, y = SoilClassifier.prepare_dataset()
    _print_class_distribution(y)

    # Stratified hold-out: the confusion matrix / per-class report come from data
    # the model never saw, so the evaluation is genuinely out-of-sample.
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=_SEED
    )

    clf = SoilClassifier()

    # Cross-validated estimate on the full data (more stable than one split).
    cv = clf.cross_validate(X, y)
    print("\n=== 5-fold stratified cross-validation ===")
    print(f"  accuracy : {cv['accuracy_mean']:.3f} +/- {cv['accuracy_std']:.3f}")
    print(f"  macro-F1 : {cv['f1_macro_mean']:.3f} +/- {cv['f1_macro_std']:.3f}")

    # Baseline: always predict the most frequent order. The model is only
    # meaningful if it clearly beats this (council rule: non-circular evaluation).
    dummy = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)
    print(f"\n  majority-class baseline accuracy: {dummy.score(X_te, y_te):.3f}")

    # Train on the training split, evaluate on the held-out test split.
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)

    print("\n=== Held-out test performance (per class) ===")
    print(classification_report(y_te, y_pred, digits=3, zero_division=0))

    labels = sorted(y.unique())
    _print_confusion(y_te, y_pred, labels)

    # Permutation importance on the held-out set: how much test accuracy drops when
    # each feature is shuffled. Preferred over impurity importance (which inflates
    # high-cardinality features) — it measures real predictive contribution.
    print("\n=== Permutation importance (held-out test, top 10) ===")
    perm = permutation_importance(
        clf.pipeline, X_te, y_te, n_repeats=10, random_state=_SEED, n_jobs=-1
    )
    order = np.argsort(perm.importances_mean)[::-1][:10]
    for idx in order:
        print(f"  {SoilClassifier.FEATURES[idx]:<16}"
              f"{perm.importances_mean[idx]:+.4f} +/- {perm.importances_std[idx]:.4f}")
