"""recommender.py — cited Penn State agronomy rule engine (NOT learned).

WHAT this module does
---------------------
Given a field's measured soil test and the crop being grown, it emits soil
adjustment recommendations (lime, phosphate, potash) by applying EXISTING,
published Penn State agronomy rules. Every recommendation carries the threshold
it crossed and a citation to the source rule. Nothing here is learned from data
and nothing is invented — the engine is a transparent lookup over thresholds an
agronomist can audit line by line.

WHY a rule engine (and not a model)
-----------------------------------
Fertility and lime recommendations are a SOLVED, regulated agronomy problem with
decades of calibration trials behind the published thresholds. Re-learning them
from our county-year yield table would be both circular and less trustworthy than
the extension guidance farmers and agronomists already rely on. The product's job
here is to APPLY trusted rules to a specific field and show its work, not to
reinvent soil fertility science.

PROVENANCE AND LIMITS (read this before trusting a number)
----------------------------------------------------------
The thresholds in `CITATIONS` / the rule constants below are encoded from Penn
State Extension guidance (the Penn State Agronomy Guide and Agricultural
Analytical Services Lab, AASL, interpretation categories). They are reproduced
here as an auditable table, NOT presented as infallible: editions change, and the
exact tabulated lime rate on a real AASL report comes from a buffer-pH lookup this
engine approximates. **Confirm any rate against the current AASL report before
field application.** Each constant names its source so a wrong value is correctable
in one place rather than hidden in arithmetic.

INPUT HONESTY (granularity / provenance, council rule)
------------------------------------------------------
SSURGO (our `ssurgo_pa_soil.csv`) provides pH, CEC, organic matter and texture,
but it does NOT carry Mehlich-3 P/K or a buffer pH — those come only from an
actual lab soil test of the specific field. So the P and K rules REQUIRE a real
soil-test input; the demo at the bottom supplies illustrative P/K values and
labels them as such. A county SSURGO summary is coarse county context, never a
field-level soil test — exactly the gap the future `calibration` probe layer fills.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger("strata.recommender")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PROCESSED = _REPO_ROOT / "data" / "processed"


# --------------------------------------------------------------------------- #
# Cited sources. Each rule below references one of these by key so every printed
# recommendation can name exactly where its threshold came from.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Citation:
    key: str
    source: str   # the publication
    rule: str     # the specific rule / threshold quoted from it


CITATIONS = {
    "ph_target": Citation(
        "ph_target",
        "Penn State Agronomy Guide, 'Soil Acidity and Liming'",
        "Maintain soil pH at ~6.0 for most agronomic grain/forage crops; legumes "
        "(alfalfa, clover) require ~6.5-6.8. Below the crop target, lime is advised.",
    ),
    "lime_buffer": Citation(
        "lime_buffer",
        "Penn State AASL lime requirement (Mehlich buffer pH method)",
        "Lime rate is set from the Mehlich buffer pH, not water pH alone; a lower "
        "buffer pH means more reserve acidity and more lime. Rate here is an "
        "approximation of that table -- confirm against the AASL report.",
    ),
    "p_mehlich3": Citation(
        "p_mehlich3",
        "Penn State AASL Mehlich-3 phosphorus interpretation",
        "Mehlich-3 P: below optimum <30 ppm (yield response to added P likely); "
        "optimum 30-50 ppm (maintenance only); above optimum 50-200 ppm (no P "
        "needed); excessive >200 ppm (environmental P-loss / P-Index concern).",
    ),
    "k_mehlich3": Citation(
        "k_mehlich3",
        "Penn State AASL Mehlich-3 potassium interpretation",
        "Mehlich-3 K: below optimum <100 ppm (response to added K likely); optimum "
        "~100-130 ppm (maintenance); above optimum >130 ppm (no K needed). Verify "
        "thresholds against the current AASL edition before field use.",
    ),
    "removal": Citation(
        "removal",
        "Penn State / IPNI crop nutrient removal coefficients",
        "Maintenance fertilizer replaces what the harvested crop removes: corn "
        "grain ~0.37 lb P2O5 and ~0.27 lb K2O per bushel; soybean ~0.80 lb P2O5 "
        "and ~1.30 lb K2O per bushel.",
    ),
}


# --------------------------------------------------------------------------- #
# Inputs / outputs
# --------------------------------------------------------------------------- #
@dataclass
class SoilTest:
    """One field's measured soil test for one crop.

    pH / CEC / organic matter can come from SSURGO county context, but
    `mehlich3_p_ppm`, `mehlich3_k_ppm`, and `buffer_ph` come ONLY from a real lab
    soil test of the field. Any of them may be None; rules that need a missing
    value say so instead of guessing.
    """
    crop: str
    ph: Optional[float] = None
    buffer_ph: Optional[float] = None
    mehlich3_p_ppm: Optional[float] = None
    mehlich3_k_ppm: Optional[float] = None
    cec: Optional[float] = None
    om_pct: Optional[float] = None
    yield_goal: Optional[float] = None   # crop-specific units (bu/acre etc.)
    # provenance only — never used in a calculation:
    county_name: Optional[str] = None
    note: Optional[str] = None


@dataclass
class Recommendation:
    category: str            # "Lime", "Phosphate (P2O5)", "Potash (K2O)"
    severity: str            # deficiency | maintenance | ok | environmental | unknown
    measured: str            # what was observed, e.g. "soil pH 5.72"
    finding: str             # the interpretation against the threshold
    action: str              # what to do
    amount: Optional[str]    # quantified rate where defensible, else None
    citation: Citation


# --------------------------------------------------------------------------- #
# The engine
# --------------------------------------------------------------------------- #
class OptimizationEngine:
    """Applies cited Penn State rules to a `SoilTest` and returns recommendations.

    Every threshold lives here as a named constant tied to a Citation, so the
    rule set is one auditable surface rather than scattered magic numbers.
    """

    # Crop target pH (lower bound to maintain). Legumes need it higher; everything
    # else defaults to the general agronomic 6.0. Keys match our NASS crop labels.
    DEFAULT_TARGET_PH = 6.0
    LEGUME_TARGET_PH = 6.6
    LEGUME_CROPS = {"alfalfa_hay_yield", "hay_yield"}  # forage legume stands

    # Mehlich-3 P interpretation breakpoints (ppm). See CITATIONS["p_mehlich3"].
    P_BELOW_OPTIMUM = 30.0
    P_OPTIMUM_HIGH = 50.0
    P_EXCESSIVE = 200.0

    # Mehlich-3 K interpretation breakpoints (ppm). See CITATIONS["k_mehlich3"].
    K_BELOW_OPTIMUM = 100.0
    K_OPTIMUM_HIGH = 130.0

    # Crop nutrient removal (lb of nutrient per unit yield) for maintenance rates.
    # See CITATIONS["removal"]. Only crops with well-established coefficients are
    # listed; others get a cited interpretation without a numeric maintenance rate.
    REMOVAL = {
        # crop:              (lb P2O5 / unit, lb K2O / unit, unit label)
        "corn_grain_yield":  (0.37, 0.27, "bu"),
        "soybean_yield":     (0.80, 1.30, "bu"),
    }

    def target_ph(self, crop: str) -> float:
        return self.LEGUME_TARGET_PH if crop in self.LEGUME_CROPS else self.DEFAULT_TARGET_PH

    # --------------------------------------------------------------- lime / pH
    def _lime_rule(self, t: SoilTest) -> Optional[Recommendation]:
        if t.ph is None:
            return None
        target = self.target_ph(t.crop)
        if t.ph >= target:
            return Recommendation(
                category="Lime",
                severity="ok",
                measured=f"soil pH {t.ph:.2f}",
                finding=f"at or above the pH {target:.1f} target for {t.crop}",
                action="No lime needed; maintain pH with routine monitoring.",
                amount=None,
                citation=CITATIONS["ph_target"],
            )

        # Below target -> lime advised. The RATE needs the Mehlich buffer pH. If we
        # have it, approximate the AASL table; if not, we are honest that SSURGO
        # pH alone cannot set a rate and a buffer-pH test is required.
        if t.buffer_ph is not None:
            amount = self._lime_rate_from_buffer(t.buffer_ph)
            cite = CITATIONS["lime_buffer"]
            action = ("Apply lime to raise pH to the crop target. Rate below is an "
                      "approximation of the AASL buffer table -- confirm on your report.")
        else:
            amount = None
            cite = CITATIONS["ph_target"]
            action = ("Lime needed to reach the crop target. A precise rate requires "
                      "a Mehlich BUFFER pH measurement (AASL test); it is NOT derivable "
                      "from SSURGO pH/CEC alone, so no tonnage is asserted here.")

        return Recommendation(
            category="Lime",
            severity="deficiency",
            measured=f"soil pH {t.ph:.2f}" + (f", buffer pH {t.buffer_ph:.2f}"
                                              if t.buffer_ph is not None else ""),
            finding=f"below the pH {target:.1f} target for {t.crop} (short by {target - t.ph:.2f})",
            action=action,
            amount=amount,
            citation=cite,
        )

    def _lime_rate_from_buffer(self, buffer_ph: float) -> str:
        """Approximate ag-lime requirement (tons CaCO3-equiv/acre, 0-15cm) from
        Mehlich buffer pH. Linear stand-in for the AASL lookup: ~0 at buffer pH
        6.8 (no reserve acidity), rising ~2 tons per 0.5 buffer-pH unit below it,
        capped at 6 tons (a single-application maximum). Labeled an ESTIMATE."""
        ref_buffer = 6.8
        tons = max(0.0, (ref_buffer - buffer_ph) / 0.5 * 2.0)
        tons = min(tons, 6.0)
        return f"~{tons:.1f} tons ag lime / acre (ESTIMATE; confirm with AASL buffer table)"

    # ------------------------------------------------------------- phosphorus
    def _phosphorus_rule(self, t: SoilTest) -> Optional[Recommendation]:
        if t.mehlich3_p_ppm is None:
            return Recommendation(
                category="Phosphate (P2O5)",
                severity="unknown",
                measured="Mehlich-3 P not provided",
                finding="cannot interpret P without a lab soil test (SSURGO has no P).",
                action="Obtain a Mehlich-3 P soil test for the field.",
                amount=None,
                citation=CITATIONS["p_mehlich3"],
            )
        p = t.mehlich3_p_ppm
        if p < self.P_BELOW_OPTIMUM:
            amount = self._maintenance_rate(t, nutrient="P2O5", build_up=True)
            return Recommendation(
                "Phosphate (P2O5)", "deficiency",
                f"Mehlich-3 P {p:.0f} ppm",
                f"below optimum (<{self.P_BELOW_OPTIMUM:.0f} ppm); yield response to added P likely",
                "Apply P2O5: build toward optimum plus replace crop removal at your yield goal.",
                amount, CITATIONS["p_mehlich3"],
            )
        if p <= self.P_OPTIMUM_HIGH:
            amount = self._maintenance_rate(t, nutrient="P2O5", build_up=False)
            return Recommendation(
                "Phosphate (P2O5)", "maintenance",
                f"Mehlich-3 P {p:.0f} ppm",
                f"in the optimum range ({self.P_BELOW_OPTIMUM:.0f}-{self.P_OPTIMUM_HIGH:.0f} ppm)",
                "Apply maintenance P2O5 to replace what the harvest removes.",
                amount, CITATIONS["p_mehlich3"],
            )
        if p <= self.P_EXCESSIVE:
            return Recommendation(
                "Phosphate (P2O5)", "ok",
                f"Mehlich-3 P {p:.0f} ppm",
                f"above optimum (>{self.P_OPTIMUM_HIGH:.0f} ppm)",
                "No P2O5 needed this season.",
                None, CITATIONS["p_mehlich3"],
            )
        return Recommendation(
            "Phosphate (P2O5)", "environmental",
            f"Mehlich-3 P {p:.0f} ppm",
            f"excessive (>{self.P_EXCESSIVE:.0f} ppm) -- P-loss / P-Index concern",
            "Apply NO P; a Penn State P-Index assessment may be required before manure.",
            None, CITATIONS["p_mehlich3"],
        )

    # -------------------------------------------------------------- potassium
    def _potassium_rule(self, t: SoilTest) -> Optional[Recommendation]:
        if t.mehlich3_k_ppm is None:
            return Recommendation(
                category="Potash (K2O)",
                severity="unknown",
                measured="Mehlich-3 K not provided",
                finding="cannot interpret K without a lab soil test (SSURGO has no K).",
                action="Obtain a Mehlich-3 K soil test for the field.",
                amount=None,
                citation=CITATIONS["k_mehlich3"],
            )
        k = t.mehlich3_k_ppm
        if k < self.K_BELOW_OPTIMUM:
            amount = self._maintenance_rate(t, nutrient="K2O", build_up=True)
            return Recommendation(
                "Potash (K2O)", "deficiency",
                f"Mehlich-3 K {k:.0f} ppm",
                f"below optimum (<{self.K_BELOW_OPTIMUM:.0f} ppm); yield response to added K likely",
                "Apply K2O: build toward optimum plus replace crop removal at your yield goal.",
                amount, CITATIONS["k_mehlich3"],
            )
        if k <= self.K_OPTIMUM_HIGH:
            amount = self._maintenance_rate(t, nutrient="K2O", build_up=False)
            return Recommendation(
                "Potash (K2O)", "maintenance",
                f"Mehlich-3 K {k:.0f} ppm",
                f"in the optimum range ({self.K_BELOW_OPTIMUM:.0f}-{self.K_OPTIMUM_HIGH:.0f} ppm)",
                "Apply maintenance K2O to replace what the harvest removes.",
                amount, CITATIONS["k_mehlich3"],
            )
        return Recommendation(
            "Potash (K2O)", "ok",
            f"Mehlich-3 K {k:.0f} ppm",
            f"above optimum (>{self.K_OPTIMUM_HIGH:.0f} ppm)",
            "No K2O needed this season.",
            None, CITATIONS["k_mehlich3"],
        )

    def _maintenance_rate(self, t: SoilTest, nutrient: str, build_up: bool) -> Optional[str]:
        """Crop-removal maintenance rate (lb nutrient/acre) at the stated yield goal.

        Only emitted for crops with established removal coefficients and when a
        yield goal is given; otherwise None (we don't fabricate a number). Build-up
        on a below-optimum soil ADDS to this, so we flag the rate as a floor."""
        coeffs = self.REMOVAL.get(t.crop)
        if coeffs is None or t.yield_goal is None:
            return None
        p2o5_per, k2o_per, unit = coeffs
        per = p2o5_per if nutrient == "P2O5" else k2o_per
        lbs = per * t.yield_goal
        tag = " + build-up to optimum (see AASL report)" if build_up else ""
        return (f"~{lbs:.0f} lb {nutrient}/acre to replace removal at "
                f"{t.yield_goal:.0f} {unit}/acre yield goal{tag}")

    # -------------------------------------------------------------- public API
    def recommend(self, t: SoilTest) -> list[Recommendation]:
        """Run every rule and return the recommendations that apply, in a fixed
        order (lime, then P, then K) for a stable, readable report."""
        recs = [
            self._lime_rule(t),
            self._phosphorus_rule(t),
            self._potassium_rule(t),
        ]
        return [r for r in recs if r is not None]


# --------------------------------------------------------------------------- #
# Helpers to build a SoilTest from our SSURGO county context (pH/CEC/OM only).
# --------------------------------------------------------------------------- #
def soiltest_from_ssurgo(
    county_name: str,
    crop: str,
    *,
    mehlich3_p_ppm: Optional[float] = None,
    mehlich3_k_ppm: Optional[float] = None,
    buffer_ph: Optional[float] = None,
    yield_goal: Optional[float] = None,
    ssurgo_path: Path | str = _PROCESSED / "ssurgo_pa_soil.csv",
) -> SoilTest:
    """Seed a SoilTest with a county's SSURGO pH/CEC/OM, leaving the lab-only
    fields (P, K, buffer pH) to be supplied by an actual soil test. Used by the
    demo and, later, by the orchestrator."""
    df = pd.read_csv(ssurgo_path, dtype={"county_ansi": "string"})
    row = df[df["areaname"].str.contains(county_name, case=False, na=False)]
    if row.empty:
        raise ValueError(f"No SSURGO row for county matching {county_name!r}.")
    r = row.iloc[0]
    return SoilTest(
        crop=crop,
        ph=float(r["topsoil_ph"]),
        cec=float(r["topsoil_cec"]),
        om_pct=float(r["topsoil_om_pct"]),
        buffer_ph=buffer_ph,
        mehlich3_p_ppm=mehlich3_p_ppm,
        mehlich3_k_ppm=mehlich3_k_ppm,
        yield_goal=yield_goal,
        county_name=county_name,
        note="pH/CEC/OM from SSURGO county summary; P/K/buffer from field soil test.",
    )


def _print_report(t: SoilTest, recs: list[Recommendation]) -> None:
    print(f"\n=== Strata soil-adjustment report -- {t.crop} ===")
    if t.county_name:
        print(f"field context: {t.county_name} County")
    obs = []
    if t.ph is not None: obs.append(f"pH {t.ph:.2f}")
    if t.buffer_ph is not None: obs.append(f"buffer pH {t.buffer_ph:.2f}")
    if t.cec is not None: obs.append(f"CEC {t.cec:.1f}")
    if t.om_pct is not None: obs.append(f"OM {t.om_pct:.1f}%")
    if t.mehlich3_p_ppm is not None: obs.append(f"M3-P {t.mehlich3_p_ppm:.0f} ppm")
    if t.mehlich3_k_ppm is not None: obs.append(f"M3-K {t.mehlich3_k_ppm:.0f} ppm")
    print("measured: " + ", ".join(obs))
    if t.note:
        print(f"(provenance: {t.note})")

    for r in recs:
        print(f"\n[{r.severity.upper()}] {r.category}")
        print(f"  measured : {r.measured}")
        print(f"  finding  : {r.finding}")
        print(f"  action   : {r.action}")
        if r.amount:
            print(f"  rate     : {r.amount}")
        print(f"  source   : {r.citation.source}")
        print(f"             {r.citation.rule}")


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    engine = OptimizationEngine()

    # Demo on REAL Chester County SSURGO soil (pH 5.72, CEC 10.5, OM 2.4%, Ultisols).
    # The Mehlich-3 P/K and buffer pH below are ILLUSTRATIVE soil-test values — SSURGO
    # does not carry them; a real field would supply them from an AASL test report.
    print("#" * 72)
    print("# NOTE: P, K, and buffer pH below are ILLUSTRATIVE lab values, not SSURGO.")
    print("#" * 72)

    corn = soiltest_from_ssurgo(
        "Chester", "corn_grain_yield",
        mehlich3_p_ppm=22, mehlich3_k_ppm=85, buffer_ph=6.5, yield_goal=160,
    )
    _print_report(corn, engine.recommend(corn))

    # Same field, alfalfa — note the HIGHER pH target makes the lime finding stronger,
    # and P/K here are illustrated in the optimum range (maintenance only).
    alfalfa = soiltest_from_ssurgo(
        "Chester", "alfalfa_hay_yield",
        mehlich3_p_ppm=40, mehlich3_k_ppm=120, buffer_ph=6.6, yield_goal=5,
    )
    _print_report(alfalfa, engine.recommend(alfalfa))

    # And a case with NO lab P/K (SSURGO-only) to show the engine refusing to guess.
    print("\n" + "-" * 72)
    print("SSURGO-only input (no field soil test): the engine declines to assert P/K.")
    bare = soiltest_from_ssurgo("Chester", "soybean_yield")
    _print_report(bare, engine.recommend(bare))
