# Strata

An affordable precision-ag data/ML product for small farms in Chester County, PA
(Exton/Malvern area). Strata does three things for a crop the farmer **already
grows**:

1. **Yield regression** — predict county-year yield from soil + weather features.
2. **Soil classification** — characterize soil taxonomic class from measurable lab
   properties (rigor: proves soil is characterizable from numbers).
3. **Optimization / recommendation** — apply EXISTING, trusted Penn State AASL
   agronomy rules to a field's measured soil and emit cited adjustment advice.

### What Strata is NOT
- Not a "which crop to plant" classifier — farmers know their crop.
- Not an invented "% of theoretical max yield" target.
- Not hardware. No rover/motors/GPS/PCB code lives here. A cheap soil probe will
  later feed in only as a **calibration layer** (`strata.calibration`); the
  product has zero dependency on any sensor existing yet.

---

## Repository layout

```
strata/
├── config/
│   └── sources.yaml         # source-agnostic loader config (no params in code)
├── data/
│   ├── raw/                 # pulled source data (gitignored; reproduce via provenance)
│   └── processed/           # analysis tables, county-year unit (gitignored)
├── src/strata/
│   ├── data_ingest.py       # [IMPLEMENTED] NASS + SSURGO + KSSL + Daymet loaders
│   ├── assemble.py          # [IMPLEMENTED] join -> county-year-crop analysis table
│   ├── soil_classifier.py   # [IMPLEMENTED] SoilClassifier (RandomForestClassifier)
│   ├── yield_model.py       # [IMPLEMENTED] YieldModel (RandomForestRegressor)
│   ├── recommender.py       # [IMPLEMENTED] OptimizationEngine (cited Penn State rules)
│   ├── calibration.py       # [IMPLEMENTED] cheap-probe vs lab calibration seam
│   └── main.py              # [IMPLEMENTED] orchestrator + readable report
├── requirements.txt
├── .env.example             # copy to .env, add NASS_API_KEY
└── README.md
```

Status: **all modules implemented and runnable**. The four data loaders, the
analysis-table assembler, the soil classifier, the yield model, the cited
recommender, the calibration seam, and the orchestrator all run. Results are
recorded in the sections below; `python -m strata.main` reproduces them offline
from `data/processed/*.csv`.

---

## Setup

```bash
python -m venv .venv
# Windows PowerShell:  .venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env        # then edit .env and set NASS_API_KEY
```

The NASS API key is read from the environment (`NASS_API_KEY`) via `.env`, so it
never lands in source or git history.

---

## Data provenance

Every source is reproduced from the record below, not committed to the repo.
Access dates will be filled in as each source is actually pulled.

| # | Source | What we take | Endpoint / location | Accessed |
|---|--------|--------------|---------------------|----------|
| 1 | USDA NASS Quick Stats | County-level PA yields, 8 crops (corn grain/silage, soybean, winter wheat, barley, oats, hay, alfalfa hay), SURVEY, year≥2000 | `https://quickstats.nass.usda.gov/api/api_GET/` | 2026-06-24 |
| 2 | USDA SSURGO via Soil Data Access (SDA) | County-level topsoil (0-25cm) texture, pH, OM, CEC + dominant Soil Taxonomy order & drainage, for all 62 PA survey areas | `https://sdmdataaccess.sc.egov.usda.gov/Tabular/post.rest` | 2026-06-24 |
| 3 | NCSS/KSSL Soil Characterization DB | Lab pedon horizons (sand/silt/clay, pH, CEC, organic C) + Soil Taxonomy order/great group, 14 NE/Mid-Atlantic states | SDA Lab Data Mart (`lab_*` tables) | 2026-06-24 |
| 4 | Daymet v4 daily weather | County-centroid daily Tmax/Tmin/precip -> growing-season (Apr-Oct) features, 2000-2023 | `https://daymet.ornl.gov/single-pixel/api/data` | 2026-06-24 |
| 5 | Penn State AASL handbooks | Mehlich-3 P/K, buffer lime, PSNT corn-N rules | Encoded as a cited rule engine (not learned) | n/a |
| 6 | Synthetic Kaggle yield CSV | Smoke-test scaffold ONLY | (scaffold) | n/a |

> Source 6 is synthetic. Any result derived from it is labeled as such and is
> never reported as a finding.

**Daymet citation (required by the data provider):** Thornton, M.M., R. Shrestha,
Y. Wei, P.E. Thornton, S-C. Kao, and B.E. Wilson. 2022. *Daymet: Daily Surface
Weather Data on a 1-km Grid for North America, Version 4 R1.* ORNL DAAC, Oak
Ridge, Tennessee, USA. https://doi.org/10.3334/ORNLDAAC/2129

#### NASS coverage notes (observed 2026-06-24)

PA county-year yields, SURVEY program, 2000+. No suppressed/withheld values in
any series (NASS only publishes county yields where disclosure rules allow, so
what we get is complete). Counts and **caveats that matter for the yield model**:

| crop | rows | year span | counties |
|------|-----:|-----------|---------:|
| corn_grain_yield   | 1509 | 2000-2025 | 68 |
| corn_silage_yield  | 1204 | 2000-2025 | 68 |
| soybean_yield      | 1246 | 2000-2025 | 59 |
| oats_yield         | 1234 | 2000-2025 | 63 |
| alfalfa_hay_yield  |  927 | 2000-2018 | 67 |
| winter_wheat_yield |  821 | 2000-2025 | 56 |
| hay_yield          |  575 | 2000-2008 | 67 |
| barley_yield       |  463 | 2000-2015 | 36 |

- **`hay_yield` (all hay) effectively ends ~2008** and **`barley` ~2015**,
  **`alfalfa` ~2018** at county level — NASS thinned these county series over
  time. They are kept for history but are not reliable recent targets.
- **Chester County specifically is thin**: 136 rows across all crops, e.g. only
  26 corn-grain county-years (2000-2025). *Implication, stated up front:* the
  yield model will train across **all PA counties**, not Chester alone — a single
  county does not supply enough county-year observations. This is the expected
  county-year granularity limit, surfaced rather than masked.

#### SSURGO coverage notes (observed 2026-06-24)

- **62 PA county survey areas**, all summarized to one county row each, **no
  missing soil properties**. Output: `data/processed/ssurgo_pa_soil.csv`.
- Each county row is an **area×component×depth-weighted** summary of 0-25cm
  topsoil (sand/silt/clay/pH/OM/CEC) plus the dominant Soil Taxonomy order and
  drainage class. Aggregation is done transparently in pandas, not a black-box
  SQL query (see `SsurgoSoilLoader.tidy`).
- *Assumption stated inline:* map-unit area weights use `STArea()` in
  square-degrees (geometry is lat/lon). Absolute area is irrelevant; only the
  *relative* within-county weighting matters, and lat/lon distortion across one
  county is negligible.
- **Join caveat:** NASS reports yields for ~68 county codes (includes "other
  (combined) counties" pseudo-records); SSURGO has 62 real counties. The
  county-year join will be an inner join on real FIPS — unmatched NASS pseudo-
  records are dropped, and we will report how many.
- *Granularity mismatch (council rule):* a single county soil summary is coarse
  context for a county-year yield, NOT a field-level claim. Field resolution is
  deferred to the future cheap-probe calibration layer.

#### KSSL lab pedon coverage (observed 2026-06-24)

- **65,791 horizon rows / 10,993 pedons** across 14 NE/Mid-Atlantic states;
  **7,754 pedons carry a Soil Taxonomy order label**. Output:
  `data/processed/kssl_pedons.csv` (layer-level; one row per horizon).
- Soil ORDER distribution (pedon-level): alfisols 2627, inceptisols 1874,
  ultisols 1307, entisols 806, spodosols 567, mollisols 407, histosols 141,
  aridisols 19, vertisols 5, oxisols 1.
- *Modeling note for `soil_classifier.py`:* 7 orders are well-populated (>100);
  aridisols/vertisols/oxisols are too rare to learn or validate and will be
  dropped or grouped (decision stated there, not hidden in ingest).
- `cec` is recovered from a documented method fallback (NH4OAc pH7 → sum-of-
  cations pH8.2), cutting layer-level CEC missingness from 66.6% to 36.7%.
- `lat`/`lon` are kept as **provenance only** — never classifier features
  (predicting soil order from coordinates would be leakage; the goal is to
  predict it from measurable soil properties).

#### Soil classifier results (`soil_classifier.py`, observed 2026-06-24)

Predicts Soil Taxonomy ORDER from measured lab properties only (texture, pH, CEC,
organic C, and depth-contrast features). Trained on **7,686 labeled pedons** (7
orders; aridisols/vertisols/oxisols dropped as too rare to validate — 25 pedons).

- **5-fold stratified CV: accuracy 0.689 ±0.009, macro-F1 0.653 ±0.010**, versus a
  majority-class baseline of **0.341** — the model roughly doubles the baseline,
  so soil order is genuinely recoverable from cheap measurable numbers.
- Held-out per-class F1: alfisols 0.79, ultisols/spodosols/inceptisols ~0.66–0.68,
  rare histosols/mollisols weakest (~0.52, confused mainly with alfisols).
- **Top permutation-importance features are `clay_increase` and `clay_max`** — the
  exact clay-with-depth signal (argillic horizon) that Soil Taxonomy keys on,
  followed by profile CEC, organic C, and pH. The model learned real soil science,
  not an artifact. `lat`/`lon` and finer tax labels were excluded to prevent leakage.

#### Daymet weather coverage (observed 2026-06-24)

- **1,488 county-year rows** (62 counties × 24 years, 2000-2023), **no missing
  features**. Outputs: `data/processed/daymet_pa_weather.csv` and the county
  centroids used (`data/processed/pa_county_centroids.csv`).
- Features per county-year over the Apr 1-Oct 31 window: `gdd_base10`,
  `precip_total_mm`, `tmax_mean_c`, `tmin_mean_c`, `heat_days_30c`.
- County represented by ONE area-weighted SSURGO centroid (no new dataset).
  *Granularity caveat:* one point per county; coarse for large/varied counties.
- **Join caveat:** Daymet ends 2023; NASS yields run to 2025. The 2024-2025
  yield rows will have no weather and drop from the modeling join (reported there).

#### Yield model results (`yield_model.py`, observed 2026-06-26)

Regresses a crop's county-year YIELD onto static county soil (0-25cm topsoil
texture/pH/OM/CEC + taxonomic order & drainage) and that year's growing-season
weather. **One model per crop** — yield units differ by crop (BU/ACRE vs
TONS/ACRE), so pooling would make the metrics meaningless. `year`, county
identity, and any per-county target statistic are excluded as leakage.

The headline rigor here is reporting **two** cross-validation views, because soil
features are county-static (every row from a county shares them):

- **Random 5-fold CV** lets a county sit in train and test, so the model can
  recognize it by its soil fingerprint — optimistic.
- **Leave-counties-out 5-fold CV (GroupKFold on FIPS)** holds every test county
  out of training — the honest estimate for a *new* farm in an unmodeled county.

Corn grain (1,138 county-years, 56 counties, BU/ACRE):

| evaluation | MAE | R² |
|------------|----:|---:|
| mean-yield baseline (held-out) | 21.7 | ~0.00 |
| random 5-fold CV | 18.3 | 0.368 |
| leave-counties-out 5-fold CV | 19.9 | 0.212 |

Per-crop (random R² vs leave-counties-out R² / MAE), crops with ≥150 county-years:

| crop | rows | counties | unit | rand R² | LCO R² | LCO MAE |
|------|-----:|---------:|------|--------:|-------:|--------:|
| corn_grain_yield   | 1138 | 56 | BU/ACRE   | 0.368 | 0.212 | 19.9 |
| oats_yield         |  977 | 52 | BU/ACRE   | 0.113 | -0.02 |  7.0 |
| soybean_yield      |  949 | 51 | BU/ACRE   | 0.380 | 0.262 |  5.2 |
| corn_silage_yield  |  883 | 56 | TONS/ACRE | 0.376 | 0.227 |  2.4 |
| alfalfa_hay_yield  |  727 | 56 | TONS/ACRE | 0.306 | 0.157 |  0.4 |
| winter_wheat_yield |  630 | 50 | BU/ACRE   | 0.344 | 0.122 |  7.5 |
| hay_yield          |  483 | 56 | TONS/ACRE | 0.410 | 0.409 |  0.3 |
| barley_yield       |  345 | 33 | BU/ACRE   | 0.396 | 0.305 |  7.3 |

- The model **clearly beats the mean-yield baseline** (R² ~0) but the random→LCO
  gap shows part of the random-CV score is a county-identity crutch. The
  leave-counties-out numbers are reported as the real-world expectation.
- **Permutation importance is dominated by weather** — `heat_days_30c`,
  `tmin_mean_c`, `precip_total_mm` — which is expected and honest: soil is static,
  so it cannot explain *year-to-year* yield swings; the year's weather can. Soil
  (CEC, OM) contributes second-order. This says yield variance at county-year
  granularity is mostly a weather story, and the README does not oversell soil.
- `oats_yield` LCO-R² is ≈0: it does not generalize to unseen counties and is
  reported as such rather than masked by the flattering random-CV number.

#### Recommendation engine (`recommender.py`)

`OptimizationEngine` applies EXISTING, cited Penn State agronomy rules to a
field's soil test and emits adjustment advice — it is a transparent lookup, not a
learned model and not the old circular IQR scorer (which is gone). Rules covered:

- **Lime / pH** — crop-specific target pH (≈6.0 general crops; ≈6.5–6.8 legumes).
  The *rate* is set from the Mehlich **buffer pH**; when only SSURGO water-pH is
  available the engine states a rate is not derivable and a buffer-pH test is
  required, rather than inventing a tonnage. When buffer pH is given it prints a
  clearly-labeled ESTIMATE to confirm against the AASL table.
- **Phosphate (P2O5)** — Mehlich-3 P interpretation (below optimum <30 ppm,
  optimum 30–50, above 50–200, excessive >200 ppm → P-Index concern).
- **Potash (K2O)** — Mehlich-3 K interpretation (below optimum <100, optimum
  ~100–130, above >130 ppm).
- **Maintenance rates** from cited crop-removal coefficients (corn grain, soybean)
  at the user's yield goal; build-up on below-optimum soils is flagged as a floor.

Every recommendation prints the measured value, the threshold it crossed, the
action, an amount where defensible, and a **citation** to the source rule. Design
choices that match the project's rigor ethos:

- **Cited, not invented / not learned.** All thresholds live in one auditable
  table (`CITATIONS` + named constants) attributed to the Penn State Agronomy
  Guide / AASL, with a prominent note that exact values must be confirmed against
  the current AASL edition before field use.
- **Input honesty.** SSURGO carries pH/CEC/OM but NOT Mehlich-3 P/K or buffer pH;
  those require a real lab soil test. The engine REFUSES to interpret P/K when they
  are absent (returns an "obtain a soil test" item) instead of guessing — the same
  county-vs-field granularity gap the future `calibration` probe layer addresses.

Run `python -m strata.recommender` for a worked report on real Chester County
SSURGO soil (pH 5.72, Ultisols) for corn and alfalfa, plus an SSURGO-only case
showing the engine declining to assert P/K.

#### Calibration seam (`calibration.py`)

`ProbeCalibration` defines the ONLY place a future cheap soil probe plugs in. Given
a handful of paired `(probe_reading, lab_value)` samples per property (pH, Mehlich-3
P, K), it fits a simple ordinary-least-squares `lab = slope·probe + intercept`
correction and reports its quality (slope, R², RMSE, and the raw probe↔lab bias the
calibration removes). A raw probe reading is then mapped to a lab-equivalent value
and fed into the **same** recommender — nothing upstream changes.

- **No hardware dependency.** Yield, soil, and recommendation modules all run today
  with no sensor. The probe is an *optional* accuracy layer, not a prerequisite.
- **Honesty.** No probe or probe data exists, so the demo fits on a clearly-labeled
  **synthetic** paired sample (a known gain/offset/noise on real lab values) purely
  to exercise the interface. Per the project rule, synthetic output is labeled and
  never reported as a finding. Run `python -m strata.calibration`.

---

## Reproducing results end to end

Everything runs **offline** from the deterministic `data/processed/*.csv` (no API
key or network needed once those exist; reproduce them from the provenance table
above). The orchestrator sequences all four capabilities into one report:

```bash
python -m strata.main      # soil classifier -> yield model -> cited reco -> calib seam
```

Individual modules are runnable on their own for detail:

```bash
python -m strata.assemble         # rebuild the county-year-crop analysis table
python -m strata.soil_classifier  # soil-order classification, CV + importance
python -m strata.yield_model      # per-crop yield regression, random vs LCO CV
python -m strata.recommender      # cited Penn State recommendations on a field
python -m strata.calibration      # synthetic probe->lab calibration seam demo
```

---

## Modeling conventions / rigor

- `random_state=42` everywhere for determinism.
- Validation on held-out **real** data; **no circular/self-referential** scoring.
- Granularity mismatch is stated inline where it occurs (county-year yield vs.
  point soil reading) rather than hidden.
- Where real PA data is thin, the output says so instead of masking it with
  synthetic fills.

---

## Dependencies policy

Core deps are in `requirements.txt`. "Heavy" libraries (GIS stacks, R bridges)
are **deliberately deferred** and will be discussed before being added — see the
comments at the bottom of `requirements.txt`.
