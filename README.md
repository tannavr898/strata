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
│   ├── main.py              # [IMPLEMENTED] orchestrator + readable report
│   └── report.py            # [IMPLEMENTED] 4-panel diagnostic dashboard PNG
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

#### Yield model results (`yield_model.py`, observed 2026-07-07)

Per-crop RandomForest regressors predict county-year yield from soil + weather.
Two honest cross-validations are reported: random k-fold (optimistic — a county's
static soil fingerprint leaks across folds) vs **leave-counties-out (LCO)**, plus a
**forward-in-time** forecast (train ≤2017, predict 2018+). `year` is a switchable
trend term: withheld = *attribution* (soil+weather only), included = *forecast*
(product default). County identity is never a feature.

Corn grain (1,138 county-years, 56 counties):
- mean-yield baseline MAE **21.7 bu/ac** → model LCO forecast MAE **~15 bu/ac**.
- LCO R²: **0.21 attribution → 0.51 forecast**; forward-forecast R² **0.35**.
- The random→LCO drop (0.68→0.51) is exactly the county-memorization the design
  warns about — LCO is the number to trust.

| crop | attribR² | fcstR² | fwdR² | fcst MAE | ship? |
|---|--:|--:|--:|--:|:--:|
| corn_grain | 0.21 | 0.51 | 0.35 | 14.9 bu/ac | ✓ |
| soybean | 0.26 | 0.51 | 0.32 | 4.2 bu/ac | ✓ |
| hay | 0.41 | 0.52 | — | 0.24 t/ac | ✓ |
| corn_silage | 0.23 | 0.44 | 0.46 | 2.1 t/ac | ✓ |
| barley | 0.31 | 0.38 | — | 6.9 bu/ac | ✓ |
| winter_wheat | 0.12 | 0.33 | — | 6.6 bu/ac | ✓ |
| alfalfa_hay | 0.16 | 0.18 | -0.00 | 0.40 t/ac | ✗ |
| oats | -0.02 | 0.07 | -0.04 | 6.6 bu/ac | ✗ |

- **Stated honestly:** `oats` and `alfalfa` are essentially unexplained by soil+
  weather (R² ≈ 0) — surfaced, not masked. Corn/soy/silage/hay/barley forecast
  usefully. The attribR²→fcstR² jump is the genetic/management yield trend.
- **Product quality gate (`ship?`).** Research reports every crop; the *product*
  offers a forecast only where the honest LCO forecast R² clears a named threshold
  (`YieldModel.PRODUCT_MIN_FORECAST_R2 = 0.30`). `oats` (0.07) and `alfalfa` (0.18)
  fall below it and are withheld with an explicit *"insufficient model skill"*
  message rather than shipped as a number a farmer would wrongly trust.
- **Permutation importance is computed leave-counties-out** (random vs LCO shown
  side by side). Doing so corrected an earlier reading: the old headline
  `dominant_taxorder ≈ 0.89` was a *label/column misalignment* — that magnitude is
  actually `year` (the trend term). Correctly labeled, taxorder is minor in **both**
  splits (~0.004). The features that genuinely shrink random→LCO are the **static
  soil numerics** (`topsoil_om` 0.12→0.07, `cec` 0.06→0.03, `ph`/`clay`→~0), i.e.
  the county-recall crutch, exactly as the between/within-county variance split
  below predicts.

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

#### Yield model results (`yield_model.py`, observed 2026-06-30)

Regresses a crop's county-year YIELD onto static county soil (0-25cm topsoil
texture/pH/OM/CEC + taxonomic order & drainage) and that year's growing-season
weather. **One model per crop** — yield units differ by crop (BU/ACRE vs
TONS/ACRE), so pooling would make the metrics meaningless. County identity and any
per-county target statistic are always excluded as leakage.

**Two modes for `year` (the `include_trend` flag).** Corn/soy county yields carry
a strong upward trend over 2000-2023 (better hybrids/management add ~1.5-2
bu/acre/yr). That is real signal, but not a soil-or-weather effect, so we report
both treatments rather than pick one:

- **Attribution mode** (`include_trend=False`) withholds `year`: how much do
  soil+weather *alone* explain? The conservative number.
- **Forecast mode** (`include_trend=True`, the **product default**) adds `year` as
  a trend term: what will the field actually yield? This is what a farmer needs and
  it roughly **doubles** honest accuracy. Critically, it is the only mode that
  forecasts **forward in time** (train on the past, predict the future): a no-trend
  model trained on ≤2017 scores **R² < 0** on 2018+ (worse than the mean, because it
  cannot see the uptrend), while the trend model reaches R² ≈ 0.35.

Each treatment is still validated two ways — **random 5-fold CV** (optimistic; a
county can sit in train and test) and **leave-counties-out CV** (GroupKFold on
FIPS; the honest new-county estimate). Plus a **forward forecast** (train ≤2017,
predict 2018+), the real-world arrow-of-time test.

Corn grain (1,138 county-years, 56 counties, BU/ACRE):

| evaluation | MAE | R² |
|------------|----:|---:|
| mean-yield baseline (held-out) | 21.7 | ~0.00 |
| attribution, leave-counties-out (soil+weather only) | 19.9 | 0.212 |
| **forecast, leave-counties-out (+trend)** | **14.9** | **0.513** |
| forecast, forward-in-time (train ≤2017) | 16.9 | 0.347 |
| same forward test **without** trend | 25.9 | **−0.42** |

Per-crop leave-counties-out R² in both modes, plus the forward forecast (crops with
≥150 county-years; `nan` = the NASS series ends before 2018 so no forward split):

| crop | rows | counties | unit | attrib R² | forecast R² | forward R² | fcst MAE |
|------|-----:|---------:|------|----------:|------------:|-----------:|---------:|
| corn_grain_yield   | 1138 | 56 | BU/ACRE   | 0.212 | 0.513 |  0.347 | 14.9 |
| oats_yield         |  977 | 52 | BU/ACRE   | -0.019 | 0.072 | -0.039 |  6.6 |
| soybean_yield      |  949 | 51 | BU/ACRE   | 0.262 | 0.510 |  0.322 |  4.2 |
| corn_silage_yield  |  883 | 56 | TONS/ACRE | 0.227 | 0.439 |  0.458 |  2.1 |
| alfalfa_hay_yield  |  727 | 56 | TONS/ACRE | 0.157 | 0.180 | -0.002 |  0.4 |
| winter_wheat_yield |  630 | 50 | BU/ACRE   | 0.122 | 0.332 |    nan |  6.6 |
| hay_yield          |  483 | 56 | TONS/ACRE | 0.409 | 0.523 |    nan |  0.2 |
| barley_yield       |  345 | 33 | BU/ACRE   | 0.305 | 0.376 |    nan |  6.9 |

- The trend term lifts **every crop** (corn 0.21→0.51, soy 0.26→0.51, silage
  0.23→0.44, wheat 0.12→0.33). It is surfaced as an explicit, switchable feature,
  not hidden, and the conservative attribution number is reported alongside it.
- **What does NOT help (tested, rejected):** gradient boosting, ExtraTrees,
  hyperparameter tuning, county lat/lon, and engineered weather/soil interactions
  were all measured against the same honest metrics and **none beat RF+trend** on
  the forward forecast; some (ExtraTrees) raise leave-counties-out R² while
  *lowering* forward R² — a mirage we declined to ship. Replacing the `year` feature
  with an explicit linear **detrend** (linear trend + RF on the residual) was also
  tested on the theory that a RandomForest cannot extrapolate `year` past its
  training range; it **slightly hurt** the forward forecast (corn 0.35→0.29) rather
  than helping — the linear trend *overshoots* on 2018+ while the year-in-RF model,
  which holds near its terminal-year level, was already close. A null result, kept
  as a finding rather than tuned away.
- **Why the ceiling is where it is (measured, not asserted).** A between/within-county
  variance decomposition of corn yield is **20% between-county / 80% within-county**
  (year to year). The 8 SSURGO soil features are **static per county** — one value,
  no time axis — so they have *zero* variance within a county and can only ever
  address the ~20% between-county share; they contribute literally nothing to the
  80% that varies year to year. That within-county 80% is ~30% a smooth time trend
  and ~70% weather+noise, so a *single county-centroid* Daymet series plus the trend
  is all that speaks to it. LCO R² ≈ 0.5 / forward R² ≈ 0.35 is therefore a
  **feature-information ceiling**, not a model-tuning one: no estimator can extract
  within-county signal the feature matrix does not contain. The next real gain needs
  a feature that varies *within* a county over time — exactly what the calibration
  seam is built to ingest.
- `oats_yield` stays ≈0 even with the trend: it does not generalize to unseen
  counties and is reported as such rather than masked.

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
- **Maintenance rates** from cited crop-removal coefficients (corn grain, soybean),
  scaled to the **expected harvest**; build-up on below-optimum soils is flagged as a
  floor. The removal basis is, in order: a **YieldModel-predicted yield** when one is
  supplied (`recommend(..., expected_yield=...)`), else the farmer's stated yield goal,
  else no rate (never fabricated). This is the one place the two modules **compose** —
  a larger predicted harvest removes more nutrients — and the printed rate names which
  basis it used, so a model estimate is never passed off as a measured target. The
  handoff is gated: `main.py` only feeds a predicted yield when the crop cleared the
  product quality gate above.

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

- **What a probe actually adds (the ceiling connection).** The yield model's honest
  ceiling exists because the feature matrix has *no soil feature that varies within a
  county over time* — SSURGO is one static snapshot per county, which is why it can
  only reach corn's ~20% between-county variance and none of the ~80% year-to-year
  (see the yield ceiling note above). A probe read at a field this season is exactly
  that missing axis: a **time-varying** soil measurement. Calibrated onto the lab
  scale it becomes the first feature able to speak to the within-county variance the
  static SSURGO layer cannot touch — so the seam is where new predictive signal
  enters, not just finer spatial resolution.
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
python -m strata.report           # render data/processed/strata_dashboard.png
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
