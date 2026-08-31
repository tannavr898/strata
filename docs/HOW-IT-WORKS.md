# How Strata Actually Works

**A mechanical walkthrough. No prior knowledge assumed.**

`METHODS.md` is written for a research audience — it states *what was done* and
assumes you already know how the machinery works. This document is the opposite.
It explains *how the machinery works*, using real data from the actual system, and
it defines every term the first time it appears.

Read this one first. Then `METHODS.md` will make sense.

---

## Table of contents

1. [The one-paragraph version](#1-the-one-paragraph-version)
2. [Following one real row through the whole system](#2-following-one-real-row-through-the-whole-system)
3. [What each file does](#3-what-each-file-does)
4. [How the yield model learns](#4-how-the-yield-model-learns)
5. [Why we test it three different ways](#5-why-we-test-it-three-different-ways)
6. [The ceiling, shown with real numbers](#6-the-ceiling-shown-with-real-numbers)
7. [The soil classifier: why it exists at all](#7-the-soil-classifier-why-it-exists-at-all)
8. [The recommender: not a model](#8-the-recommender-not-a-model)
9. [The calibration seam](#9-the-calibration-seam)
10. [Glossary](#10-glossary)

---

## 1. The one-paragraph version

Strata downloads four free public datasets about Pennsylvania farming, glues them
into one big table, trains a model to predict corn yield from that table, and then
uses the prediction — plus a farmer's soil test — to print fertilizer advice with a
citation to the Penn State rule behind each recommendation. It also refuses to
answer when it doesn't have enough information, which turns out to be one of the
more important things it does.

That's it. Everything below is detail on those steps.

---

## 2. Following one real row through the whole system

The clearest way to understand the pipeline is to follow a single real piece of
data from download to advice. Let's use **Chester County corn in 2023**.

### Step 1 — NASS tells us what was harvested

The USDA runs a survey called NASS Quick Stats. We ask it one question:

> *"How many bushels of corn per acre did each Pennsylvania county harvest, every
> year since 2000?"*

It answers with rows like this:

```
county = CHESTER,  year = 2023,  crop = corn grain,  yield = 175.4 bu/acre
```

That number — **175.4** — is the thing we want to predict. In machine learning
this is called the **target**. Everything else we collect exists to explain it.

Chester's corn history looks like this:

| year | yield (bu/acre) |
|---|---|
| 2000 | 134.8 |
| 2001 | 101.4 |
| 2002 | 72.2 ← bad year |
| … | … |
| 2021 | 204.7 |
| 2022 | 210.6 |
| 2023 | 175.4 |

Two things are visible immediately, and both matter later: yields **trend upward**
over 23 years, and they **bounce around a lot** year to year.

### Step 2 — SSURGO tells us what the soil is like

The USDA also publishes a soil survey called SSURGO. We ask it:

> *"What is the soil like in the top 25 cm of Chester County?"*

The raw answer is thousands of overlapping soil map polygons, each with multiple
layers. That's unusable, so we **average it down to one summary per county**,
weighted by how much area each soil type covers. For Chester we get:

```
sand 35.18%   silt 47.28%   clay 17.52%
pH 5.72       organic matter 2.41%    CEC 10.53
soil order = Ultisols      drainage = Well drained
```

Eight numbers describing Chester's dirt. (What pH and CEC mean is in the
[glossary](#10-glossary).)

**The critical property of these eight numbers: there is only one set of them.**
SSURGO is a survey, not a time series. Chester's pH is 5.72 — full stop. Not 5.72
in 2023 and something else in 2001. Just 5.72. Remember this; §6 turns on it.

### Step 3 — Daymet tells us what the weather did

Daymet is a NASA/ORNL dataset of daily weather on a 1 km grid. We pick one point
to represent each county, then ask:

> *"What was the temperature and rainfall every single day from 2000 to 2023?"*

That's ~8,700 days per county — far too granular. So we **squeeze each growing
season (April–October) down to five numbers**:

| feature | plain meaning |
|---|---|
| `gdd_base10` | total accumulated warmth — how much "growing weather" the crop got |
| `precip_total_mm` | total rain |
| `tmax_mean_c` | average daily high |
| `tmin_mean_c` | average daily low |
| `heat_days_30c` | number of days hot enough (≥30 °C) to stress the plant |

For Chester in 2023: `gdd 1736.7`, `rain 812.4 mm`, `high 23.8 °C`, `low 12.12 °C`,
`23 hot days`.

**Unlike soil, these change every year.** That's the whole point of including them:

| year | warmth | rain | hot days | → yield |
|---|---|---|---|---|
| 2018 | 1930.0 | 1209.7 | 35 | 167.3 |
| 2019 | 1960.8 | 961.0 | 36 | 195.7 |
| 2020 | 1773.5 | 923.0 | 36 | 193.2 |
| 2021 | 1919.4 | 950.0 | 42 | 204.7 |
| 2022 | 1817.4 | 867.7 | 39 | 210.6 |
| 2023 | 1736.7 | 812.4 | 23 | 175.4 |

### Step 4 — Glue it into one row

`assemble.py` matches these three sources on **county + year** and produces one
flat row:

```
crop              corn_grain_yield
county            CHESTER (FIPS 029)
year              2023
─────────────────────────────────── the answer we want to predict
yield_value       175.4 bu/acre
─────────────────────────────────── soil (same every year)
topsoil_sand_pct  35.18
topsoil_silt_pct  47.28
topsoil_clay_pct  17.52
topsoil_ph        5.72
topsoil_om_pct    2.41
topsoil_cec       10.53
dominant_taxorder Ultisols
dominant_drainage Well drained
─────────────────────────────────── weather (changes every year)
gdd_base10        1736.7
precip_total_mm   812.4
tmax_mean_c       23.8
tmin_mean_c       12.12
heat_days_30c     23.0
```

**That is one training example.** Thirteen input numbers, one answer — shown
condensed here. The shipped model actually gets a fuller set on the same row: the
weather also **split into sub-seasons** (was there a *July* drought during
pollination?) and a **satellite greenness reading** of the crop (NDVI). Those extra
time-varying numbers are what let it break past its old accuracy ceiling — the full
story is in §6. The eight soil + five season-weather numbers above are the easiest
slice to picture, so we follow those.

Do this for every county and every year and you get **1,314 rows for corn** across
66 counties and 24 years. That table is what the model learns from.

### Step 5 — The model learns the pattern

The model is a **random forest** (explained in §4). You hand it all 1,314 rows and
it works out rules like *"hot dry years in low-organic-matter counties yield less."*

It never sees county names. It only sees the thirteen numbers.

### Step 6 — It makes a prediction

Give it Chester's soil and 2023's weather, and it predicts **179 bu/acre**. The
actual was 175.4, so it was about 4 bushels high — well inside its typical honest
error of ~13.6 bu/acre (leave-counties-out).

### Step 7 — The prediction becomes advice

The predicted harvest is handed to the recommender, which reasons:

> *179 bushels of corn will leave this field. Corn removes about 0.37 lb of
> phosphate per bushel. So 179 × 0.37 ≈ 66 lb of phosphate must be replaced.*

Combined with the farmer's soil test (Mehlich-3 P = 22 ppm, which is below Penn
State's optimum of 30), it prints:

```
[DEFICIENCY] Phosphate (P2O5)
  measured : Mehlich-3 P 22 ppm
  finding  : below optimum (<30 ppm); yield response to added P likely
  action   : Apply P2O5: build toward optimum plus replace crop removal
  rate     : ~66 lb P2O5/acre to replace removal at 179 bu/acre
             predicted yield (YieldModel) + build-up to optimum
  source   : Penn State AASL Mehlich-3 phosphorus interpretation
```

**That is the full journey**: public download → one table row → learned prediction
→ cited fertilizer advice.

---

## 3. What each file does

```
data_ingest.py   →  assemble.py  →  yield_model.py  ─┐
                                                     ├→ main.py → report
soil_classifier.py (separate track) ─────────────────┤
recommender.py  ←  calibration.py ───────────────────┘
```

| File | Takes in | Puts out | In one sentence |
|---|---|---|---|
| `data_ingest.py` | Web APIs | CSV files | Downloads the public datasets and cleans them (yields, soil, lab pedons, weather + sub-season windows, soil moisture, and — via Earth Engine — satellite NDVI) |
| `assemble.py` | Those CSVs | 1 combined CSV | Glues yields + soil + weather + NDVI into one row per county-year-crop (soil-moisture & drought layers ride along but were tested-null) |
| `yield_model.py` | Combined CSV | Predictions + scores | Learns to predict yield, and honestly measures how well it does |
| `soil_classifier.py` | Lab pedon CSV | Accuracy scores | Proves soil type is identifiable from cheap measurements |
| `recommender.py` | A soil test | Printed advice | Applies published Penn State rules — no learning involved |
| `calibration.py` | Probe/lab pairs | A correction formula | Where a future cheap sensor would plug in |
| `main.py` | everything | Text report | Runs all of it in order |
| `report.py` | everything | PNG figure | Draws the four diagnostic charts |

`config/sources.yaml` holds every setting — which crops, which years, which
states — so nothing is buried in code.

---

## 4. How the yield model learns

### What a random forest is

A **decision tree** is a flowchart of yes/no questions:

```
Was it a hot year (heat_days > 35)?
├── yes → Was rainfall under 900 mm?
│         ├── yes → predict 165 bu/acre
│         └── no  → predict 185 bu/acre
└── no  → Is organic matter above 3%?
          ├── yes → predict 195 bu/acre
          └── no  → predict 175 bu/acre
```

The tree builds itself by repeatedly finding the split that best separates high
yields from low ones.

One tree is unreliable — it latches onto quirks in the data. So a **random forest**
builds **400 trees**, each on a slightly different random slice of the data and
random subset of features, then averages their answers. Averaging cancels out
individual trees' mistakes. That's the whole trick.

### Why one model per crop

Corn is measured in **bushels/acre** (~175). Hay is measured in **tons/acre** (~3).
If you pooled them, an average error of "10" would be trivial for corn and
catastrophic for hay. Separate models keep every number in its own units.

### The `year` switch

Yields rose ~2.08 bu/acre every year from 2000–2023 — better seed genetics and
better management. That's real, but it's not soil or weather.

So the model can be run two ways, and we report both:

| mode | `year` included? | answers |
|---|---|---|
| **attribution** | no | "How much do soil and weather *alone* explain?" (honest, conservative) |
| **forecast** | yes | "What will this field actually yield?" (what a farmer needs) |

Hiding `year` inside the model would be dishonest. Excluding it entirely would make
the model useless for forecasting. So it's an explicit switch, and both results are
published.

---

## 5. Why we test it three different ways

This is the part most likely to feel abstract, so here's the intuition.

**The problem:** if you test a model on data it already studied, you learn nothing
about whether it actually understands anything. It's an open-book exam using the
exact questions from the study guide.

We test three ways, each harder and more realistic than the last.

### Test 1 — Random split (too easy)

Shuffle all 1,314 rows, train on 80%, test on 20%.

**Score: R² = 0.69.** Looks great. It's misleading.

Why: Chester County appears ~24 times (once per year), and **every one of those
rows has identical soil numbers** (pH 5.72, CEC 10.53…). So Chester rows land in
both the training and test sets. The model doesn't need to understand agronomy — it
just memorizes *"when I see pH 5.72 and CEC 10.53, that's the ~180 bu/acre county."*

It's recognizing a fingerprint, not doing science.

### Test 2 — Leave-counties-out (honest)

Hold out **entire counties**. Train on 53 counties, test on 13 the model has never
seen — then rotate.

**Score: R² = 0.60.**

This is the real question: *a new farm walks in from a county we never trained on —
can we predict its yield?* No fingerprint to memorize.

**The 0.69 → 0.60 drop is exactly the memorization we just described**, measured.

### Test 3 — Forward in time (hardest, most realistic)

Train only on 2000–2017. Predict 2018–2023.

**Score: R² = 0.46.**

This mirrors reality: a farmer forecasting next season has only the past. Tests 1
and 2 both let the model see 2022 while predicting 2015, which is time travel.
(This 0.46 is itself a recent win — it was **0.28** before we added the time-varying
features in §6.)

### The punchline

The same model is "0.69 good," "0.60 good," or "0.46 good" depending purely on how
you test it. Papers that report only the first number aren't lying, but they're
answering an easier question than the one that matters.

> **What is R²?** It's "what fraction of the variation did we explain?"
> **1.0** = perfect. **0.0** = no better than always guessing the average.
> **Negative** = worse than guessing the average. Take away the year-trend from the
> forward forecast and it drops to a weak **+0.09** — and on the older, weaker
> ingredient list (before satellite greenness) it was actually **−0.44**, worse than
> guessing. Adding a direct picture of the crop is what pulled it back above zero.

---

## 6. The ceiling, shown with real numbers

This is the project's most important finding, and it's much simpler than the paper
makes it sound.

Look at six years of Chester County, with **all** the columns:

| year | pH | organic matter | CEC | warmth | rain | hot days | **yield** |
|---|---|---|---|---|---|---|---|
| 2018 | 5.72 | 2.41 | 10.53 | 1930.0 | 1209.7 | 35 | **167.3** |
| 2019 | 5.72 | 2.41 | 10.53 | 1960.8 | 961.0 | 36 | **195.7** |
| 2020 | 5.72 | 2.41 | 10.53 | 1773.5 | 923.0 | 36 | **193.2** |
| 2021 | 5.72 | 2.41 | 10.53 | 1919.4 | 950.0 | 42 | **204.7** |
| 2022 | 5.72 | 2.41 | 10.53 | 1817.4 | 867.7 | 39 | **210.6** |
| 2023 | 5.72 | 2.41 | 10.53 | 1736.7 | 812.4 | 23 | **175.4** |

**Stare at the soil columns.** They never change. Not once. Meanwhile yield swings
from 167 to 211 — a 43-bushel spread.

**A number that is identical every year cannot explain something that changes every
year.** The soil columns are literally incapable of telling you why 2022 beat 2018.
Only the weather columns and the trend can do that.

### Putting a number on it

Split all the variation in corn yield into two buckets:

- **Variation between counties** (Lancaster is generally better than Cameron): **20%**
- **Variation within a county, year to year** (Chester 2022 vs Chester 2018): **80%**

Our eight soil features are stuck working on that 20% slice. They contribute
**nothing** to the 80%. And that 80% has only two things speaking to it: one
weather point per county, and a trend line.

**So the model isn't underperforming because we picked the wrong algorithm. It's
underperforming because the ingredients can't answer the question.** We tested six
alternatives — gradient boosting, ExtraTrees, tuning, coordinates, interaction
terms, detrending — and none helped, exactly as this predicts.

### The evidence this is right

When we test features under the honest leave-counties-out split (bigger number =
the model relies on it more):

| feature | random split | honest split | what happened |
|---|---|---|---|
| `year` (the trend) | 0.452 | 0.442 | **held** — the single biggest signal |
| `ndvi_jul` (satellite greenness) | 0.224 | 0.226 | **held** — a real observation of the crop |
| `precip_jul_mm` (July rain) | 0.032 | 0.029 | **held** — timing genuinely matters |
| `topsoil_om_pct` | 0.099 | 0.054 | **halved** — was mostly county fingerprint |
| `topsoil_cec` | 0.029 | 0.008 | **shrank** — county fingerprint |
| `topsoil_clay_pct` | 0.009 | 0.000 | contributes nothing real |

The **time-varying** features (the trend, satellite greenness, July rain) keep their
value on unseen counties; the **static soil** features lose half or more. That's the
fingerprint effect showing up per-feature, precisely where the theory says it should —
what changes over time generalizes, what's frozen per county only helps you recognize
that county.

### Why this matters

It converts *"our model is mediocre"* into *"here is the exact information budget,
and here is the one thing that could improve it."* And that one thing is a soil
measurement **that changes over time** — which is exactly what a cheap field probe
would provide (§9).

### Update (2026-07-24): we found the missing kind of feature — and it worked

The paragraph above makes a bet: the only thing that helps is a feature that
**changes from year to year within a county.** We tested that bet by adding four
free public data sources that do change over time, and kept only what improved the
*honest* forecast (the "predict a future year we've never seen" test):

| what we gave the model | forward-forecast score (corn) |
|---|---|
| the old ingredients (soil + whole-season weather) | 0.28 |
| **+ weather broken into sub-seasons** (was there a *July* drought during pollination?) | 0.39 |
| **+ satellite greenness of the actual crop** (NDVI, through July) | **0.46** |

- **Splitting the weather by month helped.** A whole-season rainfall total hides a
  dry July — the worst possible time for corn — so giving the model each part of the
  season let it *see* the timing that decides the year.
- **Watching the crop from space helped more.** A satellite greenness reading is a
  picture of how the actual crop is doing, which captures things weather never can
  (planting choices, pests, management). We only use readings **through July**, so
  we're never peeking at the near-harvest crop and cheating.
- **Two other free sources did *nothing*** — a "drought index" and a satellite
  *soil-moisture* estimate. Why? Both are really just repackaged rainfall-and-
  temperature, which the model already had. They weren't *new* information.

That last point is the important one for the probe. Free data got us from 0.28 to
0.46 — real progress — but it plateaus, because everything free is ultimately built
from the same weather. **A real probe in the ground measures the actual water in
*this* field** — something no weather-based estimate can reproduce. So the ceiling
didn't turn out to be a wall; it turned out to be a staircase, and the probe is the
next step up (§9).

---

## 7. The soil classifier: why it exists at all

This module confuses people because it doesn't help the farmer. It isn't supposed
to. **It's a test of whether the whole project's assumption is true.**

The assumption: *cheap, measurable soil numbers actually contain real information
about soil.* If that's false, everything else is built on sand.

So we test it head-on. We take 7,686 real lab-analyzed soil profiles, hide their
official classification, and ask a model to recover it from measurements alone
(clay content, pH, CEC, organic carbon, and how those change with depth).

- **Result: 69.8% accurate**, versus **34.1%** for always guessing the most common
  type. Roughly double.
- More importantly, **the features it relies on most are clay content and how much
  clay increases with depth.**

That second point is the real result. Soil scientists classify soil partly by
looking for an **argillic horizon** — a layer where clay has washed down and
accumulated. The model, given no instruction, independently discovered that clay
increase with depth is the key signal.

We deliberately deleted latitude and longitude from its inputs, so it can't have
cheated by memorizing geography. It genuinely learned soil science from numbers.

**Conclusion: the assumption holds.** Cheap numbers carry real soil information,
so the rest of the system is legitimate.


## 8. The recommender: not a model

Nothing here is learned. It's a lookup table with citations.

**Why not learn it?** Three reasons:

1. **It's circular.** Our yield data came from farmers who were *already following
   Penn State's recommendations*. Learning from those yields would just rediscover
   the advice, distorted.
2. **It's already solved.** Penn State ran decades of controlled trials. Our
   observational data can't compete with that.
3. **It's free.** The rules are published.

So the engine holds thresholds like:

```
Mehlich-3 phosphorus:
  below 30 ppm  → deficient, apply phosphate
  30–50 ppm     → optimum, maintenance only
  50–200 ppm    → plenty, apply none
  above 200 ppm → excessive, environmental concern
```

Each one names its source, so an agronomist can audit every line without evaluating
a model.

### The part worth noticing: it refuses

SSURGO gives us pH, CEC, and organic matter — but **not** phosphorus or potassium.
Those require an actual lab test of the actual field.

So when they're missing, the engine does *not* guess from county averages:

```
[UNKNOWN] Phosphate (P2O5)
  measured : Mehlich-3 P not provided
  finding  : cannot interpret P without a lab soil test (SSURGO has no P)
  action   : Obtain a Mehlich-3 P soil test for the field.
```

Same for lime: without a buffer-pH test it says a rate **cannot be derived** rather
than inventing tonnage.

Getting fertilizer wrong costs real money and pollutes real water. Saying "I don't
know, go test" is the correct answer, and the system is built to give it.

The same logic gates the yield forecast: crops whose honest score falls below 0.30
are **withheld** entirely, with a message explaining that a forecast would be no
better than quoting the county average. Currently **withheld**: oats (0.17), alfalfa
(0.25), and barley (0.28). The crops that **clear** the gate are corn grain (0.60),
soybeans (0.58), corn silage (0.55), hay (0.41), and — newly, after the time-varying
features lifted it over the bar — winter wheat (0.32).

---

## 9. The calibration seam

Lab soil tests cost money and take time. A cheap electronic probe would be great —
but cheap probes are *biased*: a probe might read pH 6.1 where the lab says 5.7.

`calibration.py` fixes that with the simplest honest method. Take a handful of
samples, measure each **both** ways, and fit a straight line:

```
lab_value = slope × probe_reading + intercept
```

Now any raw probe reading converts to a lab-equivalent number, which flows into the
**unchanged** recommender.

**Why this is the right next step** — and this connects back to §6: the ceiling
exists because no soil measurement in the system changes over time. A probe reading
taken *in a specific field in a specific season* is exactly that missing thing. It's
not just finer detail; it's the only available input that could address the 80% of
variation that soil data currently cannot touch.

The 2026-07-24 update to §6 makes this argument *sharper*, not weaker. Free
time-varying data (weather timing + satellite greenness) already pushed the forecast
from 0.28 to 0.46 — so the mechanism is proven. But the two free *soil* estimates we
tried (a drought index and satellite soil moisture) added nothing, because they're
just repackaged weather. A probe is the one thing that measures the **actual water in
the actual field** — independent of the weather everything else is built from. That's
the specific, remaining signal it would contribute.

**Honesty note:** no probe exists yet. The demo runs on clearly-labeled fake data
just to prove the interface works. Fake results are never reported as findings.

---

## 10. Glossary

**Bushel** — a volume measure for grain. Corn yields run 100–210 bushels per acre
in Pennsylvania.

**CEC (cation exchange capacity)** — how well soil holds onto nutrients. Higher =
nutrients stay put instead of washing away. Chester's is 10.53.

**County-year** — one county in one year, e.g. "Chester 2023." The basic unit of
our table.

**Cross-validation** — testing a model on data it didn't learn from, repeatedly,
by rotating which slice is held back.

**FIPS** — the federal ID number for a county. Chester is `029`. Used to match
records across datasets.

**Feature** — an input number the model uses to predict. The shipped model uses
about two dozen: static soil, growing-season weather (whole-season *and* split into
sub-season windows), satellite NDVI, and the year trend.

**Leakage** — when a model accidentally sees information it shouldn't, making it
look better than it is. County fingerprinting (§5) is our main example.

**Mehlich-3** — a standard lab procedure for measuring plant-available phosphorus
and potassium. Requires a real lab; SSURGO doesn't have it.

**Organic matter (OM)** — decomposed plant material in soil. Improves water
retention and fertility. Chester's is 2.41%.

**Pedon** — one complete soil profile dug and analyzed from surface down through
its layers. The soil classifier uses 7,686 of them.

**Permutation importance** — how to tell if a feature matters: scramble that
column and see how much worse predictions get. Big drop = important feature.

**pH** — acidity. Below 7 is acidic. Most crops want ~6.0; Chester is 5.72, which
is why the system recommends lime.

**R²** — fraction of variation explained. 1.0 perfect, 0.0 = no better than
guessing the average, negative = worse than that.

**Random forest** — 400 decision trees, each trained on a random slice, whose
answers are averaged.

**Soil order** — the top level of soil classification (Alfisols, Ultisols, etc.).
Chester is Ultisols: weathered, acidic, typical of the humid southeast.

**SSURGO** — the USDA's detailed soil survey. Static — it has no time dimension,
which is the root of the ceiling in §6.

**Target** — the thing being predicted. Here, yield.

---

## Where to go next

- **Want the formal writeup?** → `METHODS.md`
- **Want to see it run?** → `python -m strata.main`
- **Want the charts?** → `python -m strata.report`, then open
  `data/processed/strata_dashboard.png`
- **Want to read code?** → start with `assemble.py`; it's the shortest and shows
  how the pieces connect.
