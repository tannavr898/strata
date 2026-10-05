# Changelog

One entry per prompt that changes code or docs, newest first. Each entry records
the date, the prompt (from `.claude/last_prompt.txt`), the files changed, what
changed and why, and metrics before and after when a change moves them. Metrics
refer to lines in [`metrics.jsonl`](metrics.jsonl) by timestamp.

Changes before this log began are summarized in `METHODS.md` §14 and in the git
history (`git log`).

---

## 2026-10-03: Field map, deletion, model rebuild — and barley crossing the gate

**Prompt.** "finsih out the rest of the app and thhen we can go over it"

### The finding that matters most

Rebuilding the models under the **installed** scikit-learn (1.8.0) rather than the
one that pickled them (1.9.0) changed the results, and **barley crossed the
product gate**: LCO forecast R² **0.388 → 0.283**, now withheld.

Same analysis table (hash `923c106626a51b00`), same seed, same code. Two earlier
builds under 1.9.0 (2026-08-31 and 2026-09-19) were byte-identical to each other,
so this is an implementation difference, not randomness.

| crop | county-years | 1.9.0 | 1.8.0 | Δ |
|---|--:|--:|--:|--:|
| corn grain | 1,314 | 0.610 | 0.601 | 0.009 |
| soybean | 1,033 | 0.586 | 0.581 | 0.005 |
| corn silage | 1,044 | 0.554 | 0.553 | 0.001 |
| winter wheat | 667 | 0.312 | 0.318 | 0.006 |
| **hay** | 567 | 0.493 | 0.408 | **0.085** |
| **barley** | 366 | 0.388 | 0.283 | **0.105 — crosses the gate** |

The two crops that moved most are the two with the fewest county-years. The honest
reading: **a leave-counties-out estimate from a few hundred county-years is not
stable to a decimal place**, and a crop sitting between 0.28 and 0.39 depending on
a library version is not reliably above a 0.30 threshold.

It is also the gate working as designed — nobody re-checked barley by hand. The
threshold is a constant the build re-applies, so the decision followed the
evidence the moment the evidence moved. `README.md` and `METHODS.md` §10 are
updated, both with the instability stated rather than just the new number.

This vindicates the version check added earlier today: the warning was not
theoretical. Note the earlier single-point verification (corn predicting 179.24
from the 1.9.0 pickle) was *not* sufficient evidence of safety — it tested a
fitted model's inference, not a refit's cross-validation.

### Files changed

| file | change |
|---|---|
| `src/strata/web/charts.py` | **new**: `field_map()`, `MapSite`, `FieldMap`, `SEQUENTIAL` ramp |
| `src/strata/web/templates/map.html` | **new**: schematic map, north arrow, scale bar, ramp legend |
| `src/strata/web/edit.py` | **new**: 4 delete routes (readings / soil test / spot / field) |
| `src/strata/web/views.py` | **new** route `GET /field/{id}/map` |
| `src/strata/web/core.py` | **new** `tidy` filter: `R^2` → R², `--` → em dash on product surfaces |
| `src/strata/web/templates/{import,field,field_form,models}.html` | delete UI; tidied notation |
| `src/strata/web/static/strata.css` | map, sequential ramp legend, destructive-action styles |
| `data/models/*`, `docs/metrics.jsonl` | rebuilt; manifest now records `sklearn_version: 1.8.0` |
| `README.md`, `docs/METHODS.md`, `docs/TECHNICAL_REPORT.md` | gate tables updated; new report §9.3, §9.4 |
| `tests/test_charts.py`, `tests/test_web_forms.py` | +11 map tests, +9 deletion tests |

### The map

A schematic in real coordinate space — no tiles, no API key, no script. The part
that had to be right is the projection: longitude is scaled by cos(latitude),
without which a Pennsylvania field renders ~23% too wide east-west and every spot
sits wrong relative to every other, defeating the one job it has (walking back to
a planted probe). One px/m scale on both axes, north up, scale bar. Verified: a
350 m true separation renders at 362 px against a 1.035 px/m scale.

It states on screen that it is **not** an interpolated surface and computes how
many more spots a believable gradient would need (7, for the seeded field) —
noting that making many spots affordable is the argument for the probe.

### Deletion

Four scoped deletes, all POST-only: a GET that destroys data can be fired by a
prefetch or a crawler. Readings delete by channel/day/source; a spot cascades to
its readings; a field requires its name typed back. Tests assert the scoping
holds — notably that one field's delete never crosses into another's, which a
missing join clause would do silently.

### Verification

- `python -m pytest`: **152 passed** (132 before; +11 map, +9 deletion).
- The scikit-learn `InconsistentVersionWarning` is gone after the rebuild.
- All 9 routes 200 against a live server, nothing logged; map inspected visually.

### Behavior change

**Barley no longer ships a yield forecast.** Otherwise additive: a new read-only
map page, delete paths that did not previously exist, and display-only typography.

### Still open

No cited interpretation for organic matter — the one gap that most limits the soil
status, and a research task rather than a coding one. `calibration` is still not
device-scoped. There is no print/PDF report (deliberately deselected).

---

## 2026-10-03: Visual review of the running app, and the fixes it found

**Prompt.** "look at everything we have built so far"

**How.** Launched the app and screenshotted every page with headless Chrome
(already installed; no driver or package added) at desktop and phone widths, in
both colour schemes — rather than inferring layout from extracted HTML text, as
every previous UI change in this project had been.

**Files changed.**

| file | change |
|---|---|
| `src/strata/web/status.py` | `alerts_for` groups a field's findings instead of emitting one alert per measure; new `_join` helper |
| `src/strata/web/static/strata.css` | alert actions stack below ~620px; `.rec-amount` no longer green; field cards distribute slack before the yield block; chart data-quality tags no longer pushed to the far right |
| `src/strata/web/charts.py` | y-axis tick density loosened (a 27-unit range was drawing only two gridlines) |
| `tests/test_soil_status.py` | 3 tests for alert grouping |

**What the review found.**

1. **The attention list repeated itself.** One field with pH, P and K all below
   optimum produced three near-identical alerts, which read as three problems
   rather than one soil test with three findings — and pushed genuinely different
   items (an unassessed field, a withheld forecast) down the page. Now grouped per
   field: *"North Field: pH, P and K below Penn State optimum"* with the measured
   values in the detail. Six alerts became four on the demo data. Excess
   phosphorus deliberately stays separate: it carries the opposite instruction
   (apply none), and folding it in with shortfalls would blur two opposite actions.

2. **Fertiliser rates were green.** Green means "no action needed" everywhere else
   in this app, so a lime or P₂O₅ rate — which appears precisely *because*
   something is wrong — was reassuring the farmer directly beneath an amber
   "deficiency" tag. Rates are now neutral with a weighted left border: a number
   to act on, not a verdict.

3. **Two gridlines across a 27-unit range** made the season chart hard to read a
   value off. The tick selector was rejecting a 5-unit step by one tick.

4. **Alert buttons were cramped on a phone.** Now full-width and stacked below the
   text under 620px.

**A finding that turned out not to be one.** Phone screenshots appeared to show
alert text clipped at the right edge, which looked like a responsive overflow bug.
Measuring it from inside the page (a same-origin probe reporting
`document.documentElement.scrollWidth` and every element extending past the
viewport) returned **viewport 420 / scrollWidth 420, nothing overflows** — the
clipping was an artefact of headless Chrome's `--window-size` without mobile
emulation, not a defect. The stacking change in (4) was kept because it genuinely
improves narrow-screen tap targets, but it fixes ergonomics, not a bug. Recorded
because "the screenshot looked wrong" is not evidence, and the probe is the
repeatable way to settle it.

**Verification.** `python -m pytest`: **132 passed** (129 before, +3). Every page
re-rendered after the fixes and inspected: overview, fields, field detail, season
(light and dark), models, calibration, and 420px phone width.

**Behavior change.** Display only. No model, rule, recommendation or stored value
changed.

**Metrics.** None moved; no evaluation run was triggered.

**Still open.** The schematic map; no cited interpretation for organic matter;
`calibration` is not device-scoped; `data/models/` is pickled under scikit-learn
1.9.0 against 1.8.0 installed; the Models page still prints the rule engine's raw
`R^2` and `--` in its withheld explanations.

---

## 2026-10-03: Season view — one channel, every spot, one growing season

**Prompt.** "keep going with the season trends"

**Files changed.**

| file | change |
|---|---|
| `src/strata/web/charts.py` | **new**: `season_chart()` plus `Series`/`Point`/`Band` geometry, and the validated categorical `SERIES_COLORS` |
| `src/strata/web/templates/season.html` | **new**: chart, legend, direct labels, phenology bands, table view |
| `src/strata/web/views.py` | **new** route `GET /field/{id}/season?channel=&year=` |
| `src/strata/db.py` | **new**: `season_readings()`, `reading_years()` |
| `src/strata/web/static/strata.css` | season chart styles; `color-scheme: light dark` on `:root` |
| `src/strata/web/templates/field.html` | "Season view" link from the readings card |
| `tests/test_charts.py` | **new**, 15 tests |
| `tests/test_web_app.py` | 5 season-view tests; fixture gains a second spot so multi-series behaviour is exercised |
| `docs/TECHNICAL_REPORT.md` | new §9.2; route table updated |

**What and why.** The app could show one spot's single channel as a sparkline, and
nothing else — there was no way to compare spots or see a season. The season view
is the farmer-facing picture of the project's own argument: a measurement that
varies **within a field** and **within a season**, the two axes the static county
snapshot lacks and the reason for the §7 ceiling.

The phenology bands behind the readings are the yield model's *own* sub-season
feature windows from `config/sources.yaml`, the same ones `assemble.py` turns into
per-window precipitation and heat-day features. Silking is drawn darker because it
is the window the corn model leans on hardest. It is the one screen where the
grower and the model visibly share a calendar.

**Colour.** Series colour is categorical — *which spot*, not *how good* — and is
kept strictly disjoint from the app's status palette, since a site line in amber
would assert something about that site. The eight slots are a published validated
order, assigned in fixed sequence, never cycled; a ninth spot is named as
unplotted rather than reusing a hue. Validated against this app's surfaces with
`validate_palette.js`: worst adjacent CVD ΔE 9.1 light / 8.4 dark, normal-vision
19.6 / 19.3 — all pass. Light mode **fails** the 3:1 contrast check, which is not
dismissable, so the template ships direct labels *and* a table view. Uncalibrated
points are hollow rather than a second colour.

**Found and fixed while building.**

- Dark-mode series colours could never have worked: the template set `--c` in an
  inline `style` attribute, and an inline style outranks the
  `@media (prefers-color-scheme: dark)` block that tried to swap it. Replaced with
  `light-dark()` emitted per series, plus `color-scheme: light dark` on `:root`.
- The two new view tests initially failed for a test-harness reason worth keeping:
  `visible_text()` strips `<svg>` by design (it models prose a reader sees), so
  band and series labels are invisible to it. Those assertions now run against the
  raw markup, and the fixture gained a second spot so the legend path is real.

**Verification.**
- `python -m pytest`: **129 passed** (109 before; +15 charts, +5 season view).
- Geometry checked numerically on the seeded field: 36 points / 3 spots, bands in
  calendar order and non-overlapping, every point inside the frame, no NaN,
  direct labels ≥ 20px apart.
- All 9 routes 200 against a live server, nothing logged.

**Behavior change.** Additive. No model, rule, or recommendation changed.

**Metrics.** None moved; no evaluation run was triggered, so `metrics.jsonl` has
no new line.

**Deviation recorded** (report §9.2): visualisation guidance wants a JS crosshair
and tooltip on line charts. The app's no-script property is documented and
deliberate, so hover is SVG `<title>` (a native browser tooltip, no script) with
the full table view beside every chart.

**Still open.** The schematic map; no cited interpretation for organic matter;
`calibration` is not device-scoped; `data/models/` is still pickled under
scikit-learn 1.9.0 against 1.8.0 installed.

---

## 2026-10-03: App shell, Overview dashboard, and soil status

**Prompt.** "Can we make the app cleaner now and more functional. I want this to
be a full blown professional App. Ask questions to get me thinking…" — followed by
the direction set in that exchange: farmer tool **and** team research instrument,
modern-SaaS shell, hamburger on mobile, season-scale trends, schematic map first
(field-level interpolation later), and "overall soil health of the different
fields" as the headline number.

**Files changed.**

| file | change |
|---|---|
| `src/strata/web/status.py` | **new**: `SoilStatus`, `Chip`, `Alert`, `soil_status()`, `alerts_for()` |
| `src/strata/web/templates/overview.html` | **new**: attention list + per-field status cards |
| `src/strata/web/templates/base.html` | rewritten: sidebar shell, Farm/Lab groups, CSS-only hamburger, build stamp, library-mismatch banner |
| `src/strata/web/static/strata.css` | chrome block replaced with the shell; overview, alert, chip and card styles added (320 → 482 lines) |
| `src/strata/web/views.py` | `/` is now the Overview; the field list moved to `/fields` |
| `src/strata/web/core.py` | `base_context` derives `nav` from the request path and exposes `library_mismatch` |
| `src/strata/artifacts.py` | `Manifest.sklearn_version`; `library_mismatch()`; a warning on load when versions differ |
| 9 templates | dropped the in-content `<h1>` now that the topbar carries the title; each declares `{% block heading %}` |
| `tests/test_soil_status.py` | **new**, 14 tests |
| `tests/test_web_app.py` | footer assertion updated to the new provenance line |
| `docs/TECHNICAL_REPORT.md` | §9 rewritten; new §9.1 "Soil status, and why it is not a score" |

**What and why.**

*Shell.* The app was a single-column page with a flat nav, and `/` redirected into
a field when only one existed — so there was no screen answering "how is my
operation doing?". A sidebar now splits the app by audience: **Farm** for the
grower, **Lab** (calibration, models) for the team's research use, which is the
dual audience that was asked for. The drawer is a CSS `:checked` selector, not
JavaScript, keeping the no-script property.

*Soil status is deliberately not a score.* The request was "overall soil health",
whose natural UI is one number. `recommender.CITATIONS` covers only pH, Mehlich-3
P and Mehlich-3 K; a composite with organic matter or biology needs weights no
publication supplies, and a circular IQR scorer was already removed from this
project once. `status.py` emits a roll-up of cited findings instead — each chip
names its citation, CEC and organic matter appear as context labelled "no rule",
and `unmeasured` is a distinct amber state rather than a quiet pass. Reasoning is
in the module docstring and report §9.1.

**Found and fixed.**

- Farmer-facing copy said *"Yield response to added pH is likely"* — a generic
  nutrient template applied to the one measure that is not an input. pH alerts now
  say liming is advised and that the rate needs a buffer pH reading.
- **scikit-learn version mismatch.** `data/models/` was pickled under 1.9.0; 1.8.0
  is installed, and sklearn warns results may be invalid. Verified harmless *for
  this build* — the served corn model still predicts Chester 2023 at 179.24 bu/ac,
  matching the recorded 179 — but the artifacts now record `sklearn_version` and
  both log and banner a mismatch. Existing manifests read `unknown` and stay
  loadable; the next `build_models.py` run records the real version.

**Behavior change.** `/` renders the Overview instead of redirecting to a field;
the field list is at `/fields`. No change to models, rules, or recommendations.

**Verification.**
- `python -m pytest`: **109 passed** (95 before; +14 in `test_soil_status.py`).
- All 10 routes return 200 against `TestClient`, including the two demo fields.
- Overview renders both states correctly: North Field shows pH/P/K chips with
  "below Penn State optimum" and CEC/OM as "no rule"; Back Lot shows "No soil test
  on file" with no yield estimate.

**Metrics.** None moved. No model, feature, or rule changed, and no evaluation run
was triggered, so `metrics.jsonl` has no new line.

**Still open.** Season trends and the schematic map (both agreed, not yet built);
no cited interpretation for organic matter; `calibration` is not device-scoped.

---

## 2026-09-19: Rewrite history to drop Claude co-author trailers

**Prompt.** "go ahead and rewrite and force-push"

**Files changed.** None. This entry records a repository-level change: every
commit on `main` was rebuilt with its `Co-Authored-By: Claude …` trailer removed.

**What happened.** Each of the 15 commits was recreated with `git commit-tree`
using the same tree, author, committer and dates, and a message with the trailer
stripped (14 trailers across 15 commits; the initial commit had none). `main` was
then force-pushed with `--force-with-lease` pinned to the previous head.

- before: `9d5a69e` · after: `7f3d9c1`
- `git diff backup/pre-trailer-strip main` is empty — no file content changed
- verified after fetching: 0 Claude trailers on `origin/main`
- local backup branch `backup/pre-trailer-strip` still points at `9d5a69e`

**Consequence.** All commit IDs changed. Anyone holding a clone must
`git fetch origin && git reset --hard origin/main` (or re-clone) instead of
pulling. Future commits carry no trailer, per `.claude/settings.json`
(`attribution.commit` / `attribution.pr` = `""`).

**Note on the Stop hook.** This turn tripped `.claude/hooks/check_docs.py`,
because the code fingerprint includes `HEAD` and a rewrite changes it. That is the
intended behavior — a commit *is* a code change — and this entry is the response.

---

## 2026-09-19: Fix stale docs; attribute commits to the user only

**Prompt.** "fix those and can you also remove the part where it says you and i
committed on the github branch and just make it me please?"

**Files changed**

| file | change |
|---|---|
| `docs/METHODS.md` | §6 headline and all-crops tables and §10 gate table regenerated from the 2026-09-19 logged runs (barley now SHIP 0.388; hay 0.493; corn 0.610; oats 0.142; alfalfa 0.268); header, provenance, and §14 changelog updated; "0.60" → "0.61" throughout; probe wording; `PYTHONPATH=src` in the verification commands |
| `README.md` | Setup now documents `PYTHONPATH=src`; oats/alfalfa bullet → 0.14 / 0.27; barley/wheat gate bullet no longer credits NDVI with a rise the 07-25 run did not show; probe wording ("hardware exists, no paired data yet") |
| `docs/HOW-IT-WORKS.md` | gate list matches the build; corn LCO 0.61 and MAE ~13.5; probe wording; `PYTHONPATH=src` on run commands |
| `src/strata/calibration.py`, `src/strata/main.py` | docstring and printed text: the probe exists, but paired probe/lab data does not yet (no behavior change) |
| `docs/TECHNICAL_REPORT.md` | §11.2/§11.3 marked resolved |
| `.claude/settings.json` | `attribution.commit` / `attribution.pr` set to `""` so Claude Code adds no co-author trailer to future commits or PRs |

**Why.** The report's §11.3 had found these discrepancies. The METHODS tables came
from a 2026-07-25 run on an earlier analysis table and disagreed with the served
build. The logged runs (`build_models`, `main`, and `python -m strata.yield_model`)
agree with `MANIFEST.json` to four decimals, so they are now the single source.

**Metrics.** No model change. New numbers quoted in the docs (corn attribution LCO
MAE 15.17, no-trend forward MAE 19.36) came from a one-off run of the same
`YieldModel` methods. Per-crop attribution and forward R² come from
`python -m strata.yield_model` on 2026-09-19.

**History rewrite.** Removing the `Co-Authored-By: Claude` trailers from the 15
pushed commits required rewriting `main` and force-pushing. The auto-mode safety
check blocked it until the user gave an explicit go-ahead; it was then done in the
turn recorded in the entry above.

---

## 2026-09-19: Technical report, documentation system, and metrics logging

**Prompt.** "our hardware is almost ready, i just want a detailed overview report
with everything", followed by the documentation-system instructions (file layout,
report contents, metrics log, CLAUDE.md policy, UserPromptSubmit/Stop hooks).

**Files changed**

| file | change |
|---|---|
| `docs/TECHNICAL_REPORT.md` | **new**: paper-style report covering architecture, data/execution flow (Mermaid), models and training, evaluation, per-module reference, web app, design decisions, limitations, and a probe integration plan |
| `docs/CHANGELOG.md` | **new**: this file |
| `docs/metrics.jsonl` | **new**: 5 backfilled rows transcribed from README/METHODS/MANIFEST (marked `"source": "backfill"`) plus the first 2 logged runs |
| `docs/scripts/make_figures.py` | **new**: regenerates `docs/figures/{gate_latest, corn_history, crop_history, variance_decomposition, join_funnel}.png` |
| `docs/figures/*.png` | 5 new generated figures (the two existing PNGs are untouched) |
| `src/strata/metrics_log.py` | **new**: `append_run(kind, metrics)` writes one JSON line with timestamp, commit, and dirty flag |
| `src/strata/main.py` | `section_soil_classifier` now returns its metrics; `section_yield_model` returns attribution/MAE/no-trend numbers too; `main()` calls `append_run("main", ...)` |
| `scripts/build_models.py` | calls `append_run("build_models", ...)` with per-crop LCO R², MAE, gate decision, and shape |
| `CLAUDE.md` | **new**: project notes and the documentation policy |
| `.claude/settings.json` | **new**: registers the two hooks |
| `.claude/hooks/{_fingerprint,save_prompt,check_docs}.py` | **new**: prompt capture and code-state snapshot; Stop-time check that exits 2 if code changed without a changelog update (honors `stop_hook_active`) |
| `.gitignore` | ignores `.claude/last_prompt.txt`, `.claude/hooks/.turn_state.json`, hook `__pycache__` |

**Why.** The team needed one accurate, current writeup of the whole system before
the probe hardware lands, plus a mechanism that keeps it accurate as the code
changes. Metrics were previously quoted by hand in three docs and had already
drifted (§11.3 of the report). They are now appended automatically by the two
evaluation entry points, and the figures are generated from that log.

**Behavior change.** None to models, rules, or the app. `main.py` and
`build_models.py` each append one line to `docs/metrics.jsonl` at the end of a run.

**Verification.**
- `python -m pytest`: 95 passed.
- Hook pipe-tests in a scratch clone: no change → 0; code changed without a
  changelog → 2; `stop_hook_active` → 0; code + changelog → 0; new untracked code
  file → 2; only `*.md` changed → 0; UTF-8 prompt saved intact.
- `scripts/build_models.py` reproduced `MANIFEST.json` (2026-08-31) to 4 decimals
  for all 8 crops.

**Metrics** (run `2026-09-19T19:02:53+00:00`, `main`; and the `build_models` run
just before it). No change from the 2026-08-31 build; this is the first *logged*
baseline.

| metric | before (backfill 2026-08-31 / 07-25) | after (logged run) |
|---|--:|--:|
| corn LCO R² | 0.610 | 0.610 |
| corn forward R² | 0.46 | 0.461 |
| corn attribution LCO R² | 0.291 (07-20, pre-NDVI) | 0.525 |
| soil classifier accuracy / macro-F1 | 0.689 / 0.653 | 0.689 / 0.653 |
| crops shipped | 6 / 8 | 6 / 8 |

**Found, not fixed** (listed in the report §11.3): METHODS §10 gate table is stale
against the current build (hay/barley); a README bullet quotes old oats/alfalfa
values; `python -m strata.main` needs `PYTHONPATH=src`.
