# AGENTS.md

Solar-glare / ocular-hazard analysis pipeline for the Map Ta Phut cargo-port PV
study. It combines SGHAT-style Green/Yellow/Red thresholds with IEC 62471 (BS EN
62471:2008) risk groups and an ICNIRP 2013 cross-check.

- `Document.docx` is the requirements spec, but it is a ZIP: extract it and parse
  `word/document.xml`; do not read it as plain text.
- Detailed history/provenance/results live in **`AGENTS_APPENDIX.md`**. Read that
  only when you need the long version; this file is the working quick-start.
- The repo is git-initialised (`main`, one initial commit) but most files are
  still untracked. Do not commit unless explicitly asked.

## Environment

- Use `.venv\Scripts\python.exe` for everything. Never install into global
  Python. Deps are pinned in `requirements.txt`; Python is 3.10.11.
- Shell is Windows PowerShell 5.1. The console codepage is 874, so set
  `$env:PYTHONIOENCODING='utf-8'` before Python that prints non-ASCII (the stage
  runner already pins this for child processes).
- There is no lint/typecheck/test framework. `scripts/sanity_checks.py` is the
  fast correctness check (plain asserts).
- `opencode.json` denies edits to
  `references/_PD_IEC_TR_62471_2_2009_lock.txt`.

## Commands (run from repo root)

```powershell
# Full pipeline (4 stages, ~27 min)
.venv\Scripts\python.exe scripts\stages\run_all.py

# Individual stages
.venv\Scripts\python.exe scripts\stages\step1_analysis.py [--case LABEL] [--no-verify]  # ~11 min
.venv\Scripts\python.exe scripts\stages\step2_validate.py [--quick] [--only NAME,...] [--fail-fast] [--force]  # ~15 min
.venv\Scripts\python.exe scripts\stages\step3_timeline.py [--case LABEL] [--no-validate]  # ~20 s
.venv\Scripts\python.exe scripts\stages\step4_report.py   # ~5 s

# Direct analysis / single case
.venv\Scripts\python.exe scripts\run_analysis.py [--case LABEL] [--brief]
.venv\Scripts\python.exe scripts\sanity_checks.py
.venv\Scripts\python.exe scripts\run_contour.py [--case LABEL] [--all] [--grid-only] [--no-html] [--hour-range START END]
.venv\Scripts\python.exe scripts\build_run_timeline.py [--case LABEL] [--check]
.venv\Scripts\python.exe scripts\build_report_html.py
```

Timeout guidance: 6-case analysis ~27 min → use 1 hr; single case ~11 min → 20
min; contour full-year ~56 min → 2 hr; validator suite up to 15 min → 30 min.

### Validator details that are load-bearing

- Stage 2 order is fast-to-slow: `sanity_checks`, `validate_sghat`,
  `validate_norm_vs_code`, `validate_iec62471`, `validate_icnirp`,
  `validate_e903`, `validate_cases`, `validate_charts`, `validate_kuantan`,
  `validate_dupplin`.
- `validate_charts.py` must be run as `--full --hit-test polygon` (the combination
  its checks were last validated with).
- `validate_kuantan.py` / `validate_dupplin.py` must be run **without**
  `--hit-test`; their committed logs use the default `circular`.
- `validate_dupplin.py` is WARN-only, never FAIL. 4 passed / 3 failed is the
  documented model limitation, not a regression.
- `validate_timeline.py` belongs after stage 3 (it reads
  `output/run_timeline.html`). It reports `[SKIP]`, never a pass, when `node` is
  not on PATH.
- `validate_pv_stack.py` and `validate_arc_index.py` exist but are manual-only;
  they are not in stage 2.

### PDF work

Use `scripts\pdf_extract.py` / `normalize_pdf_text.py` under
`.opencode\skills\pdf-to-txt\`. Never use PowerShell `Get-Content`/`Set-Content`
on the UTF-8-no-BOM extracts (PowerShell 5.1 mangles them). House style is in
`references/_FORMAT_62471.txt`.

## Current config (check before assuming)

`config/site.yaml` is the committed study; `config/site_contour.yaml` is a
separate, very large ground-grid config that `run_contour.py` uses.

- PV-A only: 4,890 × 2 m, 12.5 m AGL, `groups: 19`.
- 65 receptors, all 40 m AGL: 15 `east1_route_*` + 44 `west*_route_*`
  (11 per route) + 4 terminal + 2 obs.
- `cases:` currently has **only `az278_e903_t15` active**; the other five are
  commented out. A plain run is therefore ONE case, even though all six
  `output/<label>/` dirs exist from a prior sweep.
- Environment: `time_freq: "min"` (525,600 steps/year), `hit_test_mode:
  "polygon"`, `reflectivity_mode: "solar_weighted"` with `glass_type:
  "smooth_arc"`, `atmospheric_attenuation: "beer_lambert"` (`visibility_km:
  20.0`), `iec62471_enabled: true`, `n_workers: 16`, `map_tiles: "esri_street"`.
- `config/site_contour.yaml` has its own `cases` (currently `az278_e903_t15` and
  `az278_e903_t20`), `hour_range: [6, 12]`, and a generated dense grid
  (`STEP_M` / `DROP_EAST_COLS`). Output goes to
  `output/contour/height_<AGL>/<label>/`.
- Stage 1 checks results against `scripts/stages/_stage_lib.py::COMMITTED`
  (9 case/receptor rows) plus per-case G/Y/R totals. The six-case sweep-total
  check only fires when all six labels are active.

## Architecture

Data flow: `scripts/run_analysis.py` → `solar_glare.analysis.run_analysis` →
`PVArray._split_groups` → `_analyze_single` per sub-panel (thread pool) →
SGHAT hazard + optional IEC 62471 → `summary` / `details` → CSVs and figures.

- `analysis.py` — per-minute config-driven pipeline. `groups` only changes the
  detection footprint; `_split_groups` sets every group's `hazard_area` to the
  full contiguous sheet.
- `geometry.py` — ENU vectors. Reflection is `r = 2(s·n)n − s`: a low morning
  (east) sun reflects **west**, a low afternoon (west) sun reflects **east**.
  Polygon detector is the validated hit test.
- `hazard.py` — SGHAT ocular thresholds, E903 solar-weighted reflectivity,
  Beer–Lambert extinction, retinal irradiance. `aoi_reflectivity` / `GLASS_FITS`
  are reference-only (no longer selectable pipeline modes).
- `solar.py` — pvlib NREL SPA sun position + DNI. Do not downgrade to the
  simplified ephemeris.
- `iec62471.py` — IEC 62471 risk groups from the already-computed `E_c`/`E_r`.
- `contour.py`, `viz.py`, `viz3d.py` — presentation only.
- `profiling.py` — opt-in timing; never feeds physics.
- `scripts/stages/` — thin launchers only; no physics/config duplication.

## Must-know gotchas

- **Detection vs hazard:** a sub-panel wins the hit test, but hazard `x` uses the
  full `hazard_area` (entire reflective sheet). Using sub-panel area creates the
  "one-pixel mirror" Red artifact.
- **`polygon` vs `circular`:** polygon is the committed detector. Circular
  under-reports elongated near-field panels badly (e.g. `obs_ptt` 58 vs 489 min).
- **No inverse-square law:** the reflected beam is collimated, so distance does
  not dilute `E_c`. Attenuation is Beer–Lambert only, evaluated at the winning
  sub-panel's centre.
- **E903 ρ:** solar-weighted Fresnel is the live reflectivity model; at glare-band
  AOI it is much higher than the old SGHAT Table 1 fit. Do not substitute it.
- **IEC α is effectively constant on this config** (~0.05085 rad) because the
  4,890 m row length and 2 m depth clamp to `[α_min, α_max]`. Do not expect it to
  vary by receptor.
- **Hazard is classified only on hit minutes**; all other minutes are `N`.
- **`validate_charts.py` baseline uses `site.yaml`'s array azimuth (`0.0`)**, not
  `site.cases`, so its internal baseline legitimately yields 0 glare. That is a
  plumbing check, not a result.
- **`output/report.md` is hand-maintained Thai prose.** Rebuild order is
  `run_analysis.py` → update `report.md` → `build_report_html.py`. Stage 4 builds
  the HTML but never edits `report.md`.
- **`run_analysis.py` does not clean stale per-pair CSVs** in the non-case
  `output/results/`. When `site.cases` is set, each case gets a fresh
  `output/<label>/`, so cases cannot cross-contaminate.
- **Result CSVs are deterministic; PNG metadata may differ.** Compare CSVs, not
  image hashes.

## Outputs

- Per case: `output/<label>/{figures,results}`. Key results: `summary.csv`,
  `glare_events.csv`, per-pair `glare_hours_*.csv`, `iec62471_summary.csv`,
  `run_profile.csv`, `run_spans.csv`.
- Cross-case: `output/run_profile_all_cases.csv`.
- Reports: `output/report.md` (source) and self-contained `output/report.html`.
- Interactive run timeline: `output/run_timeline.html`.
- Validation logs: `output/validation/*.txt`; stage audit:
  `output/validation/stage_runs.txt`.
- Non-pipeline artifacts to ignore: `backup_report/`, root `charts_tail.txt`,
  root `contour_grid.csv`, `output/ref_figs/`, `output/validation/lost_city_*`,
  `output/_probe_*.npy`, `output/contour.7z`.

## Where to find details

- Exact current numbers and the six-case write-up: `output/report.md`,
  `output/<label>/results/summary.csv`.
- Validator/stage contracts: docstrings in `scripts/stages/*.py` and the
  validator scripts themselves.
- Physics/reference provenance and long historical notes:
  **`AGENTS_APPENDIX.md`**, module docstrings, and `references/`.
