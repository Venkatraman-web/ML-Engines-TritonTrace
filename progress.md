# Forward Propagation — Progress Tracker

## Overall status

```text
Status            : NOT STARTED (master 548-particle / 20:35 UTC spec)
Current task      : F00
Completed         : 0 / 9
Blocked           : 0
Failed            : 0
```

## ow-0009 pilot — overall status (2026-09-25)

```text
Status      : COMPLETE (F00-F07, all PASS)
Case        : ow-0009 (371 parcels, 2019-04-28 15:38:41 UTC -> 2019-05-01 15:38:41 UTC)
Outputs     : output/ow-0009/  (trajectory, hourly diagnostics, hotspot alerts,
              MP4 animation + 4 representative frames, OpenOil reference +
              comparison, run_summary.json)
Note        : this pilot is intentionally separate from, and does not complete, the
              master 548-particle/20:35 UTC spec tracked immediately above and in the
              "Task status" F00-F09 section further below, which remains NOT STARTED.
```

## Pilot case audit — F00-ow0009 (2026-09-25)

A separate, narrower request ("F00 — Audit the First Forward-Propagation Case")
asked for forward propagation to be built and validated **only** for a single
real observed case, `ow-0009`, as a pilot before touching the master 548-particle
spec below. This section records that audit. No simulation code was written;
this is inspection + documentation only, per the request.

**This case does not match the master lock** (548 particles @ 2019-04-28 20:35
UTC). It is a deliberately narrower, real, independently-loadable case used to
pilot the pipeline. The master spec in this file (below) and in `README.md`/
`tasks.md` is unchanged and still governs the full F00-F09 run.

### Locked ow-0009 case (from actual data, not invented)

```text
Source            : data/database_payload.json -> spills[] where
                     image_filename == "ow-0009.jpg"
Observation time  : 2019-04-28 15:38:41 UTC
Parcels           : 371 (ids 0..370, all unique, matches total_parcels)
Parcel fields     : id, lat, lon, mass_kg, confidence
Parcel lon range  : 34.818565 - 34.898924
Parcel lat range  : 34.561746 - 34.633203  (inside AOI)
mass_kg range     : 111.86 - 342.23
confidence range  : 0.2507 - 0.4045
image_filename    : ow-0009.jpg   (metadata string only, see blocker below)
mask_filename     : mask_ow-0009.png (metadata string only, see blocker below)
area_pixels       : 30970
bounding_box      : upper_left (34.653566, 34.970993) / bottom_right (34.558784, 34.807723)
```

Sanity check: nearest-grid-cell CMEMS lookup shows 0/371 parcels sit on a
NaN (land) cell at t0 — the observed slick starts entirely over water, as
expected.

### Input inventory

| Input | Status | Notes |
|---|---|---|
| `data/28th_april/ocean.nc` (CMEMS `uo`,`vo`) | OK | time 2019-04-28 00:30 -> 2019-05-01 23:30 (96 steps), lat 149 (ascending, 30.19-36.35), lon 216 (ascending, 27.12-36.08). Matches README-stated grid/coverage. NaN fraction = 27.5% of cells, **identical at every one of the 96 timesteps** -> the NaN mask is a genuine static land/sea mask, not per-timestep noise. |
| `data/28th_april/wind.nc` (ERA5 `u10`,`v10`) | OK | time 2019-04-28 00:00 -> 2019-05-01 23:00 (96 steps), latitude 25 (**descending**, 36.25-30.25), longitude 36 (ascending, 27.25-36.0). Zero NaNs. Single `expver='0001'` value (no ERA5 mixed-expver ambiguity to resolve). |
| `regional_alert_geofences_wide.geojson` | OK | 22 features confirmed, 11 unique hotspot names confirmed, each with a "Watch Zone" (100 km) and "Critical Strike Zone" (40 km) polygon, 33-vertex rings. Matches spec exactly. |
| `data/database_payload.json` | OK | 5 spills total (`ow-0001`, `ow-0005`, `ow-0008`, `ow-0009`, `ow-0010`); `ow-0009` entry located and used as above. |
| `ow-0009.jpg` / `mask_ow-0009.png` (actual image files) | **MISSING** | Searched the full `SIH-2026` tree (excluding venvs) for these exact filenames and any `*0009*.jpg/.png` — not present anywhere. Only referenced as metadata strings inside `database_payload.json`. Not a blocker for physics (parcel lon/lat/mass/confidence already extracted numerically), but blocks anything that wants the source image/mask for overlay or provenance display. |

### Path mismatch vs. CLAUDE.md / README (blocker to flag, not fixed here)

`CLAUDE.md` and `README.md` specify forcing paths `data/ocean.nc` and
`files/28th_june/wind.nc`. The actual files on disk are
`data/28th_april/ocean.nc` and `data/28th_april/wind.nc` (no `files/`
directory exists in this repo at all). Contents match the documented
variables/coverage, so this is a path/directory-naming drift, not a data
problem. Needs a decision before F01 writes a forcing adapter with a hard-coded
path: either update the spec docs to the real paths, or add a symlink/copy at
the documented paths. Not changed here since this task is audit-only.

### Reusable Stage III (`iste/`) modules — audited in `C:\Users\venka\OneDrive\Desktop\SIH-2026\project-v5\iste\`

Stage III is a **backward-trajectory / source-attribution** pipeline (traces an
observed slick backward to candidate spill origins), not a forward-propagation
pipeline, so reuse is at the primitive level, not the orchestration level.

Directly reusable as-is:
- `iste/interpolation.py`: `_bracket()` (handles ascending *and* descending
  monotonic coordinate arrays generically — this is the ERA5 descending-latitude
  case), `bilinear_spatial_interpolate()` (true continuous bilinear, not
  nearest-neighbor; already returns NaN + `valid=False` rather than silently
  guessing), `linear_temporal_interpolate()` / `time_bracket()` /
  `interpolate_field()` (exact-time interpolation, not snapping — this is
  exactly F01's requirement).
- `iste/environment.py`: `load_field_dataset()` for generic NetCDF loading that
  preserves NaNs (never fills with 0).
- `iste/geometry.py`: `MEAN_EARTH_RADIUS_M`, `metres_per_degree_latitude`,
  `metres_per_degree_longitude`, `metres_to_degrees` for displacement
  integration (u,v * dt -> dlon,dlat).
- `iste/ais_matching.py::haversine_distance_km`: great-circle distance —
  directly usable for the hotspot `minimum_particle_distance_km` metric (no
  distance function exists in `geometry.py` itself).

Adaptable (reusable pattern, not reusable as-is):
- `iste/boundary.py::ParticleState` / `apply_boundary_policy()`: the
  permanent-freeze-on-invalid mechanics match the required "strand and keep
  last position" semantics, but its validity source
  (`netcdf_validity_check()`) currently equates "land" with "any NaN in the
  4-corner bilinear stencil" — which is exactly the failure mode CLAUDE.md
  Section 6 prohibits (a coastal water particle must not be auto-stranded just
  because its interpolation stencil touches a masked cell). Must be replaced
  with land classification based on the particle's own nearest-cell status in
  the **static** CMEMS mask (see below), with the 3x3->5x5 fallback sitting in
  front of it for genuine water points.
- `iste/trajectory.py`: the "fixed checkpoint schedule + freeze-on-invalid +
  record every step regardless" storage pattern is reusable; the backward
  stepping and per-particle log-weight machinery is not needed.

Not applicable (wrong physics/purpose, do not reuse the logic, only note them):
- `iste/step.py::backward_euler_maruyama_step` — backward-time (`-U`), includes
  a stochastic diffusion term forbidden by CLAUDE.md.
- `iste/forward_propagation.py`, `iste/forward_transport.py` — despite the
  name, these forward-propagate backward-traced *candidate source points* to
  check them against the observed slick (source-attribution QA), not the
  observed slick forward into the future. Their equation also carries a
  stochastic diffusion term and reads ocean forcing only (no ERA5/wind term at
  all). Only the orchestration idioms (vectorized multi-particle stepping,
  checkpoint-schedule construction landing exactly on the end time) are worth
  looking at as a pattern reference.
- `iste/observation.py` — a backward-pipeline input adapter expecting
  `particle_id, latitude, longitude, observation_pixel_id, timestamp` plus a
  `log_weight`. Does not load `database_payload.json` and has no ow-0009
  logic; a new small loader is needed for `spills[].parcels[]`
  (`id, lat, lon, mass_kg, confidence`).

Confirmed gaps (must be built new, nothing in `iste/` covers these):
- The CLAUDE.md Section 6 NaN-fallback decision tree (`DIRECT` /
  `NEIGHBOR_FALLBACK` / `INVALID_ENVIRONMENT` / `LAND_STRANDED`,
  distance-weighted 3x3->5x5 neighbor search) does not exist anywhere in
  `iste/` — grepped for "fallback"/"neighbor"/"NaN"/"stranded"/"land"; the
  only fallback hits are unrelated statistical fallbacks in the phase3 modules.
- No `u_oil = uo + 0.03*u10` (or any current+wind combination) logic exists
  anywhere in `iste/`. `NetCDFEnvironmentalFieldProvider.U()` in
  `environment.py` explicitly hard-locks `U=(uo,vo)` and documents "windage is
  excluded from U in v1" — it never even reads the wind dataset. This must be
  written new.
- No external coastline/shapefile source exists in the repo (no
  shapely/geopandas usage, no Natural Earth or coastline file found). This
  turns out not to be a blocker: the CMEMS `uo`/`vo` NaN mask was verified to
  be **exactly time-invariant across all 96 timesteps** (0 cells change status),
  i.e. it is already a genuine static land/sea mask at CMEMS grid resolution
  (~0.042° ≈ 4.6 km). Recommend using "nearest CMEMS cell permanently NaN" as
  the land classification source (decoupled from the interpolation-stencil NaN
  check used for the fallback decision), rather than introducing a new
  coastline dependency. Flagging this as a design decision for F02, not
  changing anything yet.

### Blockers / mismatches to resolve before F01 starts

1. Forcing file paths in `CLAUDE.md`/`README.md` (`data/ocean.nc`,
   `files/28th_june/wind.nc`) do not match the files on disk
   (`data/28th_april/ocean.nc`, `data/28th_april/wind.nc`). Needs a decision:
   update docs vs. add paths matching the spec.
2. `ow-0009.jpg` / `mask_ow-0009.png` do not exist as files anywhere in the
   `SIH-2026` tree; only referenced as metadata strings. Not a physics
   blocker (parcel coordinates already extracted), but flagged in case the
   application layer expects to display the source image.
3. Land classification design decision needed for F02: use the static CMEMS
   NaN mask (verified time-invariant) as ground truth for "land", rather than
   equating any NaN in a bilinear stencil with land (that is Stage III's
   current, CLAUDE.md-noncompliant, behavior in `boundary.py`).
4. This ow-0009 case (371 parcels @ 15:38:41 UTC) is intentionally different
   from the master lock (548 particles @ 20:35 UTC). Per the originating
   request this is expected and this section documents it; the master F00-F09
   lock elsewhere in this file is unchanged.

### Status

```text
F00-ow0009 audit: COMPLETE
Code written     : none (audit/inspection only, as instructed)
Next step        : F01-ow0009 (below)
```

---

## Pilot case — F01-ow0009 (2026-09-25): forcing adapter

### Objective

Build the environmental forcing layer (`(lon, lat, time) -> uo, vo, u10, v10`) for the
ow-0009 pilot case only, reusing Stage III's `iste.interpolation`/`iste.environment`
machinery. Particle propagation is explicitly out of scope for this task.

### Files changed

```text
fwd/__init__.py                    new — package marker
fwd/stage3_bridge.py               new — sys.path bridge to import iste from
                                          ../project-v5 without copying/modifying it
fwd/forcing.py                     new — ForcingAdapter, ForcingResult, FallbackConfig
scripts/f01_validate_forcing.py    new — manual validation script (not a test file)
```

Per the request, actual on-disk paths were used directly:
`data/28th_april/ocean.nc`, `data/28th_april/wind.nc` (these are `ForcingAdapter`'s
defaults). This resolves blocker 1 from the F00-ow0009 audit *for this module*; the
stale paths in `CLAUDE.md`/`README.md` (`data/ocean.nc`, `files/28th_june/wind.nc`)
are still unchanged in the docs themselves and should be corrected before/along with
F09's final documentation pass.

### Implementation summary

- `ForcingAdapter.__init__` loads both NetCDFs once via `iste.environment.load_field_dataset`
  (unmodified Stage III code — reused, not copied), then derives the static
  ocean-valid mask (`ocean_valid_mask = finite(uo) & finite(vo)` at every one of the 96
  timesteps) and re-verifies at load time that the mask is exactly time-invariant
  (`mask_is_time_invariant`), rather than silently trusting the F00 audit's finding.
- `ForcingAdapter.query(lon, lat, time)` calls `iste.interpolation.interpolate_field`
  (unmodified) for `uo`, `vo`, `u10`, `v10` — this already gives continuous bilinear
  spatial interpolation + exact-time linear temporal interpolation, and already handles
  ERA5's descending latitude generically (via `iste.interpolation._bracket`, which
  works on ascending or descending monotonic coordinate arrays).
- If the direct ocean lookup is invalid (a NaN in the 4-corner bilinear stencil at
  either bracketing timestamp, or point out of grid/time range), `_neighbor_fallback`
  implements CLAUDE.md section 6 exactly: try a 3x3 window of `ocean_valid_mask`-true
  cells first, then 5x5; require >= 4 valid neighbors within `max_radius_m`
  (configurable, default 20 km); weight `1/(d^2+epsilon)` with distances computed in
  metres via `iste.geometry.metres_per_degree_latitude/longitude` (reused, not
  reimplemented); blend the two bracketing timestamps by the same time fraction as the
  direct path. Only cells from the static land mask may contribute — land cells are
  never used as fallback donors.
- If no window meets the minimum-neighbor/radius requirement, or the query point/time
  is out of range, the result is `INVALID_ENVIRONMENT` with `uo`/`vo` (or `u10`/`v10`)
  reported as `NaN` — never silently substituted with 0.
- Per requirement 7, this module makes no stranding/land-classification *decision*; it
  only exposes the static mask for F02 to reuse, and documents that boundary in
  `fwd/forcing.py`'s module docstring.

### Validation

Command: `python scripts/f01_validate_forcing.py`

Evidence (abridged; full output inspected manually):

```text
ocean grid shape (lat, lon): (149, 216)
wind lat[0]=36.25 > lat[-1]=30.25  (descending, confirmed)
static ocean-valid mask, fraction water: 0.7249  (matches F00's 27.5% NaN finding)
mask time-invariant (re-verified here): True

1) all 371 ow-0009 parcels @ 2019-04-28 15:38:41 UTC:
   DIRECT=371  NEIGHBOR_FALLBACK=0  INVALID_ENVIRONMENT=0
   uo/vo speed range: [0.1132, 0.2729] m/s  (physically reasonable Levantine surface current)
   all 371 parcels' u10/v10 finite, wind_mode=DIRECT for the inspected sample

2) independent exact-time interpolation cross-check (hand-computed bilinear+linear
   blend from a fresh xarray load, not reusing iste's own arrays):
   CMEMS bracket for 15:38:41 -> 15:30:00 / 16:30:00 (confirmed, not snapped)
   ERA5  bracket for 15:38:41 -> 15:00:00 / 16:00:00 (confirmed, not snapped)
   adapter uo=-0.116669  vs independent reference uo=-0.116668  (match)
   adapter u10=0.718500  vs independent reference u10=0.718500  (exact match)

3) CMEMS NaN / land-boundary points (25 coastal water/land boundary midpoints found
   programmatically from the mask, not hand-picked):
   all 25 -> NEIGHBOR_FALLBACK (example: lon=29.0000 lat=30.8333, window=5,
   neighbors=8, uo=+0.05647 vo=+0.00490 -- finite, physically reasonable)
   deep-land point with zero valid neighbors in a 5x5 window (found programmatically):
   lon=27.2083 lat=30.2708 -> INVALID_ENVIRONMENT, uo=nan vo=nan (never 0)

4) out-of-domain point (0,0): ocean_mode=wind_mode=INVALID_ENVIRONMENT, all NaN
   (confirms no extrapolation outside the grid)

ALL CHECKS PASSED
```

### Issues discovered

- None blocking. One design note: `ForcingResult` reports a single `ocean_mode` for
  the `uo`/`vo` pair jointly (not one mode per variable) because, in this dataset, `uo`
  and `vo` share exactly the same NaN pattern (both derive from the same CMEMS
  ocean/land mask) — verified as a side effect of the boundary-point sweep in check 3,
  where every fallback and every direct lookup treated both variables identically.
  Documented here rather than silently assumed.
- `max_radius_m` default (20 km) is a placeholder "configurable" value, not a value
  taken from any spec document (none is given) — flagging for review in case the
  application wants a different default; it did not bind in any of the validated
  points above (the widest window used was 5x5 ≈ within ~10 km of the query point).

### Status

```text
F01-ow0009: COMPLETE
Files changed : fwd/__init__.py, fwd/stage3_bridge.py, fwd/forcing.py,
                scripts/f01_validate_forcing.py
Validation    : python scripts/f01_validate_forcing.py -> ALL CHECKS PASSED
Next step     : F02-ow0009 (below)
```

---

## Pilot case — F02-ow0009 (2026-09-25): one-hour forward particle step

### Objective

Implement one deterministic one-hour forward motion step for a single particle
(`u_oil = uo + 0.03*u10`, `v_oil = vo + 0.03*v10`, current_drift_factor = 1.0),
reusing F01's `ForcingAdapter` and Stage III's displacement conversion, with permanent
coastline stranding. Multi-hour propagation over all 371 parcels is explicitly out of
scope for this task.

### Files changed

```text
fwd/forcing.py                  modified — `_nearest_cell`/`_in_spatial_range` made
                                 public (`nearest_cell`/`in_spatial_range`) so
                                 fwd.boundary can reuse the same grid lookup instead
                                 of duplicating it; no other change, no behavior change
fwd/boundary.py                 new — is_land(adapter, lon, lat): land/coast
                                 classification from F01's static ocean_valid_mask
fwd/step.py                     new — ParticleState, StepRecord, step_particle,
                                 step_all
scripts/f02_validate_step.py    new — manual validation script (not a test file)
```

### Implementation summary

- `fwd/boundary.py::is_land` classifies a point as land if its nearest CMEMS grid cell
  is permanently masked (`ForcingAdapter.ocean_valid_mask`, from F01) or the point is
  outside the CMEMS grid entirely. This is the "proper land/coast classification"
  CLAUDE.md section 7 asks for, kept deliberately separate from `fwd.forcing`'s NaN
  interpolation-stencil check (a coastal water particle whose bilinear stencil touches
  a masked cell gets `NEIGHBOR_FALLBACK` from F01, not an incorrect land classification
  here) — resolving the exact failure mode the F00 audit flagged in Stage III's own
  `iste/boundary.py` (which conflates "any NaN in the interpolation stencil" with
  "land").
- `fwd/step.py::step_particle` implements one Euler step:
  1. no-op if already `stranded`/`invalid` (`ALREADY_STRANDED`, position unchanged) --
     requirement 7;
  2. defensive land check at the *current* position (`LAND_STRANDED` if it somehow
     already is land);
  3. `adapter.query(lon, lat, t)` (F01) for `uo, vo, u10, v10`; if either the ocean or
     wind lookup is `INVALID_ENVIRONMENT`, strand at the current position (no velocity
     can be computed) and record `forcing_mode=INVALID_ENVIRONMENT`;
  4. `NEIGHBOR_FALLBACK` is explicitly allowed to drive propagation (requirement 5) --
     only `INVALID_ENVIRONMENT` and land trigger stranding;
  5. `oil_u = 1.0*uo + 0.03*u10`, `oil_v = 1.0*vo + 0.03*v10` (current_drift_factor and
     wind_drift_factor both explicit parameters, not hidden constants);
  6. displacement via `iste.geometry.metres_to_degrees(oil_u*dt, oil_v*dt, old_lat)`
     (unmodified Stage III code, reused per requirement 2) -- no diffusion, no
     randomness, no Stokes term, no vertical motion added anywhere;
  7. land check at the *candidate next* position -- the actual coastline-crossing
     check. If it is land, the particle strands **at its last valid position** (never
     moved onto the land cell), with the attempted position kept in the record
     (`attempted_lon`/`attempted_lat`) purely for audit purposes.
- Every call returns a `StepRecord` with `old_lon/old_lat, new_lon/new_lat, uo, vo,
  u10, v10, oil_u, oil_v, valid, stranded, forcing_mode` (requirement 9), plus the
  optional attempted-position fields when a step is rejected for landing on land.

### Validation

Command: `python scripts/f02_validate_step.py`

Evidence (abridged; full output inspected manually):

```text
1) normal open-water parcel (id 0), real 1h step from 2019-04-28 15:38:41 UTC:
   old=(34.878516,34.633203) -> new=(34.874773,34.635703)
   uo=-0.11667 vo=+0.09917 u10=+0.71850 v10=-0.73174
   oil_u=-0.095114 oil_v=+0.077214  forcing_mode=DIRECT  valid=True stranded=False
   displacement magnitude over 1h: 441.04 m (physically reasonable for ~0.12 m/s |oil velocity|)
   direction check: sign(new_lon-old_lon)==sign(oil_u), sign(new_lat-old_lat)==sign(oil_v) -- confirmed

2) parcel closest to the coast of the 371 (id 346): 76.8 km from the nearest CMEMS
   land cell -- see "Issues discovered" below. Real 1h step: forcing_mode=DIRECT,
   finite result, no stranding (as expected at 76.8 km with a ~0.1 m/s current).

3) independent displacement-formula/units cross-check (hand-derived
   dx_m/(R*cos(lat)*pi/180) formula, not reusing iste.geometry internally): matches
   iste.geometry.metres_to_degrees to 1e-12 deg for three test vectors, including the
   parcel-0 real oil_u/oil_v*dt vector. 1000 m northward at lat 34.6 -> 0.00899 deg,
   matching 1000/metres_per_degree_latitude() exactly.

4) land-crossing/stranding path, probed with a deliberately large synthetic dt (40h,
   clearly not the real 1h step -- see below) at a real coastal boundary cell
   (lon=29.0000, lat=30.8542, Nile Delta area): attempted position (29.353648,
   30.761357) independently confirmed is_land()=True; particle kept at its original
   position (29.000000,30.854166), not moved onto land -- confirmed exactly equal.

5) re-stepping the now-stranded particle with a normal step: forcing_mode=
   ALREADY_STRANDED, position exactly unchanged -- confirmed (requirement 7).

ALL CHECKS PASSED
```

### Issues discovered

- All 371 ow-0009 parcels are 76.8-80.3 km from the nearest CMEMS-resolution land
  cell (computed via great-circle distance to every masked grid cell) -- this slick
  starts well offshore, not near any coast. This means the real one-hour step for
  every real ow-0009 parcel is expected to be `DIRECT` (confirmed: sections 1-2 above),
  and the actual land-crossing/stranding logic could not be honestly demonstrated with
  a real parcel and the real 1-hour step in this task's scope. It was instead validated
  with (a) a real coastal boundary cell elsewhere in the AOI and (b) a deliberately
  large, clearly-labelled synthetic `dt` (40h) passed through the same, unmodified
  `step_particle` code path -- not a separate/parallel implementation. Flagging this
  for later tasks: a full 72-hour run of ow-0009 may or may not reach any land within
  the AOI at all, since the initial slick centroid is ~77 km offshore and observed
  current speeds are only ~0.1-0.3 m/s (max plausible unforced drift over 72 h is a few
  tens of km); this is a legitimate scientific outcome to check for, not a validation
  gap, when F04-equivalent full propagation is implemented.
- No other blockers.

### Status

```text
F02-ow0009: COMPLETE
Files changed : fwd/forcing.py (2 methods made public, no behavior change),
                fwd/boundary.py, fwd/step.py, scripts/f02_validate_step.py
Validation    : python scripts/f02_validate_step.py -> ALL CHECKS PASSED
Next step     : F03-ow0009 (below)
```

---

## Pilot case — F03-ow0009 (2026-09-25): 72-hour forward simulation

### Objective

Run the full 72-hour, 1-hour-cadence forward simulation for all 371 real ow-0009
parcels by looping F02's `step_particle` hourly, and persist the result in a clean
machine-readable format for the later hotspot engine and animation. Hotspot alerts and
animation are explicitly out of scope for this task.

### Files changed

```text
fwd/simulate.py                new — load_ow0009_parcels, run_simulation
                                (pure function: no hard-coded I/O paths)
scripts/f03_run_simulation.py  new — runs the simulation, writes output, validates
                                all required invariants (not a test file)
output/ow-0009/trajectory.parquet          new — 27,083 rows (371 particles x 73 states)
output/ow-0009/hourly_diagnostics.parquet  new — 72 rows (one per propagation step)
```

### Implementation summary

- `run_simulation` builds one `ParticleState` per real parcel (id/lon/lat from
  `load_ow0009_parcels`), records the unstepped `t0` row (`forcing_mode="INITIAL"`,
  `uo/vo/u10/v10/oil_u/oil_v` = NaN since no step has run yet -- this is the only
  `forcing_mode` value invented outside F01/F02's DIRECT/NEIGHBOR_FALLBACK/
  INVALID_ENVIRONMENT/LAND_STRANDED/ALREADY_STRANDED vocabulary, used purely to mark
  "no step taken" bookkeeping, not a new physics/forcing concept), then calls F02's
  `step_particle` once per particle per hour for 72 hours, appending one trajectory
  row and rolling up one diagnostics row per step. No new stranding, NaN-fallback, or
  velocity logic was written -- this module is orchestration only, over F01/F02
  unchanged.
- `mass_kg`/`confidence` (requirement 4) are carried on every row by joining from the
  original parcel dict at construction time (verified unchanged per particle across
  all 73 rows, see validation below).
- Trajectory columns match README section 12's recommended fields exactly:
  `particle_id, time, lon, lat, valid, stranded, forcing_mode, uo, vo, u10, v10,
  oil_u, oil_v` plus `mass_kg, confidence`.
- Output: a single `output/ow-0009/trajectory.parquet` (long format, one row per
  particle per hour, as requested) plus one small `hourly_diagnostics.parquet`
  (72 rows: `active_count, stranded_count, fallback_count, invalid_forcing_count,
  newly_stranded_count`) for requirement 8 -- kept as a second file only because it is
  a ~72-row summary table, not a duplicate of the trajectory data.

### Validation

Command: `python scripts/f03_run_simulation.py`

Evidence (abridged; full output inspected manually):

```text
simulation complete in 34.2s wall time (371 parcels x 72 steps = 26,712 step_particle calls)
wrote output/ow-0009/trajectory.parquet  (27083 rows)
wrote output/ow-0009/hourly_diagnostics.parquet  (72 rows)

1) distinct timestamps: 73
   first: 2019-04-28 15:38:41   final: 2019-05-01 15:38:41  (exact match to spec)
   all 371 parcel IDs present, unchanged, at all 73 timestamps; no duplicate
   particle_id/time rows; mass_kg/confidence unchanged per particle over the run

2) active-particle oil speed range: [0.0573, 0.5714] m/s -- no non-finite lon/lat/
   oil_u/oil_v found on any active particle at any timestamp

3) particles that stranded at some point during the run: 0
   (no violations checked because none stranded -- see "Issues discovered" below)

4) final (2019-05-01 15:38:41): active=371 stranded=0 (total=371)
   forcing_mode counts across all 72 x 371 = 26,712 steps: {'DIRECT': 26712}
   (NEIGHBOR_FALLBACK=0, LAND_STRANDED=0, INVALID_ENVIRONMENT=0, ALREADY_STRANDED=0)
   hourly diagnostics: active_count=371 / stranded_count=0 at every one of the 72 rows

ALL CHECKS PASSED
```

Additional manual inspection (net displacement, not asserted in the script but
reviewed): the slick's centroid moved from `(34.8848, 34.6119)` at t0 to
`(34.4203, 35.0704)` at t72 -- net great-circle displacement per particle ranges
64.1-72.5 km (mean 66.3 km) over 72 hours, i.e. drift to the north-northwest at an
average ~0.9 km/h, consistent with the [0.06, 0.57] m/s instantaneous oil-speed range
above. All final positions remain well inside the CMEMS grid's spatial coverage
(lon 27.12-36.08, lat 30.19-36.35), consistent with zero `INVALID_ENVIRONMENT` events.

### Issues discovered

- **No particle stranded and no NEIGHBOR_FALLBACK was ever used across the entire
  72-hour run** -- every one of the 26,712 steps was `DIRECT`. This is the expected
  continuation of the F02 finding (all 371 parcels start 76.8-80.3 km from the nearest
  CMEMS land cell) combined with the observed ~66 km mean net drift: the slick moves a
  comparable distance to its initial offshore clearance but stays within the AOI and
  never gets within a few CMEMS grid cells (~4.6 km) of any masked/land cell. This is a
  legitimate scientific outcome, not a validation gap -- but it also means **this run
  does not exercise the NEIGHBOR_FALLBACK or LAND_STRANDED code paths end-to-end**
  (those were exercised individually in F01/F02's own validation scripts against
  synthetic/boundary points, not via this particular case's real trajectory). Flagging
  this for whoever runs the eventual OpenOil reference comparison or hotspot-alert
  validation: ow-0009 alone will not demonstrate coastline stranding in the final
  deliverable; a case whose parcels start closer to shore would be needed for that.
- No other blockers.

### Status

```text
F03-ow0009: COMPLETE
Files changed : fwd/simulate.py, scripts/f03_run_simulation.py,
                output/ow-0009/trajectory.parquet, output/ow-0009/hourly_diagnostics.parquet
Validation    : python scripts/f03_run_simulation.py -> ALL CHECKS PASSED
Next step     : F04-ow0009 (below)
```

---

## Pilot case — F04-ow0009 (2026-09-25): hotspot alert engine

### Objective

Evaluate the F03 trajectory against all 11 hotspot geofences (22 polygons) from
`regional_alert_geofences_wide.geojson`, per hourly recorded state, using the actual
polygon geometries and every active particle (never the slick centroid). Animation is
explicitly out of scope for this task.

### Files changed

```text
fwd/hotspots.py                    new — HotspotThresholds, Hotspot, load_hotspots,
                                    evaluate_hotspot_alerts
scripts/f04_run_hotspot_alerts.py  new — runs the engine, writes output, validates,
                                    reports (not a test file)
output/ow-0009/hotspot_alerts.parquet  new — 803 rows (11 hotspots x 73 states)
```

### Implementation summary

- `load_hotspots` groups the 22 GeoJSON features by `name`, keeping the two zone
  rings' actual vertex coordinates (`critical_ring`/`watch_ring`) and building a
  `matplotlib.path.Path` per ring for true point-in-polygon containment
  (`matplotlib` was already a project dependency; no new library was added, and no
  circle is reconstructed from `radius_km` anywhere in the containment test itself --
  `radius_km` is kept only as metadata, per requirement 2/6).
- Zone selection is by `HotspotThresholds.critical_zone_level`/`watch_zone_level`
  (default `"Critical Strike Zone"`/`"Watch Zone"`, the actual GeoJSON `zone_level`
  strings) rather than a hard-coded "40"/"100" check anywhere in the alert logic --
  satisfying requirement 10's "configurable thresholds" without contradicting
  requirement 2's "use actual polygons, don't recreate circles from radius."
- Audit finding used as a design justification (verified numerically before writing
  the module): for every one of the 11 hotspots, the watch and critical rings are
  exactly concentric (vertex-mean center offset 0.000000 deg) with ring radii matching
  their `radius_km` label to within the expected geodesic-circle-approximation
  tolerance (99.66-100.11 km / 39.92-39.99 km). This justifies deriving one "hotspot
  location" (`_ring_center`) from the polygon itself for the minimum-distance metric,
  rather than inventing a center.
- `evaluate_hotspot_alerts` filters to active particles (`valid & ~stranded`,
  requirement 8) per hourly group, tests polygon containment for both rings, and
  applies `CRITICAL > WATCH > CLEAR` per requirement 4. Distances use
  `iste.ais_matching.haversine_distance_km` (reused, not reimplemented). Multiple
  hotspots are evaluated fully independently per hour (requirement 5) --
  `first_watch_time`/`first_critical_time` are running per-hotspot state, set once and
  carried forward on every later row for that hotspot, never reset.

### Validation

Command: `python scripts/f04_run_hotspot_alerts.py`

Evidence (abridged; full output inspected manually):

```text
803 rows = 11 hotspots x 73 timestamps, no duplicate hotspot_name/time rows,
status values confined to {CLEAR, WATCH, CRITICAL}

confirmed: particles_inside_40km <= particles_inside_100km at every row (the critical
polygon is geometrically nested inside the watch polygon for every hotspot, so a
particle counted in the critical zone must also be counted in the watch zone -- an
integrity check beyond what was strictly required, added because the concentric-ring
finding above made it cheap and meaningful)

confirmed: first_watch_time/first_critical_time are stable once set, and
first_watch_time <= first_critical_time wherever both are set

Per-hotspot summary (min_distance_km = minimum over the whole 72h run):
   Port Said (Suez Canal)              CLEAR only                 min_dist=432.23 km
   Alexandria Port                     CLEAR only                 min_dist=582.09 km
** Vasiliko Oil Terminal               CLEAR, WATCH                min_dist= 97.83 km  first_watch=2019-04-30 10:38:41
** Beirut Port                         CLEAR, WATCH                min_dist= 95.42 km  first_watch=2019-04-28 15:38:41 (t0)
   Ceyhan Oil Terminal                 CLEAR only                 min_dist=239.80 km
   Haifa Port                          CLEAR only                 min_dist=192.93 km
   Akamas Peninsula (Turtle Nesting)   CLEAR only                 min_dist=182.00 km
** Palm Islands Nature Reserve         CLEAR, WATCH                min_dist= 80.58 km  first_watch=2019-04-28 15:38:41 (t0)
   Rosh HaNikra Marine Reserve         CLEAR only                 min_dist=164.18 km
   Nile Delta Coastal Lagoons          CLEAR only                 min_dist=436.06 km
** Cape Greco MPA                      WATCH, CRITICAL             min_dist= 23.88 km  first_watch=2019-04-28 15:38:41 (t0)
                                                                                        first_critical=2019-04-30 06:38:41

Simultaneous alerts: 50 of the 73 hours have more than one hotspot in WATCH/CRITICAL
at once (e.g. Beirut Port + Cape Greco MPA + Palm Islands Nature Reserve
simultaneously WATCH from t0 through 2019-04-29 11:38:41).

ALL CHECKS PASSED
```

### Manual inspection report (as requested)

- **WATCH/CRITICAL/CLEAR per hotspot**: 4 of 11 hotspots were ever WATCH or CRITICAL
  during the 72-hour run -- Vasiliko Oil Terminal (WATCH only), Beirut Port (WATCH
  only), Palm Islands Nature Reserve (WATCH only), and Cape Greco MPA (reached
  CRITICAL). The remaining 7 (Port Said, Alexandria Port, Ceyhan Oil Terminal, Haifa
  Port, Akamas Peninsula, Rosh HaNikra Marine Reserve, Nile Delta Coastal Lagoons)
  stayed CLEAR for the entire run.
- **First alert times**: Beirut Port and Palm Islands Nature Reserve were already in
  WATCH at `t0` (2019-04-28 15:38:41 UTC) -- the observed slick starts within 100 km of
  both. Cape Greco MPA was also WATCH at `t0` and reached CRITICAL at
  **2019-04-30 06:38:41 UTC** (~39 hours into the run). Vasiliko Oil Terminal entered
  WATCH later, at 2019-04-30 10:38:41 UTC.
- **Simultaneous alerts**: yes, extensively -- 50 of the 73 recorded hours have 2 or 3
  hotspots simultaneously in WATCH/CRITICAL (see the timestamp list logged by the
  script; e.g. Beirut Port + Cape Greco MPA + Palm Islands Nature Reserve together for
  the first ~20 hours).
- **Minimum distance per hotspot** (km, over the full run): Port Said 432.2, Alexandria
  582.1, Vasiliko 97.8, Beirut 95.4, Ceyhan 239.8, Haifa 192.9, Akamas 182.0, Palm
  Islands 80.6, Rosh HaNikra 164.2, Nile Delta 436.1, **Cape Greco 23.9 (inside the
  40 km critical zone)**.

### Issues discovered

- No blockers. One cross-check: these results are consistent with the rough
  polygon-vertex-mean distance estimates computed independently during the F00 audit
  (Cape Greco ~81 km, Palm Islands ~85 km, Beirut ~100 km from the slick centroid at
  t0) -- the per-particle minimum-distance numbers here (23.9 km at closest approach,
  80.6 km and 95.4 km at t0 for Palm Islands/Beirut) are of the same order and
  direction, giving independent corroboration that the geometry pipeline (F00's rough
  centroid estimate vs. F04's exact per-particle polygon containment) is self-consistent.
- `particles_inside_40km`/`particles_inside_100km` are literal containment counts for
  each ring independently (i.e. `particles_inside_100km` is not adjusted to exclude
  particles already counted in the critical zone) -- documented here since the field
  names could be read either way; the "nested" interpretation matches the polygons'
  actual (verified concentric) geometry and was the simpler, more literal reading of
  requirement 7's field names.

### Status

```text
F04-ow0009: COMPLETE
Files changed : fwd/hotspots.py, scripts/f04_run_hotspot_alerts.py,
                output/ow-0009/hotspot_alerts.parquet
Validation    : python scripts/f04_run_hotspot_alerts.py -> ALL CHECKS PASSED
Next step     : F05-ow0009 (below)
```

---

## Pilot case — F05-ow0009 (2026-09-25): 3-day forward-propagation animation

### Objective

Render the F03 trajectory + F04 hotspot alert timeline + the actual hotspot GeoJSON
into a clear, smooth 72-hour MP4 animation plus representative PNG frames, and
document where an application would eventually consume this output. No new scientific
states or alert events; only particle position may be smoothed for playback.

### Files changed

```text
fwd/animate.py                  new — LandBackground, interpolate_positions,
                                 status_by_hotspot_at, render_frame
scripts/f05_build_animation.py  new — builds representative frames + full animation,
                                 validates (not a test file)
output/ow-0009/ow0009_forward_animation.mp4   new — 289 frames @ 10 fps, 28.9 s, 0.20 MB
output/ow-0009/frames/initial.png             new
output/ow-0009/frames/watch.png               new
output/ow-0009/frames/critical_cape_greco.png new
output/ow-0009/frames/final.png               new
```

### Implementation summary

- **Basemap / coastline**: no coastline shapefile or map-projection library
  (cartopy/shapely) exists in this environment (re-confirmed here; only `matplotlib`
  was already available). Reused F01's static CMEMS land/sea mask
  (`ForcingAdapter.ocean_valid_mask`) as a `pcolormesh` background plus a contour line
  at the water/land boundary for a crisp coastline edge -- the same land data source
  used for stranding decisions in F02, so the animation's coastline is exactly
  consistent with the physics, not a separately-sourced approximation. Map axis is
  plain equirectangular lon/lat with `aspect = 1/cos(mean_latitude)` (requirement 7:
  geographically correct without a full projection library).
- **Hotspots**: `fwd.hotspots.load_hotspots` (F04, unmodified) supplies the actual
  polygon rings; rendered as `matplotlib.patches.Polygon` directly from those
  coordinates -- never a circle reconstructed from `radius_km`. Per-hour status comes
  from `hotspot_alerts.parquet` via `status_by_hotspot_at` (an exact-match lookup on
  the real recorded hour, never interpolated). Styling matches README section 10
  exactly: CLEAR = outline only; WATCH = yellow fill on the 100 km ring; CRITICAL =
  red fill on the 40 km ring with the 100 km yellow context retained underneath.
  Hotspot name labels are drawn at each ring's own vertex-mean center (reusing F04's
  already-verified concentric-ring finding), colored/bolded by status, and clipped to
  the axes (`clip_on=True`) so an off-AOI hotspot (Ceyhan Oil Terminal's center sits
  just north of the AOI, per the F04 audit) does not bleed its label into the figure
  margin -- caught and fixed during this task's own visual validation, see "Issues
  discovered."
- **Particles**: `interpolate_positions` linearly blends each particle's `lon`/`lat`
  between the two bracketing real hourly states at a configurable
  `substeps_per_hour` (4 used here -> 289 rendered instants from the 73 recorded
  hours) for smooth playback; `valid`/`stranded` are carried from the floor hour only
  (categorical, never interpolated), satisfying requirement 6. Active particles are
  small blue dots; stranded particles (none occur in this dataset, see "Issues
  discovered") would render as a distinct brown "x" marker with its own legend entry,
  and -- because F02 guarantees a stranded particle's two bracketing hourly positions
  are numerically identical -- the same interpolation code produces zero apparent
  motion for a stranded particle automatically, with no special-casing needed.
- **Output**: representative frames are rendered directly from the exact recorded
  hourly `trajectory_df` rows (no interpolation involved) at the four requested
  moments; the full sequence is rendered to a temporary directory and assembled into
  one MP4 with `ffmpeg` (already installed and on `PATH`, verified in this task) via
  `subprocess`, then the temporary PNGs are deleted, leaving only the MP4 and the four
  representative frames in `output/ow-0009/`.

### Requirement 9 -- application integration point

A dedicated research agent searched the entire `SIH-2026` workspace (all project
versions `project-v2` through `project-v5`, `ocean-and-wind-files`, `ais_*` data
directories) for any existing web/mobile/desktop application: no `package.json`,
`pubspec.yaml`, `next.config.*`, `vite.config.*`, `streamlit_app.py`, `manage.py`, or
`angular.json` anywhere; no `public/`, `static/`, `pages/`, `components/`, or `app/`
directory outside library internals; no code referencing `database_payload.json`,
`trajectory`, `hotspot`, `alert`, or `.geojson` outside the data-science pipelines
already known. **No frontend/application exists anywhere in this workspace.**
`project-v4/CLAUDE.md` explicitly forbids adding one ("a scientific prototype, not a
production platform"). Conclusion: there is currently no integration point to
document because there is no consuming application yet -- the deliverable for now is
the standalone MP4 (`output/ow-0009/ow0009_forward_animation.mp4`, directly playable)
and the machine-readable parquet outputs (`trajectory.parquet`,
`hotspot_alerts.parquet`) for a future application (or the hackathon demo itself) to
read directly. Per the task's own instruction ("do not rewrite the application
frontend yet"), nothing was built here beyond this finding.

### Validation

Command: `python scripts/f05_build_animation.py`

Evidence (abridged; full output inspected manually):

```text
1) representative frames, all 371 particles present at each exact recorded hour:
   initial              2019-04-28T15:38:41  active=371 stranded=0
     statuses: Beirut Port=WATCH, Palm Islands Nature Reserve=WATCH, Cape Greco MPA=WATCH
   watch                2019-04-29T12:38:41  active=371 stranded=0
     statuses: Palm Islands Nature Reserve=WATCH, Cape Greco MPA=WATCH
   critical_cape_greco  2019-04-30T06:38:41  active=371 stranded=0
     statuses: Cape Greco MPA=CRITICAL
   final                2019-05-01T15:38:41  active=371 stranded=0
     statuses: Cape Greco MPA=CRITICAL   (still critical at the end of the run)

2) interpolated 289 frame-times from the 73 recorded hours (4 substeps/hour) --
   matches (73-1)*4+1 exactly

3) rendered 289/289 frames (111.7 s), encoded to MP4 with ffmpeg (0.20 MB, 28.9 s of
   video covering the full 72 h at 10 fps) -- ALL CHECKS PASSED
```

**Visual inspection (as requested)** of all four representative PNGs (viewed directly,
not just asserted programmatically):
- **Initial frame**: Cyprus, the Levantine coast, and the Nile Delta/Suez coastline
  all render recognizably from the CMEMS land mask. The 371-particle cluster is a
  tight blue dot cluster south of Cyprus. Three hotspot circles (Cape Greco, Palm
  Islands, Beirut Port) already show yellow WATCH fill, all clearly labelled; the
  other 8 show unfilled gray outlines only (CLEAR); legend and timestamp are both
  present and readable.
- **WATCH frame** (2019-04-29 12:38): the particle cluster has moved north/into the
  Cape Greco watch circle; Beirut Port has dropped out of WATCH (matches F04's
  computed alert timeline -- Beirut Port was only WATCH from t0 through ~11:38 on
  2019-04-29, consistent with the simultaneous-alert list from F04).
- **Cape Greco CRITICAL frame** (2019-04-30 06:38): the particle cluster sits inside
  the red 40 km critical disc, which is drawn on top of the retained yellow 100 km
  context ring exactly per the required styling; label is bold red; no other hotspot
  is active at this hour, matching F04's recorded per-hour status.
- **Final frame** (2019-05-01 15:38): particles remain inside Cape Greco's critical
  zone at the very end of the run (min distance stayed well under 40 km through hour
  72); map, legend, and coastline remain consistent with the earlier frames -- no
  scientific-state drift introduced by the animation layer.

The initial label-clipping bug (see below) was caught by this same visual inspection
step before being fixed and the animation rebuilt.

### Issues discovered

- **No particle was ever stranded during this run** (confirmed again here from the
  same F03 trajectory), so the "stranded particle" marker/legend entry and the
  interpolation's stranded-particle no-motion behavior could not be visually
  demonstrated with real ow-0009 data in this deliverable -- consistent with the same
  caveat already recorded in F02/F03. The code path is implemented and reasoned about
  above (identical bracketing positions -> zero apparent motion) but not exercised by
  this specific case's real output.
- **Fixed during this task**: the first render had "Ceyhan Oil Terminal"'s label
  (whose polygon center sits just north of the AOI, per the F04 audit) bleeding into
  the figure margin above the plot frame, because `ax.text` is not clipped to the
  axes by default in matplotlib. Fixed by passing `clip_on=True`; the animation was
  regenerated after the fix and re-inspected (see the two representative-frame
  screenshots above, both from the corrected run).
- No other blockers.

### Status

```text
F05-ow0009: COMPLETE
Files changed : fwd/animate.py, scripts/f05_build_animation.py,
                output/ow-0009/ow0009_forward_animation.mp4,
                output/ow-0009/frames/{initial,watch,critical_cape_greco,final}.png
Validation    : python scripts/f05_build_animation.py -> ALL CHECKS PASSED
                + manual visual inspection of all 4 representative frames
Next step     : F06-ow0009 (below)
```

---

## Pilot case — F06-ow0009 (2026-09-25): OpenOil/OpenDrift reference comparison

### Objective

Validate the F03 ow-0009 forward simulation against a real OpenOil/OpenDrift
reference run using the same 371 initial positions, start time, duration, timestep,
forcing, and drift factors, and report quantitative position error at
+1h/+6h/+12h/+24h/+48h/+72h. Do not change F01-F05, and do not tune anything to force
agreement.

### Environment finding (important, not a blocker)

The Python interpreter this whole project (`fwd/*`, all prior scripts) has been run
under (`python3` / `py -3.14`, `C:\Users\venka\AppData\Local\Python\pythoncore-3.14-64`)
does **not** have `opendrift` installed. A second interpreter on this machine,
`py -3.13` (`C:\...\Programs\Python\Python313`), **does** have a real
`opendrift==1.14.11` installed, with `numpy`/`xarray`/`pandas`/`matplotlib`/`netCDF4`
also available there. This makes a genuine OpenOil reference run possible without
installing anything new. Consequently `scripts/f06_openoil_reference.py` must be run
with `py -3.13` specifically; every other script in this project continues to run
under the usual `python3`. This is documented here rather than silently assumed, per
task instructions to report any blocker/mismatch -- it is a one-time environment fact,
not a code change.

### Files changed

```text
scripts/f06_openoil_reference.py   new — builds + runs the real OpenOil reference
                                    simulation (run with `py -3.13`); self-contained,
                                    does not import fwd.* (requirement 2)
scripts/f06_compare_reference.py   new — loads both trajectories, computes per-
                                    checkpoint great-circle error, investigates
                                    (run with the usual python3)
output/ow-0009/openoil_reference.parquet    new — 27,083 rows (371 x 73), OpenOil output
output/ow-0009/openoil_comparison.parquet   new — 6 rows, one per checkpoint
```

### OpenOil reference configuration

Built from real, unmodified `data/28th_april/ocean.nc`/`wind.nc` -- OpenDrift's
`reader_netCDF_CF_generic` auto-detected both files with **zero file changes needed**:
`uo`/`vo` already carry CF `standard_name`s (`eastward_sea_water_velocity`/
`northward_sea_water_velocity`) that OpenDrift's built-in alias table maps straight to
`x_sea_water_velocity`/`y_sea_water_velocity`; `u10`/`v10` have `standard_name:
unknown` (a GRIB-conversion artifact, noted in the F00 audit) but OpenDrift's own
`add_standard_name_for_surface_grib_variables` helper recognizes the literal variable
names `u10`/`v10` and assigns `eastward_wind`/`northward_wind` automatically -- verified
in this task by inspecting OpenDrift's own reader source
(`opendrift/readers/__init__.py`, `opendrift/readers/basereader/__init__.py`), not
assumed.

Config, matching this project's locked physics as closely as OpenOil's own surface
allows (full rationale in `scripts/f06_openoil_reference.py`'s own docstring):

```text
seed:current_drift_factor = 1     (OpenOil default -- already matches)
seed:wind_drift_factor    = 0.03  (OpenOil default IS 0.03 -- already matches; set explicitly anyway)
drift:advection_scheme    = euler (OpenOil default -- already matches)
drift:stokes_drift        = False (set; no wave forcing available to either implementation)
drift:current_uncertainty = 0     (set; OpenOil default 0.05 is a stochastic term this project's own model has none of)
drift:wind_uncertainty    = 0     (set; OpenOil default 0.5, same reason)
drift:vertical_mixing     = False (set; keeps elements exactly at the surface, matching this project's strictly-2D transport)
processes:evaporation/emulsification/dispersion = False (set; surface-transport-only comparison, no weathering)
general:coastline_action  = stranding (left at OpenOil's default, using its own built-in GSHHG landmask --
                             irrelevant here since F03 already established this case never nears land)
```

371 real parcel lon/lat/id from `database_payload.json` seeded directly (`z=0`,
`time=2019-04-28T15:38:41`); `time_step=time_step_output=3600s`, `duration=72h`.
OpenOil's `trajectory` output dimension was verified (smoke-tested on 5 particles
first) to preserve seeding order exactly, so `particle_id` was mapped back
index-for-index -- checked programmatically in the script (`assert list(trajectory) ==
range(371)`), not assumed. The reference run itself completed in 3.9s wall time with
**0 elements ever deactivated** (matches F03's own finding that this case never
approaches land).

### Comparison results

Command: `py -3.13 scripts/f06_openoil_reference.py && python scripts/f06_compare_reference.py`

```text
Checkpoint   mean error (km)   median error (km)   max error (km)
+1h              0.001              0.001              0.002
+6h              0.005              0.005              0.013
+12h             0.010              0.009              0.021
+24h             0.038              0.036              0.063
+48h             0.100              0.096              0.152
+72h             0.201              0.169              0.645

mean-of-means across all 6 checkpoints: 0.059 km
worst single checkpoint max error: 0.645 km (at +72h, particle 367)
```

For scale: the slick's own mean net displacement over the same 72 h (from F03) is
~66.3 km. The +72h mean error (0.201 km) is **~0.3% of that displacement**; the worst
single-particle error (0.645 km) is **~1%**.

### Investigation (requirement 5 -- where the (tiny) discrepancy comes from)

- **t0 seeding check**: mean 0.000152 km, max 0.000267 km at the very first recorded
  state, where both implementations are seeded from the literally identical 371 real
  parcel positions. This ~15 cm residual is consistent with `OpenOil`'s output being
  stored as `float32` internally (confirmed: `o.result` dtype is `float32`) versus this
  project's `float64` throughout -- not a bug, just float precision, and two orders of
  magnitude below every later checkpoint's error.
- **+1h single-step error** (0.001 km mean, 0.002 km max): already nonzero after just
  one Euler step, ruling out "the error only appears after many compounding hours" --
  there is a small, real per-step numerical difference, not a delayed-onset bug. Given
  F01's own independent cross-check already showed this project's bilinear+temporal
  CMEMS/ERA5 interpolation matches a hand-computed reference to ~1e-6 (see F01's
  progress notes), interpolation itself is an unlikely dominant cause of even this
  1-2 m per-step difference.
  - **Time handling**: ruled out as a contributor -- both implementations use the
    same `time_step=time_step_output=3600s` and the same start time; F01 already
    independently verified exact-time (non-snapped) interpolation.
  - **Coastline handling**: ruled out -- 0 particles ever deactivated in either
    implementation (confirmed above and in F03).
  - **Wind-drift implementation**: unlikely primary cause -- `wind_drift_factor=0.03`
    and `current_drift_factor=1` are identical constants in both, and OpenOil's
    `drift:relative_wind` defaults to `False` (absolute wind, not wind-relative-to-
    current), matching this project's `u_oil = uo + 0.03*u10` formula exactly (not
    `uo + 0.03*(u10-uo)`).
  - **Coordinate conversion -- identified as the most likely dominant cause**:
    reading OpenDrift's own source (`opendrift/models/basemodel/__init__.py`,
    `update_positions`, ~line 4630) shows it converts velocity to a new position via
    `pyproj.Geod(ellps='WGS84').fwd(...)` -- a full **WGS84 ellipsoidal** geodesic
    forward calculation. This project's own `iste.geometry.metres_to_degrees`
    (reused unchanged from Stage III, per requirement 2) uses a **spherical-Earth**
    approximation with a single mean radius (`MEAN_EARTH_RADIUS_M = 6,371,000 m`) --
    and `iste/geometry.py`'s own docstring already documents this explicitly: "a
    standard first-order geodesic approximation, not a full ellipsoidal (WGS84)
    geodesic calculation... No geodesic library (e.g. `pyproj`) is installed in this
    project's [main] environment." That is a real, pre-existing, and already-
    documented Stage III design constraint (not something introduced or silently
    changed by this task), and its expected magnitude -- a fraction of a percent,
    since WGS84's flattening is ~1/298 -- matches the observed ~0.3-1% error exactly.
  - **Other**: `float32` storage in OpenOil's output (see t0 check) plausibly
    compounds a small amount into the later checkpoints alongside the
    spherical-vs-ellipsoidal difference, but is far too small on its own (15 cm) to
    explain the 72h numbers.
- **Growth shape**: `mean_error_km / elapsed_hours` rises smoothly from 0.0009 at +1h
  to 0.0028 at +72h (not a sudden jump at any one checkpoint), consistent with a
  small, systematic per-step geometric difference compounding roughly linearly over
  many identical steps -- exactly the signature expected from a constant
  spherical-vs-ellipsoidal conversion bias, not a one-off event (e.g. a coastline
  encounter, which does not occur here) or a random/stochastic divergence (this
  project has no randomness, and the OpenOil run was configured with all stochastic
  terms disabled to match).

### Verdict

**Scientifically acceptable.** Sub-kilometer agreement with a real OpenOil/OpenDrift
run across the full 72-hour horizon (0.2 km mean / 0.65 km max error, ~0.3-1% of net
displacement), with the residual discrepancy traced to a specific, pre-existing,
already-documented approximation (spherical vs. WGS84-ellipsoidal geodesic
conversion in Stage III's `iste.geometry`) rather than any interpolation, time-
handling, wind-drift, or coastline-handling bug. No parameter was tuned to reduce this
error (requirement 6) -- the OpenOil config above was chosen entirely to match this
project's *locked* physics, not to minimize the comparison result, and was fixed
before the comparison script was ever run.

### Issues discovered

- No blockers. One environment note (see above): the OpenOil reference script
  requires the `py -3.13` interpreter on this machine, not the `python3` used
  everywhere else in this project; documented in the script's own docstring and here
  so the run is reproducible by someone else on this machine.
- This comparison, like F02/F03's own validation, could not exercise the
  coastline-handling code path (no particle nears land in this specific case) --
  consistent with, and now further confirmed by, the OpenOil reference also showing 0
  deactivations. A future case whose parcels start closer to shore would be needed to
  compare coastline/stranding behavior specifically.

### Status

```text
F06-ow0009: COMPLETE
Files changed : scripts/f06_openoil_reference.py, scripts/f06_compare_reference.py,
                output/ow-0009/openoil_reference.parquet,
                output/ow-0009/openoil_comparison.parquet
Validation    : py -3.13 scripts/f06_openoil_reference.py
                && python scripts/f06_compare_reference.py -> sub-km agreement at all
                6 checkpoints, verdict ACCEPTABLE (see investigation above)
Next step     : F07-ow0009 (below)
```

---

## Pilot case — F07-ow0009 (2026-09-25): finalize the pilot

### Objective

Verify the complete F01-F06 chain is mutually consistent (not just each stage's own
already-passed validation in isolation), confirm the specific facts this task lists,
produce one machine-readable JSON run summary for future application integration, and
document the run's forcing/physics/policy/known-limitations in one place. No
scientific model change; no frontend.

### Files changed

```text
scripts/f07_finalize.py            new — cross-checks all outputs, writes the summary
                                    (not a test file)
output/ow-0009/run_summary.json    new — machine-readable run summary (requirement 4)
```

### Verification performed (requirement 1-3)

Command: `python scripts/f07_finalize.py`

```text
0) Inventory: all 10 expected output/ow-0009/ files present (trajectory, hourly
   diagnostics, hotspot alerts, OpenOil reference + comparison, MP4, 4 frames) --
   sizes printed and confirmed non-empty.

1) Loaded trajectory (27,083 rows), hourly_diagnostics (72 rows), hotspot_alerts
   (803 rows), openoil_comparison (6 rows).

2) Core facts (recomputed fresh, not read from an earlier progress.md entry):
   72 propagation hours between 2019-04-28 15:38:41 and 2019-05-01 15:38:41 : OK
   73 trajectory states                                                     : OK
   371 parcel IDs, exactly {0..370}                                         : OK
   final time = 2019-05-01 15:38:41 UTC                                     : OK
   100 km WATCH / 40 km CRITICAL, re-derived from the actual loaded polygons
     of all 11 hotspots (not hard-coded)                                    : OK

3) Cross-consistency (this is what F07 adds beyond each stage's own validation):
   - hourly_diagnostics' active_count/stranded_count at every one of its 72 rows was
     RECOMPUTED independently straight from trajectory.parquet's valid/stranded
     columns and compared -- 0 mismatches.
   - hotspot_alerts' 803 rows cover exactly the same 73 timestamps as trajectory
     (set-equality check, not just a count match).
   - ffprobe-verified the actual MP4 file: 289 frames @ 10 fps, 28.9s -- matches
     F05's own (73-1)*4+1 substep arithmetic exactly, confirming the shipped video
     file itself (not just the build log) has the expected frame count.

4) Cape Greco MPA CRITICAL event, re-checked directly from hotspot_alerts.parquet:
   first CRITICAL at 2019-04-30 06:38:41 UTC (unchanged since F04), minimum distance
   23.88 km, status still CRITICAL at the final hour (2019-05-01 15:38:41).
   Hotspots ever WATCH/CRITICAL: Beirut Port, Cape Greco MPA, Palm Islands Nature
   Reserve, Vasiliko Oil Terminal. Hotspots ever CRITICAL: Cape Greco MPA only.

5) OpenOil validation (F06) carried forward unchanged and re-printed from
   openoil_comparison.parquet: +72h mean error 0.201 km, max 0.645 km, verdict
   ACCEPTABLE.

ALL CHECKS PASSED
```

### `run_summary.json` (requirement 4)

Written to `output/ow-0009/run_summary.json`. Top-level keys: `case_id`,
`generated_at_utc`, `experiment` (start/end/duration/timestep/states/parcel count),
`forcing` (CMEMS/ERA5 sources+files+variables), `physics` (formula, drift factors,
Stokes/weathering flags, advection scheme), `nan_coastline_policy` (the 5-mode
vocabulary + fallback rule + land-mask source), `hotspots` (geojson path, 11 count,
100/40 km, alert logic, which hotspots triggered, Cape Greco's first-critical time and
minimum distance), `outputs` (relative paths to every artifact), `final_state`
(371 active / 0 stranded), `openoil_validation` (all 6 checkpoints' mean/max error,
verdict, primary residual cause), and `known_limitations` (see below -- the same list,
machine-readable).

### Documentation (requirement 5)

- **Forcing sources**: CMEMS `uo`/`vo` from `data/28th_april/ocean.nc` (149 lat x 216
  lon, hourly, 2019-04-28 00:30 -> 2019-05-01 23:30); ERA5 `u10`/`v10` from
  `data/28th_april/wind.nc` (25 lat x 36 lon descending-latitude, hourly, 2019-04-28
  00:00 -> 2019-05-01 23:00). Both used on their native grids, both interpolated to the
  exact requested particle timestamp (verified independently in F01), never snapped.
- **OpenOil physics used**: `u_oil = current_drift_factor*uo + wind_drift_factor*u10`,
  `v_oil = current_drift_factor*vo + wind_drift_factor*v10`, with
  `current_drift_factor=1.0`, `wind_drift_factor=0.03`, Euler advection, no Stokes
  drift, no weathering/vertical processes/diffusion (F02/F03) -- these are also
  OpenOil's own defaults for the first two, confirmed while building F06's reference
  run.
- **NaN/coastline policy**: CMEMS NaNs are never globally filled. A point's land/water
  classification comes from the *static* CMEMS `uo`/`vo` NaN mask at its nearest grid
  cell (verified time-invariant across all 96 forcing timesteps in F00/F01), kept
  deliberately separate (F01/F02) from the *interpolation-stencil* NaN check, which
  instead triggers a 3x3-then-5x5 inverse-distance-weighted valid-ocean-neighbor
  fallback (minimum 4 neighbors, weight `1/(d^2+epsilon)`, configurable max radius).
  A particle whose current position, or whose proposed next position, is land strands
  permanently at its last valid position (never on the land cell); a stranded particle
  never moves again (F02, requirement 7, re-verified in F03's full run).
- **Hotspot logic**: all 22 polygons (11 hotspots x 2 zones) loaded directly from
  `regional_alert_geofences_wide.geojson` and tested with true point-in-polygon
  containment (never a reconstructed circle); per hour, per hotspot: any active
  particle in the 40 km critical polygon -> CRITICAL; elif any active particle in the
  100 km watch polygon -> WATCH; else CLEAR; evaluated against active particles only
  (valid and not stranded), never the slick centroid (F04).
- **Known WGS84-vs-spherical-coordinate limitation**: this project's reused Stage III
  displacement conversion (`iste.geometry.metres_to_degrees`) uses a spherical-Earth
  approximation with a single mean radius, while the real OpenOil/OpenDrift reference
  used for validation performs a full WGS84-ellipsoidal geodesic calculation
  (`pyproj.Geod`). This is a pre-existing, already-documented Stage III design
  constraint (no geodesic library installed in this project's main environment), not
  introduced by this task, and was identified in F06 as the most likely dominant cause
  of the (small) OpenOil comparison residual.
- **OpenOil reference validation result**: sub-kilometer agreement with a real
  OpenOil/OpenDrift run across all 6 checkpoints (+1h through +72h); mean error grows
  from 0.001 km at +1h to 0.201 km at +72h (max 0.645 km), i.e. roughly 0.3-1% of the
  slick's own ~66 km net 72h displacement -- verdict ACCEPTABLE, no parameters were
  tuned to achieve this (F06).

### Remaining non-blocking limitations (carried into `run_summary.json`)

1. Spherical-vs-WGS84-ellipsoidal displacement approximation (above) -- primary
   residual cause vs. the OpenOil reference, not fixed (Stage III reuse policy).
2. This specific case's real 72h run never exercises `NEIGHBOR_FALLBACK` or
   `LAND_STRANDED` end-to-end (the slick stays >60 km offshore the whole time) --
   those paths were validated separately against synthetic/real boundary points in
   F01-F03, but not by ow-0009's own real trajectory.
3. Land classification is sourced from the static CMEMS NaN mask (~4.6 km grid
   resolution), not an independent coastline shapefile (none exists in this
   environment).
4. No frontend/application exists anywhere in the `SIH-2026` workspace (confirmed by
   a dedicated search in F05) -- these outputs are standalone files for now, per this
   task's own instruction not to build one.
5. `CLAUDE.md`/`README.md` still document forcing paths (`data/ocean.nc`,
   `files/28th_june/wind.nc`) that do not match the actual paths used throughout
   F01-F07 (`data/28th_april/ocean.nc`, `data/28th_april/wind.nc`) -- flagged since
   F00, the docs themselves were never edited (out of scope for every task so far).
6. `ow-0009.jpg`/`mask_ow-0009.png` (the source slick image/mask) do not exist as
   files anywhere in the workspace, only as metadata strings (F00) -- not a blocker
   for the physics (parcel coordinates were already extracted numerically), but noted
   again here since a future application might expect the source image.

### Status

```text
F00-ow0009: COMPLETE
F01-ow0009: COMPLETE
F02-ow0009: COMPLETE
F03-ow0009: COMPLETE
F04-ow0009: COMPLETE
F05-ow0009: COMPLETE
F06-ow0009: COMPLETE
F07-ow0009: COMPLETE
Files changed : scripts/f07_finalize.py, output/ow-0009/run_summary.json
Validation    : python scripts/f07_finalize.py -> ALL CHECKS PASSED
Final output directory : output/ow-0009/
Next step     : none -- ow-0009 pilot (F00-F07) is complete; the master 548-particle/
                20:35 UTC spec (this file's "Locked experiment" / "Task status"
                sections below) has not been started and is a separate, larger task
```

---

## Post-F07 revision — animation map zoom fix (2026-09-25)

### Objective

User feedback after F07: the animation (`ow0009_forward_animation.mp4`) and all four
representative frames used the full CMEMS AOI (8.97 x 6.18 deg, ~950 km wide) as the
map extent, so the slick -- which only ever spans a few tens of km -- rendered as a
barely-visible speck of a few pixels. Requested fix: rescale so the propagation/slick
is clearly visible, applied to every affected file.

### Files changed

```text
fwd/animate.py                  modified — added compute_view_bounds(); render_frame
                                 now takes a view_bounds parameter instead of a
                                 hard-coded module-level AOI constant (kept as
                                 FULL_AOI, documented as a fallback/reference only,
                                 no longer used by the animation itself); particle
                                 marker size increased (s=5->16) for the tighter view
scripts/f05_build_animation.py  modified — computes view_bounds once from the full
                                 trajectory and passes it to every render_frame call
                                 (representative frames + all 289 sequence frames)
output/ow-0009/ow0009_forward_animation.mp4        regenerated
output/ow-0009/frames/{initial,watch,critical_cape_greco,final}.png   regenerated
```

No scientific data changed -- `trajectory.parquet`, `hotspot_alerts.parquet`, and
every other non-visualization output are untouched; this only changes what portion of
the map is drawn.

### Approach (including one false start, corrected before finishing)

First attempt: zoom to the union of the full trajectory bounding box *plus* the full
40 km critical ring of any hotspot that reaches CRITICAL (Cape Greco MPA), reasoning
that the most important alert event should never be clipped. This was implemented,
rendered, and then caught by this task's own visual re-inspection: Cape Greco's
critical ring extends substantially west into Cyprus (land), so forcing the whole ring
into view pulled the frame away from the slick and let land dominate most of the
image -- the opposite of "clearly show the propagation and slick." This was reverted
before producing the final deliverable.

Final approach: `compute_view_bounds(trajectory_df)` bounds the view to the full 72h
trajectory only (every particle, every recorded hour, so the extent is fixed for the
whole video rather than zooming/panning frame to frame), with proportional padding
(40% of the trajectory's own span per axis, minimum 0.15 deg) for surrounding
context. Hotspot zones -- including Cape Greco's critical ring -- render however much
of them falls inside this trajectory-centered window and are clipped otherwise (the
same clipping already accepted for Ceyhan Oil Terminal's label in F05's original
build); a zone's own color fill still makes an active alert unambiguous even when its
label or full circle is partially cut off.

Resulting view: **1.04 x 1.00 deg**, vs. the previous 8.97 x 6.18 deg -- roughly a 53x
reduction in displayed area, computed fresh from this run's actual trajectory rather
than a hand-picked constant (so it would adapt correctly if ow-0009's parcels/drift
ever changed).

### Validation

`python scripts/f05_build_animation.py` -> ALL CHECKS PASSED (same structural checks
as the original F05 run: 289 interpolated frame-times, MP4 written and non-empty).
`ffprobe` confirms the regenerated MP4: 289 frames @ 10 fps, 28.9 s (unchanged from
before -- only the framing changed, not the frame count/timing).

All four representative frames were re-inspected visually after the final rebuild:
individual particles are now clearly distinguishable as separate dots (previously a
single indistinct cluster a few pixels wide), the drift path is visible as the cluster
elongates into a diagonal line by day 3, and the Cape Greco critical-zone crossing
(12/371 particles, 39.66 km min distance) reads clearly against the red zone at an
appropriate scale.

### Status

```text
Zoom fix: COMPLETE
Files changed : fwd/animate.py, scripts/f05_build_animation.py,
                output/ow-0009/ow0009_forward_animation.mp4,
                output/ow-0009/frames/{initial,watch,critical_cape_greco,final}.png
Validation    : python scripts/f05_build_animation.py -> ALL CHECKS PASSED
                + manual visual re-inspection of all 4 representative frames
```

---

## Post-F07 revision 2 — particle drift-trail visualization (2026-09-25)

### Objective

User feedback: even after the zoom fix, "the movement does not look natural" and
"I cannot clearly visualize the trajectories of the particles." Requested: confirm
the oil-slick source data is correct, and fix the visualization so trajectories are
clearly visible.

### Data confirmation (requested alongside the visualization fix)

The user pasted a slice of `database_payload.json`'s ow-0009 parcels (ids 0-5, 105,
316, etc.) and asked for confirmation this is the data being used. Checked
programmatically against `data/database_payload.json`: every pasted id's
`lat`/`lon`/`mass_kg`/`confidence` matches exactly (e.g. id 0:
`lat=34.633203, lon=34.878516, mass_kg=193.27, confidence=0.3376`, identical in both).
Confirmed: yes, this is the exact real data used throughout F01-F07.

### Root cause of "unnatural" movement (diagnosed before changing anything)

Extracted actual mid-sequence frames from the (already-zoom-fixed) MP4 with `ffmpeg`
and inspected them directly, rather than guessing. Two real, compounding causes, not
a simulation bug:
1. All 371 parcels start within ~3 km of each other -- smaller than one CMEMS grid
   cell -- so under this project's locked no-diffusion physics (CLAUDE.md forbids
   adding spread/randomness) they drift under nearly identical current/wind and
   barely separate visually; they read as a single blob rather than 371 distinct
   moving points. This is correct physics, not a defect.
2. **The animation only ever drew each particle's current position, never its path.**
   A dot that jumps from point to point with no trail does not read as a
   "trajectory" to a viewer -- there was nothing on screen showing where a particle
   had been, only where it currently is.

Root cause 2 is the one actually fixed here; root cause 1 is inherent to the locked
physics and not something to "fix" (adding artificial spread would be exactly the
kind of undocumented scientific change CLAUDE.md prohibits).

### Files changed

```text
fwd/animate.py                  modified — added precompute_particle_paths() and
                                 build_trail_segments(); render_frame() gained an
                                 optional trail_segments parameter (LineCollection,
                                 drawn under the current-position dots); legend
                                 gained a "Parcel drift path (since t0)" entry
scripts/f05_build_animation.py  modified — precomputes paths once, builds trail
                                 segments per frame, passes them to every
                                 render_frame call (representative + all 289
                                 sequence frames)
output/ow-0009/ow0009_forward_animation.mp4        regenerated
output/ow-0009/frames/{initial,watch,critical_cape_greco,final}.png   regenerated
```

No scientific data changed -- this only draws already-computed F03 positions as a
connected line; it does not re-simulate, smooth, or alter any trajectory value.

### Implementation

`precompute_particle_paths` sorts each particle's real hourly `(time, lon, lat)` once
(not per frame -- avoids re-filtering the full 27k-row table 289 times).
`build_trail_segments(paths, up_to_time, current_positions)` slices each particle's
path up to `up_to_time` via `np.searchsorted` (O(371) per frame) and appends that
particle's current (possibly sub-hour-interpolated) position as the trail's leading
tip, so the trail always lines up exactly with the plotted dot. `render_frame` draws
all 371 segments in one `matplotlib.collections.LineCollection` call (thin, `alpha=0.4`)
underneath the current-position scatter -- where particles move near-identically the
overlapping semi-transparent lines read as one solid streak; where they diverge (e.g.
the one parcel that separates from the main cluster, visible in every representative
frame) the individual path becomes its own visible curve.

Performance impact: ~0.45s/frame (was ~0.42s/frame without trails) -- negligible;
full 289-frame build still completes in ~131s.

### Validation

`python scripts/f05_build_animation.py` -> ALL CHECKS PASSED (same structural checks:
289 frames, MP4 non-empty). Re-extracted sample frames from the rebuilt MP4 with
`ffmpeg` (not just the 4 representative PNGs) and inspected them directly: the trail
visibly grows frame-by-frame through the video, the S-curve drift path is clearly
traceable, and the previously-mysterious "stray" particle's own curved path back to
the main cluster's origin is now visible and explained (it is a real parcel from a
different part of the originally observed slick footprint, not a rendering artifact).

### Status

```text
Drift-trail fix: COMPLETE
Files changed : fwd/animate.py, scripts/f05_build_animation.py,
                output/ow-0009/ow0009_forward_animation.mp4,
                output/ow-0009/frames/{initial,watch,critical_cape_greco,final}.png
Validation    : python scripts/f05_build_animation.py -> ALL CHECKS PASSED
                + manual visual inspection of representative frames AND freshly
                extracted mid-sequence frames from the rebuilt MP4
Data confirmation : ow-0009 parcel data verified identical to database_payload.json
                (user-pasted sample cross-checked programmatically)
```

---

## Post-F07 revision 3 — hotspot alert threshold change + animation intro zoom (2026-09-25)

### Objective

Two user-requested changes:
1. Hotspot alerts should trigger when **~30% of the active slick** is inside a zone,
   not when a single particle enters it.
2. The animation should open zoomed out to show all 11 hotspots for ~3 seconds, then
   zoom in to the trajectory-focused scale used for the rest of the video.

### Change 1: hotspot alert threshold -- a real, documented spec deviation

**This changes locked alert logic.** README section 10 / CLAUDE.md explicitly lock
"ANY active particle inside a zone triggers it," and CLAUDE.md section 13 lists
"hotspot thresholds" and "alert logic" among the things that must never be changed
*silently*. This change is not silent: it was explicitly requested by the user, and is
recorded here in full per section 13's own instruction ("if a scientifically necessary
change appears, document it in progress.md... before proceeding"). It applies only to
this ow-0009 pilot's `fwd.hotspots` module, not to `README.md`/`CLAUDE.md` themselves
(left unedited, consistent with every prior task in this pilot).

**Before implementing, checked what the change would actually do** (queried the
existing `hotspot_alerts.parquet` for the maximum fraction of the slick ever inside
each hotspot's zones): Beirut Port reaches 59.0% (100km), Cape Greco MPA reaches 100%
(both zones, eventually), Palm Islands Nature Reserve reaches 100% (100km), Vasiliko
Oil Terminal reaches only 6.5% (100km) at its absolute peak. This correctly predicted
that Vasiliko would drop out entirely under a 30% rule and the other three would still
trigger, just later -- confirmed after implementing (see Results below).

**Files changed:**
```text
fwd/hotspots.py   modified — HotspotThresholds gained min_fraction_critical/
                  min_fraction_watch (both default 0.30, configurable, not
                  hard-coded); evaluate_hotspot_alerts now takes a `thresholds`
                  parameter and compares (particles in zone)/(active particles) to
                  the relevant fraction instead of testing "any particle"; two new
                  output columns, fraction_inside_40km/fraction_inside_100km, added
                  for audit transparency now that the decision itself depends on them
output/ow-0009/hotspot_alerts.parquet   regenerated (python scripts/f04_run_hotspot_alerts.py)
```
`particles_inside_40km`/`particles_inside_100km` keep their original literal-count
meaning; the new fraction columns and the status decision are the only things that
changed. The polygon containment test itself, the distance metric, and "never use the
slick centroid" are all unchanged -- only the aggregation rule.

**Results** (re-ran `scripts/f04_run_hotspot_alerts.py`):
```text
Vasiliko Oil Terminal: WATCH -> CLEAR (entire run) -- never reached 30% (peaked at 6.5%)
Beirut Port:           still WATCH, first_watch_time unchanged (t0) -- reaches 59.0%
Palm Islands Nature Reserve: still WATCH, first_watch_time unchanged (t0) -- reaches 100%
Cape Greco MPA:        still reaches CRITICAL, but first_critical_time shifted from
                        2019-04-30 06:38:41 (old rule, only 3.2% of the slick) to
                        2019-04-30 18:38:41 (new rule, first hour >=30%, actually 40.2%)
                        -- a real, correctly-later alert under the new, less
                        trigger-happy rule; Cape Greco eventually reaches 100% of the
                        slick inside its critical zone later in the run
```
Simultaneous-alert count dropped from 50 to 32 of the 73 hours (Vasiliko's extra
alert hours are gone; the core Beirut+Cape Greco+Palm Islands overlap is unchanged).

### Change 2: animation opening sequence -- regional overview, then eased zoom

Pure visualization change, no scientific data touched.

**Files changed:**
```text
fwd/animate.py                  modified — added compute_hotspot_overview_bounds()
                                 (bounds all 11 hotspots' 100km watch rings + small
                                 padding), ease_in_out() (cubic smoothstep easing),
                                 interpolate_view_bounds() (eases the map view between
                                 two extents); render_frame() gained an optional
                                 overview_label parameter (replaces the "active/
                                 stranded" subtitle during the opening sequence)
scripts/f05_build_animation.py  modified — opening sequence: 3.0s (30 frames) held
                                 on the wide regional-overview bounds at t0, then 1.5s
                                 (15 frames) eased zoom-in to the existing trajectory-
                                 tight bounds, still frozen at t0 (the propagation
                                 clock does not advance until the zoom finishes);
                                 the "critical_cape_greco" representative-frame
                                 timestamp is now computed dynamically from
                                 hotspot_alerts.parquet (cape_greco_first_critical_
                                 hour()) instead of a hard-coded literal, so it can't
                                 go stale the next time the threshold above is tuned
output/ow-0009/ow0009_forward_animation.mp4        regenerated (334 frames total: 45
                                                    opening + 289 propagation, 33.4s)
output/ow-0009/frames/{initial,watch,critical_cape_greco,final}.png   regenerated
```

During the 3.0s hold, the map shows all 11 hotspots (labelled, with their real t0
WATCH/CLEAR status already colored in) with the subtitle "Regional hotspot overview --
11 monitored zones"; during the 1.5s zoom, the subtitle reads "Zooming to the ow-0009
slick..." while the view eases (cubic smoothstep, not linear) from the wide bounds to
the trajectory-tight bounds used for the rest of the video. Both phases are frozen at
t0 -- no particle motion, no new alert events -- only the camera moves.

### Validation

`python scripts/f04_run_hotspot_alerts.py` -> ALL CHECKS PASSED (same structural
invariants as before: 803 rows, particles_inside_40km <= particles_inside_100km at
every row, first_watch_time <= first_critical_time).

`python scripts/f05_build_animation.py` -> ALL CHECKS PASSED. Visually re-inspected
(via `ffmpeg`-extracted frames, not just the 4 representative PNGs): the opening wide
shot clearly shows all 11 labelled hotspots including the 5 that never trigger in this
run (Port Said, Alexandria, Ceyhan, Haifa, Akamas, Rosh HaNikra, Nile Delta -- visible
now for the first time in this animation, since the main sequence's tight zoom cuts
most of them out of frame); the zoom transition is smooth (eased, not a jump cut); the
corrected `critical_cape_greco.png` now genuinely shows CRITICAL status (previously,
after Change 1 alone, this frame's old fixed timestamp would have shown a stale WATCH
status -- caught and fixed as part of this same revision, before considering it done).

`python scripts/f07_finalize.py` -> ALL CHECKS PASSED and `run_summary.json`
regenerated: `hotspots.logic`/`min_fraction_critical`/`min_fraction_watch` reflect the
new rule, `triggered_hotspots` no longer includes Vasiliko Oil Terminal,
`cape_greco_first_critical_time_utc` updated to 18:38:41, and a new
`known_limitations` entry documents this deviation explicitly. F07's own hard-coded
"Cape Greco reaches CRITICAL at exactly 06:38:41" assertion was relaxed to check the
*fact* (reaches CRITICAL) without pinning the *exact time* to one now-configurable
threshold's old value; the MP4 frame-count check changed from `== 289` to `>= 289`
now that the shipped file includes the opening sequence.

### Status

```text
Hotspot threshold + animation intro revision: COMPLETE
Files changed : fwd/hotspots.py, fwd/animate.py, scripts/f05_build_animation.py,
                scripts/f07_finalize.py, output/ow-0009/hotspot_alerts.parquet,
                output/ow-0009/ow0009_forward_animation.mp4,
                output/ow-0009/frames/{initial,watch,critical_cape_greco,final}.png,
                output/ow-0009/run_summary.json
Validation    : f04/f05/f07 scripts all re-run, ALL CHECKS PASSED; animation frames
                re-inspected visually (opening overview, eased zoom, corrected
                critical frame)
Deviation recorded : hotspot alert rule changed from "any particle" to ">=30% of
                active particles" per hotspot per zone -- explicit, user-directed,
                documented here per CLAUDE.md section 13; not applied to README.md/
                CLAUDE.md themselves, only to this pilot's fwd.hotspots module
```

---

## Post-F07 revision 4 — overview shot showed blank area outside the CMEMS grid (2026-09-25)

### Objective

User feedback on the new opening overview shot (revision 3): when zoomed out to show
all 11 hotspots, part of the frame (north/east, around Ceyhan Oil Terminal's ring)
extended past the CMEMS grid's actual coverage and rendered as blank white space (no
land/water background data exists there). Requested: either show land/ocean there too,
or zoom out only as far as the AOI, not beyond it.

### Fix

Chose the second option (zoom no wider than the AOI) -- there is no data source in
this project for land/water outside the CMEMS grid (no coastline shapefile exists
here, per the F00 audit; showing a real basemap under Ceyhan's ring would require a
new dependency this project doesn't have), so clamping the shot to the AOI is the
correct fix rather than trying to fabricate a background for the uncovered area.

**Files changed:**
```text
fwd/animate.py                  modified — added clip_view_bounds() (intersects two
                                 view-bounds dicts); compute_hotspot_overview_bounds()
                                 gained an optional clamp parameter, applied after
                                 computing the hotspot-ring-based bounds
scripts/f05_build_animation.py  modified — passes the CMEMS grid's actual lon/lat
                                 extent (from LandBackground, i.e. the real basemap
                                 data's own coverage, not a hard-coded duplicate of
                                 the AOI constant) as compute_hotspot_overview_bounds'
                                 clamp
output/ow-0009/ow0009_forward_animation.mp4        regenerated (still 334 frames, 33.4s)
output/ow-0009/frames/{initial,watch,critical_cape_greco,final}.png   regenerated
output/ow-0009/run_summary.json                    regenerated (file size only)
```

Effect: the overview bounds' east/south/north edges are now clamped to the CMEMS
grid's exact extent (west was already inside it). Ceyhan Oil Terminal's ring -- the
one hotspot whose center sits outside the AOI (per the F00 audit) -- is now mostly
cropped out of the opening shot rather than showing with blank space around it; this
is an acceptable tradeoff since Ceyhan never triggers any alert in this run (confirmed
in F04's results) and the fix's actual purpose (show every hotspot that can matter,
with a fully-drawn basemap under all of them) is preserved for the other 10.

### Validation

`python scripts/f05_build_animation.py` -> ALL CHECKS PASSED. Re-extracted the first
overview frame from the rebuilt MP4 with `ffmpeg` and inspected it directly: the
entire visible frame is now land or water (no blank area), all 10 in-AOI hotspots
remain labelled and visible. `python scripts/f07_finalize.py` -> ALL CHECKS PASSED,
`run_summary.json` regenerated (only file-size/timestamp fields changed).

### Status

```text
Overview-clamp fix: COMPLETE
Files changed : fwd/animate.py, scripts/f05_build_animation.py,
                output/ow-0009/ow0009_forward_animation.mp4,
                output/ow-0009/frames/{initial,watch,critical_cape_greco,final}.png,
                output/ow-0009/run_summary.json
Validation    : python scripts/f05_build_animation.py -> ALL CHECKS PASSED
                + manual visual re-inspection of the opening overview frame
                python scripts/f07_finalize.py -> ALL CHECKS PASSED
```

---

## New case — ow-0008 pilot (2026-09-25)

### Objective

User request: run the complete forward-propagation pipeline for a second real case,
ow-0008 (548 parcels, `data/14th_june/ocean.nc`+`wind.nc`), producing every output
type ow-0009 has (trajectory, hourly diagnostics, hotspot alerts, MP4 animation + 4
representative frames, OpenOil reference + comparison, run_summary.json, CSV
exports), in `output/ow-0008/`, "perfectly... no scope of errors."

### Data confirmation (before writing any code)

User-pasted parcel data cross-checked programmatically against
`data/database_payload.json`: `image_filename="ow-0008.jpg"`, `timestamp=
"2019-06-14T20:35:58"`, `total_parcels=548`, `bounding_box` all match exactly;
spot-checked ids 0, 1, 9, 313, 547 -- all match exactly. This is the same
548-parcel/June-2019 case identified (but not used) back in the F00 audit as the
dataset whose parcel *count* matches CLAUDE.md's master "548 particles" lock while
its date does not (2019-06-14, not 2019-04-28) -- still true and still not a
resolution of that master-spec discrepancy; this is a separate pilot case, exactly
like ow-0009 was.

### Pre-flight checks that caught two real issues before they became bugs

1. **ERA5 wind time-dimension name differs between cases.** Inspecting
   `data/14th_june/wind.nc` before writing any code found its time coordinate is
   named `valid_time`, not `time` as in ow-0009's `data/28th_april/wind.nc`
   (different GRIB-to-NetCDF conversion). `ForcingAdapter` hard-coded `time_dim="time"`
   for wind, which would have failed outright (or worse, silently misbehaved) on this
   file. **Fixed in `fwd/forcing.py`**: added `_detect_wind_time_dim()`, which
   inspects the file's own coordinates and picks whichever of `("time", "valid_time")`
   is present, raising a clear `KeyError` if neither is -- auto-detected, not a manual
   parameter a future caller could forget to set correctly. Regression-verified this
   does not change ow-0009's behavior (its wind.nc still resolves to `"time"`).
2. **ow-0008 starts much closer to shore than ow-0009.** Checked distance-to-nearest-
   CMEMS-land-cell for all 548 parcels before running anything: 28.97-30.06 km (mean
   29.5 km), vs ow-0009's 76.8-80.3 km. This meant the run could plausibly exercise
   `NEIGHBOR_FALLBACK`/`LAND_STRANDED` for the first time on a real case (it did not,
   see below, but the check was made in advance rather than assumed away).

### Generalizing the shared `fwd/` modules (not duplicating them)

Per CLAUDE.md's "reuse compatible components... do not create dozens of micro-files,"
the case-specific hard-coding in the shared modules was generalized rather than
copy-pasting a parallel `fwd_ow0008/` package:

```text
fwd/cases.py         new — CaseConfig/CASES/get_case: one source of truth per case
                      (image_filename, ocean/wind paths, observation_time,
                      total_parcels), verified against database_payload.json and the
                      actual NetCDF files, not invented
fwd/forcing.py        modified — _detect_wind_time_dim() (see above); no behavior
                      change for existing callers
fwd/simulate.py       modified — load_ow0009_parcels() is now a thin wrapper around
                      a new generic load_spill_parcels(image_filename, ...); run_
                      simulation() was already fully generic (no change needed)
fwd/animate.py        modified — render_frame() gained a case_label parameter
                      (defaults to "ow-0009" for backward compatibility) used in the
                      title instead of a hard-coded string
fwd/hotspots.py, fwd/boundary.py, fwd/step.py   unchanged -- already fully generic
                      (operate on trajectory_df/adapter/hotspots, no case-specific
                      hard-coding beyond docstring mentions)
```

**Regression-verified after every change**, not assumed safe: re-ran
`f01_validate_forcing.py`, `f02_validate_step.py` (ALL CHECKS PASSED, unchanged), and
re-ran `f03_run_simulation.py`/`f04_run_hotspot_alerts.py` for ow-0009 and compared
output checksums before/after -- **bit-for-bit identical**
(`sum(lon)=937883.6131319827`, `sum(lat)=940967.5339131828` before and after F03;
identical `particles_inside_40km`/`particles_inside_100km`/alert-row sums before and
after F04). The refactor changed nothing about ow-0009's own results.

### New per-case scripts (mirroring the proven ow-0009 scripts, not inventing new logic)

```text
scripts/ow0008_run_simulation.py     F02+F03 equivalent: run + validate 72h sim
scripts/ow0008_run_hotspot_alerts.py F04 equivalent: hotspot alerts + validate
scripts/ow0008_build_animation.py    F05 equivalent: animation + 4 frames + validate
                                      (representative-hour selection is fully
                                      data-driven -- pick_representative_hours() --
                                      not hard-coded to a specific hotspot, since
                                      which hotspot(s) ow-0008 would trigger was not
                                      known in advance; fixed one bug in its own first
                                      draft before running it for real -- see below)
scripts/ow0008_openoil_reference.py  F06 part 1 (py -3.13): real OpenOil run
scripts/ow0008_compare_reference.py  F06 part 2: checkpoint comparison, tolerant of a
                                      particle being deactivated/NaN in the OpenDrift
                                      reference at some checkpoint (not assumed
                                      impossible, unlike ow-0009's version, given how
                                      close to shore this case starts)
scripts/ow0008_finalize.py           F07 equivalent: cross-consistency + run_summary.json
```

A bug in `ow0008_build_animation.py`'s first draft was caught before running it for
real: `pick_representative_hours` originally tagged "the first non-CLEAR row" as the
WATCH frame regardless of whether that row's actual status was WATCH or CRITICAL --
harmless for ow-0009 (where the first alert really was a WATCH), but wrong here,
where Beirut Port is CRITICAL from the very first hour. Fixed to select the first
true WATCH-status row and the first true CRITICAL-status row independently.

### Results

**Simulation** (`python scripts/ow0008_run_simulation.py`): 548 parcels, 73 states,
first=2019-06-14 20:35:58, final=2019-06-17 20:35:58 (exactly +72h) -- all invariants
passed. Active-particle oil speed range [0.0627, 0.3753] m/s. **0 particles stranded**
and **all 39,456 steps were `DIRECT`** (0 `NEIGHBOR_FALLBACK`, 0 `LAND_STRANDED`) --
despite starting only ~29 km offshore, the slick's real drift never actually crossed
within a CMEMS grid cell of the static land mask. Net displacement per particle over
72h: 13.3-16.3 km (mean 14.6 km), much smaller than ow-0009's ~66 km -- consistent
with the smaller max instantaneous speed observed.

**Hotspot alerts** (`python scripts/ow0008_run_hotspot_alerts.py`, same >=30% rule now
in effect for both cases): **Beirut Port is CRITICAL from t0 through the final hour**
(100% of the slick inside the 40 km zone at every recorded hour, min distance 20.01 km
to the port); **Palm Islands Nature Reserve is WATCH for the entire run** (min
distance 51.73 km). The other 9 hotspots stay CLEAR throughout. This is a
substantively different regime from ow-0009 (which only marginally grazed CRITICAL
near the very end of its run) -- ow-0008 is a genuine, sustained, full-slick critical
event from the first observation, and is the first real case in this project where
the CRITICAL alert path is exercised by a real trajectory rather than only by
synthetic/boundary-point tests.

**Animation** (`python scripts/ow0008_build_animation.py`): 334 frames (45 opening +
289 propagation), 33.4s, same structure as ow-0009's (regional overview -> eased zoom
-> trajectory-zoomed 72h sequence with drift trails). Visually inspected the initial
and final representative frames directly: the drift trail reveals the slick tracing a
**looping/spiraling path** (consistent with an inertial oscillation in the coastal
current) over the 3 days while remaining inside Beirut Port's 40 km critical zone the
entire time -- a striking, physically plausible pattern that the trail-visualization
feature (added earlier for ow-0009 per user feedback) made directly visible; with only
point markers this structure would not have been apparent at all.

**OpenOil validation** (`py -3.13 scripts/ow0008_openoil_reference.py` then
`python scripts/ow0008_compare_reference.py`): OpenDrift's reference run also shows
**0 deactivations** across all 548 particles (agreeing with our own 0-stranded
result). Sub-100-meter agreement at every checkpoint: mean error grows from 0.001 km
at +1h to **0.041 km (41 m) at +72h**, max 0.049 km -- tighter absolute agreement than
ow-0009's (0.201 km at +72h), consistent with ow-0008's smaller net displacement (14.6
km mean vs ow-0009's 66.3 km) and the same spherical-vs-WGS84-ellipsoidal residual
cause identified in the original F06 investigation.

**Finalize** (`python scripts/ow0008_finalize.py`): all cross-consistency checks
passed (hourly_diagnostics recomputed independently from trajectory and matched;
hotspot_alerts timestamps match trajectory exactly; MP4 frame count verified via
`ffprobe`); `run_summary.json` written.

**CSV exports**: all 5 parquet files exported to `output/ow-0008/csv/*.csv`, matching
the convenience exports already provided for ow-0009.

### Full output parity with ow-0009 (user's explicit ask)

```text
output/ow-0008/trajectory.parquet                       (40,004 rows = 548 x 73)
output/ow-0008/hourly_diagnostics.parquet                (72 rows)
output/ow-0008/hotspot_alerts.parquet                     (803 rows = 11 x 73)
output/ow-0008/ow0008_forward_animation.mp4               (334 frames, 33.4s, 0.38 MB)
output/ow-0008/frames/initial.png
output/ow-0008/frames/watch_palm_islands_nature_reserve.png
output/ow-0008/frames/critical_beirut_port.png
output/ow-0008/frames/final.png
output/ow-0008/openoil_reference.parquet                  (40,004 rows)
output/ow-0008/openoil_comparison.parquet                  (6 rows)
output/ow-0008/run_summary.json
output/ow-0008/csv/*.csv                                   (5 files, mirroring output/ow-0009/csv/)
```

### Status

```text
ow-0008 pilot: COMPLETE
Files changed : fwd/cases.py (new), fwd/forcing.py, fwd/simulate.py, fwd/animate.py
                (all generalized, zero behavior change for ow-0009, regression-
                verified with matching before/after checksums),
                scripts/ow0008_run_simulation.py, ow0008_run_hotspot_alerts.py,
                ow0008_build_animation.py, ow0008_openoil_reference.py,
                ow0008_compare_reference.py, ow0008_finalize.py (all new),
                output/ow-0008/** (new, full parity with output/ow-0009/**)
Validation    : every stage's own script ran with ALL CHECKS PASSED; ow-0009
                regression-checked at every shared-module change point (F01/F02
                scripts pass, F03/F04 outputs byte-identical before/after)
Key finding   : Beirut Port CRITICAL for the entire 72h run (100% of slick inside
                40km zone throughout) -- the first real case in this project to
                exercise a sustained CRITICAL alert end-to-end
```

---

## Post-ow-0008 revision — zoom, drift-pattern investigation, implementation-origin question (2026-09-25)

### Objective

Three user follow-ups on the ow-0008 pilot: (1) the visualizations were still not
zoomed in enough to clearly see particle movement; (2) the drift pattern looked
"off," as if particles were circling back to the same position repeatedly, prompting
a request to confirm this isn't a bug; (3) a direct question of whether this project
uses the OpenDrift library to implement the propagation itself, or builds it from
scratch.

### (2) investigated first, before touching any code

Before changing anything, pulled one particle's raw `uo`/`vo` (CMEMS current, straight
from `trajectory.parquet`, untouched by any of this project's own code) over the full
72h and inspected it directly: the current direction rotates smoothly through a full
circle roughly every 21-25 hours (e.g. particle 0: SW at t0, W by +7h, NE by +12h, SE
by +16h, S by +21h, back to SW-ish by +25h). Computed the theoretical inertial-
oscillation period at this latitude (34.07N): `f = 2*Omega*sin(lat) = 8.170e-5 rad/s`,
`T = 2*pi/f = 21.36 hours` -- matching the observed rotation almost exactly. This is
the textbook signature of a near-inertial oscillation (Coriolis-driven rotary current,
common in the Mediterranean after a wind event -- the ERA5 wind in this same window
shows a multi-m/s gust), confirmed from the raw forcing data itself, not from
anything this project's advection code computed. **Not a bug** -- the looping
particle paths are real physics, correctly advected by `fwd.step` along a genuinely
rotating current field. Told the user this with the supporting numbers rather than
just asserting it.

### (1) the zoom bug, found and fixed

`fwd.animate.compute_view_bounds`'s padding had a fixed floor, `min_padding_deg=0.15`,
sized against ow-0009's own trajectory span (~0.58 deg): the proportional 40% padding
rule always exceeded that floor for ow-0009, so it was invisible there. ow-0008's
much tighter spiral (raw span only ~0.237 x 0.106 deg) is smaller than the floor
itself, so the fixed 0.15 deg dominated on *both* axes regardless of the trajectory's
actual size -- padding out to 0.537 x 0.406 deg, nearly 2.3x/3.8x the trajectory's own
span, which is why the spiral looked small and hard to read.

**Fix**: reduced `min_padding_deg` to `0.03` (`fwd/animate.py`) -- small enough to
only act as a degenerate-case floor, letting the proportional 40% rule govern real
trajectories of any size. Verified before touching the rebuild: recomputed
`compute_view_bounds` for both cases with the new floor -- ow-0009's bounds were
**identical to 4 decimal places** (as expected, its proportional padding was never
near the old floor), ow-0008's shrank from 0.537 x 0.406 deg to 0.427 x 0.191 deg (a
53% reduction on the latitude axis, which had been the more floor-dominated one).

**Files changed:**
```text
fwd/animate.py                                    modified — min_padding_deg 0.15 -> 0.03
output/ow-0008/ow0008_forward_animation.mp4       regenerated
output/ow-0008/frames/*.png                       regenerated (all 4)
output/ow-0008/run_summary.json                   regenerated (file sizes only)
```
ow-0009's animation/frames were deliberately **not** touched -- confirmed unaffected
above, so there was nothing to regenerate there.

**Validation**: `python scripts/ow0008_build_animation.py` -> ALL CHECKS PASSED,
`python scripts/ow0008_finalize.py` -> ALL CHECKS PASSED. Visually re-inspected the
initial and final representative frames: the looping/spiral drift pattern now fills
the frame clearly, both the tight initial 548-particle cluster and the full 3-day
spiral are legible at a glance.

### (3) implementation-origin question, answered directly

Confirmed in the reply (not just here): the actual forward-propagation pipeline
(`fwd/step.py`'s Euler integration and `u_oil = uo + 0.03*u10` formula,
`fwd/forcing.py`'s NaN-fallback decision tree, `fwd/boundary.py`'s land
classification, `fwd/hotspots.py`, `fwd/animate.py`) is this project's own
from-scratch code -- it reuses only small numerical primitives (bilinear/temporal
interpolation, Earth-distance conversion) from the pre-existing Stage III (`iste`)
module in this repo, never OpenDrift. OpenDrift/OpenOil is used exactly once, in F06,
purely as an independent third-party validation reference in a separate script under
a separate Python installation -- its output feeds only `openoil_comparison.parquet`
and never `trajectory.parquet`/`hotspot_alerts.parquet`/the animation, which are
entirely the from-scratch implementation's output.

### Status

```text
Zoom fix + drift-pattern investigation: COMPLETE
Files changed : fwd/animate.py, output/ow-0008/ow0008_forward_animation.mp4,
                output/ow-0008/frames/*.png, output/ow-0008/run_summary.json
Validation    : ow0008_build_animation.py + ow0008_finalize.py -> ALL CHECKS PASSED;
                ow-0009 view_bounds confirmed identical (4 decimal places) before/
                after the floor change -- no regression
Finding       : the ow-0008 spiral drift pattern is a real near-inertial oscillation
                (theoretical period 21.36h at this latitude matches the observed
                current rotation almost exactly) -- confirmed from raw CMEMS uo/vo,
                not a propagation bug
```

---

## OpenOil/OpenDrift reference animations (2026-09-25)

### Objective

User request: produce standalone animations of what the real OpenOil/OpenDrift F06
reference run itself looked like (not this project's own trajectory), for both
ow-0009 and ow-0008, stored separately from `output/` so there is no ambiguity about
which implementation produced which video.

### Implementation

`scripts/build_openoil_reference_animations.py` (new) reuses `fwd.animate`'s exact
rendering machinery (same basemap, hotspot-zone styling, drift trails, legend,
opening regional-overview/eased-zoom sequence as this project's own animations) but
is driven entirely by `output/<case>/openoil_reference.parquet` (OpenDrift's own
`lon`/`lat` per particle per hour) instead of this project's `trajectory.parquet`.
Hotspot alert status shown is **recomputed** against OpenDrift's own positions via
`fwd.hotspots.evaluate_hotspot_alerts` (not copied from this project's own
`hotspot_alerts.parquet`) -- coloring OpenDrift's positions with our own alert
decisions would have been inconsistent. Verified beforehand that both reference
files have `status == 0` (OpenDrift's "active" code) for every row in both cases, so
every particle could be treated as active/not-stranded throughout without further
translation.

Output stored in a new top-level `openoil_reference_animations/` directory (a
sibling of `output/`, not nested inside it), one subfolder per case, each with the
MP4 plus representative frames -- deliberately separate from `output/ow-0009/` and
`output/ow-0008/` per the user's explicit request.

### Results

```text
openoil_reference_animations/ow-0009/openoil_ow0009_reference_animation.mp4  (334 frames, 33.4s, 0.43 MB)
openoil_reference_animations/ow-0009/frames/{initial,watch_beirut_port,critical_cape_greco_mpa,final}.png
openoil_reference_animations/ow-0008/openoil_ow0008_reference_animation.mp4  (334 frames, 33.4s, 0.48 MB)
openoil_reference_animations/ow-0008/frames/{initial,watch_palm_islands_nature_reserve,critical_beirut_port,final}.png
```

**Bonus consistency check** (not the original goal, but a useful side effect):
hotspot alerts recomputed against OpenDrift's own trajectory match this project's
own alert results almost exactly in both cases -- ow-0009: same 3 hotspots triggered
(Beirut Port, Cape Greco MPA, Palm Islands Nature Reserve), Cape Greco's first-
CRITICAL hour against the OpenOil reference is 2019-04-30 18:38:41, identical to
this project's own result; ow-0008: same 2 hotspots triggered (Beirut Port,
Palm Islands Nature Reserve), both CRITICAL/WATCH from t0 in both implementations.
Visually inspected the final frame of both: nearly indistinguishable from this
project's own final frames (same S-curve path for ow-0009, same spiral for ow-0008),
consistent with the sub-100m position agreement already established in F06.

### Status

```text
OpenOil reference animations: COMPLETE
Files changed : scripts/build_openoil_reference_animations.py (new),
                openoil_reference_animations/** (new, top-level, sibling of output/)
Validation    : both cases built with all structural assertions passing (particle
                counts, timestamp counts, non-empty MP4s); visually inspected final
                frames against this project's own equivalents
```

---

## ow-0008 zoom match + hotspot status timeline CSV (2026-09-25)

### ow-0008 animation re-zoomed to match ow-0009

User follow-up: ow-0008's animation (fixed for tightness in the previous revision)
was now judged *too* zoomed in relative to ow-0009's, and asked to be matched to
ow-0009's own scale for visual comparability rather than independently
tight-cropped. Added `fwd.animate.compute_view_bounds_with_span(trajectory_df,
lon_span, lat_span)` -- a fixed-span view centered on a trajectory's own bounding-box
center -- and used it in `scripts/ow0008_build_animation.py` with ow-0009's own view
span (computed dynamically from ow-0009's `trajectory.parquet` via the normal
`compute_view_bounds`, never hard-coded). Rebuilt `output/ow-0008/`'s animation +
frames; regenerated `run_summary.json`. Visually confirmed the final frame now shows
the same absolute zoom level as ow-0009's (full coastline, both nearby hotspot
zones, and the complete spiral still legible).

`scripts/build_openoil_reference_animations.py` was updated the same way
(`build_case` gained an optional `reference_span` parameter) so the OpenOil
reference animations would stay zoom-consistent with the main ones, but the user
interrupted that rebuild mid-run before it reached ow-0008 -- **stopped via
`TaskStop`, not left to finish**. Current state:
`openoil_reference_animations/ow-0009/` is untouched/still valid (its own view never
changes, since ow-0009 is the zoom reference for both scripts); a stray incomplete
`_anim_frames_tmp/` directory left by the interrupted run was deleted (never a
deliverable, just mid-render litter); `openoil_reference_animations/ow-0008/`
**was deliberately left at its old, independently-tight zoom, not yet matching
ow-0009** -- not rebuilt per the user's explicit instruction. Flagging this
explicitly since it means the two `openoil_reference_animations/` videos are
currently at *different* zoom levels from each other (unlike the main
`output/ow-0009/`+`output/ow-0008/` pair, which are now matched) -- to be finished
only if/when asked.

### Hotspot status timeline CSV (new deliverable)

User request: one CSV per case with all 11 hotspots as rows and every recorded
timestamp as a column, holding just the status (CLEAR/WATCH/CRITICAL) -- a
wide/pivoted view of the existing long-format `hotspot_alerts.parquet`/`.csv`
(11 hotspots x 73 rows) rather than new data.

`scripts/build_hotspot_status_timeline.py` (new) pivots each case's own
`hotspot_alerts.parquet` (`hotspot_name` x `time` -> `status`), reindexes rows to
the GeoJSON's own hotspot order (not alphabetical), sorts columns ascending in
time, and validates the result (11 x 73 shape, no missing cells, every value one of
CLEAR/WATCH/CRITICAL) before writing -- it does not recompute or alter any alert
decision, only reshapes already-validated values.

```text
output/ow-0009/csv/hotspot_status_timeline.csv   (11 rows x 73 timestamp columns)
output/ow-0008/csv/hotspot_status_timeline.csv   (11 rows x 73 timestamp columns)
```

Status counts (out of 73 hours) confirm the same figures already established in
F04/F07: ow-0009 -- Beirut Port WATCH x20, Palm Islands WATCH x32, Cape Greco WATCH
x51 + CRITICAL x22, all other 8 hotspots CLEAR the entire run; ow-0008 -- Beirut Port
CRITICAL x73 (the entire run), Palm Islands WATCH x73 (the entire run), all other 9
hotspots CLEAR the entire run.

### Status

```text
Zoom-match + status-timeline CSV: COMPLETE (openoil_reference_animations/ow-0008
    intentionally left un-rebuilt, see above)
Files changed : fwd/animate.py (compute_view_bounds_with_span, new),
                scripts/ow0008_build_animation.py,
                scripts/build_openoil_reference_animations.py,
                scripts/build_hotspot_status_timeline.py (new),
                output/ow-0008/ow0008_forward_animation.mp4 + frames/*.png + run_summary.json,
                output/ow-0009/csv/hotspot_status_timeline.csv (new),
                output/ow-0008/csv/hotspot_status_timeline.csv (new)
Validation    : ow0008_build_animation.py + ow0008_finalize.py -> ALL CHECKS PASSED;
                build_hotspot_status_timeline.py -> ALL CHECKS PASSED (shape/
                completeness/value validity asserted for both cases)
```

---

## Full-physics OpenOil/OpenDrift deliverable — separate from the locked model (2026-09-26)

### Request

After the locked, from-scratch model (`output/ow-0009/`, `output/ow-0008/`) was
finalized and the user explicitly said they were "not planning to do it further" on
it, the user asked for a **new, separate** deliverable: the real OpenOil/OpenDrift
engine run end-to-end (not just used as a reference-comparison check as in F06/F08),
with weathering (evaporation, emulsification, dispersion, using each parcel's real
`mass_kg`) and horizontal diffusion enabled, beautified/maximized output, for both
ow-0009 and ow-0008, stored in a new top-level directory so it never mixes with or
modifies the locked model. Immediate follow-up constraint: no ensemble runs,
probability cones, or Monte Carlo — a single deterministic-seed run per case only,
with explicit care taken against implementation errors.

This is **not** a change to the locked model. `output/ow-0009/`, `output/ow-0008/`,
and all `fwd/*` modules are untouched. This is a full substitution of OpenDrift's own
physics engine, run independently, for exploration/comparison purposes.

### Implementation

`scripts/openoil_full_physics_run.py` (must run under `py -3.13`, the only
interpreter with `opendrift` installed) seeds real OpenOil (`opendrift==1.14.11`)
with each case's real 371/548 parcel positions and the real total observed mass
(`sum(parcel.mass_kg)` from `data/database_payload.json`), using the project's own
CMEMS/ERA5 forcing files, `wind_drift_factor=0.03` (matching the locked model's
locked wind factor), and runs the full 72h/hourly simulation with OpenOil's own
default weathering (evaporation, emulsification, dispersion; biodegradation stays
OFF by default), default current/wind uncertainty, default vertical mixing, and the
real GSHHG coastline for stranding — the one explicit override is
`environment:fallback:horizontal_diffusivity = 10.0 m2/s` (a literature coastal-ocean
ballpark, not calibrated to this region, documented as such) to get real random-walk
horizontal diffusion, since OpenOil's own default is 0.

Key API finding: OpenOil's `mass_oil` element property is `'seed': False` — it
cannot be set per-element by the caller; `seed_elements()` unconditionally
recomputes it internally from `m3_per_hour` and the oil density it derives from the
oil type at 285K. So the real total mass is seeded correctly (not the per-parcel
`mass_kg` values, which reflect image-classification confidence rather than a
physically metered quantity) via `m3_per_hour = total_mass_kg / probed_density` and
`number = N`; verified numerically before the real run that this reproduces
`total_mass_kg` to within 1 kg.

Non-determinism: weathering-driven diffusion and OpenOil's current/wind uncertainty
terms are inherently stochastic. `OpenOil(seed=20260926)` (which calls
`np.random.seed()` internally, confirmed by reading OpenDrift's source) makes this
specific run reproducible on rerun with the same seed, but it is explicitly **not**
bit-comparable to the locked model's own full determinism — documented in both the
script's own docstring and each case's `run_summary.json` (`"deterministic": false`).
This is a deliberate, single run — no ensembles, no probability cones, no Monte
Carlo, per the user's explicit instruction.

`scripts/build_openoil_full_physics_outputs.py` (normal `python3`) builds the
beautified deliverable from each case's `trajectory.parquet`: recomputes hotspot
alerts against the OpenOil trajectory (reusing `fwd.hotspots.evaluate_hotspot_alerts`
unchanged), writes `hotspot_alerts.parquet`/csv and a pivoted
`hotspot_status_timeline.csv`, and renders a two-panel animation (334 frames: 3s
regional-overview hold + 1.5s zoom-in intro + 289 propagation frames) — left panel:
map with basemap/hotspot zones/drift trails/particles colored by
`fraction_evaporated`; right panel: stacked-area oil mass budget (remaining /
evaporated / dispersed / biodegraded, as % of total seeded mass) over time.

### Bugs found and fixed during this build

1. **Matplotlib `RecursionError` after ~300 frames.** Original renderer called
   `fig.colorbar(sc, ax=ax_map, ...)` fresh every frame on a figure reused across
   the whole ~334-frame loop; each call inserts a new axes into `ax_map`'s gridspec
   without removing the previous one, so gridspec's internal position bookkeeping
   grows unboundedly every frame (confirmed via traceback pointing at the exact
   `fig.colorbar(...)` call, and via observed per-frame render time climbing from
   ~0.2s to ~6.4s before the crash). Fixed by creating one persistent `cax` axes
   once per figure (3-column gridspec: map / colorbar / budget panel) and calling
   `cax.clear(); fig.colorbar(sc, cax=cax)` every frame instead — verified via a
   10-frame smoke test (`len(fig.axes)` stayed constant, flat per-frame timing)
   before re-running the full build.
2. **Colorbar/budget-panel label overlap** (cosmetic). The colorbar's "Fraction
   evaporated" label and the budget panel's "% of initial oil mass" y-axis label sat
   too close together. Fixed by widening the colorbar gridspec column
   (`width_ratios=[2.0, 0.06, 1]`) and adding `wspace=0.55`. Verified on the actual
   production output (not just a standalone test frame) by cropping and visually
   inspecting the label region of both cases' `final.png` after the full rebuild —
   confirmed no overlap in either case.

### Validation

Ran full structural/physical sanity checks against both cases' final
`trajectory.parquet` (not just inspected that files exist):

```text
ow-0009: 73 states, 371 particles, 0 duplicate particle/time records,
         0 non-finite coordinates among active particles,
         mass balance (mass_oil+evaporated+dispersed+biodegraded) holds at every
         timestamp to within 8.8e-8 relative (float32 rounding),
         evaporated/dispersed/biodegraded totals monotonic non-decreasing at every
         timestamp (0 violations) -- correct weathering irreversibility,
         0 stranded at final hour, z stayed at 0.0 throughout (no subsurface
         entrainment materialized this run),
         implied inter-hour speed range 0.007-3.236 km/h (physically reasonable)

ow-0008: 73 states, 548 particles, 0 duplicate records, 0 non-finite coordinates,
         mass balance holds to within 7.3e-8 relative, evaporated/dispersed/
         biodegraded monotonic non-decreasing (0 violations), 0 stranded at final
         hour, z stayed 0.0, implied speed range 0.010-2.605 km/h
```

Mass budget results (`run_summary.json`, matches trajectory data exactly):

```text
ow-0009: seeded 88,715.45 kg -> final: 79,246.41 oil remaining (89.3%),
         4,577.49 evaporated (5.2%), 4,891.56 dispersed (5.5%), 0 biodegraded
ow-0008: seeded 106,833.62 kg -> final: 93,048.66 oil remaining (87.1%),
         5,028.62 evaporated (4.7%), 8,756.35 dispersed (8.2%), 0 biodegraded
```

Hotspot results (recomputed against the OpenOil trajectory, same 30%-fraction rule
as the locked model): ow-0009 triggers Beirut Port, Cape Greco MPA (CRITICAL), Palm
Islands; ow-0008 triggers Beirut Port (CRITICAL), Palm Islands — broadly consistent
with the locked model's own F04/F07 results for the same two cases, as expected
since both use the same forcing and the same wind/current drift factors.

### Status

```text
Full-physics OpenOil deliverable: COMPLETE
Files changed : scripts/openoil_full_physics_run.py (new),
                scripts/build_openoil_full_physics_outputs.py (new),
                openoil_full_physics_output/ow-0009/ (trajectory.parquet, csv/,
                    hotspot_alerts.parquet, frames/*.png, animation mp4,
                    run_summary.json),
                openoil_full_physics_output/ow-0008/ (same set)
Validation    : structural + mass-balance + monotonicity + speed-range checks above,
                run under python3 against both cases' final trajectory.parquet ->
                ALL CHECKS PASSED; visual inspection of final.png (both cases)
                confirmed correct rendering and no label overlap after the fix
Notes/blockers: non-deterministic by nature (real OpenOil weathering + diffusion +
                current/wind uncertainty), made repeatable via fixed seed 20260926;
                explicitly a single run per case, no ensembles/probability
                cones/Monte Carlo per user instruction; does not modify the locked
                model in any way
```

---

## Locked experiment

```text
Initial observation : 2019-04-28 20:35 UTC
Initial particles   : 548
Duration            : 72 hours
Propagation step    : 1 hour
Recorded states     : 73
Final time          : 2019-05-01 20:35 UTC
```

---

## Verified upstream inputs

### CMEMS

```text
File       : data/ocean.nc
Variables  : uo, vo
Coverage   : 2019-04-28 00:30 -> 2019-05-01 23:30
Grid       : 149 lat x 216 lon
```

The source contains NaNs over masked/non-ocean cells. These are preserved.

### ERA5

```text
File       : files/28th_june/wind.nc
Variables  : u10, v10
Coverage   : 2019-04-28 00:00 -> 2019-05-01 23:00
Grid       : 25 latitude x 36 longitude
Resolution : 0.25 degree
```

### Hotspots

```text
File       : regional_alert_geofences_wide.geojson
Features   : 22
Hotspots   : 11
Zones      : 100 km Watch + 40 km Critical per hotspot
```

---

## Locked physics

```text
u_oil = uo + 0.03*u10
v_oil = vo + 0.03*v10

current drift factor : 1.0
wind drift factor    : 0.03
Stokes drift         : OFF
Weathering           : OFF
Vertical processes   : OFF
Horizontal diffusion : OFF
Coastline             : STRAND / STOP
```

---

## Locked NaN policy

```text
Never globally fill CMEMS NaNs.

Land/coast -> strand permanently.
Water + valid current -> direct interpolation.
Water + NaN stencil -> 3x3/5x5 valid-ocean fallback.
Fallback -> inverse-distance weighting.
Minimum valid neighbors -> 4.
No trustworthy fallback -> explicit environmental failure.
```

---

## Locked hotspot policy

```text
40 km  -> CRITICAL -> RED
100 km -> WATCH    -> YELLOW
>100km -> CLEAR
```

Decision is based on **any active particle**, not the slick centroid.

Multiple hotspots can alert simultaneously.

---

# Task status

## F00 — Lock experiment, inputs, architecture

```text
Status: NOT STARTED
```

Implementation:

Validation/evidence:

Files changed:

Notes/blockers:

---

## F01 — Forcing adapter and interpolation pipeline

```text
Status: NOT STARTED
```

Implementation:

Validation/evidence:

Files changed:

Notes/blockers:

---

## F02 — Land classification and CMEMS NaN fallback

```text
Status: NOT STARTED
```

Implementation:

Validation/evidence:

Files changed:

Notes/blockers:

---

## F03 — OpenOil-style one-hour particle motion

```text
Status: NOT STARTED
```

Implementation:

Validation/evidence:

Files changed:

Notes/blockers:

---

## F04 — Complete 548-particle 72-hour simulation

```text
Status: NOT STARTED
```

Implementation:

Validation/evidence:

Files changed:

Notes/blockers:

---

## F05 — Hotspot alert engine

```text
Status: NOT STARTED
```

Implementation:

Validation/evidence:

Files changed:

Notes/blockers:

---

## F06 — Diagnostics and application-ready outputs

```text
Status: NOT STARTED
```

Implementation:

Validation/evidence:

Files changed:

Notes/blockers:

---

## F07 — 3-day visualization and animation

```text
Status: NOT STARTED
```

Implementation:

Validation/evidence:

Files changed:

Notes/blockers:

---

## F08 — OpenOil/OpenDrift reference comparison

```text
Status: NOT STARTED
```

Implementation:

Validation/evidence:

Files changed:

Notes/blockers:

---

## F09 — Final integration and completion

```text
Status: NOT STARTED
```

Implementation:

Validation/evidence:

Files changed:

Notes/blockers:

---

# Validation policy

No standalone test files are allowed for this section.

Use only:

```text
.py implementation/validation scripts
.ipynb scientific validation notebooks
CLI output / logged evidence
```

Do not create a `tests/` directory.

---

# Final completion record

```text
Final run command:

Final output directory:

Trajectory output:

Hotspot alert output:

Animation output:

OpenOil reference result:

Final status:
```
