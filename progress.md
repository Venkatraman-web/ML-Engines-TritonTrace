# SIH Backtracking Progress

## Current status

Planning stage. No repository scientific code has been changed by Claude Code yet.

## T00 — Repository inspection (DONE)

### Repository map (actual, as found)

```text
project-v5/
├── CLAUDE.md                        # working rules (this file governs workflow)
├── README.md                        # scientific source of truth (Stages I-VII, ISTE)
├── ISTE_Claude_Architecture_v1.0.txt  # longer-form version of the same ISTE spec
├── tasks.md                         # task list — describes a DIFFERENT, simpler RLPT plan
├── progress.md                      # this file
├── memory.md                        # long-term memory — also describes the simpler RLPT plan
└── data/
    ├── ocean.nc                     # Copernicus MEDSEA surface currents
    └── wind.nc                      # ERA5 10 m wind
```

No `.py` or `.ipynb` files exist yet. No `src/`, `notebooks/`, `tests/`, or package config
(`pyproject.toml`/`requirements.txt`) exist. Not a git repository (`git init` has not been run).

### Environment

- Python 3.14.3 (Windows).
- Installed and importable: `numpy` 2.5.1, `scipy` 1.18.1, `xarray` 2026.7.0, `pandas` 3.0.5,
  `netCDF4` 1.7.4, `h5netcdf` 1.8.1, `dask` 2026.8.0, `matplotlib` 3.11.0. This covers everything
  README.md Part 4 requires for Stage III (NumPy + SciPy), plus convenient NetCDF I/O.
- No `pytest` check performed (not needed for an inspection-only task).

### Data files — verified structure

**`data/ocean.nc`** (Copernicus MEDSEA, hourly mean surface velocity):
- dims: `time=336, latitude=149, longitude=216`
- vars: `uo`, `vo` (m/s, float32)
- time range: `2019-10-01T00:00` → `2019-10-14T23:00` (336 hourly steps, 14 days)
- latitude **ascending** (30.19 → 36.35)
- `_FillValue = 1e20`; xarray's default `mask_and_scale=True` already converts these to `NaN`
  on open (verified: raw open with `mask_and_scale=False` shows `1e20` sentinel; default open
  shows 2,974,608 / 10,813,824 values as `NaN`, i.e. land/no-data cells). So masking is available
  "for free" via `xr.open_dataset` defaults, but a loader must not silently rely on that default —
  it should be explicit per CLAUDE.md ("do not hard-code a particular NetCDF layout").

**`data/wind.nc`** (ERA5 10 m wind, via cfgrib→CDM/CF):
- dims: `valid_time=336, latitude=25, longitude=36` (also carries `expver`, `number` coords)
- vars: `u10`, `v10` (m/s, float32)
- same 336 timestamps as ocean file, and `time == valid_time` element-wise (verified equal)
- latitude **descending** (36.25 → 30.25) — opposite of ocean file
- no NaNs found in `u10`

Confirms `memory.md`'s existing "Verified environmental datasets" section; no corrections needed there.

### No existing backtracking/adjoint code

There is no code to reuse — Stage III (or any prior RLPT prototype) has not been started.

### Critical conflict found — RESOLVED 2026-09-22

Resolved: option 1 was chosen. `tasks.md` and `memory.md` have been rewritten so the task
sequence and durable-fact record target `README.md`'s Stage III (Weighted Backward Adjoint)
architecture instead of the superseded RLPT plan. See "ISTE architecture (current)" below and the
rewritten `tasks.md`/`memory.md`. Original conflict description kept for the record:

`README.md` (declared scientific source of truth) and `ISTE_Claude_Architecture_v1.0.txt` describe
a **Bayesian inverse-source pipeline**: Stage III is a *weighted* backward adjoint SDE
(`dX* = (-U + grad(K)) dtheta + sqrt(2K) dB`) with a dynamic particle weight
(`d(ln w)/dtheta = -div(U)`), a coherent forcing ensemble (`E≈50`), per-pixel/per-member
Monte Carlo transport estimates `c_hat*_{j,e}(x,t)`, and downstream Stages IV-VII
(surrogate likelihood, tempered SMC, delayed-acceptance MCMC, multi-modal posterior). Both
documents explicitly say **not** to replace this with naive deterministic backtracking, citing
ill-posed backward diffusion, flow-compression artifacts, and the need for clean-water evidence.

`tasks.md` and `memory.md`, however, currently describe a **different, simpler** engine: a
"Reverse Lagrangian Particle Tracking (RLPT)" model with deterministic transport
(`x_new = x_old - u_total*dt + random_diffusion`, no `div(U)` weight term, no `grad(K)` term, no
forcing ensemble, no observation operator/likelihood, no SMC/MCMC), producing a KDE-based
candidate-source estimate directly. This is close to the exact pattern both README.md and
`ISTE_Claude_Architecture_v1.0.txt` warn against building as a replacement methodology.

Per CLAUDE.md, README.md governs and this simplification must not be silently adopted. T00 is
inspection-only, so no code or scientific assumption has been implemented — but **T01 cannot
safely proceed** until this is resolved, since T01-T13 in `tasks.md` are written for the RLPT plan,
not for the Stage III weighted-adjoint architecture in README.md. Options (not decided here):

1. Treat `tasks.md`/`memory.md`'s RLPT sections as stale/superseded and rewrite the task list so
   T01+ build toward README.md's Stage III (weighted adjoint + forcing ensemble), or
2. Confirm the RLPT plan is an intentionally separate/earlier prototype track that README.md's
   Stage III description does not currently govern.

No file was rewritten to resolve this — flagging only, as instructed.

### Minimal implementation plan (mechanical next steps, pending conflict resolution above)

Once the architecture question above is resolved, per README.md Stage III the next tasks are:
1. A normalized environmental field interface (`U(x,t)`, `K(x,t)`, `grad(K)(x,t)`, `div(U)(x,t)`,
   `validity(x,t)`) over both grids, without hard-coding either NetCDF's layout into the equations.
2. Explicit bilinear spatial + linear temporal interpolation, handling the ascending/descending
   latitude difference and the differing ocean/wind resolutions.
3. Explicit lat/lon-degree ↔ metre conversion (never mixed directly).
4. The Euler-Maruyama backward step for `dX*` including the `grad(K)` and diffusion terms.
5. The particle weight ODE (`d ln w/dtheta = -div(U)`) computed from the same `U` used in transport.
6. Explicit land/invalid-domain boundary policy (stop-on-invalid, as README.md's prototype policy).
7. The forcing-ensemble perturbation mechanism (must be explicitly specified, not invented ad hoc).
8. Stage III diagnostics: total/valid/invalid particle counts, total weight, effective sample size.

## T01 — Environmental field loader + Stage III interface (DONE)

### Files added

- `iste/__init__.py` — package marker.
- `iste/environment.py` — reusable T01 logic:
  - `load_field_dataset(path, variables, *, time_dim, lat_dim, lon_dim, fill_threshold=1e19)`:
    generic NetCDF loader; reads only the caller-specified variables/coordinate names (so it
    works unchanged on `ocean.nc`'s `time`/`latitude`/`longitude` and `wind.nc`'s
    `valid_time`/`latitude`/`longitude`), masks values with `|raw| >= fill_threshold` to `NaN`
    explicitly (not via `xarray`'s automatic `mask_and_scale`), and never modifies the source
    file.
  - `GriddedFieldSource` / `GriddedFieldMetadata` / `VariableMetadata`: loaded data plus the
    T01 diagnostic summary (dims, timestamp range, coordinate direction, missing counts).
  - `EnvironmentalFieldProvider(ABC)`: the Stage III interface from `README.md` —
    `U(x,t)`, `K(x,t)`, `grad_K(x,t)`, `div_U(x,t)`, `validity(x,t)` — declared as abstract
    methods only. No interpolation, no derivative operators, no `K` formula, and no
    ocean/wind (windage) combination rule are implemented; a subclass cannot be instantiated
    until all five methods are implemented, so the interface cannot be silently used as if
    particle propagation already worked.
- `tests/test_environment.py` — 15 `unittest` cases (stdlib only, no new dependency): real
  `ocean.nc`/`wind.nc` loading + diagnostics, a synthetic-layout file with different
  dimension names to prove the loader is not hard-coded to either project file, explicit
  fill-value masking, missing-dimension/variable error handling, and confirmation that
  `EnvironmentalFieldProvider` (and any incomplete subclass) cannot be instantiated.
- `notebooks/T01_field_provider_check.ipynb` — manual-check artifact (see below).

### Tests

```text
python -m unittest tests.test_environment -v
Ran 15 tests in 4.504s
OK
```

### Manual check

`notebooks/T01_field_provider_check.ipynb` loads both real files through `load_field_dataset`,
prints the required diagnostics, cross-checks timestamp alignment and latitude direction, and
demonstrates that `EnvironmentalFieldProvider` cannot be instantiated. Its code cells were also
run directly as a script to confirm output (notebook itself has no cached execution outputs —
inspect by running it):

```text
time:        time  n=336  2019-10-01T00:00:00 -> 2019-10-14T23:00:00
latitude:    latitude  n=149  [30.1875, 36.3542]  ascending
longitude:   longitude  n=216  [27.1250, 36.0833]
  uo       shape=(336, 149, 216)  missing=2974608/10813824
  vo       shape=(336, 149, 216)  missing=2974608/10813824

time:        valid_time  n=336  2019-10-01T00:00:00 -> 2019-10-14T23:00:00
latitude:    latitude  n=25  [30.2500, 36.2500]  descending
longitude:   longitude  n=36  [27.2500, 36.0000]
  u10      shape=(336, 25, 36)  missing=0/302400
  v10      shape=(336, 25, 36)  missing=0/302400

timestamps equal: True
Cannot instantiate bare interface: Can't instantiate abstract class EnvironmentalFieldProvider
without an implementation for abstract methods 'K', 'U', 'div_U', 'grad_K', 'validity'
```

All values match the T00 inspection findings exactly; no discrepancy found.

### Known limitations / explicitly deferred (not implemented in T01)

- No spatial or temporal interpolation (T02).
- No `grad(K)`/`div(U)` finite-difference operators (T03).
- No concrete `K` value/formula — `README.md` gives none for Stage III v1; must be supplied
  explicitly by a later task/config.
- No decision yet on how `U` combines ocean + wind (windage) — flagged as an open item; there
  is currently no dedicated task for this decision in `tasks.md` (T02 covers interpolation,
  T05 covers the backward step that consumes an already-combined `U`) — this should be
  resolved explicitly before or during T02/T05, not assumed.
- No particle propagation, boundary/validity policy, KDE, likelihood, forcing ensemble, SMC,
  MCMC, or AIS code.

## T01.5 — Define transport velocity U (DONE)

Documentation-only task, added after T01 review. No code changed.

### Reconciled definition

`README.md` states `U(x,t)` is "the combined horizontal transport field."
`ISTE_Claude_Architecture_v1.0.txt` (Part 2, Stage III V1 Assumptions, item 1) elaborates the
same assumption: `U(x,t)` is "the horizontal ocean-surface velocity field after all selected
deterministic transport components (including any explicitly modeled windage contribution)
have been combined." Reconciled reading: the mandatory base component is the ocean-surface
current (`uo`, `vo`); windage (from `u10`, `v10`) is described as optionally "selected" and
"explicitly modeled," not automatic. No other deterministic transport component (Stokes drift,
tidal correction, Ekman spiral, etc.) is named in either document.

### Windage decision — RESOLVED 2026-09-22

**Locked for Stage III v1:**

```text
U(x,t) = (uo(x,t), vo(x,t))
```

`U` is the Copernicus ocean-surface current only; **windage is excluded from `U` in Stage III
v1**. This was decided explicitly (by the project owner) because neither `README.md` nor
`ISTE_Claude_Architecture_v1.0.txt` specifies a windage coefficient, turning angle, or
combination formula, and per `CLAUDE.md` no such formula was invented to fill that gap.

This excludes windage from the transport-velocity definition `U` only — it does **not** discard
wind data from the project. `u10`/`v10` remain loaded and verified (T01) and stay available in
`data/wind.nc`, e.g. for a possible future windage-inclusive version of `U`, or for
`README.md` Stage I's separate wind-dependent detection probability `pi_j` (an unrelated use of
wind data). T02/T05 must implement `U` against `uo`/`vo` only; introducing a windage term later
requires a new, explicit decision — not a silent addition.

### Units / sign convention

Restated from existing docs, not newly decided: `U` is in m/s (all four source variables are
m/s, verified in T00/T01); the backward SDE's drift term is `-U` under `theta = t0 - t`
(`README.md`, already recorded in `memory.md`); `U * dt` (metres) must be converted to lat/lon
degrees explicitly, never mixed directly (already a standing numerical requirement).

### Manual check

Documentation review: the definition above was checked word-for-word against `README.md`'s
Stage III v1 assumptions and `ISTE_Claude_Architecture_v1.0.txt` Part 2 item 1; the windage
question is recorded as open, not answered.

### Known limitations

No interpolation, particle propagation, `K`, `grad(K)`, `div(U)`, or forcing-ensemble work was
done in this task — only the `U` definition was locked. If a future version wants to add
windage, that requires a new explicit decision (coefficient/formula), not an extension of this
one.

## T02 — Interpolation layer (DONE)

### Files added

- `iste/interpolation.py` — purely numerical, layout-agnostic interpolation:
  - `bilinear_spatial_interpolate(lat_coords, lon_coords, field, lat_query, lon_query)`:
    bilinear interpolation of one 2-D field slice; `lat_coords` (or `lon_coords`) may be
    ascending or descending, handled identically via a shared internal bracketing helper,
    so it works unchanged on the ocean grid (ascending, 149x216) and the wind grid
    (descending, 25x36).
  - `linear_temporal_interpolate(time_coords, values, query_time)`: linear interpolation
    in time of an array whose leading axis is time (1-D or with trailing axes), for any
    query timestamp(s) — used for the hourly → sub-hourly interpolation Stage III will
    need later; no step size (e.g. 15 minutes) is hard-coded.
  - `interpolate_field(source, variable, lat, lon, time)`: the composed convenience query
    — given (lat, lon, t), returns the interpolated value from a `GriddedFieldSource`.
  - Shared validity policy: a result is `NaN`/`valid=False` whenever any of the source
    values it depends on is itself `NaN` (masked/missing) or the query point/time falls
    outside the covered range — invalid source data never silently becomes a
    valid-looking interpolated number. This is an interpolation/data-coverage validity
    signal only, distinct from the future Stage III particle boundary/land policy (T04).
- `iste/environment.py` — added `NetCDFEnvironmentalFieldProvider(EnvironmentalFieldProvider)`:
  implements `U(x,t)` and `validity(x,t)` using `interpolate_field` against the T01.5-locked
  definition `U(x,t) = (uo(x,t), vo(x,t))` (ocean current only; wind data is never read by
  this class). `K`, `grad_K`, `div_U` still raise `NotImplementedError` — no formula/value
  invented, deferred to T03 (and `K` remains an open decision).
- `tests/test_interpolation.py` — 26 `unittest` cases: hand-computable synthetic spatial
  interpolation (ascending/descending latitude, different resolutions, exact grid points,
  vectorized queries), out-of-domain and missing-cell invalidation, non-monotonic-coordinate
  rejection, hand-computable synthetic temporal interpolation (quarter/three-quarter hour,
  exact timestamp, out-of-range, missing-value propagation, vectorized queries,
  non-ascending-time rejection), the composed `interpolate_field` on a synthetic dataset,
  real-data spot checks against both files (interior-midpoint average for ocean, exact grid
  point for wind's descending-latitude grid, a real land/masked cell), and
  `NetCDFEnvironmentalFieldProvider` (`U` matches direct `uo`/`vo` interpolation, validity,
  `K`/`grad_K`/`div_U` raise `NotImplementedError`, constructor requires both `uo` and `vo`).
- `notebooks/T02_interpolation_check.ipynb` — manual-check artifact (see below).

### Tests

```text
python -m unittest tests.test_interpolation tests.test_environment -v
Ran 41 tests in 6.205s
OK
```

### Manual check

`notebooks/T02_interpolation_check.ipynb` runs the hand-computable synthetic spatial/temporal
examples, the missing-cell/out-of-domain examples, the composed synthetic space+time query,
real-data spot checks on both files, a side-by-side plot of the raw ocean grid vs. a finer
bilinearly-interpolated grid (land cells visibly blank, not filled in), and the
`NetCDFEnvironmentalFieldProvider.U`/`validity`/`K` demo. All cells were also run directly as a
script to confirm output before publishing the notebook:

```text
ascending-latitude grid:  expected 2.5, got 2.5, valid=True
descending-latitude grid: expected 2.5, got 2.5, valid=True
query touching NaN corner:    value=nan, valid=False
query away from NaN corner:   value=4.5, valid=True
query outside lat range:      value=nan, valid=False
00:15 -> 12.5 (expected 12.5) | 00:45 -> 17.5 (expected 17.5) | 01:30 -> 25.0 (expected 25.0)
combined space+time query: expected 3.75, got 3.75, valid=True
ocean interior midpoint: expected 0.009012, got 0.009012, valid=True
wind exact grid point:   expected 0.994263, got 0.994263, valid=True
land/masked ocean cell:  value=nan, valid=False
fine-grid valid fraction: 0.714 (land/out-of-domain left invalid, not interpolated)
provider.U(x, t):        [ 0.00901181 -0.0121389 ]  (matches direct uo/vo interpolation)
provider.K(x, t):        raises NotImplementedError, as intended
```

### Known limitations / explicitly deferred (not implemented in T02)

- No `grad(K)`/`div(U)` finite-difference operators (T03).
- No concrete `K` value/formula — still an open decision.
- `interpolate_field` accepts one shared scalar query time with one-or-many spatial query
  points (the pattern later stages need: many particles, one shared simulation instant);
  it does not accept a different time per spatial point. This was a deliberate scope choice
  to keep T02 "computationally lightweight" and avoid building large/irregular particle-style
  arrays — `linear_temporal_interpolate` itself is fully general (any query-time array), so
  this restriction lives only in the `interpolate_field` convenience wrapper, not the
  underlying primitives.
- No particle propagation, boundary/validity policy, KDE, likelihood, forcing ensemble, SMC,
  MCMC, or AIS code.
- No metre/degree conversion is implemented or needed here — T02 interpolates in geographic
  coordinates only, as required; physical displacement conversion belongs to the transport
  step (T05+).

## T03 — Spatial derivative operators: grad(K), div(U) (DONE)

### Files added

- `iste/geometry.py` — explicit, documented spherical-Earth degree→metre conversion
  (`MEAN_EARTH_RADIUS_M`, `metres_per_degree_latitude`, `metres_per_degree_longitude`),
  isolated in its own module since it is needed by both T03's derivatives and future T05
  particle displacement. No geodesic library (e.g. `pyproj`) is installed in this
  environment, so a documented spherical approximation (WGS84 mean radius) is used, with
  the radius passed as an explicit parameter, not a hidden constant.
- `iste/derivatives.py` — grid-level operators (one `(n_lat, n_lon)` slice at a time):
  - `spatial_gradient(lat_coords, lon_coords, field)`: physical-space gradient
    (`d_dx`, `d_dy`, `valid`) of any supplied scalar field. Chooses no value/formula for
    `K` — it operates on whatever field is passed in.
  - `divergence_2d(lat_coords, lon_coords, u_component, v_component)`: divergence of any
    supplied 2-D vector field; called with the T01.5-locked `U = (uo, vo)` — this
    function itself is generic and adds no windage or other component.
  - Both use `numpy.gradient(..., edge_order=2)` (central differences interior,
    2nd-order-accurate one-sided differences at the two edges of each axis, never
    extrapolating beyond the grid), work identically for ascending or descending
    latitude, and propagate `NaN`/invalidity to exactly the finite-difference stencil
    that touches a masked/missing cell (including the cell itself), no farther.
- `iste/environment.py`: unchanged — `NetCDFEnvironmentalFieldProvider.grad_K`/`.div_U`
  still raise `NotImplementedError` (see "known limitations" below).
- `tests/test_geometry.py` — 6 `unittest` cases for the degree→metre conversion.
- `tests/test_derivatives.py` — 15 `unittest` cases: analytically-known constant
  gradient, analytically-known spatially-varying gradient (including exact edge-column
  values), ascending-vs-descending-latitude equivalence, `NaN`-propagation to exactly the
  touched stencil cells (and no farther), input-validation errors, analytically-known
  constant divergence, an *exactly* zero-divergence field (by construction, independent
  of interior/edge accuracy), invalid-cell propagation for divergence, a signature check
  confirming `divergence_2d` has no windage/third-component parameter, and a real-ocean
  sanity check comparing `divergence_2d`'s output against an independently hand-computed
  finite difference at a real, fully-valid grid point.

### Tests

```text
python -m unittest tests.test_geometry tests.test_derivatives tests.test_environment tests.test_interpolation -v
Ran 61 tests in 6.675s
OK
```

### Manual check

`notebooks/T03_derivatives_check.ipynb` runs the analytically-checkable synthetic gradient
and divergence cases (including edge columns and an invalid-cell "+"-pattern inspection),
the ascending/descending-latitude equivalence check, a real-data `div(U)` visualization, and
the independent hand-computed real-data sanity check. All cells were also run directly as a
script first to confirm output:

```text
constant-gradient field:      d_dx max|.|=0.0 (expect 0); d_dy = 8.993216e-06 everywhere
                               (expected 1/metres_per_degree_latitude() = 8.993216e-06); valid everywhere
spatially-varying field:      max|computed - analytic| over whole grid incl. edges = 0.0; d_dy max|.|=0.0
descending-latitude grid:     same physical answer as ascending (d_dx~0, d_dy=8.993216e-06)
NaN propagation ("+" pattern around the masked cell):
  [[1 1 1 1 1]
   [1 1 0 1 1]
   [1 0 0 0 1]
   [1 1 0 1 1]
   [1 1 1 1 1]]
known constant divergence:    range [1.99999999999998, 2.00000000000002] (expected 2.0 everywhere)
exact zero-divergence field:  max|actual| = 1.6e-15 (expected 0.0)
real ocean div(U) @ t=100:    valid fraction 0.706, range [-4.9e-05, 3.3e-05] 1/s (sanity check only)
real-data hand-check:         hand-computed 1.28981e-06 vs divergence_2d 1.28980e-06 (valid=True)
```

### Known limitations / explicitly deferred (not implemented in T03)

- No concrete `K` value/formula — still an open decision (unchanged from T01/T01.5);
  `spatial_gradient` is ready to accept a `K` field once one is decided, but nothing here
  proposes one.
- `NetCDFEnvironmentalFieldProvider.grad_K(x,t)`/`.div_U(x,t)` (continuous-`(x,t)` queries,
  combining these grid-level operators with T02's interpolation, analogous to how `U(x,t)`
  was wired in T02) are **not** implemented — T03's requirements scoped grad(K) to "only
  the generic derivative machinery" on a gridded field, so this was intentionally left for
  a later task rather than added as an unrequested extra.
- The Earth-distance conversion is a spherical approximation (single mean radius), not a
  full WGS84 ellipsoidal geodesic calculation — documented in `iste/geometry.py`; no
  geodesic library is installed in this environment.
- The finite-difference grids used are the ones supplied by the caller (ocean 149x216,
  wind 25x36, or any synthetic grid) — no regridding/resampling between the two datasets'
  resolutions is implemented or needed here.
- No particle propagation, Euler-Maruyama, particle weights, boundary/trajectory logic,
  forcing ensemble, KDE, likelihood, SMC, MCMC, or AIS code.

## T04 — Boundary and particle-validity policy (DONE)

### Files added

- `iste/boundary.py`:
  - `ParticleState`: minimal vectorized particle representation (`lat`, `lon`, `valid`
    arrays) — not the full Stage III particle record (no weight/`observation_pixel_id`/
    `forcing_member_id` yet, those are later tasks). `initialize(lat0, lon0)` builds a
    fully-valid state; `valid_count`/`invalid_count`/`valid_particle_fraction` are small
    diagnostic properties for later stages to build on.
  - `apply_boundary_policy(state, proposed_lat, proposed_lon, is_valid_location)`: the
    exact `README.md` stop-on-invalid policy — an already-invalid particle is left
    completely unchanged; a still-valid particle whose proposed step is valid moves and
    stays valid; a still-valid particle whose proposed step is invalid (land or outside
    the domain) is marked invalid and does **not** move (stays at its last valid
    position). No reflection/absorption/clipping/wrapping/re-entry is implemented.
    Particles are never removed from the returned state.
  - `netcdf_validity_check(source, variable, time)`: a thin adapter that answers
    "is `(lat, lon)` valid" using `iste.interpolation.interpolate_field`'s existing T02
    validity output — this module never redefines what "valid" means.
- `tests/test_boundary.py` — 11 `unittest` cases: valid step (moves, stays valid), land
  step (invalid, stopped, position unchanged), out-of-domain step (same), an
  already-invalid particle ignoring further proposals (including across repeated
  attempts), invalid particles retained in the array (count unchanged) plus diagnostic
  properties, four particles with independent outcomes in one vectorized call, shape
  validation errors, and confirmation that `netcdf_validity_check` agrees exactly with
  `interpolate_field`'s own validity output (both on a synthetic masked cell and via the
  boundary policy end-to-end).
- `notebooks/T04_boundary_check.ipynb` — manual-check artifact (see below).

### Tests

```text
python -m unittest tests.test_boundary -v
Ran 11 tests in 0.014s
OK

python -m unittest discover -s tests -v   (full suite)
Ran 72 tests in 7.930s
OK
```

### Manual check

`notebooks/T04_boundary_check.ipynb` shows the four required before/after cases (valid
particle, particle hitting land, particle leaving the domain, already-invalid particle
attempted again), a combined multi-particle call showing invalid particles are retained
(not dropped), and one real-data example against `data/ocean.nc`. All cells were also run
directly as a script first to confirm output:

```text
1. valid step:          before (1,1) valid=True  -> after (2,2) valid=True
2. land step:            before (3,3) valid=True  -> after (3,3) valid=False  (unmoved)
3. out-of-domain step:   before (9,9) valid=True  -> after (9,9) valid=False  (unmoved)
4. already-invalid, valid-looking next proposal: stays (3,3) valid=False (unmoved, ignored)
5. combined 3-particle call: lat=[2,3,9] lon=[2,3,9] valid=[True,False,False],
   n_particles=3, valid_count=1, invalid_count=2, valid_particle_fraction=0.333
6. real ocean grid: particle proposing a step onto a real masked/land cell -> invalid,
   stayed at its starting position
```

### Known limitations / explicitly deferred (not implemented in T04)

- `apply_boundary_policy` only decides the outcome of an *already-proposed* step — it does
  not generate that proposal. Generating it via the backward SDE (`-U + grad(K)` drift,
  `sqrt(2K)` diffusion) is T05.
- No particle weights, `grad(K)`/`div(U)` usage, forcing ensemble, observation likelihood,
  KDE, SMC, MCMC, or AIS code.
- Domain/land validity is exactly T02's interpolation validity (masked cell or outside
  grid extent) — no separate coastline/bathymetry model exists.

## T05 — One Stage III backward Euler-Maruyama step (DONE)

### Files added / changed

- `iste/step.py` (new):
  - `WeightedParticleState`: T04's `ParticleState` (lat, lon, valid) plus `log_weight` —
    still not the full Stage III particle record (no `observation_pixel_id`/
    `forcing_member_id` yet).
  - `backward_euler_maruyama_step(state, t, dt_seconds, ocean_source, k_lat, k_lon,
    k_field, rng, domain_variable="uo", radius_m=...)`: exactly one step of
    `X_new = X_old + (-U + grad(K))*dt + sqrt(2*K*dt)*xi` and
    `log_w_new = log_w_old - div(U)*dt`, per particle, reusing:
    - T02's `interpolate_field` for `U = (uo, vo)` (full hourly-to-instant space+time
      interpolation) and `bilinear_spatial_interpolate` to evaluate derivative grids at a
      particle's position;
    - T03's `spatial_gradient`/`divergence_2d` for `grad(K)`/`div(U)` (`div(U)` computed
      from the same `uo`/`vo` snapshots as the transport term, blended by the same
      time-interpolation fraction as `U` — see `time_bracket` below);
    - T04's `apply_boundary_policy`/`netcdf_validity_check` unchanged, for the proposed
      destination;
    - T03's `iste.geometry` conversion (via a new `metres_to_degrees` helper) for the
      metre→degree displacement.
    Already-invalid particles are returned completely unchanged. A still-valid particle
    whose *current* position has any unavailable physics input (missing `U`, `K`,
    `grad(K)`, or `div(U)`) is marked invalid this step with position/weight frozen —
    the same "stop propagation, retain last state" policy as an invalid destination, just
    triggered by unavailable physics rather than an unavailable destination. A rejected
    destination also freezes the log-weight (the weight update is committed only together
    with an accepted move, never applied to a step that didn't happen). No renormalization
    of weights is performed.
- `iste/geometry.py`: added `metres_to_degrees(dx_m, dy_m, latitude_deg, radius_m=...)` —
  the explicit inverse of the existing degree→metre conversion (T03), evaluated at the
  displacement's starting latitude.
- `iste/interpolation.py`: added a public `time_bracket(time_coords, query_time)` —
  a thin wrapper exposing the same internal bracket logic `interpolate_field`/
  `linear_temporal_interpolate` already use, so `div(U)`'s two hourly snapshots can be
  blended with the identical fraction `U` itself uses, without duplicating that logic.
  No existing behavior changed.
- `tests/test_step.py` (new) — 13 `unittest` cases covering every item in T05's required
  list: diffusion-only (matches an independently-reconstructed exact RNG draw), correct
  backward transport direction, a hand-computable metre→degree conversion check, `grad(K)`
  drift with correct sign/magnitude (isolated via `K=0` at the query point), zero/
  convergent/divergent `div(U)` weight changes with correct sign, same-seed
  reproducibility, different-seed divergence, an invalid-destination case (frozen state,
  frozen weight), an already-invalid particle (no movement), a "physics unavailable at the
  current position" edge case, and `dt_seconds <= 0` rejection.
- `tests/test_geometry.py` — added 3 cases for `metres_to_degrees` (exact inverse of the
  per-degree factors, sign preservation, latitude-dependent vectorized behavior).
- `tests/test_interpolation.py` — added 3 cases for `time_bracket` (matches
  `linear_temporal_interpolate`'s own bracketing, out-of-range flagging, scalar-only input).
- `notebooks/T05_backward_step_check.ipynb` — manual-check artifact (see below).

### Tests

```text
python -m unittest tests.test_step -v
Ran 13 tests in 0.036s
OK

python -m unittest discover -s tests -v   (full suite)
Ran 91 tests in 7.279s
OK
```

### Manual check

`notebooks/T05_backward_step_check.ipynb` shows: a deterministic step matching the
analytic expectation exactly; a stochastic (diffusion-only) step matching an
independently-reconstructed RNG draw exactly; same-seed vs. different-seed reproducibility;
zero/convergent/divergent `div(U)` weight changes with correct sign; `grad(K)` drift moving
up-gradient with the exact analytic magnitude; an invalid-destination case (particle
stopped at a masked land cell, weight frozen); and an already-invalid particle that does
not move. All cells were also run directly as a script first to confirm exact output:

```text
1. deterministic step:  actual (20.00809389, 109.97846664) == analytic (same)
2. diffusion-only:      actual lat 19.99900772 == independently-reconstructed 19.99900772; log_w unchanged (0.0)
3. reproducibility:     same seed identical -> True; different seed differs -> True
4. weight update:       zero-div unchanged (1.5); convergent +8.613e-06; divergent -8.613e-06 (opposite signs, as expected)
5. grad(K) drift:       actual lon 110.00000824 == analytic 110.00000824
6. invalid destination: valid=False, frozen at (10.0, 110.0)
7. already invalid:     valid=False, frozen at (15.0, 105.0)
```

### Known limitations / explicitly deferred (not implemented in T05)

- One step only — no multi-step trajectory (T06), no `observation_pixel_id` tagging (T07),
  no forcing ensemble (T08), no multi-particle/multi-member run (T09), no diagnostics
  module (T10), no storage (T11), no validation suite (T12), no sign-off artifact (T13).
- `K` still has no defined physical value/formula — every `K` field used is synthetic/test
  data, supplied explicitly by the caller, exactly as required.
- `K` is treated as a single time-instant grid with no time-interpolation (only `U`/`div(U)`
  are time-interpolated between the bracketing hourly snapshots) — an explicit,
  documented simplification since `K` currently has no real time-varying source; would
  need revisiting if/when a real time-varying `K` is introduced.
- No KDE, likelihood, SMC, MCMC, or AIS code exists in this repository.

## T06 — Single-particle backward trajectory (DONE)

### Files added

- `iste/trajectory.py` (new):
  - `BackwardTrajectory`: recorded time-indexed history (`timestamps`, `lat`, `lon`,
    `log_weight`, `valid`), always length `n_steps + 1`; `first_invalid_index` reports
    where (if at all) the particle stopped.
  - `run_backward_trajectory(lat0, lon0, t0, n_steps, dt_seconds, ocean_source, k_lat,
    k_lon, k_field, rng, log_weight0=0.0, domain_variable="uo", radius_m=...)`: loops,
    calling T05's `backward_euler_maruyama_step` once per step for one particle — no
    equations are reproduced or duplicated here. The checkpoint timestamp schedule
    (`t0, t0-dt, t0-2dt, ...`) is computed directly from `n_steps`/`dt` (not accumulated
    step-by-step), so it can never drift from what `n_steps`/`dt` imply, and it does not
    depend on particle validity. Once a particle is invalid, no further T05 calls are
    made for it (its outcome would be a guaranteed no-op) — its last valid
    position/weight is carried forward unchanged into every remaining recorded state,
    still explicitly marked invalid; the array is never truncated. `n_steps` and
    `dt_seconds` are ordinary parameters — nothing hard-codes 7 days, 15 minutes, or 672
    steps.
- `tests/test_trajectory.py` (new) — 13 `unittest` cases: correct state count, exact
  backward timestamp schedule, elapsed time = `n_steps * dt`, a zero-step edge case, a
  fully deterministic trajectory matching hand-computed positions (K=0, constant
  velocity), same-seed reproducibility across a multi-step run, weight evolution (linear
  accumulation under sustained divergence; exactly constant under zero divergence across
  every step), a particle hitting land partway through (stops, frozen, still fully
  recorded), the "stays valid throughout" case, no-NaN confirmation on a fully valid run,
  and `n_steps < 0` / `dt_seconds <= 0` rejection.
- `notebooks/T06_trajectory_check.ipynb` — manual-check artifact (see below).

### Tests

```text
python -m unittest tests.test_trajectory -v
Ran 13 tests in 0.154s
OK

python -m unittest discover -s tests -v   (full suite)
Ran 104 tests in 13.705s
OK
```

### Manual check

`notebooks/T06_trajectory_check.ipynb` runs an 8-step (2-hour) trajectory on a controlled
field, prints the full checkpoint table, plots the trajectory and log-weight-over-time, then
runs a second case where the particle is pushed straight into a masked land cell and
confirms every state from the invalidation point onward is frozen and marked invalid while
the full 7-state array is still present. All cells were also run directly as a script first
to confirm output (excerpt):

```text
n_states=9 (expected 9), elapsed=7200s (expected 8*900s), all valid
step 0: 2020-01-01T06:00 lat=20.00000 lon=110.00000 log_w=0.000000 valid=True
...
step 8: 2020-01-01T04:00 lat=20.00483 lon=109.96627 log_w=-7.6e-22    valid=True

land-hitting run: n_states=7, first_invalid_index=1
valid: [True, False, False, False, False, False, False]
frozen from index 1 onward: True
```

### Known limitations / explicitly deferred (not implemented in T06)

- One particle only — no `observation_pixel_id` tagging (T07), no forcing ensemble (T08),
  no multi-particle/multi-member run (T09), no diagnostics module (T10), no storage (T11),
  no validation suite (T12), no sign-off artifact (T13).
- Inherits T05's own documented simplification: `K` is a single time-instant grid (not
  time-interpolated across the trajectory's steps), while `U`/`div(U)` are properly
  interpolated between the bracketing hourly snapshots at every step.
- No KDE, likelihood, SMC, MCMC, or AIS code exists in this repository.

## T07 — Backtracking input adapter / particle interface (PENDING_REVIEW)

### Correction (2026-09-22): the original T07 scaffold was superseded

The first version of T07 built a synthetic `ObservationFootprint` and *generated*
particles from it (`initialize_tagged_particles`). That design was corrected because it
was redundant: satellite segmentation and particle generation are owned by an **upstream
teammate module**, not by this backtracking codebase. Continuing to generate particles
here would have created a second, competing particle-generation path that could later be
confused with the teammate's real pipeline.

`iste/observation.py` was rewritten in place (not left alongside the old version) to
remove `ObservationPixel`, `ObservationFootprint`, `TaggedParticleSet`, and
`initialize_tagged_particles` entirely, and replace them with a thin **adapter** that
starts *after* upstream particles already exist. `TaggedTrajectory`/`tag_trajectory`
were kept (extended with an optional `particle_id`) since they were never part of the
generation path — they only attach an identifier to an already-computed T06 trajectory.
`tests/test_observation.py` and `notebooks/T07_observation_pixel_check.ipynb` were
likewise rewritten to match; no synthetic-footprint generation is demonstrated anywhere
in this task's artifacts any more.

### Files changed

- `iste/observation.py` (rewritten):
  - `REQUIRED_UPSTREAM_FIELDS = ("particle_id", "latitude", "longitude",
    "observation_pixel_id", "timestamp")` — the minimum contract this backtracking module
    requires from upstream; an optional `valid` field is honored if present (defaults to
    `True`).
  - `AdaptedParticleSet` + `adapt_upstream_particles(records, log_weight0=0.0)`: validates
    a sequence of upstream particle records (plain mappings, e.g. dicts — the upstream
    schema has not been formally specified yet, so no richer schema is invented) and
    repackages them into a T05 `WeightedParticleState` (unmodified) plus the preserved
    `particle_id`/`observation_pixel_id`/`timestamp` arrays. Missing required fields raise
    `KeyError` naming the record index and field(s); empty input or duplicate
    `particle_id`s raise `ValueError`.
  - `TaggedTrajectory`/`tag_trajectory(trajectory, observation_pixel_id, particle_id=None)`:
    unchanged in spirit from before, now also carries `particle_id` — attaches identifiers
    to an already-run T06 `BackwardTrajectory` (unmodified) as constant wrapper
    attributes, immune to anything that happens during propagation.
  - T05 (`iste/step.py`) and T06 (`iste/trajectory.py`) were not modified.
- `tests/test_observation.py` (rewritten) — 13 `unittest` cases: valid-payload adaptation,
  position/timestamp/particle_id/observation_pixel_id preservation, missing-required-field
  rejection (checked individually for every required field, with the error naming the
  record index and field), empty-payload and duplicate-particle_id rejection, an upstream
  `valid=False` record being preserved (not dropped) and `valid` defaulting to `True` when
  absent, one adapted particle successfully run through T06's unmodified
  `run_backward_trajectory` with both identifiers confirmed on the resulting
  `TaggedTrajectory`, and the already-invalid upstream particle case (faithfully
  represented in the adapted output; T06 is correctly not invoked for it, since T06 has no
  "start already invalid" mode and was not modified to add one).
- `notebooks/T07_observation_pixel_check.ipynb` (rewritten) — manual-check artifact (see
  below); no longer demonstrates footprint/particle generation.

### Tests

```text
python -m unittest tests.test_observation -v
Ran 13 tests in 0.036s
OK

python -m unittest discover -s tests -v   (full suite)
Ran 117 tests in 8.173s
OK
```

### Manual check

`notebooks/T07_observation_pixel_check.ipynb` defines a 4-record mock upstream payload
(one already marked `valid=False`), adapts it and prints every field, shows a
missing-field record being rejected, confirms `particle_id`/`observation_pixel_id`
preservation, runs one adapted particle through T06 and confirms both identifiers are
still attached afterward, and shows the already-invalid record is preserved without being
run through T06. All cells were also run directly as a script first to confirm output:

```text
required fields: ('particle_id', 'latitude', 'longitude', 'observation_pixel_id', 'timestamp')
4 particles adapted; p3/px-2 correctly shows valid=False (preserved, not dropped)
missing-field record correctly rejected: "upstream particle record 0 is missing required
  field(s): ['observation_pixel_id']"
particle_id / observation_pixel_id preserved: True, True
handoff through T06: particle_id=p1, observation_pixel_id=px-1, n_states=5, all valid=True
already-invalid record: particle_id=p3, observation_pixel_id=px-2, valid=False (not run through T06)
```

### Known limitations / explicitly deferred (not implemented in T07)

- No satellite segmentation, oil/clean classification, observation-footprint generation,
  particle generation, Stage I detection probability, or observation likelihood — this
  module only validates and adapts records a caller already has.
- The upstream schema has not been formally specified yet; `adapt_upstream_particles`
  covers only the smallest contract needed to drive T05/T06 (`particle_id`, `latitude`,
  `longitude`, `observation_pixel_id`, `timestamp`, optional `valid`). A richer schema
  (e.g. thickness, confidence, detection metadata) may need a follow-up once the teammate's
  actual output format is known — not invented here.
- `run_backward_trajectory` (T06) has no "start already invalid" mode; an upstream
  particle marked invalid is faithfully represented in `AdaptedParticleSet` but a caller
  must skip invoking T06 for it rather than expect T06 to accept a pre-invalid start.
- No forcing ensemble, KDE, SMC, MCMC, or AIS code.

## T08 — Forcing ensemble: decision gate (PENDING_REVIEW — investigation only, no code)

Per `CLAUDE.md`/the task instructions, T08 did **not** write forcing-ensemble code. This
section records the investigation and recommendation only; implementation is a separate,
later step gated on this decision being approved and the open parameter checklist (see
`tasks.md`'s T08 section) being resolved.

### 1. What `README.md`/`memory.md` already require

`README.md` ("Forcing uncertainty"): the full architecture uses a coherent forcing
ensemble, initial target `E ≈ 50`; perturbations represent structural uncertainty
(coherent current bias/displacement, coherent wind bias, spatially/temporally correlated
errors) and are explicitly *not* interchangeable with independent particle-level Brownian
noise; "the exact perturbation mechanism/covariance must be explicit before
implementation." `ISTE_Claude_Architecture_v1.0.txt` Part 2 adds: "Each forcing ensemble
member `e` must remain internally coherent," and the perturbation model/covariance "should
be documented." `memory.md` already records this and additionally records the T01.5-locked
decision that `U = (uo, vo)` excludes windage in Stage III v1.

### 2. Available data inspected

Re-inspected both project NetCDF files specifically for ensemble structure:

- **`data/ocean.nc`**: CMEMS `MEDSEA_MULTIYEAR_PHY_006_004`, source `"MFS E3R1I"`
  (Mediterranean Sea Physical Reanalysis INTERIM) — a **single deterministic reanalysis
  realization**. Dimensions are `time, latitude, longitude` only; no member/realization/
  ensemble dimension exists anywhere in the file or its metadata.
- **`data/wind.nc`**: ERA5 via cfgrib. Carries GRIB ensemble-metadata coordinates
  `number` and `expver`, but empirically `number` is a **scalar `0`** (not an array of
  member indices) and `expver` is uniformly `"0001"` — this is the standard placeholder
  for ERA5's **deterministic** product, not its separate ensemble-of-data-assimilations
  (EDA) product (which would carry `number` values 1–10). **Neither file contains an
  ensemble; both are single realizations.**
- No ocean-current ensemble, no atmospheric ensemble, no model-error statistics file, and
  no drifter/in-situ observation data exist anywhere in this project's `data/` directory
  or repository.

### 3. Compatibility of known external ensemble products (not downloaded; evaluated only
   for whether pursuing them would be worthwhile)

- **CMEMS `GLOBAL_MULTIYEAR_PHY_ENS_001_031` ("GREP")**: a genuine multi-member (4-member)
  global ocean reanalysis ensemble that does cover the Mediterranean and the 2019 period.
  **Rejected on resolution grounds**: it is a **monthly-mean** product (~0.25°), and
  Stage III needs hourly-to-sub-hourly forcing over a 7–14 day backward window — a monthly
  field cannot resolve the day-to-day/hour-to-hour current variability the backward
  adjoint needs, and disaggregating a monthly ensemble to hourly resolution would itself
  require inventing an unsupported temporal model. Fails requirement 3's resolution
  criterion.
- **ERA5 ensemble-of-data-assimilations (EDA, 10 members, ~0.5°, 3-hourly)**: a real,
  obtainable wind ensemble covering 2019 and the Mediterranean. **Not a solution to
  Stage III v1's forcing uncertainty as currently scoped**: `U = (uo, vo)` excludes
  windage entirely (T01.5), so perturbing wind has no effect on `U` unless/until a
  windage term is reintroduced into `U` — a separate decision not made here. Recorded as
  a real option for a *future* windage-inclusive `U`, not for the current locked
  definition.
- **No historical Mediterranean hourly ocean-current ensemble reanalysis product is known
  to the author to exist and be readily obtainable for the 2019 period at MFS-comparable
  resolution.** (This is a knowledge-based assessment, not a live catalogue search — this
  tool environment has no internet access to verify against the current CMEMS/Copernicus
  catalogue; if you have access to check the catalogue directly, that should be treated as
  more authoritative than this assessment.)
- **Drifter/in-situ observations** (e.g. CMEMS in-situ TAC, NOAA Global Drifter Program)
  are not an ensemble at all, but could in principle be used later to *calibrate* a
  surrogate perturbation's amplitude (comparing MFS reanalysis currents against observed
  drifter velocities). This is future calibration work, not a currently available dataset
  in this project, and not something to invent numbers from without actually doing that
  comparison.
- **CMEMS product QUID/validation documentation** for `MEDSEA_MULTIYEAR_PHY_006_004`
  (the exact product already in use) typically publishes independent validation
  statistics (e.g. RMSE against altimetry/drifters/moorings). This is flagged as the most
  defensible *future* source for an amplitude parameter in Option B, since it would be a
  citable, product-specific error estimate rather than a guessed number — but it has not
  been read/verified here (no internet access in this environment), so no specific number
  is proposed or assumed.

### 4. Options considered

| Option | Description | Verdict |
|---|---|---|
| A — physically-sourced ensemble | Use an existing, already-compatible ensemble product (ocean-current or, if windage were reintroduced, wind) | **Not available.** No ensemble exists in project data; no known compatible historical Mediterranean hourly current ensemble was identified; the one real ensemble product considered (ERA5 EDA) doesn't affect the locked `U`; the one real ocean ensemble considered (GREP) fails on temporal resolution. |
| B — documented lightweight surrogate ocean-current perturbation model | A statistically-parameterized, spatially/temporally correlated perturbation applied to `(uo, vo)`, with every parameter explicitly identified as an assumption and flagged for later validation | **Recommended path**, by elimination of A, and because `README.md` explicitly anticipates this as the fallback ("must be explicit before implementation," not "must come from a real ensemble product"). **Not implemented in T08** — only the decision to pursue it, and the checklist of what must still be decided, are recorded here. |

### 5. Recommended option and why

**Option B**, because Option A has no currently available or readily-identifiable
compatible instantiation for the one thing Stage III v1 actually needs perturbed —
ocean-current uncertainty in `(uo, vo)` — at the Mediterranean/2019/hourly resolution this
project requires. This is a decision by elimination, not a preference for inventing
parameters; per `README.md`, if Option B is pursued, every parameter must be treated
explicitly as an assumption, not a validated physical quantity, until it can be checked
(e.g. against the QUID error statistics or drifter comparisons noted above, in a later
task).

### 6. Explicit distinction between the three uncertainty sources (README.md; not
   interchangeable)

- **Physically sourced ensemble uncertainty (Option A):** perturbations that come from an
  actual ensemble of independently-produced current/wind realizations (e.g. multiple
  reanalysis systems, or an EDA product) — not available here (see above).
- **Surrogate/model-error uncertainty (Option B, if pursued):** a *statistically assumed*
  spatially/temporally correlated perturbation applied to the single deterministic
  `(uo, vo)` field, standing in for unknown/unquantified reanalysis model error. Each
  member `e` must be internally coherent (one smooth perturbed field, not per-cell noise).
  This has not been implemented; none of its parameters (form, covariance, amplitude, `E`,
  etc.) have been decided.
- **Particle-level Brownian diffusion (already implemented, T05):** the `sqrt(2K) dB`
  term in the backward SDE — independent per-particle random-walk noise, representing
  unresolved sub-grid turbulent diffusion, not coherent forcing bias. `README.md` is
  explicit that this is a *different* mechanism from forcing-ensemble uncertainty and one
  must not be substituted for the other.

These three are tracked as conceptually distinct in this project and must not be merged,
regardless of which forcing-ensemble option is eventually implemented.

### 7. Computational feasibility note (for the future implementation step, not decided now)

The ocean grid is `149 × 216 ≈ 32,000` cells; a naive per-member dense spatial covariance
matrix over the full grid (`32,000 × 32,000`) is infeasible to construct or factor
(Cholesky) on a normal personal computer. If Option B proceeds, the implementation should
prefer a low-rank/low-dimensional or streamable spatially-correlated random-field
generation method (e.g. a small truncated spectral/basis-function representation, or a
separable/Markov structure in space and time) over a dense full-grid covariance approach —
flagged here as a constraint on the eventual design, not a chosen method.

---

## T08 continued — proposed prototype surrogate design (2026-09-22)

> **Superseded — never implemented.** This §§8–17 streamfunction+coherent-bias design was
> the first proposal. The project owner then specified a **different, simpler** model to
> actually build (coarse-grid random noise + Gaussian spatial smoothing + Ornstein-
> Uhlenbeck temporal evolution, explicitly **not** divergence-free) — see "T08
> implementation" below for what was actually built. This section is kept only as a
> record of the design process; none of its equations are in the codebase.

**Status: proposed design only. Not implemented. Not validated. Every numeric parameter
below is explicitly labeled `SOURCED` or `ASSUMED (prototype)`.** This section refines the
recommendation above into an actual mathematical design, per your follow-up request, still
without writing any forcing-ensemble code.

### 8. Perturbation representation

For forcing member `e`, the perturbed transport velocity is:

```text
U_e(x,t) = U(x,t) + dU_e(x,t)
dU_e(x,t) = b_e(t) + curl(psi_e(x,t))            [curl(psi) = (-d(psi)/dy, d(psi)/dx)]
```

- `U(x,t) = (uo(x,t), vo(x,t))` is the existing, unmodified T01.5-locked base field.
- `b_e(t) = (b_e^u(t), b_e^v(t))` is a spatially-uniform "coherent bias/displacement" term
  — directly representing `README.md`'s "coherent current bias/displacement" bullet.
- `psi_e(x,t)` is a scalar streamfunction; `curl(psi_e)` is the standard fluid-dynamics
  streamfunction-to-velocity identity (`SOURCED` — standard vector calculus, not invented)
  and represents `README.md`'s "spatially correlated errors" bullet as coherent,
  eddy-like rotational structure rather than raw component-wise noise.

**Why a streamfunction, not independent component-wise noise (design rationale):** both
`b_e(t)` (spatially constant) and `curl(psi_e)` (a curl) are **exactly divergence-free by
construction** — `div(dU_e) = 0` identically, for any `b_e`, any `psi_e`. This means the
perturbation mechanism itself never manufactures spurious `div(U)` weight drift (T05's
weight update `log_w_new = log_w_old - div(U)*dt`) purely from the added noise — any
weight change under a perturbed member still reflects the base field's own divergence
structure, not an artifact of how the ensemble was generated. This is a deliberate design
choice favoring physical coherence over the simplest-possible implementation.

### 9. Spatial correlation structure

`psi_e(x,t) = sum_{k=1}^{M} c_{e,k}(t) * phi_k(x)`

- `phi_k(x)`: a small, truncated 2-D cosine basis over the domain's bounding box
  (`phi_{mn}(lon,lat) = cos(m*pi*(lon-lon_min)/Lx) * cos(n*pi*(lat-lat_min)/Ly)`, the
  standard Neumann eigenbasis of the Laplacian on a rectangle — `SOURCED`, standard
  spectral-method basis, not invented) with total wavenumber
  `kappa_{mn} = sqrt((m*pi/Lx)^2 + (n*pi/Ly)^2)`.
- `M` (number of modes): **`ASSUMED (prototype)`, M ~ 12–16.** A pure computational/cost
  choice (not a physical claim) — small enough to stay cheap (see §12), large enough to
  represent mesoscale-like spatial structure; not validated.
- Mode variance falls off with wavenumber: `Var[c_{e,k}] ~ exp(-kappa_k^2 * L_spatial^2 / 2)`
  — a Gaussian spectral density, which is the Fourier-domain counterpart of a Gaussian
  spatial correlation function (Wiener–Khinchin theorem — `SOURCED`, standard
  spectral-synthesis technique for generating a spatially correlated Gaussian random
  field, e.g. as used broadly in geostatistics/turbulence synthetic-field generation).
- `L_spatial` (correlation length): **`ASSUMED (prototype)`, order 10–30 km.** Reasoned
  from general oceanographic knowledge that Mediterranean Sea mesoscale
  eddies/deformation-radius scales are characteristically small (~10–20 km) compared to
  open-ocean values — this is domain-knowledge-informed, not arbitrary, but it is **not**
  a citation-verified number for this specific reanalysis product's error field, so it
  remains a prototype assumption pending validation.

### 10. Temporal correlation structure

Every stochastic coefficient (`b_e^u`, `b_e^v`, and each `c_{e,k}`) evolves independently
as a discrete Ornstein–Uhlenbeck / AR(1) process:

```text
a(t+dt) = a(t) * exp(-dt/tau) + sigma_a * sqrt(1 - exp(-2*dt/tau)) * xi,   xi ~ N(0,1)
```

This is the standard, exact discretization of a continuous-time OU process at any step
`dt` (`SOURCED` — standard stochastic-process result, not invented), giving exponential
temporal autocorrelation `exp(-|dt|/tau)` by construction — directly representing
`README.md`'s "temporally correlated errors" bullet.

- `tau` (correlation time): **`ASSUMED (prototype)`, order 0.5–2 days.** Reasoned from
  general knowledge that synoptic/mesoscale ocean-current error structures typically
  persist on a multi-hour-to-few-day timescale — again domain-knowledge-informed, not a
  verified citation for this product, so a prototype assumption. For simplicity, `v1`
  uses one shared `tau` across all modes and `b_e`, rather than a mode-dependent value
  (larger-scale modes physically tend to persist longer than small-scale ones) — flagged
  as a known simplification, not a claim that this is physically exact.
- Each mode's/bias term's own stationary marginal variance `sigma_a^2` is set so the
  *combined* field reproduces the target spatial correlation length (§9) and the target
  overall amplitude (§11) — i.e. `sigma_a` per mode is a derived quantity, not an
  independent free parameter once `M`, `L_spatial`, and `sigma_U` are fixed.

### 11. Amplitude parameterization

Target marginal standard deviation of the total perturbation velocity magnitude:
`sigma_U`. **`ASSUMED (prototype)`, order 0.05–0.10 m/s (5–10 cm/s).** This order of
magnitude is consistent with commonly reported surface-current RMSE magnitudes for
regional ocean reanalyses validated against drifters in the broader oceanographic
literature the author is aware of — **but this has not been verified against
`MEDSEA_MULTIYEAR_PHY_006_004`'s own QUID/validation documentation or against drifter data
for this specific domain/period**, so it must be treated as illustrative only, not
trusted, until that check is done (see §3, "Validation needed" below).

### 12. Number of forcing members

`E ~ 50` — **carried forward from `README.md`'s own stated "initial target," not a new
number invented here.** `README.md` itself already flags this as an "initial target," so
it inherits that same caveat; it may be revisited once compute cost (§13) and posterior
sensitivity (a later task, T12-style) are actually measured.

### 13. RNG / reproducibility

One explicit `numpy.random.Generator` per member, e.g. via
`numpy.random.SeedSequence(project_seed).spawn(E)` — giving each member `e` an
independent, reproducible stream, consistent with the explicit-RNG convention already used
throughout this project (T05/T06). `SOURCED` — standard, already-established project
practice, not a new invention.

### 14. Application to particles / coherence across particles within a member

`dU_e(x,t)` is a function of `(x,t)` and the member index `e` only — **never of particle
identity.** Every particle propagating under forcing member `e` and querying the field at
the same `(x,t)` gets the identical perturbed value, by construction — this is precisely
what "coherent" means here, and it is structurally distinct from T05's particle-level
Brownian diffusion (`sqrt(2K) dB`), which is independent per particle, per step, and must
stay untouched and separate (§6, unchanged from the original decision-gate investigation).

### 15. Computational cost estimate

Two very different costs, kept deliberately separate:

- **Cheap, precomputable part — the coefficient time series.** `M + 2` scalar OU
  processes (§10) per member, stepped along a fixed time grid (e.g. hourly, matching
  `uo`/`vo`'s native cadence — `336` hourly steps for the current 14-day dataset).
  Total size: `~16 coefficients × 336 steps × 50 members ≈ 269,000` floats
  `≈ 2 MB` — trivial to hold fully in memory for every member at once.
- **Expensive part — evaluating the grid-sized field.** Evaluating `phi_k(x)` (and its
  two analytic partial derivatives, needed for `curl`) on the full ocean grid
  (`149 × 216 ≈ 32,000` cells) costs `O(M × n_grid) ≈ 16 × 32,000 ≈ 500,000` floating-point
  multiply-adds per field evaluation — a sub-millisecond vectorized NumPy operation.
  **This should be done on demand, at query time** (mirroring how T02/T05 already query
  `uo`/`vo` on demand rather than precomputing every possible trajectory position) —
  **not** precomputed and stored for every member at every hourly step, since that would
  cost `336 hours × 50 members × 32,000 cells × 2 components × 8 bytes ≈ 8.6 GB`, which is
  not trivial on a normal personal computer. On-demand evaluation avoids this entirely at
  negligible per-call cost.
- **Net assessment:** feasible on a normal personal computer, provided the implementation
  follows the "precompute the cheap coefficients, evaluate the expensive field lazily"
  split described above rather than precomputing/storing full perturbed fields for all
  members.

### 16. Sourced vs. assumed — summary table

| Parameter | Status | Basis |
|---|---|---|
| Streamfunction/divergence-free perturbation form | `SOURCED` | standard vector calculus identity |
| Truncated 2-D cosine spatial basis | `SOURCED` | standard Laplacian eigenbasis / spectral method |
| Gaussian spectral falloff ↔ Gaussian spatial correlation | `SOURCED` | Wiener–Khinchin theorem, standard random-field synthesis |
| OU/AR(1) temporal process, exact discretization | `SOURCED` | standard stochastic-process result |
| Per-member independent RNG streams | `SOURCED` | existing project convention (T05/T06) |
| `E ≈ 50` | carried over | `README.md`'s own stated initial target, not newly invented |
| Number of spatial modes `M ~ 12–16` | `ASSUMED (prototype)` | computational-cost choice, not physical |
| Spatial correlation length `L_spatial ~ 10–30 km` | `ASSUMED (prototype)` | general Mediterranean mesoscale-eddy-scale knowledge, not a specific citation |
| Temporal correlation time `tau ~ 0.5–2 days` | `ASSUMED (prototype)` | general synoptic/mesoscale persistence knowledge, not a specific citation |
| Amplitude `sigma_U ~ 0.05–0.10 m/s` | `ASSUMED (prototype)` | general reanalysis-RMSE order of magnitude, not verified for this product |
| One shared `tau` across all modes | `ASSUMED (prototype), known simplification` | ease of v1 implementation, not a physical claim |

### 17. Validation needed before this design is trusted

- Replace `sigma_U` with a value derived from `MEDSEA_MULTIYEAR_PHY_006_004`'s own
  QUID/validation statistics, or from a direct drifter-vs-reanalysis comparison in the
  study domain/period.
- Replace `L_spatial`/`tau` with values estimated from the reanalysis field's own spatial/
  temporal autocorrelation structure (computable directly from `data/ocean.nc` itself,
  without needing an external ensemble) or from published Mediterranean mesoscale
  statistics, rather than the general order-of-magnitude reasoning used here.
- Sensitivity-test `M` and `E` (a later, T12-style task) to confirm the posterior/behavior
  is not materially sensitive to these computational-cost choices within a feasible range.
- Confirm `E ≈ 50` remains adequate once actual per-member compute cost (including the
  full Stage III backward-adjoint run per member, not just the perturbation field itself)
  is measured.

### (End of the superseded §§8–17 streamfunction proposal — not implemented.)

---

## T08 implementation (2026-09-22) — PROTOTYPE, NOT VALIDATED (DONE)

The project owner specified the model actually implemented, superseding §§8–17 above.
**Status: DONE — implemented and tested; explicitly not validated (approved 2026-09-22).**

### Model actually built

```text
U_e(x,t) = U(x,t) + delta_U_e(x,t)
```

with `U = (uo, vo)` the unchanged T01.5-locked base field (windage still excluded — this
module never reads `data/wind.nc`), and `delta_U_e` generated as:

1. independent Gaussian random values on a COARSE regular grid (never a dense spatial
   covariance matrix);
2. each coarse cell's coefficient evolved in time by an Ornstein-Uhlenbeck (OU) process
   (exact discretization: `a[i+1] = a[i]*phi + sigma*sqrt(1-phi^2)*xi`, `phi=exp(-dt/tau)`);
3. Gaussian spatial smoothing of the (time-interpolated) coarse state, applied **before**
   interpolation to any finer grid, per instruction;
4. bilinear interpolation (`iste.interpolation.bilinear_spatial_interpolate`, T02, reused
   unchanged) of the smoothed coarse field to the requested `(lat, lon)`.

**`delta_U_e` is deliberately NOT forced to be divergence-free** — confirmed by test
(`TestDivergenceIsNotForcedToZero`) that the perturbed field's `div` (via T03's
`divergence_2d`, applied directly to the actual perturbed field, exactly as instructed)
has nonzero spread, not near-zero. If `div(U_e)` is needed later (e.g. inside a per-member
Stage III weight update), it must be computed the same way — from the real perturbed
field — not assumed or precomputed by this module.

### Files added

- `iste/forcing_ensemble.py` (new):
  - `ForcingEnsembleConfig`: `n_members` (default 50, carried over from `README.md`'s own
    initial target), `spatial_correlation_km` (default 20.0, **PROTOTYPE**),
    `correlation_time_seconds` (default 86400.0 ≈ 1 day, **PROTOTYPE**), `sigma_u`
    (default 0.05 m/s, **PROTOTYPE**), `coarse_resolution_deg` (default 0.25, a
    configuration choice), `master_seed` (default 0). All fields explicit constructor
    parameters — nothing hidden; invalid values (non-positive scales, zero members, etc.)
    raise `ValueError`.
  - `ForcingEnsemble(ocean_source, config)`: builds a coarse lat/lon grid covering (with
    padding) the base ocean grid's extent; `perturbation(member_id, lat, lon, t)` returns
    `(delta_u, delta_v, valid)`; `perturbed_velocity(member_id, lat, lon, t)` returns
    `(u_e, v_e, valid) = U(x,t) + delta_U_e(x,t)`, reusing T02's `interpolate_field` for
    the base field unchanged.
  - Per-member state (`_member_state`) is generated **lazily** (only on first use of a
    given `member_id`) and **cached** — `cached_member_ids()`/`member_state_nbytes()`
    expose this for inspection/testing. Reproducibility: one
    `numpy.random.SeedSequence(master_seed).spawn(n_members)` per ensemble, with each
    member further split into independent `u`/`v` sub-streams — same seed always
    reproduces the identical member; different `member_id`/`master_seed` differ.
  - Does not import or modify `iste/step.py` (T05) or `iste/trajectory.py` (T06) at all.
- `tests/test_forcing_ensemble.py` (new) — 19 `unittest` cases: reproducibility (same
  seed/member across fresh instances, repeated queries, different members/seeds differ),
  spatial correlation observable (nearby points more correlated than distant ones across
  40 independent members; smoothing measurably reduces cell-to-cell roughness vs. raw
  noise), temporal correlation observable (short lag more correlated than long lag),
  perturbation scale same order of magnitude as `sigma_u`, memory discipline (touching one
  member never generates others; per-member size is far smaller than a naive
  full-grid-times-all-members estimate against the *real* 149×216 ocean grid; 50 members
  generated without excessive memory), config validation, `perturbed_velocity` correctly
  composes base + perturbation, and confirmation the perturbed field's divergence is not
  forced to zero.
- `notebooks/T08_forcing_ensemble_check.ipynb` (new) — manual-check artifact (see below).

### Tests

```text
python -m unittest tests.test_forcing_ensemble -v
Ran 19 tests in 0.86s
OK

python -m unittest discover -s tests -v   (full suite)
Ran 136 tests in 8.236s
OK
```

### Manual check

`notebooks/T08_forcing_ensemble_check.ipynb` (every section headed **PROTOTYPE — NOT
VALIDATED**) shows, against the real `data/ocean.nc`: unperturbed `U` vs. 3 example
perturbed members as quiver plots; a perturbation-magnitude histogram against target
`sigma_U`; an empirical spatial-correlation-vs-distance curve (40 members); an empirical
temporal-correlation-vs-lag curve (40 members); reproducibility (same config → identical,
different seed → different); and the approximate memory footprint. All cells were also run
directly as a script first to confirm output:

```text
coarse grid shape: (30, 41)  (vs. full ocean grid (149, 216))
smoothing sigma (coarse cells): 0.719
target sigma_U = 0.05 m/s, empirically observed std = 0.017 m/s (smoothing reduces it, as documented)
spatial correlation by offset (km): {0: 1.0, 10: 0.938, 20: 0.801, 40: 0.455, 80: -0.034, 160: -0.065}
temporal correlation by lag (hours): {0: 1.0, 1: 0.967, 3: 0.867, 6: 0.793, 12: 0.64, 24: 0.36, 48: -0.087}
  (24h ≈ the 1-day correlation_time target; theoretical exp(-1) = 0.368, observed 0.36 -- close match)
reproducibility: same config -> identical; different master_seed -> different
memory: one member ≈ 6.6 MB; all 50 (if all cached at once) ≈ 315 MB; vs. a naive
  full-ocean-grid x all-members x all-timestamps estimate ≈ 8.1 GB (never constructed)
```

### Known limitations / explicitly deferred (not implemented in T08)

- **PROTOTYPE, NOT VALIDATED**: `spatial_correlation_km=20`, `correlation_time_seconds≈1
  day`, and `sigma_u=0.05 m/s` remain exactly the assumptions specified by the task, not
  independently verified against QUID/drifter data — see §17 above for the validation plan.
- The empirical spatial/temporal correlation curves are approximate (smoothing +
  interpolation only approximately preserve the raw target parameters), as documented in
  `iste/forcing_ensemble.py`'s docstrings — not an exact match to 20 km / 1 day by design.
- `div(U_e)` is not computed or wired into anything here — only confirmed (by test) to be
  nonzero, as instructed; actually using it inside a per-member Stage III run is T09's job.
- No integration of forcing members into T05/T06 (that is T09's "multi-particle,
  multi-forcing-member Stage III run", explicitly not started here); T05 and T06 were not
  modified.
- `u` and `v` perturbation components are generated independently (separate RNG streams,
  no imposed cross-component correlation) — a further simplification not requested to be
  changed; a real coherent eddy field would correlate them (e.g. via a streamfunction, as
  in the superseded §§8–17 proposal), but the task explicitly specified the simpler,
  non-divergence-free model instead.
- No KDE, likelihood, SMC, MCMC, or AIS code exists in this repository.

## T09 — Small multi-particle, multi-forcing-member Stage III run (DONE)

### Configuration used (small development smoke test, not production scale)

```text
particles:            4 (mock upstream payload via T07's adapt_upstream_particles;
                          1 already marked invalid upstream)
forcing members used: 3 of 5 configured (ForcingEnsembleConfig n_members=5)
n_steps:              8
dt_seconds:           900.0  (15-minute checkpoints)
K:                     zero synthetic field (still undefined -- no formula invented)
master_seed:          42 (run_ensemble); 7 (ForcingEnsemble)
```

### How a forcing member is injected without touching T05/T06

`iste/ensemble_run.py`'s `_PerturbedComponentArray` is a small lazy stand-in for one
velocity component's time series: indexing it at time step `i` computes
`base[i] + delta_u_e(x, t_i)` (T08's `ForcingEnsemble.perturbation`, evaluated over the
*fine* ocean grid only for that one requested timestep, cached per timestep actually
touched) and returns a real NumPy array. Two of these (`uo`, `vo`) are wrapped in an
ordinary `iste.environment.GriddedFieldSource` (`make_perturbed_source`) — T05/T06 consume
it completely unmodified and unaware anything differs from a normal source. One such
source is built **once per forcing member** (not per particle) and reused for every
particle assigned to that member, so coherence (every particle under member `e` sees the
identical perturbation realization at the same query time) is a structural guarantee, not
an added check. The real, already-loaded base `uo`/`vo` arrays (T01) are referenced by the
lazy wrapper, never copied.

### Files added

- `iste/ensemble_run.py` (new):
  - `_PerturbedComponentArray` / `make_perturbed_source`: the lazy per-member injection
    mechanism described above.
  - `EnsembleRunRecords`: flat parallel arrays, one row per `(particle, forcing_member,
    checkpoint)` — `particle_id`, `forcing_member_id`, `observation_pixel_id`,
    `timestamp`, `lat`, `lon`, `log_weight` (README's "particle_weight," kept in
    log-space for the same numerical-stability reason T05/T06 already use — `weight =
    exp(log_weight)`), `valid`.
  - `run_ensemble(particles, forcing_member_ids, n_steps, dt_seconds, ocean_source,
    ensemble, k_lat, k_lon, k_field, master_seed, domain_variable="uo")`: loops
    `(particle, forcing_member)` pairs, calling T06's `run_backward_trajectory`
    unmodified for each. An upstream-invalid particle (T07) gets exactly one frozen,
    invalid row per forcing member (T06 has no "start already invalid" mode, unchanged
    from T07's documented handoff behavior) rather than being propagated or dropped.
    RNG streams are fully reproducible and independent per `(particle, forcing_member)`,
    derived from `master_seed` via nested `numpy.random.SeedSequence.spawn` (one child per
    particle, further split into one grandchild per forcing member, indexed by the
    ensemble's own member-id space so results are stable regardless of which subset of
    members is requested). Only one forcing member's perturbed-source cache and one
    particle's trajectory are held at a time — nothing precomputes/stores every member's
    full particle history simultaneously.
  - `compute_diagnostics(records)` / `compute_diagnostics_by_member(records)`: the
    minimum Stage III diagnostics (`total_particles`, `valid_particles`,
    `invalid_particles`, `total_weight`, `effective_sample_size` — Kish's ESS,
    `(sum w)^2 / sum(w^2)`, standard, not invented — `valid_particle_fraction`), computed
    at each `(particle, forcing_member)` trajectory's **final** checkpoint, overall and
    broken down per forcing member. This is explicitly the minimum needed to sanity-check
    this small run, not T10's dedicated diagnostics module.
  - Does not import or modify `iste/step.py` (T05) or `iste/trajectory.py` (T06); does not
    generate or segment particles (uses T07's `AdaptedParticleSet` as input only).
- `tests/test_ensemble_run.py` (new) — 15 `unittest` cases: the perturbed source is a lazy
  view (no array copies, cached per timestep, base+perturbation composition correct),
  record count/field-length/particle-id/pixel-id correctness, forcing-member-id
  correctness and coherence (nearby particles under the same member produce
  same-length, structurally related trajectories), an upstream-invalid particle getting
  exactly one frozen row per member, a particle hitting land mid-run staying recorded and
  frozen (never truncated), hand-computable diagnostics on a tiny synthetic record set,
  per-member diagnostics breakdown, same/different-`master_seed` reproducibility (with a
  documented note on why `K=0` alone can't distinguish seeds, since the forcing-ensemble
  instance itself is reused identically on purpose across both runs in that test),
  confirmation that a perturbed source is built once per member (not per particle), and a
  small real-`data/ocean.nc` smoke test.

### Tests

```text
python -m unittest tests.test_ensemble_run -v
Ran 15 tests in 2.2s
OK

python -m unittest discover -s tests -v   (full suite)
Ran 151 tests in 10.356s
OK
```

### Manual check

`notebooks/T09_ensemble_run_check.ipynb` runs the configuration above against the real
`data/ocean.nc`, prints every particle/forcing-member's checkpoint count and pixel id,
plots one particle's trajectory across its 3 forcing members and two nearby particles
under the same member (visibly tracking together), prints the overall and per-member
diagnostic summary, and confirms the already-invalid upstream particle is retained (not
dropped) with exactly one row per member. All cells were also run directly as a script
first to confirm output:

```text
4 particles x 3 members x 9 checkpoints
runtime: 0.479 s for 84 records
records memory: 3.2 KB
forcing-ensemble cached state (3 members): 18.92 MB
p3 (upstream-invalid): 3 rows (1 per member), all valid=False, pixel_id='px-1' preserved
overall diagnostics: total_particles=12, valid=9, invalid=3, total_weight=9.248,
  effective_sample_size=8.999, valid_particle_fraction=0.75
per-member diagnostics: each member: total=4, valid=3, invalid=1, valid_fraction=0.75
  (consistent across members, as expected for this small coherent-perturbation test)
```

### Runtime / memory

- **Runtime:** 0.48 s for 4 particles × 3 members × 9 checkpoints (84 records) on the
  real ocean grid, including on-demand generation of 3 forcing members' coarse-grid state.
- **Memory:** records themselves are a few KB (flat arrays, tiny at this scale); the 3
  used forcing members' cached coarse-grid perturbation state totals ≈19 MB (measured);
  base `uo`/`vo` arrays are referenced, not duplicated, by any of the 3 per-member
  perturbed-source views; no full perturbed grid is ever materialized for any member.

### Known limitations / explicitly deferred (not implemented in T09)

- Smoke-test scale only (4 particles, 3 of 5 configured members, 8 steps) — not `E=50` at
  production particle counts, as instructed.
- `_PerturbedComponentArray`'s per-timestep cache is unbounded within one run (bounded in
  practice by the small number of distinct timesteps a small run actually visits) — a
  production-scale run may want an explicit eviction policy; not needed or built here.
- No T10 dedicated diagnostics module (only the minimum diagnostics needed to sanity-check
  this run were computed here); no KDE, observation likelihood, SMC, MCMC, or AIS.
- `K` remains an undefined, synthetic zero field — no formula invented.
- The forcing-ensemble perturbation used throughout remains the T08 **PROTOTYPE — NOT
  VALIDATED** model (20 km / 1 day / 0.05 m/s).

## T10 — Stage III diagnostics module (DONE)

### Goal

Turn the minimum Stage III diagnostics already exercised ad hoc inside T09's
`iste.ensemble_run` into a small, reusable, independently tested module operating on
explicit `weights`/`valid` arrays — no trajectory physics, no ensemble-run logic
duplicated.

### Implementation

`iste/diagnostics.py` (new):

- `StageIIIDiagnostics` — dataclass with the six required fields (`total_particles`,
  `valid_particles`, `invalid_particles`, `total_weight`, `effective_sample_size`,
  `valid_particle_fraction`).
- `compute_stage3_diagnostics(weights, valid)` — the core function. `weights` is linear
  (i.e. `exp(log_weight)`), `valid` a same-length boolean array. Explicit contract on
  which particles contribute to each diagnostic:
  - `total_particles`/`valid_particles`/`invalid_particles` count every row by the
    `valid` flag alone.
  - `total_weight` and `effective_sample_size` (Kish's ESS, `(sum w)^2 / sum(w^2)`,
    standard, not invented) are computed from **valid particles' weights only** — an
    invalid particle's stored weight never contributes, whatever its value.
  - `valid_particle_fraction = valid_particles / total_particles`.
- Edge cases handled explicitly (not left to raise an opaque numpy error or silently
  produce a misleading number):
  - **empty input** → all counts `0`, `total_weight = effective_sample_size = 0.0`,
    `valid_particle_fraction = nan` (0/0 is genuinely undefined).
  - **all invalid** → `total_weight = effective_sample_size = 0.0`,
    `valid_particle_fraction = 0.0` (well-defined: 0 out of a known nonzero total).
  - **zero total weight** among valid particles → `effective_sample_size = 0.0`, guarded
    explicitly rather than computing `0/0`.
  - **non-finite weight** (`NaN`/`Inf`): ignored if the particle is invalid (never
    contributes regardless); **raises `ValueError`** if the particle is valid — a
    non-finite weight on a particle claimed valid indicates an upstream bug that must
    surface, not be silently absorbed into a plausible-looking diagnostic (per
    `CLAUDE.md`'s "do not silently invent").
- `compute_stage3_diagnostics_from_log_weights(log_weights, valid)` — convenience
  wrapper for callers using T05/T06/T09's log-space weight convention; converts via
  `exp` then delegates to `compute_stage3_diagnostics` (identical edge-case handling).

**Reuse, not duplication:** `iste/ensemble_run.py` (T09) was refactored so
`compute_diagnostics`/`compute_diagnostics_by_member` select each `(particle,
forcing_member)` trajectory's final-checkpoint row (`_final_row_mask`, unchanged,
ensemble-specific) and then call `compute_stage3_diagnostics_from_log_weights` instead of
a private, duplicate ESS implementation (`_diagnostics_from_final_state` was deleted).
`ensemble_run.RunDiagnostics` is now an alias of `iste.diagnostics.StageIIIDiagnostics`
so no caller-facing behavior changed. This means the ESS/edge-case arithmetic now lives
in exactly one place in the codebase.

### Tests

`tests/test_diagnostics.py` (new) — 14 `unittest` cases: all-valid with equal weights
(ESS == N), all-valid with unequal weights (hand-computed ESS = 1.6), mixed valid/invalid
(hand-computed ESS = 49/29, invalid particle's weight excluded), all-invalid, zero total
weight (both all-zero and mixed), empty input (`valid_particle_fraction` is `NaN`),
non-finite weight on an invalid particle (ignored, no error), non-finite (`NaN` and
`Inf` separately) weight on a valid particle (raises `ValueError`), mismatched
`weights`/`valid` shapes (raises `ValueError`), and the log-weight wrapper (matches the
linear-weight computation, plus an overflow case where a huge log-weight on a valid
particle raises).

```text
python -m unittest tests.test_diagnostics -v
Ran 14 tests in 0.006s
OK

python -m unittest discover -s tests -v   (full suite, including the T09 refactor)
Ran 165 tests in 12.395s
OK
```

### Manual check

`notebooks/T10_diagnostics_check.ipynb` — tiny hand-computable arrays with the expected
value worked out by hand next to the actual function output: (1) all-valid unequal
weights (`[1, 3]` → ESS = 1.6), (2) mixed validity (`[2, 3(invalid), 5]` → ESS = 49/29 ≈
1.6897, with the invalid particle's weight visibly excluded), (3) equal weights showing
the ESS formula computed by hand (`sum(w)`, `sum(w^2)`) side by side with the module's
output (ESS == N == 4), (4) all five edge cases (empty, all-invalid, zero-weight,
non-finite-on-invalid, non-finite-on-valid raising), (5) the log-weight wrapper. All
cells were also run as a standalone script first to confirm exact output before
publishing (see the numbers above, reproduced identically).

### Known limitations / explicitly deferred (not implemented in T10)

- No storage/indexing layer (`T11`), no controlled numerical validation suite (`T12`), no
  sign-off artifact (`T13`).
- No KDE, observation likelihood, SMC, MCMC, or AIS.
- The module only validates that `weights` and `valid` have matching shapes and that
  every *valid* particle's weight is finite; it does not otherwise sanity-check its
  inputs (e.g. does not require weights to be non-negative) — callers are responsible for
  passing genuine particle weights.

## T11 — Memory-efficient lookup-table storage (DONE)

### Goal

Compact storage/indexing for the per-checkpoint Stage III records T09 produces, without
duplicating environmental fields, supporting lookup by checkpoint timestamp,
`observation_pixel_id`, and `forcing_member_id`.

### Storage representation

`iste/lookup_storage.py`'s `ParticleRecordStore`:

- **Columnar (structure-of-arrays):** one Python list per required field
  (`particle_id`, `forcing_member_id`, `observation_pixel_id`, `timestamp`, `lat`, `lon`,
  `log_weight`, `valid`) while appending (cheap amortized append); one NumPy array per
  field, built lazily and cached until the next append, for reading. Only `log_weight` is
  stored — no separate linear `weight` column (requirement 8) — matching T05/T06/T09's
  own log-space convention.
- **Indices:** three plain `dict`s (`timestamp` → row indices, `observation_pixel_id` →
  row indices, `forcing_member_id` → row indices), also built lazily and cached, giving
  average O(1) lookup by key and O(k) retrieval for k matching rows.
- **No environmental-field duplication, by construction, not by discipline:** this module
  never imports `iste.environment`/`iste.forcing_ensemble` and holds nothing beyond the
  eight record fields.
- **Query results reuse T09's own type:** every lookup method returns an
  `iste.ensemble_run.EnsembleRunRecords` (T09's record type) — no new record type was
  invented, and a round trip through the store (`add_records` → `all_records()`) yields
  the same type that went in.
- **Checkpoint-timestamp indexing (requirement 6):** index keys are the actual `int64`
  nanosecond values of whatever fixed checkpoint schedule T06 generated for each
  particle — the store indexes whatever timestamps its input carries, rather than
  assuming a particular schedule.
- **Duplicate/repeated-checkpoint handling (requirement 10), explicitly defined:** the
  store is append-only and never deduplicates. Re-adding the same batch produces
  duplicate rows (both become retrievable) — deliberate, since the store cannot tell a
  legitimate new record at the same timestamp from an accidental re-add, so it does not
  silently guess either way.

### Files added

- `iste/lookup_storage.py` (new): `ParticleRecordStore` (`add_records`, `add_record`,
  `all_records`, `checkpoints`, `records_at_checkpoint`, `records_for_pixel`,
  `records_for_member`, `nbytes`), plus `naive_list_of_dicts_nbytes(records)` — a
  `sys.getsizeof`-based, explicitly-approximate estimate of what a naive "list of one
  dict per record" representation would cost, used only for the T11 memory-comparison
  benchmark (never used to represent records internally).
- `tests/test_lookup_storage.py` (new) — 20 `unittest` cases: adding records (batch and
  single), retrieving by checkpoint (including an empty result for an unmatched
  timestamp), retrieving by `observation_pixel_id` and `forcing_member_id` (including
  confirming the two never cross-contaminate), mixed valid/invalid records retrievable
  through every lookup path, exact field-by-field preservation, empty storage (all
  lookups return a correctly-typed empty `EnsembleRunRecords`), the explicit
  no-deduplication behavior on a re-added batch, a memory-footprint comparison
  (`ParticleRecordStore.nbytes()` < `naive_list_of_dicts_nbytes`) on a small T09-run
  output, and a full T09-integration round-trip test (record count, IDs/timestamps,
  positions/weights, and validity all unchanged, plus per-member lookup counts matching a
  direct mask on the original records).
- Neither `iste/ensemble_run.py` (T09) nor any other existing module was modified.

### Tests

```text
python -m unittest tests.test_lookup_storage -v
Ran 20 tests in 0.055s
OK

python -m unittest discover -s tests -v   (full suite)
Ran 185 tests in 6.518s
OK
```

### Manual check

`notebooks/T11_lookup_storage_check.ipynb`:

1. States the chosen storage representation.
2. A tiny 3-record synthetic set (two particles, one invalid) with indexing/lookup
   examples (`checkpoints()`, `records_at_checkpoint`, `records_for_pixel`,
   `records_for_member`).
3. Confirms the invalid record is retrievable through every lookup path.
4. Runs T09's small configuration against real `data/ocean.nc` (4 particles, one already
   invalid upstream, × 3 forcing members × 9 checkpoints = 84 records), stores it, and
   compares every field before vs. after the round trip.
5. Confirms the invalid particle (`p3`) round-trips as exactly 3 frozen, invalid rows.
6. Observation-pixel and forcing-member lookup counts on the same run.
7. The naive-vs-compact memory-footprint benchmark.

All cells were also run as a standalone script first to confirm exact output (including
catching and fixing a notebook-authoring mistake — see Known limitations):

```text
T09 run produced 84 records
store build time: 0.065 ms
round trip: all 8 fields match exactly (particle_id, forcing_member_id,
  observation_pixel_id, timestamp, lat, lon, log_weight, valid)
p3 rows: 3, all valid=False
pixel px-0: 54 records (p0, p1); pixel px-1: 30 records (p2, p3)
forcing_member_id 0/1/2: 28 records each
observation checkpoint (t0): 12 records (= 4 particles x 3 members)
compact store: 12180 bytes (11.89 KB); naive list-of-dicts (approx): 44072 bytes
  (43.04 KB); ratio: 3.62x
```

### Record count round-trip

`records.n_records == store.all_records().n_records` (84 == 84) confirmed on the T09
integration run, plus exact field-by-field equality (not just count) for every one of the
eight fields — see `tests/test_lookup_storage.py`'s `TestT09IntegrationRoundTrip` and the
notebook §4.

### Memory benchmark

On the 84-record T09 run above: `ParticleRecordStore.nbytes()` ≈ 12.2 KB vs.
`naive_list_of_dicts_nbytes` ≈ 43.0 KB for the same data — the compact columnar
representation used **≈3.6× less** (approximate) memory than a naive one-dict-per-record
list. Both figures are `sys.getsizeof`-based approximations (Python object accounting is
inherently approximate — no `pympler`/`tracemalloc`-based exact measurement was used —
labeled as such in the module docstring, tests, and notebook).

### Known limitations / explicitly deferred (not implemented in T11)

- No T12 (controlled validation suite) or T13 (sign-off); no KDE, likelihood, SMC, MCMC,
  or AIS.
- Demonstrated at small scale only (a handful of particles/members/checkpoints), not a
  production-scale (`E≈50`, full particle-cloud, multi-day) storage benchmark — no such
  run was attempted; no multiprocessing/GPU used anywhere.
- The store keeps its column lists/arrays entirely in memory (no on-disk/streaming
  persistence layer) — appropriate at this scale; a later production-scale task may need
  to revisit this if record counts grow far beyond what fits comfortably in memory.
- `checkpoints()` sorts ascending in real time, so its **first** entry is the *oldest*
  timestamp — i.e. the far end of the backward window, not the observation instant `t0`.
  This tripped up an early draft of the manual-check notebook (which assumed index 0 was
  `t0`); the notebook was corrected to look up `t0` directly. Any later caller wanting
  "the observation checkpoint" should look it up by its known `t0` value, not assume it is
  `checkpoints()[0]`.
- `naive_list_of_dicts_nbytes` and `ParticleRecordStore.nbytes()` are both
  `sys.getsizeof`-based approximations, not exact memory measurements; they are a fair,
  reproducible basis for the *relative* comparison this task asked for, not an absolute
  memory-usage guarantee.

## T12 — Stage III controlled numerical validation suite (DONE)

### Goal

Validate the already-implemented Stage III transport/weighting machinery against
controlled synthetic cases with analytically known expected behavior. This is
**validation, not new scientific functionality** — no second transport implementation
was created; every case calls T05/T06/T03 directly.

### Files added

- `tests/test_stage3_validation.py` (new): a single source of truth, `ALL_CASES` (12
  case-builder functions), each returning a `ValidationCase` (`category`, `name`,
  `expected`, `computed`, `tolerance`, `notes`, plus derived `error`/`passed`
  properties). `run_all_cases()` builds the list; both the `unittest` assertions in this
  file and the T12 notebook's validation table read from this exact same source, so they
  cannot silently disagree.
  - `TestControlledValidationSuite`: asserts every case passes its own stated tolerance
    (with a full expected/computed/error dump on failure, never a bare "it ran"), that
    all 7 `tasks.md`-required categories are covered, and that every case actually states
    an independent `expected` value with explanatory `notes` (not a value copied from
    `computed`).
  - `TestRealDataSanityCheck`: a single, clearly separate test against real
    `data/ocean.nc` — explicitly a sanity check only (no NaN explosion among valid
    particles, diagnostics finite and in `[0, 1]`), never used as "truth" to validate
    against (instruction 13); intentionally excluded from `ALL_CASES`/the table.
- `notebooks/T12_stage3_validation_check.ipynb` (new): manual-check artifact.
- No existing module (`iste/step.py`, `iste/trajectory.py`, `iste/derivatives.py`, etc.)
  was modified — T12 is purely additive validation.

### The 12 controlled cases (all 7 required categories + an explicit sign-convention case)

| Category | Case | Expected | Computed | Result |
|---|---|---|---|---|
| zero divergence | uniform U → log-weight unchanged | 1.5 | 1.5 | PASS |
| backward sign convention | constant eastward U → particle moves WEST by exactly `-U*dt` | 114.978467 | 114.978467 | PASS |
| known convergent flow | `div(U)<0` → log-weight increases by exactly `-div(U)*dt` | 8.613343e-06 | 8.613343e-06 | PASS |
| known divergent flow | `div(U)>0` → log-weight decreases by exactly `-div(U)*dt` | -8.613343e-06 | -8.613343e-06 | PASS |
| constant diffusivity K | `grad(K)=0` for a constant K field | 0.0 | 1.84e-20 | PASS |
| constant diffusivity K | U=0, constant K → displacement == `sqrt(2*K*dt)*xi` exactly | 115.001614 | 115.001614 | PASS |
| spatially varying diffusivity K | numerical `grad(K)` matches analytic gradient (linear K) | 9.570381e-04 | 9.570381e-04 | PASS |
| spatially varying diffusivity K | T05 step reproduces the expected `+grad(K)*dt` drift | 115.000008 | 115.000008 | PASS |
| particle validity | valid destination at every step → stays valid | 1.0 | 1.0 | PASS |
| particle validity | leaving the domain → invalid at the exact predicted step (5) | 5.0 | 5.0 | PASS |
| particle validity | invalid particle retained, not truncated (n_states unchanged) | 1.0 | 1.0 | PASS |
| particle-weight behavior | log-weight accumulates exactly linearly under constant divergence | -4.306671e-05 | -4.306671e-05 | PASS |

**How each "expected" value is derived** (never invented, never copied from the
computed value): every case uses a synthetic field simple enough to be exact by hand —
spatially uniform flow (zero divergence exactly), a field exactly linear in longitude
(constant divergence / constant gradient exactly, so even T03's finite-difference
operators are exact, not merely approximately close), or a fully deterministic
zero-noise/zero-drift composition (the pure-diffusion and land/out-of-domain cases).
Converting a known physical value (velocity, gradient, divergence) to an expected degree
displacement or weight change reuses `iste.geometry`'s already-validated (T03) metre/
degree conversion — unit arithmetic, not a competing transport model (instruction 12).

**The "particle validity — leaving the domain" case is the most notable one**: rather
than only checking *that* the particle eventually becomes invalid, its exact triggering
*step index* (5) is predicted by hand beforehand (from the known per-step displacement
and the grid's `LON.min()`) and the implementation is checked against that exact
prediction — not just a qualitative "yes it stopped somewhere."

### Tests

```text
python -m unittest tests.test_stage3_validation -v
Ran 4 tests in 1.1s
OK

python -m unittest discover -s tests -v   (full suite)
Ran 189 tests in 12.5s
OK
```

### Manual check

`notebooks/T12_stage3_validation_check.ipynb`: the full validation table (rendered via
`pandas`, imported straight from `tests.test_stage3_validation.run_all_cases()` — the
same function the automated tests call), the analytic reasoning behind each case, three
visualizations (convergent-vs-divergent weight-response bar chart; the out-of-domain
particle's longitude-vs-step trace with the predicted invalid step marked; the
weight-accumulation line vs. the computed trajectory's log-weight at each checkpoint,
confirming an exact match), and the separate, clearly-labeled real-data sanity check. All
cells were also run as a standalone script first (matplotlib `Agg` backend) to confirm
exact output before publishing.

### Numerical errors / tolerances

All 12 cases matched their analytic expectation to floating-point precision (error
`0.0` or, for the constant-K-gradient case, `1.8e-20` — numerical noise far below the
`1e-9` tolerance). Tolerances used: `1e-9` for cases with a continuous (float) expected
value (a numerical-validation budget for double-precision arithmetic on these small
synthetic fields, not a physical-precision claim), and `0.0` (exact match required) for
cases whose expected value is an exact integer/boolean (step index, valid flag,
retained-count check) — these have no floating-point rounding to budget for.

### Known limitations / explicitly deferred (not implemented in T12)

- No T13 (Stage III sign-off); no Stage IV–VII.
- Real-data checks are sanity checks only (no NaN explosion, diagnostics finite/
  in-range) — never treated as ground truth, per instruction 13.
- Land/NaN-cell invalidation (as opposed to leaving the covered lat/lon extent) is not
  re-validated here with an exact predicted trigger step — it is already covered by T04/
  T05's own dedicated unit tests (`tests/test_boundary.py`,
  `tests/test_step.py::TestInvalidDestination`); T12 instead adds the out-of-domain case
  with an exact analytically-predicted trigger step, which existing tests did not have.
- Deliberately lightweight throughout: single particles, ≤6 steps, tiny synthetic grids
  — no production `E=50` run, no large memory allocation, no multiprocessing/GPU.

## T13 — Stage III sign-off artifact (DONE — Stage III signed off 2026-09-22)

### Goal

Produce the final human-inspectable sign-off notebook for the Stage III Weighted
Backward Adjoint engine. **This is a sign-off/integration task, not a new scientific
implementation** — every computation in the notebook calls the already-approved T05–T12
components directly. The first review of the resulting notebook found one genuine
interface-integration gap (`NetCDFEnvironmentalFieldProvider`'s `K`/`grad_K`/`div_U`),
which was then fixed — see "Flagged gap ... FIXED" below; that fix touched `iste/
environment.py` and `iste/step.py` (a refactor, not an equation change) and
`tests/test_interpolation.py`.

### Files added / changed

- `notebooks/T13_stage3_signoff.ipynb` (new) — the sign-off artifact, later updated with
  a new §E once the interface gap was fixed (see below).
- `iste/environment.py` (modified, interface-integration fix): added
  `interpolated_divergence`/`interpolated_k_value_and_gradient` helpers; wired
  `NetCDFEnvironmentalFieldProvider.K`/`grad_K`/`div_U` to them; added optional
  `k_lat`/`k_lon`/`k_field` constructor parameters.
- `iste/step.py` (modified, interface-integration fix): refactored to call the same two
  new helpers instead of its own inline duplicate computation — no equation change.
- `tests/test_interpolation.py` (modified): `TestNetCDFEnvironmentalFieldProvider`
  updated with 7 new/changed tests for the fix (see below).

### What the notebook shows

- **Explicit backward-time convention** (`theta = t0 - t`), stated up front, plus a
  concrete check that a real-data trajectory's checkpoint timestamps strictly decrease.
- **A. Backward trajectories** — a 2-particle, real-`data/ocean.nc` example (8
  checkpoints, 2 forcing members) plotted with the observation instant `t0` marked, plus
  T12's controlled unambiguous-direction case (constant eastward `U` → particle moves
  west) reused directly to confirm the sign is right, not just plausible-looking.
- **B. Weight evolution** — a controlled multi-step case (spatially linear `U`, hence
  spatially constant `div(U)`) where the log-weight is shown falling exactly on the
  analytic line `-div(U)*dt*n` (max error `0.0`), plus a real-data weight-evolution plot
  labeled explicitly as a sanity check only.
- **C. Validity/boundary handling** — one particle that stays valid through every
  checkpoint, and one that becomes invalid at the exact step T12 predicted analytically
  (step 5), with an explicit check that the invalid trajectory's record length is
  unchanged (`n_states == n_steps+1`, never truncated) and every checkpoint from the
  first invalid one onward stays present and frozen.
- **D. Small lookup-table sample** — a sample of `c_hat*_{j,e}(x,t)` records from the §A
  run, retrieved via T11's `ParticleRecordStore.records_for_pixel`, showing
  `particle_id`, `forcing_member_id`, `observation_pixel_id`, `timestamp`, position, and
  `particle_weight`, explicitly labeled "Monte Carlo transport estimate, NOT an exact
  probability or proof of source." A round-trip check and T10 diagnostics on the same run
  are shown alongside it.
- **Compact summary table** — all 12 of T12's controlled cases (category, expected,
  observed, pass/fail, error/tolerance) plus 4 new T13-level integration rows
  (timestamps-decrease check, stays-valid check, invalid-retained check, lookup-table
  round-trip check) — **16/16 rows PASS**.
- **grad(K)/div(U) recap** — explicitly points back to T12's already-validated cases
  rather than recomputing a duplicate check.
- **E. Environmental field interface, now fully wired** (added after the first review) —
  `U`/`validity`/`div_U` demonstrated working with no `K` configured; `K`/`grad_K`
  demonstrated raising a clear `ValueError` with no `K` and working correctly once one is
  supplied (exact match to the analytic gradient of a linear test field); `validity`
  shown to agree with T04's own `netcdf_validity_check`; `div_U` shown to match exactly
  what T05's live `backward_euler_maruyama_step` computes for the same input.
- **Stage III Definition-of-Done checklist** — one row per `tasks.md` bullet; the
  environmental-interface row is now `DONE` (was `PARTIAL` before the fix).
- **Final interpretation section**, using the required wording verbatim: *"Stage III is a
  validated prototype transport component, not a standalone proof of source location. The
  posterior source inference is performed in later stages,"* plus explicit statements that
  no KDE/transport-concentration is computed or treated as proof of source anywhere in
  this repository, and that Stage III alone has not confirmed any spill source.

### Tests

Initially, no new `.py` module was added and no new automated tests were written (this
is an integration/sign-off notebook, not new scientific code) — the full suite passed
unchanged at 189 tests. The subsequent interface-integration fix (see "Flagged gap ...
FIXED" below) added 6 net new tests (7 new, 1 removed) to
`tests/test_interpolation.py::TestNetCDFEnvironmentalFieldProvider`:

```text
python -m unittest discover -s tests -v
Ran 195 tests in 10.6s
OK
```

Every notebook cell's logic was independently verified by running the equivalent code as
a standalone script (`matplotlib` `Agg` backend) before publishing — output matched
exactly what is shown in the notebook (36 records for the §A/D real-data run, weight
line-fit max error `0.0`, invalid trajectory `n_states=7`/`first_invalid_index=5`, exact
round trip, 16/16 summary rows PASS).

### Manual check

`notebooks/T13_stage3_signoff.ipynb` — see "What the notebook shows" above. This *is*
the manual-check artifact; per `CLAUDE.md`, T13 (and therefore Stage III as a whole) only
reaches `DONE` once you have inspected and approved it.

### Flagged gap (reported at first review) — FIXED (2026-09-22 interface-integration correction)

The first review of this notebook found: `iste.environment.NetCDFEnvironmentalFieldProvider`
(the T01/T02 concrete environmental-field-interface class) implemented `U` and `validity`,
but its `K`, `grad_K`, and `div_U` methods raised `NotImplementedError`, even though
T05/T06/T09's live pipeline already computed the same quantities internally via
`iste.derivatives.spatial_gradient`/`divergence_2d` and `iste.interpolation` directly.

**Fix applied — interface-integration correction only, no equation change:**

- Two small shared helper functions were added to `iste/environment.py`:
  - `interpolated_divergence(ocean_source, lat, lon, t, radius_m)` — `div(U)` at a
    continuous query point, via T03's `divergence_2d` on the two hourly `uo`/`vo`
    snapshots bracketing `t` (T02's `time_bracket`), linearly blended and bilinearly
    interpolated — the exact computation `iste.step` needed inline.
  - `interpolated_k_value_and_gradient(k_lat, k_lon, k_field, lat, lon, radius_m)` — `K`
    and `grad(K)` at a continuous query point, via T03's `spatial_gradient` on an
    explicitly supplied, time-invariant `K` grid plus T02's bilinear interpolation.
- `iste/step.py`'s `backward_euler_maruyama_step` (T05) was **refactored** — not
  reimplemented — to call these exact same two helpers instead of its own inline copy of
  the same bracket-and-blend / gradient-and-interpolate math. The equations themselves
  (`X_new = X_old + (-U + grad(K))*dt + sqrt(2*K*dt)*xi`, `log_w_new = log_w_old -
  div(U)*dt`) are byte-for-byte unchanged; only how the intermediate `div(U)`/`K`/`grad(K)`
  values are computed was consolidated into one shared implementation. Every existing
  T05/T06 test (hand-computed expectations, unmodified) still passes.
- `NetCDFEnvironmentalFieldProvider.__init__` gained optional `k_lat`/`k_lon`/`k_field`
  constructor parameters (all three together, or none — partial supply raises
  `ValueError` at construction). `div_U` needs only `U`, so it works with no `K`
  configured. `K`/`grad_K` now call `interpolated_k_value_and_gradient`; if no `K` grid
  was supplied, they raise a clear `ValueError` ("K was not configured for this
  provider... No default K is invented here") instead of the previous generic
  `NotImplementedError` — `K` is still never given a physical formula anywhere.
- `tests/test_interpolation.py::TestNetCDFEnvironmentalFieldProvider` was updated: the
  old `test_K_grad_K_div_U_raise_not_implemented` (which encoded the now-fixed broken
  behavior) was replaced by 7 new tests — `div_U`/`K`/`grad_K` matching direct T02/T03
  computation, `validity` matching T04's `netcdf_validity_check`, `div_U` matching what
  T05's live `backward_euler_maruyama_step` actually computes for the identical
  synthetic field already used in `tests/test_step.py`'s convergent-flow case (a direct
  regression proof the refactor didn't change T05's output), a clear-error check for
  missing `K`, and a construction-time check for partial `K` arguments.

```text
python -m unittest tests.test_interpolation -v
Ran 35 tests in 3.4s
OK

python -m unittest discover -s tests -v   (full suite, including the T05 refactor)
Ran 195 tests in 10.6s
OK
```

`notebooks/T13_stage3_signoff.ipynb` was updated (not rebuilt) with a new §E
demonstrating the provider now working end-to-end — `U`/`validity`/`div_U` with no `K`
configured, `K`/`grad_K` raising a clear error with no `K` and working once one is
supplied, and `provider.div_U` matching T05's own live step output exactly for the same
input (`-9.570380633260542e-09` vs `-9.570380633260543e-09`, matching to float64
precision) — plus an updated Definition-of-Done checklist (the environmental-interface
row is now `DONE`) and a "Previously-flagged gap — now RESOLVED" note replacing the old
flagged-gap cell.

### Known limitations / explicitly deferred (not implemented in T13)

- No Stage IV (surrogate likelihood), Stage V (tempered SMC), Stage VI (DA-MCMC), Stage
  VII (posterior output), AIS, or any new KDE/source-attribution logic — none exist
  anywhere in this repository.
- No application/frontend/service code was added.
- The forcing ensemble (T08) remains an explicitly labeled PROTOTYPE, NOT VALIDATED
  surrogate; T13's sign-off is of the Stage III *engine's correctness*, not a claim that
  T08's physical parameter values (20 km / 1 day / 0.05 m/s) are validated.
- The `NetCDFEnvironmentalFieldProvider.K`/`grad_K`/`div_U` gap flagged at the first
  review is now fixed (see above) — the environmental field interface exposes all five
  of `U`/`K`/`grad(K)`/`div(U)`/`validity` and is genuinely functional.
- Deliberately lightweight: 2 real-data particles, 2–3 forcing members, ≤8-step windows,
  tiny synthetic grids reused from T12 — no production `E=50` run, no large memory
  allocation, no multiprocessing/GPU.
- **T13 (and therefore Stage III as a whole) was `DONE` after your manual approval on
  2026-09-22** — see T13.5 immediately below for a subsequent compatibility correction
  discovered after a frozen Stage III→Stage IV spec was supplied.

## T13.5 — Stage III → Stage IV compatibility correction (DONE — approved 2026-09-22)

### Goal

`final_stage_III_IV_specification.txt` was supplied as the authoritative frozen
Stage III → Stage IV contract after T13/T14/T15 were already approved/implemented. This
task verifies the signed-off Stage III implementation actually satisfies that contract's
requirements, making the smallest possible correction where a genuine mismatch is found
— **not** a Stage III redesign, and **not** T14/T15 work.

### Files inspected

`iste/boundary.py` (T04), `iste/step.py` (T05), `iste/trajectory.py` (T06),
`iste/ensemble_run.py` (T09), `iste/lookup_storage.py` (T11), `iste/environment.py`
(K/grad_K/div_U interface), `iste/forcing_ensemble.py` (forcing-member coherence).

### Files changed

- `iste/step.py`: `WeightedParticleState` gained `death_lat`/`death_lon`/`death_reason`
  (optional, default `None` — fully backward compatible); `backward_euler_maruyama_step`
  now classifies and records the exact death event (position + reason) for any particle
  newly invalid that step, and gained an optional `divergence_source` parameter (default
  `None` = old behavior).
- `iste/trajectory.py`: `BackwardTrajectory` gained scalar
  `death_time`/`death_lat`/`death_lon`/`death_reason` fields (one death event per
  trajectory, not per checkpoint); `run_backward_trajectory` captures them exactly once,
  at the checkpoint the rejected step targeted, and gained a pass-through
  `divergence_source` parameter.
- `iste/ensemble_run.py`: `EnsembleRunRecords` gained `death_time`/`death_lat`/
  `death_lon`/`death_reason` (repeated per row within a `(particle, forcing_member)`
  group) and `survival_mark` (`int8`, varies per row, mirrors `valid`); `run_ensemble`
  now passes `divergence_source=ocean_source` (the base field) while advection still uses
  `perturbed_sources[member_id]` (`U_e`).
- `iste/lookup_storage.py`: `ParticleRecordStore` stores/round-trips the 5 new columns;
  `add_records` normalizes a legacy batch's `None` death fields to length-matched
  "not tracked" defaults; `nbytes()`/`naive_list_of_dicts_nbytes` updated to account for
  the new columns.
- `tests/test_stage3_stage4_compatibility.py` (new): 20 focused tests, Parts C/D/E/G/H.
- `notebooks/T13_5_compatibility_check.ipynb` (new): tiny manual check (death-time alive/
  dead/excluded demonstration + the div(U) vs div(U_e) demonstration) — no Stage IV
  content.
- `tasks.md`: new T13.5 entry, `PENDING_REVIEW`.

### Current Stage III compatibility status

| Area | Status before T13.5 | Status after T13.5 |
|---|---|---|
| Death metadata (`death_time`/`death_lat`/`death_lon`/`death_reason`/`survival_mark`) | **Missing entirely** — confirmed by direct code inspection, not inferred | **Added**, additive/backward-compatible |
| `K(x,t)` interface | Already correct (accepts `t`, constant-in-time by design) | Unchanged; now explicitly tested (`K(x,t1) == K(x,t2)`) |
| `div(U)` weight interface | **Direct mismatch** — used `div(U_e)` (perturbed) instead of `div(U)` (base) whenever advected through a per-member source | **Fixed** via `divergence_source` |
| Forcing-member coherence | Already correct (one coherent realization per member, reused across the whole trajectory, reproducible under `master_seed`) | Unchanged; unaffected by the `divergence_source` fix |
| T11 storage | Already round-trips the original 8 fields exactly | Now also round-trips the 5 new fields exactly |

### Exact corrections made

1. **Death schema** (Part B): additive fields as listed above; `death_time` is always the
   exact checkpoint timestamp the rejected step targeted (never a fabricated
   interpolated instant, since this integrator's checkpoint spacing IS its integration
   step size); `death_lat`/`death_lon` are the actual invalidating position (the rejected
   proposed destination for a boundary failure, or the current position for a
   physics-unavailable failure) — deliberately different from the trajectory's own
   `lat`/`lon`, which still freeze at the particle's last VALID position, per T04's
   unchanged policy. `death_reason` ∈ `{"land", "domain_exit", "physics_unavailable",
   "already_invalid_upstream"}`.
2. **`div(U)` fix** (Part E): confirmed via direct inspection (not intuition) that
   `interpolated_divergence` was always called against whatever single `ocean_source` a
   caller passed — for T09's per-member calls, that source is the perturbed `U_e`, so the
   weight update was silently computing `div(U_e)`, not `div(U)`. Verified numerically
   (see below) that the two differ meaningfully for a controlled field. Fixed with a new,
   optional `divergence_source` parameter; `run_ensemble` now passes the base field
   explicitly while advection continues to use `U_e`.

### Focused tests and actual results

```text
python -m unittest tests.test_stage3_stage4_compatibility -v
Ran 20 tests in 0.16s
OK
```

Representative numeric results (from the test run and independent verification scripts):

- Land-cell death: particle starting at `(20.0, 120.0)` with a land cell at
  `(20.0, 100.0)`, one large step → `death_reason="land"`, `death_lat=20.0`,
  `death_lon=102.0` (inside the land cell's own bilinear stencil) — while the
  trajectory's own frozen `lat`/`lon` stay at the pre-step position `(20.0, 120.0)`.
- Domain-exit death: particle walked west by an exact, hand-computed step size →
  `death_reason="domain_exit"`, `death_lat=20.0`, `death_lon=99.0` (just past
  `LON.min()=100.0`).
- Physics-unavailable death: `K` missing at the particle's own current cell →
  `death_reason="physics_unavailable"`, death position equals the CURRENT (not
  proposed) position.
- `div(U)` vs `div(U_e)`, same controlled linear field: `div(U)` (base) =
  `9.570e-09`, `div(U_e)` (perturbed) = `-1.059e-06` — provably different. After the
  fix, `backward_euler_maruyama_step`'s implied divergence (`-log_weight_change/dt`)
  when advecting through the perturbed field but with `divergence_source` set to the
  base field matches `div(U)` (`9.570e-09`) exactly, not `div(U_e)`.
- `K(x, t1) == K(x, t2)` confirmed exactly for two different real timestamps on the same
  constant-`K` prototype configuration.
- Toy temporal-death scenario (Part G/H): 10-minute checkpoints `0..60`min; particle dies
  exactly at the 50-minute checkpoint (an actual integration result, not fabricated);
  round-tripped through `ParticleRecordStore` with all death metadata preserved exactly;
  query-time comparison against the stored `death_time` alone correctly treats the
  particle as alive at 45 min, dead at 50 min, and dead after 50 min.

### Full regression result

```text
Previous full-suite count: 238 tests, all passing (before this task)
New full-suite count:      258 tests, all passing (238 pre-existing + 20 new T13.5 tests)
Failures: 0
```

Every one of the 238 pre-existing tests passes completely unmodified — the
`divergence_source`/death-metadata additions are fully backward compatible (new
dataclass fields all default to `None`/sentinels; no existing call site required a
change; the default `divergence_source=None` reproduces the exact pre-T13.5 numeric
behavior for every caller that only ever used one source).

### Remaining incompatibility (not fixed, flagged only)

- `final_stage_III_IV_specification.txt`'s Stage IV formula (Sec. 1/19/20,
  `mean_e(L_e^beta)`, beta applied PER MEMBER before marginalizing) is the **opposite
  ordering** from `README.md`'s Stage IV formula (`[mean_e(L_e)]^beta`, beta applied
  AFTER marginalizing), which is what T14's already-approved `iste.likelihood` currently
  implements. **T13.5 does not touch T14** per its explicit scope (T14 modification is
  out of scope for this task) — this is flagged here for your attention, not resolved.
  Reconciling this is Stage IV work, not a Stage III compatibility fix.
- `final_stage_III_IV_specification.txt` also introduces substantial NEW Stage IV scope
  beyond what T14/T15 implemented (`M_eff`, `rho_oil`, kernel `K_b`, the `c_hat*` Monte
  Carlo estimator formula itself, `beta = J_eff/J`, member responsibilities/ESS, the
  Stage IV query service, the backward-window/source-time prior) — none of this was
  implemented here, per T13.5's explicit "do not implement T14/T15/Stage IV" scope.

### Known limitations

- Death metadata is only as precise as this integrator's checkpoint granularity (which
  currently equals its integration step size) — `death_time` always coincides with an
  actual checkpoint, never a genuinely sub-step interpolated instant, and this was not
  changed (changing integration step size independent of checkpoint output would be new
  transport-algorithm work, out of scope here).
- `death_reason` distinguishes 4 categories; it does not further subdivide
  `physics_unavailable` (e.g. "K missing" vs "div(U) missing" are not distinguished from
  each other) — the frozen contract only asks to distinguish land/domain-exit/other.
- **T13.5 is `DONE` (approved 2026-09-22).** T14 was not modified. T15 was not modified.
  The beta-ordering conflict flagged above (README.md's `[mean_e(L_e)]^beta`, implemented
  by T14, vs. the frozen spec's `mean_e(L_e^beta)`) remains unresolved and requires an
  explicit decision before further Stage IV work proceeds.

## Project pivot notice (2026-09-22)

`README.md` and `tasks.md` were subsequently rewritten (outside this codebase's own task
turns) to replace the Stage IV surrogate-likelihood/SMC/MCMC plan with a new **Phase 2 —
Candidate generation + forward validation** design (KD-tree/KDE candidate discovery from
Stage III particle histories, followed by forward-propagation validation against the
actual satellite slick). Under the new `tasks.md`, task numbers were reassigned: what is
labeled **T14**/**T15** in the two sections immediately below (`iste/likelihood.py`'s
Stage IV surrogate likelihood, and `iste/detection_model.py`'s `h -> pi` bridge) are
**no longer the active `tasks.md` T14/T15** — the current `tasks.md` T14 is "Historical
particle slice and Phase 2 input adapter" (`iste/particle_query.py`, added below, after
this notice). The two modules below remain in the codebase (unmodified, still tested,
still passing) as a historical/possibly-reusable artifact, but are not part of the active
Phase 2 task sequence. Do not confuse the two "T14"s when reading this file's history.

## T14 (superseded numbering) — Stage IV surrogate-likelihood core (DONE)

**Stage III is signed off; this is the first Stage IV task.**

> **2026-09-22 addendum — formula corrected, description below is now historical.** The
> write-up immediately below (and the `p_tilde = [(1/E)*sum_e L_e]^beta` formula/
> "`log_mean_member_likelihood`, then `beta *`" ordering it describes) reflects
> `iste/likelihood.py` as it stood before `final_stage_III_IV_specification.txt` was
> supplied. That spec requires the opposite ordering, `p_tilde = (1/E)*sum_e(L_e^beta)`
> (`log p_tilde = logsumexp_e(beta*log_L_e) - log(E)`); `iste/likelihood.py` and
> `tests/test_likelihood.py` were corrected to this ordering (beta applied per forcing
> member before marginalizing; `Stage4LikelihoodResult` gained `tempered_log_member_
> likelihood`, `responsibilities`, and `member_ess` fields; the frozen toy example
> `log_L=[-0.1761,-5.4150], beta=0.5 -> log_p_tilde≈-0.7109, r≈[0.932,0.068],
> ESS≈1.14` is now a regression test). All 35 tests in `tests/test_likelihood.py` pass
> against the corrected formula. This correction was made under this now-superseded task
> slot before the Phase 2 pivot; it is recorded here for accuracy, not as active Phase 2
> work.

### Goal

Given a candidate source hypothesis's per-pixel, per-forcing-member detection
probabilities `pi_j,e` and the observed pixel labels `y_j`, compute how well the
candidate explains the observations, exactly per `README.md`'s Stage IV formula.

### Implementation

`iste/likelihood.py` (new):

```text
P(y_j | h_j,e) = (1-epsilon) * pi_j,e^y_j * (1-pi_j,e)^(1-y_j) + epsilon/2
L_e            = product_j P(y_j | h_j,e)
p_tilde(y | Theta,V) = [ (1/E) * sum_e L_e ]^beta
```

- `Stage4LikelihoodResult` — `log_pixel_likelihood` (`n_pixels x n_members`),
  `log_member_likelihood` (`n_members`, `= sum_j log P`), `log_mean_member_likelihood`
  (`= log mean_e(L_e)`, via stable log-mean-exp), `log_surrogate_likelihood`
  (`= beta * log_mean_member_likelihood`).
- `compute_stage4_log_likelihood(y, pi, epsilon, beta)` — the main entry point.
  Validates: `y` binary and 1-D; `pi` 2-D `(n_pixels, n_members)` with matching leading
  dimension, finite, in `[0, 1]`; `epsilon` finite and in `[0, 1]`; `beta` finite and
  `>= 0`. Computes `P` directly (vectorized over pixels x members), then
  `log_pixel_likelihood = log(P)` (an intentional, documented `-inf` when `P == 0`, e.g.
  `epsilon == 0` and a pixel flatly contradicted by a member — not an error). `log L_e`
  is a **sum** over pixels (never a product of the linear-space probabilities). The
  ensemble mean is computed via `_log_mean_exp` (shift-by-running-max log-sum-exp,
  divided by `E`), and **only then** is `beta` multiplied in — preserving the Stage IV
  critical rule (`[mean_e(L_e)]^beta`, never `mean_e(L_e^beta)`) by construction, not
  just by convention.
- This module is a **consumer**: `pi_j,e` is accepted as a given input (presumed derived
  from Stage III's `c_hat*_{j,e}(x,t)` by some other, not-yet-built step) and is never
  computed here; `y_j` is accepted as given (from the upstream Stage I system). No Stage
  III transport code was touched, and no Stage V/VI/VII/AIS logic exists.

### Tests

`tests/test_likelihood.py` (new) — 25 `unittest` cases: one-pixel/one-member hand
calculation, multiple pixels (including a `log_pixel_likelihood` sums to
`log_member_likelihood` consistency check), multiple forcing members, epsilon's floor
contribution at both extremes (`y` maximally right and maximally wrong), `beta=1`
identity, `beta` scaling linearly across several values, `beta=0` giving
`log_surrogate_likelihood == 0`, **a dedicated correct-vs-incorrect tempering-order
regression test** (hand-computed `L_e = [0.64, 0.04]`, `beta=2` — confirms the module
matches `[mean_e(L_e)]^beta = 0.1156` and differs from `mean_e(L_e^beta) = 0.2056` by
more than `0.05`; plus 20 random-case cross-checks against an independent linear-space
reference), numerical stability (2000 pixels with `pi=0.001` — the naive linear-space
product underflows to exactly `0.0` while the module stays finite; `_log_mean_exp`
directly on a wildly unbalanced `[-5, -50000, -3]` log-likelihood array; the
all-members-impossible case giving `-inf`, never `NaN`), and invalid-input rejection
(non-binary `y`, out-of-range/non-finite `pi`, invalid `epsilon`/`beta`, mismatched
ensemble dimensions, wrong array rank, empty inputs, zero forcing members).

```text
python -m unittest tests.test_likelihood -v
Ran 25 tests in 0.17s
OK

python -m unittest discover -s tests -v   (full suite)
Ran 220 tests in 12-13s
OK
```

### Manual check

`notebooks/T14_surrogate_likelihood_check.ipynb` — a tiny 3-pixel, 3-member
hand-computable example (per-pixel `P(y_j|h_j,e)` table with a hand-verified spot check,
each member's `log L_e`/`L_e`, the log ensemble mixture cross-checked against a naive
unshifted computation, the final tempered log-surrogate likelihood), an explicit
correct-vs-incorrect ordering demonstration (`[mean(L_e)]^beta = 0.5352` vs
`mean(L_e^beta) = 0.4623`, confirming the module matches the former), and two
numerical-stability examples (2000 tiny-probability pixels; a wildly unbalanced 3-member
log-likelihood array). All cells were run as a standalone script first to confirm exact
output before publishing.

### Known limitations / explicitly deferred (not implemented in T14)

- Does not compute `pi_j,e` from Stage III's lookup-table/transport-estimate output —
  that mapping (transport estimate → detection probability) is a separate, not-yet-built
  step; T14 only consumes `pi_j,e` once it exists.
- No Stage V (tempered SMC), Stage VI (Delayed-Acceptance MCMC), Stage VII (posterior/
  HPD output), or AIS — none exist anywhere in this repository.
- No per-candidate-hypothesis orchestration over a real source-space grid — T14 exposes
  one clean function; a later task will call it many times over candidate hypotheses.
- Examples are tiny synthetic arrays only (≤ 2000 pixels, ≤ 3 forcing members) — no
  production-scale run.

## T15 (superseded numbering) — Stage IV observation-model bridge, `h_j,e` → `pi_j,e` (PENDING_REVIEW)

### Goal

Provide the mapping T14 needs from a per-`(observation pixel, forcing member)`
predicted-oil quantity `h_j,e` to the detection probability `pi_j,e` its likelihood
formula consumes.

### Existing upstream investigation (performed before writing any code)

Checked, before implementing anything:

- `data/` — only `ocean.nc`/`wind.nc` (Stage III environmental fields); no satellite
  imagery, segmentation output, or detection-probability raster.
- `iste/observation.py` (T07)'s own docstring — already lists "Stage I detection
  probability (`pi_j`)" among what it explicitly does not implement/duplicate (a
  previously-documented gap, not newly found here).
- All of `iste/` — no code computes or references a numeric `pi_j`/`h_j` value, an
  oil-thickness model, or a wind-dependent detectability curve.
- `iste.likelihood` (T14) — already documented `pi_j,e` as something it *consumes*,
  confirming this bridge was deliberately left for a separate task.
- No `.json`/`.tif`/`.geojson`/`.csv` files exist anywhere in the repository that could
  hold upstream segmentation/detection output.

**Conclusion: no upstream teammate module, data file, or existing code provides
`pi_j,e` or the raw ingredients to compute it.** `ISTE_Claude_Architecture_v1.0.txt`
explicitly states the functional form of `pi_j` "must be explicitly specified in the
implementation rather than invented silently" — nothing has been specified anywhere
(by an upstream teammate or otherwise), so this task falls into requirement 4's
fallback path: implement the smallest transparent, fully configurable **prototype**
mapping, clearly labeled PROTOTYPE / NOT VALIDATED throughout.

### Implementation

`iste/detection_model.py` (new):

- `ObservationModelConfig(midpoint, steepness)` — a frozen dataclass holding the two
  PROTOTYPE/ASSUMED logistic-curve parameters; rejects non-finite values and
  `steepness <= 0` (which would make `pi` non-monotonic or constant) at construction.
- `compute_detection_probability(h, config)` — `pi(h) = sigmoid(steepness*(h -
  midpoint))`, via a numerically stable sigmoid (branch on sign of the argument, no
  overflow for any finite `h`). Elementwise: output shape exactly matches input shape
  (any shape, typically `(n_pixels, n_members)`), so pixel/forcing-member association is
  preserved purely by array position, matching how T14's own `pi` argument already
  works — no separate identifier bookkeeping needed. Deterministic (pure function, no
  randomness). Rejects non-finite `h`.
- **Deliberately has no floor/ceiling parameter** — T14's `epsilon` already supplies
  that role at the `P(y_j|h_j,e)` level; a second floor here would double it up.
- Does **not** compute `h_j,e` from Stage III's lookup table (no KDE/aggregation logic
  — `h` is accepted as a given input array); does **not** modify `iste.step`/
  `iste.trajectory` (T05/T06), `U`/`K`/`grad(K)`/`div(U)`, the forcing ensemble, or
  `iste.likelihood` (T14) — all completely untouched.

### Tests

`tests/test_detection_model.py` (new) — 18 `unittest` cases: valid input → `pi` in
`[0,1]` (1-D and 2-D shapes), a hand-computable example (`pi(midpoint)=0.5` exactly;
`pi(midpoint±ln(3)/steepness)=0.75/0.25` exactly; an asymmetric-steepness case),
monotonicity (strictly increasing across a wide `h` range; a steeper curve pushes
further from 0.5 for the same `h` offset), edge cases (extreme `h` saturates to
`0.0`/`1.0` without overflow; moderately extreme `h` stays strictly inside `(0,1)`;
scalar-shaped and empty arrays), invalid-parameter/input rejection (non-positive
steepness, non-finite config fields, non-finite `h`), deterministic output (repeated
calls identical; no hidden state across calls), pixel/member association preserved
(row/column order confirmed via an asymmetric `2x2` example), and **compatibility with
T14 proven by an actual read-only call into `compute_stage4_log_likelihood`** (not just
a shape assertion).

```text
python -m unittest tests.test_detection_model -v
Ran 18 tests in 0.15-0.23s
OK

python -m unittest discover -s tests -v   (full suite)
Ran 238 tests in 12-13s
OK
```

### Manual check

`notebooks/T15_observation_model_check.ipynb` — the upstream-model investigation
write-up, a mock `(3 pixels x 3 members)` `h_j,e` input contract, the resulting `pi_j,e`
table, a hand-computable spot check (`pi(5)=0.5`, `pi(5±ln3)=0.75/0.25`), parameter-effect
plots (varying `steepness` and `midpoint` independently), a monotonicity confirmation,
and an end-to-end compatibility check feeding T15's `pi` output directly into T14's
`compute_stage4_log_likelihood` (T14 itself untouched). All cells verified via a
standalone script (matplotlib `Agg` backend) before publishing.

### Sourced vs assumed parameters

| Parameter | Status | Notes |
|---|---|---|
| `midpoint` | **ASSUMED (PROTOTYPE)** | No physical source; a configuration value only. |
| `steepness` | **ASSUMED (PROTOTYPE)** | No physical source; must be `> 0` for monotonicity. |
| functional form (logistic) | **ASSUMED (PROTOTYPE)** | The standard, simplest monotone S-shaped probability mapping; not derived from SAR backscatter/wind-mixing physics. |

No parameter here is claimed physically validated. If/when a real upstream Stage I
detection-probability model becomes available, this module should be replaced (or given
a second, real-model-backed implementation selectable via configuration), not assumed
correct.

### Known limitations / explicitly deferred (not implemented in T15)

- Does not compute `h_j,e` from Stage III's real lookup-table output for a candidate
  source hypothesis — that mapping (transport estimate → predicted-oil quantity) is a
  separate, not-yet-built step.
- No Stage V (tempered SMC), Stage VI (Delayed-Acceptance MCMC), Stage VII (posterior/
  HPD output), or AIS — none exist anywhere in this repository.
- No satellite segmentation, oil/clean classification, or observation-pixel generation —
  those remain upstream inputs, exactly as established by T07.
- The logistic functional form itself, not just its parameters, is a prototype choice —
  a real upstream model might use a different functional family entirely.

## Phase 2 — T14 — Historical particle slice / Phase 2 input adapter (PENDING_REVIEW)

**This is the current, active `tasks.md` T14** (see the project-pivot notice above).

### Goal

Per `tasks.md`'s T14 and `README.md` Sec. 10 (steps 1-4 of 7 — kernel evaluation and
`c_hat*` summation, steps 5-7, are explicitly a later task's scope, not this one), provide
a reusable query service: given already-computed Stage III output and an arbitrary
historical timestamp `t_s` inside the backward window, return the complete `J x E`
particle state at `t_s` — interpolated position/log-weight, death-time-based exclusion,
full metadata preserved — without rerunning Stage III and without interpolating an
already-aggregated `c_hat*` field.

### Implementation

`iste/particle_query.py` (new):

- `HistoricalParticleSlice` — one row per `(particle_id, forcing_member_id)` group with a
  defined state at `query_time`: `particle_id`, `observation_pixel_id`,
  `forcing_member_id`, interpolated `lat`/`lon`/`log_weight`, `valid` (whether the
  particle has died by `query_time`), and its whole-trajectory `death_time`/`death_lat`/
  `death_lon`/`death_reason` (T13.5 metadata), carried through regardless of whether the
  death has "happened yet" relative to `query_time`.
- `query_particle_slice(records, query_time)` — the main entry point. Groups an
  `iste.ensemble_run.EnsembleRunRecords` batch by `(particle_id, forcing_member_id)`,
  then for each group:
  - **reuses `iste.interpolation.time_bracket` directly** (no second bracketing
    implementation) to locate the two bracketing checkpoints and interpolation fraction
    — works unmodified on a backward trajectory's strictly-descending checkpoint
    timestamps, exactly as T05 already uses it for `div(U)` time-blending;
  - linearly interpolates `lat`/`lon`/`log_weight` between the bracket (exact stored
    values when `query_time` matches a checkpoint exactly, since `frac` becomes `0`/`1`);
  - a group whose own stored checkpoint range does not cover `query_time` is omitted
    entirely (no data exists to interpolate from) — a single-row group (a T07
    upstream-already-invalid particle, never propagated) is a special case: it has a
    defined (frozen, invalid) state for any `query_time` at or before its one stored
    instant, and no defined state after it;
  - **death-time exclusion is a separate, final decision** (`valid = query_time >
    death_time`, using the same "forward time decreases as backward time increases"
    convention established in T13.5) applied AFTER interpolation — interpolating between
    two post-death (frozen, identical) checkpoints already reproduces the correct frozen
    position with no special-casing;
  - excluded (dead-at-`query_time`) particles are **returned with `valid=False`, never
    silently dropped**, consistent with this project's standing "never silently drop
    invalid particles" discipline (T04/T06/T09/T11).
- Never imports `iste.step`/`iste.trajectory`/`iste.forcing_ensemble`/`iste.environment`
  — cannot rerun Stage III transport even by accident (this is checked by a test that
  greps the module's own source).

### Tests

`tests/test_particle_query.py` (new) — 20 `unittest` cases against hand-built synthetic
checkpoint data (2 observation pixels, 2 forcing members, 4 particles: one that stays
valid throughout, one that dies exactly at a checkpoint with a distinct death position,
one independent valid one, and one T07 upstream-already-invalid single-row particle):
interpolation at a checkpoint midpoint and at a non-midpoint fraction (both hand-computed
exactly); exact-checkpoint queries (including at `t0` itself); death-time exclusion
(alive strictly before death, dead exactly at death, dead after death, a never-died
particle always valid, the upstream-invalid particle excluded from `t0` onward);
metadata preservation (identifiers, death metadata exact match, death metadata visible
even while the particle is still alive, `query_time` recorded on the result);
multiple-pixel/multiple-member association correctness (no cross-contamination);
out-of-window queries (excluded, not fabricated); no-Stage-III-module-import and
call-repeatability (no hidden state) checks; and an empty-input case.

```text
python -m unittest tests.test_particle_query -v
Ran 20 tests in 0.01-0.09s
OK

python -m unittest discover -s tests -v   (full suite)
Ran 288 tests in ~13.6s
OK
```

### Known limitations / explicitly deferred (not implemented in this T14)

- Does not evaluate the fixed kernel `K_b` or sum weighted contributions into `c_hat*`
  (`README.md` Sec. 10 steps 5-7) — that is explicitly a later Phase 2 task's scope,
  consuming this module's `HistoricalParticleSlice` output.
- Does not build any KD-tree, KDE, support heat map, candidate peak/core, or forward
  transport — none of Phase 2's later tasks (T15+ in the new numbering) were touched.
- The old, now-superseded-numbering `iste/likelihood.py` (Stage IV surrogate likelihood)
  and `iste/detection_model.py` (`h -> pi`) were not modified by this task.

## Phase 2 — T15 — KD-tree neighborhood engine (PENDING_REVIEW)

### Goal

Per `tasks.md`'s T15 and `README.md` Sec. 13, provide a reusable spatial-neighborhood
acceleration structure over one T14 `HistoricalParticleSlice`'s valid particles, with
distance-safe local/projected coordinates and configurable fixed-radius / k-nearest-
neighbor queries, returning complete particle metadata. No KDE, support scoring,
candidate extraction, or source inference — acceleration structure only.

### Implementation

`iste/kdtree_neighborhood.py` (new):

- `project_to_local_metres(lat, lon, lat_ref, lon_ref, radius_m)` — an equirectangular
  local tangent-plane projection to `(x_m, y_m)` metres, reusing `iste.geometry`'s
  already-validated `metres_per_degree_latitude`/`metres_per_degree_longitude` (T03) —
  never treating degrees directly as metres, no new/invented projection.
- `ParticleNeighborhoodIndex` / `build_particle_neighborhood_index(slice_, lat_ref=None,
  lon_ref=None, radius_m=...)` — builds a `scipy.spatial.cKDTree` (an existing project
  dependency, not a reimplemented KD-tree) over the slice's VALID particles only
  (`README.md`: "build a KD-tree over the valid particle positions"); `lat_ref`/`lon_ref`
  default to the valid particles' own mean position (explicit, documented, overridable).
- `query_radius(index, lat, lon, radius_m)` / `query_knn(index, lat, lon, k)` — both
  return a `NeighborhoodResult` with the full T14 metadata (`particle_id`,
  `observation_pixel_id`, `forcing_member_id`, `lat`/`lon`/`log_weight`, `valid`, and all
  4 death-metadata fields) for every match, **sorted by ascending distance (ties broken
  by `particle_id`)** for deterministic, reproducible ordering regardless of the
  underlying tree's internal (unordered) result order. `query_knn` never fabricates
  extra results when fewer than `k` particles exist.

### Tests

`tests/test_kdtree_neighborhood.py` (new) — 20 `unittest` cases: projected-coordinate
correctness (reference point → origin, 1-degree lat/lon offsets matching `iste.geometry`
exactly, longitude metres shrinking at higher latitude, a sanity check that degrees are
never used directly as metres); radius queries matching an independently-written brute-
force reference (membership, order, and exact distances) including a rejected
zero/negative radius and an empty-result case; kNN matching brute-force top-`k` exactly,
returning fewer than `k` when fewer are available (never fabricated), and a rejected
`k<1`; multi-pixel/multi-member metadata preservation (including death metadata) for
both radius and kNN queries; invalid particles excluded from the index entirely (and
never returned even at a huge radius), including an all-invalid empty-index case;
deterministic/reproducible ordering across repeated calls for both query types, plus an
exact-tie case confirming the `particle_id` tiebreak.

```text
python -m unittest tests.test_kdtree_neighborhood -v
Ran 20 tests in 0.01s
OK

python -m unittest discover -s tests -v   (full suite)
Ran 308 tests in ~13s
OK
```

### Exact neighbor agreement and runtime comparison (tasks.md's own T15 check)

Verified exact agreement (membership, order, and floating-point distance) between
`query_radius`/`query_knn` and an independent brute-force reference on random synthetic
point sets (see tests above) — 0 discrepancies.

Runtime, 20,000-particle index, single query: KD-tree build 9.05 ms; one radius query
(148 matches) 1.24 ms vs. a hand-optimized single-shot numpy brute-force scan 0.57 ms.
200 repeated radius queries against the same index: KD-tree wrapper 61.7 ms total (0.31
ms/query) vs. the same numpy brute-force scan repeated 200 times, 38.8 ms total (0.19
ms/query) — **the brute-force reference was faster in this specific micro-benchmark.**
Isolating the underlying algorithm from this module's own Python-level overhead (result
sorting + full-metadata mapping) explains why: 200 raw `cKDTree.query_ball_point` calls
alone took 5.6 ms vs. 105 ms for the equivalent raw per-query brute-force distance loop
(**~19x faster**) — `scipy.spatial.cKDTree` itself is working exactly as expected. This
module's own convenience wrapper (building a fully-typed, deterministically-sorted
`NeighborhoodResult` with complete metadata for every match) adds Python-level overhead
that, at this particle count (~20K) and match count (~150/query), can dominate the raw
algorithmic saving — reported honestly rather than claiming a speedup that a direct
`query_radius`/`query_knn` benchmark did not actually show. A batched `cKDTree` query
(200 points at once, scipy's native batch API) was just as fast as one call (5.0 ms
total) — a likely direction for a later task if per-query overhead matters at scale;
this module deliberately does not add batched querying now, since it is not required by
`tasks.md`'s T15 scope.

### Known limitations / explicitly deferred (not implemented in T15)

- No KDE, support-map construction (`D`/`W`/`V`), candidate-peak/core extraction, or any
  later Phase 2 task — none were touched.
- The local equirectangular projection is a first-order, single-reference-point
  approximation (matching T03's own documented spherical-Earth/no-geodesic-library
  limitation) — appropriate for one regional point cloud at one instant, not a general
  large-area or multi-region projection.
- This module's convenience wrapper (sorted, fully-typed results) has measurable
  Python-level overhead at ~20K particles/~150 matches per query, as reported above; a
  later task may want a lower-overhead / batched query path if profiling shows this
  matters at production scale.
- `iste.particle_query` (T14, current numbering) was not modified.

## Phase 2 — T16 — Weighted support-feature construction (PENDING_REVIEW)

### Goal

Per `tasks.md`'s T16 and `README.md` Sec. 14/15, construct the three Phase 2 candidate-
support features (`D`, `W`, `V`) from a T14 particle slice queried through the T15
KD-tree engine, and combine them into a single `Support` value via an explicit,
configurable function — clearly labeled Phase 2 candidate-support data, never `c_hat*`,
a likelihood, or a probability density.

### Implementation

`iste/support_features.py` (new):

- `SupportFeatureConfig` — frozen dataclass: `d_scale`/`w_scale`/`v_scale` (feature
  scaling divisors, default `1.0` = no scaling, must be `> 0`) and `weight_d`/`weight_w`/
  `weight_v` (linear-combination coefficients for the default `f`, default `1.0` each,
  may be `0.0` to exclude a feature). Explicitly labeled PROTOTYPE/CONFIGURATION, not a
  physically validated normalization.
- `linear_support_function(d, w, v, config)` — the shipped default `f(D,W,V)`:
  `weight_d*(D/d_scale) + weight_w*(W/w_scale) + weight_v*(V/v_scale)`.
- `compute_support_features(index, lat, lon, *, radius_m=None, k=None, config=None,
  combine_fn=linear_support_function)` — queries T15's `query_radius`/`query_knn`
  directly (exactly one of `radius_m`/`k` required, no second spatial-query
  implementation), then computes:
  - `D` = `neighborhood.n_particles` (already valid-only, structurally, since T15's
    index only ever contains valid particles);
  - `W` = `sum(exp(neighborhood.log_weight))` — exponentiated only at this point of
    actual linear-space need, matching `iste.diagnostics`/`iste.likelihood`'s existing
    convention; the stored `log_weight` itself is never modified;
  - `V` = `len(set(neighborhood.observation_pixel_id))` — distinct pixel COUNT, not
    particle count.
  Returns a `SupportFeatures` carrying `D`/`W`/`V`/`support` plus the full T15
  `NeighborhoodResult` (so `particle_id`/`observation_pixel_id`/`forcing_member_id`
  identity of every contributing particle stays available, per `README.md` Sec. 14.3).
- `compute_support_map(index, query_lats, query_lons, ...)` — `compute_support_features`
  evaluated at each of several query points, in order — the Phase 2 "candidate-support
  heat map" as a flat list, not a rasterized grid or peak/core extraction (T17+'s scope).
- `combine_fn` is a free, pluggable `(D, W, V, config) -> float` callable — a caller can
  replace `linear_support_function` entirely, not just tune its config.

### Exact support/scaling configuration implemented

```text
D(x,y,t) = neighborhood.n_particles                          (valid particles only, by construction)
W(x,y,t) = sum(exp(log_weight_i)) over the neighborhood        (linear Stage III weight sum)
V(x,y,t) = |{distinct observation_pixel_id in the neighborhood}|

default f(D,W,V) = weight_d*(D/d_scale) + weight_w*(W/w_scale) + weight_v*(V/v_scale)
default config:   d_scale=w_scale=v_scale=1.0, weight_d=weight_w=weight_v=1.0
                  (i.e. Support = D + W + V, unscaled, unweighted, by default)
```

Every one of these six numbers is an explicit, overridable configuration value (per
`README.md` Sec. 15/26's "feature scaling must be explicit" / "support feature scaling"
and "support-combination function" are both listed as open Phase 2 parameters) — none is
claimed to be scientifically validated or physically meaningful; `combine_fn` itself is
also freely replaceable.

### Tests

`tests/test_support_features.py` (new) — 22 `unittest` cases: `D` matches a hand-
controlled neighborhood count (and is `0` with nothing nearby); `W` is the exact linear
sum of `exp(log_weight)` (with an explicit check that it is NOT a sum of raw log-weights,
and that it is never normalized/clamped to `[0,1]`); `V` counts distinct pixels, not
particles (5 particles across 3 pixels → `D=5`, `V=3`; 3 particles sharing one pixel →
`V=1`); invalid particles contribute nothing to `D`/`W`/`V` even with an enormous weight
(and an all-invalid case gives all-zero features); independent configuration checks
(changing weights alone changes `W` but not `D`/`V`; adding same-pixel particles changes
`D` but not `V`; widening the radius changes `D` and `V` together); explicit support-
function checks (default config reduces to `D+W+V`; zeroing `weight_w` removes `W`'s
influence exactly; changing `d_scale` changes `support`; a custom `combine_fn` is
actually invoked, not ignored; invalid config values rejected); full neighborhood
metadata (`particle_id`/`observation_pixel_id`/`forcing_member_id`) available on the
result; a controlled two-region synthetic case where the denser/heavier/more-diverse
region ranks higher on `D`, `W`, `V`, AND `support` simultaneously; `compute_support_map`
preserves query order; and input validation (`radius_m`/`k` mutual exclusivity,
mismatched `compute_support_map` lengths).

```text
python -m unittest tests.test_support_features -v
Ran 22 tests in 0.017s
OK

python -m unittest discover -s tests -v   (full suite)
Ran 330 tests in ~11.3s
OK
```

### Known limitations / explicitly deferred (not implemented in T16)

- No local-maxima detection, near-duplicate suppression, candidate-core definition, or
  `candidate_id`/`support_score`/etc. record construction — all T17's scope.
- No rasterized/gridded heat map (`compute_support_map` returns a flat list at the
  queried points only, not an interpolated raster).
- `W`'s `exp(log_weight)` summation has no explicit numerical-stability guard (e.g. a
  shift-by-max trick) — acceptable at Stage III's typical log-weight magnitudes (small,
  near `0`, per the existing controlled validation suite), but flagged as a possible
  future concern if a much longer backward window or stronger divergence ever produces
  extreme log-weights.
- `iste.particle_query` (T14) and `iste.kdtree_neighborhood` (T15) were not modified.

## Phase 2 — T17 — Candidate peak and candidate-core extraction (PENDING_REVIEW)

### Goal

Per `tasks.md`'s T17 and `README.md` Sec. 16, extract discrete candidate peaks/cores
from a T16 Phase 2 support map at one historical timestamp: local-maxima detection,
near-duplicate suppression, multi-mode preservation, timestamp attachment, and an
explicit configurable candidate core around each retained peak — while never collapsing
to a centroid and never treating the whole support surface as "the source."

### Implementation

`iste/candidate_extraction.py` (new):

- `CandidateExtractionConfig` — frozen dataclass, PROTOTYPE/CONFIGURATION:
  `peak_neighborhood_cells` (odd `>=3`, local-maximum comparison window in grid cells,
  default `3`), `min_support` (candidate-peak threshold, default `-inf` = no explicit
  floor), `min_separation_m` (non-maximum-suppression distance, default `0.0`),
  `core_radius_m` (fixed core radius, default `1000.0`). All validated explicitly.
- `CoreRegion` — the candidate core itself: a fixed-radius circle
  (`center_lat`/`center_lon`/`radius_m`) around the peak. The smallest-complexity core
  definition satisfying "define an explicit configurable high-support candidate core" —
  not a contour/support-threshold-based core (see limitations).
- `Candidate` — `candidate_id`, `x_peak`/`y_peak` (README's own `(x,y)` = lon/lat
  naming), `t_s`, `core_region`, `support_score`/`density_score`/
  `weighted_support_score`/`pixel_diversity_score` (the exact `D`/`W`/`V`/`Support`
  values T16 computed at the peak), and `peak_support_features` (the full T16
  `SupportFeatures` — including its T15 `NeighborhoodResult`, i.e. every contributing
  particle's `particle_id`/`observation_pixel_id`/`forcing_member_id` — preserved for
  later core sampling, T18/T19).
- `find_candidate_peaks(support_map, lat_grid, lon_grid, t_s, config=None)` — the core
  entry point, working from an already-computed T16 support map alone (per this task's
  own scope): reshapes the map's `support`/`density` values onto the caller's regular
  grid, finds local maxima via `scipy.ndimage.maximum_filter` (an existing dependency,
  not a hand-rolled peak-finder), rejects any peak with zero contributing particles
  (`density == 0` — "no candidate from invalid/empty input," regardless of
  `min_support`), then greedily suppresses near-duplicates in descending-support order
  using `iste.kdtree_neighborhood.project_to_local_metres` (T15's own distance-safe
  projection, never raw degree differences) for the separation check.
- `find_candidate_peaks_from_index(index, lat_grid, lon_grid, t_s, ...)` — a thin
  convenience wrapper that builds the grid's query points, calls T16's
  `compute_support_map`, and hands the result to `find_candidate_peaks`; the actual
  peak/core-extraction logic lives entirely in the support-map-only function above.

### Exact peak/core configuration

```text
local-maximum detection: scipy.ndimage.maximum_filter, size=peak_neighborhood_cells (default 3x3,
  i.e. immediate 8-connected grid neighbors), mode="nearest"
peak acceptance:  support == local_max  AND  isfinite(support)  AND  support > min_support (default -inf)
                  AND  density > 0  (never a candidate from an empty neighborhood)
non-max suppression: greedy, descending support order; a peak is dropped if any
  ALREADY-KEPT peak is closer than min_separation_m (default 0.0 -- no suppression
  beyond exact coincidence), using T15's project_to_local_metres for distance
candidate core: CoreRegion(center=peak, radius_m=core_radius_m)  (default 1000.0 m, fixed-radius circle)
candidate_id: f"{ISO-8601 t_s}_cand{i:03d}"  (i = rank by descending support after suppression)
```

Every one of `peak_neighborhood_cells`/`min_support`/`min_separation_m`/`core_radius_m`
is an explicit `CandidateExtractionConfig` field (`README.md` Sec. 26's "candidate peak
threshold" / "candidate core definition" / "candidate merge/separation threshold" are
all listed as open Phase 2 parameters) — none is claimed physically validated.

### Tests

`tests/test_candidate_extraction.py` (new) — 17 `unittest` cases: two well-separated
peaks both found, and confirmation the result never collapses to a single centroid;
near-duplicate suppression with a small threshold keeping both grid-separated peaks, a
large threshold suppressing the weaker one (keeping the correct, higher-support
survivor), and confirmation the threshold is what changes the outcome; three distinct
far-apart modes all retained simultaneously; `t_s` attached exactly, with different
timestamps producing different `candidate_id`s; core region using the configured radius
exactly, centered exactly on the peak, and rejecting invalid radii; candidate scores
matching the underlying `SupportFeatures` exactly (including the `neighborhood`
provenance); no candidates from an empty grid or an all-zero-density support map, and a
mismatched-length rejection; reproducibility across repeated calls including an exact-tie
support case; and one small end-to-end integration test running real T14/T15/T16
machinery over two synthetic particle clusters, confirming two candidates with correct
per-cluster pixel diversity.

```text
python -m unittest tests.test_candidate_extraction -v
Ran 17 tests in 0.11s
OK

python -m unittest discover -s tests -v   (full suite)
Ran 347 tests in ~13s
OK
```

### Known limitations / explicitly deferred (not implemented in T17)

- `core_region` is a fixed-radius circle only — not an adaptive contour/support-
  threshold-based region (e.g. "everywhere within X of the peak where support stays
  above some fraction of the peak's own value"); a possible future refinement if T18/T19
  need a more elaborate core shape.
- No core-particle sampling (`N_forward = min(X,Y)`), forward propagation, AIS,
  likelihood evaluation, or posterior inference — all later tasks' scope.
- Never uses or references Stage III `c_hat*` — this module has no knowledge of it.
- `find_candidate_peaks` requires the support map to already be laid out on a REGULAR
  grid (row-major, matching `lat_grid x lon_grid`); it does not accept or interpolate an
  irregular/scattered set of query points.
- `iste.particle_query` (T14), `iste.kdtree_neighborhood` (T15), and
  `iste.support_features` (T16) were not modified.

### Manual-check notebook

`notebooks/T17_candidate_extraction_check.ipynb` (new) — a controlled multimodal
synthetic historical particle slice (3 well-separated Gaussian clusters, deliberately
different sizes/weight-scales/pixel-diversity so the resulting candidates are
distinguishable by `D`/`W`/`V`, not just location) is run through the real T14 → T15 →
T16 → T17 chain — `HistoricalParticleSlice` constructed directly (synthetic input
standing in for a T14 query result), `build_particle_neighborhood_index` (T15),
`compute_support_map` (T16, default `SupportFeatureConfig`), `find_candidate_peaks`
(T17). No scientific logic is duplicated in the notebook — every computation calls
directly into the `.py` modules. The notebook produces a figure with a full-domain
support heatmap (red stars = candidate peaks, red circles = candidate-core boundaries)
plus one tightly-zoomed panel per candidate (the core boundary, a few km, is not visible
against the ~700 km full domain), with the historical timestamp `t_s` labeled in the
figure's title, and a closing "Interpretation limits" cell restating that this is a
screening artifact, not a source estimate. Executed end-to-end via
`jupyter nbconvert --execute` with zero errors: exactly 3 candidates recovered, one per
synthetic cluster, matching the known cluster centres. Re-ran the full test suite
afterward (notebook-only change, no `.py` edits) — still 347/347 passing.

## Phase 2 — T18 — Forward particle budget `N_forward = min(X, Y)` (PENDING_REVIEW)

### Goal

Per `tasks.md`'s T18 and `README.md`'s "Locked rule: `N_forward = min(X,Y)`" (Sec. 11.2)
and Sec. 17's opening definition: for every T17 candidate, determine `Y` (valid Stage III
particles inside the candidate's core), take the configured budget `X`, and compute
`N_forward = min(X, Y)` — never rejecting a candidate for `Y < X`, and explicitly marking
`Y = 0` candidates as not forward-validatable rather than dropping them.

### Implementation

`iste/forward_budget.py` (new):

- `ForwardBudgetConfig` — frozen dataclass holding `X` (the configured
  forward-validation particle budget, `README.md` Sec. 26's "X forward particle
  budget"), validated as a positive `int`. Explicit configuration, not a hidden
  constant; documented as a computational particle-count budget only — never a physical
  statement about oil mass/parcel count (`README.md` Sec. 11.2).
- `ForwardBudget` — frozen dataclass result: `candidate_id`, `t_s`, `X`, `Y`,
  `N_forward`, `forward_validatable` (`False` iff `Y == 0`).
- `compute_forward_particle_budget(candidate, index, config)` — computes one
  candidate's `Y` by re-querying `iste.kdtree_neighborhood.query_radius` at the
  candidate's own `core_region.center_lat`/`center_lon`/`radius_m` (T17's candidate
  core, not the support map's own query radius/k — see below), then
  `N_forward = min(config.X, Y)`.
- `compute_forward_particle_budgets(candidates, index, config)` — the same, applied to
  a list of candidates that all share one T15 `index`/historical timestamp; each
  candidate's result is computed independently from its own `core_region`.

**Key design decision — `Y` is re-queried at `core_region`, never reused from T16's
`density_score`.** A candidate's `density_score` (T16 `D`) was computed at the peak
using whatever radius/k the *support map* happened to be evaluated with, which is a
separate, independently-configured value from `core_region.radius_m` (T17's own
candidate-core definition). `README.md` Sec. 16 is explicit that `core_region` — not the
support map's own query radius — "is the high-support region used to select
representative particles for forward validation," so `Y` must come from a fresh
`query_radius` call at the core's own radius. `tests/test_forward_budget.py`'s
`TestYUsesCoreRegionNotSupportMapDensity` constructs a case where the two radii give
different counts and asserts `Y != density_score` there, so this distinction is
regression-tested, not just documented.

**`Y` counts only valid particles "for free."** A T15 `ParticleNeighborhoodIndex` only
ever indexes a slice's valid particles (T15's own existing policy) — so every particle
`query_radius` returns is already guaranteed valid, and no extra validity filter is
needed inside T18 itself.

### Exact X/Y/N_forward behavior

```text
Y = query_radius(index, core_region.center_lat, core_region.center_lon, core_region.radius_m).n_particles
X = config.X   (explicit ForwardBudgetConfig, validated positive int)
N_forward = min(X, Y)

Y > X  → N_forward = X,  forward_validatable = True
Y = X  → N_forward = X,  forward_validatable = True
Y < X  → N_forward = Y,  forward_validatable = True   (never rejected for Y < X)
Y = 0  → N_forward = 0,  forward_validatable = False  (ForwardBudget record still returned)
```

### Tests

`tests/test_forward_budget.py` (new) — 16 `unittest` cases: all three `min(X,Y)`
branches (`Y>X`, `Y=X`, `Y<X`, the last confirmed NOT rejected); `Y=0` on both a
particle-free core and a completely empty slice, both explicitly `forward_validatable =
False` while still returning a full record; `candidate_id`/`t_s`/`X`/`Y`/`N_forward`
preserved exactly; two candidates in one timestamp independent of each other and of
processing order, and two separate timestamps (two independent slices/indices) not
cross-contaminating; invalid particles placed inside the core radius correctly excluded
from `Y`; repeated calls and the batch helper deterministic/identical; the
radius-mismatch test described above; `ForwardBudgetConfig` validation (rejects
non-positive and non-integer `X`); and one end-to-end integration test running real
T14/T15/T16/T17 machinery over two differently-sized synthetic clusters into T18.

```text
python -m unittest tests.test_forward_budget -v
Ran 16 tests in 0.06s
OK

python -m unittest discover -s tests -v   (full suite)
Ran 363 tests in ~9s
OK
```

### Known limitations / explicitly deferred (not implemented in T18)

- No particle *selection*/sampling — `compute_forward_particle_budget` only counts `Y`
  and computes `N_forward`; choosing which `N_forward` particles to actually use
  (weighted-by-Stage-III-weight sampling without replacement) is T19's scope.
- No forward propagation, AIS, likelihood evaluation, or posterior inference.
- Never reads or references Stage III `c_hat*`.
- `iste.particle_query` (T14), `iste.kdtree_neighborhood` (T15),
  `iste.support_features` (T16), and `iste.candidate_extraction` (T17) were not
  modified — T18 only calls their existing public functions.
- `compute_forward_particle_budget`/`compute_forward_particle_budgets` assume the
  caller passes the T15 `index` matching the candidate's own `t_s`; multi-timestamp
  candidate lists must be grouped by `t_s` and called once per group (documented in the
  module's docstring, matching T17's own `find_candidate_peaks_from_index` convention of
  one `index` per one `t_s`) — T18 does not itself validate that `index` and
  `candidate.t_s` agree.

## Phase 2 — T19 — Weighted core-particle sampling (PENDING_REVIEW)

### Goal

Per `tasks.md`'s T19 and `README.md` Sec. 17's second half: for every T17 candidate,
select `N_forward = min(X, Y)` particles from its core for forward validation — weighted
by the original Stage III particle weights and without replacement by default when
`Y >= X`, or all `Y` particles retained (unweighted, never rejected) when `Y < X` —
while never modifying the original Stage III weights and never double-counting them
into a forward-KDE weighting scheme that belongs to a later stage.

### Implementation

`iste/core_sampling.py` (new):

- `SampledCoreParticles` — frozen dataclass result: T18's own `candidate_id`/`t_s`/`X`/
  `Y`/`N_forward`/`forward_validatable`, plus `resampled` (`True` iff `Y >= X`, i.e.
  weighted-without-replacement resampling actually ran) and `sampling_seed`
  (`README.md` Sec. 25's own output-artifact field), plus one array per required
  metadata field for every selected particle: `particle_id`, `observation_pixel_id`,
  `forcing_member_id`, `lat`/`lon`, `log_weight` (original, untouched), `valid`,
  `death_time`/`death_lat`/`death_lon`/`death_reason`.
- `sample_core_particles(candidate, index, config, seed, *, replace=False)` — the core
  entry point. Reuses T18's `compute_forward_particle_budget` for `X`/`Y`/`N_forward`/
  `forward_validatable` (no duplicated budget logic) and T15's `query_radius` directly
  (at `candidate.core_region.center_lat`/`center_lon`/`radius_m`) for the actual
  particle pool to sample from. Branches exactly on `README.md`'s locked rule:
  `Y == 0` → empty selection; `Y >= X` → `rng.choice(Y, size=X, replace=replace,
  p=selection_probabilities)`; `0 < Y < X` → all `Y` particles, unweighted.
- `_selection_probabilities(log_weight)` — original `log_weight` → a selection-
  probability vector via numerically-stable `exp(log_weight - max)` / normalize (the
  same log-space-to-linear-weight convention `iste.support_features`/`iste.likelihood`
  already use); falls back to a uniform distribution only in the degenerate case where
  every particle's weight is exactly zero (`log_weight` non-finite everywhere) — an
  explicit, documented fallback so sampling can still proceed, not a scientific claim.
- `sample_core_particles_for_candidates(candidates, index, config, seeds, *,
  replace=False)` — the same, applied to a list of candidates sharing one T15
  `index`/timestamp, with one EXPLICIT seed per candidate (`seeds` must match
  `candidates` in length) — no seed is ever silently derived or reused across
  candidates.

**Avoiding double-counting Stage III weights.** This module does not itself compute or
attach any forward-KDE weight (that would start implementing the forward predicted KDE,
explicitly out of scope: "Do not implement predicted slick/KDE comparison yet"). It
instead records `resampled` on every result so a later stage building the forward KDE
knows which convention to apply: `resampled=True` → that later stage should use equal
computational weight per selected particle by default; `resampled=False` → all original
particles were retained, and any normalization of their relative Stage III weights for
that later stage's own forward-KDE construction must happen there, not in T19 — `log_weight`
itself is never modified, scaled, or renormalized anywhere in this module.

### Exact sampling behavior

```text
Y == 0            -> empty selection (all arrays length 0), forward_validatable=False, resampled=False
Y >= X            -> N_forward = X particles via rng.choice(Y, size=X, replace=replace, p=selection_probabilities(log_weight)), resampled=True
0 < Y < X         -> N_forward = Y, ALL Y particles selected (np.arange(Y), original query order), unweighted, resampled=False; never rejected

selection_probabilities(log_weight):
    p_i ∝ exp(log_weight_i - max(log_weight))   (numerically stable, matches existing project convention)
    degenerate fallback: if every log_weight is non-finite (all weights zero), p_i = 1/n uniformly

replace: bool = False   (explicit override; without replacement is the default per README.md Sec. 17)
```

### How RNG/reproducibility is handled

`sample_core_particles` takes a required `seed: int` (never optional, never a shared/
global RNG) and constructs a fresh `np.random.default_rng(seed)` inside the call, so:
- the exact same `(candidate, index, config, seed)` always produces the exact same
  selection (verified by `TestReproducibility.test_same_seed_gives_identical_selection`
  and `test_same_candidate_sampled_independently_and_reproducibly_multiple_times`, the
  latter interleaving a different-seeded call in between to confirm no shared state
  leaks between calls);
- different seeds can (and empirically do, across 10 trial seeds) produce different
  selections;
- `sampling_seed` is recorded on the result itself, so the selection is reproducible
  from the returned record alone without needing to separately track what seed was used.

### Tests

`tests/test_core_sampling.py` (new) — 18 `unittest` cases covering all 11 requested
requirements: `Y>X`/`Y=X`/`Y<X` exact selected counts (the `Y=X` case additionally
confirms every original particle appears, since a size-`Y` without-replacement draw from
a `Y`-particle pool has no choice); `Y=0` explicit empty/unvalidated selection with every
array length 0 and no exception; same-seed determinism and different-seeds divergence;
without-replacement (no duplicate `particle_id`s) by default, plus a `replace=True` case
that does produce duplicates (confirming the flag actually reaches numpy's sampler);
higher-weight-particle selection-frequency tests over 800 and 300 Monte Carlo trials
respectively (a monotonic weight/frequency relationship, and a zero-weight-vs-dominant-
weight extreme case); full metadata-field-by-field equality against the source slice for
every selected particle, plus a death-metadata pass-through check; original `log_weight`
proven byte-identical on the source slice after sampling (both branches exercised) and on
every selected particle (no normalization applied); the batch helper's seeds-length
validation and its equivalence to individual calls; and the all-zero-weight uniform
fallback not raising.

```text
python -m unittest tests.test_core_sampling -v
Ran 18 tests in 0.14s
OK

python -m unittest discover -s tests -v   (full suite)
Ran 381 tests in ~7-8s
OK
```

### Manual-check notebook

`notebooks/T19_core_sampling_check.ipynb` (new) — two synthetic candidate cores at one
historical timestamp, deliberately built to exercise both branches of the locked rule:
candidate A (25 particles, budget `X=8` → `Y>=X`, weighted resampling) and candidate B
(5 particles, `X=8` → `Y<X`, all retained). Every candidate/budget/sampling computation
calls directly into `iste.candidate_extraction`/`iste.support_features`/
`iste.forward_budget`/`iste.core_sampling` — no logic duplicated in the notebook. The
figure plots every core particle (marker size ∝ original Stage III weight, unmodified)
with selected particles in red and unselected in gray, the core boundary as a blue
circle, and each panel's title showing `Y`/`X`/`N_forward`/`resampled`/`sampling_seed`.
Executed end-to-end via `jupyter nbconvert --execute` with zero errors; both in-notebook
asserts (`n_selected==8, resampled` for A; `n_selected==5, not resampled` for B) passed.
Closing "Interpretation limits" cell restates that no forward propagation/KDE/`c_hat*`
is computed here. Full suite re-run afterward (notebook-only change) — still 381/381.

### Known limitations / explicitly deferred (not implemented in T19)

- No forward-KDE weight is computed or attached — only the `resampled` flag is provided
  for a later stage to consult; that later stage owns any actual weight normalization.
- No forward propagation, predicted-slick construction, forward scoring, or posterior
  inference.
- Never reads or references Stage III `c_hat*`.
- `iste.particle_query` (T14) through `iste.forward_budget` (T18) were not modified —
  T19 only calls their existing public functions.
- The degenerate all-zero-weight → uniform-probability fallback in
  `_selection_probabilities` is an explicit implementation choice for a pathological
  input (documented in the module docstring and tested), not a scientifically validated
  interpretation of a zero-weight particle's true selection likelihood.
- `sample_core_particles`/`sample_core_particles_for_candidates` assume `index` matches
  each candidate's own `t_s`, exactly as T18 already assumes — not re-validated here.

## Phase 2 — T20 — Forward transport operator (PENDING_REVIEW)

### Goal

Per `tasks.md`'s T20 and `README.md` Sec. 18: implement the elementary one-step Phase 2
forward Euler-Maruyama transport operator — a NEW prototype, explicitly not described as
the mathematical inverse of the frozen Stage III backward adjoint — that a later task
(T21) will loop from `t_s` to `t_obs`.

### Forward operator equations (exact)

```text
U_e(x,t) = U(x,t) + delta_U_e(x,t)        (T08/T09's existing coherent forcing-member field)

X_new = X_old + U_e(X_old,t) * dt + sqrt(2*K(X_old,t)*dt) * xi,   xi ~ N(0, I_2)

dx_m = u_e * dt + sqrt(max(2*K, 0) * dt) * xi_x
dy_m = v_e * dt + sqrt(max(2*K, 0) * dt) * xi_y
(dlon_deg, dlat_deg) = iste.geometry.metres_to_degrees(dx_m, dy_m, lat_old)   (T03, unchanged)
lat_new = lat_old + dlat_deg
lon_new = lon_old + dlon_deg
```

Units/sign convention: `u_e`/`v_e` in m/s (eastward/northward), `K` in m^2/s, `dt` in
seconds (an increment of FORWARD time `t`, `dt > 0` — the opposite time-direction
convention from Stage III's backward `theta`, but the same positivity requirement);
`dx_m`/`dy_m` are metre displacements, converted to degrees only via
`iste.geometry.metres_to_degrees` (never added to degree coordinates directly). Sign is
`+U_e` (physical forward advection), never `-U`; there is **no `grad(K)` term** (Stage
III's backward step has `+grad(K)`, this forward prototype does not); there is **no
weight update** at all (`log_w_new = log_w_old - div(U)*dt` is exclusively a Stage
III/backward concept — `README.md`'s Phase 2 forward sections never mention a weight
update during forward propagation).

### Implementation

`iste/forward_transport.py` (new):

- `ForwardParticleState` — position + validity (T04's `ParticleState` shape) plus each
  particle's ORIGINAL Stage III `forcing_member_id` (provenance only, preserved
  unmodified through every step — NOT used to select which `U_e` a step applies).
  Deliberately no `log_weight` field.
- `forward_euler_maruyama_step(state, t, dt_seconds, ocean_source, k_lat, k_lon,
  k_field, rng, domain_variable="uo", radius_m=MEAN_EARTH_RADIUS_M)` — one step, applied
  to every particle in `state` independently.

**Reuse, not duplication.** `ocean_source` is expected to already BE the perturbed
`U_e = U + delta_U_e` field for whichever forcing member is being forward-propagated —
built externally via T09's existing `iste.ensemble_run.make_perturbed_source(base,
ensemble, member_id)`, unmodified; this module simply reads `uo`/`vo` from whatever
source it is handed via `iste.interpolation.interpolate_field`, exactly as `iste.step`
does for its own (differently-signed) transport term. `K` is read via
`iste.interpolation.bilinear_spatial_interpolate` directly (the same primitive
`iste.environment.interpolated_k_value_and_gradient` itself calls internally for `k_val`)
rather than via that K+gradient wrapper, since this module has no use for `grad(K)` and
should not have its own validity mask polluted by an unrelated `grad(K)`-unavailability
failure at a point where `K` itself is perfectly valid. Boundary/domain validity reuses
`iste.boundary.apply_boundary_policy`/`netcdf_validity_check` completely unchanged — the
same "stop propagation, retain last valid state," "once invalid, always invalid" policy
Stage III uses; an already-invalid particle is never silently treated as valid.
`iste.step`/`iste.trajectory` are not imported, modified, or duplicated.

### Tests

`tests/test_forward_transport.py` (new) — 17 `unittest` cases covering all 10 requested
requirements: zero-velocity+K=0 exact position preservation; forward (`+U_e`) sign and
magnitude (including a hand-computed metre→degree check); fixed-seed reproducibility
plus an exact reference-RNG-draw match; different-seed divergence; K=0 giving a
seed-independent (pure-drift) result; metre-to-lat/lon conversion matching
`iste.geometry` exactly; invalid/domain-exit handling (land destination, already-invalid
particle never re-validated, missing-`K`-at-current-position, and a mixed valid/invalid
batch where only the valid particle steps); a `grad(K)`-absence check (a linear `K`
field that is exactly zero at the query point produces ZERO displacement, unlike T05's
backward step, which would drift up the gradient even there); `forcing_member_id`
preserved across both a multi-particle valid step and a freshly-invalidated particle;
one fully hand-computed combined drift+diffusion synthetic case; and `dt_seconds`
validation.

```text
python -m unittest tests.test_forward_transport -v
Ran 17 tests in 0.04s
OK

python -m unittest discover -s tests -v   (full suite)
Ran 398 tests in ~14s
OK
```

### Manual check

This task's own "Progress report" ask (equations, synthetic validation results, exact
units/sign convention) is inherently textual/numeric rather than spatial — unlike T17/T19
(candidate heat maps / core scatter plots), a notebook adds little beyond what the
equations above and the hand-computed test assertions already show directly and
exactly, so (matching the precedent set by T14/T15/T16/T18, which also did not receive
notebooks) no `.ipynb` was produced for T20; the "synthetic validation results" required
by the task are the automated tests above, directly inspectable and already passing
exactly against hand computation (see `TestControlledSyntheticCase` and
`TestMetreToDegreeConversion` in particular).

### Known limitations / explicitly deferred (not implemented in T20)

- No multi-step trajectory loop (`t_s -> t_obs`) — that is T21's scope; this module is
  only the elementary single-step operator T21 will loop, exactly as `iste.step.
  backward_euler_maruyama_step` is Stage III's own elementary operator for
  `iste.trajectory.run_backward_trajectory` to loop.
- No multi-member/candidate orchestration, predicted-slick/KDE construction, or forward
  comparison metrics.
- Never reads or references Stage III `c_hat*`.
- No forward-run death-event provenance (`death_lat`/`death_lon`/`death_reason` for a
  NEW invalidation that happens DURING forward propagation, as distinct from each
  particle's existing Stage III backward death metadata) is tracked by this module —
  not requested by T20's own requirements list, and more naturally belongs to T21's
  multi-step loop if a later task needs it; left as an explicit deferral, not a silent
  omission.
- `iste.step`/`iste.trajectory`/`iste.boundary`/`iste.environment`/`iste.forcing_ensemble`/
  `iste.ensemble_run` were not modified — T20 only calls their existing public functions.

## Phase 2 — T21 — Forward propagation orchestration (PENDING_REVIEW)

### Goal

Per `tasks.md`'s T21 and `README.md` Sec. 19: for one candidate at one historical
timestamp, forward-propagate its T19-selected core particles from `t_s` to exactly
`t_obs`, once per configured forcing member (`Candidate C -> U_1 -> predicted slick_1
-> ... -> U_E -> predicted slick_E`), each run internally coherent (no forcing
redrawn per particle/timestep), recording the exact forcing-member subset used.

### Implementation

`iste/forward_propagation.py` (new):

- `ForwardCandidateMemberRun` — the full checkpoint history (`lat`/`lon`/`valid`, shape
  `(n_steps+1, n_particles)`, one row per checkpoint from `t_s` to `t_obs` inclusive) of
  ALL of one candidate's T19 core particles, run under ONE forcing member. Also carries
  `candidate_id`/`t_s`/`t_obs`/`dt_seconds`/`n_steps`/`timestamps`, each particle's
  `particle_id`/`observation_pixel_id`/`forcing_member_id` (ORIGINAL Stage III
  provenance)/`original_log_weight` (both copied straight through from T19, unmodified),
  and `forward_forcing_member_id` (the member THIS run used — deliberately a separate
  field from `forcing_member_id`; see below). `first_invalid_step(i)` mirrors T06's
  `BackwardTrajectory.first_invalid_index`, per particle.
- `ForwardCandidateRuns` — one candidate's complete result: `candidate_id`/`t_s`/`t_obs`,
  `forcing_member_ids` (the exact recorded subset), and `runs: dict[member_id ->
  ForwardCandidateMemberRun]`.
- `forward_propagate_candidate_member(sampled, t_obs, dt_seconds, ocean_source_for_member,
  forward_forcing_member_id, k_lat, k_lon, k_field, rng, ...)` — the per-member checkpoint
  loop: builds an exact `t_s -> t_obs` schedule from integer-nanosecond arithmetic (as
  many full `dt_seconds` steps as fit, plus one final PARTIAL step covering whatever
  remains, so the run always lands exactly on `t_obs` regardless of whether `(t_obs -
  t_s)` divides evenly by `dt_seconds`), then calls T20's
  `forward_euler_maruyama_step` once per checkpoint for ALL particles in the batch at
  once (T20 is already vectorized). Skips the step call once every particle is already
  invalid (a would-be no-op), matching T06's own efficiency/bookkeeping pattern.
- `forward_propagate_candidate(sampled, t_obs, dt_seconds, base_ocean_source, ensemble,
  forcing_member_ids, seeds, k_lat, k_lon, k_field, ...)` — the per-candidate
  orchestration: builds each member's perturbed `U_e` field EXACTLY ONCE (via
  `iste.ensemble_run.make_perturbed_source`, T08/T09, unmodified) and calls
  `forward_propagate_candidate_member` once per `(member_id, seed)` pair. `seeds` must
  have one explicit entry per `forcing_member_ids` (no seed silently shared/derived
  across members, matching T19's own `sample_core_particles_for_candidates` convention).

**Key design decision — `forcing_member_id` vs. `forward_forcing_member_id` (recorded in
`memory.md`).** `README.md` Sec. 19 propagates the SAME selected particle set once per
configured member, not each particle under its own original member — so a particle's
`forcing_member_id` (its Stage III backward-run provenance, preserved unmodified) is
never read to choose which `U_e` a step uses; that choice is `forward_forcing_member_id`,
an explicit orchestration-level parameter applied uniformly to the whole batch. This
module never conflates the two.

### Tests

`tests/test_forward_propagation.py` (new) — 21 `unittest` cases covering all 11
requested requirements: exact termination at `t_obs`; multiple members producing
separate, generally-divergent results with correct `forward_forcing_member_id` each;
forcing-member coherence (an instrumented `make_perturbed_source` call-count check
confirms exactly ONE build per member, never per step, plus a co-located-particles
identical-drift check under K=0); full particle metadata preserved exactly against the
T19 input, including the original-vs-forward member-id distinction; seed reproducibility
(single-member-run and full-orchestration levels) and seed divergence; invalid particles
freezing at their last valid position and no longer moving (plus an all-invalid-batch
short-circuit and a zero-particle-batch edge case); land/domain invalidation via a
grid-edge exit; no-overshoot for both evenly-dividing and non-evenly-dividing `dt`
(confirming the final PARTIAL step lands exactly on `t_obs`), a `t_s == t_obs` zero-step
case, and a `t_obs < t_s` rejection; explicit forcing-member-subset recording and a
seeds-length-mismatch rejection; and two independence tests (two candidates, two
historical timestamps) confirming no shared/leaked state between separate calls.

```text
python -m unittest tests.test_forward_propagation -v
Ran 21 tests in 0.45s
OK

python -m unittest discover -s tests -v   (full suite)
Ran 419 tests in ~12s
OK
```

### Manual-check notebook

`notebooks/T21_forward_propagation_check.ipynb` (new) — one synthetic 15-particle
candidate core forward-propagated under 3 forcing members via a single
`forward_propagate_candidate` call (no scientific logic duplicated in the notebook).
The figure shows the starting core (open circles at `t_s`), every particle's full
trajectory line (one color per forcing member), and each member's final positions at
`t_obs` (filled circles). Executed end-to-end via `jupyter nbconvert --execute` with
zero errors; in-notebook asserts confirm the recorded forcing-member subset and exact
termination at `t_obs` for every member. Closing "Interpretation limits" cell restates
that no KDE/error-metric/`c_hat*` is computed here, and that the trajectory colors
represent forcing-member divergence under a still-PROTOTYPE ensemble model, not a
validated uncertainty estimate.

### Known limitations / explicitly deferred (not implemented in T21)

- No predicted-slick/KDE/footprint construction (T22) or forward error/consistency
  metric (T23).
- No AIS or formal source inference.
- Never reads or references Stage III `c_hat*`.
- No forward-run death-event provenance (`death_lat`/`death_lon`/`death_reason` for a
  NEW invalidation during forward propagation) is tracked — same explicit deferral as
  T20; `first_invalid_step` lets a caller find WHEN a particle died from the recorded
  `valid` array, but not the exact invalidating position/reason.
- Does not itself loop over every candidate/timestamp in a full Phase 2 run — this
  module is the per-candidate orchestration layer; looping it across all retained
  candidates is left to whichever higher-level pipeline/task assembles the full run
  (consistent with "the smallest correct orchestration layer" scope given for T21).
- `iste.core_sampling` (T19), `iste.forward_transport` (T20), `iste.ensemble_run`
  (T09), and `iste.forcing_ensemble` (T08) were not modified — T21 only calls their
  existing public functions.

## Phase 2 — T22 — Predicted vs. observed slick footprint (PENDING_REVIEW)

### Goal

Per `tasks.md`'s T22 and `README.md` Sec. 20/21: at `t_obs`, build a predicted-slick KDE
footprint for every candidate/member from T21's forward particles, and an independent
observed-slick reference KDE from the upstream observation particles (T07) — as two
always-separate representations, comparable without requiring equal particle counts.

### Predicted KDE/footprint representation used

`iste/slick_footprint.py` (new): a standard isotropic 2-D Gaussian KDE (`gaussian_
kernel_2d`, PROTOTYPE/CONFIGURATION — explicit and pluggable via `FootprintGridConfig.
kernel_fn`), evaluated on a caller-supplied shared regular `lat_grid x lon_grid`:

```text
density(grid_point) = (1 / n_valid) * sum_over_valid_particles( gaussian_kernel_2d(distance_m, bandwidth_m) )
gaussian_kernel_2d(d, h) = exp(-0.5 * (d/h)^2) / (2*pi*h^2)
```

`distance_m` is always computed via `iste.kdtree_neighborhood.project_to_local_metres`
(T15's distance-safe local projection, reused unchanged) — never raw lat/lon degree
differences. Dividing by `n_valid` (not the total/raw particle count) is exactly what
decouples the footprint's shape/scale from particle count, satisfying "the comparison
must not require equal particle counts" (Sec. 20/22) and the task's own check
("particle-count changes do not by themselves change a normalized footprint score") —
proved exactly by `TestDifferentParticleCounts.test_exact_duplication_leaves_
normalized_density_unchanged` (an exact particle-set duplication leaves the density
bit-for-bit unchanged).

`PredictedSlickFootprint` carries `candidate_id`/`t_s`/`t_obs`/`forward_forcing_
member_id` (T21 provenance, preserved exactly) plus `density` (shape `(len(lat_grid),
len(lon_grid))`, row-major, matching T17's `support_grid` convention), `bandwidth_m`,
`n_particles`, `n_valid_particles`, `has_valid_particles`. Built ONLY from
`ForwardCandidateMemberRun.final_lat`/`final_lon`/`final_valid` (the `t_obs` checkpoint
T21 already guarantees is the trajectory's last row) — never from any earlier
checkpoint, and never from Stage III `c_hat*` (this module has no knowledge of it at
all).

### Observed-reference construction

`build_observed_slick_footprint(adapted: iste.observation.AdaptedParticleSet, t_obs,
config)` — uses ONLY `adapted.state.lat`/`state.lon`/`state.valid` (T07's upstream
observation particles, unmodified) through the SAME `_compute_density_grid` core as the
predicted footprint. `ObservedSlickFootprint` deliberately has NO `candidate_id`/
`forward_forcing_member_id` field at all (not merely left unset) — it is independent of
every candidate by construction. Per `README.md` Sec. 21 ("the backward historical
particles are not used as the independent truth reference"), `build_observed_slick_
footprint`'s only particle-bearing input type is `AdaptedParticleSet`; a source-level
`ast`-based test (`TestObservedReferenceIndependent`) confirms the module never imports
`iste.particle_query` or references `HistoricalParticleSlice`/`ParticleNeighborhoodIndex`
at all.

### Configuration

`FootprintGridConfig` (frozen dataclass): `lat_grid`/`lon_grid` (1-D, non-empty, SHARED
between the predicted and observed calls being compared so they land on identical grid
cells), `bandwidth_m` (metres, must be finite and `> 0` — an explicit Phase 2
calibration parameter, `README.md` Sec. 26, never a hidden constant), `lat_ref`/
`lon_ref` (local-projection reference point, defaults to the grid's own mean lat/lon),
`kernel_fn` (pluggable, defaults to `gaussian_kernel_2d`). All validated explicitly in
`__post_init__`.

### Tests

`tests/test_slick_footprint.py` (new) — 19 `unittest` cases covering all 10 requested
requirements: a synthetic cluster's footprint peak landing near the expected grid cell;
observed-reference independence (both a behavioral check and the `ast`-based source-
import check above); independent/different particle counts for predicted vs. observed,
plus the exact-duplication invariance proof; invalid particles proven to not affect
density at all (even when moved to a wildly different position); different bandwidths
producing different densities while the source `ForwardCandidateMemberRun`'s own
`lat`/`lon` arrays remain byte-unchanged; a hand-computed exact Gaussian value at a
known metre distance, plus a high-latitude test proving longitude-degree spacing is
correctly geometry-converted (not treated as raw metres); candidate/member provenance
preserved exactly, including the multi-member `build_predicted_slick_footprints`
convenience wrapper; deterministic repeated calls; explicit all-`NaN` + `has_valid_
particles=False` representation for both an all-invalid predicted run and an
all-invalid observed reference (never silently zero/valid); a controlled two-cluster
predicted-vs-observed example with visually/numerically distinct peaks (automated proxy
for the notebook's own visual inspection); and `FootprintGridConfig` validation.

```text
python -m unittest tests.test_slick_footprint -v
Ran 19 tests in 0.06s
OK

python -m unittest discover -s tests -v   (full suite)
Ran 438 tests in ~11-13s
OK
```

### Manual-check notebook

`notebooks/T22_slick_footprint_check.ipynb` (new) — an observed slick reference (40
synthetic upstream observation particles) alongside two synthetic candidates' predicted
footprints (standing in for real T21 output): candidate A placed close to the observed
slick, candidate B placed well away from it, so the predicted-vs-observed relationship
this layer exists to support is visually obvious. Three clearly-labeled, shared-colour-
scale panels (observed / candidate A / candidate B), each predicted panel titled with
its `candidate_id` and forcing-member identity, `t_obs` shown in the figure's overall
title. Every footprint is built by calling directly into `iste.slick_footprint`/
`iste.observation` — no logic duplicated in the notebook. Executed end-to-end via
`jupyter nbconvert --execute` with zero errors. Closing "Interpretation limits" cell
restates that this is a footprint-construction artifact, not a comparison result, and
that the predicted KDE is not a source-probability map.

### Known limitations / explicitly deferred (not implemented in T22)

- No predicted-vs-observed comparison/error metric (T23's scope) — footprints are built
  and kept strictly separate, never combined into a score here.
- No candidate ranking/filtering, AIS, or formal source inference.
- Never reads or references Stage III `c_hat*`.
- The Gaussian KDE kernel/bandwidth is an explicit PROTOTYPE/CONFIGURATION choice
  (`README.md` Sec. 26 lists "KDE bandwidth" as an open Phase 2 parameter) — not a
  scientifically validated density-estimation method; `kernel_fn` is pluggable
  specifically so this can be revisited without changing this module's structure.
- `_compute_density_grid`'s pairwise `(n_grid_points x n_valid_particles)` distance
  matrix is the simplest correct implementation, not optimized for very large grids or
  particle counts (no KD-tree/coarse-graining acceleration, unlike T15's own
  neighborhood engine) — acceptable at T22's current small-scale prototype stage; a
  future task can optimize this without changing the module's public interface if
  needed.
- `iste.forward_propagation` (T21), `iste.observation` (T07), and `iste.kdtree_
  neighborhood` (T15) were not modified — T22 only calls their existing public
  functions.

## Phase 2 — T23 — Forward-vs-observed accuracy (PENDING_REVIEW)

### Goal

Per `tasks.md`'s T23 and `README.md` Sec. 22: for each candidate/forcing member, an
explicit, configurable spatial-consistency error/score between T22's predicted and
observed footprints, retaining both a per-member error (lower = better) and consistency
(higher = better), never silently scoring an invalid forward result as a good match.

### Exact metric(s) implemented

`iste/forward_accuracy.py` (new) — five metric families (`README.md` Sec. 22's list,
"where practical"), each individually testable/documented, selected via
`ComparisonConfig.metric`:

```text
footprint_overlap    consistency = area-weighted IoU(pred_region, obs_region)         error = 1 - consistency
observed_coverage    consistency = area(pred_region ∩ obs_region) / area(obs_region)  error = 1 - consistency
false_positive_area  error       = area(pred_region \ obs_region) / area(pred_region)  consistency = 1 - error
spatial_distance      error       = metres(pred_centroid, obs_centroid)  consistency = exp(-error / consistency_distance_scale_m)
combined_overlap_fp  consistency = w_overlap*IoU + w_fp*(1 - false_positive_fraction)  error = 1 - consistency
```

`pred_region`/`obs_region` are each footprint's own `density` grid thresholded at
`region_threshold_fraction * (that footprint's own peak density)` — relative to its OWN
peak (predicted/observed footprints can have very different absolute density scales),
via a new `_footprint_region_mask` helper. Region/centroid AREA and DISTANCE arithmetic
always uses physically-correct metres — `_grid_cell_areas_m2` (new, via
`iste.geometry.metres_per_degree_latitude`/`metres_per_degree_longitude`, so cells
shrink correctly at higher latitude) for the three area-based metrics, and
`iste.kdtree_neighborhood.project_to_local_metres` (T15, reused) for `spatial_distance`'s
density-weighted-centroid distance. `combined_overlap_fp` is the one implemented
"documented combination" (Sec. 22's "or a documented combination") — it does not
duplicate `footprint_overlap`/`false_positive_area`'s own logic, only combines their
already-computed values with explicit weights.

### Score direction

`error`/`consistency` are always both returned and always complementary
(`error = 1 - consistency` for the four `[0,1]`-bounded metrics; `spatial_distance`'s
`error` is raw metres with its own independent `exp(-error/scale)` consistency
transform) — **lower error = better candidate, higher consistency = better candidate**,
unambiguous for every metric, verified directly by
`TestMetricDirectionUnambiguous`.

### Configuration

`ComparisonConfig` (frozen dataclass, all fields validated in `__post_init__`):
`metric` (required, one of `KNOWN_METRICS` — no default, always an explicit choice),
`region_threshold_fraction` (default `0.1`, must be in `(0, 1]`),
`consistency_distance_scale_m` (default `50_000.0` m, must be `> 0`),
`combination_weight_overlap`/`combination_weight_fp` (default `0.5`/`0.5`, must be
`>= 0`). All PROTOTYPE/CONFIGURATION calibration choices (`README.md` Sec. 22: "the
Phase 2 comparison metric is a configurable calibration component"), never claimed
scientifically validated.

### Tests

`tests/test_forward_accuracy.py` (new) — 23 `unittest` cases covering all 10 requested
requirements: identical footprints hit the best possible result for every metric;
clearly displaced footprints are worse than identical for every metric; partial overlap
sits strictly between the identical and fully-displaced extremes; an end-to-end
particle-count-independence proof using REAL T22 footprint construction (exact particle
duplication → identical comparison result for every metric, not just at the footprint
level); an all-invalid predicted footprint AND an all-invalid observed reference both
explicitly `valid=False` with `NaN` `error`/`consistency` (plus an explicit "never
silently 0/1" check); error/consistency complementarity for every metric, including
`spatial_distance`'s independent decay transform verified against a hand-computed
value; different metrics/threshold-fractions/combination-weights all deterministically
changing the result; candidate/member/timestamp provenance preserved, including the
multi-member batch helper; deterministic repeated calls; and — requirement 10 — each of
`footprint_overlap_iou`/`predicted_false_positive_fraction`/`observed_region_coverage`
tested standalone with hand-computed 3×3 grid examples, plus a test proving
`combined_overlap_fp`'s result matches a manual recombination of those already-tested
component values exactly. Also: a mismatched-grid rejection and full `ComparisonConfig`
validation.

```text
python -m unittest tests.test_forward_accuracy -v
Ran 23 tests in 0.02s
OK

python -m unittest discover -s tests -v   (full suite)
Ran 461 tests in ~11-14s
OK
```

### Manual-check notebook

`notebooks/T23_forward_accuracy_check.ipynb` (new) — the same observed-reference/
close-match/displaced-match synthetic scenario as T22's own notebook, extended with a
per-member metric table (all five metric families, both candidates) built entirely by
calling `iste.forward_accuracy.compare_predicted_to_observed` — no logic duplicated in
the notebook. Confirms the close match scores strictly better than the displaced match
on every metric (asserted in-notebook). Executed end-to-end via
`jupyter nbconvert --execute` with zero errors. Closing "Interpretation limits" cell
restates that this is a comparison-score artifact, not a ranking/filtering result, and
that every metric is an explicit calibration choice.

### Known limitations / explicitly deferred (not implemented in T23)

- No aggregation across forcing members into a single candidate-level score — that is
  T24's scope (`candidate_forward_error`/`candidate_forward_consistency`, arithmetic
  mean of valid per-member results).
- No candidate ranking/filtering (T25), AIS, or formal source inference.
- Never reads or references Stage III `c_hat*`.
- `region_threshold_fraction`'s peak-relative thresholding is a simple, explicit,
  PROTOTYPE region definition — not an adaptive/statistically-derived contour; a future
  task could revisit this without changing `ComparisonConfig`'s structure.
- `iste.slick_footprint` (T22) and every earlier Phase 2 module were not modified — T23
  only reads `PredictedSlickFootprint`/`ObservedSlickFootprint`'s existing public
  fields.

## Phase 2 — T24 — Candidate-level forward accuracy (PENDING_REVIEW)

### Goal

Per `tasks.md`'s T24 and `README.md` Sec. 23: for one candidate/time, aggregate its T23
per-forcing-member `ForwardAccuracyResult`s into a single candidate-level screening
score while retaining every member-level result unchanged, never silently turning an
invalid member into a numeric score, and explicitly marking a candidate
"forward-unvalidated" if no member was valid.

### Exact aggregation formula

```text
candidate_forward_error       = arithmetic_mean( error       for member in VALID members )
candidate_forward_consistency = arithmetic_mean( consistency for member in VALID members )

no valid members -> forward_validation_status = "unvalidated"; both aggregates = NaN (never a numeric fallback)
>= 1 valid member -> forward_validation_status = "validated"
```

Plain, unweighted `numpy.mean` over valid members only — no weighting by backward KDE
support (`support_score`/`density_score`/`weighted_support_score`/`pixel_diversity_score`,
T16/T17) or any other hidden factor. `iste/candidate_forward_accuracy.py` (new)'s
`aggregate_candidate_forward_accuracy` takes ONLY a `dict[int,
iste.forward_accuracy.ForwardAccuracyResult]` as input — it has no parameter through
which backward support data (or Stage III `c_hat*`) could even be passed in, so the
"do not mix backward KDE support into forward error" requirement is a structural
guarantee, not just a convention (`TestBackwardSupportCannotAlterAggregate` asserts the
function's own signature has exactly one parameter).

### Invalid-member handling

A member's own T23 `valid`/`invalid_reason` fields are read, never recomputed. Every
member id appears in EXACTLY ONE of `CandidateForwardAccuracy.forcing_members_validated`
(valid members, sorted) or `invalid_member_reasons` (invalid members, mapped to their
own T23 `invalid_reason` string) — never both, never neither. `member_results` retains
EVERY member's full, unmodified `ForwardAccuracyResult` (a shallow-copied dict, so later
mutation of the caller's own dict cannot retroactively change an already-returned
aggregate) regardless of validity, so no per-member diagnostic is ever lost even though
invalid members are excluded from the numeric mean.

### Tests

`tests/test_candidate_forward_accuracy.py` (new) — 18 `unittest` cases covering all 10
requested requirements: 2-member and 4-member exact arithmetic-mean checks; one-valid-
plus-one-invalid correctly excluding the invalid member; multiple invalid members'
reasons preserved by id; an all-invalid input giving `"unvalidated"`/`NaN` (plus an
explicit "never fabricates 0 or infinity" check); member-order independence (forward vs.
reversed dict insertion order give identical results); per-member values preserved
exactly (`assertIs` identity checks against the original `ForwardAccuracyResult`
objects, plus a mutate-after-call test proving the returned dict is a real copy); the
structural backward-support-independence proof (signature inspection + an identical-
values-different-candidate-identity test); two independent candidate/time groups not
interfering, plus three explicit consistency-validation rejections (mismatched
`candidate_id`/`t_s` across members, and a dict key not matching its own value's
`forward_forcing_member_id`); deterministic repeated calls; and an empty-input
rejection.

```text
python -m unittest tests.test_candidate_forward_accuracy -v
Ran 18 tests in <0.01s
OK

python -m unittest discover -s tests -v   (full suite)
Ran 479 tests in ~11-14s
OK
```

### Manual-check notebook

`notebooks/T24_candidate_forward_accuracy_check.ipynb` (new) — one candidate's 5
synthetic per-member T23 results (4 valid with distinct error/consistency values, 1
explicitly invalid), aggregated via `aggregate_candidate_forward_accuracy` (no logic
duplicated in the notebook). Grouped bar charts (error / consistency) show every
member-level value, the invalid member marked with a gray bar and "INVALID" label
excluded from the mean, and a dashed horizontal line for
`candidate_forward_error`/`candidate_forward_consistency` — visually distinguishing
member-level values from the aggregated candidate-level score. Executed end-to-end via
`jupyter nbconvert --execute` with zero errors; in-notebook asserts confirm the
aggregate matches a hand-computed mean and `forward_validation_status == "validated"`.
Closing "Interpretation limits" cell restates this is a screening-score aggregation, not
a filtering/ranking result, and that it never reads backward support or `c_hat*`.

### Known limitations / explicitly deferred (not implemented in T24)

- No candidate filtering/ranking — that is T25's scope (`README.md` Sec. 24:
  "Forward-propagation accuracy is the primary candidate-filtering and ranking signal").
- No formal likelihood/posterior inference or AIS.
- Never reads or references Stage III `c_hat*`.
- This is explicitly a Phase 2 **screening score**, not the later formal
  forcing-ensemble likelihood marginalization (`README.md` Sec. 23) — the plain
  arithmetic mean is a locked Phase 2 default, not a claim of statistical optimality.
- `iste.forward_accuracy` (T23) and every earlier Phase 2 module were not modified —
  T24 only reads `ForwardAccuracyResult`'s existing public fields.

## Phase 2 — T25 — Primary forward-accuracy candidate filtering/ranking (PENDING_REVIEW)

### Goal

Per `tasks.md`'s T25 and `README.md` Sec. 24: use T24's candidate-level forward
accuracy as the PRIMARY filter/ranking signal over the candidate set, in either
threshold or top-K mode, ranking survivors by ascending forward error, merging
near-duplicate survivors only after filtering, and never letting backward KDE support
influence the decision.

### Exact filtering mode(s)

`iste/candidate_screening.py` (new) — `ScreeningConfig.mode`, exactly one, required:

```text
"threshold"  reject every candidate with candidate_forward_error > error_threshold; retain the rest (<=  survives)
"top_k"      retain exactly the top_k VALIDATED candidates with the lowest candidate_forward_error
             (fewer than top_k valid candidates -> all of them, never an error)
```

Candidates with `forward_validation_status != "validated"` (T24: no valid forcing-member
result) are rejected BEFORE either mode runs and never reach the threshold/top-K
comparison at all — never a fabricated score or rank.

### Ranking behavior

Surviving candidates (after filtering AND merging — see below) are ranked by ascending
`candidate_forward_error`, tie-broken by `candidate_id` for full determinism; `rank` is
contiguous starting at `1`. `filter_and_rank_candidates` takes ONLY `list[CandidateRecord]`
+ `ScreeningConfig` — there is no parameter through which a backward support value
(`support_score`/`density_score`/`weighted_support_score`/`pixel_diversity_score`) could
reach the accept/reject or ranking logic, so "a candidate with weaker KDE support may
outrank a stronger KDE candidate" (`README.md` Sec. 24) holds structurally, not just by
convention (`TestBackwardSupportCannotChangeRanking` proves both the behavior and the
function's own signature).

### Duplicate-merge behavior

Near-duplicate merging runs strictly AFTER filtering, on survivors only, via a greedy
forward-error-ascending non-maximum suppression (the same structural pattern as T17's
own peak suppression, independently implemented here for candidate positions/forward
error rather than grid cells/support) using `iste.kdtree_neighborhood.
project_to_local_metres` (T15, reused) for physically-correct distances. Merging is
scoped to ONE historical timestamp `t_s` at a time — candidates from different `t_s` are
NEVER merged into each other regardless of spatial proximity, preserving distinct source
times unconditionally. `ScreeningConfig.merge_separation_m` defaults to `0.0` (no
merging at all); a merged-away candidate's `rejection_reason` names the surviving
candidate it was merged into, for auditability. Final `rank` numbers are assigned to the
POST-MERGE survivor set only, so a merged-away candidate never receives a rank.

### Provenance preserved

`CandidateRecord` carries every `README.md` Sec. 25 field (`candidate_id`/`x_peak`/
`y_peak`/`t_s`/`support_score`/`density_score`/`weighted_support_score`/
`pixel_diversity_score`/`core_definition`/`core_particle_count_Y`/
`forward_particle_count_N_forward`/`sampling_seed`/`forcing_members_validated`/
`forward_error_by_member`/`forward_consistency_by_member`/`candidate_forward_error`/
`candidate_forward_consistency`), assembled from the real T17 `Candidate`, T18
`ForwardBudget`, T19 `sampling_seed`, and T24 `CandidateForwardAccuracy` objects via the
new `build_candidate_record` helper (which cross-checks `candidate_id`/`t_s` agreement
across all four and raises on mismatch, never silently reconciling). `CandidateDecision`
wraps the SAME (never recomputed) `CandidateRecord` object with `accepted`/`rank`/
`rejection_reason` — `filter_and_rank_candidates` returns exactly one decision per input
record, so every rejected candidate (unvalidated, threshold/top-K-rejected, or merged)
remains fully inspectable for auditability, matching "preserve rejected candidates/
rejection reason where useful."

### Tests

`tests/test_candidate_screening.py` (new) — 24 `unittest` cases covering all 12
requested requirements: threshold-mode reject-above/retain-at-or-below (the `<=`
boundary tested explicitly); top-K keeping exactly K when enough valid candidates exist,
and all of them when fewer exist; ranking order by ascending error; unvalidated
candidates rejected with no rank in BOTH modes; multiple historical times and multiple
spatial modes both surviving; the backward-support-independence proof (behavioral +
signature-inspection); near-duplicate merging exercised from five angles (merge occurs
when configured, does NOT occur without a configured threshold, does NOT occur for
far-apart candidates even with a threshold, rejected/filtered-out candidates are never
considered for merging, and merging never crosses `t_s` groups); complete provenance
preservation (object-identity `assertIs` check); deterministic repeated calls and a
tie-break-by-`candidate_id` check; full `ScreeningConfig` validation; and an integration
test building a `CandidateRecord` from real T17/T18/T24 objects via
`build_candidate_record`, plus its mismatch-rejection path.

```text
python -m unittest tests.test_candidate_screening -v
Ran 24 tests in 0.005s
OK

python -m unittest discover -s tests -v   (full suite)
Ran 503 tests in ~11-14s
OK
```

### Manual-check notebook

`notebooks/T25_candidate_screening_check.ipynb` (new) — six synthetic candidates
exercising every part of T25's contract at once: a weak-backward-support/good-forward
candidate that survives and ranks first, a strong-backward-support/bad-forward
candidate that is rejected anyway (demonstrating forward accuracy overriding backward
support without ever mixing the two scores), two ~70 m-apart near-duplicates (the
better one survives, the other is merged into it), one fully unvalidated candidate
(rejected with no rank despite strong backward support), and one candidate at a
different historical timestamp (preserved independently). Two panels: a spatial map
(accepted = ranked circle, rejected = X, colored by `t_s`) and a forward-error-vs-
backward-support scatter with the configured threshold drawn as a vertical line,
visually proving the accept/reject split tracks only the x-axis (forward error). No
scientific logic duplicated in the notebook — every decision comes from
`filter_and_rank_candidates`. Executed end-to-end via `jupyter nbconvert --execute`
with zero errors; all five in-notebook assertions passed.

### Known limitations / explicitly deferred (not implemented in T25)

- No final formal source inference or AIS attribution — "the final Phase 2 artifact is
  a forward-screened candidate set, not a posterior" (`README.md` Sec. 25).
- No backward prefilter mechanism is implemented here — `README.md` Sec. 24 describes
  such a prefilter (if ever configured) as happening BEFORE forward validation, i.e. at
  candidate-selection time (T17/T18's scope), not inside this module; T25 simply has no
  parameter through which one could be injected.
- Filtering/ranking/top-K selection all operate GLOBALLY across every candidate/time in
  one `filter_and_rank_candidates` call (not per-`t_s` sub-selection) — an explicit,
  documented design choice; a per-timestamp variant could be added as a future
  `ScreeningConfig` option without changing this module's core logic.
- Never reads or references Stage III `c_hat*`.
- `iste.candidate_extraction` (T17), `iste.forward_budget` (T18), `iste.core_sampling`
  (T19), and `iste.candidate_forward_accuracy` (T24) were not modified — T25 only reads
  their existing public fields via `build_candidate_record`.

## Phase 2 — T26 — End-to-end integration and candidate artifact (PENDING_REVIEW)

### Goal

Per `tasks.md`'s T26 and `README.md` Sec. 29's locked compact contract: integrate
T14–T25 into ONE reproducible Phase 2 run, producing a machine-readable forward-screened
candidate artifact with complete provenance, over an explicit historical-timestamp grid,
without rerunning Stage III or duplicating any T14–T25 scientific logic.

### End-to-end pipeline executed

`iste/phase2_pipeline.py` (new) — `run_phase2_pipeline(records, base_ocean_source,
ensemble, k_lat, k_lon, k_field, adapted_observation, config: Phase2RunConfig) ->
Phase2Result`, calling, per historical timestamp then per candidate, exactly:

```text
T14 query_particle_slice -> T15 build_particle_neighborhood_index
  -> T16 compute_support_map -> T17 find_candidate_peaks
  -> [per candidate] T18 compute_forward_particle_budget -> T19 sample_core_particles
  -> T20/T21 forward_propagate_candidate -> T22 build_predicted_slick_footprints
     (+ build_observed_slick_footprint once, shared)
  -> T23 compare_predicted_footprints_to_observed -> T24 aggregate_candidate_forward_accuracy
  -> build_candidate_record (T25 input)
-> [after every timestamp] T25 filter_and_rank_candidates (once, over every candidate/time)
```

`iste.step`/`iste.trajectory` (Stage III) are never imported or called — `records` is an
already-computed `iste.ensemble_run.EnsembleRunRecords` batch, run exactly once by the
test/notebook setup (never per-candidate, never inside the pipeline itself).
`Phase2RunConfig` bundles every explicit `README.md` Sec. 26 parameter (historical
timestamp grid, KD-tree radius, support/extraction/forward-budget/comparison/screening
sub-configs, forcing-member set, forward `dt`, footprint grid/bandwidth, `master_seed`),
validated in `__post_init__` (non-empty timestamp grid/forcing-member set, positive
radii/dt).

`Phase2Result` retains every intermediate artifact (`TimestampPipelineResult`: `t_s`,
T14 `slice_`, T15 `index`, T16 `support_map`; `CandidatePipelineResult`: T17 `candidate`,
T18 `budget`, T19 `sampled`/`sampling_seed`, T21 `forward_runs`/`member_seeds`, T22
`predicted_footprints`, T23 `member_results`, T24 `forward_accuracy`, T25-input
`record`) alongside the shared T22 `observed_footprint` and the final T25 `decisions` —
so a notebook (or any other consumer) can inspect every stage without recomputing
anything, satisfying "call the integrated Phase 2 pipeline" literally for visualization.

### Reproducibility

Every T19/T21 RNG draw is derived from a single `config.master_seed` via
`numpy.random.SeedSequence.spawn` — the SAME deterministic-derivation idiom
`iste.forcing_ensemble.ForcingEnsemble`/`iste.ensemble_run.run_ensemble` already use
(not a new scheme). Seeds are spawned in one fixed order: historical timestamps in the
configured grid's own order; candidates within each timestamp in T17's own deterministic
descending-support order; forcing members within each candidate in the configured
`forcing_member_ids` order. No global/implicit RNG state is read anywhere in this
module. Verified directly: two independent end-to-end runs from completely separate
`ocean`/`ensemble`/`run_records`/`adapted` objects (built via two separate calls to the
same scenario-construction helper) produce byte-identical `decisions` (candidate_id,
accepted, rank, error, sampling_seed all equal) — see `TestReproducibility` below.

### Machine-readable artifact

`serialize_phase2_artifact(decisions, config) -> dict` / `save_phase2_artifact(...,
path)` / `load_phase2_artifact(path) -> dict` — a plain JSON object with every
`README.md` Sec. 25 required field per candidate (`x_s`/`y_s` matching the README's own
naming, `core_definition` as a nested `{center_lat, center_lon, radius_m}`, `t_s` as an
ISO datetime string, `accepted`/`rank`/`rejection_reason` from T25), plus a `config`
block recording every `Phase2RunConfig` field (including `master_seed`) for
reproducibility. `NaN` (an unvalidated candidate's `candidate_forward_error`/
`candidate_forward_consistency`) round-trips losslessly through Python's `json` module
(`allow_nan=True`, the default).

### Configuration used (manual-check notebook / integration tests)

```text
LAT=15..26 (23 pts), LON=95..106 (23 pts); ocean uo=0.15 m/s, vo=0.05 m/s, 8 hourly checkpoints
ForcingEnsembleConfig(n_members=4, spatial_correlation_km=40, correlation_time_s=7200, sigma_u=0.05, coarse_resolution_deg=0.5, master_seed=55)
K = 1.0 m^2/s (uniform)
upstream observation: 2 clusters (px-0 ~20.0N/100.0E n=12, px-1 ~22.5N/103.0E n=10) at t_obs
Stage III: run_ensemble(forcing_member_ids=[0,1], n_steps=4, dt=900s, master_seed=99)
historical_timestamps = (t_obs-900s, t_obs-1800s)
neighborhood_radius_m=15000, support_config=SupportFeatureConfig() defaults
extraction_config=CandidateExtractionConfig(peak_neighborhood_cells=3, min_separation_m=50000, core_radius_m=15000)
forward_budget_config=ForwardBudgetConfig(X=8); forcing_member_ids=(0,1); forward_dt_seconds=900
footprint grid = support grid; footprint_bandwidth_m=5000
comparison_config=ComparisonConfig(metric="combined_overlap_fp")
screening_config=ScreeningConfig(mode="threshold", error_threshold=0.35, merge_separation_m=10000)
master_seed=42
```

### Tests

`tests/test_phase2_pipeline.py` (new) — 14 `unittest` cases covering all 10 focused
requirements (requirement 11, the full regression suite, is a separate `unittest
discover` run, not a single test function): full pipeline execution; every stage proven
invoked via `isinstance`/structural assertions on `Phase2Result`'s nested dataclasses;
both configured historical timestamps represented in both `result.timestamps` and
`result.decisions`; both spatial clusters (two well-separated synthetic upstream
observation groups) surviving as 4 distinct accepted candidates across 2 timestamps;
forcing-member provenance (`forcing_members_validated` and each `ForwardCandidateRuns.
forcing_member_ids`) matching the configured set; two independent reproducibility tests
(same objects reused, and completely independent object construction); a full JSON
serialize/reload round-trip checking every required field's value against the original
in-memory decision (with correct `NaN` handling); a plain-`json.load` independent
readability check; every accepted candidate's complete forward-validation provenance
(non-empty `forcing_members_validated`, finite error/consistency, a real rank); an
unvalidated-candidate scenario (tiny `core_radius_m` relative to the peak-detection
radius, forcing `Y=0` via grid-quantization) confirming `NaN` scores/no rank/no silent
zero; a full manual replay of one candidate's ENTIRE T18→T24 chain via the individual
functions directly (same seeds), confirming byte-identical results against the
pipeline's own computation; and `Phase2RunConfig` validation.

```text
python -m unittest tests.test_phase2_pipeline -v
Ran 14 tests in ~11-12s
OK

python -m unittest discover -s tests -v   (full regression suite, requirement 11)
Ran 517 tests in ~23-28s
OK
```

### Notebook/artifact

`notebooks/T26_phase2_pipeline_check.ipynb` (new) — runs a tiny real Stage III backward
pass (`run_ensemble`, unmodified, executed exactly once) then the full integrated
pipeline over 2 historical timestamps × 2 spatial clusters, and visualizes, in one
figure, every stage the task asked for: the T16/T17 support/KDE map with candidate
peaks/cores; T19 selected source-core particles plus T20/T21 forward trajectories, with
a `Y`/`X`/`N_forward` table for every candidate/time; the T22 observed reference next to
each candidate's predicted footprint; a T23 per-member forward-accuracy table; and a
combined T24/T25 table showing candidate-level aggregate error/consistency alongside
final accept/reject status, rank, and rejection reason (the configured threshold is
deliberately tight enough that 2 of the 4 candidates are rejected, so both outcomes are
visible). A final cell serializes and reloads the artifact and asserts every required
field is present. No scientific logic is duplicated anywhere in the notebook — every
plotted value comes directly from `run_phase2_pipeline`'s own return value. Executed
end-to-end via `jupyter nbconvert --execute` with zero errors. Closing "Interpretation
limits" cell explicitly separates backward support (provenance) / forward consistency
(the primary screening signal) / candidate filtering (a forward-screened set, not a
posterior).

### Known limitations / explicitly deferred (not implemented in T26)

- No formal source inference or AIS attribution — the output is a forward-screened
  candidate set, handed to a later, not-yet-implemented stage.
- Per-member forward-propagation RNG seeds (`CandidatePipelineResult.member_seeds`) are
  retained in the in-memory `Phase2Result` for audit/notebook convenience but are NOT
  written into the JSON artifact itself (only `config.master_seed` is) — full
  reproducibility of any run follows from `master_seed` plus the documented
  deterministic spawn order, so this is not a reproducibility gap, but a consumer
  wanting each individual seed explicitly listed per candidate/member in the artifact
  itself would need a small schema extension.
- `FootprintGridConfig.kernel_fn` (a Python callable) is not itself part of the
  serialized artifact's `config` block (T22's own config isn't separately echoed there,
  since the pipeline builds it internally from `footprint_bandwidth_m`/grids already
  recorded) — not a gap for this task's own required field list, noted for completeness.
- Never reads or references Stage III `c_hat*`.
- `iste.particle_query` (T14) through `iste.candidate_screening` (T25) were not
  modified — T26 only calls their existing public functions.

## Real-data validation notebook — `main_actual_data_pipeline.ipynb` (not a numbered task; user-directed real-data run)

### Goal

Run the full T07 + Stage III + T14–T25 pipeline end-to-end on the actual upstream oil-
slick detection data (not synthetic), reusing every existing `.py` module unchanged, as
a real-data validation of the whole Phase 2 build (T14–T26).

### Real input used

`data/database_payload.json`'s `ow-0008.jpg` spill record (also present standalone as
`image filename ow 0008.txt`): 548 real detected oil parcels (`id`/`lat`/`lon`/
`mass_kg`/`confidence`), true detection timestamp `2019-06-14T20:35:58`, bounding box
`(34.0075–34.1424°N, 35.2081–35.3200°E)`.

### KNOWN DATA MISMATCH (discovered, confirmed with the user before proceeding)

`data/ocean.nc`/`data/wind.nc` (the only real ocean-current data in the repository)
cover `2019-10-01`–`2019-10-14` only. The true `ow-0008` observation is from June 2019;
none of the 5 real spill records in `database_payload.json` (Jan/Apr/Jun/Sep/Dec 2019)
fall inside the Oct 1–14 window either. Per the user's explicit choice (asked via
`AskUserQuestion` given this project's "do not use synthetic fallback data if real data
fails, stop and report" instruction), the run proceeds using the real 548 parcels
together with the only real ocean data available, with `t_obs` set to `ocean.nc`'s own
last checkpoint (`2019-10-14T23:00:00`) instead of the true observation time — loudly
labeled throughout the notebook as a pipeline-mechanics validation, **not** a physically
accurate reconstruction of the true June 2019 event.

### What the notebook does

`notebooks/main_actual_data_pipeline.ipynb` (new) — pure orchestration/visualization,
zero scientific logic duplicated: T07 `adapt_upstream_particles` on the real 548 parcels
(`particle_id`/`observation_pixel_id` = the parcel's own real `id`, since each parcel is
already one real mask-segmentation cell) → `iste.ensemble_run.run_ensemble` (Stage III,
unmodified) for ALL 548 real particles × 6 forcing members × 16 backward steps (dt=30
min, 8 h window) → saves the COMPLETE real particle history (not just a demo subset) via
NPZ (T11's `ParticleRecordStore` plus on-disk persistence) → selects and backward-tracks
10 real particles deterministically (`numpy.random.default_rng(2024)`), with a saved
MP4 animation → T14→T15→T16→T17 sweep over the real Stage III checkpoint schedule itself
(16 historical timestamps, no invented spacing) on a data-driven grid (observation bbox
∪ actual real backward-particle extent, padded) → selects the top 10 candidates by
T17's own `support_score` field (not a new ranking) → `iste.phase2_pipeline.
run_phase2_pipeline` (T26, unmodified) restricted to those candidates' own historical
timestamps for T18–T25 → predicted-vs-observed visualization (best/worst of the 10) →
final T25-ranked top-10 table → `save_phase2_artifact` (T26's own serialization,
reused, not a new format).

### Configuration used

```text
forcing ensemble: 50 members configured (README.md's own default); 6 used (0-5), an EXPLICIT
  RECORDED development subset (real 548-particle × 50-member × many-step run is impractical
  in this session; never a silent substitution)
K = 3.0 m^2/s (explicit PROTOTYPE constant, no real K field exists)
dt = 1800 s, n_steps = 16 (backward), forward dt = 1800 s (to t_obs)
neighborhood_radius_m = core_radius_m = 1000 m; peak_neighborhood_cells=5; min_separation_m=1500 m
footprint_bandwidth_m = 400 m; comparison metric = "combined_overlap_fp"
X (forward particle budget) = 20; screening: top_k=10, merge_separation_m=1500 m
master_seed = 42 (Stage III), 123 (Phase 2 T19/T21 seed derivation)
```

### Result (one real run, this session)

548/548 particles adapted; Stage III backward run: 55,896 rows, 100% valid at every
checkpoint (calm real Oct 2019 currents, no boundary deaths); T14–T17 found 16 real
hotspots (exactly 1 per historical timestamp — the real detected slick stays one
cohesive spatial mode across the 8 h window, a genuine finding, not a code limitation);
all 10 selected candidates forward-validated and accepted, forward error ranging
0.171 (best, closest-in-time candidate) to 0.429 (worst, furthest-back candidate) —
a physically sensible monotonic-ish trend. Total pipeline runtime ≈ 121 s (Stage III
109 s + T14–T17 5.3 s + T18–T25 6.1 s). Executed via `jupyter nbconvert --execute` with
a 900 s timeout; zero cell errors; all 8 end-to-end validation checks (Section 14 of the
request) passed.

### Outputs saved (`results/actual_data/`, 14 files)

`observation_summary.json`, `stage3_particle_history.npz` (complete real history, ~1.0
MB), `stage3_history_summary.txt`, `demo10_backward_trajectories.npz`,
`03_backward_animation.mp4`, `01_observation.png`, `02_backward_trajectories_10particles.png`,
`04_hotspot_heatmap.png`, `05_predicted_vs_observed.png`, `06_final_top10_table.png`,
`hotspots.json` (all 16), `final_top10_candidates.json` (T26's own serialization format),
`07_multimember_static.png`, `08_multimember_animation.mp4`.

### Follow-up: the same 10 real particles under multiple forcing members (user-requested)

Added after the initial run: a static 6-panel comparison (one panel per forcing member,
same particle = same color across panels) and a synchronized 6-panel MP4 animation,
both showing the SAME 10 demonstration particles' backward trajectories under **all 6**
configured forcing members (0–5) at once — directly visualizing the spread introduced
purely by each member's coherent perturbation (`delta_U_e`, T08). Reuses `records`
(already computed by the single real Stage III run — no additional backward transport
was run); purely a notebook-level visualization addition, no `.py` module touched.
Re-executed the full notebook end-to-end afterward (`jupyter nbconvert --execute`,
900 s timeout) — zero errors, all 14 output files present, full regression suite
still 517/517 passing.

### Follow-up 2: extended 3-day/1-hour backward window + Monte Carlo density heatmap + top-10 details (user-requested)

The user asked for the full 7/14-day backward window at 30-min steps; given the ~75-90 min
cost of the full 672-step/30-min run over 548 real particles, `AskUserQuestion` offered three
scoped alternatives and the user chose a **3-day backward window at 1-hour steps (72 steps)**
— an explicit, recorded reduction, not a silent one. Rebuilt the notebook
(`build_main_notebook_v3.py`) with `DT_SECONDS=3600.0`, `N_STEPS=72`, plus:

- **I/O robustness**: `save_with_retry`/`save_animation_with_retry`/`write_json_with_retry`
  helpers wrap every file write in the notebook, retrying on transient `OSError`s caused by
  OneDrive's background sync locking a just-written file (hit twice during earlier attempts,
  on two different unrelated output files — confirmed transient, not a code bug, since
  identical code had succeeded in prior runs). This is the only non-scientific helper code in
  the notebook.
- **Stage III result caching**: before rerunning the ~15-20 min real backward computation, the
  notebook checks for an existing `stage3_particle_history.npz` matching the exact expected
  row count (`548 * 6 * 73 = 240,024`) and loads it instead of recomputing. This reuses the
  SAME already-computed real result (not synthetic fallback data) purely to avoid re-paying an
  expensive deterministic computation after an unrelated I/O hiccup — on this run, an earlier
  interrupted attempt had already produced and saved a valid 240,024-row cache, so the full
  rerun completed in ~4 minutes wall-clock (cache hit) instead of ~15-20 minutes.
- **New Section 8 — Monte Carlo particle density vs. T16 support/KDE heatmap**
  (`09_montecarlo_density.png`): a 2×2 panel at two representative historical timestamps
  (nearest to `t_obs`, furthest back in the 3-day window) showing, top row, a plain
  `matplotlib.pyplot.hist2d` binning of every valid particle's position across all 6 forcing
  members combined (pure descriptive statistics, not a KDE/scientific algorithm) and, bottom
  row, the SAME particles through the existing unmodified `iste.support_features.
  compute_support_map` (T16). Both panels use the same real particle positions; only the
  representation differs, and neither is a probability density.

Re-executed via `jupyter nbconvert --execute` (3600 s timeout); zero cell errors (confirmed via
both the true unpiped exit code and absence of any `CellExecutionError`/error-output cell in
the saved notebook). Full regression suite re-run: 517/517 passing; no `.py` module touched.

### Result (3-day/72-step run, this session)

T14-T17 sweep over 72 real historical timestamps (112.8 s) found 280 real hotspots/candidates
(vs. 16 in the original 16-step run — expected, since the window is now 4.5x longer). Top 10
selected by `support_score` all cluster on the single most recent historical timestamps
(`2019-10-14T13:00`-`22:00`, i.e. closest to `RUN_T_OBS`) — consistent with support naturally
concentrating near the anchor time. All 10 forward-validated and accepted (T18-T25, 22.3 s);
`found_ids == top10_ids` assertion confirmed the pipeline processed exactly the pre-selected 10,
no extras. Forward errors ranged 0.2255 (rank 1, `2019-10-14T22:00:00_cand000`, lat=34.0735,
lon=35.2681) to 0.6807 (rank 10, `2019-10-14T16:00:00_cand000`, lat=34.0885, lon=35.3131) — a
mostly monotonic trend with time-distance from `t_obs`, physically sensible for this real-but-
mismatched-forcing scenario. All 9 end-to-end validation checks (including the new montecarlo-
density-heatmap check) passed.

### Outputs saved (`results/actual_data/`, 15 files — adds `09_montecarlo_density.png`)

Same 14 files as the initial run (see above; `stage3_particle_history.npz`, `hotspots.json`,
`final_top10_candidates.json`, etc. now reflect the 3-day/72-step/280-candidate data) plus
`09_montecarlo_density.png` (new Section 8).

### Follow-up 3: visible land/invalid-domain background layer (user-requested)

The user asked whether land is being ignored in the trajectory plots/animations. Answer:
the underlying boundary policy (T04's `apply_boundary_policy`/`netcdf_validity_check`,
unmodified, called inside every Stage III step) is fully active and would freeze any
particle stepping onto land — it just never triggered in this run (100% valid at all 73
checkpoints; the real spill's AOI, padded, contains 0 land/invalid cells out of `ocean.nc`'s
8,853 total land/invalid-domain cells). But the plots themselves drew no visible land layer
at all, so this was invisible either way. Added `add_land_background(ax)` (a helper computing
`land_mask_grid = np.all(np.isnan(ocean.data["uo"]), axis=0)` — a cell NaN at every one of
the 336 hourly timesteps, i.e. the SAME mask T04's boundary policy already checks against,
not a new land model) and applied it as a gray background layer to the single-member static
trajectory plot (`02_...png`), single-member animation (`03_...mp4`), multi-member static
comparison (`07_...png`), and multi-member animation (`08_...mp4`). Not added to
`01_observation.png` (would require loading `ocean.nc` before Section 1, reordering the
notebook) or to `04_hotspot_heatmap.png`/`05_predicted_vs_observed.png`/
`09_montecarlo_density.png` (their own dense pcolormesh/hist2d layers already cover the full
plotted AOI, so a land layer underneath would be invisible there anyway). Re-executed
end-to-end (cache hit, ~2 min); zero errors; regression suite 517/517 passing; no `.py`
module touched.

### Known limitations

- The environmental-forcing/observation date mismatch above is the dominant limitation —
  this run is a genuine real-DATA (real parcels, real currents) but NOT real-EVENT
  (mismatched dates) validation.
- The 3-day/1-hour backward window is an explicit user-scoped reduction from the full 14-day
  dataset (672-step/30-min would take ~75-90 min); recorded for auditability, not silent.
- Only 1 spatial mode found per historical timestamp in this particular real case (no
  multimodal real example was available in the current dataset/time window) — T17's own
  multimodal capability is separately validated on synthetic data
  (`notebooks/T17_candidate_extraction_check.ipynb`).
- This real spill's AOI never actually contains land, so the new land-background layer is
  present but visually blank (all-water) in every current output file — it would only show
  gray shading for a different AOI/spill location that does border land.
- The trajectory/animation plotting code still does not visually distinguish a particle that
  goes invalid (frozen per T04) from one still normally propagating — flagged to the user,
  not yet implemented (user chose to add only the land background, not this, in this
  follow-up).
- No `.py` module was modified for any part of this real-data notebook work (original run,
  multi-member follow-up, 3-day/density-heatmap follow-up, or this land-background follow-up).

## Verified datasets

- Ocean NetCDF: `uo`, `vo`, 336 hourly timestamps, 14 days.
- Wind NetCDF: `u10`, `v10`, same 336 hourly timestamps, 14 days.
- Timestamps align exactly.
- Ocean latitude ascending; wind latitude descending.
- Ocean fill values around `1e20` must be masked.
- Ocean and wind grids have different spatial resolutions.
- No diffusivity (`K`) field exists in either file — must be supplied explicitly by config.

## ISTE architecture (current, replaces the old "Locked prototype design" RLPT diagram)

Full pipeline per `README.md` (first implementation milestone is Stage III only):

```text
Sentinel-1 observation
        ↓
Stage I   — Observation Operator                    [future, blocked]
        ↓
Stage II  — Coherent Forcing Ensemble                [future, blocked]
        ↓
Stage III — Weighted Backward Adjoint                [CURRENT — T01-T13]
        ↓
Stage IV  — Surrogate Likelihood                     [future, blocked]
        ↓
Stage V   — Grid Screen + Tempered SMC                [future, blocked]
        ↓
Stage VI  — Delayed-Acceptance MCMC                  [future, blocked]
        ↓
Stage VII — Time-indexed Posterior / AIS-ready output [future, blocked]
```

Stage III mechanics (see `memory.md` for the full detail and equations):

```text
per observation pixel j, per forcing member e
        ↓
particle released, backward-adjoint SDE
  dX* = (-U + grad(K)) dtheta + sqrt(2K) dB
        ↓
dynamic weight: d(ln w)/dtheta = -div(U)
        ↓
explicit boundary/validity policy (stop-on-invalid)
        ↓
checkpointed particle records (~15-30 min), retaining
  observation_pixel_id, forcing_member_id, weight, validity
        ↓
Monte Carlo transport estimate c_hat*_{j,e}(x,t)   [not an exact probability]
        ↓
diagnostics: total/valid/invalid particles, total weight, ESS, valid_particle_fraction
```

Stage IV–VII are not designed in detail yet; they remain blocked until Stage III (T00–T13) is
signed off, per `README.md`'s Scope rule and `CLAUDE.md`.

## Repository implementation rule

```text
.py  → reusable scientific/application logic
.ipynb → experiments, plots, debugging, visual inspection
```

Do not put the core model exclusively into notebooks.

## Task status

- T00 — Inspect repository: `DONE`
- T01 — Environmental field loader + Stage III interface: `DONE`
- T01.5 — Define transport velocity U: `DONE` (locked: `U = (uo, vo)`, windage excluded from
  v1, see above)
- T02 — Interpolation layer: `DONE`
- T03 — Spatial derivative operators (grad(K), div(U)): `DONE`
- T04 — Boundary and particle-validity policy: `DONE`
- T05 — One backward Euler-Maruyama step: `DONE`
- T06 — Single-particle backward trajectory: `DONE`
- T07 — Backtracking input adapter / particle interface: `DONE`
- T08 — Forcing ensemble: `DONE` (prototype coarse-grid/OU/smoothing surrogate — 20 km /
  1 day / 0.05 m/s explicitly labeled PROTOTYPE, NOT VALIDATED)
- T09 — Small multi-particle, multi-forcing-member run: `DONE`
- T10 — Stage III diagnostics module: `DONE`
- T11 — Memory-efficient lookup-table storage: `DONE`
- T12 — Stage III controlled numerical validation suite: `DONE`
- T13 — Stage III sign-off artifact: `DONE` — **Stage III signed off 2026-09-22**
- T13.5 — Stage III → Stage IV compatibility correction: `DONE` — approved 2026-09-22
  (see T13.5 section above)
- **2026-09-22 — project pivot:** `README.md`/`tasks.md` were rewritten to replace the
  Stage IV surrogate-likelihood/SMC/MCMC plan with **Phase 2 — Candidate generation +
  forward validation**; task numbers were reassigned. The items below reflect the
  *current* `tasks.md`.
- T14 (superseded numbering) — Stage IV surrogate-likelihood core: `DONE` (formula since
  corrected to the frozen spec's ordering; see the addendum in its section above) —
  no longer an active task slot.
- T15 (superseded numbering) — Stage IV observation-model bridge (`h_j,e` → `pi_j,e`):
  `PENDING_REVIEW` — no longer an active task slot.
- **T14 (current numbering) — Historical particle slice / Phase 2 input adapter:**
  `PENDING_REVIEW` (awaiting your sign-off; see "Phase 2 — T14" section above)
- **T15 (current numbering) — KD-tree neighborhood engine:** `PENDING_REVIEW` (awaiting
  your sign-off; see "Phase 2 — T15" section above)
- **T16 (current numbering) — Weighted support-feature construction (D/W/V/Support):**
  `PENDING_REVIEW` (awaiting your sign-off; see "Phase 2 — T16" section above)
- **T17 (current numbering) — Candidate peak and candidate-core extraction:**
  `PENDING_REVIEW` (awaiting your sign-off; see "Phase 2 — T17" section above)
- Phase 2 T18–T26 (core sampling, forward transport, forward propagation, predicted/
  observed KDEs, forward accuracy, aggregation, filtering/ranking, end-to-end artifact):
  not yet started

## Phase 3 — Dispersion-aware, persistence-aware source ranking

> Added 2026-09-24 per explicit user request, after the real-data notebook surfaced a
> genuine ranking bias: raw KDE/support magnitude is structurally highest near `t_obs`
> (particles still tightly clustered around the observed slick) and decays as particles
> disperse further back, so naively ranking by raw support favors the initial condition
> over the true historical source. Full 13-task breakdown (T27-T39) added to `tasks.md`
> under "Phase 3". This phase is additive on top of the unchanged, already-implemented
> T14-T26 pipeline; it does not modify Stage III or silently combine backward support into
> T25's forward-accuracy score (that composite step is explicit, in T36, later).

### T27 — Extend real-data analysis grid to 73 hourly timestamps (T0-T72) (PENDING_REVIEW)

### Files added

- `iste/analysis_grid.py` — `generate_analysis_grid(t_obs, window_hours=72.0,
  interval_hours=1.0)`: pure timestamp-grid generation only (no particle/KDE logic, no
  Stage III module imported). Returns `T0..T_{window_hours}` as a strictly-descending
  `datetime64[ns]` array, `T0 = t_obs`, spacing = `interval_hours`. Validates
  `window_hours`/`interval_hours` are positive and that `window_hours` is an exact multiple
  of `interval_hours` (raises `ValueError` otherwise, not silent rounding). The physical
  Stage III integration timestep is completely separate and unconstrained by this function
  — a caller with finer physical checkpoints queries this grid's timestamps via the
  existing, unmodified T14 `iste.particle_query.query_particle_slice`, which already
  interpolates; no second interpolation exists here.
- `tests/test_analysis_grid.py` — 13 `unittest` cases: default 72h/1h grid has exactly 73
  entries, `T0 == t_obs`, `T72 == t_obs - 72h`, strictly 1-hour descending spacing, correct
  dtype, a custom 6h/2h configuration (4 timestamps), a fractional-hour interval (0.5h),
  rejection of a non-divisible window, rejection of zero/negative `window_hours`/
  `interval_hours`, and string-convertible `t_obs` input.
- `notebooks/T27_analysis_grid_check.ipynb` — manual-check artifact (see below).

### Tests

```text
python -m unittest tests.test_analysis_grid -v
Ran 13 tests in 0.001s
OK

python -m unittest discover -s tests -v   (full suite)
Ran 530 tests in 16.302s
OK
```

### Manual check

`notebooks/T27_analysis_grid_check.ipynb` generates the default 72h/1h grid against the
real dataset's `RUN_T_OBS` (`2019-10-14T23:00:00`), prints all 73 timestamps, confirms
`len==73`/`T0==t_obs`/`T72==t_obs-72h`/exact 1-hour spacing, cross-checks the generated
grid against the real Stage III checkpoints already cached on disk
(`results/actual_data/stage3_particle_history.npz`, from the earlier 3-day/1-hour real run)
— **exact match confirmed** (both are 73 timestamps, identical set) — demonstrates a
non-default 6h/2h configuration (4 timestamps), and confirms malformed configuration
(non-divisible window, non-positive window/interval) is rejected with `ValueError`. All
cells executed via `jupyter nbconvert --execute`, zero errors:

```text
n analysis timestamps: 73  (expected 73)
T0  (observed slick): 2019-10-14T23:00:00.000000000
T1  (1h backward):    2019-10-14T22:00:00.000000000
T72 (72h backward):   2019-10-11T23:00:00.000000000
assertions passed: exactly 73 timestamps, T0 == t_obs, T72 == t_obs - 72h
spacing (hours) -- min=-1.0, max=-1.0 (expect exactly -1.0 everywhere)

real Stage III checkpoints on disk: 73
T27 grid matches real Stage III checkpoints exactly: True

window_hours=6, interval_hours=2 -> 4 timestamps (expect 4)
correctly rejected non-divisible window: window_hours (72.0) must be an exact multiple of interval_hours (5.0); got 14.4 steps, which is not an integer
correctly rejected non-positive window_hours: window_hours must be > 0, got -1.0
```

### Known limitations

- Timestamp-grid generation only — no diagnostics, KDE, normalization, persistence,
  convergence, dispersion, screening, AIS, or composite-score logic (T28-T39, not yet
  started).
- No Stage III module was read or modified.
- The cross-check against real Stage III checkpoints matches exactly only because the
  existing real-data run happens to also use 1-hour physical checkpoints; the module itself
  does not depend on or assume this coincidence (T14's existing interpolation would bridge
  a finer physical grid, not this module).

**T27 approved 2026-09-24** (implicit, via "proceed with T28").

### T28 — Per-timestamp diagnostics module (PENDING_REVIEW)

### Files added

- `iste/phase3_diagnostics.py` — `TimestampDiagnostics` dataclass +
  `compute_timestamp_diagnostics(records, t_s, *, support_lat_grid, support_lon_grid,
  neighborhood_radius_m, support_config=None, extraction_config=None)` +
  `compute_diagnostics_table(records, analysis_grid, ...)`. Computes, per timestamp: valid
  particle count, lon/lat mean/variance, 2x2 covariance matrix, RMS spatial spread from
  centroid (metres, via T15's existing `project_to_local_metres` — no new distance
  formula), convex-hull area (`scipy.spatial.ConvexHull`, `None` for <3 particles or
  degenerate/collinear points, never fabricated), and five T16-support-grid summaries:
  `kde_max`, `kde_integrated_mass` (grid-cell-area-weighted sum of the clipped-nonnegative
  support grid), `kde_entropy` (Shannon entropy, nats, of that grid normalized to sum to
  1), `kde_effective_area_m2` (`exp(entropy) * cell_area`), and `kde_bandwidth_m` (simply
  the `neighborhood_radius_m` already passed to T16's `compute_support_map`, echoed
  through — not a new bandwidth concept). `n_significant_modes` reuses T17's
  `find_candidate_peaks` directly (candidate count IS the mode count; no second peak
  detector). This project has no separate literal Gaussian KDE — per the module's own
  docstring, "KDE" here means exactly T16's existing `Support(x,y,t)=f(D,W,V)` field,
  consistent with how the real-data notebook already used that term. T14/T15/T16/T17 are
  all called unmodified; no Stage III module is imported.
- `tests/test_phase3_diagnostics.py` — 13 `unittest` cases: a synthetic tight cluster vs. a
  synthetic dispersed cluster (same particle count/centroid) produces the expected ordering
  on every diagnostic (dispersed has larger spread/variance/entropy/effective-area/hull-
  area, tighter has higher `kde_max`); empty-particle-set returns explicit NaN/None/0
  without crashing; <3 particles and exactly-collinear particles both correctly return
  `convex_hull_area_m2=None`; two well-separated synthetic clusters yield
  `n_significant_modes >= 2`; `kde_bandwidth_m` echoes the configured radius exactly;
  `compute_diagnostics_table` preserves grid length/order.
- `notebooks/T28_phase3_diagnostics_check.ipynb` — manual-check artifact (see below).

### Tests

```text
python -m unittest tests.test_phase3_diagnostics -v
Ran 13 tests in 0.259s
OK

python -m unittest discover -s tests -v   (full suite)
Ran 543 tests in 27.991s
OK
```

### Manual check

`notebooks/T28_phase3_diagnostics_check.ipynb` loads the real cached Stage III result (548
particles x 6 members x 73 checkpoints), regenerates the T27 grid from its own `t_obs`,
computes the full 73-row diagnostic table (110.5s), prints it, plots each diagnostic vs.
hours-before-observation, and reports explicit correlation statistics. All cells executed
via `jupyter nbconvert --execute`, zero errors. **Confirms the user-reported bias**
quantitatively on real data:

```text
correlation(hours_before_obs, spatial_spread_km)  = +0.986  (spread grows with age)
correlation(hours_before_obs, kde_max)             = -0.909  (KDE max falls with age)
spread_km  at T0=0.9149, at T72=5.0139  (5.5x growth)
kde_max    at T0=4745.00, at T72=855.31  (~82% drop)
```

This is the exact pattern the user described (tight/high-KDE near T0, dispersed/lower-KDE
further back) — now measured, not just observed visually, on this real dataset.

### Known limitations

- Pure diagnostics — no normalization (T29), initial-condition correction (T30),
  persistence (T31), convergence (T32), dispersion-aware scoring (T33), screening
  normalization (T34), AIS (T35), composite score (T36), clustering (T37), uncertainty
  (T38), or comparison visualization (T39) implemented yet — T28 measures the bias, it does
  not correct it.
- `kde_entropy`/`kde_effective_area_m2` are descriptive-statistics summaries of the T16
  support grid, not a claim that `Support` is a validated probability density.
- No Stage III module, and no existing T14-T17/T25/T26 module, was modified.

**T28 approved 2026-09-24.**

### T29 — Cross-time KDE normalization (PENDING_REVIEW)

### Files added

- `iste/phase3_normalization.py` — `NormalizationResult` dataclass +
  `normalize_support_grid(support_grid, t_s, *, percentile=95.0)` +
  `normalize_support_grids_across_time(support_grids_by_t_s, *, percentile=95.0)`. Exact
  formula: `normalized_support(x,y,t) = Support(x,y,t) / percentile(Support_t[Support_t >
  0], P)`, computed independently per timestamp (never pooled across timestamps), using
  only the grid's own strictly-positive cells (excludes the large structural-zero
  background of a fixed search grid, which empirically can make a whole-grid 95th
  percentile exactly 0 for a tight cluster). Default `P=95`, documented rationale in the
  module docstring: more noise-robust than the raw max (100th percentile), directly
  interpretable, and fully overridable — not claimed as a uniquely correct statistical
  answer, per CLAUDE.md's "never silently invent a threshold" guardrail. Returns explicit
  `NaN` (never a fabricated value or crash) when a grid has no positive cell at all, or
  when the computed percentile itself is `<=0`. Operates purely on an already-computed T16
  support grid (plain `numpy` array); no particle/Stage III/candidate-detection logic, no
  re-invocation of `compute_support_map`.
- `tests/test_phase3_normalization.py` — 10 `unittest` cases: hand-computable percentile
  match on a small grid, all-zero and empty-grid graceful `NaN` handling, invalid-percentile
  rejection, per-timestamp independence (scaling one timestamp's grid by 10x does not
  change another timestamp's own normalized result), and the core demonstration —
  synthetic tight vs. dispersed clusters (90 particles each) with a confirmed >4x raw KDE
  peak difference are normalized to within a factor of 3 of each other (log-ratio strictly
  smaller than the raw log-ratio).
- `notebooks/T29_kde_normalization_check.ipynb` — manual-check artifact (see below).

### Tests

```text
python -m unittest tests.test_phase3_normalization -v
Ran 10 tests in 0.649s
OK

python -m unittest discover -s tests -v   (full suite)
Ran 553 tests in 26.311s
OK
```

### Manual check

`notebooks/T29_kde_normalization_check.ipynb` computes raw + normalized support grids at 8
representative real timestamps (T0, T6, T12, T24, T36, T48, T60, T72) from the real cached
Stage III result, prints a raw-vs-normalized table, and plots side-by-side heatmaps. All
cells executed via `jupyter nbconvert --execute`, zero errors (15.7s). **Confirms the
normalization substantially removes the T28-measured scale bias on real data:**

```text
hours_before_obs  raw_kde_max  reference(p95)  normalized_max
               0      4745.00         4295.20           1.105
               6      3961.17         3327.43           1.190
              12      3176.76         2512.04           1.265
              24      2001.91         1713.17           1.169
              36      1466.93         1079.37           1.359
              48      1189.38          832.70           1.428
              60      1045.17          685.39           1.525
              72       855.31          583.33           1.466

raw_kde_max        range: [855.31, 4745.00]  ratio=5.55x
normalized_kde_max  range: [1.105, 1.525]  ratio=1.38x
```

Raw KDE-max ratio across the window drops from 5.55x to 1.38x after normalization — the
dispersion-driven scale difference T28 measured is substantially (not perfectly) removed.

### Known limitations

- A documented PROTOTYPE/CONFIGURATION choice, not a uniquely correct statistical answer;
  `percentile` is fully overridable.
- Normalization alone makes cross-time comparison meaningful — it does not itself rank
  candidates (T30-T39 not yet implemented).
- Manual-check notebook covers 8 sample timestamps (not all 73) to keep runtime small; the
  underlying function works identically at any timestamp/grid.
- No Stage III module, and no existing T14-T17/T25/T26 module, was modified.

**T29 approved 2026-09-24.** User's explicit instruction: the observed `+0.923`
correlation between `hours_before_observation` and `normalized_max` on the real dataset
must be carried into T36's diagnostics/sensitivity analysis unmodified, not artificially
removed at T29 (recorded in `tasks.md`'s T29 entry and `memory.md`).

### T30 — Initial-condition influence field (PENDING_REVIEW)

### Revision (2026-09-24): the first design was rejected before approval

The original `exp(-d/spatial_scale) * exp(-age/temporal_scale)` formulation (reported
below in the original submission) was reviewed and rejected before sign-off: since T36
plans to subtract this influence from the composite score, a term that decays purely and
monotonically with age (`~1.0` at T0, `~0.05` at T72) would mechanically favor older
timestamps regardless of any actual evidence — precisely the "do not force older timestamps
to score higher by construction" constraint this phase exists to enforce. A follow-up
conceptual review (see the full discussion in this session's transcript) further found that
even a corrected age-free formula anchored to the *fixed* observed-slick point would
reintroduce an analogous bias along a spatial axis: real bulk advection (the whole particle
population drifting up to ~48 km from the observed point by T72, confirmed on this real
dataset) would make every old-timestamp candidate look artificially "distinct" purely
because the current carried the whole population away — not because that candidate is
actually a distinct spatial mode.

**Corrected design: Mahalanobis distance to the particle cloud's OWN centroid at `t`.**
The cloud at any timestamp already originated at the observed slick (Stage III's boundary
condition), so its own mean position is the model's own physical answer to "where would an
undifferentiated continuation of the observed slick be by now, under the real currents used
here" — it nets out advection automatically, which a fixed reference point cannot.
Anisotropy (real advected plumes are elongated along the flow direction, not circular) is
handled via the full covariance matrix rather than a scalar spread, avoiding a spurious
directional bias a single RMS number would introduce.

### Files changed (rewritten)

- `iste/phase3_initial_bias.py` — rewritten. `TimestampCloudStats` (mean/covariance of a
  timestamp's own valid particles, in local projected metres via T15's
  `project_to_local_metres`, computed once per timestamp via `compute_timestamp_cloud_stats`
  using T14's `query_particle_slice` unmodified) + `InitialBiasInfluence` +
  `compute_initial_bias_influence(cloud_stats, candidate_lat, candidate_lon, *, k=1.0)` +
  `compute_initial_bias_influence_for_candidates(records, candidates, *, k=1.0, ...)` (batch
  over T17 `Candidate` objects, grouping by shared `t_s` to avoid redundant Stage III
  queries). Exact formula:
  `d_M = sqrt((x_c - mu(t))^T @ Sigma(t)^-1 @ (x_c - mu(t)))`,
  `influence = exp(-0.5*(d_M/k)^2)`. **Contains no age term anywhere** — the only place
  time enters is through `mu(t)`/`Sigma(t)` being the actual measured state of that
  timestamp's own particles, not an assumed clock. `k` (default `1.0`, a dimensionless
  sigma-multiplier) is the only free parameter, replacing the previous two-constant design
  entirely. Explicit, tested fallback for singular/near-singular covariance (eigenvalue
  ratio `<= 1e-6`, default): isotropic variance `= trace(Sigma)/2`, still expressed in
  "sigma units" via the same formula. Explicit `NaN` (never fabricated) for `<2` particles
  or exactly-zero variance (all particles coincide) — flagged via an explicit
  `covariance_state` field (`"full_rank"` / `"singular_fallback"` /
  `"insufficient_particles"` / `"zero_variance"`), never silently treated as a valid 0.
- `tests/test_phase3_initial_bias.py` — rewritten, 17 `unittest` cases: full-rank
  covariance (centroid candidate → `influence≈1.0`, distant candidate → near-zero,
  anisotropy respected — same raw-degree offset along the wider vs. narrower axis produces
  different, correctly-ordered Mahalanobis distances, monotonicity), an explicit no-age-term
  proof (two candidates at *different* timestamps, each placed exactly at their own
  timestamp's centroid, receive *identical* `influence=1.0` regardless of the timestamps'
  relative age/scale), insufficient-particle handling (0 and 1 particle → `NaN`), singular/
  degenerate covariance (exactly 2 particles, exactly-collinear particles → isotropic
  fallback, both finite and non-crashing; all-identical particles → `zero_variance` → `NaN`),
  `k` validation and effect (larger `k` widens the falloff), and batch-over-candidates
  behavior (cloud stats reused across candidates sharing a timestamp, empty list). One test
  corrected after an initial failure for the same float64-underflow reason as before
  (a candidate thousands of km from a very tight cloud legitimately underflows `exp()` to
  exactly `0.0` — ordinary floating-point behavior for an input far outside this pipeline's
  realistic operating range, not a discontinuity in the formula).
- `notebooks/T30_initial_bias_check.ipynb` — rewritten (see manual check below).

### Tests

```text
python -m unittest tests.test_phase3_initial_bias -v
Ran 17 tests in 0.031s
OK

python -m unittest discover -s tests -v   (full suite)
Ran 570 tests in 24.742s
OK
```

### Manual check

`notebooks/T30_initial_bias_check.ipynb` computes `TimestampCloudStats` at all 73 real
analysis timestamps (T0-T72, 24.8s) and the Mahalanobis influence for all 280 real
candidates from the T14-T17 3-day sweep (`results/actual_data/hotspots.json`). All cells
executed via `jupyter nbconvert --execute`, zero errors:

```text
covariance_state distribution across all 73 real timestamps: {'full_rank': 73}
computed influence for all 280 real candidates (0 NaN -- degenerate-covariance timestamps)

correlation(age_hours, influence)             = -0.497
correlation(age_hours, mahalanobis_distance)   = +0.432
(compare to the REJECTED design, which had correlation exactly -1.000 with age by construction)
```

**This is the key validation result.** The rejected design had `correlation(age, influence)
== -1.000` *by construction* — a deterministic, guaranteed relationship baked into the
formula regardless of any data. The revised design has no age term at all, yet a real,
moderate `-0.497` correlation remains on this actual dataset. That residual correlation is
qualitatively different in kind from the rejected one: it is not guaranteed by the formula
and could reflect genuine physics (e.g. the real cloud's dominant support mode staying
closer, in Mahalanobis units, to its own centroid at earlier timestamps than at later ones,
plausibly related to how the cloud's anisotropy/shape evolves as advection stretches it).
Per the user's explicit instruction for the analogous T29 finding, **this residual
correlation must be carried into T36's diagnostics/sensitivity analysis as-is, not
artificially suppressed here** (recorded in `memory.md`).

All 73 real timestamps had sufficient, well-conditioned particle data
(`covariance_state='full_rank'` universally) — the singular/insufficient-particle fallback
paths are exercised and verified only by the synthetic unit tests, not by this real run.

### Known limitations

- `k` is a documented PROTOTYPE/CONFIGURATION choice (default `1.0`), not a validated form
  — fully overridable; the `exp(-0.5*(d_M/k)^2)` (Gaussian-density-consistent) shape was
  chosen over a considered-and-rejected linear alternative (`exp(-d_M/k)`), documented in
  the module docstring.
- The isotropic covariance fallback (used only when the cloud is singular/near-singular,
  never observed on the real dataset) loses directional information — an explicit, tested
  approximation, not a claim of full accuracy.
- Not yet combined with anything else (T36) and does not itself filter any candidate
  (T31-T39 not yet implemented).
- No Stage III module, and no existing T14-T17/T25/T26 module, was modified.

**T30 approved 2026-09-24.** User's exact framing: "influence is strongly candidate-
dependent at the same ages rather than being mechanically determined by age" — the
`r=-0.497` empirical correlation is kept as a diagnostic only, never used as an age
correction (recorded in `memory.md`).

### T31 — Temporal persistence tracking (PENDING_REVIEW)

### Files added

- `iste/phase3_persistence.py` — `PersistenceTrack` dataclass (`candidate_ids`, `t_s_list`,
  `support_scores`, `lat_list`/`lon_list`, plus derived properties `earliest_t_s`/
  `latest_t_s`/`persistence_count`/`persistence_duration_hours`/`mean_support`/
  `max_support`/`support_variance` and a `temporal_continuity(interval_hours=1.0)` method)
  + `track_candidate_persistence(candidates, *, spatial_matching_radius_m=2000.0,
  max_gap_hours=1.0, interval_hours=1.0)`. A simple, deterministic, documented-as-PROTOTYPE
  greedy nearest-neighbor multi-object tracker: at each of the real 73 hourly timestamps
  (processed in ascending real-time order), every active track is matched to its nearest
  available candidate within `spatial_matching_radius_m` (nearest-distance-first greedy
  assignment, each track/candidate matched at most once per timestamp); a track surviving
  up to `max_gap_hours` of EXTRA tolerance beyond one normal `interval_hours` step (default:
  tolerates exactly one missed intermediate hourly detection) before being closed; every
  candidate ends up in exactly one track. **Contains no age term anywhere** — a track deep
  in the window and an identically-shaped track near `t_obs` produce identical statistics
  (explicitly tested). Distance uses the same `project_to_local_metres` convention (T15)
  every other module in this project already uses. `spatial_matching_radius_m=2000.0`
  (2 km) is grounded in T30's own measured real-data finding (~48 km/72h average advection,
  ~0.67 km/hour, so 2 km gives ~3x margin over one hour's typical drift). Both parameters
  explicitly labeled PROTOTYPE/CONFIGURATION, fully overridable.
- `tests/test_phase3_persistence.py` — 15 `unittest` cases: 5 consecutive hourly candidates
  form one high-persistence track vs. a single-timestamp spike (low persistence, `count=1`,
  `duration=0h`) — directly matching the task's required check, with an explicit assertion
  that persistence (not raw support magnitude — the spike candidate was given a deliberately
  huge `support_score`) is what distinguishes them; spatial-radius gating (far-apart
  candidates at adjacent hours don't merge, near ones do); gap tolerance (default bridges
  exactly one missed hour; `max_gap_hours=0.0` does not bridge; a gap beyond tolerance does
  not bridge); per-track statistics (mean/max/variance of support, hand-computed); multiple
  T17 modes at the same timestamp never merge into one track; an explicit no-age-term proof
  (two structurally identical tracks at very different ages produce identical persistence
  statistics); and confirmation every input candidate is assigned to exactly one output
  track (round-trip completeness), plus empty-input and validation-error cases.
- `notebooks/T31_persistence_check.ipynb` — manual-check artifact (see below). One
  presentational bug was caught and fixed during authoring: an early draft cell computed
  "age of track's earliest member" using `latest_t_s` instead of `earliest_t_s` (a labeling
  error in the notebook only, not in the tested library code) — corrected and re-executed
  before finalizing.

### Tests

```text
python -m unittest tests.test_phase3_persistence -v
Ran 15 tests in 0.006s
OK

python -m unittest discover -s tests -v   (full suite)
Ran 585 tests in 25.540s
OK
```

### Manual check

`notebooks/T31_persistence_check.ipynb` tracks all 280 real candidates from the T14-T17
3-day sweep (`results/actual_data/hotspots.json`) across the real 73-timestamp window. All
cells executed via `jupyter nbconvert --execute`, zero errors:

```text
280 real candidates -> 44 persistence tracks
every candidate assigned to exactly one track: confirmed (280 == 280)

track_id    count  duration_h  continuity  mean_supp   max_supp   supp_var  earliest_age_h  latest_age_h
track0000      72        71.0       1.000    1894.78    4675.43 1124523.26            72.0           1.0
track0002      27        26.0       1.000     417.43     545.14    3931.44            72.0          46.0
track0008      19        18.0       1.000       3.33       5.48       1.58            72.0          54.0
...

correlation(age_of_earliest_member, persistence_count) = +0.280 -- reported as a
diagnostic, not corrected for (no age term exists in the tracking algorithm itself)
```

The dominant real track (`track0000`) spans 72 of the 73 timestamps (T1-T72,
`temporal_continuity=1.000`, no gaps at all) — this is the single, essentially-unimodal
support region T14-T17 found at nearly every historical timestamp in this real run
(consistent with the earlier finding that this particular real cloud stayed largely
unimodal). The other 43 tracks are much shorter (secondary/transient candidates), correctly
separated from the dominant persistent feature by the tracker. The `+0.280` age correlation
is real but modest, and — per the standing instruction already applied to the analogous
T29/T30 findings — is reported here as a diagnostic for T36, not treated as evidence to
correct for or suppress.

### Review validation (2026-09-24, requested before approval — no code changed)

Two checks requested before T31 sign-off, both added as new cells (Sections 7-9) appended
to `notebooks/T31_persistence_check.ipynb`; `iste/phase3_persistence.py` itself was **not**
modified for this validation.

**1. Empirical basis for `spatial_matching_radius_m` (not derived from the ~48 km/72h total
advection figure alone, per request).** For every strictly-adjacent (1h-apart) real
timestamp pair, computed the nearest-neighbor distance from every candidate at `t` to its
closest candidate at `t+1h` (no radius applied) — 71 adjacent pairs, 279 displacement
samples:

```text
median : 0.667 km        fraction below 0.5 km: 0.315
mean   : 1.258 km        fraction below 1.0 km: 0.677
p90    : 3.616 km        fraction below 2.0 km: 0.835
p95    : 4.690 km        fraction below 3.0 km: 0.878
p99    : 7.791 km
max    : 9.656 km
```

Sensitivity analysis (unmodified tracker, `spatial_matching_radius_m` in {1, 2, 3} km):

```text
 radius_km  n_tracks  longest_h   n>=6h  n>=12h  n>=24h  n>=48h   mean_h
       1.0        88       25.0      11       2       1       0     2.22
       2.0        44       71.0      12       6       2       1     5.55
       3.0        38       71.0      13       6       2       1     6.66
```

At 1 km the dominant track fragments (longest drops to 25h) because some of its own real
steps exceed 1 km (see below). Between 2 km and 3 km the dominant track's duration is
IDENTICAL (71h both times) — the only change is 6 additional secondary tracks merging
(`n_tracks` 44→38), not any extension of the real dominant trajectory. This is evidence that
2 km already fully captures the genuine trajectory's own step sizes, and that increasing to
3 km starts merging smaller/secondary candidates rather than recovering missed genuine
continuations — i.e. 2 km sits at a reasonable point in the sensitivity curve, though the
~16.5% of raw nearest-neighbor pairs exceeding 2 km (largely reflecting hours where a true
continuation doesn't exist and the "nearest" candidate is a spurious, unrelated one, not
missed genuine matches) is noted for the record.

**2. Deep-dive on `track0000` (default 2 km) — genuine trajectory vs. greedy artifact.**

```text
n consecutive-member steps: 71
max match distance  : 1362.6 m
mean match distance : 621.8 m
p95 match distance  : 1063.9 m
total displacement (path length): 44.15 km
steps using the max_gap_hours allowance (real gap > 1h): 0
```

Every one of the 71 steps occurred at an exact 1-hour interval — the gap-bridging allowance
was never used once. The full per-step table (all 72 members) shows smooth, small
(333-1363 m) position changes every hour, always well under the 2 km matching radius with
real margin. **Correction (2026-09-24, per explicit user instruction):** the table also
shows `support_score` increasing almost monotonically across the 72 steps; this was
initially described as additional corroborating evidence of a genuine trajectory, which the
user correctly rejected — `support_score`'s trend is not an independent signal here (it is
exactly what T28/T29 already showed is expected/confounded with age across this whole real
dataset), so it must not be cited as evidence of physical genuineness. **The actual,
sufficient validation basis for `track0000` being a coherent candidate-track (not a
validation of physical source identity) is strictly: the consecutive hourly matching
itself, the bounded match distances (max 1362.6 m, well under the 2 km radius), zero
gap-bridges used, and the 1 km/2 km/3 km sensitivity analysis above** (fragmentation at
1 km, stability from 2 km to 3 km). Plot with per-step distance annotations saved to
`results/actual_data/T31_longest_track_annotated.png`.

**Standing caveat, recorded per explicit user instruction:** persistence-track continuity
(this task's output) is evidence of candidate-TRACK continuity — a spatially/temporally
coherent sequence of detections — not proof of physical source identity. This distinction
must be preserved through T36 and any later interpretation of persistence results.

### Known limitations

- `spatial_matching_radius_m`/`max_gap_hours` are documented PROTOTYPE/CONFIGURATION
  choices, not validated constants — fully overridable.
- The tracker is a simple greedy nearest-neighbor assignment, not a claim of globally
  optimal multi-object tracking (e.g. no lookahead/backtracking if a locally-greedy choice
  turns out suboptimal a few steps later).
- The raw consecutive-displacement distribution (Section 7) mixes genuine-continuation
  distances with spurious forced nearest-neighbor distances (hours where no true
  continuation exists); it should be read as an upper-bound/context check, not a pure
  "true match distance" distribution.
- Persistence alone does not rank/score candidates (T32-T39 not yet implemented).
- No Stage III module, and no existing T14-T17/T25/T26 module, was modified.

**T31 approved 2026-09-24.** Standing caveat recorded (see `memory.md`): persistence-track
continuity is evidence of candidate-track continuity, never proof of physical source
identity; a track's own `support_score` trend must never be cited as independent
corroborating evidence of physical genuineness (an earlier draft of this section made that
exact mistake for `track0000` and was corrected above per explicit user instruction).

### T32 — Spatial convergence scoring (PENDING_REVIEW)

### Files added

- `iste/phase3_convergence.py` — `ConvergenceWeights` (5 explicit, configurable,
  equal-by-default weights) + `ConvergenceScore` dataclass +
  `compute_convergence_score(candidate, *, total_valid_particles_at_t,
  total_distinct_pixels_at_t, total_distinct_members_at_t, temporal_consistency=None,
  weights=...)` + batch `compute_convergence_scores_for_candidates(records, candidates, *,
  persistence_tracks=None, weights=...)`. Composite of 5 components, reusing existing
  outputs wherever possible: `particle_fraction = D/total_valid_particles_at_t` (D from
  T16's own attached `SupportFeatures`); `pixel_diversity_fraction = V/
  total_distinct_pixels_at_t` (V is T16's own pixel-diversity feature, reused unchanged —
  NOT recomputed); `member_diversity_fraction` (the one genuinely NEW quantity — distinct
  `forcing_member_id` count in the candidate's already-attached
  `NeighborhoodResult.forcing_member_id`, since T16 only computes pixel diversity, not
  forcing-member diversity); `concentration_score = exp(-within_core_spread_m/
  core_radius_m)` (`within_core_spread_m` = RMS of `NeighborhoodResult.distance_m`, already
  computed by T15 — no new distance formula; `core_radius_m` is T17's own
  `CoreRegion.radius_m`, not a new free parameter); and (optional) `temporal_consistency` =
  the candidate's own T31 `PersistenceTrack.temporal_continuity()`, reused via a
  `candidate_id` lookup when `persistence_tracks` is supplied — omitted and the remaining
  weights renormalized (never fabricated) when unavailable, mirroring this project's
  standing "fail gracefully" discipline for optional evidence. Equal weighting (`1.0` each)
  is the default, documented as the simplest non-arbitrary PROTOTYPE/CONFIGURATION choice,
  fully overridable. Same screening-quantity caveat as every other backward-support
  diagnostic in this project (explicit in the module docstring): high convergence is
  evidence of independent-trajectory agreement, never proof of physical source identity.
- `tests/test_phase3_convergence.py` — 12 `unittest` cases: the core required check
  (synthetic homogeneous group — all particles same single pixel/member — vs. synthetic
  diverse group — same particle COUNT, many distinct pixels/members — at IDENTICAL raw
  density/concentration; diverse scores strictly higher on `convergence_score`, confirming
  it is distinguishable from raw KDE), concentration ordering (tighter neighborhood scores
  higher) and an exact hand-computed concentration check against `core_radius_m`, optional
  `temporal_consistency` (absent → still a defined score via renormalization; present →
  changes the score), weight validation (negative rejected) and a zero-weight-effectively-
  excludes-a-component check, missing-`peak_support_features`/non-positive-totals rejection,
  and batch wiring (per-timestamp totals computed once and reused, T31 track lookup by
  `candidate_id`, empty-list handling).
- `notebooks/T32_convergence_check.ipynb` — manual-check artifact (see below).

### Tests

```text
python -m unittest tests.test_phase3_convergence -v
Ran 12 tests in 0.007s
OK

python -m unittest discover -s tests -v   (full suite)
Ran 597 tests in 14.864s
OK
```

### Manual check

`notebooks/T32_convergence_check.ipynb` re-runs T14→T15→T16→T17 on the real cached Stage
III result (regenerating `Candidate` objects WITH their attached `SupportFeatures`/
`NeighborhoodResult`, since `hotspots.json` only stores summary fields — same T14-T17 call
sequence already validated in the main real-data notebook, 67.1s), reuses T31's
`track_candidate_persistence` unchanged for the temporal-consistency component, and computes
convergence scores for all 280 real candidates. All cells executed via `jupyter nbconvert
--execute`, zero errors:

```text
regenerated 280 real candidates (with attached support features) in 67.1s
280 candidates -> 44 persistence tracks (T31, reused unchanged)
computed convergence scores for 280 real candidates
280/280 had a temporal_consistency value (all should, via T31)

candidate_id                   particles  p_frac pix_div_f mem_div_f concentr temp_cons convergence
2019-10-11T23:00:00_cand000          340   0.103     0.491     0.833    0.502     1.000       0.586
2019-10-12T02:00:00_cand011            1   0.000     0.002     0.167    0.370     1.000       0.308
2019-10-14T22:00:00_cand000         2123   0.646     0.761     1.000    0.534     1.000       0.788

correlation(raw particle count, convergence_score) = +0.881
```

The `+0.881` correlation (strong but `< 1.0`) confirms `convergence_score` tracks raw
density (as expected — density genuinely is real evidence) while NOT being a pure rescaling
of it: e.g. `2019-10-12T02:00:00_cand011` has only 1 particle (near-zero raw density) yet
still scores `convergence_score=0.308`, entirely from its concentration and (perfect)
temporal-consistency components — exactly the kind of distinguishable-from-raw-KDE behavior
the task requires, demonstrated on real data as well as the required synthetic unit test.

### Known limitations

- Equal component weighting is a documented PROTOTYPE/CONFIGURATION default, not a
  validated scheme — fully overridable.
- `within_core_spread_m` reuses the neighborhood already computed at T16's support-map
  radius, which happens to equal `core_radius_m` in this real run's configuration but is not
  guaranteed to in general.
- A screening quantity, like every other backward-support diagnostic in this project — never
  proof of physical source identity.
- Not yet combined into T36's final composite score; T33-T39 not yet implemented.
- No Stage III module, and no existing T14-T17/T25/T26/T31 module, was modified.

### T32 review diagnostic: ablation / independence analysis (2026-09-24, no code changed)

Requested before approval. All 17 new cells appended to
`notebooks/T32_convergence_check.ipynb`; every ablation used the module's own existing
`ConvergenceWeights` mechanism (setting one weight to `0.0`) — `iste/phase3_convergence.py`
itself was not modified.

**Pairwise correlations (280 real candidates).** Pearson/Spearman both computed; the
striking finding is that `pixel_diversity_fraction` and `member_diversity_fraction` are
themselves very strongly correlated with raw `particle_count` on this real dataset (Pearson
0.871/0.807, **Spearman 0.999/0.910**) — on this data, wherever more particles support a
candidate, proportionally more distinct pixels/forcing-members are represented too, so the
"diversity" components are not empirically independent of density here, even though they
are conceptually and computationally independent (T32 never derives one from the other).
`convergence_score` itself correlates 0.881 (Pearson) / 0.869 (Spearman) with raw
`particle_count`.

**Leave-one-component-out ablation (via `weight_X=0.0`):**

```text
removed component               corr_vs_original  corr_vs_density  n_top10_changed
particle_fraction                         0.9962           0.8366               12
pixel_diversity_fraction                  0.9960           0.8780                0
member_diversity_fraction                 0.9899           0.9113                0
concentration_score                       0.9987           0.8856                0
temporal_consistency                      0.9977           0.8856                0
```

Removing `particle_fraction` is the ONLY ablation that changes top-10 membership (12 of 20
symmetric-difference entries); removing any of the other four leaves the top-10 completely
unchanged.

**Variance/std of every component, and variance-decomposition of `convergence_score`**
(exact, since `convergence_score` is a coefficient-1/5 linear combination for all 280 real
candidates — every one had a `temporal_consistency` value):

```text
component                   std      var_share_of_convergence_score
particle_fraction          0.127     0.140
pixel_diversity_fraction   0.309     0.382
member_diversity_fraction  0.336     0.411
concentration_score        0.053     0.045
temporal_consistency       0.056     0.025
```

**Synthesis (answering the review's core question directly).** `particle_fraction` is NOT
the literal dominant term by variance share (14%) — `pixel_diversity_fraction` (38%) and
`member_diversity_fraction` (41%) together explain ~79% of `convergence_score`'s variance,
i.e. formally the diversity terms drive the ranking more than the raw-count term does.
**However**, because those same diversity terms are themselves almost perfectly rank-
correlated with raw particle count on this real dataset (Spearman 0.999/0.910), the
PRACTICAL effect is that ~93% of the score's variance traces back to terms that all move
together with "how much particle support exists here" — not to genuinely orthogonal
evidence. The two components designed to be structurally independent of density
(`concentration_score`, `temporal_consistency`) contribute only ~7% combined, because both
have very low variance across real candidates (concentration clusters near 0.45±0.05,
temporal_consistency clusters near 0.98±0.06) — among the top-15 candidates specifically,
`member_diversity_fraction` is pinned at exactly `1.0` and `temporal_consistency` at `1.0`
for every one of them (a ceiling effect), leaving `particle_fraction`/
`pixel_diversity_fraction` as the ONLY things still varying within the top tier — which is
exactly why removing `particle_fraction` (not any other component) is the one ablation that
reshuffles the top-10.

Checked explicitly whether any top-10 real candidate reaches high convergence via particle
count alone while scoring below the dataset median on diversity/concentration/persistence:
**none did** — every one of the real top-10 candidates was at or above the dataset median on
all four other components. This is reassuring for THIS dataset, but is a property of this
dataset's own strong count/diversity correlation, not a structural guarantee from the
scoring formula itself — the synthetic unit test (`test_multiple_independent_groups_score_
higher_than_one_homogeneous_group`) remains the actual structural proof that the formula
CAN distinguish density from diversity when they diverge; the real data simply doesn't
happen to exercise that divergence much in this particular 3-day window.

**Bottom line:** T32 is not literally reducible to particle density (the density term alone
is only 14% of score variance), but on this real dataset it is close to a monotonic function
of density because the diversity terms happen to track density so closely here, and the
genuinely orthogonal components (concentration, temporal persistence) are currently too
low-variance on this dataset to meaningfully assert independent influence on ranking. Default
weights were NOT changed per explicit instruction; this diagnostic is reported for review,
not acted upon.

Artifacts: `results/actual_data/T32_correlation_matrices.png`,
`results/actual_data/T32_ablation_top15_stacked.png`.

**T32 approved 2026-09-24.** Weights/implementation unchanged. Recorded limitation carried
forward: `convergence_score` must NOT currently be treated as an evidence source
independent from density/concentration in T36 — on this real dataset it behaves close to
(not identical to) a density-weighted score (see the ablation diagnostic above and
`memory.md`).

### T33 — Dispersion-aware scoring (PENDING_REVIEW — flagged finding, see below)

### Files added

- `iste/phase3_dispersion.py` — `DispersionAwareScore` dataclass +
  `compute_dispersion_aware_score(candidate, cloud_stats, *, neighborhood_radius_m)` +
  batch `compute_dispersion_aware_scores_for_candidates(records, candidates, *,
  neighborhood_radius_m)`. Analytically-grounded formula:
  `sigma_iso_sq(t) = trace(Sigma(t))/2` (T30's own per-timestamp covariance, reused
  unchanged — not recomputed); `p_within_r(t) = 1 - exp(-r^2/(2*sigma_iso_sq(t)))`
  (closed-form Rayleigh CDF — the probability mass of an isotropic 2-D Gaussian within
  radius `r` of its own mean); `D_expected(t) = N_total_valid_at_t * p_within_r(t)`;
  `dispersion_aware_score = D_actual_at_peak / D_expected(t)`. `r` is the SAME radius
  already used to build the candidate's T16 support map, not a new parameter. Isotropic
  approximation documented as an explicit simplification (mirrors T30's own singular-
  covariance fallback convention), not a claim of exact anisotropic accuracy — T30's own
  anisotropy test already shows real clouds are NOT isotropic. Score is unbounded above (a
  likelihood-ratio-style quantity). Explicit, tested `NaN` for undefined dispersion states
  (reusing T30's `covariance_state`) or zero particles — never fabricated.
- `tests/test_phase3_dispersion.py` — 9 `unittest` cases: exact match against the
  hand-derived analytic Rayleigh formula on a real-shaped synthetic cloud; the core required
  check (two timestamps with very different raw density, deliberately scaled to the SAME
  relative ratio against their own expected density, score approximately equal); an explicit
  demonstration that the SAME raw density scores HIGHER at a more-dispersed timestamp than a
  less-dispersed one (the literal "don't read lower raw density as automatically weaker
  evidence" requirement); `NaN` handling for insufficient particles/zero variance/no
  particles; radius validation; and batch wiring (cloud stats reused per shared timestamp,
  order preserved, empty-list handling).
- `notebooks/T33_dispersion_check.ipynb` — manual-check artifact (see below).

### Tests

```text
python -m unittest tests.test_phase3_dispersion -v
Ran 9 tests in 0.023s
OK

python -m unittest discover -s tests -v   (full suite)
Ran 606 tests in 15.809s
OK
```

### Manual check — real-data result requires review before this task is approved

`notebooks/T33_dispersion_check.ipynb` re-runs T14→T17 across all 72 real historical
timestamps (89.7s), takes each timestamp's top candidate, and computes both its raw density
and its dispersion-aware score against T30's own per-timestamp cloud stats. All cells
executed via `jupyter nbconvert --execute`, zero errors. All 72 timestamps had
`covariance_state == "full_rank"`.

```text
correlation(hours_before_obs, raw_density)              = -0.875 (Pearson) / -0.998 (Spearman)
correlation(hours_before_obs, dispersion_aware_score)    = +0.863 (Pearson) / +0.862 (Spearman)

raw_density        range: [336.00, 2123.00]   ratio=6.32x
dispersion_score   range: [0.8536, 2.6524]    ratio=3.11x
crossover (score < 1.0 -> >= 1.0): between hour 34 and hour 35 (roughly mid-window)

hours_before_obs  raw_density  n_total  sigma_iso_m  expected_D  dispersion_score
               1      2123.00     3288        668.9     2212.37           0.9596
              24       766.00     3288       1252.7      897.08           0.8539
              48       444.00     3288       2098.5      352.92           1.2581
              72       340.00     3288       3545.9      128.19           2.6524
```

**This is flagged, not accepted as a success.** Raw density still shows its familiar strong
negative age correlation; the dispersion-aware score shows an almost equally strong
correlation in age, just POSITIVE — the "undifferentiated blob" baseline (`D_expected`)
falls faster with age (17.3x, T1->T72) than the real top candidate's own raw density does
(6.3x), so the ratio necessarily climbs with age. This is materially different from every
other Phase 3 task's residual-correlation finding so far (T29's `+0.923`, T30's `-0.497`,
T31's `+0.280` — all modest, non-guaranteed residuals kept as diagnostics): here the
magnitude (`|0.86|`) is comparable to the ORIGINAL raw-density bias (`|0.88|`) this task
exists to remove, just with the sign flipped.

**Two competing, unresolved interpretations, recorded for review rather than silently
picked:**

1. **Genuine physical signal:** real advected/sheared particle clouds do not stay
   isotropic-Gaussian as they spread (T30 already confirmed real clouds are anisotropic);
   a genuine coherent source region's local peak may plausibly decay much more slowly than
   an idealized single-Gaussian-blob null predicts, so a growing dispersion-aware score
   could correctly reflect increasingly distinctive local structure persisting within an
   ever-more-spread-out overall envelope.
2. **Model-mismatch artifact:** the isotropic-Gaussian `D_expected` model may simply be a
   poor statistical null for how these real "top candidate" densities actually decay,
   systematically inflating the ratio with age for reasons unrelated to genuine
   concentration — i.e. T33 may have replaced one age-driven mechanical bias with another,
   opposite one, exactly the failure mode T30's first (rejected) design fell into.

This diagnostic does not resolve which interpretation is correct — it is reported honestly,
per the standing "carry residual correlations forward as diagnostics, do not spin as
success" instruction, but this one is flagged as requiring explicit review given its
magnitude, rather than assumed acceptable by default.

### Known limitations

- Isotropic-Gaussian approximation for `D_expected` — explicit, documented, but its
  goodness-of-fit to real (confirmed anisotropic) particle clouds has not been separately
  validated; the flagged age-correlation finding above may be partly attributable to this.
- `dispersion_aware_score` is unbounded above, not `[0,1]`-normalized.
- A screening quantity, like every other backward-support diagnostic in this project — never
  proof of physical source identity.
- Not yet combined into T36's final composite score; T34 (screening normalization) through
  T39 not yet implemented.
- No Stage III module, and no existing T14-T17/T25/T26/T30 module, was modified.

### T33 review diagnostic 2: isotropic vs. anisotropic expected-concentration model (2026-09-24, no code changed)

Requested before approval, in response to the flagged `+0.863` age correlation above. 17
new cells appended to `notebooks/T33_dispersion_check.ipynb`; `iste/phase3_dispersion.py`
was **not** modified. Per explicit instruction, the formulation was NOT selected based on
which produces a smaller age correlation — selection criterion was goodness-of-fit against
the real, model-free empirical within-radius fraction.

**1. Real particle-cloud anisotropy (72 timestamps, T30's covariance):** mean axis ratio
`sqrt(lam1/lam2) = 1.82` (max `4.33`) — real clouds ARE meaningfully anisotropic, confirming
T30's own earlier finding. `det(Sigma)` ("covariance area") correlates `+0.927` with age
(matches T28's spread-growth finding); axis ratio itself correlates only mildly with age
(`+0.375`) — clouds get somewhat, not dramatically, more elongated as they age; orientation
is only weakly age-related (`-0.117`, presumably tracks local current direction instead).

**2-4. Isotropic vs. anisotropic (Monte Carlo, 200k draws, fixed seed) vs. REAL empirical
within-radius fraction — the actual goodness-of-fit test:**

```text
isotropic model error   (p_iso - p_empirical):   mean=+0.0052  MAE=0.0252  std=0.0277
anisotropic model error (p_aniso - p_empirical): mean=+0.0204  MAE=0.0264  std=0.0290
correlation(age, isotropic model error)   = -0.895
correlation(age, anisotropic model error) = -0.819

                                  Pearson   Spearman
current ISOTROPIC score             0.863      0.862
diagnostic ANISOTROPIC score        0.923      0.893
```

**The anisotropic model is NOT more accurate against real data — it is very slightly WORSE**
(larger mean bias, larger MAE, larger std) than the simpler isotropic model, and its
dispersion-aware score's age correlation is also slightly WORSE (0.923/0.893 vs. 0.863/
0.862), not better. Both models' own prediction errors correlate strongly NEGATIVELY with
age (`-0.895`/`-0.819`) — i.e. BOTH increasingly UNDERESTIMATE the real cloud's true
concentration as it ages, regardless of covariance shape treatment.

**5 (synthetic A/B/C).** On controlled synthetic Gaussians, the isotropic formula clearly
DOES underestimate true within-circle mass for eccentric shapes (A: isotropic, `p_iso=0.3935`
vs `p_aniso=0.3952`, matching as expected; B: `axis_ratio=6`, `p_iso=0.1025` vs
`p_aniso=0.2231`, more than 2x underestimate). C (same as B, rotated 45°) reproduces B
exactly, confirming the expected rotation-invariance of mass-within-a-circle — a correctness
check on the MC estimator, not a real-data finding. So the isotropic simplification is a
DEMONSTRABLE, real mathematical limitation in the abstract — it just doesn't happen to be
what's driving the real-data age correlation (see below).

**5D and 6.** Both pass cleanly: the ratio-based `dispersion_aware_score` construction
itself is scale-invariant (`D-small`/`D-large`, very different covariance magnitude, both
score exactly `2.0000` when actual density is fixed at 2x expected) and shape/scale-
invariant across two very different synthetic anisotropic timestamps (raw density differs
`4.344x`, scores both exactly `1.7000`). The RATIO construction is not the problem.

**Synthesis.** The isotropic-vs-anisotropic axis is demonstrably NOT the main driver of the
flagged `+0.86` age correlation — switching to a (more mathematically correct, for the
Gaussian-shape question) anisotropic null model does not reduce it, and very slightly
worsens both the goodness-of-fit and the age correlation. The more fundamental finding is
that **neither Gaussian null model (isotropic or anisotropic) is a good statistical
description of how these real particle clouds' peak concentration evolves** — both
increasingly underestimate real concentration as the cloud ages, at a similar rate. This
points toward explanation (1) from the first diagnostic (genuine persisting coherent
structure exceeding any Gaussian-shaped null, growing relatively more pronounced with age)
being closer to the real story than explanation (2) (isotropic-specific model mismatch) —
though this diagnostic still does not fully resolve WHY real clouds retain more
concentration than a Gaussian predicts; it only rules out "wrong covariance shape" as the
explanation. **T33 remains PENDING_REVIEW; no formulation was selected or changed.**

### T33 review diagnostic 3 (final): empirical within-timestamp null (2026-09-24, no code changed)

Requested before approval, to determine whether the `+0.86` age correlation is caused by
the Gaussian expected-concentration model specifically, or persists under a fully
model-free baseline. 13 new cells appended to `notebooks/T33_dispersion_check.ipynb`;
`iste/phase3_dispersion.py` was **not** modified; no production weights changed.

**Empirical null — exact definition.** For every valid particle `i` at timestamp `t`:
`local_count_i` = particles (self-inclusive) within radius `r` (the SAME
`neighborhood_radius_m` already used by T16/T33 — no new parameter) of particle `i`'s own
real position, computed by self-querying the SAME T15 `ParticleNeighborhoodIndex` KD-tree
already built for that timestamp (reused, not rebuilt).
`D_expected_empirical(t) = median_i(local_count_i)`;
`empirical_null_score = D_actual_at_candidate_peak / D_expected_empirical(t)`. Zero
distributional assumption, zero new hyperparameters, preserves the timestamp's own exact
particle count/positions/extent/dispersion by construction (it IS the real data).

**Results (72 real timestamps, 129.9s, zero NaN/degenerate cases):**

```text
                                 Pearson  Spearman     mean   median      p10      p90      min      max
current GAUSSIAN score             0.863     0.862    1.298    1.056    0.883    2.291    0.854    2.652
EMPIRICAL-NULL score                0.951     0.943    1.908    2.030    1.424    2.370    1.242    2.426

(D_expected_gaussian - D_expected_empirical) / D_expected_empirical:
  mean=+0.581  std=0.323
  correlation with age: -0.510
```

**The empirical-null score's age correlation (`0.951`/`0.943`) is STRONGER than the current
Gaussian score's (`0.863`/`0.862`), not weaker or comparable.** A fully model-free baseline,
derived only from the real particle population with zero distributional assumption and zero
new hyperparameters, shows an even more pronounced positive age trend. Sample rows show why:
from hour 1 to hour 72, the top candidate's raw density falls 6.3x (2123→340), while the
SAME population's own median local density falls ~11.2x (1709→153) — the dominant candidate
pulls further ahead of the "typical" member of its own population as the window goes back in
time, regardless of which null model (Gaussian or empirical) is used to measure that gap.

**Conclusion — category B (per the required categorical framework):**

> **"The age trend persists under an empirical null; it is not primarily a Gaussian-null
> artifact."**

This is decisive, not merely suggestive: the empirical null was specifically constructed to
remove every trace of the Gaussian assumption (already implicated and then ruled out by
diagnostic 2's isotropic-vs-anisotropic comparison), and the trend not only persisted but
strengthened. Combined with diagnostic 2's finding (anisotropic Gaussian was not more
accurate against real data, and both Gaussian nulls under-predict real concentration
increasingly with age), this now robustly supports the "genuine structural persistence"
interpretation from diagnostic 1 over "model-mismatch artifact": on this real dataset, the
dominant candidate's relative prominence over the rest of its own population genuinely grows
as the backward window extends, independent of which reasonable null-model choice is used to
measure it. **This is strong evidence, within this dataset and this metric, that the trend
is a real property of this dominant support region rather than an artifact of T33's specific
formula choice** — phrased deliberately as dataset/metric-specific evidence, not as an
absolute, universally model-independent proof (a different real spill, forcing configuration,
or window length has not been tested). It does not by itself mean T33's score is safe to use
unmodified as an independent T36 input; it means the `+0.86`-ish age dependence is a
data-driven finding to be handled deliberately (e.g. explicit age-conditioning, or treated as
a distinct diagnostic dimension rather than blended in directly), not a modeling bug to fix
by picking a different null.

**T33 — CLOSED 2026-09-24: DONE, ACCEPTED WITH MAJOR CAVEAT.** Final recorded conclusion
(exact phrasing): "the dispersion-aware score shows a strong positive age trend on the real
dataset, and that trend persists — and actually strengthens — under the distribution-free
empirical within-timestamp null. Therefore the trend is not primarily attributable to the
Gaussian null assumption or covariance anisotropy." This is strong evidence within this
dataset/metric, not an absolute "model-independent" proof. No production code or weights
were changed by any of the three review diagnostics
(`iste/phase3_dispersion.py` untouched throughout).

### T34 — Screening-signal normalization: DESIGN SPECIFICATION (2026-09-24, no code written)

Per explicit instruction, this section is specification-only. `iste/phase3_screening.py`
does not exist yet; no T29-T33 production module was inspected for modification, only for
understanding what T34 must and must not duplicate.

#### 1. What "the screening signal" actually is in this codebase (inspection findings)

The original master spec (Stage 7) describes "a screening signal represented approximately
as `Support = F(D,W,V) x T16`" as if it were conceptually distinct from "KDE" (Stage 2,
→ T29). **Inspection confirms these are, in this codebase, the exact same underlying
object:** `iste.support_features.compute_support_map`'s `Support(x,y,t) = f(D,W,V)` field —
already established project-wide (see `memory.md`'s "KDE terminology note") as *the* thing
this project calls "KDE" throughout Phase 3. T17's `Candidate.support_score` is that same
field's value *exactly at the candidate's own peak grid cell* (by construction of
`find_candidate_peaks`, which sets `support_score=sf.support` from the peak's own
`SupportFeatures`). T29's `normalize_support_grid` already normalizes this exact field,
per-timestamp, via `reference_value = percentile(grid[grid>0], 95)`.

**This is the central design fact T34 must respect: there is only one raw signal here, and
T29 has already built one principled, approved, per-timestamp-relative normalization of it.**
Building a second, independently-derived normalization of the same signal would create two
inconsistent "truths" about identical evidence — precisely the double-counting risk flagged.

#### 2. What T34 IS

A thin, provenance-preserving **adapter that packages T29's already-computed, already-
approved per-timestamp normalization at the CANDIDATE level**, not a new normalization
scheme:

```text
screening_score(candidate) = candidate.support_score / reference_value(candidate.t_s)
```

where `reference_value(t)` is exactly T29's `NormalizationResult.reference_value`
(percentile-95-of-positive-grid-cells at that timestamp), obtained by calling T29's existing
`iste.phase3_normalization.normalize_support_grid`/`normalize_support_grids_across_time`
(imported and reused, never reimplemented). Because `candidate.support_score` already equals
the support grid's own value at the candidate's peak cell (T17's construction), this formula
is **mathematically identical** to reading T29's own `normalized_grid` off at the candidate's
peak position — T34 adds no new transform, just candidate-level packaging plus provenance
(raw `support_score`, the `reference_value` actually used, and the `D`/`W`/`V` breakdown
already on `Candidate`, carried through unchanged for audit).

#### 3. What T34 is explicitly NOT

- **Not** a second, independently-fitted normalization of the Support field (would compete
  with, and risk silently diverging from, T29's own).
- **Not** forced into `[0,1]`. T29's own output naturally lands `>= ~1` for a peak value
  (since a candidate IS the local maximum, and the reference is the 95th percentile, which
  is `<=` the maximum almost always) — real T29 sample values ranged `~1.10-1.53`. Inventing
  a SECOND rescale on top of T29's already-justified one, purely to force `[0,1]`, would add
  an unjustified extra assumption without new information. `screening_score` will be
  reported in whatever range T29's formula naturally produces, documented explicitly. Per
  `tasks.md`'s own rewritten wording ("`[0,1]` (or another data-justified range)"), this is
  within the task's original latitude.
- **Not** an independent evidence source from T29's normalized KDE, for T36's purposes.
  T36 must treat `screening_score` and "T29's normalized_support at a grid point" as **the
  same underlying evidence**, not two things to be separately weighted and summed — this is
  the single most important note this spec carries forward to T36's own design.
- **Not** a place to reconcile the different ranges across T29-T34 (T30/T31 ~`[0,1]`, T32
  ~`[0,1]`-ish, T33 unbounded above, T34 ~`>=1`-ish). That reconciliation is explicitly
  deferred to T36, which must design a single, holistic combination strategy for
  heterogeneously-scaled evidence — flagged here as a known open item, not solved
  piecemeal by each individual Phase 3 task.
- **Not** a place to add, correct for, or de-trend any age relationship. `screening_score`
  is a function of `(support_score, reference_value)` only — no `age`/`t_obs` argument
  anywhere in its signature. Per explicit instruction, if a real-data age correlation is
  observed in the manual check (plausible, since it's a close derivative of T29's own
  `+0.923`-correlated `normalized_max` finding), it will be **reported as a diagnostic,
  exactly like every other Phase 3 residual correlation — never suppressed, never corrected,
  never used to justify picking a different reference statistic.**
- **Does not modify** `iste/support_features.py` (T16) or `iste/phase3_normalization.py`
  (T29) — pure reuse via import.

#### 4. Information T34 must preserve

- `candidate_id`, `t_s` (standard Phase 3 identifiers, matching T30-T33's convention).
- `raw_support_score` (unmodified `candidate.support_score`) — for audit/provenance.
- `reference_value` actually used (for reproducibility — which exact T29 percentile was
  applied to this candidate).
- `density_score`/`weighted_support_score`/`pixel_diversity_score` (D/W/V), carried through
  unchanged from `Candidate` — downstream inspection should not need to re-fetch these from
  the original candidate object separately.
- **Exact per-timestamp candidate ordering.** Dividing by a single positive per-timestamp
  constant (`reference_value(t) > 0`, guaranteed whenever T29 doesn't return `NaN`) is a
  strictly monotonic transform — two candidates at the SAME `t_s` must have
  `screening_score` in the exact same relative order as their `raw_support_score`. This is
  the task's own explicitly required check.

#### 5. What T34 must NOT double-count (explicit cross-task accounting)

- **T29 <-> T34: same signal, by design** (see above) — T36 must combine them as one
  evidence axis, not two.
- **T30 (initial-bias influence), T31 (persistence), T32 (convergence), T33 (dispersion-
  aware)**: all four are structurally independent of the T16/T29/T34 Support lineage (T30
  uses Mahalanobis distance to the cloud centroid; T31 uses cross-timestamp spatial
  matching; T32 partially reuses T16's `V` diversity feature only, not the combined
  `Support`; T33 compares `D` against a dispersion-derived expectation, not against T29's
  percentile). No double-counting risk identified between T34 and T30-T33 — each measures
  a genuinely different thing. (T32's own real-data ablation already showed its `V`/`D`-
  derived components are empirically density-correlated on this dataset — a separate,
  already-recorded caveat, not a T34 concern.)

#### 5 continued. Tests to be written (`tests/test_phase3_screening.py`, not yet created)

1. **Hand-computable formula match:** a small synthetic support grid + a known `t_s` ->
   `screening_score` exactly equals `raw_support_score / reference_value`, cross-checked
   against T29's own `normalize_support_grid` output for the identical grid (bit-for-bit
   consistency between T34 and T29 — the core "not a second competing formula" guarantee).
2. **Monotonicity within a timestamp:** two synthetic candidates at the SAME `t_s`, different
   `support_score` -> `screening_score` preserves their exact relative order.
3. **Cross-candidate consistency:** two candidates sharing a timestamp use the IDENTICAL
   `reference_value` (no per-candidate recomputation drift).
4. **No age dependence structurally:** confirm the function signature takes no `age`/`t_obs`
   parameter; confirm two synthetic candidates with identical `(support_score,
   reference_value)` pairs but attached to very different (fabricated) timestamps produce
   IDENTICAL `screening_score` (mirroring T30/T33's own "no age term" proof pattern).
5. **Graceful `NaN` propagation:** if T29's `reference_value` is `NaN` (its own documented
   degenerate case — all-zero/empty grid), `screening_score` must also be `NaN`, never
   fabricated or silently treated as `0`.
6. **Batch behavior:** per-timestamp `reference_value` computed once and reused across every
   candidate sharing that `t_s` (mirrors T30/T31/T32/T33's batch convention); candidate
   order/count preserved; empty-list handling.
7. **Real-data manual check (not a unit test, the required progress-report deliverable):**
   raw `support_score` vs. `screening_score` side-by-side for the real 280-candidate set;
   explicit confirmation that within-timestamp ordering is unchanged (a real-data version of
   test 2); explicit, un-corrected report of `screening_score`'s correlation with age
   (Pearson + Spearman), labeled as a diagnostic per the standing project instruction.

#### 6. Open question flagged for the user, not decided here

Given `screening_score` is mathematically a close derivative of T29's own `normalized_max`
(which already showed `+0.923` age correlation on this real dataset), it is very likely
`screening_score` will show a comparably strong age correlation in the real-data manual
check. Per instruction, this will be reported, not corrected. Flagging in advance so it is
not a surprise at review time.

### T34 — Screening-signal normalization: IMPLEMENTATION (PENDING_REVIEW)

Implemented exactly per the approved specification above — no design changes during
implementation.

### Files added

- `iste/phase3_screening.py` — `ScreeningScore` dataclass +
  `compute_screening_score(candidate, normalization_result)` + batch
  `compute_screening_scores_for_candidates(candidates, support_grids_by_t_s, *,
  percentile=95.0)`. `screening_score = candidate.support_score / reference_value`, where
  `reference_value` is T29's own `NormalizationResult.reference_value` (imported/reused from
  `iste.phase3_normalization.normalize_support_grid`, never reimplemented). Raises
  `ValueError` if a candidate's `t_s` doesn't match the supplied `NormalizationResult`'s
  timestamp (never silently mixes references across timestamps). `NaN` propagates from T29's
  own degenerate case, never fabricated. No `age`/`t_obs` parameter anywhere. Batch function
  computes each distinct timestamp's `NormalizationResult` once and reuses it (matches
  T30-T33's convention), raising `KeyError` if `support_grids_by_t_s` is missing an entry for
  some candidate's timestamp (explicit, not silent).
- `tests/test_phase3_screening.py` — 12 `unittest` cases: hand-computable formula match plus
  bit-for-bit consistency against T29's own `normalized_grid` at the peak cell (the core
  "not a competing formula" guarantee); within-timestamp monotonic-order preservation;
  raw-score provenance preservation; shared `reference_value` across candidates at the same
  timestamp; a structural no-age-dependence proof (function signature has no `age`/`t_obs`
  parameter, and two candidates with identical `(support_score, reference_value)` at very
  different fabricated timestamps produce identical `screening_score`); `NaN` propagation
  from T29's degenerate all-zero-grid case; a mismatched-timestamp rejection; and batch
  behavior (reuse across shared timestamp, order preserved, missing-grid `KeyError`,
  empty-list handling).
- `notebooks/T34_screening_check.ipynb` — manual-check artifact (see below).

### Tests

```text
python -m unittest tests.test_phase3_screening -v
Ran 12 tests in 0.009s
OK

python -m unittest discover -s tests -v   (full suite)
Ran 618 tests in 25.942s
OK
```

### Manual check

`notebooks/T34_screening_check.ipynb` re-runs T14→T17 on the real cached Stage III result
(regenerating all 280 real candidates AND their support grids, 113.0s — `hotspots.json`
alone doesn't carry the grids T29/T34 need), computes `screening_score` for every one, and
performs the required ordering check. All cells executed via `jupyter nbconvert --execute`,
zero errors:

```text
computed screening scores for 280 real candidates
0 NaN (degenerate-timestamp) screening scores

timestamps with >=2 candidates checked: 44
ordering violations: 0
confirmed: within-timestamp candidate ordering preserved exactly, for every timestamp with >=2 candidates

correlation(age, raw_support_score):    Pearson=-0.725  Spearman=-0.413
correlation(age, screening_score):      Pearson=-0.378  Spearman=+0.013
```

The required monotonicity check passes with **zero violations across all 44 real
timestamps with 2+ candidates** — `screening_score` never reorders candidates within a
timestamp, confirmed on real data (not just the synthetic unit test). The age-correlation
diagnostic (reported, not corrected, per standing instruction) is notable: `screening_score`
shows a WEAKER age correlation than raw `support_score` (Pearson `-0.378` vs. `-0.725`), and
essentially ZERO rank correlation (Spearman `+0.013` vs. `-0.413`) — a different outcome than
the flagging anticipated in the specification (which expected a correlation comparable to
T29's own `+0.923` finding). The difference is explained by scope: T29's `+0.923` figure was
computed only on each timestamp's TOP candidate; this real-data check spans all 280
candidates (most of them small, secondary, low-support detections), so the per-timestamp
`reference_value` normalization appears to cancel out most of the cross-timestamp scale
trend when averaged over the full, mixed candidate population. Reported as-is, exactly as
computed — no interpretation bias applied, no attempt to explain this away as "the fix
worked" (consistent with the standing discipline throughout Phase 3).

### Known limitations

- `screening_score` is mathematically a thin derivative of T29's own per-timestamp
  normalization, NOT independent evidence — **T36 must treat `screening_score` and T29's
  normalized KDE as ONE evidence axis, not two.**
- Not forced into `[0,1]`; kept in T29's own natural range, an explicit design choice (see
  specification above).
- The reported age correlation is dataset/scope-specific (whole real candidate population,
  not just top candidates); it is a diagnostic finding, not validated as a general property.
- No Stage III module, and no existing T14-T17/T25/T26/T29 module, was modified.

**T34 approved 2026-09-24** after explicit provenance verification (direct source
inspection of `iste/phase3_screening.py`): `reference_value` sourced only from T29;
`screening_score = support_score / reference_value` exactly; no age/weighting logic; no
second normalization. See conversation record for the full line-by-line verification.

### T35 — AIS evidence hook (PENDING_REVIEW)

### Files added

- `iste/phase3_ais.py` — `VesselTrackSegment` (documented placeholder shape, never
  instantiated anywhere in this codebase), `AISDataSource(ABC)` (abstract interface, one
  `@abstractmethod` `query_vessels_near`, mirrors `iste.environment.
  EnvironmentalFieldProvider`'s existing "fix the contract now, implement later"
  convention — cannot be instantiated, confirmed by test), `AISEvidence` dataclass, and
  `compute_ais_evidence(candidate, ais_source=None, *, search_radius_m=5000.0,
  time_window_hours=3.0)`: when `ais_source is None` (the ONLY path this project can
  exercise — no AIS data exists anywhere), returns `ais_available=False` with every
  AIS-specific field `None` (never a fabricated `0.0`) and an explicit `reason` string; if a
  concrete `ais_source` were ever supplied, raises `NotImplementedError` rather than
  computing an unvalidated formula — implementing real AIS scoring logic now, with nothing
  to check it against, would mean inventing an untested observation model (`CLAUDE.md`
  guardrail). Also `renormalize_weighted_mean(components: list[tuple[weight, value_or_None]])
  -> float`: a small, general, reusable utility generalizing the exact optional-component
  renormalization pattern T32's `compute_convergence_score` already used ad hoc, packaged
  standalone for T36 (and any future optional-evidence combination) to reuse — drops `None`
  components, renormalizes remaining weights, returns `NaN` (never fabricated) if nothing is
  available or total weight is zero.
- `tests/test_phase3_ais.py` — 17 `unittest` cases: `AISDataSource` cannot be instantiated
  bare or via an incomplete subclass; the unavailable sentinel is returned without raising,
  with every AIS field `None` and an explanatory `reason`, for both the default and explicit
  `ais_source=None` calls; candidate identity preserved; supplying a concrete source raises
  `NotImplementedError`; and a thorough `renormalize_weighted_mean` suite (plain weighted
  mean, unequal weights, one-`None`-component renormalization over the remainder — including
  the exact "AIS present vs. absent, both well-defined" requirement using a synthetic test
  double explicitly labeled as not real evidence — all-`None`/empty/zero-total-weight `NaN`
  cases, and weight validation).
- `notebooks/T35_ais_check.ipynb` — manual-check artifact (see below).

### Tests

```text
python -m unittest tests.test_phase3_ais -v
Ran 17 tests in 0.003s
OK

python -m unittest discover -s tests -v   (full suite)
Ran 635 tests in 24.374s
OK
```

### Manual check

`notebooks/T35_ais_check.ipynb` confirms `data/` contains no AIS/vessel file, runs
`compute_ais_evidence` over all 280 real candidates, confirms the abstract interface cannot
be silently used, and demonstrates `renormalize_weighted_mean` with/without a synthetic AIS
value. All cells executed via `jupyter nbconvert --execute`, zero errors:

```text
data/ contents: ['database_payload.json', 'ocean.nc', 'wind.nc']
confirmed: no AIS/vessel data file present

ran compute_ais_evidence over 280 real candidates -- zero exceptions raised
ais_available=True:  0
ais_available=False: 280
ais_score is None:   280 (must equal total candidate count -- never a fabricated 0.0)

correctly cannot instantiate the bare abstract interface: Can't instantiate abstract class
  AISDataSource without an implementation for abstract method 'query_vessels_near'
correctly raises NotImplementedError if a source were ever supplied

composite score with AIS ABSENT (renormalized over 5 available components): 0.6100
composite score with AIS PRESENT (synthetic test double, 6 components):      0.5667
both well-defined (no NaN, no crash) -- confirms the renormalization mechanics T36 will rely on.
```

### Known limitations

- No AIS data exists in this project; the AIS-present code path is intentionally
  unimplemented (`NotImplementedError`), not a placeholder formula.
- `renormalize_weighted_mean` is a general utility, not AIS-specific — its "present" case
  demonstration used a synthetic placeholder value, explicitly not real evidence.
- No Stage III module, and no existing T14-T17/T25/T26/T29-T34 module, was modified.

### T36 — Composite multi-feature source score: DESIGN SPECIFICATION (2026-09-24, no code written)

Per explicit instruction, specification-only. No `iste/phase3_composite.py`, no tests, no
notebook, no T37 work. `iste/phase3_ais.py`/`phase3_convergence.py`/`phase3_dispersion.py`/
`phase3_initial_bias.py`/`phase3_normalization.py`/`phase3_persistence.py`/
`phase3_screening.py` (T29-T35) were inspected (source + real-data behavior) but NOT
modified. A read-only inspection script computed real-data statistics (`t36_inspection.py`,
scratchpad only, not checked in) by calling T29-T35's existing public functions on the real
280-candidate dataset; this is data-gathering for the design, not a T36 implementation.

#### 1. Existing component inventory (from source inspection, not just task descriptions)

| Component | Module | Candidate-level output field | Natural range (definition) |
|---|---|---|---|
| Support (raw) | T16/T17 | `Candidate.support_score` | Unbounded, raw units |
| T29 normalized KDE | `phase3_normalization.py` | `NormalizationResult.reference_value` (per-timestamp scalar) | Reference scale for `support_score` |
| T30 initial-condition influence | `phase3_initial_bias.py` | `InitialBiasInfluence.influence` | `(0, 1]`, Mahalanobis-based |
| T31 persistence | `phase3_persistence.py` | `PersistenceTrack.persistence_count` (int, hours) / `.temporal_continuity()` | count: `[1, 73]`; continuity: `(0, 1]` |
| T32 convergence | `phase3_convergence.py` | `ConvergenceScore.convergence_score` | Equal-weighted mean of ~`[0,1]` sub-terms |
| T33 dispersion-aware | `phase3_dispersion.py` | `DispersionAwareScore.dispersion_aware_score` | `[0, inf)`, likelihood-ratio-style |
| T34 screening | `phase3_screening.py` | `ScreeningScore.screening_score` | `= support_score / T29_reference_value` |
| T35 AIS | `phase3_ais.py` | `AISEvidence.ais_score` | Always `None` in this project (no data) |

**Confirmed by source inspection: T34 introduces no new computation.**
`iste/phase3_screening.py` line 94: `screening_score = float(candidate.support_score) /
reference_value`, where `reference_value` (line 90) is read directly from the
`NormalizationResult` T29's own `normalize_support_grid` returns (imported at line 48, never
reimplemented). There is no `np.percentile` call, no independent statistic, anywhere in
`phase3_screening.py` (confirmed by grep during T34's own review). **T29 and T34 are the
same evidence lineage — see section 6.**

#### 2. Real-data distribution summary (280 real candidates, full source-lineage recomputation)

```text
                        count      mean       std     min     10%     25%      50%       75%        90%        95%        max
raw_support             280.0  541.1449  973.5004  2.5485  2.7022  2.8368   5.8352  864.1301  1735.7832  2802.3910  4675.4308
density                 280.0  216.9429  418.1796  1.0000  1.0000  1.0000   2.0000  339.2500   656.3000  1159.5500  2123.0000
screening_score         280.0    0.4262    0.5884  0.0020  0.0033  0.0040   0.0075    1.1713     1.4223     1.4832     1.5438
initial_bias_influence  280.0    0.2810    0.4067  0.0000  0.0001  0.0005   0.0036    0.8038     0.9807     0.9910     0.9997
persistence_count       280.0   27.2429   27.3283  1.0000  2.0000  7.0000  14.0000   72.0000    72.0000    72.0000    72.0000
persistence_continuity  280.0    0.9761    0.0562  0.6667  0.9167  1.0000   1.0000    1.0000     1.0000     1.0000     1.0000
convergence_score       280.0    0.4220    0.1604  0.2435  0.2952  0.3117   0.3255    0.5618     0.6970     0.7482     0.7899
dispersion_score        280.0    0.4284    0.6472  0.0014  0.0024  0.0047   0.0076    0.8818     1.2630     1.9625     2.6524
```

`ais_available = True` for **0 / 280** candidates. `initial_bias_influence`'s
`covariance_state` is `"full_rank"` for all 280 (no degenerate-covariance candidates on this
real dataset — see the NaN-handling caveat in section 4).

**Key observation:** `raw_support`, `density`, `screening_score`, `initial_bias_influence`,
and `dispersion_score` are all heavily right-skewed/bimodal (median far below mean, e.g.
`screening_score` median `0.0075` vs. mean `0.4262`) — most of the 280 candidates are minor,
secondary detections at their own timestamp, with a smaller set of dominant candidates
carrying most of the mass. `persistence_continuity` and `convergence_score` are much less
skewed (median close to mean).

#### 3. Pairwise correlation / association analysis (candidate-level, real data)

**Pearson:**

```text
                        raw_support  density  screening_score  initial_bias_influence  persistence_count  persistence_continuity  convergence_score  dispersion_score
raw_support                   1.000    0.996            0.762                   0.839              0.810                   0.235              0.915             0.575
screening_score               0.762    0.717            1.000                   0.979              0.933                   0.305              0.934             0.927
initial_bias_influence        0.839    0.796            0.979                   1.000              0.949                   0.291              0.973             0.852
persistence_count             0.810    0.770            0.933                   0.949              1.000                   0.312              0.929             0.798
persistence_continuity        0.235    0.220            0.305                   0.291              0.312                   1.000              0.353             0.278
convergence_score             0.915    0.881            0.934                   0.973              0.929                   0.353              1.000             0.775
dispersion_score               0.575    0.536            0.927                   0.852              0.798                   0.278              0.775             1.000
```

**Spearman:**

```text
                        raw_support  density  screening_score  initial_bias_influence  persistence_count  persistence_continuity  convergence_score  dispersion_score
raw_support                   1.000    0.945            0.839                   0.819              0.602                   0.229              0.795             0.777
screening_score               0.839    0.909            1.000                   0.773              0.655                   0.268              0.767             0.976
initial_bias_influence        0.819    0.849            0.773                   1.000              0.669                   0.367              0.791             0.764
persistence_count             0.602    0.703            0.655                   0.669              1.000                   0.326              0.753             0.665
persistence_continuity        0.229    0.314            0.268                   0.367              0.326                   1.000              0.538             0.305
convergence_score             0.795    0.869            0.767                   0.791              0.753                   0.538              1.000             0.764
dispersion_score               0.777    0.871            0.976                   0.764              0.665                   0.305              0.764             1.000
```

`screening_score` vs. `raw_support`: Pearson `0.7623`, Spearman `0.8392` (the T29/T34
equivalence is a *formula* identity, not a correlation — the imperfect correlation here is
expected, since `screening_score` divides by a *per-timestamp* reference, so it is not a
monotonic transform of `raw_support` *across* timestamps, only *within* one).

**Age correlations (candidate-level, all 280, not just top-candidate-per-timestamp):**

```text
  raw_support               Pearson=-0.725  Spearman=-0.413
  density                   Pearson=-0.728  Spearman=-0.344
  screening_score           Pearson=-0.378  Spearman=+0.013
  initial_bias_influence    Pearson=-0.497  Spearman=-0.316
  persistence_count         Pearson=-0.489  Spearman=-0.289
  persistence_continuity    Pearson=-0.076  Spearman=-0.091
  convergence_score         Pearson=-0.593  Spearman=-0.342
  dispersion_score          Pearson=-0.132  Spearman=+0.110
```

**Important scope note on `dispersion_score`'s age behavior:** T33's own closed diagnostic
(approved 2026-09-24) established a strong age correlation (`Pearson +0.863`/`Spearman
+0.862`, strengthening to `+0.951`/`+0.943` under the empirical null) — but that analysis
was computed on the **top candidate per timestamp only** (72 points). Across the **full
280-candidate population** T36 will actually score, the age correlation of
`dispersion_score` is much weaker (`Pearson -0.132`/`Spearman +0.110`). Both findings are
correct and are reported here together — they describe different populations (the dominant
trajectory alone vs. every detected candidate). This is not a reversal of T33's closed
finding; it is additional, population-scoped information relevant specifically to T36's use
case. `screening_score` shows the identical pattern (T29's top-candidate `+0.923` vs. this
population-level `-0.378`/`+0.013`), already noted in T34's own approved write-up.

#### 4. Double-counting assessment

**This is the central, and largest, finding of this design phase — larger than the
already-known T29/T34 identity.** Every candidate-level evidence component EXCEPT
`persistence_continuity` shows strong pairwise association with every other component on
this real dataset (Pearson typically `0.72-0.98`, Spearman typically `0.60-0.98`). Using the
required careful language: this is a **strong empirical association** on this specific
280-candidate, single-spill, real dataset — it does not prove formal statistical dependence,
and a different spill/window/forcing configuration might show a different structure.

The physical/structural reason this association exists: T17 retains **multiple local modes
per timestamp**, but in this real run one dominant mode plus several minor secondary
detections is the typical pattern (confirmed by the skewed distributions in section 2).
Nearly every candidate-level metric responds, directly or indirectly, to "how large/dominant
is this candidate's supporting particle population" — so on a population shaped like this
one (few dominant candidates, many minor ones), metrics that are *conceptually* independent
(density, Mahalanobis-closeness-to-centroid, cross-forcing-member/pixel diversity,
dispersion-relative concentration, track length) end up *empirically* highly correlated,
because they are all responding to the same underlying dominant-vs-minor split.

**`persistence_continuity` is the one clear exception** (`0.22-0.37` with everything else) —
it answers a genuinely different question ("how unbroken is this candidate's own track,"
independent of how big the candidate is) and is also the most age-neutral component
(`Pearson -0.076`/`Spearman -0.091`). **`persistence_count`**, by contrast, IS part of the
correlated cluster (`0.60-0.95`) — it is more a proxy for "size/dominance" than for
"continuity" specifically (a long-lived track is usually also a big one, on this dataset).

**Design consequence:** summing several of these strongly-associated components with
independent, comparably large weights would not combine five independent lines of evidence
— it would count the *same* underlying "dominance" signal five times under five different
names, which is precisely the double-counting failure mode this task must avoid. This drives
the weighting philosophy in sections 7-11 below: the highly-associated components
(`convergence_score`, `dispersion_score`, and — separately, see section 7 — `initial_bias_
influence`) receive deliberately modest weights specifically *because of* this empirical
structure, not because their definitions are considered unimportant.

#### 5. Proposed normalization for each component

| Component | Proposed treatment | Preserves | Changes | Why appropriate | New temporal bias? |
|---|---|---|---|---|---|
| `screening_score` (T29/T34) | Used as-is, no further rescale | Per-timestamp relative ordering (T29's own design) | Nothing new | Real range (`0.002-1.54`) is already roughly comparable in order of magnitude to the other active components (see section 8) — unlike raw `support_score` (`2.5-4675`), which is exactly why T29/T34 exist | No — inherits T29's already-reported, already-accepted residual (see T29's approval) |
| `persistence_continuity` | Used as-is | Its own `(0,1]` bound | Nothing | Already bounded, already low variance is an honest data property, not something to "fix" | No — near-zero age correlation empirically |
| `convergence_score` | Used as-is | Its own approx-`[0,1]` range (already an equal-weighted mean of bounded sub-terms, T32's own design) | Nothing | Already bounded and interpretable | Inherits T32's own reported age correlation (not independently re-derived here) |
| `dispersion_score` | Used as-is | Its own `[0, inf)` range | Nothing | Real range (`0.001-2.65`) comparable order of magnitude to the others; forcing it into `[0,1]` via e.g. a sigmoid would be an unjustified nonlinear transform this task explicitly warns against | No new bias added; T33's own established age finding (both scopes) carried forward as-is |
| `initial_bias_influence` | **Not weighted in the default composite — diagnostic field only** | Full precision, reported per candidate | N/A (not in the sum) | See section 7 — sign is not scientifically justified given the data | N/A |
| `persistence_count` | **Not weighted in the default composite — diagnostic field only** | Full precision, reported per candidate | N/A | Empirically redundant with the "dominance" cluster; `persistence_continuity` carries T31's actual composite contribution | N/A |
| `ais_score` | Used as-is when available (never today) | N/A | N/A | Always dropped + renormalized (section 9) | N/A |

**No min-max rescaling anywhere.** Min-max would (a) compress the real, meaningful skew in
`screening_score`/`dispersion_score` into an arbitrary `[0,1]` window whose edges are set by
this ONE dataset's extreme values (not a stable, reusable scale across future runs), and (b)
was explicitly warned against. The four active components' real ranges (`0.002-1.54`,
`0.667-1.0`, `0.244-0.79`, `0.001-2.65`) are close enough in order of magnitude that no
rescaling is applied; this is stated as a data-justified decision, re-checked against
whatever real distribution a future run produces, not assumed universally true.

#### 6. Proposed treatment of T29/T34 (required)

**`support_evidence(candidate) := screening_score(candidate)`** (T34's own field, which is
*defined as* `support_score / T29_reference_value`). T29's `normalized_support` grid and
T34's `screening_score` are **one evidence axis**, entering the composite exactly once, via
a single weight `w_support`. T34 is retained purely as the candidate-level provenance-
carrying adapter it was designed to be (raw `support_score`, the `reference_value` used, and
D/W/V carried through) — it does not introduce a second term. `iste.phase3_normalization`'s
grid-level output is not separately re-entered into the composite anywhere.

#### 7. Proposed treatment of T30 (required — full reasoning, not inherited from the original spec)

**What `initial_bias_influence` actually represents** (per T30's own approved module
docstring and design record): the Mahalanobis distance from a candidate to *its own
timestamp's whole particle population's centroid*, exponentiated. High `influence` means the
candidate is statistically indistinguishable from an ordinary, undifferentiated member of
the bulk population at that instant — **not** "unlikely source," per T30's own explicit,
already-recorded caveat.

**Why simply subtracting it (the original spec's convention) is not justified by this
data:** `initial_bias_influence` correlates `0.839` (Pearson) with `raw_support` and `0.973`
with `convergence_score` — among the STRONGEST pairwise associations in the whole matrix.
This is structurally expected: the dominant candidate at a timestamp usually *is* close to
the population's own centroid (it typically contains the bulk of the mass), so "close to
centroid" and "large/dominant/well-supported" are nearly the same fact on this dataset.
Subtracting `initial_bias_influence` with any material weight would therefore **penalize
almost exactly the candidates every other evidence component favors**, for a reason
(closeness to centroid) that is a near-mechanical consequence of being the dominant
candidate, not independent evidence of implausibility. That would make top-rank candidates
unstable (large positive terms nearly cancelled by a large negative term driven by the same
underlying factor) and would not have a data-supported justification.

**Why simply adding it is equally unjustified:** would just reinforce the same "dominance"
signal already carried by `support_evidence`/`convergence_score`, worsening the
double-counting problem in section 4 rather than adding independent information.

**Proposed resolution: `initial_bias_influence` remains a reported DIAGNOSTIC field on every
candidate (default weight `w_initial = 0.0`), not part of the summed composite.** This is
directly authorized by the task's own framing ("If it should remain diagnostic rather than
directly scored, explain that alternative"). The weight is configurable and can be set
non-zero (either sign) for exploratory sensitivity analysis outside the three required
configurations, but no default sign is asserted, because neither is defensible from the
current data or from T30's own established interpretation.

#### 8. Proposed treatment of T33 (required)

`dispersion_score` enters the composite as an ordinary **positively-signed** evidence term
(`w_dispersion > 0`; higher = more concentrated than generic dispersion predicts, per T33's
own established, approved interpretation) with **no age-correction term of any kind** —
confirmed absent by construction (the formula only reads `(support_score, reference_value)`-
analogous inputs, never `age`/`t_obs`; same discipline as T30/T33's own approved modules).
Both of the following facts are carried forward together, not selectively:

- T33's own closed diagnostic: top-candidate-only age correlation is strong
  (`Pearson +0.863` to `+0.951` depending on null model).
- This design phase's finding: population-level (all 280 candidates) age correlation is
  much weaker (`Pearson -0.132`/`Spearman +0.110`).

Neither fact is used to justify adding a corrective term. The weight (`w_dispersion`) is
kept moderate, not because of the age relationship, but because of `dispersion_score`'s own
strong empirical association with `screening_score` (`0.927`) and `initial_bias_influence`
(`0.852`) noted in section 4 — the same double-counting reasoning applied to every
"dominance-cluster" component.

#### 9. Proposed AIS-missing behavior (required)

AIS unavailability is represented as `ais_score = None` (T35's own existing sentinel,
unmodified) and is combined via T35's own existing, already-tested
`renormalize_weighted_mean(components: list[tuple[weight, value_or_None]])` utility, reused
directly (no new renormalization logic written for T36):

```text
active = [(w_i, v_i) for (w_i, v_i) in components if v_i is not None]
source_score = sum(w_i * v_i for w_i, v_i in active) / sum(w_i for w_i, v_i in active)
```

When `ais_score is None` (every candidate in this project today), the `(w_ais, None)` pair
is dropped from `active` entirely — its configured weight contributes to neither the
numerator nor the denominator, so the remaining active weights are automatically
renormalized to sum to their own total (mathematically equivalent to re-scaling them
proportionally so they sum to 1 among themselves). **AIS absence never contributes 0 to the
score and never lowers it relative to an AIS-available candidate scored identically on every
other component** — this is the direct, already-verified (T35's own test suite) behavior of
the reused utility. If real AIS data is integrated later, the SAME formula and SAME default
weight `w_ais` (proposed 0.20, see section 11) apply without any T36 code change — only
`ais_source` becoming non-`None` and returning real evidence instead of raising
`NotImplementedError`.

#### 10. Exact composite mathematical formula

```text
support_evidence(c)        = screening_score(c)                          [T29/T34, one axis]
persistence_evidence(c)    = temporal_continuity(track(c))                [T31]
convergence_evidence(c)    = convergence_score(c)                         [T32]
dispersion_evidence(c)     = dispersion_aware_score(c)                    [T33]
ais_evidence(c)             = ais_score(c)  if ais_available(c) else None  [T35]
initial_bias_evidence(c)    = initial_bias_influence(c)   -- DIAGNOSTIC ONLY, weight 0 by default
persistence_count_evidence(c) = persistence_count(c)      -- DIAGNOSTIC ONLY, weight 0 by default

components(c) = [
    (w_support,             support_evidence(c)),
    (w_persistence,         persistence_evidence(c)),
    (w_convergence,         convergence_evidence(c)),
    (w_dispersion,          dispersion_evidence(c)),
    (w_ais,                 ais_evidence(c)),                # None unless real AIS data exists
    (w_initial,             initial_bias_evidence(c)),        # default weight 0.0
    (w_persistence_count,   persistence_count_evidence(c)),   # default weight 0.0
]

source_score(c) = renormalize_weighted_mean(components(c))   # T35's own utility, reused unchanged
```

**Score range:** no fixed upper bound (inherits `dispersion_evidence`'s and
`support_evidence`'s unboundedness); no fixed probabilistic interpretation.
**`source_score` is an evidence-ranking score, not a probability, confidence, likelihood, or
certainty of source identity** — stated explicitly per the task's own requirement. It is
valid for ranking candidates *within one run/configuration*, not as an absolute scale
comparable across different spills, windows, or weight configurations.

T25's separate forward-accuracy filtering/ranking is untouched and not folded into this
score, per the task's own framing.

#### 11. Default weight configuration (real-data-informed, not outcome-tuned)

```text
w_support             = 0.35
w_persistence          = 0.20
w_convergence           = 0.10
w_dispersion            = 0.15
w_ais                    = 0.20   (contributes 0 today; always renormalized away, AIS absent in this project)
w_initial (T30)          = 0.00   (diagnostic only -- see section 7)
w_persistence_count      = 0.00   (diagnostic only -- redundant with the dominance cluster)
                          -----
                    sum = 1.00
```

**These weights are heuristic and configurable — not claimed as scientifically validated.**

#### 12. Rationale for every default weight (one line each)

- `w_support = 0.35`: the largest single weight because `support_evidence` is the most
  directly interpretable, longest-established signal (T16→T17→T29→T34 lineage) and the one
  every other Phase 3 component was built to refine or cross-check, not because it "worked
  best" on any outcome.
- `w_persistence = 0.20`: the second-largest weight specifically *because*
  `persistence_continuity` is the one component with weak empirical association with
  everything else (section 4) and near-zero age correlation — genuinely additive
  information, not a restatement of dominance.
- `w_convergence = 0.10`: kept small because of its very strong empirical association with
  `support_evidence` (`0.934`) and `initial_bias_influence` (`0.973`) on this dataset — real,
  distinct-by-construction evidence (independent-trajectory diversity), but downweighted to
  avoid reinforcing the same latent "dominance" signal already carried by `w_support`.
- `w_dispersion = 0.15`: moderate weight; real, distinct-by-construction correction for
  cross-timestamp dispersion, but also strongly support-associated (`0.927`) on this
  dataset, and carries the known age-behavior caveat from section 8 (population-level weak,
  top-candidate-level strong) — kept below `w_support`/`w_persistence` pending further
  investigation, not zeroed, since it does represent a genuinely different DEFINITION
  (concentration relative to expected dispersion) even where empirically correlated here.
- `w_ais = 0.20`: a substantial *reserved* weight reflecting AIS's intended importance as
  genuinely independent, non-derived evidence once real data exists — deliberately not a
  token/negligible weight, specifically so AIS will not be effectively irrelevant on the day
  it becomes available; contributes exactly 0 to any score computed in this project today.
- `w_initial (T30) = 0.00`: no default sign is scientifically justified (section 7) — kept
  as a reported diagnostic, not scored, until a specific, data-supported interpretation is
  established.
- `w_persistence_count = 0.00`: empirically redundant with the dominance cluster (section
  4); `persistence_continuity` already carries T31's intended composite contribution.

#### 13. Three sensitivity-analysis configurations (proposed, not yet run)

Weights chosen from the evidence-role/scale reasoning above, not from inspecting which
candidates would enter/leave any top-10 — no such check was performed before proposing
these.

```text
                          DEFAULT   PERSISTENCE-EMPHASIS   DISPERSION-EMPHASIS
w_support                  0.35            0.25                   0.25
w_persistence               0.20            0.40                   0.15
w_convergence                0.10            0.10                   0.10
w_dispersion                 0.15            0.10                   0.35
w_ais                         0.20            0.15                   0.15
w_initial (T30)                0.00            0.00                   0.00
w_persistence_count            0.00            0.00                   0.00
                              ----            ----                   ----
sum                          1.00            1.00                   1.00
```

- **PERSISTENCE/TRAJECTORY-EMPHASIS**: doubles `w_persistence` (0.20->0.40) at the expense
  of `w_support`/`w_dispersion`, to test how much the ranking shifts toward long,
  well-tracked candidates over raw-magnitude-dominant ones.
- **DISPERSION/PHYSICAL-CONSISTENCY-EMPHASIS**: more than doubles `w_dispersion`
  (0.15->0.35) at the expense of `w_support`/`w_persistence`, to test how much the ranking
  shifts toward candidates whose concentration most exceeds generic dispersion expectations.

At implementation time, each configuration's real-data top-10, rank changes, score changes,
overlap with the DEFAULT top-10, and inter-configuration rank correlation (Spearman) would
be computed and reported — not done in this specification-only phase.

#### 14. Synthetic test plan (`tests/test_phase3_composite.py`, not yet created)

1. **Monotonicity** — for each of `support_evidence`/`persistence_evidence`/
   `convergence_evidence`/`dispersion_evidence`, increasing that component alone (holding
   all others fixed) strictly increases `source_score`, given a positive weight.
2. **Weight normalization** — for a candidate with all components available, active weights
   sum to the configured total; `source_score` equals the plain weighted mean.
3. **Missing AIS** — `ais_score=None` produces a well-defined, non-`NaN` `source_score`
   using only the other active components; equals `renormalize_weighted_mean` computed by
   hand over the non-AIS terms.
4. **T29/T34 non-duplication** — a structural test confirming the composite's component
   list contains exactly ONE term derived from the Support/T29/T34 lineage (e.g., by
   constructing two synthetic candidates that differ only in `support_score` and confirming
   the resulting `source_score` difference matches a single `w_support`-weighted
   contribution, not a doubled one).
5. **Raw-KDE domination (synthetic)** — a synthetic candidate with very high `support_
   evidence` but deliberately weak `persistence_evidence`/`convergence_evidence`/
   `dispersion_evidence` must NOT automatically outrank a candidate with more balanced
   evidence across all components at comparable `support_evidence` — demonstrating the
   composite is not simply `support_evidence` relabeled.
6. **Persistence evidence** — holding all else equal, a candidate with strictly higher
   `persistence_continuity` ranks strictly higher (given `w_persistence > 0`).
7. **T33 age behavior (structural)** — confirm, via `inspect.signature`, that the composite
   function accepts no `age`/`t_obs` parameter anywhere in its call chain (mirrors the
   T30/T34 structural no-age-term proofs already established).
8. **T30 sign (structural)** — confirm the DEFAULT configuration's `w_initial == 0.0`
   exactly, and confirm `initial_bias_influence` is still reported on every result object
   (diagnostic, not silently dropped).

#### 15. Raw-KDE domination test plan (real data, not yet run)

At implementation time: compute `source_score` (DEFAULT config) for all 280 real candidates;
compare its ranking against the existing raw `support_score` (T17) ranking via Spearman rank
correlation, top-10 overlap (set intersection size), which candidates enter/leave the
top-10, and per-candidate rank-change magnitude. Given the real correlation structure
already found in section 3 (`support_evidence` alone already correlates `0.762-0.934` with
several other active components), some residual correlation between the new composite and
raw Support is EXPECTED and would not by itself indicate failure — the test's purpose is to
confirm the composite is not IDENTICAL to raw-Support ranking, and to show concretely which
candidates the added components (persistence/convergence/dispersion) actually move.

#### 16. Known limitations

- The double-counting analysis (section 4) is based on ONE real spill/window/forcing
  configuration (280 candidates, 3-day/1-hour window, `ow-0008`, 6-member subset) — the
  correlation structure driving the proposed weights is dataset-specific, described with
  "strong empirical association" language throughout, never claimed as a general/formal
  independence result.
- **Implementation gap discovered during this design phase, not yet fixed (no code
  written):** `renormalize_weighted_mean` (T35) currently filters only `value is None`, not
  `NaN`. `initial_bias_influence`/`dispersion_aware_score` CAN return `NaN` for degenerate-
  covariance timestamps (T30/T33's own documented behavior) — on the current real dataset
  this never triggers (all 280 candidates have `covariance_state="full_rank"`), but T36's
  implementation must explicitly treat `NaN` the same as `None` when building
  `components(c)`, or a single degenerate candidate would silently `NaN`-poison its entire
  `source_score`. Flagged here for the implementation task, not fixed now.
- `w_ais`'s real effect is entirely untested (no AIS data exists) — its value is a
  documented placeholder, not empirically informed.
- No claim that this weighting scheme is optimal, calibrated, or validated against any
  ground truth (none exists for this project).

#### 17. Scientific caveats (explicitly restated, per instruction)

1. Weights are heuristic/configurable, not trained or externally validated.
2. `source_score` is not a calibrated probability — an evidence-ranking score only.
3. Correlated evidence streams (section 4) can cause effective double-counting even when
   implemented as separate, individually well-defined features — addressed here by
   deliberate weight allocation, not by removing any component's definition.
4. T33's population-level weak age correlation does not overturn or "fix" its own
   top-candidate-level strong finding (section 3) — both are true, of different scopes.
5. AIS absence is missing information, never evidence against a candidate (section 9).
6. T34 is not independent of T29 — one evidence axis, entered once (section 6).
7. The existing Stage III physical backtracking model (T00-T13.5) is completely unchanged
   by this design; T36 operates purely on already-computed T14-T35 outputs.
8. A high `source_score` is evidence-based ranking, not proof of actual physical source
   identity, exactly like every other Phase 2/3 screening quantity in this project.

#### 18. Files that would be changed during implementation (not created now)

- `iste/phase3_composite.py` (new) — `CompositeWeights`, `SourceScore` dataclass,
  `compute_source_score`/`compute_source_scores_for_candidates`, reusing T29-T35's public
  functions and T35's `renormalize_weighted_mean` directly.
- `tests/test_phase3_composite.py` (new) — the 8-item synthetic plan in section 14.
- `notebooks/T36_composite_check.ipynb` (new) — real-data sensitivity table (section 13) and
  raw-KDE domination comparison (section 15).
- `tasks.md`/`progress.md`/`memory.md` updates, as with every prior Phase 3 task.
- **No existing file (`iste/phase3_normalization.py`, `phase3_initial_bias.py`,
  `phase3_persistence.py`, `phase3_convergence.py`, `phase3_dispersion.py`,
  `phase3_screening.py`, `phase3_ais.py`, or any T00-T26 module) would be modified.**

#### 19. Explicit statement

**No production code has been modified during this design phase.** `t36_inspection.py`
(scratchpad-only, not part of the repository) called existing T29-T35 public functions
read-only to gather the real-data numbers in sections 2-3; it wrote no output into `iste/`,
`tests/`, or `notebooks/`. T29 through T35 remain exactly as approved. T37 has not been
started.

### T36 DESIGN AMENDMENTS (2026-09-24, no code written) — supersedes parts of sections 9, 7, 8, 5 above

Four amendments requested before any implementation; still specification-only. Nothing in
`iste/` or `tests/` changed. Sections 1-4, 6, 10-19 of the original spec above are
unaffected; sections 5, 7, 8, 9 are amended/extended as follows.

#### Amendment 1 — explicit AIS active-weight renormalization formula (replaces section 9's prose description)

```text
AVAILABLE(c) = { i : evidence_i(c) is not None }         # T35's own filter, reused unchanged

w_eff_i(c) = w_i / sum_{j in AVAILABLE(c)} w_j             for i in AVAILABLE(c)
             0                                              for i not in AVAILABLE(c)

source_score(c) = sum_{i in AVAILABLE(c)} w_eff_i(c) * evidence_i(c)
                 = sum_{i in AVAILABLE(c)} w_i * evidence_i(c)  /  sum_{j in AVAILABLE(c)} w_j
```

This is exactly T35's already-tested `renormalize_weighted_mean`, restated with the
requested `w_eff` notation — no new mechanism. `AVAILABLE(c)` is determined by whether each
component's VALUE is `None`, never by its configured weight — a component with
`w_i = 0.0` (e.g. `initial_bias_influence`, `persistence_count`) is still `AVAILABLE`
whenever its value is defined (always, for full-rank timestamps); it simply contributes `0`
to both numerator and denominator either way, so its presence/absence in `AVAILABLE` has no
numerical effect. Only AIS is actually ever excluded from `AVAILABLE` in this project today.

**Worked example, DEFAULT configuration, AIS unavailable (every candidate in this project
today) — configured weights vs. effective weights, reported side by side:**

```text
component               w_i (configured)   in AVAILABLE?   w_eff_i (renormalized)
support_evidence              0.35               yes              0.35/0.80 = 0.4375
persistence_evidence           0.20               yes              0.20/0.80 = 0.2500
convergence_evidence            0.10               yes              0.10/0.80 = 0.1250
dispersion_evidence              0.15               yes              0.15/0.80 = 0.1875
initial_bias_evidence (T30)       0.00               yes              0.00/0.80 = 0.0000
persistence_count_evidence         0.00               yes              0.00/0.80 = 0.0000
ais_evidence                        0.20               NO (None)         --  (excluded, not 0)
                                    ----                                  ----
                              sum(configured, all 7) = 1.00      sum(w_eff over AVAILABLE) = 1.0000
```

`sum_{j in AVAILABLE} w_j = 0.35+0.20+0.10+0.15+0.00+0.00 = 0.80` (AIS's `0.20` excluded from
the denominator entirely, not treated as `0` evidence). The implementation's output object
for every candidate will report BOTH the configured weight vector (fixed, from
`CompositeWeights`) AND the effective (`w_eff`) vector actually used for that candidate, so a
reader can always see whether/how renormalization occurred — required so AIS absence is
visibly "excluded" rather than silently "zeroed," satisfying "AIS must never be treated as
negative evidence" directly in the reported data structure, not just in prose.

#### Amendment 2 — T30 rationale, primary vs. supporting reasons (replaces section 7's "why subtraction/addition" framing)

**Revised primary reason (scientific, not statistical):** `initial_bias_influence` measures
Mahalanobis-distance consistency between a candidate and its own timestamp's whole particle
ensemble — a purely geometric/statistical-consistency quantity. **It has no scientifically
justified DIRECTIONAL interpretation as source evidence.** T30's own design record already
establishes that high influence means "consistent with being an ordinary member of the
bulk," and explicitly refuses to equate that with either "more likely" or "less likely" to
be a genuine source — the module was deliberately built to answer a geometric consistency
question, not a source-likelihood question. Assigning it either `+w_initial` or `-w_initial`
would require importing a directional claim (e.g. "geometric outliers are more/less
plausible sources") that T30 was never designed or validated to support. **This absence of a
justified sign — not the correlation structure — is the primary reason
`w_initial = 0.0` by default.**

**Supporting (secondary) reason, kept from the original analysis:** the strong empirical
correlation with `support_evidence` (`0.839`) and `convergence_evidence` (`0.973`) means
that even if a directional interpretation existed, including it at material weight would
additionally risk double-counting the dominance cluster (section 4) — this is offered as
*additional* justification for why, even under a hypothetical future directional
interpretation, a large weight would still be inappropriate without first addressing the
correlation structure. It is not, on its own, the reason the sign is left unassigned.

`initial_bias_influence` continues to be computed and reported on every candidate
(diagnostic field, full precision) — never silently dropped — so a future task that
*does* establish a scientifically justified directional interpretation can assign
`w_initial != 0` without recomputing anything.

#### Amendment 3 — T33 explicit ranking-vs-probability statement (extends section 8)

Added verbatim to the specification: **`dispersion_evidence`'s contribution to
`source_score` is evidence-ranking information, not a calibrated probability, and not an
age-neutral likelihood.** Three distinct claims, stated separately so none is
accidentally implied by omission:

- Not a calibrated probability: `dispersion_aware_score` is a likelihood-ratio-*style*
  quantity (unbounded above, `~1` = matches generic-dispersion expectation) but was never
  fit or validated against any ground-truth source-location dataset — no calibration
  exists, per T33's own approval record.
- Not age-neutral: both of section 3's findings (strong top-candidate-level age
  correlation; weak population-level age correlation) are real properties of this
  component that persist into the composite unmodified — `w_dispersion > 0` does not
  imply or require age-neutrality, and none is claimed.
- Evidence-ranking only: like every other component, its role is to shift relative
  ordering among candidates given the configured weight, not to assert an absolute,
  cross-run-comparable score.

#### Amendment 4 — exact normalization function for every active component (replaces section 5's summary table for these four; section 5's `initial_bias`/`persistence_count`/`ais` rows are unchanged)

**`support_evidence(c)`**

```text
formula:        support_evidence(c) = support_score(c) / reference_value(t_s(c))
                 reference_value(t) = percentile_95({Support(x,y,t) : Support(x,y,t) > 0})
                 [T29's own formula, iste.phase3_normalization.normalize_support_grid, reused unchanged]
input range:     support_score(c) in [0, infinity)  (raw T16 Support units)
output range:    empirically [0.002, 1.544] on real data; theoretically (0, infinity)
monotonicity:    strictly increasing in support_score(c) for FIXED t_s (reference_value(t_s)
                 is a positive constant per timestamp -- T34's own tested property).
                 NOT monotonic in support_score ACROSS different timestamps by design (a
                 modest candidate at a "quiet" timestamp can outscore a larger candidate at
                 a "busy" one) -- this is the entire mechanism T29 uses to remove the
                 cross-timestamp KDE-magnitude bias T28 measured.
fit scope:       PER-TIMESTAMP (reference_value recomputed independently from each
                 timestamp's own grid; never fit globally across timestamps).
temporal bias:   mitigated by design, not eliminated -- T29's own approved residual
                 (+0.923 top-candidate) and this design phase's population-level finding
                 (-0.378 Pearson / +0.013 Spearman) both stand, reported not corrected.
rationale:       reuses T29/T34's already-approved formula unchanged; T36 adds no transform.
```

**`persistence_evidence(c)`**

```text
formula:         persistence_evidence(c) = temporal_continuity(track(c))
                 = persistence_count(track(c)) / (round(persistence_duration_hours(track(c))
                   / interval_hours) + 1),  interval_hours = 1.0 (T27's locked resolution)
                 [T31's own formula, iste.phase3_persistence.PersistenceTrack.temporal_continuity, reused unchanged]
input range:     persistence_count in {1, ..., 73} (integer hourly slots);
                 persistence_duration_hours in [0, 72]
output range:    (0, 1] exactly, by construction
monotonicity:    for FIXED persistence_duration_hours, strictly increasing in
                 persistence_count; for FIXED persistence_count, strictly decreasing in
                 persistence_duration_hours. Not a function of a single scalar input alone
                 -- both quantities are emergent properties of T31's own tracking algorithm,
                 not directly chosen per candidate.
fit scope:       PER-TRACK (a third category, distinct from "global" or "per-timestamp" --
                 the ratio is intrinsic to the one T31 track a candidate belongs to,
                 self-referential to that track's own span, not fit against any other
                 candidate or timestamp).
temporal bias:   empirically the WEAKEST of any component on real data (Pearson -0.076,
                 Spearman -0.091, section 3) -- no age term anywhere in T31's tracking or
                 continuity formula (confirmed by source inspection during T31's approval).
rationale:       reuses T31's already-approved formula unchanged; T36 adds no transform.
```

**`convergence_evidence(c)`**

```text
formula:         convergence_evidence(c) = convergence_score(c)
                 = [ w_pf*particle_fraction + w_pd*pixel_diversity_fraction
                     + w_md*member_diversity_fraction + w_cc*concentration_score
                     + w_tc*temporal_consistency ] / [sum of active sub-weights],
                   all sub-weights = 1.0 by T32's own default ConvergenceWeights
                 [T32's own formula, iste.phase3_convergence.compute_convergence_score, reused unchanged]
input range:     each sub-term in approximately [0,1]: particle_fraction/pixel_diversity_
                 fraction/member_diversity_fraction are true fractions in [0,1];
                 concentration_score = exp(-x) in (0,1]; temporal_consistency (T31's
                 continuity) in (0,1] when available.
output range:    approximately [0,1] (weighted mean of [0,1]-ish terms); empirically
                 [0.2435, 0.7899] on real data.
monotonicity:    strictly increasing in EACH sub-component individually, holding the others
                 fixed (guaranteed by construction of a positive-weight mean -- T32's own
                 tested property); therefore also strictly increasing in convergence_score
                 as used here, trivially (identity pass-through).
fit scope:       MIXED, inherited unchanged from T32: particle_fraction/pixel_diversity_
                 fraction/member_diversity_fraction are normalized PER-TIMESTAMP (divided by
                 that timestamp's own totals); concentration_score is PER-CANDIDATE (uses
                 the candidate's own core_radius_m); temporal_consistency is PER-TRACK (T31).
temporal bias:   this design phase's new candidate-level finding: Pearson -0.593 / Spearman
                 -0.342 (section 3) -- reported here for the first time at this granularity;
                 carried forward as a diagnostic, no correction applied.
rationale:       reuses T32's already-approved formula and default sub-weights unchanged;
                 T36 adds no transform.
```

**`dispersion_evidence(c)`**

```text
formula:         dispersion_evidence(c) = D_actual(c) / D_expected(t_s(c))
                 D_expected(t) = N_valid(t) * (1 - exp(-r^2 / (2*sigma_iso_sq(t))))
                 sigma_iso_sq(t) = trace(Sigma(t)) / 2   [T30's own covariance, reused]
                 [T33's own formula, iste.phase3_dispersion.compute_dispersion_aware_score, reused unchanged]
input range:     D_actual(c) in [0, infinity) (T16 density count at the candidate's peak);
                 D_expected(t) in (0, N_valid(t)] (bounded above -- a probability times a count)
output range:    [0, infinity), unbounded above (a likelihood-ratio-style "surprise"
                 quantity, per T33's own approved docstring); empirically [0.0014, 2.6524]
                 on real data.
monotonicity:    strictly increasing in D_actual(c) holding t_s fixed; strictly DECREASING
                 in D_expected(t) holding D_actual fixed -- equivalently, increasing in the
                 timestamp's own dispersion (sigma_iso_sq) for fixed D_actual/N/r. This
                 directional behavior is exactly T33's designed purpose (a fixed raw density
                 at a MORE dispersed timestamp scores HIGHER, per T33's own tested
                 "same raw density scores higher when more dispersed" property).
fit scope:       PER-TIMESTAMP (Sigma(t) and N_valid(t) both computed fresh per timestamp
                 from that timestamp's own particle population via T30's
                 compute_timestamp_cloud_stats, reused; no global fitting across timestamps).
temporal bias:   both section 3 findings carried forward together (strong top-candidate-
                 level Pearson +0.863 to +0.951; weak population-level Pearson -0.132 /
                 Spearman +0.110) -- reported, not corrected; no age/t_obs parameter exists
                 anywhere in the formula (confirmed by source inspection at T33's approval).
rationale:       reuses T33's already-approved (with major caveat, see Amendment 3) formula
                 unchanged; T36 adds no transform.
```

**No component in this amendment introduces a new mathematical transform.** Every formula
above is a direct citation of an already-approved T29-T33 function; T36's only new
contribution is the WEIGHTED COMBINATION (Amendment 1's `w_eff`/`source_score` formula), not
any per-component rescaling.

Weight values are unchanged from the original specification (section 11): still proposed,
still not implemented, still not claimed as validated.

**No production code has been modified by this amendment.** T29-T35 unchanged. T37 not
started.

### T36 — Composite multi-feature source score: IMPLEMENTATION (PENDING_REVIEW)

Implemented exactly per the approved DESIGN SPECIFICATION + DESIGN AMENDMENTS above — no
design changes during implementation, beyond the already-flagged, already-approved-for-
fixing `NaN`-vs-`None` handling gap (fixed inside `phase3_composite.py` only, `iste/
phase3_ais.py` itself untouched).

#### Files added

- `iste/phase3_composite.py` — `CompositeWeights` (7 fields, `w_initial`/
  `w_persistence_count` default `0.0`, all validated `>= 0`, matching every other Phase 3
  `*Weights` dataclass's convention), `DEFAULT_WEIGHTS`/`PERSISTENCE_EMPHASIS_WEIGHTS`/
  `DISPERSION_EMPHASIS_WEIGHTS` module constants, `SourceScore` dataclass (reports both
  `configured_weights` and `effective_weights` per candidate, per Amendment 1),
  `compute_source_score` (single candidate, takes each T29/T30/T31/T32/T33/T35 result
  object explicitly — validates every `candidate_id` matches, raises `ValueError`
  otherwise) and `compute_source_scores_for_candidates` (batch, wires up T34/T31/T32/T33/T35
  via each module's own unmodified batch function, plus T30 via a per-timestamp-reused
  `compute_timestamp_cloud_stats` loop matching T30's own established pattern). The
  `NaN`-vs-`None` gap flagged in the design spec's "Known limitations" is fixed here (a
  private `_nan_to_none` helper converts `NaN` evidence to `None` before passing into T35's
  `renormalize_weighted_mean`, since `0.0 * float('nan') == nan` in IEEE arithmetic would
  otherwise poison the whole sum even at zero weight) — `iste/phase3_ais.py` itself was not
  modified; the fix lives entirely in the new module.
- `tests/test_phase3_composite.py` — 34 `unittest` cases covering the full approved test
  plan: monotonicity (all 4 active components), active-weight normalization (hand-computed
  match + effective-weights-sum-to-1), missing-AIS behavior (well-defined score, `w_eff_
  ais=0`, AIS absence never lowers a score relative to an identical AIS-absent candidate),
  T29/T34 non-duplication (structural — exactly one `support`-named weight field — and
  numerical — a `support_score` delta enters the composite exactly once), raw-KDE-domination
  synthetic case, persistence-evidence ordering, no-age-correction (structural signature
  check + identical-evidence-different-timestamp behavioral proof), T30 zero-weight/sign
  handling (default zero, no effect on score, still reported, nonzero weight DOES take
  effect, negative weight rejected by validation), all 3 sensitivity configurations (sum to
  1, intended weight shifted in the intended direction, T30/persistence_count still zero),
  `NaN`-safety (the fixed gap, both zero-weight and nonzero-weight cases), candidate-identity
  mismatch rejection, and an end-to-end batch-wiring test against a small synthetic Stage III
  dataset (not the real 280-candidate one, which is the notebook's job).

#### Tests

```text
python -m unittest tests.test_phase3_composite -v
Ran 34 tests in 0.040s
OK

python -m unittest discover -s tests -v   (full suite)
Ran 669 tests in 24.222s
OK
```

#### Manual check (real data) — `notebooks/T36_composite_check.ipynb`

Regenerated all 280 real candidates + 72 support grids (104.9s), computed DEFAULT-config
`source_score` for all of them (73.0s), plus the two sensitivity configurations (147.3s
combined). All cells executed via `jupyter nbconvert --execute`, zero errors, zero `NaN`
scores.

```text
configured weights: {support:0.35, persistence:0.20, convergence:0.10, dispersion:0.15,
                      ais:0.20, initial_bias:0.00, persistence_count:0.00}
effective weights (AIS unavailable, identical for all 280 real candidates):
                     {support:0.4375, persistence:0.25, convergence:0.125, dispersion:0.1875,
                      ais:0.00, initial_bias:0.00, persistence_count:0.00}
sum of effective weights: 1.000000
candidates with AIS available: 0 / 280

source_score distribution (280 candidates, 0 NaN):
  mean=0.5636  std=0.3953  min=0.2006  p10=0.2693  p25=0.2909  median=0.2935
  p75=1.0331  p90=1.1936  p95=1.3148  max=1.4620

Spearman(source_score, raw_support_score), all 280 candidates: 0.7156
mean |rank change| = 46.29, max |rank change| = 153 (out of 280)

Sensitivity (Spearman rank correlation between configurations):
  Spearman(DEFAULT, PERSISTENCE-EMPHASIS)             = 0.9992
  Spearman(DEFAULT, DISPERSION-EMPHASIS)              = 0.9982
  Spearman(PERSISTENCE-EMPHASIS, DISPERSION-EMPHASIS) = 0.9962
  PERSISTENCE-EMPHASIS overlap with DEFAULT top-10: 9/10
  DISPERSION-EMPHASIS overlap with DEFAULT top-10:  10/10
```

**Major flagged finding — top-10 overlap with raw Support is 0/10:**

```text
top-10 by raw T17 support_score:          all 2019-10-14 13:00-22:00 (ages 1-10h)
top-10 by T36 source_score (DEFAULT):     all 2019-10-11 23:00 - 2019-10-12 08:00 (ages 63-72h)
```

Every candidate in both top-10 lists is the SAME single dominant T31 track's own `cand000`
member at a different hour — the DEFAULT composite ranking is a complete inversion of the
raw-Support ranking WITHIN that one track, not a comparison across different candidates.
**Mechanism (verified against each component's own already-approved behavior, not a new
finding requiring further investigation):**

1. `support_evidence` (T29/T34) is already flattened across this track's own members by
   T29's per-timestamp percentile normalization (T29's own approved real-data check found
   this track's normalized values cluster `~1.1-1.5` regardless of hour) — no longer
   strongly favors recent hours the way raw `support_score` does.
2. `persistence_evidence` is IDENTICAL across every member of this one track
   (`temporal_continuity=1.000` for all 72 members, since it's a single unbroken track) — a
   tie, contributes no differentiation among these particular candidates.
3. `dispersion_evidence` is exactly T33's own already-approved, already-flagged
   top-candidate-level finding (`Pearson +0.863` to `+0.951` with age) — rising sharply from
   `~0.96` near `t_obs` to `~2.65` near `T72` for this exact track. With persistence tied and
   support flattened, this is the dominant remaining differentiator, and it favors OLD hours.
4. `convergence_evidence` pulls the opposite direction (candidate-level age correlation
   `-0.593, Pearson`, section 3 of the design spec) but at a smaller effective weight
   (`0.125` vs. dispersion's `0.1875`) and, empirically, a smaller magnitude swing across
   this track's own members — insufficient to counteract (3).

**This is not a bug and not an accidentally-reintroduced age term** — `compute_source_score`
and `compute_source_scores_for_candidates` were structurally confirmed (test suite) to
accept no `age`/`t_obs` parameter anywhere. It is the correct, direct, already-anticipated
consequence of including T33's own approved dispersion evidence (with a nonzero weight, as
specified) for a population where the age-correlated component happens to dominate the
tie-broken remainder. **This finding is reported for review, not resolved or endorsed as
correct/desirable here** — whether a top-10 built almost entirely from the OLD end of the
single dominant track (rather than distinct spatial candidates) is a useful or intended
outcome of this weighting is a judgment call for the user, not something this task decides.

#### Pairwise correlations among active evidence components + `source_score` (this run, 280 candidates)

```text
Pearson:
              support  persistence  convergence  dispersion  initial_bias  source_score
support         1.000        0.305        0.934       0.927         0.979         0.994
persistence     0.305        1.000        0.353       0.278         0.291         0.337
convergence     0.934        0.353        1.000       0.775         0.973         0.910
dispersion      0.927        0.278        0.775       1.000         0.852         0.960
source_score    0.994        0.337        0.910       0.960         0.959         1.000
```

`source_score` correlates very strongly with `support_evidence` (`0.994`) ACROSS THE WHOLE
280-candidate population (dominated by the large minor-vs-dominant split, section 4 of the
design spec) even though, AT THE TOP TIER specifically, the ranking inverts relative to raw
`support_score` (see above) — these are not contradictory: `support_evidence` (T29/T34-
normalized) is a different quantity from raw `support_score`, and population-level
correlation is a different question from top-10 ranking order.

#### Known limitations

- The zero-top-10-overlap finding (above) has not been independently validated against any
  ground truth (none exists for this project) — it demonstrates the composite is NOT simply
  raw Support relabeled, per the task's own test requirement, but does not by itself
  validate that the resulting ranking is more physically correct.
- Weights remain heuristic/configurable, not scientifically validated (restated per every
  prior Phase 3 task).
- `w_ais`'s real effect remains entirely untested — no AIS data exists.
- No claim that `source_score` is calibrated, and **the top-ranked candidate is NOT proven
  to be the actual physical source** — restated explicitly per instruction.
- No Stage III module, and no existing T14-T35 module, was modified.

### T36 review diagnostic: dominant-track member-by-member breakdown (2026-09-24, no production weights changed)

Requested before further review, to determine whether the top-10 inversion (above) reflects
meaningful relative concentration or a T33-specific artificial age reward. `iste/
phase3_composite.py` was NOT modified for this diagnostic; sections F/G below construct
diagnostic-only `CompositeWeights` instances passed into the existing, unmodified
`compute_source_score` — `DEFAULT_WEIGHTS` itself is never changed. 17 new cells appended to
`notebooks/T36_composite_check.ipynb` (one import bug in the first appended cell was caught
and fixed before the final successful execution).

**Dominant track:** `track0000`, 72 members (T1-T72), `continuity=1.000` throughout.

**A/B — within-track Spearman correlation with age (raw evidence == weighted contribution,
confirming a positive constant multiplier never changes rank correlation):**

```text
support_evidence       Spearman=+0.9088
convergence_evidence   Spearman=-0.9925
dispersion_evidence    Spearman=+0.8621
persistence_evidence   constant (continuity=1.0 for all 72 members of one unbroken track)
```

**Correction to the prior turn's explanation:** `support_evidence` is NOT flat within this
track (as previously suggested) — it is itself strongly age-correlated (`+0.91`) within
this specific track, second only to `convergence_evidence`'s strength. `persistence_
evidence` is the only literally flat (zero-variance) component here.

**C/D/E — exact decomposition of the oldest(72h)-vs-newest(1h) composite score gap
(`+0.4290` total):**

```text
T33 (dispersion)   contribution: +0.3174  (74.0%)
T29/T34 (support)  contribution: +0.1369  (31.9%)
T32 (convergence)  contribution: -0.0253  (-5.9%, partial offset)
T31 (persistence)  contribution: +0.0000  (0.0%, tied)
sum = +0.4290 (exact)
```

T33 is the single largest contributor, but NOT the only one pulling toward older ages —
T29/T34's own already-approved residual trend contributes nearly a third of the gap; T32 is
the only component pulling the other way, and only partially.

**F — T33 removed entirely (diagnostic weights, transparently renormalized, production
`DEFAULT_WEIGHTS` untouched):**

```text
effective weights without T33: support=0.538, persistence=0.308, convergence=0.154, rest=0
Spearman(age, score) WITHOUT T33 = +0.8712   (WITH T33 = +0.9032)
oldest rank WITHOUT T33 = 21/72              (WITH T33 = 1/72)
newest rank WITHOUT T33 = 71/72              (WITH T33 = 67/72)
```

**Central finding: removing T33 completely barely changes the overall age-correlated
SHAPE of the within-track ranking (`+0.87` vs. `+0.90` Spearman)** — it mainly determines
whether the single oldest candidate specifically reaches literal rank 1 (drops to 21/72
without T33) rather than driving the age-correlation pattern itself, which is substantially
present even at `w_dispersion=0`.

**G — diagnostic T33-weight sweep (0, 0.05, 0.10, 0.15, 0.20; other weights rescaled
proportionally to preserve their relative ratios; `DEFAULT_WEIGHTS` never modified):**

```text
w_dispersion=0.00: Spearman(age,score)=+0.8712  oldest_rank=21/72  newest_rank=71/72
w_dispersion=0.05: Spearman(age,score)=+0.9077  oldest_rank= 4/72  newest_rank=71/72
w_dispersion=0.10: Spearman(age,score)=+0.9088  oldest_rank= 1/72  newest_rank=68/72
w_dispersion=0.15: Spearman(age,score)=+0.9032  oldest_rank= 1/72  newest_rank=67/72
w_dispersion=0.20: Spearman(age,score)=+0.8950  oldest_rank= 1/72  newest_rank=64/72
```

The overall Spearman correlation stays in a narrow `+0.87` to `+0.91` band across the entire
sweep — not strongly sensitive to T33's specific weight. T33's main effect is pushing the
single oldest candidate to literal rank 1 once its weight reaches `~0.10`, not creating the
age-correlated shape from nothing.

**Physical diagnostics (sample rows, full 72 in the notebook/CSV):**

```text
age  covariance_det   candidate_density  population_median_density  T33_expected  T33_ratio
72h  5.14e13          340                153                        128.2         2.652
36h  8.20e12          547                266                        516.2         1.060
6h   2.01e11          1796               1337                       1905.4        0.943
```

Both `candidate_local_density` and `population_median_local_density` DECLINE with age (the
candidate's own absolute density falls from 1796 near `t_obs` to 340 at T72) — the candidate
does NOT become more concentrated in absolute terms further back. What rises is the RATIO:
the Gaussian-modeled `expected_density` declines even FASTER than the actual candidate
density does, so `t33_dispersion_ratio` climbs. A fully model-free cross-check —
`empirical_ratio = candidate_local_density / population_median_local_density` (no Gaussian
assumption at all) — shows the SAME qualitative pattern and an EVEN STRONGER age correlation
(`Spearman +0.9431` vs. T33's own `+0.8621`), directly consistent with T33's own earlier
closed empirical-null diagnostic (which also found the model-free version stronger, not
weaker, than the Gaussian one).

**Answer to the stated purpose, stated carefully:** the evidence supports "T33 (and, to a
lesser but real extent, T29/T34) reflect a genuine, model-robust pattern in how this
particular dominant candidate's density compares to its own timestamp's population, which
happens to increase toward older timestamps in this real run" rather than "T33 is a Gaussian-
model artifact manufacturing an age reward" (the model-free ratio shows the same, even
stronger, pattern; the anisotropic-vs-isotropic diagnostic already ruled out model-shape
artifacts). It does **not** support "T33 alone creates the observed top-10 inversion" — the
age-correlated ranking shape is substantially present even with T33's weight at exactly
zero, and T29/T34 contributes nearly a third of the specific oldest-vs-newest gap. **This
diagnostic does not conclude that older or newer timestamps are more likely to be the true
source, and does not establish source identity** — it characterizes a measured property of
this real dominant candidate's evolving relationship to its own population, nothing more.

Artifacts: `results/actual_data/T36_dominant_track_diagnostic.png`,
`t36_track_diagnostic.csv` (scratchpad, full 72-row table).

**T36 approved 2026-09-24.** All three review diagnostics (design spec + amendments,
real-data validation, dominant-track member-by-member breakdown) are closed; `iste/
phase3_composite.py` and `DEFAULT_WEIGHTS` stand as implemented, unmodified since approval.

### T37 — Spatial clustering of candidates (PENDING_REVIEW)

### Files added

- `iste/phase3_clustering.py` — `CandidateCluster` dataclass (centroid, `spatial_radius_m`,
  `area_m2` via `scipy.spatial.ConvexHull` with the same `<3`-points/degenerate `None`
  fallback T28 already established, `total_particle_support`, `supported_timestamps`/
  `n_distinct_timestamps`/`earliest_t_s`/`latest_t_s`/`temporal_span_hours`, `mean_
  normalized_kde`/`peak_normalized_kde` — T29/T34's `support_evidence`, the ONLY axis for
  that evidence lineage, per T36's own established rule — `mean_convergence_score`, `mean_
  dispersion_score`, `mean_initial_bias_influence`, `mean_ais_score` (`None` when no member
  has AIS available, never fabricated), `peak_source_score`/`mean_source_score`, and
  `confidence_uncertainty_note` — an explicit statement that T38 is not yet implemented,
  not a fabricated placeholder number) + `cluster_candidates(candidates, source_scores, *,
  eps_m=2000.0, min_samples=1)`. Uses `sklearn.cluster.DBSCAN` (already an available
  dependency; confirmed installed, `1.9.0`) with a precomputed pairwise distance matrix built
  from the SAME local-tangent-plane projection T15/T17/T30/T31/T36 already use
  (`iste.kdtree_neighborhood.project_to_local_metres`, reused unchanged). `eps_m` defaults to
  T31's own `spatial_matching_radius_m` (2 km) — reused, not re-derived. `min_samples=1` is a
  deliberate choice so DBSCAN never labels a candidate as noise, consistent with Phase 2/3's
  standing "never silently drop a candidate" policy; documented in the module docstring as
  mathematically equivalent to single-linkage hierarchical clustering at the same threshold
  (both reduce to the same connected-components-within-`eps` partition) — DBSCAN was picked
  for its direct scikit-learn API, not because it produces a different result from the
  hierarchical alternative. Output is sorted by `peak_source_score` descending (ranked
  clusters, per the master spec's "rank CLUSTERS rather than individual pixels").
- `tests/test_phase3_clustering.py` — 21 `unittest` cases: the two required checks (near-
  duplicate candidates — including a transitively-chained 5-point sequence — collapse into
  one cluster; well-separated candidates remain distinct, confirmed via unique cluster
  membership), no-candidate-dropped (an isolated candidate forms its own singleton, never
  excluded as noise), geometry (hand-computed centroid, `area_m2=None` for `<3` members and
  for exactly-collinear members, defined and positive for a real triangle), evidence
  aggregation (hand-computed `total_particle_support` sum, `normalized_kde` mean/peak,
  `source_score` mean/peak, `AIS` mean-over-available-only and `None`-when-none-available,
  temporal span/distinct-timestamp count), the confidence-placeholder's explicit deferred-
  to-T38 wording, ranked-output ordering, and validation (mismatched lengths/candidate-id
  order, non-positive `eps_m`, empty input, single-candidate edge case).
- `notebooks/T37_clustering_check.ipynb` — manual-check artifact (see below).

### Tests

```text
python -m unittest tests.test_phase3_clustering -v
Ran 21 tests in 0.035s
OK

python -m unittest discover -s tests -v   (full suite)
Ran 690 tests in 22.630s
OK
```

### Manual check (real data)

`notebooks/T37_clustering_check.ipynb` regenerates all 280 real candidates (79.6s), computes
T36 `DEFAULT_WEIGHTS` composite scores (46.0s, reused unchanged), then clusters them.
Executed via `jupyter nbconvert --execute`, zero errors.

```text
DEFAULT_EPS_M = 2000.0 m
280 raw candidates -> 10 distinct spatial clusters
every candidate accounted for exactly once: confirmed (280 == 280)

cluster size distribution: min=1, max=147, mean=28.00, median=2.5
n singleton clusters (1 member): 4
n clusters with >=5 members: 4
```

Top cluster (`cluster0000`): **147 members** (52.5% of all real candidates), `spatial_
radius_m=24439.1` (~24.4 km!), `n_distinct_timestamps=72`, `temporal_span_hours=71.0`,
`peak_source_score=1.4620`, `total_particle_support=56840.0`.

**Real-data finding requiring explicit awareness (not silently accepted as unambiguously
correct):** `cluster0000` is essentially the ENTIRE 71-hour dominant T31 track (whose
consecutive hourly steps were separately measured, at T31 approval, to be at most `1362.6 m`
apart — always well under `eps_m=2000`), transitively chained by DBSCAN's `min_samples=1`
single-linkage-equivalent behavior into one nominal "cluster" that in reality spans ~24 km —
a multi-hour drift corridor, not a compact source region in the intuitive sense the word
"cluster" suggests. This is not a bug: the task's own literal check (near-duplicates
collapse; well-separated candidates remain distinct) is satisfied exactly as specified, and
this is the well-known, well-documented "chaining" behavior of single-linkage-style
clustering applied to a trajectory of closely-spaced sequential points. It is reported here
prominently, per this session's standing discipline, rather than presented as an
unqualified success.

### Known limitations

- `eps_m`/`min_samples` are documented, reused (not re-derived) PROTOTYPE/CONFIGURATION
  choices — fully overridable; a smaller `eps_m` or a non-chaining algorithm (e.g. a
  different linkage criterion, or DBSCAN with `min_samples>1`) would very likely split the
  dominant track's trajectory into multiple sub-regions instead of one 24 km chain — not
  attempted here, since changing the algorithm/threshold to force a different real-data
  outcome would be exactly the "tune to a desired result" this whole session has avoided.
- Cluster-level `peak_source_score`/`mean_source_score` inherit every caveat already
  established for T36's per-candidate `source_score` — an evidence-ranking score, not a
  probability, not proof of source identity.
- `confidence_uncertainty_note` explicitly defers to T38 — no confidence/uncertainty number
  is computed by this task.
- No Stage III module, and no existing T14-T36 module, was modified.

### T38 — Uncertainty / bootstrap stability analysis (DONE, approved 2026-09-24)

### Files added

- `iste/phase3_uncertainty.py` -- `compute_total_available_particles`, `compute_spatial_
  uncertainty_radius_km`, `compute_temporal_uncertainty_hours`, `BootstrapConfig`/
  `BootstrapResult`/`run_bootstrap_stability`, `ClusterUncertainty`/`compute_cluster_
  uncertainty_table`, `AgeUncertaintyDiagnostic`/`diagnose_uncertainty_vs_age`. Reads T17
  `Candidate`/T36 `SourceScore`/T37 `CandidateCluster`/Stage III `EnsembleRunRecords`
  read-only; does not modify or re-derive any T14-T37 value.
- `tests/test_phase3_uncertainty.py` -- 21 `unittest` cases.
- `notebooks/T38_uncertainty_check.ipynb` -- manual-check artifact (see below).

### Design summary (see the module's own docstring for full derivations)

- **Particle identity:** `(particle_id, forcing_member_id)` -- the SAME grouping key T14's
  `query_particle_slice` already uses (one physical particle under one forcing-ensemble
  member is one distinct stochastic realization). Repeated hourly appearances of the same
  pair are counted once, never once per hour.
- **Part A, `particle_support_fraction`:** numerator = size of the UNION, over a cluster's
  member candidates, of the distinct particle-identity pairs in each candidate's own
  already-computed T16 `peak_support_features.neighborhood` (no re-query). Denominator =
  distinct particle-identity pairs across the WHOLE Stage III `records` batch (the fixed,
  timestamp-independent particle-member pool of the entire run — chosen over a per-
  timestamp count specifically so the fraction stays comparable across clusters whose
  members sit at different timestamps).
- **Part B, `persistence_fraction`:** `cluster.n_distinct_timestamps / total_timestamps`,
  `total_timestamps` defaulting to 73 (T27's own `generate_analysis_grid(window_hours=72,
  interval_hours=1)` length, T0..T72 inclusive — verified directly from `iste/
  analysis_grid.py`, not assumed). No minimum-duration threshold, no age-based reward.
- **Part C, `uncertainty_radius_km`:** the 90th percentile (configurable,
  `DEFAULT_SPATIAL_PERCENTILE`) of the SAME pooled distinct-particle set's distance from
  the cluster's own T37 centroid, in km. Explicitly documented as NOT T37's
  `spatial_radius_m` renamed: that field is the extent of a handful of discrete candidate
  PEAK positions; `uncertainty_radius_km` is the spread of the (typically far larger)
  underlying PARTICLE evidence. A percentile-of-actual-positions measure was chosen over a
  covariance/ellipse form specifically to avoid stacking another Gaussian/isotropy
  assumption on top of T30/T33's own, and because it reuses Part A's pooled-particle set
  exactly (no second, differently-defined particle pool).
- **Part D, `temporal_uncertainty_hours`:** T36-`source_score`-weighted standard deviation
  (hours) of the cluster's own member candidate timestamps (equal-weight fallback if any
  weight is negative/non-finite/sums to <=0, documented). Explicitly NOT T37's
  `temporal_span_hours` (a raw max-min extent) -- this is an evidence-weighted spread.
  `0.0` for a single-member cluster, documented as "no internal spread to measure," not a
  claim of exact certainty.
- **Part E, bootstrap -- practicality assessed BEFORE implementing, per instruction:** a
  full bootstrap re-running Stage III or rebuilding the whole T16 support grid per
  replicate was assessed and REJECTED as impractical: regenerating the real dataset's 280
  candidates measured ~62-80s per pass across this session's runs, and T14's
  `query_particle_slice` loops in pure Python over all 240,024 real records per call;
  repeating either per bootstrap replicate (even 100-200 replicates) would cost tens of
  minutes to hours, and Part E's own instructions separately prohibit rebuilding the whole
  simulation per replicate. **What WAS implemented instead (bootstrap WAS actually run, not
  skipped):** each candidate's own already-computed T16 neighborhood (already-found nearby
  particles, never re-queried) is resampled WITH REPLACEMENT independently per candidate
  per replicate -- `D_boot` = count of DISTINCT particle-identity pairs in the resample
  (repeats not double-counted, mirroring the identity rule used everywhere else in this
  module), `W_boot` = sum of `exp(log_weight)` over all resampled draws (a SUM statistic,
  counted with multiplicity, standard for resampling a weighted sum), `V_boot` = distinct
  `observation_pixel_id` count among the resample -- combined via the candidate's own
  already-used `linear_support_function`/`SupportFeatureConfig` (T16, unchanged, no new
  combination formula). The T29/T34 normalization reference value is NOT re-derived per
  replicate (that would require the same expensive whole-grid rebuild being avoided);
  instead the already-computed `screening_score` is algebraically SCALED by the bootstrap/
  original raw-support ratio, which is exact given T34's own `screening_score = support /
  reference_value` formula and a per-timestamp-constant `reference_value`. Every OTHER
  evidence component (T31/T32/T33/T35) is held at its original value per replicate --
  **this scopes the bootstrap to testing T29/T34 support-evidence stability specifically,
  not the full six-component pipeline**, stated explicitly in the module docstring and the
  notebook, never hidden. Recombination reuses T35's own unmodified
  `renormalize_weighted_mean` with the same configured weights -- no new formula.
  Because candidate POSITIONS never change (only their evidence VALUES are perturbed),
  cluster membership is identical to the original T37 assignment in every replicate by
  construction -- no spatial re-matching rule was needed, and this is stated explicitly
  rather than silently assumed away.
- **`BootstrapConfig`:** `n_replicates=200`, `random_seed=42`, `top_k=10` -- explicit,
  configurable, PROTOTYPE/CONFIGURATION, not scientifically validated.
  `candidate_stability_percent` = 100 * (replicates in which the cluster owned >=1 of that
  replicate's top-`top_k` candidates) / `n_replicates`. Reported as `None` ("unavailable"),
  never `0.0`, whenever bootstrap was not run.
- **Language discipline (per explicit instruction):** every user-facing string uses
  "candidate stability" / "ranking stability" / "cluster recurrence" / "uncertainty" --
  never "probability of source identity." `STABILITY_DISCLAIMER` and T36's own
  `EVIDENCE_RANKING_DISCLAIMER` are both carried onto every `ClusterUncertainty` row.

### Tests

```text
python -m unittest tests.test_phase3_uncertainty -v
Ran 21 tests in 0.056s
OK

python -m unittest discover -s tests   (full suite)
Ran 711 tests in 14.289s
OK
```

Covers: hand-computed `particle_support_fraction` (union-of-neighborhoods numerator, whole-
run denominator, shared-particle de-duplication, denominator inflated by non-candidate
particles), `persistence_fraction` (default-73 denominator, scaling with distinct
timestamps), spatial uncertainty (grows with actual particle spread, `NaN` for an empty
pool, percentile validation), temporal uncertainty (`0.0` for one member, hand-computed
weighted std for two equally-weighted members), bootstrap (reproducible under a fixed seed,
stays within `[0, 100]`, a dominant high-support candidate is MORE stable than a
deliberately weak one, no NaN/Inf), graceful degradation (`None` not `0.0` when no
bootstrap; `"unavailable"` AIS status), no mutation of T36 `source_score`/T37 cluster
membership, no NaN/Inf on valid input, `total_timestamps` defaults to 73, and an explicit
"age is not hard-coded into the formula" test (two candidates at very different ages but
IDENTICAL particle spread get IDENTICAL `uncertainty_radius_km`).

### Manual check (real data)

`notebooks/T38_uncertainty_check.ipynb` reproduces T37's own real-data setup (Sections 1-3,
unchanged: 280 candidates regenerated in 61.7s, T36 `DEFAULT_WEIGHTS` scores computed in
39.9s, clustered into 10 T37 clusters), then runs T38 on top. Executed via `jupyter
nbconvert --execute`, zero errors.

```text
total_available_particles (distinct (particle_id, forcing_member_id) pairs) = 3288
bootstrap performed: True, n_replicates_run=200, measured runtime: 2.88s
T36 source_score values unchanged: confirmed
T37 cluster IDs/membership unchanged: confirmed

cluster_id   spatial_radius_km  uncertainty_radius_km  temporal_uncertainty_hours  supporting_particle_count  particle_support_fraction  supported_timestamp_count  persistence_fraction  candidate_stability_percent
cluster0000             24.439                 25.157                       20.76                       3015                    0.91697                         72                0.9863                        100.0
cluster0002             10.034                  8.423                        9.01                        377                    0.11466                         34                0.4658                          0.0
cluster0006              6.301                  2.625                        4.15                         12                    0.00365                         15                0.2055                          0.0
cluster0004              0.000                  0.843                        0.00                          3                    0.00091                          1                0.0137                          0.0
cluster0005              0.000                  0.957                        0.00                          3                    0.00091                          1                0.0137                          0.0
cluster0009              0.361                  0.971                        0.50                          1                    0.00030                          2                0.0274                          0.0
cluster0008              0.000                  0.954                        0.00                          1                    0.00030                          1                0.0137                          0.0
cluster0003              1.774                  1.258                        3.57                          2                    0.00061                         11                0.1507                          0.0
cluster0007              0.000                  0.923                        0.00                          1                    0.00030                          1                0.0137                          0.0
cluster0001              1.071                  1.201                        0.82                          1                    0.00030                          3                0.0411                          0.0

number of T37 clusters: 10; all with finite uncertainty metrics; 0 NaN, 0 Inf
all T37 clusters accounted for exactly once: confirmed (280 == 280)
all 73 T0..T72 analysis timestamps confirmed present
```

Every AIS status is `"unavailable"` (no AIS data exists in this project, per T35).

**Part F age/spread diagnostic (10 real clusters -- a small sample, reported honestly, not
tuned to a particular trend):**

```text
pearson(age_hours, uncertainty_radius_km)   = 0.4141   spearman = 0.4545
pearson(age_hours, temporal_uncertainty_hours) = 0.4397   spearman = 0.4788
pearson(mean_particle_spread_km, uncertainty_radius_km)  = 0.9985   spearman = 0.9636
```

The age correlations are moderate (not strong, not zero, not contradictory) -- reported as-
is. The near-1.0 spread-vs-uncertainty-radius correlation is EXPECTED, not an independent
finding: `uncertainty_radius_km` is itself computed as a percentile of that same pooled-
particle spread distribution (Part C), so this is a consistency/sanity check on the
implementation, not a second, independent validation of the uncertainty measure.

**Dominant cluster (`cluster0000`, 147 members, the same T37 DBSCAN-chained 71-hour
corridor flagged at T37 approval):** `uncertainty_radius_km=25.16` km -- correctly larger
than, and consistent with, T37's own `spatial_radius_km=24.44` (both reflect the real
corridor's real width; `uncertainty_radius_km` was NOT artificially shrunk to look more
"source-like"). `candidate_stability_percent=100.0` -- read per the notebook's own explicit
caveat: this means SOME point along the 147-member corridor consistently lands in the
bootstrap top-10 across resamples, not that the whole 24 km corridor is one compact,
confirmed source.

### Known limitations

- `DEFAULT_TOTAL_TIMESTAMPS=73` and `DEFAULT_SPATIAL_PERCENTILE=90.0` are explicit,
  documented, fully overridable PROTOTYPE/CONFIGURATION choices, not scientifically
  validated constants.
- The bootstrap's scope is deliberately narrower than the full six-component T36 pipeline
  (T29/T34 support evidence only is resampled; T31/T32/T33/T35 held fixed per replicate) --
  a stated, not hidden, limitation driven by the measured impracticality of a full-pipeline
  bootstrap on this dataset's runtime.
- The T29/T34 normalization reference value is scaled, not re-derived, per bootstrap
  replicate (see Design summary above).
- Part F's correlations are computed over only 10 real clusters -- too small a sample to
  treat as a strong statistical result either way.
- `candidate_stability_percent` for a DBSCAN-chained corridor cluster (like `cluster0000`)
  reflects recurrence of ANY member along the corridor, not a single compact point --
  stated explicitly in both the notebook and this file.
- No T14-T37 module was modified; T36 `source_score` values and T37 cluster IDs/membership
  are confirmed byte-for-byte unchanged by the real-data notebook (Section 6).

### T39 — Visualization suite + old-vs-new ranking comparison (PENDING_REVIEW)

### Files added

- `notebooks/T39_visualization_comparison.ipynb` -- the only new artifact. No new `.py`
  module: this task is pure orchestration/plotting over already-approved T14-T38 output
  (candidate/score/cluster/uncertainty recomputation reuses `iste.phase3_composite.
  compute_source_scores_for_candidates`, `iste.phase3_clustering.cluster_candidates`,
  `iste.phase3_uncertainty.{run_bootstrap_stability,compute_cluster_uncertainty_table}`,
  and `iste.phase3_diagnostics.compute_diagnostics_table` unchanged), per CLAUDE.md's own
  ".ipynb = experiments, plots, debugging, manual validation" rule.
- `results/actual_data/T39_old_vs_new_ranking.csv`, `T39_stage13_diagnostics_table.csv`,
  `T39_selected_snapshots.png`, `T39_persistence_heatmap.png`, `T39_final_map.png`.

### Design summary

- **Old method (`T17-raw-support baseline`):** exactly `main_actual_data_pipeline.ipynb`'s
  own documented Section 10 selection rule -- rank all candidates by raw T17
  `support_score` (`Support(x,y,t)=f(D,W,V)`) descending, no cross-timestamp correction.
- **New method:** rank the same candidates by T36 `source_score`, unmodified.
- **Comparison table:** the union of each method's own top-10 (20 unique candidates on
  this real run since the two sets did not overlap at all), with `support_duration_hours`
  from T31's own `PersistenceTrack.persistence_duration_hours`, `distance_from_observed_km`
  from the candidate peak to the observed slick's own mean parcel position (via the
  project's standing `project_to_local_metres` convention), `particle_support` = T16's
  `density_score`, `AIS_support` = T35's `ais_evidence` (`"unavailable"`, never fabricated),
  `screening_support` = T29/T34's `support_evidence`.
- **Selected-snapshot panels:** extends `main_actual_data_pipeline.ipynb`'s existing
  4-panel hotspot-heatmap pattern to the 8 explicitly requested snapshots (T0, T0-6h,
  T0-12h, T0-24h, T0-36h, T0-48h, T0-60h, T0-72h), red star = new-method top-10 candidate.
- **Persistence heatmap:** top-10 T37 clusters (rows) x 73 hourly slots (columns), color =
  each present member candidate's own T36 `source_score`; blank cells = no member at that
  hour (never fabricated as 0).
- **Final map:** observed slick (548 real parcels), a convex-hull outline of ALL pooled
  valid-particle positions across every T0-T72 timestamp (236,736 pooled positions, purely
  descriptive spatial extent), top-10 T37 cluster centroids, each cluster's own T38
  `uncertainty_radius_km` (P90) drawn as a circle, and an explicit "AIS tracks:
  unavailable" label (never a fabricated/empty-looking AIS layer presented as data).
- **Stage 13/T28 diagnostic table:** `iste.phase3_diagnostics.compute_diagnostics_table`
  called directly over the full 73-point analysis grid (a necessary second sweep, since
  T28's function does not accept a precomputed support map -- an accepted, documented,
  one-time cost).

### Manual check (real data)

Executed via `jupyter nbconvert --execute`, zero errors.

```text
observed slick: ow-0008.jpg, 548 parcels, reference point (mean) = (34.07660, 35.26621)
loaded 240024 real rows; full analysis grid has 73 timestamps (T0..T72)
regenerated 280 real candidates in 70.5s; pooled 236736 valid particle positions for the envelope
computed T36 scores + T37 clusters (10 clusters) in 39.2s
computed T38 uncertainty for 10 clusters (bootstrap runtime 2.92s)
computed Stage 13/T28 diagnostics for 73 timestamps in 59.1s
```

**Old-vs-new comparison (union of top-10s, 20 unique candidates, 0 overlap):**

```text
old-top-10 candidates NOT in new-top-10: 10
new-top-10 candidates NOT in old-top-10: 10
candidates in BOTH top-10 sets: 0
old_rank=1 candidate: 2019-10-14T22:00:00_cand000 (t_s=T0-2h, support_score=4675.4308)
new_rank=1 candidate: 2019-10-11T23:00:00_cand000 (t_s=T0-71h, source_score=1.4620)
```

The full new-top-10 (all from the dominant 71-hour T37/T38 track/cluster already flagged
at those approvals) has `new_score` 1.383-1.462; the full old-top-10 (all from T0-1h
through T0-10h) has `old_score` 855-997. `distance_from_observed_km` for the new-top-10
ranges ~32-37 km (far from the observed slick, consistent with a 65-71 hour-old backward
position); for the old-top-10 (not shown in full here, see `T39_old_vs_new_ranking.csv`)
the closest-to-observed candidates (T0-2h through T0-13h) score HIGHEST under the OLD
method purely from raw KDE concentration, and correspondingly LOW under the new method
(`new_rank` 45-71) once T29's per-timestamp normalization removes that concentration bias
-- e.g. the `old_rank=1` candidate (`support_score=4675.4`) drops to `new_rank=71`
(`source_score=1.0331`).

**Validation checklist:** all 6 checks passed (zero-error execution, all 73 timestamps
present, comparison table >=10 candidates, genuine re-ranking confirmed, T37 cluster count
unchanged at 10, AIS reported `"unavailable"` never fabricated).

**Written explanation (required deliverable) -- why early timestamps no longer dominate
purely from KDE concentration:** T28's own real-data diagnostic already measured that the
raw T16 support/KDE grid's peak is far larger near `t_obs` purely because the particle
cloud is still tightly clustered shortly after release, not because early timestamps are
better source candidates. Left uncorrected (`old_score`/`old_rank`), this mechanically
favors whichever timestamp has the tightest cloud, which is structurally always near `T0`.
T29 fixes this by dividing each timestamp's own support grid by that SAME timestamp's own
95th-percentile-of-positive-cells reference value -- comparing each candidate to its own
timestamp's concentration level, not to T0's raw scale. T31-T35 then add temporal
persistence, convergence, dispersion-relative concentration, and (when available) AIS
evidence, none of which contain an age term (verified at each task's own approval). T36's
`source_score` is the renormalized weighted combination of exactly these corrected,
age-free evidence streams -- and the REAL-DATA RESULT above (0/10 top-10 overlap, not a
constant rescaling) demonstrates this is a genuine re-ranking under corrected evidence,
not simply the old order divided by a constant.

**This describes what the corrected evidence produces on this real dataset, not a claim
that the new top-ranked candidate is the proven physical source** -- `source_score` remains
an evidence-ranking score (T36's own `EVIDENCE_RANKING_DISCLAIMER`), and this is the exact
same dominant age-correlated track T36's own approved dominant-track diagnostic already
found to be a real, multi-component evidence pattern in this dataset, not a T29/T33-
specific artifact -- reported honestly here again, not tuned away.

### Known limitations

- Comparison table shows the union of each method's own top-10 (20 candidates here), not
  all 280 -- the full per-candidate ranking is saved to `T39_old_vs_new_ranking.csv`.
- Persistence heatmap covers only the top-10 T37 clusters (the full set on this dataset).
- Particle envelope is a purely descriptive spatial extent (convex hull of all pooled
  valid particle positions across the whole window), not a claim about source location.
- `distance_from_observed_km`'s reference point (observed slick's mean parcel position) is
  an explicit, documented choice, not a new geodesic formula.
- The Stage 13/T28 diagnostic table required a second full 73-timestamp sweep (accepted,
  one-time cost, not a claim of the cheapest possible implementation).
- No T14-T38 module was modified.

### EVENT/DATA CORRECTION — June 2019 full pipeline rerun (2026-09-24)

`data/ocean.nc`/`data/wind.nc` were replaced (by the project owner) with real June 2019
ocean-current/wind data, correcting the event/data mismatch flagged at the original run
(the October-anchored run used `data/ocean.nc`'s own last checkpoint as `t_obs` because,
at that time, no ocean data existed for the true June 2019 observation window — see
`main_actual_data_pipeline.ipynb`'s own Section 2). This entry documents a full pipeline
rerun for the corrected event, NOT a redesign of any algorithm/equation/weight/clustering
parameter — every T05/T06/T09/T29-T39 formula is reused byte-for-byte unchanged.

### Files added

- `notebooks/T_june2019_stage3_run.ipynb` — Stage III backward transport for the June 2019
  event (T07 adapt, crop+load the new `ocean.nc`, `ForcingEnsemble`/`K`/`run_ensemble`,
  save particle history). New file layout differences from the new `ocean.nc` (confirmed
  by direct inspection, not assumed): dims renamed `latitude`/`longitude` → `lat`/`lon`;
  time axis `2019-06-07T00:30` .. `2019-06-14T23:30` (192 hourly steps, 30-minute-offset
  "hourly-mean-centered-at-time" convention, unlike the October file's on-the-hour steps);
  grid `380×1016` covering nearly the whole Mediterranean (`lat 30.19..45.98`,
  `lon -6.0..36.29`), vs. the October file's regional `149×216` extent.
- `notebooks/T_june2019_phase3_pipeline.ipynb` — regenerates T17 candidates, T36 composite
  scores, T37 clusters, T38 uncertainty/bootstrap, and a T39-style old-vs-new comparison +
  visualization, all against the NEW June particle history — reusing
  `iste.phase3_composite.compute_source_scores_for_candidates`,
  `iste.phase3_clustering.cluster_candidates`, `iste.phase3_uncertainty.{run_bootstrap_
  stability,compute_cluster_uncertainty_table}` unchanged (identical `DEFAULT_WEIGHTS`,
  `DEFAULT_EPS_M`, bootstrap config to the October runs).
- `result/` (new top-level output directory, separate from `results/actual_data/`):
  `ocean_june2019_region.nc` (a REGIONAL CROP of the new `ocean.nc` — see below),
  `stage3_particle_history_june2019.npz`, `stage3_history_summary_june2019.txt`,
  `june2019_cluster_table.csv`, `june2019_old_vs_new_ranking.csv`,
  `june2019_run_summary.json`, `june2019_selected_snapshots.png`, `june2019_final_map.png`.

### Regional crop (data-practicality step, not an algorithm change)

The new `ocean.nc`'s `380×1016` grid spans nearly the whole Mediterranean — far larger
than any region relevant to this one spill's 72h drift, and loading it in full (~2.4 GB as
float64×2 vars) was a genuine memory risk on this machine (as little as ~490 MB free RAM
was observed mid-session). The Stage III notebook crops `ocean.nc` to the EXACT SAME
regional bounding box the ORIGINAL October `ocean.nc` already used (`lat 30.15..36.40`,
`lon 27.05..36.15` — reused, not newly invented, and already known to comfortably contain
this spill's location plus 72h of drift), writes it to `result/ocean_june2019_region.nc`,
then calls T01's own unmodified `load_field_dataset` on the cropped file — no loader code
or physical equation was touched. The cropped grid is `150×218`, almost identical in size
to the original October `149×216` grid.

### Manual check (real data)

**Stage III (`T_june2019_stage3_run.ipynb`):**

```text
T0  (t_obs) = 2019-06-14T20:35:58
T72         = 2019-06-11T20:35:58
ocean.nc covers the full [T72, T0] window: True (margin before T72: 116.10h; margin after T0: 2.90h)
T07 adapted 548 real particles
Running Stage III backward transport: 548 particles x 6 members x 72 steps (3.0 days) ...
done in 763.1s (12.7 min) -> 240024 (particle, member, checkpoint) rows
n_timestamps (checkpoints): 73; valid rows per checkpoint: 3288/3288 at every single one
  of the 73 checkpoints (100% valid throughout the whole 72h window)
```

All 7 Stage III validation checks passed: 73 distinct checkpoints, `T0`/`T72` exact,
1-hour spacing throughout, zero October timestamps in the new history, record count
matches `548×6×73`, `DT_SECONDS`/`N_STEPS`/seeds unchanged from the original run.

**Phase 3 (`T_june2019_phase3_pipeline.ipynb`):**

```text
loaded 240024 June-2019 real rows; full analysis grid has 73 timestamps
regenerated 271 June-2019 real candidates in 80.4s; pooled 236736 valid particle positions
computed T36 scores + T37 clusters (4 clusters) in 49.5s
computed T38 uncertainty for 4 clusters (bootstrap runtime 4.09s)

cluster_id  n_members  spatial_radius_km  uncertainty_radius_km  n_distinct_timestamps  candidate_stability_percent
cluster0000        267             29.558                 30.073                     72                        100.0
cluster0003          1              0.000                  0.919                      1                          0.0
cluster0002          2              0.866                  0.677                      2                          0.0
cluster0001          1              0.000                  0.981                      1                          0.0
```

All 6 Phase 3 validation checks passed (73 timestamps, exact `T0`/`T72`, zero October
timestamps, every candidate accounted for exactly once, AIS reported `"unavailable"`).

**Old-vs-new comparison (June event):** 0/10 overlap between the raw-`support_score`
top-10 and the T36 `source_score` top-10 — the SAME qualitative pattern the October run
showed (T29's per-timestamp normalization genuinely re-ranks candidates, not merely
rescales them), now confirmed independently on a physically-correct June 2019 currents
run: old `rank=1` (`2019-06-14T19:35:58`, `support_score=4735.7`, `T0-1h`) drops to
`new_rank=72`; new `rank=1` (`2019-06-12T04:35:58`, `source_score=1.1931`, `T0-64h`) is
part of the same kind of dominant multi-hour cluster (`cluster0000`, 267/271 candidates)
already seen structurally in the October run.

### Known limitations / findings from this rerun

- `data/wind.nc` is NOT read by Stage III on this run either — per the pre-existing,
  unmodified T01.5 decision (`iste/forcing_ensemble.py`'s own docstring: "this module
  never reads `data/wind.nc`"), only `ocean.nc`'s `uo`/`vo` drive `U` and the stochastic
  forcing ensemble. The new `wind.nc` was still verified to cover the correct June 2019
  window (for future use), but its replacement had no computational effect on this rerun.
- The new `ocean.nc`'s full (uncropped) grid is ~90% native NaN (`66,727,992` of
  `74,127,360` cells in `uo`) — the product's real data footprint is far smaller than its
  nominal `380×1016` bounding rectangle. The CROPPED region actually used for Stage III
  had 100% valid particles at all 73 checkpoints — no NaN/invalid-domain issue was
  encountered by any real particle trajectory in this run.
- `ocean.nc`'s new time axis is offset 30 minutes from `wind.nc`'s (hourly-mean-centered-
  at-time vs. on-the-hour) — cosmetic only, since `wind.nc` is unused; documented in
  `tests/test_environment.py`'s updated `test_time_covers_same_window_as_ocean_file`.
- T00-T13/T17-T26's own historical manual-check notebooks (run against the October data at
  the time) are left untouched — only what T27-T39 actually needs was rerun, per the
  task's own explicit scope ("reuse the existing T27-T39 implementation exactly").
- `results/actual_data/` (the October outputs, including `stage3_particle_history.npz`)
  was confirmed byte-identical/untouched (file mtime unchanged) after this rerun.

### Test suite updates (required — the real `data/ocean.nc`/`data/wind.nc` files changed)

5 test files directly load the real `data/ocean.nc`/`data/wind.nc` files by path
(`test_derivatives.py`, `test_ensemble_run.py`, `test_environment.py`,
`test_interpolation.py`, `test_stage3_validation.py`) and had hardcoded facts about the
OLD October file (dim names `latitude`/`longitude`, shape `(336,149,216)`, time range,
`missing_count`). These are DATA-FACT fixtures re-verified directly against the new real
files (not assumptions, not algorithm changes) — same treatment as the notebook's own
`OCEAN_PATH` dim-name update:
  - `lat_dim`/`lon_dim` for `ocean.nc` calls: `"latitude"/"longitude"` → `"lat"/"lon"`
    (`wind.nc` calls are unchanged — it still genuinely uses `"latitude"/"longitude"`).
  - `test_environment.py`'s exact shape/time-range/`missing_count` values updated to the
    new file's own re-verified numbers (`(192,380,1016)`; `2019-06-07T00:30`..
    `2019-06-14T23:30`; `missing_count("uo")=66,727,992`, `wind` shape `(192,25,36)`).
  - `test_time_matches_ocean_file` renamed `test_time_covers_same_window_as_ocean_file`
    and rewritten to check window/resolution match plus the new, real 30-minute offset,
    instead of exact equality (which no longer holds for the new files).
  - All other real-data-dependent assertions (hand-computed derivative/interpolation spot
    checks at dynamically-located or previously-verified-still-valid points) needed only
    the dim-name fix; verified individually against the new files.
- Full suite: **711/711 passed** (`python -m unittest discover -s tests`, 53.1s).

## Token/complexity rule

Keep implementation compact. One task per Claude Code turn. Stage III is signed off; Stage
IV is in progress (T14). Do not build Stage V-VII, AIS, frontend, services, databases, or
unrelated architecture until the relevant stage is reached and signed off.
