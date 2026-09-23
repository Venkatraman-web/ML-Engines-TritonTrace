# SIH ISTE — Long-Term Memory

> **2026-09-22 — superseded methodology notice.** This file previously described a simpler
> "Reverse Lagrangian Particle Tracking (RLPT)" engine (deterministic/stochastic reverse
> trajectories with no particle weighting, no forcing ensemble, and a grid-based KDE used
> directly as the candidate-source estimate). That design conflicted with `README.md` (the
> declared scientific source of truth) and `ISTE_Claude_Architecture_v1.0.txt`, both of which
> explicitly reject plain deterministic/stochastic backtracking as the inference engine, citing
> ill-posed backward diffusion, flow-compression artifacts, and the need for clean-water
> evidence and forcing-structure uncertainty. The RLPT-specific sections have been removed. Only
> durable ISTE scientific decisions and independently verified dataset facts are retained below.

## Goal

Build the ISTE (Inverse Source Term Estimation) system for one Sentinel-1 oil-slick observation,
producing the posterior:

```text
p(x_s, y_s, t_s, V | y)
```

where `x_s, y_s` are source location, `t_s` is source time, and `V` is source magnitude/volume,
given the observed satellite pixel data `y`. The final output is a time-indexed set of posterior
modes and uncertainty regions, not a single deterministic source point.

The first implementation milestone is **Stage III — Weighted Backward Adjoint** only
(`README.md`). Stages IV–VII (surrogate likelihood, tempered SMC, delayed-acceptance MCMC,
posterior/AIS-ready output) are later, separately signed-off milestones.

## Terminology

- **Observation pixel:** a rasterized cell of the satellite image, `y_j = 1` (oil) or `y_j = 0`
  (clean); Stage III particles are released per pixel (oil-positive and a clean-water buffer) and
  must retain `observation_pixel_id`.
- **Particle:** a Stage III Monte Carlo sample of the backward-adjoint SDE; it carries a dynamic
  weight, not just a position.
- **Forcing ensemble member:** one internally-coherent perturbed realization of the wind/current
  forcing field (index `e`), representing structural forcing uncertainty — distinct from
  per-particle Brownian diffusion.
- **RNG seed:** reproducibility control only. Never use "seed" ambiguously.

**Architectural decision (T09, 2026-09-22): forcing members are injected into T05/T06 via
a lazy `GriddedFieldSource`-compatible view, never by modifying T05/T06 or materializing a
full perturbed grid.** `iste/ensemble_run.py`'s `_PerturbedComponentArray`/
`make_perturbed_source` wrap `iste.forcing_ensemble.ForcingEnsemble` so that indexing
`source.data["uo"][i]` computes `base[i] + delta_u_e(x, t_i)` on demand, for only the one
requested timestep, cached per timestep actually visited — this works because T05/T02
only ever access an `ocean_source` through single-timestep indexing
(`source.data[var][i]`) and `interpolate_field`/`divergence_2d`, never anything requiring
`source` to literally be a `GriddedFieldSource` instance in a type-checked sense. One such
view is built once per forcing member (not per particle) and reused for every particle
assigned to it, which is what makes "every particle under member `e` sees the identical
perturbation" a structural guarantee rather than a bolted-on check. Any later task that
needs a per-member environmental field (T10 diagnostics, T11 storage, a full-scale T12+
run) should reuse this same lazy-view pattern rather than precomputing/storing full
perturbed grids per member, which is exactly the memory blow-up T08's design was meant to
avoid. Diagnostics in T09 are computed at each `(particle, forcing_member)` trajectory's
**final** recorded checkpoint (not every checkpoint) — this is the established convention
future diagnostics work (T10) should follow or deliberately and explicitly extend, not
silently redefine.

**Architectural decision (T13 interface-integration fix, 2026-09-22): `NetCDFEnvironmentalFieldProvider`
is now the genuine, functional access path to `U`/`K`/`grad_K`/`div_U`/`validity`, backed
by the SAME shared helper functions the live T05 pipeline calls — not a second,
independent implementation.** (This entry supersedes an earlier note recorded at the
first T13 review, which flagged that `K`/`grad_K`/`div_U` were unimplemented stubs the
live pipeline bypassed entirely; that gap is now fixed.) Two small functions in
`iste/environment.py` — `interpolated_divergence(ocean_source, lat, lon, t, radius_m)`
and `interpolated_k_value_and_gradient(k_lat, k_lon, k_field, lat, lon, radius_m)` — wrap
T02's interpolation and T03's `divergence_2d`/`spatial_gradient` to answer "`div(U)`/`K`/
`grad(K)` at this continuous `(x,t)`," and both `NetCDFEnvironmentalFieldProvider.div_U`/
`K`/`grad_K` AND `iste.step.backward_euler_maruyama_step` (T05) call these same two
functions rather than each containing its own copy of the bracket-and-blend /
gradient-and-interpolate math. **Any later task (Stage IV+, or further Stage III
maintenance) that needs `div(U)`, `K`, or `grad(K)` at a point should call these two
`iste.environment` helpers (or the provider methods that wrap them) rather than
reimplementing the computation inline a third time** — this is now the established
single-implementation pattern for "T02+T03 composed to answer a query at one point,"
analogous to T10's single-implementation pattern for diagnostics arithmetic. `K` remains
externally supplied only in both the provider and T05: passing no `K` grid to
`NetCDFEnvironmentalFieldProvider`'s constructor makes `K`/`grad_K` raise a `ValueError`
stating exactly that, never a silently invented default; `div_U`/`U`/`validity` need no
`K` and work regardless.

**Architectural decision (T10, 2026-09-22): Stage III diagnostics arithmetic
(`total/valid/invalid_particles`, `total_weight`, `effective_sample_size`,
`valid_particle_fraction`) lives in exactly one place, `iste/diagnostics.py`, and every
other module computes diagnostics by calling into it rather than reimplementing the
formula.** T09's `iste/ensemble_run.py` originally had its own private
`_diagnostics_from_final_state` duplicating Kish's ESS formula; T10 extracted that into
`iste.diagnostics.compute_stage3_diagnostics`/`compute_stage3_diagnostics_from_log_weights`
and refactored `ensemble_run.compute_diagnostics`/`compute_diagnostics_by_member` to
delegate to it (they still own the ensemble-specific "which row is each
`(particle, forcing_member)` group's final checkpoint" selection — that part is not
diagnostics logic and stays local). Any later task that needs these six diagnostics
(T11 storage round-trips, T12's controlled validation suite, T13 sign-off) must call
`iste.diagnostics`, not recompute ESS itself. The module also fixes an explicit, durable
policy for a data-quality edge case: **a non-finite (`NaN`/`Inf`) weight on a particle
marked invalid is harmless and ignored** (invalid particles never contribute to
weight-based diagnostics regardless of their stored weight's value), but **a non-finite
weight on a particle marked valid raises `ValueError`** rather than silently propagating
into `total_weight`/`effective_sample_size` — per `CLAUDE.md`'s "do not silently invent,"
a non-finite weight on a claimed-valid particle indicates an upstream bug that must
surface, not be absorbed into a plausible-looking number. Later stages must not weaken
this to a silent skip/zero-fill.

**Architectural decision (T11, 2026-09-22): the Stage III lookup-table store
(`iste/lookup_storage.py`'s `ParticleRecordStore`) is columnar (one array per field, plus
dict-based indices), holds only the eight required record fields, and both accepts and
returns T09's own `EnsembleRunRecords` type rather than inventing a parallel one — a
round trip through the store yields the same type that went in.** Because it never
imports `iste.environment`/`iste.forcing_ensemble`, it cannot duplicate an ocean/wind/K/
forcing-grid field even by accident. The store is deliberately append-only and never
deduplicates: re-adding the same batch of records produces duplicate rows on purpose,
since the store has no way to distinguish a legitimate new record from an accidental
re-add and does not silently guess either way — later tasks must not rely on the store
silently deduplicating anything. **Gotcha future tasks must remember:**
`ParticleRecordStore.checkpoints()` sorts ascending in real time, so its first entry is
the *oldest* timestamp — i.e. the far end of the backward window — not the observation
instant `t0`; this tripped up an early draft of T11's own manual-check notebook. A later
task wanting "the observation checkpoint" must look it up by its known `t0` value
directly, never assume `checkpoints()[0] == t0`.

**Architectural decision (T13.5, 2026-09-22): exact particle death metadata
(`death_time`, `death_lat`, `death_lon`, `death_reason`) and `survival_mark` are now
tracked end-to-end, additively, through `WeightedParticleState` → `BackwardTrajectory` →
`EnsembleRunRecords` → `ParticleRecordStore`, distinct from — and never overwriting —
the particle's own "last valid position" (`lat`/`lon`), per
`final_stage_III_IV_specification.txt`'s Sec. 6/9 requirement that Stage IV be able to
interpolate a particle's position/log-weight between checkpoints and exclude it from its
EXACT death time onward, without interpolating an already-aggregated field.** `death_time`
is always the exact checkpoint timestamp a rejected step targeted — never a fabricated
sub-step-interpolated instant — because this integrator's checkpoint spacing currently
equals its integration step size; if a future task ever decouples those two (finer
internal integration than checkpoint output), `death_time` can then legitimately fall
strictly between two stored checkpoints, and the schema already supports that (it is a
scalar timestamp field, independent of the checkpoint array). `death_reason` is one of
`"land"` (destination inside the domain's own coverage but masked/missing data),
`"domain_exit"` (destination outside the domain's covered extent), `"physics_unavailable"`
(K/grad(K)/div(U) missing at the particle's CURRENT position, so no destination was even
computed), or `"already_invalid_upstream"` (T07/T09's pre-existing-invalid particles,
whose "death" predates the backward window and is recorded at `t0` with no integration
event to justify anything more precise). All five new fields are additive with safe
defaults (`None`/`NaT`/`NaN`/`""`/mirroring `valid`), so every pre-T13.5 test and caller
continues to work completely unmodified.

**Architectural decision (T07, corrected 2026-09-22): satellite segmentation and particle
generation are owned by an upstream teammate module, not by this backtracking codebase.**
This backtracking codebase (`iste/`) starts *after* upstream particles already exist. An
earlier version of T07 built a synthetic observation footprint and generated particles
from it internally — that was superseded as redundant and removed, because it created a
second, competing particle-generation path that could be confused with the teammate's
real pipeline. `iste/observation.py` now provides only a minimal **adapter**
(`adapt_upstream_particles`) that validates upstream particle records (plain mappings
with at least `particle_id`, `latitude`, `longitude`, `observation_pixel_id`, `timestamp`,
and an optional `valid` flag — the smallest contract needed to drive T05/T06, not a
speculative full schema, since the upstream schema has not been formally specified) and
hands them off into T05's `WeightedParticleState`. Do not reintroduce footprint/particle
generation into this codebase; if a richer upstream schema or a real integration point is
needed later, extend the adapter's accepted fields, not this codebase's own generation
logic. `observation_pixel_id` (and `particle_id`) are tracked in a separate parallel
array/wrapper next to T05's `WeightedParticleState`/T06's `BackwardTrajectory` (both
unmodified), rather than added as a new field on those structures — later tasks (T08's
forcing-member id, T09's combined multi-particle/multi-member run) should follow the same
pattern of pairing immutable per-particle tags alongside the existing state/trajectory
structures instead of growing them in place.

## Verified environmental datasets

### Ocean (`data/ocean.nc`, Copernicus MEDSEA)

- variables: `uo`, `vo`; units: m/s; dtype float32
- 336 hourly records, `2019-10-01T00:00` → `2019-10-14T23:00` UTC (14-day span)
- grid: 149 (latitude) × 216 (longitude)
- latitude **ascending**
- `_FillValue = 1e20`; `xarray`'s default `mask_and_scale=True` converts these to `NaN`
  automatically on open (verified: `mask_and_scale=False` shows the raw `1e20` sentinel; default
  open shows 2,974,608 / 10,813,824 values as `NaN` — i.e. land/no-data cells). A loader must
  still mask explicitly rather than silently depend on this default, per `CLAUDE.md`.

### Wind (`data/wind.nc`, ERA5 10 m wind via cfgrib→CDM/CF)

- variables: `u10`, `v10`; units: m/s; dtype float32
- same 336 hourly timestamps as the ocean file (`time == valid_time`, verified element-wise equal)
- grid: 25 (latitude) × 36 (longitude) — different resolution from the ocean grid
- latitude **descending** — opposite orientation from the ocean file
- no `NaN`s found in `u10` in the inspected file

### Cross-dataset facts

- hourly timestamps align exactly between the two files;
- spatial grids differ in resolution between ocean and wind;
- latitude orientation differs between ocean (ascending) and wind (descending);
- no diffusivity (`K`) field is present in either file — `K` must be supplied explicitly by
  configuration/experiment, not read from data or invented as a formula.

## Stage III — Weighted Backward Adjoint (per `README.md`)

Backward time: `theta = t0 - t` (sign convention must stay explicit throughout the codebase).

SDE:

```text
dX* = (-U + grad(K)) dtheta + sqrt(2K) dB
```

Discrete Euler–Maruyama step:

```text
X_new = X_old + (-U + grad(K)) * dt + sqrt(2*K*dt) * xi,   xi ~ N(0, I_2)
```

Dynamic particle weight:

```text
d(ln w) / dtheta = -div(U)
```
```text
log_w_new = log_w_old - div(U) * dt
```

**Clarified 2026-09-22 (T13.5, per `final_stage_III_IV_specification.txt` Sec. 5):
`div(U)` here means the divergence of the BASE deterministic current `U`, never of a
forcing-member-perturbed `U_e = U + delta_U_e`, even though the particle's own advection
during that step uses `U_e`.** This was a genuine, confirmed bug before T13.5: because
`iste.step.backward_euler_maruyama_step` originally computed divergence from whichever
single `ocean_source` argument it was given, and T09's `run_ensemble` passes the
perturbed per-member source for advection, the weight update was silently computing
`div(U_e)`, not `div(U)`, for every ensemble run (confirmed numerically to differ
substantially, not a rounding-level discrepancy). Fixed via a new, optional
`divergence_source` parameter (default `None` preserves the old single-source behavior
for any non-ensemble caller); `iste.ensemble_run.run_ensemble` now passes the base field
explicitly. Any future code that runs backward transport under a forcing-member
perturbation must do the same — pass the UNPERTURBED base source as `divergence_source`
— rather than letting divergence be silently computed from whatever source happens to be
used for advection.

Stage III v1 assumptions (from `README.md`):

- horizontal 2-D model;
- `U(x,t)` is the combined horizontal transport field, decided and documented explicitly, not
  implicit; **locked for Stage III v1 (2026-09-22): `U(x,t) = (uo(x,t), vo(x,t))`, the Copernicus
  ocean-surface current only. Windage is excluded from `U` in Stage III v1** because neither
  `README.md` nor `ISTE_Claude_Architecture_v1.0.txt` specifies a windage coefficient, turning
  angle, or combination formula, and none was invented to fill that gap. This excludes windage
  from the transport-velocity definition only — `u10`/`v10` remain loaded, verified project data
  (see below) and stay available for other uses (e.g. a possible future windage-inclusive `U`
  version, or `README.md` Stage I's separate wind-dependent detection probability `pi_j`).
  Introducing windage into `U` later requires a new explicit decision (coefficient/formula), not
  a silent addition;
- `K(x,t)` is scalar and isotropic (no tensor/anisotropic diffusivity in v1);
- `B` is 2-D Brownian motion;
- boundary/invalid-particle handling is explicit — a valid initial prototype policy is: a
  proposed step into an invalid/land cell marks the particle invalid, stops its propagation, and
  retains invalid status for diagnostics; particles leaving the valid domain also become invalid.

The weight correction is intended to account for compression/expansion under the chosen
transport operator — it is **not** an unconditional mass-conservation guarantee. Its validity
depends on the transport operator, stochastic convention, diffusivity definition, boundary
conditions, and numerical discretization actually used, and must be tested numerically (zero
divergence, known convergent/divergent flow) rather than assumed correct because it runs.

Forcing ensemble: initial target `E ≈ 50` perturbed, internally-coherent forcing members,
representing coherent current/wind bias and spatially/temporally correlated errors — distinct
from independent particle-level diffusion. **The exact perturbation mechanism and its
covariance/parameterization has not yet been decided and must be made explicit before
implementation** (per `README.md`); do not invent one silently.

**Decision (T08, 2026-09-22): no physically-sourced forcing ensemble is currently
available, so a documented surrogate ocean-current perturbation model (README.md's
"Option B") is the recommended path — not yet implemented.** Both `data/ocean.nc` (CMEMS
MEDSEA `MFS E3R1I` reanalysis) and `data/wind.nc` (ERA5, confirmed via its `number=0`,
`expver="0001"` metadata to be the deterministic product) are single realizations with no
ensemble-member dimension. No known historical Mediterranean hourly ocean-current
ensemble reanalysis compatible with the 2019 period/domain/resolution was identified
(CMEMS's global ensemble reanalysis product, GREP, exists but is monthly-mean, failing
Stage III's hourly-resolution requirement). A real wind ensemble (ERA5's
ensemble-of-data-assimilations product) exists but is *not* a solution here: `U = (uo, vo)`
excludes windage in Stage III v1 (T01.5), so a wind ensemble does not perturb `U` unless
windage is reintroduced into `U` later — do not conflate "a wind ensemble exists" with
"Stage III forcing uncertainty is solved." If/when Option B is implemented, every
parameter (perturbation form, spatial/temporal covariance, amplitude, `E`, RNG method,
computational approach, validation strategy — see `tasks.md`'s T08 checklist) must be
treated explicitly as an assumption pending validation, ideally with the amplitude
informed by a citable source (e.g. the CMEMS product's own QUID validation statistics, or
a drifter comparison) rather than an arbitrary guess. A dense full-grid spatial covariance
matrix (the ocean grid has ~32,000 cells) is computationally infeasible on a normal
personal computer; a low-rank/low-dimensional or streamable generation method should be
preferred once this is implemented.

**Superseded design note:** an earlier same-day proposal recommended a divergence-free
streamfunction+coherent-bias form (`dU_e = b_e(t) + curl(psi_e(x,t))`). The project owner
then specified a different, simpler model to actually build instead (below); the
streamfunction proposal was never implemented and no longer describes the codebase.

**Implemented prototype (T08, 2026-09-22) — in `iste/forcing_ensemble.py`; PROTOTYPE,
NOT VALIDATED.** `U_e(x,t) = U(x,t) + delta_U_e(x,t)`, with `delta_U_e` generated as:
independent Gaussian random values on a COARSE grid (never a dense covariance matrix) →
each coarse cell evolved in time by an Ornstein-Uhlenbeck process → Gaussian spatial
smoothing of the coarse state → bilinear interpolation (T02, reused unchanged) to any
query point. **`delta_U_e` is deliberately NOT forced to be divergence-free** (this is the
opposite choice from the superseded streamfunction proposal) — if `div(U_e)` is ever
needed, it must be computed from the actual perturbed field via T03's `divergence_2d`, not
assumed; this module does not compute or wire in a divergence value itself. `u`/`v`
perturbation components are generated independently (separate RNG streams, no imposed
cross-component correlation) — a further simplification, not a claim that real ocean
eddies behave this way.

Explicitly labeled **PROTOTYPE ASSUMPTIONS, not validated**: `spatial_correlation_km =
20.0`, `correlation_time_seconds = 86400` (~1 day), `sigma_u = 0.05` m/s — reasoned from
general Mediterranean/oceanographic domain knowledge, **not verified against
`MEDSEA_MULTIYEAR_PHY_006_004`'s own QUID validation statistics or drifter data**. Do not
treat these as validated; see `progress.md`'s T08 section for the validation plan.
`n_members` (`E`) defaults to `50`, carried over unchanged from `README.md`'s own already-
flagged "initial target," not a new number. `coarse_resolution_deg` and `master_seed` are
configuration/reproducibility choices, not physical claims. Empirically, the implemented
model's temporal correlation at the 1-day lag came out ≈0.36, close to the OU process's own
theoretical `exp(-1) ≈ 0.368` — an internal-consistency check, not external validation.

Computationally: only a compact per-member coarse-grid coefficient time series is cached
(measured ≈6.6 MB/member on the real ocean grid, ≈315 MB for all 50 members if cached at
once); the expensive grid-sized field is generated on demand at query time, never
precomputed/stored for all members × all timestamps (≈8 GB if done naively) — mirroring how
`uo`/`vo` are already queried on demand elsewhere in this project. T05 (`iste/step.py`) and
T06 (`iste/trajectory.py`) were not modified; wiring forcing members into the backward step
is T09's job, not yet done.

Stage III output is a Monte Carlo transport estimate `c_hat*_{j,e}(x,t)` per observation pixel
`j` and forcing member `e` — not an exact probability. Store particle records at approximately
15–30 minute checkpoints (checkpoint interval is a storage/output cadence, separate from the
numerical integration step size `dt`, which is its own explicit configuration parameter — do not
conflate the two).

Track as diagnostics: total/valid/invalid particles, total weight, effective sample size,
valid_particle_fraction.

**Implementation decision (T06, 2026-09-22): the checkpoint timestamp grid is always fixed
by `t0`/`n_steps`/`dt` and never depends on particle validity.** A trajectory always has
exactly `n_steps + 1` recorded states at `t0, t0-dt, t0-2dt, ...`, computed directly from
`n_steps`/`dt` rather than accumulated step-by-step; once a particle becomes invalid, its
last valid position/weight is carried forward unchanged into every remaining state on this
same fixed grid (still marked invalid), never truncating the array. This keeps every
particle's recorded timestamps aligned regardless of when it stopped, which matters as
soon as multiple particles/trajectories need to be compared or combined at a shared
checkpoint time (T09 onward) — later tasks must preserve this fixed, validity-independent
schedule rather than only recording timestamps up to where a particle actually moved.

**Implementation decision (T05, 2026-09-22): the weight update commits only together with
an accepted move.** When a proposed step is rejected by the boundary policy (or a
particle's current position has unavailable physics input), the particle's log-weight is
frozen at its pre-step value, exactly like its position — the weight is never advanced for
a step that did not actually happen. This is not stated in those words by `README.md`, but
is the direct extension of its "stop propagation, retain last valid state" boundary policy
to the weight trajectory, and every later task building on the backward step (T06 onward)
must keep this coupling, not silently decouple weight updates from accepted moves.

## Numerical requirements

- never add metre displacement directly to lat/lon degrees;
- use a consistent Earth-surface conversion or an already-installed geodesic library;
  **decided in T03: no geodesic library (e.g. `pyproj`) is installed in this project's
  environment, so degree↔metre conversion uses a documented spherical-Earth approximation
  (WGS84 mean radius, ~6,371,000 m) in `iste/geometry.py` — metres-per-degree-latitude is
  constant, metres-per-degree-longitude scales by `cos(latitude)`. The radius is an
  explicit function parameter with this documented default, not a hidden constant. This is
  the standing conversion for all Stage III code (T03's derivatives now, T05's particle
  displacement later) unless a geodesic library is added and a new decision supersedes
  it;**
- mask ocean fill values before interpolation, explicitly (not only by relying on `xarray`
  defaults);
- explicitly handle the different latitude ordering in ocean and wind arrays;
- keep spatial and temporal interpolation as separate, independently testable functions;
- do not hard-code either NetCDF file's specific layout (dimension names/order, coordinate
  names) into the transport/scientific equations — the environmental field interface
  (`U`, `K`, `grad(K)`, `div(U)`, `validity`) is the seam that isolates equations from data
  layout.

## Stochastic model

Random diffusion (the `sqrt(2K) dB` term) is reproducible through an explicit RNG object/seed.
Same seed + same inputs must reproduce the same stochastic result; a different seed represents a
different stochastic realization. This reproducibility requirement applies to the forcing
ensemble's perturbations as well, once that mechanism is defined.

## Validation priorities

Always validate numerical behavior, not just whether the program runs:

- masking
- coordinate orientation
- spatial interpolation
- temporal interpolation
- backward-time sign convention
- metre-to-coordinate conversion
- `grad(K)` / `div(U)` operator correctness against analytic test fields
- particle-weight behavior under zero/known-convergent/known-divergent flow
- boundary/validity handling
- reproducibility (RNG seed)
- forcing-ensemble member coherence (once the perturbation mechanism is defined)
- effective-sample-size / valid-particle-fraction behavior

## Stage IV observation model — PROTOTYPE, NOT VALIDATED (T15, 2026-09-22)

**No upstream teammate module, data file, or existing code anywhere in this repository
provides the Stage I detection probability `pi_j`/`pi_j,e`, or the raw ingredients
(a specified oil-thickness/backscatter model, a wind-dependent detectability curve, or
numeric parameters for one) needed to compute it** — confirmed by inspection before T15
was implemented (checked `data/`, all of `iste/`, and for any `.json`/`.tif`/`.geojson`/
`.csv` upstream artifact; found none). `ISTE_Claude_Architecture_v1.0.txt`'s Stage I
section explicitly requires this functional form to be "explicitly specified in the
implementation rather than invented silently" — since nothing has been specified by an
upstream source, `iste/detection_model.py` implements a **PROTOTYPE, NOT VALIDATED**
bridge from `h_j,e` (a per-pixel-per-forcing-member predicted-oil quantity, presumed
derived from Stage III's `c_hat*_{j,e}(x,t)` by a separate, not-yet-built step — T15
does not compute `h_j,e` itself) to `pi_j,e`:

```text
pi(h) = sigmoid( steepness * (h - midpoint) )
```

Both `midpoint` and `steepness` are **ASSUMED, configuration-only parameters with no
physical source** — same discipline as T08's forcing-ensemble prototype parameters
(20 km / 1 day / 0.05 m/s). `steepness` must be `> 0` (pi monotonically increasing in
`h` is a hard physical requirement: more predicted oil must never make a pixel less
likely to be observed as oil). **This mapping has no floor/ceiling of its own** — T14's
`epsilon` already supplies the "prediction is never perfectly certain" floor at the
`P(y_j|h_j,e)` level; a second floor here would double up that role. If/when a real
upstream Stage I model or dataset becomes available, `iste/detection_model.py` should be
replaced or given a second, real-model-backed implementation selectable via
configuration — its current PROTOTYPE output must never be treated as validated.

## Project pivot (2026-09-22): Stage IV surrogate-likelihood plan superseded by Phase 2

**`README.md`/`tasks.md` were rewritten to replace the Stage IV surrogate-likelihood/
Stage V (SMC)/Stage VI (MCMC)/Stage VII (posterior) plan with a new "Phase 2 — Candidate
generation + forward validation" design**: Stage III backward particle histories feed a
KD-tree/KDE candidate-discovery step, candidates are forward-propagated under a coherent
forcing member to `t_obs`, and agreement with the actual satellite slick is the primary
filtering/ranking signal — not a Bayesian surrogate-likelihood/SMC/MCMC posterior. Task
numbers were reassigned under the new `tasks.md`; the current T14 is "Historical particle
slice and Phase 2 input adapter" (`iste/particle_query.py`), not the surrogate-likelihood
module. **Every entry below this notice that discusses "Stage IV"/"Stage V"/"Stage VI"/
"Stage VII" (the surrogate likelihood, `beta` tempering, delayed-acceptance MCMC, etc.)
describes the SUPERSEDED plan and is retained only as historical record** — `iste/
likelihood.py` and `iste/detection_model.py` still exist and still pass their tests, but
are no longer part of the active task sequence. Consult `progress.md`'s "Project pivot
notice" and its "Phase 2 — T14" section for the current, active plan.

## T19 core sampling: forward-KDE weight normalization boundary (2026-09-23)

`README.md` Sec. 17 leaves it ambiguous whether the `Y < X` branch's "relative Stage III
weights may be normalized only for the forward predicted KDE" happens inside core
sampling itself or in the later stage that actually builds that KDE. The user explicitly
resolved this when specifying T19: **any weight normalization for the forward predicted
KDE must be confined entirely to whichever later stage constructs that KDE — T19
(`iste/core_sampling.py`) must never normalize, scale, or otherwise modify `log_weight`
under any circumstance**, in both branches. `iste.core_sampling.sample_core_particles`
instead returns a `resampled` boolean (`True` iff `Y >= X`, i.e. weighted-without-
replacement resampling ran) so the future forward-KDE-building task can decide its own
weighting convention (equal computational weight per particle when `resampled=True`;
optionally normalize the original, still-untouched weights when `resampled=False`).
Any later Phase 2 task that builds the forward predicted KDE must implement its own
weight handling from this `resampled` flag and the untouched `log_weight` values — it
must not assume T19 already did any normalization.

## T20 forward transport: forcing_member_id is provenance, not a per-particle U_e selector (2026-09-23)

`README.md` Sec. 19 says forward propagation runs "For each configured forcing member:
Candidate C -> U_1 -> predicted slick_1 -> U_2 -> ... -> U_E -> predicted slick_E" — i.e.
the SAME selected particle set (from T19) is propagated once per forcing member, each
time under ONE uniform `U_e` applied to every particle in the batch, producing `E`
separate predicted slicks per candidate. This means a particle's own `forcing_member_id`
(its Stage III backward-run provenance, carried through T14-T19 unchanged) is NOT used
to pick which `U_e` field that particle's own forward step uses — the forward member is
instead an external choice the orchestrating task (T21+) makes once for the whole batch,
by building the corresponding perturbed source (`iste.ensemble_run.
make_perturbed_source(base, ensemble, member_id)`) and handing it to
`iste.forward_transport.forward_euler_maruyama_step`. `ForwardParticleState.
forcing_member_id` therefore only ever carries the particle's ORIGINAL Stage III
member id through as a passive bookkeeping/provenance field, preserved unmodified by
every forward step; a future task must not read it back to decide per-particle forward
physics, or it will silently violate Sec. 19's "propagate the whole selected set under
EACH member in turn" design and instead (incorrectly) propagate each particle only under
its own original backward member.

Separately: **the Phase 2 forward step has NO weight-update term at all.** Unlike Stage
III's backward step (`log_w_new = log_w_old - div(U)*dt`), `README.md`'s forward
propagation sections (18-25) never mention updating a particle weight during forward
propagation — `iste.forward_transport.ForwardParticleState` deliberately carries no
`log_weight` field. A future task must not add a `div(U_e)`-based weight update to
forward propagation by analogy with Stage III without a new, explicit instruction to do
so — it would be an invented physical term, not something `README.md` specifies.

## T22 slick footprint: KDE normalized by valid-particle count, not raw density (2026-09-23)

`README.md` Sec. 20/22 requires that "the comparison must operate on the spatial
footprint/distribution and must not require equal particle counts," and separately
states the check "particle-count changes do not by themselves change a normalized
footprint score" — but does not itself specify the normalization mechanism. `iste.
slick_footprint._compute_density_grid` resolves this by dividing the summed Gaussian-
kernel contributions by `n_valid` (the count of VALID particles only, never the raw/
total count): `density = (1/n_valid) * sum_valid_particles(kernel(distance, bandwidth))`.
This is what makes the resulting footprint's shape/scale independent of how many
particles happened to produce it — proven exactly by an exact-duplication test (2N
identical particle positions give a bit-for-bit identical density to N). **Any later
task that consumes `PredictedSlickFootprint`/`ObservedSlickFootprint.density` (T23's
comparison metric, in particular) must treat `density` values as already
count-normalized** — it must not re-normalize by particle count again, and must not
assume `density` sums/integrates to exactly 1 (a Gaussian KDE's mass exceeds 1 near the
domain boundary of a finite grid; the normalization guarantee here is specifically
"count-independence," not "the grid sums to 1"). `density` is all-`NaN` (never `0.0`)
whenever a footprint has zero valid particles (`has_valid_particles=False`) — T23 must
check that flag explicitly rather than assuming a numeric `density` is always safe to
compare/subtract.

## T24 aggregation: `forcing_members_validated` means VALID members, not the attempted subset (2026-09-23)

`README.md` Sec. 25's output-artifact schema lists `forcing_members_validated` next to
`forward_error_by_member`/`forward_consistency_by_member` without itself defining what
the field contains. `iste.candidate_forward_accuracy.CandidateForwardAccuracy` resolves
this as: **`forcing_members_validated` is exactly the sorted tuple of member ids whose
T23 `ForwardAccuracyResult.valid` was `True`** (i.e. the ones that actually contributed
to `candidate_forward_error`/`candidate_forward_consistency`) — NOT the full set of
members a pipeline attempted to run forward propagation/comparison for. That broader
"attempted subset" concept already has its own name and home: T21's
`ForwardCandidateRuns.forcing_member_ids` (the `README.md` Sec. 19/26 "development run
may use an explicit subset of forcing members ... must be recorded" concept). A future
task (T25 in particular, which consumes `CandidateForwardAccuracy`) must not confuse the
two: if it needs "how many members were attempted" separately from "how many validated
successfully," the former lives on the T21 result, not on T24's
`forcing_members_validated`, and the latter (plus every invalid member's own reason) is
`CandidateForwardAccuracy.member_results`/`invalid_member_reasons`.

## T25 screening: threshold/top-K and merging are GLOBAL, not per-historical-timestamp (2026-09-23)

`README.md` Sec. 24 describes filtering/ranking as a flat "PRIMARY filter/ranking" step
over "every candidate/time" without specifying whether the threshold/top-K selection
(and the near-duplicate merge step) operate independently per historical timestamp
`t_s`, or globally across the whole candidate set spanning every `t_s`.
`iste.candidate_screening.filter_and_rank_candidates` resolves this as: **threshold and
top-K selection, and the final rank numbers, are GLOBAL across every candidate/`t_s` in
one call** — e.g. `top_k=5` keeps the 5 lowest-forward-error candidates in the ENTIRE
input, not 5 per timestamp. The one exception is near-duplicate merging, which is
explicitly scoped to compare only candidates that share the SAME `t_s` (candidates from
different historical timestamps are never merged into each other, regardless of spatial
proximity — this part is a hard requirement, not a default). A future task (T26's
end-to-end integration, in particular) that wants per-timestamp top-K/threshold
selection instead must call `filter_and_rank_candidates` once per `t_s` group itself
(grouping the input `CandidateRecord`s by `t_s` beforehand) — the module does not do
this automatically, and assuming it does would silently produce a differently-shaped
candidate set than intended (e.g. one very strong historical timestamp could consume
the entire global `top_k` budget, leaving none for other timestamps).

## T26 pipeline: reproducibility depends on a FIXED seed-spawn order — do not parallelize candidate/member processing without preserving it (2026-09-23)

`iste.phase2_pipeline.run_phase2_pipeline` derives every T19 (core-sampling) and T21
(forward-propagation) RNG seed from a single `Phase2RunConfig.master_seed` via
`numpy.random.SeedSequence.spawn`, called from ONE shared `SeedSequence` object in a
FIXED, sequential order: historical timestamps in the configured grid's own order; for
each timestamp, candidates in T17's own deterministic descending-support order; for each
candidate, forcing members in the configured `forcing_member_ids` order. Reproducibility
("same inputs + same configuration + same RNG seeds -> identical Phase 2 artifact") is
provided ENTIRELY by this fixed spawn order — there is no per-candidate or per-member
seed stored anywhere in the persisted JSON artifact (only `master_seed` is). **Any future
change that processes candidates or forcing members out of this fixed order — e.g.
parallelizing the per-candidate loop across threads/processes, or batching multiple
timestamps concurrently — will silently break reproducibility** (the same `master_seed`
would then produce a different, order-dependent assignment of spawned seeds to
candidates/members) unless that future change explicitly preserves this sequential spawn
order (e.g. by pre-spawning all needed seeds up front, sequentially, before dispatching
any parallel work). `adapted_observation` (T07's `AdaptedParticleSet`) is deliberately
passed into `run_phase2_pipeline` as a SEPARATE argument from `records` (the T09 Stage
III output it already seeded) — the pipeline uses it a second, independent time to build
T22's observed-slick reference directly from the upstream particles, never by re-deriving
it from `records`/the backward histories (`README.md` Sec. 21).

## Real-data environmental forcing date mismatch: user-approved standing decision (2026-09-23)

**None of the real upstream oil-slick observation records currently in this repository
overlap in time with the only real ocean-current data available.** `data/ocean.nc`/
`data/wind.nc` cover `2019-10-01`–`2019-10-14` only; the 5 real spill records in
`data/database_payload.json` are dated Jan/Apr/Jun/Sep/Dec 2019 — none fall inside that
window. When this was discovered while building `notebooks/main_actual_data_pipeline.ipynb`
(using `ow-0008.jpg`, June 2019), the user was asked directly (via a blocking question,
per this project's own "do not use synthetic fallback data if real data fails — stop and
report" instruction) and **explicitly chose to proceed using the real observation
particles together with the only real ocean/wind data available (Oct 2019), with `t_obs`
substituted to `ocean.nc`'s own last checkpoint**, rather than stopping or waiting for
matching-date environmental data. This is a **pipeline-mechanics validation on real data,
not a physically accurate reconstruction of the true event** — every output from that run
is loudly labeled with this caveat, never presented as if the dates aligned. **Any future
real-data run using one of these same 5 spill records (or any other observation whose
timestamp isn't covered by whatever environmental data is available) will hit the exact
same mismatch** — this decision (proceed with the best-available real environmental data,
loudly labeled, rather than stopping) is the user's own standing preference for this
project, discovered once and worth re-applying rather than re-asking, unless the user
says otherwise or matching-date environmental data becomes available in the repository.

## Project boundaries

**Stage III was signed off 2026-09-22** (see `tasks.md`'s T13 entry); the (now
superseded-numbering) Stage IV surrogate-likelihood task was completed under the old
plan (`iste/likelihood.py`). Until each later Phase 2 task is itself reached and signed
off, do not build ahead of it — see the pivot notice above for the current plan:

- Stage V (grid screen + tempered SMC)
- Stage VI (delayed-acceptance MCMC)
- Stage VII (posterior/AIS-ready output)
- satellite segmentation / particle generation (owned by an upstream teammate module —
  this codebase only adapts/consumes already-generated particles, see T07)
- AIS attribution
- frontend
- databases
- services
- forecasting
- waves
- 3D hydrodynamics
- machine-learning forecasting

**Architectural decision (T14, 2026-09-22): `iste/likelihood.py`'s
`compute_stage4_log_likelihood` accepts `pi_j,e` (the per-pixel, per-forcing-member
detection probability) as a given input array — it does not compute `pi_j,e` from Stage
III's `c_hat*_{j,e}(x,t)` transport estimate.** That mapping (transport estimate →
detection probability) is a separate, not-yet-built step; whichever later task builds it
must produce a `(n_pixels, n_members)` array and hand it to `compute_stage4_log_likelihood`,
not have T14 (or any future Stage IV task) reach back into `iste.ensemble_run`/
`iste.lookup_storage` to derive it inline. Likewise `y_j` (observed pixel labels) is
accepted as given, from the upstream Stage I system — T14 does not implement
segmentation/classification.

Mathematical requirements preserved so far, now enforced in code (not just recorded as a
future intention):

- **Stage IV tempering** (`iste/likelihood.py`, T14, implemented): `beta` is applied to
  the *marginalized* forcing-ensemble mixture, `[ (1/E) sum_e L_e ]^beta` — never
  `mean_e(L_e^beta)`. The implementation computes everything in the log domain
  (`log L_e` as a sum over pixels, a stable shift-by-max log-mean-exp for
  `log mean_e(L_e)`, `beta` multiplied in only after that), so the correct ordering is
  structural, not just conventional. `log_surrogate_likelihood`/`log_mean_member_likelihood`
  can legitimately be `-inf` (e.g. `epsilon == 0` and a pixel flatly contradicted by
  every forcing member) — this is a real, valid result, not a bug; any later caller
  (Stage V/VI) must handle `-inf` (e.g. as "this hypothesis is rejected"), never treat it
  as a NaN-like error state.
  **UNRESOLVED CONFLICT flagged 2026-09-22 (T13.5): `final_stage_III_IV_specification.txt`
  (supplied after T14 was approved) specifies the OPPOSITE beta ordering,
  `mean_e(L_e^beta)` (beta applied to each member's likelihood BEFORE marginalizing,
  because that document defines `beta = J_eff/J` as a within-member spatial-correlation
  correction, not a post-marginalization tempering exponent) — directly contradicting
  `README.md`'s formula, which is what T14 actually implements. T13.5 explicitly did NOT
  modify T14 (out of its scope) and only flagged this for the project owner's attention.
  Any future Stage IV work must get an explicit decision on which formula is authoritative
  before proceeding — do not silently pick one.**
- **Stage VI correction** (not yet built): must use Delayed-Acceptance MCMC (surrogate
  screen → full-physics run only on surrogate-accepted proposals → full-physics
  accept/reject), not ordinary importance sampling/reweighting.

## Code organization

Reusable scientific/application logic belongs in `.py` modules.

Jupyter notebooks are only for experiments, debugging, plots, and manual/visual inspection.

## Claude Code workflow

Work one task at a time from `tasks.md`.

After each task:

1. run the required check;
2. update `tasks.md`;
3. update `progress.md`;
4. update this file only if a durable scientific/architectural decision/fact changed;
5. stop and report concisely.

Avoid speculative abstractions and unnecessary dependencies. Do not silently replace the
specified Stage III weighted-adjoint methodology with simpler deterministic backtracking.
