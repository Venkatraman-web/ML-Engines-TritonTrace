# SIH ISTE Tasks — Lightweight, Sequential Plan

> **2026-09-22 — architecture correction.** This task list previously described a simpler
> "Reverse Lagrangian Particle Tracking (RLPT)" engine (deterministic/stochastic reverse
> trajectories + grid-based KDE, no particle weighting, no forcing ensemble). That plan
> conflicted with `README.md` (the declared scientific source of truth) and with
> `ISTE_Claude_Architecture_v1.0.txt`, both of which explicitly reject naive deterministic
> backtracking as the inference engine. The RLPT plan is **superseded**. The task sequence
> below targets `README.md`'s Stage III — Weighted Backward Adjoint — as the first
> implementation milestone, per `CLAUDE.md`.

## Rules for Claude Code

Work on **exactly one task per turn**.

For each task:

1. Read only the files needed for that task.
2. Make the smallest correct implementation.
3. Run the task's check.
4. Mark the task `DONE` only when the check passes AND it has been manually approved
   (per `CLAUDE.md` status convention: `TODO` → `IN_PROGRESS` → `PENDING_REVIEW` → `DONE`).
5. Update `progress.md` with files changed, check result, and actual numerical/visual result.
6. Update `memory.md` only when a durable scientific or architectural fact was discovered/decided.
7. Stop and report. Do not continue to the next task automatically.

Keep responses short. Do not redesign the whole project while completing a single task.

Do not implement Stage IV, V, VI, or VII until Stage III has been signed off (`README.md`
"Scope rule").

---

## T00 — Inspect repository
**Status:** DONE

Inspect:

- repository structure;
- Python version;
- installed/referenced scientific packages;
- location of the two NetCDF files;
- any existing backtracking/adjoint code that should be reused.

Do not modify scientific code.

**Check:** report only what was actually found.

**Progress report:** repository map + minimal implementation plan.

---

## T01 — Build minimal environmental field loader
**Status:** DONE

Create a small reusable loader for:

```text
uo, vo
u10, v10
lat/lon
UTC time
```

Requirements:

- preserve raw NetCDF files;
- mask ocean fill values around `1e20`;
- normalize only in memory;
- expose dataset metadata cleanly;
- do not hard-code either file's specific NetCDF layout into any transport/scientific equation —
  the loader is an adapter, later stages consume a normalized interface.

**Check:** print dimensions, timestamp range, coordinate directions, and missing counts.

**Progress report:** compact diagnostic table/output.

---

## T01.5 — Define transport velocity U
**Status:** DONE

Documentation-only task. No interpolation, particle propagation, `K`, `grad(K)`, `div(U)`, or
forcing-ensemble code is implemented here.

### Purpose

Explicitly document the scientific definition of the Stage III transport velocity `U(x,t)`
used by the backward adjoint (`dX* = (-U + grad(K)) dtheta + sqrt(2K) dB`), reconciling
`README.md`'s wording with `ISTE_Claude_Architecture_v1.0.txt`'s more detailed elaboration of
the same Stage III v1 assumption, before T02/T05 need to consume it.

### What `README.md` / `ISTE_Claude_Architecture_v1.0.txt` actually specify

- `README.md` (Stage III v1 assumptions): `U(x,t)` is "the combined horizontal transport
  field."
- `ISTE_Claude_Architecture_v1.0.txt` (Part 2, Stage III V1 Assumptions, item 1), elaborating
  the same assumption: `U(x,t)` is "the horizontal ocean-surface velocity field after all
  selected deterministic transport components (including any explicitly modeled windage
  contribution) have been combined."

Reconciled reading: the mandatory base component of `U` is the horizontal **ocean-surface
current** (`uo`, `vo` in `data/ocean.nc`). Windage — a wind-driven drift contribution derived
from `u10`/`v10` in `data/wind.nc` — is described as something that may be "selected" and
"explicitly modeled," not as a component that is automatically included. Neither document names
any other deterministic transport component (no Stokes drift, tidal correction, Ekman spiral,
etc. is mentioned anywhere in either document).

### Decision: windage inclusion — RESOLVED 2026-09-22

**Locked for Stage III v1:**

```text
U(x,t) = (uo(x,t), vo(x,t))
```

i.e. `U` is the Copernicus ocean-surface current only. **Windage is excluded from the Stage III
v1 definition of `U`** — not implemented, not approximated, not defaulted to zero-with-a-hidden-
hook. This was a project decision, made explicitly because neither `README.md` nor
`ISTE_Claude_Architecture_v1.0.txt` specifies a windage formula or coefficient, and per
`CLAUDE.md`'s "do not silently invent ... forcing perturbations," no such formula was invented to
fill that gap.

**This excludes windage from `U` in v1 only — it does not discard wind data from the project.**
`u10`/`v10` remain loaded and verified (T01) and stay available in `data/wind.nc` for other uses
(e.g. a possible future windage-inclusive `U` version, or `README.md` Stage I's wind-dependent
detection-probability `pi_j`, which is a separate, unrelated use of wind data from the transport
velocity `U`). Only `U`'s Stage III v1 definition excludes wind.

T02/T05 must implement `U` against `uo`/`vo` only; no windage coefficient/formula may be added
without a new explicit decision recorded here.

### Units and sign convention (already established, not new)

- Units: `uo`, `vo`, `u10`, `v10` are all m/s (verified in T00/T01 against the NetCDF variable
  attributes and value ranges). `U(x,t)` is therefore m/s.
- Sign convention: `README.md` fixes backward time as `theta = t0 - t` and writes the drift term
  as `-U` in the backward SDE — i.e. the backward-time step moves *against* the physical
  (forward-time) velocity `U`. This is already recorded in `memory.md`'s Stage III section; T01.5
  does not change it, only restates it for completeness next to the `U` definition.
- Metre/degree handling: `U * dt` yields a metre displacement, which must be converted to
  lat/lon degrees explicitly (never mixed directly) — already a standing numerical requirement
  in `memory.md`, unaffected by the windage decision above.

### Check

Manual documentation review only: confirm the reconciled definition above matches `README.md`'s
and `ISTE_Claude_Architecture_v1.0.txt`'s wording, and that the windage question is recorded as
open rather than answered.

### Progress report

This section plus the corresponding `progress.md` entry.

---

## T02 — Implement and test interpolation
**Status:** DONE

Create separate functions for:

```text
bilinear spatial interpolation
linear temporal interpolation
```

They must handle:

- different ocean/wind grids;
- ocean latitude ascending;
- wind latitude descending.

Per `README.md`: spatial and temporal interpolation are separate concerns, and this interface
must expose `U(x,t)` — do not silently treat latitude/longitude degrees as metres anywhere here.

**Check:** use tiny synthetic grids with hand-computable values. Test both spatial interpolation
and hourly → sub-hourly temporal interpolation.

**Progress report:** one ocean example + one wind example + expected vs actual value.

---

## T03 — Implement grad(K) and div(U) operators
**Status:** DONE

Per `README.md`, Stage III needs, in addition to `U(x,t)`:

```text
K(x,t)       — scalar, isotropic diffusivity
grad(K)(x,t) — spatial gradient of K
div(U)(x,t)  — divergence of the transport field
```

`K` is not present in either NetCDF file and has no formula specified in `README.md`; for v1 it
must be an explicit, externally supplied parameter/field (e.g. a configurable constant or a
user-supplied test function) — do not invent a diffusivity formula.

Implement numerical gradient and divergence operators against the same grid/interpolation
interface from T01/T02, without hard-coding a particular NetCDF layout into the operators.

**Check:** analytic test fields with hand-computable gradient/divergence (e.g. linear and
quadratic synthetic fields) compared against the numerical operator output.

**Progress report:** operator output vs analytic expectation for each test field.

---

## T04 — Boundary and validity policy
**Status:** DONE

Implement the explicit prototype boundary policy from `README.md`:

```text
proposed step enters invalid/land cell
→ mark particle invalid
→ stop its propagation
→ retain invalid status for diagnostics
```

Particles leaving the valid ocean/data domain must also become explicitly invalid.

**Check:** synthetic domain with a land/invalid region and out-of-bounds area; verify a particle
proposing a step into an invalid cell is marked invalid and its propagation stops, and that
invalid status is retained (not silently dropped) for diagnostics.

**Progress report:** before/after state for a particle that hits land and one that leaves the
domain.

---

## T05 — Implement one Stage III backward step (Euler–Maruyama)
**Status:** DONE

Implement the discrete Stage III step from `README.md`:

```text
X_new = X_old + (-U + grad(K)) * dt + sqrt(2*K*dt) * xi,   xi ~ N(0, I_2)
log_w_new = log_w_old - div(U) * dt
```

with the backward-time convention `theta = t0 - t` kept explicit, `dt` as an explicit
configurable parameter (not hard-coded), the T04 boundary/validity policy applied, and an
explicit RNG object/seed for `xi` (reproducibility: same seed + same inputs → same step;
different seed → different stochastic outcome).

Do not mix metre displacement directly with lat/lon degrees.

**Check:** run 1–4 steps on controlled synthetic velocity/diffusivity fields, verifying both:
- transport direction/displacement matches the known field, and
- weight update matches the known field's divergence (zero-divergence flow → weight unchanged;
  a known convergent/divergent flow → weight decreases/increases as expected).

**Progress report:** before/after position, weight, and validity for each sample step, plus the
synthetic field used.

---

## T06 — Run one weighted particle over a configurable backward window
**Status:** DONE

Run one particle (with weight and validity tracked) through a configurable number of T05 steps
over a configurable backward window and `dt`. Do not hard-code the window length or `dt` as
universal constants — they are experiment parameters.

Store its trajectory, weight, and validity at each stored state, including timestamps.

**Check:** no NaNs while valid; propagation stops correctly after the particle becomes invalid;
weight stays finite; elapsed backward time matches `step_count * dt`.

**Progress report:** trajectory plot + weight-over-time plot + validity summary.

---

## T07 — Backtracking input adapter / particle interface
**Status:** PENDING_REVIEW

> **2026-09-22 — corrected.** The original version of this task built a synthetic
> observation footprint and generated/initialized particles from it. That was
> superseded as redundant: satellite segmentation and particle generation are owned by an
> upstream teammate module, not by this backtracking codebase. T07 now starts *after*
> upstream particles already exist.

`README.md`: "Stage III particles must retain `observation_pixel_id`... Stage III does not need
to implement the final observation likelihood yet."

Provide a minimal adapter/contract for particles coming from the upstream
segmentation/particle-generation module: validate that each upstream particle record
carries at least `particle_id`, `latitude`, `longitude`, `observation_pixel_id`, and a
`timestamp` (an optional upstream `valid` flag is honored if present), and repackage that
into T05's `WeightedParticleState` plus the preserved identifiers. Do not implement or
duplicate satellite segmentation, oil/clean classification, observation-footprint
generation, particle generation, Stage I detection probability, or observation
likelihood. Do not modify T05/T06.

**Check:** a valid mock upstream payload adapts correctly; a record missing any required
field is rejected explicitly; `particle_id`/`observation_pixel_id` survive unchanged
through adaptation and through a full T06 trajectory; an already-invalid upstream
particle is preserved (not dropped, not silently marked valid).

**Progress report:** mock upstream payload + adapted-particle table + preservation checks.

---

## T08 — Specify and implement the forcing ensemble
**Status:** DONE (prototype implemented; NOT validated)

> **2026-09-22 — decision gate completed.** Per `README.md`, "the exact perturbation
> mechanism/covariance must be explicit before implementation." Before writing any
> forcing-ensemble code, T08 first had to determine whether a physically-sourced ensemble
> product already exists and is usable, or whether a documented surrogate perturbation
> model is required instead. See `progress.md`'s T08 section for the full investigation.
>
> **Result: no compatible physically-sourced ensemble is currently available.**
> `data/ocean.nc` (CMEMS MEDSEA MFS E3R1I reanalysis) and `data/wind.nc` (ERA5,
> `expver=0001`, `number=0` — the deterministic product, not an ensemble) are both single
> deterministic realizations; no ensemble-member dimension exists in either file, and no
> known compatible historical Mediterranean hourly current ensemble was identified.
> **Recommended path: Option B** — a documented, explicitly-parameterized surrogate
> ocean-current perturbation model, with every parameter treated as an assumption
> requiring later validation, not a physical ensemble product. This has **not** been
> implemented, and none of its parameters have been decided (see the checklist below) —
> only the decision to pursue this path has been made.

Create `E` perturbed wind/current forcing fields, representing coherent spatial/temporal
forcing errors. These perturbations should NOT simply be replaced by additional
independent particle-level Brownian noise. Initial target `E ≈ 50`.

**Because `U = (uo, vo)` is locked for Stage III v1 (T01.5) and windage is excluded, only
ocean-current perturbations affect `U` right now — a wind ensemble (even a real one, e.g.
ERA5's ensemble-of-data-assimilations product) would not by itself supply Stage III v1
forcing uncertainty until/unless a windage term is reintroduced into `U`, which is a
separate, not-yet-made decision. Do not conflate "a wind ensemble exists" with "Stage III
forcing uncertainty is solved."**

### Implemented prototype (2026-09-22) — PROTOTYPE, NOT VALIDATED

> **Superseded design note:** an earlier design pass (same day) proposed a
> divergence-free streamfunction+bias model. The project owner then specified a
> **different, simpler** model to actually implement — coarse-grid random noise, Gaussian
> spatial smoothing, OU temporal evolution, explicitly **not** divergence-free. The
> streamfunction proposal was never implemented; the model below is what exists in code.

Implemented in `iste/forcing_ensemble.py` (`ForcingEnsembleConfig`, `ForcingEnsemble`).
Full write-up, empirical checks, and the sourced/assumed accounting are in `progress.md`'s
T08 section.

```text
model:      U_e(x,t) = U(x,t) + delta_U_e(x,t), U = (uo, vo) unchanged, windage excluded
generation: independent random values on a COARSE grid (never a dense covariance matrix)
            -> Ornstein-Uhlenbeck evolution in time, per coarse cell
            -> Gaussian spatial smoothing of the coarse state
            -> bilinear interpolation (T02, reused) to any query point
NOT forced divergence-free (per instruction) -- div(U_e), if ever needed, must be
  computed from the actual perturbed field via T03's divergence_2d, not assumed here
PROTOTYPE ASSUMPTIONS (labeled, not validated):
  spatial_correlation_km = 20.0
  correlation_time_seconds = 86400.0  (~1 day)
  sigma_u = 0.05  (m/s)
carried over, not invented: n_members (E) = 50 (README.md's own "initial target")
configuration choices, not physical claims: coarse_resolution_deg, master_seed
memory discipline: only a compact per-member coarse-grid coefficient time series is ever
  cached (measured ~6.6 MB/member on the real ocean grid, ~315 MB for all 50 cached at
  once); the grid-sized field is generated on demand, never precomputed/stored for all
  members x all timestamps (~8 GB if done naively)
```

**Check:** `tests/test_forcing_ensemble.py` — reproducibility (same seed -> identical;
different seed/member -> different), spatial correlation observable and decaying with
distance, temporal correlation observable and decaying with lag (empirically close to the
theoretical `exp(-lag/tau)`), perturbation scale same order of magnitude as `sigma_u`,
memory bounded, 50 members generated without full-grid storage, and confirmation that the
perturbed field's divergence is NOT forced to zero.

**Progress report:** `notebooks/T08_forcing_ensemble_check.ipynb` — unperturbed vs.
perturbed-member quiver plots, magnitude histogram vs. target `sigma_U`, empirical
spatial/temporal correlation curves, reproducibility, and memory footprint, all labeled
**PROTOTYPE — NOT VALIDATED**.

---

## T09 — Small multi-particle, multi-forcing-member Stage III run
**Status:** DONE

For a small pixel set (T07) and a small forcing ensemble (T08), run the Stage III backward
adjoint per observation pixel and per forcing member, producing per-checkpoint records:

```text
timestamp
location
particle_weight
observation_pixel_id
forcing_member_id
validity
```

Store at approximately 15–30 minute checkpoints (configurable). Start small; do not optimize for
the full particle/ensemble count yet.

**Check:** runtime, record counts match expectations, and diagnostics
(`total_particles, valid_particles, invalid_particles, total_weight, effective_sample_size,
valid_particle_fraction`) are computed and look sane on this small test case.

**Progress report:** run summary table + diagnostics.

---

## T10 — Stage III diagnostics module
**Status:** DONE

Implement reusable functions for:

```text
total_particles
valid_particles
invalid_particles
total_weight
effective_sample_size
valid_particle_fraction
```

**Check:** hand-computable synthetic weight/validity arrays produce the expected diagnostic
values.

**Progress report:** synthetic input + expected vs actual diagnostic values.

---

## T11 — Memory-efficient lookup-table storage
**Status:** DONE

Implement compact storage/indexing (e.g. array-based or spatial hashing) for the per-checkpoint
particle records across pixels and forcing members, per `README.md`'s "use a memory-efficient
storage/indexing strategy." Do not duplicate the environmental fields inside the particle
history.

**Check:** round-trip storage/retrieval correctness on the T09 small run; note memory footprint.

**Progress report:** storage layout description + round-trip check result.

---

## T12 — Stage III controlled numerical validation suite
**Status:** DONE

Build the controlled tests explicitly requested in `README.md` / `ISTE_Claude_Architecture_v1.0.txt`:

```text
zero divergence
known convergent flow
known divergent flow
constant diffusivity
spatially varying diffusivity
particle validity
particle-weight behavior
```

**Check:** each controlled test's numerical result matches its analytically expected behavior —
not merely "the code executes."

**Progress report:** one result row per controlled test (expected vs actual).

---

## T13 — Stage III sign-off artifact
**Status:** DONE — **Stage III is signed off (2026-09-22).**

> **2026-09-22 — interface-integration gap fixed, then approved.** The first T13 review
> found that `iste.environment.NetCDFEnvironmentalFieldProvider.K`/`grad_K`/`div_U` raised
> `NotImplementedError` even though the live T05 pipeline already computed the same
> quantities internally. Fixed by extracting two shared helper functions
> (`iste.environment.interpolated_divergence`, `interpolated_k_value_and_gradient`) that
> both the provider and `iste.step.backward_euler_maruyama_step` now call — no Stage III
> equation changed, no T05/T06 test's hand-computed expectation changed. See
> `progress.md`'s T13 section and the sign-off notebook's §E for the full write-up. You
> approved the corrected notebook and signed off Stage III; Stage IV work may now begin
> (see T14 below).

Produce a notebook that shows, on controlled and small real-data cases:

- sample particle trajectories with correct backward-time direction;
- weight evolution and its relationship to flow divergence;
- validity/boundary handling;
- a small-scale lookup-table sample (`c_hat*_{j,e}(x,t)`), explicitly labeled as a Monte Carlo
  transport estimate, not an exact probability or proof of source.

**Check:** manual inspection confirms the backward-time sign convention, transport direction,
weight behavior, and diagnostics are all correct, and that no language overstates the result as a
validated or confirmed source.

**Progress report:** final notebook + one-paragraph interpretation, explicitly noting this is the
Stage III sign-off gate.

This was the **Stage III sign-off** required by `README.md`/`CLAUDE.md` before any Stage IV
work could begin. **Approved 2026-09-22 — Stage IV work may now proceed, one task at a time.**

---

## T13.5 — Stage III → Stage IV compatibility correction
**Status:** DONE — approved 2026-09-22

A frozen authoritative handoff spec, `final_stage_III_IV_specification.txt`, was supplied
after T13/T14/T15 were already approved/implemented. T13.5 is a compatibility/contract
verification pass between the completed, signed-off Stage III implementation and that
frozen contract — **not** a Stage III redesign and **not** T14/T15 work. Per the frozen
contract's requirements, this task:

- added explicit particle death metadata (`death_time`, `death_lat`, `death_lon`,
  `death_reason` ∈ `{"land", "domain_exit", "physics_unavailable",
  "already_invalid_upstream"}`) and a `survival_mark` (`1`/`0`, mirrors `valid`) to
  `iste.step.WeightedParticleState`, `iste.trajectory.BackwardTrajectory`,
  `iste.ensemble_run.EnsembleRunRecords`, and `iste.lookup_storage.ParticleRecordStore`
  — all additive, all backward compatible (new fields default to `None`/sentinels; every
  pre-T13.5 test passes unmodified);
- found and fixed a genuine contract mismatch: the weight update's `div(U)` term was
  actually computing `div(U_e)` (a forcing-member-perturbed divergence) whenever a
  particle was advected through T09's per-member perturbed source, not `div(U)` (the
  base current) as the frozen contract's Sec. 5 requires. Fixed via a new, optional
  `divergence_source` parameter on `iste.step.backward_euler_maruyama_step`/
  `iste.trajectory.run_backward_trajectory` (defaults to `None` = old behavior; T09's
  `run_ensemble` now passes the base field explicitly);
- verified (no code change needed) that the `K(x,t)` interface already correctly accepts
  a time argument and returns a constant value across timestamps, per the frozen
  contract's Sec. 3 requirement that a constant `K` still be expressed as a valid
  function of time;
- verified (no code change needed) that forcing-member coherence, existing boundary/
  invalidation semantics, and existing checkpoint scheduling already match the frozen
  contract and were left untouched.

See `progress.md`'s T13.5 section for the full before/after write-up, files
inspected/changed, and exact test results.

**Does NOT implement:** T14/T15 changes, Stage V/VI/VII, `c_hat*`, `h`, `pi`, likelihood,
KDE, SMC, MCMC, or AIS.

---

## Phase 2 — Candidate generation + forward validation

> **2026-09-22 — LOCKED Phase 2 design.**
>
> T00–T13.5 remain unchanged. Stage III weighted backward transport is complete and
> signed off. Phase 2 is the current implementation layer between Stage III and the
> later formal source-inference stage.
>
> Phase 2 has two distinct jobs:
>
> 1. **Candidate discovery:** use historical Stage III particle support to find KDE/core
>    candidate source regions at every configured historical timestamp.
> 2. **Candidate validation/filtering:** forward-propagate a representative particle set
>    from every retained candidate/time to `t_obs` and use agreement with the actual
>    satellite slick as the primary filtering/ranking signal.
>
> The locked flow is:
>
> ```text
> STAGE III
>     weighted backward particle histories
>             ↓
>     historical timestamp t_s
>             ↓
>     KD-tree neighborhoods
>             ↓
>     density + weighted support + cross-pixel diversity
>             ↓
>     KDE / support heat map
>             ↓
>     local maxima / candidate cores
>             ↓
>     for every candidate/time:
>         Y = valid particles inside candidate core
>         N_forward = min(X, Y)
>             ↓
>     weighted core sampling / selection
>             ↓
>     FORWARD PROPAGATION under coherent forcing member U_e
>             ↓
>     predicted slick at t_obs
>             ↓
>     compare with actual satellite slick
>             ↓
>     forward error / consistency score
>             ↓
>     PRIMARY candidate filtering / ranking
>             ↓
>     Phase 2 candidate set
>             ↓
>     later formal source inference
> ```
>
> **Important locked rule:** a candidate with fewer than `X` available core particles is
> **not rejected merely because it has fewer particles**. Use `N_forward = min(X, Y)`.
>
> The candidate core is a Phase 2 screening representation. It is not automatically the
> formal physical source distribution. The later formal source model remains downstream.

### Global Phase 2 rules

1. Work on exactly one task per turn.
2. Preserve Stage III particle trajectories, weights, forcing-member coherence, validity,
   death semantics, and `c_hat*` definition.
3. Historical particles are possible transport states, not historical observations.
4. Preserve `observation_pixel_id` and `forcing_member_id` throughout Phase 2.
5. Invalid particles contribute nothing after their exact death time.
6. Candidate-support values, KDE values, and forward-error values are diagnostics/screening
   quantities, not posterior probabilities.
7. Do not rerun Stage III backward transport for every candidate.
8. Do not independently redraw forcing per particle or timestep. One forcing member must
   remain coherent throughout a forward run.
9. The actual satellite observation at `t_obs` is the independent validation reference.
10. Every new scientific parameter must be explicit/configurable.
11. Phase 2 forward propagation is a **new explicitly documented prototype model**. It must
    not be silently described as the mathematical inverse of the backward adjoint.
12. Phase 2 remains modular so that candidate-generation or forward-validation details can
    later be calibrated without rewriting Stage III.

---

# Stage IV — KDE / support-map candidate generation

### T14 — Historical particle slice and Phase 2 input adapter
**Status:** PENDING_REVIEW

For every configured historical timestamp `t_s`, retrieve all valid Stage III particle
states at that timestamp.

The query layer must:

- locate the bracketing Stage III checkpoints;
- linearly interpolate particle positions;
- linearly interpolate particle log-weights;
- exclude particles at/after exact death time;
- preserve `particle_id`;
- preserve `observation_pixel_id`;
- preserve `forcing_member_id`;
- preserve weight and death metadata.

Do not rerun Stage III.

**Check:** synthetic multi-pixel/multi-member lookup reproduces the expected valid slice
with exact identifiers and weights.

**Progress report:** timestamp, counts, and representative particle table.

---

### T15 — KD-tree neighborhood engine
**Status:** PENDING_REVIEW

For each historical timestamp, build a KD-tree over valid particle positions.

Requirements:

- use a distance-safe local/projected coordinate representation;
- support configurable fixed-radius or k-nearest-neighbor queries;
- return complete particle metadata;
- never discard observation-pixel or forcing-member identity.

The KD-tree is an acceleration structure, not the statistical inference itself.

**Check:** compare KD-tree neighbors against brute-force neighbors on controlled point clouds.

**Progress report:** exact neighbor agreement and runtime comparison.

---

### T16 — Weighted KDE / support-feature construction
**Status:** PENDING_REVIEW

At every historical timestamp, construct three spatial support features:

```text
D(x,y,t) = local particle count / density
W(x,y,t) = sum of Stage III particle weights
V(x,y,t) = number of distinct observation pixels represented
```

Use valid particles only.

The Stage III weights remain unnormalized transport/Jacobian-type weights. They are not
probabilities that a particle is correct.

Combine the three features using an explicit configurable support function:

```text
Support(x,y,t) = f(D, W, V)
```

Feature scaling must be explicit so numerical units do not accidentally determine the
ranking.

The output is a Phase 2 candidate-support heat map, not `c_hat*`, not a likelihood, and
not a source posterior.

**Check:** synthetic regions with independently varied density, weight, and pixel diversity
produce the expected support ordering.

**Progress report:** D/W/V maps plus combined support map.

---

### T17 — Candidate peaks and candidate-core extraction
**Status:** PENDING_REVIEW

At every historical timestamp:

1. detect local maxima of the support heat map;
2. suppress near-duplicate maxima;
3. retain multiple spatial modes;
4. attach the historical timestamp to every candidate;
5. define a high-support candidate core around each maximum.

The entire KDE is not automatically treated as the source.

Each candidate stores at minimum:

```text
candidate_id
x_peak
y_peak
t_s
core_region
support_score
density_score
weighted_support_score
pixel_diversity_score
```

**Check:** a multimodal synthetic cloud yields multiple candidate cores and does not collapse
to a centroid.

**Progress report:** annotated support heat maps with candidate peaks and core boundaries.
Manual-check artifact: `notebooks/T17_candidate_extraction_check.ipynb` (added; awaiting
manual review — see progress.md "Phase 2 — T17" section).

---

# Stage V — Core sampling for forward validation

### T18 — Define and implement `N_forward = min(X, Y)`
**Status:** PENDING_REVIEW

For every candidate core at every historical timestamp, define:

```text
X = configured particle budget representing the current slick
Y = number of valid Stage III particles inside the candidate core
N_forward = min(X, Y)
```

This rule is locked.

Consequences:

```text
Y > X  → select X particles
Y = X  → use X particles
Y < X  → use all Y particles
```

A candidate must **not** be rejected simply because `Y < X`.

If `Y = 0`, the candidate cannot be forward-validated and is marked invalid for Phase 2.

`X` is a computational particle budget, not a physical amount of oil.

**Check:** synthetic candidates with core sizes both above and below X produce exactly
`min(X,Y)` forward particles, with no low-Y candidate silently discarded.

**Progress report:** candidate/core size versus selected forward particle count.
Implemented in `iste/forward_budget.py`; see progress.md "Phase 2 — T18" section
(awaiting manual review).

---

### T19 — Weighted core-particle sampling
**Status:** PENDING_REVIEW

When `Y >= X`, select `X` particles from the candidate core using the available Stage III
particle weights as sampling weights, without replacement by default.

When `Y < X`, use all `Y` core particles.

The selected particles must preserve:

```text
position
observation_pixel_id
forcing_member_id
particle_id
original Stage III weight
```

To avoid double-counting Stage III weights:

- if `Y >= X`, the selected sample is treated as an empirical representative sample and
  the forward predicted KDE uses equal computational particle weight unless the forward
  scoring configuration explicitly specifies otherwise;
- if `Y < X`, all core particles are retained and their relative Stage III weights may be
  normalized **only for constructing the forward predicted KDE**, not fed back into Stage III.

The selection RNG must be reproducible.

**Check:** same seed produces identical selections; different seeds can produce different
samples; high-weight particles receive higher selection probability.

**Progress report:** source-core plot showing selected versus unselected particles.
Implemented in `iste/core_sampling.py`; manual-check artifact:
`notebooks/T19_core_sampling_check.ipynb`. See progress.md "Phase 2 — T19" section
(awaiting manual review).

---

### T20 — Lock and implement the Phase 2 forward transport operator
**Status:** PENDING_REVIEW

This task explicitly defines the new forward-validation model. It is **not** a silent
reuse of the Stage III backward adjoint.

For forcing member `e`:

```text
U_e(x,t) = U(x,t) + delta_U_e(x,t)
```

where `U` and the coherent forcing perturbation are the existing Stage III v1 fields.

The Phase 2 forward stochastic transport prototype is:

```text
X_new = X_old
        + U_e(X_old,t) * dt
        + sqrt(2*K(X_old,t)*dt) * xi

xi ~ N(0, I_2)
```

The forward prototype therefore uses:

- forward physical advection by `+U_e`;
- the configured Stage III diffusivity `K`;
- stochastic diffusion `sqrt(2*K*dt)`;
- no `grad(K)` drift term in the forward prototype;
- explicit metre-to-lat/lon conversion;
- the same land/domain validity policy;
- the same reproducibility discipline.

This forward equation is a **Phase 2 prototype assumption**, not a claim that it is the exact
mathematical inverse of the Stage III adjoint. It must be labeled separately from the frozen
Stage III backward equation.

**Check:** controlled constant-flow cases have the expected forward displacement; zero-current
cases have zero advective displacement; diffusion statistics match the configured K; fixed seed
is reproducible.

**Progress report:** forward operator equations, synthetic validation results, and explicit
units/sign convention. Implemented in `iste/forward_transport.py`; see progress.md
"Phase 2 — T20" section (awaiting manual review).

---

# Stage VI — Forward propagation and predicted slick generation

### T21 — Forward-propagate every candidate/time under the forcing ensemble
**Status:** PENDING_REVIEW

For **every candidate core at every historical timestamp that reaches T21**, forward-propagate
its `N_forward` particles from `t_s` to `t_obs`.

For validation, use each configured forcing member independently:

```text
Candidate C at t_s
   |
   +-- U_1 → predicted slick_1
   +-- U_2 → predicted slick_2
   +-- ...
   +-- U_E → predicted slick_E
```

Each run must keep its forcing member coherent for the entire trajectory.

Do not generate a new forcing realization for each particle or timestep.

Development runs may use an explicit subset of forcing members, but the subset must be stored
with the result.

Apply land/domain invalidation during the forward run and stop at exactly `t_obs`.

**Check:** controlled forward cases, reproducibility, correct forcing-member identity, valid
termination at `t_obs`, and correct invalidation behavior.

**Progress report:** forward trajectory plots for representative candidate/time/member tuples.
Implemented in `iste/forward_propagation.py`; manual-check artifact:
`notebooks/T21_forward_propagation_check.ipynb`. See progress.md "Phase 2 — T21" section
(awaiting manual review).

---

### T22 — Build the predicted slick KDE and observed-slick reference KDE
**Status:** PENDING_REVIEW

At `t_obs`, construct a predicted slick KDE/particle-footprint representation for every
candidate and forcing member.

Also construct the actual-observation reference representation from the upstream observation
particles at `t_obs`.

The comparison objects are therefore:

```text
predicted slick KDE_e(t_obs)
          vs
observed slick particle KDE(t_obs)
```

The predicted KDE is the simulated slick footprint. It is not a source probability map.

Because predicted and observed particle counts can differ, comparison must operate on the
spatial distribution/footprint representation rather than requiring equal particle counts.

**Check:** controlled synthetic releases produce the expected predicted footprint and observed
reference; particle-count changes do not by themselves change a normalized footprint score.

**Progress report:** predicted and observed KDE overlays for representative candidates.
Implemented in `iste/slick_footprint.py`; manual-check artifact:
`notebooks/T22_slick_footprint_check.ipynb`. See progress.md "Phase 2 — T22" section
(awaiting manual review).

---

# Stage VII — Forward consistency, filtering, and Phase 2 handoff

### T23 — Define and implement forward-vs-observed accuracy
**Status:** PENDING_REVIEW

Implement a modular spatial-consistency score comparing predicted slick KDEs with the actual
observed slick at `t_obs`.

The implementation must retain at least:

```text
per-member forward error
per-member forward consistency
```

The comparison metric is a Phase 2 calibration choice and must be explicit/configurable.
Candidate implementations may use footprint overlap, observed coverage, false-positive area,
spatial distance, or a documented combination.

The score direction must be unambiguous:

```text
lower error = better candidate
higher consistency = better candidate
```

**Check:** hand-built good/bad predicted footprints produce the expected ordering.

**Progress report:** metric definition, examples, and per-member scores.
Implemented in `iste/forward_accuracy.py`; manual-check artifact:
`notebooks/T23_forward_accuracy_check.ipynb`. See progress.md "Phase 2 — T23" section
(awaiting manual review).

---

### T24 — Aggregate forcing-member forward accuracy
**Status:** PENDING_REVIEW

For each candidate/time, aggregate the per-forcing-member forward-validation results into a
single candidate-level forward-accuracy score while retaining all member-level diagnostics.

Phase 2 default aggregation:

```text
candidate_forward_error
    = arithmetic mean of valid per-member forward errors
```

and:

```text
candidate_forward_consistency
    = arithmetic mean of valid per-member consistency scores
```

If a forcing member produces no valid predicted footprint, record the reason. Do not silently
replace it with a good score.

If no forcing member yields a valid forward result, the candidate is rejected as
forward-unvalidated.

This aggregation is a **Phase 2 screening score**, not the later formal forcing-ensemble
likelihood marginalization.

**Check:** synthetic member errors aggregate exactly as specified; invalid members do not
silently become zeros or infinities without an explicit policy.

**Progress report:** candidate table containing all member scores and aggregate score.
Implemented in `iste/candidate_forward_accuracy.py`; manual-check artifact:
`notebooks/T24_candidate_forward_accuracy_check.ipynb`. See progress.md
"Phase 2 — T24" section (awaiting manual review).

---

### T25 — Primary forward-accuracy filtering and candidate ranking
**Status:** PENDING_REVIEW

Use **forward propagation accuracy as the primary candidate-filtering and ranking signal**.

For every candidate/time:

1. retain its backward KDE/support provenance;
2. retain per-member forward results;
3. compute candidate-level forward error/consistency;
4. reject candidates above the configured forward-error threshold, or retain the configured
   best-scoring candidates in top-K mode;
5. rank surviving candidates so that lower forward error is preferred;
6. preserve distinct spatial modes and source times;
7. merge near-duplicate survivors only after forward validation.

The decision logic is therefore:

```text
KDE candidate found
      ↓
forward validation
      ↓
actual-slick agreement
      ↓
forward accuracy
      ↓
PRIMARY filter / ranking
```

Backward support answers:

> "Why was this candidate proposed?"

Forward accuracy answers:

> "Does this candidate reproduce the observed slick well enough to survive?"

Do not silently combine backward support into the forward error unless that becomes an
explicit future design decision.

Do not interpret the forward score as a posterior probability or final source likelihood.

**Check:** synthetic candidate sets rank and filter exactly according to the configured forward
accuracy; weak backward support does not automatically override strong forward performance;
near duplicates consolidate only after validation.

**Progress report:** final ranked candidate table with source time/location, backward support,
per-member forward errors, aggregate forward error, and accept/reject reason.
Implemented in `iste/candidate_screening.py`; manual-check artifact:
`notebooks/T25_candidate_screening_check.ipynb`. See progress.md "Phase 2 — T25"
section (awaiting manual review).

---

### T26 — Phase 2 end-to-end integration and candidate artifact
**Status:** PENDING_REVIEW

Integrate T14–T25 into one reproducible Phase 2 run.

The output must be a machine-readable forward-screened candidate set containing:

```text
candidate_id
x_s
y_s
t_s
support_score
density_score
weighted_support_score
pixel_diversity_score
core_definition
core_particle_count_Y
forward_particle_count_N_forward
sampling_seed
forcing_members_validated
forward_error_by_member
forward_consistency_by_member
candidate_forward_error
candidate_forward_consistency
accepted/rejected
rank
```

The artifact must preserve enough provenance to reproduce candidate discovery, source-particle
selection, forward validation, and filtering.

**Check:** full small-data run is reproducible; serialization/reload preserves all candidate
fields; no accepted candidate lacks forward-validation provenance.

**Progress report:** one human-inspectable notebook plus machine-readable candidate artifact.
Implemented in `iste/phase2_pipeline.py`; manual-check artifact:
`notebooks/T26_phase2_pipeline_check.ipynb`. See progress.md "Phase 2 — T26" section
(awaiting manual review).

---

# Phase 2 Definition of Done

Phase 2 is complete when:

- Stage III particles can be queried at arbitrary configured historical timestamps without
  rerunning backward transport;
- KD-tree neighbor results agree with brute-force reference tests;
- density, Stage III weighted support, and distinct observation-pixel diversity are computed;
- support features are combined into a configurable KDE/support heat map;
- multiple local candidate modes can be retained;
- candidate cores are explicitly defined;
- `N_forward = min(X,Y)` is enforced for every candidate/time;
- candidates with `Y < X` are not rejected solely because they have fewer core particles;
- weighted, reproducible core-particle selection is implemented;
- the Phase 2 forward operator is explicitly implemented as
  `X_new = X_old + U_e dt + sqrt(2Kdt) xi`;
- forward runs use coherent forcing members;
- every retained candidate/time is forward-validated unless removed by an explicit prefilter;
- predicted slick KDEs are constructed at `t_obs`;
- predicted slicks are compared with the actual satellite slick using an explicit metric;
- per-member forward accuracy is aggregated into candidate-level forward accuracy;
- forward-propagation accuracy/error is the primary filtering/ranking signal;
- backward KDE support remains provenance/diagnostic information and is not silently turned
  into a likelihood or posterior;
- distinct candidate modes and source times remain available after filtering;
- a machine-readable Phase 2 candidate artifact exists with complete provenance;
- every major task has automated checks plus a human-inspectable validation artifact.

---

# Current Phase 2 conceptual contract

```text
ACTUAL SATELLITE SLICK at t_obs
        ↓
STAGE III
    weighted backward histories
        ↓
for each historical timestamp t_s
        ↓
    KD-tree
        ↓
    density + weight + pixel diversity
        ↓
    KDE / support heat map
        ↓
    local maxima + candidate cores
        ↓
for every candidate/time
        ↓
    Y = core particles
    N_forward = min(X,Y)
        ↓
    weighted source-core selection
        ↓
    for each forcing member e
        ↓
    forward SDE:
        X_new = X_old + U_e dt + sqrt(2Kdt) xi
        ↓
    predicted slick KDE at t_obs
        ↓
    compare with actual observed slick
        ↓
    per-member forward error
        ↓
    mean across valid members
        ↓
    candidate-level forward accuracy
        ↓
    PRIMARY filtering / ranking
        ↓
FORWARD-SCREENED CANDIDATE SET
        ↓
LATER FORMAL SOURCE INFERENCE
```

The locked Phase 2 principle is:

> **Backward transport discovers where and when candidate origins exist. The KDE/support
> representation defines candidate cores. Up to `X` particles are selected from each core,
> with `N_forward = min(X,Y)`. Those particles are forward-propagated under coherent forcing
> members to `t_obs`, and agreement with the actual satellite slick is the primary signal used
> to filter and rank candidates.**

Phase 2 forward propagation is a prototype forward-transport assumption and is explicitly
separate from the frozen Stage III backward adjoint mathematics.

---

# Phase 3 — Dispersion-aware, persistence-aware source ranking

> **2026-09-24 — added, per explicit user request.** The real-data validation run (see
> `progress.md`'s "Real-data validation notebook" section) surfaced a genuine ranking
> problem: because particles released near `t_obs` are still tightly clustered around the
> observed slick, raw KDE/support magnitude is highest at the *most recent* historical
> timestamps and decays as particles disperse going further back — so a naive "top 10 by
> raw support" ranking (T17/T25's current behavior) is structurally biased toward the
> initial condition, not necessarily toward the true historical source. The user was
> explicit that the fix must **not** be "discard the first day" or "force old timestamps to
> rank higher" — it must make the ranking robust to the natural growth of dispersion/
> uncertainty with backward time, while preserving the complete 73-timestamp (T0-T72,
> 1-hour resolution) window and the existing T14-T26 pipeline unchanged underneath.
>
> This phase is additive: it does not replace or silently reinterpret T17's candidate
> extraction or T25's forward-accuracy filtering. It adds a new, explicitly-scored,
> explicitly-configurable ranking layer on top, and every new statistical choice (the
> initial-condition decay form, the composite-score weights, the persistence-clustering
> radius, etc.) must be recorded as an explicit assumption — per CLAUDE.md's guardrail
> against silently inventing uncertainty models/thresholds/probability interpretations —
> not treated as a validated physical claim.
>
> Work one task at a time, per the project's standing rule. No task below may combine
> backward support directly into T25's forward-accuracy score without that being the
> explicit subject of its own task (T36).

### T27 — Extend real-data analysis grid to 73 hourly timestamps (T0-T72)
**Status:** DONE — approved 2026-09-24 (implicit approval via "proceed with T28")

Storage/query layer only. Do **not** change the Stage III physical integration timestep —
if the existing backward Euler-Maruyama step needs a smaller `dt` for numerical stability,
keep it; only the **stored/query analysis grid** must resolve to exactly 1-hour steps from
`t_obs` back to `t_obs - 72h`.

Requirements:

- a small, reusable helper that generates the analysis timestamp grid
  `T0, T1, ..., T72` (73 timestamps, `T0 = t_obs`, `Tk = t_obs - k*3600s`) given `t_obs` and
  the window/interval as explicit parameters (never hard-coded to this project's specific
  `t_obs`);
- if the physical checkpoint spacing already equals 1 hour (as in the current real-data run,
  `dt=3600s`), the analysis grid coincides with the Stage III checkpoint grid directly, no
  interpolation needed;
- if the physical checkpoint spacing is finer than 1 hour, the analysis grid must reuse
  T14's existing interpolation-based historical query (`query_particle_slice`/equivalent),
  never a new ad hoc interpolation;
- confirm no Stage III module (`iste/step.py`, `iste/trajectory.py`, `iste/ensemble_run.py`)
  is modified.

**Check:** unit test confirming `generate_analysis_grid(t_obs, window_hours=72,
interval_hours=1)` returns exactly 73 timestamps, `grid[0] == t_obs`,
`grid[-1] == t_obs - 72h`, and monotonic 1-hour spacing; confirm it raises/rejects a
window not evenly divisible by the interval rather than silently rounding.

**Progress report:** generated grid printed in full (73 timestamps) + confirmation that
`len(grid) == 73`.

---

### T28 — Per-timestamp diagnostics module
**Status:** DONE — approved 2026-09-24

For every one of the 73 analysis timestamps, compute and store: particle count, lon/lat
mean, lon/lat variance, covariance matrix, spatial spread (e.g. RMS distance from centroid),
convex-hull area (if practical), KDE max, KDE integrated mass, KDE entropy, KDE effective
area, KDE bandwidth used, and number of significant KDE modes (reuse T17's existing peak
detection for the mode count — do not invent a second peak-detection algorithm).

**Check:** synthetic tight-cluster vs. synthetic dispersed-cluster inputs produce the
expected ordering on each diagnostic (dispersed cluster has larger spread/entropy/effective
area, smaller KDE max, tight cluster the reverse).

**Progress report:** the Stage 13 diagnostic table (one row per T0-T72 timestamp) on the
real 3-day dataset already produced by the real-data notebook.

---

### T29 — Cross-time KDE normalization
**Status:** DONE — approved 2026-09-24. **Carry-forward instruction for T36:** the
observed `+0.923` correlation between `hours_before_observation` and `normalized_max` on
the real dataset (see progress.md) must be reported in T36's diagnostics/sensitivity
analysis, not artificially removed or hidden at this stage.

Implement and document one specific percentile-based normalization,
e.g. `normalized_kde(x,y,t) = KDE(x,y,t) / percentile(KDE_t, 95)`, applied independently at
each of the 73 timestamps so KDE magnitude is comparable across timestamps with very
different particle dispersion.

**Check:** synthetic tight vs. dispersed clusters, after normalization, have comparable
normalized-peak magnitude (within a documented tolerance) despite very different raw KDE
peaks — demonstrating the normalization removes the dispersion-driven scale difference this
phase exists to fix.

**Progress report:** raw vs. normalized KDE side-by-side for a sample of T0-T72 timestamps
on the real dataset, plus the exact formula used.

---

### T30 — Initial-condition influence field
**Status:** DONE — approved 2026-09-24

> **2026-09-24 — revised after review.** The original `exp(-d/spatial_scale) *
> exp(-age/temporal_scale)` formulation was rejected: a monotonic-in-age term, subtracted
> in T36, would mechanically favor older timestamps regardless of actual evidence — exactly
> the constraint this phase must respect. Replaced with a Mahalanobis/dispersion-relative
> formulation that contains **no age term anywhere**. See `progress.md`/`memory.md` for the
> full review discussion and the recommended-alternatives analysis.

`d_M(candidate, t) = sqrt((x_c - mu(t))^T @ Sigma(t)^-1 @ (x_c - mu(t)))`,
`influence = exp(-0.5*(d_M/k)^2)`, where `mu(t)`/`Sigma(t)` are the ACTUAL MEASURED mean/
covariance of that timestamp's own valid particles (T14, projected into local metres via
T15), and `k` (default 1.0) is the only free parameter. Distance is measured to the
particle cloud's OWN centroid at `t` (nets out real bulk advection), not to the fixed
observed-slick point. Explicit, tested fallback for singular/near-singular covariance
(isotropic variance) and for `<2` particles (`NaN`, never fabricated). This is a
diagnostic/correction *input* for T36, not a rejection rule — no candidate is dropped here.

**Check:** confirm the field decreases smoothly and monotonically with Mahalanobis distance
(no hard cutoff/step anywhere), confirm no age-based term/branch exists anywhere in the
code, and confirm the singular-covariance/insufficient-particle fallbacks are explicit and
tested (never a silent NaN-as-valid or fabricated value).

**Approved 2026-09-24.** Empirically confirmed the age-correlation went from exactly
`-1.000` (guaranteed by the rejected design's construction) to `-0.497` (a real,
non-guaranteed, data-driven residual) on the real dataset — this residual must be carried
into T36's diagnostics/sensitivity analysis as a diagnostic, not treated as something to
correct for or suppress (per explicit user instruction, recorded in `memory.md`).

**Progress report:** influence field values for a spread of (distance, age) pairs on the
real dataset, plus explicit justification for the chosen `spatial_scale`/`temporal_scale`
defaults (labeled ASSUMED, not validated).

---

### T31 — Temporal persistence tracking
**Status:** DONE — approved 2026-09-24. **Standing caveat (user-specified):**
persistence-track continuity is evidence of CANDIDATE-TRACK continuity (a spatially/
temporally coherent detection sequence), not proof of physical source identity. The
`track0000` validation's monotonic-support-trend observation must NOT be treated as
independent evidence that a track is "physically genuine" — the actual validation basis for
that finding is strictly: consecutive hourly matching, bounded match distances, zero
gap-bridges used, and the 2 km sensitivity analysis (see `progress.md`'s review-validation
section and `memory.md`).

Cluster T17 candidate hotspots spatially within a configurable matching radius, then track
clusters across neighboring hourly timestamps (not exact-pixel matching, since particle
clouds move). Compute per cluster: `persistence_count`, `persistence_duration` (in hours,
since resolution is 1 hour), `mean_support`, `max_support`, `support_variance`,
`temporal_continuity`.

**Check:** a synthetic candidate present at 5 consecutive hourly timestamps scores high
persistence; a synthetic one-time spike at a single timestamp scores low persistence.

**Progress report:** persistence table for the real T0-T72 candidate set. Implemented in
`iste/phase3_persistence.py` (greedy nearest-neighbor tracker, no age term anywhere); see
`progress.md`'s T31 section (awaiting manual review).

---

### T32 — Spatial convergence scoring
**Status:** DONE — approved 2026-09-24. **Recorded limitation (user-specified):** on the
real 280-candidate dataset, `convergence_score` is strongly correlated with raw particle
density (Pearson 0.881, Spearman 0.869), and `pixel_diversity_fraction`/
`member_diversity_fraction` are themselves strongly density-correlated (Spearman
0.999/0.910). The ablation showed only removing `particle_fraction` changes top-10
membership; removing any other single component does not. **T32 must NOT currently be
treated as an evidence source independent from density/concentration in T36** — weights/
implementation were NOT changed; this is a documented dataset-specific finding for T36 to
account for (see `progress.md`'s ablation-diagnostic section and `memory.md`).

For each candidate region: number/fraction of particles entering it, diversity of
originating trajectories/observation pixels (reuse T16's existing pixel-diversity feature,
do not reimplement), spatial concentration, and consistency across neighboring timestamps.
Must be distinguishable from raw KDE concentration (T29) in the test case.

**Check:** synthetic case where multiple independent trajectory groups converge on one
region scores higher convergence than a region reached by one homogeneous group at equal
KDE magnitude.

**Progress report:** convergence scores for the real T0-T72 candidate set. Implemented in
`iste/phase3_convergence.py` (5-component equal-weighted composite: particle_fraction,
T16's own pixel_diversity reused, new member_diversity, concentration via T17's own
core_radius_m, T31's temporal_continuity reused when available); see `progress.md`'s T32
section (awaiting manual review). Same screening-quantity caveat as T31 applies: a high
convergence score is evidence of independent-trajectory agreement, never proof of physical
source identity.

---

### T33 — Dispersion-aware scoring
**Status:** DONE — ACCEPTED WITH MAJOR CAVEAT, approved 2026-09-24 (closed). **Final
recorded conclusion:** the dispersion-aware score shows a strong positive age trend on the
real dataset, and that trend persists — and actually strengthens — under a distribution-
free empirical within-timestamp null (Pearson `0.951`/Spearman `0.943` vs. the Gaussian
score's `0.863`/`0.862`). The trend is therefore not primarily attributable to the Gaussian
null assumption or to covariance anisotropy. This is **strong evidence within this dataset
and this metric**, not an absolute, universally model-independent proof. `iste/
phase3_dispersion.py` was not modified by any of the three review diagnostics — see
`progress.md`/`memory.md` for the full three-part diagnostic record.

Compare each timestamp's actual KDE concentration against the *expected* concentration for
that timestamp's particle dispersion (e.g. `KDE_peak / expected_KDE_peak_given_variance`),
so a lower raw KDE at T72 is not automatically read as weaker evidence than a higher raw KDE
at T1.

**Check:** two synthetic timestamps with equal *relative* concentration-given-dispersion but
very different raw KDE magnitude (because of different particle spread) receive
approximately equal dispersion-aware scores.

**Progress report:** dispersion-aware score vs. raw KDE, side-by-side, across T0-T72 on the
real dataset — the key diagnostic for whether this phase actually fixes the reported bias.

**Implemented in `iste/phase3_dispersion.py` (Rayleigh-CDF-based expected-density model,
reusing T30's covariance unchanged). Real-data result requires review before approval:**
raw density correlates `-0.875` (Pearson) / `-0.998` (Spearman) with age, as always found;
but `dispersion_aware_score` correlates `+0.863` (Pearson) / `+0.862` (Spearman) with
age — comparably strong in MAGNITUDE to the original bias, just flipped in SIGN (crosses
from below the "undifferentiated blob" baseline to above it around hour 34-35, reaching
2.65x by T72). This is a materially different situation from T29/T30/T31/T32's residual
correlations (which were modest and non-guaranteed): here the age-dependence is nearly as
strong as what T33 exists to remove. Two competing, currently unresolved explanations are
recorded in `progress.md`/`memory.md` (genuine physical signal vs. isotropic-Gaussian
model-mismatch artifact) — **not resolved, flagged for explicit review**, not silently
accepted as "the fix works."

---

### T34 — Screening-signal normalization
**Status:** DONE — approved 2026-09-24. Verified by direct source inspection: `reference_value`
sourced only from T29's `normalize_support_grid`; no independent percentile recomputation;
`screening_score == support_score / reference_value` exactly; no age/weighting logic; no
second normalization/rescale. 12/12 focused tests, 618/618 full suite, 280/280 real
candidates scored (0 NaN), 0 ordering violations across 44 real multi-candidate timestamps.

First inspect the existing `Support = f(D,W,V)` implementation (`iste/support_features.py`)
and its actual real-run value distribution (already have this from the real-data notebook).
Normalize it to `[0,1]` (or another data-justified range) using a method chosen from
inspecting that actual distribution — do not assume min-max is appropriate without checking.
Do not modify `iste/support_features.py`/T16 itself.

**Check:** normalized screening score preserves the existing candidate ordering (a
monotonic transform) and falls within the declared range for the real dataset.

**Progress report:** raw vs. normalized screening values for the real T0-T72 candidate set,
plus written justification for the normalization method chosen.

---

### T35 — AIS evidence hook
**Status:** DONE — approved 2026-09-24 (closed, T36 authorized to proceed).

No AIS data currently exists anywhere in this project (confirmed: `data/` contains only
`ocean.nc`, `wind.nc`, `database_payload.json`). Implement the interface/hook only: a
function signature that would compute distance-to-track/vessel-count/duration/compatibility
evidence *if* AIS data were supplied, which returns an explicit `ais_available=False` /
`"AIS unavailable"` sentinel now, and confirm downstream (T36) renormalizes the remaining
components correctly when this sentinel is returned. Do not fabricate or simulate AIS data.

**Check:** calling the hook with no AIS source configured returns the unavailable sentinel
without raising; confirm T36's composite score is well-defined (weights renormalize) both
with and without this component present.

**Progress report:** confirmation that the pipeline runs end-to-end with AIS absent, and the
exact renormalization behavior demonstrated. Implemented in `iste/phase3_ais.py`
(`AISDataSource` abstract interface, `compute_ais_evidence` unavailable-sentinel hook,
`renormalize_weighted_mean` general reusable utility for T36). See `progress.md`'s T35
section for the full write-up (awaiting manual review).

---

### T36 — Composite multi-feature source score
**Status:** DONE — approved 2026-09-24. Implemented exactly per the approved specification + amendments
(`iste/phase3_composite.py`). **Flagged finding requiring review, not yet interpreted as
success or failure:** on the real dataset, the DEFAULT-configuration top-10 has **zero
overlap** with the raw-Support top-10 — both top-10 lists are hourly members of the SAME
single dominant T31 track, raw Support favoring recent hours (1-10h), the composite favoring
old ones (63-72h). **Follow-up member-by-member diagnostic (2026-09-24, no weights
changed):** decomposed the oldest-vs-newest score gap exactly — T33 contributes 74.0%,
T29/T34 (support, NOT flat within this track, contrary to an earlier turn's claim)
contributes 31.9%, T32 partially offsets at -5.9%. Removing T33 entirely (diagnostic weights
only) barely changes the overall age-correlation shape (Spearman `+0.87` vs `+0.90`) — T33
mainly determines whether the single oldest candidate reaches literal rank 1, not the
age-pattern's existence. A fully model-free empirical density ratio shows the same pattern
even more strongly than T33's own Gaussian-based ratio, consistent with T33's own prior
closed empirical-null finding. See `progress.md`'s T36 sections for the full real-data
report and diagnostic; requires your review before any further use.

Combine T29 (normalized KDE), T31 (persistence), T32 (convergence), T33 (dispersion-aware),
T34 (normalized screening), T35 (AIS, if available), and T30 (initial-condition influence,
subtracted) into one configurable weighted `source_score`. All weights explicit,
configurable, defaulted, and documented with a one-line rationale each — not claimed as
validated. Provide a sensitivity check showing how the top-10 ranking changes under at least
2-3 alternative weight sets.

This is the task that combines backward-derived evidence into a single ranking signal; it
does **not** modify T25's separate forward-accuracy filtering/ranking, which remains
independent and is not folded into this score.

**Check:** synthetic case confirms the score is monotonic in each component individually
(holding others fixed); confirm the ranking is not simply reproducing raw-KDE order (i.e.
the fix in T29/T33 actually changes candidate order vs. the T17-only baseline) on the real
dataset.

**Progress report:** default weight configuration + written rationale per weight; the
sensitivity table.

---

### T37 — Spatial clustering of candidates
**Status:** PENDING_REVIEW

Cluster candidate hotspots (post T36 scoring) using a configurable geodesic-distance
threshold (DBSCAN or hierarchical — pick one, document why) so the final output is distinct
regions, not near-duplicate points. Report per cluster: centroid, spatial radius, area,
supporting-particle count, supported timestamps, persistence duration, mean/peak normalized
KDE, convergence score, screening score, AIS score, initial-condition influence, final
source score, and a confidence/uncertainty metric (from T38).

**Check:** synthetic near-duplicate candidates collapse into one cluster; synthetic
well-separated candidates remain distinct.

**Progress report:** final clustered candidate table on the real dataset, replacing the
current "top 10 nearly-identical points" behavior.

Implemented in `iste/phase3_clustering.py` (DBSCAN, `min_samples=1`, precomputed local-
metre distance matrix, `eps_m` default reused from T31's `spatial_matching_radius_m`; one
"screening score" field intentionally NOT separately reported, per T36's own T29/T34
single-axis rule; confidence/uncertainty explicitly deferred to T38, not fabricated). **Real-
data finding requiring awareness (not necessarily a defect):** with `min_samples=1`, DBSCAN's
transitive chaining merged 147 of 280 real candidates — essentially the entire 71-hour
drifting dominant T31 track, whose consecutive hourly steps are each well under `eps_m` —
into ONE cluster spanning `~24.4 km`. This satisfies the task's own literal check (near-
duplicates collapse, well-separated candidates stay distinct) but means the top cluster is
an elongated multi-hour trajectory corridor, not a compact "source area" in the intuitive
sense. See `progress.md`'s T37 section for the full real-data report.

---

### T38 — Uncertainty / bootstrap stability analysis
**Status:** DONE (approved 2026-09-24)

For each candidate cluster, report `uncertainty_radius_km`, `temporal_uncertainty_hours`,
`particle_support_fraction`, `persistence_fraction`. If a bootstrap (resample particles,
recompute cluster scores, measure top-10 stability across resamples) is computationally
practical within this session's runtime budget, implement it and report `candidate
stability` as a bootstrap-derived percentage; otherwise document explicitly why it isn't
practical here and report the non-bootstrap uncertainty measures only — do not claim
bootstrap validation that wasn't actually run.

**Check:** confirm uncertainty widens (not narrows) for candidates supported at older/more
dispersed timestamps, consistent with growing backward-integration uncertainty.

**Progress report:** uncertainty table for the final ranked clusters.

Implemented in `iste/phase3_uncertainty.py`. Reuses T36 `SourceScore`/T37 `CandidateCluster`
output read-only; does not modify T14-T37. Particle identity = `(particle_id,
forcing_member_id)` (T14's own grouping key). `particle_support_fraction` = distinct pooled
neighborhood particles per cluster / distinct particle-member pairs in the whole run.
`persistence_fraction` = `n_distinct_timestamps / 73` (T27's own T0..T72 grid length).
`uncertainty_radius_km` = 90th-percentile (configurable) distance of the SAME pooled
particles from the cluster centroid — explicitly NOT T37's `spatial_radius_m` renamed
(that is peak-position extent only). `temporal_uncertainty_hours` = T36-score-weighted std
of member timestamps — explicitly NOT `temporal_span_hours`. **Bootstrap WAS run** (not
skipped): a full Stage III/whole-grid-per-replicate bootstrap was assessed as impractical
(~62s/candidate-regen pass measured this run; repeating per replicate would cost tens of
minutes to hours) and explicitly rejected; instead each candidate's own already-computed
T16 neighborhood is resampled with replacement (200 replicates, seed 42, top_k=10, ~2.9s
real-data runtime), scoped explicitly to test T29/T34 support-evidence stability only (T31-
T33/T35 held fixed per replicate — documented limitation, not hidden). Real-data run: 10/10
clusters got finite uncertainty metrics, 0 NaN/0 Inf, T36 scores and T37 membership
confirmed byte-for-byte unchanged, all 73 T0-T72 timestamps confirmed present. Part F
diagnostic (10 real clusters, small sample): age vs. `uncertainty_radius_km`
Spearman=0.45, age vs. `temporal_uncertainty_hours` Spearman=0.48 (both moderate, reported
honestly, not tuned), particle-spread vs. `uncertainty_radius_km` Spearman=0.96 (expected,
since `uncertainty_radius_km` is itself a percentile of that same spread — reported as a
consistency check, not independent confirmation). Dominant `cluster0000` (147 members, the
same T37 DBSCAN-chained 71-hour corridor) has `uncertainty_radius_km=25.16` (correctly
reflecting the corridor's real width, not shrunk to look compact) and
`candidate_stability_percent=100.0` (interpreted per the notebook's explicit caveat: some
point along the corridor is consistently top-ranked under resampling, not that the corridor
is one compact source). See `progress.md`'s T38 section for the full real-data table and
known limitations.

---

### T39 — Visualization suite + old-vs-new ranking comparison
**Status:** PENDING_REVIEW

Extends (does not replace) the existing real-data notebook's visualization cells.
Selected-snapshot panels (T0, T0-6h, T0-12h, T0-24h, T0-36h, T0-48h, T0-60h, T0-72h — not
all 73), a persistence heatmap (x=hours before observation, y=candidate cluster,
color=normalized score), a final map (observed slick, particle envelope, top clusters,
uncertainty radii, AIS tracks if available), the required Stage 13 diagnostic table, and a
direct old-method-vs-new-method comparison table (candidate, old_rank, new_rank, old_score,
new_score, support_duration, distance_from_observed, particle_support, AIS_support,
screening_support) run on the same real dataset already produced by the T27-T38 chain.

**Check:** notebook executes end-to-end with zero errors on the real dataset; comparison
table demonstrates concretely what changed vs. the T17-raw-support baseline.

**Progress report:** notebook path, comparison table, and a written explanation of why
early timestamps are no longer allowed to dominate purely from KDE concentration.

Implemented in `notebooks/T39_visualization_comparison.ipynb` only — no new `.py` module
(pure orchestration/plotting over already-approved T14-T38 output, per CLAUDE.md's own
".ipynb = experiments, plots, debugging, manual validation" rule). Extends, does not
replace, `main_actual_data_pipeline.ipynb`'s existing hotspot-heatmap panel pattern and
observed-slick plot. Real-data run: 280 candidates, 10 T37 clusters, all 73 T0-T72
diagnostic-table rows present, zero errors, all 6 validation checks passed. **Old-vs-new
comparison found 0/10 overlap between the old (`raw support_score`) top-10 and new
(`T36 source_score`) top-10** — old `rank=1` is `T0-2h` (`2019-10-14T22:00`,
`support_score=4675.4`), new `rank=1` is `T0-71h` (`2019-10-11T23:00`, `source_score=1.462`,
the same dominant cluster already flagged at T37/T38 review) — a genuine re-ranking, not a
rescaling of the old order. See `progress.md`'s T39 section for the full table and the
written "why early timestamps no longer dominate" explanation.

---

## Phase 3 constraints (apply to every task above)

1. Do not remove or discard any of the 73 T0-T72 analysis timestamps at any stage.
2. Do not hard-code a specific "ignore first N hours" rule anywhere.
3. Do not force older timestamps to score higher by construction.
4. Preserve the Stage III physical integration timestep and equations unchanged.
5. Preserve T14-T26 unchanged; this phase adds a new layer on top, it does not silently
   rewrite T17/T25's existing behavior.
6. Every new numeric parameter (scale, radius, weight, threshold) must be an explicit,
   documented, configurable default — never a silent invented constant.
7. Fail gracefully (explicit "unavailable"/renormalize) when AIS or other optional evidence
   is missing — never crash, never fabricate.
8. Each task gets its own check, manual-check artifact, and `tasks.md`/`progress.md` update,
   per the project's standing one-task-per-turn workflow.
