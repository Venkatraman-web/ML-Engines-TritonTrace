# SIH 2026 — Stage III + Phase 2 Candidate-Generation Contract

## Purpose

This README is the authoritative implementation contract for the completed **Stage III weighted backward transport** and the current **Phase 2 candidate-generation + forward-validation/filtering** pipeline.

Preserve the Stage III mathematics and terminology below. Phase 2 adds the explicitly defined KDE/KD-tree candidate-generation and forward-accuracy screening procedure. Do not silently substitute a different backtracking or candidate-ranking formulation.

`README_stage3_stage4_frozen.md` is the scientific source of truth for Stage III–IV. `CLAUDE.md` controls how Claude Code works.

---

## 1. Project boundary

The upstream teammate/module already owns:

- satellite-image segmentation/classification;
- detected-oil region / clean-background extraction;
- creation of the initial observation particles.

This repository **must not duplicate those functions**.

Our component starts with already-generated observation particles and performs:

```text
upstream observation particles
        ↓
Stage III weighted backward transport
        ↓
Monte Carlo transport estimate c_hat*_{j,e}(x,t)
        ↓
Stage IV candidate-source likelihood
        ↓
source inference (later stage)
```

Stage IV is a likelihood/scoring component. It does not itself produce the final posterior mode unless a later inference stage consumes it.

---

# 2. Stage III — Weighted Backward Monte Carlo Transport

## 2.1 Goal

For every observed satellite/receptor pixel `j` and forcing member `e`, propagate multiple Monte Carlo particles backward from the single satellite observation time `t_obs` and estimate:

```text
c_hat*_{j,e}(x,t)
```

This is a **Monte Carlo transport estimate**.

It is NOT automatically:

- an exact probability;
- a normalized probability density;
- a likelihood;
- an oil concentration;
- a historical satellite observation;
- a final source probability;
- a final source posterior.

## 2.2 Observation pixel and particle multiplicity

`j` identifies an observation/receptor pixel `A_j` in the satellite image.

One observation pixel produces **multiple** Monte Carlo particles. For fixed `(j,e,t)`, those particles form a weighted cloud:

```text
pixel j, forcing member e
    ├── particle 1: X1, w1
    ├── particle 2: X2, w2
    ├── particle 3: X3, w3
    └── ...
```

Do not collapse `one pixel = one point`.

The particle cloud is what is used to construct `c_hat*`.

## 2.3 Backward history interpretation

The only actual satellite observation in the prototype is at:

```text
t_obs
```

A backward particle at an earlier time `t < t_obs` is a **possible historical transport state**. It is not an observation that oil was actually seen there.

Backward time convention:

```text
theta = t_obs - t
```

A positive simulation step `dt` therefore moves from `t` to `t - dt`.

---

## 3. Stage III environmental interface

The transport layer must consume an environment interface exposing:

```text
U(x,t)
K(x,t)
grad K(x,t)
div U(x,t)
validity(x,t)
```

Spatial and temporal interpolation are separate concerns.

Never treat latitude/longitude degrees as metres. Physical displacement and kernel distances must use metres, with explicit geographic conversion.

### Locked transport velocity for Stage III v1

```text
U(x,t) = (uo(x,t), vo(x,t))
```

Windage is excluded from Stage III v1. `u10/v10` are not discarded from the project; they are simply not part of `U` in this prototype.

The Stage III backward drift uses the sign written below. Do not silently switch to forward transport.

---

## 4. Stage III forcing ensemble

The current project does not have a compatible historical hourly ocean-current ensemble, so Stage III v1 uses a **surrogate forcing ensemble**.

For forcing member `e`:

```text
U_e(x,t) = U(x,t) + delta_U_e(x,t)
```

Prototype perturbation construction:

1. coarse-grid random values;
2. Ornstein–Uhlenbeck temporal evolution per coarse cell;
3. Gaussian spatial smoothing;
4. bilinear interpolation onto the requested field grid.

Prototype values:

```text
spatial correlation target ≈ 20 km
temporal correlation        = 86400 s (~1 day)
sigma_u                     = 0.05 m/s
E                            ≈ 50
```

These values are **PROTOTYPE / NOT VALIDATED**. They are configuration parameters, not established physical truths.

The same forcing member must remain coherent across all particles using that member. Particles must not independently redraw a new current field at each timestep.

Particle Brownian diffusion and forcing-member uncertainty are separate concepts.

---

## 5. Stage III backward SDE step

Use the Euler–Maruyama-style prototype step:

```text
X_new = X_old + (-U + grad(K)) * dt
                + sqrt(2*K*dt) * xi
```

with:

```text
xi ~ N(0, I_2)
```

`dt > 0` is an increment in backward time `theta`.

Physical units must be respected:

- `U`: m/s
- `K`: m²/s
- `grad(K)`: m/s
- `sqrt(2*K*dt)`: m

Then convert metre displacement to lat/lon explicitly.

---

## 6. Stage III particle weights

The frozen prototype weight update is:

```text
log_w_new = log_w_old - div(U) * dt
```

or over multiple steps:

```text
log_w_n = log_w_0 - sum_i div(U(x_i,t_i)) * Delta t
```

Interpretation:

```text
div(U) < 0  → log-weight increases
             → weight increases

div(U) > 0  → log-weight decreases
             → weight decreases

div(U) = 0  → weight unchanged
```

This is a flow/Jacobian-type density correction. It is **not** an observation likelihood and is not directly the probability that the source is correct.

Weights are unnormalized and are used in the `c_hat*` construction below. Do not renormalize them merely because invalid particles were removed.

> Implementation note: the contract uses `U` in the weight equation. Do not silently change the scientific convention while refactoring the code; if a member-specific divergence field is passed internally, it must be reconciled explicitly with the frozen contract rather than altered implicitly.

---

## 7. Stage III invalid particles and boundaries

Particles must never silently cross land/invalid cells.

If a proposed step enters invalid/land or leaves the valid ocean/data domain:

```text
mark invalid
stop propagation
retain invalid status
```

Retain enough information to identify:

```text
death time
death position
death reason code
```

A dead particle contributes nothing after its exact death time.

Where possible distinguish:

- land exit;
- spatial/data-domain exit;
- other invalidity.

Weathering is inactive in the prototype.

---

## 8. Stage III time window and checkpoints

Prototype backward window:

```text
T_w = 48 hours
```

Production target:

```text
T_w = 168 hours
```

Source-time prior:

```text
t_s ~ Uniform[t_obs - T_w, t_obs]
```

A source time outside the window has prior probability zero. Do not describe an out-of-window source as having “zero likelihood”; it is excluded by the source-time prior.

All final source claims are conditional on:

```text
source age <= T_w
```

Checkpoints are approximately 15–30 minutes apart.

The prototype must retain a fixed checkpoint schedule independent of particle validity. Invalid trajectories are not truncated out of the timeline.

For resumable production runs, preserve enough state for continuation, including position, weight, validity, any Markov turbulent state if used, and RNG state.

---

## 9. Stage III Monte Carlo transport estimate

For observation pixel `j`, forcing member `e`, location `x`, and time `t`:

```text
c_hat*_{j,e}(x,t)
  = (|A_j| / N_j)
    * sum_{p in (j,e)}
        w_p(t) * K_b(x - X_p*(t))
```

where:

```text
A_j       = observation/receptor pixel footprint
|A_j|     = geodesic pixel area in m²
N_j       = number of released particles for pixel j
w_p(t)    = unnormalized particle weight
X_p*(t)   = backward particle position
K_b       = fixed kernel with bandwidth b
```

The resulting `c_hat*` is **dimensionless**.

Land-invalid particles contribute zero. Do not renormalize remaining particles merely because some died.

### Kernel requirements

Kernel distance must be measured in metres.

Kernel shape and bandwidth are open calibration/configuration parameters. They must not be silently invented as new scientific contract values.

---

## 10. Stage III → Phase 2 particle query service

Phase 2 must be able to query the Stage III particle histories for any historical candidate time inside the Stage III time window:

```text
c_hat*_{j,e}(x_s,t_s)
```

for all observation pixels `j` and forcing members `e`.

The query service must:

1. locate the bracketing Stage III checkpoints;
2. linearly interpolate particle positions;
3. linearly interpolate particle log-weights;
4. exclude particles at/after their exact death time;
5. evaluate the fixed kernel at the interpolated position;
6. sum the weighted particle contributions;
7. return the complete `J × E` transport-estimate set.

**Do not interpolate already-aggregated `c_hat*` fields when particle-level interpolation is available.**

**Do not rerun Stage III transport for every candidate source query.**

This reuse property is essential for computational feasibility.

---


# 11. Phase 2 — Candidate generation + forward-accuracy filtering

Phase 2 is the locked implementation layer between Stage III weighted backward transport
and the later formal source-inference stage.

Its two jobs are:

1. **Candidate discovery:** use Stage III historical particle support to identify KDE/core
   candidate source regions at every configured historical timestamp.
2. **Candidate validation/filtering:** forward-propagate a representative particle set from
   every candidate/time to `t_obs` and use agreement with the actual satellite slick as the
   primary filtering and ranking signal.

This does **not** redefine the Stage III mathematics, and it does **not** claim that a Phase 2
score is a posterior probability or final source likelihood.

---

## 11.1 Locked Phase 2 pipeline

```text
STAGE III
    weighted backward particle histories
            ↓
    historical timestamp t_s
            ↓
    KD-tree neighborhoods
            ↓
    density + weighted support + cross-pixel diversity
            ↓
    KDE / support heat map
            ↓
    local maxima / candidate cores
            ↓
    every candidate/time:
        Y = valid particles inside core
        N_forward = min(X,Y)
            ↓
    weighted core sampling / selection
            ↓
    forward propagation under coherent U_e
            ↓
    predicted slick at t_obs
            ↓
    compare with actual satellite slick
            ↓
    forward error / consistency
            ↓
    PRIMARY candidate filtering / ranking
            ↓
    forward-screened Phase 2 candidate set
            ↓
    later formal source inference
```

### Locked rule: `N_forward = min(X,Y)`

Let:

```text
X = configured forward-validation particle budget
Y = number of valid Stage III particles inside the candidate core
```

Then:

```text
N_forward = min(X,Y)
```

Therefore:

```text
Y > X  → use X particles
Y = X  → use X particles
Y < X  → use all Y particles
Y = 0  → candidate cannot be forward-validated
```

A candidate is **not rejected merely because `Y < X`**.

`X` is a computational representation budget. It is not a physical statement about oil mass
or the number of physical oil parcels.

---

## 11.2 Phase 2 conceptual boundary

The KDE/core exists to answer:

> **Where and when should we test a possible source?**

Forward propagation answers:

> **Can that proposed origin reproduce the actual observed slick?**

The later formal source-inference stage answers:

> **What source hypothesis has the formal posterior/likelihood support after applying the
> project's source model and priors?**

The Phase 2 KDE/core is therefore a **screening representation**, not automatically the
formal source distribution.

---

# 12. Historical particle slices

For every configured historical timestamp `t_s`, retrieve the valid Stage III particle states.

The query must:

1. locate bracketing Stage III checkpoints;
2. interpolate particle positions linearly;
3. interpolate particle log-weights linearly;
4. exclude particles at/after their exact death time;
5. preserve `particle_id`;
6. preserve `observation_pixel_id`;
7. preserve `forcing_member_id`;
8. preserve Stage III weight and death metadata.

Do not rerun Stage III for candidate generation.

A backward particle at `t_s` is a possible historical transport state, not a historical satellite
observation.

---

# 13. KD-tree candidate-generation engine

At each historical timestamp, build a KD-tree over the valid particle positions.

The KD-tree is an efficient spatial-neighborhood structure. It is not the source-inference model.

The implementation must use a distance-safe local/projected coordinate representation. Latitude
and longitude degrees must not be treated directly as Euclidean metres.

The neighborhood strategy is configurable:

```text
fixed-radius
or
k-nearest-neighbor
```

The complete metadata of each returned particle must remain available.

---

# 14. Candidate-support features

Each neighborhood has three primary candidate-support signals.

## 14.1 Particle density

```text
D(x,y,t) = number of valid nearby particles
```

This measures spatial concentration of backward trajectories.

## 14.2 Weighted particle support

```text
W(x,y,t) = sum of Stage III particle weights in the neighborhood
```

Stage III weights remain the unnormalized flow/Jacobian-type correction defined by the Stage III
contract. They are **not** probabilities that a particle is correct and are not source probabilities.

## 14.3 Cross-pixel diversity

```text
V(x,y,t) = number of distinct observation_pixel_id values represented
           in the neighborhood
```

This distinguishes a region supported by many particles from one observation pixel from a region
supported by many different observed pixels.

The observation-pixel identity must remain attached to every particle through this entire stage.

---

# 15. KDE / support heat map

For each historical timestamp, construct a candidate-support heat map from:

```text
D(x,y,t) = density
W(x,y,t) = weighted support
V(x,y,t) = cross-pixel diversity
```

The combined support is:

```text
Support(x,y,t) = f(D,W,V)
```

where `f` and all feature scaling/normalization are explicit Phase 2 configuration choices.

The heat map is:

```text
candidate-generation support
```

It is **not**:

```text
Stage III c_hat*
probability density
historical source posterior
observation likelihood
proof of source location
```

High support means that the backward transport histories are concentrated/weighted and supported
by multiple observed pixels. It does not prove that the source was there.

---

# 16. Candidate peaks and candidate cores

At every historical timestamp:

1. identify local maxima of the support heat map;
2. suppress near-duplicate maxima;
3. retain multiple spatial modes;
4. attach timestamp `t_s` to each candidate;
5. define a high-support candidate core around each maximum.

The **entire KDE is not automatically treated as the source**.

The candidate core is the high-support region used to select representative particles for forward
validation.

Each candidate stores:

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

---

# 17. Core sampling for forward validation

Let:

```text
X = configured forward-validation particle budget
Y = number of valid Stage III particles inside the candidate core
```

Use:

```text
N_forward = min(X,Y)
```

When `Y >= X`, select `X` particles using the Stage III weights as sampling weights, without
replacement by default.

When `Y < X`, use all `Y` available core particles. Never reject the candidate merely because
`Y < X`.

The selection must preserve:

```text
particle_id
observation_pixel_id
forcing_member_id
position
original Stage III weight
```

The sampling RNG must be reproducible.

To avoid double-counting Stage III weights, a weighted resampled set (`Y >= X`) is represented
by equal computational forward weights by default. When `Y < X` and all original particles are
retained, their relative Stage III weights may be normalized only for the forward predicted KDE.
This does not modify or renormalize the Stage III weights themselves.

---

# 18. Phase 2 forward transport model

## 18.1 Forcing field

For forcing member `e`:

```text
U_e(x,t) = U(x,t) + delta_U_e(x,t)
```

where `U` and the coherent perturbation `delta_U_e` are the already-defined Stage III v1 fields.

The same member field remains coherent throughout the complete forward propagation for all
particles using that member.

## 18.2 Forward equation

Phase 2 uses the following **new forward-transport prototype**:

```text
X_new = X_old
        + U_e(X_old,t) * dt
        + sqrt(2*K(X_old,t)*dt) * xi

xi ~ N(0, I_2)
```

Therefore the forward prototype contains:

- physical forward advection by `+U_e`;
- diffusion controlled by the configured `K`;
- stochastic displacement `sqrt(2*K*dt)`;
- **no `grad(K)` drift term** in the forward prototype;
- explicit metre-to-lat/lon conversion;
- explicit land/domain validity handling;
- reproducible RNG.

This equation is a **Phase 2 prototype assumption**. It is deliberately not described as the
mathematical inverse of the Stage III backward adjoint.

The Stage III backward equation remains frozen and is not changed by this forward model.

---

# 19. Forward propagation

For **every retained KDE candidate at every historical timestamp**, propagate the selected
`N_forward` particles from:

```text
t_s → t_obs
```

For each configured forcing member:

```text
Candidate C
   |
   +-- U_1 → predicted slick_1
   +-- U_2 → predicted slick_2
   +-- ...
   +-- U_E → predicted slick_E
```

A development run may use an explicit subset of forcing members; the selected subset must be
recorded.

Do not redraw forcing independently per particle or timestep.

Apply the Stage III-compatible land/domain invalidation policy during forward propagation.
Stop exactly at `t_obs`.

---

# 20. Predicted slick representation

At `t_obs`, the forward particles form the predicted slick representation.

For every candidate/member retain:

```text
predicted particle positions
predicted validity
forcing_member_id
predicted KDE / footprint
```

The predicted KDE is a representation of the simulated slick footprint. It is not a source
probability map.

Because the candidate sample size may differ from the upstream observed-particle count, the
comparison must operate on the spatial footprint/distribution and must not require equal particle
counts.

---

# 21. Actual observed reference

The independent reference is the actual satellite observation at `t_obs`.

Where upstream observation particles are available, construct the observed reference KDE/footprint
from those observation particles.

The comparison objects are therefore:

```text
predicted slick KDE_e(t_obs)
          vs
observed slick reference KDE/footprint(t_obs)
```

The backward historical particles are not used as the independent truth reference.

---

# 22. Forward-vs-observed accuracy

For each candidate and forcing member, calculate an explicit spatial-consistency error/score
between the predicted slick and the actual observed slick.

The Phase 2 comparison metric is a configurable calibration component. Supported metric families
may include:

```text
footprint overlap
observed-region coverage
predicted false-positive area
spatial distance
or a documented combination
```

The score direction must be explicit:

```text
lower error = better
higher consistency = better
```

Retain both member-level results and the exact metric/configuration used.

---

# 23. Candidate-level forward accuracy

For every candidate/time, aggregate valid forcing-member forward errors using the Phase 2 default:

```text
candidate_forward_error
    = arithmetic mean of valid per-member forward errors
```

and:

```text
candidate_forward_consistency
    = arithmetic mean of valid per-member consistency scores
```

If a forcing member produces no valid forward footprint, record the reason. Do not silently turn
an invalid result into a favorable score.

If no forcing member produces a valid result, the candidate is forward-unvalidated and is rejected.

This is a Phase 2 physical-consistency score. It is **not** the frozen Stage IV likelihood and it
is not the later forcing-ensemble marginalization.

---

# 24. Primary candidate filtering and ranking

**Forward-propagation accuracy is the primary candidate-filtering and ranking signal after KDE
candidate extraction.**

For every candidate/time:

1. keep the backward-support provenance;
2. forward-propagate the candidate under the configured forcing members;
3. compare predicted slicks with the actual satellite slick;
4. compute candidate-level forward accuracy;
5. reject candidates above the configured forward-error threshold, or retain the configured best
   candidates in top-K mode;
6. rank surviving candidates so lower error is better;
7. merge near-duplicates only after forward validation.

Therefore:

```text
KDE candidate found
      ↓
forward propagation
      ↓
predicted slick
      ↓
actual-slick agreement
      ↓
forward accuracy
      ↓
PRIMARY filter / ranking
```

Backward KDE support answers:

> Why was this candidate proposed?

Forward accuracy answers:

> Does this candidate reproduce the observed slick well enough to survive?

Do not silently combine backward support into the forward error.

A candidate with weaker KDE support may outrank a stronger KDE candidate if it has substantially
better forward agreement, subject to any explicit backward prefilter configured before forward
validation.

---

# 25. Phase 2 output artifact

Every candidate must retain enough provenance to reproduce the decision:

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

The final Phase 2 artifact is a **forward-screened candidate set**, not a posterior.

---

# 26. Phase 2 parameters

The following are explicit/configurable and must not be silently treated as universal physical
truths:

```text
historical timestamp grid
KD-tree neighborhood strategy
KD-tree radius / k
KDE bandwidth
support feature scaling
support-combination function
candidate peak threshold
candidate core definition
candidate merge/separation threshold
X forward particle budget
sampling RNG seed/state
forcing-member validation subset
forward dt
forward integrator settings
forward comparison metric
forward error threshold or top-K rule
```

The Stage III forcing parameters remain provisional as already documented.

---

# 27. Important conceptual boundaries

### Phase 2 KDE is not Stage III `c_hat*`

`c_hat*` remains the Stage III per-observation-pixel/per-forcing-member Monte Carlo transport
estimate.

The Phase 2 KDE is an additional aggregate candidate-discovery representation.

### Candidate core is not automatically the formal source

The core is a screening region from which representative particles are selected.

The entire KDE is not declared to be the source.

### Forward validation is not the final posterior

Forward error is a physical-consistency screening quantity.

It is not a posterior probability and not the formal Stage IV surrogate likelihood.

### Historical particles are not observations

The independent observation remains at `t_obs`.

### High density does not prove source

High support may arise from transport compression, diffusivity, forcing structure, boundaries,
or Monte Carlo variation.

Likewise, low forward error is evidence of model-data consistency, not proof of the true source.

---

# 28. Validation philosophy

Every Phase 2 scientific milestone requires:

1. automated numerical tests;
2. controlled synthetic cases with known behavior;
3. a human-inspectable notebook/artifact;
4. explicit separation of controlled validation from real-data sanity checks.

The project must distinguish:

```text
backward KDE support
    ≠
forward consistency
    ≠
formal likelihood
    ≠
posterior probability
```

---

# 29. Locked compact Phase 2 contract

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
    Y = valid particles in core
    N_forward = min(X,Y)
        ↓
    weighted source-core selection
        ↓
    for each forcing member e
        ↓
    FORWARD PROTOTYPE
        X_new = X_old + U_e dt + sqrt(2Kdt) xi
        ↓
    predicted slick at t_obs
        ↓
    compare with actual satellite slick
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

### Locked Phase 2 principle

> **Use backward transport to discover where and when candidate origins exist. Build a KDE/support
> map and extract candidate cores. For every candidate/time, take `N_forward = min(X,Y)` particles,
> with weighted selection when enough core particles exist. Forward-propagate those particles under
> coherent forcing members using the explicit Phase 2 forward transport prototype. Compare the
> predicted slick at `t_obs` with the actual satellite slick. Use forward-propagation accuracy as
> the primary Phase 2 filter and ranking signal.**

Phase 2 forward propagation is explicitly a prototype transport assumption and must remain separate
from the frozen Stage III backward-adjoint mathematics.
# ML-Engines-TritonTrace
