"""T20 -- Stage VI: the Phase 2 forward transport operator.

Implements exactly `README.md` Sec. 18.2's locked Phase 2 forward-validation SDE, one
Euler-Maruyama step at a time:

    X_new = X_old + U_e(X_old,t) * dt + sqrt(2*K(X_old,t)*dt) * xi,   xi ~ N(0, I_2)

**This is a NEW Phase 2 prototype, deliberately NOT the mathematical inverse of the
frozen Stage III backward adjoint (`iste.step.backward_euler_maruyama_step`).** The two
differ in three ways, all locked by `README.md` Sec. 18.2:

    1. sign: `+U_e` (physical forward advection), never `-U`;
    2. NO `grad(K)` drift term at all (Stage III's backward step includes `+grad(K)`;
       this forward prototype does not);
    3. NO Stage III weight update (`log_w_new = log_w_old - div(U)*dt` is exclusively a
       Stage III/backward concept -- `README.md`'s Phase 2 forward-propagation sections
       (18-25) never mention a weight update during forward propagation at all, so this
       module carries no `log_weight` field and never touches one).

`iste.step.backward_euler_maruyama_step` is not imported, modified, or duplicated by
this module -- the two are structurally similar (same Euler-Maruyama skeleton, same
boundary/RNG/unit-conversion building blocks) because they both build on the SAME
shared, already-tested primitives (`iste.boundary`, `iste.interpolation`,
`iste.geometry`), not because either reuses or wraps the other.

**`U_e(x,t) = U(x,t) + delta_U_e(x,t)`, the existing Stage III coherent
forcing-member representation, is supplied by the CALLER as an already-perturbed
`ocean_source`** -- built via `iste.ensemble_run.make_perturbed_source(ocean_source,
ensemble, member_id)` (T09), exactly the same mechanism T09's own backward ensemble run
already uses to inject `U_e` into `iste.interpolation.interpolate_field` without
duplicating any forcing-ensemble logic. This module reads `uo`/`vo` from whatever
`ocean_source` it is given via `interpolate_field`, precisely as `iste.step` does; it
never calls `iste.forcing_ensemble` directly. `K` is read via the same
`bilinear_spatial_interpolate` primitive `iste.environment.
interpolated_k_value_and_gradient` itself uses internally for its own `k_val` -- but
this module calls that primitive DIRECTLY for `K` alone (not the K+gradient wrapper),
since it has no use for `grad(K)` and does not want its own validity mask polluted by
an unrelated `grad(K)`-availability failure at a point where `K` itself is perfectly
valid.

**`forcing_member_id` on `ForwardParticleState` is each particle's ORIGINAL Stage III
provenance field** (carried over unmodified from T14-T19), preserved through every
forward step purely as a passive bookkeeping value -- it is NOT used to select which
`U_e` this step applies (that is the caller's `ocean_source` choice, applied uniformly
to the whole batch, matching `README.md` Sec. 19: "For each configured forcing member:
Candidate C -> U_1 -> predicted slick_1 -> U_2 -> ... -> U_E -> predicted slick_E" --
ALL selected particles are propagated under EACH member in turn, not each under its own
original member).

Boundary/domain validity reuses `iste.boundary.apply_boundary_policy`/
`netcdf_validity_check` unchanged -- the SAME land/domain policy Stage III uses ("stop
propagation, retain last valid state," "once invalid, always invalid"): an already-
invalid particle is never silently treated as valid, and a particle is never advanced
past a step whose destination or physics inputs are unavailable.

Does not implement multi-step forward trajectories (T21), multi-member/candidate
orchestration, predicted-slick/KDE construction, or forward comparison metrics -- this
module is the elementary one-step operator those later tasks will loop and orchestrate,
exactly as `iste.step.backward_euler_maruyama_step` is Stage III's own elementary
one-step operator for `iste.trajectory.run_backward_trajectory` to loop.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from iste.boundary import ParticleState, apply_boundary_policy, netcdf_validity_check
from iste.environment import GriddedFieldSource
from iste.geometry import MEAN_EARTH_RADIUS_M, metres_to_degrees
from iste.interpolation import bilinear_spatial_interpolate, interpolate_field


@dataclass
class ForwardParticleState:
    """T20 particle state: position + validity (T04's `ParticleState` shape) plus each
    particle's ORIGINAL Stage III `forcing_member_id` (provenance only -- see module
    docstring). Deliberately carries no `log_weight`: the Phase 2 forward prototype has
    no weight-update term.
    """

    lat: np.ndarray
    lon: np.ndarray
    valid: np.ndarray
    forcing_member_id: np.ndarray

    @classmethod
    def initialize(cls, lat0, lon0, forcing_member_id0, valid0=None) -> "ForwardParticleState":
        """Create a new state at given starting positions. `forcing_member_id0` is
        required (no default) -- it must come from the particles' own Stage III record
        (e.g. `iste.core_sampling.SampledCoreParticles.forcing_member_id`), never
        invented here. `valid0` defaults to all-valid; pass explicitly to seed a state
        with some particles already invalid (never silently treated as valid)."""
        base = ParticleState.initialize(lat0, lon0)
        forcing_member_id0 = np.broadcast_to(
            np.atleast_1d(np.asarray(forcing_member_id0)), base.lat.shape
        ).copy()
        if valid0 is not None:
            valid0 = np.broadcast_to(np.atleast_1d(np.asarray(valid0, dtype=bool)), base.lat.shape).copy()
        else:
            valid0 = base.valid
        return cls(lat=base.lat, lon=base.lon, valid=valid0, forcing_member_id=forcing_member_id0)

    def as_particle_state(self) -> ParticleState:
        return ParticleState(lat=self.lat, lon=self.lon, valid=self.valid)

    @property
    def n_particles(self) -> int:
        return self.lat.shape[0]


def forward_euler_maruyama_step(
    state: ForwardParticleState,
    t,
    dt_seconds: float,
    ocean_source: GriddedFieldSource,
    k_lat: np.ndarray,
    k_lon: np.ndarray,
    k_field: np.ndarray,
    rng: np.random.Generator,
    domain_variable: str = "uo",
    radius_m: float = MEAN_EARTH_RADIUS_M,
) -> ForwardParticleState:
    """One Phase 2 forward Euler-Maruyama step (`README.md` Sec. 18.2), applied to every
    particle in `state` independently. See the module docstring for the equation and the
    sign/term differences from Stage III's backward step.

    `ocean_source` must already represent `U_e(x,t) = U(x,t) + delta_U_e(x,t)` for
    whichever forcing member this call is propagating under -- typically
    `iste.ensemble_run.make_perturbed_source(base_ocean_source, ensemble, member_id)`
    (T08/T09), unmodified. This function itself never selects or looks up a forcing
    member; it only reads `uo`/`vo` from whatever source it is given, exactly like
    `iste.step.backward_euler_maruyama_step` does for its own (differently-signed)
    transport term.

    `t` is the current forward-time instant (a `numpy.datetime64`) this step starts
    from, used only to look up `ocean_source`'s hourly snapshots -- it is not itself
    advanced by this function; a future multi-step loop (T21) steps `t` externally.
    `dt_seconds` must be `> 0` (an increment of forward time `t`, the opposite
    convention from Stage III's backward-time `theta` increment, but the same sign
    constraint).

    `K` has no defined physical formula (same open decision as Stage III, T01.5/T03);
    `k_lat`/`k_lon`/`k_field` must be supplied explicitly by the caller, exactly as
    `iste.step` requires -- this function never invents a `K` value.

    Already-invalid particles are returned completely unchanged (never silently treated
    as valid, matching `iste.boundary`'s own policy). A still-valid particle whose
    current position has unavailable physics (`U_e` or `K` not defined there) is marked
    invalid this step and kept at its pre-step position -- the same "stop propagation,
    retain last valid state" behavior as an invalid destination. `forcing_member_id` is
    always preserved unchanged for every particle, valid or not.
    """
    if dt_seconds <= 0:
        raise ValueError("dt_seconds must be > 0 (an increment of forward time t)")

    n = state.n_particles
    still_active = state.valid

    proposed_lat = state.lat.copy()
    proposed_lon = state.lon.copy()
    physics_ok = np.ones(n, dtype=bool)

    if np.any(still_active):
        lat_a = state.lat[still_active]
        lon_a = state.lon[still_active]

        u_val, u_valid = interpolate_field(ocean_source, "uo", lat_a, lon_a, t)
        v_val, v_valid = interpolate_field(ocean_source, "vo", lat_a, lon_a, t)
        k_val, k_valid = bilinear_spatial_interpolate(k_lat, k_lon, k_field, lat_a, lon_a)

        ok = u_valid & v_valid & k_valid

        xi = rng.standard_normal(size=(len(lat_a), 2))

        # +U_e, times dt: forward physical advection (never -U_e). No grad(K) term.
        drift_x_m = np.where(ok, u_val * dt_seconds, 0.0)
        drift_y_m = np.where(ok, v_val * dt_seconds, 0.0)
        diffusion_scale = np.where(ok, np.sqrt(np.maximum(2.0 * k_val, 0.0) * dt_seconds), 0.0)
        dx_m = drift_x_m + diffusion_scale * xi[:, 0]
        dy_m = drift_y_m + diffusion_scale * xi[:, 1]

        dlon_deg, dlat_deg = metres_to_degrees(dx_m, dy_m, lat_a, radius_m=radius_m)

        proposed_lat_active = np.where(ok, lat_a + dlat_deg, lat_a)
        proposed_lon_active = np.where(ok, lon_a + dlon_deg, lon_a)

        proposed_lat[still_active] = proposed_lat_active
        proposed_lon[still_active] = proposed_lon_active
        physics_ok[still_active] = ok

    domain_check = netcdf_validity_check(ocean_source, domain_variable, t)
    position_before = state.as_particle_state()
    position_after = apply_boundary_policy(position_before, proposed_lat, proposed_lon, domain_check)

    final_valid = position_after.valid & physics_ok

    return ForwardParticleState(
        lat=position_after.lat,
        lon=position_after.lon,
        valid=final_valid,
        forcing_member_id=state.forcing_member_id.copy(),
    )
