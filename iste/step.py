"""T05 — one Stage III backward Euler-Maruyama particle step (`README.md`).

Implements exactly one configurable backward-time step of the locked Stage III v1 SDE:

    X_new = X_old + (-U + grad(K)) * dt + sqrt(2*K*dt) * xi,   xi ~ N(0, I_2)
    log_w_new = log_w_old - div(U) * dt

with backward time `theta = t0 - t`; `dt_seconds > 0` is an increment of `theta`, so the
deterministic transport contribution is `-U * dt` (never `+U * dt`) -- see `README.md` and
`memory.md`'s Stage III section for the sign convention.

`U(x,t) = (uo(x,t), vo(x,t))` is the T01.5-locked Stage III v1 transport velocity
(windage excluded), evaluated via `iste.interpolation.interpolate_field` against an
`iste.environment.GriddedFieldSource` -- the same full hourly-to-instant space+time
interpolation (T02) already used for `NetCDFEnvironmentalFieldProvider.U`.

`K` has no defined physical formula (open decision, see T01.5/T03); this module never
invents one. Callers supply `K` explicitly as a scalar diffusivity grid
`(k_lat, k_lon, k_field)` at one time instant -- time-interpolation of `K` is out of scope
for T05 (K is currently synthetic/test-only, with no real time-varying source), so this is
an explicit, documented simplification, not a silent one.

`grad(K)` and `div(U)` are computed via `iste.environment.interpolated_k_value_and_gradient`
/`interpolated_divergence` -- small shared helpers (built on T03's
`spatial_gradient`/`divergence_2d` and T02's interpolation) that this module and
`iste.environment.NetCDFEnvironmentalFieldProvider.grad_K`/`div_U` both call, so the
provider-level interface and this live pipeline are guaranteed to compute the same
numbers, never two independent copies of the same math. `div(U)` is computed from the
SAME `uo`/`vo` grid snapshots used for the transport term (the two hourly snapshots
bracketing `t`, blended by the same time-interpolation fraction T02 uses for `U` itself).

Unit handling: `U`, `grad(K)`, and the diffusion term all contribute a metre displacement
once multiplied by `dt_seconds` (`K` in m^2/s makes `grad(K)` m/s, dimensionally consistent
with `U`); the resulting metre displacement is converted to a `(dlat_deg, dlon_deg)` offset
via `iste.geometry.metres_to_degrees`'s documented spherical-Earth conversion (T03) --
metres are never added directly to degree coordinates.

T04's `apply_boundary_policy` is reused unchanged to decide whether the proposed new
position is accepted -- this module does not redefine or duplicate the boundary policy.
Weights follow README.md's rule exactly: `log_w_new = log_w_old - div(U) * dt`, with no
renormalization and no reinterpretation of the weight as automatically conserved
probability mass -- both explicitly out of scope for T05.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from iste.boundary import ParticleState, apply_boundary_policy, netcdf_validity_check
from iste.environment import GriddedFieldSource, interpolated_divergence, interpolated_k_value_and_gradient
from iste.geometry import MEAN_EARTH_RADIUS_M, metres_to_degrees
from iste.interpolation import interpolate_field


@dataclass
class WeightedParticleState:
    """T05 particle state: T04's position/validity plus the dynamic Stage III log-weight.

    Still not the full Stage III particle record (no `observation_pixel_id`/
    `forcing_member_id` yet -- later tasks); T05 adds only what the weight update needs.

    `death_lat`/`death_lon`/`death_reason` (T13.5, added for `final_stage_III_IV_specification.txt`
    compatibility): the exact position and classified reason at which a particle became
    invalid, distinct from `lat`/`lon` (which freeze at the particle's *last valid*
    position, per T04's existing, unchanged policy). `None` for a hand-built state that
    never sets them (backward compatible with any code that only ever supplied
    `lat`/`lon`/`valid`/`log_weight`); `.initialize()` and `backward_euler_maruyama_step`
    always populate real arrays (`NaN`/`""` = "not dead") going forward. See
    `_classify_boundary_death_reason` below for the reason codes.
    """

    lat: np.ndarray
    lon: np.ndarray
    valid: np.ndarray
    log_weight: np.ndarray
    death_lat: np.ndarray | None = None
    death_lon: np.ndarray | None = None
    death_reason: np.ndarray | None = None

    @classmethod
    def initialize(cls, lat0, lon0, log_weight0=None) -> "WeightedParticleState":
        """Create a new, fully-valid state. `log_weight0` defaults to 0 for every
        particle (i.e. weight 1) if not given; a scalar broadcasts to all particles."""
        base = ParticleState.initialize(lat0, lon0)
        if log_weight0 is None:
            log_weight0 = np.zeros(base.n_particles, dtype=np.float64)
        else:
            log_weight0 = np.broadcast_to(
                np.atleast_1d(np.asarray(log_weight0, dtype=np.float64)), base.lat.shape
            ).copy()
        n = base.n_particles
        return cls(
            lat=base.lat, lon=base.lon, valid=base.valid, log_weight=log_weight0,
            death_lat=np.full(n, np.nan), death_lon=np.full(n, np.nan),
            death_reason=np.array([""] * n, dtype=object),
        )

    def as_particle_state(self) -> ParticleState:
        """The T04 position/validity view of this state (drops the log-weight)."""
        return ParticleState(lat=self.lat, lon=self.lon, valid=self.valid)

    @property
    def n_particles(self) -> int:
        return self.lat.shape[0]


def _classify_boundary_death_reason(
    domain_lat: np.ndarray, domain_lon: np.ndarray, lat: np.ndarray, lon: np.ndarray
) -> np.ndarray:
    """T13.5 (`final_stage_III_IV_specification.txt` compatibility): classify a rejected
    proposed position as `"land"` (inside the source grid's own lat/lon coverage, but the
    field data there is masked/missing -- the README's "invalid/land cell" case) or
    `"domain_exit"` (outside the grid's covered extent entirely). This reproduces exactly
    the same `in_range` vs "masked-but-in-range" distinction `iste.interpolation`
    computes internally, without needing to change T02's public API -- a point is
    in-range here iff it lies within `[min(domain_lat), max(domain_lat)] x
    [min(domain_lon), max(domain_lon)]`, identical to `iste.interpolation._bracket`'s own
    `in_range` check for monotonic coordinates."""
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    in_range = (
        (lat >= np.min(domain_lat)) & (lat <= np.max(domain_lat))
        & (lon >= np.min(domain_lon)) & (lon <= np.max(domain_lon))
    )
    return np.where(in_range, "land", "domain_exit").astype(object)


def backward_euler_maruyama_step(
    state: WeightedParticleState,
    t,
    dt_seconds: float,
    ocean_source: GriddedFieldSource,
    k_lat: np.ndarray,
    k_lon: np.ndarray,
    k_field: np.ndarray,
    rng: np.random.Generator,
    domain_variable: str = "uo",
    radius_m: float = MEAN_EARTH_RADIUS_M,
    divergence_source: GriddedFieldSource | None = None,
) -> WeightedParticleState:
    """One Stage III v1 backward Euler-Maruyama step, applied to every particle in
    `state` independently. See the module docstring for the equations and unit handling.

    `t` is the current forward-time instant (a `numpy.datetime64`) this step starts from,
    used only to look up `ocean_source`'s hourly snapshots -- it is not itself advanced by
    this function; callers step `theta`/`t` externally (T06's multi-step loop will do
    this). `dt_seconds` must be `> 0` (an increment of backward time `theta`).

    `domain_variable` selects which `ocean_source` variable defines the land/domain mask
    for T04's boundary policy (`iste.boundary.netcdf_validity_check`); default `"uo"`.
    `rng` must be a `numpy.random.Generator` (e.g. `numpy.random.default_rng(seed)`) --
    the same seed and inputs reproduce the same stochastic step; a different seed
    normally gives a different one.

    `divergence_source` (T13.5, `final_stage_III_IV_specification.txt` Sec. 5): the
    frozen weight equation `log_w_new = log_w_old - div(U) * dt` uses the BASE
    deterministic current `U`, not a forcing-member-perturbed `U_e`, even though
    advection itself uses whatever `ocean_source` the caller passes (which, for an
    ensemble member, is the perturbed `U_e`). Defaults to `None`, meaning "use
    `ocean_source` for divergence too" -- the original, pre-T13.5 behavior, unchanged for
    every caller that only ever has one (unperturbed) source (all of T05/T06's own
    tests). Only T09's `run_ensemble` passes a real, different `divergence_source` (the
    un-perturbed base field), so `U_e` still drives the particle's motion while `div(U)`
    (not `div(U_e)`) still drives its weight, per the frozen contract.

    Already-invalid particles are returned completely unchanged (position, log-weight).
    A still-valid particle whose *current* position has any invalid input (missing `U`,
    missing/out-of-grid `K` or its gradient, or missing `div(U)`) cannot have a step
    computed for it; it is marked invalid this step, with its position and log-weight left
    at their pre-step values -- the same "stop propagation, retain last valid state"
    policy as an invalid destination, just triggered by unavailable physics rather than an
    unavailable destination. A particle whose destination is rejected by
    `apply_boundary_policy` also keeps its pre-step log-weight (the weight update is
    committed together with the move, never applied to a step that did not happen).

    Also records (T13.5), for any particle that becomes invalid THIS step, its exact
    `death_lat`/`death_lon` (the actual invalidating position -- the *proposed*,
    rejected destination for a boundary failure, or the particle's current position for a
    physics-unavailable failure -- deliberately NOT the same as the returned `lat`/`lon`,
    which stay frozen at the particle's last VALID position per T04's existing, unchanged
    policy) and a classified `death_reason` (`"land"`, `"domain_exit"`, or
    `"physics_unavailable"`). Already-dead particles keep their previously-recorded death
    fields unchanged (never overwritten); still-valid particles keep the "not dead"
    sentinel (`NaN`/`""`).
    """
    if dt_seconds <= 0:
        raise ValueError("dt_seconds must be > 0 (an increment of backward time theta)")

    n = state.n_particles
    still_active = state.valid
    div_source = divergence_source if divergence_source is not None else ocean_source

    proposed_lat = state.lat.copy()
    proposed_lon = state.lon.copy()
    candidate_log_weight = state.log_weight.copy()
    physics_ok = np.ones(n, dtype=bool)

    death_lat = state.death_lat.copy() if state.death_lat is not None else np.full(n, np.nan)
    death_lon = state.death_lon.copy() if state.death_lon is not None else np.full(n, np.nan)
    death_reason = (
        state.death_reason.copy() if state.death_reason is not None else np.array([""] * n, dtype=object)
    )

    if np.any(still_active):
        lat_a = state.lat[still_active]
        lon_a = state.lon[still_active]

        u_val, u_valid = interpolate_field(ocean_source, "uo", lat_a, lon_a, t)
        v_val, v_valid = interpolate_field(ocean_source, "vo", lat_a, lon_a, t)

        k_val, dK_dx, dK_dy, k_valid = interpolated_k_value_and_gradient(
            k_lat, k_lon, k_field, lat_a, lon_a, radius_m=radius_m
        )
        div_val, div_valid = interpolated_divergence(div_source, lat_a, lon_a, t, radius_m=radius_m)

        ok = u_valid & v_valid & k_valid & div_valid

        xi = rng.standard_normal(size=(len(lat_a), 2))

        # -U + grad(K), times dt: the backward-time drift (never +U).
        drift_x_m = np.where(ok, (-u_val + dK_dx) * dt_seconds, 0.0)
        drift_y_m = np.where(ok, (-v_val + dK_dy) * dt_seconds, 0.0)
        diffusion_scale = np.where(ok, np.sqrt(np.maximum(2.0 * k_val, 0.0) * dt_seconds), 0.0)
        dx_m = drift_x_m + diffusion_scale * xi[:, 0]
        dy_m = drift_y_m + diffusion_scale * xi[:, 1]

        dlon_deg, dlat_deg = metres_to_degrees(dx_m, dy_m, lat_a, radius_m=radius_m)

        proposed_lat_active = np.where(ok, lat_a + dlat_deg, lat_a)
        proposed_lon_active = np.where(ok, lon_a + dlon_deg, lon_a)
        candidate_log_weight_active = np.where(
            ok, state.log_weight[still_active] - div_val * dt_seconds, state.log_weight[still_active]
        )

        proposed_lat[still_active] = proposed_lat_active
        proposed_lon[still_active] = proposed_lon_active
        candidate_log_weight[still_active] = candidate_log_weight_active
        physics_ok[still_active] = ok

    domain_check = netcdf_validity_check(ocean_source, domain_variable, t)
    position_before = state.as_particle_state()
    position_after = apply_boundary_policy(position_before, proposed_lat, proposed_lon, domain_check)

    final_valid = position_after.valid & physics_ok
    committed_move = still_active & physics_ok & position_after.valid
    final_log_weight = np.where(committed_move, candidate_log_weight, state.log_weight)

    # T13.5: classify and record the exact death event for any particle newly invalid
    # this step -- never touches a particle that was already invalid before this call.
    newly_invalid = still_active & ~final_valid
    if np.any(newly_invalid):
        physics_failed = newly_invalid & ~physics_ok
        boundary_failed = newly_invalid & physics_ok
        if np.any(physics_failed):
            death_lat[physics_failed] = state.lat[physics_failed]
            death_lon[physics_failed] = state.lon[physics_failed]
            death_reason[physics_failed] = "physics_unavailable"
        if np.any(boundary_failed):
            death_lat[boundary_failed] = proposed_lat[boundary_failed]
            death_lon[boundary_failed] = proposed_lon[boundary_failed]
            death_reason[boundary_failed] = _classify_boundary_death_reason(
                ocean_source.lat, ocean_source.lon, proposed_lat[boundary_failed], proposed_lon[boundary_failed]
            )

    return WeightedParticleState(
        lat=position_after.lat, lon=position_after.lon, valid=final_valid, log_weight=final_log_weight,
        death_lat=death_lat, death_lon=death_lon, death_reason=death_reason,
    )
