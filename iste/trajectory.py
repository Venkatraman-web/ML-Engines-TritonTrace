"""T06 — one weighted particle propagated over a configurable backward window.

This module adds no new physics: it repeatedly calls T05's
`iste.step.backward_euler_maruyama_step` -- the equations

    X_new = X_old + (-U + grad(K)) * dt + sqrt(2*K*dt) * xi
    log_w_new = log_w_old - div(U) * dt

with the locked `U = (uo, vo)` -- are not duplicated here. T06 only adds the loop, the
fixed backward-time checkpoint schedule, and the "stop calling the step once invalid, but
keep recording states" bookkeeping.

Backward time is explicit: the trajectory starts at the observation time `t0` and each
accepted step moves to `t - dt` (`README.md`: `theta = t0 - t`, so increasing `theta`
means decreasing `t`). The checkpoint timestamps are a *fixed* schedule
`t0, t0 - dt, t0 - 2*dt, ..., t0 - n_steps*dt`, computed directly from `n_steps`/`dt`
(never accumulated step-by-step, so floating-point drift cannot make a timestamp
contradict the configured step count and `dt`) -- this schedule does not depend on
whether the particle is still valid.

`n_steps` and `dt_seconds` are explicit experiment parameters; nothing here hard-codes a
7-day window, a 15-minute step, or any other "universal" trajectory length.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from iste.environment import GriddedFieldSource
from iste.geometry import MEAN_EARTH_RADIUS_M
from iste.step import WeightedParticleState, backward_euler_maruyama_step


@dataclass
class BackwardTrajectory:
    """The recorded time-indexed history of one backward-propagated particle.

    Every array has length `n_steps + 1` (the initial state at `t0` plus one entry per
    configured step) -- this length never changes regardless of when/whether the
    particle becomes invalid; only `lat`/`lon`/`log_weight`/`valid` reflect that.

    `death_time`/`death_lat`/`death_lon`/`death_reason` (T13.5,
    `final_stage_III_IV_specification.txt` Sec. 6/9 compatibility): scalars describing
    the ONE death event this trajectory experienced, if any -- deliberately NOT
    per-checkpoint arrays, since a particle dies at most once. `death_time` is the exact
    checkpoint timestamp the rejected step was targeting (the finest time resolution this
    integrator actually resolves, since checkpoint spacing and integration step size are
    currently identical here -- see the module docstring); `death_lat`/`death_lon` are the
    exact invalidating position (which may differ from `lat[-1]`/`lon[-1]`, the last
    VALID position the trajectory arrays freeze at -- see
    `iste.step.backward_euler_maruyama_step`'s own `death_lat`/`death_lon` docs for why
    those are deliberately different quantities). All four default to "never died"
    sentinels (`None`/`NaT` and `NaN`/`""`) for a particle that stayed valid throughout.
    """

    timestamps: np.ndarray  # datetime64[ns], shape (n_steps+1,), t0, t0-dt, t0-2dt, ...
    lat: np.ndarray
    lon: np.ndarray
    log_weight: np.ndarray
    valid: np.ndarray
    death_time: np.datetime64 | None = None
    death_lat: float = float("nan")
    death_lon: float = float("nan")
    death_reason: str = ""

    @property
    def n_states(self) -> int:
        return len(self.timestamps)

    @property
    def first_invalid_index(self) -> int | None:
        """Index of the first invalid state, or `None` if the particle stayed valid
        throughout. Every state at or after this index is invalid (propagation, once
        stopped, never resumes)."""
        invalid_positions = np.flatnonzero(~self.valid)
        return int(invalid_positions[0]) if invalid_positions.size else None


def run_backward_trajectory(
    lat0: float,
    lon0: float,
    t0,
    n_steps: int,
    dt_seconds: float,
    ocean_source: GriddedFieldSource,
    k_lat: np.ndarray,
    k_lon: np.ndarray,
    k_field: np.ndarray,
    rng: np.random.Generator,
    log_weight0: float = 0.0,
    domain_variable: str = "uo",
    radius_m: float = MEAN_EARTH_RADIUS_M,
    divergence_source: GriddedFieldSource | None = None,
) -> BackwardTrajectory:
    """Propagate exactly one weighted particle backward from `(lat0, lon0)` at `t0`, for
    `n_steps` steps of `dt_seconds` each, by repeatedly calling T05's
    `backward_euler_maruyama_step` -- see the module docstring for the equations, which
    are not reproduced or duplicated here.

    `n_steps` (>= 0) and `dt_seconds` (> 0) are the caller's explicit experiment
    parameters -- e.g. a 7-day window at a 15-minute step is `n_steps=672,
    dt_seconds=900.0`, but neither value is assumed or hard-coded by this function.

    `ocean_source`, `k_lat`/`k_lon`/`k_field`, `domain_variable`, and `radius_m` are
    passed straight through to each T05 step call (see its docstring for their meaning).
    `divergence_source` (T13.5) is likewise passed straight through unchanged -- `None`
    (the default) preserves the original behavior (divergence computed from
    `ocean_source`, exactly as before T13.5) for every caller that does not need to
    distinguish an advection field from a divergence field. `rng` is a single
    `numpy.random.Generator` shared across all steps of this one trajectory (its running
    state advances step by step, exactly as a single-particle, multi-step reproducible
    simulation requires).

    Returns a `BackwardTrajectory` with one recorded state per step, `t0` included.
    Once the particle becomes invalid, no further T05 step calls are made for it -- its
    last valid position and weight are simply carried forward, unchanged, into every
    remaining recorded state (still explicitly marked invalid), rather than being dropped.
    The trajectory's `death_time`/`death_lat`/`death_lon`/`death_reason` (T13.5) are set
    exactly once, at the step where invalidity was first detected -- `death_time` is that
    step's target checkpoint timestamp (the finest resolution available, since this
    integrator's checkpoint spacing IS its integration step size; see the module
    docstring), never fabricated to some interpolated in-between instant.
    """
    if n_steps < 0:
        raise ValueError("n_steps must be >= 0")
    if dt_seconds <= 0:
        raise ValueError("dt_seconds must be > 0 (an increment of backward time theta)")

    t0 = np.datetime64(t0, "ns")
    dt_ns = int(round(dt_seconds * 1e9))

    n_states = n_steps + 1
    timestamps = t0 - np.arange(n_states, dtype=np.int64) * np.timedelta64(dt_ns, "ns")

    lat_hist = np.empty(n_states, dtype=np.float64)
    lon_hist = np.empty(n_states, dtype=np.float64)
    log_weight_hist = np.empty(n_states, dtype=np.float64)
    valid_hist = np.empty(n_states, dtype=bool)

    state = WeightedParticleState.initialize(lat0=[lat0], lon0=[lon0], log_weight0=log_weight0)
    lat_hist[0] = state.lat[0]
    lon_hist[0] = state.lon[0]
    log_weight_hist[0] = state.log_weight[0]
    valid_hist[0] = state.valid[0]

    death_time = None
    death_lat = float("nan")
    death_lon = float("nan")
    death_reason = ""

    for i in range(1, n_states):
        if state.valid[0]:
            step_start_t = timestamps[i - 1]
            state = backward_euler_maruyama_step(
                state, step_start_t, dt_seconds, ocean_source, k_lat, k_lon, k_field, rng,
                domain_variable=domain_variable, radius_m=radius_m, divergence_source=divergence_source,
            )
            if death_time is None and not state.valid[0]:
                # first step at which invalidity was detected -- record the death event
                # exactly once, using T05's own death_lat/lon/reason for this particle.
                death_time = timestamps[i]
                death_lat = float(state.death_lat[0]) if state.death_lat is not None else float("nan")
                death_lon = float(state.death_lon[0]) if state.death_lon is not None else float("nan")
                death_reason = str(state.death_reason[0]) if state.death_reason is not None else ""
        # else: already invalid -- `state` is left exactly as it is (T04/T05's own
        # "once invalid, stays invalid, does not move" policy), and no further T05 step
        # is even called, since its outcome would be a no-op by construction.
        lat_hist[i] = state.lat[0]
        lon_hist[i] = state.lon[0]
        log_weight_hist[i] = state.log_weight[0]
        valid_hist[i] = state.valid[0]

    return BackwardTrajectory(
        timestamps=timestamps, lat=lat_hist, lon=lon_hist, log_weight=log_weight_hist, valid=valid_hist,
        death_time=death_time, death_lat=death_lat, death_lon=death_lon, death_reason=death_reason,
    )
