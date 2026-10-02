"""F02 -- deterministic one-hour forward particle step for the ow-0009 pilot.

Combines `fwd.forcing.ForcingAdapter` (F01) and `fwd.boundary.is_land` (F02) into the
single OpenOil-style horizontal motion equation CLAUDE.md locks:

    u_oil = current_drift_factor*uo + wind_drift_factor*u10
    v_oil = current_drift_factor*vo + wind_drift_factor*v10
    (current_drift_factor = 1.0, wind_drift_factor = 0.03)

No Stokes drift, no diffusion, no randomness, no weathering, no vertical motion --
exactly one deterministic Euler displacement per call, in geographic coordinates, via
`iste.geometry.metres_to_degrees` (unmodified Stage III code) for the metre -> degree
conversion.

Land/coastline stranding (CLAUDE.md section 7/9) is checked twice per step:
1. at the particle's *current* position (defensive -- should not trigger for a particle
   that was already accepted as water on the previous step, but a particle must never
   silently keep moving once it is on land);
2. at the particle's *proposed next* position -- the actual coastline-crossing check.
   If the next position would be on land, the particle stops permanently **at its last
   valid position**, never at the land cell itself (CLAUDE.md section 7: "Retain the
   last position in trajectory history"; do not use OpenDrift's `previous` behavior).

Once a particle is `stranded`, every subsequent call to `step_particle` is a no-op that
returns the same position unchanged (requirement 7: "Never move stranded particles
again").
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from fwd.boundary import is_land
from fwd.forcing import INVALID_ENVIRONMENT, ForcingAdapter
from fwd.stage3_bridge import ensure_iste_importable

ensure_iste_importable()

from iste.geometry import metres_to_degrees  # noqa: E402

CURRENT_DRIFT_FACTOR = 1.0
WIND_DRIFT_FACTOR = 0.03
DT_SECONDS = 3600.0

# Step-level outcomes not produced by fwd.forcing (which deliberately knows nothing
# about land, per its own docstring) -- these are this module's own vocabulary for why
# a step did not produce ordinary motion.
LAND_STRANDED = "LAND_STRANDED"
ALREADY_STRANDED = "ALREADY_STRANDED"


@dataclass
class ParticleState:
    particle_id: int
    lon: float
    lat: float
    valid: bool = True
    stranded: bool = False


@dataclass(frozen=True)
class StepRecord:
    """One audit record per particle per step (requirement 9)."""

    particle_id: int
    time_start: np.datetime64
    time_end: np.datetime64
    old_lon: float
    old_lat: float
    new_lon: float
    new_lat: float
    uo: float
    vo: float
    u10: float
    v10: float
    oil_u: float
    oil_v: float
    valid: bool
    stranded: bool
    forcing_mode: str  # DIRECT | NEIGHBOR_FALLBACK | INVALID_ENVIRONMENT | LAND_STRANDED | ALREADY_STRANDED
    attempted_lon: float | None = None  # set only when a land-crossing was rejected
    attempted_lat: float | None = None


def _dead_record(
    state: ParticleState, time_start: np.datetime64, time_end: np.datetime64, mode: str,
    uo: float = float("nan"), vo: float = float("nan"),
    u10: float = float("nan"), v10: float = float("nan"),
    oil_u: float = float("nan"), oil_v: float = float("nan"),
    attempted_lon: float | None = None, attempted_lat: float | None = None,
) -> StepRecord:
    return StepRecord(
        particle_id=state.particle_id,
        time_start=time_start,
        time_end=time_end,
        old_lon=state.lon,
        old_lat=state.lat,
        new_lon=state.lon,
        new_lat=state.lat,
        uo=uo, vo=vo, u10=u10, v10=v10, oil_u=oil_u, oil_v=oil_v,
        valid=False,
        stranded=True,
        forcing_mode=mode,
        attempted_lon=attempted_lon,
        attempted_lat=attempted_lat,
    )


def step_particle(
    adapter: ForcingAdapter,
    state: ParticleState,
    time_start,
    dt_seconds: float = DT_SECONDS,
    wind_drift_factor: float = WIND_DRIFT_FACTOR,
    current_drift_factor: float = CURRENT_DRIFT_FACTOR,
) -> tuple[ParticleState, StepRecord]:
    """Advance one particle by `dt_seconds` (default one hour), or leave it unchanged
    if it is already stranded. Returns `(new_state, record)`."""
    time_start = np.datetime64(time_start, "ns")
    time_end = time_start + np.timedelta64(int(dt_seconds), "s")

    if state.stranded or not state.valid:
        return state, _dead_record(state, time_start, time_end, ALREADY_STRANDED)

    old_lon, old_lat = state.lon, state.lat

    if is_land(adapter, old_lon, old_lat):
        dead_state = replace(state, valid=False, stranded=True)
        return dead_state, _dead_record(state, time_start, time_end, LAND_STRANDED)

    forcing = adapter.query(old_lon, old_lat, time_start)

    if forcing.ocean_mode == INVALID_ENVIRONMENT or forcing.wind_mode == INVALID_ENVIRONMENT:
        dead_state = replace(state, valid=False, stranded=True)
        return dead_state, _dead_record(
            state, time_start, time_end, INVALID_ENVIRONMENT,
            uo=forcing.uo, vo=forcing.vo, u10=forcing.u10, v10=forcing.v10,
        )

    oil_u = current_drift_factor * forcing.uo + wind_drift_factor * forcing.u10
    oil_v = current_drift_factor * forcing.vo + wind_drift_factor * forcing.v10

    dlon_deg, dlat_deg = metres_to_degrees(oil_u * dt_seconds, oil_v * dt_seconds, old_lat)
    candidate_lon = old_lon + float(dlon_deg)
    candidate_lat = old_lat + float(dlat_deg)

    if is_land(adapter, candidate_lon, candidate_lat):
        dead_state = replace(state, valid=False, stranded=True)
        return dead_state, _dead_record(
            state, time_start, time_end, LAND_STRANDED,
            uo=forcing.uo, vo=forcing.vo, u10=forcing.u10, v10=forcing.v10,
            oil_u=oil_u, oil_v=oil_v,
            attempted_lon=candidate_lon, attempted_lat=candidate_lat,
        )

    new_state = ParticleState(state.particle_id, candidate_lon, candidate_lat, valid=True, stranded=False)
    record = StepRecord(
        particle_id=state.particle_id,
        time_start=time_start,
        time_end=time_end,
        old_lon=old_lon, old_lat=old_lat,
        new_lon=candidate_lon, new_lat=candidate_lat,
        uo=forcing.uo, vo=forcing.vo, u10=forcing.u10, v10=forcing.v10,
        oil_u=oil_u, oil_v=oil_v,
        valid=True, stranded=False,
        forcing_mode=forcing.ocean_mode,
    )
    return new_state, record


def step_all(
    adapter: ForcingAdapter,
    states: list[ParticleState],
    time_start,
    dt_seconds: float = DT_SECONDS,
) -> tuple[list[ParticleState], list[StepRecord]]:
    """Advance a list of particles by one step each (a convenience loop over
    `step_particle`, not a multi-hour propagation loop -- that is a later task)."""
    new_states: list[ParticleState] = []
    records: list[StepRecord] = []
    for state in states:
        new_state, record = step_particle(adapter, state, time_start, dt_seconds=dt_seconds)
        new_states.append(new_state)
        records.append(record)
    return new_states, records
