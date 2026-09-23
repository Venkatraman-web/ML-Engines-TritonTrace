"""T04 — Stage III boundary and particle-validity policy (`README.md`).

`README.md`'s prototype policy:

    proposed step enters invalid/land cell
        -> mark particle invalid
        -> stop its propagation
        -> retain invalid status for diagnostics

and: "Particles leaving the valid ocean/data domain must also become explicitly invalid."

This module implements exactly that stop-on-invalid policy -- no shoreline reflection,
absorption, clipping, wrapping, or re-entry is implemented, per the task instructions.

It does **not** redefine what "valid" means: whether a proposed location is inside the
valid ocean/data domain is answered by an injected `is_valid_location` callable, which
for real data is `netcdf_validity_check` below -- a thin adapter over
`iste.interpolation.interpolate_field`'s existing T02 validity semantics (a location is
invalid if it is masked/missing source data, or outside the grid's covered extent). This
keeps the boundary policy itself independent of any NetCDF layout, and tests can supply a
trivial synthetic `is_valid_location` function instead.

`ParticleState` is the minimal per-particle representation the policy needs (position +
validity flag, vectorized over `n` particles). It is deliberately not the full Stage III
particle record (no weight, `observation_pixel_id`, or `forcing_member_id` yet -- those
belong to later tasks); T04 defines only what this task needs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import numpy as np

from iste.interpolation import interpolate_field

if TYPE_CHECKING:
    from iste.environment import GriddedFieldSource

ValidityCheck = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass
class ParticleState:
    """Minimal vectorized particle state for the T04 boundary/validity policy.

    - `lat`, `lon`: each particle's current position -- for an invalid particle this is
      its *last valid* position (invalid particles never move again).
    - `valid`: `True` while a particle is still propagating; once `False`, a particle
      never becomes `True` again ("once invalid, a particle stays invalid").

    Invalid particles are never removed from these arrays -- all `n_particles` entries
    always remain present so later diagnostics can count valid/invalid particles and the
    valid fraction.
    """

    lat: np.ndarray
    lon: np.ndarray
    valid: np.ndarray

    @classmethod
    def initialize(cls, lat0, lon0) -> "ParticleState":
        """Create a new, fully-valid particle state at the given starting positions."""
        lat0 = np.atleast_1d(np.asarray(lat0, dtype=np.float64))
        lon0 = np.atleast_1d(np.asarray(lon0, dtype=np.float64))
        if lat0.shape != lon0.shape:
            raise ValueError("lat0 and lon0 must have the same shape")
        return cls(lat=lat0.copy(), lon=lon0.copy(), valid=np.ones(lat0.shape, dtype=bool))

    @property
    def n_particles(self) -> int:
        return self.lat.shape[0]

    @property
    def valid_count(self) -> int:
        return int(np.sum(self.valid))

    @property
    def invalid_count(self) -> int:
        return int(np.sum(~self.valid))

    @property
    def valid_particle_fraction(self) -> float:
        return float(np.mean(self.valid)) if self.n_particles else float("nan")


def apply_boundary_policy(
    state: ParticleState,
    proposed_lat: np.ndarray,
    proposed_lon: np.ndarray,
    is_valid_location: ValidityCheck,
) -> ParticleState:
    """Apply the Stage III v1 stop-on-invalid boundary policy to one proposed step.

    For each particle, independently:

    - **already invalid** (`state.valid[i]` is `False`): left completely unchanged --
      same position, still invalid. A proposed step is not even evaluated for it.
    - **still valid**, and `is_valid_location` says its proposed position is valid: the
      particle moves to `(proposed_lat[i], proposed_lon[i])` and remains valid.
    - **still valid**, but its proposed position is invalid (enters a land/invalid cell,
      or leaves the valid domain): the particle is marked invalid and does **not** move
      -- it stays at its last valid position, and its propagation stops here. This status
      is retained in the returned state, not dropped.

    Returns a new `ParticleState` (the input `state` is not mutated in place); the number
    of particles is always preserved.
    """
    proposed_lat = np.asarray(proposed_lat, dtype=np.float64)
    proposed_lon = np.asarray(proposed_lon, dtype=np.float64)
    if proposed_lat.shape != state.lat.shape or proposed_lon.shape != state.lon.shape:
        raise ValueError(
            "proposed_lat/proposed_lon must match state's particle count "
            f"({state.lat.shape}); got {proposed_lat.shape} and {proposed_lon.shape}"
        )

    still_active = state.valid
    step_valid = np.zeros_like(state.valid)
    if np.any(still_active):
        step_valid[still_active] = np.asarray(
            is_valid_location(proposed_lat[still_active], proposed_lon[still_active]), dtype=bool
        )

    new_lat = state.lat.copy()
    new_lon = state.lon.copy()
    new_valid = state.valid.copy()

    move_mask = still_active & step_valid
    new_lat[move_mask] = proposed_lat[move_mask]
    new_lon[move_mask] = proposed_lon[move_mask]

    newly_invalid_mask = still_active & ~step_valid
    new_valid[newly_invalid_mask] = False
    # newly-invalid particles' lat/lon are left as the copied (last valid) values above.

    return ParticleState(lat=new_lat, lon=new_lon, valid=new_valid)


def netcdf_validity_check(source: "GriddedFieldSource", variable: str, time) -> ValidityCheck:
    """Build an `is_valid_location(lat, lon) -> bool array` callable, backed by
    `iste.interpolation.interpolate_field`'s existing T02 validity semantics at a fixed
    time `time`, using `variable` (e.g. `"uo"`) as the domain/land reference field.

    This does not redefine what "valid" means -- a location is exactly as valid as T02's
    interpolation already says it is (not masked/missing, inside the grid's extent).
    """

    def _check(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        _, valid = interpolate_field(source, variable, lat, lon, time)
        return valid

    return _check
