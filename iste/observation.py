"""T07 (revised 2026-09-22) — backtracking input adapter / particle interface.

**Correction notice:** the original T07 implementation built a synthetic observation
footprint and generated/initialized particles from it. That was redundant: satellite
segmentation and particle generation are owned by an upstream teammate module, not by
this backtracking codebase. This module was rewritten to remove that generation path
entirely and replace it with a thin adapter that starts *after* upstream particles
already exist.

This module does **not** implement or duplicate:

- satellite segmentation,
- oil/clean classification,
- observation-footprint generation,
- particle generation/initialization,
- Stage I detection probability (`pi_j`),
- observation likelihood.

It provides exactly two things:

1. `adapt_upstream_particles` -- validates a sequence of upstream particle records (plain
   mappings, e.g. dicts, since the upstream schema has not been formally specified yet)
   and repackages them into a T05 `WeightedParticleState` (unmodified) plus the upstream
   identifiers this backtracking module must preserve: `particle_id`,
   `observation_pixel_id`, and `timestamp`. Missing required fields are rejected
   explicitly, not silently defaulted.

2. `TaggedTrajectory` / `tag_trajectory` -- attaches an `observation_pixel_id` (and,
   preserved from upstream, a `particle_id`) to an already-run T06 `BackwardTrajectory`
   (unmodified), so the identifier survives a full backward trajectory unchanged,
   including one that becomes invalid.

Neither T05 (`iste/step.py`) nor T06 (`iste/trajectory.py`) is modified or duplicated by
this module -- it only validates upstream input and connects it to their existing,
unchanged structures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from iste.step import WeightedParticleState
from iste.trajectory import BackwardTrajectory

REQUIRED_UPSTREAM_FIELDS = ("particle_id", "latitude", "longitude", "observation_pixel_id", "timestamp")
"""The minimum fields this backtracking module requires from an upstream particle
record. `valid` is accepted if present but is optional (defaults to `True`)."""


@dataclass
class AdaptedParticleSet:
    """The result of adapting upstream particle records into this backtracking module's
    representation.

    `state` is exactly T05's `WeightedParticleState` -- unmodified, not duplicated.
    `particle_id`, `observation_pixel_id`, and `timestamp` are the upstream values,
    preserved unchanged and in the same order as `state`'s arrays (never merged into
    `WeightedParticleState` itself, so T05/T06 stay untouched by this adapter).
    """

    state: WeightedParticleState
    particle_id: np.ndarray  # upstream identifiers, preserved as given (dtype=object)
    observation_pixel_id: np.ndarray  # upstream pixel identifiers, preserved as given
    timestamp: np.ndarray  # datetime64[ns], the observation/starting time per particle

    @property
    def n_particles(self) -> int:
        return self.state.n_particles


def adapt_upstream_particles(
    records: Sequence[Mapping[str, Any]], log_weight0: float = 0.0
) -> AdaptedParticleSet:
    """Validate and convert upstream particle records into an `AdaptedParticleSet`.

    Each element of `records` must be a mapping (e.g. a plain `dict`) containing at
    least:

    ```text
    particle_id            -- any hashable upstream identifier
    latitude, longitude    -- starting position, degrees
    observation_pixel_id   -- the upstream observation pixel this particle came from
    timestamp               -- observation/starting time (numpy.datetime64-convertible)
    ```

    and may optionally contain:

    ```text
    valid                   -- bool; validity/state info already determined upstream,
                                if any. Defaults to True (a normal starting particle)
                                when absent.
    ```

    This is deliberately the smallest interface that satisfies what T05/T06 need to run
    -- since the upstream schema has not been formally specified yet, no additional
    fields are assumed, invented, or silently accepted-and-ignored beyond these.

    Raises `KeyError` (naming the record index and the missing field(s)) if any required
    field is absent, and `ValueError` if `records` is empty or `particle_id` values are
    not unique.
    """
    if len(records) == 0:
        raise ValueError("records must contain at least one upstream particle")

    n = len(records)
    particle_id: list[Any] = []
    lat = np.empty(n, dtype=np.float64)
    lon = np.empty(n, dtype=np.float64)
    observation_pixel_id: list[Any] = []
    timestamp = np.empty(n, dtype="datetime64[ns]")
    valid = np.empty(n, dtype=bool)

    for i, record in enumerate(records):
        missing = [f for f in REQUIRED_UPSTREAM_FIELDS if f not in record]
        if missing:
            raise KeyError(f"upstream particle record {i} is missing required field(s): {missing}")
        particle_id.append(record["particle_id"])
        lat[i] = float(record["latitude"])
        lon[i] = float(record["longitude"])
        observation_pixel_id.append(record["observation_pixel_id"])
        timestamp[i] = np.datetime64(record["timestamp"], "ns")
        valid[i] = bool(record.get("valid", True))

    if len(set(particle_id)) != len(particle_id):
        raise ValueError(f"particle_id values must be unique; got {particle_id}")

    state = WeightedParticleState.initialize(lat0=lat, lon0=lon, log_weight0=log_weight0)
    state.valid[:] = valid  # honor upstream validity/state info, if any was supplied

    return AdaptedParticleSet(
        state=state,
        particle_id=np.array(particle_id, dtype=object),
        observation_pixel_id=np.array(observation_pixel_id, dtype=object),
        timestamp=timestamp,
    )


@dataclass
class TaggedTrajectory:
    """A T06 `BackwardTrajectory` (unmodified) paired with the `observation_pixel_id`
    (and, if known, the upstream `particle_id`) its one particle was released from. Both
    identifiers are constant attributes of this wrapper -- never part of the per-timestep
    trajectory arrays -- so they cannot be silently altered as the particle's
    position/weight/validity evolve, including once the particle becomes invalid."""

    trajectory: BackwardTrajectory
    observation_pixel_id: Any
    particle_id: Any = None


def tag_trajectory(
    trajectory: BackwardTrajectory, observation_pixel_id: Any, particle_id: Any = None
) -> TaggedTrajectory:
    """Attach `observation_pixel_id` (and optionally `particle_id`) to an already-computed
    T06 `BackwardTrajectory` (produced by `iste.trajectory.run_backward_trajectory`,
    unmodified)."""
    return TaggedTrajectory(trajectory=trajectory, observation_pixel_id=observation_pixel_id, particle_id=particle_id)
