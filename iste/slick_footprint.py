"""T22 -- Stage VI/VII boundary: predicted slick footprint and independent observed
reference.

Implements exactly `README.md` Sec. 20/21: at `t_obs`, build

    predicted slick KDE_e(t_obs)          -- from T21's forward particles, per candidate/member
    observed slick reference KDE(t_obs)   -- from the upstream observation particles (T07)

as two SEPARATE, never-merged representations, both evaluated on the SAME caller-
supplied regular grid so a later task (T23) can compare them cell-by-cell without
requiring equal particle counts.

**The predicted KDE is a simulated-slick footprint, not a source-probability map, a
posterior, or a likelihood** (`README.md` Sec. 20). It is built ONLY from T21's forward
particles at `t_obs` (`ForwardCandidateMemberRun.final_lat`/`final_lon`/`final_valid`),
never from Stage III `c_hat*`, which this module never reads or references.

**The observed reference is built ONLY from the upstream observation particles**
(`iste.observation.AdaptedParticleSet`, T07, unmodified) -- `README.md` Sec. 21 is
explicit that "the backward historical particles are not used as the independent truth
reference," so `build_observed_slick_footprint` never imports or accepts
`iste.particle_query.HistoricalParticleSlice`, `iste.kdtree_neighborhood.
ParticleNeighborhoodIndex`, or any T14-T21 candidate/forward type -- its only particle
input type is `AdaptedParticleSet`.

**KDE construction.** Both footprints use the same small, explicit, PROTOTYPE/
CONFIGURATION Gaussian-kernel density estimate (`README.md` Sec. 26: "KDE bandwidth" is
an explicit open Phase 2 parameter, not a validated physical choice): for each valid
particle, an isotropic 2-D Gaussian kernel of bandwidth `bandwidth_m` (metres) is placed
at that particle's position and evaluated at every point of the caller's regular
`lat_grid x lon_grid`, summed, and divided by the NUMBER OF VALID PARTICLES -- this
normalization is exactly what decouples the resulting footprint's shape/scale from the
particle COUNT (`README.md`'s own check: "particle-count changes do not by themselves
change a normalized footprint score"), satisfying "the comparison must not require equal
particle counts" (Sec. 20/22) without inventing a new interpretation of that
requirement. `kernel_fn` is a pluggable callable (default `gaussian_kernel_2d`) so a
caller can substitute a different kernel entirely, matching T16/T17's own established
"combine_fn"/`config` pluggability discipline.

**Distances are always computed via `iste.kdtree_neighborhood.project_to_local_metres`**
(T15's own distance-safe local/projected coordinates, reused unchanged) -- latitude/
longitude degrees are never treated directly as Euclidean metres anywhere in this
module.

Does not implement the predicted-vs-observed comparison/error metric (T23), candidate
ranking/filtering, AIS, or formal source inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from iste.forward_propagation import ForwardCandidateMemberRun
from iste.kdtree_neighborhood import project_to_local_metres
from iste.observation import AdaptedParticleSet


def gaussian_kernel_2d(distance_m: np.ndarray, bandwidth_m: float) -> np.ndarray:
    """The default KDE kernel: a standard isotropic 2-D Gaussian,
    `exp(-d^2 / (2*h^2)) / (2*pi*h^2)`. PROTOTYPE/CONFIGURATION (see module docstring),
    not a claim of scientific validation -- `combine_fn`/`kernel_fn`-style pluggability
    (here, `FootprintGridConfig.kernel_fn`) lets a caller substitute any other kernel."""
    return np.exp(-0.5 * (distance_m / bandwidth_m) ** 2) / (2.0 * np.pi * bandwidth_m**2)


@dataclass(frozen=True)
class FootprintGridConfig:
    """Explicit, shared KDE grid/bandwidth configuration used for BOTH the predicted and
    observed footprints, so the two are directly comparable cell-by-cell by a later task.

    - `lat_grid`/`lon_grid`: 1-D regular query-grid coordinate arrays (the SAME arrays
      should be reused for every footprint being compared in one Phase 2 run).
    - `bandwidth_m`: the Gaussian kernel's bandwidth, in metres. Must be finite and > 0.
      An explicit Phase 2 calibration parameter (`README.md` Sec. 26), never a hidden
      constant.
    - `lat_ref`/`lon_ref`: the local-projection reference point (`iste.kdtree_
      neighborhood.project_to_local_metres`). Defaults to the grid's own mean
      latitude/longitude if not given -- an explicit, documented default, not a hidden
      one; pass explicitly to pin the projection to a fixed reference across multiple
      footprints/timestamps instead.
    - `kernel_fn`: `(distance_m, bandwidth_m) -> weight`, defaults to
      `gaussian_kernel_2d`.
    """

    lat_grid: np.ndarray
    lon_grid: np.ndarray
    bandwidth_m: float
    lat_ref: float | None = None
    lon_ref: float | None = None
    kernel_fn: Callable[[np.ndarray, float], np.ndarray] = gaussian_kernel_2d

    def __post_init__(self) -> None:
        lat_grid = np.asarray(self.lat_grid, dtype=np.float64)
        lon_grid = np.asarray(self.lon_grid, dtype=np.float64)
        if lat_grid.ndim != 1 or lat_grid.size == 0:
            raise ValueError(f"lat_grid must be a non-empty 1-D array; got shape {lat_grid.shape}")
        if lon_grid.ndim != 1 or lon_grid.size == 0:
            raise ValueError(f"lon_grid must be a non-empty 1-D array; got shape {lon_grid.shape}")
        if not np.isfinite(self.bandwidth_m) or self.bandwidth_m <= 0:
            raise ValueError(f"bandwidth_m must be finite and > 0; got {self.bandwidth_m!r}")


@dataclass
class PredictedSlickFootprint:
    """One candidate/member's predicted-slick KDE footprint at `t_obs`
    (`README.md` Sec. 20) -- a simulated-slick representation, NOT a source-probability
    map, posterior, or likelihood.

    `density` has shape `(len(lat_grid), len(lon_grid))` (row-major, matching T17's own
    `support_grid` convention). `density` is all-`NaN` (never zero, never fabricated)
    when `has_valid_particles` is `False` -- an explicitly undefined footprint, not a
    silently-valid empty one.
    """

    candidate_id: str
    t_s: np.datetime64
    t_obs: np.datetime64
    forward_forcing_member_id: int
    lat_grid: np.ndarray
    lon_grid: np.ndarray
    density: np.ndarray
    bandwidth_m: float
    n_particles: int
    n_valid_particles: int
    has_valid_particles: bool


@dataclass
class ObservedSlickFootprint:
    """The independent observed-slick reference KDE footprint at `t_obs`
    (`README.md` Sec. 21), built ONLY from upstream observation particles -- carries no
    `candidate_id`/`forcing_member_id` at all (by construction, not merely by omission),
    since it is independent of every candidate/member (`README.md`: "the backward
    historical particles are not used as the independent truth reference").

    `density` has shape `(len(lat_grid), len(lon_grid))`, all-`NaN` when
    `has_valid_particles` is `False` -- same explicit-undefined convention as
    `PredictedSlickFootprint`.
    """

    t_obs: np.datetime64
    lat_grid: np.ndarray
    lon_grid: np.ndarray
    density: np.ndarray
    bandwidth_m: float
    n_particles: int
    n_valid_particles: int
    has_valid_particles: bool


def _compute_density_grid(
    particle_lat: np.ndarray, particle_lon: np.ndarray, particle_valid: np.ndarray, config: FootprintGridConfig,
) -> tuple[np.ndarray, int, int]:
    """Shared KDE core for both predicted and observed footprints: project the grid and
    the VALID particles into the same local metres (`iste.kdtree_neighborhood.
    project_to_local_metres`), evaluate `config.kernel_fn` pairwise, sum, and divide by
    the valid-particle count. Returns `(density_grid, n_particles_total,
    n_valid_particles)`; `density_grid` is all-`NaN` if there are zero valid particles.
    """
    particle_lat = np.asarray(particle_lat, dtype=np.float64)
    particle_lon = np.asarray(particle_lon, dtype=np.float64)
    valid_mask = np.asarray(particle_valid, dtype=bool)
    n_total = len(particle_lat)
    n_valid = int(np.sum(valid_mask))

    lat_ref = config.lat_ref if config.lat_ref is not None else float(np.mean(config.lat_grid))
    lon_ref = config.lon_ref if config.lon_ref is not None else float(np.mean(config.lon_grid))

    lon_mesh, lat_mesh = np.meshgrid(config.lon_grid, config.lat_grid)  # both shape (n_lat, n_lon)
    grid_x, grid_y = project_to_local_metres(lat_mesh.ravel(), lon_mesh.ravel(), lat_ref, lon_ref)

    if n_valid == 0:
        return np.full(lat_mesh.shape, np.nan), n_total, n_valid

    px, py = project_to_local_metres(particle_lat[valid_mask], particle_lon[valid_mask], lat_ref, lon_ref)

    dx = grid_x[:, None] - px[None, :]  # (n_grid_points, n_valid_particles)
    dy = grid_y[:, None] - py[None, :]
    distance_m = np.sqrt(dx**2 + dy**2)
    kernel_vals = config.kernel_fn(distance_m, config.bandwidth_m)
    density_flat = np.sum(kernel_vals, axis=1) / n_valid  # normalize by VALID count only
    return density_flat.reshape(lat_mesh.shape), n_total, n_valid


def build_predicted_slick_footprint(
    run: ForwardCandidateMemberRun, config: FootprintGridConfig
) -> PredictedSlickFootprint:
    """Build one candidate/member's predicted-slick KDE footprint at `t_obs`, from T21's
    `ForwardCandidateMemberRun` -- uses ONLY `run.final_lat`/`final_lon`/`final_valid`
    (the particles' state at exactly `t_obs`, since `run.timestamps[-1] == run.t_obs` by
    T21's own contract); does not read or need any earlier checkpoint of the trajectory.
    """
    density, n_total, n_valid = _compute_density_grid(run.final_lat, run.final_lon, run.final_valid, config)
    return PredictedSlickFootprint(
        candidate_id=run.candidate_id,
        t_s=run.t_s,
        t_obs=run.t_obs,
        forward_forcing_member_id=run.forward_forcing_member_id,
        lat_grid=np.asarray(config.lat_grid, dtype=np.float64),
        lon_grid=np.asarray(config.lon_grid, dtype=np.float64),
        density=density,
        bandwidth_m=config.bandwidth_m,
        n_particles=n_total,
        n_valid_particles=n_valid,
        has_valid_particles=n_valid > 0,
    )


def build_predicted_slick_footprints(
    runs: dict[int, ForwardCandidateMemberRun], config: FootprintGridConfig
) -> dict[int, PredictedSlickFootprint]:
    """`build_predicted_slick_footprint` for every member in `runs` (e.g.
    `iste.forward_propagation.ForwardCandidateRuns.runs`), one footprint per member,
    keyed by `forward_forcing_member_id`."""
    return {member_id: build_predicted_slick_footprint(run, config) for member_id, run in runs.items()}


def build_observed_slick_footprint(
    adapted: AdaptedParticleSet, t_obs, config: FootprintGridConfig
) -> ObservedSlickFootprint:
    """Build the independent observed-slick reference KDE footprint at `t_obs`, from
    T07's `AdaptedParticleSet` (the upstream observation particles) -- ONLY
    `adapted.state.lat`/`state.lon`/`state.valid` are used; `t_obs` is the caller's own
    explicit observation timestamp (not inferred from `adapted.timestamp`, which is a
    per-particle array whose homogeneity this module does not assume or validate).

    Never reads a `HistoricalParticleSlice`, a `ForwardCandidateMemberRun`, or any other
    T14-T21 candidate/backward-history type -- see the module docstring for why.
    """
    density, n_total, n_valid = _compute_density_grid(
        adapted.state.lat, adapted.state.lon, adapted.state.valid, config
    )
    return ObservedSlickFootprint(
        t_obs=np.datetime64(t_obs, "ns"),
        lat_grid=np.asarray(config.lat_grid, dtype=np.float64),
        lon_grid=np.asarray(config.lon_grid, dtype=np.float64),
        density=density,
        bandwidth_m=config.bandwidth_m,
        n_particles=n_total,
        n_valid_particles=n_valid,
        has_valid_particles=n_valid > 0,
    )
