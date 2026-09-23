"""T15 -- Phase 2 KD-tree neighborhood engine.

Implements exactly `README.md` Sec. 13's KD-tree candidate-generation engine: at one
historical timestamp (one T14 `HistoricalParticleSlice`), build a spatial-neighborhood
acceleration structure over the VALID particle positions, and answer fixed-radius or
k-nearest-neighbor queries against it, always returning each matched particle's complete
metadata.

**This module is an acceleration structure only.** It contains no KDE, no candidate-
support scoring (`D`/`W`/`V`, `README.md` Sec. 14), no candidate-peak/core extraction,
and no source inference -- those are later tasks (T16+) that will consume this module's
query results, not reimplement spatial search themselves.

**Distance-safe local/projected coordinates (`README.md` Sec. 13: "Latitude and
longitude degrees must not be treated directly as Euclidean metres"):** every particle
position is projected into a local equirectangular tangent-plane coordinate system
(`x` = eastward metres, `y` = northward metres) centered at an explicit reference
latitude/longitude, reusing `iste.geometry`'s own already-validated metre-per-degree
conversions (T03) -- the SAME spherical-Earth approximation the rest of this project
already uses, not a new or invented projection. This is a first-order approximation
appropriate for one regional point cloud at one instant (not a full geodesic/UTM
projection -- no geodesic library is installed in this environment, matching T03's own
documented limitation); it is not reused for, and does not replace, Stage III's own
metre/degree conversions.

**Reuses `scipy.spatial.cKDTree`** (already an available project dependency, per
`iste/geometry.py`'s own documented environment) rather than implementing a new KD-tree
from scratch.

Does not implement KDE, support-map construction, candidate-peak/core extraction, or any
Phase 2 task beyond T15's own KD-tree engine.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from iste.geometry import MEAN_EARTH_RADIUS_M, metres_per_degree_latitude, metres_per_degree_longitude
from iste.particle_query import HistoricalParticleSlice


def project_to_local_metres(
    lat: np.ndarray, lon: np.ndarray, lat_ref: float, lon_ref: float,
    radius_m: float = MEAN_EARTH_RADIUS_M,
) -> tuple[np.ndarray, np.ndarray]:
    """Equirectangular local tangent-plane projection of `(lat, lon)` (degrees) to
    `(x_m, y_m)` (metres eastward/northward) centered at `(lat_ref, lon_ref)`, via
    `iste.geometry.metres_per_degree_latitude`/`metres_per_degree_longitude` (T03) --
    never treating degrees directly as metres.

    `x_m = (lon - lon_ref) * metres_per_degree_longitude(lat_ref)`,
    `y_m = (lat - lat_ref) * metres_per_degree_latitude()`. Longitude's conversion factor
    is evaluated once at `lat_ref` (not re-evaluated per point), which is the standard
    small-area equirectangular approximation and matches how `iste.geometry.
    metres_to_degrees` already evaluates its own conversion at one reference latitude.
    """
    lat = np.atleast_1d(np.asarray(lat, dtype=np.float64))
    lon = np.atleast_1d(np.asarray(lon, dtype=np.float64))
    dlat_per_deg = metres_per_degree_latitude(radius_m)
    dlon_per_deg = metres_per_degree_longitude(lat_ref, radius_m)
    x_m = (lon - lon_ref) * dlon_per_deg
    y_m = (lat - lat_ref) * dlat_per_deg
    return x_m, y_m


@dataclass
class ParticleNeighborhoodIndex:
    """A KD-tree built over one `HistoricalParticleSlice`'s VALID particles only
    (`README.md` Sec. 13: "build a KD-tree over the valid particle positions"), in local
    projected metres. Purely an acceleration structure -- holds no support/KDE/candidate
    logic.

    `source_row` maps each point actually in the tree back to its row index in the
    original `slice_` (so query results can recover every field, including the ones not
    duplicated onto this index: `death_time`/`death_lat`/`death_lon`/`death_reason`).
    """

    slice_: HistoricalParticleSlice
    source_row: np.ndarray  # shape (n_indexed,): row index into slice_'s arrays
    x_m: np.ndarray
    y_m: np.ndarray
    lat_ref: float
    lon_ref: float
    radius_m: float
    _tree: cKDTree | None

    @property
    def n_particles(self) -> int:
        return len(self.source_row)


@dataclass
class NeighborhoodResult:
    """Complete metadata for the particles a KD-tree query matched, sorted by ascending
    `distance_m` (ties broken by `particle_id`) for deterministic, reproducible ordering
    regardless of the underlying tree's internal (unordered) result order. Every field
    from the source `HistoricalParticleSlice`/T14 is preserved -- `particle_id`,
    `observation_pixel_id`, `forcing_member_id` identity is never discarded, and
    validity/death metadata are carried through unchanged (queries only ever run against
    already-valid particles, so `valid` is always `True` here, but the field is kept for
    a consistent record shape)."""

    particle_id: np.ndarray
    observation_pixel_id: np.ndarray
    forcing_member_id: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    log_weight: np.ndarray
    valid: np.ndarray
    death_time: np.ndarray
    death_lat: np.ndarray
    death_lon: np.ndarray
    death_reason: np.ndarray
    distance_m: np.ndarray

    @property
    def n_particles(self) -> int:
        return len(self.particle_id)


def build_particle_neighborhood_index(
    slice_: HistoricalParticleSlice,
    lat_ref: float | None = None,
    lon_ref: float | None = None,
    radius_m: float = MEAN_EARTH_RADIUS_M,
) -> ParticleNeighborhoodIndex:
    """Build a `ParticleNeighborhoodIndex` over `slice_`'s VALID particles only (invalid
    particles are excluded from the tree, per `README.md` Sec. 13 -- they are never
    matched by any query, though they remain fully present, elsewhere, in `slice_`
    itself, per T14's own "never silently drop invalid particles" policy).

    `lat_ref`/`lon_ref` (the local projection's reference point) default to the mean
    latitude/longitude of the VALID particles being indexed -- an explicit, documented
    configuration choice (not a hidden constant); pass them explicitly to pin the
    projection to a fixed reference (e.g. the candidate-search region's center) across
    multiple timestamps/calls instead.
    """
    valid_mask = slice_.valid
    source_row = np.flatnonzero(valid_mask)
    lat = slice_.lat[source_row]
    lon = slice_.lon[source_row]

    if lat_ref is None:
        lat_ref = float(np.mean(lat)) if len(lat) else 0.0
    if lon_ref is None:
        lon_ref = float(np.mean(lon)) if len(lon) else 0.0

    if len(lat) == 0:
        x_m = np.array([], dtype=np.float64)
        y_m = np.array([], dtype=np.float64)
        tree = None
    else:
        x_m, y_m = project_to_local_metres(lat, lon, lat_ref, lon_ref, radius_m)
        tree = cKDTree(np.column_stack([x_m, y_m]))

    return ParticleNeighborhoodIndex(
        slice_=slice_, source_row=source_row, x_m=x_m, y_m=y_m,
        lat_ref=lat_ref, lon_ref=lon_ref, radius_m=radius_m, _tree=tree,
    )


def _select(index: ParticleNeighborhoodIndex, local_indices: np.ndarray, distance_m: np.ndarray) -> NeighborhoodResult:
    """Map local tree-point indices (`0..index.n_particles-1`) back to `index.slice_`'s
    original rows, build the full-metadata `NeighborhoodResult`, and sort it by ascending
    distance (ties broken by `particle_id`) for deterministic ordering."""
    slice_ = index.slice_
    rows = index.source_row[local_indices]

    order = np.lexsort((slice_.particle_id[rows].astype(str), distance_m))
    rows = rows[order]
    distance_m = distance_m[order]

    return NeighborhoodResult(
        particle_id=slice_.particle_id[rows],
        observation_pixel_id=slice_.observation_pixel_id[rows],
        forcing_member_id=slice_.forcing_member_id[rows],
        lat=slice_.lat[rows],
        lon=slice_.lon[rows],
        log_weight=slice_.log_weight[rows],
        valid=slice_.valid[rows],
        death_time=slice_.death_time[rows],
        death_lat=slice_.death_lat[rows],
        death_lon=slice_.death_lon[rows],
        death_reason=slice_.death_reason[rows],
        distance_m=distance_m,
    )


def _empty_result() -> NeighborhoodResult:
    return NeighborhoodResult(
        particle_id=np.array([], dtype=object),
        observation_pixel_id=np.array([], dtype=object),
        forcing_member_id=np.array([], dtype=np.int64),
        lat=np.array([], dtype=np.float64),
        lon=np.array([], dtype=np.float64),
        log_weight=np.array([], dtype=np.float64),
        valid=np.array([], dtype=bool),
        death_time=np.array([], dtype="datetime64[ns]"),
        death_lat=np.array([], dtype=np.float64),
        death_lon=np.array([], dtype=np.float64),
        death_reason=np.array([], dtype=object),
        distance_m=np.array([], dtype=np.float64),
    )


def query_radius(index: ParticleNeighborhoodIndex, lat: float, lon: float, radius_m: float) -> NeighborhoodResult:
    """Fixed-radius neighborhood query: every valid indexed particle within `radius_m`
    metres (inclusive) of `(lat, lon)`, in the index's own local projected coordinates.

    `radius_m` must be `> 0`. Returns an empty `NeighborhoodResult` if the index has no
    particles or none fall within the radius -- never raises for "no neighbors found".
    """
    if radius_m <= 0:
        raise ValueError(f"radius_m must be > 0; got {radius_m!r}")
    if index._tree is None:
        return _empty_result()

    qx, qy = project_to_local_metres(lat, lon, index.lat_ref, index.lon_ref, index.radius_m)
    query_point = np.array([qx[0], qy[0]])
    local_indices = np.asarray(index._tree.query_ball_point(query_point, r=radius_m), dtype=np.int64)
    if local_indices.size == 0:
        return _empty_result()

    points = np.column_stack([index.x_m[local_indices], index.y_m[local_indices]])
    distance_m = np.sqrt(np.sum((points - query_point) ** 2, axis=1))
    return _select(index, local_indices, distance_m)


def query_knn(index: ParticleNeighborhoodIndex, lat: float, lon: float, k: int) -> NeighborhoodResult:
    """k-nearest-neighbor query: the `k` valid indexed particles closest to `(lat, lon)`,
    in the index's own local projected coordinates. If fewer than `k` particles are
    indexed, returns only the ones that actually exist (never fabricates extra results).

    `k` must be `>= 1`.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1; got {k!r}")
    if index._tree is None or index.n_particles == 0:
        return _empty_result()

    qx, qy = project_to_local_metres(lat, lon, index.lat_ref, index.lon_ref, index.radius_m)
    query_point = np.array([qx[0], qy[0]])
    k_eff = min(k, index.n_particles)
    distance_m, local_indices = index._tree.query(query_point, k=k_eff)
    distance_m = np.atleast_1d(np.asarray(distance_m, dtype=np.float64))
    local_indices = np.atleast_1d(np.asarray(local_indices, dtype=np.int64))
    return _select(index, local_indices, distance_m)
