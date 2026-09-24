"""T28 -- Phase 3 per-timestamp particle-distribution and support/"KDE" diagnostics.

For every analysis timestamp (T27's `generate_analysis_grid` output, or any other
timestamp), computes a fixed set of diagnostics describing (a) the raw valid-particle
distribution and (b) the existing T16 support/"KDE" heat map at that instant. This module
never reruns Stage III transport, never redefines particle validity, and never invents a
second density or peak-detection algorithm:

- the particle slice is T14's existing, unmodified `query_particle_slice`;
- the neighborhood index is T15's existing, unmodified `build_particle_neighborhood_index`;
- the "KDE"/support field is T16's existing, unmodified `compute_support_map`
  (`Support(x,y,t) = f(D,W,V)`) -- this project has no separate literal Gaussian KDE, and
  this module does not add one. `kde_bandwidth_m` below is simply the neighborhood radius
  already passed to `compute_support_map` (T16's own smoothing-scale parameter), echoed
  through for the diagnostic table -- not a new bandwidth concept;
- mode counting reuses T17's existing, unmodified `find_candidate_peaks` directly (the
  candidate count IS `n_significant_modes` -- no second peak detector).

Per `README.md`/`CLAUDE.md`'s "not a probability density" discipline (see
`iste.support_features`'s own module docstring), `kde_entropy`/`kde_effective_area_m2`
below are plain descriptive-statistics summaries of the *normalized, clipped-nonnegative*
support grid (an entropy-of-a-distribution calculation applied to that grid, exactly as one
would to any nonnegative array that sums to something positive) -- they are explicitly NOT
claims that `Support` is a validated probability density, only a convenient, well-defined,
unit-consistent way to summarize how concentrated vs. spread out that grid is at each
timestamp. This is exactly the kind of diagnostic Phase 3 needs to *see* the initial-
condition KDE-concentration bias the phase exists to correct (T29+); it does not correct
anything by itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from iste.candidate_extraction import CandidateExtractionConfig, find_candidate_peaks
from iste.ensemble_run import EnsembleRunRecords
from iste.geometry import metres_per_degree_latitude, metres_per_degree_longitude
from iste.kdtree_neighborhood import build_particle_neighborhood_index, project_to_local_metres
from iste.particle_query import query_particle_slice
from iste.support_features import SupportFeatureConfig, compute_support_map


@dataclass
class TimestampDiagnostics:
    """All T28-required diagnostics for one analysis timestamp. Every field is a plain
    descriptive statistic of the valid-particle distribution and/or the T16 support grid
    at this instant -- none of them is a ranking score (that is T29+)."""

    t_s: np.datetime64
    n_particles: int

    # raw valid-particle distribution
    lon_mean: float
    lat_mean: float
    lon_var: float
    lat_var: float
    cov_lonlat: np.ndarray  # 2x2, units deg^2; NaN-filled if n_particles < 2
    spatial_spread_m: float  # RMS distance of valid particles from their own centroid
    convex_hull_area_m2: float | None  # None if n_particles < 3 or points are degenerate

    # T16 support/"KDE" grid summaries
    kde_max: float
    kde_integrated_mass: float
    kde_entropy: float  # nats; entropy of the normalized, clipped-nonnegative support grid
    kde_effective_area_m2: float  # exp(entropy) * grid_cell_area_m2
    kde_bandwidth_m: float  # the neighborhood_radius_m used to build the support grid
    n_significant_modes: int  # len(T17's find_candidate_peaks(...)) on this grid


def _grid_cell_area_m2(lat_grid: np.ndarray, lon_grid: np.ndarray) -> float:
    """Approximate area of one regular support-grid cell, in m^2, evaluated at the grid's
    mean latitude (small-area equirectangular approximation, same convention T15/T03 use
    elsewhere -- never mixing degrees and metres directly). Assumes (and does not verify
    beyond `np.diff`) a regularly-spaced grid, which is how every existing Phase 2 caller
    (T17/T26/the real-data notebook) already constructs `support_lat_grid`/
    `support_lon_grid`."""
    if len(lat_grid) < 2 or len(lon_grid) < 2:
        return float("nan")
    dlat_deg = float(np.mean(np.diff(lat_grid)))
    dlon_deg = float(np.mean(np.diff(lon_grid)))
    lat_mean = float(np.mean(lat_grid))
    m_per_deg_lat = metres_per_degree_latitude()
    m_per_deg_lon = metres_per_degree_longitude(lat_mean)
    return abs(dlat_deg) * m_per_deg_lat * abs(dlon_deg) * m_per_deg_lon


def compute_timestamp_diagnostics(
    records: EnsembleRunRecords,
    t_s,
    *,
    support_lat_grid: np.ndarray,
    support_lon_grid: np.ndarray,
    neighborhood_radius_m: float,
    support_config: SupportFeatureConfig | None = None,
    extraction_config: CandidateExtractionConfig | None = None,
) -> TimestampDiagnostics:
    """Compute every T28 diagnostic for one historical timestamp `t_s`, from already-
    computed Stage III `records` (T09/`run_ensemble` output). Does not rerun Stage III.

    `support_lat_grid`/`support_lon_grid` and `neighborhood_radius_m` are the same explicit
    parameters every existing Phase 2 caller (T17/T26) already supplies -- this task
    invents no new grid/radius convention.
    """
    t_s = np.datetime64(t_s, "ns")
    if support_config is None:
        support_config = SupportFeatureConfig()
    if extraction_config is None:
        extraction_config = CandidateExtractionConfig()

    slice_ = query_particle_slice(records, t_s)  # T14
    valid_mask = slice_.valid
    lat = slice_.lat[valid_mask]
    lon = slice_.lon[valid_mask]
    n_particles = int(len(lat))

    if n_particles == 0:
        lon_mean = lat_mean = lon_var = lat_var = float("nan")
        cov_lonlat = np.full((2, 2), np.nan)
        spatial_spread_m = float("nan")
        convex_hull_area_m2 = None
    else:
        lon_mean = float(np.mean(lon))
        lat_mean = float(np.mean(lat))
        lon_var = float(np.var(lon))
        lat_var = float(np.var(lat))
        if n_particles >= 2:
            cov_lonlat = np.cov(np.vstack([lon, lat]))
        else:
            cov_lonlat = np.full((2, 2), np.nan)

        x_m, y_m = project_to_local_metres(lat, lon, lat_ref=lat_mean, lon_ref=lon_mean)
        spatial_spread_m = float(np.sqrt(np.mean(x_m**2 + y_m**2)))

        convex_hull_area_m2 = None
        if n_particles >= 3:
            try:
                hull = ConvexHull(np.column_stack([x_m, y_m]))
                convex_hull_area_m2 = float(hull.volume)  # 2-D ConvexHull: .volume == area
            except QhullError:
                convex_hull_area_m2 = None  # degenerate (collinear/coincident) points

    # T15 -> T16: the existing, unmodified support/"KDE" grid at this timestamp.
    index = build_particle_neighborhood_index(slice_)
    lon_mesh, lat_mesh = np.meshgrid(support_lon_grid, support_lat_grid)
    support_map = compute_support_map(
        index, lat_mesh.ravel(), lon_mesh.ravel(),
        radius_m=neighborhood_radius_m, config=support_config,
    )
    support_grid = np.array([sf.support for sf in support_map], dtype=np.float64).reshape(
        len(support_lat_grid), len(support_lon_grid)
    )

    kde_max = float(np.max(support_grid)) if support_grid.size else float("nan")

    cell_area_m2 = _grid_cell_area_m2(support_lat_grid, support_lon_grid)
    clipped = np.clip(support_grid, 0.0, None)
    total = float(np.sum(clipped))
    kde_integrated_mass = total * cell_area_m2

    if total > 0.0:
        p = clipped / total
        nz = p > 0.0
        kde_entropy = float(-np.sum(p[nz] * np.log(p[nz])))
    else:
        kde_entropy = 0.0
    kde_effective_area_m2 = float(np.exp(kde_entropy)) * cell_area_m2

    # T17: reuse the existing peak/mode extractor directly -- its candidate count IS the
    # number of significant modes; no second peak-detection algorithm.
    candidates = find_candidate_peaks(
        support_map, support_lat_grid, support_lon_grid, t_s, config=extraction_config
    )
    n_significant_modes = len(candidates)

    return TimestampDiagnostics(
        t_s=t_s,
        n_particles=n_particles,
        lon_mean=lon_mean, lat_mean=lat_mean, lon_var=lon_var, lat_var=lat_var,
        cov_lonlat=cov_lonlat, spatial_spread_m=spatial_spread_m,
        convex_hull_area_m2=convex_hull_area_m2,
        kde_max=kde_max, kde_integrated_mass=kde_integrated_mass, kde_entropy=kde_entropy,
        kde_effective_area_m2=kde_effective_area_m2, kde_bandwidth_m=float(neighborhood_radius_m),
        n_significant_modes=n_significant_modes,
    )


def compute_diagnostics_table(
    records: EnsembleRunRecords,
    analysis_grid: np.ndarray,
    *,
    support_lat_grid: np.ndarray,
    support_lon_grid: np.ndarray,
    neighborhood_radius_m: float,
    support_config: SupportFeatureConfig | None = None,
    extraction_config: CandidateExtractionConfig | None = None,
) -> list[TimestampDiagnostics]:
    """`compute_timestamp_diagnostics` evaluated at every timestamp in `analysis_grid`
    (T27's `generate_analysis_grid` output, in order) -- the Stage 13/T28 diagnostic
    table."""
    return [
        compute_timestamp_diagnostics(
            records, t_s,
            support_lat_grid=support_lat_grid, support_lon_grid=support_lon_grid,
            neighborhood_radius_m=neighborhood_radius_m,
            support_config=support_config, extraction_config=extraction_config,
        )
        for t_s in analysis_grid
    ]
