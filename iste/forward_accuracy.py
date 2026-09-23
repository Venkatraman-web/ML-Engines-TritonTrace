"""T23 -- Stage VII: forward-vs-observed accuracy (per candidate/forcing member).

Implements exactly `README.md` Sec. 22: for one candidate/forcing member, compute an
explicit, configurable spatial-consistency error/score between its T22
`PredictedSlickFootprint` and the independent T22 `ObservedSlickFootprint` --

    lower error = better candidate
    higher consistency = better candidate      (both always unambiguous in direction)

Reuses T22's already-built footprint/grid representation directly -- this module never
reimplements or re-evaluates a KDE; it only reads `PredictedSlickFootprint.density`/
`ObservedSlickFootprint.density` (both already evaluated on the SAME shared grid, per
T22's own contract) and `iste.geometry`'s existing metre-per-degree conversions /
`iste.kdtree_neighborhood.project_to_local_metres` for physically-correct area/distance
arithmetic.

**Metric families implemented** (`README.md` Sec. 22's list, "where practical"):

    footprint_overlap       -- area-weighted Intersection-over-Union of the two
                                footprints' thresholded regions
    observed_coverage       -- fraction of the OBSERVED region's area also covered by
                                the predicted region (a recall-style score)
    false_positive_area     -- fraction of the PREDICTED region's area that lies
                                OUTSIDE the observed region
    spatial_distance        -- metres between the two footprints' density-weighted
                                centroids (via `project_to_local_metres`), with an
                                explicit, configurable exponential decay turning that
                                distance into a consistency score
    combined_overlap_fp     -- a documented weighted combination of `footprint_overlap`
                                and `false_positive_area` (both already individually
                                implemented/tested standalone -- this metric does not
                                duplicate their logic, only combines their already-
                                computed values)

Every metric is a PROTOTYPE/CONFIGURATION choice (`README.md` Sec. 22: "the Phase 2
comparison metric is a configurable calibration component"), not a claim of scientific
validation -- `ComparisonConfig.metric` selects which one runs, and every threshold/
scale parameter is an explicit, validated field, never a hidden constant.

**Region extraction** (needed for the three area-based metrics): a footprint's own
`density` grid is thresholded at `region_threshold_fraction * nanmax(density)` --
relative to ITS OWN peak, not a shared absolute density value (predicted and observed
footprints can have very different absolute density scales, e.g. from different
particle counts or KDE bandwidths before T22's own count-normalization). This threshold
is an explicit `ComparisonConfig` field.

**Invalid handling.** If either footprint's `has_valid_particles` is `False`, the
comparison is explicitly `valid=False` with `error`/`consistency` both `NaN` and a
recorded `invalid_reason` -- never silently scored as a good (or bad) match
(`README.md`/this task: "do not silently treat invalid forward results as a good
match").

Does not aggregate across forcing members (T24), filter/rank candidates (T25), implement
AIS or formal source inference, or read/reference Stage III `c_hat*`. Does not modify
`iste.slick_footprint` (T22) or any earlier Phase 2 module.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from iste.geometry import MEAN_EARTH_RADIUS_M, metres_per_degree_latitude, metres_per_degree_longitude
from iste.kdtree_neighborhood import project_to_local_metres
from iste.slick_footprint import ObservedSlickFootprint, PredictedSlickFootprint

KNOWN_METRICS = (
    "footprint_overlap",
    "observed_coverage",
    "false_positive_area",
    "spatial_distance",
    "combined_overlap_fp",
)


@dataclass(frozen=True)
class ComparisonConfig:
    """Explicit, validated Phase 2 comparison configuration.

    - `metric`: one of `KNOWN_METRICS` (required, no default -- the metric choice is
      always an explicit caller decision, never a hidden default).
    - `region_threshold_fraction`: fraction of a footprint's OWN peak density (`(0, 1]`)
      defining its thresholded "region" for the three area-based metrics. Default
      `0.1` (10% of peak) -- a PROTOTYPE calibration choice, not a validated value.
    - `consistency_distance_scale_m`: length scale (metres, `> 0`) for
      `spatial_distance`'s `consistency = exp(-distance_m / scale)` transform. Default
      `50_000.0` (50 km) -- PROTOTYPE, freely overridable.
    - `combination_weight_overlap`/`combination_weight_fp`: non-negative weights for
      `combined_overlap_fp`'s `consistency = w_overlap*IoU + w_fp*(1 - false_positive_
      fraction)`. Defaults `0.5`/`0.5`. Not forced to sum to 1 (an explicit caller
      choice), but a sum of 1 keeps `consistency` in `[0, 1]`.
    """

    metric: str
    region_threshold_fraction: float = 0.1
    consistency_distance_scale_m: float = 50_000.0
    combination_weight_overlap: float = 0.5
    combination_weight_fp: float = 0.5

    def __post_init__(self) -> None:
        if self.metric not in KNOWN_METRICS:
            raise ValueError(f"metric must be one of {KNOWN_METRICS}; got {self.metric!r}")
        if not (0.0 < self.region_threshold_fraction <= 1.0):
            raise ValueError(f"region_threshold_fraction must be in (0, 1]; got {self.region_threshold_fraction!r}")
        if not np.isfinite(self.consistency_distance_scale_m) or self.consistency_distance_scale_m <= 0.0:
            raise ValueError(
                f"consistency_distance_scale_m must be finite and > 0; got {self.consistency_distance_scale_m!r}"
            )
        if self.combination_weight_overlap < 0.0 or self.combination_weight_fp < 0.0:
            raise ValueError("combination_weight_overlap/combination_weight_fp must be >= 0")


@dataclass
class ForwardAccuracyResult:
    """One candidate/member's forward-vs-observed comparison result.

    `error`/`consistency` are both `NaN` iff `valid` is `False` (see module docstring)
    -- never a fabricated numeric score for an undefined comparison. When `valid` is
    `True`, `error` and `consistency` are always both defined and mutually consistent
    (lower error <=> higher consistency) for every metric implemented here.
    """

    candidate_id: str
    t_s: np.datetime64
    t_obs: np.datetime64
    forward_forcing_member_id: int
    metric: str
    config: ComparisonConfig
    error: float
    consistency: float
    valid: bool
    invalid_reason: str


def _footprint_region_mask(density: np.ndarray, threshold_fraction: float) -> np.ndarray:
    """Boolean mask of cells at or above `threshold_fraction` of `density`'s own peak.
    All-`False` if `density` is entirely `NaN` or its peak is `<= 0` (degenerate)."""
    if np.all(np.isnan(density)):
        return np.zeros(density.shape, dtype=bool)
    peak = np.nanmax(density)
    if not np.isfinite(peak) or peak <= 0.0:
        return np.zeros(density.shape, dtype=bool)
    filled = np.nan_to_num(density, nan=0.0)
    return filled >= threshold_fraction * peak


def _grid_cell_areas_m2(lat_grid: np.ndarray, lon_grid: np.ndarray, radius_m: float = MEAN_EARTH_RADIUS_M) -> np.ndarray:
    """Physical area (m^2) of every grid cell of a regular `lat_grid x lon_grid`, shape
    `(len(lat_grid), len(lon_grid))` -- accounts for longitude-degree spacing shrinking
    at higher latitude (`iste.geometry.metres_per_degree_longitude`), never treating
    degrees as uniform metres. Per-point spacing is estimated via `numpy.gradient` (
    correct for a uniform grid, a reasonable local estimate at the edges of a
    near-uniform one)."""
    lat_grid = np.asarray(lat_grid, dtype=np.float64)
    lon_grid = np.asarray(lon_grid, dtype=np.float64)
    dlat_deg = np.gradient(lat_grid) if len(lat_grid) > 1 else np.array([1.0])
    dlon_deg = np.gradient(lon_grid) if len(lon_grid) > 1 else np.array([1.0])
    cell_height_m = np.abs(dlat_deg) * metres_per_degree_latitude(radius_m)  # (n_lat,)
    cell_width_m = np.abs(dlon_deg[None, :]) * metres_per_degree_longitude(lat_grid, radius_m)[:, None]  # (n_lat, n_lon)
    return cell_height_m[:, None] * cell_width_m


def footprint_overlap_iou(pred_mask: np.ndarray, obs_mask: np.ndarray, cell_area_m2: np.ndarray) -> float:
    """Area-weighted Intersection-over-Union of two region masks. `NaN` if the union has
    zero area (both masks empty)."""
    union_area = np.sum(cell_area_m2[pred_mask | obs_mask])
    if union_area <= 0.0:
        return float("nan")
    intersection_area = np.sum(cell_area_m2[pred_mask & obs_mask])
    return float(intersection_area / union_area)


def observed_region_coverage(pred_mask: np.ndarray, obs_mask: np.ndarray, cell_area_m2: np.ndarray) -> float:
    """Fraction of the OBSERVED region's area also covered by the predicted region
    (recall-style). `NaN` if the observed region has zero area."""
    obs_area = np.sum(cell_area_m2[obs_mask])
    if obs_area <= 0.0:
        return float("nan")
    covered_area = np.sum(cell_area_m2[pred_mask & obs_mask])
    return float(covered_area / obs_area)


def predicted_false_positive_fraction(pred_mask: np.ndarray, obs_mask: np.ndarray, cell_area_m2: np.ndarray) -> float:
    """Fraction of the PREDICTED region's area that lies OUTSIDE the observed region.
    `NaN` if the predicted region has zero area."""
    pred_area = np.sum(cell_area_m2[pred_mask])
    if pred_area <= 0.0:
        return float("nan")
    fp_area = np.sum(cell_area_m2[pred_mask & ~obs_mask])
    return float(fp_area / pred_area)


def _density_weighted_centroid(density: np.ndarray, lat_grid: np.ndarray, lon_grid: np.ndarray) -> tuple[float, float] | None:
    """`(lat, lon)` density-weighted centroid of a footprint (`NaN` cells treated as
    zero weight). `None` if the total weight is zero (undefined centroid)."""
    if np.all(np.isnan(density)):
        return None
    weight = np.nan_to_num(density, nan=0.0)
    total = np.sum(weight)
    if total <= 0.0:
        return None
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
    lat_c = float(np.sum(weight * lat_mesh) / total)
    lon_c = float(np.sum(weight * lon_mesh) / total)
    return lat_c, lon_c


def spatial_distance_m(
    predicted: PredictedSlickFootprint, observed: ObservedSlickFootprint, radius_m: float = MEAN_EARTH_RADIUS_M
) -> float | None:
    """Metres between the predicted and observed footprints' density-weighted
    centroids, via `iste.kdtree_neighborhood.project_to_local_metres` (never raw
    lat/lon degree differences). `None` if either centroid is undefined."""
    pred_c = _density_weighted_centroid(predicted.density, predicted.lat_grid, predicted.lon_grid)
    obs_c = _density_weighted_centroid(observed.density, observed.lat_grid, observed.lon_grid)
    if pred_c is None or obs_c is None:
        return None
    x, y = project_to_local_metres(obs_c[0], obs_c[1], lat_ref=pred_c[0], lon_ref=pred_c[1], radius_m=radius_m)
    return float(np.sqrt(x[0] ** 2 + y[0] ** 2))


def _validate_matching_grids(predicted: PredictedSlickFootprint, observed: ObservedSlickFootprint) -> None:
    if not np.array_equal(predicted.lat_grid, observed.lat_grid) or not np.array_equal(predicted.lon_grid, observed.lon_grid):
        raise ValueError(
            "predicted and observed footprints must share the exact same lat_grid/lon_grid "
            "(build both from the same iste.slick_footprint.FootprintGridConfig)"
        )


def compare_predicted_to_observed(
    predicted: PredictedSlickFootprint, observed: ObservedSlickFootprint, config: ComparisonConfig,
) -> ForwardAccuracyResult:
    """`README.md` Sec. 22`'s per-candidate/member comparison: one explicit,
    configurable spatial-consistency error/score between `predicted` and `observed`.

    Requires `predicted.lat_grid`/`lon_grid` to exactly equal `observed`'s (T22's own
    "same `FootprintGridConfig`" convention) -- raises `ValueError` otherwise, since
    comparing footprints on different grids would be meaningless.
    """
    _validate_matching_grids(predicted, observed)

    base_kwargs = dict(
        candidate_id=predicted.candidate_id,
        t_s=predicted.t_s,
        t_obs=predicted.t_obs,
        forward_forcing_member_id=predicted.forward_forcing_member_id,
        metric=config.metric,
        config=config,
    )

    if not predicted.has_valid_particles:
        return ForwardAccuracyResult(
            **base_kwargs, error=float("nan"), consistency=float("nan"),
            valid=False, invalid_reason="predicted footprint has no valid particles",
        )
    if not observed.has_valid_particles:
        return ForwardAccuracyResult(
            **base_kwargs, error=float("nan"), consistency=float("nan"),
            valid=False, invalid_reason="observed reference has no valid particles",
        )

    if config.metric == "spatial_distance":
        distance_m = spatial_distance_m(predicted, observed)
        if distance_m is None:
            return ForwardAccuracyResult(
                **base_kwargs, error=float("nan"), consistency=float("nan"),
                valid=False, invalid_reason="undefined density-weighted centroid",
            )
        error = distance_m
        consistency = float(np.exp(-distance_m / config.consistency_distance_scale_m))
        return ForwardAccuracyResult(**base_kwargs, error=error, consistency=consistency, valid=True, invalid_reason="")

    cell_area_m2 = _grid_cell_areas_m2(predicted.lat_grid, predicted.lon_grid)
    pred_mask = _footprint_region_mask(predicted.density, config.region_threshold_fraction)
    obs_mask = _footprint_region_mask(observed.density, config.region_threshold_fraction)

    if config.metric == "footprint_overlap":
        iou = footprint_overlap_iou(pred_mask, obs_mask, cell_area_m2)
        consistency, error = iou, (1.0 - iou if np.isfinite(iou) else float("nan"))
    elif config.metric == "observed_coverage":
        coverage = observed_region_coverage(pred_mask, obs_mask, cell_area_m2)
        consistency, error = coverage, (1.0 - coverage if np.isfinite(coverage) else float("nan"))
    elif config.metric == "false_positive_area":
        fp_fraction = predicted_false_positive_fraction(pred_mask, obs_mask, cell_area_m2)
        error, consistency = fp_fraction, (1.0 - fp_fraction if np.isfinite(fp_fraction) else float("nan"))
    elif config.metric == "combined_overlap_fp":
        iou = footprint_overlap_iou(pred_mask, obs_mask, cell_area_m2)
        fp_fraction = predicted_false_positive_fraction(pred_mask, obs_mask, cell_area_m2)
        if not (np.isfinite(iou) and np.isfinite(fp_fraction)):
            consistency, error = float("nan"), float("nan")
        else:
            consistency = config.combination_weight_overlap * iou + config.combination_weight_fp * (1.0 - fp_fraction)
            error = 1.0 - consistency
    else:  # pragma: no cover -- unreachable, ComparisonConfig validates `metric`
        raise AssertionError(f"unhandled metric {config.metric!r}")

    if not np.isfinite(consistency):
        return ForwardAccuracyResult(
            **base_kwargs, error=float("nan"), consistency=float("nan"),
            valid=False, invalid_reason="metric undefined (degenerate empty region)",
        )
    return ForwardAccuracyResult(**base_kwargs, error=float(error), consistency=float(consistency), valid=True, invalid_reason="")


def compare_predicted_footprints_to_observed(
    predicted_footprints: dict[int, PredictedSlickFootprint], observed: ObservedSlickFootprint, config: ComparisonConfig,
) -> dict[int, ForwardAccuracyResult]:
    """`compare_predicted_to_observed` for every member in `predicted_footprints` (e.g.
    `iste.slick_footprint.build_predicted_slick_footprints`'s output) against the SAME
    `observed` reference, keyed by `forward_forcing_member_id`."""
    return {
        member_id: compare_predicted_to_observed(predicted, observed, config)
        for member_id, predicted in predicted_footprints.items()
    }
