"""T30 -- Phase 3 initial-condition influence field (Mahalanobis/dispersion-relative).

Measures whether a candidate is statistically consistent with being an ordinary,
undifferentiated member of the SAME backward-propagated particle population at that same
timestamp `t`, rather than a genuinely separated spatial mode. This is the corrected design
(2026-09-24) after review identified that an earlier `exp(-age/temporal_scale)` formulation
mechanically favored older timestamps once subtracted in T36 (a monotonic-in-age function,
subtracted from every candidate identically, changes the ranking purely as a function of the
clock, regardless of actual evidence -- exactly the "do not force older timestamps to score
higher by construction" constraint this phase must respect). See
`notebooks/T30_initial_bias_check.ipynb` and the corresponding `progress.md`/`memory.md`
entries for the full review discussion.

**This module contains NO age term anywhere -- not explicitly, not implicitly.** The only
place backward time enters is through `mu(t)`/`Sigma(t)` being the ACTUAL MEASURED mean/
covariance of that timestamp's own valid particles (via T14's `query_particle_slice`,
unmodified) -- a descriptive fact about the data, not an assumed clock. Two candidates at
different ages with the same Mahalanobis distance from their own timestamp's centroid
receive IDENTICAL influence; there is no remaining mechanism by which age alone can shift
the score.

**Reference choice (distance to the CLOUD's own centroid at `t`, not to the fixed observed
slick):** the entire particle population at any timestamp already originated at the
observed slick (Stage III's own boundary condition), so `mu(t)` is literally the model's own
physical answer to "where would an undifferentiated continuation of the observed slick be,
on average, at time `t`, under the real currents used in this simulation" -- it already nets
out genuine bulk advective drift (confirmed up to ~48 km on the real dataset), which a fixed
reference point would NOT net out, reintroducing the exact same class of mechanical bias
(this time along a spatial axis instead of a temporal one) that this task exists to remove.

Formula:

    d_M(candidate, t) = sqrt( (x_c - mu(t))^T @ Sigma(t)^-1 @ (x_c - mu(t)) )
    influence(candidate, t) = exp( -0.5 * (d_M(candidate, t) / k)^2 )

- `x_c`, `mu(t)`: candidate and particle-cloud-centroid positions, in LOCAL PROJECTED
  METRES (via `iste.kdtree_neighborhood.project_to_local_metres`, the same convention
  T15/T17/T23 already use -- never mixing degrees and metres directly).
- `Sigma(t)`: the 2x2 covariance of that timestamp's valid particle positions, ALSO in
  local metres^2 (computed directly from metre-projected positions here, not reused from
  T28's `cov_lonlat`, which is in raw degrees^2 and would not be metrically meaningful for
  a Mahalanobis distance).
- `k` (default `1.0`): a dimensionless sigma-multiplier -- the ONLY free parameter, replacing
  the earlier two-constant (`spatial_scale_m`, `temporal_scale_hours`) design entirely. The
  `-0.5*(d_M/k)^2` form matches the natural relationship between Mahalanobis distance and a
  multivariate-Gaussian density (`d_M` is literally in standard-deviation units under that
  model), so `k=1` means "candidates within about one Mahalanobis standard deviation of the
  ensemble's own centroid at that instant are statistically consistent with pure
  undifferentiated continuation." PROTOTYPE/CONFIGURATION choice, not a validated form --
  fully overridable, documented alternative (`exp(-d_M/k)`, linear rather than quadratic)
  considered and rejected in favor of the Gaussian-density-consistent quadratic form.

**Singular/degenerate covariance fallback (explicit, tested, never a silent NaN-as-if-valid
or fabricated number):**

- `n_particles < 2`: no dispersion estimate is possible at all -- `covariance_state =
  "insufficient_particles"`, `mahalanobis_distance`/`influence` are `NaN`.
- `n_particles >= 2` but the covariance is singular or near-singular (its smaller eigenvalue
  is `<= singular_eig_ratio_tol` (default `1e-6`) times its larger eigenvalue -- e.g. exactly
  or nearly collinear particles): `covariance_state = "singular_fallback"`. Falls back to an
  ISOTROPIC variance `sigma_iso^2 = trace(Sigma(t)) / 2` (the mean single-axis variance),
  and `d_M` becomes the plain Euclidean distance divided by `sigma_iso` -- still expressed in
  "sigma units," still combined via the same `exp(-0.5*(d/k)^2)` formula, just without a
  directional (anisotropic) correction, which the data cannot support in this case.
- If even the isotropic fallback variance is (numerically) zero -- every particle coincides
  exactly -- `covariance_state = "zero_variance"`, `mahalanobis_distance`/`influence` are
  `NaN`. No dispersion-relative distance can be defined against a single point.
- Otherwise `covariance_state = "full_rank"`: the proper anisotropic Mahalanobis distance
  is used, via `numpy.linalg.solve` (not an explicit matrix inverse, for numerical
  stability).

This module is a diagnostic/correction INPUT only -- it does not reject, filter, or rank any
candidate. `NaN` influence values (the degenerate cases above) must be handled explicitly by
T36, never silently treated as 0 or as a passing value.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from iste.candidate_extraction import Candidate
from iste.ensemble_run import EnsembleRunRecords
from iste.kdtree_neighborhood import project_to_local_metres
from iste.particle_query import query_particle_slice

DEFAULT_K = 1.0
DEFAULT_SINGULAR_EIG_RATIO_TOL = 1e-6

_COVARIANCE_STATES = ("full_rank", "singular_fallback", "insufficient_particles", "zero_variance")


@dataclass(frozen=True)
class TimestampCloudStats:
    """The valid-particle mean/covariance at one Stage III historical timestamp, in local
    projected metres -- the reference distribution T30's Mahalanobis distance is measured
    against. Computed once per timestamp and reused across every candidate sharing that
    timestamp (T17 can retain multiple candidate modes per `t_s`)."""

    t_s: np.datetime64
    n_particles: int
    lat_ref: float
    lon_ref: float
    centroid_x_m: float
    centroid_y_m: float
    covariance_m2: np.ndarray | None  # 2x2, metres^2; None iff n_particles < 2
    covariance_state: str  # one of _COVARIANCE_STATES
    isotropic_variance_m2: float | None  # set iff covariance_state == "singular_fallback"


def compute_timestamp_cloud_stats(
    records: EnsembleRunRecords,
    t_s,
    *,
    singular_eig_ratio_tol: float = DEFAULT_SINGULAR_EIG_RATIO_TOL,
) -> TimestampCloudStats:
    """Compute the valid-particle mean/covariance at `t_s` from already-computed Stage III
    `records` (via T14's `query_particle_slice`, unmodified -- no rerun of Stage III).
    """
    if not (np.isfinite(singular_eig_ratio_tol) and singular_eig_ratio_tol > 0.0):
        raise ValueError(f"singular_eig_ratio_tol must be finite and > 0, got {singular_eig_ratio_tol}")

    t_s = np.datetime64(t_s, "ns")
    slice_ = query_particle_slice(records, t_s)
    valid = slice_.valid
    lat = slice_.lat[valid]
    lon = slice_.lon[valid]
    n_particles = int(len(lat))

    if n_particles == 0:
        lat_ref, lon_ref = float("nan"), float("nan")
    else:
        lat_ref, lon_ref = float(np.mean(lat)), float(np.mean(lon))

    if n_particles < 2:
        return TimestampCloudStats(
            t_s=t_s, n_particles=n_particles, lat_ref=lat_ref, lon_ref=lon_ref,
            centroid_x_m=float("nan"), centroid_y_m=float("nan"),
            covariance_m2=None, covariance_state="insufficient_particles",
            isotropic_variance_m2=None,
        )

    x_m, y_m = project_to_local_metres(lat, lon, lat_ref=lat_ref, lon_ref=lon_ref)
    centroid_x_m = float(np.mean(x_m))  # ~0.0 by construction (ref == particle mean)
    centroid_y_m = float(np.mean(y_m))
    covariance_m2 = np.cov(np.vstack([x_m, y_m]))

    eigvals = np.linalg.eigvalsh(covariance_m2)  # ascending: [min_eig, max_eig]
    min_eig, max_eig = float(eigvals[0]), float(eigvals[1])

    if max_eig <= 0.0:
        covariance_state = "zero_variance"
        isotropic_variance_m2 = None
    elif min_eig <= singular_eig_ratio_tol * max_eig:
        covariance_state = "singular_fallback"
        isotropic_variance_m2 = float(np.trace(covariance_m2)) / 2.0
        if isotropic_variance_m2 <= 0.0:
            covariance_state = "zero_variance"
            isotropic_variance_m2 = None
    else:
        covariance_state = "full_rank"
        isotropic_variance_m2 = None

    return TimestampCloudStats(
        t_s=t_s, n_particles=n_particles, lat_ref=lat_ref, lon_ref=lon_ref,
        centroid_x_m=centroid_x_m, centroid_y_m=centroid_y_m,
        covariance_m2=covariance_m2, covariance_state=covariance_state,
        isotropic_variance_m2=isotropic_variance_m2,
    )


@dataclass(frozen=True)
class InitialBiasInfluence:
    """The T30 initial-condition influence field evaluated at one candidate, against one
    timestamp's `TimestampCloudStats`."""

    candidate_lat: float
    candidate_lon: float
    t_s: np.datetime64
    covariance_state: str
    mahalanobis_distance: float  # NaN iff covariance_state in {"insufficient_particles", "zero_variance"}
    k: float
    influence: float  # NaN iff mahalanobis_distance is NaN; else in (0, 1]


def compute_initial_bias_influence(
    cloud_stats: TimestampCloudStats,
    candidate_lat: float,
    candidate_lon: float,
    *,
    k: float = DEFAULT_K,
) -> InitialBiasInfluence:
    """Evaluate the T30 influence field for one candidate against one timestamp's
    already-computed `TimestampCloudStats` (see `compute_timestamp_cloud_stats`).
    """
    if not (np.isfinite(k) and k > 0.0):
        raise ValueError(f"k must be finite and > 0, got {k}")

    if cloud_stats.covariance_state in ("insufficient_particles", "zero_variance"):
        return InitialBiasInfluence(
            candidate_lat=float(candidate_lat), candidate_lon=float(candidate_lon),
            t_s=cloud_stats.t_s, covariance_state=cloud_stats.covariance_state,
            mahalanobis_distance=float("nan"), k=float(k), influence=float("nan"),
        )

    x_c, y_c = project_to_local_metres(
        candidate_lat, candidate_lon, lat_ref=cloud_stats.lat_ref, lon_ref=cloud_stats.lon_ref
    )
    delta = np.array([x_c[0] - cloud_stats.centroid_x_m, y_c[0] - cloud_stats.centroid_y_m])

    if cloud_stats.covariance_state == "full_rank":
        d_m_sq = float(delta @ np.linalg.solve(cloud_stats.covariance_m2, delta))
        d_m = float(np.sqrt(max(d_m_sq, 0.0)))
    else:  # "singular_fallback"
        euclidean = float(np.sqrt(delta[0] ** 2 + delta[1] ** 2))
        d_m = euclidean / float(np.sqrt(cloud_stats.isotropic_variance_m2))

    influence = float(np.exp(-0.5 * (d_m / k) ** 2))

    return InitialBiasInfluence(
        candidate_lat=float(candidate_lat), candidate_lon=float(candidate_lon),
        t_s=cloud_stats.t_s, covariance_state=cloud_stats.covariance_state,
        mahalanobis_distance=d_m, k=float(k), influence=influence,
    )


def compute_initial_bias_influence_for_candidates(
    records: EnsembleRunRecords,
    candidates: list[Candidate],
    *,
    k: float = DEFAULT_K,
    singular_eig_ratio_tol: float = DEFAULT_SINGULAR_EIG_RATIO_TOL,
) -> list[InitialBiasInfluence]:
    """`compute_initial_bias_influence` evaluated for every T17 `Candidate` in
    `candidates`, in order. `TimestampCloudStats` is computed once per DISTINCT `t_s` and
    reused across every candidate sharing that timestamp (T17 can retain multiple modes per
    timestamp) -- no redundant Stage III queries."""
    stats_by_t_s: dict[np.datetime64, TimestampCloudStats] = {}
    results = []
    for c in candidates:
        t_s = np.datetime64(c.t_s, "ns")
        if t_s not in stats_by_t_s:
            stats_by_t_s[t_s] = compute_timestamp_cloud_stats(
                records, t_s, singular_eig_ratio_tol=singular_eig_ratio_tol
            )
        results.append(
            compute_initial_bias_influence(stats_by_t_s[t_s], c.y_peak, c.x_peak, k=k)
        )
    return results
