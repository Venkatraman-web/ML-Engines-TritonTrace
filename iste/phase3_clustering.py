"""T37 -- Phase 3 spatial clustering of candidates.

Groups T36-scored candidates into distinct spatial regions, so the final Phase 3 output is
a small set of CLUSTERS (per README.md/the master spec's Stage 10: "rank CLUSTERS rather
than individual pixels"), not many near-duplicate points at slightly different positions/
timestamps.

**Distance metric.** Pairwise distances use the SAME local-tangent-plane projection
convention T15/T17/T30/T31 already use throughout this project
(`iste.kdtree_neighborhood.project_to_local_metres`, reused unchanged) -- one shared
reference point (the mean of all candidate positions being clustered), not a new geodesic
formula. This is consistent with, not more/less accurate than, every other distance
computation in this codebase.

**Why DBSCAN, not hierarchical (explicit, documented choice, per the task's own
requirement):**

- DBSCAN's core semantics -- transitively connect points within `eps` of each other -- is
  exactly "collapse near-duplicate candidates, keep well-separated candidates distinct",
  without needing to pre-specify a target cluster count.
- `min_samples=1` is used deliberately: every candidate must end up in SOME cluster --
  Phase 2/3's standing policy throughout this project is to never silently drop a candidate
  (T14/T17/T31 all preserve every input explicitly). With `min_samples=1`, DBSCAN never
  labels a point as noise; a genuinely isolated candidate simply becomes its own
  single-member cluster.
- With `min_samples=1`, DBSCAN's result is mathematically IDENTICAL to single-linkage
  hierarchical clustering cut at the same distance threshold (`scipy.cluster.hierarchy.
  fcluster(..., criterion="distance")`) -- both reduce to the same connected-components-
  within-`eps` partition. DBSCAN (`scikit-learn`, already an available dependency) was
  chosen for its direct, single-call API for this exact use case; this is not claimed to
  produce a scientifically different grouping from the hierarchical alternative.

**Evidence aggregation.** Every per-cluster evidence field is aggregated (mean, or peak =
max) from the cluster's member candidates' ALREADY-COMPUTED T36 `SourceScore` objects -- no
evidence is recomputed, no new normalization is introduced. Per T36's own established rule
(T29/T34 are one evidence axis), this module reports exactly ONE "normalized KDE"
field (from `SourceScore.support_evidence`) -- it does NOT separately report a "screening
score" as if it were independent evidence, since that would silently reintroduce the exact
double-counting T36 was built to avoid.

**Confidence/uncertainty ("from T38").** T38 (bootstrap/stability analysis) has not been
implemented yet. Per CLAUDE.md's "do not implement future stages early," this module does
NOT fabricate a bootstrap-derived confidence number. It reports the raw cluster geometry
(`spatial_radius_m`, `n_members`, `temporal_span_hours`) T38 will need, and leaves
`confidence_uncertainty_note` as an explicit statement that this is deferred, not a
placeholder score standing in for a real one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import ConvexHull, QhullError
from sklearn.cluster import DBSCAN

from iste.candidate_extraction import Candidate
from iste.kdtree_neighborhood import project_to_local_metres
from iste.phase3_composite import SourceScore

DEFAULT_EPS_M = 2000.0  # reused, not re-derived: T31's own spatial_matching_radius_m
CONFIDENCE_DEFERRED_NOTE = (
    "Confidence/uncertainty quantification is T38's own task (bootstrap/stability "
    "analysis), not yet implemented. This field intentionally reports no fabricated "
    "number; spatial_radius_m/n_members/temporal_span_hours are provided as the raw "
    "geometry T38 will need."
)


@dataclass(frozen=True)
class CandidateCluster:
    """One T37 spatial cluster of candidates, with per-cluster evidence aggregated from
    the member candidates' already-computed T36 `SourceScore` objects."""

    cluster_id: str
    member_candidate_ids: list[str]
    n_members: int

    centroid_lat: float
    centroid_lon: float
    spatial_radius_m: float  # max distance from centroid to any member
    area_m2: float | None  # convex hull area; None if <3 members or degenerate (collinear)

    total_particle_support: float  # sum of member density_score (T16 D)
    supported_timestamps: list[np.datetime64]  # sorted distinct t_s among members
    n_distinct_timestamps: int
    earliest_t_s: np.datetime64
    latest_t_s: np.datetime64
    temporal_span_hours: float

    mean_normalized_kde: float  # T29/T34's support_evidence -- ONE axis, see module docstring
    peak_normalized_kde: float
    mean_convergence_score: float
    mean_dispersion_score: float
    mean_initial_bias_influence: float
    mean_ais_score: float | None  # None iff no member has AIS available

    peak_source_score: float  # this cluster's "final source score" -- its best member
    mean_source_score: float

    confidence_uncertainty_note: str = CONFIDENCE_DEFERRED_NOTE


def _pairwise_distance_matrix_m(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    lat_ref, lon_ref = float(np.mean(lats)), float(np.mean(lons))
    x_m, y_m = project_to_local_metres(lats, lons, lat_ref=lat_ref, lon_ref=lon_ref)
    coords = np.column_stack([x_m, y_m])
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt(np.sum(diff**2, axis=-1))


def cluster_candidates(
    candidates: list[Candidate],
    source_scores: list[SourceScore],
    *,
    eps_m: float = DEFAULT_EPS_M,
    min_samples: int = 1,
) -> list[CandidateCluster]:
    """Cluster `candidates` spatially (DBSCAN, precomputed local-metre distance matrix,
    see module docstring) and aggregate each cluster's evidence from the parallel
    `source_scores` list (`source_scores[i]` must correspond to `candidates[i]`).

    Parameters
    ----------
    candidates, source_scores : parallel lists, same length and order
        `source_scores[i].candidate_id` must equal `candidates[i].candidate_id` (checked
        explicitly, never silently mismatched).
    eps_m : float
        DBSCAN's distance threshold, metres. Default reuses T31's own
        `spatial_matching_radius_m` (2 km) -- see module docstring.
    min_samples : int
        DBSCAN's `min_samples`. Default `1` so no candidate is ever dropped as noise.

    Returns
    -------
    list[CandidateCluster]
        Sorted by `peak_source_score`, descending (ranked clusters).
    """
    if len(candidates) != len(source_scores):
        raise ValueError(
            f"candidates ({len(candidates)}) and source_scores ({len(source_scores)}) "
            "must have the same length"
        )
    for i, (c, s) in enumerate(zip(candidates, source_scores)):
        if c.candidate_id != s.candidate_id:
            raise ValueError(
                f"candidates[{i}].candidate_id ({c.candidate_id!r}) does not match "
                f"source_scores[{i}].candidate_id ({s.candidate_id!r})"
            )
    if not (np.isfinite(eps_m) and eps_m > 0.0):
        raise ValueError(f"eps_m must be finite and > 0, got {eps_m}")
    if min_samples < 1:
        raise ValueError(f"min_samples must be >= 1, got {min_samples}")

    if not candidates:
        return []

    lats = np.array([c.y_peak for c in candidates], dtype=np.float64)
    lons = np.array([c.x_peak for c in candidates], dtype=np.float64)

    if len(candidates) == 1:
        labels = np.array([0])
    else:
        distance_matrix = _pairwise_distance_matrix_m(lats, lons)
        labels = DBSCAN(eps=eps_m, min_samples=min_samples, metric="precomputed").fit_predict(
            distance_matrix
        )

    clusters: list[CandidateCluster] = []
    for cluster_num, label in enumerate(sorted(set(labels.tolist()))):
        member_idx = np.flatnonzero(labels == label)
        member_candidates = [candidates[i] for i in member_idx]
        member_scores = [source_scores[i] for i in member_idx]
        n_members = len(member_idx)

        member_lats = lats[member_idx]
        member_lons = lons[member_idx]
        centroid_lat = float(np.mean(member_lats))
        centroid_lon = float(np.mean(member_lons))

        x_m, y_m = project_to_local_metres(
            member_lats, member_lons, lat_ref=centroid_lat, lon_ref=centroid_lon
        )
        spatial_radius_m = float(np.max(np.sqrt(x_m**2 + y_m**2))) if n_members > 0 else 0.0

        area_m2 = None
        if n_members >= 3:
            try:
                hull = ConvexHull(np.column_stack([x_m, y_m]))
                area_m2 = float(hull.volume)  # 2-D ConvexHull: .volume == area
            except QhullError:
                area_m2 = None  # degenerate (collinear/coincident) points

        total_particle_support = float(sum(c.density_score for c in member_candidates))

        t_s_values = sorted({np.datetime64(c.t_s, "ns") for c in member_candidates})
        earliest_t_s, latest_t_s = t_s_values[0], t_s_values[-1]
        temporal_span_hours = float((latest_t_s - earliest_t_s) / np.timedelta64(1, "h"))

        normalized_kde_vals = [s.support_evidence for s in member_scores]
        convergence_vals = [s.convergence_evidence for s in member_scores]
        dispersion_vals = [s.dispersion_evidence for s in member_scores]
        initial_bias_vals = [s.initial_bias_evidence for s in member_scores]
        ais_vals = [s.ais_evidence for s in member_scores if s.ais_evidence is not None]
        source_score_vals = [s.source_score for s in member_scores]

        clusters.append(
            CandidateCluster(
                cluster_id=f"cluster{cluster_num:04d}",
                member_candidate_ids=[c.candidate_id for c in member_candidates],
                n_members=n_members,
                centroid_lat=centroid_lat, centroid_lon=centroid_lon,
                spatial_radius_m=spatial_radius_m, area_m2=area_m2,
                total_particle_support=total_particle_support,
                supported_timestamps=t_s_values, n_distinct_timestamps=len(t_s_values),
                earliest_t_s=earliest_t_s, latest_t_s=latest_t_s,
                temporal_span_hours=temporal_span_hours,
                mean_normalized_kde=float(np.mean(normalized_kde_vals)),
                peak_normalized_kde=float(np.max(normalized_kde_vals)),
                mean_convergence_score=float(np.mean(convergence_vals)),
                mean_dispersion_score=float(np.mean(dispersion_vals)),
                mean_initial_bias_influence=float(np.mean(initial_bias_vals)),
                mean_ais_score=float(np.mean(ais_vals)) if ais_vals else None,
                peak_source_score=float(np.max(source_score_vals)),
                mean_source_score=float(np.mean(source_score_vals)),
            )
        )

    clusters.sort(key=lambda cl: -cl.peak_source_score)
    return clusters
