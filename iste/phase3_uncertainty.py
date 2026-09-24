"""T38 -- Phase 3 uncertainty / bootstrap stability layer.

Adds an UNCERTAINTY layer on top of the already-computed, already-approved T37
`CandidateCluster` objects. **This module does not change source-ranking logic.** It
computes descriptive stability/uncertainty measures from data T29-T37 already produced,
and (if practical) a bootstrap stability check. It never recomputes T29-T36 using an
alternative formula, never re-clusters (T37's cluster IDs/membership are read-only inputs
here), and never modifies any T14-T37 module.

**Distance/particle-identity conventions reused, not reinvented:**

- Distances use the same local-tangent-plane projection every other Phase 2/3 module uses
  (`iste.kdtree_neighborhood.project_to_local_metres`).
- "Distinct particle identity" uses `(particle_id, forcing_member_id)` as the atomic key --
  the SAME grouping key T14's `query_particle_slice` already uses internally (one physical
  particle propagated under one forcing-ensemble member is one distinct stochastic
  realization; the same `particle_id` under a DIFFERENT `forcing_member_id` is a genuinely
  different trajectory, per `iste.ensemble_run`'s own per-member propagation). Repeated
  HOURLY appearances of the SAME `(particle_id, forcing_member_id)` pair (e.g. the same
  particle supporting a cluster's candidates at several different timestamps) are counted
  ONCE, never once per hour.

================================================================================
PART A -- particle_support_fraction
================================================================================

    numerator   = |UNION over the cluster's member candidates of the DISTINCT
                   (particle_id, forcing_member_id) pairs appearing in that candidate's
                   own already-computed T16 `peak_support_features.neighborhood`|
    denominator = |DISTINCT (particle_id, forcing_member_id) pairs across the ENTIRE
                   Stage III `records` batch| -- the whole simulation's particle-member
                   pool, fixed for the whole run (NOT a per-timestamp count, which varies
                   with particle death and would make the fraction incomparable across
                   clusters whose members sit at different timestamps).

This directly answers "what share of the whole particle-member ensemble's evidence is
concentrated in this cluster's own candidate neighborhoods" -- reusing the neighborhood
data T16/T17 already computed (`peak_support_features`), never re-querying Stage III.

================================================================================
PART B -- persistence_fraction
================================================================================

    persistence_fraction = cluster.n_distinct_timestamps / total_timestamps

`n_distinct_timestamps` is read directly from the T37 `CandidateCluster` (no
recomputation of T31's tracking). `total_timestamps` defaults to `DEFAULT_TOTAL_TIMESTAMPS`
(73 -- T27's own canonical analysis-grid length for its own defaults, `window_hours=72.0`,
`interval_hours=1.0`, giving T0..T72 inclusive = 73 timestamps: see
`iste.analysis_grid.generate_analysis_grid`), fully overridable if a caller used a
different window/interval. No minimum-duration threshold, no age-based reward/penalty --
purely descriptive.

================================================================================
PART C -- uncertainty_radius_km (percentile-based, distinct from T37's spatial_radius_m)
================================================================================

`spatial_radius_m` (T37) is the geometric extent of the cluster's few discrete CANDIDATE
PEAK positions. `uncertainty_radius_km` here is a different quantity: the spread of the
underlying PARTICLE EVIDENCE (typically far more points than there are candidate peaks)
supporting those peaks.

    uncertainty_radius_km = percentile(
        { distance_from_cluster_centroid(p) : p in the SAME pooled distinct-particle set
          used for Part A's numerator },
        spatial_percentile,
    ) / 1000.0

Distances are computed in local projected metres from the cluster's own already-computed
centroid (T37's `centroid_lat`/`centroid_lon`, reused unchanged). `spatial_percentile`
defaults to `DEFAULT_SPATIAL_PERCENTILE = 90.0` -- an explicit, configurable, documented
choice (not a covariance/ellipse form): a percentile-of-actual-particle-positions measure
was chosen over a covariance-ellipse measure because it requires no Gaussian/isotropy
assumption on top of the already-established T30/T33 machinery, is directly interpretable
("90% of the distinct particles supporting this cluster's evidence lie within X km of its
centroid"), and reuses the exact same pooled-particle set Part A already computed (no
second, differently-defined particle pool).

================================================================================
PART D -- temporal_uncertainty_hours (distinct from T37's temporal_span_hours)
================================================================================

`temporal_span_hours` (T37) is the raw max-min real-time extent of the cluster's member
timestamps. `temporal_uncertainty_hours` here is the EVIDENCE-WEIGHTED spread of those
same member timestamps, using each member candidate's own already-computed T36
`source_score` as its weight (no recomputation of T36):

    hours_i        = (candidate_i.t_s - cluster.earliest_t_s) / 1 hour   [translation only]
    w_i            = candidate_i's own T36 source_score (equal-weight fallback, documented,
                     if any weight is negative/non-finite or all weights sum to <= 0)
    weighted_mean  = sum(w_i * hours_i) / sum(w_i)
    weighted_var   = sum(w_i * (hours_i - weighted_mean)^2) / sum(w_i)
    temporal_uncertainty_hours = sqrt(weighted_var)

A single-member cluster has no internal temporal spread to measure at all --
`temporal_uncertainty_hours = 0.0` in that case describes the absence of multi-timestamp
evidence, not a claim of zero uncertainty in an absolute sense (documented on the result).

================================================================================
PART E -- bootstrap stability (practicality assessed BEFORE implementing; see below)
================================================================================

**Why a full bootstrap (re-running Stage III / rebuilding the whole T16 support grid per
replicate) is NOT practical:** regenerating the real dataset's 280 candidates (rebuilding
the T16 support grid across the whole spatial search area, at all ~72 candidate-bearing
timestamps) measured ~80 seconds for ONE pass (see `notebooks/T37_clustering_check.ipynb`).
`iste.particle_query.query_particle_slice` (T14) also loops in pure Python over every row
of the Stage III `records` batch (240,024 rows on the real dataset) each time it is called.
Repeating either of these per bootstrap replicate (even 100-200 replicates) would cost tens
of minutes to hours -- explicitly NOT practical within this task's runtime budget, and
Part E's own instructions prohibit rebuilding the whole simulation per replicate anyway.

**What IS practical, and what this module implements instead:** each candidate's own T16
`peak_support_features.neighborhood` (the specific particles already found within its own
support-map neighborhood -- already computed, never re-queried) is resampled WITH
REPLACEMENT, independently per candidate per replicate (cheap: a handful of NumPy draws per
candidate, no spatial re-query, no Stage III access). From the resample:

    D_boot = number of DISTINCT (particle_id, forcing_member_id) pairs actually present in
             the resample (<= original D; repeated draws of the same particle do not count
             twice, mirroring the "same particle counted once" rule used throughout this
             module -- otherwise resampling would mechanically inflate density)
    W_boot = sum of exp(log_weight) over ALL n resampled draws, counting a repeated
             particle's weight once per draw (W is a SUM statistic; this is the standard
             way a weighted-sum statistic behaves under resampling)
    V_boot = number of distinct observation_pixel_id values among the n resampled draws

    support_boot = the candidate's OWN `SupportFeatures.config`, combined via the SAME
                   `iste.support_features.linear_support_function` already used to compute
                   the ORIGINAL `support` value -- no new combination formula.

The T29/T34 normalization reference value is NOT re-derived from scratch per replicate
(doing so would require rebuilding the whole-grid percentile, i.e. exactly the expensive
step being avoided); instead the already-computed original `screening_score` (T36's
`support_evidence`) is SCALED by the ratio of the bootstrap raw support to the original raw
support -- `boot_support_evidence = support_evidence * (support_boot / candidate.support_score)`
-- which is algebraically exact given T34's own formula (`screening_score = support /
reference_value`, a per-timestamp CONSTANT that this scaling correctly holds fixed) and
introduces no new normalization. Every OTHER evidence component (persistence/convergence/
dispersion/AIS/initial_bias/persistence_count) is held at its already-computed T31-T35
value, unchanged -- **this bootstrap tests the stability of the T29/T34 support/density
evidence specifically under particle-resampling variability, not the full six-component
pipeline**, a scope limitation stated explicitly, never hidden. The composite bootstrap
score itself is recombined via T35's own unmodified `renormalize_weighted_mean`, with the
SAME configured weights the original run used -- no new combination formula.

**Cluster identity under resampling:** because this bootstrap perturbs each candidate's own
EVIDENCE VALUE (never its position, and never which candidates exist), candidate-to-cluster
membership is IDENTICAL to the original T37 assignment in every replicate, by construction.
No spatial re-matching rule is needed (unlike a hypothetical bootstrap that re-ran candidate
extraction and could produce candidates at new positions) -- this is stated explicitly
rather than silently assumed.

**Bootstrap parameters (explicit, configurable, `BootstrapConfig`):**
`n_replicates` (default 200), `random_seed` (default 42), `top_k` (default 10 -- an
explicit, PROTOTYPE/CONFIGURATION choice, not scientifically validated). For each
replicate, every candidate's bootstrap composite score is recomputed, the top `top_k`
candidates (by that replicate's bootstrap score) are found, and every T37 cluster that owns
at least one of those top-`top_k` candidates in that replicate is credited with one
"appearance." `candidate_stability_percent` for a cluster = 100 * (number of replicates in
which it appeared) / `n_replicates`. This is CANDIDATE/CLUSTER RECURRENCE across resampled
replicates -- explicitly NOT a probability that the cluster is the true source (see
`STABILITY_DISCLAIMER`).

If bootstrap was not run (e.g. explicitly disabled, or a future caller judges it
impractical for a different dataset), `candidate_stability_percent` is reported as `None`
("unavailable") on every cluster -- never fabricated as `0.0`.

================================================================================
PART F -- age/dispersion diagnostic
================================================================================

`diagnose_uncertainty_vs_age` (see below) is a VALIDATION diagnostic only: it reports the
Pearson/Spearman correlation between each cluster's `uncertainty_radius_km` /
`temporal_uncertainty_hours` and (a) the cluster's own backward age (hours from `t_obs` to
its `earliest_t_s`) and (b) a measure of particle spatial spread (the same pooled-particle
distance-from-centroid distribution Part C already computed, its own mean). It reports
whatever correlation the real data actually shows -- weak, strong, or contradictory -- and
never adjusts the uncertainty formulas to manufacture a particular trend.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from iste.candidate_extraction import Candidate
from iste.ensemble_run import EnsembleRunRecords
from iste.kdtree_neighborhood import project_to_local_metres
from iste.phase3_ais import renormalize_weighted_mean
from iste.phase3_clustering import CandidateCluster
from iste.phase3_composite import EVIDENCE_RANKING_DISCLAIMER, SourceScore
from iste.support_features import linear_support_function

DEFAULT_TOTAL_TIMESTAMPS = 73  # T27's own default grid length: T0..T72 inclusive
DEFAULT_SPATIAL_PERCENTILE = 90.0
DEFAULT_BOOTSTRAP_REPLICATES = 200
DEFAULT_RANDOM_SEED = 42
DEFAULT_TOP_K = 10

STABILITY_DISCLAIMER = (
    "candidate_stability_percent is the percentage of bootstrap replicates (particle "
    "resampling within each candidate's own already-computed support neighborhood) in "
    "which this cluster's own top-scoring member remained among the configured top-K "
    "candidates. It is CANDIDATE STABILITY / RANKING STABILITY / CLUSTER RECURRENCE under "
    "resampling -- it is NOT a probability that this cluster is the true physical source."
)


def _distinct_particle_keys(particle_id: np.ndarray, forcing_member_id: np.ndarray) -> set[tuple]:
    return {(pid, int(m)) for pid, m in zip(particle_id.tolist(), forcing_member_id.tolist())}


def compute_total_available_particles(records: EnsembleRunRecords) -> int:
    """Denominator for Part A: the number of DISTINCT `(particle_id, forcing_member_id)`
    pairs across the WHOLE Stage III `records` batch -- the fixed, timestamp-independent
    particle-member pool of the entire simulation."""
    return len(_distinct_particle_keys(records.particle_id, records.forcing_member_id))


def _pooled_cluster_particles(
    cluster: CandidateCluster, candidates_by_id: dict[str, Candidate]
) -> dict[tuple, tuple[float, float]]:
    """Union, over the cluster's member candidates, of the DISTINCT
    `(particle_id, forcing_member_id)` pairs in each candidate's own already-computed T16
    neighborhood -- `{key: (lat, lon)}`, first-seen position kept for each distinct key.
    Reused unchanged by both Part A (its size) and Part C (its position distribution)."""
    pooled: dict[tuple, tuple[float, float]] = {}
    for cid in cluster.member_candidate_ids:
        nbhd = candidates_by_id[cid].peak_support_features.neighborhood
        for pid, member, lat, lon in zip(
            nbhd.particle_id.tolist(), nbhd.forcing_member_id.tolist(),
            nbhd.lat.tolist(), nbhd.lon.tolist(),
        ):
            key = (pid, int(member))
            if key not in pooled:
                pooled[key] = (float(lat), float(lon))
    return pooled


def compute_spatial_uncertainty_radius_km(
    cluster: CandidateCluster,
    pooled_particles: dict[tuple, tuple[float, float]],
    *,
    percentile: float = DEFAULT_SPATIAL_PERCENTILE,
) -> float:
    """Part C: `percentile`-th percentile of pooled-particle distance from the cluster's
    own centroid, in km. `NaN` if `pooled_particles` is empty."""
    if not (0.0 <= percentile <= 100.0):
        raise ValueError(f"percentile must be in [0, 100], got {percentile}")
    if not pooled_particles:
        return float("nan")
    lats = np.array([v[0] for v in pooled_particles.values()], dtype=np.float64)
    lons = np.array([v[1] for v in pooled_particles.values()], dtype=np.float64)
    x_m, y_m = project_to_local_metres(lats, lons, lat_ref=cluster.centroid_lat, lon_ref=cluster.centroid_lon)
    dist_m = np.sqrt(x_m**2 + y_m**2)
    return float(np.percentile(dist_m, percentile)) / 1000.0


def compute_temporal_uncertainty_hours(
    cluster: CandidateCluster,
    candidates_by_id: dict[str, Candidate],
    source_scores_by_id: dict[str, SourceScore],
) -> float:
    """Part D: score-weighted standard deviation (hours) of the cluster's own member
    candidate timestamps, weighted by each member's own T36 `source_score`. `0.0` for a
    single-member cluster (no internal spread to measure, not a claim of zero uncertainty)."""
    if cluster.n_members <= 1:
        return 0.0

    earliest = cluster.earliest_t_s
    hours = np.array(
        [
            float((np.datetime64(candidates_by_id[cid].t_s, "ns") - earliest) / np.timedelta64(1, "h"))
            for cid in cluster.member_candidate_ids
        ],
        dtype=np.float64,
    )
    weights = np.array(
        [source_scores_by_id[cid].source_score for cid in cluster.member_candidate_ids],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0) or float(np.sum(weights)) <= 0.0:
        weights = np.ones_like(hours)  # documented equal-weight fallback

    weighted_mean = float(np.average(hours, weights=weights))
    weighted_var = float(np.average((hours - weighted_mean) ** 2, weights=weights))
    return float(np.sqrt(max(weighted_var, 0.0)))


@dataclass(frozen=True)
class BootstrapConfig:
    """Part E's explicit, configurable bootstrap parameters. PROTOTYPE/CONFIGURATION,
    not scientifically validated defaults."""

    n_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES
    random_seed: int = DEFAULT_RANDOM_SEED
    top_k: int = DEFAULT_TOP_K

    def __post_init__(self) -> None:
        if self.n_replicates < 1:
            raise ValueError(f"n_replicates must be >= 1, got {self.n_replicates}")
        if self.top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {self.top_k}")


@dataclass(frozen=True)
class BootstrapResult:
    """Part E's output. `performed=False` -> `stability_percent_by_cluster` is `{}` and
    `reason` explains why; callers must report `None` ("unavailable"), never `0.0`, for
    every cluster in that case."""

    performed: bool
    reason: str
    config: BootstrapConfig | None
    runtime_seconds: float
    n_replicates_run: int
    stability_percent_by_cluster: dict[str, float]


def _nan_to_none(value: float | None) -> float | None:
    """Mirrors `iste.phase3_composite._nan_to_none`'s documented NaN-poisons-zero-weight-sum
    fix -- duplicated here as a 3-line utility (not imported from that module's private
    name) rather than recomputing any T36 formula."""
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def _resample_candidate_support(candidate: Candidate, rng: np.random.Generator) -> float:
    """One bootstrap replicate's resampled raw `support` value for one candidate -- see
    module docstring Part E for the exact D_boot/W_boot/V_boot definitions."""
    nbhd = candidate.peak_support_features.neighborhood
    n = nbhd.n_particles
    if n == 0:
        return 0.0
    idx = rng.integers(0, n, size=n)
    pid_sample = nbhd.particle_id[idx]
    member_sample = nbhd.forcing_member_id[idx]
    d_boot = len(_distinct_particle_keys(pid_sample, member_sample))
    w_boot = float(np.sum(np.exp(nbhd.log_weight[idx])))
    v_boot = len(set(nbhd.observation_pixel_id[idx].tolist()))
    config = candidate.peak_support_features.config
    return float(linear_support_function(d_boot, w_boot, v_boot, config))


def run_bootstrap_stability(
    candidates: list[Candidate],
    source_scores: list[SourceScore],
    clusters: list[CandidateCluster],
    *,
    config: BootstrapConfig = BootstrapConfig(),
) -> BootstrapResult:
    """Part E: run the practical particle-resampling bootstrap described in the module
    docstring and report `candidate_stability_percent` per T37 cluster. Never modifies
    `candidates`/`source_scores`/`clusters`."""
    t_start = time.time()

    candidates_by_id = {c.candidate_id: c for c in candidates}
    scores_by_id = {s.candidate_id: s for s in source_scores}
    cluster_of_candidate: dict[str, str] = {}
    for cl in clusters:
        for cid in cl.member_candidate_ids:
            cluster_of_candidate[cid] = cl.cluster_id

    rng = np.random.default_rng(config.random_seed)
    appearance_counts = {cl.cluster_id: 0 for cl in clusters}

    for _ in range(config.n_replicates):
        rep_scores: list[tuple[str, float]] = []
        for c in candidates:
            orig = scores_by_id[c.candidate_id]
            orig_support_evidence = orig.support_evidence
            if (
                orig_support_evidence is None
                or not np.isfinite(orig_support_evidence)
                or c.support_score == 0.0
            ):
                boot_support_evidence = orig_support_evidence
            else:
                boot_raw_support = _resample_candidate_support(c, rng)
                boot_support_evidence = orig_support_evidence * (boot_raw_support / c.support_score)

            cw = orig.configured_weights
            components = {
                "support": (cw["support"], _nan_to_none(boot_support_evidence)),
                "persistence": (cw["persistence"], _nan_to_none(orig.persistence_evidence)),
                "convergence": (cw["convergence"], _nan_to_none(orig.convergence_evidence)),
                "dispersion": (cw["dispersion"], _nan_to_none(orig.dispersion_evidence)),
                "ais": (cw["ais"], orig.ais_evidence),
                "initial_bias": (cw["initial_bias"], _nan_to_none(orig.initial_bias_evidence)),
                "persistence_count": (cw["persistence_count"], _nan_to_none(orig.persistence_count_evidence)),
            }
            boot_score = renormalize_weighted_mean(list(components.values()))
            rep_scores.append((c.candidate_id, boot_score))

        valid = [(cid, s) for cid, s in rep_scores if np.isfinite(s)]
        valid.sort(key=lambda item: -item[1])
        top_ids = {cid for cid, _ in valid[: config.top_k]}
        top_clusters = {cluster_of_candidate[cid] for cid in top_ids if cid in cluster_of_candidate}
        for clid in top_clusters:
            appearance_counts[clid] += 1

    runtime = time.time() - t_start
    stability = {
        clid: 100.0 * count / config.n_replicates for clid, count in appearance_counts.items()
    }
    return BootstrapResult(
        performed=True, reason="bootstrap executed (see module docstring for scope/limitations)",
        config=config, runtime_seconds=runtime, n_replicates_run=config.n_replicates,
        stability_percent_by_cluster=stability,
    )


@dataclass(frozen=True)
class ClusterUncertainty:
    """One T37 cluster's T38 uncertainty/stability report. Every field is traceable to the
    particles, timestamps, T36 candidate scores, and T37 cluster it was derived from (see
    module docstring); nothing here is fabricated. `source_score_caveat` repeats T36's own
    disclaimer since this table is likely read on its own."""

    cluster_id: str
    centroid_lat: float
    centroid_lon: float
    spatial_radius_km: float  # T37's own spatial_radius_m / 1000 -- geometric extent, NOT uncertainty
    uncertainty_radius_km: float
    uncertainty_radius_percentile: float
    temporal_uncertainty_hours: float
    supporting_particle_count: int  # Part A numerator
    total_available_particles: int  # Part A denominator
    particle_support_fraction: float
    supported_timestamp_count: int
    total_timestamps: int
    persistence_fraction: float
    persistence_duration_hours: float
    mean_source_score: float
    peak_source_score: float
    candidate_stability_percent: float | None  # None == "unavailable" -- never 0.0
    ais_status: str
    source_score_caveat: str = EVIDENCE_RANKING_DISCLAIMER
    stability_caveat: str = STABILITY_DISCLAIMER


def compute_cluster_uncertainty_table(
    clusters: list[CandidateCluster],
    candidates: list[Candidate],
    source_scores: list[SourceScore],
    records: EnsembleRunRecords,
    *,
    total_timestamps: int = DEFAULT_TOTAL_TIMESTAMPS,
    spatial_percentile: float = DEFAULT_SPATIAL_PERCENTILE,
    bootstrap_result: BootstrapResult | None = None,
) -> list[ClusterUncertainty]:
    """Compute Parts A-D (+ Part E's `candidate_stability_percent`, if `bootstrap_result`
    is given) for every T37 `cluster` in `clusters`, reusing `candidates`/`source_scores`
    (T17/T36 output, unmodified) and `records` (Stage III output, unmodified, queried only
    for Part A's denominator). Does not re-cluster, does not modify `clusters`."""
    if total_timestamps < 1:
        raise ValueError(f"total_timestamps must be >= 1, got {total_timestamps}")
    if not (0.0 <= spatial_percentile <= 100.0):
        raise ValueError(f"spatial_percentile must be in [0, 100], got {spatial_percentile}")

    candidates_by_id = {c.candidate_id: c for c in candidates}
    source_scores_by_id = {s.candidate_id: s for s in source_scores}
    total_available_particles = compute_total_available_particles(records)

    results = []
    for cluster in clusters:
        pooled = _pooled_cluster_particles(cluster, candidates_by_id)
        n_pooled = len(pooled)

        particle_support_fraction = (
            n_pooled / total_available_particles if total_available_particles > 0 else float("nan")
        )
        persistence_fraction = cluster.n_distinct_timestamps / total_timestamps
        uncertainty_radius_km = compute_spatial_uncertainty_radius_km(
            cluster, pooled, percentile=spatial_percentile
        )
        temporal_uncertainty_hours = compute_temporal_uncertainty_hours(
            cluster, candidates_by_id, source_scores_by_id
        )

        candidate_stability_percent = None
        if bootstrap_result is not None and bootstrap_result.performed:
            candidate_stability_percent = bootstrap_result.stability_percent_by_cluster.get(
                cluster.cluster_id
            )

        results.append(
            ClusterUncertainty(
                cluster_id=cluster.cluster_id,
                centroid_lat=cluster.centroid_lat, centroid_lon=cluster.centroid_lon,
                spatial_radius_km=cluster.spatial_radius_m / 1000.0,
                uncertainty_radius_km=uncertainty_radius_km,
                uncertainty_radius_percentile=spatial_percentile,
                temporal_uncertainty_hours=temporal_uncertainty_hours,
                supporting_particle_count=n_pooled,
                total_available_particles=total_available_particles,
                particle_support_fraction=particle_support_fraction,
                supported_timestamp_count=cluster.n_distinct_timestamps,
                total_timestamps=total_timestamps,
                persistence_fraction=persistence_fraction,
                persistence_duration_hours=cluster.temporal_span_hours,
                mean_source_score=cluster.mean_source_score,
                peak_source_score=cluster.peak_source_score,
                candidate_stability_percent=candidate_stability_percent,
                ais_status="unavailable" if cluster.mean_ais_score is None else "available",
            )
        )
    return results


@dataclass(frozen=True)
class AgeUncertaintyDiagnostic:
    """Part F's honest correlation report -- descriptive only, never used to retune Parts
    A-D's formulas."""

    n_clusters: int
    age_hours: list[float]  # per cluster: hours from t_obs to cluster.earliest_t_s
    mean_particle_spread_km: list[float]  # per cluster: mean pooled-particle distance (km)
    uncertainty_radius_km: list[float]
    temporal_uncertainty_hours: list[float]
    pearson_age_vs_uncertainty_radius: float
    spearman_age_vs_uncertainty_radius: float
    pearson_age_vs_temporal_uncertainty: float
    spearman_age_vs_temporal_uncertainty: float
    pearson_spread_vs_uncertainty_radius: float
    spearman_spread_vs_uncertainty_radius: float


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or np.all(a == a[0]) or np.all(b == b[0]):
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or np.all(a == a[0]) or np.all(b == b[0]):
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def diagnose_uncertainty_vs_age(
    clusters: list[CandidateCluster],
    uncertainty_table: list[ClusterUncertainty],
    candidates: list[Candidate],
    t_obs,
) -> AgeUncertaintyDiagnostic:
    """Part F: honest correlation diagnostic between backward age / particle spread and
    the Part C/D uncertainty measures, on the real T37 clusters. Reports whatever the data
    shows; never adjusts Parts A-D to manufacture a particular trend."""
    candidates_by_id = {c.candidate_id: c for c in candidates}
    t_obs = np.datetime64(t_obs, "ns")

    age_hours: list[float] = []
    mean_spread_km: list[float] = []
    unc_radius: list[float] = []
    temp_unc: list[float] = []

    for cluster, row in zip(clusters, uncertainty_table):
        assert cluster.cluster_id == row.cluster_id
        age_hours.append(float((t_obs - cluster.earliest_t_s) / np.timedelta64(1, "h")))
        pooled = _pooled_cluster_particles(cluster, candidates_by_id)
        if pooled:
            lats = np.array([v[0] for v in pooled.values()], dtype=np.float64)
            lons = np.array([v[1] for v in pooled.values()], dtype=np.float64)
            x_m, y_m = project_to_local_metres(
                lats, lons, lat_ref=cluster.centroid_lat, lon_ref=cluster.centroid_lon
            )
            mean_spread_km.append(float(np.mean(np.sqrt(x_m**2 + y_m**2))) / 1000.0)
        else:
            mean_spread_km.append(float("nan"))
        unc_radius.append(row.uncertainty_radius_km)
        temp_unc.append(row.temporal_uncertainty_hours)

    age_arr = np.array(age_hours)
    spread_arr = np.array(mean_spread_km)
    radius_arr = np.array(unc_radius)
    temp_arr = np.array(temp_unc)

    finite_rs = np.isfinite(age_arr) & np.isfinite(radius_arr)
    finite_ts = np.isfinite(age_arr) & np.isfinite(temp_arr)
    finite_sr = np.isfinite(spread_arr) & np.isfinite(radius_arr)

    return AgeUncertaintyDiagnostic(
        n_clusters=len(clusters), age_hours=age_hours, mean_particle_spread_km=mean_spread_km,
        uncertainty_radius_km=unc_radius, temporal_uncertainty_hours=temp_unc,
        pearson_age_vs_uncertainty_radius=_pearson(age_arr[finite_rs], radius_arr[finite_rs]),
        spearman_age_vs_uncertainty_radius=_spearman(age_arr[finite_rs], radius_arr[finite_rs]),
        pearson_age_vs_temporal_uncertainty=_pearson(age_arr[finite_ts], temp_arr[finite_ts]),
        spearman_age_vs_temporal_uncertainty=_spearman(age_arr[finite_ts], temp_arr[finite_ts]),
        pearson_spread_vs_uncertainty_radius=_pearson(spread_arr[finite_sr], radius_arr[finite_sr]),
        spearman_spread_vs_uncertainty_radius=_spearman(spread_arr[finite_sr], radius_arr[finite_sr]),
    )
