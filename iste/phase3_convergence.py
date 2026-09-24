"""T32 -- Phase 3 spatial convergence scoring.

Measures whether multiple, genuinely INDEPENDENT particle trajectories converge onto a
candidate's core region, as distinct from raw density/KDE concentration (T29). A candidate
reached by many particles that all trace back to the SAME single observation pixel and the
SAME single forcing-member realization is much weaker evidence than the same particle COUNT
reached by particles from many distinct pixels/forcing members -- even though raw density
(`D`) and T29's normalized KDE would be IDENTICAL in both cases, by construction. This
module exists specifically to make that distinction visible, per the task's own requirement
("must be distinguishable from raw KDE concentration").

Composite of five explicit, individually-interpretable components, each reused from an
existing project module wherever possible -- nothing here re-implements T15's neighborhood
query, T16's `D`/`W`/`V` features, or T31's persistence tracking:

    particle_fraction     = D / total_valid_particles_at_t
                             [D and the neighborhood are T16's own `SupportFeatures`/T15's
                             `ParticleNeighborhoodIndex` output, reused unchanged]
    pixel_diversity_frac  = V / total_distinct_pixels_at_t
                             [V is T16's own pixel-diversity feature, reused unchanged --
                             NOT recomputed here]
    member_diversity_frac = distinct_forcing_members_in_neighborhood / total_distinct_members_at_t
                             [NEW: T16 does not compute forcing-member diversity, only
                             observation-pixel diversity -- this is the one genuinely new
                             quantity in this module, derived directly from the neighborhood
                             T16/T17 already queried (`NeighborhoodResult.forcing_member_id`),
                             not a new spatial query]
    concentration_score   = exp(-within_core_spread_m / core_radius_m)
                             [within_core_spread_m is the RMS of `NeighborhoodResult.
                             distance_m`, itself already computed by T15 -- no new distance
                             formula; core_radius_m is T17's own `CoreRegion.radius_m`, not
                             a new free parameter]
    temporal_consistency  = the candidate's own T31 `PersistenceTrack.temporal_continuity()`
                             [OPTIONAL -- reused from T31 if the caller supplies a matching
                             track; a candidate not (yet) tracked still gets a score from the
                             other four components, never a crash, never a fabricated value]

`convergence_score` is the weighted mean of whichever components are actually available
(equal weight by default -- the simplest non-arbitrary starting point, PROTOTYPE/
CONFIGURATION, fully overridable via `ConvergenceWeights`), renormalized over the available
weights when `temporal_consistency` is `None` -- mirroring this project's standing "fail
gracefully, renormalize transparently" discipline for optional evidence (the same policy
Stage 8/T35's AIS hook will use).

**Caveat (explicit, carried over from T31's review):** a high convergence score is evidence
of independent-trajectory agreement on a region, not proof of physical source identity --
it remains a Phase 2/3 screening quantity, exactly like every other backward-support
diagnostic in this project.

Note on the reused neighborhood's radius: `peak_support_features.neighborhood` was built at
T16's own support-map smoothing radius (`neighborhood_radius_m`, the same radius passed to
`compute_support_map`), which in this project's real runs happens to equal T17's
`core_radius_m`, but is not guaranteed to in general. This module reuses that
ALREADY-COMPUTED neighborhood as-is (never re-querying T15) rather than requiring an exact
`core_radius_m`-delimited population -- a documented simplification, not a claim that the
reused neighborhood is always exactly the candidate's formal core.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from iste.candidate_extraction import Candidate
from iste.ensemble_run import EnsembleRunRecords
from iste.particle_query import query_particle_slice
from iste.phase3_persistence import PersistenceTrack


@dataclass(frozen=True)
class ConvergenceWeights:
    """Explicit, configurable weights for `convergence_score`'s five components. Equal
    weighting (`1.0` each) is the default -- the simplest non-arbitrary starting point when
    no other information favors one component over another. PROTOTYPE/CONFIGURATION, not a
    validated weighting scheme."""

    weight_particle_fraction: float = 1.0
    weight_pixel_diversity: float = 1.0
    weight_member_diversity: float = 1.0
    weight_concentration: float = 1.0
    weight_temporal_consistency: float = 1.0

    def __post_init__(self) -> None:
        for name in (
            "weight_particle_fraction", "weight_pixel_diversity", "weight_member_diversity",
            "weight_concentration", "weight_temporal_consistency",
        ):
            value = getattr(self, name)
            if not (np.isfinite(value) and value >= 0.0):
                raise ValueError(f"{name} must be finite and >= 0, got {value!r}")


DEFAULT_WEIGHTS = ConvergenceWeights()


@dataclass(frozen=True)
class ConvergenceScore:
    """T32 convergence evidence for one candidate."""

    candidate_id: str
    t_s: np.datetime64

    particle_count: int
    total_valid_particles_at_t: int
    particle_fraction: float

    pixel_diversity: int
    total_distinct_pixels_at_t: int
    pixel_diversity_fraction: float

    distinct_forcing_members: int
    total_distinct_members_at_t: int
    member_diversity_fraction: float

    within_core_spread_m: float
    core_radius_m: float
    concentration_score: float

    temporal_consistency: float | None  # None iff no matching T31 track was supplied

    convergence_score: float


def compute_convergence_score(
    candidate: Candidate,
    *,
    total_valid_particles_at_t: int,
    total_distinct_pixels_at_t: int,
    total_distinct_members_at_t: int,
    temporal_consistency: float | None = None,
    weights: ConvergenceWeights = DEFAULT_WEIGHTS,
) -> ConvergenceScore:
    """Compute T32 convergence evidence for one T17 `Candidate`.

    Parameters
    ----------
    candidate : Candidate
        Must carry a non-`None` `peak_support_features` (T16/T17's own attached
        neighborhood) -- this is always true for a `Candidate` produced by
        `find_candidate_peaks`.
    total_valid_particles_at_t, total_distinct_pixels_at_t, total_distinct_members_at_t : int
        The FULL historical slice's own totals at this candidate's timestamp (not the
        neighborhood's totals) -- the denominators that make `particle_fraction`/
        `pixel_diversity_fraction`/`member_diversity_fraction` true fractions of "everything
        available at this instant," not just of the local neighborhood. Typically supplied
        from T14's `HistoricalParticleSlice` at the same `t_s` (see
        `compute_convergence_scores_for_candidates` for the batch convenience that wires
        this up automatically).
    temporal_consistency : float, optional
        The candidate's own T31 `PersistenceTrack.temporal_continuity()`, if the candidate
        has already been tracked. `None` (default) if unavailable -- the component is then
        omitted and the remaining weights renormalized, never fabricated.
    weights : ConvergenceWeights
        Explicit, configurable component weights (see module docstring).
    """
    if candidate.peak_support_features is None:
        raise ValueError("candidate.peak_support_features must not be None for T32 scoring")
    for name, value in (
        ("total_valid_particles_at_t", total_valid_particles_at_t),
        ("total_distinct_pixels_at_t", total_distinct_pixels_at_t),
        ("total_distinct_members_at_t", total_distinct_members_at_t),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be > 0, got {value}")

    neighborhood = candidate.peak_support_features.neighborhood
    particle_count = neighborhood.n_particles
    pixel_diversity = candidate.pixel_diversity_score  # T16's own V, reused unchanged
    distinct_forcing_members = len(set(neighborhood.forcing_member_id.tolist()))

    particle_fraction = float(particle_count) / float(total_valid_particles_at_t)
    pixel_diversity_fraction = float(pixel_diversity) / float(total_distinct_pixels_at_t)
    member_diversity_fraction = float(distinct_forcing_members) / float(total_distinct_members_at_t)

    core_radius_m = candidate.core_region.radius_m
    if particle_count > 0:
        within_core_spread_m = float(np.sqrt(np.mean(neighborhood.distance_m**2)))
    else:
        within_core_spread_m = float("nan")
    concentration_score = (
        float(np.exp(-within_core_spread_m / core_radius_m))
        if particle_count > 0 else 0.0
    )

    components = [
        (weights.weight_particle_fraction, particle_fraction),
        (weights.weight_pixel_diversity, pixel_diversity_fraction),
        (weights.weight_member_diversity, member_diversity_fraction),
        (weights.weight_concentration, concentration_score),
    ]
    if temporal_consistency is not None:
        components.append((weights.weight_temporal_consistency, temporal_consistency))

    total_weight = sum(w for w, _ in components)
    if total_weight > 0.0:
        convergence_score = sum(w * v for w, v in components) / total_weight
    else:
        convergence_score = float("nan")

    return ConvergenceScore(
        candidate_id=candidate.candidate_id, t_s=np.datetime64(candidate.t_s, "ns"),
        particle_count=particle_count, total_valid_particles_at_t=total_valid_particles_at_t,
        particle_fraction=particle_fraction,
        pixel_diversity=pixel_diversity, total_distinct_pixels_at_t=total_distinct_pixels_at_t,
        pixel_diversity_fraction=pixel_diversity_fraction,
        distinct_forcing_members=distinct_forcing_members,
        total_distinct_members_at_t=total_distinct_members_at_t,
        member_diversity_fraction=member_diversity_fraction,
        within_core_spread_m=within_core_spread_m, core_radius_m=core_radius_m,
        concentration_score=concentration_score,
        temporal_consistency=temporal_consistency, convergence_score=float(convergence_score),
    )


def compute_convergence_scores_for_candidates(
    records: EnsembleRunRecords,
    candidates: list[Candidate],
    *,
    persistence_tracks: list[PersistenceTrack] | None = None,
    weights: ConvergenceWeights = DEFAULT_WEIGHTS,
) -> list[ConvergenceScore]:
    """`compute_convergence_score` evaluated for every candidate in `candidates`, in order.

    Per-timestamp totals (`total_valid_particles_at_t`, etc.) are computed once per DISTINCT
    `t_s` (via T14's `query_particle_slice`, unmodified) and reused across every candidate
    sharing that timestamp -- no redundant Stage III queries, mirroring T30's own batch
    convenience.

    `persistence_tracks`, if supplied (typically T31's `track_candidate_persistence` output
    on the SAME `candidates`), is used to look up each candidate's own
    `temporal_continuity()` by `candidate_id`; a candidate not found in any track (should not
    happen if `persistence_tracks` was built from the same candidate list, but handled
    explicitly rather than assumed) gets `temporal_consistency=None`.
    """
    track_by_candidate_id: dict[str, PersistenceTrack] = {}
    if persistence_tracks is not None:
        for track in persistence_tracks:
            for cid in track.candidate_ids:
                track_by_candidate_id[cid] = track

    totals_by_t_s: dict[np.datetime64, tuple[int, int, int]] = {}

    def _totals_at(t_s: np.datetime64) -> tuple[int, int, int]:
        if t_s not in totals_by_t_s:
            slice_ = query_particle_slice(records, t_s)
            valid = slice_.valid
            n_valid = int(np.sum(valid))
            n_pixels = len(set(slice_.observation_pixel_id[valid].tolist())) if n_valid else 0
            n_members = len(set(slice_.forcing_member_id[valid].tolist())) if n_valid else 0
            totals_by_t_s[t_s] = (n_valid, n_pixels, n_members)
        return totals_by_t_s[t_s]

    results = []
    for c in candidates:
        t_s = np.datetime64(c.t_s, "ns")
        n_valid, n_pixels, n_members = _totals_at(t_s)
        track = track_by_candidate_id.get(c.candidate_id)
        temporal_consistency = track.temporal_continuity() if track is not None else None
        results.append(
            compute_convergence_score(
                c, total_valid_particles_at_t=n_valid, total_distinct_pixels_at_t=n_pixels,
                total_distinct_members_at_t=n_members, temporal_consistency=temporal_consistency,
                weights=weights,
            )
        )
    return results
