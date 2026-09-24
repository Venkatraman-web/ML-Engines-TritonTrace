"""T35 -- Phase 3 AIS evidence hook.

**UPDATE (TASK AIS-6, 2026-09-24): real historical AIS data now exists** (AIS-2 through
AIS-5 validated `east_med_massive_random_ais.csv` into clean event-window tracks). This
module's own long-documented precondition for implementing the AIS-present evidence path
-- "future work, gated on real data actually being integrated" -- is now satisfied, and
that path is implemented below. `AISDataSource`, `VesselTrackSegment`, and `AISEvidence`
are UNCHANGED (the same contract fixed at T35's own approval); `compute_ais_evidence`'s
`ais_source is None` branch is also UNCHANGED (still the exact `ais_available=False`
sentinel, never a fabricated score) -- only the previously-`NotImplementedError` branch
now has a real, documented implementation. See that function's own docstring for the full
formula. A concrete `AISDataSource` backed by the real AIS-2 tracks lives in
`iste.ais_source.RealAISDataSource` (a new, separate module -- not defined here, keeping
this module unchanged in every other respect).

**Candidate-level AIS evidence is DELIBERATELY INDEPENDENT of Support (TASK AIS-6
requirement 3):** it is built ONLY from (a) synchronized spatial proximity to real AIS
positions and (b) repeated/persistent temporal co-location across a small window of
nearby analysis-grid offsets -- never from AIS-5's own separate `support_evidence`
dimension (which exists purely as one of AIS-5's OWN vessel-ranking dimensions, at the
vessel level, and is never read by this module). This avoids double-counting the T29/T34
Support evidence T36 already weights separately via `w_support`.

Also provides `renormalize_weighted_mean`, a small, general, reusable utility for combining
several `(weight, value_or_None)` evidence components into one score, dropping any `None`
(unavailable) components and renormalizing the remaining weights -- generalizing the exact
pattern T32's `compute_convergence_score` already used ad hoc for its optional
`temporal_consistency` component, packaged here as a standalone, tested utility so T36 (and
this module's own new AIS-present path below) can reuse it rather than re-deriving the same
renormalization logic per task.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from iste.candidate_extraction import Candidate

AIS_UNAVAILABLE_REASON = (
    "AIS unavailable: no AIS data source is configured or exists in this project "
    "(data/ contains only ocean.nc, wind.nc, database_payload.json)"
)


@dataclass(frozen=True)
class VesselTrackSegment:
    """Placeholder shape for one vessel-track sample a future concrete `AISDataSource`
    would return. Documents the fields Stage 8's evidence computation would need
    (`README.md`/master-spec: distance to track, vessel count, presence duration,
    trajectory compatibility, temporal compatibility) -- NO instance of this class is ever
    constructed anywhere in this codebase, since no real AIS data exists to populate one."""

    vessel_id: str
    lat: float
    lon: float
    t: np.datetime64


class AISDataSource(ABC):
    """Abstract interface for a future AIS vessel-track data source. Fixes the CONTRACT
    only -- every method is `@abstractmethod`, so this class cannot be instantiated, and no
    subclass exists in this project (mirrors `iste.environment.EnvironmentalFieldProvider`'s
    identical "declare now, implement when real data exists" convention). A future concrete
    implementation must be built and validated against real AIS data before
    `compute_ais_evidence`'s AIS-present code path can be implemented -- not invented here.
    """

    @abstractmethod
    def query_vessels_near(
        self, lat: float, lon: float, t: np.datetime64, radius_m: float, time_window_hours: float
    ) -> list[VesselTrackSegment]:
        """Return vessel track samples within `radius_m` of `(lat, lon)` and within
        `time_window_hours` of `t`. Must be implemented by a concrete, real-data-backed
        subclass; not implemented here."""
        raise NotImplementedError


@dataclass(frozen=True)
class AISEvidence:
    """T35 AIS evidence for one candidate. Every AIS-specific field is `None` whenever
    `ais_available` is `False` -- never a fabricated `0.0` or default value standing in for
    "no evidence.\""""

    candidate_id: str
    t_s: np.datetime64
    ais_available: bool
    distance_to_nearest_track_m: float | None
    n_vessels_nearby: int | None
    presence_duration_hours: float | None
    trajectory_compatibility_score: float | None
    ais_score: float | None
    reason: str


DEFAULT_AIS_SPATIAL_SCALE_KM = 10.0  # reused unchanged from AIS-3/4/5's own established close-encounter definition (iste.ais_matching.DEFAULT_CLOSE_ENCOUNTER_DISTANCE_KM) -- not a new threshold
DEFAULT_AIS_TEMPORAL_CHECK_OFFSETS_HOURS = tuple(range(-6, 7))  # +/-6h at 1h steps -- reuses the project's own locked 1h analysis-grid resolution (T27) as the step size, not a new one
DEFAULT_AIS_SUBWEIGHTS = {"spatial": 0.5, "temporal_consistency": 0.5}  # explicit, configurable, equal by default; deliberately excludes any Support-like dimension (TASK AIS-6 requirement 3)


def compute_ais_evidence(
    candidate: Candidate,
    ais_source: AISDataSource | None = None,
    *,
    search_radius_m: float = 5000.0,
    time_window_hours: float = 3.0,
    temporal_check_offsets_hours: tuple[float, ...] = DEFAULT_AIS_TEMPORAL_CHECK_OFFSETS_HOURS,
    spatial_scale_km: float = DEFAULT_AIS_SPATIAL_SCALE_KM,
    ais_subweights: dict[str, float] = None,
) -> AISEvidence:
    """Compute T35 AIS evidence for one candidate, if AIS data is available.

    Parameters
    ----------
    candidate : Candidate
        A T17 candidate.
    ais_source : AISDataSource, optional
        A concrete AIS data source (e.g. `iste.ais_source.RealAISDataSource`, backed by
        the real AIS-2 event-window tracks). `None` (the default) returns the explicit
        unavailable sentinel below, unchanged from T35's original behavior.
    search_radius_m, time_window_hours : float
        T35's own original, unchanged PROTOTYPE defaults (5 km / 3 h) -- the radius/window
        passed to EACH individual `ais_source.query_vessels_near` call below.
    temporal_check_offsets_hours : tuple[float, ...]
        TASK AIS-6: the candidate's own `t_s` is queried at each of these hour OFFSETS
        (default +/-6h at 1h steps) -- not just at `t_s` itself -- so that "repeated
        temporal consistency"/"encounter persistence" (AIS-6 requirement 3) can be
        measured, not just a single instant's proximity. Explicit, configurable.
    spatial_scale_km : float
        Exponential-decay scale for `distance_to_nearest_track_m` -> spatial evidence,
        reused unchanged from AIS-3/4/5's own `DEFAULT_CLOSE_ENCOUNTER_DISTANCE_KM`.
    ais_subweights : dict, optional
        `{"spatial": w, "temporal_consistency": w}` -- explicit, configurable sub-weights
        for combining this candidate-level evidence's own two dimensions (via
        `renormalize_weighted_mean`, reused unchanged). Defaults to
        `DEFAULT_AIS_SUBWEIGHTS` (0.5/0.5). Deliberately has NO "support" key -- this
        function never reads or combines any Support-like quantity (AIS-6 requirement 3).

    Returns
    -------
    AISEvidence
        `ais_available=False` (every AIS-specific field `None`) when `ais_source is None`
        -- unchanged from T35's original behavior. When `ais_source` is supplied:
        `ais_available=True` always (the data source itself is genuinely available), but
        `ais_score`/`distance_to_nearest_track_m`/etc. are `None` if literally no AIS
        vessel was found near this candidate at ANY checked temporal offset -- missing
        AIS evidence for THIS candidate, never fabricated as a low/zero score (AIS-6
        requirement 5; consumed downstream by T36's existing, unmodified renormalization).

    Formula (TASK AIS-6, real-data-backed; independent of Support -- requirement 3)
    --------------------------------------------------------------------------------
        for h in temporal_check_offsets_hours:
            segments(h) = ais_source.query_vessels_near(candidate.lat, candidate.lon,
                                                          t_s + h, search_radius_m, time_window_hours)
        distance_to_nearest_track_m = min great-circle distance over ALL segments found,
            at ANY offset (haversine, via iste.ais_matching.haversine_distance_km, reused)
        spatial_evidence            = exp(-(distance_to_nearest_track_m/1000) / spatial_scale_km)
        n_vessels_nearby             = count of DISTINCT vessel_id across all offsets
        presence_duration_hours      = span (hours) between the earliest and latest offset
                                        that had >=1 vessel nearby (0 if only one such offset)
        trajectory_compatibility_score (== temporal_consistency_evidence)
                                      = (n offsets with >=1 vessel nearby) / len(temporal_check_offsets_hours)
        ais_score = renormalize_weighted_mean([
            (ais_subweights["spatial"], spatial_evidence),
            (ais_subweights["temporal_consistency"], trajectory_compatibility_score),
        ])
    `ais_score` is `None` (not 0.0) if no vessel was found at any offset -- genuinely
    missing evidence for this one candidate, not negative evidence.
    """
    if ais_source is None:
        return AISEvidence(
            candidate_id=candidate.candidate_id, t_s=np.datetime64(candidate.t_s, "ns"),
            ais_available=False, distance_to_nearest_track_m=None, n_vessels_nearby=None,
            presence_duration_hours=None, trajectory_compatibility_score=None, ais_score=None,
            reason=AIS_UNAVAILABLE_REASON,
        )

    from iste.ais_matching import haversine_distance_km  # local import: avoids a module-level dependency from T35 on the new AIS-N modules for the (still fully supported) ais_source=None path

    if ais_subweights is None:
        ais_subweights = DEFAULT_AIS_SUBWEIGHTS

    t_s = np.datetime64(candidate.t_s, "ns")
    min_distance_m: float | None = None
    vessel_ids: set[str] = set()
    offsets_with_vessel: list[float] = []

    for h in temporal_check_offsets_hours:
        t_query = t_s + np.timedelta64(int(round(h * 3600)), "s")
        segments = ais_source.query_vessels_near(candidate.y_peak, candidate.x_peak, t_query, search_radius_m, time_window_hours)
        if not segments:
            continue
        offsets_with_vessel.append(float(h))
        for seg in segments:
            vessel_ids.add(seg.vessel_id)
            d_m = float(haversine_distance_km(candidate.y_peak, candidate.x_peak, seg.lat, seg.lon)) * 1000.0
            if min_distance_m is None or d_m < min_distance_m:
                min_distance_m = d_m

    if min_distance_m is None:
        return AISEvidence(
            candidate_id=candidate.candidate_id, t_s=t_s, ais_available=True,
            distance_to_nearest_track_m=None, n_vessels_nearby=0,
            presence_duration_hours=None, trajectory_compatibility_score=None, ais_score=None,
            reason="AIS source available, but no vessel found within search_radius_m at any checked temporal offset for this candidate",
        )

    spatial_evidence = float(np.exp(-(min_distance_m / 1000.0) / spatial_scale_km))
    temporal_consistency_evidence = len(offsets_with_vessel) / len(temporal_check_offsets_hours)
    presence_duration_hours = float(max(offsets_with_vessel) - min(offsets_with_vessel)) if len(offsets_with_vessel) > 1 else 0.0

    ais_score = renormalize_weighted_mean([
        (ais_subweights["spatial"], spatial_evidence),
        (ais_subweights["temporal_consistency"], temporal_consistency_evidence),
    ])

    return AISEvidence(
        candidate_id=candidate.candidate_id, t_s=t_s, ais_available=True,
        distance_to_nearest_track_m=min_distance_m, n_vessels_nearby=len(vessel_ids),
        presence_duration_hours=presence_duration_hours, trajectory_compatibility_score=temporal_consistency_evidence,
        ais_score=ais_score, reason="computed from real AIS data via ais_source",
    )


def renormalize_weighted_mean(components: list[tuple[float, float | None]]) -> float:
    """Weighted mean of `(weight, value)` pairs, dropping any pair whose value is `None`
    (an unavailable evidence component) and renormalizing the remaining weights so they
    still sum to the same total -- the general form of the "fail gracefully, renormalize
    transparently" pattern T32's `compute_convergence_score` already applied ad hoc to its
    optional `temporal_consistency` component. Intended for reuse by T36 (and any future
    optional-evidence combination, including AIS once real data exists) rather than each
    task re-deriving the same renormalization.

    Parameters
    ----------
    components : list[tuple[float, float | None]]
        `(weight, value_or_None)` pairs. Weights must be finite and `>= 0`.

    Returns
    -------
    float
        The renormalized weighted mean of the available (non-`None`) components. `NaN` if
        no component is available (all `None`) or if every available component has zero
        weight -- never a fabricated `0.0`.
    """
    for weight, _ in components:
        if not (np.isfinite(weight) and weight >= 0.0):
            raise ValueError(f"all weights must be finite and >= 0, got {weight!r}")

    available = [(w, v) for w, v in components if v is not None]
    total_weight = sum(w for w, _ in available)
    if not available or total_weight <= 0.0:
        return float("nan")
    return sum(w * v for w, v in available) / total_weight
