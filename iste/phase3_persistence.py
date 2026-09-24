"""T31 -- Phase 3 temporal persistence tracking.

Tracks T17 candidate hotspots across neighboring hourly analysis timestamps into
"persistence tracks" -- a real historical source region should show sustained support
across several consecutive hours, not merely a single-timestamp KDE spike. Uses spatial
matching (a configurable radius) rather than exact-position/exact-pixel matching, since a
genuinely persisting region's peak position drifts slightly frame to frame as the particle
cloud continues to be advected.

**This module contains no age-based scoring.** It only groups/counts candidates by their
own spatial+temporal adjacency; nothing here rewards or penalizes a track for being near or
far from `t_obs` -- a track spanning T50-T54 and one spanning T2-T6 are scored by exactly
the same rules.

Algorithm (a simple, deterministic, greedy multi-object tracker -- not a claim of optimal
assignment, documented as PROTOTYPE/CONFIGURATION):

1. Group candidates by their own `t_s`, process timestamps in ascending real-time order
   (i.e. oldest analysis timestamp first, `T72 -> ... -> T0`; the direction is
   functionally arbitrary but fixed here for determinism).
2. At each timestamp, try to extend every currently-open track with the nearest available
   candidate within `spatial_matching_radius_m` (metres, via
   `iste.kdtree_neighborhood.project_to_local_metres` -- the same local-projection
   convention used everywhere else in this project), using greedy nearest-distance-first
   assignment (each track matched to at most one candidate per timestamp, each candidate to
   at most one track).
3. A track that finds no match within `max_gap_hours` of its own last member is closed
   (finalized); every unmatched candidate starts a new track.
4. Every candidate belongs to exactly one track.

`spatial_matching_radius_m` (default `2000.0`, 2 km) is grounded in T30's own measured
real-data finding that the particle cloud's centroid drifts up to ~48 km over the full
72-hour window (~0.67 km/hour average) -- 2 km gives roughly a 3x margin over that average
single-hour drift for frame-to-frame matching. `max_gap_hours` (default `1.0`) is EXTRA
tolerance beyond one normal `interval_hours` step, so a track survives exactly one missed
intermediate hourly detection (e.g. a real region's support momentarily dipping below T17's
local-maxima threshold for a single hour) before being considered ended -- the smallest
possible non-zero tolerance at this project's locked 1-hour analysis resolution. Both
PROTOTYPE/CONFIGURATION, not validated constants, fully overridable (including
`max_gap_hours=0.0` for strict no-gap tracking).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from iste.candidate_extraction import Candidate
from iste.kdtree_neighborhood import project_to_local_metres

DEFAULT_SPATIAL_MATCHING_RADIUS_M = 2000.0
DEFAULT_MAX_GAP_HOURS = 1.0
DEFAULT_INTERVAL_HOURS = 1.0


def _metres_between(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance-safe metre distance between two lat/lon points, via T15's own local
    projection (`project_to_local_metres`) -- the same convention every other distance
    calculation in this project already uses; no new formula."""
    x, y = project_to_local_metres(lat2, lon2, lat_ref=lat1, lon_ref=lon1)
    return float(np.sqrt(x[0] ** 2 + y[0] ** 2))


@dataclass
class PersistenceTrack:
    """One spatially-and-temporally tracked candidate region across neighboring hourly
    analysis timestamps."""

    track_id: str
    candidate_ids: list[str] = field(default_factory=list)
    t_s_list: list[np.datetime64] = field(default_factory=list)
    support_scores: list[float] = field(default_factory=list)
    lat_list: list[float] = field(default_factory=list)
    lon_list: list[float] = field(default_factory=list)

    @property
    def earliest_t_s(self) -> np.datetime64:
        return min(self.t_s_list)

    @property
    def latest_t_s(self) -> np.datetime64:
        return max(self.t_s_list)

    @property
    def persistence_count(self) -> int:
        """Number of distinct hourly timestamps this track has a candidate at."""
        return len(self.t_s_list)

    @property
    def persistence_duration_hours(self) -> float:
        """Real-time span between the track's earliest and latest member, in hours."""
        return float((self.latest_t_s - self.earliest_t_s) / np.timedelta64(1, "h"))

    @property
    def mean_support(self) -> float:
        return float(np.mean(self.support_scores))

    @property
    def max_support(self) -> float:
        return float(np.max(self.support_scores))

    @property
    def support_variance(self) -> float:
        return float(np.var(self.support_scores))

    def temporal_continuity(self, interval_hours: float = DEFAULT_INTERVAL_HOURS) -> float:
        """Fraction of the expected hourly slots within `[earliest_t_s, latest_t_s]` that
        this track actually has a member at -- `1.0` for a track with no gap (including
        every single-timestamp track, trivially), `< 1.0` whenever `max_gap_hours` bridged
        one or more genuinely missed hourly detections."""
        expected_slots = round(self.persistence_duration_hours / interval_hours) + 1
        return float(self.persistence_count) / float(expected_slots)


def track_candidate_persistence(
    candidates: list[Candidate],
    *,
    spatial_matching_radius_m: float = DEFAULT_SPATIAL_MATCHING_RADIUS_M,
    max_gap_hours: float = DEFAULT_MAX_GAP_HOURS,
    interval_hours: float = DEFAULT_INTERVAL_HOURS,
) -> list[PersistenceTrack]:
    """Track T17 `Candidate` hotspots across neighboring hourly analysis timestamps into
    `PersistenceTrack`s. `candidates` may span multiple timestamps and multiple spatial
    modes per timestamp (T17's own multimodal output) -- every candidate ends up in exactly
    one track.

    Parameters
    ----------
    candidates : list[Candidate]
        T17 candidates, from any/all historical timestamps (order does not matter -- this
        function sorts internally).
    spatial_matching_radius_m, max_gap_hours : float
        Explicit, configurable tracker parameters (see module docstring for defaults/
        rationale). Must be finite; `spatial_matching_radius_m > 0`, `max_gap_hours >= 0`.
    interval_hours : float
        The analysis grid's own interval (default `1.0`, Phase 3's locked resolution).
        `max_gap_hours` is interpreted as EXTRA tolerance beyond one normal `interval_hours`
        step -- i.e. a track's real-time gap since its last member may grow up to
        `interval_hours + max_gap_hours` before the track is closed, so the default
        (`max_gap_hours=1.0`, `interval_hours=1.0`) tolerates exactly one missed
        intermediate hourly detection, not zero (ordinary frame-to-frame continuation
        already spends one `interval_hours` step and must not itself count against the
        tolerance).
    """
    if not (np.isfinite(spatial_matching_radius_m) and spatial_matching_radius_m > 0.0):
        raise ValueError(
            f"spatial_matching_radius_m must be finite and > 0, got {spatial_matching_radius_m}"
        )
    if not (np.isfinite(max_gap_hours) and max_gap_hours >= 0.0):
        raise ValueError(f"max_gap_hours must be finite and >= 0, got {max_gap_hours}")
    if not (np.isfinite(interval_hours) and interval_hours > 0.0):
        raise ValueError(f"interval_hours must be finite and > 0, got {interval_hours}")

    if not candidates:
        return []

    by_t_s: dict[np.datetime64, list[Candidate]] = {}
    for c in candidates:
        t_s = np.datetime64(c.t_s, "ns")
        by_t_s.setdefault(t_s, []).append(c)
    distinct_t_s = sorted(by_t_s.keys())

    active: list[PersistenceTrack] = []
    finished: list[PersistenceTrack] = []
    next_track_num = 0

    def _new_track_id() -> str:
        nonlocal next_track_num
        tid = f"track{next_track_num:04d}"
        next_track_num += 1
        return tid

    max_gap_ns = np.timedelta64(
        int(round((interval_hours + max_gap_hours) * 3600 * 1e9)), "ns"
    )

    for t_s in distinct_t_s:
        here = list(by_t_s[t_s])

        # close any active track whose gap tolerance is already exceeded at this timestamp
        # (timestamps advance monotonically, so once exceeded it can never be matched again)
        still_active = []
        for tr in active:
            if (t_s - tr.latest_t_s) > max_gap_ns:
                finished.append(tr)
            else:
                still_active.append(tr)
        active = still_active

        # candidate pairs within the matching radius and gap tolerance, nearest-first
        pairs = []
        for ti, tr in enumerate(active):
            for ci, c in enumerate(here):
                d = _metres_between(tr.lat_list[-1], tr.lon_list[-1], c.y_peak, c.x_peak)
                if d <= spatial_matching_radius_m:
                    pairs.append((d, ti, ci))
        pairs.sort(key=lambda item: item[0])

        matched_tracks: set[int] = set()
        matched_candidates: set[int] = set()
        for d, ti, ci in pairs:
            if ti in matched_tracks or ci in matched_candidates:
                continue
            matched_tracks.add(ti)
            matched_candidates.add(ci)
            tr = active[ti]
            c = here[ci]
            tr.candidate_ids.append(c.candidate_id)
            tr.t_s_list.append(t_s)
            tr.support_scores.append(c.support_score)
            tr.lat_list.append(c.y_peak)
            tr.lon_list.append(c.x_peak)

        for ci, c in enumerate(here):
            if ci in matched_candidates:
                continue
            tr = PersistenceTrack(track_id=_new_track_id())
            tr.candidate_ids.append(c.candidate_id)
            tr.t_s_list.append(t_s)
            tr.support_scores.append(c.support_score)
            tr.lat_list.append(c.y_peak)
            tr.lon_list.append(c.x_peak)
            active.append(tr)

    finished.extend(active)
    finished.sort(key=lambda tr: tr.track_id)
    return finished
