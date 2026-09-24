"""TASK AIS-3 -- spatial-temporal matching of AIS vessels to backtracked Phase 3
candidates (RAW MATCHING EVIDENCE ONLY).

Matches the already-validated AIS-2 event-window+AOI vessel tracks against the
EXISTING June-2019 Phase 3 candidate outputs (T27-T39, read-only). **This module does
not rank vessels, does not compute an AIS evidence score, and is never imported by
`iste.phase3_ais` (T35) or `iste.phase3_composite` (T36)** -- it only computes raw
distance/time-difference features between AIS observations and candidate positions,
per this task's own explicit scope.

**Core principle (this task's own instruction):** a vessel is spatially consistent
with a candidate only when its position and the candidate's position are close at
APPROXIMATELY THE SAME PHYSICAL TIME. Every match this module produces carries both
its spatial distance AND its time difference explicitly -- there is no code path here
that matches a vessel to a candidate purely because their tracks pass through the same
region at different times.

**Distance: great-circle (Haversine), never planar lat/lon Euclidean, per this task's
own explicit instruction** -- unlike this project's other distance code (`iste.
kdtree_neighborhood.project_to_local_metres`, a local tangent-plane approximation
appropriate for one small region around a single reference point), AIS vessels and
candidates can be arbitrarily far apart across the whole project AOI, where a single-
reference-point planar approximation degrades. `haversine_distance_km` implements the
standard closed-form great-circle formula, reusing `iste.geometry.MEAN_EARTH_RADIUS_M`
(the SAME Earth-radius constant already used throughout this project, not a new one).

**No AIS position is ever interpolated.** Every AIS observation used in a match is a
REAL, already-recorded `(lat, lon, timestamp)` row from the AIS-2 output; `time_diff_
hours` measures how far the real observation actually is from the candidate's own
timestamp, never a synthesized in-between position.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from iste.geometry import MEAN_EARTH_RADIUS_M

# Explicit, documented, configurable diagnostic parameters (this task's own
# requirement 6/8: "these thresholds are diagnostic bins only, NOT vessel-ranking
# rules" -- never hidden constants, never used to filter/rank/score anything).
DEFAULT_TIME_TOLERANCE_HOURS = 2.0  # widest tolerance level computed; exact/1h/2h all derived from the same underlying time_diff_hours
DEFAULT_DISTANCE_BINS_KM = (1.0, 2.0, 5.0, 10.0)
DEFAULT_CLOSE_ENCOUNTER_DISTANCE_KM = 10.0  # diagnostic definition of a "close encounter" for track-level consecutive-checkpoint analysis (item 7) -- the widest of the four distance bins, chosen for visibility, not validated as a physical threshold
DEFAULT_EXACT_TOLERANCE_SECONDS = 1.0  # two timestamps are "exact" if within this many seconds (floating-point/second-resolution safety margin only, not a real tolerance widening)


def haversine_distance_km(
    lat1, lon1, lat2, lon2, radius_m: float = MEAN_EARTH_RADIUS_M
) -> np.ndarray:
    """Standard closed-form great-circle distance (Haversine formula), vectorized over
    numpy arrays/broadcastable scalars. Reuses `iste.geometry.MEAN_EARTH_RADIUS_M` --
    the same Earth-radius convention already used throughout this project. NOT the
    project's usual local-tangent-plane `project_to_local_metres` -- that projection is
    only accurate near ONE fixed reference point, unsuitable for arbitrary vessel-
    candidate pairs that may be hundreds of km apart across the whole AOI."""
    lat1r, lon1r, lat2r, lon2r = (
        np.radians(np.asarray(lat1, dtype=np.float64)),
        np.radians(np.asarray(lon1, dtype=np.float64)),
        np.radians(np.asarray(lat2, dtype=np.float64)),
        np.radians(np.asarray(lon2, dtype=np.float64)),
    )
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    a = np.clip(a, 0.0, 1.0)  # guards against tiny negative values from floating-point roundoff
    c = 2.0 * np.arcsin(np.sqrt(a))
    return (radius_m * c) / 1000.0


def load_phase3_candidates(hotspots_json_path: str, cluster_lookup: dict[str, str] | None = None) -> pd.DataFrame:
    """TASK AIS-3 requirement 1: load the EXISTING June-2019 Phase 3 candidate output
    (`result/hotspots.json`, T17's own already-produced, unmodified candidate records
    -- read-only, never regenerated here) into a flat DataFrame preserving exactly:
    candidate identifier, candidate timestamp, latitude, longitude, T16 Support, and
    (if `cluster_lookup` is supplied, e.g. `{candidate_id: cluster_id}` from an already-
    computed T37 clustering) the candidate's own cluster identifier."""
    import json
    with open(hotspots_json_path, "r", encoding="utf-8") as f:
        hotspots = json.load(f)
    rows = []
    for h in hotspots:
        rows.append(dict(
            candidate_id=h["candidate_id"],
            t_s=pd.Timestamp(h["t_s"], tz="UTC"),
            lat=float(h["lat"]), lon=float(h["lon"]),
            support_score=float(h["support_score"]),
            cluster_id=(cluster_lookup or {}).get(h["candidate_id"]),
        ))
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class TimestampAlignmentReport:
    """TASK AIS-3 requirement 2: descriptive alignment between the Phase 3 checkpoint
    grid and the real AIS observation timestamps -- computed once, up front, before
    any matching is attempted, so the matching results below can be interpreted
    correctly (e.g. whether "exact" matches are even structurally possible)."""

    n_candidate_timestamps: int
    n_distinct_ais_timestamps: int
    n_ais_timestamps_exactly_on_candidate_grid: int
    fraction_ais_timestamps_on_grid: float
    typical_offset_seconds: float  # median |AIS_timestamp - nearest_candidate_timestamp|
    max_offset_seconds: float


def analyze_timestamp_alignment(
    candidate_t_s: pd.Series, ais_timestamps: pd.Series
) -> TimestampAlignmentReport:
    """Compares the distinct candidate checkpoint timestamps against the distinct real
    AIS observation timestamps -- does the AIS reporting grid line up with the Phase 3
    hourly analysis grid, and by how much does it typically differ."""
    # Timestamps are normalized to a fixed ns resolution explicitly -- pandas may
    # otherwise represent `datetime64[UTC]` columns at us/ns/etc. resolution
    # depending on how they were constructed, and raw integer (`.asi8`) arithmetic
    # would silently mix units; `pd.Timestamp` subtraction (`Timedelta`) is used
    # instead, which is resolution-safe regardless of the underlying storage unit.
    cand_ts = sorted(pd.to_datetime(pd.Series(candidate_t_s), utc=True).unique())
    ais_ts = sorted(pd.to_datetime(pd.Series(ais_timestamps), utc=True).unique())
    cand_ts_set = set(cand_ts)

    on_grid = [t in cand_ts_set for t in ais_ts]
    n_on_grid = int(sum(on_grid))

    cand_ts_list = [pd.Timestamp(t) for t in cand_ts]
    offsets_seconds = np.array([
        min(abs((pd.Timestamp(t) - c).total_seconds()) for c in cand_ts_list)
        for t in ais_ts
    ])

    return TimestampAlignmentReport(
        n_candidate_timestamps=len(cand_ts),
        n_distinct_ais_timestamps=len(ais_ts),
        n_ais_timestamps_exactly_on_candidate_grid=n_on_grid,
        fraction_ais_timestamps_on_grid=float(n_on_grid / len(ais_ts)) if len(ais_ts) else float("nan"),
        typical_offset_seconds=float(np.median(offsets_seconds)) if len(offsets_seconds) else float("nan"),
        max_offset_seconds=float(np.max(offsets_seconds)) if len(offsets_seconds) else float("nan"),
    )


def compute_vessel_candidate_matches(
    candidates_df: pd.DataFrame,
    ais_df: pd.DataFrame,
    *,
    max_time_tolerance_hours: float = DEFAULT_TIME_TOLERANCE_HOURS,
    exact_tolerance_seconds: float = DEFAULT_EXACT_TOLERANCE_SECONDS,
) -> pd.DataFrame:
    """TASK AIS-3 requirements 3-5: for every candidate, find every AIS observation
    (any vessel) within `max_time_tolerance_hours` of the candidate's own timestamp,
    compute the great-circle distance and the signed/absolute time difference, and tag
    which tolerance level(s) it satisfies. Returns ONE ROW PER (candidate, AIS
    observation) PAIR within the widest tolerance -- the full, auditable raw-evidence
    table (requirement 8: "preserve all matched records so the result is auditable").

    `is_exact`/`is_within_1h`/`is_within_2h` are kept as SEPARATE explicit boolean
    columns (requirement 8: "keep exact timestamp matches separate from time-tolerant
    matches") -- a row can satisfy more than one (a within-1h match also satisfies
    within-2h), so a caller filtering for a specific tolerance uses the matching column
    directly, never infers it from `time_diff_hours` alone.
    """
    ais_ts_sorted = ais_df.copy()
    ais_ts_sorted["timestamp"] = pd.to_datetime(ais_ts_sorted["timestamp"], utc=True)
    ais_ts_sorted = ais_ts_sorted.sort_values("timestamp").reset_index(drop=True)
    # Resolution-safe: seconds-since-epoch via `.timestamp()` rather than raw
    # integer (`.asi8`) arithmetic, which is resolution-dependent (see
    # `analyze_timestamp_alignment`'s own docstring note on this).
    ais_t_epoch_s = np.array([pd.Timestamp(t).timestamp() for t in ais_ts_sorted["timestamp"]])

    tol_s = max_time_tolerance_hours * 3600.0
    exact_s = exact_tolerance_seconds

    rows = []
    for _, cand in candidates_df.iterrows():
        cand_t_epoch_s = pd.Timestamp(cand["t_s"]).timestamp()
        diff_s = ais_t_epoch_s - cand_t_epoch_s
        within = np.abs(diff_s) <= tol_s
        if not np.any(within):
            continue
        matched = ais_ts_sorted.loc[within]
        distances_km = haversine_distance_km(
            cand["lat"], cand["lon"], matched["lat"].to_numpy(), matched["lon"].to_numpy()
        )
        time_diff_hours = np.abs(diff_s[within]) / 3600.0
        is_exact = np.abs(diff_s[within]) <= exact_s

        for j, (_, arow) in enumerate(matched.iterrows()):
            rows.append(dict(
                mmsi=int(arow["mmsi"]), vessel_name=arow["vessel_name"], vessel_type=arow["vessel_type"],
                imo=int(arow["imo"]),
                candidate_id=cand["candidate_id"], candidate_t_s=cand["t_s"],
                candidate_lat=cand["lat"], candidate_lon=cand["lon"],
                candidate_support=cand["support_score"], candidate_cluster_id=cand.get("cluster_id"),
                ais_t_s=arow["timestamp"], ais_lat=arow["lat"], ais_lon=arow["lon"],
                ais_sog=arow["sog"], ais_cog=arow["cog"],
                distance_km=float(distances_km[j]), time_diff_hours=float(time_diff_hours[j]),
                is_exact=bool(is_exact[j]),
                is_within_1h=bool(time_diff_hours[j] <= 1.0),
                is_within_2h=bool(time_diff_hours[j] <= 2.0),
            ))
    return pd.DataFrame(rows)


def summarize_vessel_matches(
    matches_df: pd.DataFrame,
    ais_df: pd.DataFrame,
    n_candidate_timestamps: int,
    *,
    distance_bins_km: tuple[float, ...] = DEFAULT_DISTANCE_BINS_KM,
) -> pd.DataFrame:
    """TASK AIS-3 requirement 6: per-vessel descriptive aggregate statistics, computed
    ONLY from `matches_df` (the raw pair table `compute_vessel_candidate_matches`
    produced) -- no new distance/time computation here, purely aggregation. Includes
    EVERY vessel present in `ais_df` (even those with zero matches at all, reported
    with `n_matches=0`/`NaN` minimum-distance fields -- never silently excluded)."""
    all_vessels = ais_df[["mmsi", "vessel_name", "vessel_type", "imo"]].drop_duplicates(subset="mmsi")

    rows = []
    for _, v in all_vessels.iterrows():
        vm = matches_df[matches_df["mmsi"] == v["mmsi"]]
        row = dict(
            mmsi=int(v["mmsi"]), vessel_name=v["vessel_name"], vessel_type=v["vessel_type"], imo=int(v["imo"]),
            n_pair_matches_within_2h=len(vm),
            n_distinct_candidate_timestamps_matched_2h=int(vm["candidate_t_s"].nunique()) if len(vm) else 0,
            n_distinct_candidate_timestamps_matched_1h=int(vm.loc[vm["is_within_1h"], "candidate_t_s"].nunique()) if len(vm) else 0,
            n_distinct_candidate_timestamps_matched_exact=int(vm.loc[vm["is_exact"], "candidate_t_s"].nunique()) if len(vm) else 0,
        )
        row["fraction_candidate_timestamps_with_2h_match"] = row["n_distinct_candidate_timestamps_matched_2h"] / n_candidate_timestamps

        if len(vm):
            best = vm.loc[vm["distance_km"].idxmin()]
            row["min_distance_km"] = float(best["distance_km"])
            row["min_distance_candidate_id"] = best["candidate_id"]
            row["min_distance_candidate_t_s"] = best["candidate_t_s"]
            row["min_distance_candidate_support"] = float(best["candidate_support"])
            row["min_distance_time_diff_hours"] = float(best["time_diff_hours"])
        else:
            row["min_distance_km"] = float("nan")
            row["min_distance_candidate_id"] = None
            row["min_distance_candidate_t_s"] = None
            row["min_distance_candidate_support"] = float("nan")
            row["min_distance_time_diff_hours"] = float("nan")

        for b in distance_bins_km:
            row[f"n_matches_within_{b:g}km"] = int((vm["distance_km"] <= b).sum()) if len(vm) else 0

        rows.append(row)

    return pd.DataFrame(rows).sort_values("min_distance_km", na_position="last").reset_index(drop=True)


@dataclass(frozen=True)
class VesselTrackAnalysis:
    """TASK AIS-3 requirement 7: track-level descriptive comparison for one vessel.
    Purely descriptive -- no score, no rank, no "best vessel" designation."""

    mmsi: int
    min_synchronized_distance_km: float
    median_synchronized_distance_km: float
    n_close_encounters: int  # matches within `close_encounter_distance_km` (any tolerance up to max_time_tolerance_hours)
    close_encounter_time_span_hours: float  # latest - earliest candidate_t_s among close encounters
    max_consecutive_close_encounter_checkpoints: int  # longest run of ADJACENT (1h-apart) candidate checkpoints with a close encounter


def analyze_vessel_track(
    mmsi: int, matches_df: pd.DataFrame, *, close_encounter_distance_km: float = DEFAULT_CLOSE_ENCOUNTER_DISTANCE_KM
) -> VesselTrackAnalysis:
    """Compares one vessel's matched observations (already time-synchronized to
    candidates in `matches_df`) across the chronological candidate timestamp sequence.
    `max_consecutive_close_encounter_checkpoints` counts a run only across CONSECUTIVE
    (exactly 1-hour-apart) candidate checkpoints -- a gap of >1h breaks the run, since
    that would no longer be "near-continuous" by this project's own 1h analysis grid."""
    vm = matches_df[matches_df["mmsi"] == mmsi]
    if len(vm) == 0:
        return VesselTrackAnalysis(
            mmsi=mmsi, min_synchronized_distance_km=float("nan"), median_synchronized_distance_km=float("nan"),
            n_close_encounters=0, close_encounter_time_span_hours=float("nan"),
            max_consecutive_close_encounter_checkpoints=0,
        )

    close = vm[vm["distance_km"] <= close_encounter_distance_km]
    n_close = len(close)
    if n_close > 0:
        close_t_s = pd.to_datetime(close["candidate_t_s"]).sort_values().unique()
        span_hours = float((close_t_s.max() - close_t_s.min()) / np.timedelta64(1, "h"))
        # longest run of consecutive (exactly 1h-apart) checkpoints among close_t_s
        deltas_h = np.diff(close_t_s).astype("timedelta64[s]").astype(np.float64) / 3600.0
        run = 1
        best_run = 1
        for d in deltas_h:
            if abs(d - 1.0) < 1e-6:
                run += 1
                best_run = max(best_run, run)
            else:
                run = 1
    else:
        span_hours = float("nan")
        best_run = 0

    return VesselTrackAnalysis(
        mmsi=mmsi,
        min_synchronized_distance_km=float(vm["distance_km"].min()),
        median_synchronized_distance_km=float(vm["distance_km"].median()),
        n_close_encounters=n_close,
        close_encounter_time_span_hours=span_hours,
        max_consecutive_close_encounter_checkpoints=best_run,
    )
