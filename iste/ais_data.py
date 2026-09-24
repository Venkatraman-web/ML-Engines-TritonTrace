"""TASK AIS-2 -- historical AIS track preparation/validation (data layer only).

Loads, inspects, filters, and validates the real historical AIS CSV
(`east_med_massive_random_ais.csv`) into clean per-vessel chronological tracks for the
event window. **This module does not rank vessels, does not compute an AIS evidence
score, and is never imported by `iste.phase3_ais` (T35) or `iste.phase3_composite`
(T36)** -- those modules are unmodified; this is preparation/validation-only, per this
task's own explicit scope.

**No position is ever invented or interpolated here.** AIS `(lat, lon, timestamp)` is
treated as the primary observed track; `sog`/`cog` are carried through as metadata/
features only, never used to reconstruct or replace an observed position (this task's
own explicit instruction -- SOG/COG cannot be assumed to perfectly reconstruct observed
displacement, e.g. course changes, current drift, or simple reporting noise between
fixes). Outlier/implausible records are FLAGGED and reported, never silently dropped;
any exclusion this module does apply (e.g. malformed coordinates) is a documented,
explicit rule, not a silent deletion.

Distance calculations reuse this project's own existing local-tangent-plane projection
convention (`iste.kdtree_neighborhood.project_to_local_metres`) -- the same one every
other distance computation in this codebase already uses, not a new geodesic formula.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from iste.kdtree_neighborhood import project_to_local_metres

REQUIRED_COLUMNS = (
    "mmsi", "timestamp", "lat", "lon", "sog", "cog",
    "vessel_name", "imo", "vessel_type", "type_code", "length", "width", "draught",
)

# Explicit, documented, configurable validity/plausibility bounds -- never hidden
# constants. None of these are used to alter/delete data automatically; see
# `flag_coordinate_validity`/`flag_speed_outliers` for how each is actually applied.
LAT_VALID_RANGE = (-90.0, 90.0)
LON_VALID_RANGE = (-180.0, 180.0)
DEFAULT_MAX_PLAUSIBLE_SPEED_KNOTS = 60.0  # generous upper bound: fast ferries/naval
# vessels rarely exceed ~45-50 knots; commercial tankers/cargo/bulk carriers (the bulk
# of this dataset's vessel_type distribution) cruise at 10-25 knots. 60 knots is a
# deliberately generous "implausible" threshold (an implied speed above this cannot be
# explained by any real vessel in this AOI), not a tight statistical cutoff.
KNOTS_TO_MPS = 0.514444


def load_ais_csv(path: str) -> pd.DataFrame:
    """Load the raw AIS CSV, parsing `timestamp` to UTC `datetime64[ns]` and validating
    every documented column is present. Does not filter, dedupe, or otherwise modify
    any row -- a faithful, typed read of the file exactly as it exists on disk."""
    df = pd.read_csv(path)
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"AIS CSV at {path!r} is missing required columns: {missing_cols}")
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


@dataclass(frozen=True)
class AISDataQualityReport:
    """TASK AIS-2 requirement 1 -- every field is a plain descriptive fact about the
    raw CSV, computed once, never inferred/assumed."""

    n_rows: int
    n_unique_mmsi: int
    n_unique_imo: int
    n_unique_vessel_name: int
    timestamp_min: pd.Timestamp
    timestamp_max: pd.Timestamp
    vessel_type_counts: dict[str, int]
    missing_value_counts: dict[str, int]
    n_duplicate_rows: int
    n_duplicate_mmsi_timestamp: int
    n_invalid_lat: int
    n_invalid_lon: int
    n_zero_zero_coords: int
    mmsi_with_multiple_names: dict[int, list[str]]
    mmsi_with_multiple_imo: dict[int, list[int]]


def compute_data_quality_report(df: pd.DataFrame) -> AISDataQualityReport:
    """TASK AIS-2 requirement 1: row count, unique MMSI/IMO/name counts, timestamp
    range, vessel-type distribution, missing values, duplicate rows, duplicate
    (mmsi, timestamp) records, and coordinate validity -- computed directly from the
    already-loaded `df` (via `load_ais_csv`), nothing assumed."""
    lat_invalid = (df["lat"] < LAT_VALID_RANGE[0]) | (df["lat"] > LAT_VALID_RANGE[1]) | df["lat"].isna()
    lon_invalid = (df["lon"] < LON_VALID_RANGE[0]) | (df["lon"] > LON_VALID_RANGE[1]) | df["lon"].isna()
    zero_zero = (df["lat"] == 0.0) & (df["lon"] == 0.0)

    dup_mmsi_ts = df.duplicated(subset=["mmsi", "timestamp"], keep=False)

    names_per_mmsi = df.groupby("mmsi")["vessel_name"].unique()
    mmsi_multi_name = {int(m): list(v) for m, v in names_per_mmsi.items() if len(v) > 1}
    imo_per_mmsi = df.groupby("mmsi")["imo"].unique()
    mmsi_multi_imo = {int(m): [int(x) for x in v] for m, v in imo_per_mmsi.items() if len(v) > 1}

    return AISDataQualityReport(
        n_rows=len(df),
        n_unique_mmsi=int(df["mmsi"].nunique()),
        n_unique_imo=int(df["imo"].nunique()),
        n_unique_vessel_name=int(df["vessel_name"].nunique()),
        timestamp_min=df["timestamp"].min(),
        timestamp_max=df["timestamp"].max(),
        vessel_type_counts=df["vessel_type"].value_counts().to_dict(),
        missing_value_counts={c: int(df[c].isna().sum()) for c in df.columns},
        n_duplicate_rows=int(df.duplicated(keep=False).sum()),
        n_duplicate_mmsi_timestamp=int(dup_mmsi_ts.sum()),
        n_invalid_lat=int(lat_invalid.sum()),
        n_invalid_lon=int(lon_invalid.sum()),
        n_zero_zero_coords=int(zero_zero.sum()),
        mmsi_with_multiple_names=mmsi_multi_name,
        mmsi_with_multiple_imo=mmsi_multi_imo,
    )


def filter_to_event_window(df: pd.DataFrame, t72, t0) -> pd.DataFrame:
    """TASK AIS-2 requirement 2 (temporal half): rows with `timestamp` in `[t72, t0]`
    inclusive. `t72`/`t0` are datetime64-convertible; the caller supplies the EXACT
    Stage III backward-window boundaries (`T27`'s own T0/T72) -- this function invents
    no window of its own."""
    t72 = pd.Timestamp(t72, tz="UTC") if pd.Timestamp(t72).tzinfo is None else pd.Timestamp(t72)
    t0 = pd.Timestamp(t0, tz="UTC") if pd.Timestamp(t0).tzinfo is None else pd.Timestamp(t0)
    mask = (df["timestamp"] >= t72) & (df["timestamp"] <= t0)
    return df.loc[mask].copy()


def filter_to_aoi(df: pd.DataFrame, *, north: float, south: float, east: float, west: float) -> pd.DataFrame:
    """TASK AIS-2 requirement 2 (spatial half): rows with `(lat, lon)` inside the
    given AOI bounding box (inclusive). The caller supplies the EXACT project AOI --
    this function invents no region of its own."""
    mask = (
        (df["lat"] >= south) & (df["lat"] <= north)
        & (df["lon"] >= west) & (df["lon"] <= east)
    )
    return df.loc[mask].copy()


@dataclass
class VesselTrack:
    """TASK AIS-2 requirement 3/8: one chronological AIS track for one vessel, over
    whatever record subset it was built from (e.g. the event-window+AOI subset).
    Every vessel-identity field is preserved unchanged from the raw CSV row -- this is
    a reorganization of the existing rows, not a re-derivation of any field."""

    mmsi: int
    imo: int
    vessel_name: str
    vessel_type: str
    type_code: int
    length: int
    width: int
    draught: float
    timestamps: np.ndarray  # datetime64[ns], ascending
    lat: np.ndarray
    lon: np.ndarray
    sog: np.ndarray
    cog: np.ndarray

    @property
    def n_observations(self) -> int:
        return len(self.timestamps)


def build_vessel_tracks(df: pd.DataFrame) -> list[VesselTrack]:
    """TASK AIS-2 requirement 3: one chronological track per MMSI, sorted ascending by
    `timestamp`. If a single MMSI's rows carry more than one distinct `vessel_name`/
    `imo` (see `AISDataQualityReport.mmsi_with_multiple_{name,imo}`), the FIRST
    chronological row's identity fields are used for the track's own identity summary
    -- the full per-row identity is not lost (still in `df`), only the track-level
    single-value fields; this choice is documented here, not silently assumed."""
    tracks: list[VesselTrack] = []
    for mmsi, g in df.groupby("mmsi", sort=False):
        g = g.sort_values("timestamp")
        first = g.iloc[0]
        tracks.append(VesselTrack(
            mmsi=int(mmsi), imo=int(first["imo"]), vessel_name=str(first["vessel_name"]),
            vessel_type=str(first["vessel_type"]), type_code=int(first["type_code"]),
            length=int(first["length"]), width=int(first["width"]), draught=float(first["draught"]),
            timestamps=g["timestamp"].values.astype("datetime64[ns]"),
            lat=g["lat"].to_numpy(dtype=np.float64), lon=g["lon"].to_numpy(dtype=np.float64),
            sog=g["sog"].to_numpy(dtype=np.float64), cog=g["cog"].to_numpy(dtype=np.float64),
        ))
    tracks.sort(key=lambda t: t.mmsi)
    return tracks


@dataclass(frozen=True)
class TrackTemporalGaps:
    """TASK AIS-2 requirement 4: per-vessel consecutive-observation interval stats.
    `interval_hours` has length `n_observations - 1` (empty for a single-observation
    track). No position is ever invented to fill a gap; this is a pure measurement of
    the gaps that already exist in the observed data."""

    mmsi: int
    n_observations: int
    interval_hours: np.ndarray
    median_interval_hours: float
    max_interval_hours: float
    n_gaps_over_threshold: int


def compute_track_temporal_gaps(
    track: VesselTrack, *, gap_threshold_hours: float = 6.0
) -> TrackTemporalGaps:
    """Consecutive-observation time intervals for one track, plus a count of gaps
    exceeding `gap_threshold_hours` (explicit, configurable -- not a silent constant;
    default 6h is 6x this dataset's own most common ~1h reporting cadence, discovered
    empirically in the manual-check notebook, not assumed beforehand)."""
    if track.n_observations < 2:
        return TrackTemporalGaps(
            mmsi=track.mmsi, n_observations=track.n_observations,
            interval_hours=np.array([]), median_interval_hours=float("nan"),
            max_interval_hours=float("nan"), n_gaps_over_threshold=0,
        )
    deltas_ns = np.diff(track.timestamps.astype("datetime64[ns]").astype("int64"))
    interval_hours = deltas_ns / 1e9 / 3600.0
    return TrackTemporalGaps(
        mmsi=track.mmsi, n_observations=track.n_observations, interval_hours=interval_hours,
        median_interval_hours=float(np.median(interval_hours)),
        max_interval_hours=float(np.max(interval_hours)),
        n_gaps_over_threshold=int(np.sum(interval_hours > gap_threshold_hours)),
    )


@dataclass(frozen=True)
class TrackPositionalJump:
    """TASK AIS-2 requirement 5: one consecutive-position displacement, its implied
    speed, and whether it exceeds `DEFAULT_MAX_PLAUSIBLE_SPEED_KNOTS`. Flagged only --
    never used, by this module, to drop the underlying row."""

    mmsi: int
    idx: int  # index of the LATER of the two positions in the track's own arrays
    t_prev: np.datetime64
    t_curr: np.datetime64
    distance_m: float
    interval_hours: float
    implied_speed_knots: float
    reported_sog_knots: float
    is_implausible: bool


def compute_track_positional_jumps(
    track: VesselTrack, *, max_plausible_speed_knots: float = DEFAULT_MAX_PLAUSIBLE_SPEED_KNOTS
) -> list[TrackPositionalJump]:
    """Consecutive-position distance (via this project's own `project_to_local_metres`,
    reused unchanged) vs. the real timestamp interval, for every step in `track`.
    `implied_speed_knots` is the OBSERVED displacement's own speed -- compared against,
    but never replaced by, the vessel's own reported `sog` (this task's own explicit
    instruction: SOG/COG are metadata, not assumed to perfectly reconstruct
    displacement). A jump is flagged `is_implausible` if its implied speed exceeds
    `max_plausible_speed_knots` -- reported, never silently dropped."""
    jumps: list[TrackPositionalJump] = []
    for i in range(1, track.n_observations):
        t_prev, t_curr = track.timestamps[i - 1], track.timestamps[i]
        interval_hours = float((t_curr - t_prev) / np.timedelta64(1, "h"))
        x_m, y_m = project_to_local_metres(
            np.array([track.lat[i]]), np.array([track.lon[i]]),
            lat_ref=float(track.lat[i - 1]), lon_ref=float(track.lon[i - 1]),
        )
        distance_m = float(np.sqrt(x_m[0] ** 2 + y_m[0] ** 2))
        if interval_hours > 0:
            implied_speed_mps = distance_m / (interval_hours * 3600.0)
            implied_speed_knots = implied_speed_mps / KNOTS_TO_MPS
        else:
            implied_speed_knots = float("inf") if distance_m > 0 else 0.0
        jumps.append(TrackPositionalJump(
            mmsi=track.mmsi, idx=i, t_prev=t_prev, t_curr=t_curr,
            distance_m=distance_m, interval_hours=interval_hours,
            implied_speed_knots=implied_speed_knots, reported_sog_knots=float(track.sog[i]),
            is_implausible=implied_speed_knots > max_plausible_speed_knots,
        ))
    return jumps


def tracks_to_dataframe(tracks: list[VesselTrack]) -> pd.DataFrame:
    """Flatten `tracks` back into one row-per-observation DataFrame (TASK AIS-2
    requirement 9's CSV/parquet output) -- preserves every vessel-identity field
    (requirement 8) alongside each observation's own position/sog/cog/timestamp."""
    rows = []
    for tr in tracks:
        for i in range(tr.n_observations):
            rows.append(dict(
                mmsi=tr.mmsi, imo=tr.imo, vessel_name=tr.vessel_name, vessel_type=tr.vessel_type,
                type_code=tr.type_code, length=tr.length, width=tr.width, draught=tr.draught,
                timestamp=tr.timestamps[i], lat=tr.lat[i], lon=tr.lon[i],
                sog=tr.sog[i], cog=tr.cog[i],
            ))
    return pd.DataFrame(rows)
