"""TASK AIS-6 -- concrete `AISDataSource` backed by the real AIS-2 event-window tracks.

Implements `iste.phase3_ais.AISDataSource`'s abstract interface (fixed at T35's own
approval, unmodified here) against the real, already-validated AIS-2 vessel-track CSV
(`result/AIS_vessel_tracks_event_window.csv`). Kept as its own small, focused module
(CLAUDE.md: "keep modules small and focused") rather than adding a concrete
implementation into `iste/phase3_ais.py` itself, which stays unchanged in every respect
except its own `compute_ais_evidence` real-path implementation.

Distances reuse `iste.ais_matching.haversine_distance_km` (great-circle, already
established by AIS-3) -- no new distance formula.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from iste.ais_matching import haversine_distance_km
from iste.phase3_ais import AISDataSource, VesselTrackSegment


class RealAISDataSource(AISDataSource):
    """Queries the real, already-validated AIS-2 event-window+AOI vessel tracks
    directly -- no synthetic/interpolated data, no re-derivation of AIS-3's own bulk
    matching pipeline (this is a genuine point-query implementation of T35's own
    abstract interface, used by `iste.phase3_ais.compute_ais_evidence`'s real path)."""

    def __init__(self, ais_track_csv_path: str):
        df = pd.read_csv(ais_track_csv_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        self._df = df

    def query_vessels_near(
        self, lat: float, lon: float, t: np.datetime64, radius_m: float, time_window_hours: float
    ) -> list[VesselTrackSegment]:
        t_ts = pd.Timestamp(t)
        if t_ts.tzinfo is None:
            t_ts = t_ts.tz_localize("UTC")
        window = self._df[
            (self._df["timestamp"] >= t_ts - pd.Timedelta(hours=time_window_hours))
            & (self._df["timestamp"] <= t_ts + pd.Timedelta(hours=time_window_hours))
        ]
        if len(window) == 0:
            return []
        d_km = haversine_distance_km(lat, lon, window["lat"].to_numpy(), window["lon"].to_numpy())
        within = window.loc[d_km <= (radius_m / 1000.0)]
        return [
            VesselTrackSegment(
                vessel_id=str(r["mmsi"]), lat=float(r["lat"]), lon=float(r["lon"]),
                t=np.datetime64(r["timestamp"].tz_convert("UTC").tz_localize(None)),
            )
            for _, r in within.iterrows()
        ]
