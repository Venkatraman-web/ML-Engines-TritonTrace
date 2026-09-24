"""Focused TASK AIS-2 checks for `iste.ais_data` (historical AIS preparation/
validation -- no ranking, no AIS score; see the module's own docstring).

Run with:
    python -m unittest tests.test_ais_data -v
"""

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from iste.ais_data import (
    DEFAULT_MAX_PLAUSIBLE_SPEED_KNOTS,
    build_vessel_tracks,
    compute_data_quality_report,
    compute_track_positional_jumps,
    compute_track_temporal_gaps,
    filter_to_aoi,
    filter_to_event_window,
    load_ais_csv,
    tracks_to_dataframe,
)

COLUMNS = ["mmsi", "timestamp", "lat", "lon", "sog", "cog", "vessel_name", "imo", "vessel_type", "type_code", "length", "width", "draught"]


def _row(mmsi, ts, lat, lon, sog=10.0, cog=90.0, name="SHIP", imo=1234567, vtype="Cargo", type_code=70, length=100, width=20, draught=8.0):
    return dict(mmsi=mmsi, timestamp=ts, lat=lat, lon=lon, sog=sog, cog=cog, vessel_name=name, imo=imo, vessel_type=vtype, type_code=type_code, length=length, width=width, draught=draught)


def _make_df(rows) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=COLUMNS)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


class TestLoadAisCsv(unittest.TestCase):
    def test_loads_and_parses_timestamp(self):
        rows = [_row(1, "2019-06-12T00:00:00Z", 34.0, 35.0)]
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "ais.csv")
            _make_df(rows).to_csv(path, index=False)
            df = load_ais_csv(path)
        self.assertEqual(len(df), 1)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df["timestamp"]))

    def test_missing_column_raises(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "bad.csv")
            pd.DataFrame({"mmsi": [1]}).to_csv(path, index=False)
            with self.assertRaises(ValueError):
                load_ais_csv(path)


class TestDataQualityReport(unittest.TestCase):
    def test_basic_counts(self):
        rows = [
            _row(1, "2019-06-12T00:00:00Z", 34.0, 35.0, name="A"),
            _row(1, "2019-06-12T01:00:00Z", 34.1, 35.1, name="A"),
            _row(2, "2019-06-12T00:00:00Z", 33.0, 34.0, name="B"),
        ]
        df = _make_df(rows)
        report = compute_data_quality_report(df)
        self.assertEqual(report.n_rows, 3)
        self.assertEqual(report.n_unique_mmsi, 2)
        self.assertEqual(report.n_duplicate_rows, 0)
        self.assertEqual(report.n_duplicate_mmsi_timestamp, 0)
        self.assertEqual(report.n_invalid_lat, 0)
        self.assertEqual(report.n_invalid_lon, 0)

    def test_detects_duplicate_mmsi_timestamp(self):
        rows = [
            _row(1, "2019-06-12T00:00:00Z", 34.0, 35.0),
            _row(1, "2019-06-12T00:00:00Z", 34.05, 35.05),  # same mmsi+timestamp, different position
        ]
        df = _make_df(rows)
        report = compute_data_quality_report(df)
        self.assertEqual(report.n_duplicate_mmsi_timestamp, 2)

    def test_detects_invalid_coordinates(self):
        rows = [
            _row(1, "2019-06-12T00:00:00Z", 999.0, 35.0),
            _row(1, "2019-06-12T01:00:00Z", 34.0, -999.0),
        ]
        df = _make_df(rows)
        report = compute_data_quality_report(df)
        self.assertEqual(report.n_invalid_lat, 1)
        self.assertEqual(report.n_invalid_lon, 1)

    def test_detects_mmsi_with_multiple_names(self):
        rows = [
            _row(1, "2019-06-12T00:00:00Z", 34.0, 35.0, name="ALPHA"),
            _row(1, "2019-06-12T01:00:00Z", 34.1, 35.1, name="BETA"),
        ]
        df = _make_df(rows)
        report = compute_data_quality_report(df)
        self.assertIn(1, report.mmsi_with_multiple_names)


class TestFilterToEventWindow(unittest.TestCase):
    def test_filters_inclusive_range(self):
        rows = [
            _row(1, "2019-06-11T20:35:58Z", 34.0, 35.0),  # exactly T72
            _row(1, "2019-06-13T00:00:00Z", 34.0, 35.0),  # inside
            _row(1, "2019-06-14T20:35:58Z", 34.0, 35.0),  # exactly T0
            _row(1, "2019-06-15T00:00:00Z", 34.0, 35.0),  # after window
            _row(1, "2019-06-10T00:00:00Z", 34.0, 35.0),  # before window
        ]
        df = _make_df(rows)
        filtered = filter_to_event_window(df, "2019-06-11T20:35:58Z", "2019-06-14T20:35:58Z")
        self.assertEqual(len(filtered), 3)


class TestFilterToAOI(unittest.TestCase):
    def test_filters_bounding_box(self):
        rows = [
            _row(1, "2019-06-12T00:00:00Z", 34.0, 35.0),   # inside AOI
            _row(1, "2019-06-12T01:00:00Z", 40.0, 35.0),   # outside (lat too high)
            _row(1, "2019-06-12T02:00:00Z", 34.0, 10.0),   # outside (lon too low)
        ]
        df = _make_df(rows)
        filtered = filter_to_aoi(df, north=36.372, south=30.1875, east=36.088, west=27.121)
        self.assertEqual(len(filtered), 1)


class TestBuildVesselTracks(unittest.TestCase):
    def test_one_track_per_mmsi_sorted_ascending(self):
        rows = [
            _row(2, "2019-06-12T01:00:00Z", 33.0, 34.0),
            _row(1, "2019-06-12T01:00:00Z", 34.1, 35.1),
            _row(1, "2019-06-12T00:00:00Z", 34.0, 35.0),
        ]
        df = _make_df(rows)
        tracks = build_vessel_tracks(df)
        self.assertEqual(len(tracks), 2)
        track1 = next(t for t in tracks if t.mmsi == 1)
        self.assertEqual(track1.n_observations, 2)
        self.assertTrue(np.all(np.diff(track1.timestamps.astype("int64")) > 0))

    def test_preserves_identity_fields(self):
        rows = [_row(1, "2019-06-12T00:00:00Z", 34.0, 35.0, imo=9999999, name="MY SHIP", vtype="Tanker", length=200, width=30, draught=12.5)]
        df = _make_df(rows)
        tracks = build_vessel_tracks(df)
        t = tracks[0]
        self.assertEqual(t.imo, 9999999)
        self.assertEqual(t.vessel_name, "MY SHIP")
        self.assertEqual(t.vessel_type, "Tanker")
        self.assertEqual(t.length, 200)
        self.assertEqual(t.width, 30)
        self.assertAlmostEqual(t.draught, 12.5)


class TestTemporalGaps(unittest.TestCase):
    def test_median_and_max_interval(self):
        rows = [
            _row(1, "2019-06-12T00:00:00Z", 34.0, 35.0),
            _row(1, "2019-06-12T01:00:00Z", 34.0, 35.0),
            _row(1, "2019-06-12T09:00:00Z", 34.0, 35.0),  # 8h gap
        ]
        df = _make_df(rows)
        track = build_vessel_tracks(df)[0]
        gaps = compute_track_temporal_gaps(track, gap_threshold_hours=6.0)
        self.assertEqual(gaps.n_observations, 3)
        np.testing.assert_allclose(gaps.interval_hours, [1.0, 8.0])
        self.assertAlmostEqual(gaps.median_interval_hours, 4.5)
        self.assertAlmostEqual(gaps.max_interval_hours, 8.0)
        self.assertEqual(gaps.n_gaps_over_threshold, 1)

    def test_single_observation_track_returns_empty_gaps(self):
        rows = [_row(1, "2019-06-12T00:00:00Z", 34.0, 35.0)]
        df = _make_df(rows)
        track = build_vessel_tracks(df)[0]
        gaps = compute_track_temporal_gaps(track)
        self.assertEqual(gaps.n_observations, 1)
        self.assertEqual(len(gaps.interval_hours), 0)
        self.assertTrue(np.isnan(gaps.median_interval_hours))


class TestPositionalJumps(unittest.TestCase):
    def test_stationary_vessel_zero_distance(self):
        rows = [
            _row(1, "2019-06-12T00:00:00Z", 34.0, 35.0),
            _row(1, "2019-06-12T01:00:00Z", 34.0, 35.0),
        ]
        df = _make_df(rows)
        track = build_vessel_tracks(df)[0]
        jumps = compute_track_positional_jumps(track)
        self.assertEqual(len(jumps), 1)
        self.assertAlmostEqual(jumps[0].distance_m, 0.0, places=3)
        self.assertAlmostEqual(jumps[0].implied_speed_knots, 0.0, places=3)
        self.assertFalse(jumps[0].is_implausible)

    def test_flags_implausible_jump_never_drops_it(self):
        # ~1 degree longitude at this latitude is roughly 92 km; in 1 minute that's
        # a wildly implausible speed for any real vessel.
        rows = [
            _row(1, "2019-06-12T00:00:00Z", 34.0, 35.0),
            _row(1, "2019-06-12T00:01:00Z", 34.0, 36.0),
        ]
        df = _make_df(rows)
        track = build_vessel_tracks(df)[0]
        jumps = compute_track_positional_jumps(track, max_plausible_speed_knots=DEFAULT_MAX_PLAUSIBLE_SPEED_KNOTS)
        self.assertEqual(len(jumps), 1)  # flagged, not dropped
        self.assertTrue(jumps[0].is_implausible)
        self.assertGreater(jumps[0].implied_speed_knots, DEFAULT_MAX_PLAUSIBLE_SPEED_KNOTS)

    def test_reported_sog_carried_through_unchanged(self):
        rows = [
            _row(1, "2019-06-12T00:00:00Z", 34.0, 35.0, sog=12.3),
            _row(1, "2019-06-12T01:00:00Z", 34.01, 35.01, sog=15.7),
        ]
        df = _make_df(rows)
        track = build_vessel_tracks(df)[0]
        jumps = compute_track_positional_jumps(track)
        self.assertAlmostEqual(jumps[0].reported_sog_knots, 15.7)


class TestTracksToDataframe(unittest.TestCase):
    def test_round_trip_row_count_and_identity(self):
        rows = [
            _row(1, "2019-06-12T00:00:00Z", 34.0, 35.0, name="A"),
            _row(1, "2019-06-12T01:00:00Z", 34.1, 35.1, name="A"),
            _row(2, "2019-06-12T00:00:00Z", 33.0, 34.0, name="B"),
        ]
        df = _make_df(rows)
        tracks = build_vessel_tracks(df)
        out = tracks_to_dataframe(tracks)
        self.assertEqual(len(out), 3)
        self.assertEqual(set(out["mmsi"]), {1, 2})
        self.assertIn("vessel_name", out.columns)
        self.assertIn("sog", out.columns)
        self.assertIn("cog", out.columns)


if __name__ == "__main__":
    unittest.main()
