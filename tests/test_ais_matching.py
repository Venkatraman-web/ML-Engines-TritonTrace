"""Focused TASK AIS-3 checks for `iste.ais_matching` (raw AIS-vessel-to-Phase3-
candidate matching evidence -- no ranking, no AIS score; see module docstring).

Run with:
    python -m unittest tests.test_ais_matching -v
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from iste.ais_matching import (
    DEFAULT_CLOSE_ENCOUNTER_DISTANCE_KM,
    analyze_timestamp_alignment,
    analyze_vessel_track,
    compute_vessel_candidate_matches,
    haversine_distance_km,
    summarize_vessel_matches,
)


class TestHaversineDistance(unittest.TestCase):
    def test_zero_distance_for_identical_points(self):
        d = haversine_distance_km(34.0, 35.0, 34.0, 35.0)
        self.assertAlmostEqual(float(d), 0.0, places=6)

    def test_known_distance_one_degree_latitude(self):
        # 1 degree of latitude is ~111.19 km everywhere on a sphere (independent of longitude)
        d = haversine_distance_km(34.0, 35.0, 35.0, 35.0)
        self.assertAlmostEqual(float(d), 111.19, delta=0.5)

    def test_symmetric(self):
        d1 = haversine_distance_km(34.0, 35.0, 36.0, 37.0)
        d2 = haversine_distance_km(36.0, 37.0, 34.0, 35.0)
        self.assertAlmostEqual(float(d1), float(d2), places=8)

    def test_vectorized_over_arrays(self):
        lat2 = np.array([34.0, 35.0, 36.0])
        lon2 = np.array([35.0, 35.0, 35.0])
        d = haversine_distance_km(34.0, 35.0, lat2, lon2)
        self.assertEqual(d.shape, (3,))
        self.assertAlmostEqual(float(d[0]), 0.0, places=6)


class TestTimestampAlignment(unittest.TestCase):
    def test_detects_on_grid_ais_timestamps(self):
        cand_ts = pd.Series([pd.Timestamp("2019-06-12T00:35:58Z"), pd.Timestamp("2019-06-12T01:35:58Z")])
        ais_ts = pd.Series([pd.Timestamp("2019-06-12T00:35:58Z"), pd.Timestamp("2019-06-12T01:35:58Z")])
        report = analyze_timestamp_alignment(cand_ts, ais_ts)
        self.assertEqual(report.n_ais_timestamps_exactly_on_candidate_grid, 2)
        self.assertAlmostEqual(report.fraction_ais_timestamps_on_grid, 1.0)
        self.assertAlmostEqual(report.typical_offset_seconds, 0.0)

    def test_detects_offset_ais_timestamps(self):
        cand_ts = pd.Series([pd.Timestamp("2019-06-12T00:35:58Z")])
        ais_ts = pd.Series([pd.Timestamp("2019-06-12T00:05:00Z")])  # ~30m58s off
        report = analyze_timestamp_alignment(cand_ts, ais_ts)
        self.assertEqual(report.n_ais_timestamps_exactly_on_candidate_grid, 0)
        self.assertGreater(report.typical_offset_seconds, 1800)  # > 30 minutes


def _candidates_df():
    return pd.DataFrame([
        dict(candidate_id="c1", t_s=pd.Timestamp("2019-06-12T00:35:58Z"), lat=34.0, lon=35.0, support_score=100.0, cluster_id="cluster0000"),
        dict(candidate_id="c2", t_s=pd.Timestamp("2019-06-12T01:35:58Z"), lat=34.1, lon=35.1, support_score=50.0, cluster_id="cluster0001"),
    ])


def _ais_df():
    return pd.DataFrame([
        dict(mmsi=1, vessel_name="A", vessel_type="Fishing", imo=111, timestamp=pd.Timestamp("2019-06-12T00:35:58Z"), lat=34.0, lon=35.0, sog=5.0, cog=90.0),  # exact match with c1, distance 0
        dict(mmsi=1, vessel_name="A", vessel_type="Fishing", imo=111, timestamp=pd.Timestamp("2019-06-12T01:35:58Z"), lat=34.1, lon=35.1, sog=5.0, cog=90.0),  # exact match with c2, distance 0
        dict(mmsi=2, vessel_name="B", vessel_type="Cargo", imo=222, timestamp=pd.Timestamp("2019-06-12T02:35:58Z"), lat=34.0, lon=35.0, sog=8.0, cog=180.0),  # far in time from both (2h from c1 exact bound is 2h... within tol, 1h from c2)
        dict(mmsi=3, vessel_name="C", vessel_type="Tanker", imo=333, timestamp=pd.Timestamp("2019-06-12T10:35:58Z"), lat=34.0, lon=35.0, sog=8.0, cog=180.0),  # way outside tolerance
    ])


class TestComputeMatches(unittest.TestCase):
    def test_exact_match_zero_distance(self):
        matches = compute_vessel_candidate_matches(_candidates_df(), _ais_df(), max_time_tolerance_hours=2.0)
        pair_rows = matches[(matches["mmsi"] == 1) & (matches["candidate_id"] == "c1")]
        self.assertEqual(len(pair_rows), 2)  # AIS obs at 00:35:58 (0h diff) and 01:35:58 (1h diff), both within +/-2h
        exact_rows = pair_rows[pair_rows["is_exact"]]
        self.assertEqual(len(exact_rows), 1)
        self.assertAlmostEqual(exact_rows.iloc[0]["distance_km"], 0.0, places=6)
        self.assertTrue(exact_rows.iloc[0]["is_within_1h"])

    def test_out_of_tolerance_vessel_excluded(self):
        matches = compute_vessel_candidate_matches(_candidates_df(), _ais_df(), max_time_tolerance_hours=2.0)
        self.assertNotIn(3, matches["mmsi"].values)

    def test_preserves_vessel_identity_and_candidate_fields(self):
        matches = compute_vessel_candidate_matches(_candidates_df(), _ais_df(), max_time_tolerance_hours=2.0)
        row = matches[(matches["mmsi"] == 1) & (matches["candidate_id"] == "c1")].iloc[0]
        self.assertEqual(row["vessel_name"], "A")
        self.assertEqual(row["vessel_type"], "Fishing")
        self.assertEqual(row["imo"], 111)
        self.assertEqual(row["candidate_cluster_id"], "cluster0000")
        self.assertAlmostEqual(row["candidate_support"], 100.0)

    def test_within_2h_not_within_1h_flagged_correctly(self):
        matches = compute_vessel_candidate_matches(_candidates_df(), _ais_df(), max_time_tolerance_hours=2.0)
        row = matches[(matches["mmsi"] == 2) & (matches["candidate_id"] == "c1")]
        self.assertEqual(len(row), 1)
        r = row.iloc[0]
        self.assertAlmostEqual(r["time_diff_hours"], 2.0, places=3)
        self.assertFalse(r["is_exact"])
        self.assertFalse(r["is_within_1h"])
        self.assertTrue(r["is_within_2h"])

    def test_no_position_interpolated_only_real_ais_rows_used(self):
        matches = compute_vessel_candidate_matches(_candidates_df(), _ais_df(), max_time_tolerance_hours=2.0)
        real_positions = set(zip(_ais_df()["lat"], _ais_df()["lon"]))
        for _, r in matches.iterrows():
            self.assertIn((r["ais_lat"], r["ais_lon"]), real_positions)


class TestSummarizeVesselMatches(unittest.TestCase):
    def test_includes_vessel_with_zero_matches(self):
        matches = compute_vessel_candidate_matches(_candidates_df(), _ais_df(), max_time_tolerance_hours=2.0)
        summary = summarize_vessel_matches(matches, _ais_df(), n_candidate_timestamps=2)
        self.assertIn(3, summary["mmsi"].values)
        row3 = summary[summary["mmsi"] == 3].iloc[0]
        self.assertEqual(row3["n_pair_matches_within_2h"], 0)
        self.assertTrue(np.isnan(row3["min_distance_km"]))

    def test_min_distance_fields_correct_for_matched_vessel(self):
        matches = compute_vessel_candidate_matches(_candidates_df(), _ais_df(), max_time_tolerance_hours=2.0)
        summary = summarize_vessel_matches(matches, _ais_df(), n_candidate_timestamps=2)
        row1 = summary[summary["mmsi"] == 1].iloc[0]
        self.assertAlmostEqual(row1["min_distance_km"], 0.0, places=6)
        self.assertEqual(row1["n_distinct_candidate_timestamps_matched_exact"], 2)

    def test_distance_bin_counts(self):
        matches = compute_vessel_candidate_matches(_candidates_df(), _ais_df(), max_time_tolerance_hours=2.0)
        summary = summarize_vessel_matches(matches, _ais_df(), n_candidate_timestamps=2, distance_bins_km=(1.0, 5.0))
        row1 = summary[summary["mmsi"] == 1].iloc[0]
        self.assertEqual(row1["n_matches_within_1km"], 2)  # both exact-distance matches


class TestAnalyzeVesselTrack(unittest.TestCase):
    def test_no_matches_returns_nan_gracefully(self):
        matches = compute_vessel_candidate_matches(_candidates_df(), _ais_df(), max_time_tolerance_hours=2.0)
        result = analyze_vessel_track(3, matches)
        self.assertEqual(result.n_close_encounters, 0)
        self.assertTrue(np.isnan(result.min_synchronized_distance_km))

    def test_consecutive_checkpoints_counted(self):
        matches = compute_vessel_candidate_matches(_candidates_df(), _ais_df(), max_time_tolerance_hours=2.0)
        result = analyze_vessel_track(1, matches, close_encounter_distance_km=DEFAULT_CLOSE_ENCOUNTER_DISTANCE_KM)
        # vessel 1 has close encounters (distance 0) at both c1 (00:35:58) and c2 (01:35:58) -- consecutive 1h-apart checkpoints
        self.assertEqual(result.max_consecutive_close_encounter_checkpoints, 2)
        self.assertAlmostEqual(result.min_synchronized_distance_km, 0.0, places=6)

    def test_no_score_or_rank_field_exists(self):
        matches = compute_vessel_candidate_matches(_candidates_df(), _ais_df(), max_time_tolerance_hours=2.0)
        result = analyze_vessel_track(1, matches)
        field_names = result.__dataclass_fields__.keys()
        for forbidden in ("score", "rank", "best"):
            self.assertFalse(any(forbidden in f.lower() for f in field_names), f"found forbidden field containing {forbidden!r}")


if __name__ == "__main__":
    unittest.main()
