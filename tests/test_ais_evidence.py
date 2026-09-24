"""Focused TASK AIS-4 checks for `iste.ais_evidence` (per-vessel/per-checkpoint AIS
evidence aggregation -- no ranking, no AIS score; see module docstring).

Run with:
    python -m unittest tests.test_ais_evidence -v
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from iste.ais_evidence import (
    compute_checkpoint_proximity,
    compute_vessel_evidence,
    run_circular_shift_null,
)
from iste.ais_matching import compute_vessel_candidate_matches


def _candidates_df():
    return pd.DataFrame([
        dict(candidate_id="c1", t_s=pd.Timestamp("2019-06-12T00:35:58Z"), lat=34.0, lon=35.0, support_score=100.0, cluster_id="cluster0000"),
        dict(candidate_id="c2", t_s=pd.Timestamp("2019-06-12T01:35:58Z"), lat=34.1, lon=35.1, support_score=50.0, cluster_id="cluster0001"),
        dict(candidate_id="c3", t_s=pd.Timestamp("2019-06-12T02:35:58Z"), lat=34.2, lon=35.2, support_score=25.0, cluster_id="cluster0001"),
    ])


def _ais_df():
    rows = [
        # vessel 1: exact matches at all 3 checkpoints, close to c1/c2/c3 (consecutive close encounters)
        dict(mmsi=1, vessel_name="A", vessel_type="Fishing", imo=111, timestamp=pd.Timestamp("2019-06-12T00:35:58Z"), lat=34.0, lon=35.0, sog=5.0, cog=90.0),
        dict(mmsi=1, vessel_name="A", vessel_type="Fishing", imo=111, timestamp=pd.Timestamp("2019-06-12T01:35:58Z"), lat=34.1, lon=35.1, sog=5.0, cog=90.0),
        dict(mmsi=1, vessel_name="A", vessel_type="Fishing", imo=111, timestamp=pd.Timestamp("2019-06-12T02:35:58Z"), lat=34.2, lon=35.2, sog=5.0, cog=90.0),
        # vessel 2: far away, no close encounters, but has observations near the grid
        dict(mmsi=2, vessel_name="B", vessel_type="Cargo", imo=222, timestamp=pd.Timestamp("2019-06-12T00:35:58Z"), lat=10.0, lon=10.0, sog=8.0, cog=180.0),
        # vessel 3: zero AIS observations anywhere near the event window (present in ais_df only via a distant timestamp, to test zero-match handling)
        dict(mmsi=3, vessel_name="C", vessel_type="Tanker", imo=333, timestamp=pd.Timestamp("2019-06-12T12:35:58Z"), lat=34.0, lon=35.0, sog=8.0, cog=180.0),
    ]
    return pd.DataFrame(rows)


def _gap_summary_df():
    return pd.DataFrame([
        dict(mmsi=1, vessel_name="A", vessel_type="Fishing", n_observations=3, median_interval_hours=1.0, max_interval_hours=1.0, n_gaps_over_threshold=0),
        dict(mmsi=2, vessel_name="B", vessel_type="Cargo", n_observations=1, median_interval_hours=np.nan, max_interval_hours=np.nan, n_gaps_over_threshold=0),
        dict(mmsi=3, vessel_name="C", vessel_type="Tanker", n_observations=1, median_interval_hours=np.nan, max_interval_hours=np.nan, n_gaps_over_threshold=0),
    ])


CANDIDATE_GRID = [pd.Timestamp("2019-06-12T00:35:58Z"), pd.Timestamp("2019-06-12T01:35:58Z"), pd.Timestamp("2019-06-12T02:35:58Z")]


class TestComputeVesselEvidence(unittest.TestCase):
    def setUp(self):
        self.matches = compute_vessel_candidate_matches(_candidates_df(), _ais_df(), max_time_tolerance_hours=2.0)
        self.evidence = compute_vessel_evidence(self.matches, _ais_df(), _gap_summary_df(), n_candidate_timestamps=3)

    def test_includes_every_vessel_even_zero_matches(self):
        self.assertEqual(set(self.evidence["mmsi"]), {1, 2, 3})

    def test_exact_columns_present_and_separate_from_tolerant(self):
        for prefix in ("exact_", "within_1h_", "within_2h_"):
            self.assertTrue(any(c.startswith(prefix) for c in self.evidence.columns), f"missing columns with prefix {prefix}")

    def test_vessel1_perfect_exact_coverage(self):
        row = self.evidence[self.evidence["mmsi"] == 1].iloc[0]
        self.assertEqual(row["exact_n_candidate_timestamps_covered"], 3)
        self.assertAlmostEqual(row["exact_fraction_candidate_timestamps_covered"], 1.0)
        self.assertAlmostEqual(row["exact_min_distance_km"], 0.0, places=6)
        self.assertEqual(row["exact_longest_consecutive_close_encounter_checkpoints"], 3)

    def test_count_normalized_fraction_prevents_observation_bias(self):
        row = self.evidence[self.evidence["mmsi"] == 1].iloc[0]
        self.assertIn("exact_frac_within_1km", row.index)
        self.assertAlmostEqual(row["exact_frac_within_1km"], row["exact_n_within_1km"] / row["exact_n_matches"])

    def test_missing_ais_reported_as_missing_not_negative(self):
        row = self.evidence[self.evidence["mmsi"] == 3].iloc[0]
        self.assertEqual(row["exact_n_matches"], 0)
        self.assertTrue(np.isnan(row["exact_min_distance_km"]))
        self.assertIn("missing information", row["data_availability_note"])
        self.assertIn("not evidence against", row["data_availability_note"])

    def test_ais2_data_quality_preserved(self):
        row = self.evidence[self.evidence["mmsi"] == 1].iloc[0]
        self.assertEqual(row["n_observations_total"], 3)
        self.assertAlmostEqual(row["median_interval_hours"], 1.0)

    def test_no_score_or_rank_column(self):
        for c in self.evidence.columns:
            self.assertNotIn("score", c.lower())
            self.assertNotIn("rank", c.lower())

    def test_best_encounter_fields_populated(self):
        row = self.evidence[self.evidence["mmsi"] == 1].iloc[0]
        self.assertEqual(row["exact_best_encounter_candidate_id"], "c1")
        self.assertAlmostEqual(row["exact_best_encounter_distance_km"], 0.0, places=6)


class TestCheckpointProximity(unittest.TestCase):
    def test_one_row_per_checkpoint(self):
        matches = compute_vessel_candidate_matches(_candidates_df(), _ais_df(), max_time_tolerance_hours=2.0)
        cp = compute_checkpoint_proximity(_candidates_df(), matches)
        self.assertEqual(len(cp), 3)
        self.assertEqual(set(pd.to_datetime(cp["t_s"])), set(CANDIDATE_GRID))

    def test_best_vessel_at_checkpoint_is_correct(self):
        matches = compute_vessel_candidate_matches(_candidates_df(), _ais_df(), max_time_tolerance_hours=2.0)
        cp = compute_checkpoint_proximity(_candidates_df(), matches)
        row = cp[pd.to_datetime(cp["t_s"]) == CANDIDATE_GRID[0]].iloc[0]
        self.assertEqual(row["best_mmsi"], 1)
        self.assertAlmostEqual(row["min_distance_km"], 0.0, places=6)


class TestCircularShiftNull(unittest.TestCase):
    def test_returns_none_for_insufficient_observations(self):
        result = run_circular_shift_null(2, _candidates_df(), _ais_df(), CANDIDATE_GRID)
        self.assertIsNone(result)

    def test_real_min_distance_matches_exact_evidence(self):
        result = run_circular_shift_null(1, _candidates_df(), _ais_df(), CANDIDATE_GRID)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result.real_min_distance_km, 0.0, places=6)

    def test_p_value_in_valid_range(self):
        result = run_circular_shift_null(1, _candidates_df(), _ais_df(), CANDIDATE_GRID)
        self.assertGreaterEqual(result.empirical_p_value, 0.0)
        self.assertLessEqual(result.empirical_p_value, 1.0)

    def test_exhaustive_shifts_count(self):
        result = run_circular_shift_null(1, _candidates_df(), _ais_df(), CANDIDATE_GRID)
        self.assertEqual(result.n_shifts_tested, len(CANDIDATE_GRID) - 1)

    def test_no_source_probability_language_in_result_fields(self):
        result = run_circular_shift_null(1, _candidates_df(), _ais_df(), CANDIDATE_GRID)
        field_names = result.__dataclass_fields__.keys()
        for forbidden in ("source_prob", "is_source", "rank"):
            self.assertNotIn(forbidden, field_names)


if __name__ == "__main__":
    unittest.main()
