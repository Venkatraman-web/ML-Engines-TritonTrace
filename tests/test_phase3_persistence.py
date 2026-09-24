"""Focused T31 checks for `iste.phase3_persistence` (Phase 3 temporal persistence
tracking).

Run with:
    python -m unittest tests.test_phase3_persistence -v
"""

from __future__ import annotations

import unittest

import numpy as np

from iste.candidate_extraction import Candidate, CoreRegion
from iste.phase3_persistence import (
    DEFAULT_INTERVAL_HOURS,
    PersistenceTrack,
    track_candidate_persistence,
)

T0 = np.datetime64("2019-10-14T23:00:00", "ns")


def _hour(k: int) -> np.datetime64:
    return T0 - np.timedelta64(k, "h")


def _candidate(lat, lon, t_s, support_score=100.0, cand_id=None) -> Candidate:
    t_s = np.datetime64(t_s, "ns")
    if cand_id is None:
        cand_id = f"{np.datetime_as_string(t_s, unit='s')}_cand000"
    return Candidate(
        candidate_id=cand_id, x_peak=lon, y_peak=lat, t_s=t_s,
        core_region=CoreRegion(center_lat=lat, center_lon=lon, radius_m=1000.0),
        support_score=support_score, density_score=support_score, weighted_support_score=support_score,
        pixel_diversity_score=1.0, peak_support_features=None,
    )


class TestFiveConsecutiveHoursScoresHighPersistence(unittest.TestCase):
    def test_five_consecutive_candidates_form_one_track(self):
        # same region (small drift each hour), 5 consecutive hourly timestamps
        candidates = [
            _candidate(34.05 + 0.0002 * k, 35.26 + 0.0002 * k, _hour(k), support_score=100.0 + k)
            for k in range(5)
        ]
        tracks = track_candidate_persistence(candidates)
        self.assertEqual(len(tracks), 1)
        track = tracks[0]
        self.assertEqual(track.persistence_count, 5)
        self.assertEqual(track.persistence_duration_hours, 4.0)
        self.assertAlmostEqual(track.temporal_continuity(), 1.0)

    def test_high_persistence_beats_single_spike_on_count_and_duration(self):
        five_hour_candidates = [
            _candidate(34.05 + 0.0002 * k, 35.26 + 0.0002 * k, _hour(k)) for k in range(5)
        ]
        spike_candidate = [_candidate(36.0, 37.0, _hour(30), support_score=99999.0)]  # far away, huge support
        tracks = track_candidate_persistence(five_hour_candidates + spike_candidate)
        self.assertEqual(len(tracks), 2)
        by_count = sorted(tracks, key=lambda t: -t.persistence_count)
        high, low = by_count[0], by_count[1]
        self.assertEqual(high.persistence_count, 5)
        self.assertEqual(low.persistence_count, 1)
        self.assertGreater(high.persistence_duration_hours, low.persistence_duration_hours)
        # confirms persistence, not raw support magnitude, is what T31 measures -- the
        # single spike has vastly higher support but must NOT score higher on persistence.
        self.assertGreater(high.persistence_count, low.persistence_count)


class TestOneTimeSpikeScoresLowPersistence(unittest.TestCase):
    def test_single_timestamp_candidate_is_its_own_track(self):
        candidates = [_candidate(34.05, 35.26, T0)]
        tracks = track_candidate_persistence(candidates)
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0].persistence_count, 1)
        self.assertEqual(tracks[0].persistence_duration_hours, 0.0)


class TestSpatialMatchingRadius(unittest.TestCase):
    def test_far_apart_candidates_at_adjacent_hours_do_not_merge(self):
        a = _candidate(34.05, 35.26, _hour(0))
        b = _candidate(40.0, 45.0, _hour(1))  # far away, well outside any reasonable radius
        tracks = track_candidate_persistence([a, b], spatial_matching_radius_m=2000.0)
        self.assertEqual(len(tracks), 2)

    def test_within_radius_at_adjacent_hours_merge(self):
        a = _candidate(34.05000, 35.26000, _hour(0))
        b = _candidate(34.05010, 35.26010, _hour(1))  # ~roughly 100s of metres away
        tracks = track_candidate_persistence([a, b], spatial_matching_radius_m=5000.0)
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0].persistence_count, 2)

    def test_radius_validation(self):
        with self.assertRaises(ValueError):
            track_candidate_persistence([_candidate(34.0, 35.0, T0)], spatial_matching_radius_m=0.0)
        with self.assertRaises(ValueError):
            track_candidate_persistence([_candidate(34.0, 35.0, T0)], spatial_matching_radius_m=-1.0)


class TestGapTolerance(unittest.TestCase):
    def test_default_gap_tolerance_bridges_one_missed_hour(self):
        # candidate at hour 0 and hour 2 (hour 1 missing) -- same location, must merge
        # under the default max_gap_hours=1.0 (extra) + interval_hours=1.0 tolerance.
        a = _candidate(34.05, 35.26, _hour(0))
        b = _candidate(34.05, 35.26, _hour(2))
        tracks = track_candidate_persistence([a, b])
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0].persistence_count, 2)
        self.assertEqual(tracks[0].persistence_duration_hours, 2.0)
        # continuity < 1.0: 2 actual members over a 3-slot span (hours 0,1,2)
        self.assertAlmostEqual(tracks[0].temporal_continuity(), 2.0 / 3.0)

    def test_strict_zero_gap_tolerance_does_not_bridge(self):
        a = _candidate(34.05, 35.26, _hour(0))
        b = _candidate(34.05, 35.26, _hour(2))
        tracks = track_candidate_persistence([a, b], max_gap_hours=0.0)
        self.assertEqual(len(tracks), 2)

    def test_gap_beyond_tolerance_does_not_bridge(self):
        a = _candidate(34.05, 35.26, _hour(0))
        b = _candidate(34.05, 35.26, _hour(5))  # 4 missed hours, exceeds default tolerance
        tracks = track_candidate_persistence([a, b])
        self.assertEqual(len(tracks), 2)

    def test_negative_max_gap_hours_rejected(self):
        with self.assertRaises(ValueError):
            track_candidate_persistence([_candidate(34.0, 35.0, T0)], max_gap_hours=-1.0)


class TestPerTrackStatistics(unittest.TestCase):
    def test_mean_max_variance_support(self):
        candidates = [
            _candidate(34.05, 35.26, _hour(0), support_score=10.0),
            _candidate(34.05, 35.26, _hour(1), support_score=20.0),
            _candidate(34.05, 35.26, _hour(2), support_score=30.0),
        ]
        tracks = track_candidate_persistence(candidates)
        track = tracks[0]
        self.assertAlmostEqual(track.mean_support, 20.0)
        self.assertAlmostEqual(track.max_support, 30.0)
        self.assertAlmostEqual(track.support_variance, np.var([10.0, 20.0, 30.0]))

    def test_multiple_modes_per_timestamp_stay_separate_tracks(self):
        # T17 explicitly supports multiple retained modes per timestamp -- two
        # well-separated candidates at the SAME t_s must never merge into one track.
        a = _candidate(34.0, 35.0, T0, cand_id="a")
        b = _candidate(40.0, 45.0, T0, cand_id="b")
        tracks = track_candidate_persistence([a, b])
        self.assertEqual(len(tracks), 2)


class TestNoAgeScoring(unittest.TestCase):
    def test_identical_track_shape_at_different_ages_scores_identically(self):
        # a 3-hour track near t_obs vs. an identical-shape 3-hour track deep in the window
        # must produce IDENTICAL persistence statistics -- nothing in T31 depends on age.
        near = [_candidate(34.05, 35.26, _hour(k), support_score=50.0) for k in range(3)]
        far = [_candidate(50.0, 60.0, _hour(60 + k), support_score=50.0) for k in range(3)]
        tracks = track_candidate_persistence(near + far)
        self.assertEqual(len(tracks), 2)
        t_near = next(t for t in tracks if t.earliest_t_s == _hour(2))
        t_far = next(t for t in tracks if t.earliest_t_s == _hour(62))
        self.assertEqual(t_near.persistence_count, t_far.persistence_count)
        self.assertEqual(t_near.persistence_duration_hours, t_far.persistence_duration_hours)
        self.assertAlmostEqual(t_near.mean_support, t_far.mean_support)
        self.assertAlmostEqual(t_near.temporal_continuity(), t_far.temporal_continuity())


class TestEveryCandidateAssignedExactlyOnce(unittest.TestCase):
    def test_all_candidate_ids_preserved_exactly_once(self):
        candidates = [
            _candidate(34.0 + 0.001 * i, 35.0, _hour(i % 10), cand_id=f"c{i}") for i in range(20)
        ]
        tracks = track_candidate_persistence(candidates)
        all_ids = [cid for t in tracks for cid in t.candidate_ids]
        self.assertEqual(sorted(all_ids), sorted(c.candidate_id for c in candidates))
        self.assertEqual(len(all_ids), len(set(all_ids)))

    def test_empty_input(self):
        self.assertEqual(track_candidate_persistence([]), [])


if __name__ == "__main__":
    unittest.main()
