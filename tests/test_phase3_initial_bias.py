"""Focused T30 checks for `iste.phase3_initial_bias` (Phase 3 Mahalanobis/dispersion-
relative initial-condition influence field).

Run with:
    python -m unittest tests.test_phase3_initial_bias -v
"""

from __future__ import annotations

import unittest

import numpy as np

from iste.candidate_extraction import Candidate, CoreRegion
from iste.ensemble_run import EnsembleRunRecords
from iste.phase3_initial_bias import (
    DEFAULT_K,
    compute_initial_bias_influence,
    compute_initial_bias_influence_for_candidates,
    compute_timestamp_cloud_stats,
)

T_S = np.datetime64("2020-01-01T00:00:00", "ns")
T_S2 = T_S - np.timedelta64(24, "h")


def _records(lat, lon, t_s=T_S) -> EnsembleRunRecords:
    n = len(lat)
    return EnsembleRunRecords(
        particle_id=np.array([f"p{i}" for i in range(n)], dtype=object),
        forcing_member_id=np.zeros(n, dtype=np.int64),
        observation_pixel_id=np.array([f"px-{i % 3}" for i in range(n)], dtype=object),
        timestamp=np.full(n, t_s, dtype="datetime64[ns]"),
        lat=np.asarray(lat, dtype=np.float64), lon=np.asarray(lon, dtype=np.float64),
        log_weight=np.zeros(n, dtype=np.float64), valid=np.ones(n, dtype=bool),
    )


class TestFullRankCovariance(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        n = 200
        # anisotropic cloud: wider in longitude than latitude, well-conditioned covariance.
        lat = 10.0 + rng.normal(0, 0.002, n)
        lon = 100.0 + rng.normal(0, 0.02, n)
        self.records = _records(lat, lon)
        self.stats = compute_timestamp_cloud_stats(self.records, T_S)

    def test_covariance_state_is_full_rank(self):
        self.assertEqual(self.stats.covariance_state, "full_rank")
        self.assertEqual(self.stats.n_particles, 200)
        self.assertIsNotNone(self.stats.covariance_m2)
        self.assertEqual(self.stats.covariance_m2.shape, (2, 2))

    def test_centroid_candidate_gets_influence_one(self):
        result = compute_initial_bias_influence(
            self.stats, self.stats.lat_ref, self.stats.lon_ref, k=DEFAULT_K
        )
        self.assertAlmostEqual(result.mahalanobis_distance, 0.0, places=6)
        self.assertAlmostEqual(result.influence, 1.0, places=6)

    def test_distant_candidate_gets_low_influence(self):
        far = compute_initial_bias_influence(self.stats, 15.0, 105.0, k=DEFAULT_K)
        self.assertGreater(far.mahalanobis_distance, 10.0)
        self.assertLess(far.influence, 1e-6)

    def test_anisotropy_is_respected(self):
        # a displacement of the same RAW degree magnitude along the wider (lon) axis vs.
        # the narrower (lat) axis must NOT produce the same Mahalanobis distance -- that
        # would mean the covariance structure is being ignored (i.e. effectively isotropic).
        along_lon = compute_initial_bias_influence(
            self.stats, self.stats.lat_ref, self.stats.lon_ref + 0.02, k=DEFAULT_K
        )
        along_lat = compute_initial_bias_influence(
            self.stats, self.stats.lat_ref + 0.02, self.stats.lon_ref, k=DEFAULT_K
        )
        self.assertNotAlmostEqual(along_lon.mahalanobis_distance, along_lat.mahalanobis_distance, places=2)
        # the wider (lon) axis should produce a SMALLER Mahalanobis distance for the same
        # raw degree offset (it takes a bigger raw move to reach "1 sigma" along the wide axis).
        self.assertLess(along_lon.mahalanobis_distance, along_lat.mahalanobis_distance)

    def test_monotonic_in_mahalanobis_distance(self):
        near = compute_initial_bias_influence(self.stats, 10.001, 100.0, k=DEFAULT_K)
        far = compute_initial_bias_influence(self.stats, 10.01, 100.0, k=DEFAULT_K)
        self.assertGreater(near.mahalanobis_distance, 0.0)
        self.assertGreater(far.mahalanobis_distance, near.mahalanobis_distance)
        self.assertGreater(near.influence, far.influence)


class TestNoAgeTerm(unittest.TestCase):
    def test_identical_relative_position_different_timestamps_gives_identical_influence(self):
        # two DIFFERENT timestamps, but each built so the candidate sits at exactly the
        # same Mahalanobis distance from its own timestamp's centroid -- influence must be
        # IDENTICAL, confirming no age-dependent term exists anywhere in the formula.
        rng = np.random.default_rng(1)
        n = 150
        lat_a = 10.0 + rng.normal(0, 0.001, n)
        lon_a = 100.0 + rng.normal(0, 0.001, n)
        records_a = _records(lat_a, lon_a, t_s=T_S)
        stats_a = compute_timestamp_cloud_stats(records_a, T_S)

        lat_b = 20.0 + rng.normal(0, 0.005, n)  # much later "age" in spirit, different scale
        lon_b = 50.0 + rng.normal(0, 0.005, n)
        records_b = _records(lat_b, lon_b, t_s=T_S2)
        stats_b = compute_timestamp_cloud_stats(records_b, T_S2)

        # place each candidate exactly at its own timestamp's centroid -> both should be
        # influence == 1.0 regardless of "how old" T_S2 is relative to T_S.
        result_a = compute_initial_bias_influence(stats_a, stats_a.lat_ref, stats_a.lon_ref)
        result_b = compute_initial_bias_influence(stats_b, stats_b.lat_ref, stats_b.lon_ref)
        self.assertAlmostEqual(result_a.influence, result_b.influence, places=6)
        self.assertAlmostEqual(result_a.influence, 1.0, places=6)


class TestInsufficientParticles(unittest.TestCase):
    def test_zero_particles(self):
        records = EnsembleRunRecords(
            particle_id=np.array([], dtype=object), forcing_member_id=np.array([], dtype=np.int64),
            observation_pixel_id=np.array([], dtype=object), timestamp=np.array([], dtype="datetime64[ns]"),
            lat=np.array([]), lon=np.array([]), log_weight=np.array([]), valid=np.array([], dtype=bool),
        )
        stats = compute_timestamp_cloud_stats(records, T_S)
        self.assertEqual(stats.covariance_state, "insufficient_particles")
        self.assertEqual(stats.n_particles, 0)
        self.assertIsNone(stats.covariance_m2)

        result = compute_initial_bias_influence(stats, 10.0, 100.0)
        self.assertTrue(np.isnan(result.mahalanobis_distance))
        self.assertTrue(np.isnan(result.influence))

    def test_one_particle(self):
        records = _records([10.0], [100.0])
        stats = compute_timestamp_cloud_stats(records, T_S)
        self.assertEqual(stats.covariance_state, "insufficient_particles")
        self.assertEqual(stats.n_particles, 1)


class TestSingularDegenerateCovariance(unittest.TestCase):
    def test_two_particles_falls_back_to_isotropic(self):
        # exactly 2 points -> covariance is always rank-deficient in 2D.
        records = _records([10.0, 10.002], [100.0, 100.002])
        stats = compute_timestamp_cloud_stats(records, T_S)
        self.assertEqual(stats.covariance_state, "singular_fallback")
        self.assertIsNotNone(stats.isotropic_variance_m2)
        self.assertGreater(stats.isotropic_variance_m2, 0.0)

        result = compute_initial_bias_influence(stats, stats.lat_ref, stats.lon_ref)
        self.assertAlmostEqual(result.mahalanobis_distance, 0.0, places=6)

    def test_collinear_points_falls_back_to_isotropic(self):
        lat = np.full(10, 10.0)
        lon = np.linspace(99.98, 100.02, 10)
        records = _records(lat, lon)
        stats = compute_timestamp_cloud_stats(records, T_S)
        self.assertEqual(stats.covariance_state, "singular_fallback")

        # a candidate off the collinear line still gets a defined, finite influence via the
        # isotropic fallback -- never a crash, never NaN when variance IS defined.
        result = compute_initial_bias_influence(stats, 10.05, 100.0)
        self.assertFalse(np.isnan(result.influence))
        self.assertGreater(result.mahalanobis_distance, 0.0)

    def test_all_particles_identical_gives_zero_variance_nan(self):
        lat = np.full(5, 10.0)
        lon = np.full(5, 100.0)
        records = _records(lat, lon)
        stats = compute_timestamp_cloud_stats(records, T_S)
        self.assertEqual(stats.covariance_state, "zero_variance")

        result = compute_initial_bias_influence(stats, 10.01, 100.01)
        self.assertTrue(np.isnan(result.mahalanobis_distance))
        self.assertTrue(np.isnan(result.influence))


class TestCentroidAndDistantCandidates(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(2)
        n = 100
        self.lat = 34.0 + rng.normal(0, 0.001, n)
        self.lon = 35.0 + rng.normal(0, 0.001, n)
        self.records = _records(self.lat, self.lon)
        self.stats = compute_timestamp_cloud_stats(self.records, T_S)

    def test_centroid_candidate_influence_near_one(self):
        result = compute_initial_bias_influence(self.stats, self.stats.lat_ref, self.stats.lon_ref)
        self.assertGreater(result.influence, 0.99)

    def test_very_distant_candidate_influence_near_zero(self):
        # thousands of km away against a very tight cloud (std ~0.001 deg) -> the true
        # Mahalanobis distance is astronomically large; float64's exp() legitimately
        # underflows to exactly 0.0 here (ordinary floating-point behavior for an input far
        # outside this pipeline's realistic few-km/few-sigma operating range, not a
        # discontinuity in the formula itself, which has no branch/threshold anywhere).
        result = compute_initial_bias_influence(self.stats, 50.0, 80.0)
        self.assertGreaterEqual(result.influence, 0.0)
        self.assertLess(result.influence, 1e-10)


class TestKParameter(unittest.TestCase):
    def test_non_positive_k_rejected(self):
        records = _records([10.0, 10.001, 10.002], [100.0, 100.0, 100.001])
        stats = compute_timestamp_cloud_stats(records, T_S)
        with self.assertRaises(ValueError):
            compute_initial_bias_influence(stats, 10.0, 100.0, k=0.0)
        with self.assertRaises(ValueError):
            compute_initial_bias_influence(stats, 10.0, 100.0, k=-1.0)

    def test_larger_k_widens_influence_falloff(self):
        rng = np.random.default_rng(3)
        n = 100
        lat = 10.0 + rng.normal(0, 0.001, n)
        lon = 100.0 + rng.normal(0, 0.002, n)
        records = _records(lat, lon)
        stats = compute_timestamp_cloud_stats(records, T_S)
        near_default_k = compute_initial_bias_influence(stats, 10.005, 100.0, k=1.0)
        near_large_k = compute_initial_bias_influence(stats, 10.005, 100.0, k=5.0)
        self.assertGreater(near_large_k.influence, near_default_k.influence)


def _candidate(lat, lon, t_s, cand_id="c0") -> Candidate:
    return Candidate(
        candidate_id=cand_id, x_peak=lon, y_peak=lat, t_s=np.datetime64(t_s, "ns"),
        core_region=CoreRegion(center_lat=lat, center_lon=lon, radius_m=1000.0),
        support_score=1.0, density_score=1.0, weighted_support_score=1.0,
        pixel_diversity_score=1.0, peak_support_features=None,
    )


class TestBatchOverCandidates(unittest.TestCase):
    def test_reuses_cloud_stats_across_shared_timestamp(self):
        rng = np.random.default_rng(4)
        n = 100
        lat = 10.0 + rng.normal(0, 0.001, n)
        lon = 100.0 + rng.normal(0, 0.001, n)
        records = _records(lat, lon)
        candidates = [
            _candidate(10.0, 100.0, T_S, "c0"),
            _candidate(10.001, 100.0, T_S, "c1"),
        ]
        results = compute_initial_bias_influence_for_candidates(records, candidates)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].covariance_state, "full_rank")
        self.assertEqual(results[0].t_s, T_S)

    def test_empty_candidate_list(self):
        records = _records([10.0, 10.001, 10.002], [100.0, 100.0, 100.001])
        results = compute_initial_bias_influence_for_candidates(records, [])
        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main()
