"""Focused T33 checks for `iste.phase3_dispersion` (Phase 3 dispersion-aware scoring).

Run with:
    python -m unittest tests.test_phase3_dispersion -v
"""

from __future__ import annotations

import unittest

import numpy as np

from iste.candidate_extraction import Candidate, CoreRegion
from iste.ensemble_run import EnsembleRunRecords
from iste.phase3_dispersion import (
    compute_dispersion_aware_score,
    compute_dispersion_aware_scores_for_candidates,
)
from iste.phase3_initial_bias import compute_timestamp_cloud_stats
from iste.support_features import SupportFeatureConfig

T_S = np.datetime64("2020-01-01T00:00:00", "ns")
T_S2 = T_S - np.timedelta64(1, "h")
R = 1000.0


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


def _candidate(lat, lon, t_s, density_score) -> Candidate:
    return Candidate(
        candidate_id="c0", x_peak=lon, y_peak=lat, t_s=np.datetime64(t_s, "ns"),
        core_region=CoreRegion(center_lat=lat, center_lon=lon, radius_m=R),
        support_score=density_score, density_score=density_score,
        weighted_support_score=density_score, pixel_diversity_score=1.0,
        peak_support_features=None,
    )


class TestHandComputableFullRank(unittest.TestCase):
    def test_matches_analytic_rayleigh_formula(self):
        rng = np.random.default_rng(0)
        n = 300
        lat = 10.0 + rng.normal(0, 0.003, n)
        lon = 100.0 + rng.normal(0, 0.003, n)
        records = _records(lat, lon)
        stats = compute_timestamp_cloud_stats(records, T_S)
        self.assertEqual(stats.covariance_state, "full_rank")

        candidate = _candidate(stats.lat_ref, stats.lon_ref, T_S, density_score=100.0)
        result = compute_dispersion_aware_score(candidate, stats, neighborhood_radius_m=R)

        sigma_iso_sq_expected = float(np.trace(stats.covariance_m2)) / 2.0
        p_expected = 1.0 - np.exp(-(R**2) / (2.0 * sigma_iso_sq_expected))
        d_expected = stats.n_particles * p_expected
        self.assertAlmostEqual(result.p_within_r, p_expected, places=9)
        self.assertAlmostEqual(result.expected_density, d_expected, places=6)
        self.assertAlmostEqual(result.dispersion_aware_score, 100.0 / d_expected, places=6)


class TestEqualRelativeConcentrationEqualScore(unittest.TestCase):
    """The core T33 requirement: two timestamps with very different raw density but equal
    RELATIVE concentration-given-dispersion score approximately equally."""

    def test_tight_and_dispersed_clouds_scaled_to_same_ratio_score_equal(self):
        rng = np.random.default_rng(1)
        n = 300

        tight_lat = 10.0 + rng.normal(0, 0.0005, n)
        tight_lon = 100.0 + rng.normal(0, 0.0005, n)
        records_tight = _records(tight_lat, tight_lon, t_s=T_S)
        stats_tight = compute_timestamp_cloud_stats(records_tight, T_S)

        dispersed_lat = 10.0 + rng.normal(0, 0.02, n)
        dispersed_lon = 100.0 + rng.normal(0, 0.02, n)
        records_dispersed = _records(dispersed_lat, dispersed_lon, t_s=T_S2)
        stats_dispersed = compute_timestamp_cloud_stats(records_dispersed, T_S2)

        # deliberately construct actual densities at a FIXED ratio (2.0x expected) for BOTH
        # timestamps, despite their very different absolute expected densities.
        cand_tight = _candidate(stats_tight.lat_ref, stats_tight.lon_ref, T_S, density_score=1.0)
        probe_tight = compute_dispersion_aware_score(cand_tight, stats_tight, neighborhood_radius_m=R)
        cand_tight_scaled = _candidate(
            stats_tight.lat_ref, stats_tight.lon_ref, T_S,
            density_score=2.0 * probe_tight.expected_density,
        )

        cand_dispersed = _candidate(stats_dispersed.lat_ref, stats_dispersed.lon_ref, T_S2, density_score=1.0)
        probe_dispersed = compute_dispersion_aware_score(cand_dispersed, stats_dispersed, neighborhood_radius_m=R)
        cand_dispersed_scaled = _candidate(
            stats_dispersed.lat_ref, stats_dispersed.lon_ref, T_S2,
            density_score=2.0 * probe_dispersed.expected_density,
        )

        result_tight = compute_dispersion_aware_score(cand_tight_scaled, stats_tight, neighborhood_radius_m=R)
        result_dispersed = compute_dispersion_aware_score(cand_dispersed_scaled, stats_dispersed, neighborhood_radius_m=R)

        # raw actual densities must be VERY different (confirms the synthetic setup)
        self.assertGreater(
            result_tight.particle_count_actual / result_dispersed.particle_count_actual, 5.0
        )
        # but dispersion-aware scores must be approximately equal (both == 2.0 by construction)
        self.assertAlmostEqual(result_tight.dispersion_aware_score, 2.0, places=6)
        self.assertAlmostEqual(result_dispersed.dispersion_aware_score, 2.0, places=6)


class TestLowerRawDensityCanScoreHigherWhenMoreDispersed(unittest.TestCase):
    def test_same_raw_density_scores_higher_at_more_dispersed_timestamp(self):
        rng = np.random.default_rng(2)
        n = 300
        tight_lat = 10.0 + rng.normal(0, 0.0008, n)
        tight_lon = 100.0 + rng.normal(0, 0.0008, n)
        records_tight = _records(tight_lat, tight_lon, t_s=T_S)
        stats_tight = compute_timestamp_cloud_stats(records_tight, T_S)

        dispersed_lat = 10.0 + rng.normal(0, 0.015, n)
        dispersed_lon = 100.0 + rng.normal(0, 0.015, n)
        records_dispersed = _records(dispersed_lat, dispersed_lon, t_s=T_S2)
        stats_dispersed = compute_timestamp_cloud_stats(records_dispersed, T_S2)

        same_raw_density = 20.0
        cand_tight = _candidate(stats_tight.lat_ref, stats_tight.lon_ref, T_S, density_score=same_raw_density)
        cand_dispersed = _candidate(
            stats_dispersed.lat_ref, stats_dispersed.lon_ref, T_S2, density_score=same_raw_density
        )
        result_tight = compute_dispersion_aware_score(cand_tight, stats_tight, neighborhood_radius_m=R)
        result_dispersed = compute_dispersion_aware_score(cand_dispersed, stats_dispersed, neighborhood_radius_m=R)

        # SAME raw density, but the dispersed timestamp's dispersion-aware score is HIGHER
        # -- exactly the required "do not read lower raw density as automatically weaker
        # evidence" behavior.
        self.assertGreater(result_dispersed.dispersion_aware_score, result_tight.dispersion_aware_score)


class TestUndefinedCases(unittest.TestCase):
    def test_insufficient_particles_gives_nan(self):
        records = _records([10.0], [100.0])
        stats = compute_timestamp_cloud_stats(records, T_S)
        self.assertEqual(stats.covariance_state, "insufficient_particles")
        cand = _candidate(10.0, 100.0, T_S, density_score=5.0)
        result = compute_dispersion_aware_score(cand, stats, neighborhood_radius_m=R)
        self.assertTrue(np.isnan(result.dispersion_aware_score))
        self.assertTrue(np.isnan(result.expected_density))

    def test_zero_variance_gives_nan(self):
        records = _records([10.0] * 5, [100.0] * 5)
        stats = compute_timestamp_cloud_stats(records, T_S)
        self.assertEqual(stats.covariance_state, "zero_variance")
        cand = _candidate(10.0, 100.0, T_S, density_score=5.0)
        result = compute_dispersion_aware_score(cand, stats, neighborhood_radius_m=R)
        self.assertTrue(np.isnan(result.dispersion_aware_score))

    def test_no_particles_gives_nan(self):
        records = EnsembleRunRecords(
            particle_id=np.array([], dtype=object), forcing_member_id=np.array([], dtype=np.int64),
            observation_pixel_id=np.array([], dtype=object), timestamp=np.array([], dtype="datetime64[ns]"),
            lat=np.array([]), lon=np.array([]), log_weight=np.array([]), valid=np.array([], dtype=bool),
        )
        stats = compute_timestamp_cloud_stats(records, T_S)
        cand = _candidate(10.0, 100.0, T_S, density_score=5.0)
        result = compute_dispersion_aware_score(cand, stats, neighborhood_radius_m=R)
        self.assertTrue(np.isnan(result.dispersion_aware_score))


class TestValidation(unittest.TestCase):
    def test_non_positive_radius_rejected(self):
        records = _records([10.0, 10.001, 10.002], [100.0, 100.001, 100.002])
        stats = compute_timestamp_cloud_stats(records, T_S)
        cand = _candidate(10.0, 100.0, T_S, density_score=5.0)
        with self.assertRaises(ValueError):
            compute_dispersion_aware_score(cand, stats, neighborhood_radius_m=0.0)
        with self.assertRaises(ValueError):
            compute_dispersion_aware_score(cand, stats, neighborhood_radius_m=-100.0)


class TestBatch(unittest.TestCase):
    def test_reuses_cloud_stats_across_shared_timestamp_and_preserves_order(self):
        rng = np.random.default_rng(3)
        n = 100
        lat = 10.0 + rng.normal(0, 0.001, n)
        lon = 100.0 + rng.normal(0, 0.001, n)
        records = _records(lat, lon)
        candidates = [
            _candidate(10.0, 100.0, T_S, density_score=5.0),
            _candidate(10.001, 100.001, T_S, density_score=10.0),
        ]
        results = compute_dispersion_aware_scores_for_candidates(records, candidates, neighborhood_radius_m=R)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].particle_count_actual, 5.0)
        self.assertEqual(results[1].particle_count_actual, 10.0)

    def test_empty_candidate_list(self):
        records = _records([10.0], [100.0])
        self.assertEqual(
            compute_dispersion_aware_scores_for_candidates(records, [], neighborhood_radius_m=R), []
        )


if __name__ == "__main__":
    unittest.main()
