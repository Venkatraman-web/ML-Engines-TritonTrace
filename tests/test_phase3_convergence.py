"""Focused T32 checks for `iste.phase3_convergence` (Phase 3 spatial convergence scoring).

Run with:
    python -m unittest tests.test_phase3_convergence -v
"""

from __future__ import annotations

import unittest

import numpy as np

from iste.candidate_extraction import Candidate, CoreRegion
from iste.ensemble_run import EnsembleRunRecords
from iste.kdtree_neighborhood import NeighborhoodResult
from iste.phase3_convergence import (
    ConvergenceWeights,
    compute_convergence_score,
    compute_convergence_scores_for_candidates,
)
from iste.phase3_persistence import PersistenceTrack
from iste.support_features import SupportFeatureConfig, SupportFeatures

T_S = np.datetime64("2020-01-01T00:00:00", "ns")
NAT = np.datetime64("NaT", "ns")


def _neighborhood(pixel_ids, member_ids, distances_m) -> NeighborhoodResult:
    n = len(pixel_ids)
    return NeighborhoodResult(
        particle_id=np.array([f"p{i}" for i in range(n)], dtype=object),
        observation_pixel_id=np.array(pixel_ids, dtype=object),
        forcing_member_id=np.array(member_ids, dtype=np.int64),
        lat=np.zeros(n), lon=np.zeros(n), log_weight=np.zeros(n),
        valid=np.ones(n, dtype=bool),
        death_time=np.full(n, NAT, dtype="datetime64[ns]"),
        death_lat=np.full(n, np.nan), death_lon=np.full(n, np.nan),
        death_reason=np.array([""] * n, dtype=object),
        distance_m=np.array(distances_m, dtype=np.float64),
    )


def _candidate(pixel_ids, member_ids, distances_m, pixel_diversity, cand_id="c0") -> Candidate:
    neighborhood = _neighborhood(pixel_ids, member_ids, distances_m)
    sf = SupportFeatures(
        query_lat=34.0, query_lon=35.0, density=neighborhood.n_particles,
        weighted_support=float(neighborhood.n_particles), pixel_diversity=pixel_diversity,
        support=float(neighborhood.n_particles), neighborhood=neighborhood,
        config=SupportFeatureConfig(),
    )
    return Candidate(
        candidate_id=cand_id, x_peak=35.0, y_peak=34.0, t_s=T_S,
        core_region=CoreRegion(center_lat=34.0, center_lon=35.0, radius_m=1000.0),
        support_score=sf.support, density_score=sf.density,
        weighted_support_score=sf.weighted_support, pixel_diversity_score=sf.pixel_diversity,
        peak_support_features=sf,
    )


class TestDiverseVsHomogeneousAtEqualDensity(unittest.TestCase):
    """The core T32 requirement: equal raw density/KDE, different diversity -> different
    convergence_score."""

    def test_multiple_independent_groups_score_higher_than_one_homogeneous_group(self):
        n = 20
        # homogeneous: all 20 particles from the SAME single pixel and SAME single member
        homogeneous = _candidate(
            pixel_ids=["px-0"] * n, member_ids=[0] * n, distances_m=[100.0] * n,
            pixel_diversity=1, cand_id="homogeneous",
        )
        # diverse: same particle COUNT (density identical), but many distinct pixels/members
        diverse = _candidate(
            pixel_ids=[f"px-{i}" for i in range(n)], member_ids=[i % 6 for i in range(n)],
            distances_m=[100.0] * n, pixel_diversity=n, cand_id="diverse",
        )

        score_homogeneous = compute_convergence_score(
            homogeneous, total_valid_particles_at_t=100, total_distinct_pixels_at_t=20,
            total_distinct_members_at_t=6,
        )
        score_diverse = compute_convergence_score(
            diverse, total_valid_particles_at_t=100, total_distinct_pixels_at_t=20,
            total_distinct_members_at_t=6,
        )

        # raw density (KDE-equivalent) is IDENTICAL by construction
        self.assertEqual(score_homogeneous.particle_count, score_diverse.particle_count)
        self.assertEqual(score_homogeneous.particle_fraction, score_diverse.particle_fraction)
        self.assertEqual(score_homogeneous.concentration_score, score_diverse.concentration_score)
        # but convergence_score must differ, with diverse scoring higher
        self.assertGreater(score_diverse.convergence_score, score_homogeneous.convergence_score)
        self.assertGreater(score_diverse.pixel_diversity_fraction, score_homogeneous.pixel_diversity_fraction)
        self.assertGreater(score_diverse.member_diversity_fraction, score_homogeneous.member_diversity_fraction)


class TestConcentration(unittest.TestCase):
    def test_tighter_neighborhood_scores_higher_concentration(self):
        tight = _candidate(
            pixel_ids=["px-0"] * 10, member_ids=[0] * 10, distances_m=[10.0] * 10, pixel_diversity=1,
        )
        loose = _candidate(
            pixel_ids=["px-0"] * 10, member_ids=[0] * 10, distances_m=[900.0] * 10, pixel_diversity=1,
        )
        score_tight = compute_convergence_score(
            tight, total_valid_particles_at_t=100, total_distinct_pixels_at_t=5, total_distinct_members_at_t=6,
        )
        score_loose = compute_convergence_score(
            loose, total_valid_particles_at_t=100, total_distinct_pixels_at_t=5, total_distinct_members_at_t=6,
        )
        self.assertGreater(score_tight.concentration_score, score_loose.concentration_score)

    def test_concentration_uses_core_radius_not_hidden_constant(self):
        cand = _candidate(pixel_ids=["px-0"] * 5, member_ids=[0] * 5, distances_m=[500.0] * 5, pixel_diversity=1)
        cand.core_region.radius_m  # sanity: attribute exists
        score = compute_convergence_score(
            cand, total_valid_particles_at_t=50, total_distinct_pixels_at_t=5, total_distinct_members_at_t=6,
        )
        expected = float(np.exp(-500.0 / cand.core_region.radius_m))
        self.assertAlmostEqual(score.concentration_score, expected, places=9)


class TestTemporalConsistencyOptional(unittest.TestCase):
    def test_none_when_not_supplied(self):
        cand = _candidate(pixel_ids=["px-0"] * 5, member_ids=[0] * 5, distances_m=[100.0] * 5, pixel_diversity=1)
        score = compute_convergence_score(
            cand, total_valid_particles_at_t=50, total_distinct_pixels_at_t=5, total_distinct_members_at_t=6,
        )
        self.assertIsNone(score.temporal_consistency)
        self.assertFalse(np.isnan(score.convergence_score))  # renormalized over 4 components, still defined

    def test_included_when_supplied_changes_score(self):
        cand = _candidate(pixel_ids=["px-0"] * 5, member_ids=[0] * 5, distances_m=[100.0] * 5, pixel_diversity=1)
        score_without = compute_convergence_score(
            cand, total_valid_particles_at_t=50, total_distinct_pixels_at_t=5, total_distinct_members_at_t=6,
        )
        score_with_low = compute_convergence_score(
            cand, total_valid_particles_at_t=50, total_distinct_pixels_at_t=5, total_distinct_members_at_t=6,
            temporal_consistency=0.0,
        )
        score_with_high = compute_convergence_score(
            cand, total_valid_particles_at_t=50, total_distinct_pixels_at_t=5, total_distinct_members_at_t=6,
            temporal_consistency=1.0,
        )
        self.assertNotEqual(score_without.convergence_score, score_with_low.convergence_score)
        self.assertGreater(score_with_high.convergence_score, score_with_low.convergence_score)


class TestWeights(unittest.TestCase):
    def test_negative_weight_rejected(self):
        with self.assertRaises(ValueError):
            ConvergenceWeights(weight_particle_fraction=-1.0)

    def test_zero_weight_excludes_component_effectively(self):
        cand_a = _candidate(pixel_ids=["px-0"] * 5, member_ids=[0] * 5, distances_m=[100.0] * 5, pixel_diversity=1)
        cand_b = _candidate(pixel_ids=["px-1"] * 5, member_ids=[1] * 5, distances_m=[999.0] * 5, pixel_diversity=1)
        # only weight concentration -> particle_fraction/pixel_diversity/member_diversity
        # differences must not affect the score at all.
        weights = ConvergenceWeights(
            weight_particle_fraction=0.0, weight_pixel_diversity=0.0,
            weight_member_diversity=0.0, weight_concentration=1.0, weight_temporal_consistency=0.0,
        )
        score_a = compute_convergence_score(
            cand_a, total_valid_particles_at_t=50, total_distinct_pixels_at_t=5,
            total_distinct_members_at_t=6, weights=weights,
        )
        score_b = compute_convergence_score(
            cand_b, total_valid_particles_at_t=999, total_distinct_pixels_at_t=999,
            total_distinct_members_at_t=999, weights=weights,
        )
        self.assertAlmostEqual(score_a.convergence_score, score_a.concentration_score, places=9)
        self.assertNotAlmostEqual(score_a.convergence_score, score_b.convergence_score, places=2)


class TestValidation(unittest.TestCase):
    def test_missing_support_features_rejected(self):
        cand = Candidate(
            candidate_id="c0", x_peak=35.0, y_peak=34.0, t_s=T_S,
            core_region=CoreRegion(center_lat=34.0, center_lon=35.0, radius_m=1000.0),
            support_score=1.0, density_score=1.0, weighted_support_score=1.0,
            pixel_diversity_score=1.0, peak_support_features=None,
        )
        with self.assertRaises(ValueError):
            compute_convergence_score(
                cand, total_valid_particles_at_t=10, total_distinct_pixels_at_t=1, total_distinct_members_at_t=1,
            )

    def test_non_positive_totals_rejected(self):
        cand = _candidate(pixel_ids=["px-0"], member_ids=[0], distances_m=[10.0], pixel_diversity=1)
        with self.assertRaises(ValueError):
            compute_convergence_score(
                cand, total_valid_particles_at_t=0, total_distinct_pixels_at_t=1, total_distinct_members_at_t=1,
            )


def _records(lat, lon, t_s=T_S, n_pixels=3, n_members=6) -> EnsembleRunRecords:
    n = len(lat)
    return EnsembleRunRecords(
        particle_id=np.array([f"p{i}" for i in range(n)], dtype=object),
        forcing_member_id=np.array([i % n_members for i in range(n)], dtype=np.int64),
        observation_pixel_id=np.array([f"px-{i % n_pixels}" for i in range(n)], dtype=object),
        timestamp=np.full(n, t_s, dtype="datetime64[ns]"),
        lat=np.asarray(lat, dtype=np.float64), lon=np.asarray(lon, dtype=np.float64),
        log_weight=np.zeros(n, dtype=np.float64), valid=np.ones(n, dtype=bool),
    )


class TestBatchWithRealRecordsAndPersistence(unittest.TestCase):
    def test_batch_wires_up_totals_and_persistence(self):
        rng = np.random.default_rng(0)
        n = 60
        lat = 10.0 + rng.normal(0, 0.001, n)
        lon = 100.0 + rng.normal(0, 0.001, n)
        records = _records(lat, lon)

        cand = _candidate(pixel_ids=["px-0"] * 5, member_ids=[0] * 5, distances_m=[100.0] * 5, pixel_diversity=1)
        track = PersistenceTrack(track_id="track0000")
        track.candidate_ids.append(cand.candidate_id)
        track.t_s_list.append(T_S)
        track.support_scores.append(cand.support_score)
        track.lat_list.append(cand.y_peak)
        track.lon_list.append(cand.x_peak)

        results = compute_convergence_scores_for_candidates(
            records, [cand], persistence_tracks=[track],
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].total_valid_particles_at_t, 60)
        self.assertEqual(results[0].total_distinct_pixels_at_t, 3)
        self.assertEqual(results[0].total_distinct_members_at_t, 6)
        self.assertAlmostEqual(results[0].temporal_consistency, track.temporal_continuity())

    def test_batch_without_persistence_gives_none_consistency(self):
        records = _records([10.0, 10.001, 10.002], [100.0, 100.001, 100.002])
        cand = _candidate(pixel_ids=["px-0"] * 3, member_ids=[0] * 3, distances_m=[10.0] * 3, pixel_diversity=1)
        results = compute_convergence_scores_for_candidates(records, [cand])
        self.assertIsNone(results[0].temporal_consistency)

    def test_empty_candidate_list(self):
        records = _records([10.0], [100.0])
        self.assertEqual(compute_convergence_scores_for_candidates(records, []), [])


if __name__ == "__main__":
    unittest.main()
