"""Focused T37 checks for `iste.phase3_clustering` (Phase 3 spatial clustering of
candidates).

Run with:
    python -m unittest tests.test_phase3_clustering -v
"""

from __future__ import annotations

import unittest

import numpy as np

from iste.candidate_extraction import Candidate, CoreRegion
from iste.phase3_clustering import CONFIDENCE_DEFERRED_NOTE, cluster_candidates
from iste.phase3_composite import CompositeWeights, SourceScore

T_S = np.datetime64("2020-01-01T00:00:00", "ns")


def _candidate(cand_id, lat, lon, t_s=T_S, support_score=100.0) -> Candidate:
    return Candidate(
        candidate_id=cand_id, x_peak=lon, y_peak=lat, t_s=np.datetime64(t_s, "ns"),
        core_region=CoreRegion(center_lat=lat, center_lon=lon, radius_m=1000.0),
        support_score=support_score, density_score=support_score * 0.5,
        weighted_support_score=support_score * 0.3, pixel_diversity_score=support_score * 0.2,
        peak_support_features=None,
    )


def _score(cand_id, t_s=T_S, support_evidence=0.5, convergence=0.5, dispersion=0.5,
           initial_bias=0.3, ais=None, source_score=0.5) -> SourceScore:
    weights = CompositeWeights()
    return SourceScore(
        candidate_id=cand_id, t_s=np.datetime64(t_s, "ns"),
        support_evidence=support_evidence, persistence_evidence=1.0,
        convergence_evidence=convergence, dispersion_evidence=dispersion,
        ais_evidence=ais, initial_bias_evidence=initial_bias, persistence_count_evidence=1.0,
        configured_weights=weights.as_dict(), effective_weights=weights.as_dict(),
        source_score=source_score,
    )


class TestNearDuplicatesCollapse(unittest.TestCase):
    def test_two_close_candidates_form_one_cluster(self):
        candidates = [
            _candidate("a", 34.0000, 35.0000),
            _candidate("b", 34.0005, 35.0005),  # ~roughly 60-70m away
        ]
        scores = [_score("a"), _score("b")]
        clusters = cluster_candidates(candidates, scores, eps_m=2000.0)
        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0].n_members, 2)
        self.assertEqual(sorted(clusters[0].member_candidate_ids), ["a", "b"])

    def test_chain_of_near_duplicates_collapses_transitively(self):
        # each point ~500m from the next -- individually within eps, chained together
        candidates = [_candidate(f"c{i}", 34.0 + i * 0.0045, 35.0) for i in range(5)]
        scores = [_score(f"c{i}") for i in range(5)]
        clusters = cluster_candidates(candidates, scores, eps_m=600.0)
        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0].n_members, 5)


class TestWellSeparatedRemainDistinct(unittest.TestCase):
    def test_far_apart_candidates_stay_in_separate_clusters(self):
        candidates = [_candidate("a", 34.0, 35.0), _candidate("b", 40.0, 45.0)]
        scores = [_score("a"), _score("b")]
        clusters = cluster_candidates(candidates, scores, eps_m=2000.0)
        self.assertEqual(len(clusters), 2)
        self.assertEqual({cl.n_members for cl in clusters}, {1})


class TestNoCandidateDropped(unittest.TestCase):
    def test_isolated_candidate_forms_its_own_singleton_cluster(self):
        candidates = [
            _candidate("a", 34.0, 35.0), _candidate("b", 34.0001, 35.0001),
            _candidate("isolated", 50.0, 60.0),
        ]
        scores = [_score("a"), _score("b"), _score("isolated")]
        clusters = cluster_candidates(candidates, scores, eps_m=2000.0)
        total_members = sum(cl.n_members for cl in clusters)
        self.assertEqual(total_members, 3)
        isolated_cluster = next(cl for cl in clusters if "isolated" in cl.member_candidate_ids)
        self.assertEqual(isolated_cluster.n_members, 1)


class TestGeometry(unittest.TestCase):
    def test_centroid_is_mean_of_members(self):
        candidates = [_candidate("a", 34.0, 35.0), _candidate("b", 34.001, 35.001)]
        scores = [_score("a"), _score("b")]
        clusters = cluster_candidates(candidates, scores, eps_m=2000.0)
        self.assertAlmostEqual(clusters[0].centroid_lat, 34.0005, places=6)
        self.assertAlmostEqual(clusters[0].centroid_lon, 35.0005, places=6)

    def test_area_none_for_fewer_than_three_members(self):
        candidates = [_candidate("a", 34.0, 35.0), _candidate("b", 34.0001, 35.0001)]
        scores = [_score("a"), _score("b")]
        clusters = cluster_candidates(candidates, scores, eps_m=2000.0)
        self.assertIsNone(clusters[0].area_m2)

    def test_area_none_for_collinear_members(self):
        candidates = [_candidate(f"c{i}", 34.0, 35.0 + i * 0.0003) for i in range(4)]
        scores = [_score(f"c{i}") for i in range(4)]
        clusters = cluster_candidates(candidates, scores, eps_m=2000.0)
        self.assertIsNone(clusters[0].area_m2)

    def test_area_defined_for_nondegenerate_triangle(self):
        candidates = [
            _candidate("a", 34.0000, 35.0000), _candidate("b", 34.0010, 35.0000),
            _candidate("c", 34.0000, 35.0010),
        ]
        scores = [_score("a"), _score("b"), _score("c")]
        clusters = cluster_candidates(candidates, scores, eps_m=2000.0)
        self.assertIsNotNone(clusters[0].area_m2)
        self.assertGreater(clusters[0].area_m2, 0.0)


class TestEvidenceAggregation(unittest.TestCase):
    def test_total_particle_support_is_sum_of_density(self):
        candidates = [_candidate("a", 34.0, 35.0, support_score=10.0), _candidate("b", 34.0001, 35.0001, support_score=20.0)]
        scores = [_score("a"), _score("b")]
        clusters = cluster_candidates(candidates, scores, eps_m=2000.0)
        self.assertAlmostEqual(clusters[0].total_particle_support, 5.0 + 10.0)  # density_score = 0.5*support_score

    def test_normalized_kde_mean_and_peak(self):
        candidates = [_candidate("a", 34.0, 35.0), _candidate("b", 34.0001, 35.0001)]
        scores = [_score("a", support_evidence=0.4), _score("b", support_evidence=0.8)]
        clusters = cluster_candidates(candidates, scores, eps_m=2000.0)
        self.assertAlmostEqual(clusters[0].mean_normalized_kde, 0.6)
        self.assertAlmostEqual(clusters[0].peak_normalized_kde, 0.8)

    def test_peak_and_mean_source_score(self):
        candidates = [_candidate("a", 34.0, 35.0), _candidate("b", 34.0001, 35.0001)]
        scores = [_score("a", source_score=0.3), _score("b", source_score=0.9)]
        clusters = cluster_candidates(candidates, scores, eps_m=2000.0)
        self.assertAlmostEqual(clusters[0].peak_source_score, 0.9)
        self.assertAlmostEqual(clusters[0].mean_source_score, 0.6)

    def test_ais_none_when_all_members_unavailable(self):
        candidates = [_candidate("a", 34.0, 35.0), _candidate("b", 34.0001, 35.0001)]
        scores = [_score("a", ais=None), _score("b", ais=None)]
        clusters = cluster_candidates(candidates, scores, eps_m=2000.0)
        self.assertIsNone(clusters[0].mean_ais_score)

    def test_ais_mean_when_some_members_available(self):
        candidates = [_candidate("a", 34.0, 35.0), _candidate("b", 34.0001, 35.0001)]
        scores = [_score("a", ais=None), _score("b", ais=0.6)]
        clusters = cluster_candidates(candidates, scores, eps_m=2000.0)
        self.assertAlmostEqual(clusters[0].mean_ais_score, 0.6)  # mean over AVAILABLE members only

    def test_temporal_span_and_supported_timestamps(self):
        candidates = [
            _candidate("a", 34.0, 35.0, t_s=T_S), _candidate("b", 34.0001, 35.0001, t_s=T_S - np.timedelta64(5, "h")),
        ]
        scores = [_score("a", t_s=T_S), _score("b", t_s=T_S - np.timedelta64(5, "h"))]
        clusters = cluster_candidates(candidates, scores, eps_m=2000.0)
        self.assertAlmostEqual(clusters[0].temporal_span_hours, 5.0)
        self.assertEqual(clusters[0].n_distinct_timestamps, 2)


class TestConfidencePlaceholder(unittest.TestCase):
    def test_note_states_deferred_not_fabricated(self):
        candidates = [_candidate("a", 34.0, 35.0)]
        scores = [_score("a")]
        clusters = cluster_candidates(candidates, scores, eps_m=2000.0)
        self.assertIn("T38", clusters[0].confidence_uncertainty_note)
        self.assertEqual(clusters[0].confidence_uncertainty_note, CONFIDENCE_DEFERRED_NOTE)


class TestRankedOutput(unittest.TestCase):
    def test_clusters_sorted_by_peak_source_score_descending(self):
        candidates = [
            _candidate("a", 34.0, 35.0), _candidate("b", 40.0, 45.0), _candidate("c", 10.0, 20.0),
        ]
        scores = [_score("a", source_score=0.3), _score("b", source_score=0.9), _score("c", source_score=0.6)]
        clusters = cluster_candidates(candidates, scores, eps_m=2000.0)
        peak_scores = [cl.peak_source_score for cl in clusters]
        self.assertEqual(peak_scores, sorted(peak_scores, reverse=True))


class TestValidation(unittest.TestCase):
    def test_mismatched_lengths_raises(self):
        candidates = [_candidate("a", 34.0, 35.0)]
        scores = [_score("a"), _score("b")]
        with self.assertRaises(ValueError):
            cluster_candidates(candidates, scores)

    def test_mismatched_candidate_id_order_raises(self):
        candidates = [_candidate("a", 34.0, 35.0)]
        scores = [_score("WRONG")]
        with self.assertRaises(ValueError):
            cluster_candidates(candidates, scores)

    def test_non_positive_eps_rejected(self):
        candidates = [_candidate("a", 34.0, 35.0)]
        scores = [_score("a")]
        with self.assertRaises(ValueError):
            cluster_candidates(candidates, scores, eps_m=0.0)

    def test_empty_input(self):
        self.assertEqual(cluster_candidates([], []), [])

    def test_single_candidate(self):
        candidates = [_candidate("a", 34.0, 35.0)]
        scores = [_score("a")]
        clusters = cluster_candidates(candidates, scores, eps_m=2000.0)
        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0].n_members, 1)


if __name__ == "__main__":
    unittest.main()
