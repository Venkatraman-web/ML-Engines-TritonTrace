"""Focused T34 checks for `iste.phase3_screening` (Phase 3 screening-signal normalization).

Run with:
    python -m unittest tests.test_phase3_screening -v
"""

from __future__ import annotations

import inspect
import unittest

import numpy as np

from iste.candidate_extraction import Candidate, CoreRegion
from iste.phase3_normalization import normalize_support_grid
from iste.phase3_screening import (
    compute_screening_score,
    compute_screening_scores_for_candidates,
)

T_S = np.datetime64("2020-01-01T00:00:00", "ns")
T_S2 = T_S - np.timedelta64(1, "h")


def _candidate(support_score, t_s=T_S, cand_id="c0") -> Candidate:
    return Candidate(
        candidate_id=cand_id, x_peak=35.0, y_peak=34.0, t_s=np.datetime64(t_s, "ns"),
        core_region=CoreRegion(center_lat=34.0, center_lon=35.0, radius_m=1000.0),
        support_score=support_score, density_score=support_score * 0.5,
        weighted_support_score=support_score * 0.3, pixel_diversity_score=support_score * 0.2,
        peak_support_features=None,
    )


class TestFormulaMatchesT29Exactly(unittest.TestCase):
    def test_hand_computable_match(self):
        grid = np.array([[0.0, 10.0], [20.0, 30.0]])
        norm_result = normalize_support_grid(grid, T_S, percentile=50.0)
        candidate = _candidate(support_score=30.0, t_s=T_S)  # matches grid's own max cell
        result = compute_screening_score(candidate, norm_result)
        self.assertAlmostEqual(result.screening_score, 30.0 / norm_result.reference_value, places=9)

    def test_bit_for_bit_consistency_with_t29_normalized_grid(self):
        rng = np.random.default_rng(0)
        grid = rng.uniform(0, 100, size=(10, 10))
        norm_result = normalize_support_grid(grid, T_S, percentile=95.0)
        peak_value = float(np.max(grid))
        candidate = _candidate(support_score=peak_value, t_s=T_S)
        result = compute_screening_score(candidate, norm_result)
        expected_from_t29_grid = peak_value / norm_result.reference_value
        self.assertAlmostEqual(result.screening_score, expected_from_t29_grid, places=9)
        # also matches T29's own normalized_grid at its max cell exactly
        self.assertAlmostEqual(result.screening_score, float(np.max(norm_result.normalized_grid)), places=9)


class TestMonotonicity(unittest.TestCase):
    def test_relative_order_preserved_within_timestamp(self):
        grid = np.array([[5.0, 15.0], [25.0, 35.0]])
        norm_result = normalize_support_grid(grid, T_S)
        low = _candidate(support_score=10.0, t_s=T_S, cand_id="low")
        high = _candidate(support_score=20.0, t_s=T_S, cand_id="high")
        r_low = compute_screening_score(low, norm_result)
        r_high = compute_screening_score(high, norm_result)
        self.assertLess(r_low.screening_score, r_high.screening_score)

    def test_raw_support_score_preserved(self):
        grid = np.array([[5.0, 15.0], [25.0, 35.0]])
        norm_result = normalize_support_grid(grid, T_S)
        cand = _candidate(support_score=17.5, t_s=T_S)
        result = compute_screening_score(cand, norm_result)
        self.assertEqual(result.raw_support_score, 17.5)


class TestSharedReferenceValue(unittest.TestCase):
    def test_two_candidates_same_timestamp_share_reference_value(self):
        grid = np.array([[5.0, 15.0], [25.0, 35.0]])
        norm_result = normalize_support_grid(grid, T_S)
        c1 = _candidate(support_score=10.0, t_s=T_S, cand_id="c1")
        c2 = _candidate(support_score=30.0, t_s=T_S, cand_id="c2")
        r1 = compute_screening_score(c1, norm_result)
        r2 = compute_screening_score(c2, norm_result)
        self.assertEqual(r1.reference_value, r2.reference_value)


class TestNoAgeDependence(unittest.TestCase):
    def test_signature_has_no_age_or_t_obs_parameter(self):
        sig = inspect.signature(compute_screening_score)
        self.assertNotIn("age", sig.parameters)
        self.assertNotIn("age_hours", sig.parameters)
        self.assertNotIn("t_obs", sig.parameters)

    def test_identical_inputs_at_very_different_timestamps_give_identical_score(self):
        grid_a = np.array([[5.0, 15.0], [25.0, 35.0]])
        grid_b = np.array([[5.0, 15.0], [25.0, 35.0]])  # identical grid shape/values
        far_future_t_s = T_S + np.timedelta64(1000, "h")  # deliberately very different "age"
        norm_a = normalize_support_grid(grid_a, T_S)
        norm_b = normalize_support_grid(grid_b, far_future_t_s)
        cand_a = _candidate(support_score=30.0, t_s=T_S)
        cand_b = _candidate(support_score=30.0, t_s=far_future_t_s)
        result_a = compute_screening_score(cand_a, norm_a)
        result_b = compute_screening_score(cand_b, norm_b)
        self.assertAlmostEqual(result_a.screening_score, result_b.screening_score, places=9)


class TestNaNPropagation(unittest.TestCase):
    def test_nan_reference_value_gives_nan_score(self):
        grid = np.zeros((3, 3))  # T29's own degenerate all-zero case -> NaN reference_value
        norm_result = normalize_support_grid(grid, T_S)
        self.assertTrue(np.isnan(norm_result.reference_value))
        cand = _candidate(support_score=5.0, t_s=T_S)
        result = compute_screening_score(cand, norm_result)
        self.assertTrue(np.isnan(result.screening_score))

    def test_mismatched_timestamp_raises(self):
        grid = np.array([[5.0, 15.0], [25.0, 35.0]])
        norm_result = normalize_support_grid(grid, T_S)
        cand = _candidate(support_score=10.0, t_s=T_S2)
        with self.assertRaises(ValueError):
            compute_screening_score(cand, norm_result)


class TestBatch(unittest.TestCase):
    def test_reuses_normalization_across_shared_timestamp_and_preserves_order(self):
        grid_t1 = np.array([[5.0, 15.0], [25.0, 35.0]])
        grid_t2 = np.array([[1.0, 2.0], [3.0, 4.0]])
        support_grids_by_t_s = {T_S: grid_t1, T_S2: grid_t2}
        candidates = [
            _candidate(support_score=10.0, t_s=T_S, cand_id="a"),
            _candidate(support_score=30.0, t_s=T_S, cand_id="b"),
            _candidate(support_score=3.0, t_s=T_S2, cand_id="c"),
        ]
        results = compute_screening_scores_for_candidates(candidates, support_grids_by_t_s)
        self.assertEqual(len(results), 3)
        self.assertEqual([r.candidate_id for r in results], ["a", "b", "c"])
        # a and b share the same t_s -> same reference_value
        self.assertEqual(results[0].reference_value, results[1].reference_value)
        # c is a different timestamp -> generally a different reference_value
        self.assertNotEqual(results[0].reference_value, results[2].reference_value)

    def test_missing_grid_for_timestamp_raises(self):
        candidates = [_candidate(support_score=10.0, t_s=T_S)]
        with self.assertRaises(KeyError):
            compute_screening_scores_for_candidates(candidates, {})

    def test_empty_candidate_list(self):
        self.assertEqual(compute_screening_scores_for_candidates([], {}), [])


if __name__ == "__main__":
    unittest.main()
