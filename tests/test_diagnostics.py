"""Focused T10 checks for `iste.diagnostics` (reusable Stage III diagnostics module).

Run with:
    python -m unittest tests.test_diagnostics -v
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from iste.diagnostics import compute_stage3_diagnostics, compute_stage3_diagnostics_from_log_weights


class TestAllValid(unittest.TestCase):
    def test_equal_weights_ess_equals_n(self):
        # four equal weights -> ESS = N (the textbook case: no concentration at all).
        weights = np.array([1.0, 1.0, 1.0, 1.0])
        valid = np.array([True, True, True, True])
        diag = compute_stage3_diagnostics(weights, valid)
        self.assertEqual(diag.total_particles, 4)
        self.assertEqual(diag.valid_particles, 4)
        self.assertEqual(diag.invalid_particles, 0)
        self.assertAlmostEqual(diag.total_weight, 4.0)
        self.assertAlmostEqual(diag.effective_sample_size, 4.0)
        self.assertAlmostEqual(diag.valid_particle_fraction, 1.0)

    def test_unequal_weights_expected_ess(self):
        # hand-computable: weights = [1, 3] -> ESS = (1+3)^2 / (1^2+3^2) = 16/10 = 1.6
        weights = np.array([1.0, 3.0])
        valid = np.array([True, True])
        diag = compute_stage3_diagnostics(weights, valid)
        self.assertAlmostEqual(diag.total_weight, 4.0)
        self.assertAlmostEqual(diag.effective_sample_size, 1.6)
        self.assertAlmostEqual(diag.valid_particle_fraction, 1.0)


class TestMixedValidity(unittest.TestCase):
    def test_mixed_valid_invalid(self):
        # weights = [2, 3, 5], valid = [True, False, True] -> only 2 and 5 contribute.
        # total_weight = 7, ESS = 7^2 / (2^2+5^2) = 49/29
        weights = np.array([2.0, 3.0, 5.0])
        valid = np.array([True, False, True])
        diag = compute_stage3_diagnostics(weights, valid)
        self.assertEqual(diag.total_particles, 3)
        self.assertEqual(diag.valid_particles, 2)
        self.assertEqual(diag.invalid_particles, 1)
        self.assertAlmostEqual(diag.total_weight, 7.0)
        self.assertAlmostEqual(diag.effective_sample_size, 49.0 / 29.0)
        self.assertAlmostEqual(diag.valid_particle_fraction, 2.0 / 3.0)

    def test_all_invalid(self):
        weights = np.array([1.0, 2.0, 3.0])
        valid = np.array([False, False, False])
        diag = compute_stage3_diagnostics(weights, valid)
        self.assertEqual(diag.total_particles, 3)
        self.assertEqual(diag.valid_particles, 0)
        self.assertEqual(diag.invalid_particles, 3)
        self.assertEqual(diag.total_weight, 0.0)
        self.assertEqual(diag.effective_sample_size, 0.0)
        self.assertAlmostEqual(diag.valid_particle_fraction, 0.0)


class TestZeroWeights(unittest.TestCase):
    def test_all_valid_zero_weight(self):
        # zero total weight must not raise a division-by-zero error.
        weights = np.array([0.0, 0.0, 0.0])
        valid = np.array([True, True, True])
        diag = compute_stage3_diagnostics(weights, valid)
        self.assertEqual(diag.total_weight, 0.0)
        self.assertEqual(diag.effective_sample_size, 0.0)
        self.assertAlmostEqual(diag.valid_particle_fraction, 1.0)

    def test_some_zero_some_positive(self):
        weights = np.array([0.0, 4.0])
        valid = np.array([True, True])
        diag = compute_stage3_diagnostics(weights, valid)
        self.assertAlmostEqual(diag.total_weight, 4.0)
        # ESS = 4^2 / (0^2 + 4^2) = 1.0
        self.assertAlmostEqual(diag.effective_sample_size, 1.0)


class TestEmptyInput(unittest.TestCase):
    def test_empty_arrays(self):
        weights = np.array([])
        valid = np.array([], dtype=bool)
        diag = compute_stage3_diagnostics(weights, valid)
        self.assertEqual(diag.total_particles, 0)
        self.assertEqual(diag.valid_particles, 0)
        self.assertEqual(diag.invalid_particles, 0)
        self.assertEqual(diag.total_weight, 0.0)
        self.assertEqual(diag.effective_sample_size, 0.0)
        self.assertTrue(math.isnan(diag.valid_particle_fraction))  # 0/0 is undefined, not 0


class TestNonFiniteWeights(unittest.TestCase):
    def test_non_finite_weight_on_invalid_particle_is_ignored(self):
        # a garbage/non-finite weight on an INVALID particle must not raise -- it never
        # contributes to weight-based diagnostics regardless of its value.
        weights = np.array([2.0, np.nan, np.inf])
        valid = np.array([True, False, False])
        diag = compute_stage3_diagnostics(weights, valid)
        self.assertEqual(diag.valid_particles, 1)
        self.assertEqual(diag.invalid_particles, 2)
        self.assertAlmostEqual(diag.total_weight, 2.0)
        self.assertAlmostEqual(diag.effective_sample_size, 1.0)

    def test_nan_weight_on_valid_particle_raises(self):
        weights = np.array([1.0, np.nan])
        valid = np.array([True, True])
        with self.assertRaises(ValueError):
            compute_stage3_diagnostics(weights, valid)

    def test_inf_weight_on_valid_particle_raises(self):
        weights = np.array([1.0, np.inf])
        valid = np.array([True, True])
        with self.assertRaises(ValueError):
            compute_stage3_diagnostics(weights, valid)

    def test_mismatched_shapes_raise(self):
        with self.assertRaises(ValueError):
            compute_stage3_diagnostics(np.array([1.0, 2.0]), np.array([True]))


class TestFromLogWeights(unittest.TestCase):
    def test_log_weight_conversion_matches_linear(self):
        log_weights = np.array([0.0, math.log(2.0), math.log(3.0)])
        valid = np.array([True, True, False])
        diag = compute_stage3_diagnostics_from_log_weights(log_weights, valid)
        # linear weights would be [1, 2, 3] with only the first two valid -> total = 3
        self.assertAlmostEqual(diag.total_weight, 3.0)
        self.assertAlmostEqual(diag.effective_sample_size, (1.0 + 2.0) ** 2 / (1.0 ** 2 + 2.0 ** 2))

    def test_non_finite_log_weight_on_invalid_particle_is_ignored(self):
        # log_weight = -inf -> exp(-inf) = 0.0, perfectly finite once converted; this also
        # exercises that an extreme (but validly converting) log-weight on an invalid
        # particle causes no issue.
        log_weights = np.array([0.0, -np.inf])
        valid = np.array([True, False])
        diag = compute_stage3_diagnostics_from_log_weights(log_weights, valid)
        self.assertAlmostEqual(diag.total_weight, 1.0)

    def test_huge_log_weight_on_valid_particle_raises(self):
        # exp(1e4) overflows to +inf in float64 -- must surface as an error, not silently
        # poison the diagnostic.
        log_weights = np.array([1.0, 1e4])
        valid = np.array([True, True])
        with self.assertRaises(ValueError):
            compute_stage3_diagnostics_from_log_weights(log_weights, valid)


if __name__ == "__main__":
    unittest.main()
