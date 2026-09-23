"""Focused T15 checks for `iste.detection_model` (Stage IV observation-model bridge,
`h_j,e -> pi_j,e`).

Run with:
    python -m unittest tests.test_detection_model -v
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from iste.detection_model import ObservationModelConfig, compute_detection_probability
from iste.likelihood import compute_stage4_log_likelihood

CONFIG = ObservationModelConfig(midpoint=5.0, steepness=1.0)


class TestValidInputGivesProbabilityInRange(unittest.TestCase):
    def test_pi_in_unit_interval(self):
        h = np.array([-100.0, -1.0, 0.0, 5.0, 10.0, 100.0])
        pi = compute_detection_probability(h, CONFIG)
        self.assertTrue(np.all(pi >= 0.0))
        self.assertTrue(np.all(pi <= 1.0))
        self.assertTrue(np.all(np.isfinite(pi)))

    def test_2d_shape_preserved(self):
        h = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        pi = compute_detection_probability(h, CONFIG)
        self.assertEqual(pi.shape, h.shape)


class TestHandComputableExample(unittest.TestCase):
    def test_midpoint_gives_exactly_half(self):
        h = np.array([5.0])
        pi = compute_detection_probability(h, CONFIG)
        self.assertAlmostEqual(pi[0], 0.5)

    def test_offset_by_ln3_over_steepness_gives_075_and_025(self):
        # sigmoid(ln(3)) = 3/4, sigmoid(-ln(3)) = 1/4 -- exact, hand-computable.
        h = np.array([5.0 + math.log(3.0), 5.0 - math.log(3.0)])
        pi = compute_detection_probability(h, CONFIG)
        self.assertAlmostEqual(pi[0], 0.75, places=9)
        self.assertAlmostEqual(pi[1], 0.25, places=9)

    def test_asymmetric_steepness_hand_computable(self):
        config = ObservationModelConfig(midpoint=0.0, steepness=2.0)
        h = np.array([0.5])  # z = 2*0.5 = 1.0
        pi = compute_detection_probability(h, config)
        expected = 1.0 / (1.0 + math.exp(-1.0))
        self.assertAlmostEqual(pi[0], expected, places=12)


class TestMonotonicity(unittest.TestCase):
    def test_pi_strictly_increasing_in_h(self):
        h = np.linspace(-20.0, 20.0, 41)
        pi = compute_detection_probability(h, CONFIG)
        self.assertTrue(np.all(np.diff(pi) > 0.0))

    def test_higher_steepness_is_sharper_around_midpoint(self):
        h = np.array([CONFIG.midpoint + 0.1])
        gentle = compute_detection_probability(h, ObservationModelConfig(midpoint=CONFIG.midpoint, steepness=0.5))
        steep = compute_detection_probability(h, ObservationModelConfig(midpoint=CONFIG.midpoint, steepness=5.0))
        # both above 0.5 (h > midpoint), but the steeper curve pushes further from 0.5
        self.assertGreater(steep[0], gentle[0])
        self.assertGreater(gentle[0], 0.5)


class TestEdgeCases(unittest.TestCase):
    def test_extreme_h_saturates_without_overflow(self):
        h = np.array([-1e6, 1e6])
        pi = compute_detection_probability(h, CONFIG)
        self.assertTrue(np.all(np.isfinite(pi)))
        self.assertAlmostEqual(pi[0], 0.0, places=9)
        self.assertAlmostEqual(pi[1], 1.0, places=9)

    def test_never_exactly_zero_or_one_for_moderately_extreme_h(self):
        # mathematically pi(h) is in the OPEN interval (0,1) for every finite h; in
        # float64 this stays distinguishable from the boundary as long as |z| stays
        # below ~37 (beyond that, exp(-|z|) underflows below float64 epsilon relative
        # to 1.0 and pi rounds to exactly 0.0/1.0 -- see test_extreme_h_saturates_...
        # below for that expected, non-error floating-point behavior).
        h = np.array([-25.0, 35.0])  # z = -30, +30 relative to midpoint=5
        pi = compute_detection_probability(h, CONFIG)
        self.assertTrue(np.all(pi > 0.0))
        self.assertTrue(np.all(pi < 1.0))

    def test_single_scalar_shaped_array(self):
        pi = compute_detection_probability(np.array([5.0]), CONFIG)
        self.assertEqual(pi.shape, (1,))

    def test_empty_array(self):
        pi = compute_detection_probability(np.array([]), CONFIG)
        self.assertEqual(pi.shape, (0,))


class TestInvalidParameterRejection(unittest.TestCase):
    def test_non_positive_steepness_rejected(self):
        with self.assertRaises(ValueError):
            ObservationModelConfig(midpoint=0.0, steepness=0.0)
        with self.assertRaises(ValueError):
            ObservationModelConfig(midpoint=0.0, steepness=-1.0)

    def test_non_finite_config_rejected(self):
        with self.assertRaises(ValueError):
            ObservationModelConfig(midpoint=float("nan"), steepness=1.0)
        with self.assertRaises(ValueError):
            ObservationModelConfig(midpoint=0.0, steepness=float("inf"))
        with self.assertRaises(ValueError):
            ObservationModelConfig(midpoint=float("inf"), steepness=1.0)

    def test_non_finite_h_rejected(self):
        with self.assertRaises(ValueError):
            compute_detection_probability(np.array([1.0, np.nan]), CONFIG)
        with self.assertRaises(ValueError):
            compute_detection_probability(np.array([1.0, np.inf]), CONFIG)


class TestDeterministicOutput(unittest.TestCase):
    def test_same_input_gives_identical_output(self):
        h = np.array([-3.0, 0.0, 2.5, 9.0])
        pi_a = compute_detection_probability(h, CONFIG)
        pi_b = compute_detection_probability(h, CONFIG)
        np.testing.assert_array_equal(pi_a, pi_b)

    def test_no_hidden_state_across_calls(self):
        h = np.array([1.0])
        first = compute_detection_probability(h, CONFIG)
        for _ in range(5):
            compute_detection_probability(np.array([-999.0]), CONFIG)
        again = compute_detection_probability(h, CONFIG)
        np.testing.assert_array_equal(first, again)


class TestPixelMemberAssociationPreserved(unittest.TestCase):
    def test_row_column_order_preserved(self):
        # h[j, e]: pixel j, member e -- pi must preserve exactly which (j, e) each value
        # belongs to (no reordering/transposition/aggregation).
        h = np.array([
            [0.0, 10.0],   # pixel 0: member 0 far below midpoint, member 1 far above
            [10.0, 0.0],   # pixel 1: reversed
        ])
        pi = compute_detection_probability(h, CONFIG)
        self.assertLess(pi[0, 0], 0.5)
        self.assertGreater(pi[0, 1], 0.5)
        self.assertGreater(pi[1, 0], 0.5)
        self.assertLess(pi[1, 1], 0.5)


class TestCompatibleWithT14(unittest.TestCase):
    """Requirement 13: compatibility with T14's expected input shape -- proven by an
    actual (read-only) call into `iste.likelihood.compute_stage4_log_likelihood`, not
    merely a shape assertion."""

    def test_output_feeds_directly_into_t14(self):
        n_pixels, n_members = 4, 3
        rng = np.random.default_rng(0)
        h = rng.normal(loc=CONFIG.midpoint, scale=3.0, size=(n_pixels, n_members))
        pi = compute_detection_probability(h, CONFIG)

        y = rng.integers(0, 2, size=n_pixels).astype(float)
        result = compute_stage4_log_likelihood(y, pi, epsilon=0.05, beta=1.0)

        self.assertEqual(result.log_pixel_likelihood.shape, (n_pixels, n_members))
        self.assertTrue(np.isfinite(result.log_surrogate_likelihood))


if __name__ == "__main__":
    unittest.main()
