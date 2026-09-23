"""Focused T14 checks for `iste.likelihood` (Stage IV surrogate-likelihood core).

**2026-09-22 correction:** these tests were rewritten to match the corrected, frozen-
contract ordering (`final_stage_III_IV_specification.txt`) -- `beta` applied PER MEMBER
before marginalizing (`mean_e(L_e^beta)`), replacing an earlier, now-superseded ordering
(`(mean_e L_e)^beta`) these tests previously encoded as correct.

Run with:
    python -m unittest tests.test_likelihood -v
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from iste.likelihood import Stage4LikelihoodResult, compute_stage4_log_likelihood


class TestOnePixel(unittest.TestCase):
    def test_single_pixel_single_member_hand_computable(self):
        # y=1, pi=0.8, epsilon=0 -> P = 0.8 exactly; single member -> L_e = P.
        # single member: mean_e(L_e^beta) == L_e^beta == P^beta (beta=1 here -> P).
        y = np.array([1.0])
        pi = np.array([[0.8]])
        res = compute_stage4_log_likelihood(y, pi, epsilon=0.0, beta=1.0)
        self.assertAlmostEqual(res.log_pixel_likelihood[0, 0], math.log(0.8))
        self.assertAlmostEqual(res.log_member_likelihood[0], math.log(0.8))
        self.assertAlmostEqual(res.log_mean_member_likelihood, math.log(0.8))
        self.assertAlmostEqual(res.tempered_log_member_likelihood[0], math.log(0.8))
        self.assertAlmostEqual(res.log_surrogate_likelihood, math.log(0.8))
        self.assertAlmostEqual(res.responsibilities[0], 1.0)
        self.assertAlmostEqual(res.member_ess, 1.0)


class TestMultiplePixels(unittest.TestCase):
    def test_two_pixels_single_member(self):
        # y=[1,0], pi=[0.8,0.3], epsilon=0 -> P = [0.8, 0.7] -> L_e = 0.8*0.7 = 0.56
        y = np.array([1.0, 0.0])
        pi = np.array([[0.8], [0.3]])
        res = compute_stage4_log_likelihood(y, pi, epsilon=0.0, beta=1.0)
        expected_log_L = math.log(0.8) + math.log(0.7)
        self.assertAlmostEqual(res.log_member_likelihood[0], expected_log_L)
        self.assertAlmostEqual(res.log_surrogate_likelihood, expected_log_L)

    def test_log_pixel_likelihood_sums_to_member_likelihood(self):
        rng = np.random.default_rng(1)
        y = rng.integers(0, 2, size=10).astype(float)
        pi = rng.uniform(0.05, 0.95, size=(10, 4))
        res = compute_stage4_log_likelihood(y, pi, epsilon=0.05, beta=0.7)
        np.testing.assert_allclose(
            res.log_member_likelihood, np.sum(res.log_pixel_likelihood, axis=0)
        )


class TestMultipleForcingMembers(unittest.TestCase):
    def test_three_members_hand_computable(self):
        y = np.array([1.0])
        pi = np.array([[0.9, 0.5, 0.1]])
        res = compute_stage4_log_likelihood(y, pi, epsilon=0.0, beta=1.0)
        expected_log_L = np.log([0.9, 0.5, 0.1])
        np.testing.assert_allclose(res.log_member_likelihood, expected_log_L)
        expected_mean = np.mean([0.9, 0.5, 0.1])
        self.assertAlmostEqual(res.log_mean_member_likelihood, math.log(expected_mean))
        # beta=1 -> tempered == untempered -> surrogate == untempered mean too.
        self.assertAlmostEqual(res.log_surrogate_likelihood, math.log(expected_mean))


class TestEpsilonContribution(unittest.TestCase):
    def test_epsilon_floor_when_prediction_is_maximally_wrong(self):
        # y=0, pi=1.0: without epsilon this is impossible (P=0); epsilon provides a floor.
        y = np.array([0.0])
        pi = np.array([[1.0]])
        res_no_eps = compute_stage4_log_likelihood(y, pi, epsilon=0.0, beta=1.0)
        self.assertTrue(np.isneginf(res_no_eps.log_pixel_likelihood[0, 0]))

        res_eps = compute_stage4_log_likelihood(y, pi, epsilon=0.2, beta=1.0)
        self.assertAlmostEqual(res_eps.log_pixel_likelihood[0, 0], math.log(0.1))  # eps/2

    def test_epsilon_floor_when_prediction_is_maximally_right(self):
        # y=1, pi=1.0, epsilon=0 -> P = 1.0 exactly (no epsilon dilution at the correct
        # extreme, since (1-eps)*1 + eps/2 with eps=0 is 1.0).
        y = np.array([1.0])
        pi = np.array([[1.0]])
        res = compute_stage4_log_likelihood(y, pi, epsilon=0.0, beta=1.0)
        self.assertAlmostEqual(res.log_pixel_likelihood[0, 0], 0.0)  # log(1.0)

        # with epsilon > 0, the correct-and-confident case is slightly diluted:
        # P = (1-eps)*1 + eps/2 = 1 - eps/2
        res_eps = compute_stage4_log_likelihood(y, pi, epsilon=0.2, beta=1.0)
        self.assertAlmostEqual(res_eps.log_pixel_likelihood[0, 0], math.log(1.0 - 0.1))


class TestBetaAppliedPerMemberBeforeMarginalizing(unittest.TestCase):
    """The corrected Stage IV ordering: beta * log_L_e per member, THEN marginalize."""

    def test_beta_equals_one_is_identity(self):
        y = np.array([1.0, 0.0, 1.0])
        pi = np.array([[0.9, 0.3], [0.1, 0.6], [0.7, 0.8]])
        res = compute_stage4_log_likelihood(y, pi, epsilon=0.05, beta=1.0)
        self.assertAlmostEqual(res.log_surrogate_likelihood, res.log_mean_member_likelihood)
        np.testing.assert_allclose(res.tempered_log_member_likelihood, res.log_member_likelihood)

    def test_beta_applied_to_each_member_before_logsumexp(self):
        y = np.array([1.0, 0.0])
        pi = np.array([[0.8, 0.3], [0.2, 0.6]])
        for beta in (0.25, 0.5, 2.0, 3.0):
            res = compute_stage4_log_likelihood(y, pi, epsilon=0.0, beta=beta)
            # tempered_log_member_likelihood must be beta * (untempered log_L_e) exactly
            np.testing.assert_allclose(res.tempered_log_member_likelihood, beta * res.log_member_likelihood)
            # and log_surrogate_likelihood must be logsumexp(tempered) - log(E), i.e.
            # NOT beta * log_mean_member_likelihood (that would be the superseded ordering).
            E = len(res.log_member_likelihood)
            m = np.max(res.tempered_log_member_likelihood)
            expected = m + math.log(np.sum(np.exp(res.tempered_log_member_likelihood - m))) - math.log(E)
            self.assertAlmostEqual(res.log_surrogate_likelihood, expected, places=9)

    def test_beta_zero_gives_log_surrogate_zero_and_uniform_responsibilities(self):
        # mean_e(L_e^0) = mean_e(1) = 1 -> log == 0, regardless of the data, including a
        # member whose untempered L_e is exactly 0 (empty-product convention: L_e^0 = 1).
        y = np.array([1.0, 0.0])
        pi = np.array([[0.8, 0.3], [0.2, 0.6]])
        res = compute_stage4_log_likelihood(y, pi, epsilon=0.0, beta=0.0)
        self.assertAlmostEqual(res.log_surrogate_likelihood, 0.0)
        np.testing.assert_allclose(res.responsibilities, [0.5, 0.5])
        self.assertAlmostEqual(res.member_ess, 2.0)

    def test_beta_zero_handles_a_neg_inf_member_without_nan(self):
        # member 0 has log_L_e = -inf (epsilon=0, pixel flatly contradicted); beta=0 must
        # still give log_surrogate=0 and uniform responsibilities, not NaN from 0*(-inf).
        y = np.array([1.0])
        pi = np.array([[0.0, 0.5]])  # member 0: P=0 exactly (log=-inf); member 1: P=0.5
        res = compute_stage4_log_likelihood(y, pi, epsilon=0.0, beta=0.0)
        self.assertTrue(np.isneginf(res.log_member_likelihood[0]))
        self.assertFalse(np.any(np.isnan(res.tempered_log_member_likelihood)))
        self.assertAlmostEqual(res.log_surrogate_likelihood, 0.0)
        np.testing.assert_allclose(res.responsibilities, [0.5, 0.5])
        self.assertAlmostEqual(res.member_ess, 2.0)


class TestCorrectVsIncorrectTempering(unittest.TestCase):
    """Prove the implementation computes `mean_e(L_e^beta)` (the frozen contract's
    ordering), and NOT the superseded `(mean_e(L_e))^beta` -- a regression test that
    fails if the ensemble marginalization and the beta exponentiation are ever
    accidentally swapped back."""

    def test_matches_mean_of_powers_not_power_of_mean(self):
        # hand-computable, epsilon=0: L_0 = 0.8*0.8 = 0.64, L_1 = 0.4*0.1 = 0.04
        y = np.array([1.0, 0.0])
        pi = np.array([[0.8, 0.4], [0.2, 0.9]])
        beta = 2.0
        res = compute_stage4_log_likelihood(y, pi, epsilon=0.0, beta=beta)

        L = np.array([0.64, 0.04])
        correct = np.mean(L ** beta)          # mean_e(L_e^beta) -- the frozen contract's formula
        incorrect = np.mean(L) ** beta        # (mean_e(L_e))^beta -- superseded, explicitly forbidden

        self.assertAlmostEqual(correct, (0.64 ** 2 + 0.04 ** 2) / 2)
        self.assertAlmostEqual(incorrect, 0.1156)
        self.assertNotAlmostEqual(correct, incorrect)  # sanity: the two must actually differ

        computed = math.exp(res.log_surrogate_likelihood)
        self.assertAlmostEqual(computed, correct, places=9)
        self.assertGreater(abs(computed - incorrect), 0.05)  # nowhere near the superseded formula

    def test_hand_checkable_hint_from_task_spec(self):
        # L_1=0.8, L_2=0.2, beta=0.5 (single pixel per member, epsilon=0, so L_e == P_e)
        # correct: mean(L_e^beta) = (sqrt(0.8) + sqrt(0.2)) / 2
        # incorrect: sqrt(mean(L_e)) = sqrt((0.8+0.2)/2) = sqrt(0.5)
        y = np.array([1.0])
        pi = np.array([[0.8, 0.2]])
        res = compute_stage4_log_likelihood(y, pi, epsilon=0.0, beta=0.5)

        expected_correct = (math.sqrt(0.8) + math.sqrt(0.2)) / 2.0
        expected_incorrect = math.sqrt(0.5)
        self.assertNotAlmostEqual(expected_correct, expected_incorrect)

        computed = math.exp(res.log_surrogate_likelihood)
        self.assertAlmostEqual(computed, expected_correct, places=9)
        self.assertGreater(abs(computed - expected_incorrect), 0.01)

    def test_matches_correct_formula_across_random_cases(self):
        rng = np.random.default_rng(42)
        for _ in range(20):
            n_pixels = rng.integers(1, 6)
            n_members = rng.integers(1, 5)
            y = rng.integers(0, 2, size=n_pixels).astype(float)
            pi = rng.uniform(0.05, 0.95, size=(n_pixels, n_members))
            epsilon = rng.uniform(0.0, 0.3)
            beta = rng.uniform(0.1, 3.0)

            res = compute_stage4_log_likelihood(y, pi, epsilon, beta)

            # reference computed independently in linear space (safe at this tiny scale)
            p = (1 - epsilon) * pi ** y[:, None] * (1 - pi) ** (1 - y[:, None]) + epsilon / 2
            L = np.prod(p, axis=0)
            expected_correct = np.mean(L ** beta)
            expected_incorrect = np.mean(L) ** beta

            computed = math.exp(res.log_surrogate_likelihood)
            self.assertAlmostEqual(computed, expected_correct, places=6)
            if not math.isclose(expected_correct, expected_incorrect, rel_tol=1e-9):
                self.assertNotAlmostEqual(computed, expected_incorrect, places=6)


class TestFrozenToyExample(unittest.TestCase):
    """`final_stage_III_IV_specification.txt` Sec. 32's hand-verified toy example. The
    equations produce this result; the rounded numbers are only the comparison tolerance,
    never hard-coded as the implementation."""

    def test_toy_example_log_p_tilde_and_responsibilities(self):
        # single pixel per member, epsilon=0 -> log_L_e == log(P_e) directly, chosen so
        # log_L_1 = -0.1761, log_L_2 = -5.4150 exactly (P_e = exp(log_L_e)).
        log_L_1, log_L_2 = -0.1761, -5.4150
        pi = np.array([[math.exp(log_L_1), math.exp(log_L_2)]])
        y = np.array([1.0])
        beta = 0.5

        res = compute_stage4_log_likelihood(y, pi, epsilon=0.0, beta=beta)
        np.testing.assert_allclose(res.log_member_likelihood, [log_L_1, log_L_2], atol=1e-9)

        self.assertAlmostEqual(res.log_surrogate_likelihood, -0.7109, places=3)
        self.assertAlmostEqual(math.exp(res.log_surrogate_likelihood), 0.4912, places=3)

        self.assertAlmostEqual(res.responsibilities[0], 0.932, places=3)
        self.assertAlmostEqual(res.responsibilities[1], 0.068, places=3)
        self.assertAlmostEqual(np.sum(res.responsibilities), 1.0, places=9)

        self.assertAlmostEqual(res.member_ess, 1.14, places=1)


class TestResponsibilitiesAndESS(unittest.TestCase):
    def test_responsibilities_sum_to_one(self):
        y = np.array([1.0, 0.0, 1.0])
        pi = np.array([[0.9, 0.3, 0.5], [0.1, 0.6, 0.4], [0.7, 0.8, 0.2]])
        res = compute_stage4_log_likelihood(y, pi, epsilon=0.05, beta=0.7)
        self.assertAlmostEqual(np.sum(res.responsibilities), 1.0, places=9)

    def test_ess_bounds_for_valid_responsibilities(self):
        y = np.array([1.0, 0.0])
        pi = np.array([[0.8, 0.3, 0.6], [0.2, 0.6, 0.4]])
        res = compute_stage4_log_likelihood(y, pi, epsilon=0.02, beta=1.0)
        self.assertGreaterEqual(res.member_ess, 1.0)
        self.assertLessEqual(res.member_ess, 3.0)

    def test_ess_equals_E_for_equal_member_likelihoods(self):
        y = np.array([1.0, 0.0])
        pi = np.array([[0.7, 0.7, 0.7], [0.4, 0.4, 0.4]])  # identical across members
        res = compute_stage4_log_likelihood(y, pi, epsilon=0.0, beta=1.0)
        np.testing.assert_allclose(res.responsibilities, [1 / 3, 1 / 3, 1 / 3], atol=1e-9)
        self.assertAlmostEqual(res.member_ess, 3.0, places=9)

    def test_larger_tempered_log_likelihood_gets_larger_responsibility(self):
        y = np.array([1.0])
        pi = np.array([[0.9, 0.2, 0.5]])  # member 0 clearly most likely
        res = compute_stage4_log_likelihood(y, pi, epsilon=0.0, beta=1.5)
        self.assertGreater(res.responsibilities[0], res.responsibilities[2])
        self.assertGreater(res.responsibilities[2], res.responsibilities[1])

    def test_responsibilities_computed_from_tempered_not_untempered(self):
        # with unequal beta, responsibilities must reflect the TEMPERED order, which can
        # differ from the untempered order when beta interacts with per-member spread.
        y = np.array([1.0, 1.0])
        # member 0: moderate single high pixel-count product; member 1: one near-1 pixel,
        # one near-0.5 pixel -- construct so beta reorders which dominates in log-space is
        # not the point; the point is responsibilities are literally softmax(tempered).
        pi = np.array([[0.6, 0.99], [0.6, 0.51]])
        beta = 3.0
        res = compute_stage4_log_likelihood(y, pi, epsilon=0.0, beta=beta)
        m = np.max(res.tempered_log_member_likelihood)
        expected_r = np.exp(res.tempered_log_member_likelihood - m)
        expected_r = expected_r / np.sum(expected_r)
        np.testing.assert_allclose(res.responsibilities, expected_r, atol=1e-12)

    def test_neg_inf_member_gets_zero_responsibility_when_another_is_finite(self):
        y = np.array([1.0])
        pi = np.array([[0.0, 0.5]])  # member 0: P=0 exactly -> log_L=-inf
        res = compute_stage4_log_likelihood(y, pi, epsilon=0.0, beta=1.0)
        self.assertTrue(np.isneginf(res.tempered_log_member_likelihood[0]))
        self.assertAlmostEqual(res.responsibilities[0], 0.0)
        self.assertAlmostEqual(res.responsibilities[1], 1.0)

    def test_all_members_neg_inf_gives_nan_responsibilities_not_misleading_finite(self):
        y = np.array([1.0])
        pi = np.array([[0.0, 0.0]])
        res = compute_stage4_log_likelihood(y, pi, epsilon=0.0, beta=1.0)
        self.assertTrue(np.all(np.isnan(res.responsibilities)))
        self.assertTrue(math.isnan(res.member_ess))


class TestVerySmallProbabilitiesAndStability(unittest.TestCase):
    def test_many_pixels_tiny_probabilities_stays_finite(self):
        # naive linear-space product would underflow to exactly 0.0 for this many pixels;
        # the log-domain implementation must stay finite.
        rng = np.random.default_rng(7)
        n_pixels = 2000
        y = rng.integers(0, 2, size=n_pixels).astype(float)
        pi = np.full((n_pixels, 3), 0.001)
        res = compute_stage4_log_likelihood(y, pi, epsilon=1e-6, beta=0.5)

        self.assertTrue(np.all(np.isfinite(res.log_member_likelihood)))
        self.assertTrue(np.isfinite(res.log_mean_member_likelihood))
        self.assertTrue(np.isfinite(res.log_surrogate_likelihood))
        self.assertTrue(np.all(np.isfinite(res.responsibilities)))
        self.assertTrue(np.isfinite(res.member_ess))

        # confirm the naive linear-space product really would have underflowed, to prove
        # this test is actually exercising the numerical-stability concern.
        naive_products = np.prod(np.exp(res.log_pixel_likelihood), axis=0)
        self.assertTrue(np.all(naive_products == 0.0))

    def test_log_mean_exp_matches_naive_computation_when_naive_is_safe(self):
        # at a small, safe scale, the stable log-mean-exp must agree with the naive
        # (unshifted) computation to floating precision.
        y = np.array([1.0, 0.0, 1.0])
        pi = np.array([[0.7, 0.6, 0.5], [0.3, 0.4, 0.5], [0.6, 0.6, 0.6]])
        res = compute_stage4_log_likelihood(y, pi, epsilon=0.02, beta=1.0)
        naive_log_mean = math.log(np.mean(np.exp(res.log_member_likelihood)))
        self.assertAlmostEqual(res.log_mean_member_likelihood, naive_log_mean, places=9)

    def test_extremely_unbalanced_members_no_nan_or_inf(self):
        # one member overwhelmingly more likely than another (huge log-likelihood gap) --
        # the shift-by-max trick must keep this finite and dominated by the likely member.
        log_L = np.array([-5.0, -50000.0, -3.0])
        from iste.likelihood import _log_mean_exp
        result = _log_mean_exp(log_L)
        self.assertTrue(np.isfinite(result))
        # dominated by the two large (less negative) terms; the -50000 term contributes ~0.
        reference = math.log((math.exp(-5.0) + math.exp(-3.0)) / 3.0)
        self.assertAlmostEqual(result, reference, places=6)

    def test_all_members_impossible_gives_neg_inf_not_nan(self):
        # epsilon=0 and a pixel flatly contradicted by every member -> every L_e == 0.
        y = np.array([1.0])
        pi = np.array([[0.0, 0.0]])
        res = compute_stage4_log_likelihood(y, pi, epsilon=0.0, beta=1.0)
        self.assertTrue(np.all(np.isneginf(res.log_member_likelihood)))
        self.assertTrue(np.isneginf(res.log_mean_member_likelihood))
        self.assertTrue(np.isneginf(res.log_surrogate_likelihood))
        self.assertFalse(np.any(np.isnan(res.log_member_likelihood)))


class TestInvalidInputs(unittest.TestCase):
    def setUp(self):
        self.y = np.array([1.0, 0.0])
        self.pi = np.array([[0.5, 0.6], [0.4, 0.3]])

    def test_non_binary_y_rejected(self):
        with self.assertRaises(ValueError):
            compute_stage4_log_likelihood(np.array([0.5, 0.0]), self.pi, 0.05, 1.0)

    def test_pi_out_of_range_rejected(self):
        bad_pi = self.pi.copy()
        bad_pi[0, 0] = 1.5
        with self.assertRaises(ValueError):
            compute_stage4_log_likelihood(self.y, bad_pi, 0.05, 1.0)

        bad_pi2 = self.pi.copy()
        bad_pi2[0, 0] = -0.1
        with self.assertRaises(ValueError):
            compute_stage4_log_likelihood(self.y, bad_pi2, 0.05, 1.0)

    def test_pi_non_finite_rejected(self):
        bad_pi = self.pi.copy()
        bad_pi[0, 0] = np.nan
        with self.assertRaises(ValueError):
            compute_stage4_log_likelihood(self.y, bad_pi, 0.05, 1.0)

        bad_pi2 = self.pi.copy()
        bad_pi2[0, 0] = np.inf
        with self.assertRaises(ValueError):
            compute_stage4_log_likelihood(self.y, bad_pi2, 0.05, 1.0)

    def test_epsilon_out_of_range_rejected(self):
        with self.assertRaises(ValueError):
            compute_stage4_log_likelihood(self.y, self.pi, -0.01, 1.0)
        with self.assertRaises(ValueError):
            compute_stage4_log_likelihood(self.y, self.pi, 1.01, 1.0)
        with self.assertRaises(ValueError):
            compute_stage4_log_likelihood(self.y, self.pi, float("nan"), 1.0)

    def test_beta_invalid_rejected(self):
        with self.assertRaises(ValueError):
            compute_stage4_log_likelihood(self.y, self.pi, 0.05, -1.0)
        with self.assertRaises(ValueError):
            compute_stage4_log_likelihood(self.y, self.pi, 0.05, float("nan"))
        with self.assertRaises(ValueError):
            compute_stage4_log_likelihood(self.y, self.pi, 0.05, float("inf"))

    def test_mismatched_ensemble_dimensions_rejected(self):
        wrong_shape_pi = np.array([[0.5, 0.6]])  # 1 pixel, but y has 2
        with self.assertRaises(ValueError):
            compute_stage4_log_likelihood(self.y, wrong_shape_pi, 0.05, 1.0)

    def test_y_wrong_ndim_rejected(self):
        with self.assertRaises(ValueError):
            compute_stage4_log_likelihood(self.pi, self.pi, 0.05, 1.0)  # 2-D y

    def test_pi_wrong_ndim_rejected(self):
        with self.assertRaises(ValueError):
            compute_stage4_log_likelihood(self.y, np.array([0.5, 0.6]), 0.05, 1.0)  # 1-D pi

    def test_empty_y_rejected(self):
        with self.assertRaises(ValueError):
            compute_stage4_log_likelihood(np.array([]), np.empty((0, 2)), 0.05, 1.0)

    def test_zero_forcing_members_rejected(self):
        with self.assertRaises(ValueError):
            compute_stage4_log_likelihood(self.y, np.empty((2, 0)), 0.05, 1.0)


if __name__ == "__main__":
    unittest.main()
