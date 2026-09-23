"""Focused T24 checks for `iste.candidate_forward_accuracy` (candidate-level forward
accuracy aggregation over forcing members).

Run with:
    python -m unittest tests.test_candidate_forward_accuracy -v
"""

from __future__ import annotations

import unittest

import numpy as np

from iste.candidate_forward_accuracy import (
    CandidateForwardAccuracy,
    aggregate_candidate_forward_accuracy,
)
from iste.forward_accuracy import ComparisonConfig, ForwardAccuracyResult

T_S = np.datetime64("2020-01-01T00:00:00", "ns")
T_OBS = np.datetime64("2020-01-01T06:00:00", "ns")
DUMMY_CONFIG = ComparisonConfig(metric="footprint_overlap")


def make_result(member_id, error=None, consistency=None, valid=True, invalid_reason="", candidate_id="cand-A", t_s=T_S, t_obs=T_OBS):
    return ForwardAccuracyResult(
        candidate_id=candidate_id, t_s=t_s, t_obs=t_obs, forward_forcing_member_id=member_id,
        metric="footprint_overlap", config=DUMMY_CONFIG,
        error=(float("nan") if not valid else error), consistency=(float("nan") if not valid else consistency),
        valid=valid, invalid_reason=invalid_reason,
    )


class TestTwoValidMembers(unittest.TestCase):
    """Requirement 1: two valid members -> arithmetic mean error/consistency."""

    def test_two_valid_members_arithmetic_mean(self):
        results = {0: make_result(0, error=0.2, consistency=0.8), 1: make_result(1, error=0.4, consistency=0.6)}
        agg = aggregate_candidate_forward_accuracy(results)
        self.assertAlmostEqual(agg.candidate_forward_error, 0.3)
        self.assertAlmostEqual(agg.candidate_forward_consistency, 0.7)
        self.assertEqual(agg.forward_validation_status, "validated")


class TestThreeOrMoreValidMembers(unittest.TestCase):
    """Requirement 2: three or more valid members -> correct arithmetic mean."""

    def test_four_valid_members(self):
        errors = [0.1, 0.2, 0.3, 0.4]
        consistencies = [0.9, 0.8, 0.7, 0.6]
        results = {i: make_result(i, error=e, consistency=c) for i, (e, c) in enumerate(zip(errors, consistencies))}
        agg = aggregate_candidate_forward_accuracy(results)
        self.assertAlmostEqual(agg.candidate_forward_error, np.mean(errors))
        self.assertAlmostEqual(agg.candidate_forward_consistency, np.mean(consistencies))


class TestOneValidOneInvalid(unittest.TestCase):
    """Requirement 3: one valid + one invalid -> aggregate uses only the valid member."""

    def test_invalid_member_excluded_from_mean(self):
        results = {
            0: make_result(0, error=0.25, consistency=0.75),
            1: make_result(1, valid=False, invalid_reason="predicted footprint has no valid particles"),
        }
        agg = aggregate_candidate_forward_accuracy(results)
        self.assertAlmostEqual(agg.candidate_forward_error, 0.25)
        self.assertAlmostEqual(agg.candidate_forward_consistency, 0.75)
        self.assertEqual(agg.forcing_members_validated, (0,))
        self.assertEqual(agg.forward_validation_status, "validated")


class TestMultipleInvalidReasonsPreserved(unittest.TestCase):
    """Requirement 4: multiple invalid members -> invalid reasons are preserved."""

    def test_invalid_reasons_recorded_per_member(self):
        results = {
            0: make_result(0, error=0.1, consistency=0.9),
            1: make_result(1, valid=False, invalid_reason="predicted footprint has no valid particles"),
            2: make_result(2, valid=False, invalid_reason="observed reference has no valid particles"),
        }
        agg = aggregate_candidate_forward_accuracy(results)
        self.assertEqual(
            agg.invalid_member_reasons,
            {1: "predicted footprint has no valid particles", 2: "observed reference has no valid particles"},
        )
        self.assertEqual(agg.forcing_members_validated, (0,))


class TestNoValidMembers(unittest.TestCase):
    """Requirement 5: no valid members -> explicitly forward-unvalidated, no fabricated
    score."""

    def test_all_invalid_gives_unvalidated_status_and_nan_aggregate(self):
        results = {
            0: make_result(0, valid=False, invalid_reason="predicted footprint has no valid particles"),
            1: make_result(1, valid=False, invalid_reason="observed reference has no valid particles"),
        }
        agg = aggregate_candidate_forward_accuracy(results)
        self.assertEqual(agg.forward_validation_status, "unvalidated")
        self.assertTrue(np.isnan(agg.candidate_forward_error))
        self.assertTrue(np.isnan(agg.candidate_forward_consistency))
        self.assertEqual(agg.forcing_members_validated, ())
        self.assertEqual(len(agg.invalid_member_reasons), 2)

    def test_never_fabricates_zero_or_infinity(self):
        results = {0: make_result(0, valid=False, invalid_reason="x")}
        agg = aggregate_candidate_forward_accuracy(results)
        self.assertFalse(agg.candidate_forward_error == 0.0)
        self.assertFalse(np.isinf(agg.candidate_forward_error))


class TestMemberOrderIndependence(unittest.TestCase):
    """Requirement 6: member order does not change the aggregate."""

    def test_dict_insertion_order_does_not_affect_result(self):
        results_forward = {
            0: make_result(0, error=0.1, consistency=0.9),
            1: make_result(1, error=0.3, consistency=0.7),
            2: make_result(2, error=0.5, consistency=0.5),
        }
        results_reversed = {2: results_forward[2], 1: results_forward[1], 0: results_forward[0]}
        agg_forward = aggregate_candidate_forward_accuracy(results_forward)
        agg_reversed = aggregate_candidate_forward_accuracy(results_reversed)
        self.assertEqual(agg_forward.candidate_forward_error, agg_reversed.candidate_forward_error)
        self.assertEqual(agg_forward.candidate_forward_consistency, agg_reversed.candidate_forward_consistency)
        self.assertEqual(agg_forward.forcing_members_validated, agg_reversed.forcing_members_validated)


class TestPerMemberValuesPreservedExactly(unittest.TestCase):
    """Requirement 7: per-member values are preserved exactly."""

    def test_member_results_dict_matches_input_exactly(self):
        r0 = make_result(0, error=0.15, consistency=0.85)
        r1 = make_result(1, valid=False, invalid_reason="predicted footprint has no valid particles")
        results = {0: r0, 1: r1}
        agg = aggregate_candidate_forward_accuracy(results)
        self.assertIs(agg.member_results[0], r0)
        self.assertIs(agg.member_results[1], r1)
        self.assertEqual(agg.forward_error_by_member, {0: 0.15})
        self.assertEqual(agg.forward_consistency_by_member, {0: 0.85})

    def test_returned_member_results_is_a_copy_not_the_caller_dict(self):
        results = {0: make_result(0, error=0.1, consistency=0.9)}
        agg = aggregate_candidate_forward_accuracy(results)
        self.assertIsNot(agg.member_results, results)
        results[1] = make_result(1, error=0.9, consistency=0.1)  # mutate caller's dict afterward
        self.assertNotIn(1, agg.member_results)  # aggregate is unaffected


class TestBackwardSupportCannotAlterAggregate(unittest.TestCase):
    """Requirement 8: backward support cannot alter candidate_forward_error -- the
    aggregation function never receives backward-KDE support data at all, so identical
    member error/consistency values always produce identical aggregates regardless of
    any (hypothetical) backward support context."""

    def test_function_signature_has_no_support_parameter(self):
        import inspect
        sig = inspect.signature(aggregate_candidate_forward_accuracy)
        self.assertEqual(list(sig.parameters.keys()), ["member_results"])

    def test_identical_member_values_give_identical_aggregate_regardless_of_candidate_identity(self):
        # two "candidates" (different candidate_id, standing in for different backward
        # support scores) with numerically IDENTICAL member error/consistency values
        # must produce numerically identical candidate_forward_error/consistency.
        results_a = {0: make_result(0, error=0.2, consistency=0.8, candidate_id="cand-HIGH-SUPPORT")}
        results_b = {0: make_result(0, error=0.2, consistency=0.8, candidate_id="cand-LOW-SUPPORT")}
        agg_a = aggregate_candidate_forward_accuracy(results_a)
        agg_b = aggregate_candidate_forward_accuracy(results_b)
        self.assertEqual(agg_a.candidate_forward_error, agg_b.candidate_forward_error)
        self.assertEqual(agg_a.candidate_forward_consistency, agg_b.candidate_forward_consistency)


class TestCandidateTimeGroupsIndependent(unittest.TestCase):
    """Requirement 9: candidate/time groups remain independent."""

    def test_two_different_candidates_do_not_interfere(self):
        results_a = {0: make_result(0, error=0.1, consistency=0.9, candidate_id="cand-A"),
                     1: make_result(1, error=0.3, consistency=0.7, candidate_id="cand-A")}
        results_b = {0: make_result(0, error=0.5, consistency=0.5, candidate_id="cand-B")}
        agg_a = aggregate_candidate_forward_accuracy(results_a)
        agg_b = aggregate_candidate_forward_accuracy(results_b)
        self.assertEqual(agg_a.candidate_id, "cand-A")
        self.assertAlmostEqual(agg_a.candidate_forward_error, 0.2)
        self.assertEqual(agg_b.candidate_id, "cand-B")
        self.assertAlmostEqual(agg_b.candidate_forward_error, 0.5)

    def test_mismatched_candidate_id_across_members_raises(self):
        results = {0: make_result(0, error=0.1, consistency=0.9, candidate_id="cand-A"),
                   1: make_result(1, error=0.2, consistency=0.8, candidate_id="cand-B")}
        with self.assertRaises(ValueError):
            aggregate_candidate_forward_accuracy(results)

    def test_mismatched_t_s_across_members_raises(self):
        other_t_s = T_S + np.timedelta64(1, "h")
        results = {0: make_result(0, error=0.1, consistency=0.9, t_s=T_S),
                   1: make_result(1, error=0.2, consistency=0.8, t_s=other_t_s)}
        with self.assertRaises(ValueError):
            aggregate_candidate_forward_accuracy(results)

    def test_key_mismatched_with_own_forward_forcing_member_id_raises(self):
        results = {0: make_result(1, error=0.1, consistency=0.9)}  # key 0, but result says member 1
        with self.assertRaises(ValueError):
            aggregate_candidate_forward_accuracy(results)


class TestDeterminism(unittest.TestCase):
    """Requirement 10: same input results produce deterministic aggregation."""

    def test_repeated_calls_produce_identical_result(self):
        results = {0: make_result(0, error=0.2, consistency=0.8), 1: make_result(1, error=0.4, consistency=0.6)}
        agg1 = aggregate_candidate_forward_accuracy(results)
        agg2 = aggregate_candidate_forward_accuracy(results)
        self.assertEqual(agg1.candidate_forward_error, agg2.candidate_forward_error)
        self.assertEqual(agg1.candidate_forward_consistency, agg2.candidate_forward_consistency)
        self.assertEqual(agg1.forcing_members_validated, agg2.forcing_members_validated)


class TestProvenancePreserved(unittest.TestCase):
    def test_candidate_t_s_t_obs_preserved(self):
        results = {0: make_result(0, error=0.1, consistency=0.9, candidate_id="cand-XYZ", t_s=T_S, t_obs=T_OBS)}
        agg = aggregate_candidate_forward_accuracy(results)
        self.assertEqual(agg.candidate_id, "cand-XYZ")
        self.assertEqual(agg.t_s, T_S)
        self.assertEqual(agg.t_obs, T_OBS)


class TestEmptyInputRejected(unittest.TestCase):
    def test_empty_dict_raises(self):
        with self.assertRaises(ValueError):
            aggregate_candidate_forward_accuracy({})


if __name__ == "__main__":
    unittest.main()
