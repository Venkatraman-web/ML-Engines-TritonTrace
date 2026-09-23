"""Focused T25 checks for `iste.candidate_screening` (primary forward-accuracy
candidate filtering and ranking).

Run with:
    python -m unittest tests.test_candidate_screening -v
"""

from __future__ import annotations

import unittest

import numpy as np

from iste.candidate_extraction import Candidate, CoreRegion
from iste.candidate_forward_accuracy import CandidateForwardAccuracy
from iste.candidate_screening import (
    CandidateRecord,
    ScreeningConfig,
    build_candidate_record,
    filter_and_rank_candidates,
)
from iste.forward_budget import ForwardBudget

T_S = np.datetime64("2020-01-01T00:00:00", "ns")
T_S_2 = np.datetime64("2020-01-01T06:00:00", "ns")
T_OBS = np.datetime64("2020-01-01T12:00:00", "ns")


def make_record(
    candidate_id, forward_error=0.3, forward_consistency=0.7, status="validated",
    x_peak=100.0, y_peak=20.0, t_s=T_S, support_score=10.0,
    forcing_members_validated=(0, 1), forward_error_by_member=None, forward_consistency_by_member=None,
):
    if forward_error_by_member is None:
        forward_error_by_member = {m: forward_error for m in forcing_members_validated}
    if forward_consistency_by_member is None:
        forward_consistency_by_member = {m: forward_consistency for m in forcing_members_validated}
    return CandidateRecord(
        candidate_id=candidate_id, x_peak=x_peak, y_peak=y_peak, t_s=t_s,
        support_score=support_score, density_score=support_score, weighted_support_score=support_score,
        pixel_diversity_score=2.0,
        core_definition=CoreRegion(center_lat=y_peak, center_lon=x_peak, radius_m=5000.0),
        core_particle_count_Y=20, forward_particle_count_N_forward=10, sampling_seed=42,
        forcing_members_validated=forcing_members_validated,
        forward_error_by_member=forward_error_by_member,
        forward_consistency_by_member=forward_consistency_by_member,
        candidate_forward_error=(float("nan") if status == "unvalidated" else forward_error),
        candidate_forward_consistency=(float("nan") if status == "unvalidated" else forward_consistency),
        forward_validation_status=status,
    )


def decisions_by_id(decisions):
    return {d.record.candidate_id: d for d in decisions}


class TestThresholdModeRejectsAboveThreshold(unittest.TestCase):
    """Requirement 1: threshold mode rejects candidates above the configured
    forward-error threshold."""

    def test_above_threshold_rejected(self):
        records = [make_record("A", forward_error=0.2), make_record("B", forward_error=0.6)]
        config = ScreeningConfig(mode="threshold", error_threshold=0.3)
        by_id = decisions_by_id(filter_and_rank_candidates(records, config))
        self.assertFalse(by_id["B"].accepted)
        self.assertIsNone(by_id["B"].rank)
        self.assertIn("exceeds", by_id["B"].rejection_reason)


class TestThresholdModeRetainsAtOrBelow(unittest.TestCase):
    """Requirement 2: threshold mode retains candidates at/below the threshold."""

    def test_at_and_below_threshold_retained(self):
        records = [make_record("A", forward_error=0.3), make_record("B", forward_error=0.1)]
        config = ScreeningConfig(mode="threshold", error_threshold=0.3)  # A is exactly AT threshold
        by_id = decisions_by_id(filter_and_rank_candidates(records, config))
        self.assertTrue(by_id["A"].accepted)
        self.assertTrue(by_id["B"].accepted)


class TestTopKKeepsExactlyK(unittest.TestCase):
    """Requirement 3: top-K mode keeps exactly the configured number when enough
    exist."""

    def test_keeps_exactly_k_when_enough_valid_candidates(self):
        records = [make_record(f"C{i}", forward_error=float(i)) for i in range(5)]
        config = ScreeningConfig(mode="top_k", top_k=2)
        decisions = filter_and_rank_candidates(records, config)
        accepted = [d for d in decisions if d.accepted]
        self.assertEqual(len(accepted), 2)
        self.assertEqual({d.record.candidate_id for d in accepted}, {"C0", "C1"})


class TestFewerThanKKeepsAll(unittest.TestCase):
    """Requirement 4: fewer than K valid candidates keeps all valid candidates."""

    def test_fewer_than_k_keeps_all_valid(self):
        records = [make_record("A", forward_error=0.1), make_record("B", forward_error=0.2),
                   make_record("C", status="unvalidated")]
        config = ScreeningConfig(mode="top_k", top_k=10)
        decisions = filter_and_rank_candidates(records, config)
        accepted = [d for d in decisions if d.accepted]
        self.assertEqual({d.record.candidate_id for d in accepted}, {"A", "B"})  # C excluded (unvalidated)


class TestRankingOrder(unittest.TestCase):
    """Requirement 5: lower forward error ranks ahead of higher forward error."""

    def test_lower_error_ranks_first(self):
        records = [make_record("A", forward_error=0.5), make_record("B", forward_error=0.1), make_record("C", forward_error=0.3)]
        config = ScreeningConfig(mode="top_k", top_k=3)
        by_id = decisions_by_id(filter_and_rank_candidates(records, config))
        self.assertEqual(by_id["B"].rank, 1)
        self.assertEqual(by_id["C"].rank, 2)
        self.assertEqual(by_id["A"].rank, 3)


class TestUnvalidatedNoFabricatedScore(unittest.TestCase):
    """Requirement 6: unvalidated candidates do not receive a fabricated score/rank."""

    def test_unvalidated_rejected_with_no_rank_in_both_modes(self):
        records = [make_record("A", status="unvalidated"), make_record("B", forward_error=0.2)]
        for config in (ScreeningConfig(mode="threshold", error_threshold=1.0), ScreeningConfig(mode="top_k", top_k=5)):
            with self.subTest(mode=config.mode):
                by_id = decisions_by_id(filter_and_rank_candidates(records, config))
                self.assertFalse(by_id["A"].accepted)
                self.assertIsNone(by_id["A"].rank)
                self.assertIn("unvalidated", by_id["A"].rejection_reason)
                self.assertTrue(np.isnan(by_id["A"].record.candidate_forward_error))


class TestMultipleHistoricalTimesRepresented(unittest.TestCase):
    """Requirement 7: multiple historical times remain represented."""

    def test_candidates_from_different_t_s_both_survive(self):
        records = [make_record("A", forward_error=0.1, t_s=T_S), make_record("B", forward_error=0.2, t_s=T_S_2)]
        config = ScreeningConfig(mode="threshold", error_threshold=0.5)
        by_id = decisions_by_id(filter_and_rank_candidates(records, config))
        self.assertTrue(by_id["A"].accepted)
        self.assertTrue(by_id["B"].accepted)
        self.assertEqual(by_id["A"].record.t_s, T_S)
        self.assertEqual(by_id["B"].record.t_s, T_S_2)


class TestMultipleSpatialModesRepresented(unittest.TestCase):
    """Requirement 8: multiple spatial modes remain represented (no collapse to one
    location)."""

    def test_well_separated_candidates_both_survive_no_merge_by_default(self):
        records = [make_record("A", forward_error=0.1, x_peak=100.0, y_peak=20.0),
                   make_record("B", forward_error=0.2, x_peak=110.0, y_peak=25.0)]
        config = ScreeningConfig(mode="threshold", error_threshold=0.5)  # merge_separation_m default 0.0
        by_id = decisions_by_id(filter_and_rank_candidates(records, config))
        self.assertTrue(by_id["A"].accepted)
        self.assertTrue(by_id["B"].accepted)


class TestBackwardSupportCannotChangeRanking(unittest.TestCase):
    """Requirement 9: backward support cannot change forward ranking when forward
    errors differ."""

    def test_weaker_support_but_better_forward_error_ranks_first(self):
        weak_support_better_forward = make_record("weak-support-good-forward", forward_error=0.1, support_score=1.0)
        strong_support_worse_forward = make_record("strong-support-bad-forward", forward_error=0.5, support_score=1000.0)
        config = ScreeningConfig(mode="top_k", top_k=2)
        by_id = decisions_by_id(filter_and_rank_candidates(
            [weak_support_better_forward, strong_support_worse_forward], config
        ))
        self.assertEqual(by_id["weak-support-good-forward"].rank, 1)
        self.assertEqual(by_id["strong-support-bad-forward"].rank, 2)

    def test_function_never_reads_support_score_for_decisions(self):
        # identical forward errors/positions, wildly different support scores -> identical decisions
        a = make_record("A", forward_error=0.2, support_score=0.001)
        b = make_record("B", forward_error=0.2, support_score=999999.0, x_peak=101.0, y_peak=21.0)
        config = ScreeningConfig(mode="threshold", error_threshold=0.5)
        by_id = decisions_by_id(filter_and_rank_candidates([a, b], config))
        self.assertTrue(by_id["A"].accepted)
        self.assertTrue(by_id["B"].accepted)


class TestNearDuplicateMergeAfterValidationOnly(unittest.TestCase):
    """Requirement 10: near-duplicate merging happens only after forward validation and
    uses the configured spatial threshold."""

    def test_close_candidates_merged_when_threshold_configured(self):
        better = make_record("better", forward_error=0.1, x_peak=100.0, y_peak=20.0)
        worse_nearby = make_record("worse-nearby", forward_error=0.3, x_peak=100.0005, y_peak=20.0005)  # ~60m away
        config = ScreeningConfig(mode="threshold", error_threshold=0.5, merge_separation_m=1000.0)
        by_id = decisions_by_id(filter_and_rank_candidates([better, worse_nearby], config))
        self.assertTrue(by_id["better"].accepted)
        self.assertEqual(by_id["better"].rank, 1)
        self.assertFalse(by_id["worse-nearby"].accepted)
        self.assertIn("merged into near-duplicate candidate 'better'", by_id["worse-nearby"].rejection_reason)

    def test_no_merge_without_configured_threshold(self):
        better = make_record("better", forward_error=0.1, x_peak=100.0, y_peak=20.0)
        worse_nearby = make_record("worse-nearby", forward_error=0.3, x_peak=100.0005, y_peak=20.0005)
        config = ScreeningConfig(mode="threshold", error_threshold=0.5)  # merge_separation_m default 0.0
        by_id = decisions_by_id(filter_and_rank_candidates([better, worse_nearby], config))
        self.assertTrue(by_id["better"].accepted)
        self.assertTrue(by_id["worse-nearby"].accepted)  # both survive -- no merging configured

    def test_far_apart_candidates_not_merged_even_with_threshold(self):
        a = make_record("A", forward_error=0.1, x_peak=100.0, y_peak=20.0)
        b = make_record("B", forward_error=0.2, x_peak=110.0, y_peak=25.0)  # very far
        config = ScreeningConfig(mode="threshold", error_threshold=0.5, merge_separation_m=1000.0)
        by_id = decisions_by_id(filter_and_rank_candidates([a, b], config))
        self.assertTrue(by_id["A"].accepted)
        self.assertTrue(by_id["B"].accepted)

    def test_rejected_candidates_never_merged(self):
        # a threshold-rejected candidate must not be considered for merging even if it
        # would be spatially close to a survivor.
        survivor = make_record("survivor", forward_error=0.1, x_peak=100.0, y_peak=20.0)
        rejected = make_record("rejected", forward_error=5.0, x_peak=100.0005, y_peak=20.0005)
        config = ScreeningConfig(mode="threshold", error_threshold=0.5, merge_separation_m=1000.0)
        by_id = decisions_by_id(filter_and_rank_candidates([survivor, rejected], config))
        self.assertIn("exceeds", by_id["rejected"].rejection_reason)
        self.assertNotIn("merged", by_id["rejected"].rejection_reason)

    def test_never_merges_across_different_t_s(self):
        a = make_record("A", forward_error=0.1, x_peak=100.0, y_peak=20.0, t_s=T_S)
        b = make_record("B", forward_error=0.2, x_peak=100.0005, y_peak=20.0005, t_s=T_S_2)  # same location, different time
        config = ScreeningConfig(mode="threshold", error_threshold=0.5, merge_separation_m=1000.0)
        by_id = decisions_by_id(filter_and_rank_candidates([a, b], config))
        self.assertTrue(by_id["A"].accepted)
        self.assertTrue(by_id["B"].accepted)  # NOT merged despite spatial proximity -- different t_s


class TestCompleteProvenancePreserved(unittest.TestCase):
    """Requirement 11: complete provenance is preserved after filtering/ranking."""

    def test_every_field_preserved_on_the_record(self):
        record = make_record("A", forward_error=0.2, forward_consistency=0.8)
        config = ScreeningConfig(mode="threshold", error_threshold=0.5)
        decision = filter_and_rank_candidates([record], config)[0]
        self.assertIs(decision.record, record)  # exact same object, nothing recomputed
        self.assertEqual(decision.record.candidate_id, "A")
        self.assertEqual(decision.record.core_particle_count_Y, 20)
        self.assertEqual(decision.record.forward_particle_count_N_forward, 10)
        self.assertEqual(decision.record.sampling_seed, 42)
        self.assertEqual(decision.record.forcing_members_validated, (0, 1))


class TestDeterminism(unittest.TestCase):
    """Requirement 12: repeated inputs produce deterministic ordering/ranking."""

    def test_repeated_calls_identical(self):
        records = [make_record("A", forward_error=0.3), make_record("B", forward_error=0.1), make_record("C", forward_error=0.2)]
        config = ScreeningConfig(mode="top_k", top_k=3)
        d1 = decisions_by_id(filter_and_rank_candidates(records, config))
        d2 = decisions_by_id(filter_and_rank_candidates(records, config))
        for cid in ("A", "B", "C"):
            self.assertEqual(d1[cid].rank, d2[cid].rank)
            self.assertEqual(d1[cid].accepted, d2[cid].accepted)

    def test_tie_break_by_candidate_id_is_stable(self):
        records = [make_record("Z", forward_error=0.2), make_record("A", forward_error=0.2)]
        config = ScreeningConfig(mode="top_k", top_k=2)
        by_id = decisions_by_id(filter_and_rank_candidates(records, config))
        self.assertEqual(by_id["A"].rank, 1)  # "A" < "Z" lexicographically, tie-break deterministic
        self.assertEqual(by_id["Z"].rank, 2)


class TestScreeningConfigValidation(unittest.TestCase):
    def test_rejects_unknown_mode(self):
        with self.assertRaises(ValueError):
            ScreeningConfig(mode="bogus")

    def test_threshold_mode_requires_error_threshold(self):
        with self.assertRaises(ValueError):
            ScreeningConfig(mode="threshold")

    def test_top_k_mode_requires_positive_top_k(self):
        with self.assertRaises(ValueError):
            ScreeningConfig(mode="top_k")
        with self.assertRaises(ValueError):
            ScreeningConfig(mode="top_k", top_k=0)

    def test_rejects_negative_merge_separation(self):
        with self.assertRaises(ValueError):
            ScreeningConfig(mode="threshold", error_threshold=1.0, merge_separation_m=-1.0)


class TestBuildCandidateRecordIntegration(unittest.TestCase):
    """Integration: build_candidate_record from real T17/T18/T24 objects."""

    def test_assembles_record_from_real_upstream_objects(self):
        core = CoreRegion(center_lat=20.0, center_lon=100.0, radius_m=5000.0)
        candidate = Candidate(
            candidate_id="cand-A", x_peak=100.0, y_peak=20.0, t_s=T_S, core_region=core,
            support_score=12.0, density_score=8.0, weighted_support_score=6.0, pixel_diversity_score=3.0,
            peak_support_features=None,
        )
        budget = ForwardBudget(candidate_id="cand-A", t_s=T_S, X=10, Y=25, N_forward=10, forward_validatable=True)
        forward_accuracy = CandidateForwardAccuracy(
            candidate_id="cand-A", t_s=T_S, t_obs=T_OBS,
            member_results={}, forcing_members_validated=(0, 1), forward_error_by_member={0: 0.2, 1: 0.3},
            forward_consistency_by_member={0: 0.8, 1: 0.7}, invalid_member_reasons={},
            candidate_forward_error=0.25, candidate_forward_consistency=0.75, forward_validation_status="validated",
        )
        record = build_candidate_record(candidate, budget, sampling_seed=7, forward_accuracy=forward_accuracy)
        self.assertEqual(record.candidate_id, "cand-A")
        self.assertEqual(record.core_particle_count_Y, 25)
        self.assertEqual(record.forward_particle_count_N_forward, 10)
        self.assertEqual(record.sampling_seed, 7)
        self.assertEqual(record.candidate_forward_error, 0.25)
        self.assertIs(record.core_definition, core)

    def test_mismatched_candidate_id_raises(self):
        core = CoreRegion(center_lat=20.0, center_lon=100.0, radius_m=5000.0)
        candidate = Candidate(
            candidate_id="cand-A", x_peak=100.0, y_peak=20.0, t_s=T_S, core_region=core,
            support_score=1.0, density_score=1.0, weighted_support_score=1.0, pixel_diversity_score=1.0,
            peak_support_features=None,
        )
        budget = ForwardBudget(candidate_id="cand-B", t_s=T_S, X=10, Y=25, N_forward=10, forward_validatable=True)
        forward_accuracy = CandidateForwardAccuracy(
            candidate_id="cand-A", t_s=T_S, t_obs=T_OBS, member_results={}, forcing_members_validated=(),
            forward_error_by_member={}, forward_consistency_by_member={}, invalid_member_reasons={},
            candidate_forward_error=float("nan"), candidate_forward_consistency=float("nan"),
            forward_validation_status="unvalidated",
        )
        with self.assertRaises(ValueError):
            build_candidate_record(candidate, budget, sampling_seed=1, forward_accuracy=forward_accuracy)


if __name__ == "__main__":
    unittest.main()
