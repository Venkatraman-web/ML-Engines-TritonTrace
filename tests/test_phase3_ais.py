"""Focused T35 checks for `iste.phase3_ais` (Phase 3 AIS evidence hook).

Run with:
    python -m unittest tests.test_phase3_ais -v
"""

from __future__ import annotations

import unittest

import numpy as np

from iste.candidate_extraction import Candidate, CoreRegion
from iste.phase3_ais import (
    AISDataSource,
    AISEvidence,
    compute_ais_evidence,
    renormalize_weighted_mean,
)

T_S = np.datetime64("2020-01-01T00:00:00", "ns")


def _candidate(cand_id="c0") -> Candidate:
    return Candidate(
        candidate_id=cand_id, x_peak=35.0, y_peak=34.0, t_s=T_S,
        core_region=CoreRegion(center_lat=34.0, center_lon=35.0, radius_m=1000.0),
        support_score=10.0, density_score=5.0, weighted_support_score=3.0,
        pixel_diversity_score=1.0, peak_support_features=None,
    )


class TestAbstractInterfaceCannotBeInstantiated(unittest.TestCase):
    def test_ais_data_source_is_abstract(self):
        with self.assertRaises(TypeError):
            AISDataSource()

    def test_incomplete_subclass_cannot_be_instantiated(self):
        class IncompleteSource(AISDataSource):
            pass

        with self.assertRaises(TypeError):
            IncompleteSource()


class TestUnavailableSentinel(unittest.TestCase):
    def test_no_source_returns_unavailable_without_raising(self):
        candidate = _candidate()
        result = compute_ais_evidence(candidate)
        self.assertIsInstance(result, AISEvidence)
        self.assertFalse(result.ais_available)

    def test_all_fields_none_when_unavailable(self):
        result = compute_ais_evidence(_candidate())
        self.assertIsNone(result.distance_to_nearest_track_m)
        self.assertIsNone(result.n_vessels_nearby)
        self.assertIsNone(result.presence_duration_hours)
        self.assertIsNone(result.trajectory_compatibility_score)
        self.assertIsNone(result.ais_score)

    def test_reason_explains_why(self):
        result = compute_ais_evidence(_candidate())
        self.assertIn("no AIS data", result.reason)

    def test_candidate_identity_preserved(self):
        candidate = _candidate(cand_id="probe-1")
        result = compute_ais_evidence(candidate)
        self.assertEqual(result.candidate_id, "probe-1")
        self.assertEqual(result.t_s, T_S)

    def test_explicit_none_source_same_as_default(self):
        result_default = compute_ais_evidence(_candidate())
        result_explicit = compute_ais_evidence(_candidate(), ais_source=None)
        self.assertEqual(result_default.ais_available, result_explicit.ais_available)


class TestConcreteSourceRealPath(unittest.TestCase):
    """TASK AIS-6: `compute_ais_evidence` now implements the real AIS-present path
    (previously deferred/NotImplementedError, per the module's own documented
    precondition -- real AIS data now exists, see AIS-2 through AIS-5)."""

    def test_source_with_no_vessels_anywhere_returns_missing_not_raising(self):
        class EmptySource(AISDataSource):
            def query_vessels_near(self, lat, lon, t, radius_m, time_window_hours):
                return []

        result = compute_ais_evidence(_candidate(), ais_source=EmptySource())
        self.assertTrue(result.ais_available)
        self.assertIsNone(result.ais_score)
        self.assertIsNone(result.distance_to_nearest_track_m)
        self.assertEqual(result.n_vessels_nearby, 0)

    def test_source_with_a_nearby_vessel_produces_a_score(self):
        from iste.phase3_ais import VesselTrackSegment

        class OneVesselSource(AISDataSource):
            def query_vessels_near(self, lat, lon, t, radius_m, time_window_hours):
                return [VesselTrackSegment(vessel_id="1", lat=lat, lon=lon, t=t)]

        result = compute_ais_evidence(_candidate(), ais_source=OneVesselSource())
        self.assertTrue(result.ais_available)
        self.assertIsNotNone(result.ais_score)
        self.assertAlmostEqual(result.distance_to_nearest_track_m, 0.0, places=3)
        self.assertGreater(result.n_vessels_nearby, 0)
        self.assertGreaterEqual(result.ais_score, 0.0)
        self.assertLessEqual(result.ais_score, 1.0001)

    def test_score_independent_of_support_no_support_field_read(self):
        # the candidate/support_score value must have zero influence on ais_score --
        # confirmed structurally: compute_ais_evidence never reads candidate.support_score.
        import inspect
        from iste import phase3_ais
        src = inspect.getsource(phase3_ais.compute_ais_evidence)
        self.assertNotIn("support_score", src)
        self.assertNotIn("candidate.support", src)


class TestRenormalizeWeightedMean(unittest.TestCase):
    def test_all_available_matches_plain_weighted_mean(self):
        result = renormalize_weighted_mean([(1.0, 2.0), (1.0, 4.0), (1.0, 6.0)])
        self.assertAlmostEqual(result, 4.0)

    def test_unequal_weights(self):
        result = renormalize_weighted_mean([(2.0, 10.0), (1.0, 4.0)])
        self.assertAlmostEqual(result, (2.0 * 10.0 + 1.0 * 4.0) / 3.0)

    def test_one_none_component_renormalizes_over_remainder(self):
        with_all = renormalize_weighted_mean([(1.0, 5.0), (1.0, 10.0), (1.0, 30.0)])
        without_middle = renormalize_weighted_mean([(1.0, 5.0), (1.0, None), (1.0, 30.0)])
        self.assertAlmostEqual(without_middle, 17.5)  # mean of remaining {5, 30}
        self.assertNotEqual(with_all, without_middle)

    def test_ais_present_vs_absent_both_well_defined(self):
        # demonstrates the exact requirement: T36's composite must be well-defined both
        # with and without the AIS component present (using a synthetic stand-in value for
        # the "present" case -- never real AIS data, which does not exist in this project).
        base_components_absent = [(1.0, 0.6), (1.0, 0.8), (1.0, 0.4)]
        synthetic_ais_present_value = 0.9  # a synthetic test double, NOT real AIS evidence
        base_components_present = base_components_absent + [(1.0, synthetic_ais_present_value)]

        score_absent = renormalize_weighted_mean(base_components_absent)
        score_present = renormalize_weighted_mean(base_components_present)

        self.assertFalse(np.isnan(score_absent))
        self.assertFalse(np.isnan(score_present))
        self.assertAlmostEqual(score_absent, (0.6 + 0.8 + 0.4) / 3.0)
        self.assertAlmostEqual(score_present, (0.6 + 0.8 + 0.4 + 0.9) / 4.0)

    def test_all_none_returns_nan(self):
        result = renormalize_weighted_mean([(1.0, None), (1.0, None)])
        self.assertTrue(np.isnan(result))

    def test_empty_list_returns_nan(self):
        result = renormalize_weighted_mean([])
        self.assertTrue(np.isnan(result))

    def test_zero_total_weight_returns_nan(self):
        result = renormalize_weighted_mean([(0.0, 5.0), (0.0, 10.0)])
        self.assertTrue(np.isnan(result))

    def test_negative_weight_rejected(self):
        with self.assertRaises(ValueError):
            renormalize_weighted_mean([(-1.0, 5.0)])

    def test_non_finite_weight_rejected(self):
        with self.assertRaises(ValueError):
            renormalize_weighted_mean([(float("inf"), 5.0)])


if __name__ == "__main__":
    unittest.main()
