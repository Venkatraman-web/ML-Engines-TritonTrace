"""Focused T36 checks for `iste.phase3_composite` (Phase 3 composite multi-feature source
score).

Run with:
    python -m unittest tests.test_phase3_composite -v
"""

from __future__ import annotations

import inspect
import unittest

import numpy as np

from iste.candidate_extraction import Candidate, CoreRegion
from iste.ensemble_run import EnsembleRunRecords
from iste.phase3_ais import AISEvidence
from iste.phase3_composite import (
    DEFAULT_WEIGHTS,
    DISPERSION_EMPHASIS_WEIGHTS,
    PERSISTENCE_EMPHASIS_WEIGHTS,
    CompositeWeights,
    compute_source_score,
    compute_source_scores_for_candidates,
)
from iste.phase3_convergence import ConvergenceScore
from iste.phase3_dispersion import DispersionAwareScore
from iste.phase3_initial_bias import InitialBiasInfluence
from iste.phase3_persistence import PersistenceTrack
from iste.phase3_screening import ScreeningScore

T_S = np.datetime64("2020-01-01T00:00:00", "ns")
T_S2 = T_S - np.timedelta64(50, "h")


def _candidate(cand_id="c0", t_s=T_S, support_score=100.0) -> Candidate:
    return Candidate(
        candidate_id=cand_id, x_peak=35.0, y_peak=34.0, t_s=np.datetime64(t_s, "ns"),
        core_region=CoreRegion(center_lat=34.0, center_lon=35.0, radius_m=1000.0),
        support_score=support_score, density_score=support_score * 0.5,
        weighted_support_score=support_score * 0.3, pixel_diversity_score=support_score * 0.2,
        peak_support_features=None,
    )


def _screening(cand_id="c0", t_s=T_S, screening_score=0.5, raw_support=100.0) -> ScreeningScore:
    return ScreeningScore(
        candidate_id=cand_id, t_s=np.datetime64(t_s, "ns"), raw_support_score=raw_support,
        reference_value=raw_support / screening_score if screening_score else float("nan"),
        screening_score=screening_score, density_score=raw_support * 0.5,
        weighted_support_score=raw_support * 0.3, pixel_diversity_score=raw_support * 0.2,
    )


def _track(cand_ids, t_s=T_S, count=None) -> PersistenceTrack:
    n = count if count is not None else len(cand_ids)
    tr = PersistenceTrack(track_id="track0000")
    for i, cid in enumerate(cand_ids):
        tr.candidate_ids.append(cid)
        tr.t_s_list.append(np.datetime64(t_s, "ns") - np.timedelta64(i, "h"))
        tr.support_scores.append(100.0)
        tr.lat_list.append(34.0)
        tr.lon_list.append(35.0)
    return tr


def _convergence(cand_id="c0", t_s=T_S, score=0.5) -> ConvergenceScore:
    return ConvergenceScore(
        candidate_id=cand_id, t_s=np.datetime64(t_s, "ns"), particle_count=10,
        total_valid_particles_at_t=100, particle_fraction=0.1, pixel_diversity=5,
        total_distinct_pixels_at_t=10, pixel_diversity_fraction=0.5, distinct_forcing_members=3,
        total_distinct_members_at_t=6, member_diversity_fraction=0.5, within_core_spread_m=500.0,
        core_radius_m=1000.0, concentration_score=0.6, temporal_consistency=0.8,
        convergence_score=score,
    )


def _dispersion(cand_id="c0", t_s=T_S, score=0.5) -> DispersionAwareScore:
    return DispersionAwareScore(
        candidate_id=cand_id, t_s=np.datetime64(t_s, "ns"), particle_count_actual=10.0,
        n_total_valid_at_t=100, covariance_state="full_rank", sigma_iso_m=500.0,
        neighborhood_radius_m=1000.0, p_within_r=0.5, expected_density=20.0,
        dispersion_aware_score=score,
    )


def _initial_bias(t_s=T_S, influence=0.3) -> InitialBiasInfluence:
    return InitialBiasInfluence(
        candidate_lat=34.0, candidate_lon=35.0, t_s=np.datetime64(t_s, "ns"),
        covariance_state="full_rank", mahalanobis_distance=1.0, k=1.0, influence=influence,
    )


def _ais(cand_id="c0", t_s=T_S, available=False, score=None) -> AISEvidence:
    return AISEvidence(
        candidate_id=cand_id, t_s=np.datetime64(t_s, "ns"), ais_available=available,
        distance_to_nearest_track_m=None, n_vessels_nearby=None, presence_duration_hours=None,
        trajectory_compatibility_score=None, ais_score=score,
        reason="AIS unavailable: no AIS data source is configured or exists in this project",
    )


def _full_call(cand_id="c0", t_s=T_S, support=0.5, persistence_continuity_track=None,
               convergence=0.5, dispersion=0.5, initial_bias=0.3, ais_score=None, weights=DEFAULT_WEIGHTS):
    cand = _candidate(cand_id, t_s)
    track = persistence_continuity_track if persistence_continuity_track is not None else _track([cand_id], t_s)
    return compute_source_score(
        cand, screening_score=_screening(cand_id, t_s, support), persistence_track=track,
        convergence_score=_convergence(cand_id, t_s, convergence),
        dispersion_score=_dispersion(cand_id, t_s, dispersion),
        initial_bias=_initial_bias(t_s, initial_bias),
        ais_evidence=_ais(cand_id, t_s, available=ais_score is not None, score=ais_score),
        weights=weights,
    )


class TestDefaultConfiguration(unittest.TestCase):
    def test_default_weight_values(self):
        self.assertEqual(DEFAULT_WEIGHTS.w_support, 0.35)
        self.assertEqual(DEFAULT_WEIGHTS.w_persistence, 0.20)
        self.assertEqual(DEFAULT_WEIGHTS.w_convergence, 0.10)
        self.assertEqual(DEFAULT_WEIGHTS.w_dispersion, 0.15)
        self.assertEqual(DEFAULT_WEIGHTS.w_ais, 0.20)
        self.assertEqual(DEFAULT_WEIGHTS.w_initial, 0.0)
        self.assertEqual(DEFAULT_WEIGHTS.w_persistence_count, 0.0)

    def test_default_weights_sum_to_one(self):
        total = sum(DEFAULT_WEIGHTS.as_dict().values())
        self.assertAlmostEqual(total, 1.0, places=9)


class TestMonotonicity(unittest.TestCase):
    def test_support_monotonic(self):
        low = _full_call(support=0.3)
        high = _full_call(support=0.8)
        self.assertGreater(high.source_score, low.source_score)

    def test_persistence_monotonic(self):
        low = _full_call(persistence_continuity_track=_track(["c0", "c1"], count=2))
        high = _full_call(persistence_continuity_track=_track(["c0"], count=1))
        # a track containing only c0 with 1 member has continuity 1.0; a 2-candidate,
        # 2-hour-apart track also has continuity 1.0 -- use explicit continuity via gap
        tr_broken = PersistenceTrack(track_id="broken")
        tr_broken.candidate_ids = ["c0", "c1"]
        tr_broken.t_s_list = [T_S, T_S - np.timedelta64(2, "h")]  # 1-hour gap skipped -> continuity < 1
        tr_broken.support_scores = [100.0, 100.0]
        tr_broken.lat_list = [34.0, 34.0]
        tr_broken.lon_list = [35.0, 35.0]
        tr_full = _track(["c0"], count=1)
        result_broken = _full_call(persistence_continuity_track=tr_broken)
        result_full = _full_call(persistence_continuity_track=tr_full)
        self.assertLess(tr_broken.temporal_continuity(), tr_full.temporal_continuity())
        self.assertLess(result_broken.source_score, result_full.source_score)

    def test_convergence_monotonic(self):
        low = _full_call(convergence=0.2)
        high = _full_call(convergence=0.9)
        self.assertGreater(high.source_score, low.source_score)

    def test_dispersion_monotonic(self):
        low = _full_call(dispersion=0.2)
        high = _full_call(dispersion=1.5)
        self.assertGreater(high.source_score, low.source_score)


class TestActiveWeightNormalization(unittest.TestCase):
    def test_score_matches_hand_computed_weighted_mean(self):
        result = _full_call(support=0.5, convergence=0.5, dispersion=0.5, ais_score=None)
        # AIS unavailable -> active = support, persistence, convergence, dispersion, initial(w=0), persistence_count(w=0)
        w_sum = DEFAULT_WEIGHTS.w_support + DEFAULT_WEIGHTS.w_persistence + DEFAULT_WEIGHTS.w_convergence + DEFAULT_WEIGHTS.w_dispersion
        expected = (
            DEFAULT_WEIGHTS.w_support * 0.5 + DEFAULT_WEIGHTS.w_persistence * 1.0
            + DEFAULT_WEIGHTS.w_convergence * 0.5 + DEFAULT_WEIGHTS.w_dispersion * 0.5
        ) / w_sum
        self.assertAlmostEqual(result.source_score, expected, places=9)

    def test_effective_weights_sum_to_one_when_ais_unavailable(self):
        result = _full_call(ais_score=None)
        self.assertAlmostEqual(sum(result.effective_weights.values()), 1.0, places=9)

    def test_effective_weight_zero_for_zero_configured_weight(self):
        result = _full_call()
        self.assertEqual(result.effective_weights["initial_bias"], 0.0)
        self.assertEqual(result.effective_weights["persistence_count"], 0.0)


class TestMissingAIS(unittest.TestCase):
    def test_score_well_defined_when_ais_unavailable(self):
        result = _full_call(ais_score=None)
        self.assertFalse(np.isnan(result.source_score))
        self.assertIsNone(result.ais_evidence)

    def test_effective_ais_weight_is_zero_when_unavailable(self):
        result = _full_call(ais_score=None)
        self.assertEqual(result.effective_weights["ais"], 0.0)
        # configured weight is still reported, unmodified
        self.assertEqual(result.configured_weights["ais"], DEFAULT_WEIGHTS.w_ais)

    def test_ais_absence_never_lowers_score_relative_to_equal_candidate(self):
        # two otherwise-identical candidates: AIS absent for both -> identical score
        a = _full_call(cand_id="a", ais_score=None)
        b = _full_call(cand_id="b", ais_score=None)
        self.assertAlmostEqual(a.source_score, b.source_score, places=9)

    def test_score_defined_when_ais_available_with_synthetic_value(self):
        # synthetic test double only (no real AIS data exists in this project)
        result = _full_call(ais_score=0.5)
        self.assertFalse(np.isnan(result.source_score))
        self.assertAlmostEqual(result.effective_weights["ais"], DEFAULT_WEIGHTS.w_ais, places=9)


class TestT29T34Duplication(unittest.TestCase):
    def test_no_separate_weight_field_for_t29(self):
        fields = CompositeWeights.__dataclass_fields__.keys()
        support_related = [f for f in fields if "support" in f]
        self.assertEqual(support_related, ["w_support"])  # exactly one axis

    def test_support_score_difference_enters_exactly_once(self):
        low = _full_call(support=0.4)
        high = _full_call(support=0.9)
        w_eff_support = low.effective_weights["support"]
        expected_diff = w_eff_support * (0.9 - 0.4)
        self.assertAlmostEqual(high.source_score - low.source_score, expected_diff, places=9)


class TestRawKDEDomination(unittest.TestCase):
    def test_high_support_weak_other_evidence_does_not_automatically_dominate(self):
        dominant_by_support_only = _full_call(
            cand_id="big", support=1.5, convergence=0.25, dispersion=0.05,
        )
        balanced = _full_call(
            cand_id="balanced", support=0.6, convergence=0.75, dispersion=1.2,
        )
        # balanced candidate's composite must not be trivially dominated by raw support alone
        self.assertGreater(balanced.source_score, dominant_by_support_only.source_score * 0.5)
        # and the huge support candidate must not automatically win just from support magnitude
        self.assertLess(
            dominant_by_support_only.source_score - balanced.source_score,
            dominant_by_support_only.support_evidence - balanced.support_evidence,
        )


class TestPersistenceEvidence(unittest.TestCase):
    def test_stronger_persistence_ranks_higher_holding_others_equal(self):
        tr_low = PersistenceTrack(track_id="low")
        tr_low.candidate_ids, tr_low.t_s_list = ["c0", "c1"], [T_S, T_S - np.timedelta64(2, "h")]
        tr_low.support_scores, tr_low.lat_list, tr_low.lon_list = [100.0, 100.0], [34.0, 34.0], [35.0, 35.0]
        tr_high = _track(["c0"], count=1)
        low = _full_call(persistence_continuity_track=tr_low)
        high = _full_call(persistence_continuity_track=tr_high)
        self.assertGreater(high.source_score, low.source_score)


class TestNoAgeCorrection(unittest.TestCase):
    def test_no_age_or_t_obs_parameter_in_signature(self):
        for fn in (compute_source_score, compute_source_scores_for_candidates):
            sig = inspect.signature(fn)
            self.assertNotIn("age", sig.parameters)
            self.assertNotIn("age_hours", sig.parameters)
            self.assertNotIn("t_obs", sig.parameters)

    def test_identical_evidence_different_timestamps_gives_identical_score(self):
        far_t_s = T_S + np.timedelta64(2000, "h")
        result_near = _full_call(cand_id="near", t_s=T_S)
        result_far = _full_call(cand_id="far", t_s=far_t_s)
        self.assertAlmostEqual(result_near.source_score, result_far.source_score, places=9)


class TestT30ZeroWeightAndSignHandling(unittest.TestCase):
    def test_default_weight_is_zero(self):
        self.assertEqual(DEFAULT_WEIGHTS.w_initial, 0.0)

    def test_initial_bias_value_has_no_effect_under_default_weight(self):
        low = _full_call(initial_bias=0.01)
        high = _full_call(initial_bias=0.99)
        self.assertAlmostEqual(low.source_score, high.source_score, places=9)

    def test_initial_bias_still_reported_despite_zero_weight(self):
        result = _full_call(initial_bias=0.42)
        self.assertAlmostEqual(result.initial_bias_evidence, 0.42, places=9)

    def test_nonzero_positive_weight_does_affect_score(self):
        weights = CompositeWeights(w_initial=0.5)
        low = _full_call(initial_bias=0.01, weights=weights)
        high = _full_call(initial_bias=0.99, weights=weights)
        self.assertNotAlmostEqual(low.source_score, high.source_score, places=6)

    def test_negative_weight_rejected(self):
        with self.assertRaises(ValueError):
            CompositeWeights(w_initial=-0.1)


class TestSensitivityConfigurations(unittest.TestCase):
    def test_all_three_sum_to_one(self):
        for weights in (DEFAULT_WEIGHTS, PERSISTENCE_EMPHASIS_WEIGHTS, DISPERSION_EMPHASIS_WEIGHTS):
            self.assertAlmostEqual(sum(weights.as_dict().values()), 1.0, places=9)

    def test_persistence_emphasis_increases_persistence_weight(self):
        self.assertGreater(PERSISTENCE_EMPHASIS_WEIGHTS.w_persistence, DEFAULT_WEIGHTS.w_persistence)

    def test_dispersion_emphasis_increases_dispersion_weight(self):
        self.assertGreater(DISPERSION_EMPHASIS_WEIGHTS.w_dispersion, DEFAULT_WEIGHTS.w_dispersion)

    def test_all_keep_t30_and_persistence_count_at_zero(self):
        for weights in (DEFAULT_WEIGHTS, PERSISTENCE_EMPHASIS_WEIGHTS, DISPERSION_EMPHASIS_WEIGHTS):
            self.assertEqual(weights.w_initial, 0.0)
            self.assertEqual(weights.w_persistence_count, 0.0)


class TestNaNHandling(unittest.TestCase):
    def test_nan_initial_bias_does_not_poison_score_at_zero_weight(self):
        result = _full_call(initial_bias=float("nan"))
        self.assertFalse(np.isnan(result.source_score))
        self.assertTrue(np.isnan(result.initial_bias_evidence))  # still reported raw

    def test_nan_dispersion_with_nonzero_weight_excluded_not_poisoning(self):
        cand = _candidate("c0", T_S)
        result = compute_source_score(
            cand, screening_score=_screening("c0", T_S, 0.5), persistence_track=_track(["c0"]),
            convergence_score=_convergence("c0", T_S, 0.5),
            dispersion_score=_dispersion("c0", T_S, float("nan")),
            initial_bias=_initial_bias(T_S, 0.3), ais_evidence=_ais("c0", T_S),
            weights=DEFAULT_WEIGHTS,
        )
        self.assertFalse(np.isnan(result.source_score))
        self.assertEqual(result.effective_weights["dispersion"], 0.0)


class TestCandidateIdentityValidation(unittest.TestCase):
    def test_mismatched_candidate_id_raises(self):
        cand = _candidate("c0", T_S)
        with self.assertRaises(ValueError):
            compute_source_score(
                cand, screening_score=_screening("WRONG", T_S), persistence_track=_track(["c0"]),
                convergence_score=_convergence("c0", T_S), dispersion_score=_dispersion("c0", T_S),
                initial_bias=_initial_bias(T_S), ais_evidence=_ais("c0", T_S),
            )

    def test_track_missing_candidate_raises(self):
        cand = _candidate("c0", T_S)
        with self.assertRaises(ValueError):
            compute_source_score(
                cand, screening_score=_screening("c0", T_S), persistence_track=_track(["other"]),
                convergence_score=_convergence("c0", T_S), dispersion_score=_dispersion("c0", T_S),
                initial_bias=_initial_bias(T_S), ais_evidence=_ais("c0", T_S),
            )


def _stage3_records(lat, lon, t_s=T_S, n_pixels=3, n_members=6) -> EnsembleRunRecords:
    n = len(lat)
    return EnsembleRunRecords(
        particle_id=np.array([f"p{i}" for i in range(n)], dtype=object),
        forcing_member_id=np.array([i % n_members for i in range(n)], dtype=np.int64),
        observation_pixel_id=np.array([f"px-{i % n_pixels}" for i in range(n)], dtype=object),
        timestamp=np.full(n, t_s, dtype="datetime64[ns]"),
        lat=np.asarray(lat, dtype=np.float64), lon=np.asarray(lon, dtype=np.float64),
        log_weight=np.zeros(n, dtype=np.float64), valid=np.ones(n, dtype=bool),
    )


class TestBatchWiring(unittest.TestCase):
    def test_batch_end_to_end_on_synthetic_stage3_data(self):
        rng = np.random.default_rng(0)
        n = 60
        lat = 10.0 + rng.normal(0, 0.001, n)
        lon = 100.0 + rng.normal(0, 0.001, n)
        records = _stage3_records(lat, lon)

        from iste.candidate_extraction import CandidateExtractionConfig, find_candidate_peaks
        from iste.kdtree_neighborhood import build_particle_neighborhood_index
        from iste.particle_query import query_particle_slice
        from iste.support_features import SupportFeatureConfig, compute_support_map

        lat_grid = np.linspace(9.995, 10.005, 11)
        lon_grid = np.linspace(99.995, 100.005, 11)
        slice_ = query_particle_slice(records, T_S)
        index = build_particle_neighborhood_index(slice_)
        lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
        support_map = compute_support_map(index, lat_mesh.ravel(), lon_mesh.ravel(), radius_m=500.0, config=SupportFeatureConfig())
        candidates = find_candidate_peaks(
            support_map, lat_grid, lon_grid, T_S,
            config=CandidateExtractionConfig(peak_neighborhood_cells=3, min_separation_m=1.0, core_radius_m=500.0, min_support=0.0),
        )
        self.assertGreater(len(candidates), 0)
        support_grids_by_t_s = {
            T_S: np.array([sf.support for sf in support_map]).reshape(len(lat_grid), len(lon_grid))
        }
        results = compute_source_scores_for_candidates(
            records, candidates, support_grids_by_t_s, neighborhood_radius_m=500.0,
        )
        self.assertEqual(len(results), len(candidates))
        for r in results:
            self.assertFalse(np.isnan(r.source_score))
            self.assertAlmostEqual(sum(r.effective_weights.values()), 1.0, places=6)

    def test_empty_candidate_list(self):
        records = _stage3_records([10.0], [100.0])
        self.assertEqual(
            compute_source_scores_for_candidates(records, [], {}, neighborhood_radius_m=500.0), []
        )


if __name__ == "__main__":
    unittest.main()
