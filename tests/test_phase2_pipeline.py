"""Focused T26 checks for `iste.phase2_pipeline` (Phase 2 end-to-end integration and
candidate artifact).

Run with:
    python -m unittest tests.test_phase2_pipeline -v
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import unittest

import numpy as np

from iste.candidate_extraction import CandidateExtractionConfig
from iste.candidate_forward_accuracy import aggregate_candidate_forward_accuracy
from iste.candidate_screening import ScreeningConfig
from iste.core_sampling import sample_core_particles
from iste.ensemble_run import run_ensemble
from iste.environment import GriddedFieldSource
from iste.forcing_ensemble import ForcingEnsemble, ForcingEnsembleConfig
from iste.forward_accuracy import ComparisonConfig, compare_predicted_footprints_to_observed
from iste.forward_budget import ForwardBudgetConfig, compute_forward_particle_budget
from iste.forward_propagation import forward_propagate_candidate
from iste.observation import adapt_upstream_particles
from iste.particle_query import query_particle_slice
from iste.phase2_pipeline import (
    CandidatePipelineResult,
    Phase2Result,
    Phase2RunConfig,
    TimestampPipelineResult,
    load_phase2_artifact,
    run_phase2_pipeline,
    save_phase2_artifact,
    serialize_phase2_artifact,
)
from iste.slick_footprint import FootprintGridConfig, build_predicted_slick_footprints
from iste.support_features import SupportFeatureConfig

LAT = np.linspace(15.0, 26.0, 23)
LON = np.linspace(95.0, 106.0, 23)
T_OBS = np.datetime64("2020-01-01T06:00:00", "ns")
N_HOURS = 8
TIME = T_OBS - (N_HOURS - 1) * np.timedelta64(3600, "s") + np.arange(N_HOURS) * np.timedelta64(3600, "s")
DT = 900.0
N_STEPS = 4
T_S_1 = T_OBS - np.timedelta64(900, "s")
T_S_2 = T_OBS - np.timedelta64(1800, "s")

ENSEMBLE_CONFIG = ForcingEnsembleConfig(
    n_members=4, spatial_correlation_km=40.0, correlation_time_seconds=2 * 3600.0,
    sigma_u=0.05, coarse_resolution_deg=0.5, master_seed=55,
)


def _make_ocean() -> GriddedFieldSource:
    uo = np.full((N_HOURS, len(LAT), len(LON)), 0.15)
    vo = np.full((N_HOURS, len(LAT), len(LON)), 0.05)
    return GriddedFieldSource(
        source_path="<synthetic>", time_name="time", time=TIME, lat_name="lat", lat=LAT,
        lon_name="lon", lon=LON, data={"uo": uo, "vo": vo}, fill_threshold=1e19,
    )


def _make_adapted_observation():
    rng = np.random.default_rng(3)
    n_a, n_b = 12, 10
    lat_a = 20.0 + rng.normal(0.0, 0.03, n_a)
    lon_a = 100.0 + rng.normal(0.0, 0.03, n_a)
    lat_b = 22.5 + rng.normal(0.0, 0.03, n_b)
    lon_b = 103.0 + rng.normal(0.0, 0.03, n_b)
    obs_lat = np.concatenate([lat_a, lat_b])
    obs_lon = np.concatenate([lon_a, lon_b])
    records = [
        {"particle_id": f"o{i}", "latitude": obs_lat[i], "longitude": obs_lon[i],
         "observation_pixel_id": f"px-{0 if i < n_a else 1}", "timestamp": T_OBS}
        for i in range(len(obs_lat))
    ]
    return adapt_upstream_particles(records)


def _build_scenario(neighborhood_radius_m=15000.0, core_radius_m=15000.0, master_seed=42, error_threshold=0.95, top_k=None, mode="threshold"):
    """A small, deterministic, two-spatial-cluster, two-historical-timestamp Phase 2
    scenario. `neighborhood_radius_m`/`core_radius_m` control whether the second
    (sparser) cluster ends up with Y > 0 (validated) or Y == 0 (unvalidated) -- used to
    exercise both code paths across different tests."""
    ocean = _make_ocean()
    ensemble = ForcingEnsemble(ocean, ENSEMBLE_CONFIG)
    K = np.full((len(LAT), len(LON)), 1.0)
    adapted = _make_adapted_observation()
    run_records = run_ensemble(
        adapted, forcing_member_ids=[0, 1], n_steps=N_STEPS, dt_seconds=DT,
        ocean_source=ocean, ensemble=ensemble, k_lat=LAT, k_lon=LON, k_field=K, master_seed=99,
    )

    support_lat_grid = np.linspace(18.0, 24.0, 61)
    support_lon_grid = np.linspace(98.0, 105.0, 71)

    if mode == "threshold":
        screening_config = ScreeningConfig(mode="threshold", error_threshold=error_threshold, merge_separation_m=10000.0)
    else:
        screening_config = ScreeningConfig(mode="top_k", top_k=top_k, merge_separation_m=10000.0)

    config = Phase2RunConfig(
        historical_timestamps=(T_S_1, T_S_2),
        t_obs=T_OBS,
        support_query_lat_grid=support_lat_grid,
        support_query_lon_grid=support_lon_grid,
        neighborhood_radius_m=neighborhood_radius_m,
        support_config=SupportFeatureConfig(),
        extraction_config=CandidateExtractionConfig(peak_neighborhood_cells=3, min_separation_m=50000.0, core_radius_m=core_radius_m),
        forward_budget_config=ForwardBudgetConfig(X=8),
        forcing_member_ids=(0, 1),
        forward_dt_seconds=DT,
        footprint_lat_grid=support_lat_grid,
        footprint_lon_grid=support_lon_grid,
        footprint_bandwidth_m=5000.0,
        comparison_config=ComparisonConfig(metric="combined_overlap_fp"),
        screening_config=screening_config,
        master_seed=master_seed,
    )
    return run_records, ocean, ensemble, K, adapted, config


class TestFullPipelineRunsEndToEnd(unittest.TestCase):
    """Requirement 1: full small synthetic pipeline runs end-to-end."""

    def test_pipeline_runs_and_returns_a_phase2_result(self):
        run_records, ocean, ensemble, K, adapted, config = _build_scenario()
        result = run_phase2_pipeline(run_records, ocean, ensemble, LAT, LON, K, adapted, config)
        self.assertIsInstance(result, Phase2Result)
        self.assertEqual(len(result.timestamps), 2)
        self.assertGreater(len(result.decisions), 0)


class TestEveryMajorStageInvoked(unittest.TestCase):
    """Requirement 2: every major T14-T25 stage is actually invoked."""

    def test_result_structure_proves_every_stage_ran(self):
        run_records, ocean, ensemble, K, adapted, config = _build_scenario()
        result = run_phase2_pipeline(run_records, ocean, ensemble, LAT, LON, K, adapted, config)

        self.assertGreater(len(result.timestamps), 0)
        for ts_result in result.timestamps:
            self.assertIsInstance(ts_result, TimestampPipelineResult)
            self.assertGreater(ts_result.slice_.n_particles, 0)  # T14
            self.assertGreater(len(ts_result.index.source_row), 0)  # T15
            self.assertGreater(len(ts_result.support_map), 0)  # T16
            self.assertGreater(len(ts_result.candidates), 0)  # T17
            for c in ts_result.candidates:
                self.assertIsInstance(c, CandidatePipelineResult)
                self.assertIsNotNone(c.budget)  # T18
                self.assertGreaterEqual(c.budget.Y, 0)
                self.assertIsNotNone(c.sampled)  # T19
                self.assertEqual(set(c.forward_runs.runs.keys()), set(config.forcing_member_ids))  # T20/T21
                self.assertEqual(set(c.predicted_footprints.keys()), set(config.forcing_member_ids))  # T22
                self.assertGreater(len(c.member_results), 0)  # T23
                self.assertIsNotNone(c.forward_accuracy)  # T24
                self.assertIsNotNone(c.record)  # T25 input
        self.assertGreater(len(result.decisions), 0)  # T25


class TestMultipleHistoricalTimestampsPreserved(unittest.TestCase):
    """Requirement 3: multiple historical timestamps are preserved."""

    def test_both_configured_timestamps_appear_in_result_and_decisions(self):
        run_records, ocean, ensemble, K, adapted, config = _build_scenario()
        result = run_phase2_pipeline(run_records, ocean, ensemble, LAT, LON, K, adapted, config)
        result_t_s = {ts.t_s for ts in result.timestamps}
        self.assertEqual(result_t_s, {T_S_1, T_S_2})
        decision_t_s = {d.record.t_s for d in result.decisions}
        self.assertEqual(decision_t_s, {T_S_1, T_S_2})


class TestMultipleCandidateModesPreserved(unittest.TestCase):
    """Requirement 4: multiple candidate modes are preserved before/after filtering
    where they survive."""

    def test_two_spatial_clusters_both_survive(self):
        run_records, ocean, ensemble, K, adapted, config = _build_scenario()
        result = run_phase2_pipeline(run_records, ocean, ensemble, LAT, LON, K, adapted, config)
        accepted = [d for d in result.decisions if d.accepted]
        self.assertEqual(len(accepted), 4)  # 2 clusters x 2 timestamps, all validated+accepted
        lons = sorted({round(d.record.x_peak) for d in accepted})
        self.assertEqual(lons, [100, 103])  # both spatial clusters represented


class TestForcingMemberProvenancePreserved(unittest.TestCase):
    """Requirement 5: forcing-member provenance is preserved."""

    def test_forcing_members_validated_matches_configured_set(self):
        run_records, ocean, ensemble, K, adapted, config = _build_scenario()
        result = run_phase2_pipeline(run_records, ocean, ensemble, LAT, LON, K, adapted, config)
        for d in result.decisions:
            if d.record.forward_validation_status == "validated":
                self.assertEqual(set(d.record.forcing_members_validated), set(config.forcing_member_ids))
        for ts_result in result.timestamps:
            for c in ts_result.candidates:
                self.assertEqual(c.forward_runs.forcing_member_ids, config.forcing_member_ids)


class TestReproducibility(unittest.TestCase):
    """Requirement 6: same seed/configuration reproduces the same final artifact."""

    def _decision_tuples(self, decisions):
        out = []
        for d in decisions:
            error = d.record.candidate_forward_error
            out.append((
                d.record.candidate_id, d.accepted, d.rank,
                "nan" if math.isnan(error) else error,
                d.record.sampling_seed,
            ))
        return out

    def test_two_runs_with_identical_inputs_produce_identical_decisions(self):
        run_records, ocean, ensemble, K, adapted, config = _build_scenario()
        result1 = run_phase2_pipeline(run_records, ocean, ensemble, LAT, LON, K, adapted, config)
        result2 = run_phase2_pipeline(run_records, ocean, ensemble, LAT, LON, K, adapted, config)
        self.assertEqual(self._decision_tuples(result1.decisions), self._decision_tuples(result2.decisions))

    def test_fully_independent_setup_reproduces_identically(self):
        # rebuild ocean/ensemble/run_records from scratch (no shared objects at all)
        run_records1, ocean1, ensemble1, K1, adapted1, config1 = _build_scenario()
        run_records2, ocean2, ensemble2, K2, adapted2, config2 = _build_scenario()
        result1 = run_phase2_pipeline(run_records1, ocean1, ensemble1, LAT, LON, K1, adapted1, config1)
        result2 = run_phase2_pipeline(run_records2, ocean2, ensemble2, LAT, LON, K2, adapted2, config2)
        self.assertEqual(self._decision_tuples(result1.decisions), self._decision_tuples(result2.decisions))


class TestSerializationRoundTrip(unittest.TestCase):
    """Requirement 7: serialization/reload preserves all required fields."""

    def test_all_required_fields_survive_round_trip(self):
        run_records, ocean, ensemble, K, adapted, config = _build_scenario()
        result = run_phase2_pipeline(run_records, ocean, ensemble, LAT, LON, K, adapted, config)

        required_fields = {
            "candidate_id", "x_s", "y_s", "t_s", "support_score", "density_score",
            "weighted_support_score", "pixel_diversity_score", "core_definition",
            "core_particle_count_Y", "forward_particle_count_N_forward", "sampling_seed",
            "forcing_members_validated", "forward_error_by_member", "forward_consistency_by_member",
            "candidate_forward_error", "candidate_forward_consistency", "accepted", "rank",
        }

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "phase2_artifact.json")
            save_phase2_artifact(result.decisions, config, path)
            reloaded = load_phase2_artifact(path)

        self.assertEqual(len(reloaded["candidates"]), len(result.decisions))
        by_id_original = {d.record.candidate_id: d for d in result.decisions}
        for entry in reloaded["candidates"]:
            self.assertTrue(required_fields.issubset(entry.keys()))
            d = by_id_original[entry["candidate_id"]]
            self.assertEqual(entry["x_s"], d.record.x_peak)
            self.assertEqual(entry["y_s"], d.record.y_peak)
            self.assertEqual(entry["core_particle_count_Y"], d.record.core_particle_count_Y)
            self.assertEqual(entry["forward_particle_count_N_forward"], d.record.forward_particle_count_N_forward)
            self.assertEqual(entry["sampling_seed"], d.record.sampling_seed)
            self.assertEqual(set(entry["forcing_members_validated"]), set(d.record.forcing_members_validated))
            self.assertEqual(entry["accepted"], d.accepted)
            self.assertEqual(entry["rank"], d.rank)
            self.assertEqual(entry["core_definition"]["radius_m"], d.record.core_definition.radius_m)
            if math.isnan(d.record.candidate_forward_error):
                self.assertTrue(math.isnan(entry["candidate_forward_error"]))
            else:
                self.assertAlmostEqual(entry["candidate_forward_error"], d.record.candidate_forward_error)

    def test_reload_is_plain_json_readable_independently(self):
        run_records, ocean, ensemble, K, adapted, config = _build_scenario()
        result = run_phase2_pipeline(run_records, ocean, ensemble, LAT, LON, K, adapted, config)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "phase2_artifact.json")
            save_phase2_artifact(result.decisions, config, path)
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)  # plain stdlib json, not this module's loader
        self.assertIn("candidates", raw)
        self.assertIn("config", raw)


class TestAcceptedCandidatesHaveForwardValidationProvenance(unittest.TestCase):
    """Requirement 8: no accepted candidate lacks forward-validation provenance."""

    def test_every_accepted_candidate_has_complete_forward_provenance(self):
        run_records, ocean, ensemble, K, adapted, config = _build_scenario()
        result = run_phase2_pipeline(run_records, ocean, ensemble, LAT, LON, K, adapted, config)
        accepted = [d for d in result.decisions if d.accepted]
        self.assertGreater(len(accepted), 0)
        for d in accepted:
            r = d.record
            self.assertEqual(r.forward_validation_status, "validated")
            self.assertGreater(len(r.forcing_members_validated), 0)
            self.assertFalse(math.isnan(r.candidate_forward_error))
            self.assertFalse(math.isnan(r.candidate_forward_consistency))
            self.assertGreater(len(r.forward_error_by_member), 0)
            self.assertIsNotNone(d.rank)


class TestInvalidCandidatesNeverFabricated(unittest.TestCase):
    """Requirement 9: invalid/unvalidated candidates are never given fabricated
    forward scores."""

    def test_unvalidated_candidate_has_nan_scores_and_no_rank(self):
        # tiny core_radius_m (much smaller than the peak-detection neighborhood_radius_m)
        # -> every candidate core ends up with zero particles (Y=0), due to the grid-
        # quantized peak location not exactly coinciding with any actual particle.
        run_records, ocean, ensemble, K, adapted, config = _build_scenario(neighborhood_radius_m=15000.0, core_radius_m=200.0)
        result = run_phase2_pipeline(run_records, ocean, ensemble, LAT, LON, K, adapted, config)
        unvalidated = [d for d in result.decisions if d.record.forward_validation_status == "unvalidated"]
        self.assertGreater(len(unvalidated), 0)
        for d in unvalidated:
            self.assertFalse(d.accepted)
            self.assertIsNone(d.rank)
            self.assertTrue(math.isnan(d.record.candidate_forward_error))
            self.assertTrue(math.isnan(d.record.candidate_forward_consistency))
            self.assertNotEqual(d.record.candidate_forward_error, 0.0)  # never silently zero
            self.assertIn("unvalidated", d.rejection_reason)


class TestIntegratedArtifactMatchesIndividualStages(unittest.TestCase):
    """Requirement 10: the integrated artifact matches the individually validated
    T14-T25 behavior -- manually replays one candidate's chain via the individual
    T14-T25 functions directly and confirms it matches the pipeline's own result."""

    def test_manual_replay_matches_pipeline_output_for_one_candidate(self):
        run_records, ocean, ensemble, K, adapted, config = _build_scenario()
        result = run_phase2_pipeline(run_records, ocean, ensemble, LAT, LON, K, adapted, config)

        ts_result = result.timestamps[0]
        c = ts_result.candidates[0]

        # T14/T15 already captured on ts_result; re-derive independently to confirm agreement
        slice_2 = query_particle_slice(run_records, ts_result.t_s)
        self.assertEqual(slice_2.n_particles, ts_result.slice_.n_particles)

        # T18
        budget2 = compute_forward_particle_budget(c.candidate, ts_result.index, config.forward_budget_config)
        self.assertEqual(budget2.Y, c.budget.Y)
        self.assertEqual(budget2.N_forward, c.budget.N_forward)

        # T19 (same seed -> identical selection)
        sampled2 = sample_core_particles(c.candidate, ts_result.index, config.forward_budget_config, seed=c.sampling_seed)
        np.testing.assert_array_equal(sampled2.particle_id, c.sampled.particle_id)
        np.testing.assert_array_equal(sampled2.lat, c.sampled.lat)

        # T20/T21 (same seeds -> identical trajectories)
        forward_runs2 = forward_propagate_candidate(
            sampled2, config.t_obs, config.forward_dt_seconds, ocean, ensemble,
            forcing_member_ids=list(config.forcing_member_ids),
            seeds=[c.member_seeds[m] for m in config.forcing_member_ids],
            k_lat=LAT, k_lon=LON, k_field=K,
        )
        for member_id in config.forcing_member_ids:
            np.testing.assert_array_equal(forward_runs2[member_id].lat, c.forward_runs[member_id].lat)
            np.testing.assert_array_equal(forward_runs2[member_id].lon, c.forward_runs[member_id].lon)

        # T22
        footprint_config = FootprintGridConfig(
            lat_grid=config.footprint_lat_grid, lon_grid=config.footprint_lon_grid, bandwidth_m=config.footprint_bandwidth_m,
        )
        predicted2 = build_predicted_slick_footprints(forward_runs2.runs, footprint_config)
        for member_id in config.forcing_member_ids:
            np.testing.assert_array_equal(predicted2[member_id].density, c.predicted_footprints[member_id].density)

        # T23
        member_results2 = compare_predicted_footprints_to_observed(predicted2, result.observed_footprint, config.comparison_config)
        for member_id in config.forcing_member_ids:
            self.assertEqual(member_results2[member_id].error, c.member_results[member_id].error)
            self.assertEqual(member_results2[member_id].consistency, c.member_results[member_id].consistency)

        # T24
        forward_accuracy2 = aggregate_candidate_forward_accuracy(member_results2)
        self.assertEqual(forward_accuracy2.candidate_forward_error, c.forward_accuracy.candidate_forward_error)
        self.assertEqual(forward_accuracy2.candidate_forward_consistency, c.forward_accuracy.candidate_forward_consistency)


class TestPhase2RunConfigValidation(unittest.TestCase):
    def test_rejects_empty_historical_timestamps(self):
        _, _, _, _, _, config = _build_scenario()
        with self.assertRaises(ValueError):
            Phase2RunConfig(
                historical_timestamps=(), t_obs=config.t_obs,
                support_query_lat_grid=config.support_query_lat_grid, support_query_lon_grid=config.support_query_lon_grid,
                neighborhood_radius_m=config.neighborhood_radius_m, support_config=config.support_config,
                extraction_config=config.extraction_config, forward_budget_config=config.forward_budget_config,
                forcing_member_ids=config.forcing_member_ids, forward_dt_seconds=config.forward_dt_seconds,
                footprint_lat_grid=config.footprint_lat_grid, footprint_lon_grid=config.footprint_lon_grid,
                footprint_bandwidth_m=config.footprint_bandwidth_m, comparison_config=config.comparison_config,
                screening_config=config.screening_config, master_seed=config.master_seed,
            )

    def test_rejects_empty_forcing_member_ids(self):
        _, _, _, _, _, config = _build_scenario()
        with self.assertRaises(ValueError):
            Phase2RunConfig(
                historical_timestamps=config.historical_timestamps, t_obs=config.t_obs,
                support_query_lat_grid=config.support_query_lat_grid, support_query_lon_grid=config.support_query_lon_grid,
                neighborhood_radius_m=config.neighborhood_radius_m, support_config=config.support_config,
                extraction_config=config.extraction_config, forward_budget_config=config.forward_budget_config,
                forcing_member_ids=(), forward_dt_seconds=config.forward_dt_seconds,
                footprint_lat_grid=config.footprint_lat_grid, footprint_lon_grid=config.footprint_lon_grid,
                footprint_bandwidth_m=config.footprint_bandwidth_m, comparison_config=config.comparison_config,
                screening_config=config.screening_config, master_seed=config.master_seed,
            )


if __name__ == "__main__":
    unittest.main()
