"""Focused T21 checks for `iste.forward_propagation` (per-candidate, per-forcing-member
forward propagation orchestration).

Run with:
    python -m unittest tests.test_forward_propagation -v
"""

from __future__ import annotations

import unittest

import numpy as np

from iste.core_sampling import SampledCoreParticles
from iste.environment import GriddedFieldSource
from iste.forcing_ensemble import ForcingEnsemble, ForcingEnsembleConfig
from iste.forward_propagation import (
    ForwardCandidateMemberRun,
    ForwardCandidateRuns,
    forward_propagate_candidate,
    forward_propagate_candidate_member,
)

NAT = np.datetime64("NaT", "ns")
LAT = np.linspace(10.0, 30.0, 9)
LON = np.linspace(100.0, 120.0, 11)
T_S = np.datetime64("2020-01-01T00:00", "ns")
N_HOURS = 12
TIME = T_S + np.arange(N_HOURS) * np.timedelta64(3600, "s")
T_OBS = T_S + np.timedelta64(3, "h")
DT = 900.0  # 15 minutes

ZERO_K = np.zeros((len(LAT), len(LON)))

TEST_ENSEMBLE_CONFIG = ForcingEnsembleConfig(
    n_members=6, spatial_correlation_km=60.0, correlation_time_seconds=2 * 3600.0,
    sigma_u=0.05, coarse_resolution_deg=0.5, master_seed=321,
)


def make_ocean(uo_val: float = 0.3, vo_val: float = -0.1) -> GriddedFieldSource:
    uo = np.full((N_HOURS, len(LAT), len(LON)), uo_val)
    vo = np.full((N_HOURS, len(LAT), len(LON)), vo_val)
    return GriddedFieldSource(
        source_path="<synthetic>", time_name="time", time=TIME, lat_name="lat", lat=LAT,
        lon_name="lon", lon=LON, data={"uo": uo, "vo": vo}, fill_threshold=1e19,
    )


def make_sampled(
    n=4, candidate_id="cand-A", t_s=T_S, lat0=20.0, lon0=110.0, spread=0.02,
    original_forcing_member_id=None, valid=None,
) -> SampledCoreParticles:
    rng = np.random.default_rng(0)
    lat = lat0 + rng.normal(0.0, spread, n)
    lon = lon0 + rng.normal(0.0, spread, n)
    particle_id = np.array([f"p{i}" for i in range(n)], dtype=object)
    observation_pixel_id = np.array([f"px-{i % 2}" for i in range(n)], dtype=object)
    forcing_member_id = (
        np.array(original_forcing_member_id, dtype=np.int64)
        if original_forcing_member_id is not None
        else np.zeros(n, dtype=np.int64)
    )
    log_weight = rng.normal(0.0, 0.5, n)
    valid_arr = np.array(valid, dtype=bool) if valid is not None else np.ones(n, dtype=bool)
    return SampledCoreParticles(
        candidate_id=candidate_id, t_s=t_s, X=n, Y=n, N_forward=n,
        forward_validatable=True, resampled=False, sampling_seed=0,
        particle_id=particle_id, observation_pixel_id=observation_pixel_id,
        forcing_member_id=forcing_member_id, lat=lat, lon=lon, log_weight=log_weight,
        valid=valid_arr,
        death_time=np.full(n, NAT, dtype="datetime64[ns]"),
        death_lat=np.full(n, np.nan), death_lon=np.full(n, np.nan),
        death_reason=np.array([""] * n, dtype=object),
    )


class TestReachesExactlyTObs(unittest.TestCase):
    """Requirement 1: one candidate/one member reaches exactly t_obs."""

    def test_final_timestamp_equals_t_obs(self):
        ocean = make_ocean()
        ensemble = ForcingEnsemble(ocean, TEST_ENSEMBLE_CONFIG)
        sampled = make_sampled()
        result = forward_propagate_candidate(
            sampled, T_OBS, DT, ocean, ensemble, forcing_member_ids=[0], seeds=[1],
            k_lat=LAT, k_lon=LON, k_field=ZERO_K,
        )
        run = result[0]
        self.assertEqual(run.timestamps[-1], T_OBS)
        self.assertEqual(run.t_obs, T_OBS)


class TestMultipleForcingMembersSeparateResults(unittest.TestCase):
    """Requirement 2: multiple forcing members produce separate forward results."""

    def test_each_member_gets_its_own_run_with_correct_id(self):
        ocean = make_ocean()
        ensemble = ForcingEnsemble(ocean, TEST_ENSEMBLE_CONFIG)
        sampled = make_sampled()
        result = forward_propagate_candidate(
            sampled, T_OBS, DT, ocean, ensemble, forcing_member_ids=[0, 1, 2], seeds=[1, 2, 3],
            k_lat=LAT, k_lon=LON, k_field=ZERO_K,
        )
        self.assertEqual(set(result.runs.keys()), {0, 1, 2})
        for member_id in (0, 1, 2):
            self.assertEqual(result[member_id].forward_forcing_member_id, member_id)

    def test_different_members_generally_give_different_final_positions(self):
        ocean = make_ocean()
        ensemble = ForcingEnsemble(ocean, TEST_ENSEMBLE_CONFIG)
        sampled = make_sampled(n=6, spread=0.0)  # co-located particles -> isolates member effect
        result = forward_propagate_candidate(
            sampled, T_OBS, DT, ocean, ensemble, forcing_member_ids=[0, 1], seeds=[9, 9],
            k_lat=LAT, k_lon=LON, k_field=ZERO_K,
        )
        # different members' coherent perturbations differ -> final positions differ
        self.assertFalse(np.allclose(result[0].final_lat, result[1].final_lat) and
                          np.allclose(result[0].final_lon, result[1].final_lon))


class TestForcingMemberCoherence(unittest.TestCase):
    """Requirement 3: forcing-member coherence preserved across the full run -- every
    particle under one member's run sees the IDENTICAL perturbed field object for every
    checkpoint (never rebuilt/redrawn mid-run)."""

    def test_perturbed_source_built_exactly_once_per_member(self):
        ocean = make_ocean()
        ensemble = ForcingEnsemble(ocean, TEST_ENSEMBLE_CONFIG)
        sampled = make_sampled(n=3)

        import iste.forward_propagation as fp_module
        call_count = {"n": 0}
        original = fp_module.make_perturbed_source

        def counting_wrapper(*args, **kwargs):
            call_count["n"] += 1
            return original(*args, **kwargs)

        fp_module.make_perturbed_source = counting_wrapper
        try:
            forward_propagate_candidate(
                sampled, T_OBS, DT, ocean, ensemble, forcing_member_ids=[0, 1], seeds=[1, 2],
                k_lat=LAT, k_lon=LON, k_field=ZERO_K,
            )
        finally:
            fp_module.make_perturbed_source = original

        self.assertEqual(call_count["n"], 2)  # exactly once per member, never per step

    def test_all_particles_in_one_run_share_identical_perturbation_realization(self):
        # Two co-located particles under the SAME member/seed must end up at the
        # IDENTICAL final position only if they experience the identical field AND
        # identical stochastic draw is not required (K=0 removes the stochastic part,
        # isolating pure coherent-drift agreement).
        ocean = make_ocean()
        ensemble = ForcingEnsemble(ocean, TEST_ENSEMBLE_CONFIG)
        sampled = make_sampled(n=2, spread=0.0)  # identical starting position
        result = forward_propagate_candidate(
            sampled, T_OBS, DT, ocean, ensemble, forcing_member_ids=[0], seeds=[5],
            k_lat=LAT, k_lon=LON, k_field=ZERO_K,
        )
        run = result[0]
        self.assertAlmostEqual(run.final_lat[0], run.final_lat[1], places=9)
        self.assertAlmostEqual(run.final_lon[0], run.final_lon[1], places=9)


class TestMetadataIntact(unittest.TestCase):
    """Requirement 4: selected particle metadata remains intact."""

    def test_metadata_matches_sampled_input_exactly(self):
        ocean = make_ocean()
        ensemble = ForcingEnsemble(ocean, TEST_ENSEMBLE_CONFIG)
        sampled = make_sampled(n=5, original_forcing_member_id=[3, 3, 3, 3, 3])
        result = forward_propagate_candidate(
            sampled, T_OBS, DT, ocean, ensemble, forcing_member_ids=[5], seeds=[1],
            k_lat=LAT, k_lon=LON, k_field=ZERO_K,
        )
        run = result[5]
        np.testing.assert_array_equal(run.particle_id, sampled.particle_id)
        np.testing.assert_array_equal(run.observation_pixel_id, sampled.observation_pixel_id)
        np.testing.assert_array_equal(run.forcing_member_id, sampled.forcing_member_id)  # ORIGINAL, not 7
        np.testing.assert_array_almost_equal(run.original_log_weight, sampled.log_weight)
        self.assertEqual(run.candidate_id, sampled.candidate_id)
        self.assertEqual(run.t_s, np.datetime64(sampled.t_s, "ns"))
        # the run's OWN forward member (5) is distinct from every particle's original (3)
        self.assertEqual(run.forward_forcing_member_id, 5)
        self.assertTrue(np.all(run.forcing_member_id == 3))


class TestReproducibility(unittest.TestCase):
    """Requirements 5-6: same seed reproduces; different seeds can differ."""

    def test_same_seed_gives_identical_trajectory(self):
        ocean = make_ocean()
        ensemble = ForcingEnsemble(ocean, TEST_ENSEMBLE_CONFIG)
        sampled = make_sampled(n=4)
        r1 = forward_propagate_candidate_member(
            sampled, T_OBS, DT, ocean, 0, LAT, LON, np.full_like(ZERO_K, 5.0), np.random.default_rng(42),
        )
        r2 = forward_propagate_candidate_member(
            sampled, T_OBS, DT, ocean, 0, LAT, LON, np.full_like(ZERO_K, 5.0), np.random.default_rng(42),
        )
        np.testing.assert_array_equal(r1.lat, r2.lat)
        np.testing.assert_array_equal(r1.lon, r2.lon)
        np.testing.assert_array_equal(r1.valid, r2.valid)

    def test_different_seed_can_give_different_trajectory(self):
        ocean = make_ocean()
        sampled = make_sampled(n=4)
        r1 = forward_propagate_candidate_member(
            sampled, T_OBS, DT, ocean, 0, LAT, LON, np.full_like(ZERO_K, 5.0), np.random.default_rng(1),
        )
        r2 = forward_propagate_candidate_member(
            sampled, T_OBS, DT, ocean, 0, LAT, LON, np.full_like(ZERO_K, 5.0), np.random.default_rng(2),
        )
        self.assertFalse(np.array_equal(r1.lat, r2.lat))

    def test_full_orchestration_same_seeds_reproducible(self):
        ocean = make_ocean()
        ensemble = ForcingEnsemble(ocean, TEST_ENSEMBLE_CONFIG)
        sampled = make_sampled(n=3)
        r1 = forward_propagate_candidate(
            sampled, T_OBS, DT, ocean, ensemble, forcing_member_ids=[0, 1], seeds=[11, 22],
            k_lat=LAT, k_lon=LON, k_field=ZERO_K,
        )
        r2 = forward_propagate_candidate(
            sampled, T_OBS, DT, ocean, ensemble, forcing_member_ids=[0, 1], seeds=[11, 22],
            k_lat=LAT, k_lon=LON, k_field=ZERO_K,
        )
        for member_id in (0, 1):
            np.testing.assert_array_equal(r1[member_id].lat, r2[member_id].lat)
            np.testing.assert_array_equal(r1[member_id].lon, r2[member_id].lon)


class TestInvalidParticlesStopContributing(unittest.TestCase):
    """Requirement 7: invalid particles stop contributing after invalidation."""

    def test_particle_frozen_after_going_invalid(self):
        # strong forward drift straight into a masked "land" cell.
        uo = np.full((N_HOURS, len(LAT), len(LON)), 80.0)
        uo[:, 4, 8] = np.nan  # land at lat=LAT[4]=20.0, lon=LON[8]=118.0, straight east
        ocean = GriddedFieldSource(
            source_path="<synthetic>", time_name="time", time=TIME, lat_name="lat", lat=LAT,
            lon_name="lon", lon=LON, data={"uo": uo, "vo": np.zeros_like(uo)}, fill_threshold=1e19,
        )
        sampled = make_sampled(n=1, lat0=20.0, lon0=110.0, spread=0.0)
        run = forward_propagate_candidate_member(
            sampled, T_OBS, DT, ocean, 0, LAT, LON, ZERO_K, np.random.default_rng(3),
        )
        first_dead = run.first_invalid_step(0)
        self.assertIsNotNone(first_dead)
        self.assertLess(first_dead, run.n_steps)
        # every checkpoint from first_dead onward stays invalid and frozen at the same position
        frozen_lat = run.lat[first_dead, 0]
        frozen_lon = run.lon[first_dead, 0]
        for row in range(first_dead, run.n_steps + 1):
            self.assertFalse(run.valid[row, 0])
            self.assertEqual(run.lat[row, 0], frozen_lat)
            self.assertEqual(run.lon[row, 0], frozen_lon)

    def test_all_particles_invalid_short_circuits_without_error(self):
        ocean = make_ocean()
        sampled = make_sampled(n=2, valid=[False, False])
        run = forward_propagate_candidate_member(
            sampled, T_OBS, DT, ocean, 0, LAT, LON, ZERO_K, np.random.default_rng(1),
        )
        self.assertTrue(np.all(~run.valid))
        # positions frozen at their T19 starting values throughout
        np.testing.assert_array_equal(run.lat[0], run.lat[-1])
        np.testing.assert_array_equal(run.lon[0], run.lon[-1])


class TestDomainLandInvalidation(unittest.TestCase):
    """Requirement 8: domain/land invalidation handled correctly (Stage III-compatible
    policy, via T20/T04 unchanged)."""

    def test_step_off_the_grid_edge_marks_invalid(self):
        uo = np.full((N_HOURS, len(LAT), len(LON)), 500.0)  # huge eastward push
        ocean = GriddedFieldSource(
            source_path="<synthetic>", time_name="time", time=TIME, lat_name="lat", lat=LAT,
            lon_name="lon", lon=LON, data={"uo": uo, "vo": np.zeros_like(uo)}, fill_threshold=1e19,
        )
        sampled = make_sampled(n=1, lat0=20.0, lon0=119.5, spread=0.0)  # near the eastern edge
        run = forward_propagate_candidate_member(
            sampled, T_OBS, DT, ocean, 0, LAT, LON, ZERO_K, np.random.default_rng(1),
        )
        self.assertFalse(run.final_valid[0])


class TestNoOvershoot(unittest.TestCase):
    """Requirement 9: the run does not overshoot t_obs, for dt that divides evenly and
    for dt that does NOT divide evenly (forcing a final partial step)."""

    def test_evenly_dividing_dt(self):
        ocean = make_ocean()
        sampled = make_sampled(n=1)
        run = forward_propagate_candidate_member(
            sampled, T_OBS, 900.0, ocean, 0, LAT, LON, ZERO_K, np.random.default_rng(1),
        )
        self.assertEqual(run.timestamps[-1], T_OBS)
        self.assertTrue(np.all(run.timestamps <= T_OBS))
        self.assertEqual(run.n_steps, int((T_OBS - T_S) / np.timedelta64(900, "s")))

    def test_non_evenly_dividing_dt_takes_a_final_partial_step(self):
        ocean = make_ocean()
        sampled = make_sampled(n=1)
        dt_seconds = 1000.0  # (3h = 10800s) / 1000s = 10.8 -> 10 full steps + 1 partial
        run = forward_propagate_candidate_member(
            sampled, T_OBS, dt_seconds, ocean, 0, LAT, LON, ZERO_K, np.random.default_rng(1),
        )
        self.assertEqual(run.timestamps[-1], T_OBS)
        self.assertTrue(np.all(run.timestamps <= T_OBS))
        self.assertEqual(run.n_steps, 11)  # 10 full 1000s steps + 1 final 800s step

    def test_t_s_equal_t_obs_takes_zero_steps(self):
        ocean = make_ocean()
        sampled = make_sampled(n=1)
        run = forward_propagate_candidate_member(
            sampled, T_S, 900.0, ocean, 0, LAT, LON, ZERO_K, np.random.default_rng(1),
        )
        self.assertEqual(run.n_steps, 0)
        self.assertEqual(len(run.timestamps), 1)
        self.assertEqual(run.timestamps[0], T_S)

    def test_t_obs_before_t_s_raises(self):
        ocean = make_ocean()
        sampled = make_sampled(n=1)
        with self.assertRaises(ValueError):
            forward_propagate_candidate_member(
                sampled, T_S - np.timedelta64(1, "h"), 900.0, ocean, 0, LAT, LON, ZERO_K, np.random.default_rng(1),
            )


class TestForcingMemberSubsetRecorded(unittest.TestCase):
    """Requirement 10: explicit forcing-member subset preserved in the output."""

    def test_subset_recorded_exactly(self):
        ocean = make_ocean()
        ensemble = ForcingEnsemble(ocean, TEST_ENSEMBLE_CONFIG)
        sampled = make_sampled(n=2)
        result = forward_propagate_candidate(
            sampled, T_OBS, DT, ocean, ensemble, forcing_member_ids=[4, 1], seeds=[1, 2],
            k_lat=LAT, k_lon=LON, k_field=ZERO_K,
        )
        self.assertEqual(result.forcing_member_ids, (4, 1))
        self.assertEqual(set(result.runs.keys()), {4, 1})

    def test_seeds_length_mismatch_raises(self):
        ocean = make_ocean()
        ensemble = ForcingEnsemble(ocean, TEST_ENSEMBLE_CONFIG)
        sampled = make_sampled(n=2)
        with self.assertRaises(ValueError):
            forward_propagate_candidate(
                sampled, T_OBS, DT, ocean, ensemble, forcing_member_ids=[0, 1], seeds=[1],
                k_lat=LAT, k_lon=LON, k_field=ZERO_K,
            )


class TestMultipleCandidatesAndTimestampsIndependent(unittest.TestCase):
    """Requirement 11: multiple candidates and multiple historical timestamps remain
    independent."""

    def test_two_candidates_do_not_interfere(self):
        ocean = make_ocean()
        ensemble = ForcingEnsemble(ocean, TEST_ENSEMBLE_CONFIG)
        sampled_a = make_sampled(n=3, candidate_id="cand-A", lat0=15.0, lon0=105.0)
        sampled_b = make_sampled(n=5, candidate_id="cand-B", lat0=25.0, lon0=115.0)

        result_a = forward_propagate_candidate(
            sampled_a, T_OBS, DT, ocean, ensemble, forcing_member_ids=[0], seeds=[1],
            k_lat=LAT, k_lon=LON, k_field=ZERO_K,
        )
        result_b = forward_propagate_candidate(
            sampled_b, T_OBS, DT, ocean, ensemble, forcing_member_ids=[0], seeds=[1],
            k_lat=LAT, k_lon=LON, k_field=ZERO_K,
        )
        self.assertEqual(result_a.candidate_id, "cand-A")
        self.assertEqual(result_b.candidate_id, "cand-B")
        self.assertEqual(result_a[0].n_particles, 3)
        self.assertEqual(result_b[0].n_particles, 5)
        # starting positions independent -> final positions independent/different regions
        self.assertLess(np.mean(result_a[0].final_lat), np.mean(result_b[0].final_lat))

    def test_two_historical_timestamps_independent(self):
        ocean = make_ocean()
        ensemble = ForcingEnsemble(ocean, TEST_ENSEMBLE_CONFIG)
        t_s_2 = T_S + np.timedelta64(1, "h")
        sampled_1 = make_sampled(n=2, candidate_id="cand-t1", t_s=T_S)
        sampled_2 = make_sampled(n=2, candidate_id="cand-t2", t_s=t_s_2)

        result_1 = forward_propagate_candidate(
            sampled_1, T_OBS, DT, ocean, ensemble, forcing_member_ids=[0], seeds=[1],
            k_lat=LAT, k_lon=LON, k_field=ZERO_K,
        )
        result_2 = forward_propagate_candidate(
            sampled_2, T_OBS, DT, ocean, ensemble, forcing_member_ids=[0], seeds=[1],
            k_lat=LAT, k_lon=LON, k_field=ZERO_K,
        )
        self.assertEqual(result_1.t_s, T_S)
        self.assertEqual(result_2.t_s, t_s_2)
        # the later-starting run propagates for a SHORTER duration -> fewer steps
        self.assertLess(result_2[0].n_steps, result_1[0].n_steps)


class TestZeroSelectedParticles(unittest.TestCase):
    """T19's Y=0 case feeding into T21: an unvalidatable candidate must not crash."""

    def test_empty_particle_set_produces_empty_run(self):
        ocean = make_ocean()
        sampled = make_sampled(n=0)
        run = forward_propagate_candidate_member(
            sampled, T_OBS, DT, ocean, 0, LAT, LON, ZERO_K, np.random.default_rng(1),
        )
        self.assertEqual(run.n_particles, 0)
        self.assertEqual(run.lat.shape, (run.n_steps + 1, 0))


if __name__ == "__main__":
    unittest.main()
