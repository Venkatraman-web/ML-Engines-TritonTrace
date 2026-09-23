"""T13.5 -- Stage III -> Stage IV compatibility correction.

Focused tests verifying the signed-off Stage III implementation is structurally
compatible with `final_stage_III_IV_specification.txt` (the frozen Stage III -> Stage IV
contract), covering:

- Part C: boundary/invalidation semantics (death position/time/reason correctness).
- Part D: the `K(x,t)` interface accepts a time argument and is consistent across times
  for the constant-K prototype.
- Part E: the weight update's divergence term is `div(U)` (the base current), not
  `div(U_e)` (a forcing-member-perturbed current), matching the frozen contract's Sec. 5
  exactly -- with a controlled synthetic field where the correct answer is known.
- Part G: storage round-trip of the new death/survival metadata, including a particle
  that dies strictly between two checkpoints.
- Part H: the critical temporal-death regression test -- Stage III's stored state must
  contain enough information for a later Stage IV query to treat a particle as alive
  before its exact death time and dead at/after it, even between two checkpoints.

This module implements NO Stage IV logic (no `c_hat*`, no `h`, no `pi`, no likelihood) --
it only verifies Stage III's own output/storage contract, per T13.5's explicit scope.

Run with:
    python -m unittest tests.test_stage3_stage4_compatibility -v
"""

from __future__ import annotations

import unittest

import numpy as np

from iste.environment import GriddedFieldSource, NetCDFEnvironmentalFieldProvider, interpolated_divergence
from iste.ensemble_run import make_perturbed_source
from iste.forcing_ensemble import ForcingEnsemble, ForcingEnsembleConfig
from iste.geometry import metres_per_degree_longitude
from iste.lookup_storage import ParticleRecordStore
from iste.step import WeightedParticleState, backward_euler_maruyama_step
from iste.trajectory import run_backward_trajectory

LAT = np.array([10.0, 20.0, 30.0])
LON = np.array([100.0, 105.0, 110.0, 115.0, 120.0, 125.0, 130.0])
TIME = np.datetime64("2020-01-01T00:00", "ns") + np.arange(8) * np.timedelta64(3600, "s")
T0 = TIME[-1]
ZERO_UV = np.zeros((3, 7))
ZERO_K = np.zeros((3, 7))


def make_ocean(uo_grid: np.ndarray, vo_grid: np.ndarray) -> GriddedFieldSource:
    uo = np.stack([uo_grid] * len(TIME))
    vo = np.stack([vo_grid] * len(TIME))
    return GriddedFieldSource(
        source_path="<synthetic>", time_name="time", time=TIME, lat_name="lat", lat=LAT,
        lon_name="lon", lon=LON, data={"uo": uo, "vo": vo}, fill_threshold=1e19,
    )


def _domain_exit_trajectory(dt_seconds: float, n_steps: int):
    """A particle pushed west by a constant, exactly-known velocity until it exits the
    domain (`lon < LON.min()`) -- gives an exact, hand-predictable `death_time` at
    whatever checkpoint spacing `dt_seconds` is configured to, reused by several tests
    below that need precise control over exactly when death occurs."""
    dlon_per_deg = metres_per_degree_longitude(20.0)
    step_deg = 3.0
    v_speed = step_deg * dlon_per_deg / dt_seconds
    ocean = make_ocean(np.full((3, 7), v_speed), np.zeros((3, 7)))
    rng = np.random.default_rng(1)
    traj = run_backward_trajectory(20.0, 114.0, T0, n_steps, dt_seconds, ocean, LAT, LON, ZERO_K, rng)
    return ocean, traj


class TestPartC_BoundaryDeathSemantics(unittest.TestCase):
    """Part C: verify death position/time/reason correctness against the actual
    T04/T05/T06 propagation flow (not inferred from variable names)."""

    def test_domain_exit_death_position_is_the_rejected_position_not_last_valid(self):
        _, traj = _domain_exit_trajectory(dt_seconds=600.0, n_steps=6)
        # the trajectory's OWN lat/lon array freezes at the last VALID position...
        self.assertEqual(traj.lon[-1], 102.0)
        # ...but death_lon is the actual (rejected) invalidating position, which is
        # DIFFERENT -- exactly the "last valid checkpoint" vs "exact death event"
        # distinction T13.5 is required to preserve.
        self.assertLess(traj.death_lon, LON.min())
        self.assertNotEqual(traj.death_lon, traj.lon[-1])

    def test_domain_exit_reason_is_domain_exit(self):
        _, traj = _domain_exit_trajectory(dt_seconds=600.0, n_steps=6)
        self.assertEqual(traj.death_reason, "domain_exit")

    def test_land_cell_death_reason_is_land_and_position_is_on_land(self):
        # Land at (lat=20, lon=100); the particle starts far enough away (lon=120, 4
        # grid columns from the land cell) that divergence's own 3-point finite-
        # difference stencil at the ORIGIN is unaffected by the NaN (isolating a
        # destination-only "land" rejection from a "physics_unavailable" one) -- dt is
        # chosen (via the known uo speed and metres-per-degree conversion) so the single
        # step's destination lands exactly at lon=102, inside the land cell's own
        # bilinear interpolation stencil (bracket [100, 105), corner (20,100)=NaN).
        uo_land = np.full((3, 7), 50.0)
        uo_land[1, 0] = np.nan  # land at (lat=20, lon=100)
        ocean = make_ocean(uo_land, np.zeros((3, 7)))
        dlon_per_deg = metres_per_degree_longitude(20.0)
        dt = 18.0 * dlon_per_deg / 50.0  # 120 -> 102 degrees of longitude in one step

        state = WeightedParticleState.initialize(lat0=[20.0], lon0=[120.0])
        rng = np.random.default_rng(9)
        new_state = backward_euler_maruyama_step(state, T0, dt, ocean, LAT, LON, ZERO_K, rng)

        self.assertFalse(new_state.valid[0])
        self.assertEqual(new_state.death_reason[0], "land")
        self.assertAlmostEqual(new_state.death_lat[0], 20.0)
        self.assertAlmostEqual(new_state.death_lon[0], 102.0)
        # the land death position must be WITHIN the domain's own coverage (that is what
        # makes it "land," as opposed to "domain_exit").
        self.assertGreaterEqual(new_state.death_lat[0], LAT.min())
        self.assertLessEqual(new_state.death_lat[0], LAT.max())
        self.assertGreaterEqual(new_state.death_lon[0], LON.min())
        self.assertLessEqual(new_state.death_lon[0], LON.max())

    def test_physics_unavailable_death_reason_and_position_is_current_position(self):
        ocean = make_ocean(ZERO_UV, ZERO_UV)
        k_field = np.full((3, 7), 10.0)
        k_field[1, 3] = np.nan  # K missing exactly at the particle's own starting cell (lat=20, lon=115)
        state = WeightedParticleState.initialize(lat0=[20.0], lon0=[115.0])
        rng = np.random.default_rng(3)
        new_state = backward_euler_maruyama_step(state, T0, 600.0, ocean, LAT, LON, k_field, rng)
        self.assertFalse(new_state.valid[0])
        self.assertEqual(new_state.death_reason[0], "physics_unavailable")
        # death position is the CURRENT (pre-step) position -- no destination was ever
        # computed, since the physics needed to compute one was unavailable.
        self.assertEqual(new_state.death_lat[0], 20.0)
        self.assertEqual(new_state.death_lon[0], 115.0)

    def test_pre_step_position_preserved_when_proposed_step_rejected(self):
        # Part C Q1: is the pre-step position preserved if the proposed step is rejected?
        _, traj = _domain_exit_trajectory(dt_seconds=600.0, n_steps=6)
        first_invalid = traj.first_invalid_index
        # the trajectory's lat/lon at and after the first invalid index freeze at the
        # PRE-step (last valid) position -- never at the rejected destination.
        np.testing.assert_array_equal(traj.lon[first_invalid:], traj.lon[first_invalid - 1])
        np.testing.assert_array_equal(traj.lat[first_invalid:], traj.lat[first_invalid - 1])

    def test_propagation_stops_immediately_after_death(self):
        # Part C Q4/Q5: propagation stops immediately; later checkpoints do not pretend
        # the particle is still alive.
        _, traj = _domain_exit_trajectory(dt_seconds=600.0, n_steps=6)
        first_invalid = traj.first_invalid_index
        self.assertIsNotNone(first_invalid)
        self.assertTrue(np.all(~traj.valid[first_invalid:]))
        self.assertTrue(np.all(traj.valid[:first_invalid]))

    def test_death_time_is_an_exact_checkpoint_never_fabricated(self):
        # Part C Q3/Q6: is the invalidating timestamp known exactly? Since this
        # integrator's checkpoint spacing IS its integration step size, the only honest
        # answer is "the checkpoint the rejected step targeted" -- never an interpolated
        # in-between instant fabricated without integration support for it.
        _, traj = _domain_exit_trajectory(dt_seconds=600.0, n_steps=6)
        first_invalid = traj.first_invalid_index
        self.assertEqual(traj.death_time, traj.timestamps[first_invalid])

    def test_out_of_range_leaving_domain_becomes_explicit_invalid_not_silent(self):
        _, traj = _domain_exit_trajectory(dt_seconds=600.0, n_steps=6)
        self.assertFalse(traj.valid[-1])  # explicitly invalid, not silently dropped
        self.assertEqual(traj.n_states, 7)  # never truncated


class TestPartD_KInterfaceIsTimeAware(unittest.TestCase):
    """Part D: K(x,t) must be queryable at an arbitrary valid timestamp and be a valid
    function of time even though the prototype value is constant -- verified, not
    assumed, and no new physical K model/time-varying formula is added."""

    def test_K_constant_across_different_timestamps(self):
        ocean = make_ocean(ZERO_UV, ZERO_UV)
        k_field = np.full((3, 7), 7.5)
        provider = NetCDFEnvironmentalFieldProvider(ocean, k_lat=LAT, k_lon=LON, k_field=k_field)
        x = np.array([115.0, 20.0])  # (lon, lat)
        k_t1 = provider.K(x, TIME[0])
        k_t2 = provider.K(x, TIME[5])
        self.assertEqual(k_t1, k_t2)
        self.assertEqual(k_t1, 7.5)

    def test_grad_K_also_constant_across_different_timestamps(self):
        ocean = make_ocean(ZERO_UV, ZERO_UV)
        C = 50.0
        k_field = C * (LON[None, :] - 115.0) * np.ones((3, 1))
        provider = NetCDFEnvironmentalFieldProvider(ocean, k_lat=LAT, k_lon=LON, k_field=k_field)
        x = np.array([115.0, 20.0])
        grad_t1 = provider.grad_K(x, TIME[0])
        grad_t2 = provider.grad_K(x, TIME[6])
        np.testing.assert_array_equal(grad_t1, grad_t2)

    def test_K_query_accepts_every_valid_timestamp_in_the_source(self):
        ocean = make_ocean(ZERO_UV, ZERO_UV)
        k_field = np.full((3, 7), 3.0)
        provider = NetCDFEnvironmentalFieldProvider(ocean, k_lat=LAT, k_lon=LON, k_field=k_field)
        x = np.array([110.0, 20.0])
        for t in TIME:
            self.assertEqual(provider.K(x, t), 3.0)


class TestPartE_DivergenceUsesBaseFieldNotPerturbedField(unittest.TestCase):
    """Part E: the frozen contract's weight update, `log_w_new = log_w_old - div(U)*dt`,
    uses the BASE deterministic current `U`, never a forcing-member-perturbed `U_e`.
    Direct inspection (not intuition) showed the pre-T13.5 implementation used whatever
    single `ocean_source` was passed for BOTH advection and divergence -- meaning a
    per-member (perturbed) call computed `div(U_e)`, not `div(U)`. Fixed via the new
    `divergence_source` parameter; these tests prove the fix with a controlled field
    where `div(U)` and `div(U_e)` are provably different numbers."""

    def _base_and_perturbed(self):
        C = 0.001
        lon_mesh = LON[None, :] * np.ones((len(LAT), 1))
        uo_base = C * (lon_mesh - 110.0)
        ocean = make_ocean(uo_base, np.zeros((3, 7)))
        ensemble = ForcingEnsemble(ocean, ForcingEnsembleConfig(
            n_members=2, sigma_u=0.5, spatial_correlation_km=60.0,
            correlation_time_seconds=3600.0, coarse_resolution_deg=0.5, master_seed=1,
        ))
        perturbed = make_perturbed_source(ocean, ensemble, member_id=0)
        return ocean, perturbed

    def test_div_U_and_div_U_e_are_provably_different_for_this_field(self):
        ocean, perturbed = self._base_and_perturbed()
        lat_q, lon_q = np.array([20.0]), np.array([115.0])
        div_base, _ = interpolated_divergence(ocean, lat_q, lon_q, T0)
        div_perturbed, _ = interpolated_divergence(perturbed, lat_q, lon_q, T0)
        self.assertFalse(np.isclose(div_base[0], div_perturbed[0]))

    def test_weight_update_uses_div_U_base_when_divergence_source_supplied(self):
        ocean, perturbed = self._base_and_perturbed()
        lat_q, lon_q = np.array([20.0]), np.array([115.0])
        div_base, _ = interpolated_divergence(ocean, lat_q, lon_q, T0)

        state = WeightedParticleState.initialize(lat0=[20.0], lon0=[115.0])
        rng = np.random.default_rng(0)
        # advection uses the PERTURBED source; divergence_source=ocean (base) --
        # exactly how iste.ensemble_run.run_ensemble now calls this (T13.5 fix).
        new_state = backward_euler_maruyama_step(
            state, T0, 900.0, perturbed, LAT, LON, ZERO_K, rng, divergence_source=ocean,
        )
        implied_div = -new_state.log_weight[0] / 900.0
        self.assertAlmostEqual(implied_div, float(div_base[0]), places=9)

    def test_regression_log_w_new_minus_log_w_old_equals_minus_div_U_times_dt(self):
        """The exact regression form requested by Part E: for a controlled field with a
        KNOWN analytic div(U), confirm `log_w_new - log_w_old == -div(U)*dt` exactly,
        when the particle is advected by a DIFFERENT (perturbed) field."""
        ocean, perturbed = self._base_and_perturbed()
        analytic_div_U = 0.001 / metres_per_degree_longitude(20.0)  # d/dx[C*(lon-110)] = C, converted to 1/m

        state = WeightedParticleState.initialize(lat0=[20.0], lon0=[115.0], log_weight0=2.0)
        rng = np.random.default_rng(5)
        dt = 900.0
        new_state = backward_euler_maruyama_step(
            state, T0, dt, perturbed, LAT, LON, ZERO_K, rng, divergence_source=ocean,
        )
        expected_log_w_new = 2.0 - analytic_div_U * dt
        self.assertAlmostEqual(new_state.log_weight[0], expected_log_w_new, places=9)

    def test_default_divergence_source_none_preserves_original_behavior(self):
        # backward compatibility: omitting divergence_source uses ocean_source for both
        # advection AND divergence, exactly as before T13.5 -- no behavior change for any
        # caller that only ever has one (unperturbed) source.
        ocean, _ = self._base_and_perturbed()
        state = WeightedParticleState.initialize(lat0=[20.0], lon0=[115.0])
        rng_a = np.random.default_rng(9)
        rng_b = np.random.default_rng(9)
        out_no_kw = backward_euler_maruyama_step(state, T0, 900.0, ocean, LAT, LON, ZERO_K, rng_a)
        out_explicit_none = backward_euler_maruyama_step(
            state, T0, 900.0, ocean, LAT, LON, ZERO_K, rng_b, divergence_source=None,
        )
        self.assertEqual(out_no_kw.log_weight[0], out_explicit_none.log_weight[0])


class TestPartG_StorageRoundTripOfDeathMetadata(unittest.TestCase):
    """Part G: the T11 storage layer must round-trip the new death/survival metadata
    exactly, including a particle that dies strictly between two checkpoints."""

    def test_round_trip_preserves_death_metadata_exactly(self):
        from iste.ensemble_run import EnsembleRunRecords

        death_t = np.datetime64("2020-01-01T00:50:00", "ns")
        records = EnsembleRunRecords(
            particle_id=np.array(["p0", "p0"], dtype=object),
            forcing_member_id=np.array([0, 0]),
            observation_pixel_id=np.array(["px", "px"], dtype=object),
            timestamp=np.array([np.datetime64("2020-01-01T00:30:00", "ns"),
                                 np.datetime64("2020-01-01T00:00:00", "ns")]),
            lat=np.array([20.0, 20.0]),
            lon=np.array([102.0, 102.0]),
            log_weight=np.array([0.0, 0.0]),
            valid=np.array([False, False]),
            death_time=np.array([death_t, death_t]),
            death_lat=np.array([20.0, 20.0]),
            death_lon=np.array([99.0, 99.0]),
            death_reason=np.array(["domain_exit", "domain_exit"], dtype=object),
            survival_mark=np.array([0, 0], dtype=np.int8),
        )
        store = ParticleRecordStore()
        store.add_records(records)
        result = store.all_records()

        np.testing.assert_array_equal(result.death_time, records.death_time)
        np.testing.assert_array_equal(result.death_lat, records.death_lat)
        np.testing.assert_array_equal(result.death_lon, records.death_lon)
        np.testing.assert_array_equal(result.death_reason, records.death_reason)
        np.testing.assert_array_equal(result.survival_mark, records.survival_mark)
        # provenance fields unchanged too
        np.testing.assert_array_equal(result.observation_pixel_id, records.observation_pixel_id)
        np.testing.assert_array_equal(result.forcing_member_id, records.forcing_member_id)

    def test_surviving_particle_has_survival_mark_one(self):
        from iste.ensemble_run import EnsembleRunRecords

        records = EnsembleRunRecords(
            particle_id=np.array(["p0"], dtype=object),
            forcing_member_id=np.array([0]),
            observation_pixel_id=np.array(["px"], dtype=object),
            timestamp=np.array([np.datetime64("2020-01-01T00:00:00", "ns")]),
            lat=np.array([20.0]), lon=np.array([110.0]), log_weight=np.array([0.0]),
            valid=np.array([True]),
            death_time=np.array([np.datetime64("NaT", "ns")]),
            death_lat=np.array([np.nan]), death_lon=np.array([np.nan]),
            death_reason=np.array([""], dtype=object), survival_mark=np.array([1], dtype=np.int8),
        )
        store = ParticleRecordStore()
        store.add_records(records)
        result = store.all_records()
        self.assertEqual(result.survival_mark[0], 1)

    def test_toy_case_checkpoint_30_60_death_50_round_trips_via_real_run(self):
        """The exact toy scenario T13.5 asks for: checkpoint 1 = 30 min, checkpoint 2 =
        60 min, death_time = 50 min -- built from an ACTUAL Stage III run (not
        hand-fabricated), then stored and retrieved."""
        ocean, traj = _domain_exit_trajectory(dt_seconds=600.0, n_steps=6)  # 10-min grid, 0..60 min
        self.assertEqual(traj.death_time, T0 - np.timedelta64(50, "m"))

        from iste.ensemble_run import EnsembleRunRecords
        n = traj.n_states
        records = EnsembleRunRecords(
            particle_id=np.array(["toy"] * n, dtype=object),
            forcing_member_id=np.zeros(n, dtype=np.int64),
            observation_pixel_id=np.array(["px-toy"] * n, dtype=object),
            timestamp=traj.timestamps, lat=traj.lat, lon=traj.lon, log_weight=traj.log_weight,
            valid=traj.valid,
            death_time=np.full(n, traj.death_time, dtype="datetime64[ns]"),
            death_lat=np.full(n, traj.death_lat), death_lon=np.full(n, traj.death_lon),
            death_reason=np.array([traj.death_reason] * n, dtype=object),
            survival_mark=traj.valid.astype(np.int8),
        )
        store = ParticleRecordStore()
        store.add_records(records)
        result = store.records_for_pixel("px-toy")

        checkpoint_30 = T0 - np.timedelta64(30, "m")
        checkpoint_60 = T0 - np.timedelta64(60, "m")
        self.assertIn(checkpoint_30, result.timestamp)
        self.assertIn(checkpoint_60, result.timestamp)
        self.assertTrue(np.all(result.death_time == T0 - np.timedelta64(50, "m")))


class TestPartH_CriticalTemporalDeathRegression(unittest.TestCase):
    """Part H: the critical temporal-death regression test. Does NOT implement Stage IV
    interpolation/c_hat*/h/pi/likelihood -- only proves Stage III's stored `death_time`
    is sufficient, by itself, to determine alive/dead-for-contribution status at any
    query time, including one strictly between two checkpoints."""

    def test_alive_before_dead_at_and_after_death_time(self):
        _, traj = _domain_exit_trajectory(dt_seconds=600.0, n_steps=6)  # checkpoints 0,10,...,60 min
        self.assertEqual(traj.death_time, T0 - np.timedelta64(50, "m"))

        # NOTE on direction: forward (real-world) time DECREASES as backward theta
        # increases (theta = t0 - t). "45 minutes back" (theta=45) is chronologically
        # LATER (closer to t0) than "50 minutes back" (theta=50, the death event) --
        # so "alive" (theta_query < theta_death, i.e. not yet as far back as death)
        # corresponds to forward-time timestamp COMPARISON: query_time > death_time.
        query_alive = T0 - np.timedelta64(45, "m")       # theta=45min < theta_death=50min -- alive
        query_dead_exact = T0 - np.timedelta64(50, "m")  # theta=50min == theta_death -- dead
        query_dead_after = T0 - np.timedelta64(55, "m")  # theta=55min > theta_death -- dead

        # the only logic exercised here is a plain comparison against the stored
        # death_time -- exactly the minimum a future Stage IV query needs, per Sec. 9's
        # "exclude the particle from its exact death time onward."
        self.assertTrue(query_alive > traj.death_time)        # alive: strictly before death (in theta)
        self.assertFalse(query_dead_exact > traj.death_time)  # dead: at death (in theta), excluded
        self.assertFalse(query_dead_after > traj.death_time)  # dead: after death (in theta), excluded

    def test_never_died_particle_has_no_death_time_and_is_always_alive(self):
        ocean = make_ocean(np.full((3, 7), 0.05), np.zeros((3, 7)))  # tiny drift, stays interior
        rng = np.random.default_rng(11)
        traj = run_backward_trajectory(20.0, 115.0, T0, 4, 600.0, ocean, LAT, LON, ZERO_K, rng)
        self.assertTrue(np.all(traj.valid))
        self.assertIsNone(traj.death_time)


if __name__ == "__main__":
    unittest.main()
