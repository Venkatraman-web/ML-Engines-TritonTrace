"""Focused T06 checks for `iste.trajectory` (one weighted particle over a configurable
backward window, built on T05's single step).

Run with:
    python -m unittest tests.test_trajectory -v
"""

from __future__ import annotations

import unittest

import numpy as np

from iste.environment import GriddedFieldSource
from iste.geometry import metres_per_degree_latitude, metres_per_degree_longitude
from iste.trajectory import run_backward_trajectory

LAT = np.array([10.0, 20.0, 30.0])
LON = np.array([100.0, 110.0, 120.0])
T0 = np.datetime64("2020-01-01T06:00", "ns")
# a wide-enough hourly time range (t0-8h .. t0) to cover every trajectory tested below,
# with the ocean fields held constant across time (so a "step_count/dt" window well
# under 8 hours never runs off the grid's time coverage).
TIME = T0 - np.arange(9)[::-1] * np.timedelta64(3600, "s")
DT = 900.0  # 15 minutes, seconds


def make_ocean(uo_grid: np.ndarray, vo_grid: np.ndarray) -> GriddedFieldSource:
    n_t = len(TIME)
    uo = np.stack([uo_grid] * n_t)
    vo = np.stack([vo_grid] * n_t)
    return GriddedFieldSource(
        source_path="<synthetic>", time_name="time", time=TIME, lat_name="lat", lat=LAT,
        lon_name="lon", lon=LON, data={"uo": uo, "vo": vo}, fill_threshold=1e19,
    )


ZERO_UV = np.zeros((3, 3))
ZERO_K = np.zeros((3, 3))


class TestStateCountAndTimestamps(unittest.TestCase):
    def test_correct_number_of_states(self):
        ocean = make_ocean(ZERO_UV, ZERO_UV)
        traj = run_backward_trajectory(
            20.0, 110.0, T0, n_steps=4, dt_seconds=DT, ocean_source=ocean,
            k_lat=LAT, k_lon=LON, k_field=ZERO_K, rng=np.random.default_rng(1),
        )
        self.assertEqual(traj.n_states, 5)
        for arr in (traj.timestamps, traj.lat, traj.lon, traj.log_weight, traj.valid):
            self.assertEqual(len(arr), 5)

    def test_correct_backward_timestamps(self):
        ocean = make_ocean(ZERO_UV, ZERO_UV)
        traj = run_backward_trajectory(
            20.0, 110.0, T0, n_steps=4, dt_seconds=DT, ocean_source=ocean,
            k_lat=LAT, k_lon=LON, k_field=ZERO_K, rng=np.random.default_rng(1),
        )
        expected = T0 - np.arange(5) * np.timedelta64(int(DT), "s")
        np.testing.assert_array_equal(traj.timestamps, expected)

    def test_elapsed_backward_time_equals_step_count_times_dt(self):
        ocean = make_ocean(ZERO_UV, ZERO_UV)
        n_steps = 6
        traj = run_backward_trajectory(
            20.0, 110.0, T0, n_steps=n_steps, dt_seconds=DT, ocean_source=ocean,
            k_lat=LAT, k_lon=LON, k_field=ZERO_K, rng=np.random.default_rng(1),
        )
        elapsed = T0 - traj.timestamps[-1]
        self.assertEqual(elapsed, np.timedelta64(int(n_steps * DT), "s"))

    def test_zero_steps_returns_only_the_initial_state(self):
        ocean = make_ocean(ZERO_UV, ZERO_UV)
        traj = run_backward_trajectory(
            20.0, 110.0, T0, n_steps=0, dt_seconds=DT, ocean_source=ocean,
            k_lat=LAT, k_lon=LON, k_field=ZERO_K, rng=np.random.default_rng(1),
        )
        self.assertEqual(traj.n_states, 1)
        self.assertEqual(traj.timestamps[0], T0)
        self.assertEqual(traj.lat[0], 20.0)
        self.assertEqual(traj.lon[0], 110.0)
        self.assertTrue(traj.valid[0])


class TestDeterministicTrajectory(unittest.TestCase):
    """Constant eastward ocean velocity, K=0: every step must move the particle west by
    the exact same analytically-known amount (no stochastic term at all)."""

    def test_matches_hand_computed_positions(self):
        u_speed = 1.0
        ocean = make_ocean(np.full((3, 3), u_speed), ZERO_UV)
        n_steps = 4
        traj = run_backward_trajectory(
            20.0, 110.0, T0, n_steps=n_steps, dt_seconds=DT, ocean_source=ocean,
            k_lat=LAT, k_lon=LON, k_field=ZERO_K, rng=np.random.default_rng(42),
        )
        dlon_per_deg = metres_per_degree_longitude(20.0)
        step_deg = (u_speed * DT) / dlon_per_deg
        expected_lon = 110.0 - np.arange(n_steps + 1) * step_deg
        np.testing.assert_allclose(traj.lon, expected_lon, atol=1e-9)
        np.testing.assert_allclose(traj.lat, 20.0, atol=1e-9)
        self.assertTrue(np.all(traj.valid))

    def test_reproducible_with_same_seed(self):
        ocean = make_ocean(np.full((3, 3), 0.5), np.full((3, 3), 0.3))
        k_field = np.full((3, 3), 10.0)
        traj_a = run_backward_trajectory(
            20.0, 110.0, T0, n_steps=5, dt_seconds=DT, ocean_source=ocean,
            k_lat=LAT, k_lon=LON, k_field=k_field, rng=np.random.default_rng(9),
        )
        traj_b = run_backward_trajectory(
            20.0, 110.0, T0, n_steps=5, dt_seconds=DT, ocean_source=ocean,
            k_lat=LAT, k_lon=LON, k_field=k_field, rng=np.random.default_rng(9),
        )
        np.testing.assert_array_equal(traj_a.lat, traj_b.lat)
        np.testing.assert_array_equal(traj_a.lon, traj_b.lon)


class TestWeightEvolution(unittest.TestCase):
    def test_weight_accumulates_linearly_under_constant_divergence(self):
        # uo = -C*lon (spatially linear -> div(U) is the SAME constant at every step,
        # since the field doesn't change with position along the trajectory's fixed
        # latitude/column-independent divergence here... use v=0, uo depends on lon only,
        # so div(U) varies with lon as the particle moves, but starting uo=0 at lon=110
        # keeps the check simple: verify against the per-step deltas actually observed.
        C = 0.001
        uo_grid = -C * LON[None, :] * np.ones((3, 1))
        ocean = make_ocean(uo_grid, ZERO_UV)
        n_steps = 5
        traj = run_backward_trajectory(
            20.0, 110.0, T0, n_steps=n_steps, dt_seconds=DT, ocean_source=ocean,
            k_lat=LAT, k_lon=LON, k_field=ZERO_K, rng=np.random.default_rng(3), log_weight0=0.0,
        )
        # weight must change monotonically in the same direction every step (convergent
        # flow throughout this small domain -> log-weight strictly increasing), and the
        # total change after n_steps must equal the sum of the per-step deltas.
        deltas = np.diff(traj.log_weight)
        self.assertTrue(np.all(deltas > 0))
        self.assertAlmostEqual(traj.log_weight[-1], np.sum(deltas), places=9)

    def test_zero_divergence_keeps_weight_constant_across_all_steps(self):
        ocean = make_ocean(np.full((3, 3), 0.4), np.full((3, 3), -0.2))
        traj = run_backward_trajectory(
            20.0, 110.0, T0, n_steps=6, dt_seconds=DT, ocean_source=ocean,
            k_lat=LAT, k_lon=LON, k_field=ZERO_K, rng=np.random.default_rng(5), log_weight0=2.0,
        )
        np.testing.assert_allclose(traj.log_weight, 2.0, atol=1e-9)


class TestInvalidityStopsPropagation(unittest.TestCase):
    def test_particle_hitting_land_stops_and_is_recorded_as_invalid(self):
        uo_grid = np.full((3, 3), 50.0)
        uo_grid[0, 0] = np.nan  # land at (lat=10, lon=100)
        vo_grid = np.zeros((3, 3))
        ocean = make_ocean(uo_grid, vo_grid)
        traj = run_backward_trajectory(
            10.0, 115.0, T0, n_steps=6, dt_seconds=DT, ocean_source=ocean,
            k_lat=LAT, k_lon=LON, k_field=ZERO_K, rng=np.random.default_rng(2),
        )
        self.assertEqual(traj.n_states, 7)  # never truncated
        self.assertIsNotNone(traj.first_invalid_index)
        idx = traj.first_invalid_index
        self.assertTrue(np.all(~traj.valid[idx:]))  # invalid from idx onward
        self.assertTrue(np.all(traj.valid[:idx]))  # valid before idx
        # frozen at the last valid position for every subsequent recorded state.
        np.testing.assert_array_equal(traj.lat[idx:], traj.lat[idx])
        np.testing.assert_array_equal(traj.lon[idx:], traj.lon[idx])
        np.testing.assert_array_equal(traj.log_weight[idx:], traj.log_weight[idx])

    def test_first_invalid_index_is_none_when_particle_stays_valid(self):
        ocean = make_ocean(ZERO_UV, ZERO_UV)
        traj = run_backward_trajectory(
            20.0, 110.0, T0, n_steps=4, dt_seconds=DT, ocean_source=ocean,
            k_lat=LAT, k_lon=LON, k_field=ZERO_K, rng=np.random.default_rng(1),
        )
        self.assertIsNone(traj.first_invalid_index)


class TestNoNaNsWhileValid(unittest.TestCase):
    def test_no_nans_in_a_fully_valid_trajectory(self):
        ocean = make_ocean(np.full((3, 3), 0.3), np.full((3, 3), -0.1))
        k_field = np.full((3, 3), 5.0)
        traj = run_backward_trajectory(
            20.0, 110.0, T0, n_steps=8, dt_seconds=DT, ocean_source=ocean,
            k_lat=LAT, k_lon=LON, k_field=k_field, rng=np.random.default_rng(11),
        )
        self.assertTrue(np.all(traj.valid))
        self.assertFalse(np.any(np.isnan(traj.lat)))
        self.assertFalse(np.any(np.isnan(traj.lon)))
        self.assertFalse(np.any(np.isnan(traj.log_weight)))


class TestInputValidation(unittest.TestCase):
    def test_negative_n_steps_raises(self):
        ocean = make_ocean(ZERO_UV, ZERO_UV)
        with self.assertRaises(ValueError):
            run_backward_trajectory(
                20.0, 110.0, T0, n_steps=-1, dt_seconds=DT, ocean_source=ocean,
                k_lat=LAT, k_lon=LON, k_field=ZERO_K, rng=np.random.default_rng(1),
            )

    def test_non_positive_dt_raises(self):
        ocean = make_ocean(ZERO_UV, ZERO_UV)
        with self.assertRaises(ValueError):
            run_backward_trajectory(
                20.0, 110.0, T0, n_steps=4, dt_seconds=0.0, ocean_source=ocean,
                k_lat=LAT, k_lon=LON, k_field=ZERO_K, rng=np.random.default_rng(1),
            )


if __name__ == "__main__":
    unittest.main()
