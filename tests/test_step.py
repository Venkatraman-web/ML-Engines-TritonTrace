"""Focused T05 checks for `iste.step` (one Stage III backward Euler-Maruyama step).

Run with:
    python -m unittest tests.test_step -v
"""

from __future__ import annotations

import unittest

import numpy as np

from iste.environment import GriddedFieldSource
from iste.geometry import metres_per_degree_latitude, metres_per_degree_longitude
from iste.step import WeightedParticleState, backward_euler_maruyama_step

LAT = np.array([10.0, 20.0, 30.0])
LON = np.array([100.0, 110.0, 120.0])
TIME = np.array(["2020-01-01T00:00", "2020-01-01T01:00"], dtype="datetime64[ns]")
T0 = TIME[0]
DT = 900.0  # 15 minutes, seconds


def make_ocean(uo_grid: np.ndarray, vo_grid: np.ndarray) -> GriddedFieldSource:
    """Two identical hourly snapshots of the given (n_lat, n_lon) uo/vo fields."""
    uo = np.stack([uo_grid, uo_grid])
    vo = np.stack([vo_grid, vo_grid])
    return GriddedFieldSource(
        source_path="<synthetic>",
        time_name="time",
        time=TIME,
        lat_name="lat",
        lat=LAT,
        lon_name="lon",
        lon=LON,
        data={"uo": uo, "vo": vo},
        fill_threshold=1e19,
    )


ZERO_UV = np.zeros((3, 3))
ZERO_K = np.zeros((3, 3))


class TestDiffusionOnly(unittest.TestCase):
    """Zero velocity + constant K: no drift at all (U=0, grad(K)=0 for a constant field),
    so the displacement must exactly equal `sqrt(2*K*dt) * xi` for the RNG's own draw."""

    def test_matches_exact_stochastic_displacement_and_weight_unchanged(self):
        ocean = make_ocean(ZERO_UV, ZERO_UV)
        k_field = np.full((3, 3), 50.0)
        state = WeightedParticleState.initialize(lat0=[20.0], lon0=[110.0])

        rng_for_step = np.random.default_rng(123)
        new_state = backward_euler_maruyama_step(state, T0, DT, ocean, LAT, LON, k_field, rng_for_step)

        # Independently reproduce the exact same draw with a fresh, identically-seeded RNG.
        rng_reference = np.random.default_rng(123)
        xi = rng_reference.standard_normal(size=(1, 2))
        scale = np.sqrt(2.0 * 50.0 * DT)
        dx_m, dy_m = scale * xi[0, 0], scale * xi[0, 1]
        dlon_per_deg = metres_per_degree_longitude(20.0)
        dlat_per_deg = metres_per_degree_latitude()
        expected_lon = 110.0 + dx_m / dlon_per_deg
        expected_lat = 20.0 + dy_m / dlat_per_deg

        self.assertTrue(new_state.valid[0])
        self.assertAlmostEqual(new_state.lon[0], expected_lon, places=9)
        self.assertAlmostEqual(new_state.lat[0], expected_lat, places=9)
        self.assertEqual(new_state.log_weight[0], 0.0)  # div(U) = 0 for a constant field


class TestBackwardTransportDirection(unittest.TestCase):
    """Constant eastward ocean velocity, K=0: the backward step must move the particle
    WEST (opposite the physical/forward velocity) -- confirms `-U`, never `+U`."""

    def test_backward_step_moves_opposite_to_forward_velocity(self):
        ocean = make_ocean(np.full((3, 3), 1.0), ZERO_UV)  # uo = +1 m/s east, vo = 0
        state = WeightedParticleState.initialize(lat0=[20.0], lon0=[110.0])
        rng = np.random.default_rng(1)
        new_state = backward_euler_maruyama_step(state, T0, DT, ocean, LAT, LON, ZERO_K, rng)

        self.assertTrue(new_state.valid[0])
        self.assertLess(new_state.lon[0], 110.0)
        self.assertEqual(new_state.lat[0], 20.0)


class TestMetreToDegreeConversion(unittest.TestCase):
    """A deterministic (K=0) pure-drift step: the resulting degree displacement must
    exactly match `iste.geometry`'s own conversion of the known metre displacement."""

    def test_hand_computable_displacement_matches_geometry_conversion(self):
        u_speed = 2.5  # m/s eastward
        v_speed = -1.0  # m/s southward
        ocean = make_ocean(np.full((3, 3), u_speed), np.full((3, 3), v_speed))
        state = WeightedParticleState.initialize(lat0=[20.0], lon0=[110.0])
        rng = np.random.default_rng(2)
        new_state = backward_euler_maruyama_step(state, T0, DT, ocean, LAT, LON, ZERO_K, rng)

        expected_dx_m = -u_speed * DT
        expected_dy_m = -v_speed * DT
        dlon_per_deg = metres_per_degree_longitude(20.0)
        dlat_per_deg = metres_per_degree_latitude()
        expected_lon = 110.0 + expected_dx_m / dlon_per_deg
        expected_lat = 20.0 + expected_dy_m / dlat_per_deg

        self.assertAlmostEqual(new_state.lon[0], expected_lon, places=9)
        self.assertAlmostEqual(new_state.lat[0], expected_lat, places=9)


class TestGradKDrift(unittest.TestCase):
    """K linear in longitude (`K(lon) = C * (lon - 110)`, so K = 0 exactly at the query
    point lon=110 -- the diffusion term is then exactly zero there too, isolating the
    `grad(K)` drift). Analytically `dK/d(lon_deg) = C` exactly (linear field), so the
    expected displacement is fully deterministic."""

    def test_drift_moves_up_the_gradient_with_correct_magnitude(self):
        C = 100.0
        k_field = C * (LON[None, :] - 110.0) * np.ones((3, 1))
        ocean = make_ocean(ZERO_UV, ZERO_UV)  # U = 0: isolate grad(K) alone
        state = WeightedParticleState.initialize(lat0=[20.0], lon0=[110.0])
        rng = np.random.default_rng(3)
        new_state = backward_euler_maruyama_step(state, T0, DT, ocean, LAT, LON, k_field, rng)

        dlon_per_deg = metres_per_degree_longitude(20.0)
        dK_dx = C / dlon_per_deg
        expected_dx_m = dK_dx * DT  # (-U + grad(K)) with U=0
        expected_lon = 110.0 + expected_dx_m / dlon_per_deg

        self.assertGreater(new_state.lon[0], 110.0)  # moved toward higher K (up-gradient)
        self.assertAlmostEqual(new_state.lon[0], expected_lon, places=9)
        self.assertEqual(new_state.lat[0], 20.0)


class TestDivergenceWeightUpdate(unittest.TestCase):
    def test_zero_divergence_leaves_weight_unchanged(self):
        # spatially uniform U -> div(U) = 0 exactly.
        ocean = make_ocean(np.full((3, 3), 0.4), np.full((3, 3), -0.2))
        state = WeightedParticleState.initialize(lat0=[20.0], lon0=[110.0], log_weight0=1.5)
        rng = np.random.default_rng(4)
        new_state = backward_euler_maruyama_step(state, T0, DT, ocean, LAT, LON, ZERO_K, rng)
        self.assertAlmostEqual(new_state.log_weight[0], 1.5)

    def test_convergent_flow_changes_weight_with_correct_sign_and_magnitude(self):
        # uo = -C * lon -> du/dx < 0 (convergent flow): div(U) = du/dx (v=0).
        C = 0.001
        uo_grid = -C * LON[None, :] * np.ones((3, 1))
        ocean = make_ocean(uo_grid, ZERO_UV)
        state = WeightedParticleState.initialize(lat0=[20.0], lon0=[110.0])
        rng = np.random.default_rng(6)
        new_state = backward_euler_maruyama_step(state, T0, DT, ocean, LAT, LON, ZERO_K, rng)

        dlon_per_deg = metres_per_degree_longitude(20.0)
        expected_div = -C / dlon_per_deg  # negative: convergent
        self.assertLess(expected_div, 0.0)
        expected_log_weight = -expected_div * DT  # log_w_new = log_w_old(0) - div(U)*dt
        self.assertGreater(expected_log_weight, 0.0)  # weight increases for convergent flow
        self.assertAlmostEqual(new_state.log_weight[0], expected_log_weight, places=6)

    def test_divergent_flow_changes_weight_with_opposite_sign(self):
        C = 0.001
        uo_grid = C * LON[None, :] * np.ones((3, 1))  # du/dx > 0: divergent
        ocean = make_ocean(uo_grid, ZERO_UV)
        state = WeightedParticleState.initialize(lat0=[20.0], lon0=[110.0])
        rng = np.random.default_rng(6)
        new_state = backward_euler_maruyama_step(state, T0, DT, ocean, LAT, LON, ZERO_K, rng)
        self.assertLess(new_state.log_weight[0], 0.0)  # weight decreases for divergent flow


class TestReproducibility(unittest.TestCase):
    def test_same_seed_gives_identical_step(self):
        ocean = make_ocean(ZERO_UV, ZERO_UV)
        k_field = np.full((3, 3), 20.0)
        out_a = backward_euler_maruyama_step(
            WeightedParticleState.initialize(lat0=[20.0], lon0=[110.0]),
            T0, DT, ocean, LAT, LON, k_field, np.random.default_rng(7),
        )
        out_b = backward_euler_maruyama_step(
            WeightedParticleState.initialize(lat0=[20.0], lon0=[110.0]),
            T0, DT, ocean, LAT, LON, k_field, np.random.default_rng(7),
        )
        self.assertEqual(out_a.lat[0], out_b.lat[0])
        self.assertEqual(out_a.lon[0], out_b.lon[0])

    def test_different_seed_gives_different_step(self):
        ocean = make_ocean(ZERO_UV, ZERO_UV)
        k_field = np.full((3, 3), 20.0)
        out_a = backward_euler_maruyama_step(
            WeightedParticleState.initialize(lat0=[20.0], lon0=[110.0]),
            T0, DT, ocean, LAT, LON, k_field, np.random.default_rng(7),
        )
        out_b = backward_euler_maruyama_step(
            WeightedParticleState.initialize(lat0=[20.0], lon0=[110.0]),
            T0, DT, ocean, LAT, LON, k_field, np.random.default_rng(8),
        )
        self.assertNotEqual(out_a.lat[0], out_b.lat[0])


class TestInvalidDestination(unittest.TestCase):
    def test_step_into_land_marks_invalid_and_freezes_state(self):
        # deterministic: a strong westward-forcing eastward ocean velocity (so the
        # backward step pushes hard west) straight into a masked "land" cell, K=0.
        uo_grid = np.full((3, 3), 50.0)  # +50 m/s east -> backward drift is strongly west
        uo_grid[0, 0] = np.nan  # land at (lat=10, lon=100)
        ocean = make_ocean(uo_grid, ZERO_UV)
        state = WeightedParticleState.initialize(lat0=[10.0], lon0=[110.0], log_weight0=0.7)
        rng = np.random.default_rng(9)
        new_state = backward_euler_maruyama_step(state, T0, DT * 20, ocean, LAT, LON, ZERO_K, rng)

        self.assertFalse(new_state.valid[0])
        self.assertEqual(new_state.lat[0], 10.0)
        self.assertEqual(new_state.lon[0], 110.0)
        self.assertEqual(new_state.log_weight[0], 0.7)  # frozen, not updated


class TestAlreadyInvalidParticle(unittest.TestCase):
    def test_already_invalid_particle_never_moves(self):
        ocean = make_ocean(np.full((3, 3), 5.0), np.full((3, 3), 5.0))
        state = WeightedParticleState(
            lat=np.array([15.0]), lon=np.array([105.0]), valid=np.array([False]), log_weight=np.array([2.5])
        )
        rng = np.random.default_rng(10)
        new_state = backward_euler_maruyama_step(state, T0, DT, ocean, LAT, LON, np.full((3, 3), 30.0), rng)

        self.assertFalse(new_state.valid[0])
        self.assertEqual(new_state.lat[0], 15.0)
        self.assertEqual(new_state.lon[0], 105.0)
        self.assertEqual(new_state.log_weight[0], 2.5)


class TestPhysicsUnavailableAtCurrentPosition(unittest.TestCase):
    def test_missing_k_at_current_position_marks_particle_invalid(self):
        ocean = make_ocean(ZERO_UV, ZERO_UV)
        k_field = np.full((3, 3), 10.0)
        k_field[1, 1] = np.nan  # (lat=20, lon=110) -- exactly the particle's own cell
        state = WeightedParticleState.initialize(lat0=[20.0], lon0=[110.0], log_weight0=0.3)
        rng = np.random.default_rng(11)
        new_state = backward_euler_maruyama_step(state, T0, DT, ocean, LAT, LON, k_field, rng)

        self.assertFalse(new_state.valid[0])
        self.assertEqual(new_state.lat[0], 20.0)
        self.assertEqual(new_state.lon[0], 110.0)
        self.assertEqual(new_state.log_weight[0], 0.3)


class TestInputValidation(unittest.TestCase):
    def test_non_positive_dt_raises(self):
        ocean = make_ocean(ZERO_UV, ZERO_UV)
        state = WeightedParticleState.initialize(lat0=[20.0], lon0=[110.0])
        rng = np.random.default_rng(0)
        with self.assertRaises(ValueError):
            backward_euler_maruyama_step(state, T0, 0.0, ocean, LAT, LON, ZERO_K, rng)
        with self.assertRaises(ValueError):
            backward_euler_maruyama_step(state, T0, -60.0, ocean, LAT, LON, ZERO_K, rng)


if __name__ == "__main__":
    unittest.main()
