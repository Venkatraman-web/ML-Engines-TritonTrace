"""Focused T20 checks for `iste.forward_transport` (Phase 2 forward Euler-Maruyama
transport operator).

Run with:
    python -m unittest tests.test_forward_transport -v
"""

from __future__ import annotations

import unittest

import numpy as np

from iste.environment import GriddedFieldSource
from iste.geometry import metres_per_degree_latitude, metres_per_degree_longitude
from iste.forward_transport import ForwardParticleState, forward_euler_maruyama_step

LAT = np.array([10.0, 20.0, 30.0])
LON = np.array([100.0, 110.0, 120.0])
TIME = np.array(["2020-01-01T00:00", "2020-01-01T01:00"], dtype="datetime64[ns]")
T0 = TIME[0]
DT = 900.0  # 15 minutes, seconds

ZERO_UV = np.zeros((3, 3))
ZERO_K = np.zeros((3, 3))


def make_ocean(uo_grid: np.ndarray, vo_grid: np.ndarray) -> GriddedFieldSource:
    """Two identical hourly snapshots of the given (n_lat, n_lon) uo/vo fields --
    standing in for an already-perturbed U_e source (e.g.
    `iste.ensemble_run.make_perturbed_source`'s output), which this module treats no
    differently from a plain base source."""
    uo = np.stack([uo_grid, uo_grid])
    vo = np.stack([vo_grid, vo_grid])
    return GriddedFieldSource(
        source_path="<synthetic U_e>",
        time_name="time",
        time=TIME,
        lat_name="lat",
        lat=LAT,
        lon_name="lon",
        lon=LON,
        data={"uo": uo, "vo": vo},
        fill_threshold=1e19,
    )


class TestZeroVelocityZeroDiffusion(unittest.TestCase):
    """Requirement 1: zero velocity + K=0 -> position unchanged exactly."""

    def test_position_unchanged(self):
        ocean = make_ocean(ZERO_UV, ZERO_UV)
        state = ForwardParticleState.initialize(lat0=[20.0], lon0=[110.0], forcing_member_id0=[3])
        rng = np.random.default_rng(0)
        new_state = forward_euler_maruyama_step(state, T0, DT, ocean, LAT, LON, ZERO_K, rng)

        self.assertTrue(new_state.valid[0])
        self.assertEqual(new_state.lat[0], 20.0)
        self.assertEqual(new_state.lon[0], 110.0)


class TestForwardAdvectionSignAndMagnitude(unittest.TestCase):
    """Requirement 2: constant positive velocity -> correct forward displacement and
    sign (opposite direction from what T05's backward step would produce)."""

    def test_eastward_current_moves_particle_east(self):
        ocean = make_ocean(np.full((3, 3), 1.0), ZERO_UV)  # +1 m/s east
        state = ForwardParticleState.initialize(lat0=[20.0], lon0=[110.0], forcing_member_id0=[0])
        rng = np.random.default_rng(1)
        new_state = forward_euler_maruyama_step(state, T0, DT, ocean, LAT, LON, ZERO_K, rng)

        self.assertGreater(new_state.lon[0], 110.0)  # +U -> moves EAST (not west, per T05)
        self.assertEqual(new_state.lat[0], 20.0)

    def test_hand_computable_displacement_matches_geometry_conversion(self):
        u_speed = 2.5  # m/s eastward
        v_speed = -1.0  # m/s southward
        ocean = make_ocean(np.full((3, 3), u_speed), np.full((3, 3), v_speed))
        state = ForwardParticleState.initialize(lat0=[20.0], lon0=[110.0], forcing_member_id0=[0])
        rng = np.random.default_rng(2)
        new_state = forward_euler_maruyama_step(state, T0, DT, ocean, LAT, LON, ZERO_K, rng)

        expected_dx_m = u_speed * DT  # +U, never -U
        expected_dy_m = v_speed * DT
        dlon_per_deg = metres_per_degree_longitude(20.0)
        dlat_per_deg = metres_per_degree_latitude()
        expected_lon = 110.0 + expected_dx_m / dlon_per_deg
        expected_lat = 20.0 + expected_dy_m / dlat_per_deg

        self.assertAlmostEqual(new_state.lon[0], expected_lon, places=9)
        self.assertAlmostEqual(new_state.lat[0], expected_lat, places=9)


class TestDiffusionReproducibility(unittest.TestCase):
    """Requirements 3-4: fixed seed reproducible; different seeds can differ."""

    def test_same_seed_gives_identical_step(self):
        ocean = make_ocean(ZERO_UV, ZERO_UV)
        k_field = np.full((3, 3), 50.0)
        out_a = forward_euler_maruyama_step(
            ForwardParticleState.initialize(lat0=[20.0], lon0=[110.0], forcing_member_id0=[0]),
            T0, DT, ocean, LAT, LON, k_field, np.random.default_rng(7),
        )
        out_b = forward_euler_maruyama_step(
            ForwardParticleState.initialize(lat0=[20.0], lon0=[110.0], forcing_member_id0=[0]),
            T0, DT, ocean, LAT, LON, k_field, np.random.default_rng(7),
        )
        self.assertEqual(out_a.lat[0], out_b.lat[0])
        self.assertEqual(out_a.lon[0], out_b.lon[0])

    def test_matches_exact_stochastic_displacement(self):
        ocean = make_ocean(ZERO_UV, ZERO_UV)
        k_field = np.full((3, 3), 50.0)
        state = ForwardParticleState.initialize(lat0=[20.0], lon0=[110.0], forcing_member_id0=[0])

        rng_for_step = np.random.default_rng(123)
        new_state = forward_euler_maruyama_step(state, T0, DT, ocean, LAT, LON, k_field, rng_for_step)

        rng_reference = np.random.default_rng(123)
        xi = rng_reference.standard_normal(size=(1, 2))
        scale = np.sqrt(2.0 * 50.0 * DT)
        dx_m, dy_m = scale * xi[0, 0], scale * xi[0, 1]
        dlon_per_deg = metres_per_degree_longitude(20.0)
        dlat_per_deg = metres_per_degree_latitude()
        expected_lon = 110.0 + dx_m / dlon_per_deg
        expected_lat = 20.0 + dy_m / dlat_per_deg

        self.assertAlmostEqual(new_state.lon[0], expected_lon, places=9)
        self.assertAlmostEqual(new_state.lat[0], expected_lat, places=9)

    def test_different_seed_gives_different_step(self):
        ocean = make_ocean(ZERO_UV, ZERO_UV)
        k_field = np.full((3, 3), 20.0)
        out_a = forward_euler_maruyama_step(
            ForwardParticleState.initialize(lat0=[20.0], lon0=[110.0], forcing_member_id0=[0]),
            T0, DT, ocean, LAT, LON, k_field, np.random.default_rng(7),
        )
        out_b = forward_euler_maruyama_step(
            ForwardParticleState.initialize(lat0=[20.0], lon0=[110.0], forcing_member_id0=[0]),
            T0, DT, ocean, LAT, LON, k_field, np.random.default_rng(8),
        )
        self.assertNotEqual(out_a.lat[0], out_b.lat[0])


class TestKZeroRemovesStochasticDisplacement(unittest.TestCase):
    """Requirement 5: K=0 removes the stochastic term -- displacement is pure drift,
    identical regardless of RNG seed."""

    def test_k_zero_gives_seed_independent_result(self):
        ocean = make_ocean(np.full((3, 3), 0.5), np.full((3, 3), -0.3))
        out_a = forward_euler_maruyama_step(
            ForwardParticleState.initialize(lat0=[20.0], lon0=[110.0], forcing_member_id0=[0]),
            T0, DT, ocean, LAT, LON, ZERO_K, np.random.default_rng(1),
        )
        out_b = forward_euler_maruyama_step(
            ForwardParticleState.initialize(lat0=[20.0], lon0=[110.0], forcing_member_id0=[0]),
            T0, DT, ocean, LAT, LON, ZERO_K, np.random.default_rng(999),
        )
        self.assertEqual(out_a.lat[0], out_b.lat[0])
        self.assertEqual(out_a.lon[0], out_b.lon[0])


class TestMetreToDegreeConversion(unittest.TestCase):
    """Requirement 6: correct metre-to-lat/lon conversion (deterministic, K=0)."""

    def test_conversion_matches_geometry_module_exactly(self):
        u_speed, v_speed = 3.0, 1.5
        ocean = make_ocean(np.full((3, 3), u_speed), np.full((3, 3), v_speed))
        state = ForwardParticleState.initialize(lat0=[20.0], lon0=[110.0], forcing_member_id0=[0])
        rng = np.random.default_rng(4)
        new_state = forward_euler_maruyama_step(state, T0, DT, ocean, LAT, LON, ZERO_K, rng)

        dlon_per_deg = metres_per_degree_longitude(20.0)
        dlat_per_deg = metres_per_degree_latitude()
        expected_lon = 110.0 + (u_speed * DT) / dlon_per_deg
        expected_lat = 20.0 + (v_speed * DT) / dlat_per_deg
        self.assertAlmostEqual(new_state.lon[0], expected_lon, places=9)
        self.assertAlmostEqual(new_state.lat[0], expected_lat, places=9)


class TestInvalidAndDomainExitHandling(unittest.TestCase):
    """Requirement 7: invalid/domain-exit handling."""

    def test_step_into_land_marks_invalid_and_freezes_position(self):
        uo_grid = np.full((3, 3), 50.0)  # strong eastward forward drift
        uo_grid[1, 2] = np.nan  # land at (lat=20, lon=120) -- straight ahead
        ocean = make_ocean(uo_grid, ZERO_UV)
        state = ForwardParticleState.initialize(lat0=[20.0], lon0=[110.0], forcing_member_id0=[5])
        rng = np.random.default_rng(9)
        new_state = forward_euler_maruyama_step(state, T0, DT * 20, ocean, LAT, LON, ZERO_K, rng)

        self.assertFalse(new_state.valid[0])
        self.assertEqual(new_state.lat[0], 20.0)
        self.assertEqual(new_state.lon[0], 110.0)  # frozen at last valid position
        self.assertEqual(new_state.forcing_member_id[0], 5)  # preserved even when newly invalid

    def test_already_invalid_particle_never_moves_and_is_never_silently_revalidated(self):
        ocean = make_ocean(np.full((3, 3), 5.0), np.full((3, 3), 5.0))
        state = ForwardParticleState(
            lat=np.array([15.0]), lon=np.array([105.0]), valid=np.array([False]),
            forcing_member_id=np.array([2]),
        )
        rng = np.random.default_rng(10)
        new_state = forward_euler_maruyama_step(state, T0, DT, ocean, LAT, LON, np.full((3, 3), 30.0), rng)

        self.assertFalse(new_state.valid[0])
        self.assertEqual(new_state.lat[0], 15.0)
        self.assertEqual(new_state.lon[0], 105.0)
        self.assertEqual(new_state.forcing_member_id[0], 2)

    def test_missing_k_at_current_position_marks_particle_invalid(self):
        ocean = make_ocean(ZERO_UV, ZERO_UV)
        k_field = np.full((3, 3), 10.0)
        k_field[1, 1] = np.nan  # (lat=20, lon=110) -- exactly the particle's own cell
        state = ForwardParticleState.initialize(lat0=[20.0], lon0=[110.0], forcing_member_id0=[0])
        rng = np.random.default_rng(11)
        new_state = forward_euler_maruyama_step(state, T0, DT, ocean, LAT, LON, k_field, rng)

        self.assertFalse(new_state.valid[0])
        self.assertEqual(new_state.lat[0], 20.0)
        self.assertEqual(new_state.lon[0], 110.0)

    def test_mixed_valid_invalid_batch_only_valid_ones_step(self):
        ocean = make_ocean(np.full((3, 3), 1.0), ZERO_UV)
        state = ForwardParticleState(
            lat=np.array([20.0, 20.0]), lon=np.array([110.0, 110.0]),
            valid=np.array([True, False]), forcing_member_id=np.array([0, 1]),
        )
        rng = np.random.default_rng(12)
        new_state = forward_euler_maruyama_step(state, T0, DT, ocean, LAT, LON, ZERO_K, rng)

        self.assertTrue(new_state.valid[0])
        self.assertGreater(new_state.lon[0], 110.0)
        self.assertFalse(new_state.valid[1])
        self.assertEqual(new_state.lon[1], 110.0)  # untouched


class TestNoGradKDrift(unittest.TestCase):
    """Requirement 8: grad(K) is not used -- a spatially-varying K with zero velocity at
    the query point (K=0 exactly there too) must produce PURE stochastic displacement
    with no deterministic drift, unlike T05's backward step (which would drift up the
    gradient)."""

    def test_linear_k_field_produces_no_deterministic_drift(self):
        C = 100.0
        k_field = C * (LON[None, :] - 110.0) * np.ones((3, 1))  # K=0 exactly at lon=110
        ocean = make_ocean(ZERO_UV, ZERO_UV)  # U = 0 too
        state = ForwardParticleState.initialize(lat0=[20.0], lon0=[110.0], forcing_member_id0=[0])
        rng = np.random.default_rng(3)
        new_state = forward_euler_maruyama_step(state, T0, DT, ocean, LAT, LON, k_field, rng)

        # K(20, 110) = 0 exactly -> diffusion_scale = 0 there too -> NO displacement at
        # all (if grad(K) were included, as in T05, the particle would move toward
        # higher K; here it must not move).
        self.assertEqual(new_state.lon[0], 110.0)
        self.assertEqual(new_state.lat[0], 20.0)


class TestForcingMemberIdPreserved(unittest.TestCase):
    """Requirement 9: forcing_member_id preserved (for valid, invalid, and mixed
    multi-particle batches)."""

    def test_preserved_across_a_step_for_multiple_particles(self):
        ocean = make_ocean(np.full((3, 3), 0.5), ZERO_UV)
        state = ForwardParticleState.initialize(
            lat0=[15.0, 25.0, 12.0], lon0=[105.0, 115.0, 101.0], forcing_member_id0=[0, 7, 42],
        )
        rng = np.random.default_rng(13)
        new_state = forward_euler_maruyama_step(state, T0, DT, ocean, LAT, LON, ZERO_K, rng)
        np.testing.assert_array_equal(new_state.forcing_member_id, [0, 7, 42])

    def test_preserved_even_for_a_freshly_invalidated_particle(self):
        uo_grid = np.full((3, 3), 50.0)
        uo_grid[1, 2] = np.nan
        ocean = make_ocean(uo_grid, ZERO_UV)
        state = ForwardParticleState.initialize(lat0=[20.0], lon0=[110.0], forcing_member_id0=[99])
        rng = np.random.default_rng(9)
        new_state = forward_euler_maruyama_step(state, T0, DT * 20, ocean, LAT, LON, ZERO_K, rng)
        self.assertFalse(new_state.valid[0])
        self.assertEqual(new_state.forcing_member_id[0], 99)


class TestControlledSyntheticCase(unittest.TestCase):
    """Requirement 10: a controlled synthetic case matches the expected forward step
    exactly (deterministic drift + a hand-verified stochastic draw combined)."""

    def test_combined_drift_and_diffusion_matches_hand_computation(self):
        u_speed, v_speed, K = 1.2, -0.4, 8.0
        ocean = make_ocean(np.full((3, 3), u_speed), np.full((3, 3), v_speed))
        k_field = np.full((3, 3), K)
        state = ForwardParticleState.initialize(lat0=[20.0], lon0=[110.0], forcing_member_id0=[0])

        rng_for_step = np.random.default_rng(55)
        new_state = forward_euler_maruyama_step(state, T0, DT, ocean, LAT, LON, k_field, rng_for_step)

        rng_reference = np.random.default_rng(55)
        xi = rng_reference.standard_normal(size=(1, 2))
        scale = np.sqrt(2.0 * K * DT)
        expected_dx_m = u_speed * DT + scale * xi[0, 0]
        expected_dy_m = v_speed * DT + scale * xi[0, 1]
        dlon_per_deg = metres_per_degree_longitude(20.0)
        dlat_per_deg = metres_per_degree_latitude()
        expected_lon = 110.0 + expected_dx_m / dlon_per_deg
        expected_lat = 20.0 + expected_dy_m / dlat_per_deg

        self.assertAlmostEqual(new_state.lon[0], expected_lon, places=9)
        self.assertAlmostEqual(new_state.lat[0], expected_lat, places=9)
        self.assertTrue(new_state.valid[0])


class TestInputValidation(unittest.TestCase):
    def test_non_positive_dt_raises(self):
        ocean = make_ocean(ZERO_UV, ZERO_UV)
        state = ForwardParticleState.initialize(lat0=[20.0], lon0=[110.0], forcing_member_id0=[0])
        rng = np.random.default_rng(0)
        with self.assertRaises(ValueError):
            forward_euler_maruyama_step(state, T0, 0.0, ocean, LAT, LON, ZERO_K, rng)
        with self.assertRaises(ValueError):
            forward_euler_maruyama_step(state, T0, -60.0, ocean, LAT, LON, ZERO_K, rng)


if __name__ == "__main__":
    unittest.main()
