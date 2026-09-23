"""Focused T04 checks for `iste.boundary` (Stage III stop-on-invalid boundary policy).

Run with:
    python -m unittest tests.test_boundary -v
"""

from __future__ import annotations

import unittest

import numpy as np

from iste.boundary import ParticleState, apply_boundary_policy, netcdf_validity_check
from iste.environment import GriddedFieldSource
from iste.interpolation import interpolate_field


def synthetic_domain_validity(lat, lon) -> np.ndarray:
    """A tiny hand-defined domain for tests: ocean is `[0, 10] x [0, 10]`, with a "land"
    square `[4, 6] x [4, 6]` carved out of the middle."""
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    in_domain = (lat >= 0.0) & (lat <= 10.0) & (lon >= 0.0) & (lon <= 10.0)
    is_land = (lat >= 4.0) & (lat <= 6.0) & (lon >= 4.0) & (lon <= 6.0)
    return in_domain & ~is_land


class TestParticleInValidOcean(unittest.TestCase):
    def test_particle_in_open_water_moves_and_stays_valid(self):
        state = ParticleState.initialize(lat0=[1.0], lon0=[1.0])
        new_state = apply_boundary_policy(state, [2.0], [2.0], synthetic_domain_validity)
        self.assertTrue(new_state.valid[0])
        self.assertEqual(new_state.lat[0], 2.0)
        self.assertEqual(new_state.lon[0], 2.0)


class TestParticleHittingLand(unittest.TestCase):
    def test_particle_entering_land_becomes_invalid_and_stops(self):
        state = ParticleState.initialize(lat0=[3.0], lon0=[3.0])
        new_state = apply_boundary_policy(state, [5.0], [5.0], synthetic_domain_validity)
        self.assertFalse(new_state.valid[0])
        # propagation stopped -> position stays at the last VALID location, not the
        # proposed (invalid) one.
        self.assertEqual(new_state.lat[0], 3.0)
        self.assertEqual(new_state.lon[0], 3.0)


class TestParticleLeavingDomain(unittest.TestCase):
    def test_particle_leaving_the_spatial_domain_becomes_invalid_and_stops(self):
        state = ParticleState.initialize(lat0=[9.0], lon0=[9.0])
        new_state = apply_boundary_policy(state, [11.0], [11.0], synthetic_domain_validity)
        self.assertFalse(new_state.valid[0])
        self.assertEqual(new_state.lat[0], 9.0)
        self.assertEqual(new_state.lon[0], 9.0)


class TestOnceInvalidStaysInvalid(unittest.TestCase):
    def test_further_steps_on_an_invalid_particle_are_no_ops(self):
        state = ParticleState.initialize(lat0=[3.0], lon0=[3.0])
        after_land = apply_boundary_policy(state, [5.0], [5.0], synthetic_domain_validity)
        self.assertFalse(after_land.valid[0])

        # Propose a perfectly valid next location -- must still be ignored: once
        # invalid, always invalid, and the particle must not move at all.
        after_second_attempt = apply_boundary_policy(
            after_land, [1.0], [1.0], synthetic_domain_validity
        )
        self.assertFalse(after_second_attempt.valid[0])
        self.assertEqual(after_second_attempt.lat[0], 3.0)
        self.assertEqual(after_second_attempt.lon[0], 3.0)

        # ... and this must remain true across further repeated attempts.
        again = apply_boundary_policy(after_second_attempt, [7.0], [7.0], synthetic_domain_validity)
        self.assertFalse(again.valid[0])
        self.assertEqual(again.lat[0], 3.0)


class TestInvalidParticlesRetainedForDiagnostics(unittest.TestCase):
    def test_invalid_particles_are_never_removed_from_the_state(self):
        state = ParticleState.initialize(lat0=[1.0, 3.0], lon0=[1.0, 3.0])
        after = apply_boundary_policy(state, [2.0, 5.0], [2.0, 5.0], synthetic_domain_validity)
        self.assertEqual(after.n_particles, 2)  # still 2 particles, none dropped
        self.assertTrue(after.valid[0])
        self.assertFalse(after.valid[1])

    def test_diagnostic_properties(self):
        state = ParticleState(
            lat=np.array([0.0, 1.0, 2.0, 3.0]),
            lon=np.array([0.0, 1.0, 2.0, 3.0]),
            valid=np.array([True, True, False, False]),
        )
        self.assertEqual(state.valid_count, 2)
        self.assertEqual(state.invalid_count, 2)
        self.assertAlmostEqual(state.valid_particle_fraction, 0.5)


class TestVectorizedMultiParticleStep(unittest.TestCase):
    def test_four_particles_independent_outcomes_in_one_call(self):
        # particle 0: moves normally; particle 1: hits land; particle 2: leaves domain;
        # particle 3: already invalid before this call.
        state = ParticleState(
            lat=np.array([1.0, 3.0, 9.0, 3.0]),
            lon=np.array([1.0, 3.0, 9.0, 3.0]),
            valid=np.array([True, True, True, False]),
        )
        proposed_lat = np.array([2.0, 5.0, 11.0, 8.0])
        proposed_lon = np.array([2.0, 5.0, 11.0, 8.0])
        new_state = apply_boundary_policy(state, proposed_lat, proposed_lon, synthetic_domain_validity)

        np.testing.assert_array_equal(new_state.valid, [True, False, False, False])
        np.testing.assert_allclose(new_state.lat, [2.0, 3.0, 9.0, 3.0])
        np.testing.assert_allclose(new_state.lon, [2.0, 3.0, 9.0, 3.0])
        self.assertEqual(new_state.valid_count, 1)
        self.assertEqual(new_state.invalid_count, 3)


class TestErrorHandling(unittest.TestCase):
    def test_mismatched_proposed_shape_raises(self):
        state = ParticleState.initialize(lat0=[1.0, 2.0], lon0=[1.0, 2.0])
        with self.assertRaises(ValueError):
            apply_boundary_policy(state, [2.0], [2.0], synthetic_domain_validity)

    def test_initialize_requires_matching_shapes(self):
        with self.assertRaises(ValueError):
            ParticleState.initialize(lat0=[1.0, 2.0], lon0=[1.0])


class TestNetCDFValidityCheckReusesT02(unittest.TestCase):
    """`netcdf_validity_check` must not redefine validity -- it must agree exactly with
    `iste.interpolation.interpolate_field`'s own validity output."""

    def setUp(self):
        lat = np.array([10.0, 20.0, 30.0])
        lon = np.array([100.0, 110.0])
        field = np.array([[1.0, np.nan], [3.0, 4.0], [5.0, 6.0]])  # one masked "land" cell
        time = np.array(["2020-01-01T00:00", "2020-01-01T01:00"], dtype="datetime64[ns]")
        self.source = GriddedFieldSource(
            source_path="<synthetic>",
            time_name="time",
            time=time,
            lat_name="lat",
            lat=lat,
            lon_name="lon",
            lon=lon,
            data={"test_var": np.stack([field, field])},
            fill_threshold=1e19,
        )
        self.t0 = time[0]

    def test_agrees_with_interpolate_field_at_a_range_of_points(self):
        check = netcdf_validity_check(self.source, "test_var", self.t0)
        lat_q = np.array([20.0, 10.0, 15.0, 5.0])  # valid, touches-NaN, interior-ok, out-of-domain
        lon_q = np.array([105.0, 105.0, 105.0, 105.0])

        checked = check(lat_q, lon_q)
        _, expected = interpolate_field(self.source, "test_var", lat_q, lon_q, self.t0)
        np.testing.assert_array_equal(checked, expected)

    def test_land_cell_makes_particle_invalid_via_boundary_policy(self):
        check = netcdf_validity_check(self.source, "test_var", self.t0)
        state = ParticleState.initialize(lat0=[20.0], lon0=[105.0])  # away from the NaN corner
        # propose a step whose bracket touches the masked (lat=10, lon=110) corner
        new_state = apply_boundary_policy(state, [10.0], [105.0], check)
        self.assertFalse(new_state.valid[0])
        self.assertEqual(new_state.lat[0], 20.0)  # stayed at the last valid position


if __name__ == "__main__":
    unittest.main()
