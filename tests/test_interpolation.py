"""Focused T02 checks for `iste.interpolation` (+ the T02 wiring in `iste.environment`).

Run with:
    python -m unittest tests.test_interpolation -v
"""

from __future__ import annotations

import os
import unittest

import numpy as np

from iste.environment import GriddedFieldSource, NetCDFEnvironmentalFieldProvider, load_field_dataset
from iste.geometry import metres_per_degree_longitude
from iste.interpolation import (
    bilinear_spatial_interpolate,
    interpolate_field,
    linear_temporal_interpolate,
    time_bracket,
)
from iste.step import WeightedParticleState, backward_euler_maruyama_step

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OCEAN_PATH = os.path.join(REPO_ROOT, "data", "ocean.nc")
WIND_PATH = os.path.join(REPO_ROOT, "data", "wind.nc")

# Hand-computable synthetic grid used throughout:
#   lat = [10, 20, 30] (ascending), lon = [100, 110]
#   field =  lat=10: [1, 2]
#            lat=20: [3, 4]
#            lat=30: [5, 6]
# Query (lat=15, lon=105) sits exactly halfway in both axes between the four corners
# {1, 2, 3, 4}, so the expected bilinear value is their plain average: 2.5.
_LAT_ASC = np.array([10.0, 20.0, 30.0])
_LON = np.array([100.0, 110.0])
_FIELD_ASC = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])


class TestBilinearSpatialInterpolateSynthetic(unittest.TestCase):
    def test_interior_point_ascending_latitude(self):
        values, valid = bilinear_spatial_interpolate(_LAT_ASC, _LON, _FIELD_ASC, 15.0, 105.0)
        self.assertTrue(valid[0])
        self.assertAlmostEqual(values[0], 2.5)

    def test_interior_point_descending_latitude_same_physical_field(self):
        # Same field, described with descending latitude (like the wind grid): row order
        # is flipped to match, but the physical answer at the same (lat, lon) must match.
        lat_desc = _LAT_ASC[::-1]  # [30, 20, 10]
        field_desc = _FIELD_ASC[::-1, :]  # row for lat=30 first
        values, valid = bilinear_spatial_interpolate(lat_desc, _LON, field_desc, 15.0, 105.0)
        self.assertTrue(valid[0])
        self.assertAlmostEqual(values[0], 2.5)

    def test_exact_grid_point_returns_exact_value(self):
        values, valid = bilinear_spatial_interpolate(_LAT_ASC, _LON, _FIELD_ASC, 20.0, 110.0)
        self.assertTrue(valid[0])
        self.assertAlmostEqual(values[0], 4.0)

    def test_vectorized_query_points(self):
        lat_q = np.array([15.0, 20.0, 15.0])
        lon_q = np.array([105.0, 110.0, 100.0])
        values, valid = bilinear_spatial_interpolate(_LAT_ASC, _LON, _FIELD_ASC, lat_q, lon_q)
        np.testing.assert_allclose(values, [2.5, 4.0, 2.0])
        self.assertTrue(np.all(valid))

    def test_out_of_domain_query_is_invalid(self):
        for lat_q, lon_q in [(5.0, 105.0), (35.0, 105.0), (15.0, 90.0), (15.0, 120.0)]:
            values, valid = bilinear_spatial_interpolate(_LAT_ASC, _LON, _FIELD_ASC, lat_q, lon_q)
            self.assertFalse(valid[0], f"expected invalid at ({lat_q}, {lon_q})")
            self.assertTrue(np.isnan(values[0]))

    def test_missing_corner_cell_propagates_as_invalid(self):
        field = _FIELD_ASC.copy()
        field[0, 0] = np.nan  # the (lat=10, lon=100) corner
        # This query's 4 corners include the NaN cell -> must be invalid.
        values, valid = bilinear_spatial_interpolate(_LAT_ASC, _LON, field, 15.0, 105.0)
        self.assertFalse(valid[0])
        self.assertTrue(np.isnan(values[0]))
        # This query's 4 corners do NOT include the NaN cell -> must stay valid.
        values2, valid2 = bilinear_spatial_interpolate(_LAT_ASC, _LON, field, 25.0, 105.0)
        self.assertTrue(valid2[0])
        self.assertFalse(np.isnan(values2[0]))

    def test_non_monotonic_coords_rejected(self):
        with self.assertRaises(ValueError):
            bilinear_spatial_interpolate(np.array([10.0, 30.0, 20.0]), _LON, _FIELD_ASC, 15.0, 105.0)

    def test_different_resolution_grid_also_works(self):
        # A different-shaped grid needs no special-casing.
        lat5 = np.linspace(0.0, 40.0, 5)  # [0, 10, 20, 30, 40]
        lon4 = np.linspace(0.0, 30.0, 4)  # [0, 10, 20, 30]
        field = np.outer(lat5, np.ones(4)) + np.outer(np.ones(5), lon4)  # f = lat + lon
        values, valid = bilinear_spatial_interpolate(lat5, lon4, field, 15.0, 15.0)
        self.assertTrue(valid[0])
        self.assertAlmostEqual(values[0], 30.0)  # 15 + 15


class TestLinearTemporalInterpolateSynthetic(unittest.TestCase):
    def setUp(self):
        self.time = np.array(
            ["2020-01-01T00:00", "2020-01-01T01:00", "2020-01-01T02:00"], dtype="datetime64[ns]"
        )
        self.values = np.array([10.0, 20.0, 30.0])  # hourly

    def test_quarter_hour_interpolation(self):
        # hourly -> 15-minute: 00:15 is 0.25 of the way from 10.0 to 20.0.
        query = np.datetime64("2020-01-01T00:15")
        value, valid = linear_temporal_interpolate(self.time, self.values, query)
        self.assertTrue(bool(valid))
        self.assertAlmostEqual(float(value), 12.5)

    def test_three_quarter_hour_interpolation(self):
        query = np.datetime64("2020-01-01T00:45")
        value, valid = linear_temporal_interpolate(self.time, self.values, query)
        self.assertAlmostEqual(float(value), 17.5)

    def test_second_interval(self):
        query = np.datetime64("2020-01-01T01:30")
        value, valid = linear_temporal_interpolate(self.time, self.values, query)
        self.assertAlmostEqual(float(value), 25.0)

    def test_exact_timestamp(self):
        query = np.datetime64("2020-01-01T01:00")
        value, valid = linear_temporal_interpolate(self.time, self.values, query)
        self.assertAlmostEqual(float(value), 20.0)

    def test_out_of_range_query_is_invalid(self):
        for query in [np.datetime64("2019-12-31T23:00"), np.datetime64("2020-01-01T03:00")]:
            value, valid = linear_temporal_interpolate(self.time, self.values, query)
            self.assertFalse(bool(valid))
            self.assertTrue(np.isnan(float(value)))

    def test_missing_bracket_value_propagates_as_invalid(self):
        values = np.array([10.0, np.nan, 30.0])
        v1, ok1 = linear_temporal_interpolate(self.time, values, np.datetime64("2020-01-01T00:30"))
        self.assertFalse(bool(ok1))
        v2, ok2 = linear_temporal_interpolate(self.time, values, np.datetime64("2020-01-01T01:30"))
        self.assertFalse(bool(ok2))

    def test_vectorized_query_times(self):
        queries = np.array(
            ["2020-01-01T00:15", "2020-01-01T00:45", "2020-01-01T01:30"], dtype="datetime64[ns]"
        )
        values, valid = linear_temporal_interpolate(self.time, self.values, queries)
        np.testing.assert_allclose(values, [12.5, 17.5, 25.0])
        self.assertTrue(np.all(valid))

    def test_non_ascending_time_coords_rejected(self):
        bad_time = self.time[::-1]
        with self.assertRaises(ValueError):
            linear_temporal_interpolate(bad_time, self.values, np.datetime64("2020-01-01T00:15"))


class TestTimeBracket(unittest.TestCase):
    """`time_bracket` is the public wrapper T05 reuses to keep a second time-dependent
    quantity (div(U)) consistent with how `interpolate_field` time-interpolates `U`."""

    def setUp(self):
        self.time = np.array(
            ["2020-01-01T00:00", "2020-01-01T01:00", "2020-01-01T02:00"], dtype="datetime64[ns]"
        )

    def test_quarter_hour_matches_linear_temporal_interpolate(self):
        query = np.datetime64("2020-01-01T00:15")
        idx_lo, idx_hi, frac, in_range = time_bracket(self.time, query)
        self.assertEqual((idx_lo, idx_hi), (0, 1))
        self.assertAlmostEqual(frac, 0.25)
        self.assertTrue(in_range)

        values = np.array([10.0, 20.0, 30.0])
        expected, _ = linear_temporal_interpolate(self.time, values, query)
        reconstructed = (1 - frac) * values[idx_lo] + frac * values[idx_hi]
        self.assertAlmostEqual(reconstructed, float(expected))

    def test_out_of_range_is_flagged(self):
        _, _, _, in_range = time_bracket(self.time, np.datetime64("2020-01-01T05:00"))
        self.assertFalse(in_range)

    def test_rejects_non_scalar_query(self):
        with self.assertRaises(ValueError):
            time_bracket(self.time, np.array(["2020-01-01T00:15", "2020-01-01T00:30"], dtype="datetime64[ns]"))


class TestInterpolateFieldCombined(unittest.TestCase):
    """Composed space+time query against a tiny synthetic GriddedFieldSource."""

    def setUp(self):
        time = np.array(
            ["2020-01-01T00:00", "2020-01-01T01:00", "2020-01-01T02:00"], dtype="datetime64[ns]"
        )
        data = {"test_var": np.stack([_FIELD_ASC * 1.0, _FIELD_ASC * 2.0, _FIELD_ASC * 3.0])}
        self.src = GriddedFieldSource(
            source_path="<synthetic>",
            time_name="time",
            time=time,
            lat_name="lat",
            lat=_LAT_ASC,
            lon_name="lon",
            lon=_LON,
            data=data,
            fill_threshold=1e19,
        )

    def test_combined_space_and_time_interpolation(self):
        # spatial-only value is 2.5 at t=0 (field*1) and 5.0 at t=1 (field*2);
        # halfway in time (00:30) must give their average: 3.75.
        value, valid = interpolate_field(self.src, "test_var", lat=15.0, lon=105.0, time=np.datetime64("2020-01-01T00:30"))
        self.assertTrue(valid)
        self.assertAlmostEqual(value, 3.75)

    def test_vectorized_spatial_points_at_one_shared_time(self):
        values, valid = interpolate_field(
            self.src, "test_var", lat=np.array([15.0, 20.0]), lon=np.array([105.0, 110.0]),
            time=np.datetime64("2020-01-01T00:00"),
        )
        np.testing.assert_allclose(values, [2.5, 4.0])
        self.assertTrue(np.all(valid))

    def test_out_of_range_time_is_invalid(self):
        value, valid = interpolate_field(self.src, "test_var", lat=15.0, lon=105.0, time=np.datetime64("2019-12-31T00:00"))
        self.assertFalse(valid)


class TestInterpolationAgainstRealData(unittest.TestCase):
    """Spot checks against the real NetCDF files, exercising both grids' latitude
    orientation and resolution."""

    @classmethod
    def setUpClass(cls):
        cls.ocean = load_field_dataset(
            OCEAN_PATH, ["uo", "vo"], time_dim="time", lat_dim="latitude", lon_dim="longitude"
        )
        cls.wind = load_field_dataset(
            WIND_PATH, ["u10", "v10"], time_dim="valid_time", lat_dim="latitude", lon_dim="longitude"
        )

    def test_ocean_interior_midpoint_matches_hand_computed_average(self):
        # Bilinear interpolation at the exact midpoint of a 2x2 all-finite block must equal
        # the plain average of those four corner values -- a real-data analogue of the
        # synthetic hand-computed tests above. (An exact grid vertex is deliberately not
        # used here: its bracket can include a NaN neighbour near the coastline, which
        # correctly makes it invalid -- that behaviour is covered by
        # `test_land_cell_is_invalid_not_silently_valid` and the strict corner policy is
        # intentional, not a bug.)
        uo = self.ocean.data["uo"]
        found = None
        for t in range(uo.shape[0]):
            finite = ~np.isnan(uo[t])
            block_ok = finite[:-1, :-1] & finite[:-1, 1:] & finite[1:, :-1] & finite[1:, 1:]
            hits = np.argwhere(block_ok)
            if hits.size:
                i_idx, j_idx = hits[0]
                found = (t, int(i_idx), int(j_idx))
                break
        self.assertIsNotNone(found, "expected at least one interior all-finite 2x2 block")
        t_idx, i_idx, j_idx = found

        lat_mid = float((self.ocean.lat[i_idx] + self.ocean.lat[i_idx + 1]) / 2)
        lon_mid = float((self.ocean.lon[j_idx] + self.ocean.lon[j_idx + 1]) / 2)
        expected = float(
            uo[t_idx, i_idx, j_idx]
            + uo[t_idx, i_idx, j_idx + 1]
            + uo[t_idx, i_idx + 1, j_idx]
            + uo[t_idx, i_idx + 1, j_idx + 1]
        ) / 4.0

        value, valid = interpolate_field(
            self.ocean, "uo", lat=lat_mid, lon=lon_mid, time=self.ocean.time[t_idx],
        )
        self.assertTrue(valid)
        self.assertAlmostEqual(value, expected, places=5)

    def test_wind_exact_grid_point_matches_raw_value_descending_latitude(self):
        self.assertFalse(self.wind.lat_ascending)
        u10 = self.wind.data["u10"]
        value, valid = interpolate_field(
            self.wind, "u10", lat=self.wind.lat[3], lon=self.wind.lon[5], time=self.wind.time[7],
        )
        self.assertTrue(valid)
        self.assertAlmostEqual(value, float(u10[7, 3, 5]), places=5)

    def test_land_cell_is_invalid_not_silently_valid(self):
        uo = self.ocean.data["uo"]
        nan_positions = np.argwhere(np.isnan(uo[0]))
        self.assertTrue(nan_positions.size > 0, "expected some masked/land cells in the ocean grid")
        i_idx, j_idx = nan_positions[0]
        value, valid = interpolate_field(
            self.ocean, "uo", lat=self.ocean.lat[i_idx], lon=self.ocean.lon[j_idx], time=self.ocean.time[0],
        )
        self.assertFalse(valid)
        self.assertTrue(np.isnan(value))


class TestNetCDFEnvironmentalFieldProvider(unittest.TestCase):
    """Confirms `NetCDFEnvironmentalFieldProvider` is fully wired -- `U`/`validity` via T02
    interpolation (unchanged since T02), `div_U` via T03's divergence machinery on the
    provider's own ocean grid, and `K`/`grad_K` via T03's gradient machinery applied to an
    explicitly supplied `K` grid -- never a physically-invented `K` formula. Also confirms
    the provider agrees with T04's own domain check and with what T05's live step function
    actually computes for the same inputs, so the interface is a genuine access path to
    the running pipeline's numbers, not a second, independently-implemented copy."""

    @classmethod
    def setUpClass(cls):
        cls.ocean = load_field_dataset(
            OCEAN_PATH, ["uo", "vo"], time_dim="time", lat_dim="latitude", lon_dim="longitude"
        )
        cls.provider = NetCDFEnvironmentalFieldProvider(cls.ocean)  # no K configured

        # A small, EXACTLY-linear-in-longitude synthetic K grid on the ocean's own lat/lon
        # grid -- deliberately linear so its analytic gradient is knowable exactly by hand
        # (the same technique T12's controlled cases use), never a physically claimed K.
        cls.k_lat = cls.ocean.lat
        cls.k_lon = cls.ocean.lon
        cls.k_slope = 100.0
        cls._k_lon_ref = float(cls.k_lon[len(cls.k_lon) // 2])
        cls.k_field = cls.k_slope * (cls.k_lon[None, :] - cls._k_lon_ref) * np.ones((len(cls.k_lat), 1))
        cls.provider_with_k = NetCDFEnvironmentalFieldProvider(
            cls.ocean, k_lat=cls.k_lat, k_lon=cls.k_lon, k_field=cls.k_field
        )

    def test_U_matches_direct_uo_vo_interpolation(self):
        lat, lon, t = 33.0, 30.0, self.ocean.time[10]
        u_expected, u_valid = interpolate_field(self.ocean, "uo", lat, lon, t)
        v_expected, v_valid = interpolate_field(self.ocean, "vo", lat, lon, t)
        self.assertTrue(u_valid and v_valid)
        result = self.provider.U(np.array([lon, lat]), t)
        np.testing.assert_allclose(result, [u_expected, v_expected])

    def test_validity_false_outside_domain(self):
        self.assertFalse(self.provider.validity(np.array([0.0, 0.0]), self.ocean.time[0]))

    def test_validity_matches_t04_netcdf_validity_check(self):
        from iste.boundary import netcdf_validity_check

        lat, lon, t = 33.0, 30.0, self.ocean.time[10]
        t04_check = netcdf_validity_check(self.ocean, "uo", t)
        t04_valid = bool(t04_check(np.array([lat]), np.array([lon]))[0])
        provider_valid = self.provider.validity(np.array([lon, lat]), t)
        self.assertTrue(provider_valid)
        self.assertEqual(provider_valid, t04_valid)

    def test_div_U_matches_direct_derivative_computation(self):
        from iste.environment import interpolated_divergence

        lat, lon, t = 33.0, 30.0, self.ocean.time[10]
        expected, valid = interpolated_divergence(self.ocean, np.array([lat]), np.array([lon]), t)
        self.assertTrue(valid[0])
        result = self.provider.div_U(np.array([lon, lat]), t)
        self.assertAlmostEqual(result, float(expected[0]))

    def test_div_U_no_longer_raises_and_matches_t05s_own_computation(self):
        """Regression check for the interface-integration fix: `div_U` needs only `U`
        (`uo`/`vo`), not `K` -- it must be fully functional even with no `K` configured,
        and must match the EXACT hand-computed expected divergence already validated in
        `tests/test_step.py::TestDivergenceWeightUpdate` (same synthetic field, unmodified
        there), proving `iste.step`'s refactor to call the shared helper did not change
        its output."""
        lat_grid = np.array([10.0, 20.0, 30.0])
        lon_grid = np.array([100.0, 110.0, 120.0])
        time_grid = np.array(["2020-01-01T00:00", "2020-01-01T01:00"], dtype="datetime64[ns]")
        C = 0.001
        uo_grid = -C * lon_grid[None, :] * np.ones((3, 1))  # matches test_step.py's convergent-flow case
        vo_grid = np.zeros((3, 3))
        synthetic_ocean = GriddedFieldSource(
            source_path="<synthetic>", time_name="time", time=time_grid, lat_name="lat", lat=lat_grid,
            lon_name="lon", lon=lon_grid,
            data={"uo": np.stack([uo_grid, uo_grid]), "vo": np.stack([vo_grid, vo_grid])},
            fill_threshold=1e19,
        )
        provider = NetCDFEnvironmentalFieldProvider(synthetic_ocean)  # no K -- div_U must still work
        result = provider.div_U(np.array([110.0, 20.0]), time_grid[0])
        self.assertTrue(np.isfinite(result))

        expected_div = -C / metres_per_degree_longitude(20.0)
        self.assertAlmostEqual(result, expected_div, places=9)

        # and iste.step's live pipeline output, still unmodified equation-wise, must
        # match the same analytic expectation (places=6, matching test_step.py's own
        # tolerance for this exact case).
        state = WeightedParticleState.initialize(lat0=[20.0], lon0=[110.0])
        rng = np.random.default_rng(6)
        new_state = backward_euler_maruyama_step(
            state, time_grid[0], 900.0, synthetic_ocean, lat_grid, lon_grid, np.zeros((3, 3)), rng
        )
        expected_log_weight = -expected_div * 900.0
        self.assertAlmostEqual(new_state.log_weight[0], expected_log_weight, places=6)

    def test_missing_k_configuration_raises_clear_error(self):
        x, t = np.array([30.0, 33.0]), self.ocean.time[0]
        with self.assertRaises(ValueError) as ctx:
            self.provider.K(x, t)
        self.assertIn("K was not configured", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx2:
            self.provider.grad_K(x, t)
        self.assertIn("K was not configured", str(ctx2.exception))

    def test_partial_k_arguments_rejected_at_construction(self):
        with self.assertRaises(ValueError):
            NetCDFEnvironmentalFieldProvider(self.ocean, k_lat=self.k_lat, k_lon=self.k_lon)  # k_field missing

    def test_K_query_matches_direct_bilinear_interpolation(self):
        # must be inside the REAL ocean grid's own lat/lon extent (k_lat/k_lon reuse it):
        # data/ocean.nc covers roughly lat [30.19, 36.35], lon [27.13, 36.08].
        lat, lon, t = 33.0, self._k_lon_ref + 1.0, self.ocean.time[0]
        expected, valid = bilinear_spatial_interpolate(self.k_lat, self.k_lon, self.k_field, lat, lon)
        self.assertTrue(valid[0])
        result = self.provider_with_k.K(np.array([lon, lat]), t)
        self.assertAlmostEqual(result, float(expected[0]))

    def test_grad_K_matches_exact_analytic_gradient(self):
        lat, lon, t = 33.0, self._k_lon_ref + 1.0, self.ocean.time[0]
        result = self.provider_with_k.grad_K(np.array([lon, lat]), t)
        analytic_dKdx = self.k_slope / metres_per_degree_longitude(lat)
        self.assertAlmostEqual(result[0], analytic_dKdx, places=6)
        self.assertAlmostEqual(result[1], 0.0, places=9)  # K has no latitude dependence here

    def test_constructor_requires_uo_and_vo(self):
        bad_source = load_field_dataset(
            OCEAN_PATH, ["uo"], time_dim="time", lat_dim="latitude", lon_dim="longitude"
        )
        with self.assertRaises(ValueError):
            NetCDFEnvironmentalFieldProvider(bad_source)


if __name__ == "__main__":
    unittest.main()
