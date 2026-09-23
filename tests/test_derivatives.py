"""Focused T03 checks for `iste.derivatives` (`grad(K)` / `div(U)` machinery).

Run with:
    python -m unittest tests.test_derivatives -v
"""

from __future__ import annotations

import os
import unittest

import numpy as np

from iste.derivatives import divergence_2d, spatial_gradient
from iste.environment import load_field_dataset
from iste.geometry import metres_per_degree_latitude, metres_per_degree_longitude

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OCEAN_PATH = os.path.join(REPO_ROOT, "data", "ocean.nc")

LAT = np.linspace(10.0, 30.0, 9)  # ascending, uniform 2.5-degree spacing
LON = np.linspace(100.0, 120.0, 11)  # ascending, uniform 2.0-degree spacing


class TestSpatialGradientKnownConstantGradient(unittest.TestCase):
    """A field linear in latitude only: `field[i, j] = lat_coords[i]` (independent of
    longitude). Analytically: `d(field)/dx = 0` and `d(field)/dy = 1 / metres_per_degree_latitude()`
    everywhere -- a single constant value, exactly recovered at every interior *and* edge
    point since central/edge_order=2 differences are exact for a linear function."""

    def test_constant_gradient_recovered_everywhere(self):
        field = np.broadcast_to(LAT[:, None], (len(LAT), len(LON))).astype(np.float64)
        d_dx, d_dy, valid = spatial_gradient(LAT, LON, field)
        self.assertTrue(np.all(valid))
        np.testing.assert_allclose(d_dx, 0.0, atol=1e-9)
        expected_d_dy = 1.0 / metres_per_degree_latitude()
        np.testing.assert_allclose(d_dy, expected_d_dy, rtol=1e-9)


class TestSpatialGradientSpatiallyVarying(unittest.TestCase):
    """A field quadratic in longitude only: `field[i, j] = lon_coords[j]**2`. Analytically
    `d(field)/d(lon_deg) = 2*lon_coords[j]` exactly (central/edge_order=2 differences are
    exact for quadratics too), so `d(field)/dx = 2*lon_coords[j] / metres_per_degree_longitude(lat_i)`
    -- a value that varies with *both* longitude (through `lon_coords[j]`) and latitude
    (through the row-dependent metric factor). `d(field)/dy = 0` everywhere."""

    def test_spatially_varying_gradient_matches_analytic_value(self):
        field = np.broadcast_to(LON[None, :] ** 2, (len(LAT), len(LON))).astype(np.float64)
        d_dx, d_dy, valid = spatial_gradient(LAT, LON, field)
        self.assertTrue(np.all(valid))
        dlon_per_deg = metres_per_degree_longitude(LAT)  # shape (n_lat,)
        expected_d_dx = (2.0 * LON[None, :]) / dlon_per_deg[:, None]
        np.testing.assert_allclose(d_dx, expected_d_dx, rtol=1e-9)
        np.testing.assert_allclose(d_dy, 0.0, atol=1e-9)

    def test_edges_are_not_degraded_or_extrapolated(self):
        # Same quadratic field: boundary columns (j=0, j=-1) must match the analytic
        # value exactly too (edge_order=2), not merely "close" as a first-order edge
        # formula would give, and must never reach outside [min(LON), max(LON)].
        field = np.broadcast_to(LON[None, :] ** 2, (len(LAT), len(LON))).astype(np.float64)
        d_dx, _, valid = spatial_gradient(LAT, LON, field)
        dlon_per_deg = metres_per_degree_longitude(LAT)
        for j in (0, -1):
            expected = (2.0 * LON[j]) / dlon_per_deg
            np.testing.assert_allclose(d_dx[:, j], expected, rtol=1e-9)
        self.assertTrue(np.all(valid[:, 0]) and np.all(valid[:, -1]))


class TestSpatialGradientDescendingLatitude(unittest.TestCase):
    def test_same_field_descending_latitude_gives_same_physical_answer(self):
        field_asc = np.broadcast_to(LAT[:, None], (len(LAT), len(LON))).astype(np.float64)
        lat_desc = LAT[::-1]
        field_desc = field_asc[::-1, :]
        d_dx, d_dy, valid = spatial_gradient(lat_desc, LON, field_desc)
        self.assertTrue(np.all(valid))
        np.testing.assert_allclose(d_dx, 0.0, atol=1e-9)
        np.testing.assert_allclose(d_dy, 1.0 / metres_per_degree_latitude(), rtol=1e-9)


class TestSpatialGradientInvalidCells(unittest.TestCase):
    def test_nan_propagates_only_to_its_stencil_neighbours(self):
        field = np.broadcast_to(LAT[:, None], (len(LAT), len(LON))).astype(np.float64).copy()
        i0, j0 = 4, 5  # an interior cell, away from every edge
        field[i0, j0] = np.nan
        d_dx, d_dy, valid = spatial_gradient(LAT, LON, field)

        # the masked cell, its immediate lat-neighbours (lat stencil at column j0), and
        # its immediate lon-neighbours (lon stencil at row i0) must all be invalid ...
        self.assertFalse(valid[i0, j0])
        self.assertFalse(valid[i0 - 1, j0])
        self.assertFalse(valid[i0 + 1, j0])
        self.assertFalse(valid[i0, j0 - 1])
        self.assertFalse(valid[i0, j0 + 1])
        # ... but a cell two rows/columns away (whose stencil never touches (i0, j0))
        # must be unaffected.
        self.assertTrue(valid[i0 - 2, j0])
        self.assertTrue(valid[i0, j0 - 2])
        self.assertTrue(np.isnan(d_dx[i0, j0]) and np.isnan(d_dy[i0, j0]))
        self.assertFalse(np.isnan(d_dy[i0 - 2, j0]))


class TestSpatialGradientErrorHandling(unittest.TestCase):
    def test_wrong_field_shape_raises(self):
        with self.assertRaises(ValueError):
            spatial_gradient(LAT, LON, np.zeros((3, 3)))

    def test_too_few_points_raises(self):
        with self.assertRaises(ValueError):
            spatial_gradient(np.array([1.0, 2.0]), LON, np.zeros((2, len(LON))))

    def test_non_monotonic_coords_raises(self):
        bad_lat = LAT.copy()
        bad_lat[3], bad_lat[4] = bad_lat[4], bad_lat[3]
        with self.assertRaises(ValueError):
            spatial_gradient(bad_lat, LON, np.broadcast_to(bad_lat[:, None], (len(LAT), len(LON))))


class TestDivergenceKnownField(unittest.TestCase):
    """`u[i,j]` = physical eastward position (metres), `v[i,j]` = physical northward
    position (metres) -- both linear in physical space by construction, so
    `du/dx = dv/dy = 1` exactly everywhere and `div(U) = 2` exactly everywhere."""

    def test_known_constant_divergence(self):
        dlon_per_deg = metres_per_degree_longitude(LAT)  # (n_lat,)
        dlat_per_deg = metres_per_degree_latitude()
        u = LON[None, :] * dlon_per_deg[:, None]  # eastward physical position (m)
        v = np.broadcast_to((LAT * dlat_per_deg)[:, None], (len(LAT), len(LON))).astype(np.float64)

        div_u, valid = divergence_2d(LAT, LON, u, v)
        self.assertTrue(np.all(valid))
        np.testing.assert_allclose(div_u, 2.0, rtol=1e-8)

    def test_zero_divergence_field(self):
        # u depends only on latitude (constant along the row) -> du/dx = 0 exactly,
        # regardless of the (even nonlinear) shape of that lat-dependence.
        u = np.broadcast_to(np.sin(np.deg2rad(LAT))[:, None], (len(LAT), len(LON))).astype(np.float64)
        # v depends only on longitude (constant down the column) -> dv/dy = 0 exactly.
        v = np.broadcast_to((LON**3)[None, :], (len(LAT), len(LON))).astype(np.float64)

        div_u, valid = divergence_2d(LAT, LON, u, v)
        self.assertTrue(np.all(valid))
        np.testing.assert_allclose(div_u, 0.0, atol=1e-9)

    def test_windage_is_never_introduced(self):
        # divergence_2d is generic: passing exactly two components (u, v) is the only
        # signature available -- there is no third "wind" argument to accidentally wire in.
        import inspect

        params = list(inspect.signature(divergence_2d).parameters)
        self.assertEqual(params, ["lat_coords", "lon_coords", "u_component", "v_component", "radius_m"])


class TestDivergenceInvalidCells(unittest.TestCase):
    def test_missing_u_or_v_cell_propagates_as_invalid(self):
        u = np.ones((len(LAT), len(LON)))
        v = np.ones((len(LAT), len(LON)))
        u[4, 5] = np.nan
        div_u, valid = divergence_2d(LAT, LON, u, v)
        self.assertFalse(valid[4, 5])
        self.assertFalse(valid[4, 4])  # u's lon-stencil neighbour
        self.assertTrue(valid[2, 5])  # unrelated row, unaffected


class TestDivergenceRealDataSanityCheck(unittest.TestCase):
    """Only a numerical sanity check against the real ocean dataset -- NOT a validation
    of the ocean dynamics themselves. Confirms `divergence_2d`'s output matches an
    independently hand-computed finite difference at one real, fully-valid grid point."""

    @classmethod
    def setUpClass(cls):
        cls.ocean = load_field_dataset(
            OCEAN_PATH, ["uo", "vo"], time_dim="time", lat_dim="latitude", lon_dim="longitude"
        )

    def test_matches_manual_finite_difference_at_one_real_point(self):
        uo = self.ocean.data["uo"]
        vo = self.ocean.data["vo"]
        t_idx = None
        i0 = j0 = None
        for t in range(uo.shape[0]):
            u_finite = ~np.isnan(uo[t])
            v_finite = ~np.isnan(vo[t])
            # need u valid at (i, j-1),(i,j),(i,j+1) and v valid at (i-1,j),(i,j),(i+1,j)
            u_ok = u_finite[1:-1, :-2] & u_finite[1:-1, 1:-1] & u_finite[1:-1, 2:]
            v_ok = v_finite[:-2, 1:-1] & v_finite[1:-1, 1:-1] & v_finite[2:, 1:-1]
            hits = np.argwhere(u_ok & v_ok)
            if hits.size:
                t_idx = t
                i0, j0 = hits[0] + 1  # offset back into full-grid indices
                break
        self.assertIsNotNone(t_idx, "expected at least one interior all-valid '+' neighbourhood")

        lat = self.ocean.lat
        lon = self.ocean.lon
        dlon_per_deg = metres_per_degree_longitude(lat[i0])
        dlat_per_deg = metres_per_degree_latitude()

        du_dlon = (uo[t_idx, i0, j0 + 1] - uo[t_idx, i0, j0 - 1]) / (lon[j0 + 1] - lon[j0 - 1])
        dv_dlat = (vo[t_idx, i0 + 1, j0] - vo[t_idx, i0 - 1, j0]) / (lat[i0 + 1] - lat[i0 - 1])
        expected_div = (du_dlon / dlon_per_deg) + (dv_dlat / dlat_per_deg)

        div_field, valid_field = divergence_2d(lat, lon, uo[t_idx], vo[t_idx])
        self.assertTrue(valid_field[i0, j0])
        self.assertAlmostEqual(div_field[i0, j0], expected_div, places=8)

    def test_output_shape_and_some_valid_coverage(self):
        uo = self.ocean.data["uo"]
        vo = self.ocean.data["vo"]
        div_field, valid_field = divergence_2d(self.ocean.lat, self.ocean.lon, uo[0], vo[0])
        self.assertEqual(div_field.shape, uo[0].shape)
        self.assertTrue(np.any(valid_field))
        self.assertTrue(np.all(np.isnan(div_field[~valid_field])))


if __name__ == "__main__":
    unittest.main()
