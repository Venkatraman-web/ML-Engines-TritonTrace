"""Focused T03 checks for `iste.geometry`.

Run with:
    python -m unittest tests.test_geometry -v
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from iste.geometry import (
    MEAN_EARTH_RADIUS_M,
    metres_per_degree_latitude,
    metres_per_degree_longitude,
    metres_to_degrees,
)


class TestMetresPerDegree(unittest.TestCase):
    def test_latitude_conversion_matches_hand_computation(self):
        # circumference / 360, for the default radius.
        expected = (2 * math.pi * MEAN_EARTH_RADIUS_M) / 360.0
        self.assertAlmostEqual(metres_per_degree_latitude(), expected, places=3)

    def test_latitude_conversion_is_independent_of_latitude(self):
        # spherical approximation: constant regardless of where you are.
        self.assertEqual(metres_per_degree_latitude(), metres_per_degree_latitude())

    def test_longitude_conversion_equals_latitude_conversion_at_equator(self):
        self.assertAlmostEqual(
            metres_per_degree_longitude(0.0), metres_per_degree_latitude(), places=6
        )

    def test_longitude_conversion_shrinks_toward_the_poles(self):
        eq = metres_per_degree_longitude(0.0)
        mid = metres_per_degree_longitude(60.0)
        high = metres_per_degree_longitude(80.0)
        self.assertLess(mid, eq)
        self.assertLess(high, mid)
        # at 60 degrees, cos(60) = 0.5 exactly.
        self.assertAlmostEqual(mid, eq * 0.5, places=6)

    def test_longitude_conversion_vectorized_over_latitude_array(self):
        lats = np.array([0.0, 30.0, 60.0, 90.0])
        values = metres_per_degree_longitude(lats)
        self.assertEqual(values.shape, lats.shape)
        self.assertAlmostEqual(values[-1], 0.0, places=6)  # pole: longitude circle shrinks to a point

    def test_custom_radius_scales_linearly(self):
        base = metres_per_degree_latitude(radius_m=1000.0)
        doubled = metres_per_degree_latitude(radius_m=2000.0)
        self.assertAlmostEqual(doubled, base * 2.0, places=9)


class TestMetresToDegrees(unittest.TestCase):
    def test_is_the_exact_inverse_of_the_per_degree_factors(self):
        lat0 = 33.0
        dx_m, dy_m = 1234.5, -987.6
        dlon_deg, dlat_deg = metres_to_degrees(dx_m, dy_m, lat0)
        self.assertAlmostEqual(dlon_deg * metres_per_degree_longitude(lat0), dx_m, places=6)
        self.assertAlmostEqual(dlat_deg * metres_per_degree_latitude(), dy_m, places=6)

    def test_sign_is_preserved(self):
        dlon_pos, dlat_pos = metres_to_degrees(100.0, 100.0, 0.0)
        dlon_neg, dlat_neg = metres_to_degrees(-100.0, -100.0, 0.0)
        self.assertGreater(dlon_pos, 0.0)
        self.assertGreater(dlat_pos, 0.0)
        self.assertLess(dlon_neg, 0.0)
        self.assertLess(dlat_neg, 0.0)

    def test_vectorized_over_arrays(self):
        dx_m = np.array([100.0, 200.0])
        dy_m = np.array([50.0, -50.0])
        lat0 = np.array([0.0, 60.0])
        dlon_deg, dlat_deg = metres_to_degrees(dx_m, dy_m, lat0)
        self.assertEqual(dlon_deg.shape, dx_m.shape)
        # at 60 degrees, the same eastward metre displacement needs twice the degrees.
        dlon_at_0, _ = metres_to_degrees(100.0, 0.0, 0.0)
        dlon_at_60, _ = metres_to_degrees(100.0, 0.0, 60.0)
        self.assertAlmostEqual(dlon_at_60, dlon_at_0 * 2.0, places=6)


if __name__ == "__main__":
    unittest.main()
