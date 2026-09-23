"""Focused T15 checks for `iste.kdtree_neighborhood` (Phase 2 KD-tree neighborhood
engine).

Run with:
    python -m unittest tests.test_kdtree_neighborhood -v
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from iste.geometry import MEAN_EARTH_RADIUS_M, metres_per_degree_latitude, metres_per_degree_longitude
from iste.kdtree_neighborhood import (
    build_particle_neighborhood_index,
    project_to_local_metres,
    query_knn,
    query_radius,
)
from iste.particle_query import HistoricalParticleSlice

QUERY_TIME = np.datetime64("2020-01-01T00:00:00", "ns")
NAT = np.datetime64("NaT", "ns")


def _make_slice(
    particle_id, observation_pixel_id, forcing_member_id, lat, lon, log_weight=None, valid=None,
    death_time=None, death_lat=None, death_lon=None, death_reason=None,
) -> HistoricalParticleSlice:
    n = len(lat)
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    return HistoricalParticleSlice(
        particle_id=np.array(particle_id, dtype=object),
        observation_pixel_id=np.array(observation_pixel_id, dtype=object),
        forcing_member_id=np.array(forcing_member_id, dtype=np.int64),
        lat=lat, lon=lon,
        log_weight=np.array(log_weight if log_weight is not None else [0.0] * n, dtype=np.float64),
        valid=np.array(valid if valid is not None else [True] * n, dtype=bool),
        death_time=np.array(death_time if death_time is not None else [NAT] * n, dtype="datetime64[ns]"),
        death_lat=np.array(death_lat if death_lat is not None else [np.nan] * n, dtype=np.float64),
        death_lon=np.array(death_lon if death_lon is not None else [np.nan] * n, dtype=np.float64),
        death_reason=np.array(death_reason if death_reason is not None else [""] * n, dtype=object),
        query_time=QUERY_TIME,
    )


def _brute_force_local_coords(lat, lon, lat_ref, lon_ref):
    dlat_per_deg = metres_per_degree_latitude()
    dlon_per_deg = metres_per_degree_longitude(lat_ref)
    x = (np.asarray(lon) - lon_ref) * dlon_per_deg
    y = (np.asarray(lat) - lat_ref) * dlat_per_deg
    return x, y


def _brute_force_neighbors(slice_: HistoricalParticleSlice, lat_ref, lon_ref, q_lat, q_lon):
    """Independent reference implementation: plain pairwise Euclidean distance in the
    same local-projected metres, over valid particles only, sorted by distance then
    particle_id -- used only by these tests, never by the module under test."""
    valid_idx = np.flatnonzero(slice_.valid)
    x, y = _brute_force_local_coords(slice_.lat[valid_idx], slice_.lon[valid_idx], lat_ref, lon_ref)
    qx, qy = _brute_force_local_coords([q_lat], [q_lon], lat_ref, lon_ref)
    d = np.sqrt((x - qx[0]) ** 2 + (y - qy[0]) ** 2)
    order = np.lexsort((slice_.particle_id[valid_idx].astype(str), d))
    return valid_idx[order], d[order]


class TestProjectedCoordinates(unittest.TestCase):
    """Requirement 6: controlled synthetic point set, hand-checkable projection."""

    def test_reference_point_projects_to_origin(self):
        x, y = project_to_local_metres(20.0, 100.0, lat_ref=20.0, lon_ref=100.0)
        self.assertAlmostEqual(x[0], 0.0)
        self.assertAlmostEqual(y[0], 0.0)

    def test_one_degree_latitude_north_matches_geometry_conversion(self):
        x, y = project_to_local_metres(21.0, 100.0, lat_ref=20.0, lon_ref=100.0)
        self.assertAlmostEqual(x[0], 0.0)
        self.assertAlmostEqual(y[0], metres_per_degree_latitude())

    def test_one_degree_longitude_east_matches_geometry_conversion_at_ref_latitude(self):
        x, y = project_to_local_metres(20.0, 101.0, lat_ref=20.0, lon_ref=100.0)
        self.assertAlmostEqual(y[0], 0.0)
        self.assertAlmostEqual(x[0], metres_per_degree_longitude(20.0))

    def test_longitude_metres_shrink_toward_higher_latitude(self):
        # same 1-degree longitude offset, but at a higher reference latitude, must
        # project to a SMALLER metre distance (never treating degrees as flat metres).
        x_low, _ = project_to_local_metres(10.0, 101.0, lat_ref=10.0, lon_ref=100.0)
        x_high, _ = project_to_local_metres(60.0, 101.0, lat_ref=60.0, lon_ref=100.0)
        self.assertGreater(x_low[0], x_high[0])
        self.assertAlmostEqual(x_low[0], metres_per_degree_longitude(10.0))
        self.assertAlmostEqual(x_high[0], metres_per_degree_longitude(60.0))

    def test_degrees_never_used_directly_as_metres(self):
        # a naive "treat degrees as metres" bug would give x=1.0 for a 1-degree offset;
        # the real conversion is many orders of magnitude larger.
        x, _ = project_to_local_metres(20.0, 101.0, lat_ref=20.0, lon_ref=100.0)
        self.assertGreater(abs(x[0]), 1000.0)


class TestRadiusQueryAgainstBruteForce(unittest.TestCase):
    def setUp(self):
        # a small grid of points around (20.0, 100.0), one excluded as invalid.
        offsets = [(-0.01, 0.0), (0.01, 0.0), (0.0, -0.01), (0.0, 0.01), (0.0, 0.0), (0.05, 0.05)]
        lat = [20.0 + dy for dy, dx in offsets]
        lon = [100.0 + dx for dy, dx in offsets]
        self.slice_ = _make_slice(
            particle_id=[f"p{i}" for i in range(6)],
            observation_pixel_id=["px-0"] * 6,
            forcing_member_id=[0] * 6,
            lat=lat, lon=lon,
            valid=[True, True, True, True, True, False],
        )
        self.index = build_particle_neighborhood_index(self.slice_)

    def test_radius_matches_brute_force_membership_and_order(self):
        result = query_radius(self.index, 20.0, 100.0, radius_m=1500.0)
        expected_idx, expected_d = _brute_force_neighbors(self.slice_, self.index.lat_ref, self.index.lon_ref, 20.0, 100.0)
        expected_idx = expected_idx[expected_d <= 1500.0]
        expected_d = expected_d[expected_d <= 1500.0]

        self.assertEqual(list(result.particle_id), list(self.slice_.particle_id[expected_idx]))
        np.testing.assert_allclose(result.distance_m, expected_d, atol=1e-6)

    def test_invalid_particle_never_returned_by_radius_query_even_if_close(self):
        # p5 is close (0.05 deg offset) but invalid -- must never appear, at any radius.
        result = query_radius(self.index, 20.0, 100.0, radius_m=1_000_000.0)
        self.assertNotIn("p5", result.particle_id.tolist())

    def test_radius_zero_or_negative_rejected(self):
        with self.assertRaises(ValueError):
            query_radius(self.index, 20.0, 100.0, radius_m=0.0)
        with self.assertRaises(ValueError):
            query_radius(self.index, 20.0, 100.0, radius_m=-5.0)

    def test_no_neighbors_in_radius_returns_empty_not_error(self):
        result = query_radius(self.index, 45.0, 45.0, radius_m=10.0)  # far away
        self.assertEqual(result.n_particles, 0)


class TestKnnAgainstBruteForce(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        n = 15
        lat = 20.0 + rng.uniform(-0.05, 0.05, size=n)
        lon = 100.0 + rng.uniform(-0.05, 0.05, size=n)
        valid = np.ones(n, dtype=bool)
        valid[3] = False
        valid[9] = False
        self.slice_ = _make_slice(
            particle_id=[f"p{i}" for i in range(n)],
            observation_pixel_id=["px-0"] * n,
            forcing_member_id=[0] * n,
            lat=lat, lon=lon, valid=valid,
        )
        self.index = build_particle_neighborhood_index(self.slice_)

    def test_knn_matches_brute_force_top_k(self):
        k = 5
        result = query_knn(self.index, 20.0, 100.0, k=k)
        expected_idx, expected_d = _brute_force_neighbors(self.slice_, self.index.lat_ref, self.index.lon_ref, 20.0, 100.0)
        self.assertEqual(list(result.particle_id), list(self.slice_.particle_id[expected_idx[:k]]))
        np.testing.assert_allclose(result.distance_m, expected_d[:k], atol=1e-6)

    def test_knn_returns_exactly_k_when_enough_valid_particles_exist(self):
        result = query_knn(self.index, 20.0, 100.0, k=4)
        self.assertEqual(result.n_particles, 4)

    def test_knn_k_larger_than_available_returns_all_available_not_fabricated(self):
        result = query_knn(self.index, 20.0, 100.0, k=1000)
        self.assertEqual(result.n_particles, self.index.n_particles)
        self.assertEqual(self.index.n_particles, 13)  # 15 - 2 invalid

    def test_knn_k_less_than_one_rejected(self):
        with self.assertRaises(ValueError):
            query_knn(self.index, 20.0, 100.0, k=0)


class TestMultiplePixelsAndMembersMetadataUnchanged(unittest.TestCase):
    def test_metadata_preserved_across_pixels_and_members(self):
        slice_ = _make_slice(
            particle_id=["a", "b", "c", "d"],
            observation_pixel_id=["px-0", "px-0", "px-1", "px-1"],
            forcing_member_id=[0, 1, 0, 1],
            lat=[20.0, 20.001, 20.002, 20.5],
            lon=[100.0, 100.001, 100.002, 100.5],
            log_weight=[0.1, 0.2, 0.3, 0.4],
            death_time=[NAT, np.datetime64("2020-01-01T00:00:00", "ns"), NAT, NAT],
            death_lat=[np.nan, 21.0, np.nan, np.nan],
            death_lon=[np.nan, 101.0, np.nan, np.nan],
            death_reason=["", "land", "", ""],
        )
        index = build_particle_neighborhood_index(slice_)
        result = query_radius(index, 20.0, 100.0, radius_m=500.0)

        by_id = {pid: i for i, pid in enumerate(result.particle_id)}
        self.assertIn("a", by_id)
        self.assertIn("b", by_id)
        self.assertIn("c", by_id)
        self.assertNotIn("d", by_id)  # far away, outside 500m

        ia, ib, ic = by_id["a"], by_id["b"], by_id["c"]
        self.assertEqual(result.observation_pixel_id[ia], "px-0")
        self.assertEqual(result.observation_pixel_id[ic], "px-1")
        self.assertEqual(result.forcing_member_id[ia], 0)
        self.assertEqual(result.forcing_member_id[ib], 1)
        self.assertAlmostEqual(result.log_weight[ia], 0.1)
        self.assertAlmostEqual(result.log_weight[ic], 0.3)
        # death metadata (for particle b, which is included since it's still spatially
        # near even though it independently has death info recorded) preserved exactly.
        self.assertEqual(result.death_reason[ib], "land")
        self.assertAlmostEqual(result.death_lat[ib], 21.0)

    def test_knn_also_preserves_metadata(self):
        slice_ = _make_slice(
            particle_id=["a", "b"], observation_pixel_id=["px-7", "px-8"], forcing_member_id=[3, 4],
            lat=[20.0, 20.001], lon=[100.0, 100.001],
        )
        index = build_particle_neighborhood_index(slice_)
        result = query_knn(index, 20.0, 100.0, k=2)
        self.assertEqual(set(result.observation_pixel_id.tolist()), {"px-7", "px-8"})
        self.assertEqual(set(result.forcing_member_id.tolist()), {3, 4})


class TestInvalidParticlesExcluded(unittest.TestCase):
    def test_invalid_particles_not_indexed_at_all(self):
        slice_ = _make_slice(
            particle_id=["a", "b"], observation_pixel_id=["px-0", "px-0"], forcing_member_id=[0, 0],
            lat=[20.0, 20.0], lon=[100.0, 100.0], valid=[True, False],
        )
        index = build_particle_neighborhood_index(slice_)
        self.assertEqual(index.n_particles, 1)
        self.assertEqual(slice_.particle_id[index.source_row[0]], "a")

    def test_all_invalid_gives_empty_index_and_safe_queries(self):
        slice_ = _make_slice(
            particle_id=["a", "b"], observation_pixel_id=["px-0", "px-0"], forcing_member_id=[0, 0],
            lat=[20.0, 20.0], lon=[100.0, 100.0], valid=[False, False],
        )
        index = build_particle_neighborhood_index(slice_)
        self.assertEqual(index.n_particles, 0)
        self.assertEqual(query_radius(index, 20.0, 100.0, radius_m=100.0).n_particles, 0)
        self.assertEqual(query_knn(index, 20.0, 100.0, k=3).n_particles, 0)


class TestReproducibleDeterministicOrdering(unittest.TestCase):
    def test_radius_query_order_identical_across_repeated_calls(self):
        rng = np.random.default_rng(3)
        n = 30
        lat = 20.0 + rng.uniform(-0.02, 0.02, size=n)
        lon = 100.0 + rng.uniform(-0.02, 0.02, size=n)
        slice_ = _make_slice(
            particle_id=[f"p{i}" for i in range(n)], observation_pixel_id=["px-0"] * n,
            forcing_member_id=[0] * n, lat=lat, lon=lon,
        )
        index = build_particle_neighborhood_index(slice_)
        r1 = query_radius(index, 20.0, 100.0, radius_m=3000.0)
        r2 = query_radius(index, 20.0, 100.0, radius_m=3000.0)
        self.assertEqual(r1.particle_id.tolist(), r2.particle_id.tolist())
        np.testing.assert_array_equal(r1.distance_m, r2.distance_m)

    def test_knn_order_identical_across_repeated_calls(self):
        rng = np.random.default_rng(4)
        n = 30
        lat = 20.0 + rng.uniform(-0.02, 0.02, size=n)
        lon = 100.0 + rng.uniform(-0.02, 0.02, size=n)
        slice_ = _make_slice(
            particle_id=[f"p{i}" for i in range(n)], observation_pixel_id=["px-0"] * n,
            forcing_member_id=[0] * n, lat=lat, lon=lon,
        )
        index = build_particle_neighborhood_index(slice_)
        r1 = query_knn(index, 20.0, 100.0, k=7)
        r2 = query_knn(index, 20.0, 100.0, k=7)
        self.assertEqual(r1.particle_id.tolist(), r2.particle_id.tolist())

    def test_ties_broken_deterministically_by_particle_id(self):
        # two particles at the exact same distance -- order must be by particle_id.
        slice_ = _make_slice(
            particle_id=["zeta", "alpha"], observation_pixel_id=["px-0", "px-0"], forcing_member_id=[0, 0],
            lat=[20.01, 19.99], lon=[100.0, 100.0],  # symmetric around (20.0, 100.0)
        )
        index = build_particle_neighborhood_index(slice_, lat_ref=20.0, lon_ref=100.0)
        result = query_radius(index, 20.0, 100.0, radius_m=10_000.0)
        self.assertEqual(result.n_particles, 2)
        self.assertAlmostEqual(result.distance_m[0], result.distance_m[1])
        self.assertEqual(list(result.particle_id), ["alpha", "zeta"])


if __name__ == "__main__":
    unittest.main()
