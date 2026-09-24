"""Focused T29 checks for `iste.phase3_normalization` (Phase 3 cross-time KDE/support-grid
normalization).

Run with:
    python -m unittest tests.test_phase3_normalization -v
"""

from __future__ import annotations

import unittest

import numpy as np

from iste.ensemble_run import EnsembleRunRecords
from iste.kdtree_neighborhood import build_particle_neighborhood_index
from iste.particle_query import query_particle_slice
from iste.phase3_normalization import (
    normalize_support_grid,
    normalize_support_grids_across_time,
)
from iste.support_features import SupportFeatureConfig, compute_support_map

T_S = np.datetime64("2020-01-01T00:00:00", "ns")
LAT_GRID = np.linspace(9.85, 10.15, 61)
LON_GRID = np.linspace(99.85, 100.15, 61)
RADIUS_M = 1000.0


def _records(lat, lon) -> EnsembleRunRecords:
    n = len(lat)
    return EnsembleRunRecords(
        particle_id=np.array([f"p{i}" for i in range(n)], dtype=object),
        forcing_member_id=np.zeros(n, dtype=np.int64),
        observation_pixel_id=np.array([f"px-{i % 3}" for i in range(n)], dtype=object),
        timestamp=np.full(n, T_S, dtype="datetime64[ns]"),
        lat=np.asarray(lat, dtype=np.float64), lon=np.asarray(lon, dtype=np.float64),
        log_weight=np.zeros(n, dtype=np.float64), valid=np.ones(n, dtype=bool),
    )


def _support_grid_for(lat, lon) -> np.ndarray:
    slice_ = query_particle_slice(_records(lat, lon), T_S)
    index = build_particle_neighborhood_index(slice_)
    lon_mesh, lat_mesh = np.meshgrid(LON_GRID, LAT_GRID)
    support_map = compute_support_map(
        index, lat_mesh.ravel(), lon_mesh.ravel(), radius_m=RADIUS_M, config=SupportFeatureConfig()
    )
    return np.array([sf.support for sf in support_map]).reshape(len(LAT_GRID), len(LON_GRID))


class TestNormalizeSupportGrid(unittest.TestCase):
    def test_simple_grid_matches_hand_computed_percentile(self):
        grid = np.array([[0.0, 10.0], [20.0, 30.0]])
        result = normalize_support_grid(grid, T_S, percentile=50.0)
        expected_ref = np.percentile([10.0, 20.0, 30.0], 50.0)  # positive cells only
        self.assertAlmostEqual(result.reference_value, expected_ref)
        np.testing.assert_allclose(result.normalized_grid, grid / expected_ref)

    def test_all_zero_grid_returns_nan_not_crash(self):
        grid = np.zeros((5, 5))
        result = normalize_support_grid(grid, T_S)
        self.assertEqual(result.n_positive_cells, 0)
        self.assertTrue(np.isnan(result.reference_value))
        self.assertTrue(np.all(np.isnan(result.normalized_grid)))

    def test_empty_grid_returns_nan_not_crash(self):
        grid = np.array([])
        result = normalize_support_grid(grid, T_S)
        self.assertTrue(np.isnan(result.reference_value))

    def test_invalid_percentile_raises(self):
        with self.assertRaises(ValueError):
            normalize_support_grid(np.ones((3, 3)), T_S, percentile=150.0)
        with self.assertRaises(ValueError):
            normalize_support_grid(np.ones((3, 3)), T_S, percentile=-1.0)

    def test_t_s_carried_through(self):
        result = normalize_support_grid(np.array([[1.0, 2.0]]), T_S)
        self.assertEqual(result.t_s, T_S)

    def test_normalized_grid_shape_matches_input(self):
        grid = np.random.default_rng(0).uniform(0, 10, size=(7, 9))
        result = normalize_support_grid(grid, T_S)
        self.assertEqual(result.normalized_grid.shape, grid.shape)


class TestPercentileIsPerTimestampOnly(unittest.TestCase):
    def test_scaling_one_timestamp_does_not_affect_another(self):
        grid_a = np.array([[1.0, 2.0, 3.0]])
        grid_b = np.array([[10.0, 20.0, 30.0]])
        result_a = normalize_support_grid(grid_a, T_S)
        result_b = normalize_support_grid(grid_b, T_S)
        # each grid's own reference value scales with its own magnitude -- no cross-
        # timestamp pooling, so normalized_max should be identical (both are self-relative).
        self.assertAlmostEqual(result_a.normalized_max, result_b.normalized_max)


class TestTightVsDispersedClusterNormalization(unittest.TestCase):
    """The core T29 demonstration: percentile normalization substantially removes the
    dispersion-driven raw-KDE-scale difference T28 measured on the real dataset."""

    def setUp(self):
        rng = np.random.default_rng(0)
        n = 90
        tight_lat = 10.0 + rng.normal(0, 0.0003, n)
        tight_lon = 100.0 + rng.normal(0, 0.0003, n)
        disp_lat = 10.0 + rng.normal(0, 0.02, n)
        disp_lon = 100.0 + rng.normal(0, 0.02, n)
        self.g_tight = _support_grid_for(tight_lat, tight_lon)
        self.g_disp = _support_grid_for(disp_lat, disp_lon)

    def test_raw_kde_peaks_differ_substantially(self):
        raw_ratio = self.g_tight.max() / self.g_disp.max()
        self.assertGreater(raw_ratio, 4.0)  # confirms the synthetic setup reproduces the bias

    def test_normalized_peaks_are_comparable(self):
        result_tight = normalize_support_grid(self.g_tight, T_S, percentile=95.0)
        result_disp = normalize_support_grid(self.g_disp, T_S - np.timedelta64(72, "h"), percentile=95.0)
        raw_ratio = self.g_tight.max() / self.g_disp.max()
        normalized_ratio = result_tight.normalized_max / result_disp.normalized_max
        # normalization must bring the ratio much closer to 1 than the raw ratio was.
        self.assertLess(abs(np.log(normalized_ratio)), np.log(3.0))
        self.assertLess(abs(np.log(normalized_ratio)), abs(np.log(raw_ratio)))


class TestNormalizeAcrossTime(unittest.TestCase):
    def test_dict_keys_and_independence(self):
        grids = {
            T_S: np.array([[1.0, 2.0], [3.0, 4.0]]),
            T_S - np.timedelta64(1, "h"): np.array([[100.0, 200.0], [300.0, 400.0]]),
        }
        results = normalize_support_grids_across_time(grids)
        self.assertEqual(set(results.keys()), set(grids.keys()))
        # both timestamps normalize to the same relative shape despite 100x different scale
        self.assertAlmostEqual(
            results[T_S].normalized_max,
            results[T_S - np.timedelta64(1, "h")].normalized_max,
        )


if __name__ == "__main__":
    unittest.main()
