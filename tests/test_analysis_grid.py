"""Focused T27 checks for `iste.analysis_grid` (Phase 3 fixed hourly analysis grid).

Run with:
    python -m unittest tests.test_analysis_grid -v
"""

from __future__ import annotations

import unittest

import numpy as np

from iste.analysis_grid import generate_analysis_grid


class TestGenerateAnalysisGrid(unittest.TestCase):
    def setUp(self):
        self.t_obs = np.datetime64("2019-10-14T23:00:00", "ns")

    def test_default_72h_1h_grid_has_73_timestamps(self):
        grid = generate_analysis_grid(self.t_obs)
        self.assertEqual(len(grid), 73)

    def test_grid_first_entry_is_t_obs(self):
        grid = generate_analysis_grid(self.t_obs)
        self.assertEqual(grid[0], self.t_obs)

    def test_grid_last_entry_is_72_hours_back(self):
        grid = generate_analysis_grid(self.t_obs)
        expected_last = self.t_obs - np.timedelta64(72, "h")
        self.assertEqual(grid[-1], expected_last)

    def test_grid_is_strictly_descending_with_1_hour_spacing(self):
        grid = generate_analysis_grid(self.t_obs)
        diffs = np.diff(grid.astype("int64"))
        expected_step_ns = -3600 * 1_000_000_000
        self.assertTrue(np.all(diffs == expected_step_ns))

    def test_grid_dtype_is_datetime64_ns(self):
        grid = generate_analysis_grid(self.t_obs)
        self.assertEqual(grid.dtype, np.dtype("datetime64[ns]"))

    def test_custom_window_and_interval(self):
        grid = generate_analysis_grid(self.t_obs, window_hours=6.0, interval_hours=2.0)
        # 6h / 2h = 3 steps -> 4 timestamps: T0, T2h, T4h, T6h
        self.assertEqual(len(grid), 4)
        self.assertEqual(grid[0], self.t_obs)
        self.assertEqual(grid[-1], self.t_obs - np.timedelta64(6, "h"))
        diffs = np.diff(grid.astype("int64"))
        self.assertTrue(np.all(diffs == -2 * 3600 * 1_000_000_000))

    def test_fractional_hour_interval(self):
        # 3h / 0.5h = 6 steps -> 7 timestamps, 30-minute spacing
        grid = generate_analysis_grid(self.t_obs, window_hours=3.0, interval_hours=0.5)
        self.assertEqual(len(grid), 7)
        self.assertEqual(grid[-1], self.t_obs - np.timedelta64(3, "h"))

    def test_non_divisible_window_raises(self):
        with self.assertRaises(ValueError):
            generate_analysis_grid(self.t_obs, window_hours=72.0, interval_hours=5.0)

    def test_zero_window_hours_raises(self):
        with self.assertRaises(ValueError):
            generate_analysis_grid(self.t_obs, window_hours=0.0, interval_hours=1.0)

    def test_negative_window_hours_raises(self):
        with self.assertRaises(ValueError):
            generate_analysis_grid(self.t_obs, window_hours=-72.0, interval_hours=1.0)

    def test_zero_interval_hours_raises(self):
        with self.assertRaises(ValueError):
            generate_analysis_grid(self.t_obs, window_hours=72.0, interval_hours=0.0)

    def test_negative_interval_hours_raises(self):
        with self.assertRaises(ValueError):
            generate_analysis_grid(self.t_obs, window_hours=72.0, interval_hours=-1.0)

    def test_t_obs_accepts_string_input(self):
        grid = generate_analysis_grid("2019-10-14T23:00:00", window_hours=2.0, interval_hours=1.0)
        self.assertEqual(len(grid), 3)
        self.assertEqual(grid[0], np.datetime64("2019-10-14T23:00:00", "ns"))


if __name__ == "__main__":
    unittest.main()
