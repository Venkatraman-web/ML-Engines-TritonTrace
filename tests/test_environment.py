"""Focused T01 checks for `iste.environment`.

Run with:
    python -m unittest tests.test_environment -v

Uses only the Python standard library's `unittest` (no new test-framework dependency).
"""

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np
import xarray as xr

from iste.environment import EnvironmentalFieldProvider, load_field_dataset

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OCEAN_PATH = os.path.join(REPO_ROOT, "data", "ocean.nc")
WIND_PATH = os.path.join(REPO_ROOT, "data", "wind.nc")


class TestLoadOceanDataset(unittest.TestCase):
    """Checks against the real ocean.nc, cross-checked against the T00 inspection findings."""

    @classmethod
    def setUpClass(cls):
        cls.src = load_field_dataset(
            OCEAN_PATH, ["uo", "vo"], time_dim="time", lat_dim="lat", lon_dim="lon"
        )

    def test_shape_and_timestamp_count(self):
        # June-2019 event-correction rerun (2026-09-24): data/ocean.nc was replaced with
        # real June 2019 currents (dims renamed latitude/longitude -> lat/lon, a much
        # larger regional extent, 192 hourly steps instead of 336). These are facts about
        # the current real file, re-verified directly against it, not assumed.
        self.assertEqual(self.src.data["uo"].shape, (192, 380, 1016))
        self.assertEqual(len(self.src.time), 192)

    def test_time_range(self):
        self.assertEqual(str(self.src.time[0])[:19], "2019-06-07T00:30:00")
        self.assertEqual(str(self.src.time[-1])[:19], "2019-06-14T23:30:00")

    def test_latitude_ascending(self):
        self.assertTrue(self.src.lat_ascending)

    def test_missing_count_matches_t00_finding(self):
        # Re-verified directly against the current (June-2019) data/ocean.nc: this file's
        # grid spans nearly the whole Mediterranean (lat 30.19..45.98, lon -6.0..36.29),
        # far beyond the product's actual data footprint/coastline, so the large majority
        # of cells are native NaN (land or out-of-domain), not |x| >= 1e19 sentinel fills.
        self.assertEqual(self.src.missing_count("uo"), 66_727_992)

    def test_describe_table_is_non_empty_and_mentions_variables(self):
        table = self.src.describe().as_table()
        self.assertIn("uo", table)
        self.assertIn("vo", table)
        self.assertIn("ascending", table)


class TestLoadWindDataset(unittest.TestCase):
    """Checks against the real wind.nc: different dim names, opposite latitude order."""

    @classmethod
    def setUpClass(cls):
        cls.src = load_field_dataset(
            WIND_PATH, ["u10", "v10"], time_dim="valid_time", lat_dim="latitude", lon_dim="longitude"
        )

    def test_shape_and_timestamp_count(self):
        # June-2019 event-correction rerun (2026-09-24): data/wind.nc was replaced with
        # real June 2019 winds -- same dim names/grid size as before, 192 hourly steps
        # instead of 336 (re-verified directly against the current file).
        self.assertEqual(self.src.data["u10"].shape, (192, 25, 36))
        self.assertEqual(len(self.src.time), 192)

    def test_latitude_descending(self):
        self.assertFalse(self.src.lat_ascending)

    def test_no_missing_values_found(self):
        self.assertEqual(self.src.missing_count("u10"), 0)
        self.assertEqual(self.src.missing_count("v10"), 0)

    def test_time_covers_same_window_as_ocean_file(self):
        # June-2019 event-correction rerun (2026-09-24): unlike the October files (which
        # shared identical on-the-hour timestamps), the new ocean.nc is an hourly-MEAN-
        # centered-at-time product offset 30 minutes from wind.nc's on-the-hour
        # valid_time -- re-verified directly against both current files, not assumed to
        # still be exactly equal. This does not affect Stage III (T01.5: wind.nc is never
        # read for U/forcing). Both cover the same calendar window at 1-hour resolution.
        ocean = load_field_dataset(
            OCEAN_PATH, ["uo"], time_dim="time", lat_dim="lat", lon_dim="lon"
        )
        self.assertEqual(len(self.src.time), len(ocean.time))
        self.assertEqual(str(self.src.time[0])[:10], str(ocean.time[0])[:10])
        self.assertEqual(str(self.src.time[-1])[:10], str(ocean.time[-1])[:10])
        offset = (ocean.time[0] - self.src.time[0]) / np.timedelta64(1, "m")
        self.assertEqual(offset, 30.0)


class TestLoaderIsLayoutAgnostic(unittest.TestCase):
    """Proves `load_field_dataset` does not hard-code either project file's layout: it
    must work against a synthetic file using entirely different dimension/variable names."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmpdir.name, "synthetic.nc")
        ds = xr.Dataset(
            data_vars={
                "custom_speed": (("T", "Y", "X"), np.array([[[1.0, 2.0], [3.0, 1e20]]])),
            },
            coords={
                "T": np.array(["2020-01-01"], dtype="datetime64[ns]"),
                "Y": np.array([10.0, 20.0]),  # ascending, unlike either project file's naming
                "X": np.array([100.0, 101.0]),
            },
        )
        ds.to_netcdf(self.path)
        ds.close()

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_synthetic_layout_loads_correctly(self):
        src = load_field_dataset(
            self.path, ["custom_speed"], time_dim="T", lat_dim="Y", lon_dim="X"
        )
        self.assertEqual(src.data["custom_speed"].shape, (1, 2, 2))
        self.assertTrue(src.lat_ascending)
        # the 1e20 cell must be masked to NaN, the rest must survive untouched
        self.assertEqual(src.missing_count("custom_speed"), 1)
        self.assertEqual(src.data["custom_speed"][0, 0, 0], 1.0)
        self.assertTrue(np.isnan(src.data["custom_speed"][0, 1, 1]))

    def test_unknown_dimension_name_raises_keyerror(self):
        with self.assertRaises(KeyError):
            load_field_dataset(
                self.path, ["custom_speed"], time_dim="not_a_real_dim", lat_dim="Y", lon_dim="X"
            )

    def test_unknown_variable_name_raises_keyerror(self):
        with self.assertRaises(KeyError):
            load_field_dataset(
                self.path, ["not_a_real_variable"], time_dim="T", lat_dim="Y", lon_dim="X"
            )


class TestEnvironmentalFieldProviderIsAbstract(unittest.TestCase):
    """T01 must define the Stage III interface without implementing it yet."""

    def test_cannot_instantiate_directly(self):
        with self.assertRaises(TypeError):
            EnvironmentalFieldProvider()  # abstract methods unimplemented

    def test_partial_subclass_still_cannot_be_instantiated(self):
        class PartialProvider(EnvironmentalFieldProvider):
            def U(self, x, t):
                return np.zeros(2)

            # K, grad_K, div_U, validity intentionally left unimplemented

        with self.assertRaises(TypeError):
            PartialProvider()

    def test_full_subclass_can_be_instantiated_but_raises_when_called(self):
        # Demonstrates the interface contract without inventing any physics: a stub
        # subclass compiles (proves the signatures are usable) but each method still
        # raises NotImplementedError, since T02/T03 have not been implemented yet.
        class StubProvider(EnvironmentalFieldProvider):
            def U(self, x, t):
                raise NotImplementedError

            def K(self, x, t):
                raise NotImplementedError

            def grad_K(self, x, t):
                raise NotImplementedError

            def div_U(self, x, t):
                raise NotImplementedError

            def validity(self, x, t):
                raise NotImplementedError

        provider = StubProvider()
        with self.assertRaises(NotImplementedError):
            provider.U(np.zeros(2), np.datetime64("2019-10-01"))
        with self.assertRaises(NotImplementedError):
            provider.K(np.zeros(2), np.datetime64("2019-10-01"))


if __name__ == "__main__":
    unittest.main()
