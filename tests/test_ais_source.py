"""Focused TASK AIS-6 checks for `iste.ais_source.RealAISDataSource`.

Run with:
    python -m unittest tests.test_ais_source -v
"""

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from iste.ais_source import RealAISDataSource

COLUMNS = ["mmsi", "timestamp", "lat", "lon", "sog", "cog", "vessel_name", "imo", "vessel_type", "type_code", "length", "width", "draught"]


def _row(mmsi, ts, lat, lon):
    return dict(mmsi=mmsi, timestamp=ts, lat=lat, lon=lon, sog=5.0, cog=90.0, vessel_name="X", imo=1, vessel_type="Cargo", type_code=70, length=1, width=1, draught=1.0)


def _make_source(rows) -> RealAISDataSource:
    df = pd.DataFrame(rows, columns=COLUMNS)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ais.csv")
        df.to_csv(path, index=False)
        return RealAISDataSource(path)


class TestRealAISDataSource(unittest.TestCase):
    def test_finds_vessel_within_radius_and_time_window(self):
        src = _make_source([_row(1, "2019-06-12T00:35:58Z", 34.0, 35.0)])
        result = src.query_vessels_near(34.0, 35.0, np.datetime64("2019-06-12T00:35:58", "ns"), radius_m=1000.0, time_window_hours=1.0)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].vessel_id, "1")

    def test_excludes_vessel_outside_radius(self):
        src = _make_source([_row(1, "2019-06-12T00:35:58Z", 40.0, 40.0)])
        result = src.query_vessels_near(34.0, 35.0, np.datetime64("2019-06-12T00:35:58", "ns"), radius_m=1000.0, time_window_hours=1.0)
        self.assertEqual(len(result), 0)

    def test_excludes_vessel_outside_time_window(self):
        src = _make_source([_row(1, "2019-06-12T10:35:58Z", 34.0, 35.0)])
        result = src.query_vessels_near(34.0, 35.0, np.datetime64("2019-06-12T00:35:58", "ns"), radius_m=1000.0, time_window_hours=1.0)
        self.assertEqual(len(result), 0)

    def test_empty_source_returns_empty_list(self):
        src = _make_source([])
        result = src.query_vessels_near(34.0, 35.0, np.datetime64("2019-06-12T00:35:58", "ns"), radius_m=1000.0, time_window_hours=1.0)
        self.assertEqual(result, [])

    def test_multiple_vessels_all_returned(self):
        src = _make_source([_row(1, "2019-06-12T00:35:58Z", 34.0, 35.0), _row(2, "2019-06-12T00:35:58Z", 34.001, 35.001)])
        result = src.query_vessels_near(34.0, 35.0, np.datetime64("2019-06-12T00:35:58", "ns"), radius_m=1000.0, time_window_hours=1.0)
        self.assertEqual(len(result), 2)
        self.assertEqual({r.vessel_id for r in result}, {"1", "2"})

    def test_no_position_interpolated_positions_match_source_rows(self):
        src = _make_source([_row(1, "2019-06-12T00:35:58Z", 34.1234, 35.5678)])
        result = src.query_vessels_near(34.1234, 35.5678, np.datetime64("2019-06-12T00:35:58", "ns"), radius_m=1000.0, time_window_hours=1.0)
        self.assertAlmostEqual(result[0].lat, 34.1234)
        self.assertAlmostEqual(result[0].lon, 35.5678)


if __name__ == "__main__":
    unittest.main()
