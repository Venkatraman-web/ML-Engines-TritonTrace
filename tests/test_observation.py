"""Focused T07 checks for `iste.observation` (backtracking input adapter / particle
interface).

Run with:
    python -m unittest tests.test_observation -v
"""

from __future__ import annotations

import unittest

import numpy as np

from iste.environment import GriddedFieldSource
from iste.observation import REQUIRED_UPSTREAM_FIELDS, adapt_upstream_particles, tag_trajectory
from iste.trajectory import run_backward_trajectory

LAT = np.array([10.0, 20.0, 30.0])
LON = np.array([100.0, 110.0, 120.0])
T0 = np.datetime64("2020-01-01T06:00", "ns")
TIME = T0 - np.arange(5)[::-1] * np.timedelta64(3600, "s")
DT = 900.0
ZERO_K = np.zeros((3, 3))


def make_ocean(uo_grid: np.ndarray, vo_grid: np.ndarray) -> GriddedFieldSource:
    n_t = len(TIME)
    uo = np.stack([uo_grid] * n_t)
    vo = np.stack([vo_grid] * n_t)
    return GriddedFieldSource(
        source_path="<synthetic>", time_name="time", time=TIME, lat_name="lat", lat=LAT,
        lon_name="lon", lon=LON, data={"uo": uo, "vo": vo}, fill_threshold=1e19,
    )


def mock_upstream_payload():
    """Stands in for the teammate's satellite-segmentation/particle-generation output --
    this module never produces records like these itself, only consumes them."""
    return [
        {"particle_id": "p0", "latitude": 20.0, "longitude": 109.0, "observation_pixel_id": "px-0", "timestamp": T0},
        {"particle_id": "p1", "latitude": 20.0, "longitude": 110.0, "observation_pixel_id": "px-1", "timestamp": T0},
        {"particle_id": "p2", "latitude": 20.0, "longitude": 111.0, "observation_pixel_id": "px-1", "timestamp": T0},
        {"particle_id": "p3", "latitude": 20.0, "longitude": 112.0, "observation_pixel_id": "px-2", "timestamp": T0, "valid": False},
    ]


class TestValidUpstreamInput(unittest.TestCase):
    def test_adapts_a_valid_payload(self):
        adapted = adapt_upstream_particles(mock_upstream_payload())
        self.assertEqual(adapted.n_particles, 4)

    def test_position_preserved(self):
        adapted = adapt_upstream_particles(mock_upstream_payload())
        np.testing.assert_allclose(adapted.state.lat, [20.0, 20.0, 20.0, 20.0])
        np.testing.assert_allclose(adapted.state.lon, [109.0, 110.0, 111.0, 112.0])

    def test_timestamp_preserved(self):
        adapted = adapt_upstream_particles(mock_upstream_payload())
        self.assertTrue(np.all(adapted.timestamp == T0))

    def test_particle_id_preserved(self):
        adapted = adapt_upstream_particles(mock_upstream_payload())
        self.assertEqual(list(adapted.particle_id), ["p0", "p1", "p2", "p3"])

    def test_observation_pixel_id_preserved(self):
        adapted = adapt_upstream_particles(mock_upstream_payload())
        self.assertEqual(list(adapted.observation_pixel_id), ["px-0", "px-1", "px-1", "px-2"])


class TestMissingRequiredFieldRejection(unittest.TestCase):
    def test_missing_field_raises_keyerror_naming_index_and_field(self):
        records = mock_upstream_payload()
        del records[2]["observation_pixel_id"]
        with self.assertRaises(KeyError) as ctx:
            adapt_upstream_particles(records)
        self.assertIn("2", str(ctx.exception))
        self.assertIn("observation_pixel_id", str(ctx.exception))

    def test_each_required_field_is_actually_enforced(self):
        for field in REQUIRED_UPSTREAM_FIELDS:
            records = mock_upstream_payload()
            del records[0][field]
            with self.assertRaises(KeyError):
                adapt_upstream_particles(records)

    def test_empty_payload_rejected(self):
        with self.assertRaises(ValueError):
            adapt_upstream_particles([])

    def test_duplicate_particle_id_rejected(self):
        records = mock_upstream_payload()
        records[1]["particle_id"] = records[0]["particle_id"]
        with self.assertRaises(ValueError):
            adapt_upstream_particles(records)


class TestInvalidUpstreamParticleRepresented(unittest.TestCase):
    def test_upstream_valid_flag_is_honored_not_dropped(self):
        adapted = adapt_upstream_particles(mock_upstream_payload())
        # record index 3 ("p3") was supplied with valid=False.
        self.assertEqual(adapted.n_particles, 4)  # still present, not dropped
        np.testing.assert_array_equal(adapted.state.valid, [True, True, True, False])
        self.assertEqual(adapted.particle_id[3], "p3")
        self.assertEqual(adapted.observation_pixel_id[3], "px-2")

    def test_valid_defaults_to_true_when_absent(self):
        records = mock_upstream_payload()
        self.assertNotIn("valid", records[0])
        adapted = adapt_upstream_particles(records)
        self.assertTrue(bool(adapted.state.valid[0]))


class TestHandoffIntoExistingBacktrackingRepresentation(unittest.TestCase):
    """Confirms the adapted output is directly usable by T06's unmodified single-particle
    runner, and that observation_pixel_id/particle_id survive a full trajectory."""

    def test_adapted_particle_runs_through_t06_and_preserves_ids(self):
        adapted = adapt_upstream_particles(mock_upstream_payload())
        i = 1  # "p1", pixel "px-1", starts valid
        ocean = make_ocean(np.full((3, 3), 0.3), np.full((3, 3), -0.1))

        traj = run_backward_trajectory(
            float(adapted.state.lat[i]), float(adapted.state.lon[i]), T0, n_steps=3, dt_seconds=DT,
            ocean_source=ocean, k_lat=LAT, k_lon=LON, k_field=ZERO_K, rng=np.random.default_rng(1),
            log_weight0=float(adapted.state.log_weight[i]),
        )
        tagged = tag_trajectory(
            traj, observation_pixel_id=adapted.observation_pixel_id[i], particle_id=adapted.particle_id[i]
        )
        self.assertEqual(tagged.observation_pixel_id, "px-1")
        self.assertEqual(tagged.particle_id, "p1")
        self.assertTrue(np.all(tagged.trajectory.valid))

    def test_upstream_invalid_particle_is_faithfully_represented_without_running_t06(self):
        # T06's single-particle runner (`run_backward_trajectory`) always starts from a
        # fresh, valid state -- it has no "start already invalid" mode, and T06 is not
        # modified here to add one. The correct handoff behavior is therefore: an
        # already-invalid upstream particle is faithfully represented in the adapted
        # output (not dropped, not silently treated as valid), and a caller simply does
        # not invoke T06 for it -- there is nothing left to propagate.
        adapted = adapt_upstream_particles(mock_upstream_payload())
        i = 3  # "p3", pixel "px-2", already invalid upstream
        self.assertFalse(bool(adapted.state.valid[i]))
        self.assertEqual(adapted.particle_id[i], "p3")
        self.assertEqual(adapted.observation_pixel_id[i], "px-2")
        self.assertEqual(float(adapted.state.lat[i]), 20.0)
        self.assertEqual(float(adapted.state.lon[i]), 112.0)


if __name__ == "__main__":
    unittest.main()
