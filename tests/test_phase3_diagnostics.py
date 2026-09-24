"""Focused T28 checks for `iste.phase3_diagnostics` (Phase 3 per-timestamp diagnostics).

Run with:
    python -m unittest tests.test_phase3_diagnostics -v
"""

from __future__ import annotations

import unittest

import numpy as np

from iste.candidate_extraction import CandidateExtractionConfig
from iste.ensemble_run import EnsembleRunRecords
from iste.phase3_diagnostics import compute_diagnostics_table, compute_timestamp_diagnostics
from iste.support_features import SupportFeatureConfig

T_S = np.datetime64("2020-01-01T00:00:00", "ns")
NAT = np.datetime64("NaT", "ns")

LAT_GRID = np.linspace(9.9, 10.1, 11)
LON_GRID = np.linspace(99.9, 100.1, 11)

# Generous config/radius so a handful of synthetic particles produce non-trivial support.
RADIUS_M = 5000.0
EXTRACTION_CONFIG = CandidateExtractionConfig(
    peak_neighborhood_cells=3, min_separation_m=1.0, core_radius_m=500.0, min_support=0.0
)


def _records(particle_lats, particle_lons, n_pixels=1) -> EnsembleRunRecords:
    n = len(particle_lats)
    particle_id = np.array([f"p{i}" for i in range(n)], dtype=object)
    pixel_id = np.array([f"px-{i % n_pixels}" for i in range(n)], dtype=object)
    return EnsembleRunRecords(
        particle_id=particle_id,
        forcing_member_id=np.zeros(n, dtype=np.int64),
        observation_pixel_id=pixel_id,
        timestamp=np.full(n, T_S, dtype="datetime64[ns]"),
        lat=np.asarray(particle_lats, dtype=np.float64),
        lon=np.asarray(particle_lons, dtype=np.float64),
        log_weight=np.zeros(n, dtype=np.float64),
        valid=np.ones(n, dtype=bool),
    )


class TestTightVsDispersedCluster(unittest.TestCase):
    """A synthetic tight cluster vs. a synthetic dispersed cluster (same particle count,
    same centroid) must produce the expected ordering on every diagnostic -- this is the
    exact ordering Phase 3 needs to be able to see (T28 does not fix the bias, only
    measures it)."""

    def setUp(self):
        n = 40
        rng = np.random.default_rng(0)
        centroid_lat, centroid_lon = 10.0, 100.0

        tight_lat = centroid_lat + rng.normal(0, 0.0005, n)
        tight_lon = centroid_lon + rng.normal(0, 0.0005, n)
        self.tight_records = _records(tight_lat, tight_lon)

        dispersed_lat = centroid_lat + rng.normal(0, 0.03, n)
        dispersed_lon = centroid_lon + rng.normal(0, 0.03, n)
        self.dispersed_records = _records(dispersed_lat, dispersed_lon)

        kwargs = dict(
            support_lat_grid=LAT_GRID, support_lon_grid=LON_GRID,
            neighborhood_radius_m=RADIUS_M,
            support_config=SupportFeatureConfig(),
            extraction_config=EXTRACTION_CONFIG,
        )
        self.tight = compute_timestamp_diagnostics(self.tight_records, T_S, **kwargs)
        self.dispersed = compute_timestamp_diagnostics(self.dispersed_records, T_S, **kwargs)

    def test_particle_count_preserved(self):
        self.assertEqual(self.tight.n_particles, 40)
        self.assertEqual(self.dispersed.n_particles, 40)

    def test_dispersed_has_larger_spatial_spread(self):
        self.assertGreater(self.dispersed.spatial_spread_m, self.tight.spatial_spread_m)

    def test_dispersed_has_larger_variance(self):
        self.assertGreater(self.dispersed.lat_var, self.tight.lat_var)
        self.assertGreater(self.dispersed.lon_var, self.tight.lon_var)

    def test_tight_has_higher_kde_max(self):
        self.assertGreater(self.tight.kde_max, self.dispersed.kde_max)

    def test_dispersed_has_larger_kde_entropy(self):
        self.assertGreater(self.dispersed.kde_entropy, self.tight.kde_entropy)

    def test_dispersed_has_larger_effective_area(self):
        self.assertGreater(self.dispersed.kde_effective_area_m2, self.tight.kde_effective_area_m2)

    def test_convex_hull_area_larger_for_dispersed(self):
        self.assertIsNotNone(self.tight.convex_hull_area_m2)
        self.assertIsNotNone(self.dispersed.convex_hull_area_m2)
        self.assertGreater(self.dispersed.convex_hull_area_m2, self.tight.convex_hull_area_m2)

    def test_kde_bandwidth_echoes_configured_radius(self):
        self.assertEqual(self.tight.kde_bandwidth_m, RADIUS_M)


class TestEmptyParticleSet(unittest.TestCase):
    """No valid particles at this timestamp -> diagnostics must not crash, and must be
    explicit (NaN/None/zero) rather than fabricating a value."""

    def test_no_valid_particles_returns_nan_not_crash(self):
        records = EnsembleRunRecords(
            particle_id=np.array([], dtype=object),
            forcing_member_id=np.array([], dtype=np.int64),
            observation_pixel_id=np.array([], dtype=object),
            timestamp=np.array([], dtype="datetime64[ns]"),
            lat=np.array([]), lon=np.array([]), log_weight=np.array([]),
            valid=np.array([], dtype=bool),
        )
        diag = compute_timestamp_diagnostics(
            records, T_S,
            support_lat_grid=LAT_GRID, support_lon_grid=LON_GRID,
            neighborhood_radius_m=RADIUS_M, extraction_config=EXTRACTION_CONFIG,
        )
        self.assertEqual(diag.n_particles, 0)
        self.assertTrue(np.isnan(diag.lon_mean))
        self.assertIsNone(diag.convex_hull_area_m2)
        self.assertEqual(diag.n_significant_modes, 0)

    def test_two_particles_no_hull_area(self):
        # < 3 particles: convex hull is undefined by this module's own rule -> None.
        records = _records([10.0, 10.001], [100.0, 100.001])
        diag = compute_timestamp_diagnostics(
            records, T_S,
            support_lat_grid=LAT_GRID, support_lon_grid=LON_GRID,
            neighborhood_radius_m=RADIUS_M, extraction_config=EXTRACTION_CONFIG,
        )
        self.assertEqual(diag.n_particles, 2)
        self.assertIsNone(diag.convex_hull_area_m2)

    def test_collinear_points_no_hull_area(self):
        # exactly collinear -> degenerate convex hull -> None, not a crash.
        lat = np.full(5, 10.0)
        lon = np.array([99.98, 99.99, 100.0, 100.01, 100.02])
        records = _records(lat, lon)
        diag = compute_timestamp_diagnostics(
            records, T_S,
            support_lat_grid=LAT_GRID, support_lon_grid=LON_GRID,
            neighborhood_radius_m=RADIUS_M, extraction_config=EXTRACTION_CONFIG,
        )
        self.assertIsNone(diag.convex_hull_area_m2)


class TestModeCountReusesT17(unittest.TestCase):
    def test_two_well_separated_clusters_yield_two_modes(self):
        rng = np.random.default_rng(1)
        cluster_a_lat = 9.95 + rng.normal(0, 0.0005, 15)
        cluster_a_lon = 99.95 + rng.normal(0, 0.0005, 15)
        cluster_b_lat = 10.05 + rng.normal(0, 0.0005, 15)
        cluster_b_lon = 100.05 + rng.normal(0, 0.0005, 15)
        lat = np.concatenate([cluster_a_lat, cluster_b_lat])
        lon = np.concatenate([cluster_a_lon, cluster_b_lon])
        records = _records(lat, lon)

        diag = compute_timestamp_diagnostics(
            records, T_S,
            support_lat_grid=LAT_GRID, support_lon_grid=LON_GRID,
            neighborhood_radius_m=RADIUS_M,
            extraction_config=CandidateExtractionConfig(
                peak_neighborhood_cells=3, min_separation_m=1.0, core_radius_m=500.0, min_support=0.0
            ),
        )
        self.assertGreaterEqual(diag.n_significant_modes, 2)


class TestDiagnosticsTable(unittest.TestCase):
    def test_table_length_matches_grid_and_preserves_order(self):
        grid = np.array(
            [T_S, T_S - np.timedelta64(1, "h"), T_S - np.timedelta64(2, "h")], dtype="datetime64[ns]"
        )
        records = _records([10.0, 10.001, 10.002], [100.0, 100.001, 100.002])
        table = compute_diagnostics_table(
            records, grid,
            support_lat_grid=LAT_GRID, support_lon_grid=LON_GRID,
            neighborhood_radius_m=RADIUS_M, extraction_config=EXTRACTION_CONFIG,
        )
        self.assertEqual(len(table), 3)
        for expected_t, diag in zip(grid, table):
            self.assertEqual(diag.t_s, expected_t)


if __name__ == "__main__":
    unittest.main()
