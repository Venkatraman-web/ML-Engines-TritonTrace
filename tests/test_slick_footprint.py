"""Focused T22 checks for `iste.slick_footprint` (predicted-slick KDE footprint and
independent observed-slick reference).

Run with:
    python -m unittest tests.test_slick_footprint -v
"""

from __future__ import annotations

import ast
import inspect
import unittest

import numpy as np

from iste.forward_propagation import ForwardCandidateMemberRun
from iste.geometry import metres_per_degree_latitude, metres_per_degree_longitude
from iste.kdtree_neighborhood import project_to_local_metres
from iste.observation import adapt_upstream_particles
from iste.slick_footprint import (
    FootprintGridConfig,
    ObservedSlickFootprint,
    PredictedSlickFootprint,
    build_observed_slick_footprint,
    build_predicted_slick_footprint,
    build_predicted_slick_footprints,
    gaussian_kernel_2d,
)

T_S = np.datetime64("2020-01-01T00:00:00", "ns")
T_OBS = np.datetime64("2020-01-01T06:00:00", "ns")
LAT_GRID = np.linspace(19.0, 21.0, 21)  # 0.1 deg spacing
LON_GRID = np.linspace(99.0, 101.0, 21)


def make_run(lat, lon, valid=None, candidate_id="cand-A", forward_forcing_member_id=0):
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    n = len(lat)
    valid_arr = np.array(valid, dtype=bool) if valid is not None else np.ones(n, dtype=bool)
    return ForwardCandidateMemberRun(
        candidate_id=candidate_id,
        t_s=T_S,
        t_obs=T_OBS,
        forward_forcing_member_id=forward_forcing_member_id,
        dt_seconds=900.0,
        n_steps=0,
        timestamps=np.array([T_OBS], dtype="datetime64[ns]"),
        particle_id=np.array([f"p{i}" for i in range(n)], dtype=object),
        observation_pixel_id=np.array([f"px-{i}" for i in range(n)], dtype=object),
        forcing_member_id=np.zeros(n, dtype=np.int64),
        original_log_weight=np.zeros(n, dtype=np.float64),
        lat=lat.reshape(1, n),
        lon=lon.reshape(1, n),
        valid=valid_arr.reshape(1, n),
    )


def make_adapted(lat, lon, valid=None):
    n = len(lat)
    valid = valid if valid is not None else [True] * n
    records = [
        {
            "particle_id": f"o{i}", "latitude": lat[i], "longitude": lon[i],
            "observation_pixel_id": f"opx-{i}", "timestamp": T_OBS, "valid": valid[i],
        }
        for i in range(n)
    ]
    return adapt_upstream_particles(records)


class TestPredictedFootprintNearExpectedLocation(unittest.TestCase):
    """Requirement 1: known synthetic forward particles -> footprint peak near expected
    location."""

    def test_peak_density_cell_matches_cluster_centre(self):
        rng = np.random.default_rng(0)
        n = 30
        lat = 20.0 + rng.normal(0.0, 0.02, n)
        lon = 100.0 + rng.normal(0.0, 0.02, n)
        run = make_run(lat, lon)
        config = FootprintGridConfig(lat_grid=LAT_GRID, lon_grid=LON_GRID, bandwidth_m=3000.0)
        fp = build_predicted_slick_footprint(run, config)

        peak_idx = np.unravel_index(np.nanargmax(fp.density), fp.density.shape)
        peak_lat = LAT_GRID[peak_idx[0]]
        peak_lon = LON_GRID[peak_idx[1]]
        self.assertAlmostEqual(peak_lat, 20.0, delta=0.15)
        self.assertAlmostEqual(peak_lon, 100.0, delta=0.15)


class TestObservedReferenceIndependent(unittest.TestCase):
    """Requirement 2: observed reference is independent of the backward candidate --
    built only from upstream observation particles, never from a HistoricalParticleSlice
    or a T21 forward run."""

    def test_observed_footprint_needs_no_candidate_data(self):
        adapted = make_adapted([20.5, 20.6, 20.4], [100.2, 100.3, 100.1])
        config = FootprintGridConfig(lat_grid=LAT_GRID, lon_grid=LON_GRID, bandwidth_m=3000.0)
        fp = build_observed_slick_footprint(adapted, T_OBS, config)
        self.assertIsInstance(fp, ObservedSlickFootprint)
        self.assertTrue(fp.has_valid_particles)
        # ObservedSlickFootprint carries no candidate_id / forcing_member_id at all
        self.assertFalse(hasattr(fp, "candidate_id"))
        self.assertFalse(hasattr(fp, "forward_forcing_member_id"))

    def test_source_never_imports_backward_candidate_types(self):
        import iste.slick_footprint as mod
        tree = ast.parse(inspect.getsource(mod))
        imported_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                imported_names.update(alias.name for alias in node.names)
            elif isinstance(node, ast.Import):
                imported_names.update(alias.name for alias in node.names)
        for forbidden in ("HistoricalParticleSlice", "ParticleNeighborhoodIndex"):
            self.assertNotIn(forbidden, imported_names)

    def test_module_does_not_import_particle_query(self):
        import iste.slick_footprint as mod
        tree = ast.parse(inspect.getsource(mod))
        imported_modules = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.add(node.module)
            elif isinstance(node, ast.Import):
                imported_modules.update(alias.name for alias in node.names)
        self.assertNotIn("iste.particle_query", imported_modules)


class TestDifferentParticleCounts(unittest.TestCase):
    """Requirement 3 + tasks.md's own check: predicted/observed work with different
    particle counts, and particle-count changes alone do not change a normalized
    footprint score -- proven exactly via exact duplication (2N identical points ->
    identical normalized density to N points)."""

    def test_predicted_and_observed_have_independent_particle_counts(self):
        run = make_run([20.0] * 5, [100.0] * 5)
        adapted = make_adapted([20.0] * 12, [100.0] * 12)
        config = FootprintGridConfig(lat_grid=LAT_GRID, lon_grid=LON_GRID, bandwidth_m=3000.0)
        predicted = build_predicted_slick_footprint(run, config)
        observed = build_observed_slick_footprint(adapted, T_OBS, config)
        self.assertEqual(predicted.n_valid_particles, 5)
        self.assertEqual(observed.n_valid_particles, 12)
        self.assertEqual(predicted.density.shape, observed.density.shape)  # comparable grid

    def test_exact_duplication_leaves_normalized_density_unchanged(self):
        rng = np.random.default_rng(1)
        n = 8
        lat = 20.0 + rng.normal(0.0, 0.02, n)
        lon = 100.0 + rng.normal(0.0, 0.02, n)
        run_single = make_run(lat, lon)
        run_doubled = make_run(np.concatenate([lat, lat]), np.concatenate([lon, lon]))

        config = FootprintGridConfig(lat_grid=LAT_GRID, lon_grid=LON_GRID, bandwidth_m=3000.0)
        fp_single = build_predicted_slick_footprint(run_single, config)
        fp_doubled = build_predicted_slick_footprint(run_doubled, config)

        self.assertEqual(fp_single.n_valid_particles, 8)
        self.assertEqual(fp_doubled.n_valid_particles, 16)
        np.testing.assert_allclose(fp_single.density, fp_doubled.density, atol=1e-12)


class TestInvalidParticlesExcluded(unittest.TestCase):
    """Requirement 4: invalid forward particles do not contribute."""

    def test_moving_an_invalid_particle_does_not_change_density(self):
        lat = [20.0, 20.0, 20.0]
        lon = [100.0, 100.0, 100.0]
        valid = [True, True, False]
        run_a = make_run(lat, lon, valid=valid)
        run_b = make_run([20.0, 20.0, 25.0], [100.0, 100.0, 108.0], valid=valid)  # invalid one moved far away

        config = FootprintGridConfig(lat_grid=LAT_GRID, lon_grid=LON_GRID, bandwidth_m=3000.0)
        fp_a = build_predicted_slick_footprint(run_a, config)
        fp_b = build_predicted_slick_footprint(run_b, config)
        np.testing.assert_array_equal(fp_a.density, fp_b.density)
        self.assertEqual(fp_a.n_valid_particles, 2)
        self.assertEqual(fp_a.n_particles, 3)


class TestConfigurationChangesOnlyRepresentation(unittest.TestCase):
    """Requirement 5: changing KDE bandwidth/config changes only the configured
    representation, not the source particle data."""

    def test_different_bandwidth_gives_different_density_but_same_source_data(self):
        run = make_run([20.0, 20.01, 19.99], [100.0, 100.01, 99.99])
        lat_before = run.lat.copy()
        lon_before = run.lon.copy()

        config_narrow = FootprintGridConfig(lat_grid=LAT_GRID, lon_grid=LON_GRID, bandwidth_m=1000.0)
        config_wide = FootprintGridConfig(lat_grid=LAT_GRID, lon_grid=LON_GRID, bandwidth_m=8000.0)
        fp_narrow = build_predicted_slick_footprint(run, config_narrow)
        fp_wide = build_predicted_slick_footprint(run, config_wide)

        self.assertFalse(np.allclose(fp_narrow.density, fp_wide.density))
        np.testing.assert_array_equal(run.lat, lat_before)  # source data untouched
        np.testing.assert_array_equal(run.lon, lon_before)


class TestPhysicalDistanceHandling(unittest.TestCase):
    """Requirement 6: physical-distance handling is correct -- exact Gaussian value at a
    hand-computed metre distance, and longitude-degree spacing that correctly shrinks at
    higher latitude (never raw degrees treated as metres)."""

    def test_exact_kernel_value_at_known_distance(self):
        # single particle exactly at the grid's reference point; evaluate at a grid
        # point placed EXACTLY bandwidth_m away along the latitude axis.
        bandwidth_m = 5000.0
        lat_ref = 20.0
        lon_ref = 100.0
        dlat_deg = bandwidth_m / metres_per_degree_latitude()
        lat_grid = np.array([lat_ref, lat_ref + dlat_deg])
        lon_grid = np.array([lon_ref])
        run = make_run([lat_ref], [lon_ref])
        config = FootprintGridConfig(
            lat_grid=lat_grid, lon_grid=lon_grid, bandwidth_m=bandwidth_m, lat_ref=lat_ref, lon_ref=lon_ref,
        )
        fp = build_predicted_slick_footprint(run, config)

        expected_at_ref = gaussian_kernel_2d(np.array([0.0]), bandwidth_m)[0]
        expected_at_offset = gaussian_kernel_2d(np.array([bandwidth_m]), bandwidth_m)[0]
        self.assertAlmostEqual(fp.density[0, 0], expected_at_ref, places=9)
        self.assertAlmostEqual(fp.density[1, 0], expected_at_offset, places=9)

    def test_longitude_spacing_uses_geometry_not_raw_degrees(self):
        # at high latitude, 1 degree of longitude is much less than 1 degree of latitude
        # in metres -- if degrees were treated as raw metres this asymmetry would vanish.
        lat_ref = 60.0
        lon_ref = 100.0
        run = make_run([lat_ref], [lon_ref])
        lat_grid = np.array([lat_ref, lat_ref + 0.05])
        lon_grid = np.array([lon_ref, lon_ref + 0.05])
        config = FootprintGridConfig(
            lat_grid=lat_grid, lon_grid=lon_grid, bandwidth_m=10000.0, lat_ref=lat_ref, lon_ref=lon_ref,
        )
        fp = build_predicted_slick_footprint(run, config)
        # 0.05 deg lon at 60N is a SMALLER metre distance than 0.05 deg lat anywhere ->
        # the density at the lon-offset point must be HIGHER (closer) than at the
        # lat-offset point.
        density_at_lat_offset = fp.density[1, 0]
        density_at_lon_offset = fp.density[0, 1]
        self.assertGreater(density_at_lon_offset, density_at_lat_offset)


class TestProvenancePreserved(unittest.TestCase):
    """Requirement 7: candidate/member provenance preserved."""

    def test_predicted_footprint_carries_candidate_and_member_identity(self):
        run = make_run([20.0], [100.0], candidate_id="cand-XYZ", forward_forcing_member_id=4)
        config = FootprintGridConfig(lat_grid=LAT_GRID, lon_grid=LON_GRID, bandwidth_m=3000.0)
        fp = build_predicted_slick_footprint(run, config)
        self.assertEqual(fp.candidate_id, "cand-XYZ")
        self.assertEqual(fp.forward_forcing_member_id, 4)
        self.assertEqual(fp.t_s, T_S)
        self.assertEqual(fp.t_obs, T_OBS)

    def test_build_predicted_slick_footprints_keys_by_member(self):
        run0 = make_run([20.0], [100.0], forward_forcing_member_id=0)
        run1 = make_run([20.5], [100.5], forward_forcing_member_id=1)
        config = FootprintGridConfig(lat_grid=LAT_GRID, lon_grid=LON_GRID, bandwidth_m=3000.0)
        fps = build_predicted_slick_footprints({0: run0, 1: run1}, config)
        self.assertEqual(set(fps.keys()), {0, 1})
        self.assertEqual(fps[0].forward_forcing_member_id, 0)
        self.assertEqual(fps[1].forward_forcing_member_id, 1)


class TestDeterminism(unittest.TestCase):
    """Requirement 8: repeated inputs produce deterministic outputs (no RNG in the KDE
    itself)."""

    def test_repeated_calls_produce_identical_density(self):
        run = make_run([20.0, 20.02, 19.98], [100.0, 100.02, 99.98])
        config = FootprintGridConfig(lat_grid=LAT_GRID, lon_grid=LON_GRID, bandwidth_m=2500.0)
        fp1 = build_predicted_slick_footprint(run, config)
        fp2 = build_predicted_slick_footprint(run, config)
        np.testing.assert_array_equal(fp1.density, fp2.density)


class TestEmptyOrAllInvalidExplicit(unittest.TestCase):
    """Requirement 9: empty/all-invalid forward results are represented explicitly, not
    silently treated as valid slicks."""

    def test_all_invalid_run_gives_nan_density_and_flag(self):
        run = make_run([20.0, 20.1], [100.0, 100.1], valid=[False, False])
        config = FootprintGridConfig(lat_grid=LAT_GRID, lon_grid=LON_GRID, bandwidth_m=3000.0)
        fp = build_predicted_slick_footprint(run, config)
        self.assertFalse(fp.has_valid_particles)
        self.assertTrue(np.all(np.isnan(fp.density)))
        self.assertEqual(fp.n_valid_particles, 0)
        self.assertEqual(fp.n_particles, 2)

    def test_zero_particle_run_gives_nan_density_and_flag(self):
        run = make_run([], [])
        config = FootprintGridConfig(lat_grid=LAT_GRID, lon_grid=LON_GRID, bandwidth_m=3000.0)
        fp = build_predicted_slick_footprint(run, config)
        self.assertFalse(fp.has_valid_particles)
        self.assertTrue(np.all(np.isnan(fp.density)))

    def test_empty_observed_reference_is_explicit(self):
        adapted = make_adapted([20.0], [100.0], valid=[False])
        config = FootprintGridConfig(lat_grid=LAT_GRID, lon_grid=LON_GRID, bandwidth_m=3000.0)
        fp = build_observed_slick_footprint(adapted, T_OBS, config)
        self.assertFalse(fp.has_valid_particles)
        self.assertTrue(np.all(np.isnan(fp.density)))


class TestPredictedVsObservedControlledExample(unittest.TestCase):
    """Requirement 10 (automated proxy for visual inspection): a controlled
    predicted-vs-observed example with known, DIFFERENT peak locations."""

    def test_predicted_and_observed_peaks_at_their_own_distinct_locations(self):
        run = make_run([19.5] * 6, [99.5] * 6)  # predicted cluster near (19.5, 99.5)
        adapted = make_adapted([20.5] * 6, [100.5] * 6)  # observed cluster near (20.5, 100.5)
        config = FootprintGridConfig(lat_grid=LAT_GRID, lon_grid=LON_GRID, bandwidth_m=2000.0)

        predicted = build_predicted_slick_footprint(run, config)
        observed = build_observed_slick_footprint(adapted, T_OBS, config)

        pred_peak = np.unravel_index(np.nanargmax(predicted.density), predicted.density.shape)
        obs_peak = np.unravel_index(np.nanargmax(observed.density), observed.density.shape)
        self.assertNotEqual(pred_peak, obs_peak)
        self.assertAlmostEqual(LAT_GRID[pred_peak[0]], 19.5, delta=0.15)
        self.assertAlmostEqual(LAT_GRID[obs_peak[0]], 20.5, delta=0.15)


class TestFootprintGridConfigValidation(unittest.TestCase):
    def test_rejects_non_positive_bandwidth(self):
        with self.assertRaises(ValueError):
            FootprintGridConfig(lat_grid=LAT_GRID, lon_grid=LON_GRID, bandwidth_m=0.0)
        with self.assertRaises(ValueError):
            FootprintGridConfig(lat_grid=LAT_GRID, lon_grid=LON_GRID, bandwidth_m=-100.0)

    def test_rejects_empty_grid(self):
        with self.assertRaises(ValueError):
            FootprintGridConfig(lat_grid=np.array([]), lon_grid=LON_GRID, bandwidth_m=1000.0)


if __name__ == "__main__":
    unittest.main()
