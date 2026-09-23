"""Focused T23 checks for `iste.forward_accuracy` (forward-vs-observed accuracy per
candidate/forcing member).

Run with:
    python -m unittest tests.test_forward_accuracy -v
"""

from __future__ import annotations

import unittest

import numpy as np

from iste.forward_accuracy import (
    ComparisonConfig,
    ForwardAccuracyResult,
    compare_predicted_footprints_to_observed,
    compare_predicted_to_observed,
    footprint_overlap_iou,
    predicted_false_positive_fraction,
    observed_region_coverage,
    spatial_distance_m,
    _footprint_region_mask,
    _grid_cell_areas_m2,
)
from iste.forward_propagation import ForwardCandidateMemberRun
from iste.observation import adapt_upstream_particles
from iste.slick_footprint import (
    FootprintGridConfig,
    ObservedSlickFootprint,
    PredictedSlickFootprint,
    build_observed_slick_footprint,
    build_predicted_slick_footprint,
)

T_S = np.datetime64("2020-01-01T00:00:00", "ns")
T_OBS = np.datetime64("2020-01-01T06:00:00", "ns")
LAT_GRID = np.linspace(19.0, 21.0, 21)  # 0.1 deg spacing
LON_GRID = np.linspace(99.0, 101.0, 21)


def make_predicted(density, candidate_id="cand-A", member_id=0, n_valid=10, has_valid=True, lat_grid=LAT_GRID, lon_grid=LON_GRID):
    return PredictedSlickFootprint(
        candidate_id=candidate_id, t_s=T_S, t_obs=T_OBS, forward_forcing_member_id=member_id,
        lat_grid=lat_grid, lon_grid=lon_grid, density=density, bandwidth_m=3000.0,
        n_particles=n_valid, n_valid_particles=(n_valid if has_valid else 0), has_valid_particles=has_valid,
    )


def make_observed(density, n_valid=10, has_valid=True, lat_grid=LAT_GRID, lon_grid=LON_GRID):
    return ObservedSlickFootprint(
        t_obs=T_OBS, lat_grid=lat_grid, lon_grid=lon_grid, density=density, bandwidth_m=3000.0,
        n_particles=n_valid, n_valid_particles=(n_valid if has_valid else 0), has_valid_particles=has_valid,
    )


def gaussian_blob(lat_grid, lon_grid, lat0, lon0, sigma_deg=0.15):
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
    return np.exp(-((lat_mesh - lat0) ** 2 + (lon_mesh - lon0) ** 2) / (2.0 * sigma_deg**2))


class TestIdenticalFootprintsBestPossible(unittest.TestCase):
    """Requirement 1: identical predicted/observed footprints give the best possible
    result for every metric family."""

    def test_all_metrics_at_their_best_for_identical_footprints(self):
        density = gaussian_blob(LAT_GRID, LON_GRID, 20.0, 100.0)
        predicted = make_predicted(density.copy())
        observed = make_observed(density.copy())

        for metric, expected_error, expected_consistency in [
            ("footprint_overlap", 0.0, 1.0),
            ("observed_coverage", 0.0, 1.0),
            ("false_positive_area", 0.0, 1.0),
            ("spatial_distance", 0.0, 1.0),
            ("combined_overlap_fp", 0.0, 1.0),
        ]:
            with self.subTest(metric=metric):
                config = ComparisonConfig(metric=metric)
                result = compare_predicted_to_observed(predicted, observed, config)
                self.assertTrue(result.valid)
                self.assertAlmostEqual(result.error, expected_error, places=6)
                self.assertAlmostEqual(result.consistency, expected_consistency, places=6)


class TestDisplacedFootprintsWorse(unittest.TestCase):
    """Requirement 2: clearly displaced footprints -> worse error / lower consistency
    than the identical case, for every metric."""

    def test_displaced_is_worse_than_identical_for_every_metric(self):
        density_ref = gaussian_blob(LAT_GRID, LON_GRID, 20.0, 100.0)
        density_displaced = gaussian_blob(LAT_GRID, LON_GRID, 20.8, 100.8)  # far away, no overlap

        for metric in ("footprint_overlap", "observed_coverage", "false_positive_area", "spatial_distance", "combined_overlap_fp"):
            with self.subTest(metric=metric):
                config = ComparisonConfig(metric=metric)
                identical_result = compare_predicted_to_observed(
                    make_predicted(density_ref.copy()), make_observed(density_ref.copy()), config
                )
                displaced_result = compare_predicted_to_observed(
                    make_predicted(density_displaced.copy()), make_observed(density_ref.copy()), config
                )
                self.assertTrue(identical_result.valid)
                self.assertTrue(displaced_result.valid)
                self.assertGreater(displaced_result.error, identical_result.error)
                self.assertLess(displaced_result.consistency, identical_result.consistency)


class TestPartialOverlap(unittest.TestCase):
    """Requirement 3: partial overlap behaves correctly -- strictly between the
    identical and fully-displaced extremes."""

    def test_partial_overlap_is_between_identical_and_displaced(self):
        density_ref = gaussian_blob(LAT_GRID, LON_GRID, 20.0, 100.0)
        density_partial = gaussian_blob(LAT_GRID, LON_GRID, 20.15, 100.15)  # nearby, partial overlap
        density_far = gaussian_blob(LAT_GRID, LON_GRID, 20.8, 100.8)

        for metric in ("footprint_overlap", "observed_coverage", "false_positive_area", "combined_overlap_fp"):
            with self.subTest(metric=metric):
                config = ComparisonConfig(metric=metric)
                r_identical = compare_predicted_to_observed(make_predicted(density_ref.copy()), make_observed(density_ref.copy()), config)
                r_partial = compare_predicted_to_observed(make_predicted(density_partial.copy()), make_observed(density_ref.copy()), config)
                r_far = compare_predicted_to_observed(make_predicted(density_far.copy()), make_observed(density_ref.copy()), config)
                self.assertLess(r_identical.error, r_partial.error)
                self.assertLess(r_partial.error, r_far.error)
                self.assertGreater(r_identical.consistency, r_partial.consistency)
                self.assertGreater(r_partial.consistency, r_far.consistency)


class TestParticleCountIndependence(unittest.TestCase):
    """Requirement 4: different particle counts do not change the result solely because
    of count -- uses REAL T22 footprint construction (exact duplication of the same
    spatial distribution) to prove this end-to-end, not just at the footprint level."""

    def test_duplicated_particle_set_gives_identical_comparison(self):
        rng = np.random.default_rng(0)
        n = 10
        lat = 20.0 + rng.normal(0.0, 0.02, n)
        lon = 100.0 + rng.normal(0.0, 0.02, n)

        def make_run(lat_arr, lon_arr):
            m = len(lat_arr)
            return ForwardCandidateMemberRun(
                candidate_id="cand-A", t_s=T_S, t_obs=T_OBS, forward_forcing_member_id=0,
                dt_seconds=900.0, n_steps=0, timestamps=np.array([T_OBS], dtype="datetime64[ns]"),
                particle_id=np.array([f"p{i}" for i in range(m)], dtype=object),
                observation_pixel_id=np.array([f"px-{i}" for i in range(m)], dtype=object),
                forcing_member_id=np.zeros(m, dtype=np.int64),
                original_log_weight=np.zeros(m, dtype=np.float64),
                lat=np.asarray(lat_arr).reshape(1, m), lon=np.asarray(lon_arr).reshape(1, m),
                valid=np.ones((1, m), dtype=bool),
            )

        obs_records = [
            {"particle_id": f"o{i}", "latitude": 20.0 + rng.normal(0, 0.02), "longitude": 100.0 + rng.normal(0, 0.02),
             "observation_pixel_id": f"opx-{i}", "timestamp": T_OBS}
            for i in range(15)
        ]
        adapted = adapt_upstream_particles(obs_records)
        grid_config = FootprintGridConfig(lat_grid=LAT_GRID, lon_grid=LON_GRID, bandwidth_m=3000.0)
        observed = build_observed_slick_footprint(adapted, T_OBS, grid_config)

        run_single = make_run(lat, lon)
        run_doubled = make_run(np.concatenate([lat, lat]), np.concatenate([lon, lon]))
        predicted_single = build_predicted_slick_footprint(run_single, grid_config)
        predicted_doubled = build_predicted_slick_footprint(run_doubled, grid_config)

        for metric in ("footprint_overlap", "observed_coverage", "false_positive_area", "spatial_distance", "combined_overlap_fp"):
            with self.subTest(metric=metric):
                config = ComparisonConfig(metric=metric)
                r_single = compare_predicted_to_observed(predicted_single, observed, config)
                r_doubled = compare_predicted_to_observed(predicted_doubled, observed, config)
                # spatial_distance sums many more floating-point terms once particles are
                # doubled (different summation order) -- use a relative tolerance there;
                # the [0,1]-bounded area metrics stay exact to high absolute precision.
                delta = abs(r_single.error) * 1e-6 + 1e-6 if metric == "spatial_distance" else 1e-9
                self.assertAlmostEqual(r_single.error, r_doubled.error, delta=delta)
                self.assertAlmostEqual(r_single.consistency, r_doubled.consistency, delta=delta)


class TestEmptyOrInvalidPredictedFootprint(unittest.TestCase):
    """Requirement 5: empty/all-invalid predicted footprint is explicitly invalid."""

    def test_predicted_without_valid_particles_is_explicitly_invalid(self):
        density_obs = gaussian_blob(LAT_GRID, LON_GRID, 20.0, 100.0)
        predicted = make_predicted(np.full(density_obs.shape, np.nan), has_valid=False)
        observed = make_observed(density_obs)
        for metric in ("footprint_overlap", "observed_coverage", "false_positive_area", "spatial_distance", "combined_overlap_fp"):
            with self.subTest(metric=metric):
                result = compare_predicted_to_observed(predicted, observed, ComparisonConfig(metric=metric))
                self.assertFalse(result.valid)
                self.assertTrue(np.isnan(result.error))
                self.assertTrue(np.isnan(result.consistency))
                self.assertIn("predicted", result.invalid_reason)

    def test_observed_without_valid_particles_is_explicitly_invalid(self):
        density_pred = gaussian_blob(LAT_GRID, LON_GRID, 20.0, 100.0)
        predicted = make_predicted(density_pred)
        observed = make_observed(np.full(density_pred.shape, np.nan), has_valid=False)
        result = compare_predicted_to_observed(predicted, observed, ComparisonConfig(metric="footprint_overlap"))
        self.assertFalse(result.valid)
        self.assertTrue(np.isnan(result.error))
        self.assertIn("observed", result.invalid_reason)

    def test_invalid_result_never_silently_a_good_match(self):
        # a naive bug might return error=0/consistency=1 (a "perfect match") for an
        # undefined comparison -- explicitly assert that never happens.
        predicted = make_predicted(np.full((21, 21), np.nan), has_valid=False)
        observed = make_observed(gaussian_blob(LAT_GRID, LON_GRID, 20.0, 100.0))
        result = compare_predicted_to_observed(predicted, observed, ComparisonConfig(metric="footprint_overlap"))
        self.assertFalse(result.valid)
        self.assertFalse(result.consistency == 1.0)
        self.assertFalse(result.error == 0.0)


class TestMetricDirectionUnambiguous(unittest.TestCase):
    """Requirement 6: metric direction is unambiguous (error and consistency always
    move in opposite directions; error = 1 - consistency for the [0,1]-bounded metrics)."""

    def test_error_and_consistency_are_complementary(self):
        density_ref = gaussian_blob(LAT_GRID, LON_GRID, 20.0, 100.0)
        density_other = gaussian_blob(LAT_GRID, LON_GRID, 20.3, 100.2)
        for metric in ("footprint_overlap", "observed_coverage", "false_positive_area", "combined_overlap_fp"):
            with self.subTest(metric=metric):
                result = compare_predicted_to_observed(
                    make_predicted(density_other), make_observed(density_ref), ComparisonConfig(metric=metric)
                )
                self.assertAlmostEqual(result.error, 1.0 - result.consistency, places=9)

    def test_spatial_distance_error_is_raw_metres_consistency_is_decay(self):
        density_ref = gaussian_blob(LAT_GRID, LON_GRID, 20.0, 100.0)
        density_other = gaussian_blob(LAT_GRID, LON_GRID, 20.2, 100.0)
        config = ComparisonConfig(metric="spatial_distance", consistency_distance_scale_m=20_000.0)
        result = compare_predicted_to_observed(make_predicted(density_other), make_observed(density_ref), config)
        self.assertGreater(result.error, 0.0)  # metres, > 0 for a real displacement
        expected_consistency = np.exp(-result.error / 20_000.0)
        self.assertAlmostEqual(result.consistency, expected_consistency, places=9)


class TestConfigurationChangesResultDeterministically(unittest.TestCase):
    """Requirement 7: configuration changes the selected metric/parameters
    deterministically."""

    def test_different_metrics_give_different_results(self):
        density_pred = gaussian_blob(LAT_GRID, LON_GRID, 20.1, 100.1)
        density_obs = gaussian_blob(LAT_GRID, LON_GRID, 20.0, 100.0)
        predicted, observed = make_predicted(density_pred), make_observed(density_obs)
        results = {
            metric: compare_predicted_to_observed(predicted, observed, ComparisonConfig(metric=metric))
            for metric in ("footprint_overlap", "observed_coverage", "false_positive_area", "spatial_distance")
        }
        values = {(r.error, r.consistency) for r in results.values()}
        self.assertGreater(len(values), 1)  # not all metrics coincidentally agree

    def test_different_threshold_fraction_changes_area_metrics(self):
        density_pred = gaussian_blob(LAT_GRID, LON_GRID, 20.1, 100.1)
        density_obs = gaussian_blob(LAT_GRID, LON_GRID, 20.0, 100.0)
        predicted, observed = make_predicted(density_pred), make_observed(density_obs)
        r_loose = compare_predicted_to_observed(predicted, observed, ComparisonConfig(metric="footprint_overlap", region_threshold_fraction=0.05))
        r_strict = compare_predicted_to_observed(predicted, observed, ComparisonConfig(metric="footprint_overlap", region_threshold_fraction=0.6))
        self.assertNotAlmostEqual(r_loose.consistency, r_strict.consistency, places=6)

    def test_different_combination_weights_change_combined_metric(self):
        density_pred = gaussian_blob(LAT_GRID, LON_GRID, 20.1, 100.1)
        density_obs = gaussian_blob(LAT_GRID, LON_GRID, 20.0, 100.0)
        predicted, observed = make_predicted(density_pred), make_observed(density_obs)
        r_overlap_heavy = compare_predicted_to_observed(
            predicted, observed, ComparisonConfig(metric="combined_overlap_fp", combination_weight_overlap=1.0, combination_weight_fp=0.0)
        )
        r_fp_heavy = compare_predicted_to_observed(
            predicted, observed, ComparisonConfig(metric="combined_overlap_fp", combination_weight_overlap=0.0, combination_weight_fp=1.0)
        )
        self.assertNotAlmostEqual(r_overlap_heavy.consistency, r_fp_heavy.consistency, places=6)


class TestProvenancePreserved(unittest.TestCase):
    """Requirement 8: candidate/member provenance preserved."""

    def test_result_carries_candidate_time_and_member_identity(self):
        density = gaussian_blob(LAT_GRID, LON_GRID, 20.0, 100.0)
        predicted = make_predicted(density, candidate_id="cand-XYZ", member_id=4)
        observed = make_observed(density)
        result = compare_predicted_to_observed(predicted, observed, ComparisonConfig(metric="footprint_overlap"))
        self.assertEqual(result.candidate_id, "cand-XYZ")
        self.assertEqual(result.forward_forcing_member_id, 4)
        self.assertEqual(result.t_s, T_S)
        self.assertEqual(result.t_obs, T_OBS)
        self.assertEqual(result.metric, "footprint_overlap")

    def test_batch_helper_keys_by_member(self):
        density = gaussian_blob(LAT_GRID, LON_GRID, 20.0, 100.0)
        predicted_0 = make_predicted(density, member_id=0)
        predicted_1 = make_predicted(density, member_id=1)
        observed = make_observed(density)
        results = compare_predicted_footprints_to_observed({0: predicted_0, 1: predicted_1}, observed, ComparisonConfig(metric="footprint_overlap"))
        self.assertEqual(set(results.keys()), {0, 1})
        self.assertEqual(results[0].forward_forcing_member_id, 0)
        self.assertEqual(results[1].forward_forcing_member_id, 1)


class TestDeterminism(unittest.TestCase):
    """Requirement 9: repeated inputs produce identical results."""

    def test_repeated_calls_identical(self):
        density_pred = gaussian_blob(LAT_GRID, LON_GRID, 20.1, 100.1)
        density_obs = gaussian_blob(LAT_GRID, LON_GRID, 20.0, 100.0)
        predicted, observed = make_predicted(density_pred), make_observed(density_obs)
        config = ComparisonConfig(metric="combined_overlap_fp")
        r1 = compare_predicted_to_observed(predicted, observed, config)
        r2 = compare_predicted_to_observed(predicted, observed, config)
        self.assertEqual(r1.error, r2.error)
        self.assertEqual(r1.consistency, r2.consistency)


class TestCompositeMetricComponentsSeparatelyTestable(unittest.TestCase):
    """Requirement 10: composite metric components are separately testable/documented."""

    def test_footprint_overlap_iou_standalone(self):
        cell_area = np.ones((3, 3))
        pred_mask = np.array([[True, True, False], [False, False, False], [False, False, False]])
        obs_mask = np.array([[True, False, False], [False, False, False], [False, False, False]])
        # intersection = 1 cell, union = 2 cells -> IoU = 0.5
        self.assertAlmostEqual(footprint_overlap_iou(pred_mask, obs_mask, cell_area), 0.5)

    def test_predicted_false_positive_fraction_standalone(self):
        cell_area = np.ones((3, 3))
        pred_mask = np.array([[True, True, False], [False, False, False], [False, False, False]])
        obs_mask = np.array([[True, False, False], [False, False, False], [False, False, False]])
        # predicted area = 2 cells, false-positive (predicted but not observed) = 1 cell -> 0.5
        self.assertAlmostEqual(predicted_false_positive_fraction(pred_mask, obs_mask, cell_area), 0.5)

    def test_observed_region_coverage_standalone(self):
        cell_area = np.ones((3, 3))
        pred_mask = np.array([[True, False, False], [False, False, False], [False, False, False]])
        obs_mask = np.array([[True, True, False], [False, False, False], [False, False, False]])
        # observed area = 2 cells, covered = 1 cell -> 0.5
        self.assertAlmostEqual(observed_region_coverage(pred_mask, obs_mask, cell_area), 0.5)

    def test_combined_metric_matches_manual_combination_of_components(self):
        density_pred = gaussian_blob(LAT_GRID, LON_GRID, 20.1, 100.1)
        density_obs = gaussian_blob(LAT_GRID, LON_GRID, 20.0, 100.0)
        predicted, observed = make_predicted(density_pred), make_observed(density_obs)
        config = ComparisonConfig(metric="combined_overlap_fp", combination_weight_overlap=0.7, combination_weight_fp=0.3)

        cell_area = _grid_cell_areas_m2(LAT_GRID, LON_GRID)
        pred_mask = _footprint_region_mask(density_pred, config.region_threshold_fraction)
        obs_mask = _footprint_region_mask(density_obs, config.region_threshold_fraction)
        iou = footprint_overlap_iou(pred_mask, obs_mask, cell_area)
        fp = predicted_false_positive_fraction(pred_mask, obs_mask, cell_area)
        expected_consistency = 0.7 * iou + 0.3 * (1.0 - fp)

        result = compare_predicted_to_observed(predicted, observed, config)
        self.assertAlmostEqual(result.consistency, expected_consistency, places=9)
        self.assertAlmostEqual(result.error, 1.0 - expected_consistency, places=9)


class TestMismatchedGridsRejected(unittest.TestCase):
    def test_different_grids_raise(self):
        density = gaussian_blob(LAT_GRID, LON_GRID, 20.0, 100.0)
        predicted = make_predicted(density)
        other_lat_grid = LAT_GRID + 1.0
        observed = make_observed(gaussian_blob(other_lat_grid, LON_GRID, 20.0, 100.0), lat_grid=other_lat_grid)
        with self.assertRaises(ValueError):
            compare_predicted_to_observed(predicted, observed, ComparisonConfig(metric="footprint_overlap"))


class TestComparisonConfigValidation(unittest.TestCase):
    def test_rejects_unknown_metric(self):
        with self.assertRaises(ValueError):
            ComparisonConfig(metric="not_a_real_metric")

    def test_rejects_invalid_threshold_fraction(self):
        with self.assertRaises(ValueError):
            ComparisonConfig(metric="footprint_overlap", region_threshold_fraction=0.0)
        with self.assertRaises(ValueError):
            ComparisonConfig(metric="footprint_overlap", region_threshold_fraction=1.5)

    def test_rejects_non_positive_distance_scale(self):
        with self.assertRaises(ValueError):
            ComparisonConfig(metric="spatial_distance", consistency_distance_scale_m=0.0)


if __name__ == "__main__":
    unittest.main()
