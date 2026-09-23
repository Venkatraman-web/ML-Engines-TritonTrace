"""Focused T18 checks for `iste.forward_budget` (Stage V forward particle budget,
`N_forward = min(X, Y)`).

Run with:
    python -m unittest tests.test_forward_budget -v
"""

from __future__ import annotations

import unittest

import numpy as np

from iste.candidate_extraction import (
    CandidateExtractionConfig,
    CoreRegion,
    find_candidate_peaks_from_index,
)
from iste.candidate_extraction import Candidate
from iste.forward_budget import (
    ForwardBudget,
    ForwardBudgetConfig,
    compute_forward_particle_budget,
    compute_forward_particle_budgets,
)
from iste.geometry import metres_per_degree_latitude
from iste.kdtree_neighborhood import build_particle_neighborhood_index
from iste.particle_query import HistoricalParticleSlice
from iste.support_features import SupportFeatureConfig, compute_support_features

NAT = np.datetime64("NaT", "ns")
T_S = np.datetime64("2020-01-01T06:00:00", "ns")
LAT0, LON0 = 10.0, 50.0
M_PER_DEG_LAT = metres_per_degree_latitude()


def _make_slice(lat, lon, valid=None, t_s=T_S):
    n = len(lat)
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    return HistoricalParticleSlice(
        particle_id=np.array([f"p{i}" for i in range(n)], dtype=object),
        observation_pixel_id=np.array([f"px-{i % 3}" for i in range(n)], dtype=object),
        forcing_member_id=np.zeros(n, dtype=np.int64),
        lat=lat,
        lon=lon,
        log_weight=np.zeros(n, dtype=np.float64),
        valid=np.array(valid if valid is not None else [True] * n, dtype=bool),
        death_time=np.full(n, NAT, dtype="datetime64[ns]"),
        death_lat=np.full(n, np.nan),
        death_lon=np.full(n, np.nan),
        death_reason=np.array([""] * n, dtype=object),
        query_time=t_s,
    )


def _north_offset_points(distances_m, lat0=LAT0, lon0=LON0):
    """Particles due north of (lat0, lon0), each exactly `d` metres away (along the
    latitude axis only, so true distance == the offset used to place it -- the same
    equirectangular projection `iste.kdtree_neighborhood` itself uses)."""
    lat = lat0 + np.asarray(distances_m, dtype=np.float64) / M_PER_DEG_LAT
    lon = np.full(len(distances_m), lon0, dtype=np.float64)
    return lat, lon


def _dummy_support_features(index):
    """A real (but otherwise irrelevant to these tests) `SupportFeatures`, needed only
    to satisfy `Candidate.peak_support_features`'s required field -- these tests build
    `Candidate`s directly rather than through `find_candidate_peaks` so each one's
    `core_region` can be placed at an exact, hand-computed distance from known particles."""
    return compute_support_features(index, LAT0, LON0, radius_m=1.0, config=SupportFeatureConfig())


def _make_candidate(candidate_id, core, index, t_s=T_S):
    sf = _dummy_support_features(index)
    return Candidate(
        candidate_id=candidate_id,
        x_peak=core.center_lon,
        y_peak=core.center_lat,
        t_s=t_s,
        core_region=core,
        support_score=sf.support,
        density_score=sf.density,
        weighted_support_score=sf.weighted_support,
        pixel_diversity_score=sf.pixel_diversity,
        peak_support_features=sf,
    )


class TestLockedRuleConsequences(unittest.TestCase):
    """Requirements 1-3: the three N_forward = min(X, Y) branches."""

    def _budget_for(self, n_inside, n_outside, X):
        inside = [50.0] * n_inside  # all well within a 1000 m core
        outside = [5000.0] * n_outside  # all well outside it
        lat, lon = _north_offset_points(inside + outside)
        slice_ = _make_slice(lat, lon)
        index = build_particle_neighborhood_index(slice_, lat_ref=LAT0, lon_ref=LON0)
        core = CoreRegion(center_lat=LAT0, center_lon=LON0, radius_m=1000.0)
        candidate = _make_candidate("cand-a", core, index)
        return compute_forward_particle_budget(candidate, index, ForwardBudgetConfig(X=X))

    def test_y_greater_than_x_selects_x(self):
        fb = self._budget_for(n_inside=10, n_outside=3, X=4)
        self.assertEqual(fb.Y, 10)
        self.assertEqual(fb.X, 4)
        self.assertEqual(fb.N_forward, 4)
        self.assertTrue(fb.forward_validatable)

    def test_y_equal_to_x_uses_x(self):
        fb = self._budget_for(n_inside=5, n_outside=2, X=5)
        self.assertEqual(fb.Y, 5)
        self.assertEqual(fb.N_forward, 5)
        self.assertTrue(fb.forward_validatable)

    def test_y_less_than_x_uses_all_y_and_is_not_rejected(self):
        fb = self._budget_for(n_inside=3, n_outside=1, X=10)
        self.assertEqual(fb.Y, 3)
        self.assertEqual(fb.N_forward, 3)
        self.assertTrue(fb.forward_validatable)  # never rejected for Y < X
        self.assertIsInstance(fb, ForwardBudget)


class TestZeroYIsUnvalidated(unittest.TestCase):
    """Requirement 4: Y = 0 is marked explicitly unvalidated, not dropped/raised."""

    def test_no_particles_in_core_marks_unvalidated(self):
        lat, lon = _north_offset_points([5000.0, 8000.0, 12000.0])
        slice_ = _make_slice(lat, lon)
        index = build_particle_neighborhood_index(slice_, lat_ref=LAT0, lon_ref=LON0)
        core = CoreRegion(center_lat=LAT0, center_lon=LON0, radius_m=1000.0)
        candidate = _make_candidate("cand-empty", core, index)

        fb = compute_forward_particle_budget(candidate, index, ForwardBudgetConfig(X=7))

        self.assertEqual(fb.Y, 0)
        self.assertEqual(fb.N_forward, 0)
        self.assertFalse(fb.forward_validatable)
        # still a full record -- not silently omitted -- with candidate_id/t_s preserved
        self.assertEqual(fb.candidate_id, "cand-empty")
        self.assertEqual(fb.t_s, T_S)

    def test_empty_slice_also_marks_unvalidated(self):
        slice_ = _make_slice(np.array([]), np.array([]))
        index = build_particle_neighborhood_index(slice_, lat_ref=LAT0, lon_ref=LON0)
        core = CoreRegion(center_lat=LAT0, center_lon=LON0, radius_m=1000.0)
        candidate = _make_candidate("cand-noparticles", core, index)

        fb = compute_forward_particle_budget(candidate, index, ForwardBudgetConfig(X=3))
        self.assertEqual(fb.Y, 0)
        self.assertEqual(fb.N_forward, 0)
        self.assertFalse(fb.forward_validatable)


class TestCandidatePreservation(unittest.TestCase):
    """candidate_id, t_s, X, Y, N_forward must all be preserved/reported."""

    def test_all_fields_preserved_exactly(self):
        lat, lon = _north_offset_points([10.0, 20.0, 30.0, 40.0])
        slice_ = _make_slice(lat, lon)
        index = build_particle_neighborhood_index(slice_, lat_ref=LAT0, lon_ref=LON0)
        core = CoreRegion(center_lat=LAT0, center_lon=LON0, radius_m=1000.0)
        candidate = _make_candidate("2020-01-01T06:00:00_cand007", core, index)

        fb = compute_forward_particle_budget(candidate, index, ForwardBudgetConfig(X=2))

        self.assertEqual(fb.candidate_id, "2020-01-01T06:00:00_cand007")
        self.assertEqual(fb.t_s, T_S)
        self.assertEqual(fb.X, 2)
        self.assertEqual(fb.Y, 4)
        self.assertEqual(fb.N_forward, 2)


class TestMultipleCandidatesAndTimestampsIndependent(unittest.TestCase):
    """Requirement 5: multiple candidates/timestamps remain independent."""

    def test_two_candidates_same_timestamp_independent(self):
        # Two well-separated cores in one slice, each with its own particle count.
        lat_a, lon_a = _north_offset_points([50.0] * 6, lat0=10.0, lon0=50.0)
        lat_b, lon_b = _north_offset_points([50.0] * 2, lat0=20.0, lon0=60.0)
        lat = np.concatenate([lat_a, lat_b])
        lon = np.concatenate([lon_a, lon_b])
        slice_ = _make_slice(lat, lon)
        index = build_particle_neighborhood_index(slice_)

        core_a = CoreRegion(center_lat=10.0, center_lon=50.0, radius_m=1000.0)
        core_b = CoreRegion(center_lat=20.0, center_lon=60.0, radius_m=1000.0)
        cand_a = _make_candidate("cand-a", core_a, index)
        cand_b = _make_candidate("cand-b", core_b, index)

        budgets_forward = compute_forward_particle_budgets([cand_a, cand_b], index, ForwardBudgetConfig(X=4))
        budgets_reversed = compute_forward_particle_budgets([cand_b, cand_a], index, ForwardBudgetConfig(X=4))

        by_id_forward = {b.candidate_id: b for b in budgets_forward}
        by_id_reversed = {b.candidate_id: b for b in budgets_reversed}

        self.assertEqual(by_id_forward["cand-a"].Y, 6)
        self.assertEqual(by_id_forward["cand-a"].N_forward, 4)
        self.assertEqual(by_id_forward["cand-b"].Y, 2)
        self.assertEqual(by_id_forward["cand-b"].N_forward, 2)
        # processing order does not change either candidate's own result
        self.assertEqual(by_id_forward["cand-a"], by_id_reversed["cand-a"])
        self.assertEqual(by_id_forward["cand-b"], by_id_reversed["cand-b"])

    def test_two_timestamps_do_not_cross_contaminate(self):
        t_s_1 = np.datetime64("2020-01-01T06:00:00", "ns")
        t_s_2 = np.datetime64("2020-01-01T12:00:00", "ns")

        lat1, lon1 = _north_offset_points([50.0] * 9)
        slice_1 = _make_slice(lat1, lon1, t_s=t_s_1)
        index_1 = build_particle_neighborhood_index(slice_1, lat_ref=LAT0, lon_ref=LON0)

        lat2, lon2 = _north_offset_points([50.0] * 2)
        slice_2 = _make_slice(lat2, lon2, t_s=t_s_2)
        index_2 = build_particle_neighborhood_index(slice_2, lat_ref=LAT0, lon_ref=LON0)

        core = CoreRegion(center_lat=LAT0, center_lon=LON0, radius_m=1000.0)
        cand_1 = _make_candidate("cand-t1", core, index_1, t_s=t_s_1)
        cand_2 = _make_candidate("cand-t2", core, index_2, t_s=t_s_2)

        fb1 = compute_forward_particle_budget(cand_1, index_1, ForwardBudgetConfig(X=5))
        fb2 = compute_forward_particle_budget(cand_2, index_2, ForwardBudgetConfig(X=5))

        self.assertEqual(fb1.Y, 9)
        self.assertEqual(fb1.N_forward, 5)
        self.assertEqual(fb1.t_s, t_s_1)
        self.assertEqual(fb2.Y, 2)
        self.assertEqual(fb2.N_forward, 2)
        self.assertEqual(fb2.t_s, t_s_2)


class TestInvalidParticlesExcludedFromY(unittest.TestCase):
    """Requirement 6: invalid particles are not counted in Y."""

    def test_invalid_particles_inside_core_do_not_count(self):
        lat, lon = _north_offset_points([50.0, 60.0, 70.0, 80.0, 90.0])
        valid = [True, True, False, False, False]  # 3 of 5 in-core particles invalid
        slice_ = _make_slice(lat, lon, valid=valid)
        index = build_particle_neighborhood_index(slice_, lat_ref=LAT0, lon_ref=LON0)
        core = CoreRegion(center_lat=LAT0, center_lon=LON0, radius_m=1000.0)
        candidate = _make_candidate("cand-mixed", core, index)

        fb = compute_forward_particle_budget(candidate, index, ForwardBudgetConfig(X=10))

        self.assertEqual(fb.Y, 2)  # only the 2 valid particles
        self.assertEqual(fb.N_forward, 2)


class TestDeterminism(unittest.TestCase):
    """Requirement 7: same inputs produce deterministic results."""

    def test_repeated_calls_are_identical(self):
        lat, lon = _north_offset_points([10.0, 200.0, 400.0, 900.0, 1500.0])
        slice_ = _make_slice(lat, lon)
        index = build_particle_neighborhood_index(slice_, lat_ref=LAT0, lon_ref=LON0)
        core = CoreRegion(center_lat=LAT0, center_lon=LON0, radius_m=500.0)
        candidate = _make_candidate("cand-repeat", core, index)
        config = ForwardBudgetConfig(X=2)

        results = [compute_forward_particle_budget(candidate, index, config) for _ in range(5)]
        self.assertTrue(all(r == results[0] for r in results))

    def test_batch_helper_matches_individual_calls(self):
        lat, lon = _north_offset_points([10.0, 200.0, 400.0])
        slice_ = _make_slice(lat, lon)
        index = build_particle_neighborhood_index(slice_, lat_ref=LAT0, lon_ref=LON0)
        core = CoreRegion(center_lat=LAT0, center_lon=LON0, radius_m=500.0)
        candidate = _make_candidate("cand-batch", core, index)
        config = ForwardBudgetConfig(X=2)

        individual = compute_forward_particle_budget(candidate, index, config)
        (batched,) = compute_forward_particle_budgets([candidate], index, config)
        self.assertEqual(individual, batched)


class TestYUsesCoreRegionNotSupportMapDensity(unittest.TestCase):
    """Y must come from re-querying `core_region` (T17's own candidate-core radius), not
    from reusing T16's `density_score` (which may have been evaluated at a different
    radius/k when the support map itself was built)."""

    def test_y_differs_from_density_score_when_radii_differ(self):
        # 3 particles within 200 m, 5 more between 200 m and 800 m.
        lat, lon = _north_offset_points([50.0, 100.0, 150.0, 300.0, 400.0, 500.0, 600.0, 700.0])
        slice_ = _make_slice(lat, lon)
        index = build_particle_neighborhood_index(slice_, lat_ref=LAT0, lon_ref=LON0)

        # Support map's own D at a SMALL radius (200 m) -> only the first 3 particles.
        sf_small_radius = compute_support_features(index, LAT0, LON0, radius_m=200.0)
        self.assertEqual(sf_small_radius.density, 3)

        # Candidate's core_region uses a LARGER radius (800 m) -> all 8 particles.
        core = CoreRegion(center_lat=LAT0, center_lon=LON0, radius_m=800.0)
        candidate = Candidate(
            candidate_id="cand-radius-mismatch",
            x_peak=LON0,
            y_peak=LAT0,
            t_s=T_S,
            core_region=core,
            support_score=sf_small_radius.support,
            density_score=sf_small_radius.density,  # deliberately the SMALL-radius D
            weighted_support_score=sf_small_radius.weighted_support,
            pixel_diversity_score=sf_small_radius.pixel_diversity,
            peak_support_features=sf_small_radius,
        )

        fb = compute_forward_particle_budget(candidate, index, ForwardBudgetConfig(X=100))

        self.assertEqual(candidate.density_score, 3)  # T16's D, at its own radius
        self.assertEqual(fb.Y, 8)  # T18's Y, re-queried at core_region's own radius
        self.assertNotEqual(fb.Y, candidate.density_score)


class TestForwardBudgetConfigValidation(unittest.TestCase):
    def test_rejects_non_positive_x(self):
        with self.assertRaises(ValueError):
            ForwardBudgetConfig(X=0)
        with self.assertRaises(ValueError):
            ForwardBudgetConfig(X=-3)

    def test_rejects_non_integer_x(self):
        with self.assertRaises(TypeError):
            ForwardBudgetConfig(X=5.5)
        with self.assertRaises(TypeError):
            ForwardBudgetConfig(X="5")

    def test_accepts_positive_int(self):
        config = ForwardBudgetConfig(X=5)
        self.assertEqual(config.X, 5)


class TestEndToEndThroughRealPeakExtraction(unittest.TestCase):
    """Integration: real T14/T15/T16/T17 machinery feeding T18, over two synthetic
    particle clusters of different sizes."""

    def test_two_clusters_produce_two_independent_budgets(self):
        rng = np.random.default_rng(11)
        lat_a = 10.0 + rng.normal(0.0, 0.0005, 12)
        lon_a = 50.0 + rng.normal(0.0, 0.0005, 12)
        lat_b = 12.0 + rng.normal(0.0, 0.0005, 4)
        lon_b = 53.0 + rng.normal(0.0, 0.0005, 4)
        lat = np.concatenate([lat_a, lat_b])
        lon = np.concatenate([lon_a, lon_b])
        slice_ = _make_slice(lat, lon)
        index = build_particle_neighborhood_index(slice_)

        lat_grid = np.linspace(9.5, 12.5, 31)
        lon_grid = np.linspace(49.5, 53.5, 41)
        candidates = find_candidate_peaks_from_index(
            index, lat_grid, lon_grid, T_S,
            radius_m=300.0,
            extraction_config=CandidateExtractionConfig(
                peak_neighborhood_cells=3, min_separation_m=50_000.0, core_radius_m=300.0
            ),
        )
        self.assertEqual(len(candidates), 2)

        budgets = compute_forward_particle_budgets(candidates, index, ForwardBudgetConfig(X=6))
        self.assertEqual(len(budgets), 2)

        by_id = {b.candidate_id: b for b in budgets}
        ys = sorted(b.Y for b in budgets)
        # larger cluster's core sees (approximately) 12 particles, smaller sees 4 --
        # exact Y depends on the peak grid cell found, so assert the qualitative split
        # (one candidate's Y >= X, the other's Y < X) rather than exact counts.
        self.assertTrue(any(b.N_forward == 6 and b.Y >= 6 for b in budgets))
        self.assertTrue(any(b.Y < 6 and b.N_forward == b.Y and b.forward_validatable for b in budgets))
        self.assertEqual(set(by_id), {c.candidate_id for c in candidates})


if __name__ == "__main__":
    unittest.main()
