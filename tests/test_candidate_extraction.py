"""Focused T17 checks for `iste.candidate_extraction` (Phase 2 candidate peak/core
extraction).

Run with:
    python -m unittest tests.test_candidate_extraction -v
"""

from __future__ import annotations

import unittest

import numpy as np

from iste.candidate_extraction import (
    CandidateExtractionConfig,
    find_candidate_peaks,
    find_candidate_peaks_from_index,
)
from iste.kdtree_neighborhood import NeighborhoodResult, build_particle_neighborhood_index
from iste.particle_query import HistoricalParticleSlice
from iste.support_features import SupportFeatures, SupportFeatureConfig

QUERY_TIME = np.datetime64("2020-01-01T00:00:00", "ns")
NAT = np.datetime64("NaT", "ns")


def _make_neighborhood(n: int, pixel_id: str = "px-0") -> NeighborhoodResult:
    return NeighborhoodResult(
        particle_id=np.array([f"p{i}" for i in range(n)], dtype=object),
        observation_pixel_id=np.array([pixel_id] * n, dtype=object),
        forcing_member_id=np.zeros(n, dtype=np.int64),
        lat=np.zeros(n), lon=np.zeros(n), log_weight=np.zeros(n),
        valid=np.ones(n, dtype=bool),
        death_time=np.full(n, NAT, dtype="datetime64[ns]"),
        death_lat=np.full(n, np.nan), death_lon=np.full(n, np.nan),
        death_reason=np.array([""] * n, dtype=object),
        distance_m=np.zeros(n),
    )


def _sf(lat, lon, density, weighted_support, pixel_diversity) -> SupportFeatures:
    support = density + weighted_support + pixel_diversity  # default linear_support_function
    return SupportFeatures(
        query_lat=lat, query_lon=lon, density=density, weighted_support=weighted_support,
        pixel_diversity=pixel_diversity, support=support,
        neighborhood=_make_neighborhood(density), config=SupportFeatureConfig(),
    )


def _support_map_from_2d(values: np.ndarray, lat_grid, lon_grid) -> list[SupportFeatures]:
    """Build a hand-controlled support map directly from a 2-D (n_lat, n_lon) array of
    `(density, weighted_support, pixel_diversity)` tuples, in the row-major order
    `find_candidate_peaks` expects."""
    n_lat, n_lon = len(lat_grid), len(lon_grid)
    out = []
    for i in range(n_lat):
        for j in range(n_lon):
            d, w, v = values[i][j]
            out.append(_sf(lat_grid[i], lon_grid[j], d, w, v))
    return out


class TestMultimodalYieldsMultipleCandidates(unittest.TestCase):
    def test_two_well_separated_peaks_both_found(self):
        lat_grid = np.linspace(0.0, 10.0, 11)
        lon_grid = np.linspace(0.0, 10.0, 11)
        values = [[(0, 0.0, 0) for _ in lon_grid] for _ in lat_grid]
        values[2][2] = (10, 10.0, 2)  # peak A at (lat=2, lon=2)
        values[8][8] = (12, 8.0, 3)   # peak B at (lat=8, lon=8)
        support_map = _support_map_from_2d(values, lat_grid, lon_grid)

        config = CandidateExtractionConfig(min_separation_m=1000.0, core_radius_m=500.0)
        candidates = find_candidate_peaks(support_map, lat_grid, lon_grid, QUERY_TIME, config)

        self.assertEqual(len(candidates), 2)
        peaks = {(c.y_peak, c.x_peak) for c in candidates}
        self.assertEqual(peaks, {(2.0, 2.0), (8.0, 8.0)})

    def test_does_not_collapse_to_a_single_centroid(self):
        # if this collapsed to a centroid, we'd see one candidate near (5,5) -- must not.
        lat_grid = np.linspace(0.0, 10.0, 11)
        lon_grid = np.linspace(0.0, 10.0, 11)
        values = [[(0, 0.0, 0) for _ in lon_grid] for _ in lat_grid]
        values[1][1] = (5, 5.0, 1)
        values[9][9] = (5, 5.0, 1)
        support_map = _support_map_from_2d(values, lat_grid, lon_grid)
        config = CandidateExtractionConfig(min_separation_m=1000.0, core_radius_m=500.0)
        candidates = find_candidate_peaks(support_map, lat_grid, lon_grid, QUERY_TIME, config)
        centroid_like = [c for c in candidates if abs(c.y_peak - 5.0) < 0.5 and abs(c.x_peak - 5.0) < 0.5]
        self.assertEqual(len(centroid_like), 0)
        self.assertEqual(len(candidates), 2)


class TestNearDuplicateSuppression(unittest.TestCase):
    def _two_close_peaks_map(self):
        # Grid spacing 0.1 deg; peaks placed 5 CELLS apart (0.5 deg, ~55.6 km near
        # lat=1.0) -- far enough apart on the GRID that both are independent raw local
        # maxima under the default 3x3 maximum_filter neighborhood (i.e. NOT grid-
        # adjacent, which would already merge them before distance-based suppression
        # ever runs). This isolates the separation-THRESHOLD's own effect.
        lat_grid = np.linspace(0.0, 2.0, 21)  # 0.1 deg spacing
        lon_grid = np.linspace(0.0, 2.0, 21)
        values = [[(0, 0.0, 0) for _ in lon_grid] for _ in lat_grid]
        values[10][10] = (10, 10.0, 1)  # stronger peak at (1.0, 1.0)
        values[10][15] = (8, 8.0, 1)    # weaker peak ~55.6 km away at (1.0, 1.5)
        return _support_map_from_2d(values, lat_grid, lon_grid), lat_grid, lon_grid

    def test_small_separation_threshold_keeps_both(self):
        support_map, lat_grid, lon_grid = self._two_close_peaks_map()
        config = CandidateExtractionConfig(min_separation_m=10_000.0, core_radius_m=50.0)  # 10 km < 55.6 km
        candidates = find_candidate_peaks(support_map, lat_grid, lon_grid, QUERY_TIME, config)
        self.assertEqual(len(candidates), 2)

    def test_large_separation_threshold_suppresses_the_weaker_one(self):
        support_map, lat_grid, lon_grid = self._two_close_peaks_map()
        config = CandidateExtractionConfig(min_separation_m=100_000.0, core_radius_m=50.0)  # 100 km > 55.6 km
        candidates = find_candidate_peaks(support_map, lat_grid, lon_grid, QUERY_TIME, config)
        self.assertEqual(len(candidates), 1)
        # the SURVIVING candidate must be the higher-support one.
        self.assertAlmostEqual(candidates[0].support_score, 21.0)  # 10+10+1

    def test_threshold_is_configurable_and_changes_the_outcome(self):
        support_map, lat_grid, lon_grid = self._two_close_peaks_map()
        few = find_candidate_peaks(
            support_map, lat_grid, lon_grid, QUERY_TIME,
            CandidateExtractionConfig(min_separation_m=100_000.0, core_radius_m=50.0),
        )
        many = find_candidate_peaks(
            support_map, lat_grid, lon_grid, QUERY_TIME,
            CandidateExtractionConfig(min_separation_m=10.0, core_radius_m=50.0),
        )
        self.assertLess(len(few), len(many))


class TestDistinctSpatialModesRemainSeparate(unittest.TestCase):
    def test_three_modes_all_retained_when_far_apart(self):
        lat_grid = np.linspace(0.0, 30.0, 31)
        lon_grid = np.linspace(0.0, 30.0, 31)
        values = [[(0, 0.0, 0) for _ in lon_grid] for _ in lat_grid]
        values[5][5] = (5, 5.0, 1)
        values[15][15] = (7, 7.0, 2)
        values[25][25] = (9, 9.0, 1)
        support_map = _support_map_from_2d(values, lat_grid, lon_grid)
        config = CandidateExtractionConfig(min_separation_m=10_000.0, core_radius_m=1000.0)
        candidates = find_candidate_peaks(support_map, lat_grid, lon_grid, QUERY_TIME, config)
        self.assertEqual(len(candidates), 3)
        positions = {(round(c.y_peak), round(c.x_peak)) for c in candidates}
        self.assertEqual(positions, {(5, 5), (15, 15), (25, 25)})


class TestTimestampPreserved(unittest.TestCase):
    def test_t_s_attached_to_every_candidate(self):
        lat_grid = np.linspace(0.0, 2.0, 3)
        lon_grid = np.linspace(0.0, 2.0, 3)
        values = [[(0, 0.0, 0) for _ in lon_grid] for _ in lat_grid]
        values[1][1] = (3, 3.0, 1)
        support_map = _support_map_from_2d(values, lat_grid, lon_grid)
        t_s = np.datetime64("2020-06-15T08:30:00", "ns")
        candidates = find_candidate_peaks(support_map, lat_grid, lon_grid, t_s)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].t_s, t_s)

    def test_different_timestamps_give_different_candidate_ids(self):
        lat_grid = np.linspace(0.0, 2.0, 3)
        lon_grid = np.linspace(0.0, 2.0, 3)
        values = [[(0, 0.0, 0) for _ in lon_grid] for _ in lat_grid]
        values[1][1] = (3, 3.0, 1)
        support_map = _support_map_from_2d(values, lat_grid, lon_grid)
        c1 = find_candidate_peaks(support_map, lat_grid, lon_grid, "2020-01-01T00:00:00")
        c2 = find_candidate_peaks(support_map, lat_grid, lon_grid, "2020-01-01T01:00:00")
        self.assertNotEqual(c1[0].candidate_id, c2[0].candidate_id)


class TestCoreBoundariesDeterministicAndConfigurable(unittest.TestCase):
    def test_core_region_uses_configured_radius(self):
        lat_grid = np.linspace(0.0, 2.0, 3)
        lon_grid = np.linspace(0.0, 2.0, 3)
        values = [[(0, 0.0, 0) for _ in lon_grid] for _ in lat_grid]
        values[1][1] = (3, 3.0, 1)
        support_map = _support_map_from_2d(values, lat_grid, lon_grid)

        c_small = find_candidate_peaks(
            support_map, lat_grid, lon_grid, QUERY_TIME, CandidateExtractionConfig(core_radius_m=500.0)
        )
        c_large = find_candidate_peaks(
            support_map, lat_grid, lon_grid, QUERY_TIME, CandidateExtractionConfig(core_radius_m=5000.0)
        )
        self.assertEqual(c_small[0].core_region.radius_m, 500.0)
        self.assertEqual(c_large[0].core_region.radius_m, 5000.0)
        self.assertEqual(c_small[0].core_region.center_lat, c_large[0].core_region.center_lat)

    def test_core_region_centered_exactly_on_the_peak(self):
        lat_grid = np.linspace(0.0, 2.0, 3)
        lon_grid = np.linspace(0.0, 2.0, 3)
        values = [[(0, 0.0, 0) for _ in lon_grid] for _ in lat_grid]
        values[2][0] = (4, 4.0, 1)
        support_map = _support_map_from_2d(values, lat_grid, lon_grid)
        candidates = find_candidate_peaks(support_map, lat_grid, lon_grid, QUERY_TIME)
        self.assertEqual(candidates[0].core_region.center_lat, candidates[0].y_peak)
        self.assertEqual(candidates[0].core_region.center_lon, candidates[0].x_peak)

    def test_invalid_core_radius_rejected(self):
        with self.assertRaises(ValueError):
            CandidateExtractionConfig(core_radius_m=0.0)
        with self.assertRaises(ValueError):
            CandidateExtractionConfig(core_radius_m=-10.0)


class TestFeatureValuesMatchT16Data(unittest.TestCase):
    def test_scores_match_the_underlying_support_features_exactly(self):
        lat_grid = np.linspace(0.0, 2.0, 3)
        lon_grid = np.linspace(0.0, 2.0, 3)
        values = [[(0, 0.0, 0) for _ in lon_grid] for _ in lat_grid]
        values[1][1] = (7, 12.5, 3)
        support_map = _support_map_from_2d(values, lat_grid, lon_grid)
        candidates = find_candidate_peaks(support_map, lat_grid, lon_grid, QUERY_TIME)

        self.assertEqual(len(candidates), 1)
        c = candidates[0]
        self.assertEqual(c.density_score, 7.0)
        self.assertEqual(c.weighted_support_score, 12.5)
        self.assertEqual(c.pixel_diversity_score, 3.0)
        self.assertAlmostEqual(c.support_score, 7 + 12.5 + 3)
        # and the peak_support_features object itself is the SAME underlying SupportFeatures.
        self.assertEqual(c.peak_support_features.density, 7)
        self.assertEqual(c.peak_support_features.neighborhood.n_particles, 7)


class TestNoCandidateFromInvalidOrEmptyInput(unittest.TestCase):
    def test_empty_grid_gives_no_candidates(self):
        candidates = find_candidate_peaks([], np.array([]), np.array([]), QUERY_TIME)
        self.assertEqual(candidates, [])

    def test_all_zero_density_gives_no_candidates(self):
        lat_grid = np.linspace(0.0, 2.0, 3)
        lon_grid = np.linspace(0.0, 2.0, 3)
        values = [[(0, 0.0, 0) for _ in lon_grid] for _ in lat_grid]  # everything empty
        support_map = _support_map_from_2d(values, lat_grid, lon_grid)
        candidates = find_candidate_peaks(support_map, lat_grid, lon_grid, QUERY_TIME)
        self.assertEqual(candidates, [])

    def test_mismatched_support_map_length_rejected(self):
        lat_grid = np.linspace(0.0, 2.0, 3)
        lon_grid = np.linspace(0.0, 2.0, 3)
        support_map = _support_map_from_2d(
            [[(0, 0.0, 0) for _ in lon_grid] for _ in lat_grid], lat_grid, lon_grid
        )
        with self.assertRaises(ValueError):
            find_candidate_peaks(support_map[:-1], lat_grid, lon_grid, QUERY_TIME)


class TestReproducibility(unittest.TestCase):
    def test_repeated_calls_give_identical_candidates(self):
        lat_grid = np.linspace(0.0, 10.0, 11)
        lon_grid = np.linspace(0.0, 10.0, 11)
        values = [[(0, 0.0, 0) for _ in lon_grid] for _ in lat_grid]
        values[2][2] = (10, 10.0, 2)
        values[8][8] = (10, 10.0, 2)  # equal support -- exercises the tie-break path too
        support_map = _support_map_from_2d(values, lat_grid, lon_grid)
        config = CandidateExtractionConfig(min_separation_m=1000.0, core_radius_m=500.0)

        r1 = find_candidate_peaks(support_map, lat_grid, lon_grid, QUERY_TIME, config)
        r2 = find_candidate_peaks(support_map, lat_grid, lon_grid, QUERY_TIME, config)

        self.assertEqual([c.candidate_id for c in r1], [c.candidate_id for c in r2])
        self.assertEqual([(c.x_peak, c.y_peak) for c in r1], [(c.x_peak, c.y_peak) for c in r2])
        self.assertEqual([c.support_score for c in r1], [c.support_score for c in r2])


class TestIntegrationWithIndexAndSupportMap(unittest.TestCase):
    """A small end-to-end check through T14/T15/T16's real machinery, not just
    hand-built support maps."""

    def test_two_clusters_of_real_particles_yield_two_candidates(self):
        rng = np.random.default_rng(0)
        n_cluster = 15
        lat1 = 20.0 + rng.normal(0, 0.003, n_cluster)
        lon1 = 100.0 + rng.normal(0, 0.003, n_cluster)
        lat2 = 25.0 + rng.normal(0, 0.003, n_cluster)
        lon2 = 105.0 + rng.normal(0, 0.003, n_cluster)
        lat = np.concatenate([lat1, lat2])
        lon = np.concatenate([lon1, lon2])
        n = len(lat)
        slice_ = HistoricalParticleSlice(
            particle_id=np.array([f"p{i}" for i in range(n)], dtype=object),
            observation_pixel_id=np.array(["px-0"] * n_cluster + ["px-1"] * n_cluster, dtype=object),
            forcing_member_id=np.zeros(n, dtype=np.int64),
            lat=lat, lon=lon, log_weight=np.zeros(n), valid=np.ones(n, dtype=bool),
            death_time=np.full(n, NAT, dtype="datetime64[ns]"),
            death_lat=np.full(n, np.nan), death_lon=np.full(n, np.nan),
            death_reason=np.array([""] * n, dtype=object), query_time=QUERY_TIME,
        )
        index = build_particle_neighborhood_index(slice_)
        lat_grid = np.linspace(19.5, 25.5, 25)
        lon_grid = np.linspace(99.5, 105.5, 25)
        config = CandidateExtractionConfig(min_separation_m=50_000.0, core_radius_m=2000.0)

        candidates = find_candidate_peaks_from_index(
            index, lat_grid, lon_grid, QUERY_TIME, radius_m=3000.0, extraction_config=config
        )
        self.assertEqual(len(candidates), 2)
        for c in candidates:
            self.assertGreater(c.density_score, 0)
            self.assertEqual(c.pixel_diversity_score, 1)  # each cluster is one pixel


if __name__ == "__main__":
    unittest.main()
