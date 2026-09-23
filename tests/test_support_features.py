"""Focused T16 checks for `iste.support_features` (Phase 2 candidate-support feature
construction: D, W, V, Support).

Run with:
    python -m unittest tests.test_support_features -v
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from iste.kdtree_neighborhood import build_particle_neighborhood_index
from iste.particle_query import HistoricalParticleSlice
from iste.support_features import (
    SupportFeatureConfig,
    compute_support_features,
    compute_support_map,
    linear_support_function,
)

QUERY_TIME = np.datetime64("2020-01-01T00:00:00", "ns")
NAT = np.datetime64("NaT", "ns")


def _make_slice(particle_id, observation_pixel_id, forcing_member_id, lat, lon, log_weight, valid=None) -> HistoricalParticleSlice:
    n = len(lat)
    valid = valid if valid is not None else [True] * n
    return HistoricalParticleSlice(
        particle_id=np.array(particle_id, dtype=object),
        observation_pixel_id=np.array(observation_pixel_id, dtype=object),
        forcing_member_id=np.array(forcing_member_id, dtype=np.int64),
        lat=np.array(lat, dtype=np.float64),
        lon=np.array(lon, dtype=np.float64),
        log_weight=np.array(log_weight, dtype=np.float64),
        valid=np.array(valid, dtype=bool),
        death_time=np.full(n, NAT, dtype="datetime64[ns]"),
        death_lat=np.full(n, np.nan),
        death_lon=np.full(n, np.nan),
        death_reason=np.array([""] * n, dtype=object),
        query_time=QUERY_TIME,
    )


class TestDensityD(unittest.TestCase):
    def test_density_matches_controlled_neighborhood_count(self):
        slice_ = _make_slice(
            particle_id=[f"p{i}" for i in range(6)],
            observation_pixel_id=["px-0"] * 6,
            forcing_member_id=[0] * 6,
            lat=[20.0, 20.001, 20.002, 20.003, 20.5, 20.6],  # first 4 close, last 2 far
            lon=[100.0] * 6,
            log_weight=[0.0] * 6,
        )
        index = build_particle_neighborhood_index(slice_)
        result = compute_support_features(index, 20.0, 100.0, radius_m=1000.0)
        self.assertEqual(result.density, 4)

    def test_density_zero_when_nothing_nearby(self):
        slice_ = _make_slice(
            particle_id=["a"], observation_pixel_id=["px-0"], forcing_member_id=[0],
            lat=[45.0], lon=[45.0], log_weight=[0.0],
        )
        index = build_particle_neighborhood_index(slice_)
        result = compute_support_features(index, 20.0, 100.0, radius_m=1000.0)
        self.assertEqual(result.density, 0)
        self.assertEqual(result.weighted_support, 0.0)
        self.assertEqual(result.pixel_diversity, 0)


class TestWeightedSupportW(unittest.TestCase):
    def test_w_is_exact_sum_of_linear_weights(self):
        # log-weights chosen so exp() gives clean numbers: 0->1, ln(2)->2, ln(3)->3
        slice_ = _make_slice(
            particle_id=["a", "b", "c"], observation_pixel_id=["px-0"] * 3, forcing_member_id=[0] * 3,
            lat=[20.0, 20.001, 20.002], lon=[100.0, 100.0, 100.0],
            log_weight=[0.0, math.log(2.0), math.log(3.0)],
        )
        index = build_particle_neighborhood_index(slice_)
        result = compute_support_features(index, 20.0, 100.0, radius_m=1000.0)
        self.assertAlmostEqual(result.weighted_support, 1.0 + 2.0 + 3.0, places=9)

    def test_w_uses_linear_not_log_weight(self):
        # a naive bug might sum log_weight directly instead of exp(log_weight); these
        # two must differ for non-trivial weights.
        slice_ = _make_slice(
            particle_id=["a", "b"], observation_pixel_id=["px-0"] * 2, forcing_member_id=[0] * 2,
            lat=[20.0, 20.001], lon=[100.0, 100.0],
            log_weight=[math.log(5.0), math.log(0.5)],
        )
        index = build_particle_neighborhood_index(slice_)
        result = compute_support_features(index, 20.0, 100.0, radius_m=1000.0)
        naive_wrong_sum = math.log(5.0) + math.log(0.5)  # sum of LOG weights (wrong)
        self.assertAlmostEqual(result.weighted_support, 5.5, places=9)
        self.assertNotAlmostEqual(result.weighted_support, naive_wrong_sum)

    def test_stage3_weights_not_normalized_or_reinterpreted(self):
        # W can legitimately exceed 1 or vary wildly -- it is not a probability and must
        # not be silently clamped/normalized to [0,1].
        slice_ = _make_slice(
            particle_id=["a", "b"], observation_pixel_id=["px-0"] * 2, forcing_member_id=[0] * 2,
            lat=[20.0, 20.001], lon=[100.0, 100.0],
            log_weight=[math.log(50.0), math.log(50.0)],
        )
        index = build_particle_neighborhood_index(slice_)
        result = compute_support_features(index, 20.0, 100.0, radius_m=1000.0)
        self.assertAlmostEqual(result.weighted_support, 100.0, places=6)


class TestPixelDiversityV(unittest.TestCase):
    def test_v_counts_distinct_pixels_not_particles(self):
        slice_ = _make_slice(
            particle_id=["a", "b", "c", "d", "e"],
            observation_pixel_id=["px-0", "px-0", "px-0", "px-1", "px-2"],
            forcing_member_id=[0, 1, 2, 0, 0],
            lat=[20.0] * 5, lon=[100.0] * 5,
            log_weight=[0.0] * 5,
        )
        index = build_particle_neighborhood_index(slice_)
        result = compute_support_features(index, 20.0, 100.0, radius_m=1000.0)
        self.assertEqual(result.density, 5)  # 5 particles
        self.assertEqual(result.pixel_diversity, 3)  # but only 3 distinct pixels

    def test_v_is_one_when_all_particles_share_a_pixel(self):
        slice_ = _make_slice(
            particle_id=["a", "b", "c"], observation_pixel_id=["px-0"] * 3, forcing_member_id=[0, 1, 2],
            lat=[20.0] * 3, lon=[100.0] * 3, log_weight=[0.0] * 3,
        )
        index = build_particle_neighborhood_index(slice_)
        result = compute_support_features(index, 20.0, 100.0, radius_m=1000.0)
        self.assertEqual(result.pixel_diversity, 1)


class TestInvalidParticlesExcluded(unittest.TestCase):
    def test_invalid_particles_contribute_nothing_to_d_w_v(self):
        slice_ = _make_slice(
            particle_id=["a", "b"], observation_pixel_id=["px-0", "px-1"], forcing_member_id=[0, 0],
            lat=[20.0, 20.0], lon=[100.0, 100.0],
            log_weight=[0.0, math.log(1000.0)],  # huge weight, but invalid
            valid=[True, False],
        )
        index = build_particle_neighborhood_index(slice_)
        result = compute_support_features(index, 20.0, 100.0, radius_m=1000.0)
        self.assertEqual(result.density, 1)
        self.assertAlmostEqual(result.weighted_support, 1.0)  # only particle "a"'s weight
        self.assertEqual(result.pixel_diversity, 1)  # only px-0, not px-1

    def test_all_invalid_gives_zero_features(self):
        slice_ = _make_slice(
            particle_id=["a", "b"], observation_pixel_id=["px-0", "px-1"], forcing_member_id=[0, 0],
            lat=[20.0, 20.0], lon=[100.0, 100.0], log_weight=[0.0, 0.0], valid=[False, False],
        )
        index = build_particle_neighborhood_index(slice_)
        result = compute_support_features(index, 20.0, 100.0, radius_m=1000.0)
        self.assertEqual(result.density, 0)
        self.assertEqual(result.weighted_support, 0.0)
        self.assertEqual(result.pixel_diversity, 0)


class TestIndependentConfigurations(unittest.TestCase):
    """Requirement 5: different input configurations can move D/W/V independently."""

    def test_changing_weights_alone_changes_w_not_d_or_v(self):
        base = _make_slice(
            particle_id=["a", "b"], observation_pixel_id=["px-0", "px-1"], forcing_member_id=[0, 0],
            lat=[20.0, 20.001], lon=[100.0, 100.0], log_weight=[0.0, 0.0],
        )
        heavier = _make_slice(
            particle_id=["a", "b"], observation_pixel_id=["px-0", "px-1"], forcing_member_id=[0, 0],
            lat=[20.0, 20.001], lon=[100.0, 100.0], log_weight=[math.log(10.0), math.log(10.0)],
        )
        r_base = compute_support_features(build_particle_neighborhood_index(base), 20.0, 100.0, radius_m=1000.0)
        r_heavier = compute_support_features(build_particle_neighborhood_index(heavier), 20.0, 100.0, radius_m=1000.0)
        self.assertEqual(r_base.density, r_heavier.density)
        self.assertEqual(r_base.pixel_diversity, r_heavier.pixel_diversity)
        self.assertNotAlmostEqual(r_base.weighted_support, r_heavier.weighted_support)

    def test_adding_particles_at_the_same_pixel_changes_d_not_v(self):
        one_pixel_1p = _make_slice(
            particle_id=["a"], observation_pixel_id=["px-0"], forcing_member_id=[0],
            lat=[20.0], lon=[100.0], log_weight=[0.0],
        )
        one_pixel_3p = _make_slice(
            particle_id=["a", "b", "c"], observation_pixel_id=["px-0"] * 3, forcing_member_id=[0, 1, 2],
            lat=[20.0, 20.0005, 20.001], lon=[100.0] * 3, log_weight=[0.0] * 3,
        )
        r1 = compute_support_features(build_particle_neighborhood_index(one_pixel_1p), 20.0, 100.0, radius_m=1000.0)
        r3 = compute_support_features(build_particle_neighborhood_index(one_pixel_3p), 20.0, 100.0, radius_m=1000.0)
        self.assertEqual(r1.pixel_diversity, r3.pixel_diversity)  # both V=1
        self.assertNotEqual(r1.density, r3.density)  # D differs (1 vs 3)

    def test_neighborhood_radius_changes_d_w_v_together(self):
        slice_ = _make_slice(
            particle_id=["a", "b", "c"], observation_pixel_id=["px-0", "px-1", "px-2"], forcing_member_id=[0, 0, 0],
            lat=[20.0, 20.001, 20.5], lon=[100.0, 100.0, 100.0],
            log_weight=[0.0, 0.0, 0.0],
        )
        index = build_particle_neighborhood_index(slice_)
        small = compute_support_features(index, 20.0, 100.0, radius_m=500.0)
        large = compute_support_features(index, 20.0, 100.0, radius_m=200_000.0)
        self.assertLess(small.density, large.density)
        self.assertLess(small.pixel_diversity, large.pixel_diversity)


class TestExplicitSupportFunction(unittest.TestCase):
    """Requirement 6: Support uses the configured scaling/function explicitly."""

    def test_default_config_is_unweighted_sum(self):
        slice_ = _make_slice(
            particle_id=["a", "b"], observation_pixel_id=["px-0", "px-1"], forcing_member_id=[0, 0],
            lat=[20.0, 20.001], lon=[100.0, 100.0], log_weight=[0.0, math.log(3.0)],
        )
        result = compute_support_features(build_particle_neighborhood_index(slice_), 20.0, 100.0, radius_m=1000.0)
        expected = result.density + result.weighted_support + result.pixel_diversity
        self.assertAlmostEqual(result.support, expected, places=9)

    def test_zeroing_a_weight_removes_that_features_influence(self):
        slice_ = _make_slice(
            particle_id=["a", "b"], observation_pixel_id=["px-0", "px-1"], forcing_member_id=[0, 0],
            lat=[20.0, 20.001], lon=[100.0, 100.0], log_weight=[0.0, math.log(999.0)],
        )
        index = build_particle_neighborhood_index(slice_)
        cfg_no_w = SupportFeatureConfig(weight_w=0.0)
        result = compute_support_features(index, 20.0, 100.0, radius_m=1000.0, config=cfg_no_w)
        self.assertAlmostEqual(result.support, result.density + result.pixel_diversity, places=6)

    def test_scaling_divisors_change_support_value(self):
        slice_ = _make_slice(
            particle_id=["a", "b", "c"], observation_pixel_id=["px-0"] * 3, forcing_member_id=[0, 1, 2],
            lat=[20.0, 20.0005, 20.001], lon=[100.0] * 3, log_weight=[0.0] * 3,
        )
        index = build_particle_neighborhood_index(slice_)
        unscaled = compute_support_features(index, 20.0, 100.0, radius_m=1000.0)
        scaled = compute_support_features(
            index, 20.0, 100.0, radius_m=1000.0, config=SupportFeatureConfig(d_scale=3.0)
        )
        self.assertNotAlmostEqual(unscaled.support, scaled.support)

    def test_custom_combine_fn_is_actually_used(self):
        slice_ = _make_slice(
            particle_id=["a"], observation_pixel_id=["px-0"], forcing_member_id=[0],
            lat=[20.0], lon=[100.0], log_weight=[0.0],
        )
        index = build_particle_neighborhood_index(slice_)

        def always_42(d, w, v, config):
            return 42.0

        result = compute_support_features(index, 20.0, 100.0, radius_m=1000.0, combine_fn=always_42)
        self.assertEqual(result.support, 42.0)

    def test_invalid_config_rejected(self):
        with self.assertRaises(ValueError):
            SupportFeatureConfig(d_scale=0.0)
        with self.assertRaises(ValueError):
            SupportFeatureConfig(w_scale=-1.0)
        with self.assertRaises(ValueError):
            SupportFeatureConfig(weight_d=float("nan"))


class TestMetadataAvailable(unittest.TestCase):
    """Requirement 7: observation-pixel/particle identity remains available."""

    def test_neighborhood_metadata_accessible_on_result(self):
        slice_ = _make_slice(
            particle_id=["a", "b"], observation_pixel_id=["px-7", "px-8"], forcing_member_id=[3, 4],
            lat=[20.0, 20.001], lon=[100.0, 100.0], log_weight=[0.0, 0.0],
        )
        index = build_particle_neighborhood_index(slice_)
        result = compute_support_features(index, 20.0, 100.0, radius_m=1000.0)
        self.assertEqual(set(result.neighborhood.observation_pixel_id.tolist()), {"px-7", "px-8"})
        self.assertEqual(set(result.neighborhood.forcing_member_id.tolist()), {3, 4})
        self.assertEqual(set(result.neighborhood.particle_id.tolist()), {"a", "b"})


class TestControlledSupportOrdering(unittest.TestCase):
    """Requirement 8: a controlled synthetic case produces the expected support
    ordering."""

    def test_denser_heavier_more_diverse_region_ranks_higher(self):
        # region A: many particles, high weight, many distinct pixels, near (20,100)
        # region B: few particles, low weight, one pixel, near (30,110) -- far apart so
        # queries don't cross-contaminate.
        slice_ = _make_slice(
            particle_id=["a0", "a1", "a2", "a3", "b0"],
            observation_pixel_id=["px-0", "px-1", "px-2", "px-3", "px-9"],
            forcing_member_id=[0, 1, 2, 3, 0],
            lat=[20.0, 20.0005, 20.001, 20.0015, 30.0],
            lon=[100.0, 100.0, 100.0005, 100.001, 110.0],
            log_weight=[math.log(5.0)] * 4 + [0.0],
        )
        index = build_particle_neighborhood_index(slice_)
        region_a = compute_support_map(index, [20.0], [100.0], radius_m=1000.0)[0]
        region_b = compute_support_map(index, [30.0], [110.0], radius_m=1000.0)[0]

        self.assertGreater(region_a.density, region_b.density)
        self.assertGreater(region_a.weighted_support, region_b.weighted_support)
        self.assertGreater(region_a.pixel_diversity, region_b.pixel_diversity)
        self.assertGreater(region_a.support, region_b.support)

    def test_support_map_preserves_query_order(self):
        slice_ = _make_slice(
            particle_id=["a"], observation_pixel_id=["px-0"], forcing_member_id=[0],
            lat=[20.0], lon=[100.0], log_weight=[0.0],
        )
        index = build_particle_neighborhood_index(slice_)
        results = compute_support_map(index, [20.0, 30.0, 45.0], [100.0, 110.0, 45.0], radius_m=1000.0)
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0].density, 1)  # near the one particle
        self.assertEqual(results[1].density, 0)
        self.assertEqual(results[2].density, 0)


class TestInputValidation(unittest.TestCase):
    def test_requires_exactly_one_of_radius_or_k(self):
        slice_ = _make_slice(
            particle_id=["a"], observation_pixel_id=["px-0"], forcing_member_id=[0],
            lat=[20.0], lon=[100.0], log_weight=[0.0],
        )
        index = build_particle_neighborhood_index(slice_)
        with self.assertRaises(ValueError):
            compute_support_features(index, 20.0, 100.0)
        with self.assertRaises(ValueError):
            compute_support_features(index, 20.0, 100.0, radius_m=1000.0, k=3)

    def test_support_map_rejects_mismatched_lengths(self):
        slice_ = _make_slice(
            particle_id=["a"], observation_pixel_id=["px-0"], forcing_member_id=[0],
            lat=[20.0], lon=[100.0], log_weight=[0.0],
        )
        index = build_particle_neighborhood_index(slice_)
        with self.assertRaises(ValueError):
            compute_support_map(index, [20.0, 21.0], [100.0], radius_m=1000.0)


if __name__ == "__main__":
    unittest.main()
