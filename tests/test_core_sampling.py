"""Focused T19 checks for `iste.core_sampling` (Stage V weighted core-particle
sampling for forward validation).

Run with:
    python -m unittest tests.test_core_sampling -v
"""

from __future__ import annotations

import unittest
from collections import Counter

import numpy as np

from iste.candidate_extraction import Candidate, CoreRegion
from iste.core_sampling import (
    SampledCoreParticles,
    sample_core_particles,
    sample_core_particles_for_candidates,
)
from iste.forward_budget import ForwardBudgetConfig
from iste.geometry import metres_per_degree_latitude
from iste.kdtree_neighborhood import build_particle_neighborhood_index
from iste.particle_query import HistoricalParticleSlice
from iste.support_features import SupportFeatureConfig, compute_support_features

NAT = np.datetime64("NaT", "ns")
T_S = np.datetime64("2020-01-01T06:00:00", "ns")
LAT0, LON0 = 10.0, 50.0
M_PER_DEG_LAT = metres_per_degree_latitude()


def _make_slice(lat, lon, log_weight=None, valid=None, t_s=T_S):
    n = len(lat)
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    return HistoricalParticleSlice(
        particle_id=np.array([f"p{i}" for i in range(n)], dtype=object),
        observation_pixel_id=np.array([f"px-{i % 3}" for i in range(n)], dtype=object),
        forcing_member_id=np.array([i % 2 for i in range(n)], dtype=np.int64),
        lat=lat,
        lon=lon,
        log_weight=np.array(log_weight if log_weight is not None else [0.0] * n, dtype=np.float64),
        valid=np.array(valid if valid is not None else [True] * n, dtype=bool),
        death_time=np.full(n, NAT, dtype="datetime64[ns]"),
        death_lat=np.full(n, np.nan),
        death_lon=np.full(n, np.nan),
        death_reason=np.array([""] * n, dtype=object),
        query_time=t_s,
    )


def _north_offset_points(distances_m, lat0=LAT0, lon0=LON0):
    lat = lat0 + np.asarray(distances_m, dtype=np.float64) / M_PER_DEG_LAT
    lon = np.full(len(distances_m), lon0, dtype=np.float64)
    return lat, lon


def _dummy_support_features(index):
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


def _core_candidate(n_particles, radius_m=1000.0, log_weight=None, valid=None):
    lat, lon = _north_offset_points([50.0] * n_particles)
    slice_ = _make_slice(lat, lon, log_weight=log_weight, valid=valid)
    index = build_particle_neighborhood_index(slice_, lat_ref=LAT0, lon_ref=LON0)
    core = CoreRegion(center_lat=LAT0, center_lon=LON0, radius_m=radius_m)
    candidate = _make_candidate("cand", core, index)
    return candidate, index


class TestBranchCounts(unittest.TestCase):
    """Requirements 1-3: exact selected-count per Y/X relationship."""

    def test_y_greater_than_x_selects_exactly_x(self):
        candidate, index = _core_candidate(n_particles=10)
        result = sample_core_particles(candidate, index, ForwardBudgetConfig(X=4), seed=1)
        self.assertEqual(result.Y, 10)
        self.assertEqual(result.N_forward, 4)
        self.assertEqual(result.n_selected, 4)
        self.assertTrue(result.resampled)

    def test_y_equal_to_x_selects_all_x(self):
        candidate, index = _core_candidate(n_particles=5)
        result = sample_core_particles(candidate, index, ForwardBudgetConfig(X=5), seed=1)
        self.assertEqual(result.Y, 5)
        self.assertEqual(result.N_forward, 5)
        self.assertEqual(result.n_selected, 5)
        self.assertTrue(result.resampled)
        # all 5 original particles must appear -- no choice possible when pool == draw size
        self.assertEqual(set(result.particle_id.tolist()), {f"p{i}" for i in range(5)})

    def test_y_less_than_x_selects_all_y_and_is_not_rejected(self):
        candidate, index = _core_candidate(n_particles=3)
        result = sample_core_particles(candidate, index, ForwardBudgetConfig(X=10), seed=1)
        self.assertEqual(result.Y, 3)
        self.assertEqual(result.N_forward, 3)
        self.assertEqual(result.n_selected, 3)
        self.assertFalse(result.resampled)
        self.assertTrue(result.forward_validatable)
        self.assertEqual(set(result.particle_id.tolist()), {"p0", "p1", "p2"})


class TestZeroYExplicitEmptySelection(unittest.TestCase):
    """Requirement 4: Y = 0 -> explicit unvalidated/empty selection, never raises."""

    def test_empty_core_returns_empty_selection(self):
        lat, lon = _north_offset_points([5000.0, 8000.0])  # outside the 1000 m core
        slice_ = _make_slice(lat, lon)
        index = build_particle_neighborhood_index(slice_, lat_ref=LAT0, lon_ref=LON0)
        core = CoreRegion(center_lat=LAT0, center_lon=LON0, radius_m=1000.0)
        candidate = _make_candidate("cand-empty", core, index)

        result = sample_core_particles(candidate, index, ForwardBudgetConfig(X=5), seed=1)

        self.assertEqual(result.Y, 0)
        self.assertEqual(result.N_forward, 0)
        self.assertEqual(result.n_selected, 0)
        self.assertFalse(result.forward_validatable)
        self.assertFalse(result.resampled)
        self.assertIsInstance(result, SampledCoreParticles)
        for arr in (
            result.particle_id, result.observation_pixel_id, result.forcing_member_id,
            result.lat, result.lon, result.log_weight, result.valid,
            result.death_time, result.death_lat, result.death_lon, result.death_reason,
        ):
            self.assertEqual(len(arr), 0)


class TestReproducibility(unittest.TestCase):
    """Requirements 5, 6, 11: seed-controlled determinism and independence."""

    def test_same_seed_gives_identical_selection(self):
        candidate, index = _core_candidate(n_particles=20)
        config = ForwardBudgetConfig(X=6)
        r1 = sample_core_particles(candidate, index, config, seed=42)
        r2 = sample_core_particles(candidate, index, config, seed=42)
        self.assertEqual(r1.particle_id.tolist(), r2.particle_id.tolist())
        self.assertEqual(r1.sampling_seed, 42)
        self.assertEqual(r2.sampling_seed, 42)

    def test_different_seeds_can_differ(self):
        candidate, index = _core_candidate(n_particles=30)
        config = ForwardBudgetConfig(X=6)
        selections = {
            seed: tuple(sample_core_particles(candidate, index, config, seed=seed).particle_id.tolist())
            for seed in range(10)
        }
        self.assertGreater(len(set(selections.values())), 1)

    def test_same_candidate_sampled_independently_and_reproducibly_multiple_times(self):
        candidate, index = _core_candidate(n_particles=15)
        config = ForwardBudgetConfig(X=5)
        # Re-run the same (candidate, seed) pair interleaved with a DIFFERENT seed to
        # confirm each call is a fresh, independent RNG draw (no shared/mutated state
        # leaking between calls), yet still reproducible for a fixed seed.
        a1 = sample_core_particles(candidate, index, config, seed=7)
        _ = sample_core_particles(candidate, index, config, seed=99)
        a2 = sample_core_particles(candidate, index, config, seed=7)
        self.assertEqual(a1.particle_id.tolist(), a2.particle_id.tolist())


class TestWithoutReplacement(unittest.TestCase):
    """Requirement 8: selection is without replacement (default)."""

    def test_no_duplicate_particles_in_selection(self):
        candidate, index = _core_candidate(n_particles=25)
        result = sample_core_particles(candidate, index, ForwardBudgetConfig(X=12), seed=3)
        ids = result.particle_id.tolist()
        self.assertEqual(len(ids), len(set(ids)))

    def test_replace_true_can_select_duplicates(self):
        # Y (4) >= X (3), so this stays in the weighted-resampling branch regardless of
        # `replace`; with replace=True and a highly skewed weight, duplicates become
        # likely -- confirms the `replace` flag actually reaches numpy's sampler.
        n = 4
        log_weight = [0.0, 0.0, 0.0, 50.0]  # last particle vastly dominant
        candidate, index = _core_candidate(n_particles=n, log_weight=log_weight)

        found_duplicate = False
        for seed in range(20):
            result = sample_core_particles(
                candidate, index, ForwardBudgetConfig(X=3), seed=seed, replace=True
            )
            self.assertEqual(result.n_selected, 3)
            ids = result.particle_id.tolist()
            if len(ids) != len(set(ids)):
                found_duplicate = True
                break
        self.assertTrue(found_duplicate, "expected at least one seed to produce a duplicate with replace=True")


class TestHigherWeightHigherSelectionProbability(unittest.TestCase):
    """Requirement 7: higher-weight particles selected more often over repeated trials."""

    def test_empirical_selection_frequency_tracks_weight(self):
        n = 6
        # exponentially increasing weights: p5 should be selected far more often than p0
        log_weight = [0.0, 0.5, 1.0, 1.5, 2.0, 4.0]
        candidate, index = _core_candidate(n_particles=n, log_weight=log_weight)
        config = ForwardBudgetConfig(X=2)  # Y=6 >= X=2 -> weighted resampling branch

        counts = Counter()
        n_trials = 800
        for seed in range(n_trials):
            result = sample_core_particles(candidate, index, config, seed=seed)
            counts.update(result.particle_id.tolist())

        freq = {pid: counts.get(pid, 0) / n_trials for pid in [f"p{i}" for i in range(n)]}
        # Monotonic (or very close to it) increase in empirical frequency with weight;
        # assert the extremes at least, robust to Monte Carlo noise.
        self.assertGreater(freq["p5"], freq["p0"])
        self.assertGreater(freq["p5"], freq["p3"])
        self.assertGreater(freq["p4"], freq["p1"])

    def test_zero_weight_particle_rarely_selected_vs_dominant_one(self):
        n = 4
        log_weight = [float("-inf"), 0.0, 0.0, 20.0]
        candidate, index = _core_candidate(n_particles=n, log_weight=log_weight)
        config = ForwardBudgetConfig(X=1)

        counts = Counter()
        n_trials = 300
        for seed in range(n_trials):
            result = sample_core_particles(candidate, index, config, seed=seed)
            counts.update(result.particle_id.tolist())

        self.assertEqual(counts.get("p0", 0), 0)  # zero weight -> never selected
        self.assertGreater(counts.get("p3", 0), n_trials * 0.9)  # dominant weight -> almost always


class TestMetadataPreservation(unittest.TestCase):
    """Requirement 9: all required particle metadata preserved for selected particles."""

    def test_all_fields_match_source_slice(self):
        lat, lon = _north_offset_points([10.0, 20.0, 30.0, 40.0, 50.0])
        log_weight = [0.1, 0.2, 0.3, 0.4, 0.5]
        death_time = [NAT, NAT, NAT, NAT, NAT]
        slice_ = _make_slice(lat, lon, log_weight=log_weight)
        index = build_particle_neighborhood_index(slice_, lat_ref=LAT0, lon_ref=LON0)
        core = CoreRegion(center_lat=LAT0, center_lon=LON0, radius_m=1000.0)
        candidate = _make_candidate("cand-meta", core, index)

        result = sample_core_particles(candidate, index, ForwardBudgetConfig(X=3), seed=5)

        self.assertEqual(result.n_selected, 3)
        for i, pid in enumerate(result.particle_id):
            src_idx = int(pid[1:])  # "p3" -> 3
            self.assertEqual(result.observation_pixel_id[i], slice_.observation_pixel_id[src_idx])
            self.assertEqual(result.forcing_member_id[i], slice_.forcing_member_id[src_idx])
            self.assertAlmostEqual(result.lat[i], slice_.lat[src_idx])
            self.assertAlmostEqual(result.lon[i], slice_.lon[src_idx])
            self.assertAlmostEqual(result.log_weight[i], slice_.log_weight[src_idx])
            self.assertEqual(result.valid[i], slice_.valid[src_idx])
            self.assertEqual(np.isnat(result.death_time[i]), np.isnat(slice_.death_time[src_idx]))

    def test_death_metadata_preserved_for_dead_but_still_indexed_case(self):
        # T15 only ever indexes valid particles, so every selected particle here has
        # valid=True and NaT/NaN death metadata by construction -- this test confirms
        # those fields are still carried through the selection untouched.
        candidate, index = _core_candidate(n_particles=4)
        result = sample_core_particles(candidate, index, ForwardBudgetConfig(X=2), seed=1)
        self.assertTrue(np.all(result.valid))
        self.assertTrue(np.all(np.isnat(result.death_time)))
        self.assertTrue(np.all(np.isnan(result.death_lat)))
        self.assertTrue(np.all(np.isnan(result.death_lon)))


class TestOriginalWeightsUnchanged(unittest.TestCase):
    """Requirement 10: original Stage III weights are unchanged (not renormalized)."""

    def test_source_slice_log_weight_untouched_after_sampling(self):
        log_weight = [0.3, -1.2, 2.5, 0.0, -0.7]
        candidate, index = _core_candidate(n_particles=5, log_weight=log_weight)
        original = np.array(log_weight, dtype=np.float64).copy()

        sample_core_particles(candidate, index, ForwardBudgetConfig(X=2), seed=1)  # Y>=X branch
        sample_core_particles(candidate, index, ForwardBudgetConfig(X=10), seed=1)  # Y<X branch

        np.testing.assert_array_equal(index.slice_.log_weight, original)

    def test_selected_log_weight_matches_source_exactly_no_normalization(self):
        log_weight = [0.3, -1.2, 2.5, 0.0, -0.7]
        candidate, index = _core_candidate(n_particles=5, log_weight=log_weight)
        result = sample_core_particles(candidate, index, ForwardBudgetConfig(X=10), seed=1)  # Y<X: all retained
        # sum of selected weights should NOT be normalized to 1 (or any fixed value) --
        # they are exactly the original values.
        for i, pid in enumerate(result.particle_id):
            src_idx = int(pid[1:])
            self.assertEqual(result.log_weight[i], log_weight[src_idx])


class TestBatchHelper(unittest.TestCase):
    def test_seeds_length_mismatch_raises(self):
        candidate, index = _core_candidate(n_particles=5)
        with self.assertRaises(ValueError):
            sample_core_particles_for_candidates([candidate], index, ForwardBudgetConfig(X=2), seeds=[])

    def test_batch_matches_individual_calls(self):
        lat_a, lon_a = _north_offset_points([50.0] * 6)
        lat_b, lon_b = _north_offset_points([50.0] * 6, lat0=20.0, lon0=60.0)
        lat = np.concatenate([lat_a, lat_b])
        lon = np.concatenate([lon_a, lon_b])
        slice_ = _make_slice(lat, lon)
        index = build_particle_neighborhood_index(slice_)
        core_a = CoreRegion(center_lat=10.0, center_lon=50.0, radius_m=1000.0)
        core_b = CoreRegion(center_lat=20.0, center_lon=60.0, radius_m=1000.0)
        cand_a = _make_candidate("cand-a", core_a, index)
        cand_b = _make_candidate("cand-b", core_b, index)
        config = ForwardBudgetConfig(X=3)

        individual_a = sample_core_particles(cand_a, index, config, seed=11)
        individual_b = sample_core_particles(cand_b, index, config, seed=22)
        batch = sample_core_particles_for_candidates([cand_a, cand_b], index, config, seeds=[11, 22])

        self.assertEqual(batch[0].particle_id.tolist(), individual_a.particle_id.tolist())
        self.assertEqual(batch[1].particle_id.tolist(), individual_b.particle_id.tolist())


class TestSelectionProbabilitiesEdgeCases(unittest.TestCase):
    def test_all_zero_weight_falls_back_to_uniform_without_raising(self):
        n = 5
        log_weight = [float("-inf")] * n
        candidate, index = _core_candidate(n_particles=n, log_weight=log_weight)
        result = sample_core_particles(candidate, index, ForwardBudgetConfig(X=3), seed=1)
        self.assertEqual(result.n_selected, 3)


if __name__ == "__main__":
    unittest.main()
