"""Focused T38 checks for `iste.phase3_uncertainty` (Phase 3 uncertainty/bootstrap
stability layer built on top of T37 clusters).

Run with:
    python -m unittest tests.test_phase3_uncertainty -v
"""

from __future__ import annotations

import unittest

import numpy as np

from iste.candidate_extraction import Candidate, CoreRegion
from iste.ensemble_run import EnsembleRunRecords
from iste.kdtree_neighborhood import NeighborhoodResult
from iste.phase3_clustering import CandidateCluster
from iste.phase3_composite import CompositeWeights, SourceScore
from iste.phase3_uncertainty import (
    BootstrapConfig,
    ClusterUncertainty,
    compute_cluster_uncertainty_table,
    compute_spatial_uncertainty_radius_km,
    compute_temporal_uncertainty_hours,
    compute_total_available_particles,
    run_bootstrap_stability,
)
from iste.support_features import SupportFeatureConfig, SupportFeatures

T0 = np.datetime64("2020-01-01T00:00:00", "ns")


def _neighborhood(pids, members, lats, lons, weights=None, pixels=None) -> NeighborhoodResult:
    n = len(pids)
    weights = weights if weights is not None else [0.0] * n
    pixels = pixels if pixels is not None else [f"px{i}" for i in range(n)]
    return NeighborhoodResult(
        particle_id=np.array(pids, dtype=object),
        observation_pixel_id=np.array(pixels, dtype=object),
        forcing_member_id=np.array(members, dtype=np.int64),
        lat=np.array(lats, dtype=np.float64), lon=np.array(lons, dtype=np.float64),
        log_weight=np.array(weights, dtype=np.float64),
        valid=np.array([True] * n, dtype=bool),
        death_time=np.array([np.datetime64("NaT", "ns")] * n, dtype="datetime64[ns]"),
        death_lat=np.array([np.nan] * n), death_lon=np.array([np.nan] * n),
        death_reason=np.array([""] * n, dtype=object),
        distance_m=np.array([0.0] * n, dtype=np.float64),
    )


def _candidate(cand_id, lat, lon, t_s=T0, support_score=10.0, n_particles=5,
               particle_prefix=None) -> Candidate:
    particle_prefix = particle_prefix or cand_id
    pids = [f"{particle_prefix}_p{i}" for i in range(n_particles)]
    members = [0] * n_particles
    lats = [lat] * n_particles
    lons = [lon] * n_particles
    nbhd = _neighborhood(pids, members, lats, lons)
    sf = SupportFeatures(
        query_lat=lat, query_lon=lon, density=n_particles, weighted_support=float(n_particles),
        pixel_diversity=n_particles, support=support_score, neighborhood=nbhd,
        config=SupportFeatureConfig(),
    )
    return Candidate(
        candidate_id=cand_id, x_peak=lon, y_peak=lat, t_s=np.datetime64(t_s, "ns"),
        core_region=CoreRegion(center_lat=lat, center_lon=lon, radius_m=1000.0),
        support_score=support_score, density_score=float(n_particles),
        weighted_support_score=float(n_particles), pixel_diversity_score=float(n_particles),
        peak_support_features=sf,
    )


def _score(cand_id, t_s=T0, support_evidence=0.5, source_score=0.5) -> SourceScore:
    weights = CompositeWeights()
    return SourceScore(
        candidate_id=cand_id, t_s=np.datetime64(t_s, "ns"),
        support_evidence=support_evidence, persistence_evidence=1.0,
        convergence_evidence=0.5, dispersion_evidence=0.5,
        ais_evidence=None, initial_bias_evidence=0.3, persistence_count_evidence=1.0,
        configured_weights=weights.as_dict(), effective_weights=weights.as_dict(),
        source_score=source_score,
    )


def _cluster_from(candidates, scores) -> CandidateCluster:
    from iste.phase3_clustering import cluster_candidates
    return cluster_candidates(candidates, scores, eps_m=5000.0)[0]


def _records_from(candidates) -> EnsembleRunRecords:
    """Build a minimal EnsembleRunRecords whose distinct (particle_id, member) pairs match
    the union of every candidate's own neighborhood particles, one row each."""
    pids, members, pixels, ts, lats, lons, lw, valid = [], [], [], [], [], [], [], []
    seen = set()
    for c in candidates:
        nbhd = c.peak_support_features.neighborhood
        for pid, m, lat, lon in zip(nbhd.particle_id, nbhd.forcing_member_id, nbhd.lat, nbhd.lon):
            key = (pid, int(m))
            if key in seen:
                continue
            seen.add(key)
            pids.append(pid); members.append(int(m)); pixels.append("px0")
            ts.append(T0); lats.append(lat); lons.append(lon); lw.append(0.0); valid.append(True)
    return EnsembleRunRecords(
        particle_id=np.array(pids, dtype=object), forcing_member_id=np.array(members, dtype=np.int64),
        observation_pixel_id=np.array(pixels, dtype=object), timestamp=np.array(ts, dtype="datetime64[ns]"),
        lat=np.array(lats, dtype=np.float64), lon=np.array(lons, dtype=np.float64),
        log_weight=np.array(lw, dtype=np.float64), valid=np.array(valid, dtype=bool),
    )


class TestParticleSupportFraction(unittest.TestCase):
    def test_fraction_matches_hand_count(self):
        c1 = _candidate("a", 34.0, 35.0, n_particles=5, particle_prefix="a")
        c2 = _candidate("b", 34.0001, 35.0001, n_particles=5, particle_prefix="b")  # distinct particles
        scores = [_score("a"), _score("b")]
        cluster = _cluster_from([c1, c2], scores)
        records = _records_from([c1, c2])  # 10 distinct particles total, all in this one cluster
        table = compute_cluster_uncertainty_table(
            [cluster], [c1, c2], scores, records, total_timestamps=73
        )
        row = table[0]
        self.assertEqual(row.total_available_particles, 10)
        self.assertEqual(row.supporting_particle_count, 10)
        self.assertAlmostEqual(row.particle_support_fraction, 1.0)

    def test_shared_particles_not_double_counted(self):
        # b reuses a's particle prefix -> identical (particle_id, member) pairs
        c1 = _candidate("a", 34.0, 35.0, n_particles=5, particle_prefix="shared")
        c2 = _candidate("b", 34.0, 35.0, t_s=T0 - np.timedelta64(1, "h"), n_particles=5, particle_prefix="shared")
        scores = [_score("a"), _score("b", t_s=T0 - np.timedelta64(1, "h"))]
        cluster = _cluster_from([c1, c2], scores)
        records = _records_from([c1, c2])
        table = compute_cluster_uncertainty_table([cluster], [c1, c2], scores, records)
        self.assertEqual(table[0].supporting_particle_count, 5)  # not 10
        self.assertEqual(table[0].total_available_particles, 5)

    def test_fraction_below_one_when_records_has_extra_particles(self):
        c1 = _candidate("a", 34.0, 35.0, n_particles=5, particle_prefix="a")
        scores = [_score("a")]
        cluster = _cluster_from([c1], scores)
        records = _records_from([c1])
        # inflate the denominator with particles not referenced by any candidate
        extra = EnsembleRunRecords(
            particle_id=np.array(["extra1", "extra2"], dtype=object),
            forcing_member_id=np.array([0, 0], dtype=np.int64),
            observation_pixel_id=np.array(["px0", "px0"], dtype=object),
            timestamp=np.array([T0, T0], dtype="datetime64[ns]"),
            lat=np.array([1.0, 2.0]), lon=np.array([1.0, 2.0]),
            log_weight=np.array([0.0, 0.0]), valid=np.array([True, True]),
        )
        combined = EnsembleRunRecords(
            particle_id=np.concatenate([records.particle_id, extra.particle_id]),
            forcing_member_id=np.concatenate([records.forcing_member_id, extra.forcing_member_id]),
            observation_pixel_id=np.concatenate([records.observation_pixel_id, extra.observation_pixel_id]),
            timestamp=np.concatenate([records.timestamp, extra.timestamp]),
            lat=np.concatenate([records.lat, extra.lat]), lon=np.concatenate([records.lon, extra.lon]),
            log_weight=np.concatenate([records.log_weight, extra.log_weight]),
            valid=np.concatenate([records.valid, extra.valid]),
        )
        table = compute_cluster_uncertainty_table([cluster], [c1], scores, combined)
        self.assertEqual(table[0].total_available_particles, 7)
        self.assertAlmostEqual(table[0].particle_support_fraction, 5.0 / 7.0)


class TestPersistenceFraction(unittest.TestCase):
    def test_default_denominator_is_73(self):
        c1 = _candidate("a", 34.0, 35.0)
        scores = [_score("a")]
        cluster = _cluster_from([c1], scores)
        records = _records_from([c1])
        table = compute_cluster_uncertainty_table([cluster], [c1], scores, records)
        self.assertEqual(table[0].total_timestamps, 73)
        self.assertAlmostEqual(table[0].persistence_fraction, 1.0 / 73.0)

    def test_scales_with_distinct_timestamps(self):
        c1 = _candidate("a", 34.0, 35.0, t_s=T0)
        c2 = _candidate("b", 34.0001, 35.0001, t_s=T0 - np.timedelta64(1, "h"))
        scores = [_score("a"), _score("b", t_s=T0 - np.timedelta64(1, "h"))]
        cluster = _cluster_from([c1, c2], scores)
        records = _records_from([c1, c2])
        table = compute_cluster_uncertainty_table([cluster], [c1, c2], scores, records)
        self.assertEqual(table[0].supported_timestamp_count, 2)
        self.assertAlmostEqual(table[0].persistence_fraction, 2.0 / 73.0)


class TestSpatialUncertainty(unittest.TestCase):
    def test_radius_grows_with_particle_spread(self):
        tight = _candidate("tight", 34.0, 35.0, n_particles=1, particle_prefix="tight")
        wide = _candidate("wide", 34.0, 35.0, n_particles=1, particle_prefix="wide")
        # override wide's neighborhood with particles spread far from the peak
        wide.peak_support_features.neighborhood.lat[:] = [34.05]
        wide.peak_support_features.neighborhood.lon[:] = [35.05]

        scores = [_score("tight"), _score("wide")]
        cluster_tight = _cluster_from([tight], [scores[0]])
        cluster_wide = _cluster_from([wide], [scores[1]])
        records = _records_from([tight, wide])
        table_tight = compute_cluster_uncertainty_table([cluster_tight], [tight], [scores[0]], records)
        table_wide = compute_cluster_uncertainty_table([cluster_wide], [wide], [scores[1]], records)
        self.assertGreater(table_wide[0].uncertainty_radius_km, table_tight[0].uncertainty_radius_km)

    def test_nan_for_no_pooled_particles(self):
        from iste.phase3_uncertainty import _pooled_cluster_particles
        c1 = _candidate("a", 34.0, 35.0, n_particles=1)
        scores = [_score("a")]
        cluster = _cluster_from([c1], scores)
        radius = compute_spatial_uncertainty_radius_km(cluster, {})
        self.assertTrue(np.isnan(radius))

    def test_percentile_validated(self):
        c1 = _candidate("a", 34.0, 35.0)
        cluster = _cluster_from([c1], [_score("a")])
        with self.assertRaises(ValueError):
            compute_spatial_uncertainty_radius_km(cluster, {("p", 0): (34.0, 35.0)}, percentile=150.0)


class TestTemporalUncertainty(unittest.TestCase):
    def test_zero_for_single_member_cluster(self):
        c1 = _candidate("a", 34.0, 35.0)
        scores = [_score("a")]
        cluster = _cluster_from([c1], scores)
        candidates_by_id = {"a": c1}
        scores_by_id = {"a": scores[0]}
        self.assertEqual(compute_temporal_uncertainty_hours(cluster, candidates_by_id, scores_by_id), 0.0)

    def test_positive_for_spread_out_equally_weighted_members(self):
        c1 = _candidate("a", 34.0, 35.0, t_s=T0)
        c2 = _candidate("b", 34.0001, 35.0001, t_s=T0 - np.timedelta64(10, "h"), particle_prefix="b")
        scores = [_score("a", t_s=T0, source_score=0.5), _score("b", t_s=T0 - np.timedelta64(10, "h"), source_score=0.5)]
        cluster = _cluster_from([c1, c2], scores)
        candidates_by_id = {"a": c1, "b": c2}
        scores_by_id = {"a": scores[0], "b": scores[1]}
        result = compute_temporal_uncertainty_hours(cluster, candidates_by_id, scores_by_id)
        self.assertAlmostEqual(result, 5.0)  # equal weights, 0 and 10 hours apart -> std = 5


class TestBootstrap(unittest.TestCase):
    def _setup(self):
        c1 = _candidate("a", 34.0, 35.0, support_score=10.0, n_particles=20, particle_prefix="a")
        c2 = _candidate("b", 50.0, 60.0, support_score=1.0, n_particles=20, particle_prefix="b")
        scores = [_score("a", support_evidence=5.0, source_score=5.0), _score("b", support_evidence=0.1, source_score=0.1)]
        from iste.phase3_clustering import cluster_candidates
        clusters = cluster_candidates([c1, c2], scores, eps_m=2000.0)
        return c1, c2, scores, clusters

    def test_reproducible_with_fixed_seed(self):
        c1, c2, scores, clusters = self._setup()
        cfg = BootstrapConfig(n_replicates=30, random_seed=7, top_k=1)
        r1 = run_bootstrap_stability([c1, c2], scores, clusters, config=cfg)
        r2 = run_bootstrap_stability([c1, c2], scores, clusters, config=cfg)
        self.assertEqual(r1.stability_percent_by_cluster, r2.stability_percent_by_cluster)

    def test_stability_within_bounds(self):
        c1, c2, scores, clusters = self._setup()
        cfg = BootstrapConfig(n_replicates=30, random_seed=1, top_k=1)
        result = run_bootstrap_stability([c1, c2], scores, clusters, config=cfg)
        for pct in result.stability_percent_by_cluster.values():
            self.assertGreaterEqual(pct, 0.0)
            self.assertLessEqual(pct, 100.0)

    def test_dominant_candidate_more_stable_than_weak_one(self):
        c1, c2, scores, clusters = self._setup()
        cfg = BootstrapConfig(n_replicates=50, random_seed=3, top_k=1)
        result = run_bootstrap_stability([c1, c2], scores, clusters, config=cfg)
        cluster_a = next(cl.cluster_id for cl in clusters if "a" in cl.member_candidate_ids)
        cluster_b = next(cl.cluster_id for cl in clusters if "b" in cl.member_candidate_ids)
        self.assertGreater(
            result.stability_percent_by_cluster[cluster_a],
            result.stability_percent_by_cluster[cluster_b],
        )

    def test_no_nan_or_inf_in_stability(self):
        c1, c2, scores, clusters = self._setup()
        cfg = BootstrapConfig(n_replicates=20, random_seed=5, top_k=2)
        result = run_bootstrap_stability([c1, c2], scores, clusters, config=cfg)
        for pct in result.stability_percent_by_cluster.values():
            self.assertTrue(np.isfinite(pct))


class TestGracefulDegradation(unittest.TestCase):
    def test_unavailable_stability_reported_as_none_not_zero(self):
        c1 = _candidate("a", 34.0, 35.0)
        scores = [_score("a")]
        cluster = _cluster_from([c1], scores)
        records = _records_from([c1])
        table = compute_cluster_uncertainty_table([cluster], [c1], scores, records, bootstrap_result=None)
        self.assertIsNone(table[0].candidate_stability_percent)

    def test_missing_ais_reported_as_unavailable(self):
        c1 = _candidate("a", 34.0, 35.0)
        scores = [_score("a")]
        cluster = _cluster_from([c1], scores)
        records = _records_from([c1])
        table = compute_cluster_uncertainty_table([cluster], [c1], scores, records)
        self.assertEqual(table[0].ais_status, "unavailable")


class TestNoMutationOfUpstream(unittest.TestCase):
    def test_t36_scores_unchanged(self):
        c1 = _candidate("a", 34.0, 35.0, support_score=10.0)
        scores = [_score("a", source_score=0.42)]
        cluster = _cluster_from([c1], scores)
        records = _records_from([c1])
        compute_cluster_uncertainty_table([cluster], [c1], scores, records)
        self.assertEqual(scores[0].source_score, 0.42)

    def test_t37_cluster_membership_unchanged(self):
        c1 = _candidate("a", 34.0, 35.0)
        c2 = _candidate("b", 34.0001, 35.0001, particle_prefix="b")
        scores = [_score("a"), _score("b")]
        cluster = _cluster_from([c1, c2], scores)
        original_members = list(cluster.member_candidate_ids)
        records = _records_from([c1, c2])
        compute_cluster_uncertainty_table([cluster], [c1, c2], scores, records)
        self.assertEqual(cluster.member_candidate_ids, original_members)


class TestNoNanInf(unittest.TestCase):
    def test_valid_input_produces_finite_core_fields(self):
        c1 = _candidate("a", 34.0, 35.0, n_particles=5)
        c2 = _candidate("b", 34.0001, 35.0001, n_particles=5, particle_prefix="b")
        scores = [_score("a"), _score("b")]
        cluster = _cluster_from([c1, c2], scores)
        records = _records_from([c1, c2])
        row = compute_cluster_uncertainty_table([cluster], [c1, c2], scores, records)[0]
        for value in (
            row.uncertainty_radius_km, row.temporal_uncertainty_hours,
            row.particle_support_fraction, row.persistence_fraction,
        ):
            self.assertTrue(np.isfinite(value))


class TestTimestampsPreserved(unittest.TestCase):
    def test_all_73_represented_in_default_denominator(self):
        c1 = _candidate("a", 34.0, 35.0)
        scores = [_score("a")]
        cluster = _cluster_from([c1], scores)
        records = _records_from([c1])
        table = compute_cluster_uncertainty_table([cluster], [c1], scores, records)
        self.assertEqual(table[0].total_timestamps, 73)


class TestAgeNotHardcoded(unittest.TestCase):
    def test_older_more_dispersed_candidate_only_wider_if_actual_spread_wider(self):
        # older candidate (further back in time) but with IDENTICAL particle spread ->
        # uncertainty_radius_km must be equal, not automatically wider from age alone.
        near = _candidate("near", 34.0, 35.0, t_s=T0, n_particles=1, particle_prefix="near")
        near.peak_support_features.neighborhood.lat[:] = [34.01]
        near.peak_support_features.neighborhood.lon[:] = [35.0]
        far = _candidate("far", 34.0, 35.0, t_s=T0 - np.timedelta64(50, "h"), n_particles=1, particle_prefix="far")
        far.peak_support_features.neighborhood.lat[:] = [34.01]
        far.peak_support_features.neighborhood.lon[:] = [35.0]

        scores_near = [_score("near", t_s=T0)]
        scores_far = [_score("far", t_s=T0 - np.timedelta64(50, "h"))]
        cluster_near = _cluster_from([near], scores_near)
        cluster_far = _cluster_from([far], scores_far)
        records = _records_from([near, far])
        row_near = compute_cluster_uncertainty_table([cluster_near], [near], scores_near, records)[0]
        row_far = compute_cluster_uncertainty_table([cluster_far], [far], scores_far, records)[0]
        self.assertAlmostEqual(row_near.uncertainty_radius_km, row_far.uncertainty_radius_km, places=6)


if __name__ == "__main__":
    unittest.main()
