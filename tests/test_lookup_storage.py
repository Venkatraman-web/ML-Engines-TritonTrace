"""Focused T11 checks for `iste.lookup_storage` (memory-efficient lookup-table storage).

Run with:
    python -m unittest tests.test_lookup_storage -v
"""

from __future__ import annotations

import unittest

import numpy as np

from iste.ensemble_run import EnsembleRunRecords, run_ensemble
from iste.environment import GriddedFieldSource
from iste.forcing_ensemble import ForcingEnsemble, ForcingEnsembleConfig
from iste.lookup_storage import ParticleRecordStore, naive_list_of_dicts_nbytes
from iste.observation import adapt_upstream_particles

LAT = np.linspace(10.0, 30.0, 9)
LON = np.linspace(100.0, 120.0, 11)
T0 = np.datetime64("2020-01-01T00:00", "ns")
T1 = T0 + np.timedelta64(900, "s")
N_HOURS = 9
TIME = T0 + np.arange(N_HOURS) * np.timedelta64(3600, "s")
DT = 900.0

TEST_ENSEMBLE_CONFIG = ForcingEnsembleConfig(
    n_members=4, spatial_correlation_km=60.0, correlation_time_seconds=2 * 3600.0,
    sigma_u=0.05, coarse_resolution_deg=0.5, master_seed=123,
)


def make_ocean() -> GriddedFieldSource:
    uo = np.full((N_HOURS, len(LAT), len(LON)), 0.2)
    vo = np.full((N_HOURS, len(LAT), len(LON)), -0.1)
    return GriddedFieldSource(
        source_path="<synthetic>", time_name="time", time=TIME, lat_name="lat", lat=LAT,
        lon_name="lon", lon=LON, data={"uo": uo, "vo": vo}, fill_threshold=1e19,
    )


def two_synthetic_records() -> EnsembleRunRecords:
    return EnsembleRunRecords(
        particle_id=np.array(["a", "b"], dtype=object),
        forcing_member_id=np.array([0, 1], dtype=np.int64),
        observation_pixel_id=np.array(["px-0", "px-1"], dtype=object),
        timestamp=np.array([T0, T0], dtype="datetime64[ns]"),
        lat=np.array([10.0, 20.0]),
        lon=np.array([100.0, 110.0]),
        log_weight=np.array([0.0, np.log(2.0)]),
        valid=np.array([True, False]),
    )


class TestAddingRecords(unittest.TestCase):
    def test_add_records_batch(self):
        store = ParticleRecordStore()
        store.add_records(two_synthetic_records())
        self.assertEqual(len(store), 2)

    def test_add_record_singly(self):
        store = ParticleRecordStore()
        store.add_record(
            particle_id="p0", forcing_member_id=0, observation_pixel_id="px-0",
            timestamp=T0, lat=1.0, lon=2.0, log_weight=0.0, valid=True,
        )
        self.assertEqual(len(store), 1)
        all_recs = store.all_records()
        self.assertEqual(all_recs.particle_id[0], "p0")
        self.assertEqual(all_recs.lat[0], 1.0)

    def test_mixed_add_record_and_add_records(self):
        store = ParticleRecordStore()
        store.add_records(two_synthetic_records())
        store.add_record(
            particle_id="c", forcing_member_id=2, observation_pixel_id="px-2",
            timestamp=T1, lat=5.0, lon=5.0, log_weight=1.0, valid=True,
        )
        self.assertEqual(len(store), 3)


class TestCheckpointLookup(unittest.TestCase):
    def test_retrieve_by_checkpoint(self):
        store = ParticleRecordStore()
        store.add_records(two_synthetic_records())
        store.add_record(
            particle_id="c", forcing_member_id=2, observation_pixel_id="px-2",
            timestamp=T1, lat=5.0, lon=5.0, log_weight=1.0, valid=True,
        )
        at_t0 = store.records_at_checkpoint(T0)
        self.assertEqual(at_t0.n_records, 2)
        self.assertEqual(set(at_t0.particle_id.tolist()), {"a", "b"})

        at_t1 = store.records_at_checkpoint(T1)
        self.assertEqual(at_t1.n_records, 1)
        self.assertEqual(at_t1.particle_id[0], "c")

    def test_checkpoint_with_no_records_returns_empty(self):
        store = ParticleRecordStore()
        store.add_records(two_synthetic_records())
        missing = store.records_at_checkpoint(T0 + np.timedelta64(999, "h"))
        self.assertEqual(missing.n_records, 0)

    def test_checkpoints_lists_sorted_unique_timestamps(self):
        store = ParticleRecordStore()
        store.add_records(two_synthetic_records())
        store.add_record(
            particle_id="c", forcing_member_id=2, observation_pixel_id="px-2",
            timestamp=T1, lat=5.0, lon=5.0, log_weight=1.0, valid=True,
        )
        cps = store.checkpoints()
        np.testing.assert_array_equal(cps, np.array([T0, T1], dtype="datetime64[ns]"))


class TestPixelAndMemberLookup(unittest.TestCase):
    def test_retrieve_by_observation_pixel_id(self):
        store = ParticleRecordStore()
        store.add_records(two_synthetic_records())
        px0 = store.records_for_pixel("px-0")
        self.assertEqual(px0.n_records, 1)
        self.assertEqual(px0.particle_id[0], "a")

    def test_retrieve_by_forcing_member_id(self):
        store = ParticleRecordStore()
        store.add_records(two_synthetic_records())
        m1 = store.records_for_member(1)
        self.assertEqual(m1.n_records, 1)
        self.assertEqual(m1.particle_id[0], "b")

    def test_pixel_and_member_do_not_cross_contaminate(self):
        store = ParticleRecordStore()
        store.add_records(two_synthetic_records())
        # "a" is pixel px-0/member 0; "b" is pixel px-1/member 1 -- filtering by one
        # never returns rows belonging only to the other's group.
        self.assertNotIn("b", store.records_for_pixel("px-0").particle_id.tolist())
        self.assertNotIn("a", store.records_for_member(1).particle_id.tolist())

    def test_unknown_pixel_or_member_returns_empty(self):
        store = ParticleRecordStore()
        store.add_records(two_synthetic_records())
        self.assertEqual(store.records_for_pixel("does-not-exist").n_records, 0)
        self.assertEqual(store.records_for_member(999).n_records, 0)


class TestMixedValidity(unittest.TestCase):
    def test_invalid_records_are_stored_and_retrievable(self):
        store = ParticleRecordStore()
        store.add_records(two_synthetic_records())
        all_recs = store.all_records()
        self.assertEqual(int(np.sum(~all_recs.valid)), 1)
        invalid_row = all_recs.particle_id[~all_recs.valid][0]
        self.assertEqual(invalid_row, "b")
        # retrievable via every lookup path too, not just all_records().
        self.assertFalse(store.records_for_pixel("px-1").valid[0])
        self.assertFalse(store.records_at_checkpoint(T0).valid[
            store.records_at_checkpoint(T0).particle_id.tolist().index("b")
        ])


class TestFieldPreservationAndRoundTrip(unittest.TestCase):
    def test_all_fields_preserved_exactly(self):
        original = two_synthetic_records()
        store = ParticleRecordStore()
        store.add_records(original)
        result = store.all_records()

        np.testing.assert_array_equal(result.particle_id, original.particle_id)
        np.testing.assert_array_equal(result.forcing_member_id, original.forcing_member_id)
        np.testing.assert_array_equal(result.observation_pixel_id, original.observation_pixel_id)
        np.testing.assert_array_equal(result.timestamp, original.timestamp)
        np.testing.assert_array_equal(result.lat, original.lat)
        np.testing.assert_array_equal(result.lon, original.lon)
        np.testing.assert_array_equal(result.log_weight, original.log_weight)
        np.testing.assert_array_equal(result.valid, original.valid)


class TestEmptyStorage(unittest.TestCase):
    def test_len_zero(self):
        store = ParticleRecordStore()
        self.assertEqual(len(store), 0)

    def test_all_records_empty(self):
        store = ParticleRecordStore()
        recs = store.all_records()
        self.assertEqual(recs.n_records, 0)
        for arr in (
            recs.particle_id, recs.forcing_member_id, recs.observation_pixel_id,
            recs.timestamp, recs.lat, recs.lon, recs.log_weight, recs.valid,
        ):
            self.assertEqual(len(arr), 0)

    def test_checkpoints_empty(self):
        store = ParticleRecordStore()
        self.assertEqual(len(store.checkpoints()), 0)

    def test_lookups_on_empty_store_return_empty(self):
        store = ParticleRecordStore()
        self.assertEqual(store.records_at_checkpoint(T0).n_records, 0)
        self.assertEqual(store.records_for_pixel("px-0").n_records, 0)
        self.assertEqual(store.records_for_member(0).n_records, 0)


class TestDuplicateCheckpointHandling(unittest.TestCase):
    def test_readding_the_same_batch_appends_rather_than_deduplicates(self):
        # explicit, documented behavior (requirement 10): this store never deduplicates.
        store = ParticleRecordStore()
        records = two_synthetic_records()
        store.add_records(records)
        store.add_records(records)
        self.assertEqual(len(store), 4)
        at_t0 = store.records_at_checkpoint(T0)
        self.assertEqual(at_t0.n_records, 4)
        self.assertEqual(sorted(at_t0.particle_id.tolist()), ["a", "a", "b", "b"])

    def test_two_different_particles_at_the_same_checkpoint_both_retrievable(self):
        store = ParticleRecordStore()
        store.add_records(two_synthetic_records())  # both rows share timestamp T0
        at_t0 = store.records_at_checkpoint(T0)
        self.assertEqual(at_t0.n_records, 2)


class TestMemoryFootprint(unittest.TestCase):
    def test_compact_store_uses_less_memory_than_naive_list_of_dicts(self):
        ocean = make_ocean()
        particles = adapt_upstream_particles(
            [
                {"particle_id": f"p{i}", "latitude": 20.0, "longitude": 109.0 + 0.1 * i,
                 "observation_pixel_id": "px-0", "timestamp": T0}
                for i in range(5)
            ]
        )
        ensemble = ForcingEnsemble(ocean, TEST_ENSEMBLE_CONFIG)
        k_field = np.zeros((len(LAT), len(LON)))
        records = run_ensemble(particles, [0, 1], 6, DT, ocean, ensemble, LAT, LON, k_field, master_seed=1)

        store = ParticleRecordStore()
        store.add_records(records)

        compact_bytes = store.nbytes()
        naive_bytes = naive_list_of_dicts_nbytes(records)
        self.assertGreater(compact_bytes, 0)
        self.assertGreater(naive_bytes, 0)
        self.assertLess(compact_bytes, naive_bytes)


class TestT09IntegrationRoundTrip(unittest.TestCase):
    """Requirement 11: use T09's small-run output as the main integration test source."""

    def test_round_trip_preserves_everything(self):
        ocean = make_ocean()
        particles = adapt_upstream_particles(
            [
                {"particle_id": "p0", "latitude": 20.0, "longitude": 109.0, "observation_pixel_id": "px-0", "timestamp": T0},
                {"particle_id": "p1", "latitude": 20.0, "longitude": 110.0, "observation_pixel_id": "px-0", "timestamp": T0},
                {"particle_id": "p2", "latitude": 20.0, "longitude": 111.0, "observation_pixel_id": "px-1", "timestamp": T0, "valid": False},
            ]
        )
        ensemble = ForcingEnsemble(ocean, TEST_ENSEMBLE_CONFIG)
        k_field = np.zeros((len(LAT), len(LON)))
        records = run_ensemble(particles, [0, 1, 2], 5, DT, ocean, ensemble, LAT, LON, k_field, master_seed=7)

        store = ParticleRecordStore()
        store.add_records(records)
        result = store.all_records()

        # record count before storage == record count after round-trip
        self.assertEqual(result.n_records, records.n_records)
        # IDs and timestamps unchanged
        np.testing.assert_array_equal(result.particle_id, records.particle_id)
        np.testing.assert_array_equal(result.forcing_member_id, records.forcing_member_id)
        np.testing.assert_array_equal(result.observation_pixel_id, records.observation_pixel_id)
        np.testing.assert_array_equal(result.timestamp, records.timestamp)
        # positions and weights unchanged
        np.testing.assert_array_equal(result.lat, records.lat)
        np.testing.assert_array_equal(result.lon, records.lon)
        np.testing.assert_array_equal(result.log_weight, records.log_weight)
        # validity unchanged
        np.testing.assert_array_equal(result.valid, records.valid)

        # and the store's own lookups agree with a direct mask on the original records
        for member_id in (0, 1, 2):
            from_store = store.records_for_member(member_id)
            direct_mask = records.forcing_member_id == member_id
            self.assertEqual(from_store.n_records, int(np.sum(direct_mask)))


if __name__ == "__main__":
    unittest.main()
