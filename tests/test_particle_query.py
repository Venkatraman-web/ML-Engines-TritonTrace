"""Focused T14 checks for `iste.particle_query` (Stage III -> Phase 2 historical
particle slice / query service).

Run with:
    python -m unittest tests.test_particle_query -v
"""

from __future__ import annotations

import unittest

import numpy as np

from iste.ensemble_run import EnsembleRunRecords
from iste.particle_query import HistoricalParticleSlice, query_particle_slice

T0 = np.datetime64("2020-01-01T01:00:00", "ns")


def _mins(m: int) -> np.datetime64:
    return T0 - np.timedelta64(m, "m")


def _add_group(cols: dict, pid, pixel_id, member_id, ts, lat, lon, lw, valid, death_time, death_lat, death_lon, death_reason):
    n = len(ts)
    cols["particle_id"].extend([pid] * n)
    cols["observation_pixel_id"].extend([pixel_id] * n)
    cols["forcing_member_id"].extend([member_id] * n)
    cols["timestamp"].append(np.asarray(ts, dtype="datetime64[ns]"))
    cols["lat"].append(np.asarray(lat, dtype=np.float64))
    cols["lon"].append(np.asarray(lon, dtype=np.float64))
    cols["log_weight"].append(np.asarray(lw, dtype=np.float64))
    cols["valid"].append(np.asarray(valid, dtype=bool))
    cols["death_time"].append(np.full(n, death_time, dtype="datetime64[ns]"))
    cols["death_lat"].append(np.full(n, death_lat, dtype=np.float64))
    cols["death_lon"].append(np.full(n, death_lon, dtype=np.float64))
    cols["death_reason"].extend([death_reason] * n)


def _build_records() -> EnsembleRunRecords:
    """Synthetic checkpoint data: two observation pixels, two forcing members.

    - p0 (px-0, member 0): stays valid throughout; checkpoints at 0/10/20/30 min back.
    - p1 (px-0, member 1): dies exactly at the 20-min checkpoint (domain_exit), frozen
      from there on; death position deliberately differs from the frozen lat/lon.
    - p2 (px-1, member 0): stays valid throughout, distinct position range.
    - p3 (px-1, member 1): a T07 upstream-already-invalid particle -- a single row at t0
      only, never propagated (matches iste.ensemble_run's own "already_invalid_upstream"
      convention).
    """
    cols: dict = {
        "particle_id": [], "observation_pixel_id": [], "forcing_member_id": [],
        "timestamp": [], "lat": [], "lon": [], "log_weight": [], "valid": [],
        "death_time": [], "death_lat": [], "death_lon": [], "death_reason": [],
    }
    ts4 = [_mins(0), _mins(10), _mins(20), _mins(30)]
    nat = np.datetime64("NaT", "ns")

    _add_group(
        cols, "p0", "px-0", 0, ts4,
        lat=[20.0, 20.1, 20.2, 20.3], lon=[100.0, 100.1, 100.2, 100.3], lw=[0.0, 0.1, 0.2, 0.3],
        valid=[True, True, True, True], death_time=nat, death_lat=np.nan, death_lon=np.nan, death_reason="",
    )
    _add_group(
        cols, "p1", "px-0", 1, ts4,
        lat=[22.0, 22.1, 22.1, 22.1], lon=[102.0, 102.1, 102.1, 102.1], lw=[0.0, 0.1, 0.1, 0.1],
        valid=[True, True, False, False], death_time=_mins(20), death_lat=21.5, death_lon=101.5,
        death_reason="domain_exit",
    )
    _add_group(
        cols, "p2", "px-1", 0, ts4,
        lat=[30.0, 30.2, 30.4, 30.6], lon=[110.0, 110.2, 110.4, 110.6], lw=[0.0, -0.1, -0.2, -0.3],
        valid=[True, True, True, True], death_time=nat, death_lat=np.nan, death_lon=np.nan, death_reason="",
    )
    _add_group(
        cols, "p3", "px-1", 1, [T0],
        lat=[25.0], lon=[105.0], lw=[0.0], valid=[False],
        death_time=T0, death_lat=25.0, death_lon=105.0, death_reason="already_invalid_upstream",
    )

    n_total = sum(len(t) for t in cols["timestamp"])
    valid_concat = np.concatenate(cols["valid"])
    return EnsembleRunRecords(
        particle_id=np.array(cols["particle_id"], dtype=object),
        forcing_member_id=np.array(cols["forcing_member_id"], dtype=np.int64),
        observation_pixel_id=np.array(cols["observation_pixel_id"], dtype=object),
        timestamp=np.concatenate(cols["timestamp"]),
        lat=np.concatenate(cols["lat"]),
        lon=np.concatenate(cols["lon"]),
        log_weight=np.concatenate(cols["log_weight"]),
        valid=valid_concat,
        death_time=np.concatenate(cols["death_time"]),
        death_lat=np.concatenate(cols["death_lat"]),
        death_lon=np.concatenate(cols["death_lon"]),
        death_reason=np.array(cols["death_reason"], dtype=object),
        survival_mark=valid_concat.astype(np.int8),
    )


def _row(slice_: HistoricalParticleSlice, particle_id: str) -> int:
    (idx,) = np.where(slice_.particle_id == particle_id)
    return int(idx[0])


class TestInterpolationBetweenCheckpoints(unittest.TestCase):
    def test_midpoint_between_two_checkpoints(self):
        records = _build_records()
        slice_ = query_particle_slice(records, _mins(5))  # halfway between 0 and 10 min
        i = _row(slice_, "p0")
        self.assertAlmostEqual(slice_.lat[i], 20.05)
        self.assertAlmostEqual(slice_.lon[i], 100.05)
        self.assertAlmostEqual(slice_.log_weight[i], 0.05)
        self.assertTrue(slice_.valid[i])

    def test_non_midpoint_fraction(self):
        records = _build_records()
        slice_ = query_particle_slice(records, _mins(24))  # 4/10 of the way from 20 to 30
        i = _row(slice_, "p0")
        self.assertAlmostEqual(slice_.lat[i], 0.6 * 20.2 + 0.4 * 20.3)
        self.assertAlmostEqual(slice_.log_weight[i], 0.6 * 0.2 + 0.4 * 0.3)


class TestExactCheckpointQuery(unittest.TestCase):
    def test_query_exactly_on_a_checkpoint_returns_exact_stored_values(self):
        records = _build_records()
        slice_ = query_particle_slice(records, _mins(10))
        i = _row(slice_, "p0")
        self.assertEqual(slice_.lat[i], 20.1)
        self.assertEqual(slice_.lon[i], 100.1)
        self.assertEqual(slice_.log_weight[i], 0.1)

    def test_query_exactly_at_t0(self):
        records = _build_records()
        slice_ = query_particle_slice(records, _mins(0))
        i = _row(slice_, "p0")
        self.assertEqual(slice_.lat[i], 20.0)
        self.assertEqual(slice_.log_weight[i], 0.0)


class TestDeathTimeExclusion(unittest.TestCase):
    def test_alive_strictly_before_death_time(self):
        # 15 min back is strictly between the 10-min (alive) and 20-min (first dead)
        # checkpoints; since death_time == 20 min back, 15 min back is still alive.
        records = _build_records()
        slice_ = query_particle_slice(records, _mins(15))
        i = _row(slice_, "p1")
        self.assertTrue(slice_.valid[i])
        # both bracketing checkpoints are frozen at the same position -> interpolation
        # correctly reproduces the last-valid position regardless of the fraction.
        self.assertAlmostEqual(slice_.lat[i], 22.1)
        self.assertAlmostEqual(slice_.lon[i], 102.1)

    def test_dead_exactly_at_death_time(self):
        records = _build_records()
        slice_ = query_particle_slice(records, _mins(20))
        i = _row(slice_, "p1")
        self.assertFalse(slice_.valid[i])
        # still present in the result (not silently dropped), frozen position retained.
        self.assertAlmostEqual(slice_.lat[i], 22.1)

    def test_dead_after_death_time(self):
        records = _build_records()
        slice_ = query_particle_slice(records, _mins(25))
        i = _row(slice_, "p1")
        self.assertFalse(slice_.valid[i])
        self.assertAlmostEqual(slice_.lat[i], 22.1)  # still frozen, not extrapolated

    def test_never_died_particle_always_valid(self):
        records = _build_records()
        for minutes in (0, 5, 10, 20, 30):
            slice_ = query_particle_slice(records, _mins(minutes))
            i = _row(slice_, "p0")
            self.assertTrue(slice_.valid[i])

    def test_upstream_invalid_particle_excluded_from_t0_onward(self):
        records = _build_records()
        for minutes in (0, 5, 30):
            slice_ = query_particle_slice(records, _mins(minutes))
            i = _row(slice_, "p3")
            self.assertFalse(slice_.valid[i])
            self.assertEqual(slice_.lat[i], 25.0)
            self.assertEqual(slice_.lon[i], 105.0)


class TestMetadataPreservation(unittest.TestCase):
    def test_identifiers_and_death_metadata_preserved_exactly(self):
        records = _build_records()
        slice_ = query_particle_slice(records, _mins(25))
        i1 = _row(slice_, "p1")
        self.assertEqual(slice_.observation_pixel_id[i1], "px-0")
        self.assertEqual(slice_.forcing_member_id[i1], 1)
        self.assertEqual(slice_.death_time[i1], _mins(20))
        self.assertAlmostEqual(slice_.death_lat[i1], 21.5)
        self.assertAlmostEqual(slice_.death_lon[i1], 101.5)
        self.assertEqual(slice_.death_reason[i1], "domain_exit")

    def test_never_died_particle_has_nat_death_metadata(self):
        records = _build_records()
        slice_ = query_particle_slice(records, _mins(5))
        i0 = _row(slice_, "p0")
        self.assertTrue(np.isnat(slice_.death_time[i0]))
        self.assertTrue(np.isnan(slice_.death_lat[i0]))
        self.assertEqual(slice_.death_reason[i0], "")

    def test_death_metadata_carried_even_while_still_alive(self):
        # p1 is ALIVE at 15 min (before its death at 20 min), but its eventual death
        # metadata must still be visible in the result -- a fact about the whole
        # trajectory, not conditioned on the current query time.
        records = _build_records()
        slice_ = query_particle_slice(records, _mins(15))
        i1 = _row(slice_, "p1")
        self.assertTrue(slice_.valid[i1])
        self.assertEqual(slice_.death_time[i1], _mins(20))
        self.assertEqual(slice_.death_reason[i1], "domain_exit")

    def test_query_time_recorded_on_result(self):
        records = _build_records()
        slice_ = query_particle_slice(records, _mins(15))
        self.assertEqual(slice_.query_time, _mins(15))


class TestMultiplePixelsAndForcingMembers(unittest.TestCase):
    def test_all_four_groups_present_with_correct_associations(self):
        records = _build_records()
        slice_ = query_particle_slice(records, _mins(5))
        self.assertEqual(slice_.n_particles, 4)

        pairs = {
            (str(pid), str(pix), int(mem))
            for pid, pix, mem in zip(slice_.particle_id, slice_.observation_pixel_id, slice_.forcing_member_id)
        }
        self.assertEqual(
            pairs,
            {("p0", "px-0", 0), ("p1", "px-0", 1), ("p2", "px-1", 0), ("p3", "px-1", 1)},
        )

    def test_pixel_and_member_never_cross_contaminate(self):
        records = _build_records()
        slice_ = query_particle_slice(records, _mins(5))
        i0 = _row(slice_, "p0")
        i2 = _row(slice_, "p2")
        self.assertNotEqual(slice_.observation_pixel_id[i0], slice_.observation_pixel_id[i2])
        self.assertAlmostEqual(slice_.lat[i2], 30.1)  # p2's own interpolated value, not p0's


class TestOutOfWindowQueries(unittest.TestCase):
    def test_query_beyond_stored_window_excludes_multi_checkpoint_groups(self):
        # beyond the 30-min stored window for p0/p1/p2's own bracketed checkpoints --
        # those three groups have no data to interpolate from and must be excluded. p3
        # (a single-row, upstream-already-invalid particle) has no upper backward bound
        # by design (see TestDeathTimeExclusion.test_upstream_invalid_particle_excluded_
        # from_t0_onward) and correctly still appears, as invalid.
        records = _build_records()
        slice_ = query_particle_slice(records, _mins(45))
        self.assertEqual(set(slice_.particle_id.tolist()), {"p3"})
        self.assertFalse(slice_.valid[_row(slice_, "p3")])

    def test_query_after_t0_excludes_the_single_row_group(self):
        records = _build_records()
        slice_ = query_particle_slice(records, T0 + np.timedelta64(5, "m"))
        self.assertNotIn("p3", slice_.particle_id.tolist())


class TestReusableAndDoesNotRerunStage3(unittest.TestCase):
    def test_no_stage3_module_imported(self):
        import iste.particle_query as pq
        with open(pq.__file__) as f:
            source = f.read()
        for forbidden in ("iste.step", "iste.trajectory", "iste.forcing_ensemble", "iste.environment"):
            self.assertNotIn(f"from {forbidden}", source)
            self.assertNotIn(f"import {forbidden}", source)

    def test_callable_multiple_times_with_different_query_times_no_side_effects(self):
        records = _build_records()
        slice_a = query_particle_slice(records, _mins(5))
        slice_b = query_particle_slice(records, _mins(25))
        # re-querying at the original time gives the identical result -- no mutation of
        # the input records or hidden state across calls.
        slice_a2 = query_particle_slice(records, _mins(5))
        np.testing.assert_array_equal(slice_a.lat, slice_a2.lat)
        np.testing.assert_array_equal(slice_a.valid, slice_a2.valid)


class TestEmptyRecords(unittest.TestCase):
    def test_empty_records_gives_empty_slice(self):
        from iste.lookup_storage import _EMPTY_RECORDS_KWARGS
        empty = EnsembleRunRecords(**_EMPTY_RECORDS_KWARGS)
        slice_ = query_particle_slice(empty, T0)
        self.assertEqual(slice_.n_particles, 0)


if __name__ == "__main__":
    unittest.main()
