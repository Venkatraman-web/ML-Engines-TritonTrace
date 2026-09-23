"""T11 -- memory-efficient lookup-table storage for Stage III particle records.

A compact, columnar (structure-of-arrays) append-only store for the per-checkpoint
`(particle, forcing_member, checkpoint)` records T09's `iste.ensemble_run.run_ensemble`
produces, plus small dict-based indices for lookup by checkpoint timestamp,
`observation_pixel_id`, and `forcing_member_id`.

Stores the eight original `README.md` Stage III lookup-table fields -- `particle_id`,
`forcing_member_id`, `observation_pixel_id`, `timestamp`, `lat`, `lon`, `log_weight`,
`valid` -- plus five fields added for `final_stage_III_IV_specification.txt`
compatibility (T13.5): `death_time`, `death_lat`, `death_lon`, `death_reason` (the exact
invalidating event for a `(particle_id, forcing_member_id)` group, repeated on every row
of that group, or the "never died" sentinel `NaT`/`NaN`/`""`), and `survival_mark`
(`int8`, `1`/`0`, mirroring `valid` at that row's own checkpoint -- an explicitly named
column, even though derivable from `valid`, matching the frozen contract's own schema).
This module never imports `iste.environment`/`iste.forcing_ensemble` and never touches an
ocean/wind/K field or a forcing-ensemble grid -- there is nothing here that could
duplicate them, by construction, not by discipline alone.

**Weight representation (requirement 8):** only `log_weight` is stored, matching T05/T06/
T09's own numerical-stability convention (`weight = exp(log_weight)`). The linear weight
is never separately stored; callers that need it compute `np.exp(log_weight)` on demand
(as `iste.diagnostics.compute_stage3_diagnostics_from_log_weights` already does).

**Storage representation:** while records are being appended, each of the eight fields is
held in its own plain Python list (cheap amortized append). Reading (any lookup, or
`nbytes()`) lazily builds one NumPy array per field plus the three lookup dicts, cached
until the next `add_records`/`add_record` call invalidates them. This keeps memory to
exactly one array per field (no per-record Python object/dict overhead once built) while
still supporting cheap incremental appends -- see `notebooks/T11_lookup_storage_check.ipynb`
for a direct memory comparison against a naive "list of one dict per record" approach.

**Checkpoint-timestamp indexing (requirement 6):** the timestamp index's keys are the
actual `int64` nanosecond values of whatever fixed checkpoint schedule T06's
`run_backward_trajectory` generated for each particle (`t0, t0 - dt, t0 - 2*dt, ...`,
independent of validity, per T06's own design) -- this store does not invent or assume any
particular schedule; it indexes whatever timestamps its input records actually carry.

**Duplicate/repeated-checkpoint handling (requirement 10):** this store is deliberately
append-only and never deduplicates. Calling `add_records` twice with the same batch simply
appends two copies of every row (both become retrievable). This is intentional and
explicit, not a bug: the store has no way to know whether a second call is a legitimate
new record at the same timestamp (e.g. a different particle) or an accidental re-add, so it
does not silently guess -- callers are responsible for not re-adding a batch they already
added.

Query methods return `iste.ensemble_run.EnsembleRunRecords` (T09's own record type), so
this module reuses T09's structure instead of inventing a parallel one, and a round trip
through the store (`add_records` then `all_records()`) yields the same type as went in.
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np

from iste.ensemble_run import EnsembleRunRecords

REQUIRED_FIELDS = (
    "particle_id", "forcing_member_id", "observation_pixel_id",
    "timestamp", "lat", "lon", "log_weight", "valid",
)

_EMPTY_RECORDS_KWARGS = dict(
    particle_id=np.array([], dtype=object),
    forcing_member_id=np.array([], dtype=np.int64),
    observation_pixel_id=np.array([], dtype=object),
    timestamp=np.array([], dtype="datetime64[ns]"),
    lat=np.array([], dtype=np.float64),
    lon=np.array([], dtype=np.float64),
    log_weight=np.array([], dtype=np.float64),
    valid=np.array([], dtype=bool),
    death_time=np.array([], dtype="datetime64[ns]"),
    death_lat=np.array([], dtype=np.float64),
    death_lon=np.array([], dtype=np.float64),
    death_reason=np.array([], dtype=object),
    survival_mark=np.array([], dtype=np.int8),
)


class ParticleRecordStore:
    """Compact append-only store + index for Stage III per-checkpoint particle records.

    Invalid particles are stored and retrievable exactly like valid ones -- `valid` is
    just another indexed/returned field, never a filter applied implicitly by this class.
    """

    def __init__(self) -> None:
        self._particle_id: list = []
        self._forcing_member_id: list = []
        self._observation_pixel_id: list = []
        self._timestamp: list = []
        self._lat: list = []
        self._lon: list = []
        self._log_weight: list = []
        self._valid: list = []
        self._death_time: list = []
        self._death_lat: list = []
        self._death_lon: list = []
        self._death_reason: list = []
        self._survival_mark: list = []

        self._arrays_cache: dict[str, np.ndarray] | None = None
        self._timestamp_index: dict[int, list] | None = None
        self._pixel_index: dict[Any, list] | None = None
        self._member_index: dict[int, list] | None = None

    def __len__(self) -> int:
        return len(self._timestamp)

    # -- writing -----------------------------------------------------------------

    def add_records(self, records: EnsembleRunRecords) -> None:
        """Append every row of a T09 `EnsembleRunRecords` batch (or of another store's
        query result, since both are the same type). Does not run any trajectory/ensemble
        logic itself -- purely appends what it is given.

        The five T13.5 death/survival fields are optional on the incoming batch (`None`
        means "not tracked," e.g. a hand-built `EnsembleRunRecords` predating T13.5) --
        when `None`, this store substitutes length-matched "not tracked" defaults
        (`NaT`/`NaN`/`""`/`survival_mark` derived from `valid`) so its own internal
        columns are always real arrays, never `None`."""
        n = records.n_records
        self._particle_id.extend(records.particle_id.tolist())
        self._forcing_member_id.extend(int(x) for x in records.forcing_member_id)
        self._observation_pixel_id.extend(records.observation_pixel_id.tolist())
        self._timestamp.extend(records.timestamp.tolist())
        self._lat.extend(float(x) for x in records.lat)
        self._lon.extend(float(x) for x in records.lon)
        self._log_weight.extend(float(x) for x in records.log_weight)
        self._valid.extend(bool(x) for x in records.valid)

        death_time = records.death_time if records.death_time is not None else np.full(n, np.datetime64("NaT", "ns"))
        death_lat = records.death_lat if records.death_lat is not None else np.full(n, np.nan)
        death_lon = records.death_lon if records.death_lon is not None else np.full(n, np.nan)
        death_reason = records.death_reason if records.death_reason is not None else np.array([""] * n, dtype=object)
        survival_mark = (
            records.survival_mark if records.survival_mark is not None else records.valid.astype(np.int8)
        )
        self._death_time.extend(np.asarray(death_time, dtype="datetime64[ns]").tolist())
        self._death_lat.extend(float(x) for x in death_lat)
        self._death_lon.extend(float(x) for x in death_lon)
        self._death_reason.extend(death_reason.tolist() if hasattr(death_reason, "tolist") else list(death_reason))
        self._survival_mark.extend(int(x) for x in survival_mark)
        self._invalidate_cache()

    def add_record(
        self, *, particle_id, forcing_member_id: int, observation_pixel_id,
        timestamp, lat: float, lon: float, log_weight: float, valid: bool,
        death_time=None, death_lat: float = float("nan"), death_lon: float = float("nan"),
        death_reason: str = "", survival_mark: int | None = None,
    ) -> None:
        """Append a single record -- a convenience for tests and small synthetic
        examples; `add_records` is the normal path for T09 output. The T13.5 death/
        survival fields all default to "not dead" (`death_time=None` -> `NaT`,
        `death_lat`/`death_lon`=`NaN`, `death_reason`=`""`); `survival_mark` defaults to
        mirroring `valid` (`1` if valid else `0`) when not given explicitly."""
        self._particle_id.append(particle_id)
        self._forcing_member_id.append(int(forcing_member_id))
        self._observation_pixel_id.append(observation_pixel_id)
        self._timestamp.append(np.datetime64(timestamp, "ns"))
        self._lat.append(float(lat))
        self._lon.append(float(lon))
        self._log_weight.append(float(log_weight))
        self._valid.append(bool(valid))
        self._death_time.append(np.datetime64(death_time, "ns") if death_time is not None else np.datetime64("NaT", "ns"))
        self._death_lat.append(float(death_lat))
        self._death_lon.append(float(death_lon))
        self._death_reason.append(death_reason)
        self._survival_mark.append(int(survival_mark) if survival_mark is not None else (1 if valid else 0))
        self._invalidate_cache()

    def _invalidate_cache(self) -> None:
        self._arrays_cache = None
        self._timestamp_index = None
        self._pixel_index = None
        self._member_index = None

    # -- internal array / index construction --------------------------------------

    def _arrays(self) -> dict[str, np.ndarray]:
        if self._arrays_cache is None:
            self._arrays_cache = {
                "particle_id": np.array(self._particle_id, dtype=object),
                "forcing_member_id": np.array(self._forcing_member_id, dtype=np.int64),
                "observation_pixel_id": np.array(self._observation_pixel_id, dtype=object),
                "timestamp": np.array(self._timestamp, dtype="datetime64[ns]"),
                "lat": np.array(self._lat, dtype=np.float64),
                "lon": np.array(self._lon, dtype=np.float64),
                "log_weight": np.array(self._log_weight, dtype=np.float64),
                "valid": np.array(self._valid, dtype=bool),
                "death_time": np.array(self._death_time, dtype="datetime64[ns]"),
                "death_lat": np.array(self._death_lat, dtype=np.float64),
                "death_lon": np.array(self._death_lon, dtype=np.float64),
                "death_reason": np.array(self._death_reason, dtype=object),
                "survival_mark": np.array(self._survival_mark, dtype=np.int8),
            }
        return self._arrays_cache

    def _timestamp_idx(self) -> dict:
        if self._timestamp_index is None:
            ts_ns = self._arrays()["timestamp"].astype("int64")
            index: dict = {}
            for i, t in enumerate(ts_ns.tolist()):
                index.setdefault(t, []).append(i)
            self._timestamp_index = index
        return self._timestamp_index

    def _pixel_idx(self) -> dict:
        if self._pixel_index is None:
            index: dict = {}
            for i, p in enumerate(self._arrays()["observation_pixel_id"].tolist()):
                index.setdefault(p, []).append(i)
            self._pixel_index = index
        return self._pixel_index

    def _member_idx(self) -> dict:
        if self._member_index is None:
            index: dict = {}
            for i, m in enumerate(self._arrays()["forcing_member_id"].tolist()):
                index.setdefault(m, []).append(i)
            self._member_index = index
        return self._member_index

    def _select(self, row_indices) -> EnsembleRunRecords:
        arrays = self._arrays()
        idx = np.asarray(row_indices, dtype=np.int64)
        return EnsembleRunRecords(
            particle_id=arrays["particle_id"][idx],
            forcing_member_id=arrays["forcing_member_id"][idx],
            observation_pixel_id=arrays["observation_pixel_id"][idx],
            timestamp=arrays["timestamp"][idx],
            lat=arrays["lat"][idx],
            lon=arrays["lon"][idx],
            log_weight=arrays["log_weight"][idx],
            valid=arrays["valid"][idx],
            death_time=arrays["death_time"][idx],
            death_lat=arrays["death_lat"][idx],
            death_lon=arrays["death_lon"][idx],
            death_reason=arrays["death_reason"][idx],
            survival_mark=arrays["survival_mark"][idx],
        )

    # -- reading -------------------------------------------------------------------

    def all_records(self) -> EnsembleRunRecords:
        """Every stored record, in insertion order."""
        if len(self) == 0:
            return EnsembleRunRecords(**_EMPTY_RECORDS_KWARGS)
        return self._select(np.arange(len(self)))

    def checkpoints(self) -> np.ndarray:
        """Sorted, unique checkpoint timestamps present in the store."""
        return np.unique(self._arrays()["timestamp"])

    def records_at_checkpoint(self, timestamp) -> EnsembleRunRecords:
        """All records (any particle, any forcing member, valid or invalid) whose
        `timestamp` exactly equals `timestamp`."""
        key = int(np.datetime64(timestamp, "ns").astype("int64"))
        return self._select(self._timestamp_idx().get(key, []))

    def records_for_pixel(self, observation_pixel_id) -> EnsembleRunRecords:
        """All records (any checkpoint, any forcing member, valid or invalid) tagged with
        this `observation_pixel_id`."""
        return self._select(self._pixel_idx().get(observation_pixel_id, []))

    def records_for_member(self, forcing_member_id: int) -> EnsembleRunRecords:
        """All records (any checkpoint, any pixel, valid or invalid) tagged with this
        `forcing_member_id`."""
        return self._select(self._member_idx().get(int(forcing_member_id), []))

    # -- memory accounting -----------------------------------------------------------

    def nbytes(self) -> int:
        """Approximate memory footprint of the stored columns, in bytes. For the two
        object-dtype (string) columns this includes each distinct string's own size (via
        `sys.getsizeof`), not just the 8-byte pointer NumPy's object array stores per
        element -- a fairer accounting than `ndarray.nbytes` alone for the naive-vs-compact
        comparison in the T11 notebook. This never includes anything about environmental
        fields, since this module holds none."""
        arrays = self._arrays()
        numeric_columns = (
            "forcing_member_id", "timestamp", "lat", "lon", "log_weight", "valid",
            "death_time", "death_lat", "death_lon", "survival_mark",
        )
        total = sum(arrays[name].nbytes for name in numeric_columns)
        for name in ("particle_id", "observation_pixel_id", "death_reason"):
            total += arrays[name].nbytes
            total += sum(sys.getsizeof(v) for v in arrays[name])
        return total


def naive_list_of_dicts_nbytes(records: EnsembleRunRecords) -> int:
    """A rough, explicitly approximate estimate (via `sys.getsizeof`, recursively over
    each dict/its values) of the memory a naive "one Python dict per record, all in a
    list" representation would use for the same data `records` holds -- used only by the
    T11 benchmark (requirement 13) to compare against `ParticleRecordStore.nbytes()`.
    `ParticleRecordStore` itself never represents records this way."""
    total = sys.getsizeof([])
    has_death_fields = records.death_time is not None
    for i in range(records.n_records):
        row = {
            "particle_id": str(records.particle_id[i]),
            "forcing_member_id": int(records.forcing_member_id[i]),
            "observation_pixel_id": str(records.observation_pixel_id[i]),
            "timestamp": records.timestamp[i].item(),
            "lat": float(records.lat[i]),
            "lon": float(records.lon[i]),
            "log_weight": float(records.log_weight[i]),
            "valid": bool(records.valid[i]),
        }
        if has_death_fields:
            row["death_time"] = records.death_time[i].item()
            row["death_lat"] = float(records.death_lat[i])
            row["death_lon"] = float(records.death_lon[i])
            row["death_reason"] = str(records.death_reason[i])
            row["survival_mark"] = int(records.survival_mark[i])
        total += sys.getsizeof(row)
        for v in row.values():
            total += sys.getsizeof(v)
    return total
