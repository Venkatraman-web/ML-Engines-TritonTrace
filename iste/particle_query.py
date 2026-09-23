"""T14 -- Stage III -> Phase 2 historical particle slice / query service.

Implements exactly `README.md` Sec. 10's particle-level part of the Stage III -> Phase 2
query service (steps 1-4 of 7; the kernel evaluation and `c_hat*` summation, steps 5-7,
are explicitly NOT this task's scope -- they belong to a later Phase 2 task that will
consume this module's output):

    1. locate the bracketing Stage III checkpoints;
    2. linearly interpolate particle positions;
    3. linearly interpolate particle log-weights;
    4. exclude particles at/after their exact death time.

This module is a **query layer over already-computed Stage III output** -- it never runs
or re-runs backward transport (`iste.step`/`iste.trajectory`/`iste.ensemble_run` are not
imported here), and it never interpolates an already-aggregated `c_hat*` field (there is
no `c_hat*`/kernel logic in this module at all). It consumes an `iste.ensemble_run.
EnsembleRunRecords` batch (the same type T09 produces and T11's `ParticleRecordStore`
round-trips) and returns, for one requested historical timestamp `t_s`, one row per
`(particle_id, forcing_member_id)` group that has a defined state at `t_s`.

**Bracketing/interpolation reuses T02's own `iste.interpolation.time_bracket` directly**
(no second bracketing/interpolation implementation) -- it already handles a strictly
monotonic time axis in either direction, and a Stage III backward trajectory's checkpoint
timestamps (`t0, t0-dt, t0-2dt, ...`) are exactly that: strictly monotonic descending.

**Death-time exclusion, per `README.md` Sec. 7/10 and T13.5's death metadata:** a
particle's position/log-weight are frozen once it becomes invalid (T04/T05's own
"stop propagation, retain last valid state" policy, unchanged), so interpolating between
two post-death checkpoints (which are frozen and therefore identical) already gives the
correct frozen answer with no special-casing needed. Whether the particle actually
*contributes* at `t_s` is a SEPARATE, final decision: a particle is excluded (`valid =
False` in the result) iff `t_s` is at or after its exact `death_time` (using the same
"forward time decreases as backward time increases" comparison established in T13.5 --
`t_s <= death_time` means the backward integration has reached or passed the death
instant). A particle that dies LATER within the window (at a smaller `t_s`, i.e. further
back) but is still alive at the CURRENT query `t_s` is returned with `valid = True`; its
(future) death metadata is still carried through, since it is a fact about the whole
trajectory, not about this one query.

**Excluded (dead-at-query-time) particles are still returned, with `valid = False`, not
silently dropped** -- consistent with every other Stage III component's "never silently
drop invalid particles" discipline. A `(particle_id, forcing_member_id)` group is only
OMITTED from the result entirely when `t_s` falls outside that group's own stored
checkpoint range (i.e. there is no data to interpolate from at all, not even a frozen
dead state) -- this is different from, and never confused with, death-time exclusion.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from iste.ensemble_run import EnsembleRunRecords
from iste.interpolation import time_bracket


@dataclass
class HistoricalParticleSlice:
    """The Stage III particle state at one requested historical timestamp `query_time`,
    one row per `(particle_id, forcing_member_id)` group whose own stored checkpoint
    range covers `query_time`. All arrays have the same length (`n_particles`).

    - `particle_id`, `observation_pixel_id`, `forcing_member_id`: preserved unchanged.
    - `lat`, `lon`: linearly interpolated particle position at `query_time` (exact,
      un-interpolated, when `query_time` matches a stored checkpoint exactly).
    - `log_weight`: linearly interpolated particle log-weight at `query_time`.
    - `valid`: `True` iff the particle has not yet died by `query_time` (i.e.
      `query_time > death_time`, or it never dies within the stored window at all).
      `False` iff `query_time` is at or after its exact `death_time` -- such rows ARE
      still included (never silently dropped), with `lat`/`lon`/`log_weight` equal to
      the particle's frozen last-valid state (which interpolation naturally reproduces,
      since post-death checkpoints are already frozen identical).
    - `death_time`/`death_lat`/`death_lon`/`death_reason`: the particle's own
      whole-trajectory death fact (T13.5), carried through unchanged regardless of
      whether it has happened yet relative to `query_time` -- `NaT`/`NaN`/`""` if the
      particle never dies within its stored window.
    - `query_time`: the requested `t_s`, repeated here for convenience/provenance.
    """

    particle_id: np.ndarray
    observation_pixel_id: np.ndarray
    forcing_member_id: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    log_weight: np.ndarray
    valid: np.ndarray
    death_time: np.ndarray
    death_lat: np.ndarray
    death_lon: np.ndarray
    death_reason: np.ndarray
    query_time: np.datetime64

    @property
    def n_particles(self) -> int:
        return len(self.particle_id)


def query_particle_slice(records: EnsembleRunRecords, query_time) -> HistoricalParticleSlice:
    """Return the complete `J x E` Stage III particle slice at historical timestamp
    `query_time`, per `README.md` Sec. 10 steps 1-4 (bracket, interpolate position,
    interpolate log-weight, exclude at/after exact death time). Does not rerun Stage III
    transport and does not evaluate any kernel or `c_hat*` -- that is a later task's
    responsibility, consuming this function's output.

    Parameters
    ----------
    records : EnsembleRunRecords
        Already-computed Stage III output (from `iste.ensemble_run.run_ensemble` directly,
        or from `iste.lookup_storage.ParticleRecordStore.all_records()`/
        `records_for_pixel()`/`records_for_member()`). Must include the death-metadata
        fields (T13.5) for death-time exclusion to be meaningful; a batch with
        `death_time is None` (predating T13.5) is treated as "no particle ever dies."
    query_time : np.datetime64-convertible
        The historical timestamp `t_s` to query, inside the Stage III backward window.

    Returns
    -------
    HistoricalParticleSlice
    """
    query_time = np.datetime64(query_time, "ns")

    if records.n_records == 0:
        return _empty_slice(query_time)

    # group row indices by (particle_id, forcing_member_id) -- O(n), no sorting/hashing
    # surprises with object-dtype particle_id (same technique as
    # iste.ensemble_run._final_row_mask).
    groups: dict[tuple, list[int]] = {}
    for i in range(records.n_records):
        key = (records.particle_id[i], int(records.forcing_member_id[i]))
        groups.setdefault(key, []).append(i)

    out_particle_id: list = []
    out_pixel_id: list = []
    out_member_id: list = []
    out_lat: list = []
    out_lon: list = []
    out_log_weight: list = []
    out_valid: list = []
    out_death_time: list = []
    out_death_lat: list = []
    out_death_lon: list = []
    out_death_reason: list = []

    for (pid, member_id), row_indices in groups.items():
        idx = np.asarray(row_indices, dtype=np.int64)
        # defensive: sort each group's own rows by timestamp descending (backward-time
        # checkpoint order), not assuming storage/insertion order.
        order = np.argsort(-records.timestamp[idx].astype("int64"))
        idx = idx[order]
        timestamps = records.timestamp[idx]

        if len(idx) == 1:
            # a single stored instant (e.g. a T07 upstream-already-invalid particle,
            # which is never propagated -- see iste.ensemble_run's "already_invalid_
            # upstream" branch): no bracket is possible. It has a defined (frozen) state
            # for any query at or before that single instant; nothing after it.
            single_t = timestamps[0]
            if query_time > single_t:
                continue
            lat = float(records.lat[idx[0]])
            lon = float(records.lon[idx[0]])
            log_weight = float(records.log_weight[idx[0]])
        else:
            idx_lo, idx_hi, frac, in_range = time_bracket(timestamps, query_time)
            if not in_range:
                continue  # this particle's own stored window does not cover query_time
            lat = (1 - frac) * records.lat[idx[idx_lo]] + frac * records.lat[idx[idx_hi]]
            lon = (1 - frac) * records.lon[idx[idx_lo]] + frac * records.lon[idx[idx_hi]]
            log_weight = (
                (1 - frac) * records.log_weight[idx[idx_lo]] + frac * records.log_weight[idx[idx_hi]]
            )

        first = idx[0]
        pixel_id = records.observation_pixel_id[first]
        if records.death_time is not None:
            death_time = records.death_time[first]
            death_lat = float(records.death_lat[first])
            death_lon = float(records.death_lon[first])
            death_reason = str(records.death_reason[first])
        else:
            death_time = np.datetime64("NaT", "ns")
            death_lat = float("nan")
            death_lon = float("nan")
            death_reason = ""

        has_died = not np.isnat(death_time)
        valid_at_query = not (has_died and query_time <= death_time)

        out_particle_id.append(pid)
        out_pixel_id.append(pixel_id)
        out_member_id.append(member_id)
        out_lat.append(float(lat))
        out_lon.append(float(lon))
        out_log_weight.append(float(log_weight))
        out_valid.append(bool(valid_at_query))
        out_death_time.append(death_time)
        out_death_lat.append(death_lat)
        out_death_lon.append(death_lon)
        out_death_reason.append(death_reason)

    return HistoricalParticleSlice(
        particle_id=np.array(out_particle_id, dtype=object),
        observation_pixel_id=np.array(out_pixel_id, dtype=object),
        forcing_member_id=np.array(out_member_id, dtype=np.int64),
        lat=np.array(out_lat, dtype=np.float64),
        lon=np.array(out_lon, dtype=np.float64),
        log_weight=np.array(out_log_weight, dtype=np.float64),
        valid=np.array(out_valid, dtype=bool),
        death_time=np.array(out_death_time, dtype="datetime64[ns]"),
        death_lat=np.array(out_death_lat, dtype=np.float64),
        death_lon=np.array(out_death_lon, dtype=np.float64),
        death_reason=np.array(out_death_reason, dtype=object),
        query_time=query_time,
    )


def _empty_slice(query_time: np.datetime64) -> HistoricalParticleSlice:
    return HistoricalParticleSlice(
        particle_id=np.array([], dtype=object),
        observation_pixel_id=np.array([], dtype=object),
        forcing_member_id=np.array([], dtype=np.int64),
        lat=np.array([], dtype=np.float64),
        lon=np.array([], dtype=np.float64),
        log_weight=np.array([], dtype=np.float64),
        valid=np.array([], dtype=bool),
        death_time=np.array([], dtype="datetime64[ns]"),
        death_lat=np.array([], dtype=np.float64),
        death_lon=np.array([], dtype=np.float64),
        death_reason=np.array([], dtype=object),
        query_time=query_time,
    )
