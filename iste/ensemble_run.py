"""T09 — small multi-particle, multi-forcing-member Stage III run.

Runs the *existing, unmodified* T05/T06 backward-adjoint pipeline for a small set of
upstream particles (T07's `AdaptedParticleSet`) across a small set of forcing members
(T08's `ForcingEnsemble`). This module adds **no new transport/weight equations** --
every displacement and weight update still comes from `iste.step.backward_euler_maruyama_step`
via `iste.trajectory.run_backward_trajectory`, called once per (particle, forcing member)
pair.

**How a forcing member is injected without modifying T05/T06:** T05 only ever accesses an
`ocean_source` through `source.data[variable][time_index]` (a single 2-D time-slice) and
`iste.interpolation.interpolate_field`/`iste.derivatives.divergence_2d`, which use that
same indexing internally. `_PerturbedComponentArray` below is a small lazy, on-demand
stand-in for one velocity component's full time series: indexing it at time step `i`
computes `base[i] + delta_u_e(x, t_i)` (via T08's `ForcingEnsemble.perturbation`,
evaluated over the *fine* ocean grid only for that one requested timestep -- never
precomputed or stored for the full time series) and returns a normal, real NumPy array.
Wrapping two of these (`uo`, `vo`) in an ordinary `iste.environment.GriddedFieldSource`
gives a drop-in "perturbed" source that T05/T06 consume completely unaware anything is
different -- no fork, no duplicated equations, and the *real* loaded base `uo`/`vo`
arrays (T01) are referenced, never copied.

Because one `GriddedFieldSource` (and its lazy component arrays) is built once per forcing
member and reused for every particle assigned to that member, **every particle sees the
identical perturbation realization at the same query time by construction** -- coherence
is a structural property of this design, not an extra check bolted on afterward.

Records retained per `(particle, forcing_member, checkpoint)`, per `README.md`'s Stage III
lookup-table fields: `timestamp`, `latitude`, `longitude`, `log_weight` (`README.md`'s
"particle_weight," kept in log-space for the same numerical-stability reason T05/T06
already do -- `weight = exp(log_weight)`), `observation_pixel_id`, `forcing_member_id`,
`validity`. Invalid particles are always retained (never dropped), matching T04/T06's own
"stop propagation, retain last valid state" policy, applied once per forcing member.

This module does **not** implement KDE, the observation likelihood, SMC, MCMC, or AIS.
`compute_diagnostics`/`compute_diagnostics_by_member` below select each
`(particle, forcing_member)` trajectory's final checkpoint row and hand it to T10's
reusable `iste.diagnostics.compute_stage3_diagnostics_from_log_weights` -- the
`total/valid/invalid_particles`, `total_weight`, `effective_sample_size`, and
`valid_particle_fraction` arithmetic itself lives there, in exactly one place, not
duplicated in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from iste.diagnostics import StageIIIDiagnostics, compute_stage3_diagnostics_from_log_weights
from iste.environment import GriddedFieldSource
from iste.forcing_ensemble import ForcingEnsemble
from iste.observation import AdaptedParticleSet
from iste.trajectory import run_backward_trajectory

# T10 note: `RunDiagnostics` is kept as a name here (aliasing T10's `StageIIIDiagnostics`)
# so existing callers/tests that refer to "this module's diagnostics type" keep working;
# the fields and the ESS/edge-case logic itself live in exactly one place, `iste.diagnostics`.
RunDiagnostics = StageIIIDiagnostics


class _PerturbedComponentArray:
    """Lazy, on-demand full-fine-grid view of one perturbed velocity component
    (`uo + delta_u_e` or `vo + delta_v_e`) for one forcing member.

    Supports only `__getitem__(time_index)` -- the single access pattern
    `iste.interpolation`/`iste.step` actually use on `GriddedFieldSource.data[var]`.
    Each requested time slice is computed once and cached (bounded by the number of
    distinct timesteps this run's trajectories actually visit -- never the full
    336-timestep base series, and never for more than one forcing member at a time in
    this array). Never duplicates or stores the full base `uo`/`vo` series itself
    (`self._base` is a reference to the already-loaded T01 array, not a copy).
    """

    def __init__(
        self, base_array: np.ndarray, ensemble: ForcingEnsemble, member_id: int, component: int,
        lat_coords: np.ndarray, lon_coords: np.ndarray, time_coords: np.ndarray,
    ):
        self._base = base_array
        self._ensemble = ensemble
        self._member_id = member_id
        self._component = component  # 0 -> eastward (u), 1 -> northward (v)
        lon_mesh, lat_mesh = np.meshgrid(lon_coords, lat_coords)  # both shape (n_lat, n_lon)
        self._lat_flat = lat_mesh.ravel()
        self._lon_flat = lon_mesh.ravel()
        self._grid_shape = (len(lat_coords), len(lon_coords))
        self._time_coords = time_coords
        self._cache: dict[int, np.ndarray] = {}

    def __getitem__(self, i: int) -> np.ndarray:
        if i in self._cache:
            return self._cache[i]
        du, dv, valid = self._ensemble.perturbation(self._member_id, self._lat_flat, self._lon_flat, self._time_coords[i])
        delta = du if self._component == 0 else dv
        perturbed = self._base[i] + delta.reshape(self._grid_shape)
        perturbed = np.where(valid.reshape(self._grid_shape), perturbed, np.nan)
        self._cache[i] = perturbed
        return perturbed


def make_perturbed_source(ocean_source: GriddedFieldSource, ensemble: ForcingEnsemble, member_id: int) -> GriddedFieldSource:
    """A `GriddedFieldSource` for forcing member `member_id`: `uo`/`vo` replaced by lazy
    `_PerturbedComponentArray`s (`U_e = U + delta_U_e`, T08). Everything else
    (`time`/`lat`/`lon`/`fill_threshold`) is the real base source's own data, unchanged --
    this is a thin view, not a copy of the environmental grids."""
    u_lazy = _PerturbedComponentArray(
        ocean_source.data["uo"], ensemble, member_id, 0, ocean_source.lat, ocean_source.lon, ocean_source.time
    )
    v_lazy = _PerturbedComponentArray(
        ocean_source.data["vo"], ensemble, member_id, 1, ocean_source.lat, ocean_source.lon, ocean_source.time
    )
    return GriddedFieldSource(
        source_path=f"{ocean_source.source_path} [perturbed, forcing_member_id={member_id}]",
        time_name=ocean_source.time_name,
        time=ocean_source.time,
        lat_name=ocean_source.lat_name,
        lat=ocean_source.lat,
        lon_name=ocean_source.lon_name,
        lon=ocean_source.lon,
        data={"uo": u_lazy, "vo": v_lazy},
        fill_threshold=ocean_source.fill_threshold,
    )


@dataclass
class EnsembleRunRecords:
    """Flat, per-`(particle, forcing_member, checkpoint)` records from a T09 run. Every
    array has the same length (`n_records`); row `k` across all arrays describes one
    checkpoint of one particle under one forcing member. Invalid particles contribute
    exactly one row per checkpoint too (never dropped) -- their `lat`/`lon`/`log_weight`
    are simply frozen from their last valid state, per T04/T05/T06's own policy.

    `death_time`/`death_lat`/`death_lon`/`death_reason` (T13.5,
    `final_stage_III_IV_specification.txt` compatibility): the exact death event for the
    `(particle_id, forcing_member_id)` group this row belongs to, REPEATED identically
    across every row of that group (it is a single event, not one per checkpoint --
    mirroring how `particle_id`/`observation_pixel_id`/`forcing_member_id` are already
    repeated per row). `death_time` is `NaT`/`death_lat`/`death_lon` are `NaN`/
    `death_reason` is `""` for a particle that never died within the run. `survival_mark`
    (`int8`, `1`/`0`) mirrors `valid` at THIS row's own checkpoint (genuinely varies per
    row, unlike the death fields) -- an explicit, separately-named column matching the
    frozen contract's own schema, even though it is fully derivable from `valid`.

    All five new fields default to `None`, meaning "not tracked" -- for hand-built
    `EnsembleRunRecords` (e.g. existing test fixtures predating T13.5) that only ever
    supply the original eight fields. `run_ensemble` (below) always populates real arrays
    for every field; `iste.lookup_storage.ParticleRecordStore.add_records` normalizes a
    `None` batch field to a length-matched "not tracked" array before storing it."""

    particle_id: np.ndarray
    forcing_member_id: np.ndarray
    observation_pixel_id: np.ndarray
    timestamp: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    log_weight: np.ndarray
    valid: np.ndarray
    death_time: np.ndarray | None = None
    death_lat: np.ndarray | None = None
    death_lon: np.ndarray | None = None
    death_reason: np.ndarray | None = None
    survival_mark: np.ndarray | None = None

    @property
    def n_records(self) -> int:
        return len(self.timestamp)


def run_ensemble(
    particles: AdaptedParticleSet,
    forcing_member_ids: Sequence[int],
    n_steps: int,
    dt_seconds: float,
    ocean_source: GriddedFieldSource,
    ensemble: ForcingEnsemble,
    k_lat: np.ndarray,
    k_lon: np.ndarray,
    k_field: np.ndarray,
    master_seed: int = 0,
    domain_variable: str = "uo",
) -> EnsembleRunRecords:
    """Run T06's unmodified `run_backward_trajectory` once per (particle, forcing member)
    pair, using `particles` (T07's `AdaptedParticleSet` -- this function never generates
    or segments particles itself) and `forcing_member_ids` (a small subset of
    `ensemble`'s configured members).

    One `make_perturbed_source` view is built per forcing member (not per particle) and
    reused for every particle assigned to it, so every particle under a given member sees
    the identical perturbation realization at the same query time, by construction.

    A particle already marked invalid by T07's upstream adapter is recorded once per
    forcing member at `t0` (invalid, unmoved) -- T06 has no "start already invalid" mode
    (unchanged from T07's documented handoff behavior), so it is never propagated.

    RNG streams are fully reproducible and independent per (particle, forcing member):
    derived from `master_seed` via nested `numpy.random.SeedSequence.spawn`, one child
    sequence per input particle, further split into one grandchild per forcing member
    (indexed by `ensemble`'s own member-id space, so results are stable regardless of
    which subset of `forcing_member_ids` is actually requested).

    Memory: only one forcing member's perturbed-source cache and one particle's
    `BackwardTrajectory` are held at a time (the loop below processes one
    (particle, member) pair fully before moving to the next); nothing precomputes or
    stores every member's full particle history simultaneously.
    """
    n_particles = particles.n_particles
    particle_seeds = np.random.SeedSequence(master_seed).spawn(n_particles)

    perturbed_sources: dict[int, GriddedFieldSource] = {
        member_id: make_perturbed_source(ocean_source, ensemble, member_id) for member_id in forcing_member_ids
    }

    particle_id_col: list = []
    forcing_member_id_col: list = []
    observation_pixel_id_col: list = []
    timestamp_col: list = []
    lat_col: list = []
    lon_col: list = []
    log_weight_col: list = []
    valid_col: list = []
    death_time_col: list = []
    death_lat_col: list = []
    death_lon_col: list = []
    death_reason_col: list = []
    survival_mark_col: list = []

    for p_idx in range(n_particles):
        pid = particles.particle_id[p_idx]
        pixel_id = particles.observation_pixel_id[p_idx]
        t0 = particles.timestamp[p_idx]
        lat0 = float(particles.state.lat[p_idx])
        lon0 = float(particles.state.lon[p_idx])
        log_weight0 = float(particles.state.log_weight[p_idx])
        upstream_valid = bool(particles.state.valid[p_idx])
        member_rngs = particle_seeds[p_idx].spawn(ensemble.config.n_members)

        for member_id in forcing_member_ids:
            n_rows = n_steps + 1
            if upstream_valid:
                rng = np.random.default_rng(member_rngs[member_id])
                # T13.5: divergence_source=ocean_source (the BASE, unperturbed field) --
                # advection still uses perturbed_sources[member_id] (U_e), but the weight
                # update's div(U) term must use the base U, per the frozen contract's
                # Sec. 5 (see iste.step.backward_euler_maruyama_step's own docs).
                traj = run_backward_trajectory(
                    lat0, lon0, t0, n_steps, dt_seconds, perturbed_sources[member_id],
                    k_lat, k_lon, k_field, rng, log_weight0=log_weight0, domain_variable=domain_variable,
                    divergence_source=ocean_source,
                )
                timestamps, lats, lons, log_weights, valids = (
                    traj.timestamps, traj.lat, traj.lon, traj.log_weight, traj.valid,
                )
                if traj.death_time is not None:
                    death_time_row = np.full(n_rows, traj.death_time, dtype="datetime64[ns]")
                    death_lat_row = np.full(n_rows, traj.death_lat, dtype=np.float64)
                    death_lon_row = np.full(n_rows, traj.death_lon, dtype=np.float64)
                    death_reason_row = np.array([traj.death_reason] * n_rows, dtype=object)
                else:
                    death_time_row = np.full(n_rows, np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
                    death_lat_row = np.full(n_rows, np.nan, dtype=np.float64)
                    death_lon_row = np.full(n_rows, np.nan, dtype=np.float64)
                    death_reason_row = np.array([""] * n_rows, dtype=object)
            else:
                # already invalid upstream -- one frozen record only, not propagated,
                # per T07's documented handoff behavior (T06 cannot start pre-invalid).
                # Its "death" (or rather, its known-invalid status) predates the backward
                # window entirely: no integration event caused it, so death_time is set
                # to t0 (the only instant its invalid status is actually known at) and
                # death_lat/lon to its only known position -- never fabricated further.
                n_rows = 1
                timestamps = np.array([t0], dtype="datetime64[ns]")
                lats = np.array([lat0])
                lons = np.array([lon0])
                log_weights = np.array([log_weight0])
                valids = np.array([False])
                death_time_row = np.array([np.datetime64(t0, "ns")], dtype="datetime64[ns]")
                death_lat_row = np.array([lat0], dtype=np.float64)
                death_lon_row = np.array([lon0], dtype=np.float64)
                death_reason_row = np.array(["already_invalid_upstream"], dtype=object)

            particle_id_col.extend([pid] * n_rows)
            forcing_member_id_col.extend([member_id] * n_rows)
            observation_pixel_id_col.extend([pixel_id] * n_rows)
            timestamp_col.append(timestamps)
            lat_col.append(lats)
            lon_col.append(lons)
            log_weight_col.append(log_weights)
            valid_col.append(valids)
            death_time_col.append(death_time_row)
            death_lat_col.append(death_lat_row)
            death_lon_col.append(death_lon_row)
            death_reason_col.append(death_reason_row)
            survival_mark_col.append(valids.astype(np.int8))

    return EnsembleRunRecords(
        particle_id=np.array(particle_id_col, dtype=object),
        forcing_member_id=np.array(forcing_member_id_col, dtype=np.int64),
        observation_pixel_id=np.array(observation_pixel_id_col, dtype=object),
        timestamp=np.concatenate(timestamp_col),
        lat=np.concatenate(lat_col),
        lon=np.concatenate(lon_col),
        log_weight=np.concatenate(log_weight_col),
        valid=np.concatenate(valid_col),
        death_time=np.concatenate(death_time_col),
        death_lat=np.concatenate(death_lat_col),
        death_lon=np.concatenate(death_lon_col),
        death_reason=np.concatenate(death_reason_col),
        survival_mark=np.concatenate(survival_mark_col),
    )


def compute_diagnostics(records: EnsembleRunRecords) -> RunDiagnostics:
    """`total_particles`/`valid_particles`/`invalid_particles`/`total_weight`/
    `effective_sample_size`/`valid_particle_fraction`, computed by T10's
    `iste.diagnostics.compute_stage3_diagnostics_from_log_weights` across every
    `(particle, forcing_member)` trajectory's FINAL recorded checkpoint -- i.e. "how many
    of this run's particle-realizations are still valid, and how concentrated are their
    weights, at the end of the configured backward window." This function's own job is
    only to select those final rows; the diagnostic arithmetic itself lives in
    `iste.diagnostics` (not duplicated here)."""
    final_mask = _final_row_mask(records)
    return compute_stage3_diagnostics_from_log_weights(records.log_weight[final_mask], records.valid[final_mask])


def compute_diagnostics_by_member(records: EnsembleRunRecords) -> dict:
    """The same diagnostics as `compute_diagnostics`, broken down per `forcing_member_id`
    -- a direct, inspectable check that each member's particles behave as an independent
    (but internally coherent) realization."""
    final_mask = _final_row_mask(records)
    result = {}
    for member_id in np.unique(records.forcing_member_id):
        member_mask = final_mask & (records.forcing_member_id == member_id)
        result[int(member_id)] = compute_stage3_diagnostics_from_log_weights(
            records.log_weight[member_mask], records.valid[member_mask]
        )
    return result


def _final_row_mask(records: EnsembleRunRecords) -> np.ndarray:
    """Boolean mask selecting each `(particle_id, forcing_member_id)` group's LAST row
    (by position -- rows are appended in chronological/run order per group)."""
    key = np.array(list(zip(records.particle_id, records.forcing_member_id)), dtype=object)
    mask = np.zeros(len(key), dtype=bool)
    # last occurrence of each distinct key, found by walking once and remembering the
    # most recent index per key -- O(n), no sorting/hashing surprises with object dtype.
    last_index: dict = {}
    for i, k in enumerate(key):
        last_index[tuple(k)] = i
    mask[list(last_index.values())] = True
    return mask
