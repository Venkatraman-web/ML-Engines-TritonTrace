"""T21 -- Stage VI: forward-propagate one candidate's T19-selected core particles under
the configured forcing ensemble.

Implements exactly `README.md` Sec. 19's orchestration diagram, for one candidate at a
time:

    Candidate C at t_s
       |
       +-- U_1 -> predicted slick_1
       +-- U_2 -> predicted slick_2
       +-- ...
       +-- U_E -> predicted slick_E

i.e. `sampled.particle_id`'s (T19's) selected core particles are propagated from `t_s`
to exactly `t_obs`, once per configured forcing member, each run independently coherent
(the SAME forcing member's `U_e` field for the ENTIRE trajectory of that run -- never
redrawn per particle or per timestep).

This module adds **no new transport physics** -- every displacement still comes from
T20's `iste.forward_transport.forward_euler_maruyama_step`, called once per checkpoint,
for all of a candidate's selected particles at once (T20 is already vectorized over
particles). This module only adds: the `t_s -> t_obs` checkpoint loop, per-member
orchestration (one `iste.ensemble_run.make_perturbed_source` build per member, reused
unchanged for that member's entire run), and the "stop exactly at `t_obs`" checkpoint
schedule.

**Exact termination at `t_obs`.** The checkpoint schedule is built once, up front, from
INTEGER nanosecond arithmetic (never accumulated step-by-step float `dt`, so floating-
point drift cannot make the final checkpoint disagree with `t_obs`): `n_full_steps =
(t_obs - t_s) // dt` steps of exactly `dt_seconds` each, plus -- if `(t_obs - t_s)` is
not an exact multiple of `dt_seconds` -- one final PARTIAL step of whatever duration
remains, so the run always lands exactly on `t_obs` regardless of the configured `dt`,
rather than silently overshooting it or requiring the caller to choose a `dt` that
divides evenly.

**`forcing_member_id` distinction preserved (`memory.md`, 2026-09-23):** every returned
`ForwardCandidateMemberRun.forcing_member_id` is each particle's ORIGINAL Stage III
provenance (T14-T19, unmodified, copied straight through from `sampled`) --
`forward_forcing_member_id` is the SEPARATE, single member this particular run was
executed under (README Sec. 19's `U_1`..`U_E` loop variable). The two are never
conflated: a particle's own original member never selects which `U_e` its forward run
uses (the whole selected set is propagated once per configured member, per Sec. 19),
and this module never reads a particle's `forcing_member_id` to make that choice.

**Original Stage III weight is preserved, never modified.** `original_log_weight` is
copied straight through from `sampled.log_weight` (T19) -- this module has no weight
term at all (matching T20; see that module's own docstring/memory.md entry for why the
Phase 2 forward prototype has no weight update).

Does not implement predicted-slick/KDE construction (T22), forward error metrics (T23),
AIS, or formal source inference -- and never reads or references Stage III `c_hat*`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from iste.core_sampling import SampledCoreParticles
from iste.ensemble_run import make_perturbed_source
from iste.environment import GriddedFieldSource
from iste.forcing_ensemble import ForcingEnsemble
from iste.forward_transport import ForwardParticleState, forward_euler_maruyama_step
from iste.geometry import MEAN_EARTH_RADIUS_M


@dataclass
class ForwardCandidateMemberRun:
    """The complete forward-propagated checkpoint history of ALL of one candidate's T19
    core particles, run from `t_s` to `t_obs` under ONE forcing member.

    `lat`/`lon`/`valid` have shape `(n_steps + 1, n_particles)` -- one row per
    checkpoint (`t_s` included as row 0), one column per selected particle, in the same
    particle order as `particle_id`/`observation_pixel_id`/`forcing_member_id`/
    `original_log_weight` (each length `n_particles`). This length never changes
    regardless of when/whether a particle becomes invalid during the run -- only
    `valid` (and the frozen `lat`/`lon`) reflect that, matching T06's own
    `BackwardTrajectory` convention.
    """

    candidate_id: str
    t_s: np.datetime64
    t_obs: np.datetime64
    forward_forcing_member_id: int
    dt_seconds: float
    n_steps: int
    timestamps: np.ndarray  # datetime64[ns], shape (n_steps+1,): t_s, ..., t_obs

    particle_id: np.ndarray
    observation_pixel_id: np.ndarray
    forcing_member_id: np.ndarray  # ORIGINAL Stage III provenance, preserved unmodified
    original_log_weight: np.ndarray  # original Stage III weight, preserved unmodified

    lat: np.ndarray
    lon: np.ndarray
    valid: np.ndarray

    @property
    def n_particles(self) -> int:
        return len(self.particle_id)

    @property
    def final_lat(self) -> np.ndarray:
        return self.lat[-1]

    @property
    def final_lon(self) -> np.ndarray:
        return self.lon[-1]

    @property
    def final_valid(self) -> np.ndarray:
        return self.valid[-1]

    def first_invalid_step(self, particle_index: int) -> int | None:
        """Checkpoint index (into `timestamps`) at which `particle_index` first became
        invalid during THIS forward run, or `None` if it stayed valid throughout --
        mirrors `iste.trajectory.BackwardTrajectory.first_invalid_index`'s convention,
        per-particle. Does not distinguish an already-invalid starting particle (row 0
        already `False`) from one invalidated mid-run; index 0 covers both."""
        invalid_rows = np.flatnonzero(~self.valid[:, particle_index])
        return int(invalid_rows[0]) if invalid_rows.size else None


@dataclass
class ForwardCandidateRuns:
    """All forward-validation runs for ONE candidate, across the explicit forcing-member
    subset actually used. `forcing_member_ids` IS the recorded subset (`README.md`
    Sec. 19/26: "A development run may use an explicit subset of forcing members; the
    selected subset must be recorded.")."""

    candidate_id: str
    t_s: np.datetime64
    t_obs: np.datetime64
    forcing_member_ids: tuple[int, ...]
    runs: dict[int, ForwardCandidateMemberRun] = field(default_factory=dict)

    def __getitem__(self, member_id: int) -> ForwardCandidateMemberRun:
        return self.runs[member_id]


def _checkpoint_step_sizes_ns(t_s: np.datetime64, t_obs: np.datetime64, dt_seconds: float) -> list[int]:
    """The forward-time checkpoint schedule's step sizes (nanoseconds), from `t_s` to
    exactly `t_obs`: as many full `dt_seconds` steps as fit, plus one final partial step
    covering whatever remains -- see module docstring. Returns `[]` if `t_s == t_obs`
    (no propagation needed, already at `t_obs`)."""
    if dt_seconds <= 0:
        raise ValueError(f"dt_seconds must be > 0; got {dt_seconds!r}")
    dt_ns = int(round(dt_seconds * 1e9))
    if dt_ns <= 0:
        raise ValueError(f"dt_seconds must correspond to a positive nanosecond step; got {dt_seconds!r}")

    total_ns = int((t_obs - t_s) / np.timedelta64(1, "ns"))
    if total_ns < 0:
        raise ValueError(f"t_obs ({t_obs}) must be >= t_s ({t_s}) for forward propagation")

    n_full_steps, remainder_ns = divmod(total_ns, dt_ns)
    step_sizes_ns = [dt_ns] * n_full_steps
    if remainder_ns > 0:
        step_sizes_ns.append(remainder_ns)
    return step_sizes_ns


def forward_propagate_candidate_member(
    sampled: SampledCoreParticles,
    t_obs,
    dt_seconds: float,
    ocean_source_for_member: GriddedFieldSource,
    forward_forcing_member_id: int,
    k_lat: np.ndarray,
    k_lon: np.ndarray,
    k_field: np.ndarray,
    rng: np.random.Generator,
    domain_variable: str = "uo",
    radius_m: float = MEAN_EARTH_RADIUS_M,
) -> ForwardCandidateMemberRun:
    """Forward-propagate ALL of `sampled`'s selected core particles together, from
    `sampled.t_s` to exactly `t_obs`, under ONE forcing member.

    `ocean_source_for_member` must already be that member's perturbed `U_e` field
    (`iste.ensemble_run.make_perturbed_source(base_ocean_source, ensemble, member_id)`)
    -- built ONCE by the caller (see `forward_propagate_candidate` below) and passed in
    unchanged, so the identical field object is reused for every checkpoint of this run
    ("do not redraw forcing independently per particle or timestep").

    `rng` is a single `numpy.random.Generator` whose state advances across every
    checkpoint of this one run (the same seed + same inputs reproduce the identical
    trajectory; a different seed normally gives a different one) -- exactly T06's own
    reproducibility discipline, extended to a vectorized multi-particle run.

    Particles already invalid in `sampled` (there should be none -- T19's particles all
    come from a T15 index that only ever holds valid particles -- but this is never
    assumed) start invalid and stay invalid throughout, never silently revalidated. Once
    EVERY particle in the batch has become invalid, no further T20 step calls are made
    (a pure no-op would otherwise be repeated every remaining checkpoint); the frozen
    state is simply carried into every remaining recorded checkpoint instead.
    """
    t_s = np.datetime64(sampled.t_s, "ns")
    t_obs = np.datetime64(t_obs, "ns")
    step_sizes_ns = _checkpoint_step_sizes_ns(t_s, t_obs, dt_seconds)
    n_steps = len(step_sizes_ns)

    timestamps = np.empty(n_steps + 1, dtype="datetime64[ns]")
    timestamps[0] = t_s
    cumulative_ns = 0
    for i, size_ns in enumerate(step_sizes_ns):
        cumulative_ns += size_ns
        timestamps[i + 1] = t_s + np.timedelta64(cumulative_ns, "ns")

    n_particles = sampled.n_selected
    lat_hist = np.empty((n_steps + 1, n_particles), dtype=np.float64)
    lon_hist = np.empty((n_steps + 1, n_particles), dtype=np.float64)
    valid_hist = np.empty((n_steps + 1, n_particles), dtype=bool)

    state = ForwardParticleState(
        lat=sampled.lat.copy(), lon=sampled.lon.copy(), valid=sampled.valid.copy(),
        forcing_member_id=sampled.forcing_member_id.copy(),
    )
    lat_hist[0] = state.lat
    lon_hist[0] = state.lon
    valid_hist[0] = state.valid

    for i, size_ns in enumerate(step_sizes_ns):
        if n_particles > 0 and np.any(state.valid):
            step_start_t = timestamps[i]
            step_dt_seconds = size_ns / 1e9
            state = forward_euler_maruyama_step(
                state, step_start_t, step_dt_seconds, ocean_source_for_member,
                k_lat, k_lon, k_field, rng, domain_variable=domain_variable, radius_m=radius_m,
            )
        # else: every particle already invalid (or there are none at all) -- no T20
        # step call is made; `state` is carried forward unchanged, exactly like T06's
        # "already invalid, do not call the step" policy.
        lat_hist[i + 1] = state.lat
        lon_hist[i + 1] = state.lon
        valid_hist[i + 1] = state.valid

    return ForwardCandidateMemberRun(
        candidate_id=sampled.candidate_id,
        t_s=t_s,
        t_obs=t_obs,
        forward_forcing_member_id=forward_forcing_member_id,
        dt_seconds=dt_seconds,
        n_steps=n_steps,
        timestamps=timestamps,
        particle_id=sampled.particle_id.copy(),
        observation_pixel_id=sampled.observation_pixel_id.copy(),
        forcing_member_id=sampled.forcing_member_id.copy(),
        original_log_weight=sampled.log_weight.copy(),
        lat=lat_hist,
        lon=lon_hist,
        valid=valid_hist,
    )


def forward_propagate_candidate(
    sampled: SampledCoreParticles,
    t_obs,
    dt_seconds: float,
    base_ocean_source: GriddedFieldSource,
    ensemble: ForcingEnsemble,
    forcing_member_ids: Sequence[int],
    seeds: Sequence[int],
    k_lat: np.ndarray,
    k_lon: np.ndarray,
    k_field: np.ndarray,
    domain_variable: str = "uo",
    radius_m: float = MEAN_EARTH_RADIUS_M,
) -> ForwardCandidateRuns:
    """`README.md` Sec. 19's per-candidate orchestration: forward-propagate `sampled`'s
    selected core particles under EACH of `forcing_member_ids`, independently.

    `forcing_member_ids` may be `ensemble`'s full member set or an explicit development
    subset -- either way it is exactly what gets recorded on the returned
    `ForwardCandidateRuns.forcing_member_ids` (`README.md`: "the selected subset must be
    recorded").

    `seeds` must have exactly one entry per `forcing_member_ids` -- each member's
    forward run gets its own EXPLICIT RNG seed (no seed silently shared or derived
    across members), matching this session's T19 `sample_core_particles_for_candidates`
    convention. Two calls with the same `sampled`/`t_obs`/`dt_seconds`/`forcing_member_ids`
    /`seeds` reproduce identical trajectories for every member; a different seed for a
    given member normally changes only that member's run.

    Exactly one `iste.ensemble_run.make_perturbed_source` build per member (never per
    particle, per timestep, or repeated across members) keeps every member's field
    coherent for its entire run.
    """
    if len(seeds) != len(forcing_member_ids):
        raise ValueError(
            f"seeds has {len(seeds)} entries but forcing_member_ids has {len(forcing_member_ids)}; "
            "need exactly one seed per forcing member"
        )

    runs: dict[int, ForwardCandidateMemberRun] = {}
    for member_id, seed in zip(forcing_member_ids, seeds):
        ocean_source_for_member = make_perturbed_source(base_ocean_source, ensemble, member_id)
        rng = np.random.default_rng(seed)
        runs[member_id] = forward_propagate_candidate_member(
            sampled, t_obs, dt_seconds, ocean_source_for_member, member_id,
            k_lat, k_lon, k_field, rng, domain_variable=domain_variable, radius_m=radius_m,
        )

    return ForwardCandidateRuns(
        candidate_id=sampled.candidate_id,
        t_s=np.datetime64(sampled.t_s, "ns"),
        t_obs=np.datetime64(t_obs, "ns"),
        forcing_member_ids=tuple(forcing_member_ids),
        runs=runs,
    )
