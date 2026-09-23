"""T19 -- Stage V: weighted core-particle sampling for forward validation.

Implements exactly `README.md` Sec. 17's second half (the sampling half; the
`N_forward = min(X, Y)` budget itself is T18's `iste.forward_budget`):

    When Y >= X: select X particles from the candidate core, using the Stage III
        particle weights as sampling weights, WITHOUT replacement by default.
    When Y < X: select ALL Y available core particles. Never reject the candidate
        merely because Y < X.

Selection preserves, for every selected particle: `particle_id`,
`observation_pixel_id`, `forcing_member_id`, position (`lat`/`lon`), the ORIGINAL Stage
III `log_weight` (never modified, never renormalized here), death metadata
(`death_time`/`death_lat`/`death_lon`/`death_reason`), and `valid`.

**Avoiding double-counting Stage III weights (`README.md` Sec. 17, this session's own
T19 spec).** This module does not itself compute or attach any forward-KDE weight --
that would start implementing the forward predicted KDE, which is explicitly out of
scope here ("Do not implement predicted slick/KDE comparison yet"). Instead it records
`resampled` (`True` iff `Y >= X`, i.e. weighted-without-replacement resampling actually
ran) on every result, so a LATER stage building the forward KDE knows which convention
to apply:

    resampled = True   -> the selected set is an empirical representative sample; that
                           later stage should use EQUAL computational weight per
                           selected particle by default.
    resampled = False  -> ALL of the core's original particles were retained (no
                           resampling occurred); that later stage MAY normalize their
                           original relative Stage III weights for its own forward-KDE
                           construction, but that normalization must happen THERE, not
                           in this module -- this module never modifies or normalizes
                           `log_weight` under any circumstance.

**Reuses T18's `compute_forward_particle_budget`** for `X`/`Y`/`N_forward`/
`forward_validatable` (no duplicated budget logic), and T15's `query_radius` directly
for the actual particle metadata to sample from -- both existing public functions,
neither modified by this module.

Does not implement forward propagation, predicted-slick/KDE construction, forward
scoring, or any use of Stage III `c_hat*` -- all later tasks' scope.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from iste.candidate_extraction import Candidate
from iste.forward_budget import ForwardBudgetConfig, compute_forward_particle_budget
from iste.kdtree_neighborhood import ParticleNeighborhoodIndex, query_radius


@dataclass(frozen=True)
class SampledCoreParticles:
    """The result of sampling one T17 `Candidate`'s core for forward validation.

    `candidate_id`/`t_s`/`X`/`Y`/`N_forward`/`forward_validatable` are T18's own budget
    fields, carried through unchanged for provenance. `resampled` and `sampling_seed`
    are this task's own additions (`README.md` Sec. 25's `forward_particle_count_
    N_forward` and `sampling_seed` output-artifact fields). Every selected-particle
    array has length `N_forward` (or `0` when `Y == 0`); the `i`-th entry of every array
    describes the same particle.
    """

    candidate_id: str
    t_s: np.datetime64
    X: int
    Y: int
    N_forward: int
    forward_validatable: bool
    resampled: bool  # True iff Y >= X (weighted-without-replacement resampling ran)
    sampling_seed: int

    particle_id: np.ndarray
    observation_pixel_id: np.ndarray
    forcing_member_id: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    log_weight: np.ndarray  # ORIGINAL Stage III log-weight -- never modified here
    valid: np.ndarray
    death_time: np.ndarray
    death_lat: np.ndarray
    death_lon: np.ndarray
    death_reason: np.ndarray

    @property
    def n_selected(self) -> int:
        return len(self.particle_id)


def _empty_selection_arrays() -> dict:
    return dict(
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
    )


def _selection_probabilities(log_weight: np.ndarray) -> np.ndarray:
    """Original Stage III `log_weight` -> a selection-probability vector (higher weight
    -> higher probability), via a numerically-stable `exp(log_weight - max)` /
    normalize -- the same log-space-to-linear-weight convention already used by
    `iste.support_features`/`iste.likelihood` (exponentiate only at the point a linear
    value is actually needed; `log_weight` itself is never touched).

    Degenerate edge case: if EVERY particle in the core has a non-finite `log_weight`
    (i.e. all Stage III weights are exactly zero), there is no informative weight signal
    to sample from; this falls back to a UNIFORM selection probability so sampling can
    still proceed rather than raising. This is an explicit, documented fallback for a
    degenerate input, not a claim about relative particle likelihood.
    """
    log_weight = np.asarray(log_weight, dtype=np.float64)
    n = len(log_weight)
    if n == 0:
        return np.array([], dtype=np.float64)
    finite_mask = np.isfinite(log_weight)
    if not np.any(finite_mask):
        return np.full(n, 1.0 / n)
    m = np.max(log_weight[finite_mask])
    linear = np.exp(log_weight - m)  # -inf entries -> exp(-inf) = 0.0, no NaN
    total = np.sum(linear)
    if total <= 0.0:
        return np.full(n, 1.0 / n)
    return linear / total


def sample_core_particles(
    candidate: Candidate,
    index: ParticleNeighborhoodIndex,
    config: ForwardBudgetConfig,
    seed: int,
    *,
    replace: bool = False,
) -> SampledCoreParticles:
    """Select `N_forward = min(config.X, Y)` particles from `candidate`'s core for
    forward validation.

    `index` must be the T15 `ParticleNeighborhoodIndex` built over the same historical
    timestamp's particle slice `candidate` came from (matching T18's own
    `compute_forward_particle_budget` requirement).

    `seed` is required (not optional) and is recorded verbatim as `sampling_seed` on the
    result, so the exact selection is reproducible from the returned record alone
    (`README.md`: "The sampling RNG must be reproducible.").

    `replace` selects with vs. without replacement when `Y >= X`; defaults to `False`
    ("without replacement by default", `README.md` Sec. 17).

    Behavior (`README.md` Sec. 17, locked):
        Y == 0          -> empty selection, `forward_validatable = False` (never raises)
        Y >= config.X   -> `X` particles drawn using Stage III weights as selection
                           probabilities (`resampled = True`)
        0 < Y < config.X -> ALL `Y` particles selected, unweighted (`resampled = False`);
                           the candidate is never rejected for this.
    """
    budget = compute_forward_particle_budget(candidate, index, config)
    core = candidate.core_region
    neighborhood = query_radius(index, core.center_lat, core.center_lon, core.radius_m)
    Y = neighborhood.n_particles
    X = budget.X

    rng = np.random.default_rng(seed)

    if Y == 0:
        selected = _empty_selection_arrays()
        resampled = False
    elif Y >= X:
        probabilities = _selection_probabilities(neighborhood.log_weight)
        selected_idx = rng.choice(Y, size=X, replace=replace, p=probabilities)
        selected = _gather(neighborhood, selected_idx)
        resampled = True
    else:  # 0 < Y < X: retain every core particle, no weighting/resampling
        selected_idx = np.arange(Y)
        selected = _gather(neighborhood, selected_idx)
        resampled = False

    return SampledCoreParticles(
        candidate_id=candidate.candidate_id,
        t_s=candidate.t_s,
        X=X,
        Y=Y,
        N_forward=budget.N_forward,
        forward_validatable=budget.forward_validatable,
        resampled=resampled,
        sampling_seed=seed,
        **selected,
    )


def _gather(neighborhood, selected_idx: np.ndarray) -> dict:
    """Pull out the selected local indices' full metadata from a T15
    `NeighborhoodResult`, unchanged (in particular `log_weight` is copied verbatim --
    never exponentiated, scaled, or otherwise modified)."""
    selected_idx = np.asarray(selected_idx, dtype=np.int64)
    return dict(
        particle_id=neighborhood.particle_id[selected_idx],
        observation_pixel_id=neighborhood.observation_pixel_id[selected_idx],
        forcing_member_id=neighborhood.forcing_member_id[selected_idx],
        lat=neighborhood.lat[selected_idx],
        lon=neighborhood.lon[selected_idx],
        log_weight=neighborhood.log_weight[selected_idx],
        valid=neighborhood.valid[selected_idx],
        death_time=neighborhood.death_time[selected_idx],
        death_lat=neighborhood.death_lat[selected_idx],
        death_lon=neighborhood.death_lon[selected_idx],
        death_reason=neighborhood.death_reason[selected_idx],
    )


def sample_core_particles_for_candidates(
    candidates: list[Candidate],
    index: ParticleNeighborhoodIndex,
    config: ForwardBudgetConfig,
    seeds: list[int],
    *,
    replace: bool = False,
) -> list[SampledCoreParticles]:
    """`sample_core_particles` for every `(candidate, seed)` pair, all drawn from the
    SAME historical timestamp / T15 `index` (see `sample_core_particles`'s docstring for
    the multi-timestamp case -- group by `t_s` and call once per group, exactly as T17's
    `find_candidate_peaks_from_index` and T18's `compute_forward_particle_budgets`
    already require).

    `seeds` must have exactly one entry per candidate -- each candidate's sampling seed
    is an explicit, separately-controllable input (no seed is silently derived or
    reused across candidates), so two candidates given the SAME seed are independently
    reproducible from that one seed, and two candidates given DIFFERENT seeds are
    sampled independently of one another.
    """
    if len(seeds) != len(candidates):
        raise ValueError(f"seeds has {len(seeds)} entries but candidates has {len(candidates)}; need exactly one seed per candidate")
    return [
        sample_core_particles(candidate, index, config, seed, replace=replace)
        for candidate, seed in zip(candidates, seeds)
    ]
