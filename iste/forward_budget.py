"""T18 -- Stage V: forward particle budget `N_forward = min(X, Y)`.

Implements exactly `README.md`'s "Locked rule: `N_forward = min(X,Y)`" (Sec. 11.2) and
Sec. 17's opening definition, for one T17 `Candidate` at a time:

    X = configured forward-validation particle budget (README.md Sec. 26: "X forward
        particle budget" -- an explicit, configurable Phase 2 parameter, never a hidden
        constant)
    Y = number of VALID Stage III particles inside the candidate's own `core_region`
    N_forward = min(X, Y)

Consequences (`README.md` Sec. 11.2, locked):

    Y > X  -> N_forward = X
    Y = X  -> N_forward = X
    Y < X  -> N_forward = Y   (never rejected merely because Y < X)
    Y = 0  -> candidate cannot be forward-validated (forward_validatable = False)

`X` is a computational particle-count budget only -- explicitly **not** a physical
statement about oil mass or the number of physical oil parcels (`README.md` Sec. 11.2).

**`Y` is computed by re-querying the T15 KD-tree, not by reusing T16's `D`.** A
candidate's `density_score` (T16 `D`, from `iste.support_features`) was computed at the
candidate's peak using whatever radius/k the *support map* was evaluated with, which may
differ from `core_region.radius_m` (T17's own, separately-configured, candidate-core
definition). Sec. 16 is explicit that `core_region` -- not the support map's own query
radius -- "is the high-support region used to select representative particles for
forward validation," so this module re-queries `iste.kdtree_neighborhood.query_radius`
at `core_region.center_lat`/`center_lon`/`radius_m` directly. Because a
`ParticleNeighborhoodIndex` (T15) only ever indexes VALID particles (T15's own "invalid
particles are never indexed" policy), every particle `query_radius` returns is already
guaranteed valid -- `Y` counting "only valid Stage III particles" falls out of T15's
existing structure rather than needing a fresh validity filter here.

**Scope boundary.** This module only counts `Y` and computes `N_forward`. It does not:
    - select/sample *which* `N_forward` particles to use (weighted core-particle
      sampling is T19's scope, `README.md` Sec. 17's second half);
    - propagate particles forward in time (T20/T21+);
    - read or reference Stage III `c_hat*` anywhere;
    - modify `iste.particle_query` (T14), `iste.kdtree_neighborhood` (T15),
      `iste.support_features` (T16), or `iste.candidate_extraction` (T17) in any way --
      this module only calls their existing public functions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from iste.candidate_extraction import Candidate
from iste.kdtree_neighborhood import ParticleNeighborhoodIndex, query_radius


@dataclass(frozen=True)
class ForwardBudgetConfig:
    """`X` -- the configured forward-validation particle budget (`README.md` Sec. 11.2/
    17/26: "X forward particle budget"). Explicit Phase 2 configuration, not a hidden
    constant; must be a positive integer particle count."""

    X: int

    def __post_init__(self) -> None:
        if isinstance(self.X, bool) or not isinstance(self.X, (int, np.integer)):
            raise TypeError(f"X must be an int; got {self.X!r} ({type(self.X).__name__})")
        if self.X <= 0:
            raise ValueError(f"X must be > 0; got {self.X!r}")


@dataclass(frozen=True)
class ForwardBudget:
    """One candidate's forward particle budget -- `README.md` Sec. 25's
    `core_particle_count_Y` / `forward_particle_count_N_forward` output fields, plus the
    `X` that produced them and enough provenance (`candidate_id`, `t_s`) to trace the
    decision back to its candidate."""

    candidate_id: str
    t_s: np.datetime64
    X: int
    Y: int
    N_forward: int
    forward_validatable: bool  # False iff Y == 0 ("cannot be forward-validated")


def compute_forward_particle_budget(
    candidate: Candidate,
    index: ParticleNeighborhoodIndex,
    config: ForwardBudgetConfig,
) -> ForwardBudget:
    """Compute one T17 `Candidate`'s `Y`, `X`, and `N_forward = min(X, Y)`.

    `index` must be the T15 `ParticleNeighborhoodIndex` built over the SAME historical
    timestamp's particle slice `candidate` came from (i.e. `index.slice_.query_time ==
    candidate.t_s`) -- exactly as T17's own `find_candidate_peaks_from_index` already
    requires one `index` per one `t_s`. This function does not itself check that
    `index`'s particles belong to `candidate.t_s`; a caller iterating candidates across
    multiple historical timestamps must group candidates by `t_s` and call this (or
    `compute_forward_particle_budgets`) once per group with that group's own index.

    A candidate is never rejected/dropped for having `Y < X` -- this function always
    returns a `ForwardBudget` (with `N_forward = Y` in that case), and even `Y == 0`
    still returns a record (with `N_forward = 0` and `forward_validatable = False`)
    rather than raising or silently omitting the candidate.
    """
    core = candidate.core_region
    neighborhood = query_radius(index, core.center_lat, core.center_lon, core.radius_m)
    Y = neighborhood.n_particles
    X = config.X
    N_forward = min(X, Y)
    return ForwardBudget(
        candidate_id=candidate.candidate_id,
        t_s=candidate.t_s,
        X=X,
        Y=Y,
        N_forward=N_forward,
        forward_validatable=Y > 0,
    )


def compute_forward_particle_budgets(
    candidates: list[Candidate],
    index: ParticleNeighborhoodIndex,
    config: ForwardBudgetConfig,
) -> list[ForwardBudget]:
    """`compute_forward_particle_budget` for every candidate in `candidates`, all drawn
    from the SAME historical timestamp / T15 `index` (see that function's docstring for
    the multi-timestamp case: group candidates by `t_s` and call once per group with
    each group's own index). Each candidate's `Y`/`N_forward` is computed independently
    from its own `core_region` -- one candidate's result never affects another's."""
    return [compute_forward_particle_budget(c, index, config) for c in candidates]
