"""T10 -- reusable Stage III diagnostics module.

Turns the minimum Stage III diagnostics already exercised ad hoc inside T09's
`iste.ensemble_run` into a small, independently tested, reusable module:

    total_particles
    valid_particles
    invalid_particles
    total_weight
    effective_sample_size
    valid_particle_fraction

This module is diagnostics-only. It does not run trajectories, does not touch
`iste.step`/`iste.trajectory`/`iste.ensemble_run`'s propagation logic, and does not know
anything about `GriddedFieldSource`, forcing members, or checkpoints -- it operates on
plain `weights`/`valid` arrays that some caller (e.g. `iste.ensemble_run`, or T13's
sign-off notebook) has already selected (for example, one array per particle at a chosen
checkpoint). `iste.ensemble_run.compute_diagnostics`/`compute_diagnostics_by_member` call
into this module rather than recomputing effective-sample-size arithmetic themselves, so
the formula/edge-case handling below is defined in exactly one place.

Which particles contribute to each diagnostic (explicit, per requirement 4):

- `total_particles` counts every row passed in, valid or not.
- `valid_particles`/`invalid_particles` split that count by the `valid` flag alone.
- `total_weight` and `effective_sample_size` are computed from **valid particles'
  weights only** -- an invalid particle's stored weight (whatever it happens to be,
  including a frozen/stale value or, if upstream ever produced one, a non-finite value)
  never contributes. This mirrors T04/T05/T06's own policy that an invalid particle's
  state is frozen for bookkeeping, not a live estimate.
- `valid_particle_fraction` is `valid_particles / total_particles`.

Effective sample size (Kish's formula, standard, not invented):

    ESS = (sum(w))^2 / sum(w^2)

computed over valid particles' weights only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class StageIIIDiagnostics:
    """The minimum Stage III diagnostics (`README.md`/`tasks.md` T10)."""

    total_particles: int
    valid_particles: int
    invalid_particles: int
    total_weight: float
    effective_sample_size: float
    valid_particle_fraction: float


def compute_stage3_diagnostics(weights: np.ndarray, valid: np.ndarray) -> StageIIIDiagnostics:
    """Compute Stage III diagnostics from explicit `weights` (linear, i.e. `exp(log_weight)`
    -- NOT log-weight) and `valid` arrays of equal length, one entry per particle.

    Edge cases, handled explicitly rather than left to raise an opaque numpy error or to
    silently produce a misleading number:

    - **empty input** (`len(weights) == 0`): `total_particles = valid_particles =
      invalid_particles = 0`, `total_weight = 0.0`, `effective_sample_size = 0.0`,
      `valid_particle_fraction = nan` (0/0 is genuinely undefined -- distinct from "zero
      valid particles out of some").
    - **all invalid**: `total_weight = 0.0`, `effective_sample_size = 0.0` (no valid
      weights to concentrate), `valid_particle_fraction = 0.0` (well-defined: 0 out of a
      known nonzero total).
    - **zero total weight** (valid particles all weight `0.0`): `effective_sample_size =
      0.0`, not `0/0` -- division by zero is guarded explicitly rather than producing NaN.
    - **non-finite weights**: a non-finite (`NaN`/`Inf`) weight on a particle marked
      **invalid** is ignored (never contributes, like any other invalid particle's
      weight). A non-finite weight on a particle marked **valid** raises `ValueError` --
      silently summing a NaN/Inf into `total_weight`/`effective_sample_size` would either
      poison the whole diagnostic (NaN) or silently overstate it (Inf); per
      `CLAUDE.md`'s "do not silently invent," an upstream bug that produced this must
      surface, not be absorbed into a plausible-looking number.

    Never treats an invalid particle as valid: `valid` is taken at face value and is the
    sole partition between "contributes to weight-based diagnostics" and "counted only in
    `total_particles`/`invalid_particles`."
    """
    weights = np.asarray(weights, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    if weights.shape != valid.shape:
        raise ValueError(f"weights and valid must have the same shape, got {weights.shape} and {valid.shape}")

    total_particles = int(weights.shape[0])
    valid_particles = int(np.sum(valid))
    invalid_particles = total_particles - valid_particles
    valid_particle_fraction = (valid_particles / total_particles) if total_particles > 0 else float("nan")

    valid_weights = weights[valid]
    if valid_weights.size and not np.all(np.isfinite(valid_weights)):
        raise ValueError(
            "non-finite weight (NaN or Inf) found among particles marked valid -- "
            "diagnostics require every valid particle's weight to be finite; fix the "
            "upstream source of the non-finite weight rather than silently ignoring it "
            "(a non-finite weight on an INVALID particle is fine and is ignored, since "
            "invalid particles never contribute to weight-based diagnostics)."
        )

    total_weight = float(np.sum(valid_weights)) if valid_weights.size else 0.0
    if valid_weights.size and total_weight > 0.0:
        effective_sample_size = float((np.sum(valid_weights) ** 2) / np.sum(valid_weights ** 2))
    else:
        effective_sample_size = 0.0

    return StageIIIDiagnostics(
        total_particles=total_particles,
        valid_particles=valid_particles,
        invalid_particles=invalid_particles,
        total_weight=total_weight,
        effective_sample_size=effective_sample_size,
        valid_particle_fraction=valid_particle_fraction,
    )


def compute_stage3_diagnostics_from_log_weights(log_weights: np.ndarray, valid: np.ndarray) -> StageIIIDiagnostics:
    """Convenience wrapper for callers that track weight in log-space (T05/T06/T09's own
    numerical-stability convention, `weight = exp(log_weight)`). Converts to linear weight
    and delegates to `compute_stage3_diagnostics` -- the edge-case handling above (in
    particular: a non-finite value only matters if the corresponding particle is valid)
    applies identically here."""
    log_weights = np.asarray(log_weights, dtype=float)
    with np.errstate(over="ignore"):
        weights = np.exp(log_weights)
    return compute_stage3_diagnostics(weights, valid)
