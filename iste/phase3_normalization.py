"""T29 -- Phase 3 cross-time normalization of the T16 support/"KDE" grid.

Implements exactly one specific, documented normalization:

    normalized_support(x, y, t) = Support(x, y, t) / percentile(Support_t[Support_t > 0], P)

applied independently at each analysis timestamp `t`, where `Support_t` is the full T16
support grid (`iste.support_features.compute_support_map`, unmodified, reshaped) evaluated
at that timestamp, and `P` is a configurable percentile (default 95). This is the T29 fix
for exactly the bias T28 measured on the real dataset: the T0 (near-observation) support
grid has a much larger absolute peak than the T72 (72h-backward) grid purely because
particles are still tightly clustered near `t_obs`, not because T0 is intrinsically a
better candidate region.

WHY PERCENTILE-OF-POSITIVE-CELLS, AND WHY 95 BY DEFAULT (the exact normalization used,
documented per CLAUDE.md's "never silently invent" guardrail):

- A support grid is evaluated over a fixed geographic search area that is almost always
  much larger than where the particles at any one instant actually are -- most grid cells
  are structural zeros (no particle within the neighborhood radius), which say nothing
  about that timestamp's *density* and would swamp a percentile computed over the whole
  grid (empirically: a 95th-percentile-of-the-WHOLE-grid can be exactly 0 for a tight
  cluster if fewer than 5% of grid cells have any support at all). Restricting the
  percentile to the grid's own strictly-positive cells uses only the cells that actually
  describe the timestamp's particle distribution.
- The percentile is computed INDEPENDENTLY per timestamp (never pooled across timestamps)
  -- this is exactly what makes different timestamps' concentration levels comparable in
  relative terms afterward.
- 95th percentile (rather than the raw max, i.e. the 100th percentile) is the standard,
  more noise-robust choice -- a single unusually-high grid cell does not by itself set the
  whole timestamp's reference scale. It is directly interpretable: "this point's support is
  X times the timestamp's own high-concentration reference level."
- Empirically verified (see `notebooks/T29_kde_normalization_check.ipynb` and this module's
  own tests): a synthetic tight cluster with an 8x larger raw support peak than a synthetic
  dispersed cluster has, after this normalization, comparable normalized peaks (within
  roughly a factor of 2, vs. the original 8x raw difference) -- demonstrating the
  normalization substantially removes, though does not perfectly equalize, the dispersion-
  driven scale difference. This is a PROTOTYPE/CONFIGURATION choice (`percentile` is fully
  overridable), not a claim of a uniquely correct statistical answer.

This module contains no particle/Stage III/candidate-detection logic; it operates purely on
an already-computed T16 support grid (a plain 2-D `numpy` array) and never recomputes the
support map itself. It does not decide any ranking (that is T36) -- it only produces a
per-timestamp-comparable field.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_PERCENTILE = 95.0


@dataclass(frozen=True)
class NormalizationResult:
    """Result of normalizing one T16 support grid at one timestamp.

    - `reference_value`: `percentile(support_grid[support_grid > 0], percentile)` -- the
      per-timestamp normalization constant. `NaN` if the grid has no strictly-positive
      cell at all (no particle-derived support anywhere at this timestamp).
    - `normalized_grid`: `support_grid / reference_value`, same shape as the input. If
      `reference_value` is `NaN`, this is an all-`NaN` array of the same shape -- never a
      fabricated value.
    - `raw_max`/`normalized_max`: convenience summaries (`NaN` if undefined).
    """

    t_s: np.datetime64
    percentile: float
    n_positive_cells: int
    reference_value: float
    normalized_grid: np.ndarray
    raw_max: float
    normalized_max: float


def normalize_support_grid(
    support_grid: np.ndarray,
    t_s,
    *,
    percentile: float = DEFAULT_PERCENTILE,
) -> NormalizationResult:
    """Normalize one T16 support grid at one timestamp, per this module's documented
    percentile-of-positive-cells formula (see module docstring).

    Parameters
    ----------
    support_grid : np.ndarray
        Any-shaped array of T16 `Support(x,y,t)` values at one timestamp (e.g. the
        `(n_lat, n_lon)` grid the real-data notebook / `iste.phase3_diagnostics` already
        build from `iste.support_features.compute_support_map`). Not recomputed here.
    t_s : datetime64-convertible
        The timestamp this grid belongs to (carried through for provenance only).
    percentile : float
        Percentile in `[0, 100]`, computed over the grid's own strictly-positive cells at
        this timestamp only (never pooled across timestamps). Default 95 (see module
        docstring). Explicit and configurable -- never a hidden constant.

    Returns
    -------
    NormalizationResult
    """
    if not (0.0 <= percentile <= 100.0):
        raise ValueError(f"percentile must be in [0, 100], got {percentile}")

    support_grid = np.asarray(support_grid, dtype=np.float64)
    t_s = np.datetime64(t_s, "ns")

    if support_grid.size == 0:
        return NormalizationResult(
            t_s=t_s, percentile=percentile, n_positive_cells=0, reference_value=float("nan"),
            normalized_grid=support_grid.copy(), raw_max=float("nan"), normalized_max=float("nan"),
        )

    positive = support_grid[support_grid > 0.0]
    raw_max = float(np.max(support_grid))

    if positive.size == 0:
        return NormalizationResult(
            t_s=t_s, percentile=percentile, n_positive_cells=0, reference_value=float("nan"),
            normalized_grid=np.full_like(support_grid, np.nan), raw_max=raw_max,
            normalized_max=float("nan"),
        )

    reference_value = float(np.percentile(positive, percentile))
    if reference_value <= 0.0:
        # possible when `percentile` is very low and most positive cells are tiny/equal to
        # a floor value; never divide by <=0 or fabricate a value.
        return NormalizationResult(
            t_s=t_s, percentile=percentile, n_positive_cells=int(positive.size),
            reference_value=reference_value, normalized_grid=np.full_like(support_grid, np.nan),
            raw_max=raw_max, normalized_max=float("nan"),
        )

    normalized_grid = support_grid / reference_value
    normalized_max = float(np.max(normalized_grid))

    return NormalizationResult(
        t_s=t_s, percentile=percentile, n_positive_cells=int(positive.size),
        reference_value=reference_value, normalized_grid=normalized_grid,
        raw_max=raw_max, normalized_max=normalized_max,
    )


def normalize_support_grids_across_time(
    support_grids_by_t_s: dict,
    *,
    percentile: float = DEFAULT_PERCENTILE,
) -> dict:
    """`normalize_support_grid` applied independently at every timestamp in
    `support_grids_by_t_s` (a mapping of timestamp -> support grid, e.g. exactly the shape
    of the real-data notebook's existing `support_grids_by_t_s` dict). Returns a matching
    dict of `NormalizationResult`, one per timestamp, each normalized using only that
    timestamp's own grid -- never a cross-timestamp-pooled reference."""
    return {
        t_s: normalize_support_grid(grid, t_s, percentile=percentile)
        for t_s, grid in support_grids_by_t_s.items()
    }
