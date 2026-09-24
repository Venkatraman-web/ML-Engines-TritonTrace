"""T34 -- Phase 3 screening-signal normalization.

**Inspection finding this module is built on:** T16's `Support(x,y,t) = f(D,W,V)`
(`iste.support_features.compute_support_map`) is the exact same underlying signal this
project already calls "KDE" throughout Phase 3 (see `memory.md`'s "KDE terminology note").
`Candidate.support_score` (T17) is that same field's value at the candidate's own peak grid
cell, by construction of `find_candidate_peaks`. T29 (`iste.phase3_normalization`) already
built one principled, approved, per-timestamp-relative normalization of this exact signal.

**This module is therefore a thin, provenance-preserving ADAPTER over T29 -- not a second,
independently-derived normalization.** Building a competing normalization of the same raw
signal would create two inconsistent "truths" about identical evidence. `screening_score` is
defined as:

    screening_score(candidate) = candidate.support_score / reference_value(candidate.t_s)

where `reference_value(t)` is exactly T29's `NormalizationResult.reference_value`
(percentile-95-of-positive-grid-cells at that timestamp), obtained via T29's existing
`normalize_support_grid`, imported and reused -- never reimplemented. Because
`candidate.support_score` already equals the support grid's own value at the candidate's
peak cell, this is mathematically identical to reading T29's own `normalized_grid` off at
the candidate's position.

**Explicitly NOT done here (see `progress.md`'s T34 design specification for the full
reasoning):**
- NOT forced into `[0,1]` -- T29's own natural range (candidates are peaks, so
  `screening_score` typically lands `>= ~1`) is preserved as-is; a second rescale would add
  an unjustified extra assumption with no new information.
- NOT treated as independent evidence from T29 for T36's purposes -- T36 must combine
  `screening_score` and T29's normalized KDE as ONE evidence axis, not two.
- NOT a place to reconcile the very different ranges across T29-T34 (~`[0,1]` for T30/T31,
  ~`[0,1]`-ish for T32, unbounded above for T33, `>=1`-ish here) -- explicitly deferred to
  T36's own holistic combination design.
- NO age term anywhere -- `screening_score` is a pure function of `(support_score,
  reference_value)`; no `age`/`t_obs` parameter exists in this module. Any age correlation
  observed on real data is reported as a diagnostic (per the standing Phase 3 discipline of
  T29-T33), never corrected or suppressed.
- Does NOT modify `iste/support_features.py` (T16) or `iste/phase3_normalization.py` (T29).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from iste.candidate_extraction import Candidate
from iste.phase3_normalization import DEFAULT_PERCENTILE, NormalizationResult, normalize_support_grid


@dataclass(frozen=True)
class ScreeningScore:
    """T34 screening evidence for one candidate -- a candidate-level packaging of T29's
    own per-timestamp normalization, with full provenance."""

    candidate_id: str
    t_s: np.datetime64
    raw_support_score: float
    reference_value: float  # T29's own reference_value for this candidate's timestamp
    screening_score: float  # raw_support_score / reference_value; NaN iff reference_value is NaN/<=0
    density_score: float  # D, carried through unchanged from Candidate
    weighted_support_score: float  # W, carried through unchanged
    pixel_diversity_score: float  # V, carried through unchanged


def compute_screening_score(
    candidate: Candidate,
    normalization_result: NormalizationResult,
) -> ScreeningScore:
    """Compute T34 screening evidence for one candidate against its own timestamp's
    already-computed T29 `NormalizationResult` (see `iste.phase3_normalization.
    normalize_support_grid`, reused unchanged).

    Parameters
    ----------
    candidate : Candidate
        A T17 candidate. `support_score` (T16's own `Support` at the peak) is used directly.
    normalization_result : NormalizationResult
        T29's own result for the SAME timestamp as `candidate.t_s` -- raises `ValueError` if
        the timestamps do not match (never silently mixes a candidate with another
        timestamp's reference).
    """
    candidate_t_s = np.datetime64(candidate.t_s, "ns")
    if candidate_t_s != normalization_result.t_s:
        raise ValueError(
            f"candidate.t_s ({candidate_t_s}) does not match normalization_result.t_s "
            f"({normalization_result.t_s})"
        )

    reference_value = normalization_result.reference_value
    if np.isnan(reference_value) or reference_value <= 0.0:
        screening_score = float("nan")
    else:
        screening_score = float(candidate.support_score) / reference_value

    return ScreeningScore(
        candidate_id=candidate.candidate_id, t_s=candidate_t_s,
        raw_support_score=float(candidate.support_score), reference_value=float(reference_value),
        screening_score=screening_score,
        density_score=float(candidate.density_score),
        weighted_support_score=float(candidate.weighted_support_score),
        pixel_diversity_score=float(candidate.pixel_diversity_score),
    )


def compute_screening_scores_for_candidates(
    candidates: list[Candidate],
    support_grids_by_t_s: dict,
    *,
    percentile: float = DEFAULT_PERCENTILE,
) -> list[ScreeningScore]:
    """`compute_screening_score` evaluated for every candidate in `candidates`, in order.
    T29's `normalize_support_grid` is called once per DISTINCT `t_s` present in `candidates`
    (looked up from `support_grids_by_t_s`, the same `{t_s: grid}` structure T29's own
    `normalize_support_grids_across_time` consumes) and reused across every candidate
    sharing that timestamp -- no redundant normalization, mirroring T30/T31/T32/T33's own
    batch convention.

    Parameters
    ----------
    candidates : list[Candidate]
        T17 candidates, any/all timestamps.
    support_grids_by_t_s : dict
        Mapping of timestamp -> the T16 support grid (plain 2-D array) at that timestamp,
        e.g. exactly the `support_grids_by_t_s` dict the real-data notebook already builds.
        Must contain an entry for every distinct `t_s` present in `candidates`.
    percentile : float
        Passed through to T29's `normalize_support_grid` (default 95, T29's own default).
    """
    results_by_t_s: dict[np.datetime64, NormalizationResult] = {}
    results = []
    for c in candidates:
        t_s = np.datetime64(c.t_s, "ns")
        if t_s not in results_by_t_s:
            if t_s not in support_grids_by_t_s:
                raise KeyError(f"support_grids_by_t_s has no entry for timestamp {t_s}")
            results_by_t_s[t_s] = normalize_support_grid(
                support_grids_by_t_s[t_s], t_s, percentile=percentile
            )
        results.append(compute_screening_score(c, results_by_t_s[t_s]))
    return results
