"""T25 -- Stage VII: primary forward-accuracy candidate filtering and ranking.

Implements exactly `README.md` Sec. 24: forward-propagation accuracy is the PRIMARY
candidate-filtering and ranking signal, applied to candidates that already carry T24's
candidate-level forward accuracy --

    KDE candidate found -> forward propagation -> predicted slick -> actual-slick
    agreement -> forward accuracy -> PRIMARY filter/ranking

**Backward KDE support answers "why was this candidate proposed?"; forward accuracy
answers "does this candidate reproduce the observed slick well enough to survive?"**
This module's filtering/ranking functions never read `support_score`/`density_score`/
`weighted_support_score`/`pixel_diversity_score` for any filtering/ranking DECISION --
those fields are carried through on `CandidateRecord`/`CandidateDecision` purely as
preserved provenance (`README.md` Sec. 25's output-artifact fields), never as an input
to `_filter`/`_rank`/`_merge`. `README.md` Sec. 24: "A candidate with weaker KDE support
may outrank a stronger KDE candidate if it has substantially better forward agreement" --
and the reverse holds unconditionally here, structurally, not just by convention: there
is no code path in this module through which a backward support value could influence
an accept/reject decision or a rank number. "Do not silently combine backward support
into the forward error" (Sec. 24) -- an explicit backward PREFILTER, if one is ever
configured, is described by `README.md` as happening BEFORE forward validation (i.e. at
candidate-selection time, T17/T18's scope), never inside this module, which only ever
sees candidates that already completed forward validation (T21-T24).

**Filtering modes** (`ScreeningConfig.mode`, exactly one, explicit, required):

    "threshold"  -- reject every candidate whose `candidate_forward_error` exceeds
                    `error_threshold`; retain the rest (`<=` survives).
    "top_k"      -- retain exactly the `top_k` VALIDATED candidates with the lowest
                    `candidate_forward_error` (fewer than `top_k` valid candidates ->
                    all of them are retained, never an error).

**Unvalidated candidates** (T24 `forward_validation_status == "unvalidated"`, i.e. no
forcing member produced a valid forward result) are rejected immediately, before either
filtering mode runs, and NEVER receive a rank or a numeric score -- `README.md`/this
task: "candidates with no valid forcing-member result remain forward-unvalidated and
must not receive a fabricated rank/score."

**Near-duplicate merging happens strictly AFTER filtering, on survivors only**, within
each historical timestamp `t_s` group independently (candidates from different `t_s`
are never merged into each other, regardless of spatial proximity -- see
`README.md`/this task: "preserve distinct spatial modes and source times ... do not
collapse all candidates to a single source/time"), using an explicit configurable metre
separation threshold (`ScreeningConfig.merge_separation_m`, default `0.0` = no merging).
Distances are computed via `iste.kdtree_neighborhood.project_to_local_metres` (T15's
distance-safe local projection, reused unchanged), and the merge is a greedy,
forward-error-ascending non-maximum suppression -- the SAME structural pattern T17's own
`find_candidate_peaks` non-max suppression uses (reused pattern, independently
implemented for this module's own coordinate/ranking data, not imported from T17's
private helpers), just on candidate positions/forward-error instead of grid cells/
support.

**Final ranks are assigned to the post-merge survivor set** (`rank` is contiguous,
starting at `1`, in ascending `candidate_forward_error` order, tie-broken by
`candidate_id` for determinism) -- a candidate merged away as a near-duplicate never
receives a rank, exactly like a threshold/top-K-rejected or unvalidated candidate; its
`CandidateDecision.rejection_reason` instead names the surviving candidate it was merged
into, for auditability.

Does not implement final formal source inference, AIS attribution, a likelihood, or a
posterior -- "the final Phase 2 artifact is a forward-screened candidate set, not a
posterior" (`README.md` Sec. 25). Never reads or references Stage III `c_hat*`. Does not
modify `iste.candidate_extraction` (T17), `iste.forward_budget` (T18),
`iste.core_sampling` (T19), or `iste.candidate_forward_accuracy` (T24) -- this module
only reads their existing public fields via `build_candidate_record`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from iste.candidate_extraction import Candidate, CoreRegion
from iste.candidate_forward_accuracy import CandidateForwardAccuracy
from iste.forward_budget import ForwardBudget
from iste.kdtree_neighborhood import project_to_local_metres

VALID_MODES = ("threshold", "top_k")


@dataclass(frozen=True)
class CandidateRecord:
    """Complete per-candidate provenance (`README.md` Sec. 25's output-artifact field
    list) gathered from earlier Phase 2 stages -- T17 (backward candidate/support), T18
    (forward particle budget), T19 (core-sampling seed), and T24 (candidate-level
    forward accuracy). Built via `build_candidate_record` below, or directly for tests/
    synthetic scenarios; this module never recomputes any of these values itself."""

    candidate_id: str
    x_peak: float
    y_peak: float
    t_s: np.datetime64
    support_score: float
    density_score: float
    weighted_support_score: float
    pixel_diversity_score: float
    core_definition: CoreRegion
    core_particle_count_Y: int
    forward_particle_count_N_forward: int
    sampling_seed: int
    forcing_members_validated: tuple[int, ...]
    forward_error_by_member: dict[int, float]
    forward_consistency_by_member: dict[int, float]
    candidate_forward_error: float
    candidate_forward_consistency: float
    forward_validation_status: str  # "validated" | "unvalidated" (T24)


def build_candidate_record(
    candidate: Candidate, budget: ForwardBudget, sampling_seed: int, forward_accuracy: CandidateForwardAccuracy,
) -> CandidateRecord:
    """Assemble one `CandidateRecord` from the actual T17/T18/T19/T24 objects it
    describes, checking they all agree on `candidate_id`/`t_s` (raises `ValueError`
    otherwise -- a caller wiring bug, never silently reconciled)."""
    if not (candidate.candidate_id == budget.candidate_id == forward_accuracy.candidate_id):
        raise ValueError(
            "candidate/budget/forward_accuracy candidate_id mismatch: "
            f"{candidate.candidate_id!r} / {budget.candidate_id!r} / {forward_accuracy.candidate_id!r}"
        )
    t_s = np.datetime64(candidate.t_s, "ns")
    if not (t_s == np.datetime64(budget.t_s, "ns") == np.datetime64(forward_accuracy.t_s, "ns")):
        raise ValueError(
            f"candidate/budget/forward_accuracy t_s mismatch: {candidate.t_s!r} / {budget.t_s!r} / {forward_accuracy.t_s!r}"
        )
    return CandidateRecord(
        candidate_id=candidate.candidate_id,
        x_peak=candidate.x_peak,
        y_peak=candidate.y_peak,
        t_s=t_s,
        support_score=candidate.support_score,
        density_score=candidate.density_score,
        weighted_support_score=candidate.weighted_support_score,
        pixel_diversity_score=candidate.pixel_diversity_score,
        core_definition=candidate.core_region,
        core_particle_count_Y=budget.Y,
        forward_particle_count_N_forward=budget.N_forward,
        sampling_seed=sampling_seed,
        forcing_members_validated=forward_accuracy.forcing_members_validated,
        forward_error_by_member=dict(forward_accuracy.forward_error_by_member),
        forward_consistency_by_member=dict(forward_accuracy.forward_consistency_by_member),
        candidate_forward_error=forward_accuracy.candidate_forward_error,
        candidate_forward_consistency=forward_accuracy.candidate_forward_consistency,
        forward_validation_status=forward_accuracy.forward_validation_status,
    )


@dataclass(frozen=True)
class ScreeningConfig:
    """Explicit, validated Phase 2 filtering/ranking configuration.

    - `mode`: `"threshold"` or `"top_k"` (required, no default).
    - `error_threshold`: required and must be finite when `mode == "threshold"`;
      candidates with `candidate_forward_error <= error_threshold` survive.
    - `top_k`: required and must be a positive int when `mode == "top_k"`; the `top_k`
      lowest-error VALIDATED candidates survive (fewer, if fewer than `top_k` exist).
    - `merge_separation_m`: metres; near-duplicate survivors within this distance of a
      lower-error survivor, AT THE SAME `t_s`, are merged into it. Default `0.0` (no
      merging). Must be finite and `>= 0`.
    """

    mode: str
    error_threshold: float | None = None
    top_k: int | None = None
    merge_separation_m: float = 0.0

    def __post_init__(self) -> None:
        if self.mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {VALID_MODES}; got {self.mode!r}")
        if self.mode == "threshold":
            if self.error_threshold is None or not np.isfinite(self.error_threshold):
                raise ValueError(f"error_threshold must be a finite number when mode='threshold'; got {self.error_threshold!r}")
        if self.mode == "top_k":
            if self.top_k is None or self.top_k <= 0:
                raise ValueError(f"top_k must be a positive int when mode='top_k'; got {self.top_k!r}")
        if not np.isfinite(self.merge_separation_m) or self.merge_separation_m < 0.0:
            raise ValueError(f"merge_separation_m must be finite and >= 0; got {self.merge_separation_m!r}")


@dataclass
class CandidateDecision:
    """One candidate's final screening outcome: the complete, unmodified
    `CandidateRecord` plus `accepted`/`rank`/`rejection_reason`. `rank` is `None` for
    every rejected candidate (unvalidated, threshold/top-K-rejected, or merged away) --
    never a fabricated number."""

    record: CandidateRecord
    accepted: bool
    rank: int | None
    rejection_reason: str


def _metres_between(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    x, y = project_to_local_metres(lat2, lon2, lat_ref=lat1, lon_ref=lon1)
    return float(np.sqrt(x[0] ** 2 + y[0] ** 2))


def _sort_key(record: CandidateRecord) -> tuple[float, str]:
    """Ascending forward error, tie-broken by `candidate_id` -- deterministic
    regardless of input order."""
    return (record.candidate_forward_error, record.candidate_id)


def _merge_near_duplicates(
    survivors: list[CandidateRecord], separation_m: float,
) -> tuple[list[CandidateRecord], list[CandidateDecision]]:
    """Greedy, forward-error-ascending non-maximum suppression within each `t_s` group
    independently -- never merges candidates from different historical timestamps. `[]`
    merge decisions and the input list unchanged if `separation_m <= 0`."""
    if separation_m <= 0.0 or len(survivors) <= 1:
        return list(survivors), []

    kept: list[CandidateRecord] = []
    merged_away: list[CandidateDecision] = []
    by_t_s: dict[np.datetime64, list[CandidateRecord]] = {}
    for record in survivors:
        by_t_s.setdefault(record.t_s, []).append(record)

    for t_s in sorted(by_t_s.keys()):
        ordered = sorted(by_t_s[t_s], key=_sort_key)  # best (lowest error) first
        group_kept: list[CandidateRecord] = []
        for record in ordered:
            winner = next(
                (k for k in group_kept if _metres_between(record.y_peak, record.x_peak, k.y_peak, k.x_peak) < separation_m),
                None,
            )
            if winner is None:
                group_kept.append(record)
            else:
                merged_away.append(
                    CandidateDecision(
                        record=record, accepted=False, rank=None,
                        rejection_reason=(
                            f"merged into near-duplicate candidate {winner.candidate_id!r} "
                            f"(same t_s, separation < {separation_m} m)"
                        ),
                    )
                )
        kept.extend(group_kept)

    return kept, merged_away


def filter_and_rank_candidates(
    records: list[CandidateRecord], config: ScreeningConfig,
) -> list[CandidateDecision]:
    """`README.md` Sec. 24's primary forward-accuracy filter/ranking, over every
    candidate in `records` (any mix of historical timestamps/spatial modes -- all
    preserved; see module docstring). Returns exactly one `CandidateDecision` per input
    record (nothing added, nothing dropped), in no particular order (sort by `rank` for
    the accepted ranking, or filter by `accepted`/`rejection_reason` for an audit view).
    """
    decisions: list[CandidateDecision] = []
    validated: list[CandidateRecord] = []
    for record in records:
        if record.forward_validation_status != "validated":
            decisions.append(
                CandidateDecision(
                    record=record, accepted=False, rank=None,
                    rejection_reason="forward-unvalidated (no valid forcing-member result)",
                )
            )
        else:
            validated.append(record)

    if config.mode == "threshold":
        survivors = [r for r in validated if r.candidate_forward_error <= config.error_threshold]
        for r in validated:
            if r.candidate_forward_error > config.error_threshold:
                decisions.append(
                    CandidateDecision(
                        record=r, accepted=False, rank=None,
                        rejection_reason=(
                            f"forward error {r.candidate_forward_error:.6g} exceeds "
                            f"configured threshold {config.error_threshold:.6g}"
                        ),
                    )
                )
    else:  # "top_k"
        ordered = sorted(validated, key=_sort_key)
        survivors = ordered[: config.top_k]
        for r in ordered[config.top_k :]:
            decisions.append(
                CandidateDecision(
                    record=r, accepted=False, rank=None,
                    rejection_reason=f"outside configured top_k={config.top_k}",
                )
            )

    merged_survivors, merge_decisions = _merge_near_duplicates(survivors, config.merge_separation_m)
    decisions.extend(merge_decisions)

    ranked = sorted(merged_survivors, key=_sort_key)
    for rank, record in enumerate(ranked, start=1):
        decisions.append(CandidateDecision(record=record, accepted=True, rank=rank, rejection_reason=""))

    return decisions
