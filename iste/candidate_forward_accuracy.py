"""T24 -- Stage VII: candidate-level forward accuracy (aggregation over forcing
members).

Implements exactly `README.md` Sec. 23: for one candidate at one historical timestamp,
aggregate its T23 per-forcing-member `ForwardAccuracyResult`s into a single
candidate-level screening score, while retaining every member-level result unchanged:

    candidate_forward_error       = arithmetic mean of VALID per-member errors
    candidate_forward_consistency = arithmetic mean of VALID per-member consistency scores

"This is a Phase 2 physical-consistency SCREENING score. It is not the frozen Stage IV
likelihood and it is not the later forcing-ensemble marginalization" (`README.md`
Sec. 23) -- a plain, unweighted arithmetic mean over valid members, nothing more.

**Invalid members are never silently turned into a numeric score.** A member whose T23
`ForwardAccuracyResult.valid` is `False` (no valid predicted/observed footprint,
recorded by T23's own `invalid_reason`) is excluded from the arithmetic mean entirely
and instead recorded, by member id, in `CandidateForwardAccuracy.invalid_member_reasons`
-- never averaged in as `0`, `NaN`-treated-as-zero, or any other fabricated value.

**If NO forcing member is valid, the candidate is explicitly marked
`forward_validation_status = "unvalidated"`** (`candidate_forward_error`/
`candidate_forward_consistency` both `NaN`) rather than producing any numeric aggregate
-- `README.md`: "If no forcing member produces a valid result, the candidate is
forward-unvalidated and is rejected."

**No hidden weighting.** This module's only input is a `dict[int,
iste.forward_accuracy.ForwardAccuracyResult]` for one candidate/time -- it has no
access to, and never reads, any T16/T17 backward-KDE support field
(`support_score`/`density_score`/`weighted_support_score`/`pixel_diversity_score`) or
Stage III `c_hat*`. The aggregation is structurally incapable of mixing backward
support into the forward error/consistency, because it never receives that data at all.

Does not implement candidate filtering/ranking (T25), formal likelihood/posterior
inference, or AIS. Does not modify `iste.forward_accuracy` (T23) or any earlier Phase 2
module -- every member's `ForwardAccuracyResult` is carried through byte-for-byte
unchanged in `CandidateForwardAccuracy.member_results`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from iste.forward_accuracy import ForwardAccuracyResult


@dataclass
class CandidateForwardAccuracy:
    """One candidate/time's aggregated forward-validation result.

    `member_results` is every T23 `ForwardAccuracyResult` handed to
    `aggregate_candidate_forward_accuracy`, unchanged, keyed by
    `forward_forcing_member_id` -- the full, authoritative per-member provenance.
    `forward_error_by_member`/`forward_consistency_by_member`/
    `forcing_members_validated` cover ONLY the members that were actually valid (never
    include a `NaN`/fabricated placeholder for an invalid one); `invalid_member_reasons`
    covers exactly the complementary (invalid) set. Every member id therefore appears in
    EXACTLY ONE of `forcing_members_validated` or `invalid_member_reasons.keys()`.
    """

    candidate_id: str
    t_s: np.datetime64
    t_obs: np.datetime64
    member_results: dict[int, ForwardAccuracyResult]
    forcing_members_validated: tuple[int, ...]
    forward_error_by_member: dict[int, float]
    forward_consistency_by_member: dict[int, float]
    invalid_member_reasons: dict[int, str]
    candidate_forward_error: float
    candidate_forward_consistency: float
    forward_validation_status: str  # "validated" | "unvalidated"


def aggregate_candidate_forward_accuracy(
    member_results: dict[int, ForwardAccuracyResult],
) -> CandidateForwardAccuracy:
    """Aggregate one candidate/time's per-member T23 results into a
    `CandidateForwardAccuracy`.

    `member_results` must be non-empty, keyed by `forcing_member_id`, with every value's
    own `candidate_id`/`t_s`/`t_obs`/`forward_forcing_member_id` consistent with its key
    and with every other entry (all members of one call must belong to the SAME
    candidate/time) -- raises `ValueError` otherwise. An empty dict is a caller error
    (there is no candidate/time to attribute a result to), distinct from "every member
    present but invalid" (handled explicitly as `forward_validation_status =
    "unvalidated"`, never an exception).

    Member order in the input dict never affects the result (the arithmetic mean is
    order-independent, and `forcing_members_validated` is always returned sorted).
    """
    if not member_results:
        raise ValueError("member_results must contain at least one member's ForwardAccuracyResult")

    candidate_ids = {r.candidate_id for r in member_results.values()}
    t_s_values = {r.t_s for r in member_results.values()}
    t_obs_values = {r.t_obs for r in member_results.values()}
    if len(candidate_ids) != 1 or len(t_s_values) != 1 or len(t_obs_values) != 1:
        raise ValueError(
            "all member_results must share the same candidate_id/t_s/t_obs "
            f"(got candidate_ids={candidate_ids}, t_s={t_s_values}, t_obs={t_obs_values})"
        )
    for member_id, result in member_results.items():
        if result.forward_forcing_member_id != member_id:
            raise ValueError(
                f"member_results key {member_id!r} does not match its own value's "
                f"forward_forcing_member_id {result.forward_forcing_member_id!r}"
            )

    candidate_id = candidate_ids.pop()
    t_s = t_s_values.pop()
    t_obs = t_obs_values.pop()

    valid_ids = sorted(member_id for member_id, r in member_results.items() if r.valid)
    invalid_reasons = {
        member_id: r.invalid_reason for member_id, r in member_results.items() if not r.valid
    }
    error_by_member = {member_id: member_results[member_id].error for member_id in valid_ids}
    consistency_by_member = {member_id: member_results[member_id].consistency for member_id in valid_ids}

    if valid_ids:
        candidate_forward_error = float(np.mean([error_by_member[m] for m in valid_ids]))
        candidate_forward_consistency = float(np.mean([consistency_by_member[m] for m in valid_ids]))
        status = "validated"
    else:
        candidate_forward_error = float("nan")
        candidate_forward_consistency = float("nan")
        status = "unvalidated"

    return CandidateForwardAccuracy(
        candidate_id=candidate_id,
        t_s=t_s,
        t_obs=t_obs,
        member_results=dict(member_results),  # shallow copy -- never mutate the caller's dict
        forcing_members_validated=tuple(valid_ids),
        forward_error_by_member=error_by_member,
        forward_consistency_by_member=consistency_by_member,
        invalid_member_reasons=invalid_reasons,
        candidate_forward_error=candidate_forward_error,
        candidate_forward_consistency=candidate_forward_consistency,
        forward_validation_status=status,
    )
