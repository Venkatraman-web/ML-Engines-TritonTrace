"""T36 -- Phase 3 composite multi-feature source score.

Implements the T36 design specification exactly as approved (see `progress.md`'s T36
"DESIGN SPECIFICATION" + "DESIGN AMENDMENTS" sections). Combines T29/T31/T32/T33/T34/T35's
already-computed, already-approved candidate-level evidence into one configurable,
transparent `source_score` per candidate. **Introduces NO new normalization/transformation
of any evidence value** -- every evidence quantity used here is read directly from its own
already-approved module's own output object; this module's only new computation is the
weighted combination itself.

**Component wiring (design spec section 10):**

    support_evidence(c)          = screening_score(c)                    [T29/T34, ONE axis]
    persistence_evidence(c)      = temporal_continuity(track(c))          [T31]
    convergence_evidence(c)      = convergence_score(c)                   [T32]
    dispersion_evidence(c)       = dispersion_aware_score(c)              [T33]
    ais_evidence(c)              = ais_score(c) if available else None    [T35]
    initial_bias_evidence(c)     = initial_bias_influence(c)   -- DIAGNOSTIC ONLY (T30)
    persistence_count_evidence(c) = persistence_count(track(c)) -- DIAGNOSTIC ONLY (T31)

**T29/T34 (design spec section 6):** `support_evidence` is `iste.phase3_screening.
ScreeningScore.screening_score` -- already, by construction, `candidate.support_score /
T29_reference_value`. T29's grid-level `normalized_support` is never separately re-entered;
there is exactly one weight (`w_support`) for this entire evidence lineage.

**T30 (design spec Amendment 2):** `initial_bias_influence` (Mahalanobis consistency with
the timestamp's own particle ensemble) has NO scientifically justified DIRECTIONAL
interpretation as source evidence -- assigning `+w_initial` or `-w_initial` would import a
directional claim T30 was never designed or validated to support. This is the PRIMARY reason
`w_initial = 0.0` by default. Its strong empirical correlation with `support_evidence`/
`convergence_evidence` (see `progress.md`) is SECONDARY, supporting evidence against
double-counting if a direction were ever established -- not the primary reason. The value is
still computed and reported on every result (diagnostic field), never dropped.

**T33 (design spec Amendment 3):** `dispersion_evidence` is scored positively with NO age
correction of any kind (no `age`/`t_obs` parameter exists anywhere in this module or in
`iste.phase3_dispersion`). Its contribution to `source_score` is evidence-ranking
information, not a calibrated probability, and not an age-neutral likelihood -- both of
T33's own established age-correlation findings (strong at the top-candidate level, weak at
the population level) persist into the composite unmodified; neither is corrected here.

**AIS (design spec Amendment 1):** `ais_evidence` is `None` whenever
`iste.phase3_ais.compute_ais_evidence` returns `ais_available=False` (always, in this
project today -- no AIS data exists). Combined via T35's own `renormalize_weighted_mean`,
reused unchanged:

    AVAILABLE(c)    = { i : evidence_i(c) is not None }
    w_eff_i(c)      = w_i / sum_{j in AVAILABLE(c)} w_j   for i in AVAILABLE(c), else 0
    source_score(c) = sum_{i in AVAILABLE(c)} w_eff_i(c) * evidence_i(c)

AIS absence is EXCLUDED from the denominator, never treated as `0` evidence. Every
`SourceScore` reports both the CONFIGURED weight vector and the EFFECTIVE (`w_eff`) vector
actually used for that candidate, so renormalization is visible in the data, not just
asserted in prose.

**`NaN` handling (a gap explicitly flagged in the design spec's "Known limitations" and
fixed here at implementation time, without modifying T30/T33 themselves):** `initial_bias_
influence`/`dispersion_aware_score` can return `NaN` for degenerate-covariance timestamps.
Because `0.0 * float('nan')` is `nan` in IEEE arithmetic, an unconverted `NaN` would poison
the whole weighted sum even at zero weight. This module converts `NaN` to `None` before
building the weighted-combination input (never before populating the reported diagnostic
fields, which always show the raw, possibly-`NaN` value for full transparency).

**`source_score` is an evidence-ranking score for comparing candidates within one run/
configuration. It is NOT a probability, confidence, likelihood, or certainty of source
identity**, and has no fixed upper bound (inherits `support_evidence`'s/`dispersion_
evidence`'s unboundedness). A high `source_score` is not proof of actual physical source
identity, exactly like every other Phase 2/3 screening quantity in this project.

T25's separate forward-accuracy filtering/ranking is untouched and not folded into this
score. The Stage III physical backtracking model (T00-T13.5) and T14-T26 are not read or
modified by this module.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from iste.candidate_extraction import Candidate
from iste.ensemble_run import EnsembleRunRecords
from iste.phase3_ais import AISDataSource, AISEvidence, compute_ais_evidence, renormalize_weighted_mean
from iste.phase3_convergence import ConvergenceScore, compute_convergence_scores_for_candidates
from iste.phase3_dispersion import DispersionAwareScore, compute_dispersion_aware_scores_for_candidates
from iste.phase3_initial_bias import (
    InitialBiasInfluence,
    compute_initial_bias_influence,
    compute_timestamp_cloud_stats,
)
from iste.phase3_persistence import PersistenceTrack, track_candidate_persistence
from iste.phase3_screening import ScreeningScore, compute_screening_scores_for_candidates

EVIDENCE_RANKING_DISCLAIMER = (
    "source_score is an evidence-ranking score for comparing candidates within one run/"
    "configuration. It is NOT a probability, confidence, likelihood, or certainty of "
    "source identity, and a high score is not proof of actual physical source identity."
)

_WEIGHT_FIELDS = (
    "w_support", "w_persistence", "w_convergence", "w_dispersion",
    "w_ais", "w_initial", "w_persistence_count",
)


@dataclass(frozen=True)
class CompositeWeights:
    """T36 configured weights (design spec section 11). Heuristic and configurable --
    NOT claimed as scientifically validated. Defaults sum to `1.0`; `w_initial` and
    `w_persistence_count` default to `0.0` (diagnostic-only, see module docstring)."""

    w_support: float = 0.35
    w_persistence: float = 0.20
    w_convergence: float = 0.10
    w_dispersion: float = 0.15
    w_ais: float = 0.20
    w_initial: float = 0.0
    w_persistence_count: float = 0.0

    def __post_init__(self) -> None:
        for name in _WEIGHT_FIELDS:
            value = getattr(self, name)
            if not (np.isfinite(value) and value >= 0.0):
                raise ValueError(f"{name} must be finite and >= 0, got {value!r}")

    def as_dict(self) -> dict[str, float]:
        return {
            "support": self.w_support, "persistence": self.w_persistence,
            "convergence": self.w_convergence, "dispersion": self.w_dispersion,
            "ais": self.w_ais, "initial_bias": self.w_initial,
            "persistence_count": self.w_persistence_count,
        }


DEFAULT_WEIGHTS = CompositeWeights()

# Design spec section 13 -- proposed from evidence-role/scale reasoning, not tuned against
# any candidate outcome.
PERSISTENCE_EMPHASIS_WEIGHTS = CompositeWeights(
    w_support=0.25, w_persistence=0.40, w_convergence=0.10, w_dispersion=0.10, w_ais=0.15,
)
DISPERSION_EMPHASIS_WEIGHTS = CompositeWeights(
    w_support=0.25, w_persistence=0.15, w_convergence=0.10, w_dispersion=0.35, w_ais=0.15,
)


def _nan_to_none(value: float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def _effective_weights(components: dict[str, tuple[float, float | None]]) -> dict[str, float]:
    """Reporting-only computation of `w_eff` (design spec Amendment 1) -- mirrors exactly
    the AVAILABLE/renormalization logic `renormalize_weighted_mean` applies internally, so
    every `SourceScore` can report both configured and effective weights. Does not
    duplicate T35's combination arithmetic (the score itself is computed by calling
    `renormalize_weighted_mean` directly); this only exposes the weight vector."""
    total_weight = sum(w for w, v in components.values() if v is not None)
    if total_weight <= 0.0:
        return {name: 0.0 for name in components}
    return {
        name: (w / total_weight if v is not None else 0.0)
        for name, (w, v) in components.items()
    }


@dataclass(frozen=True)
class SourceScore:
    """T36 composite evidence for one candidate. `source_score` is an evidence-ranking
    score, NOT a probability/confidence/likelihood/certainty of source identity (see
    `EVIDENCE_RANKING_DISCLAIMER` and this module's own docstring). Diagnostic fields
    (`initial_bias_evidence`, `persistence_count_evidence`) are always populated with the
    raw (possibly `NaN`) underlying value, even though their default configured weight is
    `0.0`."""

    candidate_id: str
    t_s: np.datetime64

    support_evidence: float
    persistence_evidence: float
    convergence_evidence: float
    dispersion_evidence: float
    ais_evidence: float | None
    initial_bias_evidence: float
    persistence_count_evidence: float

    configured_weights: dict[str, float]
    effective_weights: dict[str, float]

    source_score: float


def compute_source_score(
    candidate: Candidate,
    *,
    screening_score: ScreeningScore,
    persistence_track: PersistenceTrack | None,
    convergence_score: ConvergenceScore,
    dispersion_score: DispersionAwareScore,
    initial_bias: InitialBiasInfluence,
    ais_evidence: AISEvidence,
    weights: CompositeWeights = DEFAULT_WEIGHTS,
) -> SourceScore:
    """Compute the T36 composite `SourceScore` for one candidate from its own
    already-computed T29(via T34)/T30/T31/T32/T33/T35 results (each passed in explicitly --
    this function does not call into any other Phase 3 module to derive them, only to
    combine them). Raises `ValueError` if any supplied result's own `candidate_id` does not
    match `candidate.candidate_id` (never silently mixes results across candidates)."""
    candidate_id = candidate.candidate_id
    t_s = np.datetime64(candidate.t_s, "ns")

    for label, obj_id in (
        ("screening_score", screening_score.candidate_id),
        ("convergence_score", convergence_score.candidate_id),
        ("dispersion_score", dispersion_score.candidate_id),
        ("ais_evidence", ais_evidence.candidate_id),
    ):
        if obj_id != candidate_id:
            raise ValueError(
                f"{label}.candidate_id ({obj_id!r}) does not match "
                f"candidate.candidate_id ({candidate_id!r})"
            )
    if persistence_track is not None and candidate_id not in persistence_track.candidate_ids:
        raise ValueError(
            f"persistence_track ({persistence_track.track_id}) does not contain "
            f"candidate {candidate_id!r}"
        )

    raw_support_evidence = screening_score.screening_score
    raw_persistence_evidence = (
        persistence_track.temporal_continuity() if persistence_track is not None else float("nan")
    )
    raw_convergence_evidence = convergence_score.convergence_score
    raw_dispersion_evidence = dispersion_score.dispersion_aware_score
    raw_ais_evidence = ais_evidence.ais_score
    raw_initial_bias_evidence = initial_bias.influence
    raw_persistence_count_evidence = (
        float(persistence_track.persistence_count) if persistence_track is not None else float("nan")
    )

    components: dict[str, tuple[float, float | None]] = {
        "support": (weights.w_support, _nan_to_none(raw_support_evidence)),
        "persistence": (weights.w_persistence, _nan_to_none(raw_persistence_evidence)),
        "convergence": (weights.w_convergence, _nan_to_none(raw_convergence_evidence)),
        "dispersion": (weights.w_dispersion, _nan_to_none(raw_dispersion_evidence)),
        "ais": (weights.w_ais, raw_ais_evidence),
        "initial_bias": (weights.w_initial, _nan_to_none(raw_initial_bias_evidence)),
        "persistence_count": (weights.w_persistence_count, _nan_to_none(raw_persistence_count_evidence)),
    }

    effective_weights = _effective_weights(components)
    source_score = renormalize_weighted_mean(list(components.values()))

    return SourceScore(
        candidate_id=candidate_id, t_s=t_s,
        support_evidence=raw_support_evidence, persistence_evidence=raw_persistence_evidence,
        convergence_evidence=raw_convergence_evidence, dispersion_evidence=raw_dispersion_evidence,
        ais_evidence=raw_ais_evidence, initial_bias_evidence=raw_initial_bias_evidence,
        persistence_count_evidence=raw_persistence_count_evidence,
        configured_weights=weights.as_dict(), effective_weights=effective_weights,
        source_score=source_score,
    )


def compute_source_scores_for_candidates(
    records: EnsembleRunRecords,
    candidates: list[Candidate],
    support_grids_by_t_s: dict,
    *,
    neighborhood_radius_m: float,
    weights: CompositeWeights = DEFAULT_WEIGHTS,
    ais_source: AISDataSource | None = None,
) -> list[SourceScore]:
    """Wire up T29(via T34)/T30/T31/T32/T33/T35 for every candidate -- calling each
    module's own existing (unmodified) batch function exactly once -- then combine via
    `compute_source_score`. No Stage III query is duplicated beyond what each called
    module's own batch function already performs internally.

    Parameters
    ----------
    records : EnsembleRunRecords
        Already-computed Stage III output.
    candidates : list[Candidate]
        T17 candidates, any/all timestamps.
    support_grids_by_t_s : dict
        `{t_s: grid}` mapping T29/T34 need (same structure the real-data notebooks build).
    neighborhood_radius_m : float
        The radius T33 needs (must match the radius `support_grids_by_t_s` was built with).
    weights : CompositeWeights
        Defaults to `DEFAULT_WEIGHTS`.
    ais_source : AISDataSource, optional
        Passed through to T35's `compute_ais_evidence`; `None` (default) is the only value
        this project can exercise today.
    """
    screening_results = compute_screening_scores_for_candidates(candidates, support_grids_by_t_s)

    tracks = track_candidate_persistence(candidates)
    track_by_candidate_id: dict[str, PersistenceTrack] = {}
    for track in tracks:
        for cid in track.candidate_ids:
            track_by_candidate_id[cid] = track

    convergence_results = compute_convergence_scores_for_candidates(
        records, candidates, persistence_tracks=tracks
    )
    dispersion_results = compute_dispersion_aware_scores_for_candidates(
        records, candidates, neighborhood_radius_m=neighborhood_radius_m
    )
    ais_results = [compute_ais_evidence(c, ais_source=ais_source) for c in candidates]

    cloud_stats_by_t_s: dict[np.datetime64, object] = {}
    initial_bias_results = []
    for c in candidates:
        t_s = np.datetime64(c.t_s, "ns")
        if t_s not in cloud_stats_by_t_s:
            cloud_stats_by_t_s[t_s] = compute_timestamp_cloud_stats(records, t_s)
        initial_bias_results.append(
            compute_initial_bias_influence(cloud_stats_by_t_s[t_s], c.y_peak, c.x_peak)
        )

    results = []
    for i, c in enumerate(candidates):
        results.append(
            compute_source_score(
                c, screening_score=screening_results[i],
                persistence_track=track_by_candidate_id.get(c.candidate_id),
                convergence_score=convergence_results[i], dispersion_score=dispersion_results[i],
                initial_bias=initial_bias_results[i], ais_evidence=ais_results[i], weights=weights,
            )
        )
    return results
