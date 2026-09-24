"""TASK AIS-5 -- transparent AIS vessel-attribution ranking (FIRST task allowed to
rank vessels).

Builds an interpretable, multi-dimensional `ais_attribution_score` from AIS-4's
already-computed per-vessel evidence table, using the SAME architectural pattern T36
already established for combining Phase 3 evidence (`iste.phase3_ais.
renormalize_weighted_mean`, reused unchanged -- not reimplemented): each evidence
dimension is either a number in `[0, 1]` (or `None` if genuinely undefined for that
vessel, e.g. zero close encounters), combined via a configurable weighted mean that
renormalizes over only the AVAILABLE dimensions, exactly as T36 already does for
candidate evidence. **This module does not modify T35/T36 (only reuses T35's own
already-approved `renormalize_weighted_mean` utility, the same reuse T36 itself
already relies on), does not modify Stage III, and does not modify T27-T39.**

**`ais_attribution_score` is an EVIDENCE-RANKING score, exactly like T36's own
`source_score` -- NOT a probability, confidence, or certainty that a vessel is the
source of the spill.** See `EVIDENCE_RANKING_DISCLAIMER` below, and this module's own
explicit separation (Part 10 of the AIS-5 task) between an "AIS evidence ranking" (what
this module produces) and a "source attribution conclusion" (explicitly NOT produced
here or anywhere in this project so far).

**Evidence dimensions (item 2), each an independent, non-overlapping measurement:**

    spatial_evidence(v)   = exp(-exact_min_distance_km(v) / spatial_scale_km)
        [continuous decay of the vessel's own best synchronized-encounter distance;
        `spatial_scale_km` defaults to the SAME `DEFAULT_CLOSE_ENCOUNTER_DISTANCE_KM`
        AIS-3/AIS-4 already established (10 km) -- reused, not reinvented]

    temporal_consecutive_evidence(v) = longest_consecutive_close_encounter_run(v) / n_candidate_timestamps
        [AIS-4's own already-computed run-length statistic, normalized by the FIXED
        number of candidate timestamps -- bounded, count-independent]

    temporal_coverage_evidence(v) = n_distinct_close_encounter_checkpoints(v) / n_candidate_timestamps
        [a genuinely DIFFERENT dimension from the run-length above: how much of the
        whole window has ANY close encounter, not just the longest unbroken streak]

    support_evidence(v) = mean(candidate_support at v's own close encounters) / reference_support
        [`reference_support` defaults to the 95th percentile of ALL candidates' own
        `support_score` -- reusing T29's own percentile-normalization convention/
        rationale at this evidence level, not inventing a new arbitrary scale]

None of these four dimensions rewards raw AIS observation count (item 4): spatial/
support are MIN/MEAN-based, both temporal dimensions are counts normalized by the FIXED
`n_candidate_timestamps` denominator, never by the vessel's own observation count.
`vessel_type` never enters any weight or transformation (item 5) -- it is carried
through the output table purely as a descriptive identity field.

**The AIS-4 circular-shift null p-value is kept as a SEPARATE diagnostic (item 3):**
`null_evidence(v) = 1 - empirical_p_value(v)` exists as a FIFTH, OPTIONAL dimension
with weight `0.0` in `DEFAULT_WEIGHTS` (primary score is independent of it); a
`NULL_SENSITIVITY_WEIGHTS` configuration is provided separately for the required
sensitivity check that DOES include it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from iste.ais_matching import DEFAULT_CLOSE_ENCOUNTER_DISTANCE_KM
from iste.phase3_ais import renormalize_weighted_mean

EVIDENCE_RANKING_DISCLAIMER = (
    "ais_attribution_score is an EVIDENCE-RANKING score for comparing vessels within "
    "one run/configuration. It is NOT a probability, confidence, likelihood, or "
    "certainty that any vessel caused or is the source of the spill. A high score is "
    "evidence of closer/more sustained spatial-temporal synchronization with the "
    "backtracked candidate corridor, nothing more. This is an AIS EVIDENCE RANKING, "
    "not a source attribution conclusion -- no vessel is or should be identified as "
    "the source based on this score alone."
)

DEFAULT_SPATIAL_SCALE_KM = DEFAULT_CLOSE_ENCOUNTER_DISTANCE_KM  # reused from AIS-3/4, not reinvented
DEFAULT_SUPPORT_REFERENCE_PERCENTILE = 95.0  # reuses T29's own percentile convention

_WEIGHT_FIELDS = ("w_spatial", "w_temporal_consecutive", "w_temporal_coverage", "w_support", "w_null")


@dataclass(frozen=True)
class AISScoreWeights:
    """Explicit, configurable, validated weights (item 6). Defaults sum to 1.0 across
    the four PRIMARY dimensions; `w_null` defaults to 0.0 -- the primary score is
    independent of the null diagnostic (item 3), which is exercised only via
    `NULL_SENSITIVITY_WEIGHTS` in the required sensitivity analysis."""

    w_spatial: float = 0.35
    w_temporal_consecutive: float = 0.20
    w_temporal_coverage: float = 0.20
    w_support: float = 0.25
    w_null: float = 0.0

    def __post_init__(self) -> None:
        for name in _WEIGHT_FIELDS:
            value = getattr(self, name)
            if not (np.isfinite(value) and value >= 0.0):
                raise ValueError(f"{name} must be finite and >= 0, got {value!r}")

    def as_dict(self) -> dict[str, float]:
        return {
            "spatial": self.w_spatial, "temporal_consecutive": self.w_temporal_consecutive,
            "temporal_coverage": self.w_temporal_coverage, "support": self.w_support, "null": self.w_null,
        }


DEFAULT_WEIGHTS = AISScoreWeights()

# Item 7 sensitivity configs -- proposed from evidence-role reasoning, not tuned
# against any vessel's own outcome (same discipline T36's own sensitivity configs used).
EQUAL_WEIGHTS = AISScoreWeights(w_spatial=0.25, w_temporal_consecutive=0.25, w_temporal_coverage=0.25, w_support=0.25)
SPATIAL_EMPHASIS_WEIGHTS = AISScoreWeights(w_spatial=0.55, w_temporal_consecutive=0.15, w_temporal_coverage=0.15, w_support=0.15)
TEMPORAL_EMPHASIS_WEIGHTS = AISScoreWeights(w_spatial=0.20, w_temporal_consecutive=0.30, w_temporal_coverage=0.30, w_support=0.20)
NULL_SENSITIVITY_WEIGHTS = AISScoreWeights(w_spatial=0.25, w_temporal_consecutive=0.15, w_temporal_coverage=0.15, w_support=0.20, w_null=0.25)


def compute_close_encounter_counts(
    matches_df: pd.DataFrame, level: str, *, close_encounter_km: float = DEFAULT_CLOSE_ENCOUNTER_DISTANCE_KM
) -> pd.Series:
    """`n_distinct_close_encounter_checkpoints` per vessel, at tolerance `level` (one of
    'exact'/'within_1h'/'within_2h') -- a count AIS-4's own table did not separately
    store (it stored the RUN length, a different statistic), recomputed here directly
    from the AIS-3 raw match table (read-only reuse, no new matching performed)."""
    col = {"exact": "is_exact", "within_1h": "is_within_1h", "within_2h": "is_within_2h"}[level]
    close = matches_df[(matches_df[col]) & (matches_df["distance_km"] <= close_encounter_km)]
    return close.groupby("mmsi")["candidate_t_s"].nunique()


def compute_mean_close_encounter_support(
    matches_df: pd.DataFrame, level: str, *, close_encounter_km: float = DEFAULT_CLOSE_ENCOUNTER_DISTANCE_KM
) -> pd.Series:
    """Mean `candidate_support` across a vessel's own close encounters at tolerance
    `level` -- the raw (un-normalized) input to `support_evidence`."""
    col = {"exact": "is_exact", "within_1h": "is_within_1h", "within_2h": "is_within_2h"}[level]
    close = matches_df[(matches_df[col]) & (matches_df["distance_km"] <= close_encounter_km)]
    return close.groupby("mmsi")["candidate_support"].mean()


def compute_ais_ranking(
    vessel_evidence_df: pd.DataFrame,
    matches_df: pd.DataFrame,
    null_df: pd.DataFrame,
    candidates_df: pd.DataFrame,
    *,
    level: str = "exact",
    weights: AISScoreWeights = DEFAULT_WEIGHTS,
    spatial_scale_km: float = DEFAULT_SPATIAL_SCALE_KM,
    close_encounter_km: float = DEFAULT_CLOSE_ENCOUNTER_DISTANCE_KM,
    support_reference_percentile: float = DEFAULT_SUPPORT_REFERENCE_PERCENTILE,
) -> pd.DataFrame:
    """TASK AIS-5 items 1-6, 9: the full ranked vessel-attribution table for ONE
    tolerance `level` ('exact' by default -- the PRIMARY basis per item 1) and ONE
    weight configuration. Every vessel from `vessel_evidence_df` (AIS-4's own output)
    is included; vessels with zero close encounters at this `level` get `None` for the
    close-encounter-dependent dimensions (temporal/support), renormalized away via
    `renormalize_weighted_mean` (T35's own utility, reused unchanged) exactly as T36
    already does for its own optional evidence components -- never a fabricated 0.
    """
    if level not in ("exact", "within_1h", "within_2h"):
        raise ValueError(f"level must be one of exact/within_1h/within_2h, got {level!r}")

    reference_support = float(np.percentile(candidates_df["support_score"], support_reference_percentile))

    n_close = compute_close_encounter_counts(matches_df, level, close_encounter_km=close_encounter_km)
    mean_support = compute_mean_close_encounter_support(matches_df, level, close_encounter_km=close_encounter_km)
    null_by_mmsi = null_df.set_index("mmsi")["empirical_p_value"] if len(null_df) else pd.Series(dtype=float)

    n_candidate_timestamps = int(candidates_df["t_s"].nunique())

    min_dist_col = f"{level}_min_distance_km"
    run_col = f"{level}_longest_consecutive_close_encounter_checkpoints"
    first_col = f"{level}_first_close_encounter_t_s"
    last_col = f"{level}_last_close_encounter_t_s"
    best_id_col = f"{level}_best_encounter_candidate_id"
    best_dist_col = f"{level}_best_encounter_distance_km"
    best_ais_t_col = f"{level}_best_encounter_ais_t_s"
    best_support_col = f"{level}_best_encounter_candidate_support"
    best_cluster_col = f"{level}_best_encounter_candidate_cluster_id"
    coverage_col = f"{level}_fraction_candidate_timestamps_covered"
    n_matches_col = f"{level}_n_matches"

    rows = []
    for _, v in vessel_evidence_df.iterrows():
        mmsi = int(v["mmsi"])
        min_dist = v[min_dist_col]
        spatial_evidence = float(np.exp(-min_dist / spatial_scale_km)) if np.isfinite(min_dist) else None

        n_close_v = int(n_close.get(mmsi, 0))
        temporal_coverage_evidence = (n_close_v / n_candidate_timestamps) if n_close_v > 0 else None
        run_len = v[run_col]
        temporal_consecutive_evidence = (float(run_len) / n_candidate_timestamps) if (n_close_v > 0 and pd.notna(run_len)) else None

        support_v = mean_support.get(mmsi)
        support_evidence = (float(support_v) / reference_support) if (support_v is not None and pd.notna(support_v)) else None

        p_value = null_by_mmsi.get(mmsi)
        null_evidence = (1.0 - float(p_value)) if (p_value is not None and pd.notna(p_value)) else None

        components = {
            "spatial": (weights.w_spatial, spatial_evidence),
            "temporal_consecutive": (weights.w_temporal_consecutive, temporal_consecutive_evidence),
            "temporal_coverage": (weights.w_temporal_coverage, temporal_coverage_evidence),
            "support": (weights.w_support, support_evidence),
            "null": (weights.w_null, null_evidence),
        }
        total_w = sum(w for w, val in components.values() if val is not None)
        effective_weights = {name: (w / total_w if val is not None and total_w > 0 else 0.0) for name, (w, val) in components.items()}
        ais_attribution_score = renormalize_weighted_mean(list(components.values()))

        rows.append(dict(
            mmsi=mmsi, vessel_name=v["vessel_name"], imo=int(v["imo"]), vessel_type=v["vessel_type"],
            ais_attribution_score=ais_attribution_score,
            spatial_evidence=spatial_evidence, temporal_consecutive_evidence=temporal_consecutive_evidence,
            temporal_coverage_evidence=temporal_coverage_evidence, support_evidence=support_evidence,
            null_evidence=null_evidence, null_p_value=(float(p_value) if p_value is not None and pd.notna(p_value) else None),
            effective_weight_spatial=effective_weights["spatial"], effective_weight_temporal_consecutive=effective_weights["temporal_consecutive"],
            effective_weight_temporal_coverage=effective_weights["temporal_coverage"], effective_weight_support=effective_weights["support"],
            effective_weight_null=effective_weights["null"],
            coverage_fraction=v[coverage_col], n_matches=v[n_matches_col],
            n_observations_total=v["n_observations_total"], max_interval_hours=v["max_interval_hours"],
            n_gaps_over_threshold=v["n_gaps_over_threshold"],
            strongest_encounter_ais_t_s=v[best_ais_t_col], strongest_encounter_distance_km=v[best_dist_col],
            strongest_matching_candidate_id=v[best_id_col], strongest_candidate_support=v[best_support_col],
            strongest_candidate_cluster_id=v[best_cluster_col],
            first_close_encounter_t_s=v[first_col], last_close_encounter_t_s=v[last_col],
        ))

    df = pd.DataFrame(rows)
    df = df.sort_values("ais_attribution_score", ascending=False, na_position="last").reset_index(drop=True)
    df.insert(0, "rank", np.where(df["ais_attribution_score"].notna(), np.arange(1, len(df) + 1), np.nan))
    return df


def compute_evidence_concentration(ranking_df: pd.DataFrame, top_k_values: tuple[int, ...] = (5, 10, 20)) -> dict:
    """TASK AIS-5 item 11: is the evidence concentrated in a small number of vessels,
    or broadly spread? Computes the top-K share of total (non-null) score mass, plus a
    Gini coefficient of the score distribution -- both standard, transparent,
    descriptive concentration measures, not a new invented metric."""
    scores = ranking_df["ais_attribution_score"].dropna().to_numpy()
    scores_sorted = np.sort(scores)
    total = float(np.sum(scores_sorted))
    result: dict = {"n_vessels_with_score": len(scores), "total_score_mass": total}
    for k in top_k_values:
        top_k_sum = float(np.sum(scores_sorted[-k:])) if len(scores_sorted) >= k else total
        result[f"top_{k}_share_of_total_score"] = (top_k_sum / total) if total > 0 else float("nan")

    # Gini coefficient (0 = perfectly even, 1 = maximally concentrated) -- standard
    # closed-form for a sorted (ascending) array.
    n = len(scores_sorted)
    if n > 0 and total > 0:
        index = np.arange(1, n + 1)
        gini = float((2.0 * np.sum(index * scores_sorted) - (n + 1) * total) / (n * total))
    else:
        gini = float("nan")
    result["gini_coefficient"] = gini
    return result
