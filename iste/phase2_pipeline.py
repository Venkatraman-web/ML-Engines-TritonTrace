"""T26 -- Stage VII: Phase 2 end-to-end integration and candidate artifact.

Integrates T14-T25 into ONE reproducible Phase 2 run (`README.md` Sec. 29's locked
compact contract):

    Stage III particle histories
    -> historical timestamp slices (T14)
    -> KD-tree neighborhoods (T15)
    -> D/W/V support features + support map (T16)
    -> candidate peaks/cores (T17)
    -> Y and N_forward = min(X,Y) (T18)
    -> weighted core sampling (T19)
    -> forward propagation under coherent forcing members (T20/T21)
    -> predicted slick + observed reference (T22)
    -> per-member forward accuracy (T23)
    -> candidate-level forward accuracy (T24)
    -> primary filtering/ranking (T25)
    -> final forward-screened Phase 2 candidate set

**This module adds no new scientific logic whatsoever.** Every displacement, weight,
KDE value, comparison score, aggregate, and filtering/ranking decision comes from the
existing T14-T25 public functions, called in the sequence above, once per historical
timestamp/candidate/forcing member. `iste.step`/`iste.trajectory` (Stage III) are never
called by this module -- it only ever READS an already-computed `iste.ensemble_run.
EnsembleRunRecords` batch via T14's `query_particle_slice` ("do not rerun Stage III for
every candidate" -- Stage III is run exactly once, upstream, by the caller, before this
pipeline starts).

**Reproducibility, without any global/implicit RNG state.** Every stochastic draw this
pipeline needs (T19's core-sampling seed, one per candidate; T21's forward-propagation
seed, one per candidate per forcing member) is derived from a SINGLE explicit
`Phase2RunConfig.master_seed` via `numpy.random.SeedSequence.spawn` -- the exact same
deterministic-derivation idiom `iste.forcing_ensemble.ForcingEnsemble` and
`iste.ensemble_run.run_ensemble` already use, not a new invented scheme. Seeds are
spawned in one fixed, deterministic order (historical timestamps in the configured
grid's own order; within each timestamp, candidates in T17's own deterministic
descending-support order; within each candidate, forcing members in the configured
`forcing_member_ids` order) -- so the SAME `Phase2RunConfig`/`master_seed`/upstream data
always reproduces the identical seed sequence, and therefore the identical final
artifact.

**Machine-readable output.** `serialize_phase2_artifact`/`load_phase2_artifact` write
and read a plain JSON object containing exactly `README.md` Sec. 25's required
candidate fields (`x_s`/`y_s` matching the README's own naming, even though T17's own
`Candidate.x_peak`/`y_peak` fields are not renamed) plus enough run configuration to
reproduce the run.

Does not implement final formal source inference or AIS attribution -- the output is a
forward-screened candidate set, not a posterior (`README.md` Sec. 25/29). Never reads or
references Stage III `c_hat*`. Does not modify `iste.particle_query` (T14) through
`iste.candidate_screening` (T25) -- every call in this module is to one of their
existing, unmodified public functions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from iste.candidate_extraction import Candidate, CandidateExtractionConfig, find_candidate_peaks
from iste.candidate_forward_accuracy import CandidateForwardAccuracy, aggregate_candidate_forward_accuracy
from iste.candidate_screening import (
    CandidateDecision,
    CandidateRecord,
    ScreeningConfig,
    build_candidate_record,
    filter_and_rank_candidates,
)
from iste.core_sampling import SampledCoreParticles, sample_core_particles
from iste.ensemble_run import EnsembleRunRecords
from iste.environment import GriddedFieldSource
from iste.forcing_ensemble import ForcingEnsemble
from iste.forward_accuracy import (
    ComparisonConfig,
    ForwardAccuracyResult,
    compare_predicted_footprints_to_observed,
)
from iste.forward_budget import ForwardBudget, ForwardBudgetConfig, compute_forward_particle_budget
from iste.forward_propagation import ForwardCandidateRuns, forward_propagate_candidate
from iste.kdtree_neighborhood import ParticleNeighborhoodIndex, build_particle_neighborhood_index
from iste.observation import AdaptedParticleSet
from iste.particle_query import HistoricalParticleSlice, query_particle_slice
from iste.slick_footprint import (
    FootprintGridConfig,
    ObservedSlickFootprint,
    PredictedSlickFootprint,
    build_observed_slick_footprint,
    build_predicted_slick_footprints,
)
from iste.support_features import SupportFeatureConfig, SupportFeatures, compute_support_map


@dataclass(frozen=True)
class Phase2RunConfig:
    """Every explicit Phase 2 configuration knob for one integrated run
    (`README.md` Sec. 26's list), bundled into one object so a whole run's
    configuration/seeds can be recorded, logged, and serialized together for
    reproducibility.

    - `historical_timestamps`: the EXPLICIT historical timestamp grid to run candidate
      discovery over -- never silently reduced by this module for convenience.
    - `t_obs`: the actual satellite-observation time every candidate is forward-
      propagated to.
    - `support_query_lat_grid`/`support_query_lon_grid`: the regular grid T16's support
      map (and T17's candidate-peak search) is evaluated on, at every historical
      timestamp.
    - `neighborhood_radius_m`: T15/T16 fixed-radius neighborhood query radius.
    - `support_config`: T16 `SupportFeatureConfig`.
    - `extraction_config`: T17 `CandidateExtractionConfig`.
    - `forward_budget_config`: T18 `ForwardBudgetConfig` (the `X` forward particle
      budget).
    - `forcing_member_ids`: the configured forcing-member set (or an explicit,
      recorded development subset) -- used, unchanged, for every candidate/time.
    - `forward_dt_seconds`: T20/T21 forward integration step.
    - `footprint_lat_grid`/`footprint_lon_grid`/`footprint_bandwidth_m`: T22
      `FootprintGridConfig`'s grid/bandwidth (shared by the predicted and observed
      footprints).
    - `comparison_config`: T23 `ComparisonConfig`.
    - `screening_config`: T25 `ScreeningConfig`.
    - `master_seed`: the SINGLE seed every T19/T21 RNG draw is deterministically
      derived from (see module docstring).
    - `domain_variable`: T20/T21's land/domain mask variable, default `"uo"`.
    """

    historical_timestamps: tuple[np.datetime64, ...]
    t_obs: np.datetime64
    support_query_lat_grid: np.ndarray
    support_query_lon_grid: np.ndarray
    neighborhood_radius_m: float
    support_config: SupportFeatureConfig
    extraction_config: CandidateExtractionConfig
    forward_budget_config: ForwardBudgetConfig
    forcing_member_ids: tuple[int, ...]
    forward_dt_seconds: float
    footprint_lat_grid: np.ndarray
    footprint_lon_grid: np.ndarray
    footprint_bandwidth_m: float
    comparison_config: ComparisonConfig
    screening_config: ScreeningConfig
    master_seed: int
    domain_variable: str = "uo"

    def __post_init__(self) -> None:
        if len(self.historical_timestamps) == 0:
            raise ValueError("historical_timestamps must be non-empty")
        if len(self.forcing_member_ids) == 0:
            raise ValueError("forcing_member_ids must be non-empty")
        if not np.isfinite(self.neighborhood_radius_m) or self.neighborhood_radius_m <= 0:
            raise ValueError(f"neighborhood_radius_m must be finite and > 0; got {self.neighborhood_radius_m!r}")
        if not np.isfinite(self.forward_dt_seconds) or self.forward_dt_seconds <= 0:
            raise ValueError(f"forward_dt_seconds must be finite and > 0; got {self.forward_dt_seconds!r}")


@dataclass
class CandidatePipelineResult:
    """Every intermediate artifact computed for ONE candidate, for inspection/notebook
    use, plus the final T25-ready `CandidateRecord`. Nothing here is recomputed --
    each field is exactly what the corresponding T17-T24 function returned."""

    candidate: Candidate  # T17
    budget: ForwardBudget  # T18
    sampled: SampledCoreParticles  # T19
    sampling_seed: int
    forward_runs: ForwardCandidateRuns  # T21 (T20 internally)
    member_seeds: dict[int, int]
    predicted_footprints: dict[int, PredictedSlickFootprint]  # T22
    member_results: dict[int, ForwardAccuracyResult]  # T23
    forward_accuracy: CandidateForwardAccuracy  # T24
    record: CandidateRecord  # T25 input


@dataclass
class TimestampPipelineResult:
    """Every intermediate artifact computed for ONE historical timestamp `t_s`."""

    t_s: np.datetime64
    slice_: HistoricalParticleSlice  # T14
    index: ParticleNeighborhoodIndex  # T15
    support_map: list[SupportFeatures]  # T16
    candidates: list[CandidatePipelineResult] = field(default_factory=list)  # T17+


@dataclass
class Phase2Result:
    """The complete integrated Phase 2 run: every stage's intermediate artifacts (for
    inspection/notebook use) plus the final T25 `decisions` -- the forward-screened
    candidate set."""

    config: Phase2RunConfig
    observed_footprint: ObservedSlickFootprint  # T22, shared across every t_s/candidate
    timestamps: list[TimestampPipelineResult]
    decisions: list[CandidateDecision]  # T25, flat across every historical timestamp


def _spawn_seed(seed_sequence: np.random.SeedSequence) -> tuple[np.random.SeedSequence, int]:
    """Deterministically spawn one child `SeedSequence` from `seed_sequence` and
    convert it to a plain `int` seed (for `numpy.random.default_rng(seed)`/T19's/T21's
    own `seed: int` parameters) -- returns the child sequence too, in case the caller
    wants to spawn further from it (unused here, kept for clarity of provenance)."""
    (child,) = seed_sequence.spawn(1)
    return child, int(child.generate_state(1)[0])


def run_phase2_pipeline(
    records: EnsembleRunRecords,
    base_ocean_source: GriddedFieldSource,
    ensemble: ForcingEnsemble,
    k_lat: np.ndarray,
    k_lon: np.ndarray,
    k_field: np.ndarray,
    adapted_observation: AdaptedParticleSet,
    config: Phase2RunConfig,
) -> Phase2Result:
    """Run the complete integrated Phase 2 pipeline (module docstring) once, over every
    historical timestamp in `config.historical_timestamps`.

    `records` is an ALREADY-COMPUTED T09 `EnsembleRunRecords` batch (Stage III is never
    rerun here). `adapted_observation` is the SAME upstream `AdaptedParticleSet` (T07)
    that seeded that Stage III run -- used here, independently, to build T22's observed-
    slick reference (`README.md` Sec. 21: never the backward historical particles).
    `base_ocean_source`/`ensemble`/`k_lat`/`k_lon`/`k_field` are the same environmental
    inputs Stage III itself used, reused unchanged for forward propagation.
    """
    footprint_config = FootprintGridConfig(
        lat_grid=config.footprint_lat_grid, lon_grid=config.footprint_lon_grid, bandwidth_m=config.footprint_bandwidth_m,
    )
    observed_footprint = build_observed_slick_footprint(adapted_observation, config.t_obs, footprint_config)

    seed_sequence = np.random.SeedSequence(config.master_seed)

    timestamp_results: list[TimestampPipelineResult] = []
    all_records: list[CandidateRecord] = []

    for t_s in config.historical_timestamps:
        t_s = np.datetime64(t_s, "ns")

        slice_ = query_particle_slice(records, t_s)  # T14
        index = build_particle_neighborhood_index(slice_)  # T15

        lon_mesh, lat_mesh = np.meshgrid(config.support_query_lon_grid, config.support_query_lat_grid)  # (n_lat, n_lon)
        support_map = compute_support_map(  # T16
            index, lat_mesh.ravel(), lon_mesh.ravel(), radius_m=config.neighborhood_radius_m, config=config.support_config,
        )
        candidates = find_candidate_peaks(  # T17
            support_map, config.support_query_lat_grid, config.support_query_lon_grid, t_s, config=config.extraction_config,
        )

        candidate_results: list[CandidatePipelineResult] = []
        for candidate in candidates:
            budget = compute_forward_particle_budget(candidate, index, config.forward_budget_config)  # T18

            _, sampling_seed = _spawn_seed(seed_sequence)
            sampled = sample_core_particles(candidate, index, config.forward_budget_config, seed=sampling_seed)  # T19

            member_seeds: dict[int, int] = {}
            for member_id in config.forcing_member_ids:
                _, member_seed = _spawn_seed(seed_sequence)
                member_seeds[member_id] = member_seed

            forward_runs = forward_propagate_candidate(  # T21 (T20 internally)
                sampled, config.t_obs, config.forward_dt_seconds, base_ocean_source, ensemble,
                forcing_member_ids=list(config.forcing_member_ids),
                seeds=[member_seeds[m] for m in config.forcing_member_ids],
                k_lat=k_lat, k_lon=k_lon, k_field=k_field, domain_variable=config.domain_variable,
            )

            predicted_footprints = build_predicted_slick_footprints(forward_runs.runs, footprint_config)  # T22

            member_results = compare_predicted_footprints_to_observed(  # T23
                predicted_footprints, observed_footprint, config.comparison_config,
            )

            forward_accuracy = aggregate_candidate_forward_accuracy(member_results)  # T24

            record = build_candidate_record(candidate, budget, sampling_seed, forward_accuracy)  # T25 input

            candidate_results.append(
                CandidatePipelineResult(
                    candidate=candidate, budget=budget, sampled=sampled, sampling_seed=sampling_seed,
                    forward_runs=forward_runs, member_seeds=member_seeds,
                    predicted_footprints=predicted_footprints, member_results=member_results,
                    forward_accuracy=forward_accuracy, record=record,
                )
            )
            all_records.append(record)

        timestamp_results.append(
            TimestampPipelineResult(t_s=t_s, slice_=slice_, index=index, support_map=support_map, candidates=candidate_results)
        )

    decisions = filter_and_rank_candidates(all_records, config.screening_config)  # T25

    return Phase2Result(config=config, observed_footprint=observed_footprint, timestamps=timestamp_results, decisions=decisions)


# --------------------------------------------------------------------------------
# Machine-readable artifact serialization
# --------------------------------------------------------------------------------

def _json_datetime64(t) -> str:
    return str(np.datetime_as_string(np.datetime64(t, "ns"), unit="ns"))


def _serialize_config(config: Phase2RunConfig) -> dict:
    return {
        "historical_timestamps": [_json_datetime64(t) for t in config.historical_timestamps],
        "t_obs": _json_datetime64(config.t_obs),
        "support_query_lat_grid": np.asarray(config.support_query_lat_grid).tolist(),
        "support_query_lon_grid": np.asarray(config.support_query_lon_grid).tolist(),
        "neighborhood_radius_m": config.neighborhood_radius_m,
        "support_config": {
            "d_scale": config.support_config.d_scale, "w_scale": config.support_config.w_scale,
            "v_scale": config.support_config.v_scale, "weight_d": config.support_config.weight_d,
            "weight_w": config.support_config.weight_w, "weight_v": config.support_config.weight_v,
        },
        "extraction_config": {
            "peak_neighborhood_cells": config.extraction_config.peak_neighborhood_cells,
            "min_support": config.extraction_config.min_support,
            "min_separation_m": config.extraction_config.min_separation_m,
            "core_radius_m": config.extraction_config.core_radius_m,
        },
        "forward_budget_config": {"X": config.forward_budget_config.X},
        "forcing_member_ids": list(config.forcing_member_ids),
        "forward_dt_seconds": config.forward_dt_seconds,
        "footprint_lat_grid": np.asarray(config.footprint_lat_grid).tolist(),
        "footprint_lon_grid": np.asarray(config.footprint_lon_grid).tolist(),
        "footprint_bandwidth_m": config.footprint_bandwidth_m,
        "comparison_config": {
            "metric": config.comparison_config.metric,
            "region_threshold_fraction": config.comparison_config.region_threshold_fraction,
            "consistency_distance_scale_m": config.comparison_config.consistency_distance_scale_m,
            "combination_weight_overlap": config.comparison_config.combination_weight_overlap,
            "combination_weight_fp": config.comparison_config.combination_weight_fp,
        },
        "screening_config": {
            "mode": config.screening_config.mode,
            "error_threshold": config.screening_config.error_threshold,
            "top_k": config.screening_config.top_k,
            "merge_separation_m": config.screening_config.merge_separation_m,
        },
        "master_seed": config.master_seed,
        "domain_variable": config.domain_variable,
    }


def _serialize_decision(decision: CandidateDecision) -> dict:
    r = decision.record
    return {
        "candidate_id": r.candidate_id,
        "x_s": r.x_peak,
        "y_s": r.y_peak,
        "t_s": _json_datetime64(r.t_s),
        "support_score": r.support_score,
        "density_score": r.density_score,
        "weighted_support_score": r.weighted_support_score,
        "pixel_diversity_score": r.pixel_diversity_score,
        "core_definition": {
            "center_lat": r.core_definition.center_lat, "center_lon": r.core_definition.center_lon,
            "radius_m": r.core_definition.radius_m,
        },
        "core_particle_count_Y": r.core_particle_count_Y,
        "forward_particle_count_N_forward": r.forward_particle_count_N_forward,
        "sampling_seed": r.sampling_seed,
        "forcing_members_validated": list(r.forcing_members_validated),
        "forward_error_by_member": {str(k): v for k, v in r.forward_error_by_member.items()},
        "forward_consistency_by_member": {str(k): v for k, v in r.forward_consistency_by_member.items()},
        "candidate_forward_error": r.candidate_forward_error,
        "candidate_forward_consistency": r.candidate_forward_consistency,
        "forward_validation_status": r.forward_validation_status,
        "accepted": decision.accepted,
        "rank": decision.rank,
        "rejection_reason": decision.rejection_reason,
    }


def serialize_phase2_artifact(decisions: list[CandidateDecision], config: Phase2RunConfig) -> dict:
    """A JSON-safe `dict` representation of the final Phase 2 forward-screened
    candidate set (`README.md` Sec. 25's required fields, `x_s`/`y_s` matching the
    README's own naming) plus the run configuration/seed needed to reproduce it
    (`config.master_seed` alone -- combined with the deterministic per-candidate/
    per-member spawn order documented in the module docstring -- reproduces every
    T19/T21 seed exactly).

    `candidate_forward_error`/`candidate_forward_consistency` may be `NaN` for an
    unvalidated candidate; Python's `json` module round-trips `NaN` losslessly (a
    documented non-strict-JSON extension), which this artifact relies on.
    """
    return {
        "phase2_artifact_version": "1",
        "config": _serialize_config(config),
        "candidates": [_serialize_decision(d) for d in decisions],
    }


def save_phase2_artifact(decisions: list[CandidateDecision], config: Phase2RunConfig, path: str) -> None:
    """Serialize (`serialize_phase2_artifact`) and write to `path` as JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serialize_phase2_artifact(decisions, config), f, indent=2, allow_nan=True)


def load_phase2_artifact(path: str) -> dict:
    """Read back a JSON file written by `save_phase2_artifact`/`serialize_phase2_artifact`.
    Returns the raw `dict` (not reconstructed `CandidateDecision` objects) -- the
    machine-readable artifact IS this dict; no Python-object reconstruction is implied
    or required for a downstream consumer."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
