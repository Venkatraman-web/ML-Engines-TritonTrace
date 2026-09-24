"""T33 -- Phase 3 dispersion-aware scoring.

Compares a candidate's actual local particle density (T16's `D`, `Candidate.density_score`)
against the density EXPECTED at that location if the WHOLE population at that timestamp
were a single, undifferentiated, isotropic-Gaussian blob with the SAME total particle count
`N` and the SAME measured spread -- not the candidate's own local neighborhood, the whole
timestamp's own dispersion state (T30's `TimestampCloudStats`, reused unchanged). This is
the direct answer to Stage 6's requirement: a lower raw density at T72 (where T28 already
measured the real population has genuinely spread out ~5.5x more than at T0) must not be
automatically read as weaker evidence than a higher raw density at T1, if T72's density is
nonetheless MORE concentrated than T72's own (larger) dispersion would predict for an
unremarkable single blob.

Formula (explicit, analytically-grounded PROTOTYPE/CONFIGURATION choice -- every step
justified, not an arbitrary transform):

    sigma_iso_sq(t) = trace(Sigma(t)) / 2
        [T30's own per-timestamp covariance (`TimestampCloudStats.covariance_m2`, or its
        `isotropic_variance_m2` fallback), REUSED unchanged -- not recomputed here]

    p_within_r(t) = 1 - exp(-r^2 / (2 * sigma_iso_sq(t)))
        [closed-form Rayleigh-CDF: the probability mass of an isotropic 2-D Gaussian that
        falls within radius r of its own mean -- standard, not invented]

    D_expected(t) = N_total_valid_at_t * p_within_r(t)
        [expected particle count within radius r of an undifferentiated single blob's own
        peak, given the SAME total particle count and the SAME measured spread as the real
        population at t]

    dispersion_aware_score = D_actual_at_candidate_peak / D_expected(t)

`r` (`neighborhood_radius_m`) is the SAME radius already used to build the candidate's T16
support map -- not a new free parameter. The isotropic approximation (rather than an
anisotropic closed form for mass-within-a-circle, which has no simple form for a general
elliptical Gaussian) mirrors T30's own singular-covariance-fallback convention
(`sigma_iso^2 = trace(Sigma)/2`) -- an explicit, documented simplification, not a claim
that real particle clouds are exactly isotropic (T30's own anisotropy check on real
full-rank covariance already shows they are not).

`dispersion_aware_score` is UNBOUNDED ABOVE -- a likelihood-ratio/Bayes-factor-style
"surprise" quantity, not a probability or a value normalized to `[0, 1]`. `~1` means the
candidate's density matches exactly what undifferentiated dispersion alone would predict at
its most concentrated point; `>1` means genuinely more concentrated than dispersion
explains; `<1` means less concentrated than even an undifferentiated blob's own peak.

Explicit, tested `NaN` (never fabricated) whenever the underlying dispersion state is
undefined -- T30's `covariance_state in {"insufficient_particles", "zero_variance"}` -- or
when there are no valid particles at all at that timestamp.

Screening-quantity caveat (same as T30/T31/T32): a high dispersion-aware score is evidence
that a candidate's concentration exceeds what generic dispersion alone would produce, never
proof of physical source identity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from iste.candidate_extraction import Candidate
from iste.ensemble_run import EnsembleRunRecords
from iste.phase3_initial_bias import TimestampCloudStats, compute_timestamp_cloud_stats


@dataclass(frozen=True)
class DispersionAwareScore:
    """T33 dispersion-aware evidence for one candidate."""

    candidate_id: str
    t_s: np.datetime64
    particle_count_actual: float  # T16's D at the candidate's own peak
    n_total_valid_at_t: int
    covariance_state: str  # mirrors TimestampCloudStats.covariance_state
    sigma_iso_m: float  # NaN iff covariance_state undefined
    neighborhood_radius_m: float
    p_within_r: float  # NaN iff undefined
    expected_density: float  # D_expected(t); NaN iff undefined
    dispersion_aware_score: float  # NaN iff undefined


def compute_dispersion_aware_score(
    candidate: Candidate,
    cloud_stats: TimestampCloudStats,
    *,
    neighborhood_radius_m: float,
) -> DispersionAwareScore:
    """Evaluate the T33 dispersion-aware score for one candidate against one timestamp's
    already-computed `TimestampCloudStats` (T30's `compute_timestamp_cloud_stats`, reused).

    Parameters
    ----------
    candidate : Candidate
        A T17 candidate; `density_score` (T16's own `D`) is used as the actual local count.
    cloud_stats : TimestampCloudStats
        The SAME timestamp's whole-population dispersion state (T30, unmodified).
    neighborhood_radius_m : float
        The radius `r` used to build the candidate's T16 support map -- must match, since it
        defines what "within radius r" means for both the actual and expected density.
    """
    if not (np.isfinite(neighborhood_radius_m) and neighborhood_radius_m > 0.0):
        raise ValueError(f"neighborhood_radius_m must be finite and > 0, got {neighborhood_radius_m}")

    particle_count_actual = float(candidate.density_score)
    n_total = cloud_stats.n_particles

    undefined = (
        n_total <= 0
        or cloud_stats.covariance_state in ("insufficient_particles", "zero_variance")
    )

    if undefined:
        sigma_iso_m = float("nan")
        p_within_r = float("nan")
        expected_density = float("nan")
        score = float("nan")
    else:
        if cloud_stats.covariance_state == "full_rank":
            sigma_iso_sq = float(np.trace(cloud_stats.covariance_m2)) / 2.0
        else:  # "singular_fallback" -- T30 guarantees isotropic_variance_m2 > 0 here
            sigma_iso_sq = cloud_stats.isotropic_variance_m2
        sigma_iso_m = float(np.sqrt(sigma_iso_sq))
        p_within_r = float(1.0 - np.exp(-(neighborhood_radius_m**2) / (2.0 * sigma_iso_sq)))
        expected_density = float(n_total) * p_within_r
        score = particle_count_actual / expected_density if expected_density > 0.0 else float("nan")

    return DispersionAwareScore(
        candidate_id=candidate.candidate_id, t_s=np.datetime64(candidate.t_s, "ns"),
        particle_count_actual=particle_count_actual, n_total_valid_at_t=n_total,
        covariance_state=cloud_stats.covariance_state, sigma_iso_m=sigma_iso_m,
        neighborhood_radius_m=float(neighborhood_radius_m), p_within_r=p_within_r,
        expected_density=expected_density, dispersion_aware_score=score,
    )


def compute_dispersion_aware_scores_for_candidates(
    records: EnsembleRunRecords,
    candidates: list[Candidate],
    *,
    neighborhood_radius_m: float,
) -> list[DispersionAwareScore]:
    """`compute_dispersion_aware_score` evaluated for every candidate in `candidates`, in
    order. `TimestampCloudStats` (T30) is computed once per DISTINCT `t_s` and reused across
    every candidate sharing that timestamp -- no redundant Stage III queries, mirroring
    T30/T31/T32's own batch convenience."""
    stats_by_t_s: dict[np.datetime64, TimestampCloudStats] = {}
    results = []
    for c in candidates:
        t_s = np.datetime64(c.t_s, "ns")
        if t_s not in stats_by_t_s:
            stats_by_t_s[t_s] = compute_timestamp_cloud_stats(records, t_s)
        results.append(
            compute_dispersion_aware_score(
                c, stats_by_t_s[t_s], neighborhood_radius_m=neighborhood_radius_m
            )
        )
    return results
