"""T14 -- Stage IV surrogate-likelihood core.

**2026-09-22 correction (T13.5/T14-correction):** `final_stage_III_IV_specification.txt`
is now the authoritative mathematical source of truth for Stage IV, superseding an
earlier, differently-ordered formula. The pixel-likelihood/per-member-likelihood layer
below is unchanged; only the ensemble-marginalization/tempering ordering was corrected --
see "Critical rule" below.

Implements exactly the Stage IV observation model and surrogate likelihood, no more and
no less:

    P(y_j | h_j,e) = (1-epsilon) * pi_j,e^y_j * (1-pi_j,e)^(1-y_j) + epsilon/2

    L_e = product_j P(y_j | h_j,e)

    p_tilde(y | Theta, M_eff) = (1/E) * sum_e [ L_e^beta ]

**Critical rule, per the frozen contract's Sec. 19/21 (corrected 2026-09-22 -- this is
the OPPOSITE ordering from an earlier, now-superseded formula this module previously
implemented):** `beta` is applied to EACH forcing member's own likelihood `L_e`
INDIVIDUALLY, before the ensemble is marginalized -- `mean_e(L_e^beta)`, never
`(mean_e(L_e))^beta`. The frozen contract defines `beta = J_eff/J` as a correction for
residual spatial correlation BETWEEN PIXELS WITHIN one forcing member, not a generic
post-hoc tempering/learning-rate applied to the whole ensemble mixture -- so it belongs
inside the per-member term, before marginalizing across members. See
`tests/test_likelihood.py::TestCorrectVsIncorrectTempering` for a regression test that
fails if this ordering is ever accidentally swapped back.

This module is a **consumer** of Stage III's outputs, not a re-implementation of
anything Stage III computes: `pi_j,e` (a detection probability derived from Stage III's
`c_hat*_{j,e}(x,t)` transport estimate for observation pixel `j` under forcing member
`e`) is accepted as an explicit input array, never recomputed here. `y_j` (the observed
pixel label, 0 or 1) is likewise accepted as given -- Stage I segmentation/classification
(which is what actually produces `y_j`) is out of scope here and is not duplicated.

**Numerical stability (the point of this whole module):** `L_e` is a product over
potentially many pixels of probabilities that can each be small -- computing it directly
in linear space would underflow to exactly `0.0` for even a modest number of pixels.
Everything here is computed in the log domain instead:

    log P(y_j | h_j,e)                     -- computed directly, once per pixel/member
    log L_e = sum_j log P(y_j|h_j,e)                               -- a sum, never a product
    tempered_log_L_e = beta * log L_e                              -- beta applied PER MEMBER
    log p_tilde = logsumexp_e(tempered_log_L_e) - log(E)           -- marginalized LAST

No probability is ever multiplied directly with another; the only place a linear-space
number appears is inside the per-pixel `P(y_j|h_j,e)` formula itself (an unavoidable,
bounded, single-pixel quantity, not a product of many terms) and inside the stable
log-sum-exp/log-softmax steps' own internal `exp(shifted)`, which is deliberately shifted
by the running maximum first so it never overflows.

**Member responsibilities and effective sample size** (frozen contract Sec. 22):
`r_e = softmax_e(tempered_log_L_e)`, `ESS_member = 1 / sum_e(r_e^2)`, computed from the
SAME tempered per-member log-likelihoods used for the final marginalization (never from
the untempered `log_L_e`). A low `ESS_member` is only a diagnostic -- it does not by
itself identify insufficient ensemble size, forcing-perturbation strength, `beta`, or
observation structure as the cause; that requires separate investigation, not performed
here.

Does not implement Stage V (tempered SMC), Stage VI (Delayed-Acceptance MCMC), Stage VII
(posterior/HPD output), or AIS -- this module exposes one clean likelihood function later
stages can call, nothing more. Does not compute `h_j,e` from Stage III's transport
estimate, and does not implement the detection-probability model `pi(h)` (T15's separate
concern).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Stage4LikelihoodResult:
    """Every intermediate quantity of the Stage IV surrogate likelihood, one candidate
    source hypothesis at a time.

    - `log_pixel_likelihood`: shape `(n_pixels, n_members)` -- `log P(y_j | h_j,e)`.
    - `log_member_likelihood`: shape `(n_members,)` -- `log L_e = sum_j log P(y_j|h_j,e)`
      (UNTEMPERED -- `beta` not yet applied).
    - `log_mean_member_likelihood`: scalar -- `log( (1/E) * sum_e L_e )`, the untempered
      (`beta=1`-equivalent) ensemble mean, computed via a stable log-mean-exp. Kept as a
      diagnostic (e.g. "what would the surrogate likelihood be with no correlation
      correction") -- it is NOT used to derive `log_surrogate_likelihood` (that would
      reproduce the superseded, incorrect ordering).
    - `tempered_log_member_likelihood`: shape `(n_members,)` -- `beta * log_L_e`, i.e.
      `log(L_e^beta)`, one value per forcing member. This is what actually feeds the
      final marginalization and the responsibilities/ESS below.
    - `log_surrogate_likelihood`: scalar -- `logsumexp_e(tempered_log_member_likelihood)
      - log(E)`, i.e. `log( mean_e(L_e^beta) )`. This is the frozen contract's Sec. 19/20
      formula: `beta` applied PER MEMBER, marginalized LAST.
    - `responsibilities`: shape `(n_members,)` -- `r_e = softmax_e(tempered_log_L_e)`,
      summing to `1.0` (within floating-point tolerance) for at least one finite member.
    - `member_ess`: scalar -- `1 / sum_e(r_e^2)`, in `[1, E]` for valid responsibilities.
    """

    log_pixel_likelihood: np.ndarray
    log_member_likelihood: np.ndarray
    log_mean_member_likelihood: float
    tempered_log_member_likelihood: np.ndarray
    log_surrogate_likelihood: float
    responsibilities: np.ndarray
    member_ess: float


def _log_mean_exp(log_values: np.ndarray) -> float:
    """`log( mean(exp(log_values)) )`, computed by shifting by the running maximum before
    exponentiating so no term ever overflows -- the standard stable log-sum-exp trick,
    divided through by `n` (i.e. log-MEAN-exp, not log-sum-exp). Equivalently,
    `logsumexp(log_values) - log(n)`.

    If every value is `-inf` (every forcing member's tempered likelihood is exactly zero
    -- a legitimate, not-an-error outcome, e.g. `epsilon == 0` and some pixel is flatly
    contradicted by every member), the shift itself is skipped and `-inf` is returned
    directly, avoiding `exp(-inf - (-inf)) = exp(nan)`.
    """
    log_values = np.asarray(log_values, dtype=np.float64)
    n = log_values.shape[0]
    m = np.max(log_values)
    if not np.isfinite(m):
        return float(m)
    shifted = log_values - m
    return float(m + np.log(np.sum(np.exp(shifted)) / n))


def _responsibilities_and_ess(tempered_log_L: np.ndarray) -> tuple[np.ndarray, float]:
    """Member responsibilities `r_e = softmax_e(tempered_log_L_e)` and the member
    effective sample size `ESS_member = 1 / sum_e(r_e^2)`, via a numerically stable
    log-softmax (shift by the running maximum, the same technique `_log_mean_exp` uses)
    over the ALREADY-TEMPERED per-member log-likelihoods -- never the untempered ones.

    If every tempered log-likelihood is `-inf` (every member has zero tempered
    likelihood), softmax is mathematically undefined (`0/0`); this returns an all-`NaN`
    responsibilities array and `NaN` ESS rather than silently guessing a finite value
    (e.g. uniform weights) -- an explicit "undefined," not a misleading result.
    """
    tempered_log_L = np.asarray(tempered_log_L, dtype=np.float64)
    n = tempered_log_L.shape[0]
    m = np.max(tempered_log_L)
    if not np.isfinite(m):
        return np.full(n, np.nan), float("nan")
    shifted = tempered_log_L - m
    log_norm = np.log(np.sum(np.exp(shifted)))
    r = np.exp(shifted - log_norm)
    ess = float(1.0 / np.sum(r ** 2))
    return r, ess


def compute_stage4_log_likelihood(
    y: np.ndarray,
    pi: np.ndarray,
    epsilon: float,
    beta: float,
) -> Stage4LikelihoodResult:
    """Compute the Stage IV surrogate likelihood (in the log domain throughout) for one
    candidate source hypothesis.

    Parameters
    ----------
    y : array, shape (n_pixels,)
        Observed pixel labels, one per observation pixel `j`. Must be binary (`0`/`1` --
        or boolean/float exactly equal to `0`/`1`). Provided by the upstream Stage I
        system; not produced or validated for "correctness" here beyond being binary.
    pi : array, shape (n_pixels, n_members)
        Detection probabilities `pi_j,e`, one row per observation pixel (matching `y`'s
        order) and one column per forcing member `e` -- the explicit forcing-member
        dimension. Each value must be finite and in `[0, 1]`. Derived from Stage III's
        transport estimate by some other (not-yet-built) step; this function only
        consumes it.
    epsilon : float
        The observation-noise floor `epsilon` in `[0, 1]`. `epsilon/2` is the minimum
        possible `P(y_j|h_j,e)` for either outcome of `y_j`, keeping the likelihood from
        ever exactly vanishing when `epsilon > 0`.
    beta : float
        The per-member spatial-correlation correction exponent (frozen contract:
        `beta = J_eff/J`), `>= 0`, applied to EACH forcing member's own likelihood `L_e`
        individually, before the ensemble is marginalized (the corrected Stage IV
        ordering -- see the module docstring's "Critical rule").

        `beta == 0.0` is handled explicitly (not via `0 * log_L_e`, which would produce
        `NaN` for a member with `log_L_e == -inf`): every member's tempered
        log-likelihood is set to exactly `0.0` (`L_e^0 = 1` for every member, by the
        standard empty-product convention, including a member whose untempered `L_e`
        happens to be exactly `0`), giving `log_surrogate_likelihood = 0.0`,
        `responsibilities = 1/E` for every member, and `member_ess = E`.

    Returns
    -------
    Stage4LikelihoodResult
    """
    y = np.asarray(y)
    pi = np.asarray(pi, dtype=np.float64)

    if y.ndim != 1:
        raise ValueError(f"y must be 1-D (one entry per observation pixel); got shape {y.shape}")
    if y.shape[0] == 0:
        raise ValueError("y must have at least one observation pixel")
    if pi.ndim != 2:
        raise ValueError(f"pi must be 2-D (n_pixels, n_forcing_members); got shape {pi.shape}")
    if pi.shape[0] != y.shape[0]:
        raise ValueError(
            f"pi.shape[0] ({pi.shape[0]}) must match len(y) ({y.shape[0]}) -- "
            "one pi row per observation pixel, matching y's order"
        )
    if pi.shape[1] == 0:
        raise ValueError("pi must have at least one forcing member (pi.shape[1] == 0)")

    y_float = y.astype(np.float64)
    if not np.all((y_float == 0.0) | (y_float == 1.0)):
        raise ValueError("y must be binary (0 or 1) for every observation pixel")

    if not np.all(np.isfinite(pi)):
        raise ValueError("pi (detection probabilities) must be finite -- found NaN/Inf")
    if np.any(pi < 0.0) or np.any(pi > 1.0):
        raise ValueError("pi (detection probabilities) must lie in [0, 1]")

    if not np.isfinite(epsilon):
        raise ValueError(f"epsilon must be finite; got {epsilon!r}")
    if not (0.0 <= epsilon <= 1.0):
        raise ValueError(f"epsilon must be in [0, 1]; got {epsilon!r}")

    if not np.isfinite(beta):
        raise ValueError(f"beta must be finite; got {beta!r}")
    if beta < 0.0:
        raise ValueError(f"beta must be >= 0 (a tempering exponent); got {beta!r}")

    # P(y_j | h_j,e) = (1-eps)*pi^y * (1-pi)^(1-y) + eps/2 -- exactly README.md's formula,
    # vectorized over (n_pixels, n_members) with y broadcast down each pixel's row.
    y_col = y_float[:, None]
    p = (1.0 - epsilon) * (pi ** y_col) * ((1.0 - pi) ** (1.0 - y_col)) + epsilon / 2.0
    # Mathematically p is always in [0, 1] given the validated ranges above; clip only to
    # absorb floating-point rounding at the boundary, never to change a real value.
    p = np.clip(p, 0.0, 1.0)

    with np.errstate(divide="ignore"):
        # log(0.0) == -inf here is a legitimate result (epsilon == 0 and a pixel flatly
        # contradicted by this member's pi), not a computation error -- never suppressed
        # or replaced with a placeholder.
        log_pixel_likelihood = np.log(p)

    # log L_e = sum_j log P(y_j|h_j,e) -- a SUM in the log domain, never a product of the
    # (potentially many, potentially tiny) linear-space probabilities. UNTEMPERED.
    log_member_likelihood = np.sum(log_pixel_likelihood, axis=0)

    # Untempered (beta=1-equivalent) ensemble mean -- a diagnostic only; NOT used below to
    # derive log_surrogate_likelihood (see module docstring's "Critical rule").
    log_mean_member_likelihood = _log_mean_exp(log_member_likelihood)

    # beta applied PER MEMBER, before marginalizing -- the corrected Stage IV ordering.
    # beta == 0 is special-cased to avoid `0 * (-inf) == NaN`; see this function's
    # docstring for the L_e^0 = 1 convention this implements.
    if beta == 0.0:
        tempered_log_member_likelihood = np.zeros_like(log_member_likelihood)
    else:
        tempered_log_member_likelihood = beta * log_member_likelihood

    # log p_tilde = logsumexp_e(beta * log_L_e) - log(E) = log(mean_e(L_e^beta)).
    log_surrogate_likelihood = _log_mean_exp(tempered_log_member_likelihood)

    # Member responsibilities/ESS from the SAME tempered quantities (frozen contract
    # Sec. 22) -- never from the untempered log_member_likelihood.
    responsibilities, member_ess = _responsibilities_and_ess(tempered_log_member_likelihood)

    return Stage4LikelihoodResult(
        log_pixel_likelihood=log_pixel_likelihood,
        log_member_likelihood=log_member_likelihood,
        log_mean_member_likelihood=log_mean_member_likelihood,
        tempered_log_member_likelihood=tempered_log_member_likelihood,
        log_surrogate_likelihood=log_surrogate_likelihood,
        responsibilities=responsibilities,
        member_ess=member_ess,
    )
