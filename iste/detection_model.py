"""T15 -- Stage IV observation-model bridge: `h_j,e` -> `pi_j,e`.

## Investigation performed before writing any code (per T15's explicit instructions)

`README.md` names "Stage I -- Observation Operator" in its architecture diagram but does
not itself specify a functional form. `ISTE_Claude_Architecture_v1.0.txt`'s "STAGE I.
OBSERVATION OPERATOR" section is more specific:

    Define a wind-dependent detection probability:
        pi_j = P(y_j = 1 | h_j)
    where h_j is the predicted oil thickness or corresponding observable quantity.
    ...
    The exact functional form of pi_j must be explicitly specified in the
    implementation rather than invented silently.

Repository inspection (this task, before implementing anything) found:

- No satellite/SAR imagery, segmentation output, or detection-probability raster
  anywhere in this repository (`data/` contains only `ocean.nc`/`wind.nc`, both
  environmental fields consumed by Stage III transport, not observation data).
- `iste/observation.py` (T07)'s own docstring already lists "Stage I detection
  probability (`pi_j`)" among the things that module explicitly does **not**
  implement or duplicate -- confirming this was a known, already-documented gap, not
  something newly discovered here.
- No code anywhere in `iste/` computes, stores, or references a numeric `pi_j`/`h_j`
  value, an oil-thickness model, or a wind-dependent detectability curve.
- `iste.likelihood` (T14) explicitly documents `pi_j,e` as "a detection probability
  derived from Stage III's `c_hat*_{j,e}(x,t)` transport estimate... accepted as an
  explicit input array, never recomputed" -- i.e. T14 already anticipated that
  something else (this task) would have to supply `pi_j,e`, and deliberately did not
  build it inline.

**Conclusion: no upstream teammate module, data file, or existing code in this
repository provides `pi_j,e` directly, or the raw physical ingredients (a specified
oil-thickness/backscatter model, a wind-dependent detectability curve, or any numeric
parameters for one) needed to compute it.** Per requirement 4 of this task, this module
therefore implements only the smallest transparent, fully configurable **prototype**
mapping needed to unblock Stage IV development -- explicitly labeled **PROTOTYPE, NOT
VALIDATED** throughout, never presented as a physically authoritative detection model.
If/when an upstream Stage I detection-probability model or dataset becomes available,
this module should be replaced (or given a second, real-model-backed implementation
selected via configuration) rather than assumed correct.

## What this module actually does

`h_j,e` is *some* scalar, per-(observation pixel, forcing member) quantity that
increases with how strongly a candidate source hypothesis's Stage III transport predicts
oil presence at pixel `j` under forcing member `e` (conceptually, something derived from
Stage III's `c_hat*_{j,e}(x,t)` Monte Carlo transport estimate for a specific candidate
source state). **Computing `h_j,e` itself from Stage III's lookup-table output is
explicitly out of scope for this module** (that is a separate, not-yet-built step, per
this task's own "T15 consumes Stage III transport output... provide the mapping
`h_j,e -> pi_j,e`" framing, and per instruction 5 forbidding any change to T14 or Stage
III transport code) -- this module accepts `h` as a given input array and returns `pi`,
nothing more.

The mapping itself is a standard monotone logistic (sigmoid) detection curve:

    pi(h) = sigmoid( steepness * (h - midpoint) )

- `midpoint`: the `h` value at which `pi = 0.5` (PROTOTYPE, ASSUMED -- no physical
  source; a configuration parameter, not a claim).
- `steepness`: how sharply `pi` transitions from near-0 to near-1 as `h` crosses
  `midpoint` (PROTOTYPE, ASSUMED, must be `> 0` so `pi` is strictly increasing in `h`,
  matching the physical requirement that higher predicted oil presence should never
  make the pixel *less* likely to be observed as oil).

A logistic curve is used because it is the standard, simplest monotone S-shaped mapping
from an unbounded real-valued predictor to a `[0, 1]` probability (the same generic
functional family Stage I's own spec names, "P(y_j=1|h_j)", without prescribing which
concrete parametric family to use) -- not a claim about SAR backscatter physics, radar
contrast, or wind-driven oil-mixing.

**This module intentionally has no floor/ceiling parameter of its own.** T14's `epsilon`
already provides the "even a maximally-wrong prediction still has some observation
noise" floor at the `P(y_j|h_j,e)` level (`README.md`'s `epsilon/2` term); adding a
second floor here would double up that role and make T14's `epsilon` harder to
interpret. `pi(h)` here mathematically spans the open interval `(0, 1)` for every finite
`h` (the logistic function's own range) -- in `float64`, a sufficiently extreme `h`
still rounds to exactly `0.0`/`1.0` once `steepness * |h - midpoint|` exceeds ~37 (where
`exp` of it underflows below double precision relative to `1.0`); this is expected
floating-point behavior, not a bug, and `compute_stage4_log_likelihood`'s own `epsilon`
still applies on top regardless.

Deliberately out of scope here (per this task's instructions): computing `h_j,e` from
Stage III's lookup table (no KDE/aggregation logic), satellite segmentation/
classification, Stage III transport physics (`U`/`K`/`grad(K)`/`div(U)`/T05/T06/the
forcing ensemble -- all untouched), T14's likelihood implementation (untouched), and
Stage V/VI/VII/AIS.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ObservationModelConfig:
    """PROTOTYPE, NOT VALIDATED logistic detection-probability parameters. Both fields
    are configuration, not physically sourced/validated constants -- see the module
    docstring's "Sourced vs assumed" accounting.

    - `midpoint`: the `h` value at which `pi(h) = 0.5`.
    - `steepness`: `d(logit(pi))/dh`; must be `> 0` (pi strictly increasing in h).
    """

    midpoint: float
    steepness: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.midpoint):
            raise ValueError(f"midpoint must be finite; got {self.midpoint!r}")
        if not np.isfinite(self.steepness):
            raise ValueError(f"steepness must be finite; got {self.steepness!r}")
        if self.steepness <= 0.0:
            raise ValueError(
                f"steepness must be > 0 so pi(h) is strictly increasing in h; got {self.steepness!r}"
            )


def _stable_sigmoid(z: np.ndarray) -> np.ndarray:
    """`1 / (1 + exp(-z))`, computed without overflow for any finite `z` (large negative
    `z` would overflow `exp(-z)` in the naive formula; large positive `z` is handled by
    the same trick applied to the mathematically equivalent `exp(z) / (1 + exp(z))`)."""
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    neg_exp = np.exp(z[~positive])
    out[~positive] = neg_exp / (1.0 + neg_exp)
    return out


def compute_detection_probability(h: np.ndarray, config: ObservationModelConfig) -> np.ndarray:
    """Map `h_j,e` (any shape -- typically `(n_pixels, n_members)`, matching T14's `pi`
    input contract exactly) to `pi_j,e` via the PROTOTYPE logistic curve above.

    - Output shape == input shape (elementwise mapping) -- the observation-pixel and
      forcing-member association is preserved purely by array position/shape, exactly
      as `iste.likelihood.compute_stage4_log_likelihood` already expects for its own
      `pi` argument; this function attaches no separate identifier bookkeeping.
    - Deterministic: a pure function of `(h, config)`, no randomness.
    - Output is guaranteed finite and in `[0, 1]` for any finite `h` (the logistic
      function's own range is `(0, 1)`; the defensive `clip` below only absorbs
      floating-point rounding at the boundary, never changes a real value).

    Raises `ValueError` if any `h` value is non-finite.
    """
    h = np.asarray(h, dtype=np.float64)
    if not np.all(np.isfinite(h)):
        raise ValueError("h must be finite everywhere -- found NaN/Inf")

    z = config.steepness * (h - config.midpoint)
    pi = _stable_sigmoid(z)
    return np.clip(pi, 0.0, 1.0)
