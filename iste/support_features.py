"""T16 -- Phase 2 candidate-support feature construction.

Implements exactly `README.md` Sec. 14/15's three candidate-support signals, computed
over a T15 `ParticleNeighborhoodIndex` (itself built from valid particles only) around
one query point, plus their explicit, configurable combination into a single `Support`
value:

    D(x,y,t) = number of valid nearby particles              (local particle density)
    W(x,y,t) = sum of Stage III particle weights nearby       (weighted particle support)
    V(x,y,t) = number of distinct observation_pixel_id values in the neighborhood
    Support(x,y,t) = f(D, W, V)

**This module produces Phase 2 candidate-support data only.** It is explicitly NOT:

    Stage III c_hat*
    a probability density
    a historical source posterior
    an observation likelihood
    proof of source location

High support means backward transport histories are spatially concentrated, weighted,
and corroborated by multiple observed pixels near this point -- nothing more.

**Stage III weights are read, never changed.** `log_weight` (T05/T06's log-space
convention, `weight = exp(log_weight)`) is exponentiated only at the point `W` actually
needs the linear sum -- the same convention `iste.diagnostics`/`iste.likelihood` already
use, not a new one. Per `README.md` Sec. 14.2, these remain unnormalized transport/
Jacobian-type corrections; `W` is explicitly NOT, and is never treated as, a probability.

**`D` only ever counts VALID particles**, structurally, not by an extra filter: this
module queries a T15 `ParticleNeighborhoodIndex`, which is itself built over a T14
`HistoricalParticleSlice`'s valid particles only (T15's own "invalid particles are never
indexed" policy) -- so every neighborhood this module ever sees already excludes invalid
particles.

**`Support = f(D, W, V)` and all feature scaling are explicit Phase 2 configuration**
(`README.md` Sec. 15/26: "`f` and all feature scaling/normalization are explicit Phase 2
configuration choices" / "support feature scaling" and "support-combination function"
are both listed as open, non-physical parameters). This module does not invent a
scientifically validated combination formula: `SupportFeatureConfig` exposes explicit,
overridable scaling divisors and linear-combination weights, and `combine_fn` is a
pluggable callable, so a caller can substitute any other combination entirely. The
shipped default (`linear_support_function`) is the smallest transparent choice, labeled
PROTOTYPE / CONFIGURATION, not a claim of scientific validation.

Does not implement candidate-peak/core extraction, forward propagation, or any later
Phase 2 task -- this module's output (`SupportFeatures`, `compute_support_map`'s list of
them) is exactly what a later peak-finding task would consume, not itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from iste.kdtree_neighborhood import NeighborhoodResult, ParticleNeighborhoodIndex, query_knn, query_radius


@dataclass(frozen=True)
class SupportFeatureConfig:
    """PROTOTYPE / CONFIGURATION: explicit scaling and linear-combination weights for
    `D`/`W`/`V`, per `README.md` Sec. 15/26 ("feature scaling must be explicit"). Not a
    scientifically validated normalization -- a configuration choice, freely overridable.

    - `d_scale`/`w_scale`/`v_scale`: divisors applied to `D`/`W`/`V` respectively before
      combination (default `1.0`, i.e. no scaling), so a caller can make the three
      features' numeric units comparable before combining them (e.g. raw particle counts
      and weight sums can otherwise be on very different scales). Must be finite and
      `> 0`.
    - `weight_d`/`weight_w`/`weight_v`: linear-combination coefficients used by the
      DEFAULT combination function (`linear_support_function`); ignored by any custom
      `combine_fn`. Must be finite (may be `0.0`, e.g. to exclude one feature entirely).
    """

    d_scale: float = 1.0
    w_scale: float = 1.0
    v_scale: float = 1.0
    weight_d: float = 1.0
    weight_w: float = 1.0
    weight_v: float = 1.0

    def __post_init__(self) -> None:
        for name, value in (("d_scale", self.d_scale), ("w_scale", self.w_scale), ("v_scale", self.v_scale)):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and > 0; got {value!r}")
        for name, value in (("weight_d", self.weight_d), ("weight_w", self.weight_w), ("weight_v", self.weight_v)):
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite; got {value!r}")


def linear_support_function(d: float, w: float, v: float, config: SupportFeatureConfig) -> float:
    """PROTOTYPE default `f(D, W, V)`: an explicit linear combination of independently
    scaled features, `weight_d*(D/d_scale) + weight_w*(W/w_scale) + weight_v*(V/v_scale)`.
    Not a scientifically validated formula -- `README.md` Sec. 15 leaves `f` as an open
    Phase 2 configuration choice; this is the smallest transparent default, freely
    replaceable via `compute_support_features`'s `combine_fn` parameter."""
    return (
        config.weight_d * (d / config.d_scale)
        + config.weight_w * (w / config.w_scale)
        + config.weight_v * (v / config.v_scale)
    )


@dataclass
class SupportFeatures:
    """Phase 2 candidate-support data at one query point, one historical timestamp
    (carried via `neighborhood`'s source slice) -- explicitly NOT `c_hat*`, a probability
    density, a source posterior, or an observation likelihood (see module docstring).

    - `query_lat`/`query_lon`: the point these features were evaluated at.
    - `density` (`D`): count of valid particles in the neighborhood.
    - `weighted_support` (`W`): sum of the neighborhood's Stage III (linear) weights.
    - `pixel_diversity` (`V`): count of DISTINCT `observation_pixel_id` values in the
      neighborhood -- not the particle count (a neighborhood of 50 particles all from one
      pixel has `V=1`; 5 particles from 5 different pixels has `V=5`).
    - `support`: `f(D, W, V)` via `combine_fn`/`config` (see `compute_support_features`).
    - `neighborhood`: the full T15 `NeighborhoodResult` this was computed from --
      `observation_pixel_id`/`particle_id`/`forcing_member_id` identity remains fully
      available here for provenance and for later Phase 2 tasks (`README.md` Sec. 14.3:
      "observation-pixel identity must remain attached ... through this entire stage").
    - `config`: the `SupportFeatureConfig` actually used.
    """

    query_lat: float
    query_lon: float
    density: int
    weighted_support: float
    pixel_diversity: int
    support: float
    neighborhood: NeighborhoodResult
    config: SupportFeatureConfig


def compute_support_features(
    index: ParticleNeighborhoodIndex,
    lat: float,
    lon: float,
    *,
    radius_m: float | None = None,
    k: int | None = None,
    config: SupportFeatureConfig | None = None,
    combine_fn: Callable[[float, float, float, SupportFeatureConfig], float] = linear_support_function,
) -> SupportFeatures:
    """Compute Phase 2 `D`/`W`/`V`/`Support` at `(lat, lon)` from `index`'s valid
    particles, using T15's neighborhood engine directly (no second spatial-query
    implementation). Exactly one of `radius_m` (fixed-radius) or `k` (k-nearest-neighbor)
    must be given, matching T15's own two supported neighborhood strategies.

    `config` defaults to `SupportFeatureConfig()` (no scaling, equal `1.0` weights) if
    not given. `combine_fn` defaults to `linear_support_function` but may be replaced
    with any other `(D, W, V, config) -> float` callable.
    """
    if (radius_m is None) == (k is None):
        raise ValueError("exactly one of radius_m or k must be given")
    if config is None:
        config = SupportFeatureConfig()

    if radius_m is not None:
        neighborhood = query_radius(index, lat, lon, radius_m)
    else:
        neighborhood = query_knn(index, lat, lon, k)

    density = neighborhood.n_particles
    # W: the linear sum of Stage III weights -- exponentiating log_weight only here, at
    # the point the sum actually needs linear space (never changing the stored weights).
    weighted_support = float(np.sum(np.exp(neighborhood.log_weight)))
    pixel_diversity = len(set(neighborhood.observation_pixel_id.tolist()))

    support = float(combine_fn(density, weighted_support, pixel_diversity, config))

    return SupportFeatures(
        query_lat=float(lat), query_lon=float(lon),
        density=density, weighted_support=weighted_support, pixel_diversity=pixel_diversity,
        support=support, neighborhood=neighborhood, config=config,
    )


def compute_support_map(
    index: ParticleNeighborhoodIndex,
    query_lats,
    query_lons,
    *,
    radius_m: float | None = None,
    k: int | None = None,
    config: SupportFeatureConfig | None = None,
    combine_fn: Callable[[float, float, float, SupportFeatureConfig], float] = linear_support_function,
) -> list[SupportFeatures]:
    """`compute_support_features` evaluated at every `(lat, lon)` pair in `query_lats`/
    `query_lons` -- the Phase 2 "candidate-support heat map" (`README.md` Sec. 15) as a
    flat list of per-point features, in the same order as the input points. Building an
    actual gridded/rasterized map, finding local maxima, or extracting candidate cores is
    explicitly out of this task's scope (T17+)."""
    query_lats = np.atleast_1d(np.asarray(query_lats, dtype=np.float64))
    query_lons = np.atleast_1d(np.asarray(query_lons, dtype=np.float64))
    if query_lats.shape != query_lons.shape:
        raise ValueError(
            f"query_lats and query_lons must have the same shape; got {query_lats.shape} and {query_lons.shape}"
        )
    return [
        compute_support_features(index, lat, lon, radius_m=radius_m, k=k, config=config, combine_fn=combine_fn)
        for lat, lon in zip(query_lats, query_lons)
    ]
