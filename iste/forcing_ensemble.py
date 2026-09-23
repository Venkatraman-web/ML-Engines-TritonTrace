"""T08 — prototype surrogate forcing-ensemble perturbation model.

*** PROTOTYPE -- NOT VALIDATED. See progress.md's T08 section for the full sourced-vs-
assumed accounting and the required validation steps before any result built on this
module is trusted. ***

Implements exactly the model specified by the project owner (2026-09-22), refining the
T08 decision-gate investigation's "Option B" (no compatible physically-sourced ensemble
is currently available -- see `progress.md`) into a concrete, lightweight
implementation:

    U_e(x,t) = U(x,t) + delta_U_e(x,t)

with `U(x,t) = (uo(x,t), vo(x,t))` the T01.5-locked base transport velocity (windage
still excluded -- this module never reads `data/wind.nc`), and `delta_U_e(x,t)` a
per-forcing-member perturbation built from:

1. independent random values on a COARSE regular grid (never the full ocean grid, and
   never a dense spatial covariance matrix);
2. an Ornstein-Uhlenbeck (OU) process evolving each coarse-grid cell's coefficient in
   time, with correlation time `correlation_time_seconds` (PROTOTYPE default ~1 day);
3. spatial Gaussian smoothing of the coarse-grid state -- inducing spatial correlation
   at approximately `spatial_correlation_km` (PROTOTYPE default 20 km) -- applied BEFORE
   interpolation to any finer grid, per instruction;
4. bilinear interpolation (`iste.interpolation.bilinear_spatial_interpolate`, T02,
   reused unchanged) of the smoothed coarse field onto the ocean grid or any query point.

Every explicitly numeric physical parameter below (`spatial_correlation_km`,
`correlation_time_seconds`, `sigma_u`) is a **labeled PROTOTYPE ASSUMPTION**, not a
validated physical or model-error estimate. `n_members` (`E`) is carried over from
`README.md`'s own stated "initial target," not newly invented. `coarse_resolution_deg`
and `master_seed` are computational/reproducibility configuration choices, not physical
claims. All are explicit constructor parameters -- nothing here is a hidden constant.

**`delta_U_e` is intentionally NOT forced to be divergence-free** (per instruction,
superseding an earlier streamfunction-based design proposal that was never implemented).
If/when `div(U_e)` is needed, it must be computed from the actual perturbed field
`U_e = U + delta_U_e` using the existing T03 `iste.derivatives.divergence_2d` operator on
that perturbed field -- this module does not compute, assume, or approximate a divergence
value for the perturbation on its own; that wiring is left to whichever later task
(T09+) actually uses forcing members inside the backward step.

Particle-level Brownian diffusion (T05's `sqrt(2K) dB` term) is untouched and entirely
separate from this module -- forcing-member coherence (every particle under member `e`
sees the identical `delta_U_e(x,t)`, by construction, since nothing here depends on
particle identity) is a different mechanism from independent per-particle noise. Neither
T05 (`iste/step.py`) nor T06 (`iste/trajectory.py`) is imported, modified, or duplicated
by this module.

Memory discipline: only a compact per-member coarse-grid coefficient time series is ever
generated or cached (lazily, on first use of a given `member_id`); the expensive
grid-sized smoothing/interpolation step happens on demand at query time and is never
precomputed or stored for the full ocean grid, for all members, or for the full time
history at once.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.ndimage import gaussian_filter

from iste.geometry import MEAN_EARTH_RADIUS_M, metres_per_degree_latitude
from iste.interpolation import bilinear_spatial_interpolate, interpolate_field, linear_temporal_interpolate

if TYPE_CHECKING:
    from iste.environment import GriddedFieldSource


@dataclass(frozen=True)
class ForcingEnsembleConfig:
    """All configurable parameters for the T08 surrogate forcing ensemble.

    `spatial_correlation_km`, `correlation_time_seconds`, and `sigma_u` are PROTOTYPE
    ASSUMPTIONS (see module docstring) -- not validated physical or model-error
    estimates. `n_members` is carried over from `README.md`'s own "initial target."
    `coarse_resolution_deg` and `master_seed` are computational/reproducibility choices.
    """

    n_members: int = 50
    spatial_correlation_km: float = 20.0  # PROTOTYPE ASSUMPTION
    correlation_time_seconds: float = 86400.0  # PROTOTYPE ASSUMPTION (~1 day)
    sigma_u: float = 0.05  # PROTOTYPE ASSUMPTION (m/s)
    coarse_resolution_deg: float = 0.25  # configuration choice, not a physical claim
    master_seed: int = 0

    def __post_init__(self):
        if self.n_members < 1:
            raise ValueError("n_members must be >= 1")
        if self.spatial_correlation_km <= 0:
            raise ValueError("spatial_correlation_km must be > 0")
        if self.correlation_time_seconds <= 0:
            raise ValueError("correlation_time_seconds must be > 0")
        if self.sigma_u < 0:
            raise ValueError("sigma_u must be >= 0")
        if self.coarse_resolution_deg <= 0:
            raise ValueError("coarse_resolution_deg must be > 0")


def _build_coarse_grid(
    lat_min: float, lat_max: float, lon_min: float, lon_max: float,
    resolution_deg: float, pad_cells: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Coarse ascending lat/lon coordinate arrays covering `[lat_min, lat_max] x
    [lon_min, lon_max]` with `pad_cells` extra cells of margin on every side, so bilinear
    interpolation (T02) never falls outside the coarse grid's own coverage for any point
    inside the base domain (T02's strict out-of-range policy would otherwise mark valid
    ocean-domain edge points as invalid purely due to insufficient coarse-grid padding)."""
    pad = pad_cells * resolution_deg
    lat = np.arange(lat_min - pad, lat_max + pad + resolution_deg / 2, resolution_deg)
    lon = np.arange(lon_min - pad, lon_max + pad + resolution_deg / 2, resolution_deg)
    return lat, lon


def _generate_coarse_ou_series(
    rng: np.random.Generator, n_time: int, coarse_shape: tuple[int, int],
    dt_seconds: float, correlation_time_seconds: float, sigma: float,
) -> np.ndarray:
    """`(n_time, *coarse_shape)` Ornstein-Uhlenbeck/AR(1) time series, independent per
    coarse-grid cell (no spatial structure at this stage -- smoothing is applied later,
    at query time, per instruction), via the standard exact OU discretization:

        a[i+1] = a[i] * phi + sigma * sqrt(1 - phi^2) * xi,   phi = exp(-dt/tau)

    with `a[0]` drawn from the process's own stationary distribution `N(0, sigma^2)`.
    """
    phi = float(np.exp(-dt_seconds / correlation_time_seconds))
    series = np.empty((n_time,) + coarse_shape, dtype=np.float64)
    series[0] = rng.normal(0.0, sigma, size=coarse_shape)
    innovation_scale = sigma * np.sqrt(max(1.0 - phi**2, 0.0))
    for i in range(1, n_time):
        series[i] = series[i - 1] * phi + rng.normal(0.0, innovation_scale, size=coarse_shape)
    return series


def _smoothing_sigma_in_cells(
    spatial_correlation_km: float, coarse_resolution_deg: float, reference_lat_deg: float
) -> float:
    """Approximate conversion of a target spatial correlation length (km) into a Gaussian
    smoothing sigma, in coarse-grid CELLS, using the T03 degree<->metre conversion at one
    representative (domain-centre) latitude -- a simplification, since the true
    metres-per-degree-longitude varies across the domain (T03). The resulting *actual*
    correlation length after smoothing + bilinear interpolation is only approximate and
    should be checked empirically (see `notebooks/T08_forcing_ensemble_check.ipynb`), not
    assumed to exactly equal `spatial_correlation_km`.
    """
    km_per_degree_lat = metres_per_degree_latitude(MEAN_EARTH_RADIUS_M) / 1000.0
    target_deg = spatial_correlation_km / km_per_degree_lat
    return target_deg / coarse_resolution_deg


class ForcingEnsemble:
    """*** PROTOTYPE -- NOT VALIDATED (see module docstring). ***

    Generates and caches, per forcing member, only a compact coarse-grid perturbation
    coefficient time series, and exposes the perturbed transport velocity
    `U_e(x,t) = U(x,t) + delta_U_e(x,t)` for any requested `member_id` and `(x, t)` --
    compatible with T02's interpolation conventions.
    """

    def __init__(self, ocean_source: "GriddedFieldSource", config: ForcingEnsembleConfig = ForcingEnsembleConfig()):
        self.ocean_source = ocean_source
        self.config = config

        lat_min, lat_max = float(np.min(ocean_source.lat)), float(np.max(ocean_source.lat))
        lon_min, lon_max = float(np.min(ocean_source.lon)), float(np.max(ocean_source.lon))
        self.coarse_lat, self.coarse_lon = _build_coarse_grid(
            lat_min, lat_max, lon_min, lon_max, config.coarse_resolution_deg
        )
        self._smoothing_sigma_cells = _smoothing_sigma_in_cells(
            config.spatial_correlation_km, config.coarse_resolution_deg, 0.5 * (lat_min + lat_max)
        )
        self._member_seeds = np.random.SeedSequence(config.master_seed).spawn(config.n_members)
        self._cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    @property
    def coarse_shape(self) -> tuple[int, int]:
        return (len(self.coarse_lat), len(self.coarse_lon))

    def _checkpoint_times(self) -> np.ndarray:
        # Reuses the base source's own hourly cadence -- consistent with how U itself is
        # time-interpolated (T02), and avoids inventing a second, independent time grid.
        return self.ocean_source.time

    def _member_state(self, member_id: int) -> tuple[np.ndarray, np.ndarray]:
        """Lazily generate (once) and cache the compact coarse-grid OU coefficient time
        series for `member_id`: `(n_checkpoints, n_coarse_lat, n_coarse_lon)` each for
        the eastward and northward perturbation components. Deterministic given
        `member_id` and `config.master_seed` -- same seed + same member_id always
        reproduces the identical series; different `member_id`s use independent,
        statistically distinct RNG streams (`numpy.random.SeedSequence.spawn`)."""
        if member_id in self._cache:
            return self._cache[member_id]
        if not (0 <= member_id < self.config.n_members):
            raise ValueError(f"member_id must be in [0, {self.config.n_members}); got {member_id}")

        u_seed, v_seed = self._member_seeds[member_id].spawn(2)
        rng_u = np.random.default_rng(u_seed)
        rng_v = np.random.default_rng(v_seed)

        time = self._checkpoint_times()
        if len(time) > 1:
            dt_seconds = float((time[1] - time[0]) / np.timedelta64(1, "s"))
        else:
            dt_seconds = self.config.correlation_time_seconds

        u_series = _generate_coarse_ou_series(
            rng_u, len(time), self.coarse_shape, dt_seconds, self.config.correlation_time_seconds, self.config.sigma_u
        )
        v_series = _generate_coarse_ou_series(
            rng_v, len(time), self.coarse_shape, dt_seconds, self.config.correlation_time_seconds, self.config.sigma_u
        )
        self._cache[member_id] = (u_series, v_series)
        return u_series, v_series

    def perturbation(self, member_id: int, lat, lon, t):
        """`(delta_u, delta_v, valid)` -- the coherent perturbation for forcing member
        `member_id` at `(lat, lon)`, single scalar time `t`. Every caller querying the
        same `member_id`/`t`/`(lat, lon)` gets the identical value (deterministic given
        the member's seed) -- this is what makes a forcing member "coherent" across every
        particle that uses it, as distinct from T05's independent per-particle Brownian
        diffusion.

        Spatial smoothing (Gaussian filter, coarse-grid cells) is applied to the
        time-interpolated coarse state BEFORE bilinear interpolation to `(lat, lon)`, per
        instruction. `delta_u`/`delta_v` are NOT forced to be divergence-free.
        """
        u_series, v_series = self._member_state(member_id)
        time = self._checkpoint_times()

        u_coarse_t, u_t_valid = linear_temporal_interpolate(time, u_series, t)
        v_coarse_t, v_t_valid = linear_temporal_interpolate(time, v_series, t)
        t_ok = bool(np.all(u_t_valid)) and bool(np.all(v_t_valid))

        u_smoothed = gaussian_filter(u_coarse_t, sigma=self._smoothing_sigma_cells, mode="reflect")
        v_smoothed = gaussian_filter(v_coarse_t, sigma=self._smoothing_sigma_cells, mode="reflect")

        delta_u, u_spatial_valid = bilinear_spatial_interpolate(self.coarse_lat, self.coarse_lon, u_smoothed, lat, lon)
        delta_v, v_spatial_valid = bilinear_spatial_interpolate(self.coarse_lat, self.coarse_lon, v_smoothed, lat, lon)

        valid = u_spatial_valid & v_spatial_valid & t_ok
        delta_u = np.where(valid, delta_u, np.nan)
        delta_v = np.where(valid, delta_v, np.nan)

        scalar_query = np.ndim(lat) == 0 and np.ndim(lon) == 0
        if scalar_query:
            return float(delta_u[0]), float(delta_v[0]), bool(valid[0])
        return delta_u, delta_v, valid

    def perturbed_velocity(self, member_id: int, lat, lon, t):
        """`U_e(x,t) = U(x,t) + delta_U_e(x,t)`: the T01.5-locked base transport velocity
        (via `iste.interpolation.interpolate_field` against `uo`/`vo`, unchanged) plus
        this member's coherent perturbation.

        Returns `(u_e, v_e, valid)` -- `valid` requires both the base field and the
        perturbation to be valid at `(lat, lon, t)`.
        """
        u_base, u_base_valid = interpolate_field(self.ocean_source, "uo", lat, lon, t)
        v_base, v_base_valid = interpolate_field(self.ocean_source, "vo", lat, lon, t)
        delta_u, delta_v, pert_valid = self.perturbation(member_id, lat, lon, t)

        scalar_query = np.ndim(lat) == 0 and np.ndim(lon) == 0
        if scalar_query:
            valid = bool(u_base_valid) and bool(v_base_valid) and bool(pert_valid)
            u_e = (u_base + delta_u) if valid else float("nan")
            v_e = (v_base + delta_v) if valid else float("nan")
            return float(u_e), float(v_e), valid

        valid = np.asarray(u_base_valid) & np.asarray(v_base_valid) & np.asarray(pert_valid)
        u_e = np.where(valid, np.asarray(u_base) + np.asarray(delta_u), np.nan)
        v_e = np.where(valid, np.asarray(v_base) + np.asarray(delta_v), np.nan)
        return u_e, v_e, valid

    def cached_member_ids(self) -> tuple[int, ...]:
        """Which members have had their (compact) coarse-grid state generated so far --
        for inspecting/testing memory discipline (nothing is generated eagerly)."""
        return tuple(sorted(self._cache.keys()))

    def member_state_nbytes(self, member_id: int) -> int:
        """Byte size of one cached member's compact coarse-grid state (generates it if
        not already cached) -- for the memory-footprint checks in the T08 notebook/tests."""
        u_series, v_series = self._member_state(member_id)
        return int(u_series.nbytes + v_series.nbytes)
