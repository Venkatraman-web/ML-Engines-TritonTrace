"""Stage III environmental field interface (T01).

This module provides:

1. `load_field_dataset` -- a generic NetCDF loader/adapter that reads a small set of named
   variables plus time/latitude/longitude coordinates from any NetCDF file, explicitly
   masks configured fill values, and returns an in-memory `GriddedFieldSource`. It does not
   assume a file is specifically "the ocean file" or "the wind file" -- the caller supplies
   which variables and coordinate names to read, so it works for both `data/ocean.nc` and
   `data/wind.nc` (which use different dimension names and different latitude orderings)
   without hard-coding either layout into this module.

2. `GriddedFieldSource` / `GriddedFieldMetadata` -- the loaded data plus a compact
   diagnostic summary (dimensions, timestamp range, coordinate directions, missing counts)
   used for T01's manual check.

3. `EnvironmentalFieldProvider` -- the abstract Stage III field interface described in
   `README.md`: `U(x, t)`, `K(x, t)`, `grad(K)(x, t)`, `div(U)(x, t)`, `validity(x, t)`.
   T01 fixed only the *contract*; `NetCDFEnvironmentalFieldProvider` (below) now
   implements every method: `U`/`validity` via T02 interpolation (unchanged since T02),
   `div_U` via `interpolated_divergence` (T02+T03), and `K`/`grad_K` via
   `interpolated_k_value_and_gradient` (T02+T03) against an explicitly supplied `K` grid.
   These two helper functions are the same numerical machinery
   `iste.step.backward_euler_maruyama_step` uses internally -- factored out here so the
   provider interface and the live T05 pipeline share one implementation, never two.

`K` is not present in either supplied NetCDF file, and `README.md` gives no formula for it
in Stage III v1 -- a concrete provider must be given an explicit `K` grid through its
constructor (`k_lat`/`k_lon`/`k_field`); if none is given, `K`/`grad_K` raise a clear
`ValueError` rather than inventing a default. Likewise, how ocean and wind velocities are
combined into `U` (windage) remains excluded per the T01.5 decision -- this module never
reads wind data for `U`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import xarray as xr

from iste.derivatives import divergence_2d, spatial_gradient
from iste.geometry import MEAN_EARTH_RADIUS_M
from iste.interpolation import bilinear_spatial_interpolate, interpolate_field, time_bracket


@dataclass(frozen=True)
class VariableMetadata:
    """Per-variable diagnostics for one loaded field."""

    name: str
    shape: tuple[int, ...]
    missing_count: int
    total_count: int
    raw_min: float
    raw_max: float


@dataclass(frozen=True)
class GriddedFieldMetadata:
    """Summary diagnostics for one loaded NetCDF source (T01 manual-check output)."""

    source_path: str
    time_name: str
    n_timestamps: int
    time_start: str
    time_end: str
    lat_name: str
    n_lat: int
    lat_ascending: bool
    lat_min: float
    lat_max: float
    lon_name: str
    n_lon: int
    lon_min: float
    lon_max: float
    variables: tuple[VariableMetadata, ...]

    def as_table(self) -> str:
        """Render a compact human-readable diagnostic table for printing/manual check."""
        lines = [
            f"source:      {self.source_path}",
            f"time:        {self.time_name}  n={self.n_timestamps}  "
            f"{self.time_start} -> {self.time_end}",
            f"latitude:    {self.lat_name}  n={self.n_lat}  "
            f"[{self.lat_min:.4f}, {self.lat_max:.4f}]  "
            f"{'ascending' if self.lat_ascending else 'descending'}",
            f"longitude:   {self.lon_name}  n={self.n_lon}  "
            f"[{self.lon_min:.4f}, {self.lon_max:.4f}]",
        ]
        for v in self.variables:
            lines.append(
                f"  {v.name:8s} shape={v.shape}  "
                f"missing={v.missing_count}/{v.total_count}  "
                f"raw_range=[{v.raw_min:.6g}, {v.raw_max:.6g}]"
            )
        return "\n".join(lines)


@dataclass
class GriddedFieldSource:
    """A masked, in-memory view of a subset of variables read from one NetCDF file.

    This is the T01 "loader": it performs I/O and explicit fill-value masking only. It
    does not interpolate and does not compute derived quantities (grad/div) -- see
    `EnvironmentalFieldProvider` for that seam.
    """

    source_path: str
    time_name: str
    time: np.ndarray  # datetime64[ns], 1-D
    lat_name: str
    lat: np.ndarray  # 1-D, native order (ascending or descending -- not normalized here)
    lon_name: str
    lon: np.ndarray  # 1-D
    data: Mapping[str, np.ndarray]  # variable name -> float64 array (time, lat, lon), NaN-masked
    fill_threshold: float

    @property
    def lat_ascending(self) -> bool:
        return bool(np.all(np.diff(self.lat) > 0))

    def missing_count(self, variable: str) -> int:
        return int(np.isnan(self.data[variable]).sum())

    def describe(self) -> GriddedFieldMetadata:
        """Compute the T01 manual-check diagnostics: dims, timestamp range, coordinate
        directions, and missing counts."""
        variables = []
        for name, arr in self.data.items():
            finite = arr[~np.isnan(arr)]
            variables.append(
                VariableMetadata(
                    name=name,
                    shape=arr.shape,
                    missing_count=self.missing_count(name),
                    total_count=arr.size,
                    raw_min=float(finite.min()) if finite.size else float("nan"),
                    raw_max=float(finite.max()) if finite.size else float("nan"),
                )
            )
        return GriddedFieldMetadata(
            source_path=self.source_path,
            time_name=self.time_name,
            n_timestamps=len(self.time),
            time_start=str(self.time[0]),
            time_end=str(self.time[-1]),
            lat_name=self.lat_name,
            n_lat=len(self.lat),
            lat_ascending=self.lat_ascending,
            lat_min=float(np.min(self.lat)),
            lat_max=float(np.max(self.lat)),
            lon_name=self.lon_name,
            n_lon=len(self.lon),
            lon_min=float(np.min(self.lon)),
            lon_max=float(np.max(self.lon)),
            variables=tuple(variables),
        )


def load_field_dataset(
    path: str,
    variables: Sequence[str],
    *,
    time_dim: str,
    lat_dim: str,
    lon_dim: str,
    fill_threshold: float = 1e19,
) -> GriddedFieldSource:
    """Load `variables` plus (`time_dim`, `lat_dim`, `lon_dim`) coordinates from a NetCDF
    file at `path`, explicitly masking values whose magnitude is >= `fill_threshold`.

    Coordinate/dimension names are supplied by the caller rather than assumed, because the
    two datasets used in this project (`data/ocean.nc`, `data/wind.nc`) use different names
    (`time` vs `valid_time`) and different latitude orderings (ascending vs descending).
    Masking is done explicitly here (an explicit magnitude threshold, not `xarray`'s
    automatic `mask_and_scale`), per `CLAUDE.md`'s instruction not to silently rely on a
    file-format default for a scientific masking decision. The raw NetCDF file itself is
    never modified -- this function only reads it.

    This function performs no spatial/temporal interpolation and computes no derived
    quantities; it is the T01 loader only.
    """
    ds = xr.open_dataset(path, mask_and_scale=False)
    try:
        for dim_name, dim_kind in ((time_dim, "time"), (lat_dim, "latitude"), (lon_dim, "longitude")):
            if dim_name not in ds.coords:
                raise KeyError(
                    f"{dim_kind} coordinate '{dim_name}' not found in {path}: {list(ds.coords)}"
                )

        time = ds[time_dim].values
        lat = ds[lat_dim].values.astype(np.float64)
        lon = ds[lon_dim].values.astype(np.float64)

        data: dict[str, np.ndarray] = {}
        for name in variables:
            if name not in ds.data_vars:
                raise KeyError(f"variable '{name}' not found in {path}: {list(ds.data_vars)}")
            raw = ds[name].values.astype(np.float64)
            data[name] = np.where(np.abs(raw) >= fill_threshold, np.nan, raw)

        return GriddedFieldSource(
            source_path=path,
            time_name=time_dim,
            time=time,
            lat_name=lat_dim,
            lat=lat,
            lon_name=lon_dim,
            lon=lon,
            data=data,
            fill_threshold=fill_threshold,
        )
    finally:
        ds.close()


def interpolated_divergence(
    ocean_source: GriddedFieldSource,
    lat: np.ndarray,
    lon: np.ndarray,
    t,
    radius_m: float = MEAN_EARTH_RADIUS_M,
) -> tuple[np.ndarray, np.ndarray]:
    """`div(U)` at continuous query points `(lat, lon)` and time `t`, from the same
    `uo`/`vo` grid `ocean_source` provides: T03's `divergence_2d` on the two hourly
    snapshots bracketing `t` (T02's `time_bracket`), linearly blended by the same time
    fraction, then bilinearly interpolated (T02) to the query points.

    This is the exact computation `iste.step.backward_euler_maruyama_step` needs for its
    weight update -- factored out here so that function and
    `NetCDFEnvironmentalFieldProvider.div_U` both call this one implementation, rather
    than each containing its own independent copy of the same bracket-and-blend math
    (`CLAUDE.md`: do not duplicate scientific/numeric logic).

    `lat`/`lon` may be scalars or arrays; returns `(div_val, valid)` broadcast to match.
    """
    lat = np.atleast_1d(np.asarray(lat, dtype=np.float64))
    lon = np.atleast_1d(np.asarray(lon, dtype=np.float64))

    i_lo, i_hi, frac, t_in_range = time_bracket(ocean_source.time, t)
    div_lo, _ = divergence_2d(
        ocean_source.lat, ocean_source.lon,
        ocean_source.data["uo"][i_lo], ocean_source.data["vo"][i_lo], radius_m=radius_m,
    )
    div_hi, _ = divergence_2d(
        ocean_source.lat, ocean_source.lon,
        ocean_source.data["uo"][i_hi], ocean_source.data["vo"][i_hi], radius_m=radius_m,
    )
    div_lo_val, div_lo_ok = bilinear_spatial_interpolate(ocean_source.lat, ocean_source.lon, div_lo, lat, lon)
    div_hi_val, div_hi_ok = bilinear_spatial_interpolate(ocean_source.lat, ocean_source.lon, div_hi, lat, lon)
    div_val = (1 - frac) * div_lo_val + frac * div_hi_val
    valid = div_lo_ok & div_hi_ok & t_in_range
    return div_val, valid


def interpolated_k_value_and_gradient(
    k_lat: np.ndarray,
    k_lon: np.ndarray,
    k_field: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    radius_m: float = MEAN_EARTH_RADIUS_M,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """`K` and `grad(K)` at continuous query points `(lat, lon)`, from an explicitly
    supplied, time-invariant `K` grid (`k_lat`, `k_lon`, `k_field`) -- `K` has no defined
    physical formula for Stage III v1 (T01.5/T03), so it is always externally supplied,
    never invented here or anywhere downstream of this function.

    T03's `spatial_gradient` on `k_field`, then T02's `bilinear_spatial_interpolate` for
    both `K` itself and its gradient -- the exact computation
    `iste.step.backward_euler_maruyama_step` needs; factored out here so that function and
    `NetCDFEnvironmentalFieldProvider.K`/`grad_K` both call this one implementation,
    rather than each containing its own independent copy.

    `lat`/`lon` may be scalars or arrays; returns `(k_val, dK_dx, dK_dy, valid)` broadcast
    to match. `K` is treated as time-invariant here, matching T05's own existing
    simplification (no real time-varying `K` source exists yet) -- this is not a new
    simplification introduced by this function.
    """
    lat = np.atleast_1d(np.asarray(lat, dtype=np.float64))
    lon = np.atleast_1d(np.asarray(lon, dtype=np.float64))

    k_val, k_valid = bilinear_spatial_interpolate(k_lat, k_lon, k_field, lat, lon)
    dK_dx_grid, dK_dy_grid, _ = spatial_gradient(k_lat, k_lon, k_field, radius_m=radius_m)
    dK_dx, dK_dx_valid = bilinear_spatial_interpolate(k_lat, k_lon, dK_dx_grid, lat, lon)
    dK_dy, dK_dy_valid = bilinear_spatial_interpolate(k_lat, k_lon, dK_dy_grid, lat, lon)
    valid = k_valid & dK_dx_valid & dK_dy_valid
    return k_val, dK_dx, dK_dy, valid


class EnvironmentalFieldProvider(ABC):
    """Stage III normalized environmental field interface (`README.md`).

    Later stages (starting at T05, the backward-adjoint step) must depend only on this
    interface -- `U`, `K`, `grad_K`, `div_U`, `validity` -- and never on `xarray`/NetCDF
    specifics, so the transport and weight equations stay independent of the data layout
    (`CLAUDE.md`: "Do not hard-code a particular NetCDF layout into scientific equations").

    This class fixes only the contract -- no method is implemented here (every method is
    `@abstractmethod`, so a subclass cannot be instantiated until it implements all five;
    this interface cannot be silently used as if it already performed particle
    propagation). `NetCDFEnvironmentalFieldProvider` below is the concrete implementation
    that actually evaluates all five at an arbitrary continuous `(x, t)`:

    - `U`, `K`, `grad_K`, `div_U`, and `validity` all require evaluating a field at an
      arbitrary continuous `(x, t)`, which needs bilinear spatial + linear temporal
      interpolation of the underlying grids (T02).
    - `grad_K` and `div_U` additionally require the finite-difference operators built in
      T03.
    - `K` additionally requires an externally supplied diffusivity definition -- `README.md`
      gives no formula for Stage III v1, so a concrete provider must be given one
      explicitly (via its constructor). Nothing here, or in `NetCDFEnvironmentalFieldProvider`,
      invents a value or formula for `K`; if none was supplied, `K`/`grad_K` raise a clear
      configuration error rather than a silent placeholder.
    - `U` additionally requires an explicit decision about how ocean and wind velocities
      are combined (windage) -- `NetCDFEnvironmentalFieldProvider` implements the
      T01.5-locked decision (`U = (uo, vo)`, windage excluded).
    """

    @abstractmethod
    def U(self, x: np.ndarray, t: np.datetime64) -> np.ndarray:
        """Combined horizontal transport velocity at `x=(lon, lat)`, time `t`.

        Units m/s, shape (2,): (u_x, u_y)."""
        raise NotImplementedError

    @abstractmethod
    def K(self, x: np.ndarray, t: np.datetime64) -> float:
        """Scalar, horizontally isotropic diffusivity at `x`, time `t`. Units m^2/s."""
        raise NotImplementedError

    @abstractmethod
    def grad_K(self, x: np.ndarray, t: np.datetime64) -> np.ndarray:
        """Spatial gradient of `K` at `x`, time `t`. Shape (2,)."""
        raise NotImplementedError

    @abstractmethod
    def div_U(self, x: np.ndarray, t: np.datetime64) -> float:
        """Divergence of `U` at `x`, time `t`, computed consistently with the same `U`
        used for transport (per `README.md`'s weight-update requirement)."""
        raise NotImplementedError

    @abstractmethod
    def validity(self, x: np.ndarray, t: np.datetime64) -> bool:
        """Whether `(x, t)` lies inside the valid ocean/data domain (not land, not
        missing data, not outside the grid extent)."""
        raise NotImplementedError


class NetCDFEnvironmentalFieldProvider(EnvironmentalFieldProvider):
    """T02/T05-integration concrete provider -- `U`, `K`, `grad_K`, `div_U`, and
    `validity` are all functional, each wired to the exact same numerical machinery the
    live Stage III pipeline (`iste.step.backward_euler_maruyama_step`) uses, never a
    second/duplicate implementation:

    - `U(x, t)`: the T01.5-locked Stage III v1 definition,
      `U(x,t) = (uo(x,t), vo(x,t))` -- the Copernicus ocean-surface current only, via
      `iste.interpolation.interpolate_field`. Windage is excluded from `U` in v1 (see
      `tasks.md`/`memory.md` T01.5); this class never reads the wind dataset. Unchanged
      from the original T02 implementation.
    - `div_U(x, t)`: `interpolated_divergence` (above) -- T03's `divergence_2d` applied to
      this provider's own `uo`/`vo` grid, time-blended and spatially interpolated exactly
      as `iste.step` does internally.
    - `K(x, t)` / `grad_K(x, t)`: `interpolated_k_value_and_gradient` (above) -- `K` has
      no defined physical formula for Stage III v1 (see
      `EnvironmentalFieldProvider.K`'s docstring), so it is never invented here. A
      concrete `K` grid must be supplied explicitly via this class's constructor
      (`k_lat`/`k_lon`/`k_field`); if it was not, `K`/`grad_K` raise a `ValueError`
      stating exactly that, rather than returning a placeholder or inventing a default.
    - `validity(x, t)`: `True` iff both `uo` and `vo` interpolate validly at `(x, t)`
      (inside the data domain, not masked/missing) -- this is the *interpolation*
      validity from T02. `iste.boundary.netcdf_validity_check` (T04) is built from this
      exact same `iste.interpolation.interpolate_field` validity semantics, so this
      method already agrees with T04's domain check for the same `(x, t)`; T04's
      stateful stop-on-invalid *policy* (which additionally depends on whether a particle
      was already invalid) is a separate, particle-history-dependent concern that does
      not fit a pure `(x, t) -> bool` signature and is intentionally not reproduced here.
    """

    def __init__(
        self,
        ocean_source: GriddedFieldSource,
        k_lat: np.ndarray | None = None,
        k_lon: np.ndarray | None = None,
        k_field: np.ndarray | None = None,
        radius_m: float = MEAN_EARTH_RADIUS_M,
    ):
        if "uo" not in ocean_source.data or "vo" not in ocean_source.data:
            raise ValueError(
                "ocean_source must include both 'uo' and 'vo' for the locked Stage III "
                "v1 definition U(x,t) = (uo(x,t), vo(x,t))"
            )
        k_supplied = (k_lat is not None, k_lon is not None, k_field is not None)
        if any(k_supplied) and not all(k_supplied):
            raise ValueError(
                "k_lat, k_lon, and k_field must be supplied together, or not supplied at all"
            )
        self._ocean = ocean_source
        self._k_lat = k_lat
        self._k_lon = k_lon
        self._k_field = k_field
        self._radius_m = radius_m

    def U(self, x: np.ndarray, t: np.datetime64) -> np.ndarray:
        lon, lat = float(x[0]), float(x[1])
        u, u_valid = interpolate_field(self._ocean, "uo", lat, lon, t)
        v, v_valid = interpolate_field(self._ocean, "vo", lat, lon, t)
        if not (u_valid and v_valid):
            return np.array([np.nan, np.nan])
        return np.array([u, v])

    def _require_k(self) -> None:
        if self._k_field is None:
            raise ValueError(
                "K was not configured for this provider. Stage III v1 has no physically "
                "sourced diffusivity formula (see T01.5/T03), so K must be supplied "
                "explicitly: NetCDFEnvironmentalFieldProvider(ocean_source, "
                "k_lat=..., k_lon=..., k_field=...). No default K is invented here."
            )

    def K(self, x: np.ndarray, t: np.datetime64) -> float:
        self._require_k()
        lon, lat = float(x[0]), float(x[1])
        k_val, _, _, valid = interpolated_k_value_and_gradient(
            self._k_lat, self._k_lon, self._k_field, lat, lon, radius_m=self._radius_m
        )
        return float(k_val[0]) if valid[0] else float("nan")

    def grad_K(self, x: np.ndarray, t: np.datetime64) -> np.ndarray:
        self._require_k()
        lon, lat = float(x[0]), float(x[1])
        _, dK_dx, dK_dy, valid = interpolated_k_value_and_gradient(
            self._k_lat, self._k_lon, self._k_field, lat, lon, radius_m=self._radius_m
        )
        if not valid[0]:
            return np.array([np.nan, np.nan])
        return np.array([dK_dx[0], dK_dy[0]])

    def div_U(self, x: np.ndarray, t: np.datetime64) -> float:
        lon, lat = float(x[0]), float(x[1])
        div_val, valid = interpolated_divergence(self._ocean, lat, lon, t, radius_m=self._radius_m)
        return float(div_val[0]) if valid[0] else float("nan")

    def validity(self, x: np.ndarray, t: np.datetime64) -> bool:
        lon, lat = float(x[0]), float(x[1])
        _, u_valid = interpolate_field(self._ocean, "uo", lat, lon, t)
        _, v_valid = interpolate_field(self._ocean, "vo", lat, lon, t)
        return bool(u_valid and v_valid)
