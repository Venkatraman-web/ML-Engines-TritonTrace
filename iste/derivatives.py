"""T03 — spatial derivative operators: `grad(K)` and `div(U)` (`README.md` Stage III).

Provides the generic finite-difference machinery Stage III needs, at the grid level (one
`(n_lat, n_lon)` field slice at a time -- matching `iste.environment.GriddedFieldSource`'s
layout and mirroring `iste.interpolation`'s separation of the spatial operator from any
particular NetCDF layout):

- `spatial_gradient(lat_coords, lon_coords, field)` -- the gradient of any supplied
  scalar field, in physical units (per metre). This module does **not** choose a value or
  formula for `K` -- pass whatever scalar field you have (a test field, or, once decided,
  an actual `K(x,t)` field); see `iste.environment.EnvironmentalFieldProvider.K` for the
  open decision.
- `divergence_2d(lat_coords, lon_coords, u_component, v_component)` -- the divergence of
  any supplied 2-D vector field. For Stage III v1, callers pass the T01.5-locked
  `U = (uo, vo)` (windage excluded); this function itself does not know or care what the
  vector field represents.

Both:

- convert degree spacing to metres via `iste.geometry` (never treating a degree of
  latitude/longitude as a constant, interchangeable unit of distance, and correctly
  accounting for longitude's latitude-dependent physical spacing);
- use `numpy.gradient` with `edge_order=2`: central (2nd-order accurate) differences at
  interior grid points, one-sided 2nd-order-accurate differences at the two edges of each
  axis -- never extrapolating beyond the supplied grid;
- work identically for ascending (ocean) or descending (wind-style) latitude coordinates,
  since `numpy.gradient` uses the actual (possibly negative) coordinate spacing;
- propagate invalidity explicitly: a derivative at grid cell `(i, j)` is `NaN`/invalid
  whenever any point in its finite-difference stencil (itself, plus its 1 or 2 neighbours
  along that axis -- 3 points at an edge, 3 points -- including itself -- at an interior
  point) is itself `NaN` (masked/missing/land). This is the same explicit validity policy
  as `iste.interpolation`: invalid/missing source data never silently produces a
  valid-looking derivative.
"""

from __future__ import annotations

import numpy as np

from iste.geometry import MEAN_EARTH_RADIUS_M, metres_per_degree_latitude, metres_per_degree_longitude


def _axis_stencil_valid(finite: np.ndarray, axis: int) -> np.ndarray:
    """Boolean mask, same shape as `finite`, `True` at grid index `i` along `axis` iff
    every point in `i`'s finite-difference stencil is `True` in `finite`:

    - interior `i` (`0 < i < n-1`): stencil is `{i-1, i, i+1}` (central difference; `i`
      itself is included so that a derivative is never reported "valid" *at* a masked
      location, even though the central-difference formula only algebraically uses its
      neighbours);
    - edge `i = 0` or `i = n-1` (`edge_order=2`): stencil is the 3 points at that edge
      (`{0,1,2}` or `{n-3,n-2,n-1}`).
    """
    n = finite.shape[axis]
    if n < 3:
        raise ValueError(f"need at least 3 points along axis {axis} for edge_order=2 differences")

    def sl(a: int, b: int) -> tuple:
        idx = [slice(None)] * finite.ndim
        idx[axis] = slice(a, b)
        return tuple(idx)

    valid = np.zeros_like(finite, dtype=bool)
    valid[sl(1, n - 1)] = finite[sl(0, n - 2)] & finite[sl(1, n - 1)] & finite[sl(2, n)]
    valid[sl(0, 1)] = finite[sl(0, 1)] & finite[sl(1, 2)] & finite[sl(2, 3)]
    valid[sl(n - 1, n)] = finite[sl(n - 3, n - 2)] & finite[sl(n - 2, n - 1)] & finite[sl(n - 1, n)]
    return valid


def _directional_finite_difference(
    coords: np.ndarray, field: np.ndarray, axis: int
) -> tuple[np.ndarray, np.ndarray]:
    """`d(field)/d(coords)` along `axis` (degrees-of-`coords` units, not yet converted to
    metres), plus the companion stencil-validity mask from `_axis_stencil_valid`.

    `coords` must be strictly monotonic (ascending or descending); `numpy.gradient`
    handles either correctly since it uses the actual (signed) spacing between points.
    """
    coords = np.asarray(coords, dtype=np.float64)
    field = np.asarray(field, dtype=np.float64)
    if coords.ndim != 1:
        raise ValueError("coords must be 1-D")
    if field.shape[axis] != coords.shape[0]:
        raise ValueError(
            f"field.shape[{axis}] ({field.shape[axis]}) must match len(coords) ({coords.shape[0]})"
        )
    diffs = np.diff(coords)
    if not (np.all(diffs > 0) or np.all(diffs < 0)):
        raise ValueError("coords must be strictly monotonic (ascending or descending)")

    deriv = np.gradient(field, coords, axis=axis, edge_order=2)
    finite = ~np.isnan(field)
    axis_valid = _axis_stencil_valid(finite, axis)
    return deriv, axis_valid


def spatial_gradient(
    lat_coords: np.ndarray,
    lon_coords: np.ndarray,
    field: np.ndarray,
    radius_m: float = MEAN_EARTH_RADIUS_M,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Physical-space gradient of a scalar `field` (shape `(n_lat, n_lon)`) on a lat/lon
    grid indexed by `lat_coords`/`lon_coords`.

    Returns `(d_dx, d_dy, valid)`, each shape `(n_lat, n_lon)`:

    - `d_dx[i, j]`: `d(field)/dx`, `x` = eastward physical distance -- units
      `[field units] / metre`.
    - `d_dy[i, j]`: `d(field)/dy`, `y` = northward physical distance -- units
      `[field units] / metre`.
    - `valid[i, j]`: `True` iff both directional finite-difference stencils at that cell
      (see module docstring) consist entirely of finite `field` values.

    `lat_coords` may be ascending (ocean grid) or descending (wind grid); `lon_coords` is
    handled the same way for generality. This function performs no interpolation and
    makes no assumption about what `field` represents -- it does not choose a value or
    formula for `K`.
    """
    field = np.asarray(field, dtype=np.float64)
    if field.shape != (len(lat_coords), len(lon_coords)):
        raise ValueError(
            f"field shape {field.shape} does not match "
            f"(len(lat_coords), len(lon_coords)) = ({len(lat_coords)}, {len(lon_coords)})"
        )

    d_dlon_deg, lon_valid = _directional_finite_difference(lon_coords, field, axis=1)
    d_dlat_deg, lat_valid = _directional_finite_difference(lat_coords, field, axis=0)

    dlat_per_deg = metres_per_degree_latitude(radius_m)
    dlon_per_deg = metres_per_degree_longitude(np.asarray(lat_coords, dtype=np.float64), radius_m)

    d_dx = d_dlon_deg / dlon_per_deg[:, None]  # varies by row (latitude)
    d_dy = d_dlat_deg / dlat_per_deg  # constant (spherical approximation)

    valid = lon_valid & lat_valid
    d_dx = np.where(valid, d_dx, np.nan)
    d_dy = np.where(valid, d_dy, np.nan)
    return d_dx, d_dy, valid


def divergence_2d(
    lat_coords: np.ndarray,
    lon_coords: np.ndarray,
    u_component: np.ndarray,
    v_component: np.ndarray,
    radius_m: float = MEAN_EARTH_RADIUS_M,
) -> tuple[np.ndarray, np.ndarray]:
    """Divergence `div(U) = d(u)/dx + d(v)/dy` of a 2-D vector field
    `(u_component, v_component)` = (eastward, northward) on the same lat/lon grid.

    Units are `[u/v units] / metre` (e.g. `1/s` if `u_component`/`v_component` are m/s).

    For Stage III v1, callers pass the T01.5-locked `U = (uo, vo)` -- windage excluded;
    this function itself is generic over any 2-D vector field and does not add or assume
    any additional velocity component.

    Returns `(div_u, valid)`. `valid[i, j]` requires only the two stencils this cell's
    divergence actually depends on: `u_component`'s eastward stencil and
    `v_component`'s northward stencil (not `u_component`'s northward stencil or
    `v_component`'s eastward stencil, which this function never uses).
    """
    u_component = np.asarray(u_component, dtype=np.float64)
    v_component = np.asarray(v_component, dtype=np.float64)
    expected_shape = (len(lat_coords), len(lon_coords))
    if u_component.shape != expected_shape or v_component.shape != expected_shape:
        raise ValueError(
            f"u_component/v_component shape must be {expected_shape}; "
            f"got {u_component.shape} and {v_component.shape}"
        )

    du_dlon_deg, u_lon_valid = _directional_finite_difference(lon_coords, u_component, axis=1)
    dv_dlat_deg, v_lat_valid = _directional_finite_difference(lat_coords, v_component, axis=0)

    dlat_per_deg = metres_per_degree_latitude(radius_m)
    dlon_per_deg = metres_per_degree_longitude(np.asarray(lat_coords, dtype=np.float64), radius_m)

    du_dx = du_dlon_deg / dlon_per_deg[:, None]
    dv_dy = dv_dlat_deg / dlat_per_deg

    valid = u_lon_valid & v_lat_valid
    div_u = np.where(valid, du_dx + dv_dy, np.nan)
    return div_u, valid
