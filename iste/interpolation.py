"""T02 — spatial and temporal interpolation (README.md: "spatial and temporal
interpolation are separate concerns").

This module is purely numerical: it interpolates NumPy arrays given 1-D coordinate
arrays, and knows nothing about `xarray`/NetCDF, `uo`/`vo`/`u10`/`v10`, or any particular
file's layout. It is deliberately generic so it works unchanged on the ocean grid
(ascending latitude, 149x216) and the wind grid (descending latitude, 25x36), and would
equally interpolate a future externally supplied `K` field once one exists -- nothing
here invents a value or formula for `K`.

Three functions are provided, matching `tasks.md` T02:

- `bilinear_spatial_interpolate` -- spatial interpolation of one 2-D field slice at a set
  of (lat, lon) query points.
- `linear_temporal_interpolate` -- linear interpolation in time of an array whose leading
  axis is time, at a set of query times.
- `interpolate_field` -- the convenience composition of the two above, answering T02's
  stated goal directly: given (lat, lon, t), return the interpolated field value from a
  `GriddedFieldSource` (see `iste.environment`).

Validity policy (shared by all three): a query point's interpolated value is `NaN` (and
its companion `valid` flag `False`) whenever any of the source values it depends on is
itself `NaN` (missing/masked ocean cell), or whenever the query point falls outside the
covered coordinate range. Invalid/missing source data never silently becomes a
valid-looking interpolated number. This validity is purely about *interpolation* input
data and grid/time coverage -- it is not the Stage III particle boundary/land policy,
which is a later, separate concern (T04).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from iste.environment import GriddedFieldSource


def _bracket(coords: np.ndarray, query: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Bracket `query` values against a strictly monotonic 1-D `coords` array.

    `coords` may be ascending or descending (this is what lets the same interpolation
    code handle the ocean grid's ascending latitude and the wind grid's descending
    latitude without special-casing either).

    Returns `(idx_lo, idx_hi, frac, in_range)`, all shaped like `query`, such that for an
    in-range query:

        value ~= (1 - frac) * coords_field[idx_lo] + frac * coords_field[idx_hi]

    `in_range` is `False` wherever `query` lies outside `[min(coords), max(coords)]`; for
    those points `frac`/`idx_lo`/`idx_hi` are still finite (clamped to the nearest edge
    interval) but callers must treat the result as invalid rather than extrapolating.
    """
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 1 or coords.shape[0] < 2:
        raise ValueError("coords must be a 1-D array with at least two points")
    diffs = np.diff(coords)
    if np.all(diffs > 0):
        ascending = True
    elif np.all(diffs < 0):
        ascending = False
    else:
        raise ValueError("coords must be strictly monotonic (ascending or descending)")

    query = np.asarray(query, dtype=np.float64)
    work = coords if ascending else coords[::-1]

    idx_hi_w = np.searchsorted(work, query, side="left")
    idx_hi_w = np.clip(idx_hi_w, 1, len(work) - 1)
    idx_lo_w = idx_hi_w - 1

    lo_val = work[idx_lo_w]
    hi_val = work[idx_hi_w]
    span = hi_val - lo_val
    frac = np.where(span != 0, (query - lo_val) / np.where(span == 0, 1.0, span), 0.0)

    in_range = (query >= work[0]) & (query <= work[-1])

    if ascending:
        idx_lo, idx_hi = idx_lo_w, idx_hi_w
    else:
        n = len(coords)
        idx_lo, idx_hi = (n - 1) - idx_lo_w, (n - 1) - idx_hi_w

    return idx_lo, idx_hi, frac, in_range


def bilinear_spatial_interpolate(
    lat_coords: np.ndarray,
    lon_coords: np.ndarray,
    field: np.ndarray,
    lat_query,
    lon_query,
) -> tuple[np.ndarray, np.ndarray]:
    """Bilinear spatial interpolation of one 2-D `field` slice (shape `(n_lat, n_lon)`,
    indexed to match `lat_coords`/`lon_coords`) at query points `(lat_query, lon_query)`.

    `lat_coords` may be ascending (ocean grid) or descending (wind grid); `lon_coords` is
    handled the same way for generality, though both project grids happen to be ascending
    in longitude. `field` may contain `NaN` for masked/invalid cells (from
    `iste.environment.load_field_dataset`).

    Returns `(values, valid)`, both 1-D arrays with one entry per query point:

    - `values[i]` is `NaN` if any of the 4 bracketing grid cells is `NaN`, or if the query
      point is outside `[min(lat_coords), max(lat_coords)] x [min(lon_coords), max(lon_coords)]`.
    - `valid[i]` is the explicit boolean companion to the above.

    This function interpolates a single time slice only; see `linear_temporal_interpolate`
    for interpolation across time, and `interpolate_field` for the combined query.
    """
    lat_q = np.atleast_1d(np.asarray(lat_query, dtype=np.float64))
    lon_q = np.atleast_1d(np.asarray(lon_query, dtype=np.float64))
    lat_q, lon_q = np.broadcast_arrays(lat_q, lon_q)
    field = np.asarray(field, dtype=np.float64)
    if field.ndim != 2:
        raise ValueError(f"field must be 2-D (n_lat, n_lon); got shape {field.shape}")
    if field.shape != (len(lat_coords), len(lon_coords)):
        raise ValueError(
            f"field shape {field.shape} does not match "
            f"(len(lat_coords), len(lon_coords)) = ({len(lat_coords)}, {len(lon_coords)})"
        )

    ilat_lo, ilat_hi, flat, lat_ok = _bracket(lat_coords, lat_q)
    ilon_lo, ilon_hi, flon, lon_ok = _bracket(lon_coords, lon_q)

    v00 = field[ilat_lo, ilon_lo]
    v01 = field[ilat_lo, ilon_hi]
    v10 = field[ilat_hi, ilon_lo]
    v11 = field[ilat_hi, ilon_hi]

    values = (
        (1 - flat) * (1 - flon) * v00
        + (1 - flat) * flon * v01
        + flat * (1 - flon) * v10
        + flat * flon * v11
    )

    corners_finite = ~(np.isnan(v00) | np.isnan(v01) | np.isnan(v10) | np.isnan(v11))
    valid = corners_finite & lat_ok & lon_ok
    values = np.where(valid, values, np.nan)
    return values, valid


def linear_temporal_interpolate(
    time_coords: np.ndarray,
    values: np.ndarray,
    query_time,
) -> tuple[np.ndarray, np.ndarray]:
    """Linear interpolation in time of `values` (leading axis = time, matching
    `time_coords`) at `query_time`.

    `time_coords` must be strictly ascending `datetime64` values (both project datasets'
    hourly timestamps are, verified in T00/T01). `values` may be 1-D (one scalar per
    timestamp, e.g. a value already resolved at a single spatial point) or have trailing
    axes after the leading time axis (e.g. one value per query point, per timestamp).

    Returns `(interpolated, valid)`:

    - `interpolated` has the query axis first, followed by `values`'s trailing shape.
    - `valid[i]` is `False` (and `interpolated[i]` is `NaN`) whenever `query_time[i]` is
      outside `[time_coords[0], time_coords[-1]]`, or whenever either of the two
      bracketing timestamps' `values` is itself `NaN`.

    This is the sub-hourly interpolation Stage III will need between the source data's
    hourly checkpoints; nothing here assumes a specific step size (e.g. 15 minutes) --
    `query_time` may be any timestamp(s) in range.
    """
    time_coords = np.asarray(time_coords, dtype="datetime64[ns]")
    if time_coords.ndim != 1 or time_coords.shape[0] < 2:
        raise ValueError("time_coords must be 1-D with at least two timestamps")
    if np.any(np.diff(time_coords) <= np.timedelta64(0, "ns")):
        raise ValueError("time_coords must be strictly ascending")

    values = np.asarray(values, dtype=np.float64)
    if values.shape[0] != time_coords.shape[0]:
        raise ValueError(
            f"values.shape[0] ({values.shape[0]}) must match len(time_coords) "
            f"({time_coords.shape[0]})"
        )

    scalar_query = np.ndim(query_time) == 0
    query_arr = np.atleast_1d(np.asarray(query_time, dtype="datetime64[ns]"))

    coords_f = time_coords.astype(np.int64).astype(np.float64)
    query_f = query_arr.astype(np.int64).astype(np.float64)
    idx_lo, idx_hi, frac, in_range = _bracket(coords_f, query_f)

    v_lo = values[idx_lo, ...]
    v_hi = values[idx_hi, ...]
    trailing_ones = (1,) * (v_lo.ndim - 1)
    frac_b = frac.reshape((-1,) + trailing_ones)
    in_range_b = in_range.reshape((-1,) + trailing_ones)

    interpolated = (1 - frac_b) * v_lo + frac_b * v_hi
    corners_finite = ~(np.isnan(v_lo) | np.isnan(v_hi))
    valid = corners_finite & in_range_b
    interpolated = np.where(valid, interpolated, np.nan)

    if scalar_query:
        return interpolated[0], valid[0]
    return interpolated, valid


def time_bracket(time_coords: np.ndarray, query_time) -> tuple[int, int, float, bool]:
    """Public wrapper around the same time-bracketing logic `interpolate_field`/
    `linear_temporal_interpolate` use internally (`_bracket`), for callers that need the
    bracketing hourly indices and interpolation fraction directly -- e.g. T05's backward
    step, to keep a second time-dependent grid quantity (`div(U)`) time-interpolated
    consistently with how `U` itself is interpolated, without duplicating the
    bracketing logic.

    Returns `(idx_lo, idx_hi, frac, in_range)` for one scalar `query_time`, such that
    `value ~= (1 - frac) * values[idx_lo] + frac * values[idx_hi]` (see `_bracket`).
    """
    if np.ndim(query_time) != 0:
        raise ValueError("time_bracket expects a single scalar query time")
    coords_f = np.asarray(time_coords, dtype="datetime64[ns]").astype(np.int64).astype(np.float64)
    query_f = np.atleast_1d(np.asarray(query_time, dtype="datetime64[ns]")).astype(np.int64).astype(np.float64)
    idx_lo, idx_hi, frac, in_range = _bracket(coords_f, query_f)
    return int(idx_lo[0]), int(idx_hi[0]), float(frac[0]), bool(in_range[0])


def interpolate_field(
    source: "GriddedFieldSource",
    variable: str,
    lat,
    lon,
    time,
) -> tuple[np.ndarray, np.ndarray]:
    """Given a spatial location (or array of locations) and a single time `t`, return the
    interpolated value(s) of `variable` from `source` at `(lat, lon, t)`.

    This is the composed T02 query: bilinear spatial interpolation at the two hourly
    timestamps bracketing `t` (`bilinear_spatial_interpolate`), then linear interpolation
    between them in time (the same bracket logic as `linear_temporal_interpolate`).

    `lat`/`lon` may be scalars or arrays (one or many spatial query points); `time` must
    be a single timestamp shared by all of them (querying many particles' positions at one
    shared simulation instant is the pattern later Stage III steps will use). Returns
    `(values, valid)` matching `lat`/`lon`'s shape (or a plain `(float, bool)` pair if
    `lat`/`lon` were scalars).

    Validity follows `bilinear_spatial_interpolate`'s policy at each of the two bracketing
    timestamps, further ANDed with `t` itself being within the source's time coverage.
    """
    if np.ndim(time) != 0:
        raise ValueError("interpolate_field expects a single scalar query time")
    if variable not in source.data:
        raise KeyError(f"variable '{variable}' not present in this GriddedFieldSource")

    field = source.data[variable]  # shape (n_time, n_lat, n_lon)

    coords_f = source.time.astype("datetime64[ns]").astype(np.int64).astype(np.float64)
    query_f = np.atleast_1d(np.asarray(time, dtype="datetime64[ns]")).astype(np.int64).astype(np.float64)
    idx_lo, idx_hi, frac, t_ok = _bracket(coords_f, query_f)
    i_lo, i_hi, f, t_in_range = int(idx_lo[0]), int(idx_hi[0]), float(frac[0]), bool(t_ok[0])

    val_lo, valid_lo = bilinear_spatial_interpolate(source.lat, source.lon, field[i_lo], lat, lon)
    val_hi, valid_hi = bilinear_spatial_interpolate(source.lat, source.lon, field[i_hi], lat, lon)

    values = (1 - f) * val_lo + f * val_hi
    valid = valid_lo & valid_hi & t_in_range
    values = np.where(valid, values, np.nan)

    scalar_query = np.ndim(lat) == 0 and np.ndim(lon) == 0
    if scalar_query:
        return float(values[0]), bool(valid[0])
    return values, valid
