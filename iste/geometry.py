"""T03 — Earth-surface distance conversions.

Physical-space spatial derivatives (`iste.derivatives`) -- and later, particle
displacement (T05+) -- require converting degree spacing on a lat/lon grid into metres.
This module isolates and documents that one conversion so it is never silently baked
into a scientific equation elsewhere (`CLAUDE.md`: "Do not hard-code a particular NetCDF
layout into scientific equations"; `memory.md`: "never add metre displacement directly
to lat/lon degrees").

Approximation used: a **spherical** Earth of a single, explicit mean radius (WGS84 mean
radius, ~6,371,000 m). This is a standard first-order geodesic approximation, not a full
ellipsoidal (WGS84) geodesic calculation. No geodesic library (e.g. `pyproj`) is
installed in this project's environment (verified: only `numpy`, `scipy`, `xarray`,
`pandas`, `netCDF4`, `h5netcdf`, `dask`, `matplotlib` are available). The radius is an
explicit function parameter with a documented default, not a hidden constant, per
`CLAUDE.md` ("prefer explicit parameters/configuration over hidden constants").
"""

from __future__ import annotations

import numpy as np

MEAN_EARTH_RADIUS_M = 6_371_000.0
"""WGS84 mean radius (metres), used for the spherical-Earth approximation in this
module. Not a hidden constant -- every function below accepts `radius_m` explicitly and
defaults to this value."""


def metres_per_degree_latitude(radius_m: float = MEAN_EARTH_RADIUS_M) -> float:
    """Metres per degree of latitude on a sphere of radius `radius_m`.

    Independent of latitude itself under the spherical approximation (meridians are
    great circles of constant radius).
    """
    return (np.pi / 180.0) * radius_m


def metres_per_degree_longitude(latitude_deg, radius_m: float = MEAN_EARTH_RADIUS_M):
    """Metres per degree of longitude at `latitude_deg`, on a sphere of radius
    `radius_m`.

    Longitude circles (parallels) shrink toward the poles by a factor of
    `cos(latitude)` -- this is the latitude-dependence that must not be ignored when
    converting east-west degree spacing to physical distance.
    """
    lat_rad = np.deg2rad(np.asarray(latitude_deg, dtype=np.float64))
    return (np.pi / 180.0) * radius_m * np.cos(lat_rad)


def metres_to_degrees(dx_m, dy_m, latitude_deg, radius_m: float = MEAN_EARTH_RADIUS_M):
    """Convert an eastward/northward metre displacement `(dx_m, dy_m)`, made starting at
    `latitude_deg`, into a `(dlon_deg, dlat_deg)` degree displacement -- the inverse of
    `metres_per_degree_longitude`/`metres_per_degree_latitude`, using the same
    spherical-Earth approximation (T03).

    This evaluates the conversion factor at the displacement's *starting* latitude only
    (a first-order approximation, appropriate for the small, single-step displacements
    Stage III computes; it does not re-linearize partway through a large displacement).
    Metres are never added directly to degree coordinates anywhere that uses this
    function -- the conversion always happens explicitly, here.
    """
    dlat_per_m = 1.0 / metres_per_degree_latitude(radius_m)
    dlon_per_m = 1.0 / metres_per_degree_longitude(latitude_deg, radius_m)
    dlon_deg = np.asarray(dx_m, dtype=np.float64) * dlon_per_m
    dlat_deg = np.asarray(dy_m, dtype=np.float64) * dlat_per_m
    return dlon_deg, dlat_deg
