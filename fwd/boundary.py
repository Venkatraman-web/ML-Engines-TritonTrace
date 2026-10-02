"""F02 -- land/coastline classification for the ow-0009 forward-propagation pilot.

Decides whether a geographic point is "land" (oil reaching it must strand there
permanently, per CLAUDE.md section 7/9) or "water" (normal propagation continues).
This is deliberately a separate concern from `fwd.forcing`'s NaN-fallback, which
answers "is there a trustworthy current estimate here", not "is this land" --
`fwd.forcing`'s own docstring says land classification is this module's job, so the two
are not conflated (a coastal water particle must not be auto-stranded just because an
interpolation stencil touching it contains a NaN corner; that case is exactly what
NEIGHBOR_FALLBACK is for).

Classification source: the F00 audit found the CMEMS `uo`/`vo` NaN mask is exactly
time-invariant across all 96 timesteps of `data/28th_april/ocean.nc` -- i.e. it is
already a genuine static land/sea mask at CMEMS grid resolution (~0.042 deg, ~4.6 km),
not merely a per-timestep interpolation artifact. `ForcingAdapter.ocean_valid_mask`
(built in F01) is reused directly as that ground truth rather than introducing a new
coastline/shapefile dependency (none exists in this project's environment).

A point is classified as land if its *nearest* CMEMS grid cell is permanently masked,
or if the point falls outside the CMEMS grid entirely (no trustworthy ocean data exists
there for propagation to continue, so it is treated the same as land for stranding
purposes).
"""

from __future__ import annotations

from fwd.forcing import ForcingAdapter


def is_land(adapter: ForcingAdapter, lon: float, lat: float) -> bool:
    """True iff `(lon, lat)` should be treated as land/unreachable for particle
    propagation purposes."""
    if not ForcingAdapter.in_spatial_range(adapter.ocean, lon, lat):
        return True
    i, j = adapter.nearest_cell(lon, lat)
    return not bool(adapter.ocean_valid_mask[i, j])
