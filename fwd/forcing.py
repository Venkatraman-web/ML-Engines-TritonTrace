"""F01 -- environmental forcing adapter for the ow-0009 forward-propagation pilot.

Combines CMEMS ocean current (`uo`, `vo`) and ERA5 wind (`u10`, `v10`) into a single
`(lon, lat, time) -> forcing` lookup, reusing Stage III's (`iste`) generic NetCDF
loading and bilinear/temporal interpolation machinery unchanged (CLAUDE.md: "Reuse
compatible Stage III components... Do not modify Stage III behavior as a side effect
of forward-propagation work").

Scope, per `tasks.md` F01: this module answers "what is the current/wind here" at an
arbitrary continuous `(lon, lat, time)`. It does **not** decide whether a particle is
stranded on land, and does **not** combine `uo`/`u10` into an oil velocity -- those are
F02 (land classification) and F03 (particle motion). The only land-related thing here
is a static CMEMS land/sea mask (`ForcingAdapter.ocean_valid_mask`), because the CMEMS
NaN-fallback policy (CLAUDE.md section 6) must never use a land cell as a fallback
neighbor; F02 may reuse this same mask for its own land classification instead of
rebuilding it, per requirement 7 ("keep forcing interpolation separate from
coastline/land handling").

CMEMS NaN policy (CLAUDE.md section 6 / `tasks.md` F02, implemented here for the
forcing layer only):

    direct bilinear+temporal interpolation valid?
      yes -> DIRECT
      no, but query point in-range and >= min_valid_neighbors ocean-valid cells found
           within a 3x3 (then 5x5) window and max_radius_m -> NEIGHBOR_FALLBACK
           (inverse-distance weighted, weight = 1/(d^2 + epsilon), land cells excluded)
      no, and no trustworthy fallback -> INVALID_ENVIRONMENT (uo/vo returned as NaN,
           never silently zero)

`uo.fillna(0)` / `vo.fillna(0)` are never used anywhere in this module.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fwd.stage3_bridge import ensure_iste_importable

ensure_iste_importable()

from iste.environment import GriddedFieldSource, load_field_dataset  # noqa: E402
from iste.geometry import metres_per_degree_latitude, metres_per_degree_longitude  # noqa: E402
from iste.interpolation import interpolate_field, time_bracket  # noqa: E402

OCEAN_PATH = "data/28th_april/ocean.nc"
WIND_PATH = "data/28th_april/wind.nc"

DIRECT = "DIRECT"
NEIGHBOR_FALLBACK = "NEIGHBOR_FALLBACK"
INVALID_ENVIRONMENT = "INVALID_ENVIRONMENT"

_WIND_TIME_DIM_CANDIDATES = ("time", "valid_time")


def _detect_wind_time_dim(wind_path: str) -> str:
    """ERA5 NetCDF exports vary in what they call the time coordinate: this
    project's ow-0009 wind file uses `time`, but the ow-0008 wind file
    (`data/14th_june/wind.nc`, a separately-downloaded/converted GRIB) uses
    `valid_time` instead -- discovered when adding the ow-0008 case, not assumed.
    Detect explicitly by inspecting the file's own coordinates rather than hard-coding
    one name, and fail loudly if neither known name is present rather than guessing."""
    import xarray as xr

    with xr.open_dataset(wind_path) as ds:
        for candidate in _WIND_TIME_DIM_CANDIDATES:
            if candidate in ds.coords:
                return candidate
    raise KeyError(
        f"Could not find a time coordinate in {wind_path} -- tried {_WIND_TIME_DIM_CANDIDATES}. "
        "Inspect the file's coordinates and add its time-dimension name to _WIND_TIME_DIM_CANDIDATES."
    )


@dataclass(frozen=True)
class FallbackConfig:
    """CLAUDE.md section 6 defaults, kept explicit/configurable rather than hidden
    constants (matching `iste.geometry`'s own convention)."""

    neighborhoods: tuple[int, ...] = (3, 5)  # window sizes (cells) to try, in order
    min_valid_neighbors: int = 4
    epsilon: float = 1e-6
    max_radius_m: float = 20_000.0  # configurable maximum search radius


@dataclass(frozen=True)
class ForcingResult:
    lon: float
    lat: float
    time: np.datetime64
    uo: float
    vo: float
    u10: float
    v10: float
    ocean_mode: str  # DIRECT | NEIGHBOR_FALLBACK | INVALID_ENVIRONMENT
    wind_mode: str  # DIRECT | INVALID_ENVIRONMENT
    ocean_fallback_window: int | None = None
    ocean_fallback_neighbors: int | None = None


class ForcingAdapter:
    """Loads CMEMS + ERA5 once, then answers many `(lon, lat, time)` forcing queries."""

    def __init__(
        self,
        ocean_path: str = OCEAN_PATH,
        wind_path: str = WIND_PATH,
        fallback: FallbackConfig = FallbackConfig(),
    ):
        self.ocean: GriddedFieldSource = load_field_dataset(
            ocean_path, ["uo", "vo"], time_dim="time", lat_dim="lat", lon_dim="lon",
        )
        wind_time_dim = _detect_wind_time_dim(wind_path)
        self.wind: GriddedFieldSource = load_field_dataset(
            wind_path, ["u10", "v10"], time_dim=wind_time_dim, lat_dim="latitude", lon_dim="longitude",
        )
        self.fallback = fallback

        # Static land/sea mask (requirement 6): a cell is ocean-valid only if uo AND vo
        # are finite at every timestep. The F00 audit verified the CMEMS NaN pattern is
        # exactly time-invariant across all 96 timesteps; `mask_is_time_invariant` below
        # re-checks that here rather than silently trusting the earlier finding.
        uo_all = self.ocean.data["uo"]
        vo_all = self.ocean.data["vo"]
        finite_all = np.isfinite(uo_all) & np.isfinite(vo_all)
        self.ocean_valid_mask = np.all(finite_all, axis=0)
        self.mask_is_time_invariant = bool(np.array_equal(self.ocean_valid_mask, finite_all[0]))

    def ocean_grid_shape(self) -> tuple[int, int]:
        return len(self.ocean.lat), len(self.ocean.lon)

    def nearest_cell(self, lon: float, lat: float) -> tuple[int, int]:
        """Nearest CMEMS grid `(lat_index, lon_index)` to `(lon, lat)`. Public because
        `fwd.boundary`'s land classification needs the same nearest-cell lookup this
        module uses internally for the NaN fallback -- one implementation, not two."""
        i = int(np.argmin(np.abs(self.ocean.lat - lat)))
        j = int(np.argmin(np.abs(self.ocean.lon - lon)))
        return i, j

    @staticmethod
    def in_spatial_range(source: GriddedFieldSource, lon: float, lat: float) -> bool:
        lat_ok = float(np.min(source.lat)) <= lat <= float(np.max(source.lat))
        lon_ok = float(np.min(source.lon)) <= lon <= float(np.max(source.lon))
        return lat_ok and lon_ok

    def _neighbor_fallback(
        self, lon: float, lat: float, t: np.datetime64
    ) -> tuple[float, float, str, int | None, int | None]:
        """Distance-weighted uo/vo from nearby ocean-valid cells only (never land),
        per CLAUDE.md section 6: 3x3 -> 5x5, >= min_valid_neighbors, weight
        1/(d^2 + epsilon), stop at max_radius_m."""
        if not self.in_spatial_range(self.ocean, lon, lat):
            return float("nan"), float("nan"), INVALID_ENVIRONMENT, None, None

        i_lo, i_hi, frac, t_in_range = time_bracket(self.ocean.time, t)
        if not t_in_range:
            return float("nan"), float("nan"), INVALID_ENVIRONMENT, None, None

        i0, j0 = self.nearest_cell(lon, lat)
        m_per_deg_lat = metres_per_degree_latitude()
        m_per_deg_lon = metres_per_degree_longitude(lat)
        n_lat, n_lon = self.ocean_valid_mask.shape

        for window in self.fallback.neighborhoods:
            half = window // 2
            i_min, i_max = max(0, i0 - half), min(n_lat - 1, i0 + half)
            j_min, j_max = max(0, j0 - half), min(n_lon - 1, j0 + half)

            ii, jj = np.meshgrid(
                np.arange(i_min, i_max + 1), np.arange(j_min, j_max + 1), indexing="ij"
            )
            valid = self.ocean_valid_mask[ii, jj]
            if int(valid.sum()) < self.fallback.min_valid_neighbors:
                continue

            dy_m = (self.ocean.lat[ii] - lat) * m_per_deg_lat
            dx_m = (self.ocean.lon[jj] - lon) * m_per_deg_lon
            dist_m = np.sqrt(dx_m**2 + dy_m**2)

            use = valid & (dist_m <= self.fallback.max_radius_m)
            n_use = int(use.sum())
            if n_use < self.fallback.min_valid_neighbors:
                continue

            weight = 1.0 / (dist_m[use] ** 2 + self.fallback.epsilon)

            def blend(field: np.ndarray) -> float:
                v_lo = float(np.sum(weight * field[i_lo][ii[use], jj[use]]) / np.sum(weight))
                v_hi = float(np.sum(weight * field[i_hi][ii[use], jj[use]]) / np.sum(weight))
                return (1 - frac) * v_lo + frac * v_hi

            uo_val = blend(self.ocean.data["uo"])
            vo_val = blend(self.ocean.data["vo"])
            return uo_val, vo_val, NEIGHBOR_FALLBACK, window, n_use

        return float("nan"), float("nan"), INVALID_ENVIRONMENT, None, None

    def query(self, lon: float, lat: float, time) -> ForcingResult:
        """Return interpolated `uo, vo, u10, v10` at one continuous `(lon, lat, time)`,
        with the CLAUDE.md section 6 NaN-fallback policy applied to the ocean current
        only (ERA5 has no masked cells in this dataset, verified in F00; wind is
        reported `INVALID_ENVIRONMENT` only if the query point/time is out of the ERA5
        grid's coverage)."""
        t = np.datetime64(time, "ns")
        lon = float(lon)
        lat = float(lat)

        uo, uo_valid = interpolate_field(self.ocean, "uo", lat, lon, t)
        vo, vo_valid = interpolate_field(self.ocean, "vo", lat, lon, t)
        if uo_valid and vo_valid:
            uo_val, vo_val = float(uo), float(vo)
            ocean_mode, fb_window, fb_neighbors = DIRECT, None, None
        else:
            uo_val, vo_val, ocean_mode, fb_window, fb_neighbors = self._neighbor_fallback(lon, lat, t)

        u10, u10_valid = interpolate_field(self.wind, "u10", lat, lon, t)
        v10, v10_valid = interpolate_field(self.wind, "v10", lat, lon, t)
        if u10_valid and v10_valid:
            u10_val, v10_val, wind_mode = float(u10), float(v10), DIRECT
        else:
            u10_val, v10_val, wind_mode = float("nan"), float("nan"), INVALID_ENVIRONMENT

        return ForcingResult(
            lon=lon,
            lat=lat,
            time=t,
            uo=uo_val,
            vo=vo_val,
            u10=u10_val,
            v10=v10_val,
            ocean_mode=ocean_mode,
            wind_mode=wind_mode,
            ocean_fallback_window=fb_window,
            ocean_fallback_neighbors=fb_neighbors,
        )
