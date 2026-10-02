"""F01 manual validation (not a test file -- CLAUDE.md forbids `tests/`/`test_*.py`).

Verifies `fwd.forcing.ForcingAdapter` against:
  1. the actual ow-0009 parcels at the actual observation time (2019-04-28 15:38:41 UTC);
  2. an independent hand-computed exact-time bilinear+linear interpolation, to prove the
     adapter interpolates to the exact requested timestamp rather than snapping to a
     forcing record;
  3. deliberately chosen CMEMS NaN/land-boundary points (a coastal water cell adjacent
     to a masked cell, and a point deep inside a masked region with no valid neighbors),
     to exercise the NEIGHBOR_FALLBACK and INVALID_ENVIRONMENT paths.

Run: python scripts/f01_validate_forcing.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from fwd.forcing import DIRECT, INVALID_ENVIRONMENT, NEIGHBOR_FALLBACK, ForcingAdapter  # noqa: E402

OBS_TIME = np.datetime64("2019-04-28T15:38:41", "ns")


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def load_ow0009_parcels() -> list[dict]:
    with open(REPO_ROOT / "data" / "database_payload.json", encoding="utf-8") as f:
        payload = json.load(f)
    spill = next(s for s in payload["spills"] if s["image_filename"] == "ow-0009.jpg")
    assert spill["timestamp"] == "2019-04-28T15:38:41"
    assert len(spill["parcels"]) == 371
    return spill["parcels"]


def independent_bilinear(lat_coords, lon_coords, field2d, lat_q, lon_q):
    """Reference bilinear interpolation, written independently of iste.interpolation,
    to cross-check the reused Stage III code rather than trusting it circularly."""
    ascending_lat = lat_coords[1] > lat_coords[0]
    work_lat = lat_coords if ascending_lat else lat_coords[::-1]
    i_hi_w = int(np.clip(np.searchsorted(work_lat, lat_q, side="left"), 1, len(work_lat) - 1))
    i_lo_w = i_hi_w - 1
    i_lo, i_hi = (i_lo_w, i_hi_w) if ascending_lat else (len(lat_coords) - 1 - i_lo_w, len(lat_coords) - 1 - i_hi_w)
    flat = (lat_q - lat_coords[i_lo]) / (lat_coords[i_hi] - lat_coords[i_lo])

    j_hi = int(np.clip(np.searchsorted(lon_coords, lon_q, side="left"), 1, len(lon_coords) - 1))
    j_lo = j_hi - 1
    flon = (lon_q - lon_coords[j_lo]) / (lon_coords[j_hi] - lon_coords[j_lo])

    v00, v01 = field2d[i_lo, j_lo], field2d[i_lo, j_hi]
    v10, v11 = field2d[i_hi, j_lo], field2d[i_hi, j_hi]
    return (1 - flat) * (1 - flon) * v00 + (1 - flat) * flon * v01 + flat * (1 - flon) * v10 + flat * flon * v11


def main() -> None:
    adapter = ForcingAdapter()

    section("Grid / mask summary")
    print("ocean grid shape (lat, lon):", adapter.ocean_grid_shape())
    print("ocean lat range:", float(adapter.ocean.lat.min()), float(adapter.ocean.lat.max()))
    print("ocean lon range:", float(adapter.ocean.lon.min()), float(adapter.ocean.lon.max()))
    print("wind lat[0], lat[-1] (must be descending):", adapter.wind.lat[0], adapter.wind.lat[-1])
    print("static ocean-valid mask, fraction water:", float(adapter.ocean_valid_mask.mean()))
    print("mask time-invariant (re-verified here):", adapter.mask_is_time_invariant)
    assert adapter.mask_is_time_invariant, "CMEMS NaN mask is not static -- fallback design assumption violated"
    assert adapter.wind.lat[0] > adapter.wind.lat[-1], "expected descending ERA5 latitude"

    section("1) ow-0009 parcels at the actual observation time (2019-04-28 15:38:41 UTC)")
    parcels = load_ow0009_parcels()
    sample = parcels[:5] + parcels[-5:]
    for p in sample:
        r = adapter.query(p["lon"], p["lat"], OBS_TIME)
        print(
            f"id={p['id']:>3} lon={p['lon']:.6f} lat={p['lat']:.6f} -> "
            f"uo={r.uo:+.5f} vo={r.vo:+.5f} u10={r.u10:+.5f} v10={r.v10:+.5f} "
            f"ocean={r.ocean_mode} wind={r.wind_mode}"
        )
        assert np.isfinite(r.uo) and np.isfinite(r.vo), f"parcel {p['id']} got non-finite ocean forcing"
        assert np.isfinite(r.u10) and np.isfinite(r.v10), f"parcel {p['id']} got non-finite wind forcing"
        assert r.wind_mode == DIRECT

    all_results = [adapter.query(p["lon"], p["lat"], OBS_TIME) for p in parcels]
    modes = [r.ocean_mode for r in all_results]
    print(
        f"all 371 parcels @ t0: DIRECT={modes.count(DIRECT)} "
        f"NEIGHBOR_FALLBACK={modes.count(NEIGHBOR_FALLBACK)} "
        f"INVALID_ENVIRONMENT={modes.count(INVALID_ENVIRONMENT)}"
    )
    assert modes.count(INVALID_ENVIRONMENT) == 0, "some ow-0009 parcels have no valid ocean forcing at t0"
    speeds = np.hypot([r.uo for r in all_results], [r.vo for r in all_results])
    print(f"uo/vo speed range across parcels @ t0: [{speeds.min():.4f}, {speeds.max():.4f}] m/s")
    assert speeds.max() < 3.0, "surface current speed implausibly high -- check units/interpolation"

    section("2) Independent exact-time interpolation cross-check (no snapping)")
    p0 = parcels[0]
    lon0, lat0 = p0["lon"], p0["lat"]
    r0 = adapter.query(lon0, lat0, OBS_TIME)

    ds = xr.open_dataset(REPO_ROOT / "data" / "28th_april" / "ocean.nc", mask_and_scale=False)
    times = ds["time"].values
    i_hi = int(np.searchsorted(times, OBS_TIME))
    i_lo = i_hi - 1
    t_lo, t_hi = times[i_lo], times[i_hi]
    print(f"CMEMS bracket for {OBS_TIME}: {t_lo} -> {t_hi}")
    assert str(t_lo)[:16] == "2019-04-28T15:30" and str(t_hi)[:16] == "2019-04-28T16:30", (
        "expected the 15:30->16:30 CMEMS bracket for a 15:38:41 query"
    )
    frac_t = (OBS_TIME - t_lo) / (t_hi - t_lo)
    uo_lo = independent_bilinear(ds["lat"].values, ds["lon"].values, ds["uo"].isel(time=i_lo).values, lat0, lon0)
    uo_hi = independent_bilinear(ds["lat"].values, ds["lon"].values, ds["uo"].isel(time=i_hi).values, lat0, lon0)
    uo_ref = (1 - frac_t) * uo_lo + frac_t * uo_hi
    print(f"adapter uo={r0.uo:.6f}  independent reference uo={float(uo_ref):.6f}")
    assert np.isclose(r0.uo, uo_ref, atol=1e-9), "adapter ocean interpolation diverges from independent reference"
    ds.close()

    dsw = xr.open_dataset(REPO_ROOT / "data" / "28th_april" / "wind.nc", mask_and_scale=False)
    wtimes = dsw["time"].values
    wi_hi = int(np.searchsorted(wtimes, OBS_TIME))
    wi_lo = wi_hi - 1
    wt_lo, wt_hi = wtimes[wi_lo], wtimes[wi_hi]
    print(f"ERA5 bracket for {OBS_TIME}: {wt_lo} -> {wt_hi}")
    assert str(wt_lo)[:16] == "2019-04-28T15:00" and str(wt_hi)[:16] == "2019-04-28T16:00", (
        "expected the 15:00->16:00 ERA5 bracket for a 15:38:41 query"
    )
    wfrac_t = (OBS_TIME - wt_lo) / (wt_hi - wt_lo)
    u10_lo = independent_bilinear(dsw["latitude"].values, dsw["longitude"].values, dsw["u10"].isel(time=wi_lo).values, lat0, lon0)
    u10_hi = independent_bilinear(dsw["latitude"].values, dsw["longitude"].values, dsw["u10"].isel(time=wi_hi).values, lat0, lon0)
    u10_ref = (1 - wfrac_t) * u10_lo + wfrac_t * u10_hi
    print(f"adapter u10={r0.u10:.6f}  independent reference u10={float(u10_ref):.6f}")
    assert np.isclose(r0.u10, u10_ref, atol=1e-9), "adapter wind interpolation diverges from independent reference"
    dsw.close()

    section("3) CMEMS NaN / land-boundary points")
    mask = adapter.ocean_valid_mask
    n_lat, n_lon = mask.shape

    boundary_points = []
    for i in range(n_lat):
        for j in range(n_lon):
            if not mask[i, j]:
                continue
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ni, nj = i + di, j + dj
                if 0 <= ni < n_lat and 0 <= nj < n_lon and not mask[ni, nj]:
                    q_lat = float((adapter.ocean.lat[i] + adapter.ocean.lat[ni]) / 2)
                    q_lon = float((adapter.ocean.lon[j] + adapter.ocean.lon[nj]) / 2)
                    boundary_points.append((q_lon, q_lat))
                    break
            if len(boundary_points) >= 25:
                break
        if len(boundary_points) >= 25:
            break

    print(f"found {len(boundary_points)} water/land boundary midpoints to probe")
    boundary_modes = []
    for q_lon, q_lat in boundary_points:
        r = adapter.query(q_lon, q_lat, OBS_TIME)
        boundary_modes.append(r.ocean_mode)
    print("boundary point ocean modes:", {m: boundary_modes.count(m) for m in set(boundary_modes)})
    assert NEIGHBOR_FALLBACK in boundary_modes, "expected at least one coastal boundary point to require NEIGHBOR_FALLBACK"
    fb_example_idx = boundary_modes.index(NEIGHBOR_FALLBACK)
    fb_lon, fb_lat = boundary_points[fb_example_idx]
    fb_result = adapter.query(fb_lon, fb_lat, OBS_TIME)
    print(
        f"NEIGHBOR_FALLBACK example: lon={fb_lon:.4f} lat={fb_lat:.4f} -> "
        f"uo={fb_result.uo:+.5f} vo={fb_result.vo:+.5f} "
        f"window={fb_result.ocean_fallback_window} neighbors={fb_result.ocean_fallback_neighbors}"
    )
    assert fb_result.ocean_fallback_neighbors is not None and fb_result.ocean_fallback_neighbors >= 4
    assert np.isfinite(fb_result.uo) and np.isfinite(fb_result.vo)

    half = 2  # 5x5
    deep_land = None
    for i in range(half, n_lat - half):
        for j in range(half, n_lon - half):
            if mask[i, j]:
                continue
            window = mask[i - half : i + half + 1, j - half : j + half + 1]
            if not window.any():
                deep_land = (i, j)
                break
        if deep_land is not None:
            break

    assert deep_land is not None, "expected at least one deep-land grid cell with zero valid ocean neighbors in a 5x5 window"
    di, dj = deep_land
    dl_lon, dl_lat = float(adapter.ocean.lon[dj]), float(adapter.ocean.lat[di])
    dl_result = adapter.query(dl_lon, dl_lat, OBS_TIME)
    print(f"deep-land example: lon={dl_lon:.4f} lat={dl_lat:.4f} -> ocean_mode={dl_result.ocean_mode} uo={dl_result.uo} vo={dl_result.vo}")
    assert dl_result.ocean_mode == INVALID_ENVIRONMENT
    assert np.isnan(dl_result.uo) and np.isnan(dl_result.vo), "INVALID_ENVIRONMENT must report NaN, never a silently substituted 0"

    section("4) Out-of-domain point (must not extrapolate)")
    far = adapter.query(0.0, 0.0, OBS_TIME)
    print("far-away point (0,0):", far)
    assert far.ocean_mode == INVALID_ENVIRONMENT and far.wind_mode == INVALID_ENVIRONMENT
    assert np.isnan(far.uo) and np.isnan(far.u10)

    section("ALL CHECKS PASSED")
    print("F01 forcing adapter: COMPLETE")


if __name__ == "__main__":
    main()
