"""F02 manual validation (not a test file -- CLAUDE.md forbids `tests/`/`test_*.py`).

Verifies `fwd.step.step_particle`/`step_all` against:
  1. a normal open-water ow-0009 parcel, one real hour (dt=3600s) from the actual
     observation time;
  2. the ow-0009 parcel closest to the coast, same real one-hour step;
  3. an independent hand-computed check of the displacement formula
     (u_oil, v_oil -> dlon, dlat) and its units, cross-checked against
     `iste.geometry.metres_to_degrees`;
  4. the land-crossing/stranding path: probing several real coastal boundary cells with
     a deliberately large (clearly-labelled, non-default) timestep so the small real
     CMEMS current is enough to cross a ~4.6 km grid cell onto land within one call,
     confirming `LAND_STRANDED` triggers and the particle is kept at its last valid
     position (not moved onto land);
  5. that re-stepping an already-stranded particle with the normal one-hour dt does not
     move it (`ALREADY_STRANDED`, requirement 7).

Run: python scripts/f02_validate_step.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from fwd.boundary import is_land  # noqa: E402
from fwd.forcing import DIRECT, NEIGHBOR_FALLBACK, ForcingAdapter  # noqa: E402
from fwd.stage3_bridge import ensure_iste_importable  # noqa: E402
from fwd.step import ALREADY_STRANDED, LAND_STRANDED, DT_SECONDS, ParticleState, step_particle  # noqa: E402

ensure_iste_importable()
from iste.geometry import metres_per_degree_latitude, metres_per_degree_longitude, metres_to_degrees  # noqa: E402

OBS_TIME = np.datetime64("2019-04-28T15:38:41", "ns")


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def load_ow0009_parcels() -> list[dict]:
    with open(REPO_ROOT / "data" / "database_payload.json", encoding="utf-8") as f:
        payload = json.load(f)
    spill = next(s for s in payload["spills"] if s["image_filename"] == "ow-0009.jpg")
    assert len(spill["parcels"]) == 371
    return spill["parcels"]


def independent_metres_to_degrees(dx_m: float, dy_m: float, lat_deg: float) -> tuple[float, float]:
    """Reference conversion, written independently of iste.geometry, to cross-check the
    reused Stage III function rather than trusting it circularly."""
    R = 6_371_000.0
    m_per_deg_lat = (np.pi / 180.0) * R
    m_per_deg_lon = (np.pi / 180.0) * R * np.cos(np.deg2rad(lat_deg))
    return dx_m / m_per_deg_lon, dy_m / m_per_deg_lat


def main() -> None:
    adapter = ForcingAdapter()
    parcels = load_ow0009_parcels()

    section("1) Normal open-water parcel, one real hour")
    p_normal = parcels[0]
    state = ParticleState(p_normal["id"], p_normal["lon"], p_normal["lat"])
    new_state, rec = step_particle(adapter, state, OBS_TIME)
    print(f"particle {rec.particle_id}: t0={rec.time_start} -> t1={rec.time_end}")
    print(f"  old=({rec.old_lon:.6f},{rec.old_lat:.6f})  new=({rec.new_lon:.6f},{rec.new_lat:.6f})")
    print(f"  uo={rec.uo:+.5f} vo={rec.vo:+.5f} u10={rec.u10:+.5f} v10={rec.v10:+.5f}")
    print(f"  oil_u={rec.oil_u:+.6f} oil_v={rec.oil_v:+.6f}  forcing_mode={rec.forcing_mode}")
    print(f"  valid={rec.valid} stranded={rec.stranded}")
    assert rec.valid and not rec.stranded
    assert rec.forcing_mode in (DIRECT, NEIGHBOR_FALLBACK)
    assert np.isclose(rec.oil_u, 1.0 * rec.uo + 0.03 * rec.u10)
    assert np.isclose(rec.oil_v, 1.0 * rec.vo + 0.03 * rec.v10)
    assert (rec.new_lon > rec.old_lon) == (rec.oil_u > 0), "displacement direction (east/west) disagrees with oil_u sign"
    assert (rec.new_lat > rec.old_lat) == (rec.oil_v > 0), "displacement direction (north/south) disagrees with oil_v sign"
    displacement_m = np.hypot(rec.oil_u, rec.oil_v) * DT_SECONDS
    print(f"  displacement magnitude over 1h: {displacement_m:.2f} m")
    assert 0 < displacement_m < 5000, "one-hour displacement implausible for a ~0.1-0.3 m/s surface current"

    section("2) Parcel closest to the coast (of the 371), one real hour")
    # Distance from each parcel to the nearest CMEMS land cell (great-circle, km) --
    # this is an honest "closest of the 371", not a claim that any ow-0009 parcel is
    # actually near shore (see printed distance below).
    land_i, land_j = np.where(~adapter.ocean_valid_mask)
    land_lat_deg = adapter.ocean.lat[land_i]
    land_lon_deg = adapter.ocean.lon[land_j]

    def haversine_km(lon1, lat1, lon2, lat2):
        R = 6371.0
        lon1r, lat1r, lon2r, lat2r = map(np.radians, [lon1, lat1, lon2, lat2])
        a = np.sin((lat2r - lat1r) / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin((lon2r - lon1r) / 2) ** 2
        return 2 * R * np.arcsin(np.sqrt(a))

    dist_to_land_km = [
        float(haversine_km(p["lon"], p["lat"], land_lon_deg, land_lat_deg).min()) for p in parcels
    ]
    coast_idx = int(np.argmin(dist_to_land_km))
    p_coast = parcels[coast_idx]
    print(
        f"chosen parcel id={p_coast['id']} lon={p_coast['lon']:.6f} lat={p_coast['lat']:.6f} "
        f"-- {dist_to_land_km[coast_idx]:.1f} km from the nearest CMEMS land cell "
        f"(all 371 parcels are {min(dist_to_land_km):.1f}-{max(dist_to_land_km):.1f} km offshore -- "
        "none of ow-0009's parcels start truly near the coast; the actual land-crossing/stranding "
        "path is validated separately in section 4 using real coastal boundary cells elsewhere "
        "in the AOI)"
    )
    state_c = ParticleState(p_coast["id"], p_coast["lon"], p_coast["lat"])
    new_state_c, rec_c = step_particle(adapter, state_c, OBS_TIME)
    print(f"  old=({rec_c.old_lon:.6f},{rec_c.old_lat:.6f})  new=({rec_c.new_lon:.6f},{rec_c.new_lat:.6f})")
    print(f"  uo={rec_c.uo:+.5f} vo={rec_c.vo:+.5f} u10={rec_c.u10:+.5f} v10={rec_c.v10:+.5f}")
    print(f"  forcing_mode={rec_c.forcing_mode}  valid={rec_c.valid} stranded={rec_c.stranded}")
    assert np.isfinite(rec_c.new_lon) and np.isfinite(rec_c.new_lat)
    assert rec_c.forcing_mode != "INVALID_ENVIRONMENT", "coastal ow-0009 parcel unexpectedly has no trustworthy forcing"

    section("3) Independent displacement-formula / units cross-check")
    for dx_m, dy_m, lat_deg in [(1000.0, 0.0, 34.6), (0.0, 1000.0, 34.6), (rec.oil_u * DT_SECONDS, rec.oil_v * DT_SECONDS, rec.old_lat)]:
        dlon_ref, dlat_ref = independent_metres_to_degrees(dx_m, dy_m, lat_deg)
        dlon_iste, dlat_iste = metres_to_degrees(dx_m, dy_m, lat_deg)
        print(f"  dx={dx_m:+.3f}m dy={dy_m:+.3f}m @lat={lat_deg} -> "
              f"independent=({float(dlon_ref):.8f},{float(dlat_ref):.8f})deg "
              f"iste=({float(dlon_iste):.8f},{float(dlat_iste):.8f})deg")
        assert np.isclose(dlon_ref, dlon_iste, atol=1e-12)
        assert np.isclose(dlat_ref, dlat_iste, atol=1e-12)
    # sanity: 1000 m northward at this latitude should be ~1000/111195 deg ~ 0.00899 deg
    _, dlat_1km = metres_to_degrees(0.0, 1000.0, 34.6)
    assert np.isclose(float(dlat_1km), 1000.0 / metres_per_degree_latitude(), atol=1e-9)
    print("  displacement formula and units confirmed consistent with an independent reference")

    section("4) Land-crossing / stranding path (deliberately large synthetic dt to force a crossing)")
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
                    boundary_points.append((float(adapter.ocean.lon[j]), float(adapter.ocean.lat[i])))
                    break

    STRESS_DT = 40 * DT_SECONDS  # far beyond the real 1h step, purely to force a crossing for this check
    stranded_example = None
    for lon, lat in boundary_points:
        s = ParticleState(-1, lon, lat)
        ns, r = step_particle(adapter, s, OBS_TIME, dt_seconds=STRESS_DT)
        if r.stranded and r.forcing_mode == LAND_STRANDED:
            stranded_example = (s, ns, r)
            break

    assert stranded_example is not None, "expected at least one coastal boundary point to strand under a large synthetic dt"
    s, ns, r = stranded_example
    print(f"  probe point lon={s.lon:.4f} lat={s.lat:.4f} (dt={STRESS_DT:.0f}s, {STRESS_DT/3600:.0f}h synthetic)")
    print(f"  attempted -> ({r.attempted_lon:.6f},{r.attempted_lat:.6f})  is_land(attempted)={is_land(adapter, r.attempted_lon, r.attempted_lat)}")
    print(f"  kept position -> ({r.new_lon:.6f},{r.new_lat:.6f})  == old position -> {(r.new_lon, r.new_lat) == (r.old_lon, r.old_lat)}")
    assert r.new_lon == r.old_lon and r.new_lat == r.old_lat, "a stranded particle must be kept at its last valid position, not moved onto land"
    assert ns.stranded and not ns.valid
    assert is_land(adapter, r.attempted_lon, r.attempted_lat), "LAND_STRANDED fired but attempted position is not actually classified as land"

    section("5) No movement after stranding (requirement 7)")
    ns2, r2 = step_particle(adapter, ns, OBS_TIME + np.timedelta64(1, "h"))
    print(f"  re-step: forcing_mode={r2.forcing_mode}  new=({r2.new_lon:.6f},{r2.new_lat:.6f})  unchanged={(r2.new_lon, r2.new_lat) == (ns.lon, ns.lat)}")
    assert r2.forcing_mode == ALREADY_STRANDED
    assert r2.new_lon == ns.lon and r2.new_lat == ns.lat
    assert ns2.lon == ns.lon and ns2.lat == ns.lat and ns2.stranded

    section("ALL CHECKS PASSED")
    print("F02 one-hour forward particle step: COMPLETE")


if __name__ == "__main__":
    main()
