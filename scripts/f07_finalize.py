"""F07 -- final verification + run summary for the ow-0009 forward-propagation pilot
(not a test file -- CLAUDE.md forbids `tests/`/`test_*.py`).

Cross-checks every F01-F06 output under `output/ow-0009/` for mutual consistency (not
just each stage's own already-passed validation in isolation), confirms the specific
facts this task asks for, and writes one concise machine-readable JSON summary
(`output/ow-0009/run_summary.json`) for future application integration. Reads only --
does not modify any F01-F06 code or output.

Run: python scripts/f07_finalize.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from fwd.hotspots import CRITICAL, WATCH, HotspotThresholds, load_hotspots  # noqa: E402

OUT = REPO_ROOT / "output" / "ow-0009"
START_TIME = np.datetime64("2019-04-28T15:38:41", "ns")
END_TIME = np.datetime64("2019-05-01T15:38:41", "ns")


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main() -> None:
    section("0) Inventory of output/ow-0009/")
    required_files = [
        "trajectory.parquet", "hourly_diagnostics.parquet", "hotspot_alerts.parquet",
        "openoil_reference.parquet", "openoil_comparison.parquet",
        "ow0009_forward_animation.mp4",
        "frames/initial.png", "frames/watch.png", "frames/critical_cape_greco.png", "frames/final.png",
    ]
    missing = [f for f in required_files if not (OUT / f).exists()]
    assert not missing, f"missing expected output files: {missing}"
    for f in required_files:
        p = OUT / f
        print(f"  {'OK':4s} {f:45s} {p.stat().st_size:>10,} bytes")

    section("1) Loading all tabular outputs")
    traj = pd.read_parquet(OUT / "trajectory.parquet")
    diag = pd.read_parquet(OUT / "hourly_diagnostics.parquet")
    alerts = pd.read_parquet(OUT / "hotspot_alerts.parquet")
    comparison = pd.read_parquet(OUT / "openoil_comparison.parquet")
    hotspots = load_hotspots()
    print(f"trajectory: {len(traj)} rows | hourly_diagnostics: {len(diag)} rows | "
          f"hotspot_alerts: {len(alerts)} rows | openoil_comparison: {len(comparison)} rows")

    section("2) Core experiment facts (requirement 3)")
    times = sorted(traj["time"].unique())
    assert len(times) == 73, f"expected 73 trajectory states, got {len(times)}"
    assert times[0] == START_TIME, times[0]
    assert times[-1] == END_TIME, times[-1]
    duration_h = int((times[-1] - times[0]) / np.timedelta64(1, "h"))
    assert duration_h == 72, duration_h
    ids = sorted(traj["particle_id"].unique())
    assert len(ids) == 371 and ids == list(range(371)), "expected exactly parcel IDs 0..370"
    print(f"  72 propagation hours: OK ({duration_h}h between {times[0]} and {times[-1]})")
    print(f"  73 trajectory states: OK")
    print(f"  371 parcel IDs (0..370): OK")
    print(f"  final time = {times[-1]} : OK (matches 2019-05-01 15:38:41 UTC)")

    watch_km = {round(h.watch_radius_km) for h in hotspots}
    critical_km = {round(h.critical_radius_km) for h in hotspots}
    assert watch_km == {100}, watch_km
    assert critical_km == {40}, critical_km
    print(f"  100 km WATCH / 40 km CRITICAL thresholds (from the actual polygons, all 11 hotspots): OK")

    section("3) Cross-consistency between trajectory / hourly_diagnostics / hotspot_alerts (requirement 2)")
    # hourly_diagnostics must agree with independently-recomputed active/stranded
    # counts straight from the trajectory table (not just trust F03's own numbers)
    mismatches = 0
    for _, row in diag.iterrows():
        at_t = traj[traj["time"] == row["time"]]
        active = int((at_t["valid"] & ~at_t["stranded"]).sum())
        stranded = int(at_t["stranded"].sum())
        if active != row["active_count"] or stranded != row["stranded_count"]:
            mismatches += 1
            print(f"  MISMATCH at {row['time']}: diagnostics say active={row['active_count']} "
                  f"stranded={row['stranded_count']}, trajectory says active={active} stranded={stranded}")
    assert mismatches == 0, f"{mismatches} hourly_diagnostics rows disagree with the trajectory table"
    print("  hourly_diagnostics active/stranded counts recomputed independently from trajectory.parquet: MATCH")

    assert len(alerts) == 11 * 73, f"expected {11*73} hotspot_alert rows, got {len(alerts)}"
    assert set(alerts["time"].unique()) == set(times), "hotspot_alerts timestamps differ from trajectory timestamps"
    print("  hotspot_alerts covers exactly the same 73 timestamps as trajectory: OK")

    from subprocess import run as _run
    try:
        probe = _run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=nb_frames,r_frame_rate", "-show_entries", "format=duration",
             "-of", "json", str(OUT / "ow0009_forward_animation.mp4")],
            capture_output=True, text=True, check=True,
        )
        probe_json = json.loads(probe.stdout)
        nb_frames = int(probe_json["streams"][0]["nb_frames"])
        fps = probe_json["streams"][0]["r_frame_rate"]
        video_duration_s = float(probe_json["format"]["duration"])
        # 289 = (73-1)*4+1 propagation substeps; the MP4 also opens with a
        # presentational regional-overview hold + eased zoom-in (see
        # scripts/f05_build_animation.py's INTRO_HOLD_FRAMES/INTRO_ZOOM_FRAMES), so
        # the shipped file has more than 289 frames -- checked as a lower bound, not
        # pinned to one exact intro length that is free to be retuned presentationally.
        print(f"  MP4 probe: {nb_frames} frames @ {fps} fps, {video_duration_s:.1f}s "
              f"(>= 289 propagation substeps expected, plus an opening overview/zoom sequence)")
        assert nb_frames >= 289, f"expected at least 289 rendered frames, ffprobe found {nb_frames}"
    except FileNotFoundError:
        print("  ffprobe not available -- skipping frame-count cross-check (file existence/size already confirmed above)")

    section("4) Cape Greco CRITICAL event preserved (requirement 3)")
    thresholds = HotspotThresholds()
    cg = alerts[alerts["hotspot_name"] == "Cape Greco MPA"].sort_values("time")
    assert (cg["status"] == CRITICAL).any(), "Cape Greco MPA never reached CRITICAL -- regression from F04"
    first_critical_time = cg.loc[cg["status"] == CRITICAL, "time"].iloc[0]
    min_distance_km = float(cg["min_distance_km"].min())
    max_fraction_critical = float(cg["fraction_inside_40km"].max())
    final_status = cg.iloc[-1]["status"]
    print(f"  alert rule in effect: >= {thresholds.min_fraction_critical:.0%} of active particles in the "
          f"40km polygon -> CRITICAL (user-directed change from the original 'any particle' rule, see progress.md)")
    print(f"  Cape Greco MPA first CRITICAL at: {first_critical_time} "
          f"(not pinned to one exact timestamp here -- it depends on the threshold above, "
          f"which is itself a deliberately configurable, user-tunable parameter)")
    print(f"  Cape Greco MPA minimum distance observed: {min_distance_km:.2f} km")
    print(f"  Cape Greco MPA maximum fraction of the slick ever inside the 40km zone: {max_fraction_critical:.1%}")
    print(f"  Cape Greco MPA status at final hour ({times[-1]}): {final_status}")

    triggered_hotspots = sorted(
        alerts.loc[alerts["status"] != "CLEAR", "hotspot_name"].unique()
    )
    critical_hotspots = sorted(
        alerts.loc[alerts["status"] == CRITICAL, "hotspot_name"].unique()
    )
    print(f"  hotspots ever WATCH or CRITICAL: {triggered_hotspots}")
    print(f"  hotspots ever CRITICAL: {critical_hotspots}")

    section("5) OpenOil validation result carried forward (F06)")
    print(comparison.to_string(index=False))
    err_72h = comparison.loc[comparison["checkpoint_hours"] == 72].iloc[0]
    print(f"  +72h mean error: {err_72h['mean_error_km']:.3f} km, max: {err_72h['max_error_km']:.3f} km")

    section("6) Writing run_summary.json")
    final_state = traj[traj["time"] == times[-1]]
    summary = {
        "case_id": "ow-0009",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "experiment": {
            "start_time_utc": pd.Timestamp(times[0]).isoformat(),
            "end_time_utc": pd.Timestamp(times[-1]).isoformat(),
            "duration_hours": duration_h,
            "timestep_hours": 1,
            "recorded_states": len(times),
            "initial_parcel_count": len(ids),
        },
        "forcing": {
            "ocean": {"source": "CMEMS", "file": "data/28th_april/ocean.nc", "variables": ["uo", "vo"]},
            "wind": {"source": "ERA5", "file": "data/28th_april/wind.nc", "variables": ["u10", "v10"]},
        },
        "physics": {
            "formula": "u_oil = current_drift_factor*uo + wind_drift_factor*u10; "
                       "v_oil = current_drift_factor*vo + wind_drift_factor*v10",
            "current_drift_factor": 1.0,
            "wind_drift_factor": 0.03,
            "stokes_drift": False,
            "weathering": False,
            "advection_scheme": "euler",
        },
        "nan_coastline_policy": {
            "modes": ["DIRECT", "NEIGHBOR_FALLBACK", "INVALID_ENVIRONMENT", "LAND_STRANDED", "ALREADY_STRANDED"],
            "fallback": "3x3 then 5x5 inverse-distance-weighted valid-ocean neighbors, "
                        "minimum 4 valid neighbors, weight=1/(d^2+epsilon)",
            "land_source": "static CMEMS uo/vo NaN mask (verified time-invariant across all 96 forcing timesteps)",
        },
        "hotspots": {
            "geojson": "regional_alert_geofences_wide.geojson",
            "hotspot_count": len(hotspots),
            "watch_km": 100,
            "critical_km": 40,
            "logic": f"(active particles inside critical polygon)/(active particles) >= "
                     f"{thresholds.min_fraction_critical:.0%} -> CRITICAL; elif (.../active) >= "
                     f"{thresholds.min_fraction_watch:.0%} of watch polygon -> WATCH; else CLEAR "
                     f"(user-directed change, post-F07, from the originally locked 'any particle' rule)",
            "min_fraction_critical": thresholds.min_fraction_critical,
            "min_fraction_watch": thresholds.min_fraction_watch,
            "triggered_hotspots": triggered_hotspots,
            "critical_hotspots": critical_hotspots,
            "cape_greco_first_critical_time_utc": pd.Timestamp(first_critical_time).isoformat(),
            "cape_greco_min_distance_km": round(min_distance_km, 3),
            "cape_greco_max_fraction_in_critical_zone": round(max_fraction_critical, 4),
        },
        "outputs": {
            "trajectory": "output/ow-0009/trajectory.parquet",
            "hourly_diagnostics": "output/ow-0009/hourly_diagnostics.parquet",
            "hotspot_alerts": "output/ow-0009/hotspot_alerts.parquet",
            "animation_mp4": "output/ow-0009/ow0009_forward_animation.mp4",
            "representative_frames": [
                "output/ow-0009/frames/initial.png",
                "output/ow-0009/frames/watch.png",
                "output/ow-0009/frames/critical_cape_greco.png",
                "output/ow-0009/frames/final.png",
            ],
            "openoil_reference": "output/ow-0009/openoil_reference.parquet",
            "openoil_comparison": "output/ow-0009/openoil_comparison.parquet",
        },
        "final_state": {
            "active_count": int((final_state["valid"] & ~final_state["stranded"]).sum()),
            "stranded_count": int(final_state["stranded"].sum()),
        },
        "openoil_validation": {
            "checkpoints_hours": comparison["checkpoint_hours"].tolist(),
            "mean_error_km": [round(v, 4) for v in comparison["mean_error_km"].tolist()],
            "max_error_km": [round(v, 4) for v in comparison["max_error_km"].tolist()],
            "verdict": "ACCEPTABLE",
            "primary_residual_cause": "spherical-Earth displacement approximation (iste.geometry) "
                                       "vs OpenDrift's WGS84-ellipsoidal geodesic forward calculation",
        },
        "known_limitations": [
            "Hotspot alert logic uses a user-directed 30% minimum-fraction-of-active-particles "
            "rule (per zone), requested and applied after the original F00-F07 pilot completed -- "
            "this is a deliberate, documented deviation from README section 10's originally locked "
            "'any single particle triggers the zone' rule (CLAUDE.md section 13 requires such "
            "deviations be documented, not silent; see progress.md's post-F07 revision entry). "
            "Both fractions are configurable via fwd.hotspots.HotspotThresholds.",
            "Spherical-Earth displacement approximation (iste.geometry.metres_to_degrees) vs "
            "OpenDrift's WGS84-ellipsoidal pyproj.Geod -- identified as the primary source of the "
            "small (<1%) OpenOil comparison residual; not fixed, per this project's reuse-Stage-III policy.",
            "This specific case's real 72h run never triggers NEIGHBOR_FALLBACK or LAND_STRANDED "
            "(the slick stays over 60 km offshore throughout) -- those code paths were validated "
            "separately against synthetic/boundary points in F01-F03, not exercised end-to-end here.",
            "Land classification source is the static CMEMS uo/vo NaN mask (~4.6 km grid resolution), "
            "not an independent coastline shapefile (none exists in this project's environment).",
            "No frontend/application exists anywhere in this workspace to consume these outputs "
            "(confirmed by a dedicated search in F05); outputs are standalone files for now.",
            "CLAUDE.md/README's documented forcing paths (data/ocean.nc, files/28th_june/wind.nc) "
            "still do not match the actual paths used throughout F01-F07 (data/28th_april/ocean.nc, "
            "data/28th_april/wind.nc) -- flagged since F00, docs not yet updated.",
        ],
    }
    summary_path = OUT / "run_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {summary_path}")

    section("ALL CHECKS PASSED")
    print("F07 finalization: COMPLETE")


if __name__ == "__main__":
    main()
