"""ow-0008 case -- final verification + run summary (F07 equivalent; not a test file
-- CLAUDE.md forbids `tests/`/`test_*.py`).

Mirrors `scripts/f07_finalize.py` exactly, pointed at `output/ow-0008/`. Cross-checks
every stage's output for mutual consistency (not just each stage's own already-passed
validation in isolation), and writes one concise machine-readable JSON summary
(`output/ow-0008/run_summary.json`). Reads only -- does not modify any prior output.

Run: python scripts/ow0008_finalize.py
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

from fwd.cases import get_case  # noqa: E402
from fwd.hotspots import CRITICAL, WATCH, HotspotThresholds, load_hotspots  # noqa: E402

CASE = get_case("ow-0008")
OUT = REPO_ROOT / "output" / CASE.case_id
START_TIME = CASE.observation_time
END_TIME = START_TIME + np.timedelta64(72, "h")


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main() -> None:
    section(f"0) Inventory of output/{CASE.case_id}/")
    frames_dir = OUT / "frames"
    frame_files = sorted(p.name for p in frames_dir.glob("*.png")) if frames_dir.exists() else []
    required_files = [
        "trajectory.parquet", "hourly_diagnostics.parquet", "hotspot_alerts.parquet",
        "openoil_reference.parquet", "openoil_comparison.parquet",
        f"{CASE.case_id.replace('-', '')}_forward_animation.mp4",
    ] + [f"frames/{f}" for f in frame_files]
    assert "initial.png" in frame_files and "final.png" in frame_files, (
        f"expected at least initial.png and final.png in frames/, found {frame_files}"
    )
    missing = [f for f in required_files if not (OUT / f).exists()]
    assert not missing, f"missing expected output files: {missing}"
    for f in required_files:
        p = OUT / f
        print(f"  {'OK':4s} {f:55s} {p.stat().st_size:>10,} bytes")

    section("1) Loading all tabular outputs")
    traj = pd.read_parquet(OUT / "trajectory.parquet")
    diag = pd.read_parquet(OUT / "hourly_diagnostics.parquet")
    alerts = pd.read_parquet(OUT / "hotspot_alerts.parquet")
    comparison = pd.read_parquet(OUT / "openoil_comparison.parquet")
    hotspots = load_hotspots()
    thresholds = HotspotThresholds()
    print(f"trajectory: {len(traj)} rows | hourly_diagnostics: {len(diag)} rows | "
          f"hotspot_alerts: {len(alerts)} rows | openoil_comparison: {len(comparison)} rows")

    section("2) Core experiment facts")
    times = sorted(traj["time"].unique())
    n = CASE.total_parcels
    assert len(times) == 73, f"expected 73 trajectory states, got {len(times)}"
    assert times[0] == START_TIME, times[0]
    assert times[-1] == END_TIME, times[-1]
    duration_h = int((times[-1] - times[0]) / np.timedelta64(1, "h"))
    assert duration_h == 72, duration_h
    ids = sorted(traj["particle_id"].unique())
    assert len(ids) == n and ids == list(range(n)), f"expected exactly parcel IDs 0..{n-1}"
    print(f"  72 propagation hours: OK ({duration_h}h between {times[0]} and {times[-1]})")
    print(f"  73 trajectory states: OK")
    print(f"  {n} parcel IDs (0..{n-1}): OK")
    print(f"  final time = {times[-1]} : OK (matches {END_TIME})")

    watch_km = {round(h.watch_radius_km) for h in hotspots}
    critical_km = {round(h.critical_radius_km) for h in hotspots}
    assert watch_km == {100}, watch_km
    assert critical_km == {40}, critical_km
    print(f"  100 km WATCH / 40 km CRITICAL thresholds (from the actual polygons, all 11 hotspots): OK")

    section("3) Cross-consistency between trajectory / hourly_diagnostics / hotspot_alerts")
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

    mp4_path = OUT / f"{CASE.case_id.replace('-', '')}_forward_animation.mp4"
    from subprocess import run as _run
    try:
        probe = _run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=nb_frames,r_frame_rate", "-show_entries", "format=duration",
             "-of", "json", str(mp4_path)],
            capture_output=True, text=True, check=True,
        )
        probe_json = json.loads(probe.stdout)
        nb_frames = int(probe_json["streams"][0]["nb_frames"])
        fps = probe_json["streams"][0]["r_frame_rate"]
        video_duration_s = float(probe_json["format"]["duration"])
        print(f"  MP4 probe: {nb_frames} frames @ {fps} fps, {video_duration_s:.1f}s "
              f"(>= 289 propagation substeps expected, plus an opening overview/zoom sequence)")
        assert nb_frames >= 289, f"expected at least 289 rendered frames, ffprobe found {nb_frames}"
    except FileNotFoundError:
        print("  ffprobe not available -- skipping frame-count cross-check (file existence/size already confirmed above)")

    section("4) Hotspot alert summary (requirement parity with ow-0009's F07)")
    triggered_hotspots = sorted(alerts.loc[alerts["status"] != "CLEAR", "hotspot_name"].unique())
    critical_hotspots = sorted(alerts.loc[alerts["status"] == CRITICAL, "hotspot_name"].unique())
    print(f"  alert rule in effect: >= {thresholds.min_fraction_critical:.0%} of active particles in the "
          f"40km polygon -> CRITICAL; >= {thresholds.min_fraction_watch:.0%} in the 100km polygon -> WATCH")
    print(f"  hotspots ever WATCH or CRITICAL: {triggered_hotspots}")
    print(f"  hotspots ever CRITICAL: {critical_hotspots}")
    hotspot_summaries = {}
    for name in triggered_hotspots:
        g = alerts[alerts["hotspot_name"] == name].sort_values("time")
        fw = g["first_watch_time"].dropna()
        fc = g["first_critical_time"].dropna()
        min_dist = float(g["min_distance_km"].min())
        final_status = g.iloc[-1]["status"]
        print(f"    {name:35s} min_dist={min_dist:7.2f} km  first_watch={fw.iloc[0] if len(fw) else 'never'}  "
              f"first_critical={fc.iloc[0] if len(fc) else 'never'}  final_status={final_status}")
        hotspot_summaries[name] = {
            "min_distance_km": round(min_dist, 3),
            "first_watch_time_utc": pd.Timestamp(fw.iloc[0]).isoformat() if len(fw) else None,
            "first_critical_time_utc": pd.Timestamp(fc.iloc[0]).isoformat() if len(fc) else None,
            "final_status": final_status,
        }

    section("5) OpenOil validation result carried forward")
    print(comparison.to_string(index=False))
    err_72h = comparison.loc[comparison["checkpoint_hours"] == 72].iloc[0]
    print(f"  +72h mean error: {err_72h['mean_error_km']:.3f} km, max: {err_72h['max_error_km']:.3f} km")
    n_excluded_total = int(comparison["n_excluded_nonfinite"].sum()) if "n_excluded_nonfinite" in comparison.columns else 0
    if n_excluded_total:
        print(f"  particle-checkpoints excluded (non-finite in one trajectory): {n_excluded_total}")

    section("6) Writing run_summary.json")
    final_state = traj[traj["time"] == times[-1]]
    final_active = int((final_state["valid"] & ~final_state["stranded"]).sum())
    final_stranded = int(final_state["stranded"].sum())
    summary = {
        "case_id": CASE.case_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "experiment": {
            "start_time_utc": pd.Timestamp(times[0]).isoformat(),
            "end_time_utc": pd.Timestamp(times[-1]).isoformat(),
            "duration_hours": duration_h,
            "timestep_hours": 1,
            "recorded_states": len(times),
            "initial_parcel_count": n,
        },
        "forcing": {
            "ocean": {"source": "CMEMS", "file": "data/14th_june/ocean.nc", "variables": ["uo", "vo"]},
            "wind": {"source": "ERA5", "file": "data/14th_june/wind.nc", "variables": ["u10", "v10"],
                     "note": "time coordinate is 'valid_time' in this file, not 'time' as in ow-0009's "
                             "wind.nc -- auto-detected by fwd.forcing._detect_wind_time_dim"},
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
            "land_source": "static CMEMS uo/vo NaN mask",
        },
        "hotspots": {
            "geojson": "regional_alert_geofences_wide.geojson",
            "hotspot_count": len(hotspots),
            "watch_km": 100,
            "critical_km": 40,
            "min_fraction_critical": thresholds.min_fraction_critical,
            "min_fraction_watch": thresholds.min_fraction_watch,
            "triggered_hotspots": triggered_hotspots,
            "critical_hotspots": critical_hotspots,
            "per_hotspot": hotspot_summaries,
        },
        "outputs": {
            "trajectory": f"output/{CASE.case_id}/trajectory.parquet",
            "hourly_diagnostics": f"output/{CASE.case_id}/hourly_diagnostics.parquet",
            "hotspot_alerts": f"output/{CASE.case_id}/hotspot_alerts.parquet",
            "animation_mp4": f"output/{CASE.case_id}/{CASE.case_id.replace('-', '')}_forward_animation.mp4",
            "representative_frames": [f"output/{CASE.case_id}/frames/{f}" for f in frame_files],
            "openoil_reference": f"output/{CASE.case_id}/openoil_reference.parquet",
            "openoil_comparison": f"output/{CASE.case_id}/openoil_comparison.parquet",
        },
        "final_state": {
            "active_count": final_active,
            "stranded_count": final_stranded,
        },
        "openoil_validation": {
            "checkpoints_hours": comparison["checkpoint_hours"].tolist(),
            "mean_error_km": [round(v, 4) for v in comparison["mean_error_km"].tolist()],
            "max_error_km": [round(v, 4) for v in comparison["max_error_km"].tolist()],
            "n_excluded_nonfinite_total": n_excluded_total,
        },
        "known_limitations": [
            "Hotspot alert logic uses a >=30% minimum-fraction-of-active-particles rule (per zone), "
            "the same user-directed rule now in effect for ow-0009 (see progress.md's post-F07 "
            "revision 3) -- a deliberate, documented deviation from README section 10's originally "
            "locked 'any single particle triggers the zone' rule.",
            "This case starts only ~29 km from the nearest CMEMS-masked land cell (vs ~77 km for "
            "ow-0009) and Beirut Port is CRITICAL from t0 through the final hour -- a genuinely "
            "different regime from ow-0009's always-offshore case.",
            "Spherical-Earth displacement approximation (iste.geometry.metres_to_degrees) vs "
            "OpenDrift's WGS84-ellipsoidal pyproj.Geod remains a known source of small residual "
            "error in the OpenOil comparison (see the printed investigation and progress.md).",
            "ERA5 wind file for this case uses 'valid_time' as its time coordinate name (not 'time' "
            "as in ow-0009's file) -- handled by auto-detection in fwd.forcing, not a hard-coded fix.",
            "Land classification source is the static CMEMS uo/vo NaN mask, not an independent "
            "coastline shapefile (none exists in this project's environment); OpenDrift's reference "
            "run uses its own real GSHHG coastline instead, a genuine methodological difference "
            "documented in this case's own comparison script/output.",
            "No frontend/application exists anywhere in this workspace to consume these outputs; "
            "outputs are standalone files for now.",
        ],
    }
    summary_path = OUT / "run_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {summary_path}")

    section("ALL CHECKS PASSED")
    print(f"{CASE.case_id} finalization: COMPLETE")


if __name__ == "__main__":
    main()
