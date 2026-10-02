"""Build the full deliverable set (hotspot alerts, beautified two-panel animation,
representative frames, run_summary.json, CSVs) for the full-physics OpenOil
weathering+diffusion runs (`scripts/openoil_full_physics_run.py`), for both ow-0009
and ow-0008. Not a test file -- CLAUDE.md forbids `tests/`/`test_*.py`.

Reads `openoil_full_physics_output/<case>/trajectory.parquet` (already produced by
the run script) -- does not call OpenOil itself, so this runs under the normal
`python3` interpreter, not py -3.13.

Hotspot alerts are recomputed against THIS run's own positions (reusing
`fwd.hotspots.evaluate_hotspot_alerts` unchanged) -- never copied from the locked
model's `output/<case>/hotspot_alerts.parquet`, since the two trajectories differ
(diffusion + weathering + a real coastline). "Active" for hotspot-evaluation purposes
is `moving == 1` (OpenDrift's own standard flag for "still being advected", robust
regardless of *why* a particle stopped -- stranding, dispersal, or anything else);
"stranded" (this project's own trajectory-schema field, reused for
`fwd.animate.render_frame`'s stranded-marker styling) is `moving == 0`.

The animation is deliberately "beautified" beyond the locked model's own style (per
request): larger figure/DPI, a second panel showing the live oil mass budget
(remaining / evaporated / dispersed / biodegraded, from the real seeded total mass),
and particles colour-coded by `fraction_evaporated` (how weathered each one is) with
a colorbar, instead of a single flat colour.

Run: python scripts/build_openoil_full_physics_outputs.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import ListedColormap, Normalize  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Polygon as MplPolygon  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from fwd.animate import (  # noqa: E402
    COLOR_COASTLINE, COLOR_CRITICAL_FILL, COLOR_LAND, COLOR_STRANDED, COLOR_WATCH_FILL,
    COLOR_ZONE_OUTLINE, build_trail_segments, compute_view_bounds, ease_in_out,
    interpolate_view_bounds, precompute_particle_paths,
)
from fwd.cases import CASES as CASE_CONFIGS  # noqa: E402
from fwd.forcing import ForcingAdapter  # noqa: E402
from fwd.hotspots import CLEAR, CRITICAL, WATCH, evaluate_hotspot_alerts, load_hotspots  # noqa: E402

OUTPUT_ROOT = REPO_ROOT / "openoil_full_physics_output"
FPS = 10
DPI = 140
FIGSIZE = (15, 8)
SUBSTEPS_PER_HOUR = 4
INTRO_HOLD_SECONDS = 3.0
INTRO_ZOOM_SECONDS = 1.5
INTRO_HOLD_FRAMES = round(INTRO_HOLD_SECONDS * FPS)
INTRO_ZOOM_FRAMES = round(INTRO_ZOOM_SECONDS * FPS)

WEATHER_CMAP = plt.get_cmap("YlOrRd")


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def _slug(name: str) -> str:
    return name.lower().replace(" ", "_").replace("(", "").replace(")", "")


def interpolate_positions_extra(traj: pd.DataFrame, substeps_per_hour: int) -> pd.DataFrame:
    """Like fwd.animate.interpolate_positions, but also linearly interpolates
    fraction_evaporated (for smooth particle-colour transitions) and carries
    moving/mass columns from the floor hour (categorical/monotonic-ish, not smoothly
    interpolated)."""
    times = sorted(traj["time"].unique())
    frames = []
    for k in range(len(times) - 1):
        t0, t1 = times[k], times[k + 1]
        g0 = traj[traj["time"] == t0].set_index("particle_id").sort_index()
        g1 = traj[traj["time"] == t1].set_index("particle_id").sort_index()
        dt_ns = int((t1 - t0) / np.timedelta64(1, "ns"))
        for s in range(substeps_per_hour):
            frac = s / substeps_per_hour
            t = t0 + np.timedelta64(int(round(dt_ns * frac)), "ns")
            lon = g0["lon"].to_numpy() * (1 - frac) + g1["lon"].to_numpy() * frac
            lat = g0["lat"].to_numpy() * (1 - frac) + g1["lat"].to_numpy() * frac
            fe = g0["fraction_evaporated"].to_numpy() * (1 - frac) + g1["fraction_evaporated"].to_numpy() * frac
            frames.append(pd.DataFrame({
                "particle_id": g0.index.to_numpy(), "time": t, "floor_time": t0,
                "lon": lon, "lat": lat, "fraction_evaporated": fe,
                "moving": g0["moving"].to_numpy(),
                "mass_oil": g0["mass_oil"].to_numpy(),
                "mass_evaporated": g0["mass_evaporated"].to_numpy(),
                "mass_dispersed": g0["mass_dispersed"].to_numpy(),
                "mass_biodegraded": g0["mass_biodegraded"].to_numpy(),
            }))
    g_last = traj[traj["time"] == times[-1]].sort_values("particle_id")
    frames.append(pd.DataFrame({
        "particle_id": g_last["particle_id"].to_numpy(), "time": times[-1], "floor_time": times[-1],
        "lon": g_last["lon"].to_numpy(), "lat": g_last["lat"].to_numpy(),
        "fraction_evaporated": g_last["fraction_evaporated"].to_numpy(),
        "moving": g_last["moving"].to_numpy(), "mass_oil": g_last["mass_oil"].to_numpy(),
        "mass_evaporated": g_last["mass_evaporated"].to_numpy(),
        "mass_dispersed": g_last["mass_dispersed"].to_numpy(),
        "mass_biodegraded": g_last["mass_biodegraded"].to_numpy(),
    }))
    return pd.concat(frames, ignore_index=True)


def status_by_hotspot_at(alerts_df: pd.DataFrame, floor_time) -> dict[str, str]:
    g = alerts_df[alerts_df["time"] == floor_time]
    return dict(zip(g["hotspot_name"], g["status"]))


def pick_representative_hours(traj: pd.DataFrame, alerts_df: pd.DataFrame) -> dict:
    times = sorted(traj["time"].unique())
    hours = {"initial": times[0], "final": times[-1]}
    watch_rows = alerts_df[alerts_df["status"] == WATCH].sort_values("time")
    if len(watch_rows):
        w = watch_rows.iloc[0]
        hours[f"watch_{_slug(w['hotspot_name'])}"] = w["time"]
    critical_rows = alerts_df[alerts_df["status"] == CRITICAL].sort_values("time")
    if len(critical_rows):
        c = critical_rows.iloc[0]
        hours[f"critical_{_slug(c['hotspot_name'])}"] = c["time"]
    return hours


def draw_beautified_frame(
    fig, ax_map, ax_budget, cax, land, hotspots, status_by_name, positions, trail_segments,
    display_time, view_bounds, case_label, mass_history, total_mass_kg, overview_label=None,
):
    ax_map.clear()
    ax_map.set_xlim(view_bounds["west"], view_bounds["east"])
    ax_map.set_ylim(view_bounds["south"], view_bounds["north"])
    mean_lat = (view_bounds["south"] + view_bounds["north"]) / 2.0
    ax_map.set_aspect(1.0 / np.cos(np.deg2rad(mean_lat)))

    lon_edges = np.concatenate([land.lon - np.diff(land.lon, prepend=land.lon[0] - (land.lon[1] - land.lon[0])) / 2,
                                 [land.lon[-1] + (land.lon[-1] - land.lon[-2]) / 2]])
    lat_edges = np.concatenate([land.lat - np.diff(land.lat, prepend=land.lat[0] - (land.lat[1] - land.lat[0])) / 2,
                                 [land.lat[-1] + (land.lat[-1] - land.lat[-2]) / 2]])
    ax_map.pcolormesh(lon_edges, lat_edges, land.ocean_valid_mask.astype(int),
                       cmap=ListedColormap(["#ded9c9", "#dbeeff"]), vmin=0, vmax=1, shading="flat", zorder=0)
    lon_mesh, lat_mesh = np.meshgrid(land.lon, land.lat)
    ax_map.contour(lon_mesh, lat_mesh, land.ocean_valid_mask.astype(float), levels=[0.5],
                    colors=COLOR_COASTLINE, linewidths=0.8, zorder=1)

    for h in hotspots:
        status = status_by_name.get(h.name, CLEAR)
        ax_map.add_patch(MplPolygon(h.watch_ring, closed=True, fill=False, edgecolor=COLOR_ZONE_OUTLINE,
                                     linewidth=1.0, zorder=2))
        if status == WATCH:
            ax_map.add_patch(MplPolygon(h.watch_ring, closed=True, facecolor=COLOR_WATCH_FILL,
                                         edgecolor=COLOR_WATCH_FILL, alpha=0.35, linewidth=1.3, zorder=2))
        elif status == CRITICAL:
            ax_map.add_patch(MplPolygon(h.watch_ring, closed=True, facecolor=COLOR_WATCH_FILL,
                                         edgecolor=COLOR_WATCH_FILL, alpha=0.20, linewidth=1.3, zorder=2))
            ax_map.add_patch(MplPolygon(h.critical_ring, closed=True, facecolor=COLOR_CRITICAL_FILL,
                                         edgecolor=COLOR_CRITICAL_FILL, alpha=0.55, linewidth=1.7, zorder=3))
        label_color = COLOR_CRITICAL_FILL if status == CRITICAL else ("#8a6100" if status == WATCH else "#555555")
        weight = "bold" if status != CLEAR else "normal"
        ax_map.text(h.center_lon, h.center_lat, h.name, fontsize=6.5, color=label_color, weight=weight,
                     ha="center", va="center", zorder=5, clip_on=True,
                     bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))

    if trail_segments:
        ax_map.add_collection(LineCollection(trail_segments, colors="#0b3d91", linewidths=0.8, alpha=0.35, zorder=3))

    active = positions[positions["moving"] == 1]
    stranded = positions[positions["moving"] == 0]
    sc = ax_map.scatter(active["lon"], active["lat"], c=active["fraction_evaporated"], cmap=WEATHER_CMAP,
                         norm=Normalize(vmin=0, vmax=1), s=22, zorder=4, linewidths=0.4, edgecolors="#222222")
    if len(stranded):
        ax_map.scatter(stranded["lon"], stranded["lat"], s=36, c=COLOR_STRANDED, marker="x", zorder=4)

    ax_map.set_xlabel("Longitude")
    ax_map.set_ylabel("Latitude")
    n_active, n_stranded = len(active), len(stranded)
    subtitle = overview_label if overview_label is not None else f"active parcels: {n_active}   stranded/stopped: {n_stranded}"
    ax_map.set_title(
        f"{case_label} -- OpenOil full physics (weathering + diffusion)      "
        f"{np.datetime_as_string(np.datetime64(display_time, 'ns'), unit='m')} UTC\n{subtitle}",
        fontsize=10.5, loc="left",
    )
    legend_handles = [
        Line2D([], [], marker="o", linestyle="", color=WEATHER_CMAP(0.15), markersize=7, markeredgecolor="#222", label="Active parcel (colour = weathering)"),
        Line2D([], [], marker="x", linestyle="", color=COLOR_STRANDED, markersize=8, label="Stranded / stopped parcel"),
        Line2D([], [], color="#0b3d91", linewidth=1.2, alpha=0.4, label="Drift path (since t0)"),
        Patch(facecolor=COLOR_WATCH_FILL, edgecolor=COLOR_ZONE_OUTLINE, alpha=0.35, label="Watch Zone (100 km)"),
        Patch(facecolor=COLOR_CRITICAL_FILL, edgecolor=COLOR_CRITICAL_FILL, alpha=0.5, label="Critical Strike Zone (40 km)"),
        Patch(facecolor="#ded9c9", edgecolor=COLOR_COASTLINE, label="Land"),
    ]
    ax_map.legend(handles=legend_handles, loc="lower left", fontsize=6.5, framealpha=0.88)

    # Reuse a single, pre-allocated colorbar axes (cax) across every frame of one
    # figure -- calling fig.colorbar(..., ax=ax_map) instead (once per frame, ~290
    # times) was the original bug: each call inserts a NEW axes into ax_map's
    # gridspec without removing the previous one, so the gridspec's own internal
    # position bookkeeping grows every frame -- confirmed by the render slowing from
    # ~0.2s/frame to ~6.4s/frame before eventually hitting a RecursionError in
    # matplotlib's gridspec code around frame ~300. fig.colorbar(sc, cax=cax) draws
    # directly into the same fixed axes every time, with no gridspec mutation at all.
    cax.clear()
    cbar = fig.colorbar(sc, cax=cax)
    cbar.set_label("Fraction evaporated", fontsize=7.5)
    cbar.ax.tick_params(labelsize=6.5)

    # -- mass budget panel --
    ax_budget.clear()
    hist_t = [np.datetime64(t, "ns") for t in mass_history["time"]]
    remaining = np.array(mass_history["mass_oil"]) / total_mass_kg * 100
    evaporated = np.array(mass_history["mass_evaporated"]) / total_mass_kg * 100
    dispersed = np.array(mass_history["mass_dispersed"]) / total_mass_kg * 100
    biodegraded = np.array(mass_history["mass_biodegraded"]) / total_mass_kg * 100
    ax_budget.stackplot(
        hist_t, remaining, evaporated, dispersed, biodegraded,
        colors=["#1f6f43", "#f4a300", "#8a5cf6", "#999999"],
        labels=["Oil remaining", "Evaporated", "Dispersed", "Biodegraded"],
        alpha=0.9,
    )
    ax_budget.set_ylim(0, 100)
    ax_budget.set_ylabel("% of initial oil mass")
    ax_budget.set_title(f"Oil mass budget (total seeded: {total_mass_kg:,.0f} kg)", fontsize=9.5)
    ax_budget.legend(loc="lower left", fontsize=6.5, framealpha=0.85)
    ax_budget.tick_params(axis="x", rotation=30, labelsize=6.5)
    ax_budget.tick_params(axis="y", labelsize=7)
    if len(hist_t) > 1:
        ax_budget.set_xlim(hist_t[0], hist_t[-1])


def build_case(case_id: str) -> None:
    section(f"{case_id}: loading full-physics OpenOil trajectory")
    case_cfg = CASE_CONFIGS[case_id]
    out_dir = OUTPUT_ROOT / case_id
    frames_dir = out_dir / "frames"
    tmp_dir = out_dir / "_anim_frames_tmp"
    mp4_path = out_dir / f"openoil_full_physics_{case_id.replace('-', '')}_animation.mp4"

    traj = pd.read_parquet(out_dir / "trajectory.parquet")
    n = case_cfg.total_parcels
    print(f"trajectory: {len(traj)} rows, {traj['time'].nunique()} timestamps, {traj['particle_id'].nunique()} particles")
    assert traj["particle_id"].nunique() == n and traj["time"].nunique() == 73

    total_mass_kg = float(traj[traj["time"] == traj["time"].min()]["mass_oil"].sum())
    print(f"total seeded mass: {total_mass_kg:.2f} kg")

    adapter = ForcingAdapter(ocean_path=str(case_cfg.ocean_path), wind_path=str(case_cfg.wind_path))
    hotspots = load_hotspots()
    land = adapter.ocean.lat, adapter.ocean.lon
    from fwd.animate import LandBackground
    land_bg = LandBackground(lon=adapter.ocean.lon, lat=adapter.ocean.lat, ocean_valid_mask=adapter.ocean_valid_mask)

    section(f"{case_id}: recomputing hotspot alerts against this trajectory")
    hotspot_input = traj[["particle_id", "time", "lon", "lat"]].copy()
    hotspot_input["valid"] = True
    hotspot_input["stranded"] = traj["moving"] == 0
    alerts_df = evaluate_hotspot_alerts(hotspot_input, hotspots)
    alerts_out = out_dir / "hotspot_alerts.parquet"
    alerts_df.to_parquet(alerts_out, index=False)
    csv_dir = out_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    alerts_df.to_csv(csv_dir / "hotspot_alerts.csv", index=False)
    triggered = sorted(alerts_df.loc[alerts_df["status"] != CLEAR, "hotspot_name"].unique())
    critical_hotspots = sorted(alerts_df.loc[alerts_df["status"] == CRITICAL, "hotspot_name"].unique())
    print(f"hotspots triggered: {triggered}  (CRITICAL: {critical_hotspots})")

    # status timeline CSV, matching the locked model's own convention
    matrix = alerts_df.pivot(index="hotspot_name", columns="time", values="status")
    matrix = matrix.reindex(index=[h.name for h in hotspots])
    matrix = matrix[sorted(matrix.columns)]
    matrix.columns = [pd.Timestamp(c).strftime("%Y-%m-%d %H:%M:%S") for c in matrix.columns]
    matrix.index.name = "hotspot_name"
    matrix.to_csv(csv_dir / "hotspot_status_timeline.csv")

    representative_hours = pick_representative_hours(traj, alerts_df)
    print(f"representative hours: {representative_hours}")

    view_bounds = compute_view_bounds(traj)
    print(f"view bounds: {view_bounds}  "
          f"({view_bounds['east']-view_bounds['west']:.3f} x {view_bounds['north']-view_bounds['south']:.3f} deg)")

    paths = precompute_particle_paths(traj[["particle_id", "time", "lon", "lat"]])

    # mass history up to each hour (for the budget panel)
    mass_by_time = traj.groupby("time")[["mass_oil", "mass_evaporated", "mass_dispersed", "mass_biodegraded"]].sum().reset_index()

    def mass_history_upto(t):
        sub = mass_by_time[mass_by_time["time"] <= t]
        return {col: sub[col].tolist() for col in ["time", "mass_oil", "mass_evaporated", "mass_dispersed", "mass_biodegraded"]}

    section(f"{case_id}: representative frames")
    frames_dir.mkdir(parents=True, exist_ok=True)
    for tag, t in representative_hours.items():
        positions = traj[traj["time"] == t]
        assert len(positions) == n
        status = status_by_hotspot_at(alerts_df, t)
        trail_segments = build_trail_segments(paths, t, positions[["particle_id", "lon", "lat"]])
        fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
        gs = fig.add_gridspec(1, 3, width_ratios=[2.0, 0.06, 1], wspace=0.55)
        ax_map, cax, ax_budget = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
        draw_beautified_frame(fig, ax_map, ax_budget, cax, land_bg, hotspots, status, positions, trail_segments,
                               t, view_bounds, case_id, mass_history_upto(t), total_mass_kg)
        path = frames_dir / f"{tag}.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"  {tag:26s} t={t}  -> {path}")

    section(f"{case_id}: building interpolated positions for smooth playback")
    t0 = time.time()
    pos_interp = interpolate_positions_extra(traj, SUBSTEPS_PER_HOUR)
    n_hours = traj["time"].nunique()
    expected_frames = (n_hours - 1) * SUBSTEPS_PER_HOUR + 1
    print(f"interpolated {pos_interp['time'].nunique()} frame-times in {time.time()-t0:.2f}s")
    assert pos_interp["time"].nunique() == expected_frames

    section(f"{case_id}: opening sequence + full frame rendering")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)
    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    gs = fig.add_gridspec(1, 3, width_ratios=[2.0, 0.06, 1], wspace=0.55)
    ax_map, cax, ax_budget = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
    frame_idx = 0
    t_start = time.time()

    t0v = traj["time"].min()
    t0_positions = traj[traj["time"] == t0v]
    t0_status = status_by_hotspot_at(alerts_df, t0v)
    t0_trail = build_trail_segments(paths, t0v, t0_positions[["particle_id", "lon", "lat"]])
    overview_bounds = {
        "west": float(land_bg.lon.min()), "east": float(land_bg.lon.max()),
        "south": float(land_bg.lat.min()), "north": float(land_bg.lat.max()),
    }
    for _ in range(INTRO_HOLD_FRAMES):
        draw_beautified_frame(fig, ax_map, ax_budget, cax, land_bg, hotspots, t0_status, t0_positions, t0_trail,
                               t0v, overview_bounds, case_id, mass_history_upto(t0v), total_mass_kg,
                               overview_label="Regional hotspot overview -- 11 monitored zones")
        fig.savefig(tmp_dir / f"frame_{frame_idx:05d}.png", bbox_inches="tight")
        frame_idx += 1
    for i in range(INTRO_ZOOM_FRAMES):
        frac = (i + 1) / INTRO_ZOOM_FRAMES
        zoom_bounds = interpolate_view_bounds(overview_bounds, view_bounds, frac)
        draw_beautified_frame(fig, ax_map, ax_budget, cax, land_bg, hotspots, t0_status, t0_positions, t0_trail,
                               t0v, zoom_bounds, case_id, mass_history_upto(t0v), total_mass_kg,
                               overview_label=f"Zooming to the {case_id} slick...")
        fig.savefig(tmp_dir / f"frame_{frame_idx:05d}.png", bbox_inches="tight")
        frame_idx += 1
    print(f"opening sequence: {frame_idx} frames in {time.time()-t_start:.1f}s")

    frame_times = sorted(pos_interp["time"].unique())
    t_start = time.time()
    for t in frame_times:
        frame_pos = pos_interp[pos_interp["time"] == t]
        floor_t = frame_pos["floor_time"].iloc[0]
        status = status_by_hotspot_at(alerts_df, floor_t)
        trail_segments = build_trail_segments(paths, floor_t, frame_pos[["particle_id", "lon", "lat"]])
        draw_beautified_frame(fig, ax_map, ax_budget, cax, land_bg, hotspots, status, frame_pos, trail_segments,
                               t, view_bounds, case_id, mass_history_upto(floor_t), total_mass_kg)
        fig.savefig(tmp_dir / f"frame_{frame_idx:05d}.png", bbox_inches="tight")
        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"  frame {frame_idx}  t={t}  elapsed={time.time()-t_start:.1f}s")
    plt.close(fig)
    total_frames = frame_idx
    print(f"rendered {total_frames} total frames in {time.time()-t_start:.1f}s")

    section(f"{case_id}: encoding MP4")
    if mp4_path.exists():
        mp4_path.unlink()
    cmd = ["ffmpeg", "-y", "-framerate", str(FPS), "-i", str(tmp_dir / "frame_%05d.png"),
           "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-pix_fmt", "yuv420p", str(mp4_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout[-2000:]); print(result.stderr[-2000:])
        raise RuntimeError("ffmpeg failed")
    assert mp4_path.exists() and mp4_path.stat().st_size > 0
    print(f"wrote {mp4_path}  ({mp4_path.stat().st_size/1e6:.2f} MB, {total_frames} frames @ {FPS}fps = {total_frames/FPS:.1f}s)")
    shutil.rmtree(tmp_dir)

    section(f"{case_id}: writing run_summary.json")
    final = traj[traj["time"] == traj["time"].max()]
    summary = {
        "case_id": case_id,
        "model": "Real OpenOil/OpenDrift (opendrift==1.14.11), NOT this project's own locked model",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "deterministic": False,
        "random_seed": 20260926,
        "experiment": {
            "start_time_utc": pd.Timestamp(traj["time"].min()).isoformat(),
            "end_time_utc": pd.Timestamp(traj["time"].max()).isoformat(),
            "duration_hours": 72,
            "recorded_states": int(traj["time"].nunique()),
            "initial_parcel_count": n,
        },
        "physics": {
            "wind_drift_factor": 0.03,
            "current_drift_factor": 1.0,
            "weathering": "ON (OpenOil defaults: evaporation, emulsification, dispersion)",
            "biodegradation": "OFF (OpenOil default)",
            "horizontal_diffusivity_m2s": 10.0,
            "current_uncertainty": "OpenOil default (0.05)",
            "wind_uncertainty": "OpenOil default (0.5)",
            "vertical_mixing": "ON (OpenOil default)",
            "coastline": "real GSHHG coastline (OpenOil default 'stranding'), not this project's CMEMS-mask proxy",
            "stokes_drift": "nominally True (OpenOil default) but functionally inert -- no wave reader attached",
        },
        "mass_budget": {
            "total_seeded_kg": round(total_mass_kg, 2),
            "final_mass_oil_kg": round(float(final["mass_oil"].sum()), 2),
            "final_mass_evaporated_kg": round(float(final["mass_evaporated"].sum()), 2),
            "final_mass_dispersed_kg": round(float(final["mass_dispersed"].sum()), 2),
            "final_mass_biodegraded_kg": round(float(final["mass_biodegraded"].sum()), 2),
        },
        "final_state": {
            "moving_count": int((final["moving"] == 1).sum()),
            "stopped_count": int((final["moving"] == 0).sum()),
        },
        "hotspots": {
            "triggered_hotspots": triggered,
            "critical_hotspots": critical_hotspots,
        },
        "outputs": {
            "trajectory": f"openoil_full_physics_output/{case_id}/trajectory.parquet",
            "hotspot_alerts": f"openoil_full_physics_output/{case_id}/hotspot_alerts.parquet",
            "animation_mp4": f"openoil_full_physics_output/{case_id}/openoil_full_physics_{case_id.replace('-', '')}_animation.mp4",
            "representative_frames": [f"openoil_full_physics_output/{case_id}/frames/{t}.png" for t in representative_hours],
        },
        "notes": [
            "This is a SEPARATE exploratory deliverable using the real OpenOil/OpenDrift engine "
            "with weathering and diffusion enabled -- it does NOT replace or modify this "
            "project's own locked, from-scratch, deterministic model under output/.",
            "Non-deterministic by nature (diffusion + current/wind uncertainty); reproducible "
            "only because a fixed random_seed was passed to OpenOil.",
            "Per-parcel mass_kg heterogeneity is not individually preserved -- OpenOil has no "
            "concept of per-element user-supplied mass (confirmed from its own source: mass_oil "
            "is marked 'seed': False and unconditionally recomputed during seeding). The real "
            "TOTAL observed mass (sum of all parcels' mass_kg) was seeded correctly and verified "
            "numerically, then split equally across parcels and weathered realistically from there.",
        ],
    }
    summary_path = out_dir / "run_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {summary_path}")
    print(f"{case_id}: DONE")


def main() -> None:
    for case_id in ("ow-0009", "ow-0008"):
        build_case(case_id)
    section("ALL CASES COMPLETE")


if __name__ == "__main__":
    main()
