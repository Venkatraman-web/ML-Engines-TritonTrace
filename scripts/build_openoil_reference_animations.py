"""Build MP4 animations (+ representative frames) of the real OpenOil/OpenDrift
reference trajectories (F06's `openoil_reference.parquet`) for both ow-0009 and
ow-0008, stored **separately** from this project's own results
(`openoil_reference_animations/<case_id>/`, a top-level sibling of `output/`) so
there is no ambiguity about which animation was produced by which implementation
(not a test file -- CLAUDE.md forbids `tests/`/`test_*.py`).

Reuses the exact same rendering machinery (`fwd.animate`) as this project's own
animations -- same basemap, hotspot-zone styling, drift trails, legend, opening
overview/zoom sequence -- so the two are visually comparable side by side. Hotspot
alert status shown here is **recomputed** against OpenDrift's own reference
positions via `fwd.hotspots.evaluate_hotspot_alerts` (not copied from this project's
own `hotspot_alerts.parquet`), since colouring OpenDrift's positions with our own
alert decisions would be inconsistent. Both `openoil_reference.parquet` files were
verified beforehand to have `status == 0` (active) for every row in both cases (no
deactivations), so every particle is treated as active/not-stranded throughout.

Run: python scripts/build_openoil_reference_animations.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from fwd.animate import (  # noqa: E402
    LandBackground, build_trail_segments, compute_hotspot_overview_bounds, compute_view_bounds,
    compute_view_bounds_with_span, interpolate_positions, interpolate_view_bounds,
    precompute_particle_paths, render_frame, status_by_hotspot_at,
)
from fwd.cases import CASES  # noqa: E402
from fwd.forcing import ForcingAdapter  # noqa: E402
from fwd.hotspots import CRITICAL, WATCH, evaluate_hotspot_alerts, load_hotspots  # noqa: E402

OUTPUT_ROOT = REPO_ROOT / "openoil_reference_animations"
SUBSTEPS_PER_HOUR = 4
FPS = 10
DPI = 110
INTRO_HOLD_SECONDS = 3.0
INTRO_ZOOM_SECONDS = 1.5
INTRO_HOLD_FRAMES = round(INTRO_HOLD_SECONDS * FPS)
INTRO_ZOOM_FRAMES = round(INTRO_ZOOM_SECONDS * FPS)


def _slug(name: str) -> str:
    return name.lower().replace(" ", "_").replace("(", "").replace(")", "")


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def pick_representative_hours(trajectory_df: pd.DataFrame, alerts_df: pd.DataFrame) -> dict[str, np.datetime64]:
    times = sorted(trajectory_df["time"].unique())
    hours: dict[str, np.datetime64] = {"initial": times[0], "final": times[-1]}
    watch_rows = alerts_df[alerts_df["status"] == WATCH].sort_values("time")
    if len(watch_rows):
        w = watch_rows.iloc[0]
        hours[f"watch_{_slug(w['hotspot_name'])}"] = w["time"]
    critical_rows = alerts_df[alerts_df["status"] == CRITICAL].sort_values("time")
    if len(critical_rows):
        c = critical_rows.iloc[0]
        hours[f"critical_{_slug(c['hotspot_name'])}"] = c["time"]
    return hours


def build_case(case_id: str, reference_span: tuple[float, float] | None = None) -> None:
    """`reference_span`, if given, is `(lon_span, lat_span)` in degrees to center on
    this case's own trajectory (`compute_view_bounds_with_span`) instead of using
    `compute_view_bounds`'s own proportional-to-this-trajectory's-size padding --
    used to match ow-0008's view to ow-0009's own span so both OpenOil reference
    animations play at the same absolute zoom level as each other (and as this
    project's own two animations, which are matched the same way)."""
    case = CASES[case_id]
    section(f"{case_id}: loading OpenOil reference trajectory")
    out_dir = OUTPUT_ROOT / case_id
    frames_dir = out_dir / "frames"
    tmp_dir = out_dir / "_anim_frames_tmp"
    mp4_path = out_dir / f"openoil_{case_id.replace('-', '')}_reference_animation.mp4"

    ref = pd.read_parquet(REPO_ROOT / "output" / case_id / "openoil_reference.parquet")
    assert (ref["status"] == 0).all(), f"{case_id}: expected all OpenDrift statuses == 0 (active); check before reusing this script"
    traj = ref[["particle_id", "time", "lon", "lat"]].copy()
    traj["valid"] = True
    traj["stranded"] = False
    n = case.total_parcels
    print(f"reference trajectory: {len(traj)} rows, {traj['time'].nunique()} timestamps, "
          f"{traj['particle_id'].nunique()} particles (expected {n})")
    assert traj["particle_id"].nunique() == n
    assert traj["time"].nunique() == 73

    adapter = ForcingAdapter(ocean_path=str(case.ocean_path), wind_path=str(case.wind_path))
    hotspots = load_hotspots()
    land = LandBackground(lon=adapter.ocean.lon, lat=adapter.ocean.lat, ocean_valid_mask=adapter.ocean_valid_mask)

    section(f"{case_id}: recomputing hotspot alerts against OpenDrift's own positions")
    alerts_df = evaluate_hotspot_alerts(traj, hotspots)
    triggered = sorted(alerts_df.loc[alerts_df["status"] != "CLEAR", "hotspot_name"].unique())
    print(f"hotspots ever WATCH/CRITICAL against the OpenOil reference trajectory: {triggered}")

    representative_hours = pick_representative_hours(traj, alerts_df)
    print(f"representative hours: {representative_hours}")

    if reference_span is not None:
        view_bounds = compute_view_bounds_with_span(traj, *reference_span)
    else:
        view_bounds = compute_view_bounds(traj)
    print(f"view bounds: {view_bounds}  "
          f"({view_bounds['east']-view_bounds['west']:.3f} x {view_bounds['north']-view_bounds['south']:.3f} deg)"
          f"{'  [matched to reference span]' if reference_span else ''}")

    paths = precompute_particle_paths(traj)
    case_label = f"{case_id} (OpenOil reference)"

    section(f"{case_id}: representative frames")
    frames_dir.mkdir(parents=True, exist_ok=True)
    for tag, t in representative_hours.items():
        positions = traj[traj["time"] == t]
        assert len(positions) == n, f"{tag}: expected {n} particles at {t}, found {len(positions)}"
        status = status_by_hotspot_at(alerts_df, t)
        trail_segments = build_trail_segments(paths, t, positions)
        fig, ax = plt.subplots(figsize=(10, 8), dpi=DPI)
        render_frame(ax, land, hotspots, status, positions, t, view_bounds=view_bounds,
                     trail_segments=trail_segments, case_label=case_label)
        path = frames_dir / f"{tag}.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"  {tag:26s} t={t}  -> {path}")

    section(f"{case_id}: building smooth-playback position interpolation")
    t0 = time.time()
    pos_interp = interpolate_positions(traj, substeps_per_hour=SUBSTEPS_PER_HOUR)
    n_hours = traj["time"].nunique()
    expected_frames = (n_hours - 1) * SUBSTEPS_PER_HOUR + 1
    n_frames = pos_interp["time"].nunique()
    print(f"interpolated {n_frames} frame-times in {time.time()-t0:.2f}s")
    assert n_frames == expected_frames

    section(f"{case_id}: opening sequence (regional overview -> eased zoom)")
    grid_extent = {
        "west": float(land.lon.min()), "east": float(land.lon.max()),
        "south": float(land.lat.min()), "north": float(land.lat.max()),
    }
    overview_bounds = compute_hotspot_overview_bounds(hotspots, clamp=grid_extent)
    t0v = traj["time"].min()
    t0_positions = traj[traj["time"] == t0v]
    t0_status = status_by_hotspot_at(alerts_df, t0v)
    t0_trail = build_trail_segments(paths, t0v, t0_positions)

    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)
    fig, ax = plt.subplots(figsize=(10, 8), dpi=DPI)
    frame_idx = 0
    t_start = time.time()

    for _ in range(INTRO_HOLD_FRAMES):
        ax.clear()
        render_frame(ax, land, hotspots, t0_status, t0_positions, t0v, view_bounds=overview_bounds,
                     trail_segments=t0_trail, overview_label="Regional hotspot overview -- 11 monitored zones",
                     case_label=case_label)
        fig.savefig(tmp_dir / f"frame_{frame_idx:05d}.png", bbox_inches="tight")
        frame_idx += 1

    for i in range(INTRO_ZOOM_FRAMES):
        frac = (i + 1) / INTRO_ZOOM_FRAMES
        zoom_bounds = interpolate_view_bounds(overview_bounds, view_bounds, frac)
        ax.clear()
        render_frame(ax, land, hotspots, t0_status, t0_positions, t0v, view_bounds=zoom_bounds,
                     trail_segments=t0_trail, overview_label=f"Zooming to the {case_id} OpenOil reference...",
                     case_label=case_label)
        fig.savefig(tmp_dir / f"frame_{frame_idx:05d}.png", bbox_inches="tight")
        frame_idx += 1
    print(f"opening sequence: {frame_idx} frames in {time.time()-t_start:.1f}s")

    section(f"{case_id}: rendering the 72h propagation frame sequence")
    frame_times = sorted(pos_interp["time"].unique())
    t_start = time.time()
    for t in frame_times:
        ax.clear()
        frame_pos = pos_interp[pos_interp["time"] == t]
        floor_t = frame_pos["floor_time"].iloc[0]
        status = status_by_hotspot_at(alerts_df, floor_t)
        trail_segments = build_trail_segments(paths, floor_t, frame_pos)
        render_frame(ax, land, hotspots, status, frame_pos, t, view_bounds=view_bounds,
                     trail_segments=trail_segments, case_label=case_label)
        fig.savefig(tmp_dir / f"frame_{frame_idx:05d}.png", bbox_inches="tight")
        frame_idx += 1
    plt.close(fig)
    total_frames = frame_idx
    print(f"rendered {len(frame_times)} propagation frames in {time.time()-t_start:.1f}s "
          f"({total_frames} total)")

    section(f"{case_id}: encoding MP4")
    if mp4_path.exists():
        mp4_path.unlink()
    cmd = [
        "ffmpeg", "-y", "-framerate", str(FPS),
        "-i", str(tmp_dir / "frame_%05d.png"),
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-pix_fmt", "yuv420p",
        str(mp4_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout[-2000:])
        print(result.stderr[-2000:])
        raise RuntimeError("ffmpeg encoding failed")
    assert mp4_path.exists() and mp4_path.stat().st_size > 0
    print(f"wrote {mp4_path}  ({mp4_path.stat().st_size/1e6:.2f} MB, {total_frames} frames @ {FPS} fps "
          f"= {total_frames/FPS:.1f}s)")

    shutil.rmtree(tmp_dir)
    print(f"{case_id}: DONE")


def main() -> None:
    # ow-0009's own OpenOil-reference view span (computed dynamically, never
    # hard-coded) is used as the common zoom level for both cases' animations, so
    # ow-0008 (whose own trajectory spans a much smaller area) doesn't end up
    # displayed at a visually mismatched, much-tighter zoom than ow-0009.
    ow0009_ref_traj = pd.read_parquet(REPO_ROOT / "output" / "ow-0009" / "openoil_reference.parquet")
    ow0009_bounds = compute_view_bounds(ow0009_ref_traj)
    reference_span = (ow0009_bounds["east"] - ow0009_bounds["west"], ow0009_bounds["north"] - ow0009_bounds["south"])
    print(f"common reference span (from ow-0009's own OpenOil reference trajectory): "
          f"{reference_span[0]:.3f} x {reference_span[1]:.3f} deg")

    build_case("ow-0009")
    build_case("ow-0008", reference_span=reference_span)
    section("ALL CASES COMPLETE")


if __name__ == "__main__":
    main()
