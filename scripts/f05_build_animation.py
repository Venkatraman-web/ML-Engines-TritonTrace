"""F05 -- build + validate the ow-0009 3-day forward-propagation animation (not a test
file -- CLAUDE.md forbids `tests/`/`test_*.py`).

Renders `output/ow-0009/trajectory.parquet` + `output/ow-0009/hotspot_alerts.parquet`
+ `regional_alert_geofences_wide.geojson` into an MP4 (via ffmpeg) plus four
representative PNG frames (initial / a WATCH frame / the Cape Greco CRITICAL frame /
final), using `fwd.animate`. No new scientific states or alert events are created --
only particle position is smoothed for playback (see `fwd.animate.interpolate_
positions`'s docstring).

The MP4 opens with a 3s regional-overview hold (all 11 hotspots visible, frozen at
t0) then a 1.5s eased zoom-in to the trajectory-focused view used for the rest of the
video (`INTRO_HOLD_SECONDS`/`INTRO_ZOOM_SECONDS`) -- both purely presentational; no
scientific state changes during the hold/zoom, the propagation clock does not advance
until the zoom finishes.

Run: python scripts/f05_build_animation.py
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
    interpolate_positions, interpolate_view_bounds, precompute_particle_paths, render_frame,
    status_by_hotspot_at,
)
from fwd.forcing import ForcingAdapter  # noqa: E402
from fwd.hotspots import load_hotspots  # noqa: E402

OUTPUT_DIR = REPO_ROOT / "output" / "ow-0009"
FRAMES_DIR = OUTPUT_DIR / "frames"
TMP_DIR = OUTPUT_DIR / "_anim_frames_tmp"
MP4_PATH = OUTPUT_DIR / "ow0009_forward_animation.mp4"

SUBSTEPS_PER_HOUR = 4
FPS = 10
DPI = 110

# Opening "regional overview -> zoom to slick" sequence, requested so a viewer sees
# where all 11 hotspots sit before the camera settles on the trajectory-zoomed view
# used for the rest of the video.
INTRO_HOLD_SECONDS = 3.0
INTRO_ZOOM_SECONDS = 1.5
INTRO_HOLD_FRAMES = round(INTRO_HOLD_SECONDS * FPS)
INTRO_ZOOM_FRAMES = round(INTRO_ZOOM_SECONDS * FPS)

FIXED_REPRESENTATIVE_HOURS = {
    "initial": np.datetime64("2019-04-28T15:38:41", "ns"),
    "watch": np.datetime64("2019-04-29T12:38:41", "ns"),
    "final": np.datetime64("2019-05-01T15:38:41", "ns"),
}


def cape_greco_first_critical_hour(alerts_df: pd.DataFrame) -> np.datetime64:
    """The actual first hour Cape Greco MPA reaches CRITICAL, read from
    `hotspot_alerts.parquet` rather than hard-coded -- this depends on
    `HotspotThresholds.min_fraction_critical` (currently 30%, a user-tunable
    parameter, see fwd.hotspots), so hard-coding a specific timestamp here would go
    stale the next time that threshold is adjusted."""
    cg = alerts_df[alerts_df["hotspot_name"] == "Cape Greco MPA"].sort_values("time")
    critical_rows = cg[cg["status"] == "CRITICAL"]
    assert len(critical_rows) > 0, "Cape Greco MPA never reaches CRITICAL in this run"
    return critical_rows["time"].iloc[0]


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def save_representative_frame(tag, t, trajectory_df, alerts_df, hotspots, land, view_bounds, paths):
    positions = trajectory_df[trajectory_df["time"] == t]
    assert len(positions) == 371, f"{tag}: expected 371 particles at {t}, found {len(positions)}"
    status = status_by_hotspot_at(alerts_df, t)
    trail_segments = build_trail_segments(paths, t, positions)
    fig, ax = plt.subplots(figsize=(10, 8), dpi=DPI)
    render_frame(ax, land, hotspots, status, positions, t, view_bounds=view_bounds, trail_segments=trail_segments)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    path = FRAMES_DIR / f"{tag}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  {tag:22s} t={t}  active={int((~positions['stranded']).sum())} "
          f"stranded={int(positions['stranded'].sum())}  "
          f"statuses={ {n: s for n, s in status.items() if s != 'CLEAR'} }  -> {path}")
    return path


def main() -> None:
    section("Loading inputs")
    adapter = ForcingAdapter()
    hotspots = load_hotspots()
    trajectory_df = pd.read_parquet(OUTPUT_DIR / "trajectory.parquet")
    alerts_df = pd.read_parquet(OUTPUT_DIR / "hotspot_alerts.parquet")
    land = LandBackground(lon=adapter.ocean.lon, lat=adapter.ocean.lat, ocean_valid_mask=adapter.ocean_valid_mask)
    print(f"trajectory: {len(trajectory_df)} rows, {trajectory_df['time'].nunique()} timestamps")
    print(f"hotspot alerts: {len(alerts_df)} rows")
    print(f"hotspots: {len(hotspots)}")

    representative_hours = dict(FIXED_REPRESENTATIVE_HOURS)
    representative_hours["critical_cape_greco"] = cape_greco_first_critical_hour(alerts_df)
    print(f"Cape Greco MPA first CRITICAL hour (recomputed from hotspot_alerts.parquet): "
          f"{representative_hours['critical_cape_greco']}")

    view_bounds = compute_view_bounds(trajectory_df)
    print(f"view bounds (zoomed to the full 72h trajectory + padding): {view_bounds}")
    print(f"  vs. full CMEMS AOI width/height: "
          f"{36.088-27.121:.2f} x {36.372-30.1875:.2f} deg  ->  "
          f"zoomed view: {view_bounds['east']-view_bounds['west']:.2f} x {view_bounds['north']-view_bounds['south']:.2f} deg")

    paths = precompute_particle_paths(trajectory_df)

    section("1) Representative frames (initial / WATCH / Cape Greco CRITICAL / final)")
    for tag, t in representative_hours.items():
        assert (trajectory_df["time"] == t).any(), f"{tag}: {t} is not an actual recorded hourly state"
        save_representative_frame(tag, t, trajectory_df, alerts_df, hotspots, land, view_bounds, paths)

    section("2) Building smooth-playback position interpolation (animation-only)")
    t0 = time.time()
    pos_interp = interpolate_positions(trajectory_df, substeps_per_hour=SUBSTEPS_PER_HOUR)
    n_hours = trajectory_df["time"].nunique()
    expected_frames = (n_hours - 1) * SUBSTEPS_PER_HOUR + 1
    n_frames = pos_interp["time"].nunique()
    print(f"interpolated {n_frames} frame-times from {n_hours} recorded hours "
          f"({SUBSTEPS_PER_HOUR} substeps/hour) in {time.time()-t0:.2f}s")
    assert n_frames == expected_frames, f"expected {expected_frames} frame-times, got {n_frames}"
    assert len(pos_interp) == n_frames * 371

    section("3a) Opening sequence: regional hotspot overview, then eased zoom to the slick")
    # clamp to the CMEMS grid's actual coverage -- some hotspot rings (Ceyhan Oil
    # Terminal's, in particular) extend past it, and showing that area renders as
    # blank white (no land/water background data there), which was reported as a bug
    grid_extent = {
        "west": float(land.lon.min()), "east": float(land.lon.max()),
        "south": float(land.lat.min()), "north": float(land.lat.max()),
    }
    overview_bounds = compute_hotspot_overview_bounds(hotspots, clamp=grid_extent)
    print(f"CMEMS grid extent (clamp): {grid_extent}")
    print(f"overview bounds (all 11 hotspots' watch rings + padding, clamped to the grid): {overview_bounds}")
    t0 = trajectory_df["time"].min()
    t0_positions = trajectory_df[trajectory_df["time"] == t0]
    t0_status = status_by_hotspot_at(alerts_df, t0)
    t0_trail = build_trail_segments(paths, t0, t0_positions)  # empty at t0 -- no history yet
    print(f"intro hold: {INTRO_HOLD_FRAMES} frames ({INTRO_HOLD_SECONDS}s)  "
          f"zoom-in: {INTRO_ZOOM_FRAMES} frames ({INTRO_ZOOM_SECONDS}s), both frozen at t0={t0}")

    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR)
    TMP_DIR.mkdir(parents=True)
    fig, ax = plt.subplots(figsize=(10, 8), dpi=DPI)
    frame_idx = 0
    t_start = time.time()

    for _ in range(INTRO_HOLD_FRAMES):
        ax.clear()
        render_frame(
            ax, land, hotspots, t0_status, t0_positions, t0, view_bounds=overview_bounds,
            trail_segments=t0_trail, overview_label="Regional hotspot overview -- 11 monitored zones",
        )
        fig.savefig(TMP_DIR / f"frame_{frame_idx:05d}.png", bbox_inches="tight")
        frame_idx += 1

    for i in range(INTRO_ZOOM_FRAMES):
        frac = (i + 1) / INTRO_ZOOM_FRAMES
        zoom_bounds = interpolate_view_bounds(overview_bounds, view_bounds, frac)
        ax.clear()
        render_frame(
            ax, land, hotspots, t0_status, t0_positions, t0, view_bounds=zoom_bounds,
            trail_segments=t0_trail, overview_label="Zooming to the ow-0009 slick...",
        )
        fig.savefig(TMP_DIR / f"frame_{frame_idx:05d}.png", bbox_inches="tight")
        frame_idx += 1
    print(f"opening sequence: {frame_idx} frames rendered in {time.time()-t_start:.1f}s")

    section("3b) Rendering the 72h propagation frame sequence")
    frame_times = sorted(pos_interp["time"].unique())
    t_start = time.time()
    for t in frame_times:
        ax.clear()
        frame_pos = pos_interp[pos_interp["time"] == t]
        floor_t = frame_pos["floor_time"].iloc[0]
        status = status_by_hotspot_at(alerts_df, floor_t)
        trail_segments = build_trail_segments(paths, floor_t, frame_pos)
        render_frame(ax, land, hotspots, status, frame_pos, t, view_bounds=view_bounds, trail_segments=trail_segments)
        fig.savefig(TMP_DIR / f"frame_{frame_idx:05d}.png", bbox_inches="tight")
        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"  frame {frame_idx}  t={t}  elapsed={time.time()-t_start:.1f}s")
    plt.close(fig)
    total_frames = frame_idx
    print(f"rendered {len(frame_times)} propagation frames in {time.time()-t_start:.1f}s "
          f"({total_frames} total frames including the opening sequence)")

    section("4) Encoding MP4 with ffmpeg")
    if MP4_PATH.exists():
        MP4_PATH.unlink()
    cmd = [
        "ffmpeg", "-y", "-framerate", str(FPS),
        "-i", str(TMP_DIR / "frame_%05d.png"),
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-pix_fmt", "yuv420p",
        str(MP4_PATH),
    ]
    print("  " + " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout[-2000:])
        print(result.stderr[-2000:])
        raise RuntimeError("ffmpeg encoding failed")
    assert MP4_PATH.exists() and MP4_PATH.stat().st_size > 0, "mp4 output missing or empty"
    print(f"wrote {MP4_PATH}  ({MP4_PATH.stat().st_size / 1e6:.2f} MB, "
          f"{total_frames} frames @ {FPS} fps = {total_frames/FPS:.1f}s video total "
          f"-- {(INTRO_HOLD_FRAMES+INTRO_ZOOM_FRAMES)/FPS:.1f}s opening sequence "
          f"+ {len(frame_times)/FPS:.1f}s covering the 72h propagation)")

    shutil.rmtree(TMP_DIR)
    print(f"removed temporary frame directory {TMP_DIR}")

    section("ALL CHECKS PASSED")
    print("F05 animation: COMPLETE")


if __name__ == "__main__":
    main()
