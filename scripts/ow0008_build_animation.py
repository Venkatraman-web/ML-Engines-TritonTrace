"""ow-0008 case -- build + validate the 3-day forward-propagation animation (F05
equivalent; not a test file -- CLAUDE.md forbids `tests/`/`test_*.py`).

Mirrors `scripts/f05_build_animation.py` (opening regional-overview hold + eased
zoom-in, drift trails, trajectory-zoomed main sequence) exactly, adapted for
ow-0008 via `fwd.cases`. Representative-hour selection is fully data-driven
(`pick_representative_hours`) rather than hard-coded to a specific hotspot/hour,
since which (if any) hotspots ow-0008 triggers is not assumed in advance.

Run: python scripts/ow0008_build_animation.py
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
from fwd.cases import get_case  # noqa: E402
from fwd.forcing import ForcingAdapter  # noqa: E402
from fwd.hotspots import CRITICAL, load_hotspots  # noqa: E402

CASE = get_case("ow-0008")
OUTPUT_DIR = REPO_ROOT / "output" / CASE.case_id
FRAMES_DIR = OUTPUT_DIR / "frames"
TMP_DIR = OUTPUT_DIR / "_anim_frames_tmp"
MP4_PATH = OUTPUT_DIR / f"{CASE.case_id.replace('-', '')}_forward_animation.mp4"

SUBSTEPS_PER_HOUR = 4
FPS = 10
DPI = 110

INTRO_HOLD_SECONDS = 3.0
INTRO_ZOOM_SECONDS = 1.5
INTRO_HOLD_FRAMES = round(INTRO_HOLD_SECONDS * FPS)
INTRO_ZOOM_FRAMES = round(INTRO_ZOOM_SECONDS * FPS)


def _slug(name: str) -> str:
    return name.lower().replace(" ", "_").replace("(", "").replace(")", "")


def pick_representative_hours(trajectory_df: pd.DataFrame, alerts_df: pd.DataFrame) -> dict[str, np.datetime64]:
    """Data-driven representative-hour selection (requirement: initial / a WATCH
    moment / a CRITICAL moment if one occurs / final), reading whatever this case's
    actual results are rather than assuming a specific hotspot triggers, unlike
    ow-0009's original hard-coded 'critical_cape_greco' tag. WATCH and CRITICAL rows
    are picked independently (not "first non-CLEAR row") so a hotspot that is already
    CRITICAL at the first non-CLEAR hour is never mislabelled as the WATCH frame."""
    times = sorted(trajectory_df["time"].unique())
    hours: dict[str, np.datetime64] = {"initial": times[0], "final": times[-1]}

    watch_rows = alerts_df[alerts_df["status"] == "WATCH"].sort_values("time")
    if len(watch_rows):
        w = watch_rows.iloc[0]
        hours[f"watch_{_slug(w['hotspot_name'])}"] = w["time"]

    critical_rows = alerts_df[alerts_df["status"] == CRITICAL].sort_values("time")
    if len(critical_rows):
        c = critical_rows.iloc[0]
        hours[f"critical_{_slug(c['hotspot_name'])}"] = c["time"]

    return hours


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def save_representative_frame(tag, t, trajectory_df, alerts_df, hotspots, land, view_bounds, paths, n_expected):
    positions = trajectory_df[trajectory_df["time"] == t]
    assert len(positions) == n_expected, f"{tag}: expected {n_expected} particles at {t}, found {len(positions)}"
    status = status_by_hotspot_at(alerts_df, t)
    trail_segments = build_trail_segments(paths, t, positions)
    fig, ax = plt.subplots(figsize=(10, 8), dpi=DPI)
    render_frame(ax, land, hotspots, status, positions, t, view_bounds=view_bounds,
                 trail_segments=trail_segments, case_label=CASE.case_id)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    path = FRAMES_DIR / f"{tag}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  {tag:26s} t={t}  active={int((~positions['stranded']).sum())} "
          f"stranded={int(positions['stranded'].sum())}  "
          f"statuses={ {n: s for n, s in status.items() if s != 'CLEAR'} }  -> {path}")
    return path


def main() -> None:
    section(f"Loading inputs for {CASE.case_id}")
    adapter = ForcingAdapter(ocean_path=str(CASE.ocean_path), wind_path=str(CASE.wind_path))
    hotspots = load_hotspots()
    trajectory_df = pd.read_parquet(OUTPUT_DIR / "trajectory.parquet")
    alerts_df = pd.read_parquet(OUTPUT_DIR / "hotspot_alerts.parquet")
    land = LandBackground(lon=adapter.ocean.lon, lat=adapter.ocean.lat, ocean_valid_mask=adapter.ocean_valid_mask)
    n = CASE.total_parcels
    print(f"trajectory: {len(trajectory_df)} rows, {trajectory_df['time'].nunique()} timestamps")
    print(f"hotspot alerts: {len(alerts_df)} rows")
    print(f"hotspots: {len(hotspots)}")

    representative_hours = pick_representative_hours(trajectory_df, alerts_df)
    print(f"representative hours selected: {representative_hours}")

    # Match ow-0009's own view span (computed dynamically from its own trajectory,
    # never hard-coded) so the two animations play at the same absolute zoom level
    # and are visually comparable side by side -- requested after ow-0008's
    # proportional-to-its-own-size zoom (compute_view_bounds) was judged too tight,
    # even though it correctly framed ow-0008's own (much smaller) trajectory.
    ow0009_traj = pd.read_parquet(REPO_ROOT / "output" / "ow-0009" / "trajectory.parquet")
    ow0009_bounds = compute_view_bounds(ow0009_traj)
    ref_lon_span = ow0009_bounds["east"] - ow0009_bounds["west"]
    ref_lat_span = ow0009_bounds["north"] - ow0009_bounds["south"]
    view_bounds = compute_view_bounds_with_span(trajectory_df, ref_lon_span, ref_lat_span)
    print(f"view bounds (matched to ow-0009's own {ref_lon_span:.3f} x {ref_lat_span:.3f} deg span, "
          f"centered on ow-0008's own trajectory): {view_bounds}")
    print(f"  vs. full CMEMS AOI width/height: "
          f"{36.088-27.121:.2f} x {36.372-30.1875:.2f} deg  ->  "
          f"zoomed view: {view_bounds['east']-view_bounds['west']:.2f} x {view_bounds['north']-view_bounds['south']:.2f} deg")

    paths = precompute_particle_paths(trajectory_df)

    section("1) Representative frames")
    for tag, t in representative_hours.items():
        assert (trajectory_df["time"] == t).any(), f"{tag}: {t} is not an actual recorded hourly state"
        save_representative_frame(tag, t, trajectory_df, alerts_df, hotspots, land, view_bounds, paths, n)

    section("2) Building smooth-playback position interpolation (animation-only)")
    t0 = time.time()
    pos_interp = interpolate_positions(trajectory_df, substeps_per_hour=SUBSTEPS_PER_HOUR)
    n_hours = trajectory_df["time"].nunique()
    expected_frames = (n_hours - 1) * SUBSTEPS_PER_HOUR + 1
    n_frames = pos_interp["time"].nunique()
    print(f"interpolated {n_frames} frame-times from {n_hours} recorded hours "
          f"({SUBSTEPS_PER_HOUR} substeps/hour) in {time.time()-t0:.2f}s")
    assert n_frames == expected_frames, f"expected {expected_frames} frame-times, got {n_frames}"
    assert len(pos_interp) == n_frames * n

    section("3a) Opening sequence: regional hotspot overview, then eased zoom to the slick")
    grid_extent = {
        "west": float(land.lon.min()), "east": float(land.lon.max()),
        "south": float(land.lat.min()), "north": float(land.lat.max()),
    }
    overview_bounds = compute_hotspot_overview_bounds(hotspots, clamp=grid_extent)
    print(f"CMEMS grid extent (clamp): {grid_extent}")
    print(f"overview bounds (all 11 hotspots' watch rings + padding, clamped to the grid): {overview_bounds}")
    t0v = trajectory_df["time"].min()
    t0_positions = trajectory_df[trajectory_df["time"] == t0v]
    t0_status = status_by_hotspot_at(alerts_df, t0v)
    t0_trail = build_trail_segments(paths, t0v, t0_positions)
    print(f"intro hold: {INTRO_HOLD_FRAMES} frames ({INTRO_HOLD_SECONDS}s)  "
          f"zoom-in: {INTRO_ZOOM_FRAMES} frames ({INTRO_ZOOM_SECONDS}s), both frozen at t0={t0v}")

    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR)
    TMP_DIR.mkdir(parents=True)
    fig, ax = plt.subplots(figsize=(10, 8), dpi=DPI)
    frame_idx = 0
    t_start = time.time()

    for _ in range(INTRO_HOLD_FRAMES):
        ax.clear()
        render_frame(
            ax, land, hotspots, t0_status, t0_positions, t0v, view_bounds=overview_bounds,
            trail_segments=t0_trail, overview_label="Regional hotspot overview -- 11 monitored zones",
            case_label=CASE.case_id,
        )
        fig.savefig(TMP_DIR / f"frame_{frame_idx:05d}.png", bbox_inches="tight")
        frame_idx += 1

    for i in range(INTRO_ZOOM_FRAMES):
        frac = (i + 1) / INTRO_ZOOM_FRAMES
        zoom_bounds = interpolate_view_bounds(overview_bounds, view_bounds, frac)
        ax.clear()
        render_frame(
            ax, land, hotspots, t0_status, t0_positions, t0v, view_bounds=zoom_bounds,
            trail_segments=t0_trail, overview_label=f"Zooming to the {CASE.case_id} slick...",
            case_label=CASE.case_id,
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
        render_frame(ax, land, hotspots, status, frame_pos, t, view_bounds=view_bounds,
                     trail_segments=trail_segments, case_label=CASE.case_id)
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
    print(f"{CASE.case_id} animation: COMPLETE")


if __name__ == "__main__":
    main()
