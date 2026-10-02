"""F05 -- 3-day forward-propagation animation for the ow-0009 pilot.

Renders the F03 trajectory (371 particles x 73 hourly states) and F04 hotspot alert
timeline as a video, reusing:
- F01's static CMEMS land/sea mask (`ForcingAdapter.ocean_valid_mask`) as the
  coastline/land background -- the only land data source available in this project
  (no coastline shapefile exists here, per the F00 audit);
- F04's actual hotspot polygons (`fwd.hotspots.load_hotspots`) for the zone geometries
  (never reconstructed from `radius_km`);
- F04's hotspot alert timeline (`hotspot_alerts.parquet`) for per-hour WATCH/CRITICAL/
  CLEAR status.

Only particle POSITION is interpolated for smooth sub-hour playback (linear
interpolation between the two bracketing real hourly states, `interpolate_positions`).
Hotspot alert status is never interpolated: every rendered frame's alert status is
exactly the real hourly value at the last recorded hour at or before that frame's
clock time (requirement 6 -- "do not create new scientific states or alert events").
No cartopy/coastline library is available in this environment (verified in F00); the
map uses a plain equirectangular lon/lat axis with a latitude-corrected aspect ratio
(`1/cos(mean_latitude)`) to stay geographically undistorted for this AOI's scale
(requirement 7), rather than a full map projection library.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon as MplPolygon

from fwd.hotspots import CLEAR, CRITICAL, WATCH, Hotspot

# Full CMEMS forcing domain -- kept only as a documented fallback/reference extent.
# The animation itself uses `compute_view_bounds` (below), which zooms to the actual
# trajectory instead: at this AOI's ~950 km width, a slick that only ever spans a few
# tens of km was reduced to a handful of pixels (reported as "very zoomed out" and
# fixed in this task).
FULL_AOI = dict(north=36.372, west=27.121, south=30.1875, east=36.088)

COLOR_WATER = "#dbeeff"
COLOR_LAND = "#ded9c9"
COLOR_COASTLINE = "#4a4a4a"
COLOR_ACTIVE = "#0b3d91"
COLOR_STRANDED = "#7a4b00"
COLOR_WATCH_FILL = "#ffd60a"
COLOR_CRITICAL_FILL = "#d7263d"
COLOR_ZONE_OUTLINE = "#777777"


@dataclass(frozen=True)
class LandBackground:
    lon: np.ndarray  # 1-D, CMEMS grid longitudes
    lat: np.ndarray  # 1-D, CMEMS grid latitudes
    ocean_valid_mask: np.ndarray  # 2-D (lat, lon), True = water, from F01


def interpolate_positions(trajectory_df: pd.DataFrame, substeps_per_hour: int) -> pd.DataFrame:
    """Animation-only sub-hour position interpolation between existing recorded hourly
    states (requirement 6). `valid`/`stranded` are carried from the floor hour
    (categorical state -- never interpolated); only `lon`/`lat` are linearly blended.
    A stranded particle's two bracketing hourly positions are identical (F02: a
    stranded particle never moves again), so this naturally produces zero apparent
    motion for it without any special-casing.

    Returns one row per particle per rendered sub-hour instant, with columns
    `particle_id, time, floor_time, lon, lat, valid, stranded`. `floor_time` is the
    real recorded hour to use for looking up hotspot alert status for that frame
    (never interpolated).
    """
    times = sorted(trajectory_df["time"].unique())
    frames = []
    for k in range(len(times) - 1):
        t0, t1 = times[k], times[k + 1]
        g0 = trajectory_df[trajectory_df["time"] == t0].set_index("particle_id").sort_index()
        g1 = trajectory_df[trajectory_df["time"] == t1].set_index("particle_id").sort_index()
        assert list(g0.index) == list(g1.index), "particle IDs differ between consecutive hourly states"
        dt_ns = int((t1 - t0) / np.timedelta64(1, "ns"))
        for s in range(substeps_per_hour):
            frac = s / substeps_per_hour
            t = t0 + np.timedelta64(int(round(dt_ns * frac)), "ns")
            lon = g0["lon"].to_numpy() * (1 - frac) + g1["lon"].to_numpy() * frac
            lat = g0["lat"].to_numpy() * (1 - frac) + g1["lat"].to_numpy() * frac
            frames.append(pd.DataFrame({
                "particle_id": g0.index.to_numpy(), "time": t, "floor_time": t0,
                "lon": lon, "lat": lat,
                "valid": g0["valid"].to_numpy(), "stranded": g0["stranded"].to_numpy(),
            }))
    g_last = trajectory_df[trajectory_df["time"] == times[-1]].sort_values("particle_id")
    frames.append(pd.DataFrame({
        "particle_id": g_last["particle_id"].to_numpy(), "time": times[-1], "floor_time": times[-1],
        "lon": g_last["lon"].to_numpy(), "lat": g_last["lat"].to_numpy(),
        "valid": g_last["valid"].to_numpy(), "stranded": g_last["stranded"].to_numpy(),
    }))
    return pd.concat(frames, ignore_index=True)


def precompute_particle_paths(trajectory_df: pd.DataFrame) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """One-time (not per-frame) precompute of each particle's own real hourly path
    (`time, lon, lat`, sorted by time) as plain arrays, so `build_trail_segments` can
    slice each particle's history-so-far with `np.searchsorted` per frame instead of
    re-filtering/re-grouping the full 27k-row trajectory table 289 times (once per
    rendered frame) -- a real correctness-neutral perf fix, not a physics change."""
    paths: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for pid, g in trajectory_df.sort_values("time").groupby("particle_id"):
        paths[int(pid)] = (g["time"].to_numpy(), g["lon"].to_numpy(), g["lat"].to_numpy())
    return paths


def build_trail_segments(
    paths: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
    up_to_time: np.datetime64,
    current_positions: pd.DataFrame,
) -> list[np.ndarray]:
    """Per-particle path from t0 through `up_to_time` (inclusive), using the real
    recorded hourly positions (never re-simulated or altered -- this only draws
    already-computed F03 positions), with each particle's current (possibly
    sub-hour-interpolated) position appended as the trail's leading tip so the trail
    always lines up exactly with that particle's plotted dot. This is what makes each
    particle's actual drift path visible frame to frame, rather than only ever
    showing its instantaneous position with no visual history."""
    up_to_time = np.datetime64(up_to_time, "ns")
    cur = current_positions.set_index("particle_id")
    segments = []
    for pid, (times, lons, lats) in paths.items():
        k = int(np.searchsorted(times, up_to_time, side="right"))
        seg_lon, seg_lat = lons[:k], lats[:k]
        if pid in cur.index:
            seg_lon = np.append(seg_lon, float(cur.loc[pid, "lon"]))
            seg_lat = np.append(seg_lat, float(cur.loc[pid, "lat"]))
        if len(seg_lon) >= 2:
            segments.append(np.column_stack([seg_lon, seg_lat]))
    return segments


def clip_view_bounds(bounds: dict, clamp: dict) -> dict:
    """Intersect `bounds` with `clamp` (both `west/east/south/north` dicts), so a view
    never extends past `clamp`'s extent. Used to keep the opening regional-overview
    shot from showing area outside the CMEMS grid's actual coverage, which has no
    land/water background data and renders as blank white (reported as a bug and
    fixed here) rather than silently drawing nothing there."""
    return {
        "west": max(bounds["west"], clamp["west"]),
        "east": min(bounds["east"], clamp["east"]),
        "south": max(bounds["south"], clamp["south"]),
        "north": min(bounds["north"], clamp["north"]),
    }


def compute_hotspot_overview_bounds(
    hotspots: list[Hotspot], padding_fraction: float = 0.08, clamp: dict | None = None
) -> dict:
    """One fixed map view that bounds the full 100 km watch ring of *every* one of the
    11 hotspots, with a small padding margin -- used for the animation's opening
    "regional overview" shot (see `ease_in_out`/the intro-then-zoom sequence built in
    `scripts/f05_build_animation.py`), so a viewer sees where every hotspot sits
    before the camera zooms in on the slick itself.

    If `clamp` is given (e.g. the CMEMS grid's actual `lon`/`lat` extent), the result
    is clipped to it (`clip_view_bounds`) so the overview never shows area with no
    land/water background data -- some hotspots (Ceyhan Oil Terminal's ring, in
    particular) extend beyond the CMEMS AOI, and showing that blank area was reported
    as a visual bug: better to zoom no wider than there is actual basemap data for,
    even if that means a hotspot's ring is only partially shown, than to show blank
    white space."""
    lon_min = min(float(h.watch_ring[:, 0].min()) for h in hotspots)
    lon_max = max(float(h.watch_ring[:, 0].max()) for h in hotspots)
    lat_min = min(float(h.watch_ring[:, 1].min()) for h in hotspots)
    lat_max = max(float(h.watch_ring[:, 1].max()) for h in hotspots)
    lon_pad = (lon_max - lon_min) * padding_fraction
    lat_pad = (lat_max - lat_min) * padding_fraction
    bounds = {
        "west": lon_min - lon_pad, "east": lon_max + lon_pad,
        "south": lat_min - lat_pad, "north": lat_max + lat_pad,
    }
    return clip_view_bounds(bounds, clamp) if clamp is not None else bounds


def ease_in_out(t: float) -> float:
    """Cubic smoothstep easing (`3t^2 - 2t^3`), `t` clamped to `[0, 1]`. Used to
    interpolate the map view smoothly between two extents (`interpolate_view_bounds`)
    rather than linearly, so a camera zoom accelerates in and decelerates out instead
    of moving at a constant, mechanical-looking rate."""
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def interpolate_view_bounds(wide: dict, tight: dict, frac: float) -> dict:
    """Ease-interpolate a map view between `wide` (`frac=0`) and `tight` (`frac=1`),
    both `west/east/south/north` dicts of the same shape as `compute_view_bounds`/
    `compute_hotspot_overview_bounds` return."""
    e = ease_in_out(frac)
    return {k: wide[k] + (tight[k] - wide[k]) * e for k in ("west", "east", "south", "north")}


def compute_view_bounds(
    trajectory_df: pd.DataFrame,
    padding_fraction: float = 0.4,
    min_padding_deg: float = 0.03,
) -> dict:
    """Compute one fixed map view (`west/east/south/north`) that tightly bounds the
    *entire* 72h trajectory (every particle, every recorded hour -- not just the
    current frame, so the map does not zoom/pan during playback), with proportional
    padding for surrounding visual context.

    This replaces using the full CMEMS AOI (`FULL_AOI`) as the map extent, which made
    a slick spanning only a few tens of km nearly invisible against a ~950 km-wide
    domain. Deliberately does NOT force any hotspot ring fully into view: an earlier
    version of this function unioned in the full 40 km critical ring of any hotspot
    that reached CRITICAL, but for a hotspot whose ring extends mostly over land (as
    Cape Greco MPA's does here, into Cyprus), that pulled the view away from the slick
    itself and let land dominate the frame -- the opposite of what "clearly show the
    propagation and slick" means. Hotspot zones instead render however much of their
    ring falls inside this trajectory-centered window (the same clipping behavior
    already used for any hotspot whose full circle doesn't fit,
    `render_frame`'s label `clip_on`) -- enough to see a slick entering/leaving a zone,
    without letting a distant zone's geometry dictate the whole map's scale.

    `min_padding_deg` is only a floor for degenerate near-zero-extent trajectories --
    it must stay small. It was originally 0.15 deg, sized against ow-0009's own
    trajectory span (~0.58 deg): harmless there (0.4x its span already exceeds 0.15),
    but for ow-0008's much tighter spiral (~0.24 x 0.11 deg) that same fixed floor
    dominated both axes and added far more padding than the 40% proportional rule
    intended -- reported as "not zoomed enough" and fixed by shrinking the floor so
    the proportional rule governs real trajectories of any size (verified this does
    not change ow-0009's own view: its 0.4x-scaled padding already exceeds 0.03 on
    both axes).
    """
    lon_min, lon_max = float(trajectory_df["lon"].min()), float(trajectory_df["lon"].max())
    lat_min, lat_max = float(trajectory_df["lat"].min()), float(trajectory_df["lat"].max())

    lon_pad = max((lon_max - lon_min) * padding_fraction, min_padding_deg)
    lat_pad = max((lat_max - lat_min) * padding_fraction, min_padding_deg)

    return {
        "west": lon_min - lon_pad, "east": lon_max + lon_pad,
        "south": lat_min - lat_pad, "north": lat_max + lat_pad,
    }


def compute_view_bounds_with_span(trajectory_df: pd.DataFrame, lon_span: float, lat_span: float) -> dict:
    """A map view of exactly `lon_span` x `lat_span` degrees, centered on
    `trajectory_df`'s own bounding-box center. Used to make two different cases'
    animations visually comparable at the *same absolute zoom level* -- e.g. matching
    ow-0008's view span to ow-0009's own (computed dynamically from ow-0009's own
    `compute_view_bounds` result, never hard-coded, so it can't go stale) -- rather
    than `compute_view_bounds`'s proportional-to-each-trajectory's-own-size padding,
    which by design gives small trajectories a tighter absolute view than large ones
    (correct for "zoom to fit this trajectory" but not for "match this other video's
    zoom level")."""
    lon_center = (float(trajectory_df["lon"].min()) + float(trajectory_df["lon"].max())) / 2.0
    lat_center = (float(trajectory_df["lat"].min()) + float(trajectory_df["lat"].max())) / 2.0
    return {
        "west": lon_center - lon_span / 2.0, "east": lon_center + lon_span / 2.0,
        "south": lat_center - lat_span / 2.0, "north": lat_center + lat_span / 2.0,
    }


def status_by_hotspot_at(hotspot_alerts_df: pd.DataFrame, floor_time: np.datetime64) -> dict[str, str]:
    g = hotspot_alerts_df[hotspot_alerts_df["time"] == floor_time]
    return dict(zip(g["hotspot_name"], g["status"]))


def _legend_handles() -> list:
    return [
        Line2D([], [], marker="o", linestyle="", color=COLOR_ACTIVE, markersize=6, label="Active oil parcel"),
        Line2D([], [], marker="x", linestyle="", color=COLOR_STRANDED, markersize=7, label="Stranded oil parcel"),
        Line2D([], [], color=COLOR_ACTIVE, linewidth=1.2, alpha=0.5, label="Parcel drift path (since t0)"),
        Patch(facecolor=COLOR_WATCH_FILL, edgecolor=COLOR_ZONE_OUTLINE, alpha=0.35, label="Watch Zone (100 km)"),
        Patch(facecolor=COLOR_CRITICAL_FILL, edgecolor=COLOR_CRITICAL_FILL, alpha=0.45, label="Critical Strike Zone (40 km)"),
        Patch(facecolor=COLOR_LAND, edgecolor=COLOR_COASTLINE, label="Land"),
    ]


def render_frame(
    ax: plt.Axes,
    land: LandBackground,
    hotspots: list[Hotspot],
    status_by_name: dict[str, str],
    positions: pd.DataFrame,
    display_time: np.datetime64,
    view_bounds: dict = FULL_AOI,
    trail_segments: list[np.ndarray] | None = None,
    overview_label: str | None = None,
    case_label: str = "ow-0009",
) -> None:
    """Draw one complete frame (basemap, hotspot zones/labels, particle trails,
    particles, legend, timestamp) into `ax`. Caller is responsible for `ax.clear()`
    before calling this for a new frame and for saving/advancing the figure
    afterward. `view_bounds` (`west/east/south/north`) sets the map extent -- pass the
    same dict (e.g. from `compute_view_bounds`) to every frame of one animation so the
    view stays fixed across playback; defaults to the full CMEMS AOI only if the
    caller doesn't override it. `trail_segments` (e.g. from `build_trail_segments`),
    if given, draws each particle's path-so-far as a thin faded line so the drift
    trajectory is visible, not just the current instantaneous position.
    `overview_label`, if given, replaces the usual "active/stranded" subtitle line
    (used for the opening regional-overview/zoom-in shot, where the particle counts
    aren't the point of the frame yet). `case_label` names the case in the title (e.g.
    `"ow-0008"`); defaults to `"ow-0009"` only for backward compatibility with
    existing calls that don't pass it."""
    display_time = np.datetime64(display_time, "ns")
    ax.set_xlim(view_bounds["west"], view_bounds["east"])
    ax.set_ylim(view_bounds["south"], view_bounds["north"])
    mean_lat = (view_bounds["south"] + view_bounds["north"]) / 2.0
    ax.set_aspect(1.0 / np.cos(np.deg2rad(mean_lat)))

    lon_edges = np.concatenate([land.lon - np.diff(land.lon, prepend=land.lon[0] - (land.lon[1] - land.lon[0])) / 2, [land.lon[-1] + (land.lon[-1] - land.lon[-2]) / 2]])
    lat_edges = np.concatenate([land.lat - np.diff(land.lat, prepend=land.lat[0] - (land.lat[1] - land.lat[0])) / 2, [land.lat[-1] + (land.lat[-1] - land.lat[-2]) / 2]])
    from matplotlib.colors import ListedColormap
    ax.pcolormesh(
        lon_edges, lat_edges, land.ocean_valid_mask.astype(int),
        cmap=ListedColormap([COLOR_LAND, COLOR_WATER]), vmin=0, vmax=1,
        shading="flat", zorder=0,
    )
    lon_mesh, lat_mesh = np.meshgrid(land.lon, land.lat)
    ax.contour(lon_mesh, lat_mesh, land.ocean_valid_mask.astype(float), levels=[0.5],
               colors=COLOR_COASTLINE, linewidths=0.7, zorder=1)

    for h in hotspots:
        status = status_by_name.get(h.name, CLEAR)
        ax.add_patch(MplPolygon(h.watch_ring, closed=True, fill=False,
                                 edgecolor=COLOR_ZONE_OUTLINE, linewidth=0.9, zorder=2))
        if status == WATCH:
            ax.add_patch(MplPolygon(h.watch_ring, closed=True, facecolor=COLOR_WATCH_FILL,
                                     edgecolor=COLOR_WATCH_FILL, alpha=0.35, linewidth=1.2, zorder=2))
        elif status == CRITICAL:
            ax.add_patch(MplPolygon(h.watch_ring, closed=True, facecolor=COLOR_WATCH_FILL,
                                     edgecolor=COLOR_WATCH_FILL, alpha=0.20, linewidth=1.2, zorder=2))
            ax.add_patch(MplPolygon(h.critical_ring, closed=True, facecolor=COLOR_CRITICAL_FILL,
                                     edgecolor=COLOR_CRITICAL_FILL, alpha=0.50, linewidth=1.5, zorder=3))

        if status == CRITICAL:
            label_color, weight = COLOR_CRITICAL_FILL, "bold"
        elif status == WATCH:
            label_color, weight = "#8a6100", "bold"
        else:
            label_color, weight = "#555555", "normal"
        ax.text(h.center_lon, h.center_lat, h.name, fontsize=5.5, color=label_color, weight=weight,
                ha="center", va="center", zorder=5, clip_on=True,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.65))

    if trail_segments:
        ax.add_collection(LineCollection(
            trail_segments, colors=COLOR_ACTIVE, linewidths=0.7, alpha=0.4, zorder=3,
        ))

    active = positions[~positions["stranded"]]
    stranded = positions[positions["stranded"]]
    ax.scatter(active["lon"], active["lat"], s=16, c=COLOR_ACTIVE, zorder=4, linewidths=0.3,
               edgecolors="white")
    if len(stranded):
        ax.scatter(stranded["lon"], stranded["lat"], s=28, c=COLOR_STRANDED, marker="x", zorder=4)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    n_active, n_stranded = len(active), len(stranded)
    subtitle = overview_label if overview_label is not None else f"active parcels: {n_active}   stranded: {n_stranded}"
    ax.set_title(
        f"{case_label} forward propagation      {np.datetime_as_string(display_time, unit='m')} UTC\n"
        f"{subtitle}",
        fontsize=10, loc="left",
    )
    ax.legend(handles=_legend_handles(), loc="lower left", fontsize=6.5, framealpha=0.85)
