"""F04 -- hotspot alert engine for the ow-0009 forward-propagation pilot.

Evaluates every active particle in an F03 trajectory against all 11 hotspot geofences
(22 polygons: one "Watch Zone" + one "Critical Strike Zone" per hotspot) from
`regional_alert_geofences_wide.geojson`, per hourly recorded state, using the actual
polygon geometries -- point-in-polygon containment via `matplotlib.path.Path`, never a
distance-to-center-vs-radius shortcut (requirement 2/6: "do not recreate circles").
The minimum-distance metric reuses `iste.ais_matching.haversine_distance_km`
(unmodified Stage III code) rather than reimplementing great-circle distance.

Zone selection is configurable by `zone_level` string (`HotspotThresholds`), not by a
hard-coded "40"/"100" anywhere in the alert logic -- the actual km values used are
whatever `radius_km` the GeoJSON itself carries for that zone (currently 40/100),
preserved as metadata on each `Hotspot` for display/labelling, never hard-coded into
the containment test itself (requirement 10).

Alert rule -- DEVIATION FROM THE ORIGINAL LOCK, explicitly requested by the user after
F07 (recorded per CLAUDE.md section 13, "no silent scientific changes": this is not
silent, it is a deliberate, documented, user-directed change to the locked spec for
this pilot). README section 10 / this task's original requirement 4 locked "ANY active
particle inside a zone triggers it." The user asked instead for a minimum-fraction-
of-the-slick rule, reasoning that a single particle grazing a 40 km circle
(observed in the original run: Cape Greco MPA first went CRITICAL with only 12/371 =
3.2% of the slick inside it) overstates the alert. New rule, evaluated independently
per hotspot per hour from the actual set of ACTIVE particles (valid and not stranded,
requirement 8) -- still never the slick centroid (requirement 6), still using real
polygon containment per particle, just a different aggregation threshold:

    (particles inside critical polygon) / (active particles) >= min_fraction_critical
                                                               -> CRITICAL
    elif (particles inside watch polygon) / (active particles) >= min_fraction_watch
                                                               -> WATCH
    else                                                       -> CLEAR

Both thresholds default to 0.30 (30%, as requested) and are configurable via
`HotspotThresholds` (requirement 10), not hard-coded in the comparison itself.

`first_watch_time`/`first_critical_time` are running "first ever reached" timestamps
per hotspot under whichever rule is active, carried forward on every subsequent hourly
record for that hotspot once set (never reset back to `None`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath

from fwd.stage3_bridge import ensure_iste_importable

ensure_iste_importable()

from iste.ais_matching import haversine_distance_km  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
GEOJSON_PATH = REPO_ROOT / "regional_alert_geofences_wide.geojson"

CRITICAL = "CRITICAL"
WATCH = "WATCH"
CLEAR = "CLEAR"


@dataclass(frozen=True)
class HotspotThresholds:
    """Which `zone_level` string in the GeoJSON counts as the critical/watch zone --
    configurable (requirement 10) rather than a hard-coded "40 km"/"100 km" check
    anywhere in the containment logic itself.

    `min_fraction_critical`/`min_fraction_watch` are the minimum share of the
    currently-active slick (particles inside the zone / active particles) required to
    trigger that zone's status -- a user-requested change from the originally locked
    "any single particle triggers it" rule (see this module's docstring). Both default
    to 0.30 (30%)."""

    critical_zone_level: str = "Critical Strike Zone"
    watch_zone_level: str = "Watch Zone"
    min_fraction_critical: float = 0.30
    min_fraction_watch: float = 0.30


@dataclass(frozen=True)
class Hotspot:
    name: str
    type: str
    liability_multiplier: float  # metadata only -- never used in the alert decision
    center_lon: float
    center_lat: float
    critical_radius_km: float  # from the GeoJSON's own `radius_km` property
    watch_radius_km: float
    critical_ring: np.ndarray  # (n, 2) lon/lat vertices, the actual polygon
    watch_ring: np.ndarray
    critical_path: MplPath
    watch_path: MplPath


def _ring_center(ring: np.ndarray) -> tuple[float, float]:
    """Mean of the (deduplicated, ring is closed) polygon vertices. Verified in the F04
    audit that, for every hotspot in this GeoJSON, the watch and critical rings share
    exactly this same center (offset 0.000000 deg) -- i.e. they are genuinely
    concentric circles, so this is a legitimate single "hotspot location" for the
    minimum-distance metric, not an approximation invented here."""
    pts = ring[:-1] if np.allclose(ring[0], ring[-1]) else ring
    return float(pts[:, 0].mean()), float(pts[:, 1].mean())


def load_hotspots(
    geojson_path: Path | None = None, thresholds: HotspotThresholds = HotspotThresholds()
) -> list[Hotspot]:
    """Load all 11 hotspots (22 polygons) from the GeoJSON, keeping the actual polygon
    geometries directly (requirement 1/2)."""
    path = geojson_path if geojson_path is not None else GEOJSON_PATH
    with open(path, encoding="utf-8") as f:
        gj = json.load(f)

    by_name: dict[str, dict] = {}
    for feat in gj["features"]:
        props = feat["properties"]
        ring = np.array(feat["geometry"]["coordinates"][0], dtype=np.float64)
        by_name.setdefault(props["name"], {})[props["zone_level"]] = (ring, props)

    hotspots = []
    for name, zones in by_name.items():
        if thresholds.critical_zone_level not in zones or thresholds.watch_zone_level not in zones:
            raise KeyError(f"hotspot '{name}' is missing a required zone_level: {list(zones)}")
        c_ring, c_props = zones[thresholds.critical_zone_level]
        w_ring, w_props = zones[thresholds.watch_zone_level]
        c_center = _ring_center(c_ring)
        w_center = _ring_center(w_ring)
        # both zones of one hotspot must be concentric -- fail loudly rather than
        # silently averaging two different centers if a future GeoJSON update breaks this
        if not np.allclose(c_center, w_center, atol=1e-6):
            raise ValueError(
                f"hotspot '{name}': watch/critical rings are not concentric "
                f"({w_center} vs {c_center}) -- cannot derive a single hotspot location"
            )
        hotspots.append(
            Hotspot(
                name=name,
                type=w_props.get("type", ""),
                liability_multiplier=float(w_props.get("liability_multiplier", float("nan"))),
                center_lon=w_center[0],
                center_lat=w_center[1],
                critical_radius_km=float(c_props["radius_km"]),
                watch_radius_km=float(w_props["radius_km"]),
                critical_ring=c_ring,
                watch_ring=w_ring,
                critical_path=MplPath(c_ring),
                watch_path=MplPath(w_ring),
            )
        )

    if len(hotspots) != 11:
        raise AssertionError(f"expected 11 hotspots, found {len(hotspots)}")
    return hotspots


def evaluate_hotspot_alerts(
    trajectory_df: pd.DataFrame,
    hotspots: list[Hotspot],
    thresholds: HotspotThresholds = HotspotThresholds(),
) -> pd.DataFrame:
    """One row per hotspot per recorded hour, per requirement 7's field list (plus
    `fraction_inside_40km`/`fraction_inside_100km`, added for audit transparency now
    that the status decision itself depends on these fractions, not just the raw
    counts)."""
    first_watch: dict[str, np.datetime64 | None] = {h.name: None for h in hotspots}
    first_critical: dict[str, np.datetime64 | None] = {h.name: None for h in hotspots}
    records = []

    for t, group in trajectory_df.groupby("time", sort=True):
        active = group[group["valid"] & ~group["stranded"]]
        lons = active["lon"].to_numpy()
        lats = active["lat"].to_numpy()
        pts = np.column_stack([lons, lats]) if len(lons) else np.empty((0, 2))
        n_active = len(pts)

        for h in hotspots:
            if n_active == 0:
                status, n_critical, n_watch, min_dist = CLEAR, 0, 0, float("nan")
                frac_critical = frac_watch = 0.0
            else:
                in_critical = h.critical_path.contains_points(pts)
                in_watch = h.watch_path.contains_points(pts)
                n_critical = int(in_critical.sum())
                n_watch = int(in_watch.sum())
                frac_critical = n_critical / n_active
                frac_watch = n_watch / n_active
                dists_km = haversine_distance_km(lats, lons, h.center_lat, h.center_lon)
                min_dist = float(dists_km.min())
                if frac_critical >= thresholds.min_fraction_critical:
                    status = CRITICAL
                elif frac_watch >= thresholds.min_fraction_watch:
                    status = WATCH
                else:
                    status = CLEAR

            if status in (WATCH, CRITICAL) and first_watch[h.name] is None:
                first_watch[h.name] = t
            if status == CRITICAL and first_critical[h.name] is None:
                first_critical[h.name] = t

            records.append({
                "hotspot_name": h.name,
                "time": t,
                "status": status,
                "particles_inside_40km": n_critical,
                "particles_inside_100km": n_watch,
                "fraction_inside_40km": frac_critical,
                "fraction_inside_100km": frac_watch,
                "min_distance_km": min_dist,
                "first_watch_time": first_watch[h.name],
                "first_critical_time": first_critical[h.name],
            })

    return pd.DataFrame(records)
