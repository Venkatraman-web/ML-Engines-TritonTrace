"""ow-0008 case -- run + validate the hotspot alert engine (F04 equivalent; not a
test file -- CLAUDE.md forbids `tests/`/`test_*.py`).

Mirrors `scripts/f04_run_hotspot_alerts.py` exactly, pointed at
`output/ow-0008/trajectory.parquet`. Uses `fwd.hotspots.evaluate_hotspot_alerts`'s
current default thresholds (>=30% of active particles per zone -- the same
user-directed rule now in effect for ow-0009, see progress.md's post-F07 revision 3),
so both cases are evaluated under one consistent, current rule.

Run: python scripts/ow0008_run_hotspot_alerts.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from fwd.hotspots import CLEAR, CRITICAL, WATCH, HotspotThresholds, evaluate_hotspot_alerts, load_hotspots  # noqa: E402

CASE_ID = "ow-0008"
TRAJECTORY_PATH = REPO_ROOT / "output" / CASE_ID / "trajectory.parquet"
OUTPUT_PATH = REPO_ROOT / "output" / CASE_ID / "hotspot_alerts.parquet"


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main() -> None:
    section("Loading inputs")
    trajectory_df = pd.read_parquet(TRAJECTORY_PATH)
    hotspots = load_hotspots()
    thresholds = HotspotThresholds()
    print(f"trajectory: {len(trajectory_df)} rows, {trajectory_df['time'].nunique()} timestamps, "
          f"{trajectory_df['particle_id'].nunique()} particles")
    print(f"hotspots loaded: {len(hotspots)}")
    print(f"alert rule: >= {thresholds.min_fraction_critical:.0%} of active particles in the critical "
          f"polygon -> CRITICAL; >= {thresholds.min_fraction_watch:.0%} in the watch polygon -> WATCH")
    for h in hotspots:
        print(f"  {h.name:35s} center=({h.center_lon:.4f},{h.center_lat:.4f}) "
              f"critical~{h.critical_radius_km:.1f}km watch~{h.watch_radius_km:.1f}km")
    assert len(hotspots) == 11

    section("Evaluating hotspot alerts for all recorded states")
    alerts_df = evaluate_hotspot_alerts(trajectory_df, hotspots, thresholds)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    alerts_df.to_parquet(OUTPUT_PATH, index=False)
    print(f"wrote {OUTPUT_PATH}  ({len(alerts_df)} rows)")

    section("1) Structural invariants")
    n_times = trajectory_df["time"].nunique()
    assert len(alerts_df) == 11 * n_times, f"expected {11*n_times} rows, got {len(alerts_df)}"
    dup = alerts_df.duplicated(subset=["hotspot_name", "time"]).sum()
    assert dup == 0, f"found {dup} duplicate hotspot_name/time rows"
    assert set(alerts_df["status"].unique()) <= {CLEAR, WATCH, CRITICAL}
    assert alerts_df["hotspot_name"].nunique() == 11
    print(f"{len(alerts_df)} rows = 11 hotspots x {n_times} timestamps, no duplicates, "
          f"status values confined to {{CLEAR, WATCH, CRITICAL}}")

    has_active = alerts_df["min_distance_km"].notna()
    nested = (alerts_df.loc[has_active, "particles_inside_40km"]
              <= alerts_df.loc[has_active, "particles_inside_100km"])
    assert nested.all(), "found an hour/hotspot where particles_inside_40km > particles_inside_100km"
    print("confirmed: particles_inside_40km <= particles_inside_100km at every row")

    frac_ok = (alerts_df.loc[has_active, "fraction_inside_40km"] <= alerts_df.loc[has_active, "fraction_inside_100km"] + 1e-9)
    assert frac_ok.all(), "fraction_inside_40km > fraction_inside_100km somewhere"
    print("confirmed: fraction_inside_40km <= fraction_inside_100km at every row")

    for name, g in alerts_df.groupby("hotspot_name"):
        g = g.sort_values("time")
        fw = g["first_watch_time"].dropna().unique()
        fc = g["first_critical_time"].dropna().unique()
        assert len(fw) <= 1 and len(fc) <= 1, f"{name}: first_watch_time/first_critical_time not stable once set"
        if len(fw) and len(fc):
            assert fw[0] <= fc[0], f"{name}: first_critical_time before first_watch_time"
    print("confirmed: first_watch_time/first_critical_time are stable once set and watch <= critical")

    section("2) Per-hotspot summary")
    any_alert = False
    for h in hotspots:
        g = alerts_df[alerts_df["hotspot_name"] == h.name].sort_values("time")
        statuses_seen = sorted(g["status"].unique(), key=[CLEAR, WATCH, CRITICAL].index)
        min_dist = g["min_distance_km"].min()
        max_frac_c = g["fraction_inside_40km"].max()
        max_frac_w = g["fraction_inside_100km"].max()
        fw = g["first_watch_time"].dropna()
        fc = g["first_critical_time"].dropna()
        fw_str = str(fw.iloc[0]) if len(fw) else "never"
        fc_str = str(fc.iloc[0]) if len(fc) else "never"
        triggered = WATCH in statuses_seen or CRITICAL in statuses_seen
        any_alert = any_alert or triggered
        marker = "**" if triggered else "  "
        print(f"{marker} {h.name:35s} statuses={statuses_seen}  min_dist={min_dist:7.2f} km  "
              f"max_frac_critical={max_frac_c:.1%}  max_frac_watch={max_frac_w:.1%}  "
              f"first_watch={fw_str}  first_critical={fc_str}")

    section("3) Simultaneous hotspot alerts (>1 hotspot WATCH/CRITICAL at the same hour)")
    active_alerts = alerts_df[alerts_df["status"] != CLEAR]
    counts_per_time = active_alerts.groupby("time")["hotspot_name"].nunique()
    simultaneous = counts_per_time[counts_per_time > 1]
    if len(simultaneous) == 0:
        print("none found -- at most one hotspot alerted per hour across the run")
    else:
        print(f"{len(simultaneous)} timestamp(s) with multiple simultaneously-alerting hotspots "
              f"(printing first 10):")
        for t, n in list(simultaneous.items())[:10]:
            names = sorted(active_alerts.loc[active_alerts["time"] == t, "hotspot_name"].unique())
            print(f"  {t}: {n} hotspots -> {names}")

    if not any_alert:
        print("\nNOTE: no hotspot reached WATCH or CRITICAL at any point in this 72-hour run.")

    section("ALL CHECKS PASSED")
    print(f"{CASE_ID} hotspot alert engine: COMPLETE")


if __name__ == "__main__":
    main()
