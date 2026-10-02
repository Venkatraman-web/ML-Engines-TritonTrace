"""Build a wide-format hotspot-alert status timeline (11 hotspots x 73 timestamps,
status only: CLEAR/WATCH/CRITICAL) for both ow-0009 and ow-0008, reshaped from each
case's existing `hotspot_alerts.parquet` (the source of truth -- this only pivots
already-computed status values, it does not recompute or alter any alert decision).
Not a test file -- CLAUDE.md forbids `tests/`/`test_*.py`.

Output: `output/<case>/csv/hotspot_status_timeline.csv` -- one row per hotspot (in
the same order as `regional_alert_geofences_wide.geojson`), one column per recorded
hour, column headers are the actual UTC timestamps (unambiguous, not just "t+N").

Run: python scripts/build_hotspot_status_timeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from fwd.hotspots import CLEAR, CRITICAL, WATCH, load_hotspots  # noqa: E402

CASES = ("ow-0009", "ow-0008")


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def build_case(case_id: str, hotspot_order: list[str]) -> None:
    section(case_id)
    alerts_path = REPO_ROOT / "output" / case_id / "hotspot_alerts.parquet"
    alerts_df = pd.read_parquet(alerts_path)
    print(f"loaded {alerts_path}  ({len(alerts_df)} rows)")

    matrix = alerts_df.pivot(index="hotspot_name", columns="time", values="status")
    matrix = matrix.reindex(index=hotspot_order)  # same order as the GeoJSON, not alphabetical
    matrix = matrix[sorted(matrix.columns)]  # columns ascending in time

    n_hotspots, n_times = matrix.shape
    assert n_hotspots == 11, f"{case_id}: expected 11 hotspot rows, got {n_hotspots}"
    assert n_times == 73, f"{case_id}: expected 73 timestamp columns, got {n_times}"
    assert matrix.isna().sum().sum() == 0, f"{case_id}: matrix has missing (hotspot, time) cells"
    valid_statuses = {CLEAR, WATCH, CRITICAL}
    bad = ~matrix.isin(valid_statuses)
    assert not bad.values.any(), f"{case_id}: found status values outside {valid_statuses}"
    print(f"matrix shape: {n_hotspots} hotspots x {n_times} timestamps -- OK, all cells filled, all statuses valid")

    # column headers: exact UTC timestamps (unambiguous), formatted without the
    # redundant trailing .000000000 nanosecond zeros
    matrix.columns = [pd.Timestamp(c).strftime("%Y-%m-%d %H:%M:%S") for c in matrix.columns]
    matrix.index.name = "hotspot_name"

    out_dir = REPO_ROOT / "output" / case_id / "csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "hotspot_status_timeline.csv"
    matrix.to_csv(out_path)
    print(f"wrote {out_path}  ({out_path.stat().st_size:,} bytes)")

    print("\nstatus counts per hotspot (out of 73 hours):")
    for name in hotspot_order:
        counts = matrix.loc[name].value_counts()
        c = int(counts.get(CLEAR, 0))
        w = int(counts.get(WATCH, 0))
        r = int(counts.get(CRITICAL, 0))
        marker = "**" if (w or r) else "  "
        print(f"{marker} {name:35s} CLEAR={c:2d}  WATCH={w:2d}  CRITICAL={r:2d}")


def main() -> None:
    hotspots = load_hotspots()
    hotspot_order = [h.name for h in hotspots]
    assert len(hotspot_order) == 11

    for case_id in CASES:
        build_case(case_id, hotspot_order)

    section("ALL CHECKS PASSED")
    print("hotspot_status_timeline.csv written for both cases")


if __name__ == "__main__":
    main()
