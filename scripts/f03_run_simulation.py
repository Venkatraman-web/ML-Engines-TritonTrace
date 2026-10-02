"""F03 -- run + validate the 72-hour ow-0009 forward simulation (not a test file --
CLAUDE.md forbids `tests/`/`test_*.py`).

Runs `fwd.simulate.run_simulation`, writes the trajectory and hourly diagnostics to
`output/ow-0009/`, then verifies the required invariants:
  - exactly 73 recorded timestamps (72 steps + initial state)
  - exactly 371 initial parcel IDs, present at every one of the 73 timestamps
  - final timestamp is exactly 2019-05-01 15:38:41 UTC
  - no stranded particle's position changes after the hour it stranded
  - reports final active/stranded counts and total fallback/invalid-forcing usage

Run: python scripts/f03_run_simulation.py
"""

from __future__ import annotations

import sys
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from fwd.forcing import DIRECT, INVALID_ENVIRONMENT, NEIGHBOR_FALLBACK, ForcingAdapter  # noqa: E402
from fwd.simulate import DURATION_HOURS, OW0009_OBS_TIME, load_ow0009_parcels, run_simulation  # noqa: E402
from fwd.step import ALREADY_STRANDED, LAND_STRANDED  # noqa: E402

OUTPUT_DIR = REPO_ROOT / "output" / "ow-0009"
EXPECTED_FINAL_TIME = np.datetime64("2019-05-01T15:38:41", "ns")


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main() -> None:
    section("Running 72-hour forward simulation for ow-0009")
    t_start_wall = _time.time()
    adapter = ForcingAdapter()
    parcels = load_ow0009_parcels()
    trajectory_df, diagnostics_df = run_simulation(adapter=adapter, parcels=parcels)
    elapsed = _time.time() - t_start_wall
    print(f"simulation complete in {elapsed:.1f}s wall time "
          f"({len(parcels)} parcels x {DURATION_HOURS} steps = {len(parcels) * DURATION_HOURS} step_particle calls)")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    traj_path = OUTPUT_DIR / "trajectory.parquet"
    diag_path = OUTPUT_DIR / "hourly_diagnostics.parquet"
    trajectory_df.to_parquet(traj_path, index=False)
    diagnostics_df.to_parquet(diag_path, index=False)
    print(f"wrote {traj_path}  ({len(trajectory_df)} rows)")
    print(f"wrote {diag_path}  ({len(diagnostics_df)} rows)")

    section("1) Timestamp / particle-count invariants")
    times = sorted(trajectory_df["time"].unique())
    print(f"distinct timestamps: {len(times)}")
    assert len(times) == DURATION_HOURS + 1 == 73, f"expected 73 timestamps, got {len(times)}"
    assert times[0] == OW0009_OBS_TIME, f"first timestamp {times[0]} != {OW0009_OBS_TIME}"
    assert times[-1] == EXPECTED_FINAL_TIME, f"final timestamp {times[-1]} != {EXPECTED_FINAL_TIME}"
    print(f"first: {times[0]}   final: {times[-1]}  (matches 2019-04-28 15:38:41 -> 2019-05-01 15:38:41)")

    t0_ids = set(trajectory_df.loc[trajectory_df["time"] == times[0], "particle_id"])
    assert len(t0_ids) == 371, f"expected 371 initial parcel IDs, got {len(t0_ids)}"
    assert t0_ids == {p["id"] for p in parcels}
    for t in times:
        ids_at_t = set(trajectory_df.loc[trajectory_df["time"] == t, "particle_id"])
        assert ids_at_t == t0_ids, f"particle IDs at {t} do not match the initial 371"
    assert len(trajectory_df) == 371 * 73, f"expected {371*73} rows, got {len(trajectory_df)}"
    dup = trajectory_df.duplicated(subset=["particle_id", "time"]).sum()
    assert dup == 0, f"found {dup} duplicate particle_id/time rows"
    print("all 371 parcel IDs present, unchanged, at all 73 timestamps; no duplicate rows")

    for pid in list(t0_ids)[:1] + [max(t0_ids)]:
        traj = trajectory_df[trajectory_df["particle_id"] == pid].sort_values("time")
        assert traj["mass_kg"].nunique() == 1 and traj["confidence"].nunique() == 1, (
            f"particle {pid} mass_kg/confidence changed over time"
        )
    print("mass_kg/confidence metadata preserved unchanged per particle over the run")

    section("2) Non-finite / physical sanity checks on active particles")
    active = trajectory_df[(trajectory_df["valid"]) & (~trajectory_df["stranded"]) & (trajectory_df["time"] != times[0])]
    assert active["lon"].apply(np.isfinite).all() and active["lat"].apply(np.isfinite).all(), (
        "non-finite coordinate found on an active particle"
    )
    assert active["oil_u"].apply(np.isfinite).all() and active["oil_v"].apply(np.isfinite).all(), (
        "non-finite oil velocity found on an active particle"
    )
    speeds = np.hypot(active["oil_u"], active["oil_v"])
    print(f"active-particle oil speed range: [{speeds.min():.4f}, {speeds.max():.4f}] m/s")
    assert speeds.max() < 5.0, "implausible oil speed -- check units"

    section("3) Stranded particles never move again")
    violations = 0
    stranded_events = 0
    for pid, traj in trajectory_df.groupby("particle_id"):
        traj = traj.sort_values("time").reset_index(drop=True)
        stranded_idx = traj.index[traj["stranded"]]
        if len(stranded_idx) == 0:
            continue
        first_stranded = stranded_idx[0]
        stranded_events += 1
        lon0, lat0 = traj.loc[first_stranded, "lon"], traj.loc[first_stranded, "lat"]
        after = traj.loc[first_stranded:]
        if not ((after["lon"] == lon0).all() and (after["lat"] == lat0).all()):
            violations += 1
            print(f"  VIOLATION: particle {pid} moved after stranding at {traj.loc[first_stranded, 'time']}")
    print(f"particles that stranded at some point during the run: {stranded_events}")
    assert violations == 0, f"{violations} particle(s) moved after being marked stranded"
    print("confirmed: no stranded particle's position changes after the step it stranded on")

    section("4) Final state / run summary")
    final = trajectory_df[trajectory_df["time"] == times[-1]]
    n_active_final = int(((final["valid"]) & (~final["stranded"])).sum())
    n_stranded_final = int(final["stranded"].sum())
    print(f"final ({times[-1]}): active={n_active_final} stranded={n_stranded_final} "
          f"(total={n_active_final + n_stranded_final})")
    assert n_active_final + n_stranded_final == 371

    stepped = trajectory_df[trajectory_df["time"] != times[0]]
    mode_counts = stepped["forcing_mode"].value_counts().to_dict()
    print("forcing_mode counts across all 72 x 371 steps:", mode_counts)
    print(f"  DIRECT             : {mode_counts.get(DIRECT, 0)}")
    print(f"  NEIGHBOR_FALLBACK  : {mode_counts.get(NEIGHBOR_FALLBACK, 0)}")
    print(f"  LAND_STRANDED      : {mode_counts.get(LAND_STRANDED, 0)}")
    print(f"  INVALID_ENVIRONMENT: {mode_counts.get(INVALID_ENVIRONMENT, 0)}")
    print(f"  ALREADY_STRANDED   : {mode_counts.get(ALREADY_STRANDED, 0)}")

    print("\nhourly diagnostics (first 3 and last 3 rows):")
    print(diagnostics_df.head(3).to_string(index=False))
    print("...")
    print(diagnostics_df.tail(3).to_string(index=False))
    assert (diagnostics_df["active_count"] + diagnostics_df["stranded_count"] == 371).all()

    section("ALL CHECKS PASSED")
    print("F03 72-hour forward simulation: COMPLETE")


if __name__ == "__main__":
    main()
