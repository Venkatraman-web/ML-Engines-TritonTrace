"""F06 -- compare our ow-0009 forward simulation against the OpenOil/OpenDrift
reference run (not a test file -- CLAUDE.md forbids `tests/`/`test_*.py`).

Loads `output/ow-0009/trajectory.parquet` (F03, this project's own implementation)
and `output/ow-0009/openoil_reference.parquet` (F06,
`scripts/f06_openoil_reference.py`, real OpenOil/OpenDrift under the same forcing,
positions, timestep and drift factors), computes the great-circle position error
between them at +1h/+6h/+12h/+24h/+48h/+72h (reusing
`iste.ais_matching.haversine_distance_km`, not a new distance formula), and reports
mean/median/max error per checkpoint plus an investigation of any significant
discrepancy. This script only reads both outputs -- it does not modify F01-F05 or the
OpenOil reference run (requirement 2/6: no tuning to force agreement).

Run: python scripts/f06_compare_reference.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from fwd.stage3_bridge import ensure_iste_importable  # noqa: E402

ensure_iste_importable()
from iste.ais_matching import haversine_distance_km  # noqa: E402

OURS_PATH = REPO_ROOT / "output" / "ow-0009" / "trajectory.parquet"
REFERENCE_PATH = REPO_ROOT / "output" / "ow-0009" / "openoil_reference.parquet"
OUTPUT_PATH = REPO_ROOT / "output" / "ow-0009" / "openoil_comparison.parquet"

START_TIME = np.datetime64("2019-04-28T15:38:41", "ns")
CHECKPOINT_HOURS = [1, 6, 12, 24, 48, 72]


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main() -> None:
    section("Loading both trajectories")
    ours = pd.read_parquet(OURS_PATH)
    ref = pd.read_parquet(REFERENCE_PATH)
    print(f"ours:      {len(ours)} rows, {ours['time'].nunique()} timestamps, {ours['particle_id'].nunique()} particles")
    print(f"reference: {len(ref)} rows, {ref['time'].nunique()} timestamps, {ref['particle_id'].nunique()} particles")
    assert set(ours["particle_id"].unique()) == set(ref["particle_id"].unique()), (
        "particle ID sets differ between our trajectory and the OpenOil reference"
    )
    if "status" in ref.columns:
        n_deactivated = int((ref.groupby("particle_id")["status"].nunique() > 1).sum())
        print(f"OpenOil reference: particles whose status ever changed (e.g. stranded): {n_deactivated}")

    checkpoint_rows = []
    per_particle_frames = []

    section("Per-checkpoint position error (great-circle, km)")
    for h in CHECKPOINT_HOURS:
        t = START_TIME + np.timedelta64(h, "h")
        o_at = ours[ours["time"] == t][["particle_id", "lon", "lat", "valid", "stranded"]].set_index("particle_id")
        r_at = ref[ref["time"] == t][["particle_id", "lon", "lat"]].set_index("particle_id")
        if len(o_at) == 0 or len(r_at) == 0:
            print(f"+{h:>2}h ({t}): MISSING timestamp in one of the two trajectories -- skipped")
            continue
        common = o_at.index.intersection(r_at.index)
        assert len(common) == 371, f"+{h}h: expected 371 common particle IDs, got {len(common)}"
        o_at = o_at.loc[common]
        r_at = r_at.loc[common]

        dist_km = haversine_distance_km(
            o_at["lat"].to_numpy(), o_at["lon"].to_numpy(),
            r_at["lat"].to_numpy(), r_at["lon"].to_numpy(),
        )
        mean_e, median_e, max_e = float(dist_km.mean()), float(np.median(dist_km)), float(dist_km.max())
        worst_pid = common[int(np.argmax(dist_km))]
        print(f"+{h:>2}h ({t}): mean={mean_e:7.3f} km  median={median_e:7.3f} km  "
              f"max={max_e:7.3f} km (particle {worst_pid})")

        checkpoint_rows.append({
            "checkpoint_hours": h, "time": t,
            "mean_error_km": mean_e, "median_error_km": median_e, "max_error_km": max_e,
            "n_particles_compared": len(common),
        })
        per_particle_frames.append(pd.DataFrame({
            "checkpoint_hours": h, "particle_id": common, "error_km": dist_km,
        }))

    checkpoints_df = pd.DataFrame(checkpoint_rows)
    per_particle_df = pd.concat(per_particle_frames, ignore_index=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    checkpoints_df.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nwrote {OUTPUT_PATH}")

    section("Overall summary")
    print(f"error growth: {checkpoints_df['mean_error_km'].tolist()}")
    print(f"mean-of-means across all 6 checkpoints: {checkpoints_df['mean_error_km'].mean():.3f} km")
    print(f"worst single checkpoint max error: {checkpoints_df['max_error_km'].max():.3f} km "
          f"(at +{checkpoints_df.loc[checkpoints_df['max_error_km'].idxmax(), 'checkpoint_hours']}h)")

    section("Investigation: where does the error come from?")
    # 1) initial-condition check: error at t=0 should be ~0 (both seeded from the same
    #    371 real parcel lon/lat) -- confirms this isn't a seeding/ID-mapping bug.
    o0 = ours[ours["time"] == START_TIME].set_index("particle_id")[["lon", "lat"]]
    r0 = ref[ref["time"] == START_TIME].set_index("particle_id")[["lon", "lat"]]
    common0 = o0.index.intersection(r0.index)
    d0 = haversine_distance_km(o0.loc[common0, "lat"], o0.loc[common0, "lon"],
                                r0.loc[common0, "lat"], r0.loc[common0, "lon"])
    print(f"t0 seeding-position check: mean={d0.mean():.6f} km, max={d0.max():.6f} km "
          f"(should be ~0 -- both seeded from the same real parcel lon/lat)")

    # 2) +1h check isolates a *single* Euler step's discrepancy, before any compounding
    #    over multiple hours -- large error already at +1h points at interpolation/
    #    coordinate-conversion/time-handling; error that only grows large much later
    #    points at compounding numerical drift instead.
    one_h = checkpoints_df[checkpoints_df["checkpoint_hours"] == 1]
    if len(one_h):
        print(f"+1h single-step error: mean={one_h['mean_error_km'].iloc[0]:.4f} km, "
              f"max={one_h['max_error_km'].iloc[0]:.4f} km")

    # 3) error growth shape: roughly linear with time -> consistent, compounding
    #    per-step differences (e.g. a small but systematic interpolation/advection
    #    scheme difference); a sudden jump at one checkpoint -> investigate that
    #    specific window (e.g. a coastline/stranding event, which F03 already ruled
    #    out for this particular case).
    growth = checkpoints_df.set_index("checkpoint_hours")["mean_error_km"]
    ratios = (growth / growth.index.to_series()).round(4)
    print("mean_error_km / elapsed_hours at each checkpoint (roughly constant -> linear/compounding growth):")
    print(ratios.to_string())

    section("DONE -- see progress.md for the full written investigation and verdict")


if __name__ == "__main__":
    main()
