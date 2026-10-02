"""ow-0008 case -- compare our forward simulation against the OpenOil/OpenDrift
reference run (F06 equivalent; not a test file -- CLAUDE.md forbids
`tests/`/`test_*.py`).

Mirrors `scripts/f06_compare_reference.py`'s structure and investigation exactly,
pointed at `output/ow-0008/`. Unlike ow-0009, this comparison tolerates a particle
being deactivated/NaN in the OpenDrift reference at some checkpoint (ow-0008 starts
only ~29 km from land, close enough that OpenDrift's real GSHHG coastline could
strand a particle our own CMEMS-mask-based simulation did not, or vice versa) --
such particles are excluded from that specific checkpoint's error statistics with an
explicit printed count, never silently dropped or averaged over as if they were
normal.

Run: python scripts/ow0008_compare_reference.py
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

CASE_ID = "ow-0008"
OURS_PATH = REPO_ROOT / "output" / CASE_ID / "trajectory.parquet"
REFERENCE_PATH = REPO_ROOT / "output" / CASE_ID / "openoil_reference.parquet"
OUTPUT_PATH = REPO_ROOT / "output" / CASE_ID / "openoil_comparison.parquet"

START_TIME = np.datetime64("2019-06-14T20:35:58", "ns")
CHECKPOINT_HOURS = [1, 6, 12, 24, 48, 72]
EXPECTED_PARCELS = 548


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
    n_deactivated_ref = 0
    if "status" in ref.columns:
        n_deactivated_ref = int((ref.groupby("particle_id")["status"].nunique() > 1).sum())
        print(f"OpenOil reference: particles whose status ever changed (e.g. stranded): {n_deactivated_ref}")
    n_stranded_ours = int(ours[ours["time"] == ours["time"].max()]["stranded"].sum())
    print(f"our own simulation: particles stranded by the final hour: {n_stranded_ours}")

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
        assert len(common) == EXPECTED_PARCELS, f"+{h}h: expected {EXPECTED_PARCELS} common particle IDs, got {len(common)}"
        o_at = o_at.loc[common]
        r_at = r_at.loc[common]

        finite = np.isfinite(o_at["lon"]) & np.isfinite(o_at["lat"]) & np.isfinite(r_at["lon"]) & np.isfinite(r_at["lat"])
        n_excluded = int((~finite).sum())
        o_f, r_f = o_at[finite], r_at[finite]
        used_ids = o_f.index

        dist_km = haversine_distance_km(
            o_f["lat"].to_numpy(), o_f["lon"].to_numpy(),
            r_f["lat"].to_numpy(), r_f["lon"].to_numpy(),
        )
        mean_e, median_e, max_e = float(dist_km.mean()), float(np.median(dist_km)), float(dist_km.max())
        worst_pid = used_ids[int(np.argmax(dist_km))]
        excl_note = f"  ({n_excluded} particle(s) excluded: non-finite position in one trajectory)" if n_excluded else ""
        print(f"+{h:>2}h ({t}): mean={mean_e:7.3f} km  median={median_e:7.3f} km  "
              f"max={max_e:7.3f} km (particle {worst_pid}){excl_note}")

        checkpoint_rows.append({
            "checkpoint_hours": h, "time": t,
            "mean_error_km": mean_e, "median_error_km": median_e, "max_error_km": max_e,
            "n_particles_compared": len(used_ids), "n_excluded_nonfinite": n_excluded,
        })
        per_particle_frames.append(pd.DataFrame({
            "checkpoint_hours": h, "particle_id": used_ids, "error_km": dist_km,
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
    if checkpoints_df["n_excluded_nonfinite"].sum() > 0:
        print(f"total particle-checkpoints excluded for non-finite position: "
              f"{int(checkpoints_df['n_excluded_nonfinite'].sum())}")

    section("Investigation: where does the error come from?")
    o0 = ours[ours["time"] == START_TIME].set_index("particle_id")[["lon", "lat"]]
    r0 = ref[ref["time"] == START_TIME].set_index("particle_id")[["lon", "lat"]]
    common0 = o0.index.intersection(r0.index)
    d0 = haversine_distance_km(o0.loc[common0, "lat"], o0.loc[common0, "lon"],
                                r0.loc[common0, "lat"], r0.loc[common0, "lon"])
    print(f"t0 seeding-position check: mean={d0.mean():.6f} km, max={d0.max():.6f} km "
          f"(should be ~0 -- both seeded from the same real parcel lon/lat)")

    one_h = checkpoints_df[checkpoints_df["checkpoint_hours"] == 1]
    if len(one_h):
        print(f"+1h single-step error: mean={one_h['mean_error_km'].iloc[0]:.4f} km, "
              f"max={one_h['max_error_km'].iloc[0]:.4f} km")

    growth = checkpoints_df.set_index("checkpoint_hours")["mean_error_km"]
    ratios = (growth / growth.index.to_series()).round(4)
    print("mean_error_km / elapsed_hours at each checkpoint (roughly constant -> linear/compounding growth; "
          "a sudden jump -> investigate that window, e.g. a coastline-handling divergence):")
    print(ratios.to_string())

    if n_deactivated_ref > 0 or n_stranded_ours > 0:
        print(f"\nCoastline-handling note: reference deactivations={n_deactivated_ref}, "
              f"our strandings={n_stranded_ours} -- since this case starts close to land, any large "
              f"late-checkpoint error is plausibly explained by the two implementations' different "
              f"land data sources (this project's CMEMS-mask vs OpenDrift's real GSHHG coastline), "
              f"not a bug in either. See progress.md for the full write-up.")

    section("DONE -- see progress.md for the full written investigation and verdict")


if __name__ == "__main__":
    main()
