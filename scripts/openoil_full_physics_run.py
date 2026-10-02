"""Real OpenOil/OpenDrift forward-propagation run WITH weathering (evaporation,
emulsification, dispersion, seeded with the real total observed oil mass) and
horizontal diffusion enabled, for both ow-0009 and ow-0008 -- not a test file
(CLAUDE.md forbids `tests/`/`test_*.py`).

**This is a deliberately DIFFERENT, separate deliverable from this project's own
locked, from-scratch, deterministic model.** It does not modify or replace anything
under `output/ow-0009/`, `output/ow-0008/`, or any `fwd/*` code -- it is a full
substitution of OpenDrift's real physics engine for comparison/exploration purposes,
requested explicitly after the locked model was finalized. Output goes to a new,
separate top-level directory (`openoil_full_physics_output/`), never mixed with the
locked model's outputs.

**Non-determinism, and how it's handled**: weathering-driven horizontal diffusion and
OpenDrift's own current/wind uncertainty terms are inherently stochastic. A FIXED
random seed (`RANDOM_SEED`) is passed to `OpenOil(seed=...)` (which calls
`np.random.seed()` internally -- confirmed by reading OpenDrift's own source) so THIS
specific run is reproducible if re-executed with the same seed, even though it is not
comparable step-by-step with the locked model's bit-exact determinism.

**How the real total observed oil mass is used** (`mass_kg` per parcel, summed):
OpenOil does NOT accept a per-element `mass_oil` array directly -- its own element
metadata marks `mass_oil` `'seed': False` ("Properties which may not be provided by
user"), and reading `OpenOil.seed_elements()`'s source confirms it unconditionally
overwrites any `mass_oil` kwarg with
`mass_oil_per_element = m3_per_hour * duration_hours(=1h, instantaneous) / number * density`.
So the real total observed mass (`sum(parcel.mass_kg)`) is seeded via `m3_per_hour`
(computed as `total_mass_kg / oil_density`, with `oil_density` probed from a throwaway
1-element seed first, since OpenOil computes it internally from the oil type at 285K)
and `number=N` (the real parcel count) -- verified in this task to correctly split the
real total mass equally across all N real parcels
(`mass_oil_per_element == total_mass_kg / N`, checked numerically before writing this
script). Per-parcel mass heterogeneity (each real parcel's own individual `mass_kg`,
which reflects image-classification confidence, not a physically metered
per-parcel amount) is not preserved -- OpenOil's own oil-budget model has no concept
of that; only the physically meaningful total is conserved and then weathered.

**Config**: almost entirely OpenOil's own defaults (weathering -- evaporation,
emulsification, dispersion -- is already True by default; `drift:current_uncertainty`/
`wind_uncertainty` are left at their defaults, contributing additional realistic
stochastic spread; `drift:vertical_mixing` is left at its default True, so dispersed
oil may realistically go subsurface, unlike the locked model's strict 2D-surface-only
assumption; `general:coastline_action` is left at its default `'stranding'`, using
OpenDrift's own real GSHHG coastline -- more accurate than this project's own
CMEMS-grid-resolution land mask). The one explicit override:
`environment:fallback:horizontal_diffusivity` is raised from OpenOil's default of 0 to
`HORIZONTAL_DIFFUSIVITY_M2S` (a representative literature ballpark for coastal ocean
turbulent diffusivity, not a value calibrated against this specific region -- flagged
honestly, not presented as precisely correct). Stokes drift is left at OpenOil's own
default (nominally `True`) but is functionally inert here: no wave reader is attached
and `drift:use_tabularised_stokes_drift` (which would estimate it from wind) is left
at its own default `False`, so wave-related environment variables fall back to 0 --
consistent with this project's standing "no Stokes drift, no wave forcing available"
reasoning, not silently reintroduced.

Must be run with the Python 3.13 interpreter that has `opendrift` installed:

    py -3.13 scripts/openoil_full_physics_run.py

Output: `openoil_full_physics_output/<case>/trajectory.parquet` (+ `.csv`) -- one row
per particle per recorded hour, columns `particle_id, time, lon, lat, z, status,
moving, mass_oil, mass_evaporated, mass_dispersed, mass_biodegraded,
fraction_evaporated, water_fraction`.
"""

from __future__ import annotations

import json
import time as _time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from opendrift.models.openoil import OpenOil
from opendrift.readers import reader_netCDF_CF_generic

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "openoil_full_physics_output"
PAYLOAD_PATH = REPO_ROOT / "data" / "database_payload.json"

RANDOM_SEED = 20260926  # fixed, for reproducibility of this stochastic run only
HORIZONTAL_DIFFUSIVITY_M2S = 10.0  # representative coastal-ocean literature ballpark, not calibrated
DURATION_HOURS = 72
TIME_STEP_SECONDS = 3600
WIND_DRIFT_FACTOR = 0.03

CASES = {
    "ow-0009": dict(
        image_filename="ow-0009.jpg",
        ocean_path=REPO_ROOT / "data" / "28th_april" / "ocean.nc",
        wind_path=REPO_ROOT / "data" / "28th_april" / "wind.nc",
        start_time=datetime(2019, 4, 28, 15, 38, 41),
        expected_parcels=371,
    ),
    "ow-0008": dict(
        image_filename="ow-0008.jpg",
        ocean_path=REPO_ROOT / "data" / "14th_june" / "ocean.nc",
        wind_path=REPO_ROOT / "data" / "14th_june" / "wind.nc",
        start_time=datetime(2019, 6, 14, 20, 35, 58),
        expected_parcels=548,
    ),
}


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def load_parcels(image_filename: str, expected_count: int) -> list[dict]:
    with open(PAYLOAD_PATH, encoding="utf-8") as f:
        payload = json.load(f)
    spill = next(s for s in payload["spills"] if s["image_filename"] == image_filename)
    assert len(spill["parcels"]) == expected_count, (len(spill["parcels"]), expected_count)
    return spill["parcels"]


def probe_oil_density(ocean_path: Path, wind_path: Path, start_time: datetime) -> float:
    """OpenOil computes oil density internally (NOAA model, at 285K) from the oil
    type -- not user-settable directly -- so it must be read back from a throwaway
    1-element seed before the real run, to correctly compute m3_per_hour for the real
    total mass below."""
    probe = OpenOil(loglevel=50, seed=RANDOM_SEED)
    probe.add_reader([
        reader_netCDF_CF_generic.Reader(str(ocean_path)),
        reader_netCDF_CF_generic.Reader(str(wind_path)),
    ])
    probe.seed_elements(lon=[34.0], lat=[34.0], time=start_time, number=1, m3_per_hour=1)
    return float(probe.elements_scheduled.density)


def run_case(case_id: str, cfg: dict) -> None:
    section(f"{case_id}: full-physics OpenOil run (weathering + diffusion)")
    parcels = load_parcels(cfg["image_filename"], cfg["expected_parcels"])
    n = len(parcels)
    ids = np.array([p["id"] for p in parcels])
    lons = np.array([p["lon"] for p in parcels])
    lats = np.array([p["lat"] for p in parcels])
    total_mass_kg = float(sum(p["mass_kg"] for p in parcels))
    print(f"parcels: {n}, real total observed mass: {total_mass_kg:.2f} kg")

    density = probe_oil_density(cfg["ocean_path"], cfg["wind_path"], cfg["start_time"])
    m3_per_hour = total_mass_kg / density
    print(f"probed oil density: {density:.2f} kg/m3  ->  m3_per_hour = {m3_per_hour:.4f} "
          f"(instantaneous seed, duration_hours=1h internally, so total seeded mass = "
          f"m3_per_hour * 1h * density = {m3_per_hour * density:.2f} kg, matching total_mass_kg)")

    o = OpenOil(loglevel=20, seed=RANDOM_SEED)
    o.add_reader([
        reader_netCDF_CF_generic.Reader(str(cfg["ocean_path"])),
        reader_netCDF_CF_generic.Reader(str(cfg["wind_path"])),
    ])
    o.set_config("environment:fallback:horizontal_diffusivity", HORIZONTAL_DIFFUSIVITY_M2S)
    print(f"config: horizontal_diffusivity={HORIZONTAL_DIFFUSIVITY_M2S} m2/s (explicit override); "
          f"everything else at OpenOil defaults (weathering ON, current/wind uncertainty ON, "
          f"vertical_mixing ON, coastline_action='stranding' with real GSHHG)")

    o.seed_elements(
        lon=lons, lat=lats, time=cfg["start_time"],
        wind_drift_factor=WIND_DRIFT_FACTOR, z=0, number=n, m3_per_hour=m3_per_hour,
    )
    seeded_mass = float(np.atleast_1d(o.elements_scheduled.mass_oil)[0]) * n
    print(f"verified: mass_oil per element = {float(np.atleast_1d(o.elements_scheduled.mass_oil)[0]):.4f} kg "
          f"x {n} = {seeded_mass:.2f} kg (should equal {total_mass_kg:.2f} kg)")
    assert abs(seeded_mass - total_mass_kg) < 1.0, "seeded total mass does not match the real observed total"

    t0 = _time.time()
    o.run(duration=timedelta(hours=DURATION_HOURS), time_step=TIME_STEP_SECONDS, time_step_output=TIME_STEP_SECONDS)
    print(f"run complete in {_time.time()-t0:.1f}s wall time")

    ds = o.result
    print(f"status_categories: {o.status_categories}")
    keep_vars = ["lon", "lat", "z", "status", "moving", "mass_oil", "mass_evaporated",
                 "mass_dispersed", "mass_biodegraded", "fraction_evaporated", "water_fraction"]
    df = ds[keep_vars].to_dataframe().reset_index()
    assert list(ds["trajectory"].values) == list(range(n)), "seeding order no longer matches trajectory index"
    id_map = {i: int(ids[i]) for i in range(n)}
    df["particle_id"] = df["trajectory"].map(id_map)
    df = df[["particle_id", "time"] + keep_vars]

    n_times = df["time"].nunique()
    print(f"result: {len(df)} rows, {n_times} timestamps, {df['particle_id'].nunique()} particles")
    assert n_times == DURATION_HOURS + 1
    assert df["particle_id"].nunique() == n

    final = df[df["time"] == df["time"].max()]
    print(f"final hour: total mass_oil={final['mass_oil'].sum():.2f} kg, "
          f"total mass_evaporated={final['mass_evaporated'].sum():.2f} kg, "
          f"total mass_dispersed={final['mass_dispersed'].sum():.2f} kg, "
          f"total mass_biodegraded={final['mass_biodegraded'].sum():.2f} kg")
    mass_balance = (final["mass_oil"] + final["mass_evaporated"] + final["mass_dispersed"]
                    + final["mass_biodegraded"]).sum()
    print(f"mass balance check: sum of all mass categories at final hour = {mass_balance:.2f} kg "
          f"(should equal total seeded mass {total_mass_kg:.2f} kg)")
    print(f"status values present: {sorted(df['status'].unique())}  "
          f"moving==0 (stranded/stopped) at final hour: {int((final['moving']==0).sum())}")
    print(f"z range across full run: [{df['z'].min():.3f}, {df['z'].max():.3f}] m "
          f"(0 = surface, negative = subsurface -- weathering-driven vertical entrainment)")

    out_dir = OUTPUT_ROOT / case_id
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / "trajectory.parquet"
    df.to_parquet(parquet_path, index=False)
    csv_dir = out_dir / "csv"
    csv_dir.mkdir(exist_ok=True)
    df.to_csv(csv_dir / "trajectory.csv", index=False)
    print(f"wrote {parquet_path} and csv/trajectory.csv")


def main() -> None:
    for case_id, cfg in CASES.items():
        run_case(case_id, cfg)
    section("ALL CASES COMPLETE")


if __name__ == "__main__":
    main()
