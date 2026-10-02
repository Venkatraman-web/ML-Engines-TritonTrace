"""ow-0008 case -- minimal OpenOil/OpenDrift reference run (F06 equivalent; not a test
file -- CLAUDE.md forbids `tests/`/`test_*.py`).

Mirrors `scripts/f06_openoil_reference.py` exactly (same config rationale -- see that
file's docstring for the full justification of every `set_config` call), pointed at
`data/14th_june/ocean.nc`/`wind.nc` and the 548 real ow-0008 parcels.

Must be run with the Python 3.13 interpreter that has `opendrift` installed:

    py -3.13 scripts/ow0008_openoil_reference.py

Unlike ow-0009 (which never approaches land), ow-0008's parcels start only ~29 km
from the nearest CMEMS-masked land cell and this project's own simulation already
found Beirut Port CRITICAL from t0 -- so OpenDrift's real GSHHG-coastline-based
stranding (`general:coastline_action`, left at its default) may behave differently
from this project's CMEMS-mask-based land classification. That is expected and is
investigated explicitly in the comparison script, not hidden.

Output: `output/ow-0008/openoil_reference.parquet`.
"""

from __future__ import annotations

import json
import time as _time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from opendrift.models.openoil import OpenOil
from opendrift.readers import reader_netCDF_CF_generic

REPO_ROOT = Path(__file__).resolve().parents[1]
OCEAN_PATH = REPO_ROOT / "data" / "14th_june" / "ocean.nc"
WIND_PATH = REPO_ROOT / "data" / "14th_june" / "wind.nc"
PAYLOAD_PATH = REPO_ROOT / "data" / "database_payload.json"
OUTPUT_PATH = REPO_ROOT / "output" / "ow-0008" / "openoil_reference.parquet"

START_TIME = datetime(2019, 6, 14, 20, 35, 58)
DURATION_HOURS = 72
TIME_STEP_SECONDS = 3600
WIND_DRIFT_FACTOR = 0.03
EXPECTED_PARCELS = 548


def load_ow0008_parcels() -> list[dict]:
    with open(PAYLOAD_PATH, encoding="utf-8") as f:
        payload = json.load(f)
    spill = next(s for s in payload["spills"] if s["image_filename"] == "ow-0008.jpg")
    assert spill["timestamp"] == "2019-06-14T20:35:58", spill["timestamp"]
    assert len(spill["parcels"]) == EXPECTED_PARCELS, len(spill["parcels"])
    return spill["parcels"]


def main() -> None:
    parcels = load_ow0008_parcels()
    lons = np.array([p["lon"] for p in parcels])
    lats = np.array([p["lat"] for p in parcels])
    ids = np.array([p["id"] for p in parcels])

    o = OpenOil(loglevel=20)
    o.add_reader([
        reader_netCDF_CF_generic.Reader(str(OCEAN_PATH)),
        reader_netCDF_CF_generic.Reader(str(WIND_PATH)),
    ])

    o.set_config("drift:stokes_drift", False)
    o.set_config("drift:current_uncertainty", 0)
    o.set_config("drift:wind_uncertainty", 0)
    o.set_config("drift:vertical_mixing", False)
    o.set_config("processes:evaporation", False)
    o.set_config("processes:emulsification", False)
    o.set_config("processes:dispersion", False)

    o.seed_elements(
        lon=lons, lat=lats, time=START_TIME,
        wind_drift_factor=WIND_DRIFT_FACTOR, z=0,
    )

    t0 = _time.time()
    o.run(
        duration=timedelta(hours=DURATION_HOURS),
        time_step=TIME_STEP_SECONDS,
        time_step_output=TIME_STEP_SECONDS,
    )
    print(f"OpenOil run complete in {_time.time()-t0:.1f}s wall time")

    ds = o.result[["lon", "lat", "status"]]
    df = ds.to_dataframe().reset_index()
    assert list(ds["trajectory"].values) == list(range(len(parcels))), (
        "OpenDrift's trajectory index no longer matches seeding order -- id mapping below would be wrong"
    )
    id_map = {i: int(ids[i]) for i in range(len(ids))}
    df["particle_id"] = df["trajectory"].map(id_map)
    df = df[["particle_id", "time", "lon", "lat", "status"]]

    n_times = df["time"].nunique()
    print(f"result: {len(df)} rows, {n_times} timestamps, {df['particle_id'].nunique()} particles")
    assert n_times == DURATION_HOURS + 1, f"expected {DURATION_HOURS+1} timestamps, got {n_times}"
    assert df["particle_id"].nunique() == EXPECTED_PARCELS

    n_deactivated = int((df.groupby("particle_id")["status"].nunique() > 1).sum())
    print(f"particles whose OpenDrift status ever changed (e.g. stranded): {n_deactivated} / {EXPECTED_PARCELS}")
    if n_deactivated > 0:
        print("NOTE: OpenDrift's real GSHHG coastline deactivated some particles this project's own "
              "CMEMS-mask-based simulation did not (or vice versa) -- investigated in the comparison script.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
