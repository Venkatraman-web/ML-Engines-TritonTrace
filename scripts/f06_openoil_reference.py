"""F06 -- minimal OpenOil/OpenDrift reference run for the ow-0009 case (not a test
file -- CLAUDE.md forbids `tests/`/`test_*.py`).

This script must be run with the Python 3.13 interpreter that has the real
`opendrift` package installed (verified during this task: `python3`/`py -3.14`, which
the rest of this project's `fwd` package uses, does NOT have `opendrift`; `py -3.13`
does):

    py -3.13 scripts/f06_openoil_reference.py

It is intentionally self-contained -- it does not import anything from `fwd.*` -- both
because `fwd` was built/validated under the other Python interpreter, and per
requirement 2 ("do not change our existing F01-F05 implementation"): this script only
reads the same source files F01-F05 already used (`data/28th_april/ocean.nc`,
`data/28th_april/wind.nc`, `data/database_payload.json`); it never imports, calls, or
modifies any F01-F05 code.

Physics/config matches this project's locked spec (README section 5, CLAUDE.md
section 3/15) as closely as OpenOil's own configuration surface allows:
- `seed:current_drift_factor` (OpenOil default 1) and `seed:wind_drift_factor`
  (explicitly set to 0.03 here) already match `u_oil = uo + 0.03*u10` exactly.
- `drift:advection_scheme` is already `'euler'` by OpenOil's own default, matching
  this project's Euler step.
- `drift:stokes_drift` is turned off (no wave forcing is available to either
  implementation).
- `drift:current_uncertainty` / `drift:wind_uncertainty` are set to 0 (OpenOil's
  defaults are 0.05 / 0.5 -- a *stochastic* perturbation that would make its output
  non-deterministic and not comparable run-to-run; this project's own simulation has
  no such term, so it is disabled here to compare like with like, not to "tune" the
  physics).
- `drift:vertical_mixing` is turned off, keeping elements exactly at the surface
  (z=0), matching this project's strictly-2D surface transport.
- `processes:evaporation`/`emulsification`/`dispersion` are turned off -- this is a
  surface-transport comparison, not an oil-weathering comparison (CLAUDE.md: "no
  weathering" in the baseline).
- `general:coastline_action` is left at OpenOil's own default (`'stranding'`, using
  OpenDrift's built-in GSHHG landmask) -- this is irrelevant to the comparison because
  F03 already established that the ow-0009 case never approaches land during this
  72-hour run (see progress.md), so no coastline-handling difference can appear in
  either implementation's output.

Output: `output/ow-0009/openoil_reference.parquet` -- one row per particle per
recorded hour (`particle_id, time, lon, lat`), `particle_id` mapped from OpenDrift's
0-indexed seeding order back to this project's real parcel IDs (verified in this
task's own smoke test that OpenDrift's `trajectory` dimension preserves seeding
order exactly).
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
OCEAN_PATH = REPO_ROOT / "data" / "28th_april" / "ocean.nc"
WIND_PATH = REPO_ROOT / "data" / "28th_april" / "wind.nc"
PAYLOAD_PATH = REPO_ROOT / "data" / "database_payload.json"
OUTPUT_PATH = REPO_ROOT / "output" / "ow-0009" / "openoil_reference.parquet"

START_TIME = datetime(2019, 4, 28, 15, 38, 41)
DURATION_HOURS = 72
TIME_STEP_SECONDS = 3600
WIND_DRIFT_FACTOR = 0.03


def load_ow0009_parcels() -> list[dict]:
    with open(PAYLOAD_PATH, encoding="utf-8") as f:
        payload = json.load(f)
    spill = next(s for s in payload["spills"] if s["image_filename"] == "ow-0009.jpg")
    assert spill["timestamp"] == "2019-04-28T15:38:41", spill["timestamp"]
    assert len(spill["parcels"]) == 371, len(spill["parcels"])
    return spill["parcels"]


def main() -> None:
    parcels = load_ow0009_parcels()
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
    assert df["particle_id"].nunique() == 371

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
