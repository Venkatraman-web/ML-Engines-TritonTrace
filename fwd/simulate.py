"""F03 -- 72-hour forward simulation runner, reused across cases (ow-0009, ow-0008, ...).

Pure orchestration: runs F02's `fwd.step.step_particle` once per hour, for every real
parcel of the given case, for 72 hours (73 recorded states including the initial one).
No new physics, stranding, or forcing logic is introduced here -- everything numerical
comes from F01 (`fwd.forcing.ForcingAdapter`) and F02 (`fwd.step`) unchanged.

`run_simulation` has no hard-coded I/O paths (CLAUDE.md: prefer explicit parameters
over hidden constants) -- writing the result to disk is the caller's job (see
`scripts/f03_run_simulation.py`).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from fwd.forcing import INVALID_ENVIRONMENT, NEIGHBOR_FALLBACK, ForcingAdapter
from fwd.step import DT_SECONDS, LAND_STRANDED, ParticleState, step_particle

REPO_ROOT = Path(__file__).resolve().parents[1]
OW0009_OBS_TIME = np.datetime64("2019-04-28T15:38:41", "ns")  # kept for backward compatibility
DURATION_HOURS = 72
INITIAL = "INITIAL"  # this module's own bookkeeping value for the unstepped t0 row


def load_spill_parcels(
    image_filename: str,
    payload_path: Path | None = None,
    expected_timestamp: str | None = None,
    expected_count: int | None = None,
) -> list[dict]:
    """The real parcels (`id, lon, lat, mass_kg, confidence`) for one named spill
    (matched by `image_filename`, e.g. `"ow-0009.jpg"`/`"ow-0008.jpg"`) from
    `database_payload.json`. If given, `expected_timestamp`/`expected_count` are
    asserted against the loaded spill -- a cheap guard against silently running the
    wrong case's data (CLAUDE.md: fail loudly rather than silently)."""
    path = payload_path if payload_path is not None else REPO_ROOT / "data" / "database_payload.json"
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    spill = next(s for s in payload["spills"] if s["image_filename"] == image_filename)
    if expected_timestamp is not None:
        assert spill["timestamp"] == expected_timestamp, (spill["timestamp"], expected_timestamp)
    if expected_count is not None:
        assert len(spill["parcels"]) == expected_count, (len(spill["parcels"]), expected_count)
    return spill["parcels"]


def load_ow0009_parcels(payload_path: Path | None = None) -> list[dict]:
    """The 371 real ow-0009 parcels -- thin, backward-compatible wrapper around
    `load_spill_parcels` for existing ow-0009 scripts."""
    return load_spill_parcels(
        "ow-0009.jpg", payload_path, expected_timestamp="2019-04-28T15:38:41", expected_count=371,
    )


def _trajectory_row(particle_id, time, lon, lat, valid, stranded, forcing_mode,
                     uo, vo, u10, v10, oil_u, oil_v, mass_kg, confidence) -> dict:
    return {
        "particle_id": particle_id, "time": time, "lon": lon, "lat": lat,
        "valid": valid, "stranded": stranded, "forcing_mode": forcing_mode,
        "uo": uo, "vo": vo, "u10": u10, "v10": v10, "oil_u": oil_u, "oil_v": oil_v,
        "mass_kg": mass_kg, "confidence": confidence,
    }


def run_simulation(
    adapter: ForcingAdapter | None = None,
    parcels: list[dict] | None = None,
    start_time=OW0009_OBS_TIME,
    duration_hours: int = DURATION_HOURS,
    dt_seconds: float = DT_SECONDS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the deterministic 72-hour forward simulation.

    Returns `(trajectory_df, hourly_diagnostics_df)`:
    - `trajectory_df`: one row per particle per recorded hour (`duration_hours + 1`
      states x `len(parcels)` particles), columns as `_trajectory_row` above.
    - `hourly_diagnostics_df`: one row per propagation step (`duration_hours` rows),
      with `active_count, stranded_count, fallback_count, invalid_forcing_count,
      newly_stranded_count`.
    """
    if adapter is None:
        adapter = ForcingAdapter()
    if parcels is None:
        parcels = load_ow0009_parcels()

    start_time = np.datetime64(start_time, "ns")
    mass_by_id = {p["id"]: p["mass_kg"] for p in parcels}
    confidence_by_id = {p["id"]: p["confidence"] for p in parcels}

    states = [ParticleState(p["id"], p["lon"], p["lat"]) for p in parcels]

    rows = [
        _trajectory_row(
            s.particle_id, start_time, s.lon, s.lat, True, False, INITIAL,
            np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
            mass_by_id[s.particle_id], confidence_by_id[s.particle_id],
        )
        for s in states
    ]

    diagnostics = []
    t = start_time
    for _ in range(duration_hours):
        t_next = t + np.timedelta64(int(dt_seconds), "s")
        new_states = []
        for s in states:
            ns, rec = step_particle(adapter, s, t, dt_seconds=dt_seconds)
            new_states.append(ns)
            rows.append(
                _trajectory_row(
                    ns.particle_id, t_next, ns.lon, ns.lat, ns.valid, ns.stranded,
                    rec.forcing_mode, rec.uo, rec.vo, rec.u10, rec.v10, rec.oil_u, rec.oil_v,
                    mass_by_id[ns.particle_id], confidence_by_id[ns.particle_id],
                )
            )

        active = sum(1 for ns in new_states if ns.valid and not ns.stranded)
        stranded = sum(1 for ns in new_states if ns.stranded)
        # per-step forcing_mode reflects this step's own lookup, so counting straight
        # from the rows just appended for this t_next gives per-step (not cumulative) counts
        step_modes = [r["forcing_mode"] for r in rows[-len(states):]]
        fallback = step_modes.count(NEIGHBOR_FALLBACK)
        invalid = step_modes.count(INVALID_ENVIRONMENT)
        newly_stranded = step_modes.count(LAND_STRANDED) + invalid
        diagnostics.append({
            "time": t_next, "active_count": active, "stranded_count": stranded,
            "fallback_count": fallback, "invalid_forcing_count": invalid,
            "newly_stranded_count": newly_stranded,
        })

        states = new_states
        t = t_next

    trajectory_df = pd.DataFrame(rows)
    diagnostics_df = pd.DataFrame(diagnostics)
    return trajectory_df, diagnostics_df
