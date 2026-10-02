"""Case registry for the forward-propagation pilot.

One source of truth for which real observed slick, forcing files, and observation
time each named case uses, so the per-case runner scripts don't each hard-code these
facts separately (and risk drifting out of sync with each other). Every value here
was verified against `data/database_payload.json` and the actual NetCDF files before
being added -- never invented (CLAUDE.md: "do not generate/randomize particles").
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class CaseConfig:
    case_id: str
    image_filename: str  # matches database_payload.json's spills[].image_filename
    ocean_path: Path
    wind_path: Path
    observation_time: np.datetime64
    total_parcels: int  # expected parcel count, asserted against the loaded data


CASES: dict[str, CaseConfig] = {
    "ow-0009": CaseConfig(
        case_id="ow-0009",
        image_filename="ow-0009.jpg",
        ocean_path=REPO_ROOT / "data" / "28th_april" / "ocean.nc",
        wind_path=REPO_ROOT / "data" / "28th_april" / "wind.nc",
        observation_time=np.datetime64("2019-04-28T15:38:41", "ns"),
        total_parcels=371,
    ),
    "ow-0008": CaseConfig(
        case_id="ow-0008",
        image_filename="ow-0008.jpg",
        ocean_path=REPO_ROOT / "data" / "14th_june" / "ocean.nc",
        wind_path=REPO_ROOT / "data" / "14th_june" / "wind.nc",
        observation_time=np.datetime64("2019-06-14T20:35:58", "ns"),
        total_parcels=548,
    ),
}


def get_case(case_id: str) -> CaseConfig:
    if case_id not in CASES:
        raise KeyError(f"Unknown case '{case_id}'. Known cases: {sorted(CASES)}")
    return CASES[case_id]
