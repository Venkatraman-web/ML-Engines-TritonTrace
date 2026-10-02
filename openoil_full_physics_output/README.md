# Full-Physics OpenOil/OpenDrift Deliverable

## 1. What this is

This directory contains a **second, independent forward-propagation run** for the
same two oil-slick cases (`ow-0009`, `ow-0008`) already handled by this project's
own locked model in `output/`. Instead of this project's from-scratch,
deterministic surface-drift implementation, this run uses the **real OpenOil model
from OpenDrift** (`opendrift==1.14.11`) end-to-end, with its full weathering
(evaporation, emulsification, dispersion) and horizontal diffusion physics turned
on — not just used as a reference check, but as the actual engine producing every
output in this directory.

**This is not a replacement for the locked model.** `output/ow-0009/`,
`output/ow-0008/`, and every `fwd/*` module are untouched. This deliverable was
built after the locked model was finalized, specifically to show what a
full-physics oil-weathering run looks like for the same two spills, using the same
forcing data and the same wind/current drift factors so the two are comparable.

---

## 2. Why it's different from the locked model

| | Locked model (`output/`) | This deliverable (`openoil_full_physics_output/`) |
|---|---|---|
| Engine | This project's own from-scratch implementation | Real OpenOil/OpenDrift (`opendrift==1.14.11`) |
| Surface drift | `u_oil = uo + 0.03*u10`, `v_oil = vo + 0.03*v10` | Same factors (`current_drift_factor=1.0`, `wind_drift_factor=0.03`) |
| Weathering (evaporation/emulsification/dispersion) | None — explicitly out of scope | **On**, using OpenOil's own NOAA-based oil-weathering model |
| Mass tracking | Not modeled | Full oil mass budget (`mass_oil`, `mass_evaporated`, `mass_dispersed`, `mass_biodegraded`) using each spill's real observed total mass |
| Horizontal diffusion / random spreading | None (fully deterministic) | On — literature-ballpark turbulent diffusivity, `10 m²/s` |
| Current/wind uncertainty | None | OpenOil defaults on (adds small stochastic current/wind noise) |
| Vertical behavior | Strict 2D surface-only | Vertical mixing on (OpenOil default) — dispersed oil may go subsurface |
| Coastline / stranding | This project's own CMEMS-grid-resolution land mask | OpenDrift's own real GSHHG global coastline |
| Determinism | Fully bit-reproducible | Reproducible only via a **fixed random seed** (see §5) |
| Hotspot alert logic | Same fraction-of-active-particles rule | Same fraction-of-active-particles rule, recomputed against this trajectory |

Everything not listed above (forcing files, AOI, time range, particle start
positions, hourly output cadence) is identical to the locked model, so the two
runs are directly comparable.

---

## 3. Inputs

Same as the locked model, per case:

```text
ow-0009: data/28th_april/ocean.nc, data/28th_april/wind.nc
         start 2019-04-28 15:38:41 UTC, 371 real observed parcels
ow-0008: data/14th_june/ocean.nc, data/14th_june/wind.nc
         start 2019-06-14 20:35:58 UTC, 548 real observed parcels
```

Initial parcel positions and each parcel's real observed `mass_kg` come from
`data/database_payload.json` — the same source-of-truth used by the locked model.

---

## 4. How the real oil mass was used

OpenOil's `mass_oil` element property cannot be set per-parcel directly by the
caller — reading OpenOil's own source confirms `mass_oil` is marked `'seed':
False` ("properties which may not be provided by user") and is unconditionally
recomputed internally during `seed_elements()` from `m3_per_hour`, an implied
1-hour seeding duration, and the oil density OpenOil itself derives from the oil
type at 285K.

So instead of trying to force per-parcel mass values, the **real total observed
mass** for each spill (`sum(parcel.mass_kg)` across all parcels) was seeded
correctly:

1. A throwaway 1-element probe seed reads back OpenOil's own internally-computed
   oil density for the configured oil type.
2. `m3_per_hour = total_mass_kg / density` is computed from that density.
3. The real run is seeded with `number = N` (the real parcel count) and that
   `m3_per_hour`, which OpenOil then splits evenly across all `N` real parcel
   positions.
4. The seeded total mass was verified numerically to match the real observed
   total to within 1 kg before the actual 72-hour run was executed.

**Limitation, stated plainly:** each individual parcel's own `mass_kg` (which
reflects image-classification confidence, not a physically metered per-parcel
quantity) is *not* individually preserved — only the physically meaningful
**total** mass is conserved and then weathered realistically from there. This is
a genuine limitation of OpenOil's own API, not an oversight in this run.

---

## 5. Configuration and reproducibility

Almost entirely OpenOil's own defaults. The one explicit override:

```text
environment:fallback:horizontal_diffusivity = 10.0 m²/s
```

(OpenOil's own default is `0`, i.e. no random-walk diffusion at all.) This value
is a representative literature ballpark for coastal-ocean turbulent diffusivity —
**not calibrated to this specific region** — flagged honestly rather than
presented as precisely correct.

Everything else uses OpenOil's built-in defaults:

- weathering (evaporation, emulsification, dispersion) — **on**
- biodegradation — **off**
- current uncertainty (0.05) and wind uncertainty (0.5) — **on**
- vertical mixing — **on** (dispersed oil may realistically move subsurface,
  unlike the locked model's strict 2D assumption)
- coastline action — `'stranding'` against OpenDrift's own real GSHHG coastline
- Stokes drift — nominally on by OpenOil default, but functionally inert here: no
  wave reader is attached and tabularised-Stokes-from-wind estimation is left at
  its own default `False`, so wave-related environment variables fall back to
  zero — consistent with this project's standing "no Stokes drift, no wave
  forcing available" position, not silently reintroduced.

**Non-determinism**: weathering-driven diffusion and OpenOil's current/wind
uncertainty terms are inherently stochastic (OpenOil calls `np.random.seed()`
internally when constructed with a `seed=` argument — confirmed by reading its
source). A fixed seed, `20260926`, was passed to `OpenOil(seed=20260926)` for
both cases, so **this specific run is exactly reproducible if re-executed with
the same seed** — but it is not, and cannot be, bit-comparable to the locked
model's own full determinism, and it is not an ensemble: this is a **single run
per case**, deliberately not a Monte Carlo / probability-cone spread.

---

## 6. Directory layout (per case)

```text
openoil_full_physics_output/<case>/
├── trajectory.parquet              one row per particle per recorded hour:
│                                    particle_id, time, lon, lat, z, status,
│                                    moving, mass_oil, mass_evaporated,
│                                    mass_dispersed, mass_biodegraded,
│                                    fraction_evaporated, water_fraction
├── hotspot_alerts.parquet          one row per hotspot per recorded hour:
│                                    hotspot_name, time, status,
│                                    particles_inside_40km/100km,
│                                    fraction_inside_40km/100km,
│                                    min_distance_km, first_watch_time,
│                                    first_critical_time
├── run_summary.json                experiment config, physics config,
│                                    final mass budget, triggered hotspots,
│                                    output file index, non-determinism notes
├── openoil_full_physics_<case>_animation.mp4
│                                    2-panel animation: left = map (basemap,
│                                    hotspot zones, drift trails, particles
│                                    coloured by fraction_evaporated, colour
│                                    scale on the side); right = stacked-area
│                                    oil mass budget over time (remaining /
│                                    evaporated / dispersed / biodegraded, %
│                                    of initial mass). Opens with a 3s regional
│                                    overview of all 11 hotspots, 1.5s eased
│                                    zoom-in to the slick, then hourly
│                                    propagation at 10 fps.
├── frames/
│   ├── initial.png                 t0 snapshot
│   ├── final.png                   t+72h snapshot
│   ├── watch_<hotspot>.png         first hour any hotspot reached WATCH
│   └── critical_<hotspot>.png      first hour any hotspot reached CRITICAL
│                                    (omitted if the case never goes CRITICAL)
└── csv/
    ├── trajectory.csv              same data as trajectory.parquet, CSV form
    ├── hotspot_alerts.csv          same data as hotspot_alerts.parquet, CSV form
    └── hotspot_status_timeline.csv 11 hotspots x 73 timestamps, status only
                                     (CLEAR/WATCH/CRITICAL), pivoted wide
```

---

## 7. Results summary

### Mass budget (kg, at t+72h)

| Case | Seeded | Oil remaining | Evaporated | Dispersed | Biodegraded |
|---|---|---|---|---|---|
| ow-0009 | 88,715.45 | 79,246.41 (89.3%) | 4,577.49 (5.2%) | 4,891.56 (5.5%) | 0 (0%) |
| ow-0008 | 106,833.62 | 93,048.66 (87.1%) | 5,028.62 (4.7%) | 8,756.35 (8.2%) | 0 (0%) |

Evaporated/dispersed/biodegraded totals are strictly monotonic non-decreasing at
every one of the 73 recorded hours for both cases (weathering is irreversible, as
it should be). Zero particles stranded on land in either case; `z` stayed at 0.0
throughout both runs (no subsurface entrainment materialized despite vertical
mixing being enabled).

### Hotspots triggered

```text
ow-0009: Beirut Port (WATCH), Cape Greco MPA (CRITICAL), Palm Islands Nature Reserve (WATCH)
ow-0008: Beirut Port (CRITICAL), Palm Islands Nature Reserve (WATCH)
```

Recomputed using this project's own hotspot alert rule (status is CRITICAL when
≥30% of active particles are inside a hotspot's 40 km zone, WATCH at ≥30% inside
the 100 km zone) applied to this trajectory — same rule as the locked model, just
evaluated against OpenOil's own particle positions instead. Results are broadly
consistent with the locked model's own findings for the same two cases, as
expected since both share the same forcing and drift factors.

---

## 8. How to reproduce

```text
# 1. Run the physics (requires the py -3.13 interpreter with opendrift installed)
py -3.13 scripts/openoil_full_physics_run.py

# 2. Build the beautified outputs (normal python3 interpreter)
python3 scripts/build_openoil_full_physics_outputs.py
```

Both scripts loop over both cases automatically. Re-running step 1 with the same
`RANDOM_SEED` (20260926, set at the top of `openoil_full_physics_run.py`)
reproduces the same trajectory, mass budget, and hotspot results.

---

## 9. Known limitations

- **Not deterministic in the same sense as the locked model** — reproducible only
  via the fixed seed, not bit-identical across different seeds.
- **Per-parcel mass heterogeneity is not preserved** — only the real observed
  total mass per case is conserved (see §4).
- **Horizontal diffusivity (10 m²/s) is a literature ballpark**, not calibrated
  against this specific stretch of the Eastern Mediterranean.
- **Single run per case only** — no ensemble, no Monte Carlo, no probability-cone
  uncertainty spread, by explicit design.
- Uses OpenDrift's own real coastline (GSHHG) for stranding, which is a different
  (generally more accurate) land boundary than the locked model's CMEMS-grid-
  resolution land mask — so stranding behavior between the two deliverables is
  not expected to match exactly even for the same drift path.
