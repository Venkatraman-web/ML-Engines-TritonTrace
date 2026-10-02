# CLAUDE.md — Forward Propagation Rules

## Mission

Implement the SIH 2026 forward-propagation application exactly according to `README.md` and `tasks.md`.

The goal is a working, reproducible pipeline from the 548 observed oil-slick elements to a validated 72-hour trajectory, hotspot-alert timeline, and clear application-ready animation.

Work task-by-task. Do not silently change scientific assumptions.

---

## 1. Hard scientific locks

```text
Start             = 2019-04-28 20:35 UTC
Initial elements  = 548
Duration          = 72 hours
Propagation step  = 1 hour
Recorded states   = 73
End               = 2019-05-01 20:35 UTC
```

The directory name `28th_june` is only a working-directory name. It does not change the experiment date.

---

## 2. Forcing locks

Use:

```text
data/ocean.nc
files/28th_june/wind.nc
```

CMEMS:

```text
uo, vo
```

ERA5:

```text
u10, v10
```

AOI:

```text
North = 36.372
West  = 27.121
South = 30.1875
East  = 36.088
```

Keep source files on native grids. Do not permanently regrid them merely for convenience.

---

## 3. Physics locks

Baseline surface horizontal velocity:

```text
u_oil = uo + 0.03*u10
v_oil = vo + 0.03*v10
```

Use current drift factor `1.0` and wind drift factor `0.03`.

No Stokes drift because no wave/Stokes forcing is available.

Do not add weathering, vertical mixing/advection, wave entrainment, stochastic diffusion, random particle motion, or extra windage unless a new task explicitly changes the specification.

OpenOil/OpenDrift is the behavioral reference. The OpenOil example documents `wind_drift_factor = 0.03` and `current_drift_factor = 1`. citeturn427526search0turn427526search1

---

## 4. Time handling

The particle clock is authoritative for simulation output.

Do not snap particle times to forcing times.

At `2019-04-28 20:35`:

```text
CMEMS: 20:30 -> 21:30
ERA5 : 20:00 -> 21:00
```

Interpolate to the exact requested timestamp. Repeat for every hourly state.

---

## 5. Spatial interpolation

Particles use continuous geographic coordinates.

Use continuous interpolation rather than nearest-neighbor as the baseline.

Correctly handle CMEMS coordinate ordering and ERA5 descending latitude.

Reuse compatible Stage III interpolation/geometry code rather than copying equivalent logic.

---

## 6. CMEMS NaN rules — mandatory

Never globally do:

```python
uo.fillna(0)
vo.fillna(0)
```

Never create artificial land currents.

Required decision flow:

```text
land/coast?
  yes -> strand/stop
  no  -> normal CMEMS interpolation
             |
           NaN?
         /       \
       no        yes
       |          |
    use current   nearby valid-ocean fallback
                     |
                 sufficient neighbors?
                  /             \
                yes              no
                 |                |
               use          explicit environmental
                            lookup failure
```

Fallback defaults:

```text
3x3 -> 5x5
minimum valid neighbors = 4
weight = 1/(d^2 + epsilon)
maximum search radius = configurable
```

Only cells with finite `uo` and `vo` may contribute.

Record:

```text
DIRECT
NEIGHBOR_FALLBACK
INVALID_ENVIRONMENT
LAND_STRANDED
```

A NaN caused by a mixed ocean/land interpolation stencil must not automatically kill a particle when the particle is independently classified as water.

---

## 7. Coastline rules

Project requirement:

> Oil reaching land/coastline stops permanently.

Do not use OpenDrift `previous` behavior.

Use proper land/coast classification.

When stranded:

```text
valid = False
stranded = True
```

Retain the last position in trajectory history.

OpenDrift documents `stranding` as deactivating particles whose next position is on land. citeturn427526search7

If a one-hour movement can jump across a narrow island/coastline, an internal collision substep is allowed only with explicit documentation while keeping the recorded output cadence hourly.

---

## 8. Hotspot rules

Input:

```text
regional_alert_geofences_wide.geojson
```

Expected structure:

```text
22 polygons
11 hotspots
100 km Watch Zone
40 km Critical Strike Zone
```

Use polygon geometries directly. Do not recreate circles from center/radius.

For every hourly state and every hotspot:

```text
if ANY active oil particle is inside the 40 km polygon:
    CRITICAL / RED
elif ANY active oil particle is inside the 100 km polygon:
    WATCH / YELLOW
else:
    CLEAR
```

Do not use the slick centroid as the sole criterion.

Record:

```text
status
minimum_particle_distance_km
watch_particle_count
critical_particle_count
first_watch_time
first_critical_time
```

Multiple hotspots may be active simultaneously.

`liability_multiplier` is metadata only and is not part of the v1 alert decision.

---

## 9. Animation rules

The main user-facing output is a clear 3-day animation.

Must show:

```text
oil slick
hotspot geofences
red/yellow alert state
UTC timestamp
coastline/land
legend
```

Scientific states remain hourly. Animation-only interpolation may be used for smooth playback, but it must not alter the simulation or create extra scientific alert events.

Optional KDE/density rendering is visualization-only and must never replace particle positions for alert computation.

---

## 10. Code-creation rules

Create implementation code only when needed.

Allowed artifacts:

```text
.py modules
.ipynb notebooks
```

Do NOT create:

```text
tests/ directory
test_*.py
*_test.py
large synthetic test-data trees
```

Validation must use assertions/checks inside normal `.py` scripts, Jupyter notebooks, or reproducible CLI runs.

Prefer a small number of coherent modules. Do not create dozens of micro-files.

---

## 11. Reuse existing code

Before creating a module, inspect the repository.

Reuse compatible Stage III components for interpolation, geometry/Earth conversion, environment access, and boundary concepts.

Do not modify Stage III behavior as a side effect of forward-propagation work.

If reuse is impossible, document why.

---

## 12. Task protocol

Read `tasks.md` completely before starting.

Execute tasks strictly in order.

For every task:

1. inspect current repository state;
2. read relevant existing modules;
3. implement the smallest complete change;
4. validate using a `.py` script or notebook as appropriate;
5. inspect numerical output;
6. fix failures instead of weakening validation;
7. update `progress.md` with files changed, validation command, evidence, and status;
8. only then continue.

Do not mark a task PASS because code exists.

---

## 13. No silent scientific changes

Never silently change:

```text
start/end time
particle count
hourly cadence
wind drift factor
current factor
NaN policy
land behavior
hotspot thresholds
alert logic
forcing files
```

If a scientifically necessary change appears, document it in `progress.md` and update the specification before proceeding.

---

## 14. Numerical sanity rules

Final runs must verify:

```text
no non-finite coordinates for active particles
no non-finite oil velocity for active particles
no particle moves through land
73 recorded states
548 initial particle IDs accounted for
no duplicate particle/time records
reasonable speed ranges
fallback usage recorded
hotspot state changes recorded
```

Do not suppress warnings without investigation.

---

## 15. Reproducibility

The baseline run must be deterministic.

Same inputs + same command = same output.

Do not introduce random motion.

---

## 16. Reference comparison

Before completion, compare against OpenOil/OpenDrift or an independently calculated equivalent using the same:

```text
initial positions
forcing
wind/current factors
interpolation assumptions
timestep
land behavior
```

Compare at:

```text
+1h
+6h
+12h
+24h
+48h
+72h
```

Report quantitative differences. Do not invent reference results.

---

## 17. Definition of done

Complete only when:

```text
all tasks PASS
progress.md updated
548 particles initialized
72 propagation steps completed
73 states stored
forcing validated
NaN fallback validated
land behavior validated
hotspot alerts validated
40 km red / 100 km yellow validated
3-day animation generated
OpenOil reference comparison completed
outputs reproducible
```
