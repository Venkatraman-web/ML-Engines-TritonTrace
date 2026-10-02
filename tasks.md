# Forward Propagation — Compact Task Plan

Claude must execute these tasks **strictly in order**.

There are intentionally only **9 tasks**. Do not split them into dozens of micro-tasks.

For each task, Claude must:

1. inspect the repository before editing;
2. implement the task;
3. validate it using a normal `.py` script or notebook when useful;
4. inspect the numerical output;
5. update `progress.md`;
6. mark PASS only when the acceptance criteria are satisfied.

**Do not create test files or a `tests/` directory.**

---

## F00 — Lock experiment, inputs, and architecture

### Objective

Establish the exact experiment and inspect existing Stage III code before creating duplicate modules.

### Lock

```text
start = 2019-04-28 20:35 UTC
particles = 548
duration = 72 h
step/output = 1 h
end = 2019-05-01 20:35 UTC
```

Inputs:

```text
data/ocean.nc
files/28th_june/wind.nc
regional_alert_geofences_wide.geojson
```

Physics:

```text
current factor = 1.0
wind factor = 0.03
Stokes = OFF
weathering/vertical physics = OFF
```

### Work

Inspect and document:

- initial 548-particle source format;
- reusable Stage III environment/interpolation/geometry modules;
- repository conventions;
- output directory convention.

### Validation

Create/update one small configuration `.py` module if required and use assertions/prints to verify the locked values.

### Acceptance

No important experiment parameter is ambiguous and no redundant architecture is created.

---

## F01 — Build the forcing adapter and interpolation pipeline

### Objective

Create one clean environmental-forcing interface for CMEMS + ERA5.

### Requirements

Support:

```text
forcing(lon, lat, time)
    -> uo, vo, u10, v10, forcing_mode
```

Implement:

- native-grid reading;
- exact-time interpolation;
- continuous spatial interpolation;
- descending ERA5 latitude;
- CMEMS interpolation using existing Stage III logic where appropriate.

At 20:35:

```text
CMEMS 20:30 -> 21:30
ERA5 20:00 -> 21:00
```

### Validation

Use a notebook or `.py` validation script with synthetic linear fields to prove time and spatial interpolation numerically.

### Acceptance

Known synthetic values are recovered within documented numerical tolerance and real forcing can be queried at arbitrary particle coordinates/times.

---

## F02 — Implement land classification and CMEMS NaN fallback

### Objective

Make environmental lookup robust near CMEMS masks without creating artificial land currents.

### Required logic

```text
land -> strand/stop
water + direct current -> use direct value
water + NaN stencil -> nearby valid-ocean fallback
no trustworthy fallback -> explicit environmental failure
```

Fallback:

```text
3x3 -> 5x5
minimum 4 valid neighbors
inverse-distance weighting 1/(d^2 + epsilon)
configurable maximum radius
```

Only finite `uo`/`vo` ocean cells may contribute.

### Validation

Use a notebook or `.py` script with synthetic masks covering:

- normal ocean;
- isolated NaN surrounded by ocean;
- coastline/land;
- large masked area;
- insufficient valid neighbors.

### Acceptance

A water point is not stopped merely because an interpolation stencil contains a NaN, but a land point cannot receive a fabricated current.

---

## F03 — Implement OpenOil-style one-hour particle motion

### Objective

Connect forcing to the horizontal oil-velocity and geographic update.

### Formula

```text
u_oil = uo + 0.03*u10
v_oil = vo + 0.03*v10
```

Use:

```text
1-hour recorded step = 3600 seconds
```

Reuse Stage III Earth-coordinate conversion where appropriate.

Land behavior:

```text
next position on land -> stranded permanently
```

### Validation

Run one particle for one hour and show:

```text
initial position
CMEMS current
ERA5 wind
wind contribution
oil velocity
displacement
final position
land status
forcing mode
```

Also perform a simple hand-calculated east-only/north-only displacement check.

### Acceptance

One-step numbers are traceable and physically dimensionally correct.

---

## F04 — Run the complete 548-particle 72-hour simulation

### Objective

Scale the validated one-step logic to the complete observed slick.

### Requirements

- load exactly 548 observed elements;
- preserve IDs and geometry;
- start at 20:35 UTC;
- run 72 one-hour propagation intervals;
- save 73 states including the initial state;
- retain inactive/stranded particle history.

### Output fields

At minimum:

```text
particle_id
time
lon
lat
valid
stranded
uo
vo
u10
v10
oil_u
oil_v
forcing_mode
```

### Validation

Use one normal `.py` run or notebook to verify:

```text
73 timestamps
548 initial IDs
no duplicated particle/time rows
no missing hourly states
no active particle with non-finite position
```

### Acceptance

The full deterministic forward run completes successfully.

---

## F05 — Build the hotspot alert engine

### Objective

Connect the trajectory output to the supplied GeoJSON alert geofences.

### Input

```text
regional_alert_geofences_wide.geojson
```

Expected:

```text
22 polygons
11 hotspots
100 km Watch
40 km Critical
```

### Logic

For every hourly state and hotspot:

```text
ANY active particle in 40 km -> CRITICAL / RED
ELSE any active particle in 100 km -> WATCH / YELLOW
ELSE -> CLEAR
```

Use polygon membership. Do not use slick centroid as the sole criterion.

Record:

```text
status
minimum particle distance
watch particle count
critical particle count
first watch time
first critical time
```

### Acceptance

Multiple hotspots can be simultaneously active and the alert state is reproducible from saved trajectories + GeoJSON alone.

---

## F06 — Produce diagnostics and application-ready data outputs

### Objective

Create reproducible machine-readable outputs for the application and debugging.

### Required diagnostics

Per hour:

```text
active count
stranded count
fallback count
environmental-failure count
mean current speed
mean wind speed
mean oil speed
slick centroid
slick bounding box
```

Per hotspot:

```text
status
minimum distance
watch/critical particle counts
first-entry timestamps
```

### Validation

Use assertions and summary tables in a notebook or `.py` script.

### Acceptance

A single run produces all required numeric outputs without manual editing.

---

## F07 — Build the clear 3-day visualization and animation

### Objective

Generate the main application-facing visualization.

### Required visual content

- moving oil slick;
- coastline/land;
- all 11 hotspots;
- 100 km yellow zone;
- 40 km red zone;
- timestamp;
- clear legend;
- visible alert-state changes.

### Alert styling

```text
CLEAR    -> outline only
WATCH    -> yellow 100 km shading
CRITICAL -> red 40 km shading + outer yellow context
```

The supplied GeoJSON geometries must be used directly.

### Smoothness

The scientific trajectory is hourly. The video renderer may add animation-only interpolation for smooth playback but must not alter the saved simulation or create additional scientific alert states.

Optional KDE/density may be rendered for visual clarity only.

### Acceptance

A clear, readable 3-day animation is generated from the saved trajectory and alert outputs.

---

## F08 — OpenOil/OpenDrift reference comparison

### Objective

Demonstrate that the implementation follows the intended OpenOil/OpenDrift behavior.

### Compare

Use equivalent:

```text
initial positions
CMEMS/ERA5 forcing
wind factor
current factor
timestep
interpolation assumptions
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

### Acceptance

Quantitative position differences are recorded and any unexplained divergence is resolved or explicitly documented.

Do not claim exact agreement without evidence.

---

## F09 — Final integration and completion

### Objective

Make the entire forward-propagation section reproducible and application-ready.

### Work

- run the complete pipeline from clean inputs;
- verify all outputs exist;
- verify diagnostics and hotspot alerts;
- inspect the animation manually;
- ensure no tests/test files were created;
- update README/CLAUDE/tasks/progress if implementation details changed;
- record final commands and results.

### Final acceptance

All previous tasks are PASS and the following are demonstrated:

```text
548 particles
72 hourly propagation intervals
73 recorded states
correct date range
CMEMS + ERA5 forcing
OpenOil-style surface drift
correct CMEMS NaN handling
coastline stranding
11 hotspot alert tracking
40 km red / 100 km yellow
3-day clear animation
reference comparison
reproducible final run
```
