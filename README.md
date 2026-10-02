# Forward Propagation — SIH 2026

## 1. Objective

Build the forward-propagation application for the observed oil slick so the application can:

1. Start from the **548 observed slick elements** at **2019-04-28 20:35 UTC**.
2. Propagate the slick forward for **72 hours**, with a **1-hour recorded timestep**.
3. Use the prepared **CMEMS ocean current** and **ERA5 10 m wind** forcing.
4. Follow the **OpenOil/OpenDrift surface horizontal drift concept** for v1.
5. Stop/strand particles that actually reach land/coastline.
6. Handle CMEMS masked cells intelligently without globally replacing NaNs.
7. Generate a clear 3-day slick animation for the application.
8. Detect proximity to all supplied regional hotspots and update their UI state to **yellow (100 km Watch)** or **red (40 km Critical)**.

This section is complete only when the simulation, alert engine, outputs, and animation have all been validated.

---

## 2. Exact experiment

```text
Initial observation : 2019-04-28 20:35 UTC
Initial particles   : 548
Propagation period  : 72 hours
Propagation steps   : 72
Recorded states     : 73
Output cadence      : 1 hour
Final state         : 2019-05-01 20:35 UTC
```

The directory name `28th_june` is only a working-directory name. It does **not** change the experiment date.

The 73 recorded timestamps are:

```text
T0  = 2019-04-28 20:35
T1  = 2019-04-28 21:35
...
T72 = 2019-05-01 20:35
```

The scientific simulation remains hourly. A video renderer may interpolate visually between saved states for smoother playback, but it must not create extra scientific simulation states or extra scientific alert events.

---

## 3. Input data

### 3.1 CMEMS ocean forcing

```text
data/ocean.nc
```

Variables:

```text
uo(time, lat, lon)  # eastward surface current, m/s
vo(time, lat, lon)  # northward surface current, m/s
```

Prepared coverage:

```text
time: 2019-04-28 00:30 -> 2019-05-01 23:30
lat : 149 points
lon : 216 points
```

CMEMS contains masked/NaN cells. Those values must be preserved in the source file.

### 3.2 ERA5 wind forcing

```text
files/28th_june/wind.nc
```

Variables:

```text
u10(time, latitude, longitude)  # 10 m eastward wind, m/s
v10(time, latitude, longitude)  # 10 m northward wind, m/s
```

Prepared coverage:

```text
time: 2019-04-28 00:00 -> 2019-05-01 23:00
latitude : 25 points, descending
longitude: 36 points, ascending
```

### 3.3 Initial slick

Use the project's actual observed slick input containing **548 elements**. Preserve the observed geometry and identifiers. Do not randomly perturb or resample it for the baseline run.

### 3.4 Hotspots

```text
regional_alert_geofences_wide.geojson
```

The file contains:

```text
22 polygon features
11 unique hotspots
2 zones per hotspot
```

Each hotspot has:

```text
100 km Watch Zone          -> yellow
40 km Critical Strike Zone -> red
```

The GeoJSON properties include `name`, `zone_level`, `type`, `radius_km`, `liability_multiplier`, and `color`. Preserve these properties. `liability_multiplier` is metadata and is **not part of the alert decision in v1**.

Hotspots:

```text
Port Said (Suez Canal)
Alexandria Port
Vasiliko Oil Terminal
Beirut Port
Ceyhan Oil Terminal
Haifa Port
Akamas Peninsula (Turtle Nesting)
Palm Islands Nature Reserve
Rosh HaNikra Marine Reserve
Nile Delta Coastal Lagoons
Cape Greco MPA
```

Use the supplied polygon geometries directly. Do not recreate approximate circles from hotspot center coordinates.

---

## 4. Common forcing AOI

```text
North = 36.372
West  = 27.121
South = 30.1875
East  = 36.088
```

CMEMS and ERA5 remain on their native grids. Do not permanently regrid the source files merely to make their dimensions identical.

---

## 5. Forward physics

For v1, implement the OpenOil/OpenDrift surface horizontal drift concept using current plus a fraction of the wind vector.

```text
current_drift_factor = 1.0
wind_drift_factor    = 0.03
```

Therefore:

```text
u_oil = uo + 0.03 * u10
v_oil = vo + 0.03 * v10
```

No Stokes drift is used because the current forcing package contains no wave/Stokes-drift forcing.

The OpenDrift documentation describes the surface element motion as current plus a fraction of wind and documents `0.03` as the OpenOil wind-drift-factor value in its OpenOil example output. citeturn427526search0turn427526search1

Use Euler as the baseline advection scheme unless the project explicitly changes the numerical scheme. OpenDrift documents Euler as an available/default scheme in its examples. citeturn427526search6

This v1 is a **surface horizontal transport experiment**, not a full oil-weathering simulation.

Do not add evaporation, emulsification, dissolution, biodegradation, vertical mixing, vertical advection, wave entrainment, Stokes drift, stochastic diffusion, or arbitrary windage to the baseline.

---

## 6. Time interpolation

The forcing clocks are different from the particle clock.

At the initial slick time:

```text
Particle: 20:35
CMEMS   : 20:30 and 21:30
ERA5    : 20:00 and 21:00
```

The forcing adapter must interpolate to the exact requested timestamp rather than snapping to a forcing record.

The same logic must work at every hourly particle state.

---

## 7. Spatial interpolation

Particles have continuous `(lon, lat)` coordinates.

### CMEMS

Use the existing Stage III interpolation machinery when compatible. Evaluate `uo` and `vo` at the particle location rather than snapping to the nearest grid cell.

### ERA5

Use continuous interpolation on the native 0.25° grid and correctly handle descending latitude.

---

## 8. CMEMS NaN / land policy

This is one of the most important parts of the implementation.

### Never do

```python
uo = uo.fillna(0)
vo = vo.fillna(0)
```

Do not create permanent artificial currents over land.

### Required distinction

```text
particle location
       |
       v
land/coast test
   /         \
LAND        WATER
 |             |
STOP       normal CMEMS interpolation
                |
             valid?
            /      \
          YES       NO
           |         |
        use it    local fallback
                     |
              enough valid ocean cells?
                  /             \
                YES              NO
                 |                |
          use fallback      environmental failure
```

### Water-location fallback

If the particle is classified as **water**, but the CMEMS interpolation stencil contains NaNs, estimate the current from nearby valid ocean cells only.

Default v1 configuration:

```text
neighborhoods              = 3x3, then 5x5
minimum_valid_neighbors    = 4
weight                      = 1 / (d^2 + epsilon)
maximum_search_radius       = configurable
```

Only cells where both `uo` and `vo` are finite may be used.

The fallback is an interpolation aid for a water location. It is **not permission to move oil across land**.

Record whether a forcing lookup used:

```text
DIRECT
NEIGHBOR_FALLBACK
INVALID_ENVIRONMENT
LAND_STRANDED
```

If no trustworthy fallback exists, do not silently substitute zero. Record an explicit environmental lookup failure and apply the documented simulation policy.

---

## 9. Coastline behavior

The desired project behavior is:

> If the oil slick reaches coastline/land, it stops/strands there permanently.

Use a proper land/coastline check. Do not use OpenDrift's `previous` behavior for this project. OpenDrift documents `stranding` as deactivating particles whose next position is on land. citeturn427526search7

For a stranded particle:

```text
valid    = False
stranded = True
```

Keep its last position in the trajectory history.

If a 1-hour movement can jump across a narrow island/coastline, an internal collision-check/substep may be added **without changing the 1-hour recorded cadence**, but only with explicit documentation in `progress.md`.

---

## 10. Hotspot alert logic

For every hourly simulation state, evaluate all 11 hotspots independently.

Use the actual polygons from the GeoJSON.

### Alert state

```text
greater than 100 km          -> CLEAR
within 100 km, outside 40 km -> WATCH / YELLOW
within 40 km                 -> CRITICAL / RED
```

Operationally, because the GeoJSON already provides the zones:

```text
if ANY active oil particle is inside the 40 km polygon:
    CRITICAL
elif ANY active oil particle is inside the 100 km polygon:
    WATCH
else:
    CLEAR
```

Do not use the slick centroid as the sole alert criterion.

This allows one hotspot to be critical while another is only on watch.

### UI appearance

Match the application's existing visual language:

```text
CLEAR:
    normal geofence outline; no active fill

WATCH:
    100 km zone = semi-transparent yellow fill/outline

CRITICAL:
    100 km boundary retained for context
    40 km zone = strong semi-transparent red fill/outline
```

The application's current legend should use:

```text
Watch Zone (100 km)
Critical Strike Zone (40 km)
```

### Alert metadata

For each hotspot and hourly state, record:

```text
status
minimum_particle_distance_km
particles_in_watch_zone
particles_in_critical_zone
first_watch_time
first_critical_time
```

Multiple hotspots may be active simultaneously.

---

## 11. Animation requirements

The main application output is a clear 3-day oil-slick animation.

The animation must show:

- oil slick evolution from 28 Apr 20:35 to 1 May 20:35;
- coastline/land;
- all hotspot geofences;
- current hotspot state;
- timestamp on every frame;
- clear visual distinction between oil and hotspot zones;
- smooth playback.

The scientific simulation remains hourly. For smooth playback, the renderer may interpolate particle positions between recorded hourly states, but alert state changes remain tied to the actual saved simulation states.

For visualization, a particle layer plus an optional smooth density/KDE layer may be used. Any KDE/density layer is a **visualization layer only** and must not replace the actual particle trajectory data or alter alert decisions.

---

## 12. Required outputs

A successful run should produce at least:

```text
Forward trajectory dataset
Hourly diagnostics
Hotspot alert timeline
Animation/video or reproducible frame sequence
Initial slick plot
Final slick plot
Trajectory/track plot
```

Recommended machine-readable output fields:

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

Hotspot output should contain one record per hotspot per recorded hour.

---

## 13. Validation approach — no test files

This project intentionally does **not** create a `tests/` directory or standalone `test_*.py` files.

Validation must use only:

- normal `.py` implementation scripts with explicit assertions where appropriate;
- Jupyter notebooks for numerical/scientific checks;
- direct CLI runs and logged evidence.

Do not create test files just for the sake of having a test suite.

Suggested notebooks/scripts when useful:

```text
notebooks/01_forward_validation.ipynb
notebooks/02_hotspot_validation.ipynb
notebooks/03_animation_review.ipynb
```

Use them only when a task genuinely benefits from them; do not create empty or redundant notebooks.

---

## 14. Completion criteria

Forward propagation is complete only when:

```text
[ ] 548 initial elements loaded correctly
[ ] exact start/end timestamps verified
[ ] 72 hourly propagation intervals completed
[ ] 73 hourly states recorded
[ ] CMEMS/ERA5 forcing validated
[ ] exact-time temporal interpolation validated
[ ] spatial interpolation validated
[ ] landmask/coastline behavior validated
[ ] NaN-hole fallback validated
[ ] OpenOil-style velocity validated
[ ] no artificial land current created
[ ] hotspot alerts validated for all 11 hotspots
[ ] 40 km red / 100 km yellow behavior validated
[ ] first-entry times recorded
[ ] complete trajectories saved
[ ] diagnostics generated
[ ] clear 3-day animation generated
[ ] reference OpenOil/OpenDrift comparison completed
[ ] final application-ready outputs reproduced from a single command
```

---

## 15. References

Official OpenDrift/OpenOil references:

- https://opendrift.github.io/tutorial.html
- https://opendrift.github.io/autoapi/opendrift/models/openoil/openoil/index.html
- https://opendrift.github.io/_modules/opendrift/models/openoil/openoil.html
- https://opendrift.github.io/gallery/example_coastline_options.html
- https://opendrift.github.io/gallery/example_openoil_sample_output.html
