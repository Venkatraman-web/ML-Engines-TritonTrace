import cdsapi
from pathlib import Path
import xarray as xr
import numpy as np

# ============================================================
# CONFIG
# ============================================================

YEAR = "2019"
MONTH = "06"

DAYS = [
    "14",
    "15",
    "16",
    "17",
]

TIMES = [
    f"{hour:02d}:00"
    for hour in range(24)
]

# Existing common Ocean + Wind AOI
# [North, West, South, East]
AOI = [
    36.372,
    27.121,
    30.1875,
    36.088,
]

# Output directory
OUTPUT_DIR = Path("./files/14th_june")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Final output
OUTPUT_FILE = OUTPUT_DIR / "wind.nc"


# ============================================================
# CDS CLIENT
# ============================================================

# Uses credentials from ~/.cdsapirc
# Do NOT hardcode your CDS API key in the script.
c = cdsapi.Client(
    url="https://cds.climate.copernicus.eu/api", key="11ba0656-221f-4992-a558-c994bd62aa77"
)


# ============================================================
# DOWNLOAD INFORMATION
# ============================================================

print("=" * 80)
print("DOWNLOADING ERA5 WIND DATA")
print("=" * 80)

print("\nPeriod:")
print("  2019-06-14 -> 2019-06-17")

print("\nVariables:")
print("  - 10m_u_component_of_wind")
print("  - 10m_v_component_of_wind")

print("\nAOI:")
print(f"  North: {AOI[0]}")
print(f"  West : {AOI[1]}")
print(f"  South: {AOI[2]}")
print(f"  East : {AOI[3]}")

print("\nOutput:")
print(f"  {OUTPUT_FILE.resolve()}")


# ============================================================
# DOWNLOAD
# ============================================================

print("\n" + "=" * 80)
print("REQUESTING ERA5 DATA FROM CDS")
print("=" * 80)

c.retrieve(
    "reanalysis-era5-single-levels",
    {
        "product_type": "reanalysis",
        "format": "netcdf",

        "variable": [
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
        ],

        "year": YEAR,
        "month": MONTH,
        "day": DAYS,
        "time": TIMES,

        # [North, West, South, East]
        "area": AOI,
    },
    str(OUTPUT_FILE),
)

print("\n✅ Download complete.")
print(OUTPUT_FILE.resolve())


# ============================================================
# OPEN DATASET
# ============================================================

print("\n" + "=" * 80)
print("INSPECTING DOWNLOADED FILE")
print("=" * 80)

ds = xr.open_dataset(
    OUTPUT_FILE,
    decode_times=True
)

print("\nOriginal dataset:")
print(ds)


# ============================================================
# NORMALIZE TIME DIMENSION
# ============================================================

print("\n" + "=" * 80)
print("NORMALIZING TIME")
print("=" * 80)

if "valid_time" in ds.dims:

    print("Found 'valid_time' dimension.")
    ds = ds.rename({
        "valid_time": "time"
    })

    print("Renamed valid_time -> time")

elif "time" in ds.dims:

    print("Found 'time' dimension.")

else:

    raise ValueError(
        "Dataset has neither 'time' nor 'valid_time' "
        f"as a dimension.\nDimensions: {ds.sizes}"
    )


# ============================================================
# VARIABLE CHECK
# ============================================================

REQUIRED_VARS = [
    "u10",
    "v10",
]

print("\nChecking required variables...")

for var in REQUIRED_VARS:

    if var not in ds.data_vars:

        raise KeyError(
            f"{var} not found.\n"
            f"Available variables: {list(ds.data_vars)}"
        )

    print(f"✅ {var} present")


# ============================================================
# COORDINATE CHECK
# ============================================================

if "latitude" not in ds.coords:
    raise KeyError("latitude coordinate not found.")

if "longitude" not in ds.coords:
    raise KeyError("longitude coordinate not found.")


# ============================================================
# LOAD
# ============================================================

ds = ds.load()


# ============================================================
# TIME VALIDATION
# ============================================================

print("\n" + "=" * 80)
print("TIME VALIDATION")
print("=" * 80)

times = ds.time.values

print("First timestamp:", times[0])
print("Last timestamp :", times[-1])
print("Count          :", len(times))

EXPECTED_HOURS = 96

if len(times) != EXPECTED_HOURS:

    raise ValueError(
        f"Expected {EXPECTED_HOURS} timestamps, "
        f"got {len(times)}"
    )

print("Timestamp count: PASS ✅")


# ============================================================
# HOURLY REGULARITY
# ============================================================

time_diff = np.diff(times)

expected_interval = np.timedelta64(1, "h")

if not np.all(
    time_diff == expected_interval
):

    raise ValueError(
        "Time axis is not strictly hourly."
    )

print("Hourly interval: PASS ✅")


# ============================================================
# DUPLICATE TIMESTAMPS
# ============================================================

if (
    len(np.unique(times))
    != len(times)
):

    raise ValueError(
        "Duplicate timestamps detected."
    )

print("Duplicate timestamps: NONE ✅")


# ============================================================
# SPATIAL EXTENT
# ============================================================

print("\n" + "=" * 80)
print("SPATIAL EXTENT")
print("=" * 80)

lat = ds["latitude"]
lon = ds["longitude"]

lat_min = float(lat.min())
lat_max = float(lat.max())

lon_min = float(lon.min())
lon_max = float(lon.max())

print(
    f"\nLatitude : {lat_min:.6f} -> {lat_max:.6f}"
)

print(
    f"Longitude: {lon_min:.6f} -> {lon_max:.6f}"
)

print("\nRequested AOI:")

print(
    f"Latitude : {AOI[2]} -> {AOI[0]}"
)

print(
    f"Longitude: {AOI[1]} -> {AOI[3]}"
)


# ============================================================
# AOI VALIDATION
# ============================================================

lat_ok = (
    lat_min >= AOI[2]
    and
    lat_max <= AOI[0]
)

lon_ok = (
    lon_min >= AOI[1]
    and
    lon_max <= AOI[3]
)

if not lat_ok:
    raise ValueError(
        "Latitude AOI validation failed."
    )

if not lon_ok:
    raise ValueError(
        "Longitude AOI validation failed."
    )

print("\nLatitude AOI : PASS ✅")
print("Longitude AOI: PASS ✅")


# ============================================================
# WIND STATISTICS
# ============================================================

print("\n" + "=" * 80)
print("WIND STATISTICS")
print("=" * 80)

for variable in REQUIRED_VARS:

    data = ds[variable]

    print(f"\n{variable}:")

    print(
        f"  min  = "
        f"{float(data.min(skipna=True)):.6f} m/s"
    )

    print(
        f"  max  = "
        f"{float(data.max(skipna=True)):.6f} m/s"
    )

    print(
        f"  mean = "
        f"{float(data.mean(skipna=True)):.6f} m/s"
    )

    print(
        f"  std  = "
        f"{float(data.std(skipna=True)):.6f} m/s"
    )

    print(
        f"  NaNs = "
        f"{int(data.isnull().sum())}"
    )


# ============================================================
# GRID INFORMATION
# ============================================================

print("\n" + "=" * 80)
print("GRID INFORMATION")
print("=" * 80)

print("Latitude points :", len(lat))
print("Longitude points:", len(lon))

print(
    "Latitude order  :",
    "ascending" if lat.values[0] < lat.values[-1]
    else "descending"
)

print(
    "Longitude order :",
    "ascending" if lon.values[0] < lon.values[-1]
    else "descending"
)


# ============================================================
# FINAL DATASET
# ============================================================

print("\n" + "=" * 80)
print("FINAL DATASET")
print("=" * 80)

print(ds)


# ============================================================
# CLOSE
# ============================================================

ds.close()

print("\n" + "=" * 80)
print("COMPLETE ✅")
print("=" * 80)

print("\nFinal file:")
print(OUTPUT_FILE.resolve())

print("\nCoverage:")
print("  Start: 2019-06-14 00:00")
print("  End  : 2019-06-17 23:00")
print("  Hours: 96")

print("\nVariables:")
print("  u10 = 10m eastward wind component")
print("  v10 = 10m northward wind component")