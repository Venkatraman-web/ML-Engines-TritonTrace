from pathlib import Path
import xarray as xr
import numpy as np

# ============================================================
# CONFIG
# ============================================================

FILE = Path("./files/14th_june/wind.nc")

REQUIRED_VARS = ["u10", "v10"]


# ============================================================
# OPEN DATASET
# ============================================================

if not FILE.exists():
    raise FileNotFoundError(
        f"File not found:\n{FILE.resolve()}"
    )

print("=" * 80)
print("OPENING WIND FILE")
print("=" * 80)

print("File:", FILE.resolve())

ds = xr.open_dataset(
    FILE,
    decode_times=True
)


# ============================================================
# BASIC DATASET
# ============================================================

print("\n" + "=" * 80)
print("DATASET")
print("=" * 80)

print(ds)


# ============================================================
# DIMENSIONS
# ============================================================

print("\n" + "=" * 80)
print("DIMENSIONS")
print("=" * 80)

for dim, size in ds.sizes.items():
    print(f"{dim:15s}: {size}")


# ============================================================
# COORDINATES
# ============================================================

print("\n" + "=" * 80)
print("COORDINATES")
print("=" * 80)

for coord in ds.coords:
    print(
        f"{coord:15s}: "
        f"shape={ds[coord].shape}, "
        f"dtype={ds[coord].dtype}"
    )


# ============================================================
# NORMALIZE TIME DIMENSION
# ============================================================

print("\n" + "=" * 80)
print("TIME DIMENSION")
print("=" * 80)

if "valid_time" in ds.dims:

    print("Found dimension: valid_time")

    ds = ds.rename({
        "valid_time": "time"
    })

    print("Renamed valid_time -> time")

elif "time" in ds.dims:

    print("Found dimension: time")

else:

    raise ValueError(
        "Dataset has neither 'time' nor 'valid_time' "
        f"as a dimension.\nDimensions: {ds.sizes}"
    )


# ============================================================
# REQUIRED VARIABLE CHECK
# ============================================================

print("\n" + "=" * 80)
print("REQUIRED VARIABLES")
print("=" * 80)

for var in REQUIRED_VARS:

    if var not in ds.data_vars:

        raise KeyError(
            f"{var} not found.\n"
            f"Available variables: {list(ds.data_vars)}"
        )

    print(f"✅ {var} present")


# ============================================================
# LAT / LON CHECK
# ============================================================

if "latitude" not in ds.coords:
    raise KeyError("latitude coordinate not found.")

if "longitude" not in ds.coords:
    raise KeyError("longitude coordinate not found.")


# ============================================================
# LOAD DATA
# ============================================================

ds = ds.load()


# ============================================================
# TIME INFORMATION
# ============================================================

print("\n" + "=" * 80)
print("TIME")
print("=" * 80)

times = ds["time"].values

print("Number of timestamps:", len(times))
print("First timestamp     :", times[0])
print("Last timestamp      :", times[-1])


# ============================================================
# TIME REGULARITY
# ============================================================

if len(times) > 1:

    time_diff = np.diff(times)

    unique_intervals = np.unique(
        time_diff
    )

    print("\nUnique time intervals:")

    for interval in unique_intervals:
        print(" ", interval)

    expected_interval = np.timedelta64(1, "h")

    if np.all(
        time_diff == expected_interval
    ):
        print("\n✅ Time axis is hourly.")
    else:
        print("\n❌ Time axis is NOT hourly.")


# ============================================================
# DUPLICATE TIMESTAMPS
# ============================================================

if (
    len(np.unique(times))
    != len(times)
):

    print(
        "\n❌ Duplicate timestamps detected."
    )

else:

    print(
        "\n✅ No duplicate timestamps."
    )


# ============================================================
# SPATIAL EXTENT
# ============================================================

print("\n" + "=" * 80)
print("SPATIAL EXTENT")
print("=" * 80)

lat = ds["latitude"].values
lon = ds["longitude"].values

lat_min = float(np.min(lat))
lat_max = float(np.max(lat))

lon_min = float(np.min(lon))
lon_max = float(np.max(lon))

print(
    f"Latitude : {lat_min:.8f} -> {lat_max:.8f}"
)

print(
    f"Longitude: {lon_min:.8f} -> {lon_max:.8f}"
)


# ============================================================
# AOI VALIDATION
# ============================================================

NORTH = 36.372
WEST  = 27.121
SOUTH = 30.1875
EAST  = 36.088

print("\nRequested AOI:")

print(
    f"Latitude : {SOUTH} -> {NORTH}"
)

print(
    f"Longitude: {WEST} -> {EAST}"
)


lat_ok = (
    lat_min >= SOUTH
    and
    lat_max <= NORTH
)

lon_ok = (
    lon_min >= WEST
    and
    lon_max <= EAST
)

if lat_ok:
    print("\n✅ Latitude AOI check passed.")
else:
    print("\n❌ Latitude AOI check failed.")

if lon_ok:
    print("✅ Longitude AOI check passed.")
else:
    print("❌ Longitude AOI check failed.")


# ============================================================
# LAT / LON ORDER
# ============================================================

print("\n" + "=" * 80)
print("COORDINATE ORDER")
print("=" * 80)

print(
    "Latitude order :",
    "ascending"
    if lat[0] < lat[-1]
    else "descending"
)

print(
    "Longitude order:",
    "ascending"
    if lon[0] < lon[-1]
    else "descending"
)


# ============================================================
# GRID SPACING
# ============================================================

print("\n" + "=" * 80)
print("GRID SPACING")
print("=" * 80)

if len(lat) > 1:

    lat_diff = np.abs(
        np.diff(lat)
    )

    print(
        "Latitude spacing:"
    )

    print(
        f"  min = {lat_diff.min():.10f}"
    )

    print(
        f"  max = {lat_diff.max():.10f}"
    )

if len(lon) > 1:

    lon_diff = np.abs(
        np.diff(lon)
    )

    print(
        "\nLongitude spacing:"
    )

    print(
        f"  min = {lon_diff.min():.10f}"
    )

    print(
        f"  max = {lon_diff.max():.10f}"
    )


# ============================================================
# VARIABLE INFORMATION
# ============================================================

print("\n" + "=" * 80)
print("VARIABLE INFORMATION")
print("=" * 80)

for variable in REQUIRED_VARS:

    data = ds[variable]

    print(f"\n[{variable}]")

    print("Dimensions :", data.dims)
    print("Shape      :", data.shape)
    print("dtype      :", data.dtype)

    print(
        "Units      :",
        data.attrs.get("units", "N/A")
    )

    print(
        "Long name  :",
        data.attrs.get("long_name", "N/A")
    )


# ============================================================
# WIND STATISTICS
# ============================================================

print("\n" + "=" * 80)
print("WIND STATISTICS")
print("=" * 80)

for variable in REQUIRED_VARS:

    data = ds[variable].values

    finite_mask = np.isfinite(data)

    finite_values = data[finite_mask]

    print(f"\n[{variable}]")

    if finite_values.size == 0:

        print("  No finite values!")

        continue

    print(
        f"  min       : "
        f"{finite_values.min():.8f} m/s"
    )

    print(
        f"  max       : "
        f"{finite_values.max():.8f} m/s"
    )

    print(
        f"  mean      : "
        f"{finite_values.mean():.8f} m/s"
    )

    print(
        f"  median    : "
        f"{np.median(finite_values):.8f} m/s"
    )

    print(
        f"  std       : "
        f"{finite_values.std():.8f} m/s"
    )

    print(
        f"  NaN count : "
        f"{np.isnan(data).sum()}"
    )

    print(
        f"  Inf count : "
        f"{np.isinf(data).sum()}"
    )

    print(
        f"  finite %  : "
        f"{100 * finite_mask.mean():.4f}%"
    )


# ============================================================
# WIND SPEED
# ============================================================

print("\n" + "=" * 80)
print("WIND SPEED")
print("=" * 80)

u10 = ds["u10"]
v10 = ds["v10"]

wind_speed = np.sqrt(
    u10 ** 2 +
    v10 ** 2
)

finite_speed = wind_speed.values[
    np.isfinite(wind_speed.values)
]

print(
    f"Min wind speed : "
    f"{finite_speed.min():.8f} m/s"
)

print(
    f"Max wind speed : "
    f"{finite_speed.max():.8f} m/s"
)

print(
    f"Mean wind speed: "
    f"{finite_speed.mean():.8f} m/s"
)

print(
    f"Median speed   : "
    f"{np.median(finite_speed):.8f} m/s"
)


# ============================================================
# NAN COUNTS PER TIMESTAMP
# ============================================================

print("\n" + "=" * 80)
print("NAN COUNTS PER TIMESTAMP")
print("=" * 80)

for variable in REQUIRED_VARS:

    print(f"\n[{variable}]")

    for i, timestamp in enumerate(times):

        data = ds[variable].isel(
            time=i
        ).values

        nan_count = np.isnan(data).sum()

        print(
            f"{timestamp} : "
            f"{nan_count} NaNs"
        )


# ============================================================
# SAMPLE VALUES
# ============================================================

print("\n" + "=" * 80)
print("SAMPLE VALUES")
print("=" * 80)

time_indices = [
    0,
    len(times) // 2,
    len(times) - 1
]

lat_indices = [
    0,
    len(lat) // 2,
    len(lat) - 1
]

lon_indices = [
    0,
    len(lon) // 2,
    len(lon) - 1
]

for ti in time_indices:

    print(
        f"\nTime: {times[ti]}"
    )

    for yi in lat_indices:

        for xi in lon_indices:

            u_value = ds["u10"].isel(
                time=ti,
                latitude=yi,
                longitude=xi
            ).item()

            v_value = ds["v10"].isel(
                time=ti,
                latitude=yi,
                longitude=xi
            ).item()

            print(
                f"  lat={lat[yi]:.6f}, "
                f"lon={lon[xi]:.6f} | "
                f"u10={u_value} | "
                f"v10={v_value}"
            )


# ============================================================
# GLOBAL ATTRIBUTES
# ============================================================

print("\n" + "=" * 80)
print("GLOBAL ATTRIBUTES")
print("=" * 80)

for key, value in ds.attrs.items():

    print(
        f"{key}: {value}"
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

print(
    f"File       : {FILE.name}"
)

print(
    f"Timestamps : {len(times)}"
)

print(
    f"Grid       : "
    f"{len(lat)} × {len(lon)}"
)

print(
    f"Latitude   : "
    f"{lat_min:.6f} -> {lat_max:.6f}"
)

print(
    f"Longitude  : "
    f"{lon_min:.6f} -> {lon_max:.6f}"
)

print(
    f"Time       : "
    f"{times[0]} -> {times[-1]}"
)

print(
    f"Variables  : "
    f"{list(ds.data_vars)}"
)


# ============================================================
# CLOSE
# ============================================================

ds.close()

print("\n✅ Wind inspection complete.")