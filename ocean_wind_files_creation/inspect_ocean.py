from pathlib import Path
import xarray as xr
import numpy as np

# ============================================================
# CONFIG
# ============================================================

FILE = Path("./files/14th_june/ocean.nc")

REQUIRED_VARS = ["uo", "vo"]


# ============================================================
# HELPERS
# ============================================================

def find_coord(ds, candidates):
    for name in candidates:
        if name in ds.coords or name in ds.variables:
            return name
    raise KeyError(
        f"Could not find any of {candidates}. "
        f"Available: {list(ds.variables)}"
    )


def print_range(ds, name):
    values = ds[name].values

    print(
        f"{name}: "
        f"{np.nanmin(values)} -> {np.nanmax(values)}"
    )


# ============================================================
# OPEN DATASET
# ============================================================

if not FILE.exists():
    raise FileNotFoundError(
        f"File not found: {FILE.resolve()}"
    )

print("=" * 80)
print("OPENING FILE")
print("=" * 80)

print("File:", FILE.resolve())

ds = xr.open_dataset(
    FILE,
    decode_times=True
)


# ============================================================
# BASIC DATASET INFO
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
# FIND LAT / LON
# ============================================================

lat_name = find_coord(
    ds,
    ["latitude", "lat", "nav_lat"]
)

lon_name = find_coord(
    ds,
    ["longitude", "lon", "nav_lon"]
)

print("\nLatitude coordinate :", lat_name)
print("Longitude coordinate:", lon_name)


# ============================================================
# LAT / LON INFORMATION
# ============================================================

print("\n" + "=" * 80)
print("SPATIAL GRID")
print("=" * 80)

lat = ds[lat_name].values
lon = ds[lon_name].values

print(
    f"Latitude range : {float(np.min(lat)):.8f} "
    f"-> {float(np.max(lat)):.8f}"
)

print(
    f"Longitude range: {float(np.min(lon)):.8f} "
    f"-> {float(np.max(lon)):.8f}"
)

print(
    "Latitude order :",
    "ascending" if lat[0] < lat[-1] else "descending"
)

print(
    "Longitude order:",
    "ascending" if lon[0] < lon[-1] else "descending"
)


# ============================================================
# GRID SPACING
# ============================================================

print("\n" + "=" * 80)
print("GRID SPACING")
print("=" * 80)

if len(lat) > 1:
    lat_diff = np.diff(lat)

    print(
        "Latitude spacing:"
    )
    print(
        f"  min = {np.min(np.abs(lat_diff)):.10f}"
    )
    print(
        f"  max = {np.max(np.abs(lat_diff)):.10f}"
    )

if len(lon) > 1:
    lon_diff = np.diff(lon)

    print(
        "Longitude spacing:"
    )
    print(
        f"  min = {np.min(np.abs(lon_diff)):.10f}"
    )
    print(
        f"  max = {np.max(np.abs(lon_diff)):.10f}"
    )


# ============================================================
# TIME INFORMATION
# ============================================================

print("\n" + "=" * 80)
print("TIME")
print("=" * 80)

times = ds["time"].values

print("Number of timestamps:", len(times))

print("First timestamp:", times[0])
print("Last timestamp :", times[-1])

if len(times) > 1:

    time_diffs = np.diff(
        times.astype("datetime64[s]")
    )

    unique_diffs = np.unique(time_diffs)

    print("\nUnique time intervals:")

    for diff in unique_diffs:
        print(" ", diff)

    # Check strict ordering
    if np.all(
        time_diffs > np.timedelta64(0, "s")
    ):
        print("\n✅ Time axis is strictly increasing.")
    else:
        print("\n❌ Time axis is NOT strictly increasing.")


# ============================================================
# VARIABLE INFORMATION
# ============================================================

print("\n" + "=" * 80)
print("VARIABLES")
print("=" * 80)

for var_name in ds.data_vars:

    var = ds[var_name]

    print(f"\n[{var_name}]")

    print("  dimensions :", var.dims)
    print("  shape      :", var.shape)
    print("  dtype      :", var.dtype)

    if "units" in var.attrs:
        print("  units      :", var.attrs["units"])

    if "long_name" in var.attrs:
        print("  long_name  :", var.attrs["long_name"])


# ============================================================
# REQUIRED VARIABLE CHECK
# ============================================================

print("\n" + "=" * 80)
print("REQUIRED VARIABLES")
print("=" * 80)

for var_name in REQUIRED_VARS:

    if var_name not in ds.data_vars:
        print(f"❌ {var_name} missing")
    else:
        print(f"✅ {var_name} present")


# ============================================================
# VALUE STATISTICS
# ============================================================

print("\n" + "=" * 80)
print("VALUE STATISTICS")
print("=" * 80)

for var_name in REQUIRED_VARS:

    if var_name not in ds:
        continue

    data = ds[var_name].values

    finite = np.isfinite(data)

    finite_values = data[finite]

    print(f"\n[{var_name}]")

    print(
        f"  min       : {np.min(finite_values):.8f}"
    )

    print(
        f"  max       : {np.max(finite_values):.8f}"
    )

    print(
        f"  mean      : {np.mean(finite_values):.8f}"
    )

    print(
        f"  median    : {np.median(finite_values):.8f}"
    )

    print(
        f"  NaN count : {np.isnan(data).sum()}"
    )

    print(
        f"  Inf count : {np.isinf(data).sum()}"
    )

    print(
        f"  finite %  : "
        f"{100 * finite.mean():.4f}%"
    )


# ============================================================
# NaN MAP PER TIMESTAMP
# ============================================================

print("\n" + "=" * 80)
print("NAN COUNTS PER TIMESTAMP")
print("=" * 80)

for var_name in REQUIRED_VARS:

    if var_name not in ds:
        continue

    print(f"\n[{var_name}]")

    for i, timestamp in enumerate(times):

        data = ds[var_name].isel(time=i).values

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

sample_time_indices = [
    0,
    len(times) // 2,
    len(times) - 1
]

sample_lat_indices = [
    0,
    len(lat) // 2,
    len(lat) - 1
]

sample_lon_indices = [
    0,
    len(lon) // 2,
    len(lon) - 1
]

for ti in sample_time_indices:

    print(f"\nTime: {times[ti]}")

    for yi in sample_lat_indices:

        for xi in sample_lon_indices:

            lat_value = lat[yi]
            lon_value = lon[xi]

            u_value = ds["uo"].isel(
                time=ti,
                **{
                    lat_name: yi,
                    lon_name: xi
                }
            ).item()

            v_value = ds["vo"].isel(
                time=ti,
                **{
                    lat_name: yi,
                    lon_name: xi
                }
            ).item()

            print(
                f"  lat={lat_value:.6f}, "
                f"lon={lon_value:.6f} | "
                f"uo={u_value} | "
                f"vo={v_value}"
            )


# ============================================================
# DATASET ATTRIBUTES
# ============================================================

print("\n" + "=" * 80)
print("GLOBAL ATTRIBUTES")
print("=" * 80)

for key, value in ds.attrs.items():
    print(f"{key}: {value}")


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

print(f"File       : {FILE.name}")
print(f"Timestamps : {len(times)}")
print(
    f"Grid       : "
    f"{len(lat)} × {len(lon)}"
)

print(
    f"Latitude   : "
    f"{lat.min():.6f} -> {lat.max():.6f}"
)

print(
    f"Longitude  : "
    f"{lon.min():.6f} -> {lon.max():.6f}"
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

print("\n✅ Inspection complete.")