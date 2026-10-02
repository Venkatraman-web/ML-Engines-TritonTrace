from pathlib import Path
import xarray as xr
import numpy as np

# ============================================================
# CONFIG
# ============================================================

INPUT_DIR = Path("./ocean_file_raw")

OUTPUT_FILE = Path("./files/14th_june/ocean.nc")

# Existing common Ocean + Wind AOI
NORTH = 36.372
WEST  = 27.121
SOUTH = 30.1875
EAST  = 36.088

DATES = [
    "20190614",
    "20190615",
    "20190616",
    "20190617",
]

REQUIRED_VARS = ["uo", "vo"]

EXPECTED_TIMESTAMPS = 96   # 4 days × 24 hours


# ============================================================
# HELPERS
# ============================================================

def find_coord(ds, candidates):
    """Find the first coordinate/variable matching the candidates."""
    for name in candidates:
        if name in ds.coords or name in ds.variables:
            return name

    raise KeyError(
        f"Could not find any of {candidates}. "
        f"Available variables: {list(ds.variables)}"
    )


def make_slice(coord, lower, upper):
    """
    Create a coordinate slice that works for both
    ascending and descending coordinates.
    """
    values = coord.values

    if values[0] < values[-1]:
        return slice(lower, upper)

    return slice(upper, lower)


# ============================================================
# FIND FILES
# ============================================================

all_files = sorted(INPUT_DIR.glob("*.nc"))

if not all_files:
    raise FileNotFoundError(
        f"No NetCDF files found in {INPUT_DIR.resolve()}"
    )

print("Available NetCDF files:")
for file_path in all_files:
    print("   ", file_path.name)


# ============================================================
# SELECT ONE FILE FOR EACH DATE
# ============================================================

selected_files = []

for date in DATES:

    matches = [
        file_path
        for file_path in all_files
        if date in file_path.name
    ]

    if len(matches) == 0:
        raise FileNotFoundError(
            f"No NetCDF file found for date {date}"
        )

    if len(matches) > 1:
        print(f"\nWARNING: Multiple files found for {date}")

        for match in matches:
            print("   ", match.name)

        raise RuntimeError(
            f"Ambiguous file selection for {date}. "
            f"Please keep only one file for this date."
        )

    selected_files.append(matches[0])


print("\nSelected files:")
for file_path in selected_files:
    print("   ", file_path.name)


# ============================================================
# OPEN + AOI FILTER EACH FILE
# ============================================================

filtered_datasets = []

for file_path in selected_files:

    print("\n" + "=" * 60)
    print(f"Processing: {file_path.name}")

    ds = xr.open_dataset(
        file_path,
        decode_times=True
    )

    # --------------------------------------------------------
    # REQUIRED VARIABLES
    # --------------------------------------------------------

    missing = [
        var
        for var in REQUIRED_VARS
        if var not in ds.data_vars
    ]

    if missing:
        ds.close()

        raise KeyError(
            f"{file_path.name} is missing variables: {missing}"
        )

    # --------------------------------------------------------
    # FIND LAT / LON
    # --------------------------------------------------------

    lat_name = find_coord(
        ds,
        ["latitude", "lat", "nav_lat"]
    )

    lon_name = find_coord(
        ds,
        ["longitude", "lon", "nav_lon"]
    )

    print(f"Latitude coordinate : {lat_name}")
    print(f"Longitude coordinate: {lon_name}")

    print(
        f"Original latitude range : "
        f"{float(ds[lat_name].min()):.6f} -> "
        f"{float(ds[lat_name].max()):.6f}"
    )

    print(
        f"Original longitude range: "
        f"{float(ds[lon_name].min()):.6f} -> "
        f"{float(ds[lon_name].max()):.6f}"
    )

    # --------------------------------------------------------
    # TIME INFO
    # --------------------------------------------------------

    print(
        f"Time range: "
        f"{ds.time.values[0]} -> "
        f"{ds.time.values[-1]}"
    )

    print(
        f"Number of timestamps: "
        f"{ds.sizes.get('time', 0)}"
    )

    # --------------------------------------------------------
    # AOI SLICES
    # --------------------------------------------------------

    lat_slice = make_slice(
        ds[lat_name],
        SOUTH,
        NORTH
    )

    lon_slice = make_slice(
        ds[lon_name],
        WEST,
        EAST
    )

    # --------------------------------------------------------
    # APPLY AOI FILTER
    # --------------------------------------------------------

    ds_filtered = ds.sel({
        lat_name: lat_slice,
        lon_name: lon_slice
    })

    # Keep only ocean current variables
    ds_filtered = ds_filtered[REQUIRED_VARS]

    # Load into memory before closing source dataset
    ds_filtered = ds_filtered.load()

    ds.close()

    # --------------------------------------------------------
    # VALIDATE AOI RESULT
    # --------------------------------------------------------

    lat_count = ds_filtered.sizes.get(lat_name, 0)
    lon_count = ds_filtered.sizes.get(lon_name, 0)

    if lat_count == 0:
        raise ValueError(
            f"No latitude points remain after AOI filtering "
            f"for {file_path.name}"
        )

    if lon_count == 0:
        raise ValueError(
            f"No longitude points remain after AOI filtering "
            f"for {file_path.name}"
        )

    print(
        f"Filtered grid: "
        f"{lat_count} lat × {lon_count} lon"
    )

    filtered_datasets.append(ds_filtered)


# ============================================================
# VERIFY GRID CONSISTENCY
# ============================================================

reference = filtered_datasets[0]

lat_name = find_coord(
    reference,
    ["latitude", "lat", "nav_lat"]
)

lon_name = find_coord(
    reference,
    ["longitude", "lon", "nav_lon"]
)

for i, ds in enumerate(
    filtered_datasets[1:],
    start=2
):

    current_lat_name = find_coord(
        ds,
        ["latitude", "lat", "nav_lat"]
    )

    current_lon_name = find_coord(
        ds,
        ["longitude", "lon", "nav_lon"]
    )

    if not np.array_equal(
        reference[lat_name].values,
        ds[current_lat_name].values
    ):
        raise ValueError(
            f"Latitude grid differs in file {i}"
        )

    if not np.array_equal(
        reference[lon_name].values,
        ds[current_lon_name].values
    ):
        raise ValueError(
            f"Longitude grid differs in file {i}"
        )


print("\n✅ All four files use the same spatial grid.")


# ============================================================
# CONCATENATE
# ============================================================

print("\nCombining datasets...")

combined = xr.concat(
    filtered_datasets,
    dim="time",
    data_vars="all",
    coords="minimal",
    compat="override",
    join="exact"
)


# ============================================================
# SORT TIME
# ============================================================

combined = combined.sortby("time")


# ============================================================
# REMOVE DUPLICATE TIMESTAMPS
# ============================================================

time_values = combined["time"].values

_, unique_indices = np.unique(
    time_values,
    return_index=True
)

unique_indices = np.sort(unique_indices)

combined = combined.isel(
    time=unique_indices
)


# ============================================================
# VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("FINAL VALIDATION")
print("=" * 60)

print(
    "Time:",
    combined.time.values[0],
    "->",
    combined.time.values[-1]
)

print(
    "Number of timestamps:",
    combined.sizes["time"]
)

print(
    "Latitude:",
    float(combined[lat_name].min()),
    "->",
    float(combined[lat_name].max())
)

print(
    "Longitude:",
    float(combined[lon_name].min()),
    "->",
    float(combined[lon_name].max())
)

print(
    "Variables:",
    list(combined.data_vars)
)


# ------------------------------------------------------------
# TIMESTAMP COUNT
# ------------------------------------------------------------

if combined.sizes["time"] != EXPECTED_TIMESTAMPS:
    raise ValueError(
        f"Expected {EXPECTED_TIMESTAMPS} timestamps "
        f"but got {combined.sizes['time']}"
    )


# ------------------------------------------------------------
# AOI VALIDATION
# ------------------------------------------------------------

lat_min = float(combined[lat_name].min())
lat_max = float(combined[lat_name].max())

lon_min = float(combined[lon_name].min())
lon_max = float(combined[lon_name].max())

if lat_min < SOUTH or lat_max > NORTH:
    raise ValueError(
        "Latitude AOI validation failed."
    )

if lon_min < WEST or lon_max > EAST:
    raise ValueError(
        "Longitude AOI validation failed."
    )


# ------------------------------------------------------------
# TIME ORDER / DUPLICATES
# ------------------------------------------------------------

if not np.all(
    np.diff(
        combined.time.values.astype("datetime64[s]")
    ) > np.timedelta64(0, "s")
):
    raise ValueError(
        "Time axis is not strictly increasing."
    )


# ------------------------------------------------------------
# CHECK NaNs
# ------------------------------------------------------------

for var in REQUIRED_VARS:

    nan_count = int(
        combined[var].isnull().sum().values
    )

    print(
        f"{var} NaN values: {nan_count}"
    )


# ============================================================
# SAVE
# ============================================================

print("\nSaving:")
print(OUTPUT_FILE.resolve())

combined.to_netcdf(
    OUTPUT_FILE,
    format="NETCDF4"
)


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 60)
print("✅ DONE")
print("=" * 60)

print(
    f"Output file:\n"
    f"{OUTPUT_FILE.resolve()}"
)

print("\nFinal dataset:")
print(combined)