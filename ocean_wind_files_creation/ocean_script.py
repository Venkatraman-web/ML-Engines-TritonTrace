from pathlib import Path
import xarray as xr


# ============================================================
# CONFIG
# ============================================================

INPUT_DIR = Path("ocean_file_raw")

OUTPUT_FILE = Path("data/ocean_april2019_aoi_22_to_28.nc")

# SAME AOI USED IN THE PREVIOUS CASE
# AOI = [north, west, south, east]
NORTH = 36.372
WEST = 27.121
SOUTH = 30.1875
EAST = 36.088

# April 22 -> April 28, 2019
DATES = [
    "20190422",
    "20190423",
    "20190424",
    "20190425",
    "20190426",
    "20190427",
    "20190428",
]

FILENAME_TEMPLATE = (
    "{date}_h-CMCC--RFVL-MFSe3r1-MED-"
    "b20200901_re-sv01.00.nc"
)


# ============================================================
# HELPERS
# ============================================================

def find_coord(ds, names):
    """
    Find the coordinate name used by the NetCDF file.
    """
    for name in names:
        if name in ds.coords:
            return name

        if name in ds.variables:
            return name

    raise ValueError(
        f"Could not find coordinate among {names}. "
        f"Available coords: {list(ds.coords)}"
    )


def make_lat_slice(lat):
    """
    Create correct latitude slice regardless of coordinate direction.
    """
    if lat.values[0] < lat.values[-1]:
        return slice(SOUTH, NORTH)

    return slice(NORTH, SOUTH)


def make_lon_slice(lon):
    """
    Create correct longitude slice regardless of coordinate direction.
    """
    if lon.values[0] < lon.values[-1]:
        return slice(WEST, EAST)

    return slice(EAST, WEST)


# ============================================================
# MAIN
# ============================================================

def main():

    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # 1. Locate files
    # --------------------------------------------------------

    files = []

    for date in DATES:

        filename = FILENAME_TEMPLATE.format(date=date)
        path = INPUT_DIR / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Missing required file:\n{path}"
            )

        files.append(path)

    print("=" * 70)
    print("FILES TO PROCESS")
    print("=" * 70)

    for path in files:
        print(path)

    # --------------------------------------------------------
    # 2. Open first file to identify coordinates
    # --------------------------------------------------------

    with xr.open_dataset(files[0]) as first_ds:

        lat_name = find_coord(
            first_ds,
            ["latitude", "lat", "nav_lat"]
        )

        lon_name = find_coord(
            first_ds,
            ["longitude", "lon", "nav_lon"]
        )

        print("\nDetected coordinates:")
        print(f"Latitude : {lat_name}")
        print(f"Longitude: {lon_name}")

        # Confirm required ocean-current variables
        for variable in ["uo", "vo"]:

            if variable not in first_ds.data_vars:
                raise ValueError(
                    f"Required variable '{variable}' "
                    f"not found in {files[0]}"
                )

        lat_slice = make_lat_slice(
            first_ds[lat_name]
        )

        lon_slice = make_lon_slice(
            first_ds[lon_name]
        )

    # --------------------------------------------------------
    # 3. Open all seven files lazily
    # --------------------------------------------------------

    print("\nOpening daily NetCDF files...")

    datasets = []

    for path in files:

        ds = xr.open_dataset(path)

        # Filter to SAME AOI before concatenation
        ds = ds.sel(
            {
                lat_name: lat_slice,
                lon_name: lon_slice,
            }
        )

        datasets.append(ds)

        print(
            f"  {path.name}: "
            f"{dict(ds.sizes)}"
        )

    # --------------------------------------------------------
    # 4. Concatenate along time
    # --------------------------------------------------------

    print("\nConcatenating along time...")

    combined = xr.concat(
        datasets,
        dim="time",
        data_vars="all",
        coords="minimal",
        compat="override",
        join="exact",
    )

    # --------------------------------------------------------
    # 5. Sort and remove duplicate timestamps
    # --------------------------------------------------------

    combined = combined.sortby("time")

    time_values = combined["time"].values

    if len(time_values) != len(set(time_values)):

        print(
            "Duplicate timestamps detected. "
            "Removing duplicates..."
        )

        import numpy as np

        _, unique_indices = np.unique(
            time_values,
            return_index=True,
        )

        unique_indices = sorted(unique_indices)

        combined = combined.isel(
            time=unique_indices
        )

    # --------------------------------------------------------
    # 6. Validate expected time coverage
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TIME VALIDATION")
    print("=" * 70)

    print(
        "First timestamp:",
        combined.time.values[0]
    )

    print(
        "Last timestamp :",
        combined.time.values[-1]
    )

    print(
        "Number of timestamps:",
        combined.sizes["time"]
    )

    # April 22-28 inclusive = 168 hourly timestamps
    if combined.sizes["time"] != 168:

        print(
            "\nWARNING:"
            f" expected approximately 168 hourly timestamps, "
            f"found {combined.sizes['time']}."
        )

    # --------------------------------------------------------
    # 7. Validate AOI
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("AOI VALIDATION")
    print("=" * 70)

    actual_lat_min = float(
        combined[lat_name].min()
    )

    actual_lat_max = float(
        combined[lat_name].max()
    )

    actual_lon_min = float(
        combined[lon_name].min()
    )

    actual_lon_max = float(
        combined[lon_name].max()
    )

    print(
        f"Latitude : {actual_lat_min:.6f} -> "
        f"{actual_lat_max:.6f}"
    )

    print(
        f"Longitude: {actual_lon_min:.6f} -> "
        f"{actual_lon_max:.6f}"
    )

    # --------------------------------------------------------
    # 8. Validate required variables
    # --------------------------------------------------------

    print("\nRequired variables:")

    for variable in ["uo", "vo"]:

        print(
            f"  {variable}: "
            f"{combined[variable].dims}, "
            f"{combined[variable].dtype}"
        )

    # --------------------------------------------------------
    # 9. Save
    # --------------------------------------------------------

    print("\nSaving combined AOI NetCDF...")

    combined.to_netcdf(
        OUTPUT_FILE,
        format="NETCDF4",
    )

    # Close all open datasets
    combined.close()

    for ds in datasets:
        ds.close()

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)

    print(
        f"Output: {OUTPUT_FILE.resolve()}"
    )


if __name__ == "__main__":
    main()