from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


# ============================================================
# CONFIG
# ============================================================

OCEAN_FILE = Path(
    "data/ocean_april2019_aoi_22_to_28.nc"
)


# ============================================================
# HELPERS
# ============================================================

def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def find_coord(ds, names):
    for name in names:
        if name in ds.coords:
            return name

        if name in ds.variables:
            return name

    raise ValueError(
        f"Could not find coordinate among {names}. "
        f"Available coordinates: {list(ds.coords)}"
    )


def inspect_variable(ds, name):
    if name not in ds:
        print(f"\n{name}: MISSING")
        return

    da = ds[name]

    print(f"\n{name}")
    print(f"  Dimensions : {da.dims}")
    print(f"  Shape      : {da.shape}")
    print(f"  Dtype      : {da.dtype}")

    if da.attrs:
        print("  Attributes :")
        for key, value in da.attrs.items():
            print(f"    {key}: {value}")

    # Work with a small NumPy representation for diagnostics.
    values = da.values

    total = values.size

    if total == 0:
        print("  No values.")
        return

    nan_count = int(np.isnan(values).sum())

    finite_count = int(
        np.isfinite(values).sum()
    )

    print(f"  Total values       : {total:,}")
    print(f"  NaN values         : {nan_count:,}")
    print(f"  Finite values      : {finite_count:,}")

    if finite_count > 0:

        finite_values = values[
            np.isfinite(values)
        ]

        print(
            f"  Min                : "
            f"{float(finite_values.min()):.10g}"
        )

        print(
            f"  Max                : "
            f"{float(finite_values.max()):.10g}"
        )

        print(
            f"  Mean               : "
            f"{float(finite_values.mean()):.10g}"
        )

        print(
            f"  Std                : "
            f"{float(finite_values.std()):.10g}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    if not OCEAN_FILE.exists():
        raise FileNotFoundError(
            f"Ocean file not found:\n{OCEAN_FILE}"
        )

    print_section("OPENING OCEAN FILE")

    print(f"File: {OCEAN_FILE.resolve()}")

    ds = xr.open_dataset(OCEAN_FILE)

    print("\nDataset:")
    print(ds)

    # ========================================================
    # 1. DIMENSIONS
    # ========================================================

    print_section("DIMENSIONS")

    for name, size in ds.sizes.items():
        print(f"{name:15s}: {size}")

    # ========================================================
    # 2. COORDINATES
    # ========================================================

    print_section("COORDINATES")

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

    # --------------------------------------------------------
    # Latitude
    # --------------------------------------------------------

    lat = ds[lat_name].values
    lon = ds[lon_name].values

    print("\nLatitude:")
    print(f"  Count      : {len(lat)}")
    print(f"  First      : {lat[0]}")
    print(f"  Last       : {lat[-1]}")
    print(f"  Min        : {lat.min()}")
    print(f"  Max        : {lat.max()}")

    if len(lat) > 1:
        lat_direction = (
            "ascending"
            if lat[0] < lat[-1]
            else "descending"
        )

        print(f"  Direction  : {lat_direction}")

        print(
            f"  Step range : "
            f"{np.diff(lat).min()} -> {np.diff(lat).max()}"
        )

    # --------------------------------------------------------
    # Longitude
    # --------------------------------------------------------

    print("\nLongitude:")
    print(f"  Count      : {len(lon)}")
    print(f"  First      : {lon[0]}")
    print(f"  Last       : {lon[-1]}")
    print(f"  Min        : {lon.min()}")
    print(f"  Max        : {lon.max()}")

    if len(lon) > 1:

        lon_direction = (
            "ascending"
            if lon[0] < lon[-1]
            else "descending"
        )

        print(f"  Direction  : {lon_direction}")

        print(
            f"  Step range : "
            f"{np.diff(lon).min()} -> {np.diff(lon).max()}"
        )

    # ========================================================
    # 3. TIME
    # ========================================================

    print_section("TIME COVERAGE")

    if "time" not in ds:

        raise RuntimeError(
            "No 'time' coordinate found."
        )

    times = pd.to_datetime(
        ds["time"].values
    )

    print(f"Count : {len(times)}")

    print(f"First : {times[0]}")
    print(f"Last  : {times[-1]}")

    # --------------------------------------------------------
    # Duplicate timestamp check
    # --------------------------------------------------------

    duplicated = times.duplicated()

    print(
        f"\nDuplicate timestamps: "
        f"{int(duplicated.sum())}"
    )

    if duplicated.any():

        print(
            "Duplicates:"
        )

        for timestamp in times[duplicated]:
            print(f"  {timestamp}")

    # --------------------------------------------------------
    # Time spacing
    # --------------------------------------------------------

    if len(times) > 1:

        deltas = pd.Series(
            times[1:] - times[:-1]
        )

        print("\nTime spacing:")

        print(
            deltas.value_counts()
        )

        expected = pd.Timedelta(
            hours=1
        )

        bad = deltas[
            deltas != expected
        ]

        print(
            f"\nNon-1-hour intervals: "
            f"{len(bad)}"
        )

        if len(bad) > 0:
            print(bad)

    # ========================================================
    # 4. DATA VARIABLES
    # ========================================================

    print_section("DATA VARIABLES")

    for name, da in ds.data_vars.items():

        print(
            f"{name:15s} "
            f"dims={da.dims} "
            f"shape={da.shape} "
            f"dtype={da.dtype}"
        )

    # ========================================================
    # 5. REQUIRED OCEAN VARIABLES
    # ========================================================

    print_section("REQUIRED VARIABLES: UO / VO")

    inspect_variable(ds, "uo")
    inspect_variable(ds, "vo")

    for variable in ["uo", "vo"]:

        if variable not in ds:
            raise RuntimeError(
                f"Required variable '{variable}' is missing."
            )

    # ========================================================
    # 6. AOI EXTENT
    # ========================================================

    print_section("AOI EXTENT")

    print(
        f"Latitude : "
        f"{float(ds[lat_name].min()):.8f} "
        f"-> "
        f"{float(ds[lat_name].max()):.8f}"
    )

    print(
        f"Longitude: "
        f"{float(ds[lon_name].min()):.8f} "
        f"-> "
        f"{float(ds[lon_name].max()):.8f}"
    )

    # ========================================================
    # 7. MISSING / FILL VALUES
    # ========================================================

    print_section("MISSING / FILL VALUE INSPECTION")

    for variable in ["uo", "vo"]:

        da = ds[variable]

        print(f"\n{variable}")

        for attr in [
            "_FillValue",
            "missing_value",
            "valid_min",
            "valid_max",
        ]:
            if attr in da.attrs:
                print(
                    f"  {attr}: "
                    f"{da.attrs[attr]}"
                )

        values = da.values

        # Check huge suspicious values.
        suspicious = np.abs(values) > 100

        print(
            f"  |value| > 100 m/s: "
            f"{int(suspicious.sum()):,}"
        )

        # Check values around the known NetCDF fill-value region.
        huge = np.abs(values) > 1e10

        print(
            f"  |value| > 1e10 : "
            f"{int(huge.sum()):,}"
        )

    # ========================================================
    # 8. FINITE / VALIDITY SUMMARY
    # ========================================================

    print_section("FINITE DATA SUMMARY")

    for variable in ["uo", "vo"]:

        values = ds[variable].values

        finite = np.isfinite(values)

        total = values.size
        finite_count = int(finite.sum())

        print(
            f"{variable}: "
            f"{finite_count:,}/{total:,} "
            f"finite "
            f"({100 * finite_count / total:.4f}%)"
        )

    # ========================================================
    # 9. SAMPLE VALUES
    # ========================================================

    print_section("SAMPLE OCEAN-CURRENT VALUES")

    # First time, middle time, last time
    sample_indices = sorted(
        set([
            0,
            len(times) // 2,
            len(times) - 1,
        ])
    )

    for index in sample_indices:

        print(
            f"\nTimestamp: {times[index]}"
        )

        u = ds["uo"].isel(
            time=index
        ).values

        v = ds["vo"].isel(
            time=index
        ).values

        valid = (
            np.isfinite(u)
            & np.isfinite(v)
        )

        if valid.any():

            speed = np.sqrt(
                u[valid] ** 2
                + v[valid] ** 2
            )

            print(
                f"  uo mean   : "
                f"{float(u[valid].mean()):.6f} m/s"
            )

            print(
                f"  vo mean   : "
                f"{float(v[valid].mean()):.6f} m/s"
            )

            print(
                f"  speed mean: "
                f"{float(speed.mean()):.6f} m/s"
            )

            print(
                f"  speed max : "
                f"{float(speed.max()):.6f} m/s"
            )

    # ========================================================
    # 10. DATASET ATTRIBUTES
    # ========================================================

    print_section("DATASET ATTRIBUTES")

    if ds.attrs:

        for key, value in ds.attrs.items():
            print(
                f"{key}: {value}"
            )

    else:

        print(
            "No global dataset attributes."
        )

    # ========================================================
    # 11. FINAL CHECKS
    # ========================================================

    print_section("FINAL CHECKS")

    checks = {}

    checks["uo_present"] = "uo" in ds
    checks["vo_present"] = "vo" in ds
    checks["time_present"] = "time" in ds
    checks["latitude_present"] = lat_name in ds
    checks["longitude_present"] = lon_name in ds
    checks["duplicate_times"] = not duplicated.any()

    if len(times) > 1:

        checks["hourly_spacing"] = bool(
            np.all(
                np.diff(times.values)
                == np.timedelta64(1, "h")
            )
        )

    else:

        checks["hourly_spacing"] = True

    for name, result in checks.items():

        print(
            f"  {'PASS' if result else 'FAIL'} "
            f"{name}"
        )

    all_pass = all(checks.values())

    print(
        "\nOverall basic inspection: "
        f"{'PASS' if all_pass else 'CHECK REQUIRED'}"
    )

    ds.close()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()