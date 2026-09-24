import cdsapi

# ============================================================
# CONFIG
# ============================================================

YEAR = "2019"
MONTH = "06"

DAYS = [
    "07", "08", "09", "10",
    "11", "12", "13", "14",
]

TIMES = [
    f"{hour:02d}:00"
    for hour in range(24)
]

AOI = [
    36.372,    # North
    27.121,    # West
    30.1875,   # South
    36.088,    # East
]

OUTPUT_FILE = "data/wind_june2019_07_to_14.nc"


# ============================================================
# CDS CLIENT
# ============================================================

# Use your existing CDS credentials configuration.
# Recommended: configure ~/.cdsapirc instead of hard-coding
# your API key in this script.
c = cdsapi.Client(
    url="https://cds.climate.copernicus.eu/api", key="11ba0656-221f-4992-a558-c994bd62aa77"
)


# ============================================================
# DOWNLOAD
# ============================================================

print("=" * 80)
print("DOWNLOADING ERA5 WIND DATA")
print("=" * 80)

print(f"\nPeriod : {YEAR}-{MONTH}-07 → {YEAR}-{MONTH}-14")

print("\nVariables:")
print("  - 10m_u_component_of_wind")
print("  - 10m_v_component_of_wind")

print("\nAOI:")
print(f"  North: {AOI[0]}")
print(f"  West : {AOI[1]}")
print(f"  South: {AOI[2]}")
print(f"  East : {AOI[3]}")

print(f"\nOutput: {OUTPUT_FILE}")


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

        # Bounding box format:
        # [North, West, South, East]
        "area": AOI,
    },
    OUTPUT_FILE,
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 80)
print("DOWNLOAD COMPLETE")
print("=" * 80)

print(f"\nSaved to:")
print(OUTPUT_FILE)

print("\nExpected coverage:")
print("  First: 2019-06-07 00:00")
print("  Last : 2019-06-14 23:00")
print("  Hours: 192")

print("\nWind data downloaded successfully.")