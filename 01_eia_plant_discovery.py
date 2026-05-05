#!/usr/bin/env python3
"""
Step 1: EIA Plant Discovery & Filtering
========================================
Downloads EIA-860 plant metadata (lat, lon, capacity, fuel type, in-service date)
and filters to solar plants with sufficient history for correlation analysis.

Outputs:
    data/eia_860_plants.csv         - All solar plants from EIA-860
    data/solar_plants_filtered.csv  - Filtered candidates (≥10yr, ≥5MW)
"""

import io
import sys
import zipfile
import requests
import pandas as pd
from pathlib import Path

import config


def download_eia860_plant_metadata() -> pd.DataFrame:
    """
    Download EIA-860 bulk data (Excel zip) and extract the Plant tab
    which contains plant_id, name, state, lat, lon, and fuel info.

    The 2_Plant_Yyyy file has columns including:
        Plant Code, Plant Name, State, County, Latitude, Longitude,
        Balancing Authority Code, NERC Region, Sector, etc.
    """
    print(f"Downloading EIA-860 data from:\n  {config.EIA_860_URL}")
    print("  (This is ~15 MB, may take a moment...)")

    response = requests.get(config.EIA_860_URL, timeout=120)
    response.raise_for_status()

    z = zipfile.ZipFile(io.BytesIO(response.content))

    # List all files to find the Plant file
    filenames = z.namelist()
    print(f"  ZIP contains {len(filenames)} files:")
    for f in filenames:
        print(f"    {f}")

    # Find the plant-level file (2_PlantYyyyy.xlsx or similar)
    plant_file = None
    for f in filenames:
        if "plant" in f.lower() and f.endswith(".xlsx"):
            # Prefer the main "2_Plant" file, not "2_3_Solar" or others
            if "2_plant" in f.lower() or ("plant" in f.lower() and "solar" not in f.lower()
                                           and "storage" not in f.lower()
                                           and "wind" not in f.lower()
                                           and "multi" not in f.lower()):
                plant_file = f
                break

    # Fallback: just grab any file with "plant" in the name
    if plant_file is None:
        for f in filenames:
            if "plant" in f.lower() and f.endswith(".xlsx"):
                plant_file = f
                break

    if plant_file is None:
        raise FileNotFoundError(
            f"Could not find plant metadata file in EIA-860 zip. Files: {filenames}"
        )

    print(f"\n  Reading plant metadata from: {plant_file}")

    with z.open(plant_file) as fh:
        # The first row is typically a header description; actual headers start at row 1-2
        # Try reading with header=1 first (common for recent EIA files)
        df = pd.read_excel(
            io.BytesIO(fh.read()),
            sheet_name=0,  # First sheet (Operable)
            header=1,      # Headers on second row (row index 1)
            dtype={"Plant Code": str}
        )

    # Standardize column names (EIA changes these slightly year to year)
    col_map = {}
    for col in df.columns:
        cl = str(col).lower().strip()
        if "plant code" in cl or "plant id" in cl:
            col_map[col] = "plant_id"
        elif "plant name" in cl:
            col_map[col] = "plant_name"
        elif cl == "state":
            col_map[col] = "state"
        elif cl == "county":
            col_map[col] = "county"
        elif "latitude" in cl:
            col_map[col] = "latitude"
        elif "longitude" in cl:
            col_map[col] = "longitude"
        elif "nameplate" in cl and "capacity" in cl:
            col_map[col] = "nameplate_capacity_mw"
        elif "balancing authority" in cl and "code" in cl:
            col_map[col] = "ba_code"
        elif "nerc" in cl and "region" in cl:
            col_map[col] = "nerc_region"
        elif "sector" in cl and "name" in cl:
            col_map[col] = "sector_name"
        elif "primary" in cl and ("purpose" in cl or "energy" in cl or "source" in cl):
            col_map[col] = "primary_source"

    df = df.rename(columns=col_map)

    print(f"  Found {len(df)} total plants in EIA-860")
    print(f"  Columns mapped: {list(col_map.values())}")

    return df


def download_eia860_zip(cache_path: Path) -> bytes:
    """Download EIA-860 zip once and cache it on disk."""
    if cache_path.exists():
        return cache_path.read_bytes()
    print(f"Downloading EIA-860 from {config.EIA_860_URL}")
    response = requests.get(config.EIA_860_URL, timeout=180)
    response.raise_for_status()
    cache_path.write_bytes(response.content)
    return response.content


def download_eia860_solar_details(z_content: bytes) -> pd.DataFrame:
    """
    Parse EIA-860 Schedule 3 (the 3_3_Solar Excel tab) at generator level.

    Returns a DataFrame with one row per (plant_id, generator_id) and the
    columns needed downstream: AC and DC nameplates, tilt and azimuth angles,
    and a single canonical tracking_type (one of fixed, single_axis,
    dual_axis, unknown).
    """
    z = zipfile.ZipFile(io.BytesIO(z_content))

    solar_file = None
    for f in z.namelist():
        # The file in the EIA-860 zip is 3_3_Solar_Y<year>.xlsx
        if "3_3_solar" in f.lower() and f.endswith(".xlsx"):
            solar_file = f
            break

    if solar_file is None:
        print("  WARNING: Could not find 3_3_Solar file in EIA-860 zip")
        return pd.DataFrame()

    print(f"  Reading solar generator details from: {solar_file}")

    with z.open(solar_file) as fh:
        df = pd.read_excel(
            io.BytesIO(fh.read()),
            sheet_name=0,
            header=1,
            dtype={"Plant Code": str, "Generator ID": str},
        )

    # Map the EIA-860 column names (which change slightly across vintages)
    # onto our canonical names. Schedule 3 column names use natural English,
    # like "Nameplate Capacity (MW)" and "Tilt Angle".
    col_map = {}
    for col in df.columns:
        cl = str(col).lower().strip()
        if "plant code" in cl:
            col_map[col] = "plant_id"
        elif cl == "generator id":
            col_map[col] = "generator_id"
        elif "nameplate" in cl and "capacity" in cl and "dc" not in cl:
            col_map[col] = "nameplate_capacity_ac_mw"
        elif "nameplate" in cl and "dc" in cl:
            col_map[col] = "nameplate_capacity_dc_mw"
        elif "tilt angle" in cl:
            col_map[col] = "tilt_angle_deg"
        elif "azimuth angle" in cl:
            col_map[col] = "azimuth_angle_deg"
        elif cl == "single-axis tracking?" or "single-axis" in cl:
            col_map[col] = "single_axis_tracking"
        elif cl == "dual-axis tracking?" or "dual-axis" in cl:
            col_map[col] = "dual_axis_tracking"
        elif cl == "fixed tilt?" or "fixed tilt" in cl:
            col_map[col] = "fixed_tilt"
        elif "operating year" in cl:
            col_map[col] = "operating_year"

    df = df.rename(columns=col_map)
    keep = [
        "plant_id", "generator_id",
        "nameplate_capacity_ac_mw", "nameplate_capacity_dc_mw",
        "tilt_angle_deg", "azimuth_angle_deg",
        "single_axis_tracking", "dual_axis_tracking", "fixed_tilt",
        "operating_year",
    ]
    df = df[[c for c in keep if c in df.columns]].copy()

    # Coerce numerics; the EIA Y/N flags are stringy
    for c in ("nameplate_capacity_ac_mw", "nameplate_capacity_dc_mw",
              "tilt_angle_deg", "azimuth_angle_deg", "operating_year"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Collapse the three tracking-type Y/N columns into one canonical label.
    def _yes(val):
        return isinstance(val, str) and val.strip().upper().startswith("Y")

    def _classify(row):
        if _yes(row.get("single_axis_tracking")):
            return "single_axis"
        if _yes(row.get("dual_axis_tracking")):
            return "dual_axis"
        if _yes(row.get("fixed_tilt")):
            return "fixed"
        return "unknown"

    df["tracking_type"] = df.apply(_classify, axis=1)
    df["plant_id"] = df["plant_id"].astype(str)

    print(f"  Parsed {len(df)} generator rows across "
          f"{df['plant_id'].nunique()} plants")
    return df


def aggregate_solar_details_to_plant(df_gen: pd.DataFrame) -> pd.DataFrame:
    """
    Roll generator-level Schedule 3 data up to one row per plant.

    Sums AC and DC capacity. Picks the dominant tilt, azimuth, and tracking
    type by capacity-weighted majority. Sets a homogeneous flag if every
    generator at the plant agrees on tracking type.
    """
    if df_gen.empty:
        return pd.DataFrame()

    rows = []
    for pid, group in df_gen.groupby("plant_id"):
        ac_sum = group["nameplate_capacity_ac_mw"].sum(skipna=True) \
            if "nameplate_capacity_ac_mw" in group.columns else None
        dc_sum = group["nameplate_capacity_dc_mw"].sum(skipna=True) \
            if "nameplate_capacity_dc_mw" in group.columns else None

        # Capacity-weighted dominant tilt/azimuth/tracking
        weights = group["nameplate_capacity_ac_mw"].fillna(0)
        if weights.sum() == 0:
            weights = pd.Series(1.0, index=group.index)

        def _weighted_mean(col):
            if col not in group.columns:
                return None
            vals = group[col]
            valid = vals.notna()
            if not valid.any():
                return None
            w = weights[valid]
            v = vals[valid]
            return float((v * w).sum() / w.sum()) if w.sum() else None

        # Tracking type: weighted mode
        track_counts = (
            group.groupby("tracking_type")["nameplate_capacity_ac_mw"]
            .sum(min_count=1).fillna(0)
        )
        dominant_tracking = (
            track_counts.idxmax() if not track_counts.empty else "unknown"
        )
        tracking_homogeneous = (
            group["tracking_type"].nunique(dropna=False) == 1
        )

        rows.append({
            "plant_id": pid,
            "n_generators": len(group),
            "nameplate_capacity_ac_mw_860": ac_sum,
            "nameplate_capacity_dc_mw_860": dc_sum,
            "dc_to_ac_ratio": (dc_sum / ac_sum) if (ac_sum and dc_sum) else None,
            "dominant_tilt_deg": _weighted_mean("tilt_angle_deg"),
            "dominant_azimuth_deg": _weighted_mean("azimuth_angle_deg"),
            "dominant_tracking": dominant_tracking,
            "generators_homogeneous": tracking_homogeneous,
        })

    return pd.DataFrame(rows)


def _safe_int(value, default=None):
    """Convert EIA API count values to int when possible."""
    try:
        if value is None or value == "":
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def get_solar_plants_via_api(api_key: str) -> pd.DataFrame:
    """
    Alternative approach: Use EIA API to discover solar plants.
    Queries the facility-fuel endpoint to find all plants with SUN fuel.

    This is a fallback if the EIA-860 download fails.
    """
    print("Discovering solar plants via EIA API...")

    url = "https://api.eia.gov/v2/electricity/facility-fuel/data/"

    all_plants = []
    offset = 0
    batch_size = 5000

    while True:
        params = {
            "api_key": api_key,
            "frequency": "annual",
            "data[0]": "generation",
            "facets[fuel2002][]": "SUN",
            "start": f"{config.START_YEAR}",
            "end": f"{config.END_YEAR}",
            "sort[0][column]": "plantCode",
            "sort[0][direction]": "asc",
            "offset": offset,
            "length": batch_size,
        }

        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()
        data = response.json()

        if "response" not in data or "data" not in data["response"]:
            break

        records = data["response"]["data"]
        if not records:
            break

        all_plants.extend(records)
        total = _safe_int(data["response"].get("total"))
        offset += len(records)

        print(f"  Fetched {len(all_plants)} / {total} records...")

        if total is not None and offset >= total:
            break

        if len(records) < batch_size:
            break

    df = pd.DataFrame(all_plants)

    if not df.empty:
        # Get unique plants with their total annual generation
        df["generation"] = pd.to_numeric(df["generation"], errors="coerce")
        df["plantCode"] = df["plantCode"].astype(str)

        # Count how many years each plant has data
        year_counts = (
            df.groupby("plantCode")["period"]
            .nunique()
            .reset_index()
            .rename(columns={"period": "num_years"})
        )

        # Get latest generation per plant. EIA has removed sector fields from
        # this endpoint in some responses, so keep them optional.
        plant_group_cols = ["plantCode", "plantName", "state"]
        for optional_col in ["sector", "sectorName"]:
            if optional_col in df.columns:
                plant_group_cols.append(optional_col)

        plant_summary = (
            df.groupby(plant_group_cols)
            .agg(
                total_generation=("generation", "sum"),
                first_year=("period", "min"),
                last_year=("period", "max"),
            )
            .reset_index()
        )

        plant_summary = plant_summary.merge(year_counts, on="plantCode")
        plant_summary = plant_summary.rename(columns={"plantCode": "plant_id", "plantName": "plant_name"})
        for optional_col in ["sector", "sectorName"]:
            if optional_col not in plant_summary.columns:
                plant_summary[optional_col] = pd.NA
        plant_summary = plant_summary[
            [
                "plant_id",
                "plant_name",
                "state",
                "sector",
                "sectorName",
                "total_generation",
                "first_year",
                "last_year",
                "num_years",
            ]
        ]

        print(f"  Found {len(plant_summary)} unique solar plants via API")

    return plant_summary if not df.empty else pd.DataFrame()


def filter_plants(
    df_plants: pd.DataFrame,
    min_capacity_mw: float = config.MIN_CAPACITY_MW,
    min_years: int = config.MIN_YEARS_OF_DATA,
    start_year: int = config.START_YEAR,
) -> pd.DataFrame:
    """
    Filter solar plants based on:
    - Minimum nameplate capacity
    - Minimum years of available data (based on in-service date)
    - Valid lat/lon coordinates (within CONUS bounds for NSRDB)
    """
    n_start = len(df_plants)
    print(f"\nFiltering {n_start} solar plants...")

    # Filter by capacity
    if "nameplate_capacity_mw" in df_plants.columns:
        df_plants["nameplate_capacity_mw"] = pd.to_numeric(
            df_plants["nameplate_capacity_mw"], errors="coerce"
        )
        df_plants = df_plants[df_plants["nameplate_capacity_mw"] >= min_capacity_mw]
        print(f"  After capacity ≥ {min_capacity_mw} MW: {len(df_plants)}")

    # Filter by valid lat/lon (CONUS bounds roughly)
    if "latitude" in df_plants.columns and "longitude" in df_plants.columns:
        df_plants["latitude"] = pd.to_numeric(df_plants["latitude"], errors="coerce")
        df_plants["longitude"] = pd.to_numeric(df_plants["longitude"], errors="coerce")

        df_plants = df_plants[
            df_plants["latitude"].between(24, 50) &    # CONUS latitude
            df_plants["longitude"].between(-125, -66)   # CONUS longitude
        ]
        print(f"  After CONUS lat/lon filter: {len(df_plants)}")

    # Estimate data history from in-service year if available
    if "num_years" in df_plants.columns:
        df_plants = df_plants[df_plants["num_years"] >= min_years]
        print(f"  After ≥ {min_years} years of data: {len(df_plants)}")

    print(f"\n  Final count: {len(df_plants)} plants (from {n_start})")

    return df_plants.reset_index(drop=True)


def main():
    """Run Step 1: Plant discovery and filtering."""

    print("=" * 70)
    print("STEP 1: EIA Plant Discovery & Filtering")
    print("=" * 70)

    if not config.EIA_API_KEY:
        print("\nERROR: EIA_API_KEY not set in .env file!")
        print("   Register at: https://www.eia.gov/opendata/register.php")
        sys.exit(1)

    # -- Strategy: Use API to discover solar plants + year counts --
    # The API approach is more reliable than parsing EIA-860 Excel files
    # because it directly tells us which plants have solar generation data.

    df_api = get_solar_plants_via_api(config.EIA_API_KEY)

    if df_api.empty:
        print("\nERROR: No solar plants found via API. Check your API key.")
        sys.exit(1)

    # Save raw discovery
    df_api.to_csv(config.EIA_860_PLANTS_CSV, index=False)
    print(f"\nSaved {len(df_api)} raw solar plants -> {config.EIA_860_PLANTS_CSV}")

    # -- Now we need lat/lon for each plant --
    # The facility-fuel endpoint doesn't include lat/lon, so we need to
    # either: (a) download EIA-860 Excel, or (b) use the operating-generator-capacity endpoint
    # We'll use approach (b) - query lat/lon from operating-generator-capacity

    print("\nFetching lat/lon from EIA operating-generator-capacity endpoint...")
    plant_ids = df_api["plant_id"].unique().tolist()

    latlon_data = fetch_plant_latlon_from_api(plant_ids, config.EIA_API_KEY)

    if not latlon_data.empty:
        df_api = df_api.merge(latlon_data, on="plant_id", how="left")

    # Filter
    df_filtered = filter_plants(df_api)

    # Save filtered
    df_filtered.to_csv(config.SOLAR_PLANTS_FILTERED_CSV, index=False)
    print(f"\nSaved {len(df_filtered)} filtered solar plants -> {config.SOLAR_PLANTS_FILTERED_CSV}")

    # Summary
    print("\n" + "-" * 50)
    print("SUMMARY")
    print("-" * 50)
    if "state" in df_filtered.columns:
        print(f"\nTop states by plant count:")
        print(df_filtered["state"].value_counts().head(10).to_string())
    if "num_years" in df_filtered.columns:
        print(f"\nYear coverage distribution:")
        print(df_filtered["num_years"].describe().to_string())

    print(f"\nStep 1 complete. {len(df_filtered)} plants ready for generation download.")


def fetch_plant_latlon_from_api(plant_ids: list, api_key: str) -> pd.DataFrame:
    """
    Fetch latitude/longitude for a list of plant IDs using the
    EIA operating-generator-capacity endpoint.

    We batch this to avoid hitting API limits.
    """
    import time

    url = "https://api.eia.gov/v2/electricity/operating-generator-capacity/data/"

    all_records = []

    # Process in batches (API allows filtering by multiple plantCode values)
    batch_size = 50
    for i in range(0, len(plant_ids), batch_size):
        batch = plant_ids[i : i + batch_size]

        params = {
            "api_key": api_key,
            "frequency": "monthly",
            "data[0]": "nameplate-capacity-mw",
            "data[1]": "latitude",
            "data[2]": "longitude",
            "facets[energy_source_code][]": config.SOLAR_FUEL_CODE,
            "facets[plantid][]": [str(pid) for pid in batch],
            "start": f"{config.END_YEAR}-12",
            "end": f"{config.END_YEAR}-12",
            "sort[0][column]": "plantid",
            "sort[0][direction]": "asc",
            "offset": 0,
            "length": 5000,
        }

        try:
            response = requests.get(url, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()

            if "response" in data and "data" in data["response"]:
                all_records.extend(data["response"]["data"])
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else "unknown"
            print(f"  Warning: batch {i//batch_size} failed with HTTP {status}")
        except Exception as e:
            print(f"  Warning: batch {i//batch_size} failed: {type(e).__name__}")

        time.sleep(config.EIA_API_DELAY_S)

        if (i // batch_size) % 10 == 0 and i > 0:
            print(f"  Fetched lat/lon for {i}/{len(plant_ids)} plants...")

    if not all_records:
        print("  WARNING: Could not fetch lat/lon data from API")
        return pd.DataFrame()

    df = pd.DataFrame(all_records)

    # Extract unique plant lat/lon and capacity
    if "latitude" in df.columns and "longitude" in df.columns:
        df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
        df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
        df["nameplate-capacity-mw"] = pd.to_numeric(
            df.get("nameplate-capacity-mw", pd.Series(dtype=float)), errors="coerce"
        )

        plant_id_col = "plantid" if "plantid" in df.columns else "plantCode"

        # Aggregate to plant level (sum of generator capacities, take first lat/lon)
        plant_info = (
            df.groupby(plant_id_col)
            .agg(
                latitude=("latitude", "first"),
                longitude=("longitude", "first"),
                nameplate_capacity_mw=("nameplate-capacity-mw", "sum"),
            )
            .reset_index()
            .rename(columns={plant_id_col: "plant_id"})
        )

        plant_info["plant_id"] = plant_info["plant_id"].astype(str)

        print(f"  Got lat/lon for {len(plant_info)} plants")
        return plant_info

    return pd.DataFrame()


if __name__ == "__main__":
    main()
