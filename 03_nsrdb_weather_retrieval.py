#!/usr/bin/env python3
"""
Step 3: NSRDB Weather Retrieval
================================
For each filtered solar plant, fetches historical GHI (Global Horizontal
Irradiance) data from NREL's NSRDB.

THIS IS THE SLOW STEP - NSRDB rate-limits to ~1 request/2 seconds,
and each plant requires pulling all historical years (1998-2024).
Expect ~1-2 minutes per plant.

The script supports checkpointing: if interrupted, re-run and it will
skip plants already downloaded.

Inputs:
    data/solar_plants_filtered.csv
    data/eia_monthly_generation.csv  (to know which plants actually have gen data)

Outputs:
    data/nsrdb_monthly_ghi.csv
"""

import sys
import time
import traceback
import pandas as pd
import numpy as np
from tqdm import tqdm

import config

# NSRDB data retrieval. If an optional local nsrdb.py wrapper is importable,
# use it; otherwise fall back to a standalone pvlib path that hits the same
# NSRDB endpoint directly.

NSRDB_AVAILABLE = False

try:
    from nsrdb import get_nsrdb
    NSRDB_AVAILABLE = True
    print("Using optional nsrdb.py wrapper.")
except ImportError:
    print("yellowstone package not available; using standalone pvlib fallback.")


def fetch_nsrdb_standalone(
    lat: float,
    lon: float,
    api_key: str,
    email: str,
    start_year: int = config.START_YEAR,
    end_year: int = config.END_YEAR,
) -> pd.DataFrame:
    """
    Standalone NSRDB retrieval using pvlib directly (no yellowstone dependency).

    Pulls historical yearly GHI from the NSRDB PSM v4 GOES-aggregated endpoint,
    aggregates to monthly sums, and returns a DataFrame.

    Returns DataFrame with columns: year, month, monthly_ghi_wh_m2
    """
    import pvlib

    # Round coordinates to avoid over-precision issues
    lat = round(lat, 2)
    lon = round(lon, 2)

    all_monthly = []

    # First, check which endpoint is available at this location
    try:
        response = __import__("requests").get(
            f"https://developer.nrel.gov/api/nsrdb/v2/site-count.json"
            f"?api_key={api_key}&wkt=POINT({lon} {lat})",
            timeout=30,
        )
        response.raise_for_status()
        endpoints = {k for k, v in response.json()["outputs"].items() if v == 1}
    except Exception as e:
        print(f"    Site check failed: {e}")
        return pd.DataFrame()

    # Determine the right URL and available years
    if "nsrdb-GOES-aggregated-v4-0-0" in endpoints:
        base_url = f"{pvlib.iotools.psm4.NSRDB_API_BASE}nsrdb-GOES-aggregated-v4-0-0-download.csv"
        available_years = list(range(1998, 2025))
    elif "nsrdb-msg-v1-0-0" in endpoints:
        base_url = f"{pvlib.iotools.psm4.NSRDB_API_BASE}nsrdb-msg-v1-0-0-download.csv"
        available_years = list(range(2005, 2023))
    else:
        print(f"    No NSRDB data available at ({lat}, {lon})")
        return pd.DataFrame()

    # Filter to requested year range
    years_to_fetch = [y for y in available_years if start_year <= y <= end_year]

    parameters = ("ghi", "dni", "air_temperature", "wind_speed")

    for year in years_to_fetch:
        try:
            df_year, metadata = pvlib.iotools.get_nsrdb_psm4_aggregated(
                latitude=lat,
                longitude=lon,
                api_key=api_key,
                email=email,
                year=year,
                time_step=config.NSRDB_TIMESTEP_MIN,
                parameters=parameters,
                leap_day=False,
                url=base_url,
            )

            # Ensure datetime index
            df_year.index = pd.to_datetime(df_year.index)

            # Monthly GHI aggregation
            # GHI values are in W/m², at hourly timestep each value represents Wh/m²
            monthly = (
                df_year.groupby(df_year.index.month)["ghi"]
                .sum()
                .reset_index()
            )
            monthly.columns = ["month", "monthly_ghi_wh_m2"]
            monthly["year"] = year

            all_monthly.append(monthly)

            # Rate limiting for NSRDB
            time.sleep(config.NSRDB_API_DELAY_S)

        except Exception as e:
            print(f"    Year {year} failed: {e}")
            time.sleep(config.NSRDB_API_DELAY_S)
            continue

    if all_monthly:
        return pd.concat(all_monthly, ignore_index=True)

    return pd.DataFrame()


def fetch_nsrdb_via_local_wrapper(
    lat: float,
    lon: float,
    api_key: str,
    email: str,
    start_year: int = config.START_YEAR,
    end_year: int = config.END_YEAR,
) -> pd.DataFrame:
    """
    Use the local nsrdb.py wrapper to fetch NSRDB data.

    This is the preferred approach when the optional wrapper is available.
    """
    weather = get_nsrdb(
        latitude_deg=lat,
        longitude_deg=lon,
        timestep_min=config.NSRDB_TIMESTEP_MIN,
        typical_year=None,
        api_key=api_key,
        email=email,
        use_lock=True,
    )

    years_available = weather.metadata["historical"]["years"]
    all_monthly = []

    for i, df_year in enumerate(weather.historical_years):
        year = years_available[i]
        if start_year <= year <= end_year:
            df_year = df_year.copy()
            df_year.index = pd.to_datetime(df_year.index)

            monthly = (
                df_year.groupby(df_year.index.month)["ghi"]
                .sum()
                .reset_index()
            )
            monthly.columns = ["month", "monthly_ghi_wh_m2"]
            monthly["year"] = year

            all_monthly.append(monthly)

    if all_monthly:
        return pd.concat(all_monthly, ignore_index=True)

    return pd.DataFrame()


def main():
    """Run Step 3: NSRDB weather retrieval for all plants."""

    print("=" * 70)
    print("STEP 3: NSRDB Weather Retrieval")
    print("=" * 70)

    if not config.NREL_API_KEY:
        print("\nERROR: NREL_API_KEY not set in .env file!")
        print("   Register at: https://developer.nrel.gov/signup/")
        sys.exit(1)

    if not config.NREL_EMAIL:
        print("\nERROR: NREL_EMAIL not set in .env file!")
        sys.exit(1)

    # Load plants
    if not config.SOLAR_PLANTS_FILTERED_CSV.exists():
        print(f"\nERROR: {config.SOLAR_PLANTS_FILTERED_CSV} not found. Run Step 1 first.")
        sys.exit(1)

    df_plants = pd.read_csv(config.SOLAR_PLANTS_FILTERED_CSV, dtype={"plant_id": str})

    # Only process plants that have EIA generation data
    if config.EIA_GENERATION_CSV.exists():
        df_gen = pd.read_csv(config.EIA_GENERATION_CSV, dtype={"plant_id": str})
        plants_with_gen = set(df_gen["plant_id"].unique())
        df_plants = df_plants[df_plants["plant_id"].isin(plants_with_gen)]
        print(f"Filtered to {len(df_plants)} plants that have EIA generation data")

    # Check lat/lon availability
    if "latitude" not in df_plants.columns or "longitude" not in df_plants.columns:
        print("\nERROR: Plant data missing latitude/longitude columns")
        sys.exit(1)

    df_plants = df_plants.dropna(subset=["latitude", "longitude"])
    print(f"Plants with valid lat/lon: {len(df_plants)}")

    # Check for existing checkpoint
    existing_plant_ids = set()
    if config.NSRDB_GHI_CSV.exists():
        df_existing = pd.read_csv(config.NSRDB_GHI_CSV, dtype={"plant_id": str})
        existing_plant_ids = set(df_existing["plant_id"].unique())
        print(f"Checkpoint: {len(existing_plant_ids)} plants already downloaded")

    plants_to_fetch = df_plants[~df_plants["plant_id"].isin(existing_plant_ids)]

    print(f"\n Fetching NSRDB data for {len(plants_to_fetch)} plants...")
    print(f"   Estimated time: ~{len(plants_to_fetch) * 1.5:.0f} minutes")
    print(f"   (NSRDB is rate-limited - this is the slow step)\n")

    # Choose retrieval method
    if NSRDB_AVAILABLE:
        fetch_func = fetch_nsrdb_via_local_wrapper
        print("   Using: local nsrdb.py wrapper\n")
    else:
        fetch_func = fetch_nsrdb_standalone
        print("   Using: Standalone pvlib fallback\n")

    all_nsrdb = []
    failed_plants = []

    for idx, (_, row) in enumerate(tqdm(
        plants_to_fetch.iterrows(),
        total=len(plants_to_fetch),
        desc="Fetching NSRDB"
    )):
        plant_id = str(row["plant_id"])
        lat = float(row["latitude"])
        lon = float(row["longitude"])
        name = row.get("plant_name", plant_id)

        try:
            df_ghi = fetch_func(
                lat=lat,
                lon=lon,
                api_key=config.NREL_API_KEY,
                email=config.NREL_EMAIL,
                start_year=config.START_YEAR,
                end_year=config.END_YEAR,
            )

            if not df_ghi.empty:
                df_ghi["plant_id"] = plant_id
                all_nsrdb.append(df_ghi)
            else:
                failed_plants.append((plant_id, name, "No data returned"))

        except Exception as e:
            failed_plants.append((plant_id, name, str(e)))
            traceback.print_exc()

        # Checkpoint every 25 plants
        if (idx + 1) % 25 == 0 and all_nsrdb:
            _save_checkpoint(all_nsrdb, existing_plant_ids)
            # Clear in-memory buffer after persisting; the existing CSV is now
            # the source of truth, so the next checkpoint should not re-append
            # the rows we already saved.
            all_nsrdb = []
            print(f"\n  Checkpoint saved ({idx + 1}/{len(plants_to_fetch)} plants)")

    # Final save
    if all_nsrdb:
        df_new = pd.concat(all_nsrdb, ignore_index=True)

        if config.NSRDB_GHI_CSV.exists():
            df_existing = pd.read_csv(config.NSRDB_GHI_CSV, dtype={"plant_id": str})
            df_all = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_all = df_new

        # Safety dedup: idempotent on clean data, repairs any accidental dups.
        df_all = df_all.drop_duplicates(subset=["plant_id", "year", "month"], keep="last")
        df_all.to_csv(config.NSRDB_GHI_CSV, index=False)
        n_plants = df_all["plant_id"].nunique()
        print(f"\nSaved NSRDB data for {n_plants} plants -> {config.NSRDB_GHI_CSV}")

    # Summary
    print(f"\n{'-' * 50}")
    print(f"SUMMARY")
    print(f"{'-' * 50}")
    print(f"Successfully fetched:  {len(plants_to_fetch) - len(failed_plants)}")
    print(f"Failed:                {len(failed_plants)}")

    if failed_plants:
        print(f"\nFailed plants:")
        for pid, name, err in failed_plants[:10]:
            print(f"  {pid} ({name}): {err}")
        if len(failed_plants) > 10:
            print(f"  ... and {len(failed_plants) - 10} more")

    print(f"\nStep 3 complete.")


def _save_checkpoint(all_nsrdb: list, existing_plant_ids: set):
    """Save intermediate NSRDB results."""
    df_new = pd.concat(all_nsrdb, ignore_index=True)

    if config.NSRDB_GHI_CSV.exists():
        df_existing = pd.read_csv(config.NSRDB_GHI_CSV, dtype={"plant_id": str})
        df_all = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_all = df_new

    # Safety dedup so checkpoints can never inflate the file even if a future
    # caller forgets to clear `all_nsrdb` after a save.
    df_all = df_all.drop_duplicates(subset=["plant_id", "year", "month"], keep="last")
    df_all.to_csv(config.NSRDB_GHI_CSV, index=False)


if __name__ == "__main__":
    main()
