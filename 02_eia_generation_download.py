#!/usr/bin/env python3
"""
Step 2: EIA Monthly Generation Download
========================================
For each filtered solar plant, downloads monthly net generation (MWh)
from the EIA API v2 facility-fuel endpoint.

Inputs:
    data/solar_plants_filtered.csv

Outputs:
    data/eia_monthly_generation.csv
"""

import sys
import time
import requests
import pandas as pd
from tqdm import tqdm

import config


def fetch_eia_monthly_generation(
    plant_id: str,
    api_key: str,
    start_year: int = config.START_YEAR,
    end_year: int = config.END_YEAR,
) -> pd.DataFrame:
    """
    Fetch monthly solar generation for a single plant from EIA API v2.

    Uses the electricity/facility-fuel/data endpoint with:
        - frequency=monthly
        - fuel2002=SUN (solar)
        - plantCode=<plant_id>

    Returns DataFrame with columns: plant_id, period, year, month, generation_mwh
    """
    url = "https://api.eia.gov/v2/electricity/facility-fuel/data/"

    params = {
        "api_key": api_key,
        "frequency": "monthly",
        "data[0]": "generation",
        "facets[plantCode][]": str(plant_id),
        "facets[fuel2002][]": config.SOLAR_FUEL_CODE,
        # Filter to a single prime mover. Without this, EIA returns BOTH a
        # per-prime-mover row (e.g. PV) AND an aggregate row with prime mover
        # "ALL" that has the same value, and our groupby below sums them,
        # doubling every value. Filtering here is the upstream fix.
        "start": f"{start_year}-01",
        "end": f"{end_year}-12",
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "offset": 0,
        "length": 5000,
    }
    if getattr(config, "SOLAR_PRIME_MOVER", None):
        params["facets[primeMover][]"] = config.SOLAR_PRIME_MOVER

    try:
        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()
        data = response.json()

        if "response" in data and "data" in data["response"]:
            records = data["response"]["data"]
            if records:
                df = pd.DataFrame(records)

                # Belt-and-suspenders: even if the API returns a primeMover
                # column with an "ALL" rollup row alongside the requested
                # prime mover, drop it so it cannot be summed in below.
                if "primeMover" in df.columns:
                    df = df[df["primeMover"].astype(str).str.upper() != "ALL"]
                    if df.empty:
                        return pd.DataFrame()

                df["generation"] = pd.to_numeric(df["generation"], errors="coerce")
                df["period"] = df["period"].astype(str)

                # Parse period (format: "YYYY-MM")
                df["date"] = pd.to_datetime(df["period"], format="%Y-%m")
                df["year"] = df["date"].dt.year
                df["month"] = df["date"].dt.month

                # Aggregate across rows within the same plant-month. After
                # the ALL filter above this is at most one row per fuel /
                # generator break, summed back to plant level.
                monthly = (
                    df.groupby(["period", "year", "month"])
                    .agg(generation_mwh=("generation", "sum"))
                    .reset_index()
                )

                monthly["plant_id"] = str(plant_id)

                return monthly[["plant_id", "period", "year", "month", "generation_mwh"]]

    except requests.exceptions.RequestException as e:
        print(f"  ERROR fetching plant {plant_id}: {e}")
    except (KeyError, ValueError) as e:
        print(f"  ERROR parsing data for plant {plant_id}: {e}")

    return pd.DataFrame()


def main():
    """Run Step 2: Download monthly generation for all filtered plants."""

    print("=" * 70)
    print("STEP 2: EIA Monthly Generation Download")
    print("=" * 70)

    if not config.EIA_API_KEY:
        print("\nERROR: EIA_API_KEY not set in .env file.")
        sys.exit(1)

    # Load filtered plants from Step 1
    if not config.SOLAR_PLANTS_FILTERED_CSV.exists():
        print(f"\nERROR: {config.SOLAR_PLANTS_FILTERED_CSV} not found.")
        print("  Run Step 1 first: python 01_eia_plant_discovery.py")
        sys.exit(1)

    df_plants = pd.read_csv(config.SOLAR_PLANTS_FILTERED_CSV, dtype={"plant_id": str})
    print(f"\nLoaded {len(df_plants)} plants from Step 1")

    # Check for existing checkpoint
    existing_plant_ids = set()
    if config.EIA_GENERATION_CSV.exists():
        df_existing = pd.read_csv(config.EIA_GENERATION_CSV, dtype={"plant_id": str})
        existing_plant_ids = set(df_existing["plant_id"].unique())
        print(f"Found existing checkpoint with {len(existing_plant_ids)} plants already downloaded")

    # Download generation for each plant
    all_generation = []
    failed_plants = []

    plants_to_fetch = [
        pid for pid in df_plants["plant_id"].unique()
        if str(pid) not in existing_plant_ids
    ]

    print(f"\nDownloading generation for {len(plants_to_fetch)} plants "
          f"(skipping {len(existing_plant_ids)} already done)...")
    print(f"Year range: {config.START_YEAR}-{config.END_YEAR}")
    print(f"Estimated time: ~{len(plants_to_fetch) * 0.7 / 60:.1f} minutes\n")

    for i, plant_id in enumerate(tqdm(plants_to_fetch, desc="Downloading EIA data")):
        df_gen = fetch_eia_monthly_generation(str(plant_id), config.EIA_API_KEY)

        if not df_gen.empty:
            all_generation.append(df_gen)
        else:
            failed_plants.append(plant_id)

        # Rate limiting
        time.sleep(config.EIA_API_DELAY_S)

        # Periodic checkpoint save (every 100 plants)
        if (i + 1) % 100 == 0 and all_generation:
            _save_checkpoint(all_generation, existing_plant_ids)
            # Clear in-memory buffer after persisting; existing CSV is now the
            # source of truth, so the next checkpoint should not re-append
            # the rows we already saved.
            all_generation = []
            print(f"\n  Checkpoint saved ({i + 1}/{len(plants_to_fetch)} plants)")

    # Final save
    if all_generation:
        df_new = pd.concat(all_generation, ignore_index=True)

        # Merge with any existing data
        if config.EIA_GENERATION_CSV.exists():
            df_existing = pd.read_csv(config.EIA_GENERATION_CSV, dtype={"plant_id": str})
            df_all = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_all = df_new

        # Safety dedup: idempotent on clean data, repairs any accidental dups.
        df_all = df_all.drop_duplicates(subset=["plant_id", "year", "month"], keep="last")
        df_all.to_csv(config.EIA_GENERATION_CSV, index=False)
        print(f"\nSaved {len(df_all)} total monthly records -> {config.EIA_GENERATION_CSV}")
    else:
        print("\nNo new generation data downloaded")

    # Summary
    if config.EIA_GENERATION_CSV.exists():
        df_all = pd.read_csv(config.EIA_GENERATION_CSV, dtype={"plant_id": str})
        n_plants = df_all["plant_id"].nunique()
        n_records = len(df_all)

        print(f"\n{'-' * 50}")
        print(f"SUMMARY")
        print(f"{'-' * 50}")
        print(f"Total plants with data:   {n_plants}")
        print(f"Total monthly records:    {n_records}")
        print(f"Failed plants:            {len(failed_plants)}")

        # Show year coverage distribution
        year_coverage = (
            df_all.groupby("plant_id")["year"]
            .nunique()
            .describe()
        )
        print(f"\nYears of data per plant:")
        print(year_coverage.to_string())

    print(f"\nStep 2 complete.")


def _save_checkpoint(all_generation: list, existing_plant_ids: set):
    """Save intermediate results as checkpoint."""
    df_new = pd.concat(all_generation, ignore_index=True)

    if config.EIA_GENERATION_CSV.exists():
        df_existing = pd.read_csv(config.EIA_GENERATION_CSV, dtype={"plant_id": str})
        df_all = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_all = df_new

    # Safety dedup so checkpoints can never inflate the file even if a future
    # caller forgets to clear `all_generation` after a save.
    df_all = df_all.drop_duplicates(subset=["plant_id", "year", "month"], keep="last")
    df_all.to_csv(config.EIA_GENERATION_CSV, index=False)


if __name__ == "__main__":
    main()
