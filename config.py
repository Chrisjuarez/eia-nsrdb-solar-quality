"""
Central configuration for the EIA-NSRDB Solar Plant Quality Pipeline.

All tunable parameters live here so you can adjust without editing pipeline scripts.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------
# API Keys
# ---------------------------------------------
EIA_API_KEY = os.getenv("EIA_API_KEY", "")
NREL_API_KEY = os.getenv("NREL_API_KEY", "")
NREL_EMAIL = os.getenv("NREL_EMAIL", "")

# ---------------------------------------------
# Paths
# ---------------------------------------------
PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "output"
PLOT_DIR = OUTPUT_DIR / "plant_timeseries_plots"

DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)

# Intermediate checkpoint files
EIA_860_ZIP_CACHE = DATA_DIR / "eia_860.zip"
EIA_860_PLANTS_CSV = DATA_DIR / "eia_860_plants.csv"
EIA_860_SOLAR_DETAILS_CSV = DATA_DIR / "eia_860_solar_details.csv"
EIA_860_PLANT_GEOMETRY_CSV = DATA_DIR / "eia_860_plant_geometry.csv"
SOLAR_PLANTS_FILTERED_CSV = DATA_DIR / "solar_plants_filtered.csv"
EIA_GENERATION_CSV = DATA_DIR / "eia_monthly_generation.csv"
NSRDB_GHI_CSV = DATA_DIR / "nsrdb_monthly_ghi.csv"
NSRDB_FEATURES_CSV = DATA_DIR / "nsrdb_monthly_features.csv"

# Final outputs
CORRELATION_RESULTS_CSV = OUTPUT_DIR / "correlation_results.csv"
TOP_CANDIDATES_CSV = OUTPUT_DIR / "top_candidates.csv"
SUMMARY_REPORT_HTML = OUTPUT_DIR / "summary_report.html"

# ---------------------------------------------
# Pipeline Parameters
# ---------------------------------------------

# Year range for analysis
START_YEAR = 2010
END_YEAR = 2024

# Minimum number of years a plant must have data for
MIN_YEARS_OF_DATA = 10

# Minimum number of monthly data points required (e.g., 10 years × 12 months × 80%)
MIN_MONTHLY_DATAPOINTS = 96

# Minimum plant capacity in MW (filter out tiny installations)
MIN_CAPACITY_MW = 5.0

# EIA fuel code for solar
SOLAR_FUEL_CODE = "SUN"

# Prime mover code to filter EIA generation data on. PV-only is the right
# choice for PV-validation work because it both excludes CSP plants (prime
# mover ST, e.g. Genesis Solar) and avoids EIA's "ALL" rollup row, which
# duplicates the per-prime-mover values and was the source of an earlier
# 2x double-counting bug. Set to None to disable the filter.
SOLAR_PRIME_MOVER = "PV"

# NSRDB timestep in minutes (60 = hourly)
NSRDB_TIMESTEP_MIN = 60

# Correlation thresholds for quality tiers
TIER1_PEARSON_R = 0.90   # Excellent
TIER2_PEARSON_R = 0.80   # Good
TIER3_PEARSON_R = 0.70   # Acceptable

# Target number of top candidates to select
TARGET_NUM_CANDIDATES = 30

# ---------------------------------------------
# Rate Limiting (seconds between API calls)
# ---------------------------------------------
EIA_API_DELAY_S = 0.5
NSRDB_API_DELAY_S = 2.0

# ---------------------------------------------
# EIA-860 Bulk Data URL (plant metadata with lat/lon)
# Updated annually - this is the latest release
# ---------------------------------------------
EIA_860_URL = "https://www.eia.gov/electricity/data/eia860/xls/eia8602024.zip"
