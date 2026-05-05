# EIA-NSRDB Solar Plant Quality Downselection

This research pipeline screens utility-scale U.S. solar plants for sites where reported monthly electricity generation closely tracks satellite-derived solar resource data. The goal is to identify plants with clean, stable generation-resource coupling that are good candidates for downstream modeling, validation, or benchmarking work.

The analysis combines public EIA generation and plant metadata with NREL NSRDB irradiance data, normalizes generation by plant capacity, computes correlation and stability metrics, and produces a ranked shortlist of high-quality solar plants.

## Key Results

Using 2010-2024 data and the current configuration, the pipeline analyzed 456 solar plants and identified 30 top candidates. In the generated run:

- Mean monthly Pearson correlation: 0.813
- Median monthly Pearson correlation: 0.864
- Tier 1 plants with Pearson R >= 0.90: 111
- Final shortlist size: 30 plants

Curated public outputs are available in `output/`:

- `summary_report.html`: HTML report with summary tables and plots
- `top_candidates.csv`: ranked shortlist
- `correlation_results.csv`: full plant-level metric table
- `correlation_histogram.png`: distribution of monthly Pearson R
- `geographic_quality_map.png`: spatial view of quality tiers
- `ghi_vs_yield_scatter.png`: generation-yield relationship plot
- `annual_vs_monthly_r.png`: annual vs. monthly correlation comparison

## Methodology

The pipeline runs in five checkpointed stages:

1. Discover and filter EIA solar plants by fuel type, capacity, location, and operating history.
2. Download monthly EIA facility-fuel generation for each filtered plant.
3. Retrieve monthly NSRDB Global Horizontal Irradiance (GHI) at each plant location.
4. Merge generation and irradiance data, compute plant-level metrics, and rank candidates.
5. Generate plots, CSV exports, and an HTML summary report.

Primary ranking features include monthly Pearson R, annual Pearson R, Spearman correlation, performance-ratio stability, and data completeness. Generation is normalized as MWh per MWac so plants of different sizes can be compared on the same basis.

## Data Sources

- EIA API v2 for monthly plant-level solar generation
- EIA-860 for plant metadata and solar plant characteristics
- NREL NSRDB for historical irradiance time series

API credentials are required for a full rerun. Local checkpoint files are intentionally excluded from git because they are large, reproducible, and may include raw public-data extracts that are better regenerated than versioned.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Add credentials to `.env`:

```bash
EIA_API_KEY=your_eia_key
NREL_API_KEY=your_nrel_key
NREL_EMAIL=your_email
```

API registration:

- EIA: https://www.eia.gov/opendata/register.php
- NREL: https://developer.nrel.gov/signup/

## Usage

Run the full pipeline:

```bash
python run_pipeline.py
```

Resume from a specific step:

```bash
python run_pipeline.py --from 3
```

Run one step only:

```bash
python run_pipeline.py --only 4
```

Individual scripts can also be run directly:

```bash
python 01_eia_plant_discovery.py
python 02_eia_generation_download.py
python 03_nsrdb_weather_retrieval.py
python 04_correlation_analysis.py
python 05_report_and_visuals.py
```

Step 3 is the slow stage because NSRDB requests are rate-limited and each plant requires historical weather retrieval. Steps 4 and 5 are fast and can be rerun after changing scoring or visualization logic.

## Project Structure

```text
.
├── 01_eia_plant_discovery.py
├── 02_eia_generation_download.py
├── 03_nsrdb_weather_retrieval.py
├── 04_correlation_analysis.py
├── 05_report_and_visuals.py
├── config.py
├── run_pipeline.py
├── requirements.txt
├── README.md
├── .env.example
└── output/
    ├── summary_report.html
    ├── top_candidates.csv
    ├── correlation_results.csv
    ├── correlation_histogram.png
    ├── geographic_quality_map.png
    ├── ghi_vs_yield_scatter.png
    └── annual_vs_monthly_r.png
```

## Notes

The repository is structured as a public research artifact. Secrets, local caches, raw/intermediate data, exploratory notebooks, Python bytecode, and non-curated generated files are ignored by git. The included outputs document one completed run, while the source pipeline remains reproducible with valid API credentials.
