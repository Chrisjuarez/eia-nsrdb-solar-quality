#!/usr/bin/env python3
"""
Step 5: Report & Visualizations
================================
Generates the report PNGs and HTML summary from the step-04 output CSVs.

Plots generated:
    1. Correlation distribution histogram (all plants)
    2. Top candidates scatter: GHI vs Normalized Yield per plant
    3. Geographic map of plant quality tiers
    4. Time-series overlays (GHI + generation) for top N plants
    5. Annual correlation vs monthly correlation scatter
    6. Composite score vs capacity scatter

Inputs:
    output/correlation_results.csv
    output/top_candidates.csv
    data/eia_monthly_generation.csv
    data/nsrdb_monthly_ghi.csv
    data/solar_plants_filtered.csv

Outputs:
    output/plant_timeseries_plots/  (per-plant timeseries PNGs)
    output/summary_report.html
    output/correlation_histogram.png
    output/ghi_vs_yield_scatter.png
    output/geographic_quality_map.png
    output/annual_vs_monthly_r.png
"""

import sys
import json
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import requests
from pathlib import Path
from datetime import datetime

import config


# Public US states GeoJSON used as a basemap. Cached locally after the first
# download so the pipeline does not need network access on subsequent runs.
US_STATES_GEOJSON_URL = (
    "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/"
    "master/data/geojson/us-states.json"
)


def _load_us_states_geojson():
    """Return a parsed GeoJSON FeatureCollection of US state boundaries.

    Caches the file under data/ so subsequent runs don't need network access.
    """
    cache_path = config.DATA_DIR / "us_states.geojson"
    if not cache_path.exists():
        try:
            text = requests.get(US_STATES_GEOJSON_URL, timeout=30).text
            cache_path.write_text(text)
        except Exception as e:
            print(f"  Could not fetch US states basemap: {e}")
            return None
    try:
        return json.loads(cache_path.read_text())
    except Exception as e:
        print(f"  Could not parse cached US states basemap: {e}")
        return None


def _draw_us_states(ax, geojson, fill_color="#f5f5f0", edge_color="#9aa0a6"):
    """Draw filled state polygons onto an axes in plain lon/lat coordinates."""
    if geojson is None:
        return
    for feature in geojson.get("features", []):
        geom = feature.get("geometry", {})
        gtype = geom.get("type")
        coords = geom.get("coordinates", [])
        if gtype == "Polygon":
            polygons = [coords]
        elif gtype == "MultiPolygon":
            polygons = coords
        else:
            continue
        for polygon in polygons:
            outer = polygon[0]
            xs = [pt[0] for pt in outer]
            ys = [pt[1] for pt in outer]
            ax.fill(xs, ys, facecolor=fill_color, edgecolor=edge_color,
                    linewidth=0.6, zorder=1)

warnings.filterwarnings("ignore", category=FutureWarning)
sns.set_theme(style="whitegrid", palette="deep", font_scale=1.1)

# Color scheme
TIER_COLORS = {
    "Tier 1 (Excellent)": "#2ecc71",
    "Tier 2 (Good)": "#3498db",
    "Tier 3 (Acceptable)": "#f39c12",
    "Below Threshold": "#e74c3c",
    "Insufficient Data": "#95a5a6",
}


def plot_correlation_distribution(df: pd.DataFrame, outdir: Path):
    """Plot 1: Histogram of Pearson R across all plants."""
    fig, ax = plt.subplots(figsize=(10, 6))

    r_vals = df["pearson_r"].dropna()

    ax.hist(r_vals, bins=30, edgecolor="white", alpha=0.8, color="#3498db")

    # Add threshold lines
    for threshold, label, color in [
        (config.TIER1_PEARSON_R, f"Tier 1 (≥{config.TIER1_PEARSON_R})", "#2ecc71"),
        (config.TIER2_PEARSON_R, f"Tier 2 (≥{config.TIER2_PEARSON_R})", "#f39c12"),
        (config.TIER3_PEARSON_R, f"Tier 3 (≥{config.TIER3_PEARSON_R})", "#e74c3c"),
    ]:
        ax.axvline(threshold, color=color, linestyle="--", linewidth=2, label=label)

    ax.set_xlabel("Monthly Pearson R (Generation vs GHI)")
    ax.set_ylabel("Number of Plants")
    ax.set_title("Distribution of GHI-Generation Correlation Across Solar Plants")
    ax.legend(loc="upper left")

    # Add stats text
    stats_text = (
        f"N = {len(r_vals)}\n"
        f"Mean R = {r_vals.mean():.3f}\n"
        f"Median R = {r_vals.median():.3f}\n"
        f"≥0.90: {(r_vals >= 0.90).sum()} plants"
    )
    ax.text(0.02, 0.75, stats_text, transform=ax.transAxes,
            fontsize=11, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.8))

    plt.tight_layout()
    path = outdir / "correlation_histogram.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_geographic_quality(df: pd.DataFrame, outdir: Path):
    """Plot 3: Geographic scatter of plants colored by quality tier, on a US basemap."""
    if "latitude" not in df.columns or "longitude" not in df.columns:
        print("  No lat/lon data - skipping geographic plot")
        return

    df_plot = df.dropna(subset=["latitude", "longitude", "pearson_r"])

    fig, ax = plt.subplots(figsize=(14, 8))

    # Draw state boundaries first so plant points sit on top.
    geojson = _load_us_states_geojson()
    _draw_us_states(ax, geojson)

    for tier, color in TIER_COLORS.items():
        mask = df_plot["quality_tier"] == tier
        if mask.any():
            ax.scatter(
                df_plot.loc[mask, "longitude"],
                df_plot.loc[mask, "latitude"],
                c=color,
                label=f"{tier} ({mask.sum()})",
                s=30 + df_plot.loc[mask, "capacity_mw"].fillna(10).clip(upper=500) * 0.1,
                alpha=0.85,
                edgecolors="white",
                linewidth=0.4,
                zorder=3,
            )

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Solar Plant Quality Tiers Across CONUS")
    ax.legend(loc="lower left", fontsize=9, framealpha=0.95)
    ax.set_xlim(-128, -64)
    ax.set_ylim(23, 51)
    ax.set_aspect(1.3)  # Approximate equal-area look at mid-CONUS latitudes

    plt.tight_layout()
    path = outdir / "geographic_quality_map.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_plant_timeseries(
    plant_id: str,
    plant_name: str,
    df_gen: pd.DataFrame,
    df_ghi: pd.DataFrame,
    pearson_r: float,
    outdir: Path,
):
    """
    Plot 4: Dual-axis timeseries for a single plant.
    Blue = GHI (left axis), Red = Generation/Yield (right axis).
    Uses a dual-axis monthly overlay to compare irradiance and generation.
    """
    gen = df_gen[df_gen["plant_id"] == plant_id].copy()
    ghi = df_ghi[df_ghi["plant_id"] == plant_id].copy()

    if gen.empty or ghi.empty:
        return

    gen["date"] = pd.to_datetime(gen["period"])
    ghi["date"] = pd.to_datetime(ghi["year"].astype(str) + "-" + ghi["month"].astype(str).str.zfill(2))

    merged = pd.merge(gen, ghi, on=["plant_id", "year", "month"], how="inner", suffixes=("_gen", "_ghi"))
    merged = merged.sort_values("date_gen")

    if len(merged) < 12:
        return

    fig, ax1 = plt.subplots(figsize=(14, 5))

    # GHI on left axis (blue)
    color_ghi = "#3498db"
    ax1.plot(merged["date_gen"], merged["monthly_ghi_wh_m2"],
             color=color_ghi, alpha=0.8, linewidth=1.5, label="Monthly GHI")
    ax1.fill_between(merged["date_gen"], merged["monthly_ghi_wh_m2"],
                     alpha=0.15, color=color_ghi)
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Monthly GHI (Wh/m²)", color=color_ghi)
    ax1.tick_params(axis="y", labelcolor=color_ghi)

    # Generation on right axis (red)
    ax2 = ax1.twinx()
    color_gen = "#e74c3c"
    ax2.plot(merged["date_gen"], merged["generation_mwh"],
             color=color_gen, alpha=0.8, linewidth=1.5, label="Monthly Generation")
    ax2.fill_between(merged["date_gen"], merged["generation_mwh"],
                     alpha=0.15, color=color_gen)
    ax2.set_ylabel("Monthly Generation (MWh)", color=color_gen)
    ax2.tick_params(axis="y", labelcolor=color_gen)

    # Title with correlation
    name_short = plant_name[:50] if plant_name else plant_id
    ax1.set_title(
        f"{name_short} (ID: {plant_id})  -  Pearson R = {pearson_r:.3f}",
        fontsize=12, fontweight="bold",
    )

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)

    plt.tight_layout()
    safe_name = "".join(c if c.isalnum() or c in "-_ " else "" for c in str(plant_id))
    path = outdir / f"timeseries_{safe_name}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()


def generate_html_report(
    df_results: pd.DataFrame,
    df_top: pd.DataFrame,
    outdir: Path,
):
    """Generate an HTML summary report with embedded tables and plot references."""

    n_total = len(df_results)
    r_vals = df_results["pearson_r"].dropna()

    tier_counts = df_results["quality_tier"].value_counts().to_dict()

    # Build top candidates table rows
    top_rows = ""
    for _, row in df_top.head(30).iterrows():
        name = str(row.get("plant_name", row["plant_id"]))[:40]
        top_rows += f"""
        <tr>
            <td>{int(row.get('rank', 0))}</td>
            <td>{row['plant_id']}</td>
            <td>{name}</td>
            <td>{row.get('state', '-')}</td>
            <td>{row.get('capacity_mw', 0):.1f}</td>
            <td><strong>{row.get('pearson_r', 0):.3f}</strong></td>
            <td>{row.get('spearman_rho', 0):.3f}</td>
            <td>{row.get('annual_pearson_r', 0):.3f}</td>
            <td>{row.get('composite_score', 0):.3f}</td>
            <td>{int(row.get('n_years', 0))}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>EIA-NSRDB Solar Plant Quality Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1100px;
            margin: 40px auto;
            padding: 0 20px;
            color: #2c3e50;
            line-height: 1.6;
        }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 40px; }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: #f8f9fa;
            border-left: 4px solid #3498db;
            padding: 15px;
            border-radius: 4px;
        }}
        .stat-card .value {{ font-size: 28px; font-weight: bold; color: #2c3e50; }}
        .stat-card .label {{ font-size: 13px; color: #7f8c8d; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-size: 13px;
        }}
        th {{ background: #34495e; color: white; padding: 10px 8px; text-align: left; }}
        td {{ padding: 8px; border-bottom: 1px solid #ecf0f1; }}
        tr:hover {{ background: #f5f6fa; }}
        .tier-badge {{
            display: inline-block;
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: bold;
            color: white;
        }}
        .plot-container {{ margin: 20px 0; text-align: center; }}
        .plot-container img {{ max-width: 100%; border: 1px solid #ecf0f1; border-radius: 4px; }}
        .note {{ background: #fef9e7; border-left: 4px solid #f39c12; padding: 15px; margin: 20px 0; }}
    </style>
</head>
<body>
    <h1>EIA-NSRDB Solar Plant Quality Report</h1>

    <h2>Summary Statistics</h2>
    <div class="stats-grid">
        <div class="stat-card">
            <div class="value">{n_total}</div>
            <div class="label">Plants Analyzed</div>
        </div>
        <div class="stat-card">
            <div class="value">{r_vals.mean():.3f}</div>
            <div class="label">Mean Pearson R</div>
        </div>
        <div class="stat-card">
            <div class="value">{r_vals.median():.3f}</div>
            <div class="label">Median Pearson R</div>
        </div>
        <div class="stat-card">
            <div class="value">{tier_counts.get('Tier 1 (Excellent)', 0)}</div>
            <div class="label">Tier 1 Plants (R ≥ {config.TIER1_PEARSON_R})</div>
        </div>
        <div class="stat-card">
            <div class="value">{tier_counts.get('Tier 2 (Good)', 0)}</div>
            <div class="label">Tier 2 Plants (R ≥ {config.TIER2_PEARSON_R})</div>
        </div>
        <div class="stat-card">
            <div class="value">{config.START_YEAR}-{config.END_YEAR}</div>
            <div class="label">Analysis Period</div>
        </div>
    </div>

    <h2>Quality Tier Distribution</h2>
    <div class="plot-container">
        <img src="correlation_histogram.png" alt="Correlation Distribution">
    </div>

    <h2>Geographic Distribution</h2>
    <div class="plot-container">
        <img src="geographic_quality_map.png" alt="Geographic Quality Map">
    </div>

    <h2>Top {min(30, len(df_top))} Candidates</h2>
    <div class="note">
        <strong>Selection criteria:</strong> Composite score combining monthly Pearson R (40%),
        annual Pearson R (20%), Spearman ρ (20%), performance ratio stability (10%),
        and data completeness (10%). Plants with R ≥ 0.90 are classified as "Excellent."
    </div>
    <table>
        <thead>
            <tr>
                <th>Rank</th>
                <th>Plant ID</th>
                <th>Name</th>
                <th>State</th>
                <th>MW</th>
                <th>Pearson R</th>
                <th>Spearman ρ</th>
                <th>Annual R</th>
                <th>Score</th>
                <th>Years</th>
            </tr>
        </thead>
        <tbody>
            {top_rows}
        </tbody>
    </table>

    <h2>Methodology</h2>

    <h3>Data sources</h3>
    <p><strong>Plant generation:</strong> EIA API v2 facility-fuel endpoint
    (<code>electricity/facility-fuel/data</code>), filtered to <code>fuel2002 = SUN</code>,
    monthly frequency. The <code>generation</code> field is net generation in MWh
    (post-inverter, after parasitic and station-service losses are subtracted).
    Multiple generators within the same plant-month are summed, then duplicate
    rows on <code>(plant_id, year, month)</code> are dropped before merging.</p>
    <p><strong>Plant capacity and location:</strong> EIA API v2
    operating-generator-capacity endpoint
    (<code>electricity/operating-generator-capacity/data</code>), pulled for the final
    month of the analysis window. <code>nameplate-capacity-mw</code> is summed across
    every solar generator at the plant. Latitude and longitude come from the same
    response.</p>
    <p><strong>Solar resource:</strong> NREL NSRDB v4 (PSM aggregated GOES product
    where available, MSG fallback otherwise) at the plant lat/lon. Hourly GHI
    (W/m²) is summed within each calendar month to give a monthly total in
    Wh/m². Years 2010 through 2024 are pulled for every plant.</p>

    <h3>Filtering</h3>
    <p>Plants must have AC nameplate capacity at or above
    {config.MIN_CAPACITY_MW} MW, valid lat/lon within CONUS bounds, and at least
    {config.MIN_YEARS_OF_DATA} years of EIA generation history within the
    analysis window ({config.START_YEAR}-{config.END_YEAR}).</p>

    <h3>Merge and per-plant aggregation</h3>
    <p>Generation and GHI are inner-joined on
    <code>(plant_id, year, month)</code>. Plants without overlapping records
    in both feeds are dropped. Each plant's row set is then handed to the
    metric routine, which produces one row of statistics per plant.</p>

    <h3>Metrics</h3>
    <p><strong>Monthly Pearson R</strong> = scipy.stats.pearsonr applied to
    paired (monthly GHI, monthly normalized yield) values, where normalized
    yield is generation_MWh divided by AC capacity_MW. Captures how tightly
    month-to-month output tracks month-to-month irradiance.</p>
    <p><strong>Spearman ρ</strong> = scipy.stats.spearmanr on the same paired
    values. Rank-based, so robust to outliers and any monotonic but
    non-linear response.</p>
    <p><strong>Annual Pearson R</strong> = Pearson on yearly sums of
    generation against yearly sums of GHI. Smooths short-term noise; needs
    at least 5 years to compute.</p>
    <p><strong>Performance ratio (PR)</strong> = monthly generation_MWh
    divided by (capacity_MW × monthly GHI in kWh/m²). PR is a unitless
    measure of how much of the resource the plant converts to electricity.
    The pipeline reports <em>mean PR</em> and <em>CV of PR</em>
    (standard deviation / mean), which captures temporal stability rather
    than absolute level. PR magnitudes can look high for plants with large
    DC overbuild relative to their reported AC nameplate, so the CV is the
    more reliable comparison signal.</p>
    <p><strong>R²</strong> = monthly Pearson R squared.</p>

    <h3>Composite score</h3>
    <p>Each plant gets a single 0-to-1 score combining the metrics:</p>
    <ul>
        <li>40 percent monthly Pearson R</li>
        <li>20 percent annual Pearson R</li>
        <li>20 percent Spearman ρ</li>
        <li>10 percent (1 minus CV of PR), so lower volatility scores higher</li>
        <li>10 percent data completeness, defined as min(1, n_years / 15)</li>
    </ul>
    <p>If a metric can't be computed for a plant (for example, fewer than five
    years for the annual R), its weight is dropped and the remaining weights
    are renormalized.</p>

    <h3>Quality tiers</h3>
    <p>Tiers are assigned solely on monthly Pearson R: Tier 1 at
    R &ge; {config.TIER1_PEARSON_R}, Tier 2 at R &ge; {config.TIER2_PEARSON_R},
    Tier 3 at R &ge; {config.TIER3_PEARSON_R}, otherwise "Below Threshold."
    Plants where Pearson R cannot be computed are tagged "Insufficient Data."</p>

</body>
</html>"""

    path = outdir / "summary_report.html"
    path.write_text(html)
    print(f"  Saved: {path}")


def main():
    """Run Step 5: Generate all visualizations and report."""

    print("=" * 70)
    print("STEP 5: Report & Visualizations")
    print("=" * 70)

    # -- Load Data --
    if not config.CORRELATION_RESULTS_CSV.exists():
        print(f"\n ERROR: {config.CORRELATION_RESULTS_CSV} not found. Run Step 4 first.")
        sys.exit(1)

    df_results = pd.read_csv(config.CORRELATION_RESULTS_CSV, dtype={"plant_id": str})
    df_top = pd.read_csv(config.TOP_CANDIDATES_CSV, dtype={"plant_id": str})

    print(f"Loaded {len(df_results)} plant results, {len(df_top)} top candidates")

    # Load raw data for timeseries plots
    df_gen = pd.DataFrame()
    df_ghi = pd.DataFrame()
    df_merged = pd.DataFrame()

    if config.EIA_GENERATION_CSV.exists() and config.NSRDB_GHI_CSV.exists():
        df_gen = pd.read_csv(config.EIA_GENERATION_CSV, dtype={"plant_id": str})
        df_ghi = pd.read_csv(config.NSRDB_GHI_CSV, dtype={"plant_id": str})

        # Build merged set for scatter plots
        df_gen["year"] = df_gen["year"].astype(int)
        df_gen["month"] = df_gen["month"].astype(int)
        df_ghi["year"] = df_ghi["year"].astype(int)
        df_ghi["month"] = df_ghi["month"].astype(int)

        df_merged = pd.merge(df_gen, df_ghi, on=["plant_id", "year", "month"], how="inner")

        # Add capacity for normalized yield
        cap_map = dict(zip(df_results["plant_id"], df_results.get("capacity_mw", pd.Series(dtype=float))))
        df_merged["capacity_mw"] = df_merged["plant_id"].map(cap_map)
        df_merged["normalized_yield"] = df_merged["generation_mwh"] / df_merged["capacity_mw"]

    outdir = config.OUTPUT_DIR

    # -- Generate Plots --
    print("\nGenerating visualizations...")

    # Plot 1: Correlation distribution
    plot_correlation_distribution(df_results, outdir)

    # Plot 2: Geographic quality map
    plot_geographic_quality(df_results, outdir)

    # Plot 4: per-plant timeseries, one per top candidate.
    if not df_gen.empty and not df_ghi.empty:
        n_plots = min(config.TARGET_NUM_CANDIDATES, len(df_top))
        print(f"\n  Generating timeseries plots for top {n_plots} plants...")
        config.PLOT_DIR.mkdir(exist_ok=True)

        # Clear stale PNGs from previous runs so the folder always reflects
        # the current top-candidates list and there are no orphaned plots.
        for old in config.PLOT_DIR.glob("timeseries_*.png"):
            try:
                old.unlink()
            except OSError:
                pass

        for _, row in df_top.head(n_plots).iterrows():
            plot_plant_timeseries(
                plant_id=row["plant_id"],
                plant_name=row.get("plant_name", row["plant_id"]),
                df_gen=df_gen,
                df_ghi=df_ghi,
                pearson_r=row.get("pearson_r", 0),
                outdir=config.PLOT_DIR,
            )

        print(f"  Saved timeseries plots -> {config.PLOT_DIR}")

    # -- HTML Report --
    print("\nGenerating HTML summary report...")
    generate_html_report(df_results, df_top, outdir)

    print(f"\nStep 5 complete. All outputs in: {outdir}")


if __name__ == "__main__":
    main()
