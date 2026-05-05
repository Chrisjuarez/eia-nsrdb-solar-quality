#!/usr/bin/env python3
"""
Step 4: Correlation Analysis & Ranking
========================================
Merges EIA monthly generation with NSRDB monthly GHI, computes normalized
yield, calculates correlation metrics per plant, and ranks plants by quality.

Metrics computed per plant:
    - Pearson R (monthly generation vs GHI)
    - Spearman ρ (monotonic relationship)
    - R² (coefficient of determination)
    - Annual Pearson R (annual totals)
    - CV of Performance Ratio (stability over time)
    - Number of data points / years

Inputs:
    data/solar_plants_filtered.csv
    data/eia_monthly_generation.csv
    data/nsrdb_monthly_ghi.csv

Outputs:
    output/correlation_results.csv   (all plants with metrics)
    output/top_candidates.csv        (top ~30 plants)
"""

import sys
import pandas as pd
import numpy as np
from scipy import stats

import config


def compute_plant_metrics(group: pd.DataFrame) -> dict:
    """
    Compute all quality metrics for a single plant's merged data.

    Args:
        group: DataFrame with columns: year, month, generation_mwh,
               monthly_ghi_wh_m2, normalized_yield, capacity_mw

    Returns:
        Dictionary of computed metrics
    """
    metrics = {}

    # Basic counts
    metrics["n_monthly_points"] = len(group)
    metrics["n_years"] = group["year"].nunique()
    metrics["year_min"] = group["year"].min()
    metrics["year_max"] = group["year"].max()
    metrics["capacity_mw"] = group["capacity_mw"].iloc[0] if "capacity_mw" in group.columns else np.nan

    # Total generation
    metrics["total_generation_gwh"] = group["generation_mwh"].sum() / 1000.0
    metrics["mean_monthly_gen_mwh"] = group["generation_mwh"].mean()

    # -- Monthly Correlation Metrics --
    ghi = group["monthly_ghi_wh_m2"].values
    gen = group["generation_mwh"].values
    nyield = group["normalized_yield"].values if "normalized_yield" in group.columns else gen

    # Drop NaN pairs
    valid = np.isfinite(ghi) & np.isfinite(nyield) & (nyield > 0) & (ghi > 0)
    ghi_clean = ghi[valid]
    nyield_clean = nyield[valid]

    if len(ghi_clean) >= 12:  # Need at least 1 year of monthly data
        # Pearson R (linear correlation)
        r, p_value = stats.pearsonr(ghi_clean, nyield_clean)
        metrics["pearson_r"] = r
        metrics["pearson_p"] = p_value
        metrics["r_squared"] = r ** 2

        # Spearman ρ (rank/monotonic correlation)
        rho, rho_p = stats.spearmanr(ghi_clean, nyield_clean)
        metrics["spearman_rho"] = rho
        metrics["spearman_p"] = rho_p

        # Linear regression slope and intercept
        slope, intercept, _, _, stderr = stats.linregress(ghi_clean, nyield_clean)
        metrics["regression_slope"] = slope
        metrics["regression_intercept"] = intercept
        metrics["regression_stderr"] = stderr

    else:
        metrics["pearson_r"] = np.nan
        metrics["pearson_p"] = np.nan
        metrics["r_squared"] = np.nan
        metrics["spearman_rho"] = np.nan
        metrics["spearman_p"] = np.nan
        metrics["regression_slope"] = np.nan
        metrics["regression_intercept"] = np.nan
        metrics["regression_stderr"] = np.nan

    # -- Annual Correlation --
    annual = group.groupby("year").agg(
        annual_gen=("generation_mwh", "sum"),
        annual_ghi=("monthly_ghi_wh_m2", "sum"),
    ).dropna()

    if len(annual) >= 5:  # Need at least 5 years for annual correlation
        r_ann, p_ann = stats.pearsonr(annual["annual_ghi"], annual["annual_gen"])
        metrics["annual_pearson_r"] = r_ann
        metrics["annual_pearson_p"] = p_ann
    else:
        metrics["annual_pearson_r"] = np.nan
        metrics["annual_pearson_p"] = np.nan

    # -- Performance Ratio Stability --
    # PR = generation_actual / generation_theoretical, where
    #   generation_theoretical_MWh = capacity_MW × GHI_kWh/m² (peak-sun-hours form)
    # GHI is stored in Wh/m² per month, so divide by 1000 to get kWh/m².
    # (Earlier `/ 1e6` was off by 1000× and put every PR value above the
    # outlier cap, which is why these columns were blank in past runs.)
    if "capacity_mw" in group.columns and group["capacity_mw"].iloc[0] > 0:
        cap = group["capacity_mw"].iloc[0]
        pr = group["generation_mwh"] / (cap * group["monthly_ghi_wh_m2"] / 1000.0)
        pr_valid = pr[np.isfinite(pr) & (pr > 0) & (pr < 5)]  # Filter outliers

        if len(pr_valid) >= 12:
            metrics["mean_performance_ratio"] = pr_valid.mean()
            metrics["std_performance_ratio"] = pr_valid.std()
            metrics["cv_performance_ratio"] = pr_valid.std() / pr_valid.mean()
        else:
            metrics["mean_performance_ratio"] = np.nan
            metrics["std_performance_ratio"] = np.nan
            metrics["cv_performance_ratio"] = np.nan
    else:
        metrics["mean_performance_ratio"] = np.nan
        metrics["std_performance_ratio"] = np.nan
        metrics["cv_performance_ratio"] = np.nan

    # -- Year-over-Year Degradation Check --
    # Compute annual capacity factor trend
    if len(annual) >= 5 and "capacity_mw" in group.columns:
        cap = group["capacity_mw"].iloc[0]
        annual["cf"] = annual["annual_gen"] / (cap * 8760)  # Capacity factor

        if len(annual["cf"].dropna()) >= 5:
            slope_cf, _, _, _, _ = stats.linregress(
                annual.index.astype(float), annual["cf"].values
            )
            metrics["annual_cf_trend"] = slope_cf  # Negative = degradation
        else:
            metrics["annual_cf_trend"] = np.nan
    else:
        metrics["annual_cf_trend"] = np.nan

    return metrics


def compute_composite_score(row: pd.Series) -> float:
    """
    Compute a composite quality score combining multiple metrics.

    Weights:
        - 40% Monthly Pearson R (primary resource-generation signal)
        - 20% Annual Pearson R (year-over-year consistency)
        - 20% Spearman ρ (robust to outliers)
        - 10% CV of Performance Ratio (lower = more stable)
        - 10% Data completeness (more years = better)

    Returns score between 0 and 1 (higher = better quality).
    """
    score = 0.0
    weights_used = 0.0

    # Monthly Pearson R (0 to 1)
    if np.isfinite(row.get("pearson_r", np.nan)):
        score += 0.40 * max(0, row["pearson_r"])
        weights_used += 0.40

    # Annual Pearson R
    if np.isfinite(row.get("annual_pearson_r", np.nan)):
        score += 0.20 * max(0, row["annual_pearson_r"])
        weights_used += 0.20

    # Spearman ρ
    if np.isfinite(row.get("spearman_rho", np.nan)):
        score += 0.20 * max(0, row["spearman_rho"])
        weights_used += 0.20

    # CV of PR (lower is better - invert and cap)
    if np.isfinite(row.get("cv_performance_ratio", np.nan)):
        cv = row["cv_performance_ratio"]
        cv_score = max(0, 1 - cv)  # CV of 0 -> score 1, CV of 1 -> score 0
        score += 0.10 * cv_score
        weights_used += 0.10

    # Data completeness (normalized to 0-1, where 15 years = 1.0)
    n_years = row.get("n_years", 0)
    completeness = min(1.0, n_years / 15.0)
    score += 0.10 * completeness
    weights_used += 0.10

    # Normalize by weights actually used
    if weights_used > 0:
        return score / weights_used
    return 0.0


def assign_quality_tier(pearson_r: float) -> str:
    """Assign quality tier based on Pearson R threshold."""
    if np.isnan(pearson_r):
        return "Insufficient Data"
    if pearson_r >= config.TIER1_PEARSON_R:
        return "Tier 1 (Excellent)"
    if pearson_r >= config.TIER2_PEARSON_R:
        return "Tier 2 (Good)"
    if pearson_r >= config.TIER3_PEARSON_R:
        return "Tier 3 (Acceptable)"
    return "Below Threshold"


def main():
    """Run Step 4: Correlation analysis and plant ranking."""

    print("=" * 70)
    print("STEP 4: Correlation Analysis & Ranking")
    print("=" * 70)

    # -- Load Data --
    for path, label in [
        (config.EIA_GENERATION_CSV, "EIA generation"),
        (config.NSRDB_GHI_CSV, "NSRDB GHI"),
        (config.SOLAR_PLANTS_FILTERED_CSV, "Plant metadata"),
    ]:
        if not path.exists():
            print(f"\nERROR: {path} not found ({label}).")
            print("  Run previous steps first.")
            sys.exit(1)

    df_gen = pd.read_csv(config.EIA_GENERATION_CSV, dtype={"plant_id": str})
    df_ghi = pd.read_csv(config.NSRDB_GHI_CSV, dtype={"plant_id": str})
    df_plants = pd.read_csv(config.SOLAR_PLANTS_FILTERED_CSV, dtype={"plant_id": str})

    # Defensive dedup - earlier versions of steps 02 and 03 had a checkpoint
    # bug that wrote duplicate rows. Drop any (plant, year, month) duplicates
    # so the merge below is 1:1 even when reading legacy CSVs.
    pre_gen, pre_ghi = len(df_gen), len(df_ghi)
    df_gen = df_gen.drop_duplicates(subset=["plant_id", "year", "month"], keep="last")
    df_ghi = df_ghi.drop_duplicates(subset=["plant_id", "year", "month"], keep="last")
    if pre_gen != len(df_gen) or pre_ghi != len(df_ghi):
        print(
            f"  WARNING: Removed duplicate rows: "
            f"EIA {pre_gen - len(df_gen):,}, NSRDB {pre_ghi - len(df_ghi):,}"
        )

    print(f"\nLoaded:")
    print(f"  EIA generation:  {len(df_gen):,} records, {df_gen['plant_id'].nunique()} plants")
    print(f"  NSRDB GHI:       {len(df_ghi):,} records, {df_ghi['plant_id'].nunique()} plants")
    print(f"  Plant metadata:  {len(df_plants)} plants")

    # -- Merge Datasets --
    print("\nMerging EIA generation with NSRDB GHI...")

    # Ensure consistent types
    df_gen["year"] = df_gen["year"].astype(int)
    df_gen["month"] = df_gen["month"].astype(int)
    df_ghi["year"] = df_ghi["year"].astype(int)
    df_ghi["month"] = df_ghi["month"].astype(int)

    df_merged = pd.merge(
        df_gen,
        df_ghi,
        on=["plant_id", "year", "month"],
        how="inner",
    )

    print(f"  Merged records: {len(df_merged):,}")
    print(f"  Plants with both datasets: {df_merged['plant_id'].nunique()}")

    # Add plant metadata (capacity, name, lat/lon)
    plant_cols = ["plant_id"]
    for col in ["plant_name", "nameplate_capacity_mw", "latitude", "longitude", "state"]:
        if col in df_plants.columns:
            plant_cols.append(col)

    df_merged = pd.merge(
        df_merged,
        df_plants[plant_cols].drop_duplicates(subset=["plant_id"]),
        on="plant_id",
        how="left",
    )

    # Compute normalized yield (MWh / MWac)
    if "nameplate_capacity_mw" in df_merged.columns:
        df_merged["capacity_mw"] = pd.to_numeric(df_merged["nameplate_capacity_mw"], errors="coerce")
        df_merged["normalized_yield"] = df_merged["generation_mwh"] / df_merged["capacity_mw"]
    else:
        df_merged["capacity_mw"] = np.nan
        df_merged["normalized_yield"] = df_merged["generation_mwh"]

    # -- Compute Per-Plant Metrics --
    print("\nComputing correlation metrics per plant...")

    results = []

    for plant_id, group in df_merged.groupby("plant_id"):
        metrics = compute_plant_metrics(group)
        metrics["plant_id"] = plant_id

        # Add metadata
        for col in ["plant_name", "latitude", "longitude", "state"]:
            if col in group.columns:
                metrics[col] = group[col].iloc[0]

        results.append(metrics)

    df_results = pd.DataFrame(results)

    # -- Composite Score & Tier --
    df_results["composite_score"] = df_results.apply(compute_composite_score, axis=1)
    df_results["quality_tier"] = df_results["pearson_r"].apply(assign_quality_tier)

    # Sort by composite score
    df_results = df_results.sort_values("composite_score", ascending=False).reset_index(drop=True)
    df_results["rank"] = range(1, len(df_results) + 1)

    # -- Save Full Results --
    # Reorder columns for readability
    priority_cols = [
        "rank", "plant_id", "plant_name", "state", "latitude", "longitude",
        "capacity_mw", "quality_tier", "composite_score",
        "pearson_r", "pearson_p", "r_squared",
        "spearman_rho", "annual_pearson_r",
        "mean_performance_ratio", "cv_performance_ratio",
        "n_monthly_points", "n_years", "year_min", "year_max",
        "total_generation_gwh",
    ]

    existing_cols = [c for c in priority_cols if c in df_results.columns]
    remaining_cols = [c for c in df_results.columns if c not in existing_cols]
    df_results = df_results[existing_cols + remaining_cols]

    df_results.to_csv(config.CORRELATION_RESULTS_CSV, index=False)
    print(f"\nSaved all {len(df_results)} plant results -> {config.CORRELATION_RESULTS_CSV}")

    # -- Select Top Candidates --
    df_top = df_results.head(config.TARGET_NUM_CANDIDATES).copy()
    df_top.to_csv(config.TOP_CANDIDATES_CSV, index=False)
    print(f"Saved top {len(df_top)} candidates -> {config.TOP_CANDIDATES_CSV}")

    # -- Summary Statistics --
    print(f"\n{'=' * 60}")
    print(f"CORRELATION ANALYSIS SUMMARY")
    print(f"{'=' * 60}")

    print(f"\nTotal plants analyzed: {len(df_results)}")
    print(f"\nQuality Tier Distribution:")
    tier_counts = df_results["quality_tier"].value_counts()
    for tier, count in tier_counts.items():
        print(f"  {tier}: {count}")

    print(f"\nPearson R Distribution:")
    r_vals = df_results["pearson_r"].dropna()
    print(f"  Mean:   {r_vals.mean():.3f}")
    print(f"  Median: {r_vals.median():.3f}")
    print(f"  Std:    {r_vals.std():.3f}")
    print(f"  Min:    {r_vals.min():.3f}")
    print(f"  Max:    {r_vals.max():.3f}")

    print(f"\nTop {min(10, len(df_top))} Candidates:")
    print("-" * 80)
    for _, row in df_top.head(10).iterrows():
        name = row.get("plant_name", row["plant_id"])
        st = row.get("state", "??")
        r = row.get("pearson_r", np.nan)
        rho = row.get("spearman_rho", np.nan)
        cap = row.get("capacity_mw", np.nan)
        ny = row.get("n_years", 0)

        print(f"  #{int(row['rank']):3d}  {name[:40]:<40s}  {st:2s}  "
              f"R={r:.3f}  ρ={rho:.3f}  {cap:6.1f}MW  {ny}yr")

    print(f"\nStep 4 complete.")


if __name__ == "__main__":
    main()
