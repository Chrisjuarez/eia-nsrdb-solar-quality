#!/usr/bin/env python3
"""
Master Pipeline Runner
======================
Runs all 5 steps of the EIA-NSRDB Solar Plant Quality Pipeline in sequence.

Usage:
    python run_pipeline.py           # Run all steps
    python run_pipeline.py --from 3  # Resume from Step 3
    python run_pipeline.py --only 4  # Run only Step 4
"""

import sys
import time
import argparse
import importlib
from datetime import datetime


STEPS = [
    ("01_eia_plant_discovery", "Step 1: EIA Plant Discovery & Filtering"),
    ("02_eia_generation_download", "Step 2: EIA Monthly Generation Download"),
    ("03_nsrdb_weather_retrieval", "Step 3: NSRDB Weather Retrieval"),
    ("04_correlation_analysis", "Step 4: Correlation Analysis & Ranking"),
    ("05_report_and_visuals", "Step 5: Report & Visualizations"),
]


def run_step(module_name: str, description: str):
    """Import and run a single pipeline step."""
    print(f"\n{'+' + '=' * 68 + '+'}")
    print(f"|  {description:<66s}|")
    print(f"{'+' + '=' * 68 + '+'}")

    start = time.perf_counter()

    try:
        module = importlib.import_module(module_name)
        module.main()
    except SystemExit as e:
        if e.code != 0:
            print(f"\n{description} failed with exit code {e.code}")
            raise
    except Exception as e:
        print(f"\n{description} failed: {e}")
        raise

    elapsed = time.perf_counter() - start
    print(f"\n  {description} completed in {elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="Run EIA-NSRDB Solar Quality Pipeline")
    parser.add_argument("--from", dest="from_step", type=int, default=1,
                       help="Start from step N (default: 1)")
    parser.add_argument("--only", type=int, default=None,
                       help="Run only step N")
    args = parser.parse_args()

    print("=" * 70)
    print("  EIA-NSRDB SOLAR PLANT QUALITY PIPELINE")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    pipeline_start = time.perf_counter()

    if args.only:
        # Run single step
        if 1 <= args.only <= len(STEPS):
            module_name, description = STEPS[args.only - 1]
            run_step(module_name, description)
        else:
            print(f"Invalid step number: {args.only}. Valid: 1-{len(STEPS)}")
            sys.exit(1)
    else:
        # Run from specified step onwards
        for i, (module_name, description) in enumerate(STEPS, 1):
            if i >= args.from_step:
                run_step(module_name, description)

    total_elapsed = time.perf_counter() - pipeline_start

    print(f"\n{'=' * 70}")
    print(f"  PIPELINE COMPLETE")
    print(f"  Total time: {total_elapsed / 60:.1f} minutes")
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
