"""
smap_annual_mean_from_s3.py
---------------------------
Compute the mean of all valid daily SMAP SPL3SMP_E soil moisture retrievals
within each calendar year for every SMAP pixel inside a requested bounding box.

Files are read DIRECTLY from S3 (no local download required) using h5py + s3fs.

Source      : s3://central-virginia-tree-canopy-project/SMAP/
Output      : One GeoTIFF per year  +  one CSV summary table
              saved to --out-dir (default: ./smap_annual_means/)

Usage
-----
    # Default bbox (Charlottesville + 6 counties)
    python smap_annual_mean_from_s3.py

    # Custom bbox  (lon_min lat_min lon_max lat_max)
    python smap_annual_mean_from_s3.py \
        --bbox -79.1721 37.3296 -77.6873 38.4755

    # Restrict to specific years
    python smap_annual_mean_from_s3.py --years 2015 2016 2019 2020

    # Use AM overpass only (default), PM only, or both
    python smap_annual_mean_from_s3.py --overpass AM
    python smap_annual_mean_from_s3.py --overpass PM
    python smap_annual_mean_from_s3.py --overpass BOTH

    # Dry-run — list matching S3 files without processing
    python smap_annual_mean_from_s3.py --dry-run

Dependencies (install once on Rivanna)
---------------------------------------
    pip install s3fs h5py numpy pandas rasterio pyproj tqdm
"""

import argparse
import io
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import boto3
import h5py
import numpy as np
import pandas as pd
import s3fs
from tqdm import tqdm

try:
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.crs import CRS
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    print("[WARN] rasterio not installed — GeoTIFF output disabled. "
          "Install with: pip install rasterio")

# ── Defaults ──────────────────────────────────────────────────────────────
DEFAULT_BUCKET    = "central-virginia-tree-canopy-project"
DEFAULT_S3_PREFIX = "SMAP/"
DEFAULT_BBOX      = (-79.1721, 37.3296, -77.6873, 38.4755)   # lon_min,lat_min,lon_max,lat_max
DEFAULT_OUT_DIR   = "./smap_annual_means"
FILL_VALUE        = -9999.0
QUALITY_MASK      = 0b11          # bits 0-1: recommended quality flag mask

# SMAP EASE-Grid 2.0 Global (EPSG:6933) — 9 km resolution
# Grid dimensions: 3856 cols × 1624 rows
EASE2_NCOLS = 3856
EASE2_NROWS = 1624
EASE2_RES_DEG_LAT = 180.0 / EASE2_NROWS   # ~0.1109 degrees
EASE2_RES_DEG_LON = 360.0 / EASE2_NCOLS   # ~0.0934 degrees


# ── Helpers ───────────────────────────────────────────────────────────────
def parse_date_from_key(key: str):
    """Extract YYYYMMDD from an S3 key like SMAP_L3_SM_P_E_20200615_R19240_001.h5"""
    m = re.search(r"_(\d{8})_", os.path.basename(key))
    return m.group(1) if m else None


def list_s3_h5_files(bucket: str, prefix: str, years: list = None) -> dict:
    """
    List all *.h5 files under s3://bucket/prefix and group them by year.
    Returns: {year_str: [s3_key, ...], ...}
    """
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    by_year = defaultdict(list)
    total = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".h5"):
                continue
            date_str = parse_date_from_key(key)
            if not date_str:
                continue
            year = date_str[:4]
            if years and year not in years:
                continue
            by_year[year].append(key)
            total += 1
    return dict(by_year), total


def build_bbox_mask(lats: np.ndarray, lons: np.ndarray,
                    lon_min: float, lat_min: float,
                    lon_max: float, lat_max: float) -> np.ndarray:
    """Return boolean mask of pixels inside the bounding box."""
    return (
        (lats >= lat_min) & (lats <= lat_max) &
        (lons >= lon_min) & (lons <= lon_max)
    )


def read_sm_from_s3(fs: s3fs.S3FileSystem, s3_path: str,
                    overpass: str, bbox_mask: np.ndarray) -> np.ndarray:
    """
    Open one HDF5 file from S3, extract soil moisture for the given overpass,
    apply quality flags, and return a masked array (NaN where invalid/outside bbox).
    """
    groups = []
    if overpass in ("AM", "BOTH"):
        groups.append("Soil_Moisture_Retrieval_Data_AM")
    if overpass in ("PM", "BOTH"):
        groups.append("Soil_Moisture_Retrieval_Data_PM")

    sm_layers = []
    with fs.open(s3_path, "rb") as f:
        raw = f.read()                          # stream entire file into memory

    with h5py.File(io.BytesIO(raw), "r") as hf:
        for grp_name in groups:
            if grp_name not in hf:
                continue
            grp = hf[grp_name]

            # Variable name differs between AM and PM
            sm_key = "soil_moisture" if "AM" in grp_name else "soil_moisture_pm"
            qf_key = "retrieval_qual_flag" if "AM" in grp_name else "retrieval_qual_flag_pm"

            if sm_key not in grp:
                continue

            sm = grp[sm_key][:]
            qf = grp[qf_key][:] if qf_key in grp else np.zeros_like(sm, dtype=np.uint16)

            # Mask fill values, poor quality, and outside bbox
            valid = (
                (sm != FILL_VALUE) &
                ((qf & QUALITY_MASK) == 0) &
                bbox_mask
            )
            sm_masked = np.where(valid, sm.astype(np.float32), np.nan)
            sm_layers.append(sm_masked)

    if not sm_layers:
        return None

    # If BOTH overpasses requested, average AM and PM
    if len(sm_layers) == 2:
        return np.nanmean(np.stack(sm_layers), axis=0)
    return sm_layers[0]


def compute_annual_means(
    bucket: str,
    prefix: str,
    bbox: tuple,
    years: list,
    overpass: str,
    out_dir: str,
    dry_run: bool,
) -> None:

    lon_min, lat_min, lon_max, lat_max = bbox
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 66)
    print("  SMAP SPL3SMP_E — Annual Mean Soil Moisture from S3")
    print("=" * 66)
    print(f"  S3 source  : s3://{bucket}/{prefix}")
    print(f"  Bbox       : lon [{lon_min}, {lon_max}]  lat [{lat_min}, {lat_max}]")
    print(f"  Years      : {', '.join(years) if years else 'all available'}")
    print(f"  Overpass   : {overpass}")
    print(f"  Output dir : {out_dir}")
    print("=" * 66)

    # ── List files ────────────────────────────────────────────────────────
    print("\nScanning S3 for *.h5 files …")
    by_year, total_files = list_s3_h5_files(bucket, prefix, years)
    if not by_year:
        print("[ERROR] No matching files found in S3.")
        sys.exit(1)

    print(f"  Found {total_files:,} files across {len(by_year)} year(s): "
          f"{', '.join(sorted(by_year.keys()))}")

    if dry_run:
        print("\n[DRY-RUN] Files per year:")
        for yr in sorted(by_year.keys()):
            print(f"  {yr} : {len(by_year[yr]):>4} files")
        return

    # ── Open S3 filesystem (anonymous = public bucket; remove anon=True for private) ──
    fs = s3fs.S3FileSystem(anon=False)   # uses ~/.aws/credentials or IAM role

    # ── Read one file to get lat/lon grids and build bbox mask ────────────
    print("\nReading coordinate grids from first file …")
    first_key = sorted(by_year[sorted(by_year.keys())[0]])[0]
    first_path = f"s3://{bucket}/{first_key}"
    with fs.open(first_path, "rb") as f:
        raw = f.read()
    with h5py.File(io.BytesIO(raw), "r") as hf:
        grp_name = "Soil_Moisture_Retrieval_Data_AM"
        lats_full = hf[grp_name]["latitude"][:]
        lons_full = hf[grp_name]["longitude"][:]

    bbox_mask = build_bbox_mask(lats_full, lons_full, lon_min, lat_min, lon_max, lat_max)
    n_pixels  = int(bbox_mask.sum())
    print(f"  SMAP pixels inside bbox : {n_pixels:,}")

    # Get row/col indices for the bbox pixels (for GeoTIFF output)
    rows_idx, cols_idx = np.where(bbox_mask)
    row_min, row_max = rows_idx.min(), rows_idx.max()
    col_min, col_max = cols_idx.min(), cols_idx.max()

    # Subset lat/lon to bbox extent
    lats_sub = lats_full[row_min:row_max+1, col_min:col_max+1]
    lons_sub = lons_full[row_min:row_max+1, col_min:col_max+1]

    # ── Process each year ─────────────────────────────────────────────────
    summary_rows = []

    for year in sorted(by_year.keys()):
        keys = sorted(by_year[year])
        print(f"\n  Processing {year}  ({len(keys)} files) …")

        daily_stack = []   # list of 2-D float32 arrays (full grid)

        for key in tqdm(keys, desc=f"  {year}", unit="file", ncols=70):
            s3_path = f"s3://{bucket}/{key}"
            try:
                sm_day = read_sm_from_s3(fs, s3_path, overpass, bbox_mask)
                if sm_day is not None:
                    # Crop to bbox extent
                    sm_crop = sm_day[row_min:row_max+1, col_min:col_max+1]
                    daily_stack.append(sm_crop)
            except Exception as e:
                tqdm.write(f"    [WARN] {os.path.basename(key)}: {e}")
                continue

        if not daily_stack:
            print(f"  [WARN] No valid data for {year} — skipping")
            continue

        # Annual mean (ignore NaN)
        annual_mean = np.nanmean(np.stack(daily_stack, axis=0), axis=0)
        valid_days  = np.sum(~np.isnan(np.stack(daily_stack, axis=0)), axis=0)

        # ── Save GeoTIFF ──────────────────────────────────────────────────
        if HAS_RASTERIO:
            tif_path = out_path / f"smap_annual_mean_{year}.tif"
            height, width = annual_mean.shape
            transform = from_bounds(
                lons_sub.min(), lats_sub.min(),
                lons_sub.max(), lats_sub.max(),
                width, height
            )
            with rasterio.open(
                tif_path, "w",
                driver    = "GTiff",
                height    = height,
                width     = width,
                count     = 1,
                dtype     = rasterio.float32,
                crs       = CRS.from_epsg(4326),
                transform = transform,
                nodata    = np.nan,
                compress  = "lzw",
            ) as dst:
                dst.write(annual_mean.astype(np.float32), 1)
                dst.update_tags(
                    year         = year,
                    product      = "SPL3SMP_E",
                    overpass     = overpass,
                    n_daily_obs  = len(daily_stack),
                    bbox         = str(bbox),
                )
            print(f"  Saved GeoTIFF : {tif_path}")

        # ── Save per-pixel CSV ────────────────────────────────────────────
        csv_path = out_path / f"smap_annual_mean_{year}.csv"
        rows = []
        for r in range(annual_mean.shape[0]):
            for c in range(annual_mean.shape[1]):
                val = annual_mean[r, c]
                if not np.isnan(val):
                    rows.append({
                        "year":          year,
                        "lat":           round(float(lats_sub[r, c]), 6),
                        "lon":           round(float(lons_sub[r, c]), 6),
                        "sm_mean_m3m3":  round(float(val), 6),
                        "valid_days":    int(valid_days[r, c]),
                    })
        df_year = pd.DataFrame(rows)
        df_year.to_csv(csv_path, index=False)
        print(f"  Saved CSV     : {csv_path}  ({len(df_year):,} pixels)")

        # ── Collect summary stats ─────────────────────────────────────────
        valid_vals = annual_mean[~np.isnan(annual_mean)]
        summary_rows.append({
            "year":            year,
            "n_files":         len(keys),
            "n_valid_days":    len(daily_stack),
            "n_pixels":        len(valid_vals),
            "sm_mean":         round(float(valid_vals.mean()), 6) if len(valid_vals) else np.nan,
            "sm_min":          round(float(valid_vals.min()),  6) if len(valid_vals) else np.nan,
            "sm_max":          round(float(valid_vals.max()),  6) if len(valid_vals) else np.nan,
            "sm_std":          round(float(valid_vals.std()),  6) if len(valid_vals) else np.nan,
        })

    # ── Write master summary CSV ──────────────────────────────────────────
    if summary_rows:
        summary_path = out_path / "smap_annual_summary.csv"
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        print(f"\n  Master summary : {summary_path}")

        print("\n" + "=" * 66)
        print("  ANNUAL MEAN SOIL MOISTURE SUMMARY  (m³/m³)")
        print("=" * 66)
        print(f"  {'Year':<6} {'Files':>6} {'Valid Days':>10} "
              f"{'Pixels':>7} {'Mean':>8} {'Min':>8} {'Max':>8} {'Std':>8}")
        print("  " + "-" * 64)
        for r in summary_rows:
            print(f"  {r['year']:<6} {r['n_files']:>6} {r['n_valid_days']:>10} "
                  f"{r['n_pixels']:>7} {r['sm_mean']:>8.4f} "
                  f"{r['sm_min']:>8.4f} {r['sm_max']:>8.4f} {r['sm_std']:>8.4f}")
        print("=" * 66)

    print(f"\nDone. All outputs saved to: {out_dir}")


# ── CLI ───────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute annual mean SMAP soil moisture from S3 HDF5 files"
    )
    p.add_argument(
        "--bucket", default=DEFAULT_BUCKET,
        help=f"S3 bucket name (default: {DEFAULT_BUCKET})"
    )
    p.add_argument(
        "--s3-prefix", default=DEFAULT_S3_PREFIX,
        help=f"S3 key prefix (default: {DEFAULT_S3_PREFIX})"
    )
    p.add_argument(
        "--bbox", nargs=4, type=float,
        metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"),
        default=list(DEFAULT_BBOX),
        help="Bounding box in WGS84 degrees (default: Charlottesville + 6 counties)"
    )
    p.add_argument(
        "--years", nargs="+", default=None,
        help="Restrict to specific years e.g. --years 2015 2019 2020"
    )
    p.add_argument(
        "--overpass", choices=["AM", "PM", "BOTH"], default="AM",
        help="SMAP overpass to use (default: AM)"
    )
    p.add_argument(
        "--out-dir", default=DEFAULT_OUT_DIR,
        help=f"Output directory for GeoTIFFs and CSVs (default: {DEFAULT_OUT_DIR})"
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="List matching S3 files without processing"
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    compute_annual_means(
        bucket   = args.bucket,
        prefix   = args.s3_prefix.rstrip("/") + "/",
        bbox     = tuple(args.bbox),
        years    = [str(y) for y in args.years] if args.years else None,
        overpass = args.overpass,
        out_dir  = args.out_dir,
        dry_run  = args.dry_run,
    )
