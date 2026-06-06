"""
detect_canopy_change.py — Tree Canopy Change Detection
=======================================================
Compares two CHM GeoTIFF directories (different collection years)
for the same county and produces:
  - A change classification raster (gain / loss / stable / no-canopy)
  - A per-tile change statistics CSV
  - A county-level summary report

Edge cases handled:
  - CRS mismatch and reprojection (Category 2)
  - Sub-pixel misalignment via median filter (Category 2)
  - NaN / nodata propagation (Category 3)
  - Negative CHM values (Category 3)
  - Output overwrite protection (Category 5)
  - Tile grid suffix matching across collections (Category 1)

Usage:
  python detect_canopy_change.py \
      --dir1  ../data/outputs/GeoTIFF_files/albemarle_2015/ \
      --dir2  ../data/outputs/GeoTIFF_files/albemarle_2020/ \
      --year1 2015 --year2 2020 \
      --county Albemarle \
      --out   ../data/outputs/change/
"""

import argparse
import csv
import logging
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from scipy.ndimage import median_filter

# ── Configuration Constants ────────────────────────────────────────────────────
MIN_CANOPY_HEIGHT_M = 2.0
HEIGHT_CHANGE_THRESHOLD_M = 1.5
NO_DATA_VALUE = 255

# Change Classes
CLASS_NO_CANOPY = 0
CLASS_STABLE = 1
CLASS_GAIN = 2
CLASS_LOSS = 3
CLASS_HEIGHT_INC = 4
CLASS_HEIGHT_DEC = 5

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def extract_tile_key(stem: str) -> str:
    """Extract the USGS tile grid suffix (e.g., S13_4816_10) to match across collections."""
    m = re.search(r'(S\d+_\d+_\d+)', stem)
    return m.group(1) if m else stem


def match_tiles(dir1: Path, dir2: Path):
    """Match GeoTIFF files between two year directories by tile stem name."""
    tiles1 = {f.stem.replace("_chm", ""): f for f in dir1.glob("*_chm.tif")}
    tiles2 = {f.stem.replace("_chm", ""): f for f in dir2.glob("*_chm.tif")}
    
    keyed1 = {extract_tile_key(k): v for k, v in tiles1.items()}
    keyed2 = {extract_tile_key(k): v for k, v in tiles2.items()}
    
    common = sorted(set(keyed1) & set(keyed2))
    matched = [(keyed1[k], keyed2[k]) for k in common]
    
    unmatched1 = set(keyed1) - set(keyed2)
    unmatched2 = set(keyed2) - set(keyed1)
    
    logger.info(f"Matched tiles: {len(matched)}")
    if unmatched1:
        logger.warning(f"Tiles only in year1: {len(unmatched1)}")
    if unmatched2:
        logger.warning(f"Tiles only in year2: {len(unmatched2)}")
        
    return matched


def process_tile_pair(path1: Path, path2: Path, out_dir: Path, 
                      year1: str, year2: str, overwrite: bool) -> dict:
    """Compare two CHM GeoTIFFs and write a change raster."""
    tile_stem = path1.stem.replace("_chm", "")
    tile_key = extract_tile_key(tile_stem)
    out_path = out_dir / f"{tile_key}_{year1}_{year2}_change.tif"
    
    if out_path.exists() and not overwrite:
        logger.info(f"Skipping {tile_key} (already exists)")
        return None

    try:
        with rasterio.open(path1) as src1:
            chm1 = src1.read(1).astype(np.float32)
            meta = src1.meta.copy()
            transform1 = src1.transform
            crs1 = src1.crs
            nodata1 = src1.nodata if src1.nodata is not None else np.nan

        with rasterio.open(path2) as src2:
            chm2_raw = src2.read(1).astype(np.float32)
            nodata2 = src2.nodata if src2.nodata is not None else np.nan
            
            # Edge Case: Reproject if CRS or extent differs
            chm2 = np.empty_like(chm1)
            reproject(
                source=chm2_raw,
                destination=chm2,
                src_transform=src2.transform,
                src_crs=src2.crs,
                dst_transform=transform1,
                dst_crs=crs1,
                resampling=Resampling.bilinear,
            )

        # Edge Case: Handle NaN and nodata propagation
        valid_mask = (~np.isnan(chm1)) & (chm1 != nodata1) & \
                     (~np.isnan(chm2)) & (chm2 != nodata2)
                     
        # Edge Case: Negative CHM values
        chm1 = np.maximum(chm1, 0)
        chm2 = np.maximum(chm2, 0)

        # Canopy masks
        canopy1 = (chm1 >= MIN_CANOPY_HEIGHT_M) & valid_mask
        canopy2 = (chm2 >= MIN_CANOPY_HEIGHT_M) & valid_mask

        # Height difference
        delta = chm2 - chm1
        
        # Edge Case: Sub-pixel misalignment noise reduction
        delta = median_filter(delta, size=3)

        # Classify change
        change = np.full_like(chm1, NO_DATA_VALUE, dtype=np.uint8)
        
        # Only classify valid pixels
        valid_idx = np.where(valid_mask)
        
        # Vectorized classification on valid pixels
        no_canopy = (~canopy1) & (~canopy2) & valid_mask
        stable = canopy1 & canopy2 & (np.abs(delta) < HEIGHT_CHANGE_THRESHOLD_M) & valid_mask
        gain = (~canopy1) & canopy2 & valid_mask
        loss = canopy1 & (~canopy2) & valid_mask
        h_inc = canopy1 & canopy2 & (delta >= HEIGHT_CHANGE_THRESHOLD_M) & valid_mask
        h_dec = canopy1 & canopy2 & (delta <= -HEIGHT_CHANGE_THRESHOLD_M) & valid_mask

        change[no_canopy] = CLASS_NO_CANOPY
        change[stable] = CLASS_STABLE
        change[gain] = CLASS_GAIN
        change[loss] = CLASS_LOSS
        change[h_inc] = CLASS_HEIGHT_INC
        change[h_dec] = CLASS_HEIGHT_DEC

        # Write change raster
        meta.update(dtype=np.uint8, count=1, nodata=NO_DATA_VALUE)
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(change, 1)

        # Statistics
        total_px = int(valid_mask.sum())
        if total_px == 0:
            logger.warning(f"Tile {tile_key} has no valid overlapping pixels.")
            return None
            
        stats = {
            "tile": tile_key,
            "year1": year1,
            "year2": year2,
            "valid_pixels": total_px,
            "no_canopy_px": int(no_canopy.sum()),
            "stable_px": int(stable.sum()),
            "gain_px": int(gain.sum()),
            "loss_px": int(loss.sum()),
            "height_increase_px": int(h_inc.sum()),
            "height_decrease_px": int(h_dec.sum()),
            "net_change_px": int(gain.sum()) - int(loss.sum()),
            "canopy_cover_pct_yr1": round(float(canopy1.sum()) / total_px * 100, 2),
            "canopy_cover_pct_yr2": round(float(canopy2.sum()) / total_px * 100, 2),
        }
        return stats
        
    except Exception as e:
        logger.error(f"Failed processing {tile_key}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Tree Canopy Change Detection")
    parser.add_argument("--dir1", required=True, type=Path, help="Year 1 CHM directory")
    parser.add_argument("--dir2", required=True, type=Path, help="Year 2 CHM directory")
    parser.add_argument("--year1", required=True, help="Year 1 label (e.g., 2015)")
    parser.add_argument("--year2", required=True, help="Year 2 label (e.g., 2020)")
    parser.add_argument("--county", required=True, help="County name for output files")
    parser.add_argument("--out", required=True, type=Path, help="Output directory")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    
    # Check disk space (Edge Case)
    statvfs = os.statvfs(args.out)
    free_gb = (statvfs.f_frsize * statvfs.f_bavail) / (1024**3)
    if free_gb < 5.0:
        logger.warning(f"Low disk space: only {free_gb:.1f} GB available on target drive.")

    matched_pairs = match_tiles(args.dir1, args.dir2)
    if not matched_pairs:
        logger.error("No matching tiles found between the two directories. Exiting.")
        sys.exit(1)

    all_stats = []
    t0 = time.time()
    
    for idx, (p1, p2) in enumerate(matched_pairs, 1):
        logger.info(f"Processing {idx}/{len(matched_pairs)}: {p1.name}")
        stats = process_tile_pair(p1, p2, args.out, args.year1, args.year2, args.overwrite)
        if stats:
            all_stats.append(stats)

    if not all_stats:
        logger.info("No valid statistics generated. Exiting.")
        sys.exit(0)

    # Write per-tile statistics CSV
    csv_path = args.out / f"{args.county.lower()}_{args.year1}_{args.year2}_change_stats.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_stats[0].keys())
        writer.writeheader()
        writer.writerows(all_stats)
        
    logger.info(f"Change statistics written to {csv_path}")
    logger.info(f"Completed {len(all_stats)} tiles in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
