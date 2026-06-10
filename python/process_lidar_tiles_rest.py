"""
process_lidar_tiles_rest.py — Parallel VGIN LiDAR LAZ Processing Pipeline
======================================================================
Incorporates:
  - Live VGIN REST API query for tile discovery (replaces CSV inventory)
  - concurrent.futures.ProcessPoolExecutor for tile-level parallelism
  - Thread-safe / process-safe skip logging via a multiprocessing Lock
  - Per-stage timing with time.perf_counter()
  - Four-level skip logic (empty / unclassified / no vegetation / below 2 m)
  - 2-meter minimum height filter at point cloud and CHM levels
  - Local maxima detection for tree canopy centroids (scipy.ndimage)
  - Progress bar via tqdm (optional — gracefully degrades if not installed)
  - Combined centroid output across all tiles
  - Run summary report
  - AWS S3 upload integration

Usage:
  python process_lidar_tiles_rest.py \
      --county Albemarle \
      --year 2015 \
      --workers 6 \
      --s3-upload \
      --dry-run          # optional: print tile list without processing

Dependencies:
  pip install laspy[laszip] numpy scipy rasterio requests tqdm beautifulsoup4 pyproj
"""

import argparse
import csv
import io
import logging
import multiprocessing
import os
import re
import sys
import time
import urllib.parse
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import List, Tuple

import laspy
import numpy as np
import requests
import rasterio
from rasterio.transform import from_origin
from scipy.spatial import cKDTree
from scipy.ndimage import maximum_filter
from pyproj import Transformer

import s3_utils

# Try importing tqdm; fall back to a no-op wrapper if not installed
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):  # type: ignore
        return iterable

# Configuration Constants
MIN_CANOPY_HEIGHT_M  = 2.0    # Minimum height above ground to be considered tree canopy
MAX_CANOPY_HEIGHT_M  = 60.0   # Maximum realistic tree height for Central Virginia
CROWN_RADIUS_M       = 3.0    # Local maxima search radius (meters) for crown peak detection
RASTER_RESOLUTION    = 1.0    # Output raster cell size in meters
DEFAULT_WORKERS      = max(1, os.cpu_count() - 2)  # Leave 2 cores for OS

# VGIN REST API Endpoint
VGIN_REST_URL = "https://vginmaps.vdem.virginia.gov/arcgis/rest/services/Download/Virginia_LiDAR_Downloads/MapServer/1/query"

# Output Paths (updated dynamically in main based on county/year)
GEOTIFF_OUTPUT_DIR   = Path("../data/outputs/GeoTIFF_files/default/")
CENTROID_OUTPUT_DIR  = Path("../data/outputs/centroids/default/")
SKIP_LOG_PATH        = Path("../data/outputs/skipped_tiles.csv")
COMBINED_CENTROID    = Path("../data/outputs/all_tiles_centroids.csv")
RUN_SUMMARY_PATH     = Path("../data/outputs/run_summary.txt")

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [PID %(process)d] %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: — VGIN REST API Integration
# ══════════════════════════════════════════════════════════════════════════════

def get_county_bbox(county_name: str) -> Tuple[float, float, float, float]:
    """
    Get the bounding box for a Virginia county in EPSG:3857 (Web Mercator).
    Returns (xmin, ymin, xmax, ymax).
    Uses the Nominatim geocoding API.
    """
    logger.info(f"Geocoding bounding box for {county_name} County, VA...")
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": f"{county_name} County, Virginia, USA",
        "format": "json",
        "limit": 1
    }
    headers = {"User-Agent": "CentralVATreeCanopyProject/1.0"}
    
    response = requests.get(url, params=params, headers=headers)
    response.raise_for_status()
    data = response.json()
    
    if not data:
        raise ValueError(f"Could not find bounding box for {county_name} County")
        
    # Nominatim returns bounds as [lat_min, lat_max, lon_min, lon_max] in WGS84
    bbox = data[0]["boundingbox"]
    lat_min, lat_max = float(bbox[0]), float(bbox[1])
    lon_min, lon_max = float(bbox[2]), float(bbox[3])
    
    # Convert WGS84 to EPSG:3857 for the ArcGIS REST query
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    xmin, ymin = transformer.transform(lon_min, lat_min)
    xmax, ymax = transformer.transform(lon_max, lat_max)
    
    logger.info(f"Bounding box (EPSG:3857): {xmin:.1f}, {ymin:.1f}, {xmax:.1f}, {ymax:.1f}")
    return (xmin, ymin, xmax, ymax)


def _extract_url_from_html(html_str: str) -> str:
    """Extract the first href URL from an HTML string."""
    if not html_str:
        return ""
    match = re.search(r'href=["\'](https?://[^"\'>]+)["\']', html_str)
    if match:
        return match.group(1)
    if html_str.strip().startswith("http"):
        return html_str.strip()
    return ""


def query_vgin_tiles(county: str, year: str = None, current_only: bool = True) -> List[Tuple[str, str]]:
    """
    Query the VGIN REST API for LiDAR tiles intersecting the given county.
    Returns a list of (url, geotiff_filename) tuples.
    """
    try:
        xmin, ymin, xmax, ymax = get_county_bbox(county)
    except Exception as e:
        logger.error(f"Failed to geocode county: {e}")
        sys.exit(1)
        
    # Build the SQL WHERE clause
    where_clauses = ["1=1"]
    if year and year.lower() != "all":
        where_clauses.append(f"ProjectYear = '{year}'")
    if current_only:
        where_clauses.append("VComment = 'Current'")
        
    where_str = " AND ".join(where_clauses)
    logger.info(f"Querying VGIN REST API with WHERE: {where_str}")
    
    params = {
        "f": "json",
        "geometry": f"{xmin},{ymin},{xmax},{ymax}",
        "geometryType": "esriGeometryEnvelope",
        "inSR": "3857",
        "spatialRel": "esriSpatialRelIntersects",
        "where": where_str,
        "outFields": "TileName,PointCloudDownload,ProjectYear,VComment",
        "returnGeometry": "false"
    }
    
    response = requests.get(VGIN_REST_URL, params=params)
    response.raise_for_status()
    data = response.json()
    
    if "error" in data:
        logger.error(f"ArcGIS REST Error: {data['error']}")
        sys.exit(1)
        
    features = data.get("features", [])
    logger.info(f"VGIN API returned {len(features)} intersecting tiles")
    
    tiles: List[Tuple[str, str]] = []
    skipped_no_url = 0
    
    for feat in features:
        attrs = feat.get("attributes", {})
        tile_name = attrs.get("TileName", "")
        html_link = attrs.get("PointCloudDownload", "")
        
        url = _extract_url_from_html(html_link)
        if not url:
            skipped_no_url += 1
            continue
            
        # Standardize output filename
        if tile_name:
            geotiff_filename = f"{tile_name}_chm.tif"
        else:
            geotiff_filename = Path(url).stem + "_chm.tif"
            
        tiles.append((url, geotiff_filename))
        
    if skipped_no_url:
        logger.info(f"Skipped {skipped_no_url} tiles with no download URL in metadata")
        
    return tiles


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: — Thread-Safe Skip Logging
# ══════════════════════════════════════════════════════════════════════════════

_skip_lock: multiprocessing.Lock = None

def _init_worker(lock: multiprocessing.Lock) -> None:
    """Initializer for each worker process."""
    global _skip_lock
    _skip_lock = lock


def _log_skipped(url: str, filename: str, reason: str,
                 total_pts: int, ground_pts: int, veg_pts: int) -> None:
    """Append a skipped-tile record to the audit CSV (process-safe)."""
    SKIP_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if _skip_lock:
        _skip_lock.acquire()
    try:
        write_header = not SKIP_LOG_PATH.exists()
        with open(SKIP_LOG_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["url", "filename", "reason",
                                  "total_points", "ground_points", "veg_points"])
            writer.writerow([url, filename, reason, total_pts, ground_pts, veg_pts])
    finally:
        if _skip_lock:
            _skip_lock.release()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: — Core Processing Function
# ══════════════════════════════════════════════════════════════════════════════

def process_file(url: str, geotiff_filename: str,
                 retries: int = 3, timeout: int = 120,
                 s3_bucket: str = None, county: str = "unknown", year: str = "unknown") -> dict:
    """Download, decompress, and process a single VGIN LAZ tile."""
    pid = os.getpid()
    result = {
        "status":      "failed",
        "filename":    geotiff_filename,
        "url":         url,
        "n_trees":     0,
        "skip_reason": "",
        "elapsed_s":   0.0,
    }

    for attempt in range(1, retries + 1):
        try:
            t_attempt_start = time.perf_counter()

            # Stage 1: HTTP Download
            t0 = time.perf_counter()
            logger.info(f"[{pid}] GET  {url}")
            response = requests.get(url, stream=True, timeout=timeout)
            response.raise_for_status()

            buffer = io.BytesIO()
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                buffer.write(chunk)
            buffer.seek(0)
            file_size_mb = buffer.tell() / 1e6
            t_download = time.perf_counter() - t0
            logger.info(f"[{pid}] Download: {t_download:.1f}s ({file_size_mb:.1f} MB)")

            # Stage 2: LAZ Decompression
            t0 = time.perf_counter()
            with laspy.open(buffer, laz_backend=laspy.LazBackend.Laszip) as fh:
                las = fh.read()
            t_decompress = time.perf_counter() - t0
            logger.info(f"[{pid}] Decompress: {t_decompress:.1f}s ({len(las.points):,} pts)")

            # Stage 3: Point Extraction
            t0 = time.perf_counter()
            x: np.ndarray = las.x.scaled_array()
            y: np.ndarray = las.y.scaled_array()
            z: np.ndarray = las.z.scaled_array()
            c: np.ndarray = np.array(las.classification)
            
            ground_mask = (c == 2)
            veg_mask    = (c == 3) | (c == 4) | (c == 5)
            
            total_points  = len(las.points)
            ground_points = int(np.sum(ground_mask))
            veg_points    = int(np.sum(veg_mask))

            # Stage 4: Skip Guards
            if ground_points < 3:
                _log_skipped(url, geotiff_filename, "Insufficient ground points", total_points, ground_points, veg_points)
                result["status"] = "skipped"
                result["skip_reason"] = "No ground"
                result["elapsed_s"] = time.perf_counter() - t_attempt_start
                return result

            if veg_points == 0:
                _log_skipped(url, geotiff_filename, "No vegetation points", total_points, ground_points, veg_points)
                result["status"] = "skipped"
                result["skip_reason"] = "No veg"
                result["elapsed_s"] = time.perf_counter() - t_attempt_start
                return result

            x_ground, y_ground, z_ground = x[ground_mask], y[ground_mask], z[ground_mask]
            x_veg, y_veg, z_veg          = x[veg_mask], y[veg_mask], z[veg_mask]

            # Stage 5: Grid Setup
            x_min, x_max = np.min(x), np.max(x)
            y_min, y_max = np.min(y), np.max(y)
            
            ncols = int(np.ceil((x_max - x_min) / RASTER_RESOLUTION))
            nrows = int(np.ceil((y_max - y_min) / RASTER_RESOLUTION))
            
            grid_x = np.linspace(x_min, x_max, ncols)
            grid_y = np.linspace(y_max, y_min, nrows)
            grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)
            grid_coords = np.column_stack((grid_xx.ravel(), grid_yy.ravel()))

            # Stage 6: DTM Interpolation (cKDTree)
            t0 = time.perf_counter()
            MAX_GROUND_PTS = 500_000
            if ground_points > MAX_GROUND_PTS:
                rng = np.random.default_rng(seed=42)
                idx_s = rng.choice(ground_points, size=MAX_GROUND_PTS, replace=False)
                x_g, y_g, z_g = x_ground[idx_s], y_ground[idx_s], z_ground[idx_s]
            else:
                x_g, y_g, z_g = x_ground, y_ground, z_ground

            kd_tree = cKDTree(np.column_stack((x_g, y_g)))
            distances, nn_idx = kd_tree.query(grid_coords, k=1, workers=1)
            
            MAX_INTERP_DIST = RASTER_RESOLUTION * 10
            valid_dtm = distances <= MAX_INTERP_DIST
            dtm_flat = np.full(grid_coords.shape[0], np.nan, dtype=np.float32)
            dtm_flat[valid_dtm] = z_g[nn_idx[valid_dtm]]
            dtm = dtm_flat.reshape(nrows, ncols)
            
            ground_reference = np.nanmean(dtm) if not np.isnan(dtm).all() else 0.0
            dtm = np.where(np.isnan(dtm), ground_reference, dtm)
            t_dtm = time.perf_counter() - t0
            logger.info(f"[{pid}] DTM: {t_dtm:.1f}s")

            # Stage 7: DSM Rasterization
            dsm = np.full((nrows, ncols), -np.inf, dtype=np.float32)
            col_idx = ((x_veg - x_min) / RASTER_RESOLUTION).astype(int)
            row_idx = ((y_max - y_veg) / RASTER_RESOLUTION).astype(int)
            
            valid = (col_idx >= 0) & (col_idx < ncols) & (row_idx >= 0) & (row_idx < nrows)
            np.maximum.at(dsm, (row_idx[valid], col_idx[valid]), z_veg[valid].astype(np.float32))
            dsm[dsm == -np.inf] = np.nan

            # Stage 8: CHM Computation
            chm = dsm - dtm
            chm = np.where(chm < MIN_CANOPY_HEIGHT_M, np.nan, chm)
            chm = np.where(chm > MAX_CANOPY_HEIGHT_M, np.nan, chm)
            chm = np.nan_to_num(chm, nan=0.0).astype(np.float32)

            # Stage 9: GeoTIFF Write
            GEOTIFF_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            out_tif = GEOTIFF_OUTPUT_DIR / geotiff_filename
            transform = from_origin(x_min, y_max, RASTER_RESOLUTION, RASTER_RESOLUTION)
            
            with rasterio.open(
                out_tif, 'w', driver='GTiff', height=nrows, width=ncols,
                count=1, dtype=chm.dtype, crs="EPSG:2284", transform=transform, nodata=0.0
            ) as dst:
                dst.write(chm, 1)

            # Stage 10: Local Maxima (Tree Centroids)
            search_radius_px = int(np.ceil(CROWN_RADIUS_M / RASTER_RESOLUTION))
            y_idx, x_idx = np.ogrid[-search_radius_px:search_radius_px+1, -search_radius_px:search_radius_px+1]
            footprint = (x_idx**2 + y_idx**2) <= search_radius_px**2
            
            chm_max = maximum_filter(chm, footprint=footprint)
            local_maxima = (chm == chm_max) & (chm >= MIN_CANOPY_HEIGHT_M)
            
            rows, cols = np.where(local_maxima)
            n_trees = len(rows)
            result["n_trees"] = n_trees

            if n_trees > 0:
                eastings = x_min + (cols * RASTER_RESOLUTION) + (RASTER_RESOLUTION / 2.0)
                northings = y_max - (rows * RASTER_RESOLUTION) - (RASTER_RESOLUTION / 2.0)
                heights = chm[rows, cols]
                
                CENTROID_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                csv_filename = out_tif.stem + "_centroids.csv"
                out_csv = CENTROID_OUTPUT_DIR / csv_filename
                
                with open(out_csv, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["easting_ft", "northing_ft", "height_m"])
                    for e, n, h in zip(eastings, northings, heights):
                        writer.writerow([f"{e:.2f}", f"{n:.2f}", f"{h:.2f}"])

            # Stage 11: S3 Upload (Per-Tile)
            if s3_bucket:
                s3_client = s3_utils.get_s3_client()
                if s3_client:
                    chm_key = s3_utils.build_s3_key("chm", county, year, out_tif.name)
                    s3_utils.upload_file(s3_client, out_tif, s3_bucket, chm_key)
                    if n_trees > 0:
                        csv_key = s3_utils.build_s3_key("centroids", county, year, out_csv.name)
                        s3_utils.upload_file(s3_client, out_csv, s3_bucket, csv_key)

            result["status"] = "success"
            result["elapsed_s"] = time.perf_counter() - t_attempt_start
            logger.info(f"[{pid}] SUCCESS: {n_trees:,} trees in {result['elapsed_s']:.1f}s")
            return result

        except requests.exceptions.RequestException as e:
            logger.warning(f"[{pid}] Attempt {attempt}/{retries} network error: {e}")
            if attempt < retries:
                time.sleep(5 * attempt)

        except Exception as e:
            logger.warning(f"[{pid}] Attempt {attempt}/{retries} processing error: {e}", exc_info=True)
            if attempt < retries:
                time.sleep(5 * attempt)

    logger.error(f"[{pid}] FAIL giving up: {url}")
    result["elapsed_s"] = time.perf_counter() - (time.perf_counter())
    return result


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: — Parallel Orchestrator
# ══════════════════════════════════════════════════════════════════════════════

def run_parallel(tile_list: List[Tuple[str, str]],
                 max_workers: int = DEFAULT_WORKERS,
                 dry_run: bool = False,
                 s3_bucket: str = None,
                 county: str = "unknown",
                 year: str = "unknown") -> None:
    """Dispatch all tiles to a ProcessPoolExecutor and collect results."""
    if dry_run:
        print(f"\n[DRY RUN] {len(tile_list)} tiles would be processed with {max_workers} workers:\n")
        for i, (url, fname) in enumerate(tile_list[:20], 1):
            print(f"  {i:4d}. {fname}")
            print(f"        {url}")
        if len(tile_list) > 20:
            print(f"  ... and {len(tile_list) - 20} more tiles")
        return

    t_run_start = time.perf_counter()
    logger.info(f"Starting parallel processing: {len(tile_list)} tiles | {max_workers} workers")

    manager    = multiprocessing.Manager()
    skip_lock  = manager.Lock()
    counters   = {"success": 0, "skipped": 0, "failed": 0, "total_trees": 0}
    all_results: list = []

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_worker,
        initargs=(skip_lock,),
    ) as executor:

        future_to_tile = {
            executor.submit(process_file, url, fname, 3, 120, s3_bucket, county, year): (url, fname)
            for url, fname in tile_list
        }

        for future in tqdm(as_completed(future_to_tile), total=len(tile_list), desc="Processing tiles"):
            try:
                result = future.result()
                all_results.append(result)
                counters[result["status"]] += 1
                counters["total_trees"]    += result.get("n_trees", 0)
            except BrokenProcessPool:
                print(f"[FATAL] Worker pool broken — likely OOM. Reduce --workers.")
                raise
            except Exception as e:
                print(f"[FAIL] Unexpected error: {e}")

    # Combine all per-tile centroid CSVs into one master file
    logger.info("Combining per-tile centroid CSVs into master file...")
    COMBINED_CENTROID.parent.mkdir(parents=True, exist_ok=True)
    with open(COMBINED_CENTROID, "w", newline="") as master:
        writer = csv.writer(master)
        writer.writerow(["tile_id", "easting_ft", "northing_ft", "height_m"])
        for centroid_csv in sorted(CENTROID_OUTPUT_DIR.glob("*_centroids.csv")):
            tile_id = centroid_csv.stem.replace("_centroids", "")
            with open(centroid_csv, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    writer.writerow([tile_id, row["easting_ft"], row["northing_ft"], row["height_m"]])
    logger.info(f"Master centroid file: {COMBINED_CENTROID}")

    # Run Summary
    t_total_run = time.perf_counter() - t_run_start
    elapsed_tiles = [r["elapsed_s"] for r in all_results if r["status"] == "success"]
    avg_tile_time = (sum(elapsed_tiles) / len(elapsed_tiles)) if elapsed_tiles else 0.0

    summary_lines = [
        "=" * 60,
        "  LIDAR PROCESSING RUN SUMMARY",
        "=" * 60,
        f"  Total tiles submitted : {len(tile_list):,}",
        f"  Successful            : {counters['success']:,}",
        f"  Skipped               : {counters['skipped']:,}",
        f"  Failed                : {counters['failed']:,}",
        f"  Total tree crowns     : {counters['total_trees']:,}",
        f"  Workers used          : {max_workers}",
        f"  Avg time per tile     : {avg_tile_time:.1f}s",
        f"  Total wall-clock time : {t_total_run:.1f}s ({t_total_run / 60:.1f} min)",
        f"  Master centroid file  : {COMBINED_CENTROID}",
        f"  Skip log              : {SKIP_LOG_PATH}",
        "=" * 60,
    ]

    for line in summary_lines:
        logger.info(line)

    RUN_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RUN_SUMMARY_PATH, "w") as f:
        f.write("\n".join(summary_lines) + "\n")

    # ── Final S3 Uploads (Master CSV and Logs) ────────────────────────────────
    if s3_bucket:
        s3_client = s3_utils.get_s3_client()
        if s3_client:
            logger.info("Uploading master outputs to S3...")
            master_key = f"centroids/{county.lower()}_{year}_centroids.csv"
            s3_utils.upload_file(s3_client, COMBINED_CENTROID, s3_bucket, master_key)
            
            summary_key = s3_utils.build_s3_key("logs", county, year, RUN_SUMMARY_PATH.name)
            skip_key = s3_utils.build_s3_key("logs", county, year, SKIP_LOG_PATH.name)
            s3_utils.upload_file(s3_client, RUN_SUMMARY_PATH, s3_bucket, summary_key)
            if SKIP_LOG_PATH.exists():
                s3_utils.upload_file(s3_client, SKIP_LOG_PATH, s3_bucket, skip_key)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5: — CLI Entry Point
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parallel VGIN LiDAR LAZ processing pipeline via REST API")
    parser.add_argument("--county", required=True, help="County name to process (e.g. 'Albemarle')")
    parser.add_argument("--year", default="all", help="Collection year to filter (e.g. '2015')")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help=f"Worker processes (default: {DEFAULT_WORKERS})")
    parser.add_argument("--current-only", action="store_true", default=True, help="Skip superceded tiles")
    parser.add_argument("--s3-upload", action="store_true", help="Upload outputs to AWS S3")
    parser.add_argument("--dry-run", action="store_true", help="Print the tile list without processing")
    return parser.parse_args()


def main():
    global GEOTIFF_OUTPUT_DIR, CENTROID_OUTPUT_DIR, COMBINED_CENTROID, SKIP_LOG_PATH, RUN_SUMMARY_PATH
    
    # Required on macOS and Windows to prevent recursive subprocess spawning
    multiprocessing.freeze_support()
    args = _parse_args()

    # Update global output paths to be year-aware
    county_tag = args.county.lower()
    year_tag = args.year
    GEOTIFF_OUTPUT_DIR = Path(f"../data/outputs/GeoTIFF_files/{county_tag}_{year_tag}/")
    CENTROID_OUTPUT_DIR = Path(f"../data/outputs/centroids/{county_tag}_{year_tag}/")
    COMBINED_CENTROID = Path(f"../data/outputs/{county_tag}_{year_tag}_centroids.csv")
    SKIP_LOG_PATH = Path(f"../data/outputs/{county_tag}_{year_tag}_skipped_tiles.csv")
    RUN_SUMMARY_PATH = Path(f"../data/outputs/{county_tag}_{year_tag}_run_summary.txt")

    # 1. Query VGIN REST API instead of reading CSV
    tile_list = query_vgin_tiles(
        county=args.county,
        year=args.year,
        current_only=args.current_only
    )

    if not tile_list:
        logger.error("No tiles found matching the specified filters. Exiting.")
        sys.exit(1)

    s3_bucket = "central-virginia-tree-canopy-project" if args.s3_upload else None

    # 2. Process tiles
    run_parallel(
        tile_list=tile_list,
        max_workers=args.workers,
        dry_run=args.dry_run,
        s3_bucket=s3_bucket,
        county=args.county,
        year=args.year
    )

if __name__ == "__main__":
    main()
