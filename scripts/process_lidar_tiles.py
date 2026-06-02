"""
process_lidar_tiles.py — Parallel VGIN LiDAR LAZ Processing Pipeline
======================================================================
Incorporates:
  - concurrent.futures.ProcessPoolExecutor for tile-level parallelism
  - Thread-safe / process-safe skip logging via a multiprocessing Lock
  - Per-stage timing with time.perf_counter()
  - Four-level skip logic (empty / unclassified / no vegetation / below 2 m)
  - 2-meter minimum height filter at point cloud and CHM levels
  - Local maxima detection for tree canopy centroids (scipy.ndimage)
  - Progress bar via tqdm (optional — gracefully degrades if not installed)
  - CSV-driven tile list input (reads from CentralVA_LiDAR_SelectedTiles.csv)
  - Combined centroid output across all tiles
  - Run summary report

Usage:
  python process_lidar_tiles.py \
      --csv  path/to/CentralVA_LiDAR_SelectedTiles.csv \
      --workers 6 \
      --county Albemarle \
      --dry-run          # optional: print tile list without processing

Dependencies:
  pip install laspy[laszip] numpy scipy rasterio requests tqdm
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
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import List, Tuple

import laspy
import numpy as np
import requests
import rasterio
from rasterio.transform import from_origin
from scipy.interpolate import griddata
from scipy.ndimage import maximum_filter
from scipy.spatial import cKDTree

# ── Try importing tqdm; fall back to a no-op wrapper if not installed ──────────
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):  # type: ignore
        return iterable

# ── Configuration Constants ────────────────────────────────────────────────────
MIN_CANOPY_HEIGHT_M  = 2.0    # Minimum height above ground to be considered tree canopy
MAX_CANOPY_HEIGHT_M  = 60.0   # Maximum realistic tree height for Central Virginia
CROWN_RADIUS_M       = 3.0    # Local maxima search radius (meters) for crown peak detection
RASTER_RESOLUTION    = 1.0    # Output raster cell size in meters
DEFAULT_WORKERS      = max(1, os.cpu_count() - 2)  # Leave 2 cores for OS

GEOTIFF_OUTPUT_DIR   = Path("/home/thq3hn/development/DS6015-006/data/outputs/GeoTIFF_files/albemarle_county/")
CENTROID_OUTPUT_DIR  = Path("/home/thq3hn/development/DS6015-006/data/outputs/centroids/")
SKIP_LOG_PATH        = Path("/home/thq3hn/development/DS6015-006/data/outputs/skipped_tiles.csv")
UNCLASS_LOG_PATH     = Path("/home/thq3hn/development/DS6015-006/logs/unclassified_tiles.log")
COMBINED_CENTROID    = Path("/home/thq3hn/development/DS6015-006/data/outputs/all_tiles_centroids.csv")
RUN_SUMMARY_PATH     = Path("/home/thq3hn/development/DS6015-006/data/outputs/run_summary.txt")

# ── Logging Setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [PID %(process)d] %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — CSV Parsing: Extract tile URLs from the VGIN export
# ══════════════════════════════════════════════════════════════════════════════

def _extract_url(cell: str) -> str:
    """
    Extract a plain URL from an ArcGIS HTML anchor cell.
    ArcGIS exports download links as:  <a href=""https://..."">Download LPC</a>
    Returns an empty string if no URL is found.
    """
    if not cell:
        return ""
    # Handle double-double-quote ArcGIS artefact: href=""https://...""
    match = re.search(r'href=["\']+(https?://[^"\'> ]+)', cell)
    if match:
        return match.group(1)
    # Plain URL (no HTML wrapper)
    if cell.strip().startswith("http"):
        return cell.strip()
    return ""


def load_tile_list(csv_path: str,
                   county_filter: str = None,
                   current_only: bool = True) -> List[Tuple[str, str]]:
    """
    Read the VGIN LiDAR inventory CSV and return a list of (url, geotiff_filename) tuples.

    Parameters
    ----------
    csv_path      : Path to CentralVA_LiDAR_SelectedTiles.csv
    county_filter : If provided, only include tiles where County matches this string
                    (case-insensitive). Example: "Albemarle"
    current_only  : If True, skip rows where VComment contains "Replaced" (superceded tiles)
    """
    tiles: List[Tuple[str, str]] = []
    skipped_superceded = 0
    skipped_no_url     = 0

    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:

            # Optional: filter to a specific county
            if county_filter:
                county_val = row.get("County", row.get("COUNTY", ""))
                if county_filter.lower() not in county_val.lower():
                    continue

            # Optional: skip superceded tiles
            if current_only:
                comment = row.get("VComment", row.get("VCOMMENT", ""))
                if "replaced" in comment.lower():
                    skipped_superceded += 1
                    continue

            # Extract the LPC (point cloud) download URL
            url = _extract_url(row.get("PointClo_2", row.get("POINTCLO_2", "")))
            if not url:
                skipped_no_url += 1
                continue

            # Build output filename from tile ID or URL basename
            tile_id = row.get("TileID", row.get("TILEID", ""))
            if tile_id:
                geotiff_filename = f"{tile_id}_chm.tif"
            else:
                geotiff_filename = Path(url).stem + "_chm.tif"

            tiles.append((url, geotiff_filename))

    logger.info(f"Tile list loaded: {len(tiles)} tiles to process")
    if skipped_superceded:
        logger.info(f"  Skipped {skipped_superceded} superceded tiles (--current-only)")
    if skipped_no_url:
        logger.info(f"  Skipped {skipped_no_url} tiles with no download URL")

    return tiles


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Thread-Safe Skip Logging
# ══════════════════════════════════════════════════════════════════════════════

# A multiprocessing Manager lock is created in the main process and passed to
# each worker so that concurrent writes to the shared skip log are serialized.
_skip_lock: multiprocessing.Lock = None   # set in initializer


def _init_worker(lock: multiprocessing.Lock) -> None:
    """Initializer for each worker process — stores the shared lock globally."""
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
# SECTION 3 — Core Processing Function (runs in each worker process)
# ══════════════════════════════════════════════════════════════════════════════

def process_file(url: str, geotiff_filename: str,
                 retries: int = 3, timeout: int = 120) -> dict:
    """
    Download, decompress, and process a single VGIN LAZ tile.

    Pipeline stages:
      1.  HTTP download (chunked streaming)
      2.  LAZ decompression via laspy + Laszip backend
      3.  Point extraction and classification diagnostic
      4.  Skip guards (empty / unclassified / no vegetation / below 2 m)
      5.  Vegetation point filtering (>= 2 m above ground)
      6.  Raster grid setup
      7.  DTM interpolation (ground points → griddata)
      8.  DSM rasterization (vegetation max-return)
      9.  CHM computation (DSM − DTM) with height thresholds
      10. Local maxima detection → tree canopy centroids
      11. GeoTIFF write
      12. Centroid CSV write

    Returns a result dict with keys:
      status        : "success" | "skipped" | "failed"
      filename      : geotiff_filename
      url           : source URL
      n_trees       : number of tree crowns detected (0 for skipped/failed)
      skip_reason   : reason string if status == "skipped", else ""
      elapsed_s     : total wall-clock seconds for this tile
    """
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

            # ── Stage 1: HTTP Download ─────────────────────────────────────────
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

            # ── Stage 2: LAZ Decompression ────────────────────────────────────
            t0 = time.perf_counter()
            with laspy.open(buffer, laz_backend=laspy.LazBackend.Laszip) as fh:
                las = fh.read()
            t_decompress = time.perf_counter() - t0
            logger.info(f"[{pid}] Decompress: {t_decompress:.1f}s ({len(las.points):,} pts)")

            # ── Stage 3: Point Extraction ─────────────────────────────────────
            t0 = time.perf_counter()
            x: np.ndarray = las.x.scaled_array()
            y: np.ndarray = las.y.scaled_array()
            z: np.ndarray = las.z.scaled_array()
            classification: np.ndarray = np.array(las.classification)

            total_points        = len(x)
            ground_points       = int(np.sum(classification == 2))
            veg_points_raw      = int(np.sum(classification >= 3))
            unclassified_points = int(np.sum(classification <= 1))
            t_extract = time.perf_counter() - t0

            logger.info(
                f"[{pid}] Points — total:{total_points:,} "
                f"ground:{ground_points:,} veg:{veg_points_raw:,} "
                f"unclass:{unclassified_points:,} | extract:{t_extract:.1f}s"
            )

            # ── Stage 4: Skip Guards ──────────────────────────────────────────

            if total_points == 0:
                logger.warning(f"[{pid}] SKIP empty_tile: {geotiff_filename}")
                _log_skipped(url, geotiff_filename, "empty_tile", 0, 0, 0)
                result.update(status="skipped", skip_reason="empty_tile",
                              elapsed_s=time.perf_counter() - t_attempt_start)
                return result

            if unclassified_points == total_points:
                logger.warning(f"[{pid}] SKIP all_unclassified: {geotiff_filename}")
                UNCLASS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
                with open(UNCLASS_LOG_PATH, "a") as log:
                    log.write(f"{url}\t{geotiff_filename}\t{total_points}\n")
                _log_skipped(url, geotiff_filename, "all_unclassified",
                             total_points, ground_points, veg_points_raw)
                result.update(status="skipped", skip_reason="all_unclassified",
                              elapsed_s=time.perf_counter() - t_attempt_start)
                return result

            if veg_points_raw == 0:
                logger.warning(f"[{pid}] SKIP no_vegetation: {geotiff_filename}")
                _log_skipped(url, geotiff_filename, "no_vegetation",
                             total_points, ground_points, 0)
                result.update(status="skipped", skip_reason="no_vegetation",
                              elapsed_s=time.perf_counter() - t_attempt_start)
                return result

            # ── Stage 5: Vegetation Height Filter (>= 2 m) ───────────────────
            t0 = time.perf_counter()
            ground_mask = classification == 2
            x_ground = x[ground_mask]
            y_ground = y[ground_mask]
            z_ground = z[ground_mask]

            veg_mask_raw = classification >= 3
            x_veg_raw    = x[veg_mask_raw]
            y_veg_raw    = y[veg_mask_raw]
            z_veg_raw    = z[veg_mask_raw]

            ground_reference = (float(np.median(z_ground)) if ground_points > 0
                                else float(z.min()))

            height_filter     = (z_veg_raw - ground_reference) >= MIN_CANOPY_HEIGHT_M
            x_veg             = x_veg_raw[height_filter]
            y_veg             = y_veg_raw[height_filter]
            z_veg             = z_veg_raw[height_filter]
            veg_points_filtered = len(x_veg)
            t_filter = time.perf_counter() - t0

            logger.info(
                f"[{pid}] Veg filter: raw={veg_points_raw:,} "
                f">=2m={veg_points_filtered:,} | {t_filter:.1f}s"
            )

            if veg_points_filtered == 0:
                logger.warning(f"[{pid}] SKIP below_height_threshold: {geotiff_filename}")
                _log_skipped(url, geotiff_filename, "below_height_threshold",
                             total_points, ground_points, veg_points_raw)
                result.update(status="skipped", skip_reason="below_height_threshold",
                              elapsed_s=time.perf_counter() - t_attempt_start)
                return result

            # ── Stage 6: Raster Grid Setup ────────────────────────────────────
            t0 = time.perf_counter()
            resolution = RASTER_RESOLUTION
            x_min, x_max = float(x.min()), float(x.max())
            y_min, y_max = float(y.min()), float(y.max())
            ncols = int(np.ceil((x_max - x_min) / resolution))
            nrows = int(np.ceil((y_max - y_min) / resolution))
            grid_x = np.linspace(x_min, x_max, ncols)
            grid_y = np.linspace(y_min, y_max, nrows)
            grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)
            t_grid = time.perf_counter() - t0
            logger.info(f"[{pid}] Grid: {nrows}×{ncols} | {t_grid:.1f}s")

            # ── Stage 7: DTM Interpolation ────────────────────────────────────
            t0 = time.perf_counter()
            if ground_points >= 3:
                # ---- Subsample ground points if count is very large ----
                # cKDTree scales as O(n log n) for build and O(m log n) for query.
                # Beyond ~500k ground points the build time grows significantly.
                # Subsampling to 500k preserves DTM accuracy while capping memory usage.
                MAX_GROUND_PTS = 500_000
                if ground_points > MAX_GROUND_PTS:
                    rng = np.random.default_rng(seed=42)
                    idx_sample = rng.choice(ground_points, size=MAX_GROUND_PTS, replace=False)
                    x_g = x_ground[idx_sample]
                    y_g = y_ground[idx_sample]
                    z_g = z_ground[idx_sample]
                    logger.info(
                        f"[{pid}] Ground pts subsampled: {ground_points:,} → {MAX_GROUND_PTS:,}"
                    )
                else:
                    x_g, y_g, z_g = x_ground, y_ground, z_ground

                # ---- Build KD-tree from ground points ----
                ground_xy = np.column_stack([x_g, y_g])
                kd_tree = cKDTree(ground_xy)

                # ---- Query nearest neighbour for every grid cell ----
                # grid_xx and grid_yy are the meshgrid arrays from Stage 6.
                # workers=-1 uses all available CPU cores for the query.
                grid_coords = np.column_stack([grid_xx.ravel(), grid_yy.ravel()])
                distances, nn_indices = kd_tree.query(grid_coords, k=1, workers=-1)

                dtm = z_g[nn_indices].reshape(nrows, ncols).astype(np.float32)

                # ---- Mask grid cells that are too far from any ground point ----
                # Cells farther than MAX_INTERP_DIST metres from the nearest ground
                # return are likely over water, rooftops, or data voids.  Setting
                # them to NaN prevents spurious negative CHM values in those areas.
                MAX_INTERP_DIST = resolution * 10   # 10 pixels — adjust if needed
                dtm[distances.reshape(nrows, ncols) > MAX_INTERP_DIST] = np.nan

                dtm_min = float(np.nanmin(dtm))
                dtm_max = float(np.nanmax(dtm))
                logger.info(
                    f"[{pid}] DTM (cKDTree NN): {time.perf_counter() - t0:.1f}s  "
                    f"(min: {dtm_min:.2f} m, max: {dtm_max:.2f} m)"
                )
            else:
                logger.warning(f"[{pid}] Fewer than 3 ground pts — using flat DTM")
                dtm = np.full((nrows, ncols), ground_reference)
                # Fewer than 3 ground points — fall back to a flat plane
                logger.warning(
                    f"[{pid}] Fewer than 3 ground pts ({ground_points}) — "
                    f"using flat DTM at {ground_reference:.2f} m"
                )
                dtm = np.full((nrows, ncols), ground_reference, dtype=np.float32)
            t_dtm = time.perf_counter() - t0
            logger.info(f"[{pid}] DTM: {t_dtm:.1f}s")

            # ── Stage 8: DSM Rasterization ────────────────────────────────────
            t0 = time.perf_counter()
            dsm     = np.full((nrows, ncols), -np.inf)
            col_idx = ((x_veg - x_min) / resolution).astype(int)
            row_idx = ((y_max - y_veg) / resolution).astype(int)
            valid   = ((col_idx >= 0) & (col_idx < ncols) &
                       (row_idx >= 0) & (row_idx < nrows))
            np.maximum.at(dsm, (row_idx[valid], col_idx[valid]), z_veg[valid])
            dsm[dsm == -np.inf] = np.nan
            t_dsm = time.perf_counter() - t0
            logger.info(f"[{pid}] DSM: {t_dsm:.1f}s")

            # ── Stage 9: CHM Computation ──────────────────────────────────────
            t0 = time.perf_counter()
            chm = dsm - dtm
            chm = np.where(chm < 0, 0, chm)
            chm = np.where(chm < MIN_CANOPY_HEIGHT_M, 0, chm)
            chm = np.where(chm > MAX_CANOPY_HEIGHT_M, np.nan, chm)
            t_chm = time.perf_counter() - t0
            logger.info(
                f"[{pid}] CHM: max={np.nanmax(chm):.1f}m "
                f"mean={np.nanmean(chm[chm >= MIN_CANOPY_HEIGHT_M]):.1f}m | {t_chm:.1f}s"
            )

            # ── Stage 10: Local Maxima Detection ─────────────────────────────
            t0 = time.perf_counter()
            search_radius_px = int(np.ceil(CROWN_RADIUS_M / resolution))
            chm_trees = np.where(chm >= MIN_CANOPY_HEIGHT_M, chm, 0)
            local_max = maximum_filter(chm_trees, size=2 * search_radius_px + 1)
            peaks     = (chm_trees == local_max) & (chm_trees >= MIN_CANOPY_HEIGHT_M)
            peak_rows, peak_cols = np.where(peaks)
            n_trees  = len(peak_rows)
            peak_x   = x_min + (peak_cols + 0.5) * resolution
            peak_y   = y_max - (peak_rows + 0.5) * resolution
            peak_heights = chm[peak_rows, peak_cols]
            t_peaks  = time.perf_counter() - t0
            logger.info(f"[{pid}] Peaks: {n_trees:,} crowns | {t_peaks:.1f}s")

            # ── Stage 11: GeoTIFF Write ───────────────────────────────────────
            t0 = time.perf_counter()
            GEOTIFF_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            geotiff_path = GEOTIFF_OUTPUT_DIR / geotiff_filename
            transform    = from_origin(x_min, y_max, resolution, resolution)
            with rasterio.open(
                geotiff_path, mode="w", driver="GTiff",
                height=nrows, width=ncols, count=1,
                dtype=chm.dtype, crs="EPSG:32618",
                transform=transform, nodata=np.nan,
            ) as dst:
                dst.write(chm, 1)
            t_write = time.perf_counter() - t0
            logger.info(f"[{pid}] GeoTIFF: {t_write:.1f}s → {geotiff_path}")

            # ── Stage 12: Centroid CSV Write ──────────────────────────────────
            t0 = time.perf_counter()
            CENTROID_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            centroid_filename = Path(geotiff_filename).stem + "_centroids.csv"
            centroid_path     = CENTROID_OUTPUT_DIR / centroid_filename
            with open(centroid_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["easting_m", "northing_m", "height_m"])
                for ex, ey, eh in zip(peak_x, peak_y, peak_heights):
                    writer.writerow([round(float(ex), 3),
                                     round(float(ey), 3),
                                     round(float(eh), 2)])
            t_csv = time.perf_counter() - t0

            t_total = time.perf_counter() - t_attempt_start
            logger.info(
                f"[{pid}] DONE {geotiff_filename} | "
                f"trees={n_trees:,} | total={t_total:.1f}s | "
                f"dl={t_download:.1f}s dcmp={t_decompress:.1f}s "
                f"dtm={t_dtm:.1f}s peaks={t_peaks:.1f}s"
            )

            result.update(status="success", n_trees=n_trees, elapsed_s=t_total)
            return result

        except requests.exceptions.RequestException as e:
            logger.warning(f"[{pid}] Attempt {attempt}/{retries} network error: {e}")
            if attempt < retries:
                time.sleep(5 * attempt)

        except Exception as e:
            logger.warning(f"[{pid}] Attempt {attempt}/{retries} processing error: {e}",
                           exc_info=True)
            if attempt < retries:
                time.sleep(5 * attempt)

    logger.error(f"[{pid}] FAIL giving up: {url}")
    result["elapsed_s"] = time.perf_counter() - (time.perf_counter())
    return result


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Parallel Orchestrator
# ══════════════════════════════════════════════════════════════════════════════

def run_parallel(tile_list: List[Tuple[str, str]],
                 max_workers: int = DEFAULT_WORKERS,
                 dry_run: bool = False) -> None:
    """
    Dispatch all tiles to a ProcessPoolExecutor and collect results.

    Parameters
    ----------
    tile_list   : List of (url, geotiff_filename) tuples
    max_workers : Number of parallel worker processes
    dry_run     : If True, print the tile list and exit without processing
    """
    if dry_run:
        print(f"\n[DRY RUN] {len(tile_list)} tiles would be processed with {max_workers} workers:\n")
        for i, (url, fname) in enumerate(tile_list[:20], 1):
            print(f"  {i:4d}. {fname}")
            print(f"        {url}")
        if len(tile_list) > 20:
            print(f"  ... and {len(tile_list) - 20} more tiles")
        return

    t_run_start = time.perf_counter()
    logger.info(f"Starting parallel processing: {len(tile_list)} tiles | "
                f"{max_workers} workers")

    # Create a Manager lock so all worker processes share one skip-log lock
    manager    = multiprocessing.Manager()
    skip_lock  = manager.Lock()

    counters   = {"success": 0, "skipped": 0, "failed": 0, "total_trees": 0}
    all_results: list = []

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_worker,
        initargs=(skip_lock,),
    ) as executor:

        # Submit all tiles
        future_to_tile = {
            executor.submit(process_file, url, fname): (url, fname)
            for url, fname in tile_list
        }

        # Collect results as they complete
        for future in tqdm(as_completed(future_to_tile),
                           total=len(tile_list),
                           desc="Processing tiles",
                           unit="tile"):
            #result = future.result()
            try:
                for future in as_completed(future_to_tile):
                    tile_info = future_to_tile[future]
                    try:
                        result = future.result()
                    except BrokenProcessPool:
                        print(f"[FATAL] Worker pool broken — likely OOM. "
                              f"Reduce --workers or increase --mem.")
                        raise
                    except Exception as e:
                        print(f"[FAIL] {tile_info['filename']}: {e}")
            except BrokenProcessPool:
                print("Restarting with fewer workers...")
            all_results.append(result)
            counters[result["status"]] += 1
            counters["total_trees"]    += result.get("n_trees", 0)

    # ── Combine all per-tile centroid CSVs into one master file ───────────────
    logger.info("Combining per-tile centroid CSVs into master file...")
    COMBINED_CENTROID.parent.mkdir(parents=True, exist_ok=True)
    with open(COMBINED_CENTROID, "w", newline="") as master:
        writer = csv.writer(master)
        writer.writerow(["tile_id", "easting_m", "northing_m", "height_m"])
        for centroid_csv in sorted(CENTROID_OUTPUT_DIR.glob("*_centroids.csv")):
            tile_id = centroid_csv.stem.replace("_centroids", "")
            with open(centroid_csv, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    writer.writerow([tile_id,
                                     row["easting_m"],
                                     row["northing_m"],
                                     row["height_m"]])
    logger.info(f"Master centroid file: {COMBINED_CENTROID}")

    # ── Run Summary ───────────────────────────────────────────────────────────
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
        f"  Total wall-clock time : {t_total_run:.1f}s  "
                f"({t_total_run / 60:.1f} min)",
        f"  Master centroid file  : {COMBINED_CENTROID}",
        f"  Skip log              : {SKIP_LOG_PATH}",
        "=" * 60,
    ]

    for line in summary_lines:
        logger.info(line)

    RUN_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RUN_SUMMARY_PATH, "w") as f:
        f.write("\n".join(summary_lines) + "\n")
    logger.info(f"Run summary written to: {RUN_SUMMARY_PATH}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — CLI Entry Point
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parallel VGIN LiDAR LAZ processing pipeline"
    )
    parser.add_argument(
        "--csv", required=True,
        help="Path to CentralVA_LiDAR_SelectedTiles.csv"
    )
    parser.add_argument(
        "--workers", type=int, default=DEFAULT_WORKERS,
        help=f"Number of parallel worker processes (default: {DEFAULT_WORKERS})"
    )
    parser.add_argument(
        "--county", default=None,
        help="Filter tiles to a specific county name (e.g. 'Albemarle')"
    )
    parser.add_argument(
        "--current-only", action="store_true", default=True,
        help="Skip superceded tiles (VComment contains 'Replaced')"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the tile list without downloading or processing"
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Required on macOS and Windows to prevent recursive subprocess spawning
    multiprocessing.freeze_support()

    args = _parse_args()

    tile_list = load_tile_list(
        csv_path=args.csv,
        county_filter=args.county,
        current_only=args.current_only,
    )

    if not tile_list:
        logger.error("No tiles found matching the specified filters. Exiting.")
        sys.exit(1)

    run_parallel(
        tile_list=tile_list,
        max_workers=args.workers,
        dry_run=args.dry_run,
    )
