"""
vgin_rest_download.py
Query the VGIN ArcGIS REST service for LiDAR tiles intersecting a county
bounding box and download the LAZ point cloud files.

Usage:
    python vgin_rest_download.py --county Albemarle --outdir ./laz_files/
    python vgin_rest_download.py --county Buckingham --year 2015 --outdir ./laz_files/
"""

import argparse
import re
import json
import urllib.request
import urllib.parse
from pathlib import Path
from pyproj import Transformer

# ── VGIN REST endpoint ─────────────────────────────────────────────────────
LAYER_URL = (
    "https://vginmaps.vdem.virginia.gov/arcgis/rest/services/"
    "Download/Virginia_LiDAR_Downloads/MapServer/1/query"
 )

# ── County bounding boxes (WGS84 lat/lon) ─────────────────────────────────
COUNTY_BBOX = {
    "Albemarle":  {"lat": (37.76, 38.30), "lon": (-79.00, -78.24)},
    "Buckingham": {"lat": (37.33, 37.77), "lon": (-78.85, -78.26)},
    "Fluvanna":   {"lat": (37.77, 38.07), "lon": (-78.38, -78.02)},
    "Greene":     {"lat": (38.14, 38.38), "lon": (-78.55, -78.28)},
    "Nelson":     {"lat": (37.68, 37.97), "lon": (-79.27, -78.72)},
    "Orange":     {"lat": (38.07, 38.38), "lon": (-78.22, -77.83)},
    "Louisa":     {"lat": (37.77, 38.07), "lon": (-78.05, -77.72)},
}

def bbox_to_web_mercator(lat_min, lat_max, lon_min, lon_max):
    """Convert WGS84 bounding box to EPSG:3857 (Web Mercator)."""
    t = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    xmin, ymin = t.transform(lon_min, lat_min)
    xmax, ymax = t.transform(lon_max, lat_max)
    return xmin, ymin, xmax, ymax

def extract_url(html_anchor):
    """Parse the direct download URL out of an HTML anchor string."""
    if not html_anchor:
        return None
    match = re.search(r'href=["\']([^"\']+)["\']', html_anchor, re.IGNORECASE)
    return match.group(1) if match else None

def query_tiles(county, year_filter=None, current_only=True):
    """Query the VGIN REST service and return tile records for the county."""
    bbox = COUNTY_BBOX[county]
    xmin, ymin, xmax, ymax = bbox_to_web_mercator(
        bbox["lat"][0], bbox["lat"][1],
        bbox["lon"][0], bbox["lon"][1]
    )

    geometry = json.dumps({
        "xmin": xmin, "ymin": ymin,
        "xmax": xmax, "ymax": ymax,
        "spatialReference": {"wkid": 102100}
    })

    # Build WHERE clause
    where_parts = ["1=1"]
    if current_only:
        where_parts.append("VComment = 'Current'")
    if year_filter:
        where_parts.append(f"ProjectYear = '{year_filter}'")
    where = " AND ".join(where_parts)

    params = urllib.parse.urlencode({
        "where":          where,
        "geometry":       geometry,
        "geometryType":   "esriGeometryEnvelope",
        "spatialRel":     "esriSpatialRelIntersects",
        "outFields":      "TileName,PointCloudDownload,DEMDownload,ProjectYear,VComment,PointCloudHost",
        "returnGeometry": "false",
        "resultRecordCount": 2000,
        "f":              "json",
    })

    url = f"{LAYER_URL}?{params}"
    with urllib.request.urlopen(url, timeout=30) as resp:
        data = json.loads(resp.read())

    tiles = []
    for feat in data.get("features", []):
        attr = feat["attributes"]
        laz_url = extract_url(attr.get("PointCloudDownload"))
        dem_url = extract_url(attr.get("DEMDownload"))
        if laz_url:
            tiles.append({
                "tile_name":    attr["TileName"],
                "laz_url":      laz_url,
                "dem_url":      dem_url,
                "project_year": attr["ProjectYear"],
                "status":       attr["VComment"],
                "host":         attr["PointCloudHost"],
            })
    return tiles

def download_file(url, dest_path, label=""):
    """Download a file with progress reporting."""
    dest_path = Path(dest_path)
    if dest_path.exists():
        print(f"  SKIP (exists): {dest_path.name}")
        return
    print(f"  Downloading {label}: {dest_path.name}")
    try:
        urllib.request.urlretrieve(url, dest_path)
        print(f"  OK: {dest_path.name} ({dest_path.stat().st_size / 1e6:.1f} MB)")
    except Exception as e:
        print(f"  ERROR: {dest_path.name} — {e}")
        if dest_path.exists():
            dest_path.unlink()   # remove partial file

def main():
    parser = argparse.ArgumentParser(description="Download VGIN LiDAR tiles via REST")
    parser.add_argument("--county",  required=True, choices=list(COUNTY_BBOX.keys()))
    parser.add_argument("--year",    default=None,  help="Filter by ProjectYear e.g. 2015")
    parser.add_argument("--outdir",  default="./laz_files", help="Output directory for LAZ files")
    parser.add_argument("--dem",     action="store_true",   help="Also download DEM GeoTIFFs")
    parser.add_argument("--dry-run", action="store_true",   help="Print URLs without downloading")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Querying VGIN REST service for {args.county} County"
          + (f" (year={args.year})" if args.year else "") + " ...")
    tiles = query_tiles(args.county, year_filter=args.year)
    print(f"Found {len(tiles)} tiles\n")

    for i, tile in enumerate(tiles, 1):
        print(f"[{i}/{len(tiles)}] {tile['tile_name']} "
              f"({tile['project_year']}, {tile['status']}, host={tile['host']})")

        laz_filename = Path(tile["laz_url"]).name
        if args.dry_run:
            print(f"  LAZ URL: {tile['laz_url']}")
            if args.dem and tile["dem_url"]:
                print(f"  DEM URL: {tile['dem_url']}")
        else:
            download_file(tile["laz_url"], outdir / laz_filename, label="LAZ")
            if args.dem and tile["dem_url"]:
                dem_filename = Path(tile["dem_url"]).name
                download_file(tile["dem_url"], outdir / dem_filename, label="DEM")

    print(f"\nDone. Files saved to: {outdir.resolve()}")

if __name__ == "__main__":
    main()

