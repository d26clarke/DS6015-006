import urllib.request, json, urllib.parse, re

VGIN_URL = "https://vginmaps.vdem.virginia.gov/arcgis/rest/services/Download/Virginia_LiDAR_Downloads/MapServer/1/query"

def find_laz_for_gps(lat: float, lon: float ) -> dict | None:
    """
    Query VGIN REST API to find the LAZ tile containing (lat, lon).
    Returns dict with TileName, ProjectYear, download URL, or None if not found.
    """
    params = urllib.parse.urlencode({
        "geometry": f"{lon},{lat}",        # x,y order (lon first)
        "geometryType": "esriGeometryPoint",
        "inSR": "4326",                    # WGS84 input
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "TileName,ProjectYear,PointCloudDownload,PointCloudFormat,VLPID",
        "returnGeometry": "false",
        "f": "json"
    })
    with urllib.request.urlopen(VGIN_URL + "?" + params, timeout=20) as r:
        data = json.load(r)

    features = data.get("features", [])
    if not features:
        return None

    attr = features[0]["attributes"]
    # Strip HTML anchor tags from the download URL field
    raw_url = attr.get("PointCloudDownload", "")
    match = re.search(r'href="([^"]+)"', raw_url)
    download_url = match.group(1) if match else raw_url

    return {
        "tile_name":    attr.get("TileName"),
        "project_year": attr.get("ProjectYear"),
        "format":       attr.get("PointCloudFormat"),
        "vlpid":        attr.get("VLPID"),
        "download_url": download_url,
    }

# Example usage:
result = find_laz_for_gps(38.0494, -78.4737)
# → {'tile_name': 'S13_4990_20', 'project_year': '2016', 'format': 'LAZ',
#    'vlpid': 29, 'download_url': 'https://rockyweb.usgs.gov/...S13_4990_20.laz'}

