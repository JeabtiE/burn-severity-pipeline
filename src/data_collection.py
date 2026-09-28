"""
Stage 1: Data collection.

Downloads Sentinel-2 surface-reflectance imagery for a given AOI and
pre/post-fire date windows via Google Earth Engine, and loads matching
GISTDA hotspot / burn-scar vector data for the same AOI.

Run this in an environment where Earth Engine is authenticated —
Google Colab is the easiest option (same setup as the workshop
notebooks this project builds on). This sandbox has no network access
to earthengine.google.com, so this file is written but not executed
here; preprocessing.py, severity_classifier.py and evaluation.py carry
the logic that IS tested locally (see tests/test_core_logic.py).
"""

import datetime


def init_gee(project_name: str):
    """Authenticate + initialize Earth Engine. Run once per session."""
    import ee
    ee.Authenticate()
    ee.Initialize(project=project_name)


def mask_s2_clouds(image):
    """
    Pixel-level cloud/shadow masking using the Scene Classification
    Layer (SCL).

    This replaces the workshop's scene-level-only filter
    (CLOUDY_PIXEL_PERCENTAGE < 20%), which still let cloudy pixels
    through inside otherwise-accepted scenes and could pull NDVI/NBR
    means toward cloud values.

    SCL classes dropped: 3 (cloud shadow), 8 (cloud, medium prob.),
    9 (cloud, high prob.), 10 (thin cirrus), 11 (snow/ice).
    """
    scl = image.select("SCL")
    bad = scl.eq(3).Or(scl.eq(8)).Or(scl.eq(9)).Or(scl.eq(10)).Or(scl.eq(11))
    return image.updateMask(bad.Not())


def get_s2_collection(aoi, start_date: str, end_date: str, max_cloud_pct: float = 40.0):
    """
    Sentinel-2 SR collection for an AOI/date range.

    The scene-level pre-filter is loosened vs. the workshop's 20%
    since per-pixel masking (mask_s2_clouds) now does the real
    cleaning — a stricter scene filter here would just throw away
    otherwise-usable scenes that have a clear AOI but a cloudy corner
    elsewhere in the tile.
    """
    import ee
    return (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud_pct))
        .map(mask_s2_clouds)
    )


def get_least_cloudy_composite(aoi, start_date: str, end_date: str):
    """Median composite over a window — used to build a clean pre-fire
    and post-fire reference image for dNBR."""
    return get_s2_collection(aoi, start_date, end_date).median().clip(aoi)


def download_prepost_fire_pair(aoi, fire_date: str, pre_window_days: int = 30,
                                post_window_days: int = 30):
    """
    Build the pre-fire and post-fire composite images bracketing a
    known fire date, ready for dNBR calculation in preprocessing.py.
    """
    fire_dt = datetime.date.fromisoformat(fire_date)
    pre_start = (fire_dt - datetime.timedelta(days=pre_window_days * 2)).isoformat()
    pre_end = (fire_dt - datetime.timedelta(days=1)).isoformat()
    post_start = (fire_dt + datetime.timedelta(days=1)).isoformat()
    post_end = (fire_dt + datetime.timedelta(days=post_window_days)).isoformat()

    pre_image = get_least_cloudy_composite(aoi, pre_start, pre_end)
    post_image = get_least_cloudy_composite(aoi, post_start, post_end)
    return pre_image, post_image


def load_gistda_vectors(hotspot_path: str, burn_scar_path: str):
    """
    Load GISTDA hotspot points / burn-scar polygons already downloaded
    from ecoplant.gistda.or.th (the same files used in the workshop
    exercises). Returns GeoDataFrames.
    """
    import geopandas as gpd
    hotspot = gpd.read_file(hotspot_path)
    burn_scar = gpd.read_file(burn_scar_path)
    return hotspot, burn_scar
