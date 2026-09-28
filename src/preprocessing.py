"""
Stage 2: Preprocessing — NBR / dNBR calculation and standard USGS
burn-severity classification.

Two flavours of the same math are provided:
  - the Earth Engine (ee.Image) version, for use directly on GEE
    images right after download_prepost_fire_pair()
  - the plain-numpy version, for use once bands have been exported to
    local arrays (e.g. via geemap.ee_to_numpy) — this is also what
    tests/test_core_logic.py runs against, since Earth Engine itself
    can't run in this sandbox.
"""

import numpy as np

# USGS / Key & Benson dNBR severity thresholds (unitless dNBR).
# Ordered low -> high; each tuple is (low, high, label), low inclusive.
SEVERITY_THRESHOLDS = [
    (-1.000, -0.500, "high_regrowth"),
    (-0.500, -0.251, "low_regrowth"),
    (-0.250, 0.099, "unburned"),
    (0.100, 0.269, "low_severity"),
    (0.270, 0.439, "moderate_low_severity"),
    (0.440, 0.659, "moderate_high_severity"),
    (0.660, 1.300, "high_severity"),
]


def compute_nbr(nir: np.ndarray, swir: np.ndarray) -> np.ndarray:
    """NBR = (NIR - SWIR) / (NIR + SWIR). For Sentinel-2: NIR=B8, SWIR=B12."""
    denom = nir + swir
    with np.errstate(divide="ignore", invalid="ignore"):
        nbr = np.where(denom == 0, np.nan, (nir - swir) / denom)
    return nbr


def compute_dnbr(nbr_pre: np.ndarray, nbr_post: np.ndarray) -> np.ndarray:
    """dNBR = NBR_pre - NBR_post. Positive values indicate vegetation loss."""
    return nbr_pre - nbr_post


def classify_severity(dnbr: np.ndarray) -> np.ndarray:
    """
    Map a dNBR array to severity-class labels using the USGS
    thresholds above. Returns an object array of strings, same shape
    as the input (NaN input -> 'nodata').
    """
    out = np.full(dnbr.shape, "nodata", dtype=object)
    valid = ~np.isnan(dnbr)
    for low, high, label in SEVERITY_THRESHOLDS:
        mask = valid & (dnbr >= low) & (dnbr < high)
        out[mask] = label
    return out


def ee_compute_nbr(image, nir_band: str = "B8", swir_band: str = "B12"):
    """Earth Engine version of compute_nbr, for use on ee.Image objects."""
    return image.normalizedDifference([nir_band, swir_band]).rename("NBR")


def ee_compute_dnbr(nbr_pre_image, nbr_post_image):
    """Earth Engine version of compute_dnbr."""
    return nbr_pre_image.subtract(nbr_post_image).rename("dNBR")
