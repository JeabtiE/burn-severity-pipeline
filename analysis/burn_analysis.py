"""
Burn-scar detection experiments: Sentinel-2 NBR vs GISTDA burn-scar polygons.

The functions below were run cell by cell in Google Colab (Earth Engine + Drive).
This file puts them in one place, in the order I used them. Authenticate once
in a fresh runtime before calling main():

    import ee; ee.Authenticate()

Usage:
    python analysis/burn_analysis.py path/to/burn_scar.shp results
"""
import os
import sys

import ee
import numpy as np
import pandas as pd
import geopandas as gpd
import requests
import rasterio
from rasterio import features
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from scipy.ndimage import binary_dilation
from shapely.geometry import box
from sklearn.metrics import roc_auc_score

EE_PROJECT = "burn-severity-pipeline"
UTM = "EPSG:32647"          # UTM 47N, metres (northern Thailand)
CELL = 10_000               # block size in metres (10 x 10 km)
PX = 20                     # pixel size in metres (Sentinel-2 SWIR)
N = CELL // PX              # pixels per block side (500)
NODATA = -9999
FOREST_LU = "ป่าไม้"         # LU_Name value for forest in the GISTDA file
MIN_PIECE_HA = 0.04         # smaller than one 20 m pixel

# Windows are end-exclusive (Earth Engine filterDate)
OLD_PRE_WIN = ("2026-03-01", "2026-04-01")
OLD_POST_WIN = ("2026-05-01", "2026-06-01")
PRE_WIN = ("2026-03-20", "2026-04-11")     # chosen from the dev block's own time series
POST_WIN = ("2026-04-14", "2026-04-23")

DEV_BLOCK = (41, 207)
TEST_BLOCKS = [(52, 220), (46, 216)]

GAP_WINDOWS = [
    ("2026-03-01", "2026-03-11"), ("2026-03-11", "2026-03-21"),
    ("2026-03-21", "2026-04-01"), ("2026-04-01", "2026-04-11"),
    ("2026-04-11", "2026-04-21"), ("2026-04-21", "2026-05-01"),
    ("2026-05-01", "2026-05-11"), ("2026-05-11", "2026-05-21"),
    ("2026-05-21", "2026-06-01"),
]


# ---------------------------------------------------------------- labels

def load_pieces(path):
    """Read burn-scar polygons, split multipart shapes, drop pieces below one pixel."""
    raw = gpd.read_file(path)
    parts = raw.explode(index_parts=False).reset_index(drop=True)
    parts["area_ha"] = parts.to_crs(epsg=32647).area / 10_000
    p = parts[parts["area_ha"] >= MIN_PIECE_HA].to_crs(epsg=32647).copy()
    return raw, parts, p


def audit(raw, parts):
    """Print the numbers that shaped the design: land use, piece sizes."""
    r = raw.copy()
    r["area_ha"] = r.to_crs(epsg=32647).area / 10_000
    share = r.groupby("LU_Name")["area_ha"].sum().sort_values(ascending=False)
    print((share / share.sum() * 100).round(1).head(8))
    if "name" in r.columns:
        named = r.loc[r["name"].notna(), "area_ha"].sum() / r["area_ha"].sum() * 100
        print(f"burned area with a forest-group name: {named:.1f}%")
    print("pieces after explode:", len(parts))
    print(parts["area_ha"].describe())
    print("pieces >= 1 ha:", (parts["area_ha"] >= 1).sum(),
          "| >= 5 ha:", (parts["area_ha"] >= 5).sum())


def grid_summary(p):
    """Sum burned area per 10 x 10 km block, largest first."""
    q = p.copy()
    c = q.geometry.centroid
    q["gx"] = (c.x // CELL).astype(int)
    q["gy"] = (c.y // CELL).astype(int)
    return q.groupby(["gx", "gy"]).agg(
        ha=("area_ha", "sum"),
        n=("area_ha", "size"),
        n1=("area_ha", lambda s: (s >= 1).sum()),
        lu=("LU_Name", lambda s: s.mode().iat[0]),
        district=("AP_TN", lambda s: s.mode().iat[0]),
    ).sort_values("ha", ascending=False)


def make_masks(p, gx, gy):
    """Pixel masks for one block: all burned, forest burned, and a ring control.

    The ring is 3-10 pixels (60-200 m) away from any burned pixel, so it shares
    haze, date and (roughly) surroundings with the burn scars.
    """
    x0, y0 = gx * CELL, gy * CELL
    tf = from_origin(x0, y0 + CELL, PX, PX)
    blk = p[p.intersects(box(x0, y0, x0 + CELL, y0 + CELL))]

    def rast(gdf):
        if len(gdf) == 0:
            return np.zeros((N, N), dtype="uint8")
        return features.rasterize(((g, 1) for g in gdf.geometry),
                                  out_shape=(N, N), transform=tf,
                                  fill=0, dtype="uint8")

    bm = rast(blk)
    fm = rast(blk[blk["LU_Name"] == FOREST_LU])
    rg = binary_dilation(bm, iterations=10) & ~binary_dilation(bm, iterations=2)
    return bm, fm, rg, tf


def save_mask(path, mask, tf):
    with rasterio.open(path, "w", driver="GTiff", height=N, width=N, count=1,
                       dtype="uint8", crs=UTM, transform=tf) as f:
        f.write(mask, 1)


# ---------------------------------------------------------------- imagery

def block_geometry(gx, gy):
    """Earth Engine region and pixel grid that line up with make_masks()."""
    x0, y0 = gx * CELL, gy * CELL
    reg = ee.Geometry.Rectangle([x0, y0, x0 + CELL, y0 + CELL],
                                proj=UTM, geodesic=False)
    grd = [PX, 0, x0, 0, -PX, y0 + CELL]
    return x0, y0, reg, grd


def mask_cloud(img):
    """Drop cloud shadow, cloud and snow pixels using the SCL band.

    SCL does not catch smoke or haze well, which matters in the burning season.
    """
    scl = img.select("SCL")
    bad = scl.eq(3).Or(scl.eq(8)).Or(scl.eq(9)).Or(scl.eq(10)).Or(scl.eq(11))
    return img.updateMask(bad.Not())


def composite(reg, start, end, max_cloud=80):
    """Median B8/B12 composite. Returns (image, image_count); image is None if empty."""
    col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
           .filterBounds(reg).filterDate(start, end)
           .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud))
           .map(mask_cloud))
    count = col.size().getInfo()
    print(f"  {start} to {end}: {count} images")
    if count == 0:
        return None, 0
    # empty pixels become -9999, never 0, so they cannot turn into fake NBR values
    return col.select(["B8", "B12"]).median().toFloat().unmask(NODATA), count


def fetch_arr(img, reg, grd):
    """Download an Earth Engine image as a numpy array (bands, rows, cols)."""
    url = img.getDownloadURL({"region": reg, "crs": UTM,
                              "crs_transform": grd, "format": "GEO_TIFF"})
    r = requests.get(url)
    r.raise_for_status()
    with MemoryFile(r.content) as mf:
        with mf.open() as f:
            return f.read().astype("float32")


def nbr(a):
    """NBR = (B8 - B12) / (B8 + B12); a has bands B8, B12 first."""
    b8, b12 = a[0], a[1]
    bad = (b8 == NODATA) | (b12 == NODATA) | ((b8 + b12) == 0)
    return np.where(bad, np.nan, (b8 - b12) / (b8 + b12 + 1e-9))


# ---------------------------------------------------------------- scoring

def score_block(d, bm, fm, rg, gx, gy):
    """AUC of dNBR against the burn mask, two ways: vs ring only, vs whole block."""
    ok = ~np.isnan(d)
    sel = ok & ((bm == 1) | rg)

    def mean_of(mask):
        v = ok & mask
        return round(float(d[v].mean()), 3) if v.any() else np.nan

    return {
        "block": f"{gx},{gy}",
        "burned_pixels": int(bm.sum()),
        "valid_pct": round(float(ok.mean()) * 100, 1),
        "auc_vs_ring": round(roc_auc_score(bm[sel], d[sel]), 3),
        "auc_vs_block": round(roc_auc_score(bm[ok], d[ok]), 3),
        "dnbr_agri": mean_of((bm == 1) & (fm == 0)),
        "dnbr_forest": mean_of(fm == 1),
        "dnbr_ring": mean_of(rg),
    }


def group_table(a, b, bm, fm, rg):
    """NBR before/after and dNBR per pixel group (a = pre image, b = post image)."""
    na, nb = nbr(a), nbr(b)
    d = na - nb
    ok = ~np.isnan(d)
    groups = {
        "outside_burn": ok & (bm == 0),
        "burned_all": ok & (bm == 1),
        "burned_forest": ok & (fm == 1),
        "burned_agri": ok & (bm == 1) & (fm == 0),
        "ring": ok & rg,
    }
    rows = []
    for name, m in groups.items():
        row = {"group": name, "n": int(m.sum())}
        if m.any():
            row.update({"nbr_pre": na[m].mean(), "nbr_post": nb[m].mean(),
                        "dnbr_mean": d[m].mean(), "dnbr_median": np.median(d[m]),
                        "dnbr_sd": d[m].std()})
        rows.append(row)
    return pd.DataFrame(rows).round(3)


def run_block(p, gx, gy, pre_win, post_win, max_cloud=80):
    """Masks + pre/post composites + dNBR score for one block."""
    bm, fm, rg, tf = make_masks(p, gx, gy)
    x0, y0, reg, grd = block_geometry(gx, gy)
    pre_img, _ = composite(reg, *pre_win, max_cloud)
    post_img, _ = composite(reg, *post_win, max_cloud)
    if pre_img is None or post_img is None:
        raise ValueError(f"no imagery for block {gx},{gy} in the chosen windows")
    a, b = fetch_arr(pre_img, reg, grd), fetch_arr(post_img, reg, grd)
    d = nbr(a) - nbr(b)                      # pre - post: higher = more change
    return score_block(d, bm, fm, rg, gx, gy), (a, b, bm, fm, rg)


# ---------------------------------------------------------------- time series

def gap_series(gx, gy, bm, rg, windows, max_cloud=80):
    """Mean NBR of burned pixels vs ring in consecutive windows.

    gap_change is the burned-minus-ring gap relative to the first window, which
    removes the constant land-cover offset. A burn would show up as a step down.
    """
    x0, y0, reg, grd = block_geometry(gx, gy)
    rows = []
    for start, end in windows:
        img, count = composite(reg, start, end, max_cloud)
        row = {"window_start": start, "window_end": end, "images": count,
               "nbr_burned": np.nan, "nbr_ring": np.nan}
        if img is not None:
            v = nbr(fetch_arr(img, reg, grd))
            ok = ~np.isnan(v)
            if (ok & (bm == 1)).any():
                row["nbr_burned"] = v[ok & (bm == 1)].mean()
            if (ok & rg).any():
                row["nbr_ring"] = v[ok & rg].mean()
        rows.append(row)
    t = pd.DataFrame(rows).set_index("window_start")
    t["gap"] = t["nbr_burned"] - t["nbr_ring"]
    t["gap_change"] = t["gap"] - t["gap"].dropna().iloc[0]
    return t.round(3)


def daily_series(gx, gy, bm, fm, rg, start="2026-02-01", end="2026-07-01"):
    """Mean NBR per acquisition date for burned agriculture, burned forest and ring.

    A group is reported on a date only if at least 30% of its pixels are valid.
    Slow: one download per date (about 60 dates).
    """
    x0, y0, reg, grd = block_geometry(gx, gy)
    col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
           .filterBounds(reg).filterDate(start, end))
    ms = col.aggregate_array("system:time_start").getInfo()
    days = sorted(set(pd.to_datetime(ms, unit="ms").strftime("%Y-%m-%d")))
    print("days with imagery:", len(days))

    groups = {"burned_agri": (bm == 1) & (fm == 0),
              "burned_forest": fm == 1,
              "ring": rg}
    rows = []
    for d in days:
        day = ee.Date(d)
        img = (col.filterDate(day, day.advance(1, "day"))
               .map(lambda i: mask_cloud(i).normalizedDifference(["B8", "B12"]).rename("nbr"))
               .mosaic().toFloat().unmask(NODATA))
        a = fetch_arr(img, reg, grd)[0]
        okd = a != NODATA
        row = {"date": d}
        for name, m in groups.items():
            v = m & okd
            enough = m.sum() > 0 and v.sum() >= 0.3 * m.sum()
            row[name] = a[v].mean() if enough else np.nan
            row["valid_pct_" + name] = round(v.sum() / max(m.sum(), 1) * 100, 1)
        rows.append(row)
    return pd.DataFrame(rows).set_index("date")


def plot_series(ts, path):
    import matplotlib.pyplot as plt
    ts[["burned_agri", "burned_forest", "ring"]].plot(marker="o", figsize=(10, 4))
    plt.ylabel("mean NBR")
    plt.title("NBR time series, block 41,207")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ---------------------------------------------------------------- run everything

def main(scar_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    ee.Initialize(project=EE_PROJECT)

    # data audit and block selection
    raw, parts, p = load_pieces(scar_path)
    audit(raw, parts)
    grid_summary(p).head(8).to_csv(f"{out_dir}/top_blocks.csv")

    # 1) first attempt: March vs May composites on the development block
    gx, gy = DEV_BLOCK
    res, (a, b, bm, fm, rg) = run_block(p, gx, gy, OLD_PRE_WIN, OLD_POST_WIN)
    print(res)
    group_table(a, b, bm, fm, rg).to_csv(
        f"{out_dir}/baseline_march_may_block_41_207.csv", index=False)

    # 2) revised windows, frozen, then scored on held-out blocks
    rows, cache = [], {}
    for gid in [DEV_BLOCK] + TEST_BLOCKS:
        res, arrs = run_block(p, *gid, PRE_WIN, POST_WIN)
        res["role"] = "development" if gid == DEV_BLOCK else "test"
        rows.append(res)
        cache[gid] = arrs
    pd.DataFrame(rows).to_csv(f"{out_dir}/holdout_results.csv", index=False)

    # 3) 10-day windows: is there any step down in burned vs ring NBR?
    out = []
    for gid, arrs in cache.items():
        t = gap_series(*gid, arrs[2], arrs[4], GAP_WINDOWS)
        t.insert(0, "block", f"{gid[0]},{gid[1]}")
        out.append(t.reset_index())
    pd.concat(out).to_csv(f"{out_dir}/window_gap_series.csv", index=False)

    # 4) per-date series for the development block (slow)
    a, b, bm, fm, rg = cache[DEV_BLOCK]
    ts = daily_series(*DEV_BLOCK, bm, fm, rg)
    ts.to_csv(f"{out_dir}/nbr_daily_series_41_207.csv")
    plot_series(ts, f"{out_dir}/nbr_daily_series_41_207.png")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
