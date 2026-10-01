"""
stage0.py : Stage 0 ทดสอบความเป็นไปได้ของโปรเจค Burning Recurrence Risk

คำถามที่ตอบ: ประวัติ hotspot ของฤดูก่อนๆ บอกได้ไหมว่าพื้นที่ไหนจะมี hotspot ในฤดูถัดไป
(ยังไม่ใช้โมเดล ML ใช้แค่ baseline ง่ายๆ เพื่อดูว่าโจทย์นี้มีสัญญาณให้ทำต่อหรือเปล่า)

วิธีใช้ใน Colab:
    !pip install h3 -q
    import sys
    sys.path.append("/content/drive/MyDrive/burn_severity_pipeline/recurrence")
    import stage0
    summary = stage0.run_all(FIRMS_DIR, OUT_DIR)
"""
import os
import glob

import numpy as np
import pandas as pd
import h3
from scipy import sparse
from sklearn.metrics import roc_auc_score, average_precision_score

# ======================================================================
# ค่าที่ล็อกแล้ว ล็อกก่อนเห็นตัวเลข recurrence ใดๆ ห้ามแก้หลังเห็นผล
# ถ้าจะแก้ ให้ถือเป็นการทดลองใหม่และบันทึกไว้ในรายงานว่าแก้เพราะอะไร
# ======================================================================
BBOX = (98.0, 18.2, 99.5, 20.0)   # west, south, east, north ตรงกับกรอบที่ขอจาก FIRMS
SEASON_MONTHS = (1, 5)            # ฤดูไฟ ม.ค.-พ.ค. (หมายเหตุ: S-NPP เริ่มเก็บ 20 ม.ค. 2012 ฤดู 2012 ขาดต้นเดือนม.ค.ไปเล็กน้อย)
MIN_CONF = 50                     # confidence ขั้นต่ำ (VIIRS: l=30, n=60, h=90 จึงตัด l ทิ้ง)
TOP = 0.10                        # "top 10% ของ cell" ที่ใช้วัด
N_HIST = 3                        # B1 นับจำนวนฤดูที่ไหม้ในช่วง 3 ปีล่าสุด
K_SMOOTH = 1                      # B2 ปรับให้เรียบด้วยเพื่อนบ้านวงที่ 1 (k-ring 1)
GATE_LIFT = 3.0                   # เกณฑ์ผ่าน: top 10% ต้องไหม้บ่อยกว่าการสุ่มอย่างน้อย 3 เท่า
GATE_CAP = 0.8                    # ถ้า 3 เท่าเกินเพดานที่เป็นไปได้ (เมื่อ cell ไหม้เกิน ~27% ของพื้นที่)
                                  # ให้ใช้ 80% ของเพดานแทน เกณฑ์ต่อฤดู = min(3, 0.8 / สัดส่วน cell ที่ไหม้)
GATE_SEASONS = 0.5                # ต้องผ่านเกณฑ์อย่างน้อยครึ่งหนึ่งของฤดูทดสอบ
STEP = 0.003                      # ระยะห่างจุดสุ่มสร้าง cell ทั้งพื้นที่ (องศา ราว 330 ม.) ถี่พอสำหรับ H3 res 8

SERIES = {
    "A_SNPP": {"sensor": "VIIRS_SNPP", "years": (2012, 2026), "res": [7, 8]},
    "B_MODIS": {"sensor": "MODIS", "years": (2003, 2026), "res": [7]},
}

SENSOR_BY_TAG = {
    "M-C61": "MODIS",
    "SV-C2": "VIIRS_SNPP",   # Suomi NPP
    "J1V-C2": "VIIRS_N20",   # JPSS-1 = NOAA-20
    "J2V-C2": "VIIRS_N21",   # JPSS-2 = NOAA-21
}

# h3 v4 ใช้ชื่อ latlng_to_cell / grid_disk ส่วน v3 ใช้ geo_to_h3 / k_ring
to_cell = getattr(h3, "latlng_to_cell", None) or h3.geo_to_h3
disk = getattr(h3, "grid_disk", None) or h3.k_ring


# ----------------------------------------------------------------------
# 1) โหลดข้อมูล
# ----------------------------------------------------------------------
def confidence_to_num(s):
    # MODIS เป็นตัวเลข 0-100 อยู่แล้ว ส่วน VIIRS เป็นตัวอักษร l / n / h
    # แปลงทีละไฟล์ก่อนรวม ไม่ให้ตัวเลขกับตัวอักษรปนกันในคอลัมน์เดียว
    num = pd.to_numeric(s, errors="coerce")
    letters = s.astype(str).str.lower().map({"l": 30, "n": 60, "h": 90})
    return num.fillna(letters).fillna(0)


def load_hotspots(folder):
    paths = sorted(glob.glob(f"{folder}/**/*.csv", recursive=True))
    print("พบไฟล์:", len(paths))
    keep = ["latitude", "longitude", "acq_date", "acq_time", "confidence", "frp", "daynight"]
    frames = []
    for p in paths:
        name = os.path.basename(p)
        tag = os.path.basename(os.path.dirname(p)).split("_")[2]   # DL_FIRE_M-C61_814331 -> M-C61
        d = pd.read_csv(p)
        n0 = len(d)
        if "type" in d.columns:
            d = d[d["type"] == 0]    # เอาเฉพาะไฟบนพืชพรรณ (ตัดภูเขาไฟ แหล่งความร้อนคงที่ จุดในทะเล)
        d = d[[c for c in keep if c in d.columns]].copy()
        d["confidence_num"] = confidence_to_num(d["confidence"])
        d["sensor"] = SENSOR_BY_TAG.get(tag, tag)
        d["product"] = "nrt" if "nrt" in name else "archive"
        print(f"  {name}: {n0} แถว -> ใช้ {len(d)} แถว")
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["acq_date"] = pd.to_datetime(df["acq_date"])
    before = len(df)
    df = df.drop_duplicates(subset=["latitude", "longitude", "acq_date", "acq_time", "sensor"])
    print(f"ตัดแถวซ้ำ {before - len(df)} แถว เหลือรวม {len(df)} แถว")
    return df


def pick_series(raw, cfg):
    # เลือกเฉพาะเซนเซอร์ของ series นี้ ช่วงฤดูไฟ confidence พอ และช่วงปีที่กำหนด
    y0, y1 = cfg["years"]
    m0, m1 = SEASON_MONTHS
    d = raw[raw["sensor"] == cfg["sensor"]]
    ok = d["acq_date"].dt.month.between(m0, m1) & (d["confidence_num"] >= MIN_CONF)
    d = d[ok & d["acq_date"].dt.year.between(y0, y1)].copy()
    d["season"] = d["acq_date"].dt.year   # ฤดูไฟอยู่ในปีปฏิทินเดียว (ม.ค.-พ.ค.) ใช้ปีตรงๆ ได้
    return d


# ----------------------------------------------------------------------
# 2) จักรวาลของ cell และตารางนับ hotspot
# ----------------------------------------------------------------------
def make_universe(res, hot_cells):
    # จักรวาล = ทุก cell ที่ครอบคลุมกรอบพื้นที่ (กำหนดจากภูมิศาสตร์ ไม่เกี่ยวกับว่า cell ไหนเคยไหม้)
    # ถ้าใช้เฉพาะ cell ที่เคยไหม้จะรั่วข้อมูลอนาคต เพราะใช้ label มากำหนดว่าใครอยู่ในกลุ่ม
    # เพิ่ม cell ของ hotspot เข้าไปด้วยกันเศษ cell ริมกรอบที่จุดสุ่มพลาด
    w, s, e, n = BBOX
    cells = {to_cell(la, lo, res)
             for la in np.arange(s, n, STEP) for lo in np.arange(w, e, STEP)}
    cells.update(hot_cells)
    return sorted(cells)


def season_counts(d, uidx, years):
    # คืน dict ปี -> เวกเตอร์จำนวน hotspot ต่อ cell (ยาวเท่าจักรวาล) ปีที่ไม่มี hotspot เลยเป็นเวกเตอร์ศูนย์
    out = {}
    groups = dict(list(d.groupby("season")))
    for y in years:
        vec = np.zeros(len(uidx))
        if y in groups:
            c = groups[y]["cell"].value_counts()
            vec[uidx.get_indexer(c.index)] = c.values
        out[y] = vec
    return out


def smooth_matrix(universe, k):
    # เมทริกซ์ W: W @ จำนวน hotspot = ค่าเฉลี่ยจำนวน hotspot ในวงเพื่อนบ้าน k ของแต่ละ cell (รวม cell ตัวเอง)
    # หารด้วยขนาดวงทั้งหมด (รวม cell นอกกรอบที่ถือว่าเป็นศูนย์) เพื่อไม่ให้ cell ริมกรอบได้เปรียบ
    pos = {c: i for i, c in enumerate(universe)}
    rows, cols, vals = [], [], []
    for i, c in enumerate(universe):
        ring = disk(c, k)
        for nb in ring:
            j = pos.get(nb)
            if j is not None:
                rows.append(i)
                cols.append(j)
                vals.append(1.0 / len(ring))
    n = len(universe)
    return sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))


# ----------------------------------------------------------------------
# 3) ตัววัดผล
# ----------------------------------------------------------------------
def top_stats(score, y, frac=TOP):
    """precision และ recall ของ cell อันดับต้น frac ส่วน
    cell ที่คะแนนเท่ากันตรงขอบตัดจะคิดเป็นค่าคาดหวัง (ไม่สุ่ม ผลซ้ำได้เสมอ)"""
    n = len(score)
    K = max(1, int(round(frac * n)))
    order = np.argsort(-score, kind="stable")
    s = score[order]
    yy = y[order]
    thr = s[K - 1]
    above = s > thr
    tied = s == thr
    m_above = above.sum()
    tp_above = yy[above].sum()
    t = tied.sum()
    pos_tied = yy[tied].sum()
    r = K - m_above                    # จำนวนที่ต้องหยิบจากกลุ่มคะแนนเท่ากัน
    tp = tp_above + r * pos_tied / t   # จำนวน cell ไหม้ที่คาดว่าจะจับได้
    return tp / K, tp / y.sum()


def recurrence_table(counts, years):
    # เทียบทุกคู่ปีติดกัน (ปี a -> ปี b)
    rows = []
    for a, b in zip(years[:-1], years[1:]):
        prev, cur = counts[a] > 0, counts[b] > 0
        if prev.sum() == 0 or cur.sum() == 0 or prev.all():
            continue
        p1 = cur[prev].mean()       # โอกาสไหม้ปี b เมื่อปี a ไหม้
        p0 = cur[~prev].mean()      # โอกาสไหม้ปี b เมื่อปี a ไม่ไหม้
        area = prev.mean()          # สัดส่วนพื้นที่ที่ไหม้ปี a (ค่าคาดหวังถ้าไม่มีการเผาซ้ำ)
        share = counts[b][prev].sum() / counts[b].sum()   # สัดส่วน hotspot ปี b ที่ตกใน cell ที่ไหม้ปี a
        rows.append({"season": b, "prev_area_share": area, "p_if_prev": p1, "p_if_not": p0,
                     "rel_risk": p1 / p0 if p0 > 0 else np.nan,
                     "hot_share_in_prev": share, "hot_lift": share / area})
    return pd.DataFrame(rows)


def concentration_table(counts, years):
    # ต้องใช้พื้นที่กี่ % ของจักรวาลถึงครอบคลุม hotspot 50% และ 80% ของแต่ละฤดู (ยิ่งน้อยยิ่งกระจุก)
    rows = []
    n = len(next(iter(counts.values())))
    for y in years:
        c = np.sort(counts[y])[::-1]
        tot = c.sum()
        if tot == 0:
            continue
        cum = np.cumsum(c) / tot
        rows.append({"season": y, "hotspots": int(tot), "burned_area_share": (c > 0).mean(),
                     "area_for_50pct": (np.searchsorted(cum, 0.5) + 1) / n,
                     "area_for_80pct": (np.searchsorted(cum, 0.8) + 1) / n})
    return pd.DataFrame(rows)


def cumulative_summary(counts, years):
    # ภาพรวมทุกฤดู: cell ส่วนน้อยรับผิดชอบ hotspot ส่วนใหญ่ไหม (H1)
    burned = np.array([counts[y] > 0 for y in years])    # แถว = ฤดู, คอลัมน์ = cell
    times = burned.sum(axis=0)                           # จำนวนฤดูที่แต่ละ cell ไหม้
    total = np.sum([counts[y] for y in years], axis=0)
    order = np.sort(total)[::-1]
    n = len(total)
    top10 = order[: max(1, int(round(TOP * n)))].sum() / order.sum()
    return {"never_burned": (times == 0).mean(),
            "burned_every_season": (times == len(years)).mean(),
            "burned_half_or_more": (times >= len(years) / 2).mean(),
            "hot_share_in_top10pct_cells": top10}


def eval_baselines(counts, years, W):
    # rolling-origin: ทำนายฤดู Y ด้วยข้อมูลก่อน Y เท่านั้น เริ่มทดสอบหลังมีประวัติครบ N_HIST ฤดู
    burned = {y: counts[y] > 0 for y in years}
    rows = []
    for i, Y in enumerate(years):
        if i < N_HIST:
            continue
        y = burned[Y].astype(int)
        P = y.sum()
        if P == 0 or P == len(y):
            continue
        prev = years[i - 1]
        scores = {
            "B0_last_year": burned[prev].astype(float),
            "B1_last3": sum(burned[t] for t in years[i - N_HIST:i]).astype(float),
            "B1_all": sum(burned[t] for t in years[:i]).astype(float),
            "B2_smooth": W @ counts[prev],
        }
        p = P / len(y)
        for name, sc in scores.items():
            prec, rec = top_stats(sc, y)
            rows.append({"season": Y, "baseline": name, "prevalence": p,
                         "roc_auc": roc_auc_score(y, sc), "pr_auc": average_precision_score(y, sc),
                         "prec_top10": prec, "recall_top10": rec, "lift_top10": prec / p,
                         "lift_max": min(1 / p, 1 / TOP)})
    return pd.DataFrame(rows)


def decision_gate(bl):
    # เลือก baseline ที่ lift เฉลี่ยสูงสุด แล้วดูว่าผ่านเกณฑ์ในกี่ % ของฤดู
    best = bl.groupby("baseline")["lift_top10"].mean().idxmax()
    b = bl[bl["baseline"] == best].copy()
    b["threshold"] = np.minimum(GATE_LIFT, GATE_CAP / b["prevalence"])
    b["pass_season"] = b["lift_top10"] >= b["threshold"]
    share = b["pass_season"].mean()
    return best, b, share, share >= GATE_SEASONS


# ----------------------------------------------------------------------
# 4) รันทั้งหมด
# ----------------------------------------------------------------------
def run_series(name, cfg, raw, out_dir):
    d = pick_series(raw, cfg)
    years = list(range(cfg["years"][0], cfg["years"][1] + 1))
    print(f"\n######## {name}: {cfg['sensor']} ฤดู {years[0]}-{years[-1]} ({len(d)} hotspot หลังกรอง) ########")
    print(d.groupby("season").size().to_string())
    summary = []
    for res in cfg["res"]:
        print(f"\n=== {name} H3 resolution {res} ===")
        d2 = d.assign(cell=[to_cell(la, lo, res) for la, lo in zip(d["latitude"], d["longitude"])])
        universe = make_universe(res, d2["cell"].unique())
        uidx = pd.Index(universe)
        counts = season_counts(d2, uidx, years)
        W = smooth_matrix(universe, K_SMOOTH)
        print("จำนวน cell ในจักรวาล:", len(universe))

        rec = recurrence_table(counts, years)
        con = concentration_table(counts, years)
        cum = cumulative_summary(counts, years)
        bl = eval_baselines(counts, years, W)
        best, b, share, ok = decision_gate(bl)

        print("\n-- recurrence รายฤดู (rel_risk > 1 = ไหม้ซ้ำมากกว่าสุ่ม) --")
        print(rec.round(3).to_string(index=False))
        print("\n-- ความกระจุกตัวรายฤดู --")
        print(con.round(3).to_string(index=False))
        print("\n-- ภาพรวมทุกฤดู --")
        print({k: round(float(v), 3) for k, v in cum.items()})
        print("\n-- baseline เฉลี่ยทุกฤดูทดสอบ --")
        print(bl.groupby("baseline")[["roc_auc", "pr_auc", "prec_top10", "recall_top10", "lift_top10"]]
              .mean().round(3).to_string())
        print(f"\n-- decision gate (baseline ที่ดีที่สุด: {best}) --")
        print(b[["season", "prevalence", "lift_top10", "lift_max", "threshold", "pass_season"]]
              .round(3).to_string(index=False))
        print(f"ผ่านเกณฑ์ {share:.0%} ของฤดูทดสอบ (ต้อง >= {GATE_SEASONS:.0%}) -> {'PASS' if ok else 'FAIL'}")

        tag = f"{name}_res{res}"
        rec.to_csv(f"{out_dir}/{tag}_recurrence.csv", index=False)
        con.to_csv(f"{out_dir}/{tag}_concentration.csv", index=False)
        bl.to_csv(f"{out_dir}/{tag}_baselines.csv", index=False)
        summary.append({"series": name, "res": res, "cells": len(universe), "test_seasons": b["season"].nunique(),
                        "median_rel_risk": rec["rel_risk"].median(), "median_prevalence": b["prevalence"].median(),
                        "best_baseline": best, "median_lift_top10": b["lift_top10"].median(),
                        "median_roc_auc": bl[bl["baseline"] == best]["roc_auc"].median(),
                        "seasons_passed": share, "gate": "PASS" if ok else "FAIL"})
    return summary


def run_all(firms_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    raw = load_hotspots(firms_dir)
    rows = []
    for name, cfg in SERIES.items():
        rows += run_series(name, cfg, raw, out_dir)
    out = pd.DataFrame(rows)
    out.to_csv(f"{out_dir}/stage0_summary.csv", index=False)
    print("\n================ สรุป Stage 0 ================")
    print(out.round(3).to_string(index=False))
    return out
