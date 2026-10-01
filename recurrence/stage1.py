"""
stage1.py : Stage 1 ทดสอบว่า ML จัดอันดับ cell ที่จะไหม้ได้ดีกว่า B1_all หรือไม่
ทุกค่าในไฟล์นี้มาจาก stage1_prereg.md ที่ล็อกเมื่อ 1 ต.ค. 2026 ห้ามแก้หลังเห็นผล
(แก้บั๊กได้ แต่ต้องบันทึกในตารางท้าย prereg)

ลำดับการใช้งาน (ดู colab_stage1.py)
    raw  = stage0.load_hotspots(FIRMS_DIR)
    data = stage1.prepare(raw, "A_SNPP", 8)
    stage1.check_b1(data, ".../A_SNPP_res8_baselines.csv")   # จุดตรวจ 9.3 ต้องผ่านก่อน
    terr = stage1.terrain_by_cell(TIF_PATHS, data)
    out  = stage1.run(data, OUT_DIR, terr)
"""
import time

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import h3
import stage0

# ======================================================================
# ค่าที่ล็อก (prereg ข้อ 4-8)
# ======================================================================
N_HIST = 3            # ต้องมีประวัติ 3 ฤดูก่อนติดป้ายได้ (Series A: ป้ายเริ่มฤดู 2015)
MIN_TRAIN = 3         # ฤดูเทรนขั้นต่ำ (Series A: ทดสอบเริ่มฤดู 2018)
YRS_CAP = 6           # yrs_since: ไม่เคยไหม้ หรือไหม้ล่าสุดเกิน 5 ฤดู = 6
NB_K = (1, 2, 3)      # วงเพื่อนบ้าน
BOOT_RES = 5          # บล็อกสำหรับ spatial bootstrap
N_BOOT = 1000
SEED = 0

PRIMARY = ("A_SNPP", 8)    # ชุดเดียวที่ใช้ตัดสิน
N_TEST_PRIMARY = 9         # ฤดูทดสอบ 2018-2026
WIN_SEASONS = 7            # เกณฑ์ข้อ 1: ชนะ >= 7 จาก 9 ฤดู
WIN_MEAN_DIFF = 0.02       # เกณฑ์ข้อ 2: ผลต่าง PR-AUC เฉลี่ย >= +0.02

HIST_FEATS = ["burn_y1", "burn_y2", "burn_y3", "cnt_y1", "cnt_y2", "cnt_y3",
              "n_burn3", "frac_burn5", "frac_burn_all", "yrs_since"]
NB_FEATS = [f"nb{k}_{v}" for k in NB_K for v in ("burn_y1", "frac_all")]
TERRAIN_FEATS = ["elev_mean", "elev_std", "slope_mean"]

# M2 (โมเดลหลัก) hyperparameter คงที่ ไม่จูน
# class_weight ไม่ใส่ = None ตาม prereg (ไม่ถ่วงคลาส)
HGB_PARAMS = dict(learning_rate=0.05, max_iter=200, max_leaf_nodes=15, min_samples_leaf=100,
                  l2_regularization=1.0, early_stopping=False, random_state=0)

# h3 v4 / v3
to_parent = getattr(h3, "cell_to_parent", None) or h3.h3_to_parent


# ----------------------------------------------------------------------
# 1) เตรียมข้อมูล
# ----------------------------------------------------------------------
def neighbor_matrix(universe, k):
    """W @ x = ค่าเฉลี่ย x ของเพื่อนบ้านใน disk วง k ไม่รวม cell ตัวเอง
    หารด้วยจำนวนเพื่อนบ้านทั้งหมด (cell นอกกรอบนับเป็นศูนย์) แบบเดียวกับ stage0.smooth_matrix"""
    pos = {c: i for i, c in enumerate(universe)}
    rows, cols, vals = [], [], []
    for i, c in enumerate(universe):
        ring = [nb for nb in stage0.disk(c, k) if nb != c]
        w = 1.0 / len(ring)
        for nb in ring:
            j = pos.get(nb)
            if j is not None:
                rows.append(i)
                cols.append(j)
                vals.append(w)
    n = len(universe)
    return sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))


def season_features(counts, years, Y, nb_mats):
    """ฟีเจอร์ของฤดู Y ใช้เฉพาะ counts ของฤดูก่อน Y (prereg ข้อ 4)"""
    hist = [t for t in years if t < Y]
    if len(hist) < N_HIST:
        raise ValueError(f"ฤดู {Y} มีประวัติไม่ถึง {N_HIST} ฤดู")
    n = len(counts[hist[0]])
    burn = {t: (counts[t] > 0).astype(float) for t in hist}
    f = {}
    for lag in (1, 2, 3):
        t = hist[-lag]
        f[f"burn_y{lag}"] = burn[t]
        f[f"cnt_y{lag}"] = np.log1p(counts[t])
    f["n_burn3"] = sum(burn[t] for t in hist[-3:])
    last5 = hist[-5:]
    f["frac_burn5"] = sum(burn[t] for t in last5) / len(last5)   # ฤดู 2015 หารด้วย 3, 2016 หารด้วย 4
    f["frac_burn_all"] = sum(burn[t] for t in hist) / len(hist)   # ใช้สัดส่วนกันสเกลเลื่อนตามปี
    ys = np.full(n, float(YRS_CAP))
    for lag, t in enumerate(reversed(hist), start=1):
        if lag >= YRS_CAP:
            break
        ys[(burn[t] > 0) & (ys == YRS_CAP)] = lag     # เก็บเฉพาะครั้งล่าสุดที่ไหม้
    f["yrs_since"] = ys
    for k in NB_K:
        f[f"nb{k}_burn_y1"] = nb_mats[k] @ f["burn_y1"]
        f[f"nb{k}_frac_all"] = nb_mats[k] @ f["frac_burn_all"]
    return pd.DataFrame(f)[HIST_FEATS + NB_FEATS]


def prepare_from_counts(counts, years, universe, res, name="custom"):
    """สร้างฟีเจอร์และป้ายทุกฤดูจาก counts (แยกออกมาเพื่อให้ทดสอบกับข้อมูลจำลองได้)"""
    t0 = time.time()
    nb_mats = {k: neighbor_matrix(universe, k) for k in NB_K}
    label_years = years[N_HIST:]
    feats = {Y: season_features(counts, years, Y, nb_mats) for Y in label_years}
    labels = {Y: (counts[Y] > 0).astype(int) for Y in label_years}
    blocks = pd.Index([to_parent(c, BOOT_RES) for c in universe])
    data = {"name": name, "res": res, "universe": universe, "years": years, "counts": counts,
            "label_years": label_years, "test_years": label_years[MIN_TRAIN:],
            "feats": feats, "labels": labels,
            "W": stage0.smooth_matrix(universe, stage0.K_SMOOTH),
            "block_idx": pd.factorize(blocks)[0]}
    print(f"[{name} res {res}] cell {len(universe)}, ป้ายฤดู {label_years[0]}-{label_years[-1]}, "
          f"ทดสอบ {data['test_years'][0]}-{data['test_years'][-1]} ({len(data['test_years'])} ฤดู) "
          f"ใช้เวลา {time.time() - t0:.0f} วินาที")
    return data


def prepare(raw, series, res):
    """ใช้ฟังก์ชันของ stage0 ทั้งหมด ป้าย จักรวาล และ counts จึงตรงกับ Stage 0 ทุกประการ"""
    cfg = stage0.SERIES[series]
    d = stage0.pick_series(raw, cfg)
    years = list(range(cfg["years"][0], cfg["years"][1] + 1))
    d = d.assign(cell=[stage0.to_cell(la, lo, res) for la, lo in zip(d["latitude"], d["longitude"])])
    universe = stage0.make_universe(res, d["cell"].unique())
    counts = stage0.season_counts(d, pd.Index(universe), years)
    return prepare_from_counts(counts, years, universe, res, name=series)


# ----------------------------------------------------------------------
# 2) ภูมิประเทศ
# ----------------------------------------------------------------------
def terrain_by_cell(tif_paths, data, cache_csv=None):
    """เฉลี่ยพิกเซล 90 ม. ของ GeoTIFF (แถบ 1 = elevation, 2 = slope) ตาม cell ที่จุดกลางพิกเซลตกอยู่
    คืน DataFrame เรียงตามจักรวาล cell ที่ไม่มีพิกเซลเป็น NaN"""
    import rasterio
    from pyproj import Transformer

    tr = Transformer.from_crs("EPSG:32647", "EPSG:4326", always_xy=True)
    parts = []
    for p in tif_paths:
        with rasterio.open(p) as src:
            if src.crs is None or src.crs.to_epsg() != 32647:
                raise ValueError(f"{p}: crs ไม่ใช่ EPSG:32647")
            elev = src.read(1, masked=True)
            slope = src.read(2, masked=True)
            ok = ~np.ma.getmaskarray(elev) & ~np.ma.getmaskarray(slope)
            # กันพิกเซลนอก region ที่ Earth Engine อาจเติมเป็น 0 แทน nodata
            # (ความสูงจริงในกรอบนี้สูงกว่า 0 ม. ทั้งหมด)
            ok &= np.ma.filled(elev, 0) > 0
            r, c = np.nonzero(ok)
            T = src.transform
            x = T.c + (c + 0.5) * T.a + (r + 0.5) * T.b    # จุดกลางพิกเซล
            y = T.f + (c + 0.5) * T.d + (r + 0.5) * T.e
            lon, lat = tr.transform(x, y)
            parts.append(pd.DataFrame({"lat": lat, "lon": lon,
                                       "elev": elev.data[r, c].astype(float),
                                       "slope": slope.data[r, c].astype(float)}))
            print(f"  {p}: ใช้ {len(r)} พิกเซล")
    px = pd.concat(parts, ignore_index=True)
    res = data["res"]
    px["cell"] = [stage0.to_cell(la, lo, res) for la, lo in zip(px["lat"].values, px["lon"].values)]
    g = px.groupby("cell")
    t = pd.DataFrame({"elev_mean": g["elev"].mean(), "elev_std": g["elev"].std(ddof=0),
                      "slope_mean": g["slope"].mean()})
    t = t.reindex(data["universe"])
    print(f"terrain res {res}: cell ที่ไม่มีพิกเซล {t['elev_mean'].isna().mean():.2%}")
    if cache_csv:
        t.to_csv(cache_csv, index_label="cell")
    return t.reset_index(drop=True)


def load_terrain_csv(path, data):
    t = pd.read_csv(path, index_col="cell").reindex(data["universe"])
    return t[TERRAIN_FEATS].reset_index(drop=True)


# ----------------------------------------------------------------------
# 3) baseline และตัววัด (นิยามเดียวกับ stage0.eval_baselines)
# ----------------------------------------------------------------------
def baseline_scores(data, T):
    years, counts = data["years"], data["counts"]
    i = years.index(T)
    burned = {t: counts[t] > 0 for t in years[:i]}
    return {"B1_all": sum(burned[t] for t in years[:i]).astype(float),
            "B1_last3": sum(burned[t] for t in years[i - stage0.N_HIST:i]).astype(float),
            "B2_smooth": data["W"] @ counts[years[i - 1]]}


def metrics(y, s):
    p = y.mean()
    prec, rec = stage0.top_stats(s, y)
    return {"prevalence": p, "roc_auc": roc_auc_score(y, s), "pr_auc": average_precision_score(y, s),
            "lift_top10": prec / p, "recall_top10": rec}


def check_b1(data, stage0_csv, tol=1e-9):
    """จุดตรวจ 9.3: baseline ที่โค้ดนี้คำนวณต้องตรงกับไฟล์ Stage 0 ทุกฤดู ไม่เกี่ยวกับ ML"""
    ref = pd.read_csv(stage0_csv).set_index(["season", "baseline"])
    rows = []
    for T in data["label_years"]:
        y = data["labels"][T]
        for name, s in baseline_scores(data, T).items():
            if (T, name) not in ref.index:
                continue
            m = metrics(y, s)
            for col in ["prevalence", "roc_auc", "pr_auc", "lift_top10", "recall_top10"]:
                rows.append({"season": T, "baseline": name, "metric": col,
                             "stage1": m[col], "stage0": ref.loc[(T, name), col]})
    df = pd.DataFrame(rows)
    df["diff"] = (df["stage1"] - df["stage0"]).abs()
    if df.empty:
        raise AssertionError("ไม่พบฤดูที่ตรงกับไฟล์ Stage 0 ตรวจ path และชื่อ series")
    worst = df["diff"].max()
    print(f"จุดตรวจ 9.3: เทียบ {len(df)} ค่า ({df['season'].nunique()} ฤดู) ผลต่างมากสุด {worst:.2e}")
    if worst > tol:
        print(df.sort_values("diff", ascending=False).head(10).to_string(index=False))
        raise AssertionError("ไม่ตรงกับ Stage 0 ห้ามรัน ML จนกว่าจะหาสาเหตุเจอ")
    print("ผ่าน: baseline ตรงกับ Stage 0")
    return df


# ----------------------------------------------------------------------
# 4) โมเดล
# ----------------------------------------------------------------------
def model_specs(has_terrain):
    """คืน dict ชื่อ -> (ฟีเจอร์, ฟังก์ชันสร้างโมเดล) และชื่อโมเดลหลัก
    ไม่มีภูมิประเทศ = แผนสำรองใน prereg ข้อ 10 (M2 ไม่มีภูมิประเทศเป็นโมเดลหลัก)"""
    base = HIST_FEATS + NB_FEATS
    full = base + TERRAIN_FEATS if has_terrain else base
    hgb = lambda: HistGradientBoostingClassifier(**HGB_PARAMS)
    lr = lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                               LogisticRegression(C=1.0, max_iter=1000))
    specs = {"M2_noterrain": (base, hgb), "M1": (full, lr)}
    if has_terrain:
        specs["M2"] = (full, hgb)
        return specs, "M2"
    return specs, "M2_noterrain"


def frame(data, Y, terrain):
    X = data["feats"][Y]
    if terrain is not None:
        X = pd.concat([X, terrain[TERRAIN_FEATS]], axis=1)
    return X


def fit_predict(data, terrain, specs):
    """rolling-origin: ฤดูทดสอบ T เทรนด้วยแถวฤดู label_years[0] ถึง T-1 เท่านั้น"""
    scores = {}
    for T in data["test_years"]:
        t0 = time.time()
        train_years = [Y for Y in data["label_years"] if Y < T]
        Xtr = pd.concat([frame(data, Y, terrain) for Y in train_years], ignore_index=True)
        ytr = np.concatenate([data["labels"][Y] for Y in train_years])
        Xte = frame(data, T, terrain)
        scores[T] = baseline_scores(data, T)
        for name, (cols, make) in specs.items():
            m = make().fit(Xtr[cols], ytr)
            scores[T][name] = m.predict_proba(Xte[cols])[:, 1]
        print(f"  ฤดูทดสอบ {T}: เทรน {train_years[0]}-{train_years[-1]} ({len(ytr)} แถว) "
              f"{time.time() - t0:.0f} วินาที")
    return scores


# ----------------------------------------------------------------------
# 5) ตัดสินและ bootstrap
# ----------------------------------------------------------------------
def verdict(table, primary, n_expected=N_TEST_PRIMARY):
    m = table[table["model"] == primary].set_index("season")
    b = table[table["model"] == "B1_all"].set_index("season")
    diff = m["pr_auc"] - b["pr_auc"]
    wins = int((diff > 0).sum())                     # เท่ากันนับว่าไม่ชนะ
    c1 = wins >= WIN_SEASONS
    c2 = diff.mean() >= WIN_MEAN_DIFF
    c3 = m["lift_top10"].mean() >= b["lift_top10"].mean()
    ok_n = len(diff) == n_expected
    return {"primary_model": primary, "test_seasons": len(diff), "expected_seasons": n_expected,
            "wins": wins, "c1_wins_ge_7": c1,
            "mean_pr_diff": diff.mean(), "c2_mean_diff_ge_0.02": c2,
            "lift_primary": m["lift_top10"].mean(), "lift_B1_all": b["lift_top10"].mean(),
            "c3_lift_not_lower": c3,
            "result": ("ML WINS" if (c1 and c2 and c3) else "ML DOES NOT WIN") if ok_n
            else "INVALID (จำนวนฤดูไม่ตรง prereg)"}


def bootstrap_diff(data, scores, model, ref="B1_all", n_boot=N_BOOT, seed=SEED):
    """spatial block bootstrap (บล็อก H3 res 5 สุ่มแบบใส่คืน) ของค่าเฉลี่ยข้ามฤดูของ
    PR-AUC(model) - PR-AUC(ref) รายงานประกอบ ไม่ใช้ตัดสิน"""
    rng = np.random.default_rng(seed)
    bidx = data["block_idx"]
    nb = bidx.max() + 1
    out = np.empty(n_boot)
    for r in range(n_boot):
        w = np.bincount(rng.integers(0, nb, nb), minlength=nb)[bidx]   # น้ำหนักต่อ cell = จำนวนครั้งที่บล็อกถูกสุ่ม
        d = []
        for T in data["test_years"]:
            y = data["labels"][T]
            d.append(average_precision_score(y, scores[T][model], sample_weight=w)
                     - average_precision_score(y, scores[T][ref], sample_weight=w))
        out[r] = np.mean(d)
    return {"model": model, "ref": ref, "n_boot": n_boot, "n_blocks": int(nb),
            "ci_low": np.percentile(out, 2.5), "ci_high": np.percentile(out, 97.5)}


# ----------------------------------------------------------------------
# 6) รันทั้งหมด
# ----------------------------------------------------------------------
def run(data, out_dir=None, terrain=None, n_boot=N_BOOT):
    t0 = time.time()
    tag = f"{data['name']}_res{data['res']}"
    is_primary = (data["name"], data["res"]) == PRIMARY
    specs, primary = model_specs(terrain is not None)
    print(f"\n######## Stage 1 {tag} ({'ชุดหลัก' if is_primary else 'ชุดรอง ไม่ใช้ตัดสิน'}) "
          f"โมเดลหลัก: {primary} ########")
    if terrain is None:
        print("ไม่มีภูมิประเทศ: ใช้แผนสำรอง prereg ข้อ 10 (ใช้ได้เฉพาะเมื่อพ้นสิ้นวันที่ 4 ต.ค. 2026)")

    scores = fit_predict(data, terrain, specs)
    rows = []
    for T, sc in scores.items():
        y = data["labels"][T]
        for name, s in sc.items():
            rows.append({"season": T, "model": name, **metrics(y, s)})
    table = pd.DataFrame(rows)

    print("\n-- ค่าเฉลี่ยทุกฤดูทดสอบ --")
    print(table.groupby("model")[["roc_auc", "pr_auc", "lift_top10", "recall_top10"]]
          .mean().round(3).to_string())
    wide = table.pivot(index="season", columns="model", values="pr_auc")
    print("\n-- PR-AUC รายฤดู --")
    print(wide.round(3).to_string())

    v = verdict(table, primary, N_TEST_PRIMARY if is_primary else len(data["test_years"]))
    boots = [bootstrap_diff(data, scores, primary, n_boot=n_boot)]
    print("\n-- bootstrap 95% CI ของผลต่าง PR-AUC เฉลี่ย (ประกอบ ไม่ใช้ตัดสิน) --")
    print(pd.DataFrame(boots).round(4).to_string(index=False))

    print(f"\n-- เกณฑ์ prereg ข้อ 7 ({primary} vs B1_all) --")
    for k, val in v.items():
        print(f"  {k}: {round(val, 4) if isinstance(val, float) else val}")
    if not is_primary:
        print("  (ชุดรอง: ตัวเลขข้างบนรายงานเพื่อเทียบเท่านั้น ไม่ใช่ผลตัดสิน)")

    if out_dir:
        table.to_csv(f"{out_dir}/{tag}_stage1_seasons.csv", index=False)
        pd.DataFrame([{**v, "is_primary_series": is_primary}]).to_csv(
            f"{out_dir}/{tag}_stage1_verdict.csv", index=False)
        pd.DataFrame(boots).to_csv(f"{out_dir}/{tag}_stage1_bootstrap.csv", index=False)
        print(f"\nบันทึกผลที่ {out_dir}/{tag}_stage1_*.csv")
    print(f"ใช้เวลารวม {time.time() - t0:.0f} วินาที")
    return {"table": table, "verdict": v, "bootstrap": boots, "scores": scores}
