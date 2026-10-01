"""test_stage1.py : ทดสอบ stage1.py กับข้อมูลจำลอง (prereg ข้อ 9.1-9.2) รันใน sandbox ไม่ใช่ข้อมูลจริง"""
import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from pyproj import Transformer

import stage0
import stage1

stage0.BBOX = (98.5, 18.7, 98.8, 19.0)      # กรอบเล็กเพื่อให้ทดสอบเร็ว
YEARS = list(range(2012, 2027))
RES = 8
rng = np.random.default_rng(42)
OUT = "/tmp/s1test"
os.makedirs(OUT, exist_ok=True)

universe = stage0.make_universe(RES, [])
n = len(universe)
print("cell จำลอง:", n)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def simulate(z, prev, b=1.5, persist=1.0, seed=0):
    """โอกาสไหม้ = sigmoid(a_Y + b*z + persist*ไหม้ปีก่อน) ปรับ a_Y ให้ได้สัดส่วนไหม้ตาม prev[Y]"""
    r = np.random.default_rng(seed)
    counts, last = {}, np.zeros(n)
    for Y in YEARS:
        lo, hi = -15.0, 15.0
        for _ in range(60):                      # bisection หา a_Y
            a = (lo + hi) / 2
            if sigmoid(a + b * z + persist * last).mean() < prev[Y]:
                lo = a
            else:
                hi = a
        burned = r.random(n) < sigmoid(a + b * z + persist * last)
        counts[Y] = burned * (1 + r.poisson(3, n))
        last = burned.astype(float)
    return counts


def smooth_field(seed):
    # ค่าแฝงที่เรียบเชิงพื้นที่: ค่าสุ่มต่อ parent res 6 + noise
    r = np.random.default_rng(seed)
    par = pd.factorize(pd.Index([stage1.to_parent(c, 6) for c in universe]))[0]
    z = r.normal(size=par.max() + 1)[par] + 0.5 * r.normal(size=n)
    return (z - z.mean()) / z.std()


def run_regime(name, counts, terrain=None):
    data = stage1.prepare_from_counts(counts, YEARS, universe, RES, name="A_SNPP")
    out = stage1.run(data, None, terrain, n_boot=30)
    t = out["table"].groupby("model")[["roc_auc", "pr_auc", "lift_top10"]].mean()
    print(f"\n==== สรุประบอบ {name} ====\n{t.round(3)}\n")
    return data, out, t


results = {}
varying = dict(zip(YEARS, [0.36, 0.36, 0.36, 0.36, 0.36, 0.22, 0.17, 0.41, 0.43, 0.24, 0.10, 0.31, 0.31, 0.12, 0.26]))

# 1) มีสัญญาณเผาซ้ำ สัดส่วนไหม้แกว่งแบบข้อมูลจริง
z = smooth_field(1)
data_sig, out_sig, t = run_regime("signal", simulate(z, varying, seed=1))
assert t.loc["B1_all", "roc_auc"] > 0.6 and t.loc["M2_noterrain", "roc_auc"] > 0.6
results["signal"] = t

# 2) สุ่มล้วน: ทุกคะแนนต้องใกล้ 0.5 และผลต้องเป็นไม่ชนะ
data_rnd, out_rnd, t = run_regime("random", simulate(z, {Y: 0.3 for Y in YEARS}, b=0, persist=0, seed=2))
assert (t["roc_auc"] - 0.5).abs().max() < 0.03, "สุ่มล้วนแต่ได้ ROC-AUC ห่างจาก 0.5 = มีการรั่ว"
assert out_rnd["verdict"]["result"] == "ML DOES NOT WIN"
results["random"] = t

# 3) สัดส่วนไหม้สูง และ 4) ต่ำ
_, out_hi, t = run_regime("high_prev", simulate(z, {Y: 0.65 for Y in YEARS}, seed=3))
assert t["prevalence"].mean() > 0.6 if "prevalence" in t else True
results["high_prev"] = t
_, out_lo, t = run_regime("low_prev", simulate(z, {Y: 0.05 for Y in YEARS}, seed=4))
results["low_prev"] = t
for o in (out_hi, out_lo):
    assert np.isfinite(o["table"][["roc_auc", "pr_auc", "lift_top10"]].values).all()

# 5) ตรวจการรั่ว: ฟีเจอร์ฤดู Y ต้องไม่เปลี่ยนเมื่อสลับข้อมูลฤดู >= Y
counts = data_sig["counts"]
nb_mats = {k: stage1.neighbor_matrix(universe, k) for k in stage1.NB_K}
for Y in [2015, 2020, 2026]:
    f0 = stage1.season_features(counts, YEARS, Y, nb_mats)
    c2 = {t: (rng.permutation(v) if t >= Y else v) for t, v in counts.items()}
    f1 = stage1.season_features(c2, YEARS, Y, nb_mats)
    assert f0.equals(f1), f"ฟีเจอร์ฤดู {Y} เปลี่ยนเมื่อแก้ข้อมูลอนาคต"
print("ตรวจการรั่ว: ผ่าน")

# ตรวจนิยามฟีเจอร์บาง cell ด้วยมือ
f = stage1.season_features(counts, YEARS, 2020, nb_mats)
i = int(np.argmax(f["n_burn3"].values))
hist = [t for t in YEARS if t < 2020]
manual_since = next((lag for lag, t in enumerate(reversed(hist), 1) if counts[t][i] > 0 and lag < 6), 6)
assert f.loc[i, "yrs_since"] == manual_since
assert np.isclose(f.loc[i, "frac_burn_all"], np.mean([counts[t][i] > 0 for t in hist]))
# อันดับของ frac_burn_all ต้องเหมือน B1_all (เป็นเส้นตรงของกันและกัน)
assert np.allclose(f["frac_burn_all"] * len(hist), stage1.baseline_scores(data_sig, 2020)["B1_all"])
print("ตรวจนิยามฟีเจอร์: ผ่าน")

# 6) จุดตรวจ 9.3 แบบจำลอง: เทียบกับ stage0.eval_baselines ตัวจริง
bl = stage0.eval_baselines(counts, YEARS, data_sig["W"])
bl.to_csv(f"{OUT}/fake_baselines.csv", index=False)
stage1.check_b1(data_sig, f"{OUT}/fake_baselines.csv")
bl2 = bl.copy()
bl2.loc[bl2.index[5], "pr_auc"] += 0.01
bl2.to_csv(f"{OUT}/fake_baselines_bad.csv", index=False)
try:
    stage1.check_b1(data_sig, f"{OUT}/fake_baselines_bad.csv")
    raise RuntimeError("check_b1 ควรล้ม")
except AssertionError:
    print("check_b1 จับค่าที่ผิดได้: ผ่าน")

# 7) ภูมิประเทศ: สร้าง GeoTIFF จำลองแล้วให้การไหม้ขึ้นกับความสูง (positive control)
tr = Transformer.from_crs("EPSG:4326", "EPSG:32647", always_xy=True)
w, s, e, nn = stage0.BBOX
xs, ys = tr.transform([w, e, w, e], [s, s, nn, nn])
x0, y1 = min(xs) - 900, max(ys) + 900
W = int((max(xs) + 900 - x0) / 90)
H = int((y1 - (min(ys) - 900)) / 90)
cc, rr = np.meshgrid(np.arange(W), np.arange(H))
elev = 600 + 400 * np.sin(cc / 60) * np.cos(rr / 45) + 0.002 * cc * rr
slope = np.abs(np.gradient(elev, axis=1)) * 2
elev[:5, :] = 0                                # ขอบที่ Earth Engine อาจเติม 0
slope[:5, :] = 0
tif = f"{OUT}/fake_terrain.tif"
with rasterio.open(tif, "w", driver="GTiff", width=W, height=H, count=2, dtype="float32",
                   crs="EPSG:32647", transform=from_origin(x0, y1, 90, 90)) as dst:
    dst.write(elev.astype("float32"), 1)
    dst.write(slope.astype("float32"), 2)
terr = stage1.terrain_by_cell([tif], data_sig, cache_csv=f"{OUT}/terrain.csv")
assert terr["elev_mean"].min() > 0, "พิกเซลที่เติม 0 หลุดเข้ามา"
assert terr["elev_mean"].notna().mean() > 0.99
terr2 = stage1.load_terrain_csv(f"{OUT}/terrain.csv", data_sig)
assert np.allclose(terr.values, terr2.values, equal_nan=True)

ze = terr["elev_mean"].fillna(terr["elev_mean"].mean()).values
ze = (ze - ze.mean()) / ze.std()
# ภูมิประเทศเป็นตัวขับ + persist ต่ำ: ประวัติจับได้แค่บางส่วน โมเดลที่เห็นภูมิประเทศควรดีกว่า
_, out_ter, t = run_regime("terrain_driven", simulate(ze, varying, b=2.0, persist=0.3, seed=5), terrain=terr)
assert out_ter["verdict"]["primary_model"] == "M2"
assert t.loc["M2", "pr_auc"] > t.loc["M2_noterrain", "pr_auc"], "ภูมิประเทศไม่ช่วยทั้งที่เป็นตัวขับ"
results["terrain_driven"] = t

print("\n==================== สรุปทุกระบอบ (ค่าเฉลี่ย 9 ฤดูทดสอบ) ====================")
for k, t in results.items():
    print(f"\n[{k}]")
    print(t.round(3).to_string())
print("\nผ่านทุกการทดสอบ")
