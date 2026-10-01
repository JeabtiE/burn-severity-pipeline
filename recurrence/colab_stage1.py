# colab_stage1.py : ลำดับการรัน Stage 1 ใน Colab (คัดลอกทีละ cell ที่คั่นด้วย # %%)
# ก่อน cell 5 ต้อง commit stage1_prereg.md เข้า GitHub แล้ว

# %% cell 1: เตรียมสภาพแวดล้อม (รันใหม่ทุกครั้งที่ runtime รีสตาร์ท)
from google.colab import drive
drive.mount("/content/drive")
!pip install h3 -q

import sys, glob, os, importlib
ROOT = "/content/drive/MyDrive/burn_severity_pipeline"
sys.path.append(f"{ROOT}/recurrence")
importlib.invalidate_caches()
import stage0, stage1
importlib.reload(stage0)
importlib.reload(stage1)

FIRMS_DIR = f"{ROOT}/data/firms"
STAGE0_DIR = f"{ROOT}/results/stage0"
OUT_DIR = f"{ROOT}/results/stage1"
os.makedirs(OUT_DIR, exist_ok=True)

# %% cell 2: โหลด hotspot ครั้งเดียว แล้วเตรียมชุดหลัก (A_SNPP res 8)
raw = stage0.load_hotspots(FIRMS_DIR)
data = stage1.prepare(raw, "A_SNPP", 8)

# %% cell 3: จุดตรวจ 9.3 (ต้องขึ้น "ผ่าน" ถ้าขึ้น AssertionError ให้หยุดแล้ววางผลกลับมา)
chk = stage1.check_b1(data, f"{STAGE0_DIR}/A_SNPP_res8_baselines.csv")

# %% cell 4: ภูมิประเทศ (หลัง terrain_export.py เสร็จแล้ว) บันทึกเป็น CSV ไว้ใช้ซ้ำ
TIF_PATHS = sorted(glob.glob("/content/drive/MyDrive/burn_severity_terrain/srtm_terrain_90m*.tif"))
print(TIF_PATHS)
terr8 = stage1.terrain_by_cell(TIF_PATHS, data, cache_csv=f"{OUT_DIR}/terrain_res8.csv")
# ถ้ารันใหม่ภายหลัง ใช้: terr8 = stage1.load_terrain_csv(f"{OUT_DIR}/terrain_res8.csv", data)

# %% cell 5: การวิเคราะห์หลัก (ผลตัดสินตาม prereg ข้อ 7)
out = stage1.run(data, OUT_DIR, terr8)

# %% cell 6: ชุดรอง (รายงานคู่กันเสมอ ไม่ใช้ตัดสิน)
for series, res in [("A_SNPP", 7), ("B_MODIS", 7)]:
    d = stage1.prepare(raw, series, res)
    stage1.check_b1(d, f"{STAGE0_DIR}/{series}_res{res}_baselines.csv")
    t = stage1.terrain_by_cell(TIF_PATHS, d, cache_csv=f"{OUT_DIR}/terrain_{series}_res{res}.csv")
    stage1.run(d, OUT_DIR, t)
