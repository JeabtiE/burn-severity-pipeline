# terrain_export.py : ส่งออกภูมิประเทศ SRTM สำหรับ Stage 1 (ตาม stage1_prereg.md ข้อ 4)
# รันใน Colab ทีละ cell (แต่ละ cell คั่นด้วย # %%)
# ผลลัพธ์: GeoTIFF 2 แถบ (elevation, slope) ความละเอียด 90 ม. EPSG:32647
# เส้นตายตาม prereg: ไฟล์ต้องอยู่ใน Drive ภายในสิ้นวันที่ 4 ต.ค. 2026

# %% cell 1: เชื่อม Earth Engine
import ee
ee.Authenticate()
ee.Initialize(project="burn-severity-pipeline")

# %% cell 2: เตรียมภาพและสั่ง export
BBOX = (98.0, 18.2, 99.5, 20.0)   # west, south, east, north ต้องตรงกับ stage0.BBOX

# geodesic=False ให้ขอบสี่เหลี่ยมเป็นเส้นตรงในพิกัด lon/lat เหมือนกรอบที่ขอจาก FIRMS
region = ee.Geometry.Rectangle(list(BBOX), proj="EPSG:4326", geodesic=False)

# SRTM 1 arc-second (ราว 30 ม.) แถบ elevation หน่วยเมตร
dem = ee.Image("USGS/SRTMGL1_003").select("elevation")

# ความชันคำนวณที่ความละเอียดเดิม 30 ม. ก่อน (หน่วยองศา)
# ถ้าย่อเป็น 90 ม. ก่อนแล้วค่อยคำนวณ ความชันจะต่ำกว่าจริงเพราะภูมิประเทศถูกปรับเรียบไปแล้ว
slope = ee.Terrain.slope(dem)

img = dem.addBands(slope).toFloat()   # แถบ: elevation, slope

# ย่อ 30 ม. -> 90 ม. ด้วยค่าเฉลี่ยจริง (reduceResolution)
# ถ้าใช้ reproject อย่างเดียว Earth Engine จะหยิบค่าแบบ nearest ไม่ใช่ค่าเฉลี่ย
proj90 = ee.Projection("EPSG:32647").atScale(90)
img90 = img.reduceResolution(reducer=ee.Reducer.mean(), maxPixels=64).reproject(crs=proj90)

# ใช้ Export.image.toDrive ไม่ใช้ getDownloadURL เพราะไฟล์ใหญ่เกินเพดานของ getDownloadURL
# folder คือชื่อโฟลเดอร์ระดับบนสุดใน MyDrive (กำหนด path ซ้อนไม่ได้)
task = ee.batch.Export.image.toDrive(
    image=img90,
    description="srtm_terrain_90m",
    folder="burn_severity_terrain",
    fileNamePrefix="srtm_terrain_90m",
    region=region,
    crs="EPSG:32647",
    scale=90,
    maxPixels=1e9,
    fileFormat="GeoTIFF",
)
task.start()
print("สั่ง export แล้ว task id:", task.id)

# %% cell 3: ดูสถานะ (รันซ้ำได้เรื่อยๆ ปกติใช้เวลาหลายนาที)
# READY = รอคิว, RUNNING = กำลังทำ, COMPLETED = เสร็จ, FAILED = ล้มเหลว (ดู error_message)
st = task.status()
print(st["state"], st.get("error_message", ""))

# %% cell 4: ตรวจไฟล์หลัง export เสร็จ (ต้อง mount Drive ก่อน)
from google.colab import drive
drive.mount("/content/drive")

import glob
import numpy as np
import rasterio

paths = glob.glob("/content/drive/MyDrive/burn_severity_terrain/srtm_terrain_90m*.tif")
print("พบไฟล์:", paths)   # ถ้าได้ [] ให้รอ Drive ซิงก์สักครู่ หรือเช็กชื่อโฟลเดอร์

# ไฟล์ใหญ่อาจถูกแบ่งเป็นหลายชิ้น (ชื่อลงท้าย -0000000000-0000000000.tif) ให้รายงานจำนวนชิ้นมาด้วย
for p in paths:
    with rasterio.open(p) as src:
        print("\n", p)
        print("ขนาด:", src.width, "x", src.height, "แถบ:", src.count, "crs:", src.crs)
        print("ขนาดพิกเซล:", src.res, "nodata:", src.nodata)
        for b, name in [(1, "elevation"), (2, "slope")]:
            a = src.read(b, masked=True)
            print(f"{name}: min {a.min():.1f} max {a.max():.1f} mean {a.mean():.1f} "
                  f"พิกเซลไม่มีข้อมูล {a.mask.mean():.2%}")
