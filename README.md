# Fire in northern Thailand from satellite data

Personal ML/data portfolio project. Each part fixes its evaluation before looking at results, and reports what the hold-out data said, including when that was negative.

| Part | Question | Status | Outcome |
|---|---|---|---|
| 1 | Can Sentinel-2 NBR map April 2026 burn scars (GISTDA polygons)? | closed | no reliable signal on held-out blocks |
| 2 | Which ~0.7 km² cells will have fire hotspots again next season? | Stage 1 closed | a pre-registered model beats a burn-frequency map on 9 of 9 seasons, by a small margin |

## Part 1: Burn-scar detection from Sentinel-2 NBR, a negative result

**Status: closed.** In the three 10 x 10 km blocks I tested in northern Thailand (April 2026), NBR computed from Sentinel-2 bands B8 and B12 at 20 m did not separate GISTDA burn-scar polygons from their surroundings once I scored it on held-out blocks. This repo keeps the pipeline, the numbers and the reasoning, including the mistakes the evaluation caught.

### Summary

- **Goal:** map burn severity and vegetation recovery from Sentinel-2, using GISTDA burn-scar polygons (April 2026) as ground truth.
- **First attempt:** dNBR from a March composite (before) and a May composite (after) scored *below* chance on the development block (AUC 0.41 to 0.46).
- **Diagnosis:** the burned polygons are mostly agricultural land that already had very low NBR before burning, and the May composite falls in the start of the rainy season, when everything greens up.
- **Revised windows:** after choosing new windows from the development block's own time series, AUC rose to 0.56 (vs ring) and 0.66 (vs whole block). On two held-out blocks with the windows frozen it fell back to chance or below (0.49 to 0.50 and 0.40 to 0.47).
- **10-day window series** on all three blocks: no block shows a drop of 0.05 or more in the burned-minus-ring NBR gap, the informal threshold I set beforehand.
- **Take-away:** for this data and this index, the improvement on the first block came from tuning on the block I evaluated on. The spatial hold-out is what exposed it.

### Data

| Item | Value |
|---|---|
| Labels | GISTDA burn-scar polygons, April 2026, month-level only (no acquisition date field). Obtained through a workshop exercise; **not redistributed here**, check the provider's terms |
| Pieces after splitting multipart shapes | 4,374 |
| Piece size | median 0.10 ha, 75th percentile 0.40 ha, max 34.6 ha |
| Pieces >= 1 ha / >= 5 ha | 453 / 41 |
| Land use by burned area | other agriculture 64.5%, maize 24.1%, forest 7.4%, rice 3.8%, other 0.2% |
| Imagery | Sentinel-2 L2A harmonised (`COPERNICUS/S2_SR_HARMONIZED`) through Google Earth Engine; bands B8, B12, SCL |

One Sentinel-2 SWIR pixel is 20 m, or 0.04 ha, so the median polygon covers only about 2 to 3 pixels. Forest accounts for just 135 ha in total, too little to train on separately.

### Method

- **Blocks.** The area is cut into a 10 x 10 km grid (UTM 47N). I took the three blocks with the most burned area: `41,207` (Mae Chaem, 252.7 ha, the development block), `46,216` (Wiang Haeng, 143.1 ha) and `52,220` (Mae Ai, 89.0 ha). See `results/top_blocks.csv`. They are the largest, not a random sample.
- **Pixel masks.** Polygons are rasterised onto the exact 20 m grid of the downloaded imagery, so each pixel has a burned / not burned label. In the development block, burned pixels are 2.52% of the block.
- **Ring control.** Comparing burned pixels with the whole block mixes in land cover, so I also compare them with a ring 3 to 10 pixels (60 to 200 m) away from any burned pixel. The ring shares haze, acquisition dates and roughly the same surroundings.
- **Score.** dNBR = NBR(before) - NBR(after), used as a per-pixel score for the burn mask, summarised by ROC AUC. AUC 0.5 is chance. Accuracy is not used because a model that predicts "not burned" everywhere would already get about 97%.
- **Cloud handling.** Per-pixel masking with the SCL band, median composites, and no-data pixels filled with -9999 rather than 0. SCL does not remove smoke or haze well.
- **Development vs test.** Block `41,207` was used to look at time series and choose windows. Blocks `52,220` and `46,216` were scored once with the windows frozen.

### Experiments and results

#### 1. First attempt: March vs May composites (development block)

AUC 0.411 vs ring, 0.460 vs whole block, 69.6% of pixels valid in both images.

| Group | Pixels | NBR before | NBR after | dNBR mean |
|---|---|---|---|---|
| Burned, agriculture | 4,222 | -0.091 | -0.093 | 0.003 |
| Burned, forest | 413 | -0.001 | -0.086 | 0.085 |
| Ring control | 16,990 | 0.249 | 0.201 | 0.048 |

Burned pixels already had much lower NBR than their surroundings in March, and burned agricultural pixels did not change at all. Compared with the ring, the forest signal is only about 0.04 against a pixel-level standard deviation of 0.14. Full table: `results/baseline_march_may_block_41_207.csv`.

#### 2. Time series, February to June (development block)

Burned groups sit at a roughly constant offset below the ring until mid-April, all groups dip together around 16 to 20 April, and all rise sharply in late May. This is consistent with the start of the rainy season, and it explains the below-chance AUC: burned agricultural pixels green up faster than the ring, so their pre-minus-post value ends up lower. My May "after fire" composite was a poor choice. Only about a third of the 64 acquisition dates had usable pixels after masking, and there is little usable data from late April to mid-May.

#### 3. Revised windows, frozen, scored on held-out blocks

Before: 20 March to 10 April. After: 14 to 22 April (the low point in the time series).

| Block | Role | Burned pixels | Valid % | AUC vs ring | AUC vs block |
|---|---|---|---|---|---|
| 41,207 | development | 6,301 | 85.4 | 0.560 | 0.661 |
| 52,220 | test | 2,333 | 100.0 | 0.486 | 0.495 |
| 46,216 | test | 3,543 | 100.0 | 0.399 | 0.470 |

Full table: `results/holdout_results.csv`.

#### 4. 10-day windows, all three blocks

I tracked the burned-minus-ring NBR gap relative to the first window. A burn should appear as a step down. The largest single-window drops were about -0.03 (`41,207`, 11 to 20 April, then it recovers), -0.035 (`52,220`, 1 to 10 May) and -0.014 (`46,216`). In the two test blocks the gap mostly drifts upward instead. Data: `results/window_gap_series.csv`.

### What this does and does not show

It shows that NBR from B8/B12 at 20 m, in median composites over short windows, gave no reliable burn signal in these three blocks.

It does **not** show that burned area cannot be detected from Sentinel-2 in general. I did not test other indices or bands, 10 m data, per-pixel change detection on the full time series, Sentinel-1 radar, or land-cover-matched controls. Explanations I could not tell apart:

1. Low fuel: fields already harvested or bare before the fire leave little to burn.
2. Small fires: median polygon of 2 to 3 pixels, plus boundary noise between the polygons and the pixel grid.
3. A short-lived signal: charcoal fades and vegetation regrows quickly, and cloud gaps leave only a few scenes per window.
4. Smoke and haze that the SCL band does not remove.

### Mistakes the evaluation caught

- I first used May as the "after fire" window. The time series showed it was dominated by green-up.
- I first read a forest dNBR of 0.085 as a clear signal. Against the ring the difference shrinks to about 0.04.
- Windows tuned on the development block gave AUC 0.56 / 0.66, and did not hold on the other blocks.

### Limitations

- Three blocks, chosen by burned area, one region and one month. The result may not extend to smaller fires, forest fires or other seasons.
- Labels are month-level polygons rasterised by pixel centre. Positional error was not measured.
- The ring control is not matched by land cover.
- Neighbouring pixels are spatially correlated, so I report point estimates only, without confidence intervals.

### Reproduce (Part 1)

Requirements: Python 3 and `earthengine-api geopandas rasterio shapely scipy scikit-learn pandas numpy matplotlib requests`.

1. Register a Google Cloud project for Earth Engine, run `ee.Authenticate()` once, and set `EE_PROJECT` in `analysis/burn_analysis.py`.
2. Put the GISTDA burn-scar file somewhere local. The script reads the fields `LU_Name`, `AP_TN` and `name`.
3. Run `python analysis/burn_analysis.py path/to/burn_scar.shp results`.

The tables in `results/` are the values from my Colab runs. The per-date series (`nbr_daily_series_41_207.csv` and the figure) are produced by the script and not committed. I ran the functions cell by cell in Colab; `analysis/burn_analysis.py` puts them together, and I checked the assembled file on synthetic polygons with the Earth Engine calls mocked.

## Part 2: Where will fire hotspots come back next season?

**Status: Stage 1 closed (1 October 2026).** Moving from pixels to area units, I asked which ~0.7 km² cells are likely to have a satellite fire detection again in the next January to May fire season. A machine-learning model beat the strongest simple baseline on all nine test seasons under criteria I locked before training, but the margin is small, and a plain map of how often each cell has burned already captures most of the value.

### Design

| Item | Value |
|---|---|
| Area | bounding box 98.0 to 99.5 E, 18.2 to 20.0 N. It covers most of Chiang Mai but cuts off the southern districts and includes parts of neighbouring provinces and Myanmar, so I call it "northern Thailand" |
| Unit | H3 cells at resolution 8 (about 0.74 km²), 37,359 cells. The cell set is fixed by geography, not by where fires happened |
| Data | NASA FIRMS archive download, 617,017 detections from all sensors, 2000 to 2026 |
| Label | a cell is "burned" in season Y if it has at least one VIIRS S-NPP detection, January to May, vegetation fire type, confidence at least nominal |
| Rule | features for season Y use only seasons before Y |

Sensors are never mixed in one count. VIIRS detects roughly 5 to 10 times more hotspots than MODIS in the same years, so a mixed series would jump with the sensor rather than with burning. Series A (VIIRS S-NPP, 2012 to 2026) is the main analysis. Series B (MODIS, 2003 to 2026, resolution 7 because a MODIS pixel is larger than a resolution 8 cell) is a robustness check. NOAA-20 and NOAA-21 are not used. Both series see the same fires from 2012 on, so their agreement shows the result does not depend on the sensor, not that it was replicated on independent events.

### Stage 0: is there a usable signal?

Before any model, I scored four baselines with rolling-origin evaluation (train on seasons before T, test on season T) and a pass/fail gate fixed in advance: B0 (burned last season), B1_last3 (seasons burned out of the last 3), B1_all (seasons burned out of all previous seasons) and B2_smooth (last season's hotspot density averaged with neighbouring cells).

| Series | Cells | Test seasons | Burned cells (median) | Best baseline | lift@10% (median) | Gate |
|---|---|---|---|---|---|---|
| A, res 7 | 5,442 | 12 | 66% | B2_smooth | 1.28 | pass (not meaningful, see below) |
| A, res 8 | 37,359 | 12 | 28% | B1_all | 2.07 | fail |
| B, res 7 | 5,439 | 21 | 26% | B1_all | 2.23 | fail |

lift@10% is the burned share among the top 10% of cells by score, divided by the burned share overall. The gate asked for lift of at least min(3, 0.8 / burned share) in at least half of the test seasons.

- **Recurrence is real but moderate.** A cell that burned last season is about twice as likely to burn again (median relative risk 1.86 to 2.11; above 1 in all 51 season pairs).
- **Long memory helps.** B1_all beat B0 in every series (res 8 PR-AUC 0.440 vs 0.327).
- **Hotspots are not very concentrated across years.** The top 10% of cells by total hotspots hold only 26% to 35% of all hotspots. Within one season detections cluster, but the clusters move between years, which fits rotational burning.
- **The gate had a flaw I designed in.** When two thirds of cells burn (series A, res 7), the threshold min(3, 0.8 / p) drops below 1.0 in four seasons, so random ranking would pass. I had only tested the gate on synthetic data with low burned shares. I did not change the gate after seeing this; I simply do not count that pass.

By the locked criteria Stage 0 failed. I then decided, as a new and explicitly time-boxed step, to test whether a model could beat B1_all at resolution 8. Choosing resolution 8 as the main setting happened after seeing Stage 0 (resolution 7 is saturated), so results for resolution 7 and series B are always reported alongside.

### Stage 1: does a model beat B1_all?

Everything below was fixed in [`recurrence/stage1_prereg.md`](recurrence/stage1_prereg.md) before training: 19 features (burn history of the cell, neighbour averages at rings 1 to 3, and elevation and slope from SRTM aggregated per cell), one primary model with fixed hyperparameters and no tuning (M2, `HistGradientBoostingClassifier`), test seasons 2018 to 2026 with rolling-origin training, and the win rule. M2 wins only if it beats B1_all on PR-AUC in at least 7 of 9 seasons, by at least +0.02 on average, without a lower mean lift@10%.

Before touching real labels, the code passed synthetic tests: random data (all scores near 0.5 and a "does not win" verdict), a leakage test, high and low burned shares, and a terrain-driven positive control. On real data it reproduced every Stage 0 baseline number exactly before any model was trained.

**Result: M2 wins on all three criteria.**

| Criterion | Result |
|---|---|
| PR-AUC above B1_all | 9 of 9 seasons |
| Mean PR-AUC difference | +0.030 (0.437 to 0.467); 95% spatial block bootstrap interval +0.024 to +0.035 |
| Mean lift@10% | 2.30 vs 2.19 |

| Season | Burned share | PR-AUC B1_all | PR-AUC M2 | Difference | lift B1_all | lift M2 |
|---|---|---|---|---|---|---|
| 2018 | 0.172 | 0.322 | 0.390 | +0.068 | 2.37 | 2.72 |
| 2019 | 0.405 | 0.577 | 0.600 | +0.023 | 1.70 | 1.70 |
| 2020 | 0.430 | 0.588 | 0.614 | +0.026 | 1.60 | 1.66 |
| 2021 | 0.241 | 0.440 | 0.482 | +0.042 | 2.27 | 2.36 |
| 2022 | 0.103 | 0.238 | 0.267 | +0.028 | 2.91 | 3.13 |
| 2023 | 0.312 | 0.494 | 0.501 | +0.008 | 1.88 | 1.87 |
| 2024 | 0.311 | 0.534 | 0.561 | +0.027 | 2.08 | 2.14 |
| 2025 | 0.124 | 0.272 | 0.288 | +0.016 | 2.66 | 2.80 |
| 2026 | 0.256 | 0.470 | 0.504 | +0.034 | 2.21 | 2.31 |

Mean over the nine seasons for every model (full table: `results/stage1/A_SNPP_res8_stage1_seasons.csv`):

| Model | ROC-AUC | PR-AUC | lift@10% | recall@10% |
|---|---|---|---|---|
| B1_all | 0.736 | 0.437 | 2.19 | 0.219 |
| B1_last3 | 0.679 | 0.377 | 1.94 | 0.194 |
| B2_smooth | 0.677 | 0.388 | 1.97 | 0.197 |
| M2 (primary) | 0.742 | 0.467 | 2.30 | 0.230 |
| M2 without terrain | 0.731 | 0.455 | 2.25 | 0.225 |
| M1, logistic regression | 0.761 | 0.487 | 2.37 | 0.236 |

Secondary series, same code with nothing changed: series A at resolution 7, M2 above B1_all in 8 of 9 seasons (+0.024, interval +0.019 to +0.029); series B (MODIS) at resolution 7, 17 of 18 seasons (+0.031, interval +0.027 to +0.036). The verdict files for these series print "ML WINS" because the code applies the "at least 7 seasons" rule to any number of seasons, which means little for 18 seasons; read the counts instead.

### How to read this

- **The gain is small.** The top 10% of cells flagged by M2 contain 23.0% of the cells that burned, against 21.9% for B1_all. The model is consistently better, not much better.
- **Terrain is what cleared the bar.** Without the terrain features, M2 beats B1_all in 8 of 9 seasons but only by +0.017 on average, which fails the +0.02 criterion. Terrain was in the primary model before any result was seen, so the verdict stands, but the margin depends on it.
- **Logistic regression did better than the primary model** in every season. M1 was a secondary model, so this is exploratory. It also showed up on synthetic data, which I generated from a logistic function. I am not swapping it in after the fact; it is a question for the next test.
- **One step happened out of order.** The pre-registration says it must be committed before training. The main analysis first ran about 40 minutes before that commit, and again after it. The file's content was fixed earlier (stored in Drive before the first run, never edited, and byte-identical to the committed version), so no criterion changed after results were seen. The deviation is logged in the pre-registration file.
- The bootstrap interval reflects which cells are scored, with the trained models held fixed. It does not cover variation between seasons or between training runs.

### Limitations

- A hotspot is a satellite fire detection, not burned area. Small, short or smoke-covered fires are missed, so the target is "detected burning again".
- Nine test seasons, and neighbouring cells are correlated, so uncertainty is larger than the cell counts suggest.
- Fire-season intensity varies a lot between years (S-NPP detections from 5,780 in 2022 to 30,903 in 2020), so I evaluate ranking within each season, not counts.
- The bounding box does not match the province; results are not district or sub-district rankings yet.
- Results are reported for cells only. The project deliberately does not work at the level of individual plots, so it cannot be used to single out individual farmers.

### Reproduce (Part 2)

1. Request FIRMS archive data for the bounding box (MODIS and VIIRS S-NPP, CSV) and put the folders under one directory. Keep any FIRMS MAP_KEY in Colab Secrets, not in code.
2. Run `recurrence/terrain_export.py` cell by cell in Colab to export SRTM elevation and slope (90 m, EPSG:32647) to Drive.
3. Run `recurrence/colab_stage1.py` cell by cell. `stage0.run_all` reproduces Stage 0; `stage1.check_b1` must pass before `stage1.run`.
4. Synthetic tests: `PYTHONPATH=recurrence python tests/test_stage1.py`.

Library versions differ slightly in gradient boosting results across scikit-learn releases. The reported numbers come from my Colab run with scikit-learn 1.6.1, numpy 2.1.3 and pandas 2.2.3 (about 5 minutes for the main analysis, including 1,000 bootstrap resamples).

## Repo layout

```
src/                         first pipeline modules (Part 1)
analysis/burn_analysis.py    Part 1: masks, ring control, held-out scoring, window series
results/                     Part 1 result tables
recurrence/stage0.py         Part 2, Stage 0: data loading, H3 cells, baselines, gate
recurrence/stage1.py         Part 2, Stage 1: features, models, verdict, bootstrap
recurrence/stage1_prereg.md  Part 2, Stage 1 pre-registration (locked 1 October 2026)
recurrence/terrain_export.py SRTM export from Earth Engine (Colab cells)
recurrence/colab_stage1.py   order of Colab cells for Stage 1
results/stage0/, results/stage1/   Part 2 result tables
tests/                       unit and synthetic tests
burning_recurrence_risk.ipynb      Part 2 Colab notebook with outputs
```
