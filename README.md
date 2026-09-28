# Burn-scar detection from Sentinel-2 NBR: a negative result, documented

**Status: closed.** In the three 10 x 10 km blocks I tested in northern Thailand (April 2026), NBR computed from Sentinel-2 bands B8 and B12 at 20 m did not separate GISTDA burn-scar polygons from their surroundings once I scored it on held-out blocks. This repo keeps the pipeline, the numbers and the reasoning, including the mistakes the evaluation caught.

## Summary

- **Goal:** map burn severity and vegetation recovery from Sentinel-2, using GISTDA burn-scar polygons (April 2026) as ground truth.
- **First attempt:** dNBR from a March composite (before) and a May composite (after) scored *below* chance on the development block (AUC 0.41 to 0.46).
- **Diagnosis:** the burned polygons are mostly agricultural land that already had very low NBR before burning, and the May composite falls in the start of the rainy season, when everything greens up.
- **Revised windows:** after choosing new windows from the development block's own time series, AUC rose to 0.56 (vs ring) and 0.66 (vs whole block). On two held-out blocks with the windows frozen it fell back to chance or below (0.49 to 0.50 and 0.40 to 0.47).
- **10-day window series** on all three blocks: no block shows a drop of 0.05 or more in the burned-minus-ring NBR gap, the informal threshold I set beforehand.
- **Take-away:** for this data and this index, the improvement on the first block came from tuning on the block I evaluated on. The spatial hold-out is what exposed it.

## Data

| Item | Value |
|---|---|
| Labels | GISTDA burn-scar polygons, April 2026, month-level only (no acquisition date field). Obtained through a workshop exercise; **not redistributed here**, check the provider's terms |
| Pieces after splitting multipart shapes | 4,374 |
| Piece size | median 0.10 ha, 75th percentile 0.40 ha, max 34.6 ha |
| Pieces >= 1 ha / >= 5 ha | 453 / 41 |
| Land use by burned area | other agriculture 64.5%, maize 24.1%, forest 7.4%, rice 3.8%, other 0.2% |
| Imagery | Sentinel-2 L2A harmonised (`COPERNICUS/S2_SR_HARMONIZED`) through Google Earth Engine; bands B8, B12, SCL |

One Sentinel-2 SWIR pixel is 20 m, or 0.04 ha, so the median polygon covers only about 2 to 3 pixels. Forest accounts for just 135 ha in total, too little to train on separately.

## Method

- **Blocks.** The area is cut into a 10 x 10 km grid (UTM 47N). I took the three blocks with the most burned area: `41,207` (Mae Chaem, 252.7 ha, the development block), `46,216` (Wiang Haeng, 143.1 ha) and `52,220` (Mae Ai, 89.0 ha). See `results/top_blocks.csv`. They are the largest, not a random sample.
- **Pixel masks.** Polygons are rasterised onto the exact 20 m grid of the downloaded imagery, so each pixel has a burned / not burned label. In the development block, burned pixels are 2.52% of the block.
- **Ring control.** Comparing burned pixels with the whole block mixes in land cover, so I also compare them with a ring 3 to 10 pixels (60 to 200 m) away from any burned pixel. The ring shares haze, acquisition dates and roughly the same surroundings.
- **Score.** dNBR = NBR(before) - NBR(after), used as a per-pixel score for the burn mask, summarised by ROC AUC. AUC 0.5 is chance. Accuracy is not used because a model that predicts "not burned" everywhere would already get about 97%.
- **Cloud handling.** Per-pixel masking with the SCL band, median composites, and no-data pixels filled with -9999 rather than 0. SCL does not remove smoke or haze well.
- **Development vs test.** Block `41,207` was used to look at time series and choose windows. Blocks `52,220` and `46,216` were scored once with the windows frozen.

## Experiments and results

### 1. First attempt: March vs May composites (development block)

AUC 0.411 vs ring, 0.460 vs whole block, 69.6% of pixels valid in both images.

| Group | Pixels | NBR before | NBR after | dNBR mean |
|---|---|---|---|---|
| Burned, agriculture | 4,222 | -0.091 | -0.093 | 0.003 |
| Burned, forest | 413 | -0.001 | -0.086 | 0.085 |
| Ring control | 16,990 | 0.249 | 0.201 | 0.048 |

Burned pixels already had much lower NBR than their surroundings in March, and burned agricultural pixels did not change at all. Compared with the ring, the forest signal is only about 0.04 against a pixel-level standard deviation of 0.14. Full table: `results/baseline_march_may_block_41_207.csv`.

### 2. Time series, February to June (development block)

Burned groups sit at a roughly constant offset below the ring until mid-April, all groups dip together around 16 to 20 April, and all rise sharply in late May. This is consistent with the start of the rainy season, and it explains the below-chance AUC: burned agricultural pixels green up faster than the ring, so their pre-minus-post value ends up lower. My May "after fire" composite was a poor choice. Only about a third of the 64 acquisition dates had usable pixels after masking, and there is little usable data from late April to mid-May.

### 3. Revised windows, frozen, scored on held-out blocks

Before: 20 March to 10 April. After: 14 to 22 April (the low point in the time series).

| Block | Role | Burned pixels | Valid % | AUC vs ring | AUC vs block |
|---|---|---|---|---|---|
| 41,207 | development | 6,301 | 85.4 | 0.560 | 0.661 |
| 52,220 | test | 2,333 | 100.0 | 0.486 | 0.495 |
| 46,216 | test | 3,543 | 100.0 | 0.399 | 0.470 |

Full table: `results/holdout_results.csv`.

### 4. 10-day windows, all three blocks

I tracked the burned-minus-ring NBR gap relative to the first window. A burn should appear as a step down. The largest single-window drops were about -0.03 (`41,207`, 11 to 20 April, then it recovers), -0.035 (`52,220`, 1 to 10 May) and -0.014 (`46,216`). In the two test blocks the gap mostly drifts upward instead. Data: `results/window_gap_series.csv`.

## What this does and does not show

It shows that NBR from B8/B12 at 20 m, in median composites over short windows, gave no reliable burn signal in these three blocks.

It does **not** show that burned area cannot be detected from Sentinel-2 in general. I did not test other indices or bands, 10 m data, per-pixel change detection on the full time series, Sentinel-1 radar, or land-cover-matched controls. Explanations I could not tell apart:

1. Low fuel: fields already harvested or bare before the fire leave little to burn.
2. Small fires: median polygon of 2 to 3 pixels, plus boundary noise between the polygons and the pixel grid.
3. A short-lived signal: charcoal fades and vegetation regrows quickly, and cloud gaps leave only a few scenes per window.
4. Smoke and haze that the SCL band does not remove.

## Mistakes the evaluation caught

- I first used May as the "after fire" window. The time series showed it was dominated by green-up.
- I first read a forest dNBR of 0.085 as a clear signal. Against the ring the difference shrinks to about 0.04.
- Windows tuned on the development block gave AUC 0.56 / 0.66, and did not hold on the other blocks.

## Limitations

- Three blocks, chosen by burned area, one region and one month. The result may not extend to smaller fires, forest fires or other seasons.
- Labels are month-level polygons rasterised by pixel centre. Positional error was not measured.
- The ring control is not matched by land cover.
- Neighbouring pixels are spatially correlated, so I report point estimates only, without confidence intervals.

## Reproduce

Requirements: Python 3 and `earthengine-api geopandas rasterio shapely scipy scikit-learn pandas numpy matplotlib requests`.

1. Register a Google Cloud project for Earth Engine, run `ee.Authenticate()` once, and set `EE_PROJECT` in `analysis/burn_analysis.py`.
2. Put the GISTDA burn-scar file somewhere local. The script reads the fields `LU_Name`, `AP_TN` and `name`.
3. Run `python analysis/burn_analysis.py path/to/burn_scar.shp results`.

The tables in `results/` are the values from my Colab runs. The per-date series (`nbr_daily_series_41_207.csv` and the figure) are produced by the script and not committed. I ran the functions cell by cell in Colab; `analysis/burn_analysis.py` puts them together, and I checked the assembled file on synthetic polygons with the Earth Engine calls mocked.

## Repo layout

```
src/                 first pipeline modules (data collection, preprocessing)
tests/
analysis/burn_analysis.py   masks, ring control, held-out scoring, window series
results/             small result tables
```
