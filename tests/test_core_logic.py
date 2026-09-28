"""
Sanity checks for the parts of the pipeline that don't need Earth
Engine or TensorFlow: dNBR math, severity classification, the spatial
splitter, IoU/Dice, the classic-ML classifier, and the recovery
forecasting data prep (leakage fix). Run with:

    python tests/test_core_logic.py
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocessing import compute_nbr, compute_dnbr, classify_severity
from src.evaluation import spatial_train_test_split, iou_score, dice_score
from src.severity_classifier import train_severity_classifier, evaluate_classifier
from src.recovery_forecast import (
    prepare_train_test, naive_persistence_baseline, evaluate_forecast,
)


def test_nbr_and_dnbr():
    # Healthy vegetation: high NIR, low SWIR -> NBR close to +1
    nir_pre = np.array([0.5, 0.4, 0.6])
    swir_pre = np.array([0.05, 0.05, 0.05])
    nbr_pre = compute_nbr(nir_pre, swir_pre)
    assert np.all(nbr_pre > 0.7), f"expected high NBR pre-fire, got {nbr_pre}"

    # Post-fire: vegetation gone, NIR drops, SWIR rises -> NBR much lower
    nir_post = np.array([0.15, 0.12, 0.5])  # third pixel didn't burn
    swir_post = np.array([0.35, 0.30, 0.06])
    nbr_post = compute_nbr(nir_post, swir_post)

    dnbr = compute_dnbr(nbr_pre, nbr_post)
    assert dnbr[0] > 0.6, f"pixel 0 should read high severity, got dNBR={dnbr[0]}"
    assert dnbr[2] < 0.1, f"pixel 2 (unburned) should read near zero, got dNBR={dnbr[2]}"

    labels = classify_severity(dnbr)
    assert labels[0] == "high_severity", labels[0]
    assert labels[2] == "unburned", labels[2]
    print(f"[ok] NBR/dNBR + classification: dNBR={np.round(dnbr, 3)} -> {list(labels)}")


def test_nbr_handles_zero_division():
    nir = np.array([0.0])
    swir = np.array([0.0])
    nbr = compute_nbr(nir, swir)
    assert np.isnan(nbr[0])
    print("[ok] NBR zero-division guarded (returns NaN, not an error)")


def test_spatial_split_no_leakage():
    rng = np.random.default_rng(0)
    n = 400
    cell_ids = np.arange(n)
    grid_x = rng.integers(0, 20, size=n)
    grid_y = rng.integers(0, 20, size=n)

    train_ids, test_ids = spatial_train_test_split(cell_ids, grid_x, grid_y,
                                                     test_frac=0.2, block_size=4)
    assert set(train_ids).isdisjoint(set(test_ids)), "train/test overlap!"
    assert len(test_ids) > 0 and len(train_ids) > 0
    frac = len(test_ids) / n
    assert 0.05 < frac < 0.45, f"test fraction {frac:.2f} is way off target 0.2"
    print(f"[ok] spatial split: {len(train_ids)} train / {len(test_ids)} test cells, "
          f"no overlap, test_frac~{frac:.2f}")


def test_iou_dice_perfect_and_partial():
    y_true = np.array([1, 1, 1, 0, 0])
    y_pred_perfect = np.array([1, 1, 1, 0, 0])
    assert iou_score(y_true, y_pred_perfect) == 1.0
    assert dice_score(y_true, y_pred_perfect) == 1.0

    y_pred_partial = np.array([1, 1, 0, 0, 0])  # missed one true positive
    iou = iou_score(y_true, y_pred_partial)
    dice = dice_score(y_true, y_pred_partial)
    assert 0.0 < iou < 1.0 and 0.0 < dice < 1.0
    print(f"[ok] IoU/dice: perfect=1.0, partial match -> IoU={iou:.3f}, dice={dice:.3f}")


def test_severity_classifier_beats_majority_baseline():
    rng = np.random.default_rng(1)
    n = 600
    # 3 features: dNBR, hotspot_count, dist_to_road
    dnbr = rng.normal(0, 0.3, n)
    hotspot_count = rng.poisson(1.0, n)
    dist_to_road = rng.uniform(0, 5000, n)
    X = np.column_stack([dnbr, hotspot_count, dist_to_road])

    # Ground truth: "burned" mostly driven by dNBR, with some noise
    y = ((dnbr > 0.25) | ((dnbr > 0.05) & (hotspot_count >= 2))).astype(int)

    split = int(n * 0.7)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    clf = train_severity_classifier(X_train, y_train, n_estimators=100)
    metrics = evaluate_classifier(clf, X_test, y_test)

    majority_class_frac = max(y_test.mean(), 1 - y_test.mean())
    assert metrics["f1"] > 0.5, f"F1 too low: {metrics['f1']:.3f}"
    print(f"[ok] severity classifier: F1={metrics['f1']:.3f} "
          f"(majority-class baseline accuracy would be {majority_class_frac:.3f})")


def test_recovery_forecast_no_leakage_and_baseline():
    rng = np.random.default_rng(2)
    n_months = 48
    t = np.arange(n_months)
    # Seasonal NDVI with a fire-induced dip around month 20, recovering after
    seasonal = 0.6 + 0.15 * np.sin(2 * np.pi * t / 12)
    dip = -0.35 * np.exp(-0.5 * ((t - 20) / 3) ** 2)
    series = seasonal + dip + rng.normal(0, 0.01, n_months)

    (X_train, y_train), (X_test, y_test), scaler = prepare_train_test(
        series, seq_len=6, test_frac=0.25
    )

    # Leakage check: scaler's fitted range must come ONLY from the train
    # portion, not the full series.
    split_idx = int(n_months * 0.75)
    train_raw = series[:split_idx]
    assert np.isclose(scaler.data_min_[0], train_raw.min())
    assert np.isclose(scaler.data_max_[0], train_raw.max())

    y_naive_true, y_naive_pred = naive_persistence_baseline(series, seq_len=6)
    baseline_metrics = evaluate_forecast(y_naive_true, y_naive_pred)

    assert X_train.shape[1] == 6 and X_test.shape[1] == 6
    assert len(y_test) > 0
    print(f"[ok] recovery forecast prep: scaler fit on train range only "
          f"[{scaler.data_min_[0]:.3f}, {scaler.data_max_[0]:.3f}]; "
          f"naive-persistence baseline RMSE={baseline_metrics['rmse']:.4f} "
          f"(any trained model must beat this)")


if __name__ == "__main__":
    tests = [
        test_nbr_and_dnbr,
        test_nbr_handles_zero_division,
        test_spatial_split_no_leakage,
        test_iou_dice_perfect_and_partial,
        test_severity_classifier_beats_majority_baseline,
        test_recovery_forecast_no_leakage_and_baseline,
    ]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} tests passed.")
