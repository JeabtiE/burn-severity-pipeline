"""
Stage 3a: Classic-ML severity/change classifier.

A fast, fully local baseline: predicts a coarse severity/change class
per grid cell from tabular features (dNBR, land-use type, hotspot
frequency, distance to road, etc). Train and run this before — or
alongside — the U-Net segmentation model in unet_segmentation.py.

Uses class_weight="balanced" because severity classes are naturally
imbalanced: most of a landscape is "unburned" in any given window, so
a classifier optimizing plain accuracy can look good while missing
almost every actual high-severity cell.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_recall_fscore_support


def train_severity_classifier(X_train: np.ndarray, y_train: np.ndarray,
                               n_estimators: int = 300, random_state: int = 42):
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    return clf


def evaluate_classifier(clf, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Weighted precision/recall/F1 plus a full per-class report. Always
    evaluate on a SPATIAL split (see evaluation.spatial_train_test_split)
    — a random split on spatially autocorrelated cells inflates every
    one of these numbers.
    """
    y_pred = clf.predict(X_test)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "report": classification_report(y_test, y_pred, zero_division=0),
    }
