"""
Stage 5: Evaluation utilities shared across the pipeline.

Includes:
  - a spatial train/test splitter (splits by grid block, not by row) —
    the fix for the workshop's random 80/20 splits, which let
    spatially-adjacent (and therefore highly correlated) cells leak
    between train and test and inflate every reported metric
  - plain-numpy IoU / Dice for segmentation masks, usable without
    TensorFlow (e.g. for post-hoc scoring of saved predictions)
"""

import numpy as np


def spatial_train_test_split(cell_ids: np.ndarray, grid_x: np.ndarray, grid_y: np.ndarray,
                              test_frac: float = 0.2, block_size: int = 5,
                              random_state: int = 42):
    """
    Splits cells into train/test by spatial block rather than
    randomly, so neighbouring (spatially correlated) cells don't end
    up on both sides of the split.

    grid_x / grid_y: integer grid coordinates for each cell in cell_ids.
    block_size: cells are grouped into block_size x block_size blocks;
                whole blocks go to train or test together.
    """
    rng = np.random.default_rng(random_state)
    block_x = grid_x // block_size
    block_y = grid_y // block_size
    block_id = np.array([f"{bx}_{by}" for bx, by in zip(block_x, block_y)])

    unique_blocks = np.unique(block_id)
    rng.shuffle(unique_blocks)
    n_test_blocks = max(1, int(len(unique_blocks) * test_frac))
    test_blocks = set(unique_blocks[:n_test_blocks])

    test_mask = np.array([b in test_blocks for b in block_id])
    return cell_ids[~test_mask], cell_ids[test_mask]


def iou_score(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
    y_pred_bin = (y_pred > threshold).astype(np.uint8)
    y_true_bin = (y_true > threshold).astype(np.uint8)
    intersection = np.logical_and(y_true_bin, y_pred_bin).sum()
    union = np.logical_or(y_true_bin, y_pred_bin).sum()
    return float(intersection) / float(union) if union > 0 else 1.0


def dice_score(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
    y_pred_bin = (y_pred > threshold).astype(np.uint8)
    y_true_bin = (y_true > threshold).astype(np.uint8)
    intersection = np.logical_and(y_true_bin, y_pred_bin).sum()
    denom = y_true_bin.sum() + y_pred_bin.sum()
    return float(2 * intersection) / float(denom) if denom > 0 else 1.0
