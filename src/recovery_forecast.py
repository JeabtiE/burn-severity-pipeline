"""
Stage 4: Post-fire vegetation recovery forecasting (NDVI time-series).

Fixes vs. the workshop's GRU notebook:
  1. The scaler is fit on the TRAIN split ONLY, then applied to test.
     The workshop fit MinMaxScaler on the full series before
     splitting, which leaks test-set min/max into training.
  2. SEQ_LEN defaults to a real sequence length (12 = ~1 year of
     monthly points), not 2 — a sequence length of 2 barely uses any
     history, which defeats the point of a recurrent model.
  3. A naive-persistence baseline is included so a GRU's improvement
     (or lack of one) can actually be judged, instead of just
     plotting predictions next to ground truth and eyeballing it.

The baseline and data-prep functions are plain numpy/sklearn and are
exercised in tests/test_core_logic.py. build_gru_model() needs
TensorFlow, imported locally so this file still imports cleanly
without it.
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error


def naive_persistence_baseline(series: np.ndarray, seq_len: int):
    """
    Predicts the next value = the last observed value. This is the
    bar any real forecasting model has to clear — if a GRU can't beat
    this, it isn't adding value.
    """
    y_true, y_pred = [], []
    for i in range(len(series) - seq_len):
        y_true.append(series[i + seq_len])
        y_pred.append(series[i + seq_len - 1])
    return np.array(y_true), np.array(y_pred)


def create_sequences(data: np.ndarray, seq_len: int):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len])
    return np.array(X), np.array(y)


def prepare_train_test(series: np.ndarray, seq_len: int = 12, test_frac: float = 0.2):
    """
    Chronological split BEFORE scaling — this is the leakage fix.
    Returns scaled train/test sequences plus the fitted scaler (needed
    to inverse-transform predictions back to NDVI units later).
    """
    split_idx = int(len(series) * (1 - test_frac))
    train_raw, test_raw = series[:split_idx], series[split_idx:]

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_raw.reshape(-1, 1)).flatten()
    test_scaled = scaler.transform(test_raw.reshape(-1, 1)).flatten()

    X_train, y_train = create_sequences(train_scaled, seq_len)
    # Bridge in the tail of train so the first test sequence has enough
    # history — without this, the first seq_len test points are unusable.
    bridge = np.concatenate([train_scaled[-seq_len:], test_scaled])
    X_test, y_test = create_sequences(bridge, seq_len)

    return (X_train, y_train), (X_test, y_test), scaler


def build_gru_model(seq_len: int, n_features: int = 1):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import GRU, Dropout, Dense

    model = Sequential([
        GRU(64, return_sequences=True, input_shape=(seq_len, n_features)),
        Dropout(0.2),
        GRU(32),
        Dropout(0.2),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def evaluate_forecast(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": mean_absolute_error(y_true, y_pred),
    }
