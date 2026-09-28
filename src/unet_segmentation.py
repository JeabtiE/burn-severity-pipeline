"""
Stage 3b: U-Net segmentation for pixel-level burn-scar masks.

Same architecture family as the workshop's Building Segmentation
notebook, with three fixes:
  1. More training epochs with early stopping (the workshop trained a
     flat 5 epochs, batch size 2 — barely enough to see the loss
     start moving, let alone converge)
  2. IoU / Dice tracked as real metrics during training and used for
     early stopping / checkpointing, not just eyeballing one random
     prediction at the end
  3. Meant to be trained and evaluated on a SPATIAL train/test split
     (see evaluation.spatial_train_test_split) — never a random
     image-level split for spatially autocorrelated tiles

All TensorFlow imports are local to each function so this file can
still be imported (and its shapes reasoned about) in environments
without TensorFlow installed — this sandbox is one of them, so the
functions below are written carefully but not executed here.
"""


def iou_metric(y_true, y_pred, smooth: float = 1e-6):
    import tensorflow.keras.backend as K
    y_pred = K.cast(y_pred > 0.5, "float32")
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred) - intersection
    return (intersection + smooth) / (union + smooth)


def dice_coef(y_true, y_pred, smooth: float = 1e-6):
    import tensorflow.keras.backend as K
    y_pred = K.cast(y_pred > 0.5, "float32")
    intersection = K.sum(y_true * y_pred)
    return (2.0 * intersection + smooth) / (K.sum(y_true) + K.sum(y_pred) + smooth)


def build_unet(input_size=(256, 256, 3)):
    from tensorflow.keras import layers, models

    inputs = layers.Input(input_size)

    c1 = layers.Conv2D(32, 3, activation="relu", padding="same")(inputs)
    c1 = layers.BatchNormalization()(c1)
    c1 = layers.Conv2D(32, 3, activation="relu", padding="same")(c1)
    p1 = layers.MaxPooling2D(2)(c1)

    c2 = layers.Conv2D(64, 3, activation="relu", padding="same")(p1)
    c2 = layers.BatchNormalization()(c2)
    c2 = layers.Conv2D(64, 3, activation="relu", padding="same")(c2)
    p2 = layers.MaxPooling2D(2)(c2)

    c3 = layers.Conv2D(128, 3, activation="relu", padding="same")(p2)
    c3 = layers.BatchNormalization()(c3)
    c3 = layers.Conv2D(128, 3, activation="relu", padding="same")(c3)

    u2 = layers.UpSampling2D(2)(c3)
    u2 = layers.Concatenate()([u2, c2])
    u2 = layers.Conv2D(64, 3, activation="relu", padding="same")(u2)

    u1 = layers.UpSampling2D(2)(u2)
    u1 = layers.Concatenate()([u1, c1])
    u1 = layers.Conv2D(32, 3, activation="relu", padding="same")(u1)

    outputs = layers.Conv2D(1, 1, activation="sigmoid")(u1)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[iou_metric, dice_coef],
    )
    return model


def train_unet(model, X_train, Y_train, X_val, Y_val,
               epochs: int = 60, batch_size: int = 8,
               checkpoint_path: str = "best_unet.keras"):
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

    callbacks = [
        EarlyStopping(monitor="val_iou_metric", mode="max", patience=10,
                      restore_best_weights=True),
        ModelCheckpoint(checkpoint_path, monitor="val_iou_metric", mode="max",
                         save_best_only=True),
    ]
    return model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
    )
