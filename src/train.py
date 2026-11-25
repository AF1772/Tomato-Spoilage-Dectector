"""
train.py
────────
Two-phase training pipeline for the binary tomato spoilage classifier.

Phase 1 — Train only the custom classification head (base CNN frozen).
Phase 2 — Unfreeze top 30% of base and fine-tune at a very low LR.

Compatible with: TensorFlow 2.20.0, NumPy 2.2.4

Key changes from TF 2.13 → TF 2.20:
  - tf.keras.utils.image_dataset_from_directory replaces ImageDataGenerator
    for the main pipeline (modern TF 2.20 approach — faster, uses tf.data)
  - Augmentation layers are now defined as a keras.Sequential model
    applied inside the training pipeline
  - Model saved with .keras extension (native Keras 3 format)
  - regularizers.L2 (capital L) replaces regularizers.l2

Run with:
    python src/train.py

Change ARCH to switch between: mobilenetv2 | resnet50 | vgg16
"""

import os
import sys
import tensorflow as tf
from tensorflow import keras

sys.path.insert(0, os.path.dirname(__file__))
from model import MODEL_BUILDERS, unfreeze_top_layers, get_model_summary

# ── Configuration ──────────────────────────────────────────────────────────────
ARCH         = "mobilenetv2"   # "mobilenetv2" | "resnet50" | "vgg16"
IMG_SIZE     = (224, 224)
BATCH_SIZE   = 32
SEED         = 42
AUTOTUNE     = tf.data.AUTOTUNE

# Phase 1 — head only
P1_LR        = 1e-3
P1_EPOCHS    = 30
P1_PATIENCE  = 5

# Phase 2 — fine-tuning
P2_LR        = 1e-5
P2_EPOCHS    = 20
P2_PATIENCE  = 3
UNFREEZE_PCT = 0.30

# Paths — absolute so PyCharm always finds them regardless of working directory
import os
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "train")
VAL_DIR   = os.path.join(BASE_DIR, "dataset", "val")
MODEL_DIR = os.path.join(BASE_DIR, "models")
LOG_DIR   = os.path.join(BASE_DIR, "logs")
# ───────────────────────────────────────────────────────────────────────────────

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR,   exist_ok=True)


# ── Augmentation layers (applied only during training) ────────────────────────
# In TF 2.20, augmentation is defined as Keras layers and applied as part of
# the tf.data pipeline. This replaces the old ImageDataGenerator approach and
# runs on GPU when available — significantly faster.
data_augmentation = keras.Sequential([
    keras.layers.RandomFlip("horizontal_and_vertical"),
    keras.layers.RandomRotation(0.083),        # 30 degrees = 30/360 = 0.083
    keras.layers.RandomZoom(0.20),
    keras.layers.RandomTranslation(0.10, 0.10),
    keras.layers.RandomBrightness(0.30),       # ±30% brightness
    keras.layers.RandomContrast(0.20),
], name="data_augmentation")


# ── Dataset loading ────────────────────────────────────────────────────────────
def load_dataset(directory: str, shuffle: bool = True):
    dataset = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="binary",
        class_names=["Fresh", "Spoiled"],
        color_mode="rgb",
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE,
        shuffle=shuffle,
        seed=SEED,
        interpolation="bilinear"
    )
    return dataset


def prepare_dataset(dataset, is_training: bool = False):
    # DO NOT normalise here — MobileNetV2/ResNet50/VGG16 each have
    # their own preprocess_input built inside model.py which expects
    # raw pixel values in [0, 255]. Normalising here would break that.

    if is_training:
        dataset = dataset.map(
            lambda x, y: (data_augmentation(x, training=True), y),
            num_parallel_calls=AUTOTUNE
        )

    return dataset.cache().prefetch(buffer_size=AUTOTUNE)


# ── Metrics ────────────────────────────────────────────────────────────────────
def get_metrics():
    return [
        keras.metrics.BinaryAccuracy(name="accuracy"),
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
        keras.metrics.AUC(name="auc", curve="ROC")
    ]


# ── Callbacks ─────────────────────────────────────────────────────────────────
def get_callbacks(phase: int):
    checkpoint_path = os.path.join(
        MODEL_DIR, f"{ARCH}_phase{phase}_best.keras"
    )
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=P1_PATIENCE if phase == 1 else P2_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.TensorBoard(
            log_dir=os.path.join(LOG_DIR, ARCH, f"phase{phase}"),
            histogram_freq=1,
            update_freq="epoch"
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1
        )
    ]


# ── Main training function ─────────────────────────────────────────────────────
def train():
    print("\n" + "=" * 56)
    print(f"  TRAINING  |  Architecture : {ARCH.upper()}")
    print(f"  TensorFlow version        : {tf.__version__}")
    print("=" * 56)

    # ── Load datasets ──────────────────────────────────────────────────────────
    print("\n── Loading datasets ──")
    raw_train = load_dataset(TRAIN_DIR, shuffle=True)
    raw_val   = load_dataset(VAL_DIR,   shuffle=False)

    print(f"\n  Class names : {raw_train.class_names}")
    print(f"  Fresh = 0, Spoiled = 1")

    train_ds = prepare_dataset(raw_train, is_training=True)
    val_ds   = prepare_dataset(raw_val,   is_training=False)

    # ── Build model ────────────────────────────────────────────────────────────
    print("\n── Building model ──")
    build_fn = MODEL_BUILDERS[ARCH]
    model, base_model = build_fn(freeze_base=True)

    # Log model summary to file
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(os.path.join(LOG_DIR, f"{ARCH}_summary.txt"), "w") as f:
        f.write(get_model_summary(model))

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 1 — Train classification head only
    # The pre-trained convolutional base is completely frozen.
    # Only the new layers (GAP, Dense, Dropout, Output) are updated.
    # This prevents destroying ImageNet-learned features before the
    # new head has learned meaningful representations.
    # ──────────────────────────────────────────────────────────────────────────
    print(f"\n{'─' * 56}")
    print(f"  PHASE 1 — Training Classification Head")
    print(f"  Base: FROZEN  |  LR: {P1_LR}  |  Max epochs: {P1_EPOCHS}")
    print(f"{'─' * 56}\n")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=P1_LR),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=get_metrics()
    )

    model.summary()

    history1 = model.fit(
        train_ds,
        epochs=P1_EPOCHS,
        validation_data=val_ds,
        callbacks=get_callbacks(1),
        verbose=1
    )

    p1_best_acc = max(history1.history.get("val_accuracy", [0]))
    print(f"\n  ✓ Phase 1 complete. Best val accuracy: {p1_best_acc:.4f}")

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 2 — Fine-tune top 30% of base layers
    # A very low learning rate (1e-5) prevents large weight updates that
    # would cause catastrophic forgetting of the general visual features
    # built during ImageNet pre-training.
    # ──────────────────────────────────────────────────────────────────────────
    print(f"\n{'─' * 56}")
    print(f"  PHASE 2 — Fine-Tuning Base Layers")
    print(f"  Unfreezing top {UNFREEZE_PCT*100:.0f}%  |  LR: {P2_LR}  |  Max epochs: {P2_EPOCHS}")
    print(f"{'─' * 56}")

    unfreeze_top_layers(base_model, UNFREEZE_PCT)

    # Must recompile after changing trainable flags
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=P2_LR),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=get_metrics()
    )

    history2 = model.fit(
        train_ds,
        epochs=P2_EPOCHS,
        validation_data=val_ds,
        callbacks=get_callbacks(2),
        verbose=1
    )

    p2_best_acc = max(history2.history.get("val_accuracy", [0]))
    print(f"\n  ✓ Phase 2 complete. Best val accuracy: {p2_best_acc:.4f}")

    # ── Save final model ───────────────────────────────────────────────────────
    final_path = os.path.join(MODEL_DIR, f"{ARCH}_final.keras")
    model.save(final_path)
    print(f"\n  ✓ Final model saved → {final_path}")
    print(f"\n  Next step: python src/evaluate.py\n")

    return model, history1, history2


if __name__ == "__main__":
    train()