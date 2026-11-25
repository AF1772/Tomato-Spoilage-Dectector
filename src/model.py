"""
model.py
────────
Builds CNN models using transfer learning (Section 3.7).
Supports MobileNetV2, ResNet-50, and VGG16 as feature extractors.
All three share the same custom binary classification head.

Compatible with: TensorFlow 2.20.0, NumPy 2.2.4

Key changes from TF 2.13 → TF 2.20:
  - Use tf.keras directly (keras 3.x is bundled with TF 2.20)
  - Model saving uses native .keras format
  - BatchNormalization behaviour is consistent under training=False

Usage:
    from model import MODEL_BUILDERS, unfreeze_top_layers
    model, base = MODEL_BUILDERS["mobilenetv2"]()
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.applications import MobileNetV2, ResNet50, VGG16

# ── Shared constants ───────────────────────────────────────────────────────────
IMG_SIZE    = (224, 224)
INPUT_SHAPE = (224, 224, 3)
L2_LAMBDA   = 1e-4
# ───────────────────────────────────────────────────────────────────────────────


def build_classification_head(base_output, l2_lambda: float = L2_LAMBDA):
    """
    Custom binary classification head used by all three architectures.

    Structure (Section 3.7.2):
        GlobalAveragePooling2D
        → Dense(256, ReLU) + BatchNorm + Dropout(0.5)
        → Dense(128, ReLU) + BatchNorm + Dropout(0.3)
        → Dense(1, Sigmoid)  ← P(Spoiled) in [0.0, 1.0]

    Why Sigmoid + 1 neuron:
        Mathematically equivalent to Softmax + 2 neurons for binary tasks,
        but simpler and directly optimised by Binary Cross-Entropy loss.
        Output is an interpretable probability: values >= 0.5 → Spoiled.

    Why GlobalAveragePooling (not Flatten):
        Drastically reduces parameters and spatial overfitting compared
        to flattening the full feature map.
    """
    x = layers.GlobalAveragePooling2D(name="gap")(base_output)

    # ── Dense block 1 ──────────────────────────────────────────────────────────
    x = layers.Dense(
            256,
            activation="relu",
            kernel_regularizer=regularizers.L2(l2_lambda),  # L2 → capital L in TF 2.20
            name="dense_256"
        )(x)
    x = layers.BatchNormalization(name="bn_256")(x)
    x = layers.Dropout(0.5, name="dropout_0_5")(x)

    # ── Dense block 2 ──────────────────────────────────────────────────────────
    x = layers.Dense(
            128,
            activation="relu",
            kernel_regularizer=regularizers.L2(l2_lambda),
            name="dense_128"
        )(x)
    x = layers.BatchNormalization(name="bn_128")(x)
    x = layers.Dropout(0.3, name="dropout_0_3")(x)

    # ── Output ─────────────────────────────────────────────────────────────────
    output = layers.Dense(1, activation="sigmoid", name="output")(x)
    return output


# ── MobileNetV2 ────────────────────────────────────────────────────────────────
def build_mobilenetv2(freeze_base: bool = True):
    """
    MobileNetV2 — depthwise separable convolutions.
    ~3.4M parameters. Best for mobile and embedded deployment.
    Expected pixel range: [-1, 1] — handled by preprocess_input.
    """
    base = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=INPUT_SHAPE
    )
    base.trainable = not freeze_base

    inputs = keras.Input(shape=INPUT_SHAPE, name="input_image")

    # Apply architecture-specific preprocessing inside the model graph.
    # This means preprocessing is part of the saved model — no need to
    # manually preprocess images during inference in the Flask app.
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base(x, training=False)
    outputs = build_classification_head(x)

    model = Model(inputs, outputs, name="MobileNetV2_Binary")
    return model, base


# ── ResNet-50 ──────────────────────────────────────────────────────────────────
def build_resnet50(freeze_base: bool = True):
    """
    ResNet-50 — residual skip connections prevent vanishing gradients.
    ~25.6M parameters. Strong accuracy with reliable convergence.
    Expected pixel range: BGR mean-subtracted — handled by preprocess_input.
    """
    base = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=INPUT_SHAPE
    )
    base.trainable = not freeze_base

    inputs = keras.Input(shape=INPUT_SHAPE, name="input_image")
    x = tf.keras.applications.resnet50.preprocess_input(inputs)
    x = base(x, training=False)
    outputs = build_classification_head(x)

    model = Model(inputs, outputs, name="ResNet50_Binary")
    return model, base


# ── VGG16 ──────────────────────────────────────────────────────────────────────
def build_vgg16(freeze_base: bool = True):
    """
    VGG16 — deep stacks of 3×3 convolutions.
    ~138M parameters. Slowest, but produces rich texture/colour features
    that are very effective for surface spoilage detection.
    Expected pixel range: BGR mean-subtracted — handled by preprocess_input.
    """
    base = VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=INPUT_SHAPE
    )
    base.trainable = not freeze_base

    inputs = keras.Input(shape=INPUT_SHAPE, name="input_image")
    x = tf.keras.applications.vgg16.preprocess_input(inputs)
    x = base(x, training=False)
    outputs = build_classification_head(x)

    model = Model(inputs, outputs, name="VGG16_Binary")
    return model, base


# ── Fine-tuning utility ────────────────────────────────────────────────────────
def unfreeze_top_layers(base_model, unfreeze_percent: float = 0.30):
    """
    Phase 2 fine-tuning: unfreeze the top N% of base model layers.

    Lower layers learn generic features (edges, colour gradients, basic
    textures) during ImageNet training on 1.2M images. These transfer
    well to tomatoes and should NOT be altered.

    Upper layers learn higher-level, task-specific patterns. These CAN
    be safely adapted to learn tomato-specific spoilage features like
    mould patches, soft spots, and discolouration.

    Args:
        base_model       : the CNN backbone (MobileNetV2 / ResNet50 / VGG16)
        unfreeze_percent : fraction of top layers to unfreeze (default 0.30)
    """
    total_layers = len(base_model.layers)
    freeze_until = int(total_layers * (1.0 - unfreeze_percent))

    base_model.trainable = True
    for i, layer in enumerate(base_model.layers):
        layer.trainable = (i >= freeze_until)

    unfrozen = sum(1 for lyr in base_model.layers if lyr.trainable)
    frozen   = total_layers - unfrozen

    print(f"\n  Layer fine-tuning summary for {base_model.name}:")
    print(f"    Total   : {total_layers}")
    print(f"    Frozen  : {frozen}  ← ImageNet features preserved")
    print(f"    Unfrozen: {unfrozen}  ← adapting to tomato features")


def get_model_summary(model) -> str:
    """Returns a string summary — useful for logging to file."""
    lines = []
    model.summary(print_fn=lambda x: lines.append(x))
    return "\n".join(lines)


# ── Registry ───────────────────────────────────────────────────────────────────
MODEL_BUILDERS = {
    "mobilenetv2" : build_mobilenetv2,
    "resnet50"    : build_resnet50,
    "vgg16"       : build_vgg16,
}