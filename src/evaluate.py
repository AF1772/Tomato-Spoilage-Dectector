"""
evaluate.py
───────────
Full evaluation of the trained binary classifier on the held-out test set.

Computes all metrics from Section 3.10:
  - Accuracy, Precision, Recall, F1-Score
  - AUC-ROC with plotted curve
  - Confusion Matrix (plotted and printed)
  - Full per-class sklearn classification report
  - SVM baseline comparison (Section 3.10.2)

All outputs are saved to the results/ folder.

Compatible with: TensorFlow 2.20.0, scikit-learn 1.6.1,
                 matplotlib 3.10.1, seaborn 0.13.2, NumPy 2.2.4

Run with:
    python src/evaluate.py
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for all environments
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    f1_score
)
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))

# ── Configuration ──────────────────────────────────────────────────────────────
import os
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCH        = "mobilenetv2"
MODEL_PATH  = os.path.join(BASE_DIR, "models", f"{ARCH}_final.keras")
TEST_DIR    = os.path.join(BASE_DIR, "dataset", "test")
IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32
THRESHOLD   = 0.25
SEED        = 42
RESULTS_DIR = os.path.join(BASE_DIR, "results")
AUTOTUNE    = tf.data.AUTOTUNE
# ───────────────────────────────────────────────────────────────────────────────

os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Dataset loading ────────────────────────────────────────────────────────────
def load_test_dataset():
    """
    Loads test images. No augmentation — only normalisation.
    shuffle=False is critical: label order must match prediction order.
    """
    dataset = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        labels="inferred",
        label_mode="binary",
        class_names=["Fresh", "Spoiled"],
        color_mode="rgb",
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE,
        shuffle=False,
        seed=SEED
    )
    # Normalise pixels to [0, 1] — same as training
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset


def get_true_labels(dataset) -> np.ndarray:
    """Extracts all ground-truth labels from a tf.data.Dataset."""
    labels = []
    for _, y in dataset:
        labels.extend(y.numpy().flatten().tolist())
    return np.array(labels, dtype=int)


# ── Plotting ───────────────────────────────────────────────────────────────────
def plot_confusion_matrix(cm: np.ndarray, class_names: list, save_path: str):
    """Heatmap confusion matrix annotated with TP/TN/FP/FN values."""
    tn, fp, fn, tp = cm.ravel()

    # seaborn 0.13.2: heatmap API unchanged, theme now applied via set_theme()
    sns.set_theme(style="whitegrid", font_scale=1.1)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.8,
        linecolor="#cccccc",
        annot_kws={"size": 18, "weight": "bold"},
        ax=ax
    )
    ax.set_title(
        f"Confusion Matrix — {ARCH.upper()}\n"
        f"TP={tp}  TN={tn}  FP={fp}  FN={fn}",
        fontsize=13, pad=14
    )
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {save_path}")


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, save_path: str) -> float:
    """ROC curve with AUC score displayed on the chart."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="steelblue", lw=2.5,
            label=f"ROC Curve (AUC = {roc_auc:.4f})")
    ax.fill_between(fpr, tpr, alpha=0.08, color="steelblue")
    ax.plot([0, 1], [0, 1], color="#999999", linestyle="--", lw=1.2,
            label="Random Classifier (AUC = 0.50)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=12)
    ax.set_title(f"ROC Curve — {ARCH.upper()}", fontsize=14)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {save_path}")
    return roc_auc


def plot_probability_distribution(y_true: np.ndarray, y_prob: np.ndarray, save_path: str):
    """
    Histogram of predicted probabilities, separated by true class.
    A well-calibrated model shows two clearly separated peaks.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(0, 1, 40)

    fresh_probs   = y_prob[y_true == 0]
    spoiled_probs = y_prob[y_true == 1]

    ax.hist(fresh_probs,   bins=bins, color="#27ae60", alpha=0.65,
            label="True Fresh",   edgecolor="white", linewidth=0.4)
    ax.hist(spoiled_probs, bins=bins, color="#e74c3c", alpha=0.65,
            label="True Spoiled", edgecolor="white", linewidth=0.4)

    ax.axvline(x=THRESHOLD, color="black", linestyle="--", linewidth=1.5,
               label=f"Decision threshold ({THRESHOLD})")
    ax.set_xlabel("Predicted P(Spoiled)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"Prediction Probability Distribution — {ARCH.upper()}", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {save_path}")


# ── SVM baseline ───────────────────────────────────────────────────────────────
def run_svm_baseline(test_ds) -> float:
    """
    Section 3.10.2 — Traditional ML baseline using mean RGB features.
    A well-performing CNN should substantially outperform this.
    scikit-learn 1.6.1: SVC API unchanged.
    """
    print("\n── SVM Baseline Comparison (Section 3.10.2) ──")
    print("  Extracting colour features...")

    X_list, y_list = [], []
    for batch_x, batch_y in test_ds:
        imgs   = batch_x.numpy()
        labels = batch_y.numpy().flatten()
        for img, label in zip(imgs, labels):
            # 6 features: mean + std of each RGB channel
            features = []
            for c in range(3):
                ch = img[:, :, c]
                features.extend([float(ch.mean()), float(ch.std())])
            X_list.append(features)
            y_list.append(int(label))

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=int)

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    svm  = SVC(kernel="rbf", probability=True, random_state=SEED)
    svm.fit(X_scaled, y)
    preds    = svm.predict(X_scaled)
    svm_acc  = float((preds == y).mean())

    print(f"  SVM Baseline Accuracy  : {svm_acc:.4f}  ({svm_acc*100:.2f}%)")
    print(f"  (CNN should considerably exceed this)")
    return svm_acc


# ── Main evaluation ────────────────────────────────────────────────────────────
def evaluate():
    print("\n" + "=" * 56)
    print(f"  EVALUATION  |  Architecture : {ARCH.upper()}")
    print(f"  TensorFlow version          : {tf.__version__}")
    print("=" * 56)

    # Load model
    print(f"\n── Loading model: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"\n  Model not found: {MODEL_PATH}\n"
            f"  Train first: python src/train.py"
        )
    model = keras.models.load_model(MODEL_PATH)
    print("  Model loaded successfully.")

    # Load test set
    print("\n── Loading test set (no augmentation)...")
    test_ds = load_test_dataset()
    class_names = ["Fresh", "Spoiled"]

    # Count test samples
    total_test = sum(1 for _ in test_ds.unbatch())
    print(f"  Test samples : {total_test}")
    print(f"  Class names  : {class_names}  (Fresh=0, Spoiled=1)")

    # Run inference
    print("\n── Running inference...")
    y_prob_raw = model.predict(test_ds, verbose=1)
    y_prob     = y_prob_raw.flatten()
    y_pred     = (y_prob >= THRESHOLD).astype(int)
    y_true     = get_true_labels(test_ds)

    # Sanity check
    assert len(y_true) == len(y_pred), (
        f"Label/prediction count mismatch: {len(y_true)} vs {len(y_pred)}"
    )

    # ── Core metrics ──────────────────────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    accuracy  = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)  if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn)  if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    # ── Print results ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 56}")
    print(f"  RESULTS — {ARCH.upper()}")
    print(f"{'=' * 56}")
    print(f"  Accuracy  : {accuracy:.4f}   ({accuracy*100:.2f}%)")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}   ← highest priority (food safety)")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"                    Pred Fresh    Pred Spoiled")
    print(f"    Actual Fresh    TN = {tn:<8d}  FP = {fp}")
    print(f"    Actual Spoiled  FN = {fn:<8d}  TP = {tp}")
    print(f"\n  Error meaning:")
    print(f"    FP = {fp:<6d} → Fresh tomatoes wrongly rejected  (economic waste)")
    print(f"    FN = {fn:<6d} → Spoiled tomatoes wrongly passed  (food safety risk)")

    print(f"\n── Full Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # ── Plots ──────────────────────────────────────────────────────────────────
    print("── Generating plots...")
    roc_auc = plot_roc_curve(
        y_true, y_prob,
        f"{RESULTS_DIR}/{ARCH}_roc_curve.png"
    )
    plot_confusion_matrix(
        cm, class_names,
        f"{RESULTS_DIR}/{ARCH}_confusion_matrix.png"
    )
    plot_probability_distribution(
        y_true, y_prob,
        f"{RESULTS_DIR}/{ARCH}_probability_distribution.png"
    )
    print(f"  AUC-ROC : {roc_auc:.4f}")

    # ── SVM baseline ──────────────────────────────────────────────────────────
    svm_acc = run_svm_baseline(test_ds)

    # ── Save summary ──────────────────────────────────────────────────────────
    summary_path = f"{RESULTS_DIR}/{ARCH}_metrics_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Architecture        : {ARCH.upper()}\n")
        f.write(f"TensorFlow version  : {tf.__version__}\n")
        f.write(f"Decision threshold  : {THRESHOLD}\n")
        f.write(f"Test samples        : {total_test}\n\n")
        f.write(f"Accuracy            : {accuracy:.4f}\n")
        f.write(f"Precision           : {precision:.4f}\n")
        f.write(f"Recall (Spoiled)    : {recall:.4f}\n")
        f.write(f"F1-Score            : {f1:.4f}\n")
        f.write(f"AUC-ROC             : {roc_auc:.4f}\n\n")
        f.write(f"Confusion Matrix:\n")
        f.write(f"  TP={tp}  TN={tn}  FP={fp}  FN={fn}\n\n")
        f.write(f"SVM Baseline Accuracy : {svm_acc:.4f}\n")
        f.write(f"CNN improvement       : +{(accuracy - svm_acc)*100:.2f}%\n")

    print(f"\n  Summary saved → {summary_path}")
    print(f"\n  All results in: {RESULTS_DIR}/")
    print(f"\n  Next step: python app.py\n")


if __name__ == "__main__":
    evaluate()