"""
prepare_dataset.py
──────────────────
Splits raw images (dataset/raw/Fresh & dataset/raw/Spoiled)
into train / val / test subsets using stratified sampling.

Split ratios  : 70% train | 15% val | 15% test
Run with      : python src/prepare_dataset.py

Compatible with: Python 3.10-3.12, scikit-learn 1.6.1, NumPy 2.2.4
"""

import os
import shutil
import random
from pathlib import Path
from sklearn.model_selection import train_test_split

# ── Configuration ──────────────────────────────────────────────────────────────
RAW_DIR     = "dataset/raw"
SPLIT_DIR   = "dataset"
CLASSES     = ["Fresh", "Spoiled"]
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
SEED        = 42
# ───────────────────────────────────────────────────────────────────────────────


def collect_images(raw_dir: str, classes: list) -> list:
    """Returns a flat list of (filepath, label) tuples for all classes."""
    data = []
    for cls in classes:
        cls_path = Path(raw_dir) / cls
        if not cls_path.exists():
            raise FileNotFoundError(
                f"\n  Folder not found: {cls_path}\n"
                f"  Please add images to:\n"
                f"    dataset/raw/Fresh/\n"
                f"    dataset/raw/Spoiled/"
            )
        images = (
            list(cls_path.glob("*.jpg"))  +
            list(cls_path.glob("*.jpeg")) +
            list(cls_path.glob("*.png"))  +
            list(cls_path.glob("*.webp")) +
            list(cls_path.glob("*.JPG"))  +
            list(cls_path.glob("*.JPEG"))
        )
        if len(images) == 0:
            raise ValueError(
                f"\n  No images found in {cls_path}\n"
                f"  Make sure images are .jpg / .jpeg / .png / .webp"
            )
        for img in images:
            data.append((str(img), cls))
        print(f"  {cls:10s} : {len(images):,} images found")
    return data


def clear_split_dirs():
    """Remove old train/val/test folders before re-splitting."""
    for split in ["train", "val", "test"]:
        split_path = Path(SPLIT_DIR) / split
        if split_path.exists():
            shutil.rmtree(split_path)


def copy_images(pairs, split_name: str):
    """Copies image files into dataset/<split>/<class>/ directories."""
    copied = 0
    for filepath, label in pairs:
        dest_dir = Path(SPLIT_DIR) / split_name / label
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_file = dest_dir / Path(filepath).name
        # Avoid overwriting if duplicate names exist across sources
        if dest_file.exists():
            stem = dest_file.stem
            suffix = dest_file.suffix
            dest_file = dest_dir / f"{stem}_{copied}{suffix}"
        shutil.copy2(filepath, dest_file)
        copied += 1
    print(f"  {split_name:8s} : {copied:,} images copied")


def verify_split(split_name: str):
    """Prints per-class counts for a split to confirm balance."""
    for cls in CLASSES:
        path = Path(SPLIT_DIR) / split_name / cls
        count = len(list(path.glob("*.*"))) if path.exists() else 0
        print(f"    {split_name}/{cls}: {count:,}")


def prepare():
    random.seed(SEED)

    print("\n" + "=" * 52)
    print("  DATASET PREPARATION")
    print("=" * 52)

    print("\n── Step 1: Collecting raw images ──")
    all_data = collect_images(RAW_DIR, CLASSES)
    total = len(all_data)
    print(f"  Total    : {total:,} images")

    if total < 20:
        print(f"\n  ⚠ Warning: Only {total} images found.")
        print(f"  Recommended minimum: 500+ per class for reliable training.")

    paths  = [d[0] for d in all_data]
    labels = [d[1] for d in all_data]

    print("\n── Step 2: Splitting dataset ──")
    # Split 1: 70% train, 30% temp
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        paths, labels,
        test_size=(VAL_RATIO + TEST_RATIO),
        stratify=labels,
        random_state=SEED
    )

    # Split 2: 50/50 from temp → 15% val, 15% test
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels,
        test_size=0.5,
        stratify=temp_labels,
        random_state=SEED
    )

    print(f"  Train : {len(train_paths):,}  ({len(train_paths)/total*100:.0f}%)")
    print(f"  Val   : {len(val_paths):,}   ({len(val_paths)/total*100:.0f}%)")
    print(f"  Test  : {len(test_paths):,}   ({len(test_paths)/total*100:.0f}%)")

    print("\n── Step 3: Clearing old split folders ──")
    clear_split_dirs()
    print("  Old folders cleared.")

    print("\n── Step 4: Copying files ──")
    copy_images(zip(train_paths, train_labels), "train")
    copy_images(zip(val_paths,   val_labels),   "val")
    copy_images(zip(test_paths,  test_labels),  "test")

    print("\n── Step 5: Verifying class balance ──")
    for split in ["train", "val", "test"]:
        verify_split(split)

    print("\n" + "=" * 52)
    print("  ✓ Dataset ready!")
    print("  Next: python src/train.py")
    print("=" * 52 + "\n")


if __name__ == "__main__":
    prepare()